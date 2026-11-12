// The message protocol of the Brasa worker, independent of where it runs.
//
// A Session owns the engine, the tokenizer and the retriever, receives request messages through
// handle() and answers through the `post` callback given to the constructor. worker.js wires it to a
// Web Worker; the tests drive it directly in Node. worker.d.ts documents every message.
//
// A question goes through the same steps as in training and evaluation (src/brasa/evaluate.py):
//   retrieve facts -> build the prompt -> generate until <|eos|>
// and every step is reported, so a page can show what the model was given and what it did with it.

import { Engine, seededRandom } from './engine.js';
import { Tokenizer } from './tokenizer.js';
import { Retriever } from './retrieval.js';
import { buildPrompt, DEFAULT_PROMPT, SPECIALS } from './prompt.js';

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/** A macrotask boundary without the 4 ms clamp of nested setTimeout: lets queued messages (abort) run. */
function makeYield() {
  if (typeof MessageChannel === 'undefined') return () => new Promise((resolve) => setTimeout(resolve, 0));
  const channel = new MessageChannel();
  const waiting = [];
  channel.port1.onmessage = () => { const wake = waiting.shift(); if (wake) wake(); };
  if (channel.port1.unref) { channel.port1.unref(); channel.port2.unref(); }   // Node: do not keep the process alive
  return () => new Promise((resolve) => { waiting.push(resolve); channel.port2.postMessage(0); });
}

async function fetchBytes(fetchImpl, url, onProgress) {
  const res = await fetchImpl(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers && res.headers.get && res.headers.get('content-length')) || 0;
  if (!res.body || !res.body.getReader) {
    const buffer = await res.arrayBuffer();
    onProgress(buffer.byteLength, total || buffer.byteLength);
    return buffer;
  }
  const reader = res.body.getReader(), chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const chunk of chunks) { out.set(chunk, o); o += chunk.length; }
  return out.buffer;
}

export class Session {
  /**
   * @param {(message: object, transfer?: Transferable[]) => void} post  receives every response message
   * @param {{ fetch?: typeof fetch, sliceMs?: number }} [deps]
   *        sliceMs: how long generation runs before giving queued messages (abort) a chance
   */
  constructor(post, deps = {}) {
    this.post = post;
    this.fetch = deps.fetch || ((...args) => fetch(...args));
    this.sliceMs = deps.sliceMs ?? 12;
    this.yieldToEventLoop = makeYield();
    this.engine = null; this.tokenizer = null; this.retriever = null; this.promptConfig = DEFAULT_PROMPT;
    this.running = null;        // { id, aborted, finished: Promise } of the generation in progress
  }

  /** Entry point: one request message in, one or more response messages out (through post). */
  async handle(message) {
    const { type, id } = message || {};
    try {
      if (type === 'abort') return this._abort(id);
      if (type === 'load') return await this._load(message);
      if (type === 'unload') {
        await this._stopRunning();
        this.engine = this.tokenizer = this.retriever = null;
        return this.post({ type: 'unloaded', id });
      }
      if (!this.tokenizer) throw new Error('nothing is loaded yet: send a "load" message first');
      if (type === 'tokenize') {
        return this.post({ type: 'tokens', id, ...this.tokenizer.tokenize(message.text, { allowSpecial: !!message.allowSpecial }) });
      }
      if (type === 'retrieve') {
        if (!this.retriever) throw new Error('no retrieval index is loaded');
        const k = message.k ?? this.promptConfig.max_facts;
        return this.post({ type: 'hits', id, hits: this.retriever.search(message.query, k, message.lang ?? null) });
      }
      if (type === 'ask') return await this._ask(message);
      if (type === 'generate') return await this._generate(id, message.tokens, message);
      throw new Error(`unknown message type: ${type}`);
    } catch (err) {
      this.post({ type: 'error', id, message: String((err && err.message) || err) });
    }
  }

  async _load(message) {
    const { id } = message;
    await this._stopRunning();
    const t0 = now();
    const get = async (source, stage) =>
      typeof source === 'string'
        ? fetchBytes(this.fetch, source, (loaded, total) => this.post({ type: 'progress', id, stage, loaded, total }))
        : source;
    const [model, wasm, retrieval] = await Promise.all([
      get(message.model, 'model'),
      message.wasm ? get(message.wasm, 'wasm').catch(() => null) : null,   // no kernel -> plain JavaScript
      message.retrieval ? get(message.retrieval, 'retrieval') : null,
    ]);
    const fetchMs = now() - t0;
    const engine = await Engine.create(model, { wasm, contextLength: message.contextLength });
    if (!engine.tokenizerSpec) throw new Error('the model file does not embed a tokenizer');
    this.engine = engine;
    this.tokenizer = new Tokenizer(engine.tokenizerSpec);
    this.promptConfig = engine.prompt || DEFAULT_PROMPT;
    this.retriever = null;
    if (retrieval) {
      const index = retrieval instanceof ArrayBuffer || ArrayBuffer.isView(retrieval)
        ? JSON.parse(new TextDecoder().decode(retrieval)) : retrieval;
      this.retriever = new Retriever(index);
    }
    this.post({
      type: 'ready', id,
      info: {
        name: engine.name, format: engine.format, config: engine.config, prompt: this.promptConfig,
        kernel: engine.kernelName, kernelError: engine.kernelError, memoryBytes: engine.memoryBytes,
        contextLength: engine.contextLength, vocabSize: this.tokenizer.vocabSize,
        specialTokens: this.tokenizer.specialIds, facts: this.retriever ? this.retriever.docs.length : 0,
        fetchMs, initMs: now() - t0 - fetchMs,
      },
    });
  }

  async _ask(message) {
    const { id, question } = message;
    if (!this.engine) throw new Error('no model is loaded');
    const mode = message.mode || (this.retriever ? 'open' : 'closed');
    if (mode === 'open' && !this.retriever) throw new Error('open-book mode needs a retrieval index');
    const cfg = this.promptConfig;
    const hits = mode === 'open' ? this.retriever.search(question, message.k ?? cfg.max_facts, message.lang ?? null) : [];
    const { ids, used } = buildPrompt(this.tokenizer, question, hits.map((h) => h.text), this.engine.contextLength, cfg);
    this.post({ type: 'context', id, mode, hits, used });
    return this._generate(id, ids, {
      ...message,
      maxNewTokens: message.maxNewTokens ?? cfg.answer_reserve,
      stopTokens: [this.tokenizer.specialIds[SPECIALS.eos]],
    });
  }

  async _generate(id, promptTokens, message) {
    if (!this.engine) throw new Error('no model is loaded');
    // One generation at a time: this one takes the slot at once (so that a later request can find and
    // abort it), then waits for the one it replaces to wind down.
    const previous = this.running;
    const run = { id, aborted: false };
    let release;
    run.finished = new Promise((resolve) => (release = resolve));
    this.running = run;
    try {
      if (previous) {
        previous.aborted = true;
        await previous.finished;
      }
      const { engine, tokenizer } = this;
      if (!engine) throw new Error('the model was unloaded');
      this.post({ type: 'prompt', id, tokens: Array.from(promptTokens, (t) => ({ id: t, text: tokenizer.tokenText(t) })) });
      const sampling = message.sampling || {};
      const inspect = message.inspect || null;
      const decoder = tokenizer.streamDecoder();
      const tokens = [];
      let text = '', stopReason = 'length';
      const t0 = now();
      let t1 = null, sliceStart = t0;
      const it = engine.stream(promptTokens, {
        maxNewTokens: message.maxNewTokens ?? 128, stopTokens: message.stopTokens || [],
        temperature: sampling.temperature ?? 0, topK: sampling.topK ?? 0, topP: sampling.topP ?? 1,
        repetitionPenalty: sampling.repetitionPenalty ?? 1,
        random: sampling.seed == null ? Math.random : seededRandom(sampling.seed), inspect,
      });
      for (;;) {
        if (run.aborted) { stopReason = 'aborted'; break; }
        const r = it.next();
        if (t1 === null) t1 = now();
        if (r.done) { stopReason = r.value; break; }
        const step = r.value;
        const delta = step.stop ? '' : decoder.push(step.token);
        text += delta;
        tokens.push(step.token);
        const event = { type: 'token', id, index: step.index, token: step.token, text: tokenizer.tokenText(step.token), delta, stop: step.stop };
        const transfer = [];
        if (inspect) {
          event.prob = step.prob;
          if (step.top) event.top = step.top.map((t) => ({ ...t, text: tokenizer.tokenText(t.id) }));
          if (step.attention) { event.attention = step.attention; transfer.push(step.attention.data.buffer); }
          if (step.norms) event.norms = step.norms;
        }
        this.post(event, transfer);
        if (now() - sliceStart > this.sliceMs) {
          await this.yieldToEventLoop();
          sliceStart = now();
        }
      }
      const tail = decoder.push([], true);
      text += tail;
      if (t1 === null) t1 = now();
      const decodeMs = now() - t1;
      this.post({
        type: 'done', id, text: text.trim(), tokens,
        stats: {
          stopReason, promptTokens: promptTokens.length, reusedTokens: engine.reusedTokens, newTokens: tokens.length,
          prefillMs: t1 - t0, decodeMs, kernel: engine.kernelName,
          tokensPerSecond: tokens.length > 1 && decodeMs > 0 ? ((tokens.length - 1) * 1000) / decodeMs : 0,
        },
      });
    } finally {
      if (this.running === run) this.running = null;
      release();
    }
  }

  _abort(target) {
    if (this.running && (target == null || target === this.running.id)) this.running.aborted = true;
  }

  async _stopRunning() {
    if (this.running) {
      this.running.aborted = true;
      await this.running.finished;
    }
  }
}
