// The message protocol of the Brasa worker, independent of where it runs.
//
// A Session owns the engine, the tokenizer and the retriever, receives request messages through
// handle() and answers through the `post` callback given to the constructor. worker.js wires it to a
// Web Worker; the tests drive it directly in Node. worker.d.ts documents every message.
//
// A question goes through the same steps as in training and evaluation (src/brasa/evaluate.py):
//   retrieve facts -> build the prompt -> generate until <|eos|>
// and every step is reported, so a page can show what the model was given and what it did with it.
//
// When the retrieval index carries the learned half (index.dense), the same model first reads the question
// as an encoder (one short forward pass) and the facts are ranked by BM25 fused with the cosine to their
// precomputed vectors. `dense: false` in ask / retrieve keeps the lexical ranking alone.
//
// By default the answer is constrained (ask: constrain, on whenever a retrieval index is loaded): the
// model chooses among curated sentences instead of writing freely, so it cannot blend two facts or
// invent a number. Open-book, the choice is among the facts it was given (constrain: 'retrieved'), so
// every answer is one of the sentences shown as context or one of the fixed answers (a greeting, "I
// can't answer that"); closed-book, or with constrain: 'kb', it is among all the facts of the knowledge
// base. `constrain: false` gives the model's free generation.
//
// With 'retrieved', the retriever's ranking also counts as a prior on the model's choice (ask: prior): a
// fact the retriever put lower has to be that much more likely for the model to be chosen. The default is
// the prior stored in the index (index.answers.prior, chosen on validation questions); `prior: false`
// leaves the choice to the model alone, and a number sets the strength.

import { Engine, seededRandom } from './engine.js';
import { Tokenizer } from './tokenizer.js';
import { AnswerTrie, Retriever, rankingPriors } from './retrieval.js';
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
    this.tries = new Map();     // language ('pt', 'en', 'any') -> AnswerTrie, built on first use
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
        this.tries.clear();
        return this.post({ type: 'unloaded', id });
      }
      if (!this.tokenizer) throw new Error('nothing is loaded yet: send a "load" message first');
      if (type === 'tokenize') {
        return this.post({ type: 'tokens', id, ...this.tokenizer.tokenize(message.text, { allowSpecial: !!message.allowSpecial }) });
      }
      if (type === 'retrieve') {
        if (!this.retriever) throw new Error('no retrieval index is loaded');
        const k = message.k ?? this.promptConfig.max_facts;
        const vector = await this._questionVector(message.query, message.dense);
        return this.post({
          type: 'hits', id, retriever: vector ? 'bm25+dense' : 'bm25',
          hits: this.retriever.search(message.query, k, message.lang ?? null, vector),
        });
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
    this.tries.clear();
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
        constrained: !!this.retriever, dense: !!(this.retriever && this.retriever.dense),
        prior: this.retriever ? this._prior(undefined) : null,
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
    const vector = mode === 'open' ? await this._questionVector(question, message.dense) : null;
    const hits = mode === 'open' ? this.retriever.search(question, message.k ?? cfg.max_facts, message.lang ?? null, vector) : [];
    const { ids, used } = buildPrompt(this.tokenizer, question, hits.map((h) => h.text), this.engine.contextLength, cfg);
    let constrain = message.constrain ?? !!this.retriever;
    if (constrain && !this.retriever) throw new Error('constrained decoding needs a retrieval index');
    if (constrain === true || (constrain === 'retrieved' && mode !== 'open')) constrain = mode === 'open' ? 'retrieved' : 'kb';
    if (constrain && constrain !== 'retrieved' && constrain !== 'kb') throw new Error(`unknown constrain option: ${constrain}`);
    const lang = message.lang ?? null;
    const shown = hits.slice(0, used);
    const prior = constrain === 'retrieved' ? this._prior(message.prior) : null;
    const constraint = !constrain ? null
      : constrain === 'kb' ? this._trie(lang)
        : AnswerTrie.fromIndex(this.retriever.index, this.tokenizer, lang, null, shown.map((h) => h.id),
          prior ? rankingPriors(shown, prior.weight, prior.kind) : null, prior ? prior.fixed : 0);
    const context = { type: 'context', id, mode, hits, used, constrained: !!constraint, constraint: constrain || null };
    if (mode === 'open') context.retriever = vector ? 'bm25+dense' : 'bm25';   // how the facts were ranked
    if (constrain === 'retrieved') context.prior = prior;                      // how much that ranking weighs in the answer
    this.post(context);
    return this._generate(id, ids, {
      ...message,
      maxNewTokens: message.maxNewTokens ?? cfg.answer_reserve,
      stopTokens: [this.tokenizer.specialIds[SPECIALS.eos]],
      constraint,
      constraintKind: constrain || null,
    });
  }

  /**
   * The question as the encoder sees it, <|bos|><|q|> question <|a|> (src/brasa/encoder.py question_ids),
   * pooled into one vector by the model; null when the index has no dense half or `dense` is false.
   * A generation in progress is stopped first: the pass overwrites the engine's context.
   */
  async _questionVector(question, dense = true) {
    const section = this.retriever && this.retriever.dense;
    if (!section || dense === false || !this.engine) return null;
    await this._stopRunning();
    const s = this.tokenizer.specialIds;
    const room = Math.min(section.maxTokens, this.engine.contextLength) - 3;
    const ids = [s[SPECIALS.bos], s[SPECIALS.q], ...this.tokenizer.encode(question).slice(0, room), s[SPECIALS.a]];
    return this.engine.embed(ids, section.pooling);
  }

  /**
   * The prior of an `ask`: { kind, weight, fixed } or null (the model alone chooses among the retrieved facts).
   * option undefined: the index's own (index.answers.prior); false or 0: none; a number: that weight with the
   * index's kind; an object: merged over the index's.
   */
  _prior(option) {
    const stored = (this.retriever && this.retriever.index.answers && this.retriever.index.answers.prior) || {};
    const base = { kind: stored.kind || 'score', weight: stored.weight || 0, fixed: stored.fixed || 0 };
    if (option === false || option === 0 || option === null) return null;
    const prior = option === undefined || option === true ? base
      : typeof option === 'number' ? { ...base, weight: option } : { ...base, ...option };
    return prior.weight || prior.fixed ? prior : null;
  }

  /** The trie of the allowed answers for one language (null: both), built once from the index. */
  _trie(lang) {
    const key = lang ?? 'any';
    if (!this.tries.has(key)) this.tries.set(key, AnswerTrie.fromIndex(this.retriever.index, this.tokenizer, lang));
    return this.tries.get(key);
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
        constraint: message.constraint || null,
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
        if (step.choices !== undefined) event.choices = step.choices;
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
          prefillMs: t1 - t0, decodeMs, kernel: engine.kernelName, constrained: !!message.constraint,
          constraint: message.constraint ? (message.constraintKind || 'custom') : null,
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
