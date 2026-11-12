// The worker protocol, driven in-process: Session is what worker.js runs, BrasaClient is the page side.
import test from 'node:test';
import assert from 'node:assert/strict';
import { Session } from '../session.js';
import { BrasaClient } from '../client.js';
import { Tokenizer } from '../tokenizer.js';
import { readBuffer, readJson, wasmBytes } from './helpers.mjs';

const tok = new Tokenizer(readJson('tokenizer.json'));
const sources = () => ({ model: readBuffer('model.i8.bin'), wasm: wasmBytes(), retrieval: readJson('retrieval.json') });

/** A Session plus the list of everything it posted. */
function harness(deps) {
  const messages = [];
  const listeners = [];
  const session = new Session((m) => { messages.push(m); listeners.forEach((l) => l(m)); }, deps);
  return { session, messages, onMessage: (l) => listeners.push(l) };
}

/** An in-process stand-in for a Worker running worker.js. */
function fakeWorker(deps) {
  let listener = null;
  const session = new Session((message) => queueMicrotask(() => listener({ data: message })), deps);
  return {
    postMessage: (message) => queueMicrotask(() => session.handle(message)),
    addEventListener: (_type, l) => { listener = l; },
  };
}

test('requests before load are rejected', async () => {
  const { session, messages } = harness();
  await session.handle({ type: 'ask', id: 1, question: 'oi' });
  assert.equal(messages[0].type, 'error');
  assert.match(messages[0].message, /load/);
});

test('load reports the model, then tokenize and retrieve work', async () => {
  const { session, messages } = harness();
  await session.handle({ type: 'load', id: 1, ...sources() });
  const ready = messages.at(-1);
  assert.equal(ready.type, 'ready');
  assert.equal(ready.info.kernel, 'wasm-simd');
  assert.equal(ready.info.name, 'brasa-tiny');
  assert.equal(ready.info.facts, 60);
  assert.equal(ready.info.specialTokens['<|eos|>'], tok.specialIds['<|eos|>']);

  await session.handle({ type: 'tokenize', id: 2, text: 'Olá  mundo' });
  assert.equal(messages.at(-1).text, 'Olá mundo');
  assert.deepEqual(messages.at(-1).tokens.map((t) => t.id), tok.encode('Olá mundo'));

  await session.handle({ type: 'retrieve', id: 3, query: 'onde o teo mora', k: 2 });
  assert.equal(messages.at(-1).hits.length, 2);
  assert.equal(messages.at(-1).hits[0].id, 'location');

  await session.handle({ type: 'nonsense', id: 4 });
  assert.equal(messages.at(-1).type, 'error');
});

test('ask (open-book): context, prompt, tokens, done', async () => {
  const { session, messages } = harness();
  await session.handle({ type: 'load', id: 1, ...sources() });
  messages.length = 0;
  await session.handle({ type: 'ask', id: 2, question: 'Onde o Téo mora?', inspect: { top: 3, attention: true, norms: true } });
  const types = messages.map((m) => m.type);
  assert.deepEqual(types.slice(0, 2), ['context', 'prompt']);
  assert.equal(types.at(-1), 'done');
  assert.ok(messages.every((m) => m.id === 2));

  const [context, prompt] = messages;
  assert.equal(context.mode, 'open');
  assert.equal(context.hits[0].id, 'location');
  assert.ok(context.used >= 1 && context.used <= context.hits.length);
  const s = tok.specialIds;
  assert.equal(prompt.tokens[0].id, s['<|bos|>']);
  assert.equal(prompt.tokens[1].id, s['<|ctx|>']);
  assert.equal(prompt.tokens.at(-1).text, '<|a|>');

  const tokens = messages.filter((m) => m.type === 'token');
  const done = messages.at(-1);
  assert.deepEqual(tokens.map((t) => t.token), done.tokens);
  assert.deepEqual(tokens.map((t) => t.index), tokens.map((_, i) => i));
  assert.equal(tokens.map((t) => t.delta).join('').trim(), done.text);
  assert.equal(done.text, tok.decode(done.tokens).trim());
  assert.equal(done.stats.newTokens, tokens.length);
  assert.equal(done.stats.promptTokens, prompt.tokens.length);
  assert.ok(['stop', 'length'].includes(done.stats.stopReason));
  assert.ok(tokens.length <= 16, 'the answer budget is the answer_reserve of the model');
  for (const t of tokens) {
    assert.equal(t.top.length, 3);
    assert.equal(typeof t.top[0].text, 'string');
    assert.ok(t.prob > 0 && t.prob <= 1);
    assert.equal(t.attention.length, prompt.tokens.length + t.index);
    assert.equal(t.norms.length, 3);
  }
});

test('ask (closed-book) sends the question alone', async () => {
  const { session, messages } = harness();
  await session.handle({ type: 'load', id: 1, ...sources() });
  messages.length = 0;
  await session.handle({ type: 'ask', id: 2, question: 'Onde o Téo mora?', mode: 'closed', maxNewTokens: 4 });
  assert.deepEqual(messages[0], { type: 'context', id: 2, mode: 'closed', hits: [], used: 0 });
  assert.equal(messages[1].tokens[1].text, '<|q|>');
  assert.ok(!('prob' in messages[2]), 'no inspection data unless asked');
  assert.ok(messages.at(-1).stats.newTokens <= 4);
});

test('without a retrieval index the default mode is closed-book', async () => {
  const { session, messages } = harness();
  await session.handle({ type: 'load', id: 1, model: readBuffer('model.f16.bin') });
  assert.equal(messages.at(-1).info.kernel, 'js');
  assert.equal(messages.at(-1).info.format, 'f16');
  await session.handle({ type: 'ask', id: 2, question: 'oi', maxNewTokens: 2 });
  assert.equal(messages.find((m) => m.type === 'context').mode, 'closed');
  await session.handle({ type: 'ask', id: 3, question: 'oi', mode: 'open' });
  assert.match(messages.at(-1).message, /retrieval index/);
  await session.handle({ type: 'retrieve', id: 4, query: 'oi' });
  assert.equal(messages.at(-1).type, 'error');
});

test('generate continues raw tokens; sampling with a seed is reproducible', async () => {
  const { session, messages } = harness();
  await session.handle({ type: 'load', id: 1, ...sources() });
  const tokens = tok.encode('O Téo');
  const run = async (id, seed) => {
    await session.handle({ type: 'generate', id, tokens, maxNewTokens: 12, sampling: { temperature: 1.2, topK: 20, topP: 0.95, repetitionPenalty: 1.1, seed } });
    return messages.at(-1);
  };
  const a = await run(2, 42), b = await run(3, 42), c = await run(4, 43);
  assert.equal(a.type, 'done');
  assert.equal(a.tokens.length, 12);
  assert.deepEqual(a.tokens, b.tokens);
  assert.notDeepEqual(a.tokens, c.tokens);
  assert.equal(b.stats.reusedTokens, tokens.length - 1, 'the second run reuses the cached prompt');
});

test('abort stops a running generation', async () => {
  const { session, messages, onMessage } = harness({ sliceMs: 0 });
  await session.handle({ type: 'load', id: 1, ...sources() });
  onMessage((m) => { if (m.type === 'token' && m.index === 2) session.handle({ type: 'abort', id: 2 }); });
  await session.handle({ type: 'generate', id: 2, tokens: tok.encode('O Téo'), maxNewTokens: 60 });
  const done = messages.at(-1);
  assert.equal(done.type, 'done');
  assert.equal(done.stats.stopReason, 'aborted');
  assert.equal(done.tokens.length, 3);
});

test('a new generation replaces the running one', async () => {
  const { session, messages } = harness({ sliceMs: 0 });
  await session.handle({ type: 'load', id: 1, ...sources() });
  const tokens = tok.encode('O Téo');
  const first = session.handle({ type: 'generate', id: 2, tokens, maxNewTokens: 60 });
  const second = session.handle({ type: 'generate', id: 3, tokens, maxNewTokens: 5 });
  await Promise.all([first, second]);
  const done = messages.filter((m) => m.type === 'done');
  assert.deepEqual(done.map((m) => [m.id, m.stats.stopReason]), [[2, 'aborted'], [3, 'length']]);
});

test('load from URLs reports download progress', async () => {
  const files = { '/m.bin': readBuffer('model.i8.bin'), '/k.wasm': wasmBytes(), '/r.json': new TextEncoder().encode(JSON.stringify(readJson('retrieval.json'))).buffer };
  const fetchStub = async (url) => {
    if (!(url in files)) return new Response('missing', { status: 404 });
    return new Response(files[url], { headers: { 'content-length': String(files[url].byteLength) } });
  };
  const { session, messages } = harness({ fetch: fetchStub });
  await session.handle({ type: 'load', id: 1, model: '/m.bin', wasm: '/k.wasm', retrieval: '/r.json' });
  assert.equal(messages.at(-1).type, 'ready');
  assert.equal(messages.at(-1).info.kernel, 'wasm-simd');
  const progress = messages.filter((m) => m.type === 'progress' && m.stage === 'model');
  assert.ok(progress.length >= 1);
  assert.equal(progress.at(-1).loaded, files['/m.bin'].byteLength);
  assert.equal(progress.at(-1).total, files['/m.bin'].byteLength);

  // a missing kernel is not fatal, a missing model is
  await session.handle({ type: 'load', id: 2, model: '/m.bin', wasm: '/nope.wasm' });
  assert.equal(messages.at(-1).info.kernel, 'js');
  await session.handle({ type: 'load', id: 3, model: '/nope.bin' });
  assert.equal(messages.at(-1).type, 'error');
  assert.match(messages.at(-1).message, /404/);
});

test('BrasaClient turns the protocol into promises and callbacks', async () => {
  const client = new BrasaClient(fakeWorker());
  const info = await client.load(sources());
  assert.equal(info.kernel, 'wasm-simd');

  const { text, tokens } = await client.tokenize('Olá, tudo bem?');
  assert.equal(text, 'Olá, tudo bem?');
  assert.deepEqual(tokens.map((t) => t.id), tok.encode('Olá, tudo bem?'));
  assert.equal((await client.retrieve('contato e-mail', 1))[0].id, 'contact');

  const seen = { context: 0, prompt: 0, deltas: [] };
  const run = client.ask('Como falo com o Téo?', { inspect: { top: 2 } }, {
    onContext: () => seen.context++, onPrompt: () => seen.prompt++, onToken: (t) => seen.deltas.push(t.delta),
  });
  const done = await run.done;
  assert.equal(seen.context, 1);
  assert.equal(seen.prompt, 1);
  assert.equal(seen.deltas.join('').trim(), done.text);

  await assert.rejects(client.generate([], {}).done, /empty prompt/);
  await client.unload();
  await assert.rejects(client.tokenize('x'), /load/);
});

test('BrasaClient.abort', async () => {
  const client = new BrasaClient(fakeWorker({ sliceMs: 0 }));
  await client.load(sources());
  let run = null;
  run = client.generate(tok.encode('O Téo'), { maxNewTokens: 60 }, { onToken: (t) => { if (t.index === 1) run.abort(); } });
  const done = await run.done;
  assert.equal(done.stats.stopReason, 'aborted');
  assert.ok(done.tokens.length < 60);
});
