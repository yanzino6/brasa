// The worker protocol, driven in-process: Session is what worker.js runs, BrasaClient is the page side.
import test from 'node:test';
import assert from 'node:assert/strict';
import { Session } from '../session.js';
import { BrasaClient } from '../client.js';
import { Tokenizer } from '../tokenizer.js';
import { AnswerTrie, Retriever } from '../retrieval.js';
import { Engine } from '../engine.js';
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
  assert.equal(ready.info.constrained, true, 'answers are constrained to the knowledge base by default');
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
  assert.deepEqual(messages[0], { type: 'context', id: 2, mode: 'closed', hits: [], used: 0, constrained: true, constraint: 'kb' });
  assert.equal(messages[1].tokens[1].text, '<|q|>');
  assert.ok(!('prob' in messages[2]), 'no inspection data unless asked');
  assert.ok(messages.at(-1).stats.newTokens <= 4);
});

test('without a retrieval index the default mode is closed-book', async () => {
  const { session, messages } = harness();
  await session.handle({ type: 'load', id: 1, model: readBuffer('model.f16.bin') });
  assert.equal(messages.at(-1).info.kernel, 'js');
  assert.equal(messages.at(-1).info.format, 'f16');
  assert.equal(messages.at(-1).info.constrained, false);
  await session.handle({ type: 'ask', id: 2, question: 'oi', maxNewTokens: 2 });
  assert.equal(messages.find((m) => m.type === 'context').mode, 'closed');
  assert.equal(messages.find((m) => m.type === 'context').constrained, false, 'no index: nothing to constrain to');
  await session.handle({ type: 'ask', id: 5, question: 'oi', constrain: true });
  assert.match(messages.at(-1).message, /retrieval index/);
  await session.handle({ type: 'ask', id: 3, question: 'oi', mode: 'open' });
  assert.match(messages.at(-1).message, /retrieval index/);
  await session.handle({ type: 'retrieve', id: 4, query: 'oi' });
  assert.equal(messages.at(-1).type, 'error');
});

test('ask is constrained to the knowledge base by default; constrain: false decodes freely', async () => {
  const { session, messages } = harness();
  await session.handle({ type: 'load', id: 1, ...sources() });
  const index = readJson('retrieval.json');
  const ask = async (id, extra) => {
    messages.length = 0;
    await session.handle({ type: 'ask', id, question: 'Onde o Téo mora?', mode: 'closed', maxNewTokens: 120, ...extra });
    return { context: messages[0], tokens: messages.filter((m) => m.type === 'token'), done: messages.at(-1) };
  };
  /** The answer is an allowed one; when the context ran out first, it is the beginning of one. */
  const curated = (trie, done) => {
    if (done.stats.stopReason === 'stop') return trie.accepts(done.tokens.slice(0, -1));
    let state = trie.start();
    for (const t of done.tokens) {
      if (!trie.allowed(state).includes(t)) return false;
      state = trie.advance(state, t);
    }
    return true;
  };

  // the weights of the fixture are random, yet the answer is a curated sentence
  const constrained = await ask(2, {});
  assert.equal(constrained.context.constrained, true);
  assert.equal(constrained.done.stats.constrained, true);
  const any = AnswerTrie.fromIndex(index, tok, null);
  assert.ok(curated(any, constrained.done), constrained.done.text);
  assert.ok(constrained.tokens.every((t) => t.choices >= 1));
  assert.ok(constrained.tokens.some((t) => t.choices === 1) && constrained.tokens.some((t) => t.choices > 1));
  assert.equal(constrained.tokens.at(-1).stop, constrained.done.stats.stopReason === 'stop');

  // one language only (open-book: the retrieved facts are in that language too)
  for (const lang of ['pt', 'en']) {
    const run = await ask(3, { lang, mode: 'open' });
    assert.ok(curated(AnswerTrie.fromIndex(index, tok, lang), run.done), `${lang}: ${run.done.text}`);
    assert.ok(run.context.hits.length > 0 && run.context.hits.every((h) => h.lang === lang));
  }

  // inspection works on forced tokens too: the model's own distribution is still reported
  const inspected = await ask(4, { inspect: { top: 3 } });
  assert.deepEqual(inspected.done.tokens, constrained.done.tokens, 'inspecting does not change the answer');
  assert.ok(inspected.tokens.every((t) => t.top.length === 3 && t.prob >= 0 && t.prob <= 1));

  // free decoding stays available and is what it was
  const free = await ask(5, { constrain: false });
  assert.equal(free.context.constrained, false);
  assert.equal(free.done.stats.constrained, false);
  assert.ok(free.tokens.every((t) => !('choices' in t)));
  assert.ok(!curated(any, free.done), 'random weights do not write a curated sentence on their own');

  // sampling stays inside the knowledge base
  const sampled = await ask(6, { sampling: { temperature: 1.5, seed: 7 } });
  assert.ok(curated(any, sampled.done), sampled.done.text);
});

test('open-book answers are constrained to the retrieved facts by default', async () => {
  const { session, messages } = harness();
  await session.handle({ type: 'load', id: 1, ...sources() });
  const index = readJson('retrieval.json');
  const ask = async (id, extra) => {
    messages.length = 0;
    await session.handle({ type: 'ask', id, question: 'Onde o Téo mora?', maxNewTokens: 200, ...extra });
    return { context: messages[0], done: messages.at(-1) };
  };
  /** The tokens are an allowed answer of the trie, or (context exhausted) the beginning of one. */
  const follows = (trie, done) => {
    if (done.stats.stopReason === 'stop') return trie.accepts(done.tokens.slice(0, -1));
    let state = trie.start();
    for (const t of done.tokens) {
      if (!trie.allowed(state).includes(t)) return false;
      state = trie.advance(state, t);
    }
    return true;
  };
  const whole = AnswerTrie.fromIndex(index, tok, null);
  const fixed = new Set(index.answers.canonical.map((c) => c.text));

  let facts = 0;
  for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {   // random weights, sampled: any allowed answer can come out
    const run = await ask(10 + seed, { sampling: { temperature: 2, seed } });
    assert.equal(run.context.constraint, 'retrieved');
    assert.equal(run.done.stats.constraint, 'retrieved');
    const shown = run.context.hits.slice(0, run.context.used).map((h) => h.id);
    assert.ok(shown.length > 0 && shown.length < index.docs.length / 4);
    assert.ok(follows(AnswerTrie.fromIndex(index, tok, null, null, shown), run.done), `not made of the retrieved facts: ${run.done.text}`);
    assert.ok(follows(whole, run.done));
    if (!fixed.has(run.done.text)) facts++;
  }
  assert.ok(facts > 0, 'some answers are retrieved facts, not only fixed answers');

  // 'kb' allows every fact; closed-book has nothing retrieved and falls back to it; false is free
  const kb = await ask(30, { constrain: 'kb' });
  assert.equal(kb.context.constraint, 'kb');
  assert.equal(kb.done.stats.constraint, 'kb');
  const closed = await ask(31, { mode: 'closed', constrain: 'retrieved', maxNewTokens: 8 });
  assert.equal(closed.context.constraint, 'kb');
  const free = await ask(32, { constrain: false, maxNewTokens: 4 });
  assert.equal(free.context.constraint, null);
  assert.equal(free.done.stats.constraint, null);
  await session.handle({ type: 'ask', id: 33, question: 'oi', constrain: 'everything' });
  assert.match(messages.at(-1).message, /unknown constrain option/);

  // one language: only the retrieved facts of that language
  const pt = await ask(34, { lang: 'pt', sampling: { temperature: 2, seed: 9 } });
  const shown = pt.context.hits.slice(0, pt.context.used).map((h) => h.id);
  assert.ok(follows(AnswerTrie.fromIndex(index, tok, 'pt', null, shown), pt.done), pt.done.text);
});

test('the retriever\'s ranking is a prior on the choice among the retrieved facts', async () => {
  const { session, messages } = harness();
  await session.handle({ type: 'load', id: 1, ...sources() });
  const stored = readJson('retrieval.json').answers.prior;
  assert.deepEqual(stored, { kind: 'score', weight: 6, fixed: 0 }, 'the fixture index ships a prior');
  assert.deepEqual(messages.at(-1).info.prior, stored);
  const ask = async (id, extra) => {
    messages.length = 0;
    await session.handle({ type: 'ask', id, question: 'Quais linguagens de programação ele usa?', maxNewTokens: 12, ...extra });
    return { context: messages[0], done: messages.at(-1) };
  };
  const factOf = (run) => {   // the retrieved fact the answer starts with (in either language), or null
    const hit = run.context.hits.slice(0, run.context.used)
      .find((h) => session.retriever.docs.some((d) => d.id === h.id && d.text.startsWith(run.done.text)));
    return hit ? hit.id : null;
  };

  const byDefault = await ask(2, {});
  assert.deepEqual(byDefault.context.prior, stored, 'the default is the prior of the index');
  const off = await ask(3, { prior: false });
  assert.equal(off.context.prior, null);
  assert.equal((await ask(4, { prior: 0 })).context.prior, null);
  assert.deepEqual((await ask(5, { prior: 2.5 })).context.prior, { ...stored, weight: 2.5 });

  // strong enough, the prior makes the answer the first hit; with a prior on the fixed answers, a fixed answer
  const first = await ask(6, { prior: { kind: 'rank', weight: 500, fixed: -900 } });
  assert.deepEqual(first.context.prior, { kind: 'rank', weight: 500, fixed: -900 });
  assert.equal(factOf(first), first.context.hits[0].id);
  const fixed = await ask(7, { prior: { weight: 0, fixed: 900 }, maxNewTokens: 60 });
  const texts = readJson('retrieval.json').answers.canonical.map((c) => c.text);
  assert.ok(texts.includes(fixed.done.text), fixed.done.text);

  // only the 'retrieved' constraint has a ranking to lean on
  const kb = await ask(8, { constrain: 'kb', prior: 500 });
  assert.equal('prior' in kb.context, false);
  const free = await ask(9, { constrain: false, prior: 500 });
  assert.equal('prior' in free.context, false);
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
  assert.equal((await client.retrieve('contato e-mail', 1, null, { dense: false }))[0].id, 'contact');
  assert.equal((await client.retrieve('contato e-mail', 3)).length, 3);

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

test('the model ranks the facts too when the index carries their vectors', async () => {
  const { session, messages } = harness();
  await session.handle({ type: 'load', id: 1, ...sources() });
  assert.equal(messages.at(-1).info.dense, true);

  // the reference: the question embedded by a separate engine, then the fused search of the index
  const engine = await Engine.create(readBuffer('model.i8.bin'), { wasm: wasmBytes() });
  const retriever = new Retriever(readJson('retrieval.json'));
  const s = tok.specialIds;
  const question = 'Quais linguagens de programação ele usa?';
  const vector = engine.embed([s['<|bos|>'], s['<|q|>'], ...tok.encode(question), s['<|a|>']], retriever.dense.pooling);
  const fused = retriever.search(question, 3, null, vector), lexical = retriever.search(question, 3);
  assert.notDeepEqual(fused.map((h) => h.score), lexical.map((h) => h.score));

  messages.length = 0;
  await session.handle({ type: 'ask', id: 2, question });
  const context = messages.find((m) => m.type === 'context');
  assert.equal(context.retriever, 'bm25+dense');
  assert.deepEqual(context.hits, fused);
  const first = messages.at(-1);
  assert.equal(first.type, 'done');
  assert.equal(first.stats.reusedTokens, 0, 'the embedding pass overwrote the context');

  messages.length = 0;
  await session.handle({ type: 'ask', id: 3, question });   // the same question again: the same answer
  assert.deepEqual(messages.at(-1).tokens, first.tokens);
  assert.deepEqual(messages.find((m) => m.type === 'context').hits, fused);

  messages.length = 0;
  await session.handle({ type: 'ask', id: 4, question, dense: false });   // the lexical fallback
  assert.equal(messages.find((m) => m.type === 'context').retriever, 'bm25');
  assert.deepEqual(messages.find((m) => m.type === 'context').hits, lexical);

  await session.handle({ type: 'retrieve', id: 5, query: question, k: 3 });
  assert.equal(messages.at(-1).retriever, 'bm25+dense');
  assert.deepEqual(messages.at(-1).hits, fused);
  await session.handle({ type: 'retrieve', id: 6, query: question, k: 3, dense: false });
  assert.equal(messages.at(-1).retriever, 'bm25');
  assert.deepEqual(messages.at(-1).hits, lexical);

  // closed-book never retrieves, so it never embeds
  messages.length = 0;
  await session.handle({ type: 'ask', id: 7, question, mode: 'closed' });
  assert.equal('retriever' in messages.find((m) => m.type === 'context'), false);
});

test('an index without fact vectors keeps the lexical ranking', async () => {
  const { session, messages } = harness();
  const index = readJson('retrieval.json');
  delete index.dense;
  await session.handle({ type: 'load', id: 1, ...sources(), retrieval: index });
  assert.equal(messages.at(-1).info.dense, false);
  messages.length = 0;
  await session.handle({ type: 'ask', id: 2, question: 'Onde o Téo mora?' });
  const context = messages.find((m) => m.type === 'context');
  assert.equal(context.retriever, 'bm25');
  assert.deepEqual(context.hits, new Retriever(index).search('Onde o Téo mora?', context.hits.length));
});
