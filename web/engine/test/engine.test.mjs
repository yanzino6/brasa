// Behaviour of the engine API on the tiny fixture model (numerical parity with PyTorch is in parity.test.mjs).
import test from 'node:test';
import assert from 'node:assert/strict';
import { Engine, argmax, chooseToken, filterLogits, parseModel, probabilityOf, sampleFrom, seededRandom, topTokens } from '../engine.js';
import { readBuffer, readJson, wasmBytes } from './helpers.mjs';

const vectors = readJson('testvec.i8.json');
const prompt = vectors.cases[0].prompt;
const make = (opts = {}) => Engine.create(readBuffer('model.i8.bin'), opts);

test('the header carries the configuration, the prompt format and the tokenizer', async () => {
  const { header } = parseModel(readBuffer('model.i8.bin'));
  assert.equal(header.format, 'i8');
  assert.equal(header.name, 'brasa-tiny');
  assert.equal(header.config.n_layer, 2);
  assert.equal(header.tokenizer.type, 'bpe');
  assert.deepEqual(Object.keys(header.prompt).sort(), ['answer_reserve', 'max_facts', 'max_question_tokens']);
  assert.throws(() => parseModel(new ArrayBuffer(64)), /not a TGPT/);
});

test('both kernels produce the same greedy tokens', async () => {
  const js = await make(), simd = await make({ wasm: wasmBytes() });
  assert.equal(js.kernelName, 'js');
  assert.equal(simd.kernelName, 'wasm-simd');
  const a = js.generate(prompt, { maxNewTokens: 20 }), b = simd.generate(prompt, { maxNewTokens: 20 });
  assert.deepEqual(a.tokens, b.tokens);
  assert.deepEqual(a.tokens, vectors.cases[0].greedy.slice(0, 20));
  assert.equal(a.stopReason, 'length');
});

test('a broken WASM kernel falls back to JavaScript', async () => {
  const engine = await make({ wasm: new Uint8Array([0, 1, 2, 3]).buffer });
  assert.equal(engine.kernelName, 'js');
  assert.match(engine.kernelError, /./);
  assert.deepEqual(engine.generate(prompt, { maxNewTokens: 8 }).tokens, vectors.cases[0].greedy.slice(0, 8));
});

test('feed() reuses the cached prefix and gives the same logits', async () => {
  const engine = await make();
  const first = Float32Array.from(engine.feed(prompt));
  assert.equal(engine.reusedTokens, 0);
  const again = Float32Array.from(engine.feed(prompt));
  assert.equal(engine.reusedTokens, prompt.length - 1);
  assert.deepEqual(again, first);

  const longer = [...prompt, argmax(first)];
  engine.feed(longer);
  assert.equal(engine.reusedTokens, prompt.length, 'extending the sequence only computes the new token');
  const fresh = await make();
  assert.deepEqual(Float32Array.from(engine.logits), Float32Array.from(fresh.feed(longer)));

  const changed = [...prompt];
  changed[3] = (changed[3] + 1) % 256;
  engine.feed(changed);
  assert.equal(engine.reusedTokens, 3);
  assert.deepEqual(Float32Array.from(engine.logits), Float32Array.from(fresh.feed(changed)));
});

test('stepping one token at a time equals one generate call', async () => {
  const engine = await make();
  const whole = engine.generate(prompt, { maxNewTokens: 10 }).tokens;
  const stepper = await make();
  const sequence = [...prompt];
  for (let i = 0; i < 10; i++) sequence.push(stepper.generate(sequence, { maxNewTokens: 1 }).tokens[0]);
  assert.deepEqual(sequence.slice(prompt.length), whole);
});

test('rewind, context limit and invalid tokens', async () => {
  const engine = await make({ contextLength: 12 });
  assert.equal(engine.contextLength, 12);
  engine.prefill(prompt.slice(0, 12));
  assert.equal(engine.position, 12);
  assert.throws(() => engine.step(1), /context is full/);
  engine.rewind(5);
  assert.equal(engine.position, 5);
  assert.throws(() => engine.rewind(6), RangeError);
  assert.throws(() => engine.step(-1), RangeError);
  assert.throws(() => engine.step(engine.config.vocab_size), RangeError);
  assert.throws(() => engine.feed(prompt.slice(0, 13)), /exceeds the context length/);
  // generation stops when the context is full
  const r = engine.generate(prompt.slice(0, 9), { maxNewTokens: 50 });
  assert.equal(r.tokens.length, 3);
  assert.equal(r.stopReason, 'length');
});

test('stop tokens end the generation and are reported', async () => {
  const engine = await make();
  const greedy = vectors.cases[0].greedy;
  const r = engine.generate(prompt, { maxNewTokens: 20, stopTokens: [greedy[4]] });
  const cut = greedy.indexOf(greedy[4]);
  assert.deepEqual(r.tokens, greedy.slice(0, cut + 1));
  assert.equal(r.stopReason, 'stop');
});

test('onToken can abort', async () => {
  const engine = await make();
  const r = engine.generate(prompt, { maxNewTokens: 20, onToken: (s) => s.index < 2 });
  assert.equal(r.tokens.length, 3);
  assert.equal(r.stopReason, 'aborted');
});

test('inspection: top-k probabilities, attention and residual norms for every step', async () => {
  const engine = await make({ wasm: wasmBytes() });
  const steps = [...engine.stream(prompt, { maxNewTokens: 6, inspect: { top: 5, attention: true, norms: true } })];
  assert.equal(steps.length, 6);
  steps.forEach((s, i) => {
    assert.equal(s.top.length, 5);
    assert.equal(s.top[0].id, s.token, 'greedy picks the most likely token');
    assert.ok(Math.abs(s.top[0].prob - s.prob) < 1e-12);
    for (let j = 1; j < 5; j++) assert.ok(s.top[j - 1].prob >= s.top[j].prob);
    assert.ok(s.top.reduce((a, t) => a + t.prob, 0) <= 1 + 1e-9);
    const { layers, heads, length, data } = s.attention;
    assert.deepEqual([layers, heads, length], [2, 4, prompt.length + i]);
    assert.equal(data.length, layers * heads * length);
    for (let r = 0; r < layers * heads; r++) {
      let sum = 0;
      for (let t = 0; t < length; t++) sum += data[r * length + t];
      assert.ok(Math.abs(sum - 1) < 1e-5, `attention row ${r} sums to ${sum}`);
    }
    assert.equal(s.norms.length, 3);
    assert.ok(s.norms.every((n) => n > 0));
  });
});

test('inspection is off by default and costs nothing', async () => {
  const engine = await make();
  const [step] = [...engine.stream(prompt, { maxNewTokens: 1 })];
  assert.deepEqual(Object.keys(step).sort(), ['index', 'stop', 'token']);
  assert.equal(engine.attention(), null);
  assert.equal(engine.residualNorms(), null);
  engine.setCapture({ attention: true });
  assert.equal(engine.attention(), null, 'nothing captured until a step runs');
  engine.feed(prompt);
  assert.equal(engine.attention().length, prompt.length);
});

test('sampling: seeded runs repeat, different seeds differ, top-k 1 is greedy', async () => {
  const engine = await make();
  const run = (seed, extra = {}) => engine.generate(prompt, { maxNewTokens: 24, temperature: 1.5, random: seededRandom(seed), ...extra }).tokens;
  assert.deepEqual(run(7), run(7));
  assert.notDeepEqual(run(7), run(8));
  assert.deepEqual(run(7, { topK: 1 }), vectors.cases[0].greedy.slice(0, 24));
});

test('filterLogits: temperature, top-k with ties, top-p, repetition penalty', () => {
  const logits = Float32Array.from([2, 1, 1, 0, -1]);
  const all = filterLogits(logits);
  assert.deepEqual(Array.from(all.ids), [0, 1, 2, 3, 4]);
  assert.ok(Math.abs(all.probs.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(Math.abs(all.probs[0] - probabilityOf(logits, 0)) < 1e-12);

  assert.deepEqual(Array.from(filterLogits(logits, { topK: 2 }).ids), [0, 1, 2], 'ties with the k-th logit are kept');
  assert.deepEqual(Array.from(filterLogits(logits, { topK: 1 }).ids), [0]);
  assert.deepEqual(Array.from(filterLogits(logits, { topP: 0.5 }).ids), [0], 'the token that crosses top-p is kept');
  assert.deepEqual(Array.from(filterLogits(logits, { topP: 0.6 }).ids), [0, 1]);

  const sharp = filterLogits(logits, { temperature: 0.1 }), flat = filterLogits(logits, { temperature: 10 });
  assert.ok(sharp.probs[0] > 0.99 && flat.probs[0] < 0.3);

  const penalised = filterLogits(logits, { repetitionPenalty: 4 }, [0, 4, 4]);
  assert.deepEqual(Array.from(penalised.ids), [1, 2, 0, 3, 4], 'logit 2 becomes 0.5, logit -1 becomes -4');
  assert.equal(chooseToken(logits, { repetitionPenalty: 4 }, [0]), 1, 'greedy also honours the penalty');
  assert.equal(chooseToken(logits), 0);
});

test('sampleFrom follows the probabilities', () => {
  const dist = { ids: Int32Array.from([5, 9]), probs: Float64Array.from([0.25, 0.75]) };
  const random = seededRandom(1);
  let nine = 0;
  for (let i = 0; i < 4000; i++) nine += sampleFrom(dist, random) === 9;
  assert.ok(Math.abs(nine / 4000 - 0.75) < 0.03);
});

test('topTokens orders by probability', () => {
  const top = topTokens(Float32Array.from([0, 3, 1, 3, 2]), 3);
  assert.deepEqual(top.map((t) => t.id), [1, 3, 4]);
  assert.ok(top[0].prob === top[1].prob && top[1].prob > top[2].prob);
  assert.equal(topTokens(Float32Array.from([1, 2]), 5).length, 2);
});
