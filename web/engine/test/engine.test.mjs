// Behaviour of the engine API on the tiny fixture model (numerical parity with PyTorch is in parity.test.mjs).
import test from 'node:test';
import assert from 'node:assert/strict';
import { Engine, argmax, chooseAllowed, chooseToken, filterLogits, parseModel, probabilityOf, sampleFrom, seededRandom, topTokens } from '../engine.js';
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

// ------------------------------------------------------------------------------ constrained decoding

/** A constraint that allows exactly these token sequences, each followed by eos. */
function listConstraint(sequences, eos) {
  const allowedAfter = (prefix) => {
    const out = new Set();
    for (const seq of sequences) {
      if (prefix.length <= seq.length && prefix.every((t, i) => t === seq[i])) out.add(prefix.length === seq.length ? eos : seq[prefix.length]);
    }
    return [...out].sort((a, b) => a - b);
  };
  return { eos, start: () => [], allowed: allowedAfter, advance: (state, token) => [...state, token] };
}

test('chooseAllowed: greedy among the allowed ids, ties to the lower id, sampling stays inside', () => {
  const logits = Float32Array.from([0.1, 5, 2, 2, -1, 4]);
  assert.equal(chooseAllowed(logits, [0, 2, 3, 4]), 2, 'the best allowed token, the lower id on a tie');
  assert.equal(chooseAllowed(logits, [4]), 4);
  assert.equal(chooseAllowed(logits, [1, 5], { temperature: 0 }), 1);
  const random = seededRandom(3);
  const seen = new Set();
  for (let i = 0; i < 200; i++) seen.add(chooseAllowed(logits, [0, 3, 4], { temperature: 2, random }));
  assert.deepEqual([...seen].sort(), [0, 3, 4], 'every allowed token can be drawn, nothing else');
  assert.equal(chooseAllowed(logits, [0, 3], { temperature: 0, repetitionPenalty: 100 }, [3]), 0);

  // a bias (one number per allowed token, in nats) is added to the logits before the choice
  assert.equal(chooseAllowed(logits, [1, 5], {}, null, [0, 0]), 1);
  assert.equal(chooseAllowed(logits, [1, 5], {}, null, [-1.5, 0]), 5, '5 - 1.5 < 4');
  assert.equal(chooseAllowed(logits, [1, 5], {}, null, [-1, 0]), 1, 'a tie goes to the lower id');
  assert.equal(chooseAllowed(logits, [0, 4], {}, null, Float64Array.from([-2, 0])), 4);
  const biased = new Set();
  for (let i = 0; i < 200; i++) biased.add(chooseAllowed(logits, [0, 3, 4], { temperature: 1, random }, null, [0, -60, 0]));
  assert.deepEqual([...biased].sort(), [0, 4], 'sampling sees the biased logits');
});

test('stream adds the bias of a constraint that carries priors, and only then', async () => {
  const engine = await make();
  const eos = 500, sequences = [[7, 8, 9], [30, 31], [41]];
  const plain = listConstraint(sequences, eos);
  const first = engine.generate(prompt, { constraint: plain }).tokens[0];
  for (const target of [7, 30, 41]) {
    const constraint = {
      ...plain, hasPrior: true,
      bias: (state) => Float64Array.from(plain.allowed(state), (t) => (state.length === 0 && t !== target ? -80 : 0)),
    };
    assert.equal(engine.generate(prompt, { constraint }).tokens[0], target, 'a strong prior decides');
  }
  const ignored = { ...plain, hasPrior: false, bias: () => { throw new Error('not consulted'); } };
  assert.equal(engine.generate(prompt, { constraint: ignored }).tokens[0], first);
});

test('stream with a constraint only produces allowed sequences and skips the head on forced tokens', async () => {
  const engine = await make({ wasm: wasmBytes() });
  const eos = 500;
  const sequences = [[7, 8, 9, 10], [7, 8, 20], [30, 31, 32, 33, 34], [30, 40]];
  const constraint = listConstraint(sequences, eos);

  const steps = [...engine.stream(prompt, { constraint })];
  const tokens = steps.map((s) => s.token);
  assert.equal(tokens.at(-1), eos);
  assert.ok(steps.at(-1).stop, 'the eos of the constraint ends the generation without being a stop token');
  assert.ok(sequences.some((seq) => seq.length === tokens.length - 1 && seq.every((t, i) => t === tokens[i])), `${tokens}`);
  assert.equal(steps[0].choices, 2);
  assert.ok(steps.slice(1).every((s) => s.choices >= 1 && s.choices <= 2));

  // the decisions are the model's: at every branch the chosen token has the larger logit
  engine.reset();
  const logits = engine.prefill(prompt);
  assert.equal(tokens[0], logits[7] >= logits[30] ? 7 : 30);

  // forced tokens are fed without the output head; asking for inspection computes it, same answer
  const inspected = [...engine.stream(prompt, { constraint, inspect: { top: 2 } })];
  assert.deepEqual(inspected.map((s) => s.token), tokens);
  assert.ok(inspected.every((s) => s.top.length === 2 && s.prob > 0));

  // the budget still applies, and the free path is untouched
  assert.equal(engine.generate(prompt, { constraint, maxNewTokens: 2 }).stopReason, 'length');
  const free = engine.generate(prompt, { maxNewTokens: 6 }).tokens;
  assert.deepEqual(free, vectors.cases[0].greedy.slice(0, 6));
  assert.deepEqual(engine.generate(prompt, { maxNewTokens: 6, constraint: null }).tokens, free);
});

test('a constraint with one allowed answer writes it whatever the weights say', async () => {
  const engine = await make();
  const only = [3, 1, 4, 1, 5, 9, 2, 6];
  const { tokens, stopReason } = engine.generate(prompt, { constraint: listConstraint([only], 499) });
  assert.deepEqual(tokens, [...only, 499]);
  assert.equal(stopReason, 'stop');
});

test('embed reads the model as an encoder and leaves the context empty', async () => {
  const engine = await make({ wasm: wasmBytes() });
  const before = engine.generate(prompt, { maxNewTokens: 12 }).tokens;
  for (const c of vectors.embed) {
    const v = engine.embed(c.tokens, c.pooling);
    assert.equal(v.length, engine.config.d_model);
    assert.ok(Math.abs(Math.hypot(...v) - 1) < 1e-5, 'a unit vector');
    assert.ok(v.every((x, i) => Math.abs(x - c.vector[i]) < 1e-5), `${c.pooling} pooling matches PyTorch`);
    assert.equal(engine.position, 0);
  }
  const mean = engine.embed(vectors.embed[0].tokens, 'mean'), last = engine.embed(vectors.embed[0].tokens, 'last');
  assert.ok(mean.some((x, i) => Math.abs(x - last[i]) > 1e-3), 'the two poolings differ');
  assert.deepEqual(Array.from(engine.embed(vectors.embed[0].tokens)), Array.from(mean), 'mean is the default');

  const after = engine.generate(prompt, { maxNewTokens: 12 });
  assert.deepEqual(after.tokens, before, 'generation is unaffected');
  assert.equal(after.reusedTokens, 0, 'the prompt is fed again from the start');

  assert.throws(() => engine.embed([]), /nothing to embed/);
  assert.throws(() => engine.embed([1, 2], 'max'), /unknown pooling/);
  assert.throws(() => engine.embed(new Array(engine.contextLength + 1).fill(1)), /exceeds the context length/);
});
