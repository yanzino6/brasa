// The JavaScript ports against the Python reference, on the committed fixture.
import test from 'node:test';
import assert from 'node:assert/strict';
import { runParity } from '../tools/parity.mjs';
import { FIXTURES, WASM_PATH } from './helpers.mjs';

test('engine, tokenizer, retriever, answer trie and prompt builder match the Python reference', async () => {
  const report = await runParity(FIXTURES, WASM_PATH);
  assert.deepEqual(Object.keys(report.engine).sort(), ['f16.js', 'f16.wasm-simd', 'i8.js', 'i8.wasm-simd']);
  for (const [name, r] of Object.entries(report.engine)) {
    assert.ok(r.ok, `${name}: ${JSON.stringify(r)}`);
    assert.ok(r.maxAbsLogitDiff < 1e-4, `${name} logits differ by ${r.maxAbsLogitDiff}`);
    // 6 prompts, each with the whole knowledge base allowed, with four facts only, and with a prior on those four
    assert.equal(r.constrainedIdentical, '18/18', `${name}: constrained decoding differs from Python`);
    assert.equal(r.constrainedWithPrior, 6);
  }
  assert.equal(report.text.tokenizer.mismatches, 0);
  assert.equal(report.text.features.mismatches, 0);
  assert.equal(report.text.retrieval.rankingMismatches, 0);
  assert.equal(report.text.retrieval.scoreMismatches, 0, 'scores must be bit-identical');
  assert.equal(report.text.prompt.mismatches, 0);
  assert.equal(report.text.trie.mismatches, 0);
  assert.equal(report.text.trie.cases, 6, 'both languages, each language, two subsets of the facts, one with priors');
  assert.ok(report.text.trie.walks >= 60 && report.text.trie.steps > 2000 && report.text.trie.biasSteps > 300);
  assert.ok(report.text.tokenizer.cases > 100 && report.text.retrieval.cases > 100);
});
