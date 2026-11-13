// Parity of the JavaScript engine, tokenizer, retriever, answer trie and prompt builder against the Python
// reference.
//
//   node web/engine/tools/parity.mjs <export-dir> [--wasm web/wasm/kernel.wasm] [--quiet]
//
// <export-dir> is written by `python -m brasa.export` (or tests/fixtures/tiny): model.<fmt>.bin,
// testvec.<fmt>.json, tokenizer.json, retrieval.json, parity.json. Prints one line per check, then a JSON
// report on the last line, and exits with status 1 if anything differs.
import fs from 'node:fs';
import path from 'node:path';
import { Engine, filterLogits, topTokens } from '../engine.js';
import { Tokenizer } from '../tokenizer.js';
import { AnswerTrie, Retriever, queryFeatures } from '../retrieval.js';
import { buildPrompt } from '../prompt.js';

import { SPECIALS } from '../prompt.js';

const LOGIT_TOL = 1e-3, ATTENTION_TOL = 1e-4, NORM_REL_TOL = 1e-4, PROB_TOL = 5e-5;
// A text vector is a unit vector: its components are compared one by one. A fused score computed from the
// engine's own vector may differ from Python's in the fifth decimal (the vector is rounded to int16), which
// can swap two facts that are tied to that precision: such a swap is not counted as a mismatch.
const EMBED_TOL = 1e-4, FUSED_TOL = 1e-4;

const toArrayBuffer = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const sameArray = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

export async function checkEngine(dir, fmt, wasm) {
  const model = toArrayBuffer(fs.readFileSync(path.join(dir, `model.${fmt}.bin`)));
  const vec = JSON.parse(fs.readFileSync(path.join(dir, `testvec.${fmt}.json`), 'utf8'));
  const engine = await Engine.create(model, wasm ? { wasm } : {});
  if (wasm && engine.kernelName !== 'wasm-simd') throw new Error(`wasm kernel did not load: ${engine.kernelError}`);
  const r = { kernel: engine.kernelName, maxAbsLogitDiff: 0, maxAbsLogit: 0, greedyIdentical: 0, cases: vec.cases.length };

  for (const c of vec.cases) {
    if (c.last_logits) {
      engine.reset();
      const logits = engine.prefill(c.prompt);
      for (let i = 0; i < logits.length; i++) {
        r.maxAbsLogitDiff = Math.max(r.maxAbsLogitDiff, Math.abs(logits[i] - c.last_logits[i]));
        r.maxAbsLogit = Math.max(r.maxAbsLogit, Math.abs(c.last_logits[i]));
      }
    }
    const { tokens } = engine.generate(c.prompt, { maxNewTokens: c.greedy.length });
    if (sameArray(tokens, c.greedy)) r.greedyIdentical++;
  }

  // decoding constrained to the knowledge base: the same tokens as constrained_answers() in Python
  if (vec.constrained) {
    const tok = new Tokenizer(JSON.parse(fs.readFileSync(path.join(dir, 'tokenizer.json'), 'utf8')));
    const index = JSON.parse(fs.readFileSync(path.join(dir, 'retrieval.json'), 'utf8'));
    const tries = new Map();
    let same = 0;
    for (const c of vec.constrained) {
      // "priors": the retrieved facts carry a prior (the retriever's ranking), added to the logits at every choice
      const key = `${c.lang ?? 'any'}|${c.only ? c.only.join(',') : ''}|${c.priors ? JSON.stringify([c.priors, c.fixed_prior]) : ''}`;
      if (!tries.has(key)) tries.set(key, AnswerTrie.fromIndex(index, tok, c.lang, null, c.only ?? null, c.priors ?? null, c.fixed_prior ?? 0));
      const trie = tries.get(key);
      const { tokens, stopReason } = engine.generate(c.prompt, { maxNewTokens: c.max_new_tokens, constraint: trie });
      const ended = tokens.at(-1) === trie.eos;
      if (sameArray(ended ? tokens.slice(0, -1) : tokens, c.tokens) && ended === (stopReason === 'stop')) same++;
    }
    r.constrainedIdentical = same;
    r.constrainedCases = vec.constrained.length;
    r.constrainedWithPrior = vec.constrained.filter((c) => c.priors).length;
  }

  // the model read as an encoder: pooled, normalised final hidden states of a few questions and facts
  if (vec.embed) {
    r.maxAbsEmbedDiff = 0;
    for (const c of vec.embed) {
      const got = engine.embed(c.tokens, c.pooling);
      for (let i = 0; i < got.length; i++) r.maxAbsEmbedDiff = Math.max(r.maxAbsEmbedDiff, Math.abs(got[i] - c.vector[i]));
      if (got.length !== c.vector.length) r.maxAbsEmbedDiff = Infinity;
    }
    r.embedCases = vec.embed.length;
  }

  // end to end: the question embedded by THIS engine, then the fused search of the index. Only for the
  // weights the fact vectors were computed with (index.dense.weights).
  const fixturesPath = path.join(dir, 'parity.json');
  const fixtures = fs.existsSync(fixturesPath) ? JSON.parse(fs.readFileSync(fixturesPath, 'utf8')) : {};
  if (fixtures.dense) {
    const index = JSON.parse(fs.readFileSync(path.join(dir, 'retrieval.json'), 'utf8'));
    if (index.dense && index.dense.weights === fmt) {
      const tok = new Tokenizer(JSON.parse(fs.readFileSync(path.join(dir, 'tokenizer.json'), 'utf8')));
      const retriever = new Retriever(index);
      const s = tok.specialIds;
      let same = 0, total = 0, bad = 0, maxDiff = 0;
      for (const c of fixtures.dense) {
        const ids = [s[SPECIALS.bos], s[SPECIALS.q], ...tok.encode(c.q).slice(0, retriever.dense.maxTokens - 3), s[SPECIALS.a]];
        const vector = engine.embed(ids, retriever.dense.pooling);
        for (const search of c.searches) {
          const got = retriever.search(c.q, search.k, search.lang, vector);
          total++;
          let identical = got.length === search.hits.length, close = identical;
          for (let i = 0; identical && i < got.length; i++) {
            const diff = Math.abs(got[i].score - search.hits[i].score);
            maxDiff = Math.max(maxDiff, diff);
            if (got[i].doc !== search.hits[i].doc) identical = false;
            if (diff > FUSED_TOL) close = false;
          }
          if (identical) same++;
          if (!close) bad++;
        }
      }
      r.denseSearches = `${same}/${total}`;
      r.denseScoreMismatches = bad;
      r.maxDenseScoreDiff = Number(maxDiff.toExponential(2));
    }
  }

  // inspection: attention of every layer and head, residual norms, top-5 probabilities
  const ins = vec.inspect;
  engine.reset();
  engine.setCapture({ attention: true, norms: true });
  const logits = engine.prefill(ins.prompt);
  const att = engine.attention(), norms = engine.residualNorms();
  let attDiff = 0, rowSumDiff = 0, normDiff = 0;
  ins.attention.forEach((layer, l) => layer.forEach((head, h) => {
    let sum = 0;
    head.forEach((w, t) => {
      const got = att.data[(l * att.heads + h) * att.length + t];
      attDiff = Math.max(attDiff, Math.abs(got - w));
      sum += got;
    });
    rowSumDiff = Math.max(rowSumDiff, Math.abs(sum - 1));
  }));
  ins.resid_norms.forEach((n, i) => { normDiff = Math.max(normDiff, Math.abs(norms[i] - n) / n); });
  const top = topTokens(logits, ins.top.length);
  r.attentionShape = [att.layers, att.heads, att.length];
  r.maxAbsAttentionDiff = attDiff;
  r.maxAttentionRowSumError = rowSumDiff;
  r.maxRelNormDiff = normDiff;
  r.topIdsIdentical = sameArray(top.map((t) => t.id), ins.top.map((t) => t.id));
  r.maxTopProbDiff = Math.max(...top.map((t, i) => Math.abs(t.prob - ins.top[i].prob)));
  engine.setCapture({});

  // sampling filters, on PyTorch's own logits so that only the filter is being tested
  const ref = Float32Array.from(vec.cases[0].last_logits);
  let samplingOk = 0, samplingProbDiff = 0;
  for (const s of vec.sampling) {
    const p = s.params;
    const got = filterLogits(ref, { temperature: p.temperature, topK: p.top_k, topP: p.top_p, repetitionPenalty: p.repetition_penalty }, s.previous);
    const byId = new Map(Array.from(got.ids, (id, i) => [id, got.probs[i]]));
    const same = byId.size === s.ids.length && s.ids.every((id) => byId.has(id));
    if (same) {
      samplingOk++;
      s.ids.forEach((id, i) => { samplingProbDiff = Math.max(samplingProbDiff, Math.abs(byId.get(id) - s.probs[i])); });
    }
  }
  r.samplingIdentical = `${samplingOk}/${vec.sampling.length}`;
  r.maxSamplingProbDiff = samplingProbDiff;

  r.ok = r.maxAbsLogitDiff <= LOGIT_TOL && r.greedyIdentical === r.cases && attDiff <= ATTENTION_TOL
    && rowSumDiff <= ATTENTION_TOL && normDiff <= NORM_REL_TOL && r.topIdsIdentical && r.maxTopProbDiff <= PROB_TOL
    && samplingOk === vec.sampling.length && samplingProbDiff <= PROB_TOL
    && (!vec.constrained || r.constrainedIdentical === r.constrainedCases)
    && (!vec.embed || r.maxAbsEmbedDiff <= EMBED_TOL) && !r.denseScoreMismatches;
  if (vec.embed) r.maxAbsEmbedDiff = Number(r.maxAbsEmbedDiff.toExponential(2));
  delete r.embedCases;
  for (const k of ['maxAbsLogitDiff', 'maxAbsAttentionDiff', 'maxAttentionRowSumError', 'maxRelNormDiff', 'maxTopProbDiff', 'maxSamplingProbDiff']) {
    r[k] = Number(r[k].toExponential(2));
  }
  r.maxAbsLogit = Number(r.maxAbsLogit.toFixed(2));
  r.greedyIdentical = `${r.greedyIdentical}/${r.cases}`;
  delete r.cases;
  if (vec.constrained) {
    r.constrainedIdentical = `${r.constrainedIdentical}/${r.constrainedCases}`;
    delete r.constrainedCases;
  }
  return r;
}

export function checkText(dir) {
  const fixtures = JSON.parse(fs.readFileSync(path.join(dir, 'parity.json'), 'utf8'));
  const tok = new Tokenizer(JSON.parse(fs.readFileSync(path.join(dir, 'tokenizer.json'), 'utf8')));
  const out = {};
  const examples = [];
  const note = (kind, what) => { if (examples.length < 5) examples.push(`${kind}: ${JSON.stringify(what).slice(0, 200)}`); };

  let bad = 0;
  for (const c of fixtures.tokenizer) {
    const opts = { allowSpecial: c.allow_special };
    const t = tok.tokenize(c.text, opts);
    const ok = sameArray(tok.encode(c.text, opts), c.ids) && t.text === c.norm
      && sameArray(t.tokens.map((x) => x.id), c.ids)
      && t.tokens.every((x, i) => x.start === c.bounds[i][0] && x.end === c.bounds[i][1])
      && tok.decode(c.ids) === c.decoded && tok.decode(c.ids, { skipSpecial: false }) === c.decoded_with_specials;
    if (!ok) { bad++; note('tokenizer', c.text); }
  }
  out.tokenizer = { cases: fixtures.tokenizer.length, mismatches: bad };

  bad = 0;
  for (const c of fixtures.features) if (!sameArray(queryFeatures(c.q), c.features)) { bad++; note('features', c.q); }
  out.features = { cases: fixtures.features.length, mismatches: bad };

  if (fixtures.retrieval) {
    const retriever = new Retriever(JSON.parse(fs.readFileSync(path.join(dir, 'retrieval.json'), 'utf8')));
    let rankBad = 0, scoreBad = 0, hits = 0;
    for (const c of fixtures.retrieval) {
      const got = retriever.search(c.q, c.k, c.lang);
      const sameRank = got.length === c.hits.length && got.every((h, i) => h.id === c.hits[i].id && h.lang === c.hits[i].lang && h.doc === c.hits[i].doc);
      if (!sameRank) { rankBad++; note('retrieval', c.q); continue; }
      hits += got.length;
      if (!got.every((h, i) => h.score === c.hits[i].score)) { scoreBad++; note('retrieval score', c.q); }
    }
    out.retrieval = { cases: fixtures.retrieval.length, hits, rankingMismatches: rankBad, scoreMismatches: scoreBad };
  }

  // fused searches, given the question vector Python computed: the same documents and the same scores
  if (fixtures.dense) {
    const retriever = new Retriever(JSON.parse(fs.readFileSync(path.join(dir, 'retrieval.json'), 'utf8')));
    let rankBad = 0, scoreBad = 0, searches = 0;
    for (const c of fixtures.dense) {
      const vector = Float32Array.from(c.vector);
      for (const s of c.searches) {
        searches++;
        const got = retriever.search(c.q, s.k, s.lang, vector);
        const sameRank = got.length === s.hits.length && got.every((h, i) => h.id === s.hits[i].id && h.lang === s.hits[i].lang && h.doc === s.hits[i].doc);
        if (!sameRank) { rankBad++; note('dense retrieval', c.q); continue; }
        if (!got.every((h, i) => h.score === s.hits[i].score)) { scoreBad++; note('dense retrieval score', c.q); }
      }
    }
    out.dense = { cases: fixtures.dense.length, searches, rankingMismatches: rankBad, scoreMismatches: scoreBad };
  }

  if (fixtures.trie) {
    const index = JSON.parse(fs.readFileSync(path.join(dir, 'retrieval.json'), 'utf8'));
    let walks = 0, steps = 0, biasSteps = 0;
    bad = 0;
    for (const c of fixtures.trie) {
      const trie = AnswerTrie.fromIndex(index, tok, c.lang, null, c.only ?? null, c.priors ?? null, c.fixed_prior ?? 0);
      const d = trie.describe();
      if (d.sentences !== c.sentences || d.nodes !== c.nodes || d.maxSentences !== c.max_sentences) { bad++; note('trie size', c.lang); }
      for (const w of c.walks) {
        walks++;
        let state = trie.start(), ok = trie.accepts(w.ids) === w.accepts;
        for (let i = 0; ok && i < w.allowed.length; i++) {
          steps++;
          ok = sameArray(Array.from(trie.allowed(state)), w.allowed[i]);
          if (ok && w.bias) { biasSteps++; ok = sameArray(Array.from(trie.bias(state)), w.bias[i]); }
          if (ok && i < w.ids.length && w.allowed[i].includes(w.ids[i]) && w.ids[i] !== trie.eos) state = trie.advance(state, w.ids[i]);
        }
        if (!ok) { bad++; note('trie', w.text); }
      }
    }
    out.trie = { cases: fixtures.trie.length, walks, steps, biasSteps, mismatches: bad };
  }

  bad = 0;
  for (const c of fixtures.prompt) {
    const got = buildPrompt(tok, c.question, c.facts, c.block_size, fixtures.prompt_config);
    if (!sameArray(got.ids, c.ids) || got.used !== c.used) { bad++; note('prompt', c.question); }
  }
  out.prompt = { cases: fixtures.prompt.length, mismatches: bad };

  out.ok = !out.tokenizer.mismatches && !out.features.mismatches && !out.prompt.mismatches
    && (!out.retrieval || (!out.retrieval.rankingMismatches && !out.retrieval.scoreMismatches))
    && (!out.trie || !out.trie.mismatches)
    && (!out.dense || (!out.dense.rankingMismatches && !out.dense.scoreMismatches));
  if (examples.length) out.examples = examples;
  return out;
}

export async function runParity(dir, wasmPath) {
  const wasm = wasmPath ? toArrayBuffer(fs.readFileSync(wasmPath)) : null;
  const report = { dir, node: process.version, engine: {}, ok: true };
  for (const fmt of ['f32', 'f16', 'i8']) {
    if (!fs.existsSync(path.join(dir, `model.${fmt}.bin`))) continue;
    for (const kernel of wasm ? ['js', 'wasm'] : ['js']) {
      const r = await checkEngine(dir, fmt, kernel === 'wasm' ? wasm : null);
      report.engine[`${fmt}.${r.kernel}`] = r;
      report.ok = report.ok && r.ok;
    }
  }
  if (!Object.keys(report.engine).length) throw new Error(`no model.<fmt>.bin in ${dir}`);
  if (fs.existsSync(path.join(dir, 'parity.json'))) {
    report.text = checkText(dir);
    report.ok = report.ok && report.text.ok;
  }
  return report;
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  const args = process.argv.slice(2);
  const wasmAt = args.indexOf('--wasm');
  const wasmPath = wasmAt >= 0 ? args[wasmAt + 1] : null;
  const quiet = args.includes('--quiet');
  const dir = args.find((a, i) => !a.startsWith('--') && (wasmAt < 0 || i !== wasmAt + 1));
  if (!dir) {
    console.error('usage: node parity.mjs <export-dir> [--wasm kernel.wasm] [--quiet]');
    process.exit(2);
  }
  const report = await runParity(dir, wasmPath);
  if (!quiet) {
    for (const [name, r] of Object.entries(report.engine)) {
      console.log(`${name.padEnd(14)} ${r.ok ? 'ok  ' : 'FAIL'} max|dlogit| ${r.maxAbsLogitDiff} (max|logit| ${r.maxAbsLogit})  greedy ${r.greedyIdentical}  `
        + `${r.constrainedIdentical ? `constrained ${r.constrainedIdentical}${r.constrainedWithPrior ? ` (${r.constrainedWithPrior} with a prior)` : ''}  ` : ''}attention ${r.maxAbsAttentionDiff}  norms ${r.maxRelNormDiff}  sampling ${r.samplingIdentical}`
        + `${r.maxAbsEmbedDiff !== undefined ? `  embed ${r.maxAbsEmbedDiff}` : ''}${r.denseSearches ? `  fused search ${r.denseSearches} (max|dscore| ${r.maxDenseScoreDiff})` : ''}`);
    }
    if (report.text) {
      const t = report.text;
      console.log(`tokenizer      ${t.tokenizer.mismatches ? 'FAIL' : 'ok  '} ${t.tokenizer.cases} cases, ${t.tokenizer.mismatches} mismatches`);
      console.log(`features       ${t.features.mismatches ? 'FAIL' : 'ok  '} ${t.features.cases} cases, ${t.features.mismatches} mismatches`);
      if (t.trie) console.log(`answer trie    ${t.trie.mismatches ? 'FAIL' : 'ok  '} ${t.trie.walks} walks, ${t.trie.steps} steps (${t.trie.biasSteps} with priors), ${t.trie.mismatches} mismatches`);
      if (t.retrieval) {
        console.log(`retrieval      ${t.retrieval.rankingMismatches || t.retrieval.scoreMismatches ? 'FAIL' : 'ok  '} ${t.retrieval.cases} queries, `
          + `${t.retrieval.hits} hits, ${t.retrieval.rankingMismatches} ranking mismatches, ${t.retrieval.scoreMismatches} score mismatches`);
      }
      if (t.dense) {
        console.log(`dense retrieval ${t.dense.rankingMismatches || t.dense.scoreMismatches ? 'FAIL' : 'ok '} ${t.dense.cases} questions, `
          + `${t.dense.searches} fused searches, ${t.dense.rankingMismatches} ranking mismatches, ${t.dense.scoreMismatches} score mismatches`);
      }
      console.log(`prompt         ${t.prompt.mismatches ? 'FAIL' : 'ok  '} ${t.prompt.cases} cases, ${t.prompt.mismatches} mismatches`);
      for (const e of t.examples || []) console.log('  ' + e);
    }
    console.log(report.ok ? 'PARITY OK' : 'PARITY FAILED');
  }
  console.log(JSON.stringify(report));
  // Not process.exit(): it can deadlock in Node 24 when V8 is still compiling in the background (the main
  // thread joins a worker that waits for a garbage collection). Let the process end by itself.
  process.exitCode = report.ok ? 0 : 1;
}
