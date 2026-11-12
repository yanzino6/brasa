// Parity of the JavaScript engine, tokenizer, retriever and prompt builder against the Python reference.
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
import { Retriever, queryFeatures } from '../retrieval.js';
import { buildPrompt } from '../prompt.js';

const LOGIT_TOL = 1e-3, ATTENTION_TOL = 1e-4, NORM_REL_TOL = 1e-4, PROB_TOL = 5e-5;

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
    && samplingOk === vec.sampling.length && samplingProbDiff <= PROB_TOL;
  for (const k of ['maxAbsLogitDiff', 'maxAbsAttentionDiff', 'maxAttentionRowSumError', 'maxRelNormDiff', 'maxTopProbDiff', 'maxSamplingProbDiff']) {
    r[k] = Number(r[k].toExponential(2));
  }
  r.maxAbsLogit = Number(r.maxAbsLogit.toFixed(2));
  r.greedyIdentical = `${r.greedyIdentical}/${r.cases}`;
  delete r.cases;
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

  bad = 0;
  for (const c of fixtures.prompt) {
    const got = buildPrompt(tok, c.question, c.facts, c.block_size, fixtures.prompt_config);
    if (!sameArray(got.ids, c.ids) || got.used !== c.used) { bad++; note('prompt', c.question); }
  }
  out.prompt = { cases: fixtures.prompt.length, mismatches: bad };

  out.ok = !out.tokenizer.mismatches && !out.features.mismatches && !out.prompt.mismatches
    && (!out.retrieval || (!out.retrieval.rankingMismatches && !out.retrieval.scoreMismatches));
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
        + `attention ${r.maxAbsAttentionDiff}  norms ${r.maxRelNormDiff}  sampling ${r.samplingIdentical}`);
    }
    if (report.text) {
      const t = report.text;
      console.log(`tokenizer      ${t.tokenizer.mismatches ? 'FAIL' : 'ok  '} ${t.tokenizer.cases} cases, ${t.tokenizer.mismatches} mismatches`);
      console.log(`features       ${t.features.mismatches ? 'FAIL' : 'ok  '} ${t.features.cases} cases, ${t.features.mismatches} mismatches`);
      if (t.retrieval) {
        console.log(`retrieval      ${t.retrieval.rankingMismatches || t.retrieval.scoreMismatches ? 'FAIL' : 'ok  '} ${t.retrieval.cases} queries, `
          + `${t.retrieval.hits} hits, ${t.retrieval.rankingMismatches} ranking mismatches, ${t.retrieval.scoreMismatches} score mismatches`);
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
