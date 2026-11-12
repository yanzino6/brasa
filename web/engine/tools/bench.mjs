// Speed of the engine in Node (the same V8 as Chrome), with the JavaScript and the WASM SIMD kernel.
//
//   node web/engine/tools/bench.mjs runs/brasa-12m/export/model.i8.bin [--wasm web/wasm/kernel.wasm]
//        [--new 224] [--prompt 16] [--repeats 3]
//
// Prints one JSON line per kernel. decodeTokPerSec is measured after the prompt, one token per step with
// the KV cache; last32 is the speed of the last 32 tokens (the context is longest there).
import fs from 'node:fs';
import { benchEngine } from './bench_core.js';
import { parseModel } from '../engine.js';

const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf('--' + name); return i >= 0 ? args[i + 1] : fallback; };
const toArrayBuffer = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
if (!args[0] || args[0].startsWith('--')) {
  console.error('usage: node bench.mjs MODEL.bin [--wasm kernel.wasm] [--new 224] [--prompt 16] [--repeats 3]');
  process.exit(2);
}
const model = toArrayBuffer(fs.readFileSync(args[0]));
const { header } = parseModel(model);
const wasmPath = opt('wasm', null);
// a deterministic pseudo-random prompt of plain (non-special) tokens
let seed = 12345;
const plain = header.config.vocab_size - 8;
const prompt = Array.from({ length: Number(opt('prompt', 16)) }, () => (seed = (seed * 1103515245 + 12345) >>> 0) % plain);

for (const wasm of wasmPath ? [null, toArrayBuffer(fs.readFileSync(wasmPath))] : [null]) {
  const res = await benchEngine({ model, wasm, prompt, maxNew: Number(opt('new', 224)), repeats: Number(opt('repeats', 3)) });
  const best = res.runs.reduce((a, b) => (b.decodeTokPerSec > a.decodeTokPerSec ? b : a));
  console.log(JSON.stringify({
    file: args[0], bytes: model.byteLength, kernel: res.kernel, params: header.config.params,
    layers: header.config.n_layer, d_model: header.config.d_model, vocab: header.config.vocab_size,
    memoryMB: res.memoryMB, initMs: res.initMs, promptTokens: res.promptTokens, newTokens: res.newTokens,
    decodeTokPerSec: best.decodeTokPerSec, first32: best.first32, last32: best.last32,
    ttftMs: best.ttftMs, prefillTokPerSec: best.prefillTokPerSec, allRuns: res.runs.map((r) => r.decodeTokPerSec),
    node: process.version,
  }));
}
