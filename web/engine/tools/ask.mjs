// Ask an exported model questions through the browser session, in Node: the answers a visitor would get.
//
//   node web/engine/tools/ask.mjs <export-dir> --questions questions.txt [--wasm web/wasm/kernel.wasm]
//        [--format i8] [--prior 24 | --no-prior] [--free] [--closed] [--lexical]
//
// questions.txt holds one question per line. Prints one JSON line per question:
//   { question, answer, mode, constraint, prior, retriever, facts: [fact ids in the prompt], tokens }
// the same fields as `python -m brasa.sample --export <dir> --json`, so that the two can be compared line by
// line (scripts/compare_answers.py): the end-to-end check that the engine answers like the evaluated model.
import fs from 'node:fs';
import path from 'node:path';
import { Session } from '../session.js';

const args = process.argv.slice(2);
const flag = (name) => args.includes('--' + name);
const opt = (name, fallback) => { const i = args.indexOf('--' + name); return i >= 0 ? args[i + 1] : fallback; };
const toArrayBuffer = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const dir = args[0];
const questionsPath = opt('questions', null);
if (!dir || dir.startsWith('--') || !questionsPath) {
  console.error('usage: node ask.mjs <export-dir> --questions FILE [--wasm kernel.wasm] [--format i8] [--prior N | --no-prior] [--free] [--closed] [--lexical]');
  process.exit(2);
}

const messages = [];
const session = new Session((m) => messages.push(m));
// The session yields to the event loop through an unreferenced MessageChannel (so that a page can abort);
// in a script nothing else keeps Node alive while it waits, so hold the loop open until the last answer.
const keepAlive = setInterval(() => {}, 1000);
const wasmPath = opt('wasm', null);
await session.handle({
  type: 'load', id: 0,
  model: toArrayBuffer(fs.readFileSync(path.join(dir, `model.${opt('format', 'i8')}.bin`))),
  wasm: wasmPath ? toArrayBuffer(fs.readFileSync(wasmPath)) : null,
  retrieval: JSON.parse(fs.readFileSync(path.join(dir, 'retrieval.json'), 'utf8')),
});
const failure = messages.find((m) => m.type === 'error');
if (failure) throw new Error(failure.message);

const extra = {};
if (flag('free')) extra.constrain = false;
if (flag('closed')) extra.mode = 'closed';
if (flag('lexical')) extra.dense = false;
if (flag('no-prior')) extra.prior = false;
else if (opt('prior', null) !== null) extra.prior = Number(opt('prior'));

const questions = fs.readFileSync(questionsPath, 'utf8').split('\n').map((q) => q.trim()).filter(Boolean);
let id = 0;
for (const question of questions) {
  messages.length = 0;
  await session.handle({ type: 'ask', id: ++id, question, ...extra });
  const context = messages.find((m) => m.type === 'context');
  const done = messages.find((m) => m.type === 'done');
  const error = messages.find((m) => m.type === 'error');
  if (error || !done) throw new Error(`${question}: ${error ? error.message : 'no answer'}`);
  console.log(JSON.stringify({
    question, answer: done.text, mode: context.mode, constraint: context.constraint, prior: context.prior ?? null,
    retriever: context.retriever ?? null, facts: context.hits.slice(0, context.used).map((h) => h.id),
    tokens: done.tokens.length,
  }));
}
clearInterval(keepAlive);
