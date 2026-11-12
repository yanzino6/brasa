// Compile-time check of the declaration files: `npm run typecheck` (needs TypeScript; nothing is emitted).
// This file is never executed. It uses the public API the way a page would.
import {
  BrasaClient, Engine, Retriever, Session, Tokenizer, buildPrompt, filterLogits, seededRandom, topTokens,
} from '../index.js';
import type {
  AttentionSnapshot, DoneMessage, GenerationStep, Hit, ModelInfo, RetrievalIndex, TokenMessage, TokenizerSpec,
  WorkerRequest, WorkerResponse,
} from '../index.js';

export async function page(modelBytes: ArrayBuffer, wasmBytes: ArrayBuffer, index: RetrievalIndex, spec: TokenizerSpec) {
  const engine: Engine = await Engine.create(modelBytes, { wasm: wasmBytes, contextLength: 256 });
  const tok = new Tokenizer(engine.tokenizerSpec ?? spec);
  const retriever = new Retriever(index);
  const hits: Hit[] = retriever.search('onde ele mora', 4, 'pt');
  const { ids, used } = buildPrompt(tok, 'onde ele mora', hits.map((h) => h.text), engine.contextLength, engine.prompt ?? undefined);
  const spans = tok.tokenize('Olá', { allowSpecial: false });
  const first: number = spans.tokens[0].start + used;

  for (const step of engine.stream(ids, { temperature: 0.8, topK: 40, topP: 0.9, repetitionPenalty: 1.1, random: seededRandom(1), inspect: { top: 5, attention: true, norms: true } })) {
    const s: GenerationStep = step;
    const attention: AttentionSnapshot | null | undefined = s.attention;
    const weight: number = attention ? attention.data[(0 * attention.heads + 0) * attention.length + 0] : 0;
    const best: string = tok.tokenText(s.top![0].id);
    void [weight, best, first];
  }
  const result = engine.generate(ids, { maxNewTokens: 32, stopTokens: [tok.specialIds['<|eos|>']], onToken: (s) => s.index < 10 });
  const reason: 'stop' | 'length' | 'aborted' = result.stopReason;
  const kept = filterLogits(engine.logits, { topK: 5 }, result.tokens);
  const top = topTokens(engine.logits, 3);
  return [reason, kept.ids.length, top[0].prob, tok.decode(result.tokens)];
}

export async function withWorker(worker: Worker) {
  const client = new BrasaClient(worker);
  const info: ModelInfo = await client.load({ model: '/brasa/model.i8.bin', wasm: '/brasa/kernel.wasm', retrieval: '/brasa/retrieval.json' },
    (p) => console.log(p.stage, p.loaded / Math.max(1, p.total)));
  const run = client.ask('Onde ele mora?', { mode: 'open', sampling: { temperature: 0.7, topP: 0.9, seed: 3 }, inspect: { top: 5 } }, {
    onContext: (c) => console.log(c.hits.map((h) => h.id), c.used),
    onPrompt: (p) => console.log(p.tokens.length),
    onToken: (t: TokenMessage) => console.log(t.delta, t.prob, t.top?.[0].text),
  });
  run.abort();
  const done: DoneMessage = await run.done;
  return [info.kernel, done.stats.tokensPerSecond, done.text];
}

export function rawProtocol(post: (m: WorkerResponse) => void) {
  const session = new Session(post, { sliceMs: 8 });
  const request: WorkerRequest = { type: 'generate', id: 1, tokens: [1, 2, 3], maxNewTokens: 1, inspect: { attention: true } };
  return session.handle(request);
}
