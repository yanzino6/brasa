// Compile-time check of the declaration files: `npm run typecheck` (needs TypeScript; nothing is emitted).
// This file is never executed. It uses the public API the way a page would.
import {
  AnswerTrie, BrasaClient, Engine, Retriever, Session, Tokenizer, buildPrompt, chooseAllowed, filterLogits, rankingPriors,
  seededRandom, topTokens,
} from '../index.js';
import type {
  AttentionSnapshot, DecodingConstraint, DecodingPrior, DoneMessage, GenerationStep, Hit, ModelInfo, RetrievalIndex, TokenMessage,
  TokenizerSpec, WorkerRequest, WorkerResponse,
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

  // constrained to the knowledge base: the trie of the index, or any object with the same four members
  const trie: DecodingConstraint<number[]> = AnswerTrie.fromIndex(index, tok, 'pt');
  const retrieved: AnswerTrie = AnswerTrie.fromIndex(index, tok, null, null, new Set(['a', 'b']));
  void retrieved;
  // the retriever's ranking as a prior on the retrieved facts
  const found: Hit[] = retriever.search('onde ele mora');
  const stored: DecodingPrior = index.answers?.prior ?? { kind: 'rank', weight: 0, fixed: 0 };
  const ranked = AnswerTrie.fromIndex(index, tok, null, null, found.map((h) => h.id), rankingPriors(found, stored.weight, stored.kind), stored.fixed);
  const bias: Float64Array | null = ranked.hasPrior ? ranked.bias(ranked.start()) : null;
  const leaning: number = chooseAllowed(engine.logits, ranked.allowed(ranked.start()), {}, null, bias);
  void leaning;

  // the model as an encoder: a question vector, fused with BM25 by an index that carries fact vectors
  const s = tok.specialIds;
  const vector: Float32Array = engine.embed([s['<|bos|>'], s['<|q|>'], ...tok.encode('onde ele mora'), s['<|a|>']], retriever.dense?.pooling);
  const fused: Hit[] = retriever.search('onde ele mora', 4, null, vector);
  const weight: number = (index.dense?.weight ?? 0) + (retriever.dense?.dim ?? 0) + fused.length;
  void weight;
  const curated = engine.generate(ids, { constraint: trie, maxNewTokens: engine.prompt?.answer_reserve });
  const forced: number = [...engine.stream(ids, { constraint: trie })].filter((s) => s.choices === 1).length;
  const pick: number = chooseAllowed(engine.logits, trie.allowed(trie.start()), { temperature: 0 });
  return [reason, kept.ids.length, top[0].prob, tok.decode(result.tokens), tok.decode(curated.tokens), forced, pick];
}

export async function withWorker(worker: Worker) {
  const client = new BrasaClient(worker);
  const info: ModelInfo = await client.load({ model: '/brasa/model.i8.bin', wasm: '/brasa/kernel.wasm', retrieval: '/brasa/retrieval.json' },
    (p) => console.log(p.stage, p.loaded / Math.max(1, p.total)));
  const run = client.ask('Onde ele mora?', { mode: 'open', sampling: { temperature: 0.7, topP: 0.9, seed: 3 }, inspect: { top: 5 } }, {
    onContext: (c) => console.log(c.hits.map((h) => h.id), c.used, c.constrained),
    onPrompt: (p) => console.log(p.tokens.length),
    onToken: (t: TokenMessage) => console.log(t.delta, t.prob, t.top?.[0].text),
  });
  run.abort();
  const done: DoneMessage = await run.done;
  const trusting = client.ask('Onde ele mora?', { prior: { kind: 'score', weight: (info.prior?.weight ?? 0) * 2 } }, {
    onContext: (c) => console.log(c.prior ? `${c.prior.kind} ${c.prior.weight}` : 'the model alone decides'),
  });
  void trusting;
  void client.ask('Onde ele mora?', { prior: false });
  void client.ask('Onde ele mora?', { prior: 4 });
  const free = client.ask('Onde ele mora?', { constrain: false, lang: 'pt', dense: false }, {
    onContext: (c) => console.log(c.retriever === 'bm25+dense' ? 'ranked by the model too' : 'lexical'),
    onToken: (t) => console.log(t.choices ?? 'free'),
  });
  const lexical: Hit[] = await client.retrieve('onde ele mora', 4, null, { dense: info.dense && false });
  void lexical;
  return [info.kernel, info.constrained, done.stats.tokensPerSecond, done.stats.constrained, done.text, (await free.done).text];
}

export function rawProtocol(post: (m: WorkerResponse) => void) {
  const session = new Session(post, { sliceMs: 8 });
  const request: WorkerRequest = { type: 'generate', id: 1, tokens: [1, 2, 3], maxNewTokens: 1, inspect: { attention: true } };
  return session.handle(request);
}
