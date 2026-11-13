// The message protocol between a page and worker.js. Every request carries an `id` chosen by the page;
// every response repeats it. client.js implements the page side.
import type { AttentionSnapshot, InspectOptions, ModelConfig, SamplingOptions, StopReason, TopToken } from './engine.js';
import type { PromptConfig } from './prompt.js';
import type { DecodingPrior, Hit, RetrievalIndex } from './retrieval.js';
import type { TokenSpan } from './tokenizer.js';

export interface WorkerSampling extends SamplingOptions {
  /** Makes sampling reproducible. Omit for Math.random. */
  seed?: number;
}

// ------------------------------------------------------------------------------------------ requests

export interface LoadRequest {
  type: 'load';
  id: number;
  /** URL of model.i8.bin / model.f16.bin, or its bytes. */
  model: string | ArrayBuffer;
  /** URL or bytes of kernel.wasm. Omitted, unreachable or unsupported: the JavaScript kernel is used. */
  wasm?: string | ArrayBuffer | null;
  /** URL, bytes or parsed contents of retrieval.json. Without it only closed-book questions work. */
  retrieval?: string | ArrayBuffer | RetrievalIndex | null;
  /** Use fewer positions than the model's block_size. */
  contextLength?: number;
}

export interface TokenizeRequest {
  type: 'tokenize';
  id: number;
  text: string;
  allowSpecial?: boolean;
}

export interface RetrieveRequest {
  type: 'retrieve';
  id: number;
  query: string;
  /** Default: the model's max_facts. */
  k?: number;
  lang?: 'pt' | 'en' | null;
  /** false: the lexical ranking alone, even when the index carries fact vectors. Default: use them. */
  dense?: boolean;
}

export interface AskRequest {
  type: 'ask';
  id: number;
  question: string;
  /** 'open' retrieves facts into the prompt (default when an index is loaded); 'closed' sends the question alone. */
  mode?: 'open' | 'closed';
  /** Facts to retrieve. Default: the model's max_facts. */
  k?: number;
  lang?: 'pt' | 'en' | null;
  /** Default: the model's answer_reserve. */
  maxNewTokens?: number;
  /** Default: greedy. */
  sampling?: WorkerSampling;
  inspect?: InspectOptions | null;
  /**
   * Constrain the answer: the model can only write curated sentences (a fact, two or three facts in a
   * row, or a fixed answer such as the refusal).
   *   'retrieved'  only the facts that are in the prompt (what the retriever found) and the fixed answers:
   *                every answer is one of the sentences shown as context. Open-book only.
   *   'kb'         any fact of the knowledge base.
   *   true         the default when a retrieval index is loaded: 'retrieved' open-book, 'kb' closed-book.
   *   false        the model's free generation.
   * With `lang` only sentences of that language are allowed.
   */
  constrain?: boolean | ConstraintKind;
  /**
   * With constrain 'retrieved': how much the retriever's ranking weighs in the model's choice among the
   * retrieved facts (DecodingPrior). Omitted: the prior stored in the index (ModelInfo.prior), chosen on
   * validation questions. false or 0: the model alone decides. A number: that weight, with the kind of the
   * index. An object: merged over the index's prior. A large weight makes the answer the first hit whenever
   * the model answers with a fact; a `fixed` above 0 makes it decline more often.
   */
  prior?: boolean | number | Partial<DecodingPrior>;
  /**
   * How the facts are ranked in open-book mode. With an index that carries fact vectors (ModelInfo.dense)
   * the model first reads the question as an encoder and the ranking is BM25 fused with the cosine;
   * false keeps the lexical ranking alone.
   */
  dense?: boolean;
}

/** How the facts of an answer were ranked: BM25 alone, or fused with the model's own ranking. */
export type RetrieverKind = 'bm25' | 'bm25+dense';

/** What a constrained answer may be made of: the retrieved facts, or the whole knowledge base. */
export type ConstraintKind = 'retrieved' | 'kb';

/**
 * Continue an arbitrary token sequence (no retrieval, no prompt building). The engine keeps the state of
 * the previous sequence, so sending the previous tokens plus a few more only computes the new ones: that
 * is how a page steps one token at a time (maxNewTokens: 1) or explores "what if this token instead".
 */
export interface GenerateRequest {
  type: 'generate';
  id: number;
  tokens: number[];
  maxNewTokens?: number;
  stopTokens?: number[];
  sampling?: WorkerSampling;
  inspect?: InspectOptions | null;
}

/** Stop the generation with this id (or whichever is running when id is omitted). No response of its own. */
export interface AbortRequest {
  type: 'abort';
  id?: number;
}

export interface UnloadRequest {
  type: 'unload';
  id: number;
}

export type WorkerRequest =
  | LoadRequest | TokenizeRequest | RetrieveRequest | AskRequest | GenerateRequest | AbortRequest | UnloadRequest;

// ------------------------------------------------------------------------------------------ responses

/** Download progress of a `load`. `total` is 0 when the server sends no Content-Length. */
export interface ProgressMessage {
  type: 'progress';
  id: number;
  stage: 'model' | 'wasm' | 'retrieval';
  loaded: number;
  total: number;
}

export interface ModelInfo {
  name: string;
  format: 'f32' | 'f16' | 'i8';
  config: ModelConfig;
  prompt: PromptConfig;
  kernel: 'wasm-simd' | 'js';
  /** Why the WASM kernel could not be used, when one was requested. */
  kernelError: string | null;
  memoryBytes: number;
  contextLength: number;
  vocabSize: number;
  specialTokens: Record<string, number>;
  /** Documents in the retrieval index (0 without one). */
  facts: number;
  /** Whether `ask` constrains answers to the knowledge base by default (a retrieval index is loaded). */
  constrained: boolean;
  /** Whether the retrieval index carries fact vectors, so that the model also ranks the facts. */
  dense: boolean;
  /** The default prior of constrained open-book answers (the one stored in the index), or null for none. */
  prior: DecodingPrior | null;
  fetchMs: number;
  initMs: number;
}

export interface ReadyMessage { type: 'ready'; id: number; info: ModelInfo }
export interface TokensMessage { type: 'tokens'; id: number; text: string; tokens: TokenSpan[] }
export interface HitsMessage { type: 'hits'; id: number; hits: Hit[]; retriever: RetrieverKind }

/** First message of an `ask`: what was retrieved and how many of those facts fitted into the prompt. */
export interface ContextMessage {
  type: 'context';
  id: number;
  mode: 'open' | 'closed';
  hits: Hit[];
  used: number;
  /** Whether this answer is constrained to curated sentences. */
  constrained: boolean;
  /** Which sentences: the retrieved facts, the whole knowledge base, or null for free generation. */
  constraint: ConstraintKind | null;
  /** Open-book only: how the facts were ranked. */
  retriever?: RetrieverKind;
  /** With constraint 'retrieved' only: the prior that ranking has on the answer, or null for none. */
  prior?: DecodingPrior | null;
}

/** The exact token sequence given to the model. */
export interface PromptMessage { type: 'prompt'; id: number; tokens: { id: number; text: string }[] }

export interface TokenMessage {
  type: 'token';
  id: number;
  /** 0 for the first generated token. */
  index: number;
  token: number;
  /** The token on its own. */
  text: string;
  /** Text that became complete with this token (empty while a multi-byte character is still open). */
  delta: string;
  /** True for the end-of-sequence token that closes the answer. */
  stop: boolean;
  /** Constrained answers: how many tokens the knowledge base allowed here (1: the token was forced). */
  choices?: number;
  /** With inspect: probability the model gave to this token. */
  prob?: number;
  /** With inspect.top. */
  top?: (TopToken & { text: string })[];
  /** With inspect.attention. The buffer is transferred, not copied. */
  attention?: AttentionSnapshot | null;
  /** With inspect.norms. */
  norms?: Float32Array | null;
}

export interface GenerationStats {
  stopReason: StopReason;
  promptTokens: number;
  reusedTokens: number;
  newTokens: number;
  prefillMs: number;
  decodeMs: number;
  tokensPerSecond: number;
  kernel: 'wasm-simd' | 'js';
  /** Whether the answer was constrained to curated sentences. */
  constrained: boolean;
  /** 'retrieved', 'kb', or null for free generation. */
  constraint: ConstraintKind | null;
}

/** Last message of an `ask` or `generate`. */
export interface DoneMessage { type: 'done'; id: number; text: string; tokens: number[]; stats: GenerationStats }
export interface UnloadedMessage { type: 'unloaded'; id: number }
export interface ErrorMessage { type: 'error'; id: number; message: string }

export type WorkerResponse =
  | ProgressMessage | ReadyMessage | TokensMessage | HitsMessage | ContextMessage | PromptMessage
  | TokenMessage | DoneMessage | UnloadedMessage | ErrorMessage;
