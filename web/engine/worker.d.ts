// The message protocol between a page and worker.js. Every request carries an `id` chosen by the page;
// every response repeats it. client.js implements the page side.
import type { AttentionSnapshot, InspectOptions, ModelConfig, SamplingOptions, StopReason, TopToken } from './engine.js';
import type { PromptConfig } from './prompt.js';
import type { Hit, RetrievalIndex } from './retrieval.js';
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
}

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
  fetchMs: number;
  initMs: number;
}

export interface ReadyMessage { type: 'ready'; id: number; info: ModelInfo }
export interface TokensMessage { type: 'tokens'; id: number; text: string; tokens: TokenSpan[] }
export interface HitsMessage { type: 'hits'; id: number; hits: Hit[] }

/** First message of an `ask`: what was retrieved and how many of those facts fitted into the prompt. */
export interface ContextMessage { type: 'context'; id: number; mode: 'open' | 'closed'; hits: Hit[]; used: number }

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
}

/** Last message of an `ask` or `generate`. */
export interface DoneMessage { type: 'done'; id: number; text: string; tokens: number[]; stats: GenerationStats }
export interface UnloadedMessage { type: 'unloaded'; id: number }
export interface ErrorMessage { type: 'error'; id: number; message: string }

export type WorkerResponse =
  | ProgressMessage | ReadyMessage | TokensMessage | HitsMessage | ContextMessage | PromptMessage
  | TokenMessage | DoneMessage | UnloadedMessage | ErrorMessage;
