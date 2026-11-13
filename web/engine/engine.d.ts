import type { TokenizerSpec } from './tokenizer.js';
import type { PromptConfig } from './prompt.js';

/** Hyper-parameters stored in the model file. */
export interface ModelConfig {
  vocab_size: number;
  /** Context length the model was trained with. */
  block_size: number;
  n_layer: number;
  n_head: number;
  d_model: number;
  mlp: 'swiglu' | 'gelu';
  mlp_hidden: number;
  dropout: number;
  rope_base: number;
  norm_eps: number;
  /** Number of parameters. */
  params: number;
}

export interface TensorEntry {
  name: string;
  shape: number[];
  dtype: 'f32' | 'f16' | 'i8';
  offset: number;
  scalesOffset?: number;
}

export interface ModelHeader {
  format: 'f32' | 'f16' | 'i8';
  name: string;
  config: ModelConfig;
  prompt: PromptConfig;
  tokenizer: TokenizerSpec | null;
  tensors: TensorEntry[];
}

export function parseModel(buffer: ArrayBuffer | Uint8Array): { header: ModelHeader; dataStart: number; buffer: ArrayBuffer };

export interface EngineOptions {
  /** Bytes of kernel.wasm (or a compiled module). Omit for the plain JavaScript kernel. */
  wasm?: ArrayBuffer | Uint8Array | WebAssembly.Module | null;
  /** Use fewer positions than block_size to save memory. */
  contextLength?: number;
}

export interface SamplingOptions {
  /** 0 (default) is greedy decoding. */
  temperature?: number;
  /** Keep the k most likely tokens; 0 (default) disables the filter. */
  topK?: number;
  /** Nucleus sampling: keep the smallest set of tokens whose probability reaches topP. Default 1 (off). */
  topP?: number;
  /** Values above 1 make tokens already generated in this answer less likely. Default 1 (off). */
  repetitionPenalty?: number;
}

export interface TopToken {
  id: number;
  /** Probability under the model's own distribution (temperature 1, nothing filtered). */
  prob: number;
  logit: number;
}

/** Attention of one position towards every position of the context, for all layers and heads. */
export interface AttentionSnapshot {
  layers: number;
  heads: number;
  /** Number of positions in the context (including the current one). */
  length: number;
  /** data[(layer * heads + head) * length + position]; each (layer, head) row sums to 1. */
  data: Float32Array;
}

export interface InspectOptions {
  /** Attach the k most likely tokens to every step. */
  top?: number;
  /** Attach the attention of every layer and head to every step. */
  attention?: boolean;
  /** Attach the residual-stream norms to every step. */
  norms?: boolean;
}

export interface StreamOptions extends SamplingOptions {
  /** Default 128; also limited by the room left in the context. */
  maxNewTokens?: number;
  /** Generation ends after producing one of these (it is still yielded, with stop: true). */
  stopTokens?: number[];
  /** Source of randomness in [0, 1) for sampling. Default Math.random; see seededRandom(). */
  random?: () => number;
  inspect?: InspectOptions | null;
  /**
   * Restrict every step to the tokens this allows and stop at its eos: decoding constrained to the
   * knowledge base (AnswerTrie in retrieval.js). Omitted or null: free decoding, at no extra cost.
   */
  constraint?: DecodingConstraint | null;
}

/** What Engine.stream needs from a constraint; S is its opaque state. */
export interface DecodingConstraint<S = unknown> {
  /** Token that ends an answer. */
  readonly eos: number;
  start(): S;
  /** Token ids that may come next, ascending. Empty when nothing may follow. */
  allowed(state: S): ArrayLike<number>;
  advance(state: S, token: number): S;
  /** Optional: when true, bias(state) is added to the logits of the allowed tokens before each choice. */
  readonly hasPrior?: boolean;
  /** One number per token of allowed(state), in nats (a prior over the allowed answers). */
  bias?(state: S): ArrayLike<number>;
}

export interface GenerationStep {
  /** 0 for the first generated token. */
  index: number;
  token: number;
  /** True when the token is one of stopTokens. */
  stop: boolean;
  /** Present with a constraint: how many tokens were allowed at this step (1: the token was forced). */
  choices?: number;
  /** Present with inspect: probability the model gave to the chosen token. */
  prob?: number;
  /** Present with inspect.top. */
  top?: TopToken[];
  /** Present with inspect.attention: attention of the position that predicted this token. */
  attention?: AttentionSnapshot | null;
  /** Present with inspect.norms: [after embedding, after block 0, ..., after the last block]. */
  norms?: Float32Array | null;
}

export type StopReason = 'stop' | 'length' | 'aborted';

export interface GenerateOptions extends StreamOptions {
  /** Called for every token; return false to stop. */
  onToken?: (step: GenerationStep) => boolean | void;
}

export interface GenerateResult {
  tokens: number[];
  stopReason: StopReason;
  promptTokens: number;
  /** Prompt tokens whose cached state was reused from the previous call. */
  reusedTokens: number;
  /** Time to process the prompt and choose the first token. */
  prefillMs: number;
  decodeMs: number;
  tokensPerSecond: number;
}

export class Engine {
  static create(modelBuffer: ArrayBuffer | Uint8Array, options?: EngineOptions): Promise<Engine>;

  readonly name: string;
  readonly format: 'f32' | 'f16' | 'i8';
  readonly config: ModelConfig;
  readonly prompt: PromptConfig | null;
  readonly tokenizerSpec: TokenizerSpec | null;
  /** 'wasm-simd' or 'js'. */
  readonly kernelName: 'wasm-simd' | 'js';
  /** Why the WASM kernel was not used although one was given (null when it loaded or none was given). */
  readonly kernelError: string | null;
  readonly memoryBytes: number;
  readonly contextLength: number;
  /** Number of tokens currently in the cache. */
  readonly position: number;
  readonly reusedTokens: number;
  /** Logits after the last step that asked for them; overwritten by the next step. */
  readonly logits: Float32Array;

  /** Start a new sequence. */
  reset(): void;
  /** Keep only the first `position` tokens of the cache. */
  rewind(position: number): void;
  /** Feed one token; returns the logits view, or null when wantLogits is false. */
  step(token: number, wantLogits?: boolean): Float32Array | null;
  /** Feed several tokens after the current position; returns the logits after the last one. */
  prefill(tokens: ArrayLike<number>): Float32Array | null;
  /** Make the cache hold exactly `tokens`, reusing the common prefix with what it already holds. */
  feed(tokens: ArrayLike<number>): Float32Array;
  /**
   * The model read as an encoder: a unit vector (d_model values) pooled from the final hidden states of
   * `tokens`, 'mean' over every position (default) or the 'last' one. Used to rank the facts by cosine
   * (Retriever.search with a vector). Overwrites the context: the next feed() starts from an empty cache.
   */
  embed(tokens: ArrayLike<number>, pooling?: 'mean' | 'last'): Float32Array;

  /** Turn the inspection buffers on or off (both off by default; off costs nothing). */
  setCapture(options?: { attention?: boolean; norms?: boolean }): void;
  /** Attention of the last step; null until a step ran with attention capture on. */
  attention(): AttentionSnapshot | null;
  /** Residual norms of the last step; null until a step ran with norm capture on. */
  residualNorms(): Float32Array | null;

  /** Token-by-token generation. The generator returns the stop reason. */
  stream(promptTokens: ArrayLike<number>, options?: StreamOptions): Generator<GenerationStep, 'stop' | 'length', void>;
  generate(promptTokens: ArrayLike<number>, options?: GenerateOptions): GenerateResult;
}

export function argmax(logits: ArrayLike<number>): number;
/** The k most likely tokens under softmax(logits), most likely first. */
export function topTokens(logits: ArrayLike<number>, k?: number): TopToken[];
/** softmax(logits)[id] */
export function probabilityOf(logits: ArrayLike<number>, id: number): number;
/**
 * The distribution a token is sampled from: repetition penalty, temperature, top-k, top-p, in this order
 * (the same as GPT.filter_logits in the PyTorch model). Kept ids, most likely first, with probabilities.
 */
export function filterLogits(
  logits: ArrayLike<number>,
  options?: SamplingOptions,
  previous?: ArrayLike<number> | null,
): { ids: Int32Array; probs: Float64Array };
export function sampleFrom(distribution: { ids: Int32Array; probs: Float64Array }, random?: () => number): number;
export function chooseToken(
  logits: ArrayLike<number>,
  options?: SamplingOptions & { random?: () => number },
  previous?: ArrayLike<number> | null,
): number;
/**
 * chooseToken among the `allowed` token ids only (ascending). Greedy ties go to the lower id.
 * bias: one number per allowed token, added to its logit first (DecodingConstraint.bias).
 */
export function chooseAllowed(
  logits: ArrayLike<number>,
  allowed: ArrayLike<number>,
  options?: SamplingOptions & { random?: () => number },
  previous?: ArrayLike<number> | null,
  bias?: ArrayLike<number> | null,
): number;
/** Deterministic random number generator for reproducible sampling. */
export function seededRandom(seed: number): () => number;
