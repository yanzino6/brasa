import type { DecodingConstraint } from './engine.js';
import type { Tokenizer } from './tokenizer.js';

/** One (fact, language) document of the index. */
export interface RetrievalDoc {
  id: string;
  lang: 'pt' | 'en';
  topic: string;
  text: string;
}

/** Contents of retrieval.json, built by src/brasa/retrieval.py. */
export interface RetrievalIndex {
  version: 1;
  type: 'bm25';
  /** fields: the weights of the BM25 fields that were added at build time (absent: one field). */
  params: { k1: number; b: number; word_weight: number; char_weight: number; ngram: number; fields?: Record<string, number> };
  docs: RetrievalDoc[];
  /** feature -> [doc index, weight, doc index, weight, ...] */
  postings: Record<string, number[]>;
  /**
   * The fixed answers of the corpus (greetings, refusals), how many facts one answer may chain and, when the
   * release chose one on validation questions, the default prior of constrained answers (DecodingPrior).
   */
  answers?: { max_sentences: number; canonical: { lang: 'pt' | 'en'; text: string }[]; prior?: DecodingPrior };
  /** The learned half, when the model also ranks the facts: one int8 vector per document. */
  dense?: DenseSection;
}

/**
 * Fact vectors produced by the model read as an encoder (src/brasa/encoder.py, Engine.embed).
 * A search that is given the vector of the question scores a document
 * bm25 / (best bm25 of the query) + weight * cosine.
 */
export interface DenseSection {
  version: 1;
  /** Values per vector (the model's d_model). */
  dim: number;
  /** How the question has to be pooled: Engine.embed(tokens, pooling). */
  pooling: 'mean' | 'last';
  /** How much the cosine counts next to the normalised lexical score. */
  weight: number;
  /** Longest question, in tokens, including <|bos|><|q|> and <|a|>. */
  max_tokens: number;
  /** vector = int8 values * scale. */
  scale: number;
  /** base64 of docs.length * dim int8 values, document by document. */
  vectors: string;
  /** The model file whose weights computed the vectors ('i8', 'f16'). */
  weights?: string;
}

/**
 * How much the retriever's ranking weighs when the model chooses among the retrieved facts. Each fact gets a
 * prior, in nats, added to the model's log-probability of that sentence (see rankingPriors):
 *   kind 'rank'   -weight * position in the ranking (0 for the first hit)
 *   kind 'score'  -weight * (1 - score / best score): a near tie costs almost nothing
 * `fixed` is the prior of every fixed answer (greetings, refusals): above 0 the model declines more often.
 */
export interface DecodingPrior {
  kind: 'rank' | 'score';
  weight: number;
  fixed: number;
}

export interface Hit {
  /** Fact id. */
  id: string;
  /** Language of the version of the fact that matched. */
  lang: 'pt' | 'en';
  score: number;
  /** The fact text shown to the model. */
  text: string;
  topic: string;
  /** Index of the document inside the index. */
  doc: number;
}

/** Anything with this method can replace the retriever. `vector` is the question as the encoder sees it. */
export interface RetrieverLike {
  search(query: string, k?: number, lang?: 'pt' | 'en' | null, vector?: ArrayLike<number> | null): Hit[];
}

/** Lower-case and strip accents. */
export function fold(text: string): string;
export function words(text: string): string[];
export function charNgrams(word: string, n?: number): string[];
/** Unique features of a query: "w:<word>" first, then "c:<n-gram>". */
export function queryFeatures(text: string, n?: number): string[];

/** The question vector as integers, floor(v * 32767 + 0.5): the dot products are then exact. */
export function quantizeQuery(vector: ArrayLike<number>): Int32Array;

export class Retriever implements RetrieverLike {
  constructor(index: RetrievalIndex);
  readonly index: RetrievalIndex;
  readonly docs: RetrievalDoc[];
  /** The decoded dense section of the index, or null when it has none. */
  readonly dense: {
    dim: number; pooling: 'mean' | 'last'; weight: number; maxTokens: number; factor: number; vectors: Int8Array;
  } | null;
  /** One score per document: BM25, or the fusion with the cosine when a question vector is given. */
  scores(query: string, vector?: ArrayLike<number> | null): Float64Array;
  /**
   * The k best facts, each once, best first. Default k = 4, any language. With `vector` (and a dense
   * section) the ranking is the fusion and every document is a candidate; without it, the lexical ranking.
   */
  search(query: string, k?: number, lang?: 'pt' | 'en' | null, vector?: ArrayLike<number> | null): Hit[];
}

export const PRIOR_KINDS: readonly ['rank', 'score'];

/**
 * The retriever's ranking of the facts in the prompt (best first) as priors: fact id -> nats. An empty Map
 * when weight is 0. A port of ranking_priors() in src/brasa/constrain.py.
 */
export function rankingPriors(hits: readonly Pick<Hit, 'id' | 'score'>[], weight: number, kind?: 'rank' | 'score'): Map<string, number>;

/** Token ids of `text` when it follows another sentence after one space. */
export function continuationIds(tok: Tokenizer, text: string): number[];

/** State of an AnswerTrie walk: sorted numbers node * maxSentences + sentencesFinished; [] after <|eos|>. */
export type TrieState = number[];

/**
 * Token trie of the answers the model may write under constrained decoding: a fact, two or three facts
 * of one language in a row, or one fixed answer alone. A port of src/brasa/constrain.py.
 */
export class AnswerTrie implements DecodingConstraint<TrieState> {
  constructor(eos: number, maxSentences?: number);
  /**
   * Facts (chained per language) and fixed answers of an index; lang keeps one language, only keeps
   * the facts with these ids (the ones a question retrieved) next to the fixed answers. priors gives
   * facts a prior by id (rankingPriors) and fixedPrior is the prior of every fixed answer.
   */
  static fromIndex(index: RetrievalIndex, tok: Tokenizer, lang?: 'pt' | 'en' | null, maxSentences?: number | null,
    only?: Iterable<string> | null, priors?: Map<string, number> | Record<string, number> | null,
    fixedPrior?: number): AnswerTrie;
  readonly eos: number;
  readonly maxSentences: number;
  /** True once a sentence was added with a prior other than 0: decoding then adds bias(state) to the logits. */
  readonly hasPrior: boolean;
  /**
   * group null: a fixed answer that stands alone; otherwise it may chain with sentences of its group.
   * prior: nats added to the model's log-probability of this sentence when it comes first in an answer.
   */
  add(tok: Tokenizer, text: string, group?: string | null, prior?: number): void;
  start(): TrieState;
  allowed(state: TrieState): Int32Array;
  /**
   * What the priors add to each token of allowed(state), in the same order: the best prior among the
   * sentences a token still leads to. The sentences that follow the first one carry no prior.
   */
  bias(state: TrieState): Float64Array;
  advance(state: TrieState, token: number): TrieState;
  /** True when ids (without <|eos|>) is exactly one allowed answer. */
  accepts(ids: ArrayLike<number>): boolean;
  describe(): { sentences: number; nodes: number; maxSentences: number; groups: string[] };
}
