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
  params: { k1: number; b: number; word_weight: number; char_weight: number; ngram: number };
  docs: RetrievalDoc[];
  /** feature -> [doc index, weight, doc index, weight, ...] */
  postings: Record<string, number[]>;
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

/** Anything with this method can replace the lexical retriever (the seam for a learned one). */
export interface RetrieverLike {
  search(query: string, k?: number, lang?: 'pt' | 'en' | null): Hit[];
}

/** Lower-case and strip accents. */
export function fold(text: string): string;
export function words(text: string): string[];
export function charNgrams(word: string, n?: number): string[];
/** Unique features of a query: "w:<word>" first, then "c:<n-gram>". */
export function queryFeatures(text: string, n?: number): string[];

export class Retriever implements RetrieverLike {
  constructor(index: RetrievalIndex);
  readonly index: RetrievalIndex;
  readonly docs: RetrievalDoc[];
  /** One score per document. */
  scores(query: string): Float64Array;
  /** The k best facts, each once, best first. Default k = 4, any language. */
  search(query: string, k?: number, lang?: 'pt' | 'en' | null): Hit[];
}
