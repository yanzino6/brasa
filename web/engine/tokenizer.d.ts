/** Contents of tokenizer.json (also embedded in the model file header). */
export interface TokenizerSpec {
  type: 'bpe';
  version: number;
  /** Merge i creates token 256 + i from the pair [left id, right id]. */
  merges: [number, number][];
  /** Control tokens; their ids follow the merges, in this order. */
  specials: string[];
  vocab_size: number;
}

export interface TokenSpan {
  id: number;
  /** Byte offsets into the UTF-8 encoding of the normalised text. */
  start: number;
  end: number;
  /** The token on its own; U+FFFD where it holds only part of a multi-byte character. */
  text: string;
}

export interface EncodeOptions {
  /** Parse "<|q|>", "<|a|>", ... as control tokens. Only for trusted text. Default false. */
  allowSpecial?: boolean;
}

export interface DecodeOptions {
  /** Drop control tokens from the output. Default true. */
  skipSpecial?: boolean;
}

/** NFC, every whitespace run collapsed to one space, ends trimmed. */
export function normalize(text: string): string;

export class Tokenizer {
  constructor(spec: TokenizerSpec);
  readonly spec: TokenizerSpec;
  readonly vocabSize: number;
  /** Number of byte and merge tokens; ids from here on are control tokens. */
  readonly nPlain: number;
  readonly specials: string[];
  /** e.g. specialIds['<|eos|>'] */
  readonly specialIds: Record<string, number>;

  isSpecial(id: number): boolean;
  encode(text: string, options?: EncodeOptions): number[];
  /** Encode and keep the token boundaries. */
  tokenize(text: string, options?: EncodeOptions): { text: string; tokens: TokenSpan[] };
  decode(ids: ArrayLike<number> | Iterable<number>, options?: DecodeOptions): string;
  decodeBytes(ids: Iterable<number>, options?: DecodeOptions): Uint8Array;
  tokenText(id: number): string;
  /** Incremental decoder: push() returns the text that became complete; push([], true) flushes. */
  streamDecoder(options?: DecodeOptions): { push(ids: number | number[], final?: boolean): string };
}
