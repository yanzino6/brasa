import type { Tokenizer } from './tokenizer.js';

/** Prompt settings a model was trained with (engine.prompt, manifest.json "prompt"). */
export interface PromptConfig {
  /** Longer questions are truncated. */
  max_question_tokens: number;
  /** Tokens kept free for the answer when facts are packed into the context. */
  answer_reserve: number;
  /** Facts shown in open-book mode. */
  max_facts: number;
}

export const SPECIALS: Readonly<{
  pad: '<|pad|>'; bos: '<|bos|>'; eos: '<|eos|>'; ctx: '<|ctx|>'; sep: '<|sep|>'; q: '<|q|>'; a: '<|a|>';
}>;

export const DEFAULT_PROMPT: Readonly<PromptConfig>;

/**
 * open-book:   <|bos|><|ctx|>fact<|sep|>fact<|q|>question<|a|>
 * closed-book: <|bos|><|q|>question<|a|>            (facts null or empty)
 * Returns the token ids and how many facts fitted into the context budget.
 */
export function buildPrompt(
  tok: Tokenizer,
  question: string,
  facts: string[] | null,
  blockSize: number,
  cfg?: PromptConfig,
): { ids: number[]; used: number };
