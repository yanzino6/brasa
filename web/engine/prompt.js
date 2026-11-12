// Prompt format: a port of build_prompt() in src/brasa/data.py. No dependencies.
//
//   open-book    <|bos|><|ctx|>fact<|sep|>fact<|sep|>fact<|q|>question<|a|>      (the model writes the answer)
//   closed-book  <|bos|><|q|>question<|a|>
//
// The numbers below are the training defaults; an exported model carries its own copy in the file header
// (engine.prompt) and in manifest.json, and that copy should be passed here.

export const SPECIALS = Object.freeze({
  pad: '<|pad|>', bos: '<|bos|>', eos: '<|eos|>', ctx: '<|ctx|>', sep: '<|sep|>', q: '<|q|>', a: '<|a|>',
});

export const DEFAULT_PROMPT = Object.freeze({ max_question_tokens: 96, answer_reserve: 96, max_facts: 4 });

/**
 * Token ids of the prompt, ending in <|a|>, and the number of facts that fitted.
 * Facts are added in the given order until the budget (blockSize - answer_reserve - question) is used up;
 * no facts (null or []) gives the closed-book format.
 *
 * @param {import('./tokenizer.js').Tokenizer} tok
 * @param {string} question
 * @param {string[] | null} facts
 * @param {number} blockSize  context length of the model
 * @param {{ max_question_tokens: number, answer_reserve: number, max_facts: number }} [cfg]
 * @returns {{ ids: number[], used: number }}
 */
export function buildPrompt(tok, question, facts, blockSize, cfg = DEFAULT_PROMPT) {
  const s = tok.specialIds;
  const qIds = tok.encode(question).slice(0, cfg.max_question_tokens);
  let budget = blockSize - cfg.answer_reserve - qIds.length - 3; // <|bos|> <|q|> <|a|>
  const ids = [s[SPECIALS.bos]];
  let used = 0;
  for (const text of (facts || []).slice(0, cfg.max_facts)) {
    const fIds = tok.encode(text);
    if (fIds.length + 1 > budget) break;
    ids.push(used === 0 ? s[SPECIALS.ctx] : s[SPECIALS.sep]);
    for (const id of fIds) ids.push(id);
    budget -= fIds.length + 1;
    used += 1;
  }
  ids.push(s[SPECIALS.q]);
  for (const id of qIds) ids.push(id);
  ids.push(s[SPECIALS.a]);
  return { ids, used };
}
