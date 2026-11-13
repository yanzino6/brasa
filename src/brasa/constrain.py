"""Decoding constrained to the knowledge base: the model may only write sentences that exist.

Free generation lets a model of a few million parameters blend two sibling facts or invent a number.
Here the candidates of every step are the tokens that keep the output a prefix of an allowed answer:

    a fact (in one language),
    two or three facts of the same language joined by a space (the format of multi-fact answers),
    or one of the fixed answers (greetings, "I can't answer that", ...), alone.

The allowed answers are stored in a token trie (AnswerTrie). The model still decides everything that is
a decision (which fact, whether to go on, whether to decline); it can no longer write a sentence nobody
curated. web/engine/retrieval.js holds the same trie and web/engine/engine.js the same greedy rule;
fixtures in every export compare them.

The retriever's opinion as a prior (optional). Every allowed sentence may carry a prior, a number of nats
added to the model's log-probability of that sentence: 0 for the retriever's first hit, less for the facts
below it (ranking_priors). Decoding applies it one token at a time: a token is worth the best prior among the
sentences it still leads to, so the comparison between two tokens is the comparison between the best
sentences behind them, and no extra forward pass is needed. With every prior at 0 nothing changes.

Two search rules (constrained_answers):
    beam = 1   greedy: the most likely allowed token at every branch. Positions with a single allowed
               token are not decisions and cost no forward pass.
    beam > 1   beam search on the sum of the model's log-probabilities of the allowed tokens (not
               renormalised), i.e. an approximation of the most probable allowed answer.

    uv run python -m brasa.evaluate --run runs/brasa-12m --modes open_book_constrained --beam 4
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import EOS, PAD

State = tuple[tuple[int, int], ...]  # ((trie node, sentences finished before this one), ...); () = done
_NO_PRIOR = float("-inf")
PRIOR_KINDS = ("rank", "score")


def ranking_priors(hits, weight: float, kind: str = "rank") -> dict[str, float]:
    """{fact id: prior in nats} from the retriever's ranking of the facts in the prompt (best first).

    "rank":  -weight * position (0 for the first hit, -weight for the second, ...).
    "score": -weight * (1 - score / best score): how far the retriever itself puts a fact below its first
             hit, so a near tie costs almost nothing and a distant fourth costs up to `weight`. Falls back to
             "rank" when the best score is not positive.
    weight 0 gives no priors at all ({}), i.e. the model alone decides."""
    if kind not in PRIOR_KINDS:
        raise ValueError(f"unknown prior {kind!r}; choose from {PRIOR_KINDS}")
    if not weight or not hits:
        return {}
    top = hits[0].score
    if kind == "score" and top > 0:
        return {h.id: -weight * (1.0 - h.score / top) for h in hits}
    return {h.id: -weight * i for i, h in enumerate(hits)}


def continuation_ids(tok, text: str) -> list[int]:
    """Token ids of `text` when it follows another sentence after one space.

    The tokenizer trims its input and never merges across a space, so the ids of "A B" are the ids of
    "A" followed by the ids B gets after any other word."""
    return tok.encode("x " + text)[len(tok.encode("x")) :]


class AnswerTrie:
    """Token trie of the answers the model is allowed to write.

    add(tok, text)             a fixed answer: allowed alone, nothing may follow it
    add(tok, text, group="pt") a sentence that may be followed by other sentences of the same group, up
                               to max_sentences in one answer
    add(..., prior=-1.5)       the sentence also carries a prior, in nats (see bias)

    A state is a set of (node, sentences finished) pairs, because one token sequence can be read in two
    ways when a sentence is a prefix of another ("A." then a new sentence, or the longer sentence)."""

    def __init__(self, eos: int, max_sentences: int = 3):
        self.eos = eos
        self.max_sentences = max(1, max_sentences)
        self.children: list[dict[int, int]] = [{}]  # node 0: the start of an answer
        self.ends: dict[int, set[str | None]] = {}  # node -> groups of the sentences that end there
        self.next_root: dict[str, int] = {}  # group -> node where a following sentence of the group starts
        self.sentences = 0
        self.has_prior = False  # True once a sentence was added with a prior other than 0
        self.best: list[float] = [_NO_PRIOR]  # node -> the best prior among the sentences that pass through it
        self.end_prior: dict[int, float] = {}  # node -> the best prior among the sentences that end there
        self._allowed: dict[State, list[int]] = {}
        self._bias: dict[State, list[float]] = {}

    def _insert(self, node: int, ids: list[int], group: str | None, prior: float) -> None:
        self.best[node] = max(self.best[node], prior)
        for t in ids:
            nxt = self.children[node].get(t)
            if nxt is None:
                nxt = len(self.children)
                self.children.append({})
                self.best.append(_NO_PRIOR)
                self.children[node][t] = nxt
            node = nxt
            self.best[node] = max(self.best[node], prior)
        self.ends.setdefault(node, set()).add(group)
        self.end_prior[node] = max(self.end_prior.get(node, _NO_PRIOR), prior)

    def add(self, tok, text: str, group: str | None = None, prior: float = 0.0) -> None:
        ids = tok.encode(text)
        if not ids:
            return
        self._allowed.clear()
        self._bias.clear()
        self.sentences += 1
        self.has_prior = self.has_prior or prior != 0.0
        self._insert(0, ids, group, prior)
        if group is not None and self.max_sentences > 1:
            if group not in self.next_root:
                self.next_root[group] = len(self.children)
                self.children.append({})
                self.best.append(_NO_PRIOR)
            self._insert(self.next_root[group], continuation_ids(tok, text), group, 0.0)  # see bias

    @classmethod
    def build(cls, tok, sentences: list[tuple], max_sentences: int = 3) -> AnswerTrie:
        """sentences: (text, group) or (text, group, prior); group None marks a fixed answer."""
        trie = cls(tok.special_ids[EOS], max_sentences)
        for text, group, *prior in sentences:
            trie.add(tok, text, group, prior[0] if prior else 0.0)
        return trie

    @classmethod
    def from_index(cls, index: dict, tok, lang: str | None = None, max_sentences: int | None = None,
                   only: set[str] | None = None, priors: dict[str, float] | None = None,
                   fixed_prior: float = 0.0) -> AnswerTrie:
        """The trie of a retrieval index (retrieval.json): its facts, chained per language, and its
        canonical answers. lang keeps one language; only keeps these fact ids (and the canonical answers).
        priors: {fact id: prior in nats} (see ranking_priors; a fact that is not named gets 0);
        fixed_prior: the prior of every canonical answer."""
        answers = index.get("answers") or {}
        priors = priors or {}
        sentences: list[tuple] = [
            (d["text"], d["lang"], priors.get(d["id"], 0.0)) for d in index["docs"]
            if (lang is None or d["lang"] == lang) and (only is None or d["id"] in only)]
        sentences += [(c["text"], None, fixed_prior) for c in answers.get("canonical", [])
                      if lang is None or c["lang"] == lang]
        return cls.build(tok, sentences, max_sentences or answers.get("max_sentences", 3))

    # -- walking --------------------------------------------------------------------------------
    def start(self) -> State:
        return ((0, 0),)

    def _moves(self, state: State):
        """(token -> next (node, count)) for every reading of the state."""
        for node, count in state:
            for t, child in self.children[node].items():
                yield t, (child, count)
            if count + 1 < self.max_sentences:
                for group in self.ends.get(node, ()):
                    if group is not None:
                        for t, child in self.children[self.next_root[group]].items():
                            yield t, (child, count + 1)

    def allowed(self, state: State) -> list[int]:
        """Token ids that may come next, ascending. <|eos|> is allowed where a sentence ends."""
        out = self._allowed.get(state)
        if out is None:
            tokens = {t for t, _ in self._moves(state)}
            if any(node in self.ends for node, _ in state):
                tokens.add(self.eos)
            out = self._allowed[state] = sorted(tokens)
        return out

    def bias(self, state: State) -> list[float]:
        """What the priors add to the log-probability of each token of allowed(state), in the same order.

        The prior is about the FIRST sentence of an answer: a token inside it is worth the best prior among
        the sentences it still leads to, and <|eos|> is worth the prior of the sentence that ends here. A
        token that starts a following sentence is worth the same as <|eos|>, and the sentences that follow
        carry no prior: whether to go on, and with which fact, stays the model's decision (a prior on those
        would make every multi-fact answer more expensive than stopping). Only differences between the
        tokens of one step matter. All zeros when no sentence carries a prior."""
        out = self._bias.get(state)
        if out is None:
            worth: dict[int, float] = {}
            for node, count in state:
                for t, child in self.children[node].items():
                    worth[t] = max(worth.get(t, _NO_PRIOR), self.best[child])
                if node in self.ends:
                    done = self.end_prior[node]
                    worth[self.eos] = max(worth.get(self.eos, _NO_PRIOR), done)
                    if count + 1 < self.max_sentences:
                        for group in self.ends[node]:
                            if group is not None:
                                for t in self.children[self.next_root[group]]:
                                    worth[t] = max(worth.get(t, _NO_PRIOR), done)
            out = self._bias[state] = [worth[t] for t in self.allowed(state)]
        return out

    def advance(self, state: State, token: int) -> State:
        """The state after `token`; () after <|eos|>. Raises when the token is not allowed."""
        if token == self.eos and any(node in self.ends for node, _ in state):
            return ()
        nxt = tuple(sorted({pair for t, pair in self._moves(state) if t == token}))
        if not nxt:
            raise ValueError(f"token {token} is not allowed here")
        return nxt

    def accepts(self, ids: list[int]) -> bool:
        """True when ids (without <|eos|>) is exactly one allowed answer."""
        state = self.start()
        for t in ids:
            if t not in self.allowed(state) or t == self.eos:
                return False
            state = self.advance(state, t)
        return any(node in self.ends for node, _ in state)

    def describe(self) -> dict:
        return {"sentences": self.sentences, "nodes": len(self.children), "max_sentences": self.max_sentences,
                "groups": sorted(self.next_root)}


# --------------------------------------------------------------------------------------- decoding
@dataclass
class _Hyp:
    tokens: list[int]
    state: State
    score: float = 0.0
    done: bool = False  # ended with <|eos|> (or ran out of budget)


def _last_logprobs(model, sequences: list[list[int]], pad: int, device, max_tokens: int) -> list[torch.Tensor]:
    """log-softmax of the next-token logits after each sequence, as float32 CPU vectors."""
    out: list[torch.Tensor | None] = [None] * len(sequences)
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and (end - start + 1) * len(sequences[order[end]]) <= max_tokens:
            end += 1
        rows = order[start:end]
        width = len(sequences[rows[-1]])
        buf = torch.full((len(rows), width), pad, dtype=torch.long)
        for r, i in enumerate(rows):
            buf[r, : len(sequences[i])] = torch.tensor(sequences[i])
        logits, _, _ = model(buf.to(device))
        last = torch.tensor([len(sequences[i]) - 1 for i in rows], device=device)
        picked = logits[torch.arange(len(rows), device=device), last].float().log_softmax(-1).cpu()
        for r, i in enumerate(rows):
            out[i] = picked[r]
        start = end
    return out  # type: ignore[return-value]


@torch.no_grad()
def constrained_answers(model, tok, prompts: list[list[int]], tries, max_new_tokens: int,
                        device: torch.device | str, beam: int = 1, max_tokens: int = 60000,
                        return_ids: bool = False):
    """Answers restricted to the allowed sentences of a trie (one trie for all prompts, or one each).

    Causal attention makes right-padding invisible, so every round re-runs the open sequences as one
    padded batch (the same trick as evaluate.generate_answers). Ties go to the lower token id. When a trie
    carries priors, AnswerTrie.bias is added to the log-probabilities of the allowed tokens at every decision
    (meant for greedy decoding: a beam would count the prior of a sentence once per branching point)."""
    model.eval()
    pad, eos = tok.special_ids[PAD], tok.special_ids[EOS]
    block = model.cfg.block_size
    if isinstance(tries, AnswerTrie):
        tries = [tries] * len(prompts)
    budgets = [max(0, min(max_new_tokens, block - len(p))) for p in prompts]

    def extend(q: int, hyp: _Hyp, token: int, logp: float) -> _Hyp:
        trie = tries[q]
        new = _Hyp(hyp.tokens + [token], trie.advance(hyp.state, token), hyp.score + logp)
        if beam == 1:  # a single allowed token is not a decision: take it without asking the model
            while new.state and len(new.tokens) < budgets[q]:
                options = trie.allowed(new.state)
                if len(options) != 1:
                    break
                new.tokens.append(options[0])
                new.state = trie.advance(new.state, options[0])
        new.done = not new.state or len(new.tokens) >= budgets[q]
        return new

    active: list[list[_Hyp]] = [[_Hyp([], t.start(), done=b == 0)] for t, b in zip(tries, budgets, strict=True)]
    finished: list[list[_Hyp]] = [[] for _ in prompts]
    for q, hyps in enumerate(active):
        finished[q] += [h for h in hyps if h.done]
        active[q] = [h for h in hyps if not h.done]
    while True:
        rows = [(q, h) for q, hyps in enumerate(active) for h in hyps]
        if not rows:
            break
        logps = _last_logprobs(model, [prompts[q] + h.tokens for q, h in rows], pad, device, max_tokens)
        candidates: list[list[tuple[float, int, int]]] = [[] for _ in prompts]  # (score, row, token)
        for r, (q, h) in enumerate(rows):
            options = tries[q].allowed(h.state)
            values = logps[r][options].tolist()
            if tries[q].has_prior:
                values = [v + b for v, b in zip(values, tries[q].bias(h.state), strict=True)]
            if beam == 1:
                best = max(range(len(options)), key=lambda i: (values[i], -options[i]))
                candidates[q].append((h.score + values[best], r, options[best]))
            else:
                candidates[q] += [(h.score + v, r, t) for v, t in zip(values, options, strict=True)]
        for q, cands in enumerate(candidates):
            if not cands:
                continue
            cands.sort(key=lambda c: (-c[0], c[1], c[2]))
            active[q] = []
            for score, r, token in cands[:beam]:
                new = extend(q, rows[r][1], token, score - rows[r][1].score)
                (finished[q] if new.done else active[q]).append(new)
            # every further token can only lower a score: stop when a finished answer leads
            if finished[q] and active[q] and max(h.score for h in finished[q]) >= max(h.score for h in active[q]):
                active[q] = []

    answers, ids = [], []
    for q in range(len(prompts)):
        pool = finished[q] or active[q]
        best = max(pool, key=lambda h: h.score) if beam > 1 else pool[0]
        gen = best.tokens[: best.tokens.index(eos)] if eos in best.tokens else best.tokens
        ids.append(gen)
        answers.append(tok.decode(gen).strip())
    return (answers, ids) if return_ids else answers
