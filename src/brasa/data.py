"""Data contract, prompt format and training batches.

A data directory holds a knowledge base and question/answer records:

    kb/facts.jsonl   {"id", "topic", "subtopic", "kind", "pt", "en", "importance", "caveat"}
    kb/topics.json   [{"slug", "type", "title_pt", "title_en", "aliases": [...]}]
    qa/*.jsonl       {"id", "lang": "pt"|"en", "q", "a", "facts": ["fact-id", ...],
                      "kind": "single"|"multi"|"chitchat"|"oos"|"meta",
                      "split": "train"|"val"|"test", "key_spans": ["..."]}

The test split holds questions whose phrasings never appear in train (held-out paraphrases of the same
facts) plus out-of-scope questions never seen. load_dataset() validates all of this and fails loudly.
An optional corpus.json ({"name", "fictional", "description", ...}) describes the corpus; it is copied
into the run and the model card, so that numbers measured on a rehearsal corpus stay labelled as such.

Each QA record becomes several training examples in two formats:

    open-book    <|bos|><|ctx|>fact<|sep|>fact<|sep|>fact<|q|>question<|a|>answer<|eos|>
    closed-book  <|bos|><|q|>question<|a|>answer<|eos|>

Open-book contexts come from the real retriever (so the model sees the distractors it will see in the
browser) with a few augmentations: sometimes the gold fact is forced in, sometimes it is removed, the
order is shuffled, a random fact is swapped in. Questions get surface noise (case, accents, typos).
The loss is computed on the answer span; DataConfig.prompt_loss_weight gives the other tokens a weight.

    uv run python -m brasa.data --data-dir data/rehearsal          # validate and print statistics
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import BOS, CTX, EOS, SEP, A, DataConfig, PromptConfig, Q

LANGS = ("pt", "en")
KINDS = ("single", "multi", "chitchat", "oos", "meta")
SPLITS = ("train", "val", "test")


class DataError(ValueError):
    """The data directory does not respect the contract."""


# --------------------------------------------------------------------------------------- records
@dataclass(frozen=True)
class Fact:
    id: str
    topic: str
    subtopic: str
    kind: str
    pt: str
    en: str
    importance: float = 1.0
    caveat: str | None = None

    def text(self, lang: str) -> str:
        return self.pt if lang == "pt" else self.en


@dataclass(frozen=True)
class Topic:
    slug: str
    type: str
    title_pt: str
    title_en: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class QARecord:
    id: str
    lang: str
    q: str
    a: str
    facts: tuple[str, ...]
    kind: str
    split: str
    key_spans: tuple[str, ...] = ()


@dataclass
class Dataset:
    facts: dict[str, Fact]
    topics: dict[str, Topic]
    qa: list[QARecord]
    path: str = ""
    meta: dict = field(default_factory=dict)  # contents of the optional corpus.json

    def split(self, name: str) -> list[QARecord]:
        return [r for r in self.qa if r.split == name]

    def topic_of(self, record: QARecord) -> str:
        return self.facts[record.facts[0]].topic if record.facts else "(none)"

    def stats(self) -> dict:
        out: dict = {"corpus": self.meta, "facts": len(self.facts), "topics": len(self.topics), "qa": len(self.qa),
                     "splits": {}}
        for s in SPLITS:
            rows = self.split(s)
            out["splits"][s] = {
                "n": len(rows),
                "by_lang": dict(sorted(Counter(r.lang for r in rows).items())),
                "by_kind": dict(sorted(Counter(r.kind for r in rows).items())),
            }
        per_fact = Counter((f, r.lang) for r in self.split("train") for f in r.facts)
        if per_fact:
            counts = sorted(per_fact.values())
            out["train_questions_per_fact_and_lang"] = {
                "min": counts[0], "median": counts[len(counts) // 2], "max": counts[-1]}
        return out


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DataError(f"{path}:{n}: invalid JSON ({e})") from e
    return rows


def load_dataset(data_dir: str | Path) -> Dataset:
    """Read and validate a data directory. Raises DataError listing every problem found."""
    root = Path(data_dir)
    problems: list[str] = []
    for required in ("kb/facts.jsonl", "kb/topics.json", "qa"):
        if not (root / required).exists():
            raise DataError(f"{root / required} is missing (expected kb/facts.jsonl, kb/topics.json, qa/*.jsonl)")

    topics: dict[str, Topic] = {}
    for t in json.loads((root / "kb/topics.json").read_text(encoding="utf-8")):
        try:
            topic = Topic(t["slug"], t.get("type", ""), t["title_pt"], t["title_en"], tuple(t.get("aliases", [])))
        except KeyError as e:
            problems.append(f"topics.json: entry {t.get('slug', '?')!r} lacks {e}")
            continue
        if topic.slug in topics:
            problems.append(f"topics.json: duplicate slug {topic.slug!r}")
        topics[topic.slug] = topic

    facts: dict[str, Fact] = {}
    for row in _read_jsonl(root / "kb/facts.jsonl"):
        try:
            fact = Fact(row["id"], row["topic"], row.get("subtopic", ""), row.get("kind", "fact"), row["pt"], row["en"],
                        float(row.get("importance") or 1.0), row.get("caveat") or None)
        except KeyError as e:
            problems.append(f"facts.jsonl: fact {row.get('id', '?')!r} lacks {e}")
            continue
        if fact.id in facts:
            problems.append(f"facts.jsonl: duplicate id {fact.id!r}")
        if fact.topic not in topics:
            problems.append(f"facts.jsonl: fact {fact.id!r} has unknown topic {fact.topic!r}")
        if not fact.pt.strip() or not fact.en.strip():
            problems.append(f"facts.jsonl: fact {fact.id!r} needs both a pt and an en text")
        facts[fact.id] = fact

    qa: list[QARecord] = []
    seen_ids: set[str] = set()
    files = sorted((root / "qa").glob("*.jsonl"))
    if not files:
        problems.append("qa/: no .jsonl files")
    for path in files:
        for row in _read_jsonl(path):
            where = f"{path.name}: record {row.get('id', '?')!r}"
            try:
                rec = QARecord(row["id"], row["lang"], row["q"], row["a"], tuple(row.get("facts", [])), row["kind"],
                               row["split"], tuple(row.get("key_spans", [])))
            except KeyError as e:
                problems.append(f"{where} lacks {e}")
                continue
            if rec.id in seen_ids:
                problems.append(f"{where}: duplicate id")
            seen_ids.add(rec.id)
            if rec.lang not in LANGS:
                problems.append(f"{where}: lang must be one of {LANGS}")
            if rec.kind not in KINDS:
                problems.append(f"{where}: kind must be one of {KINDS}")
            if rec.split not in SPLITS:
                problems.append(f"{where}: split must be one of {SPLITS}")
            if not rec.q.strip() or not rec.a.strip():
                problems.append(f"{where}: empty question or answer")
            problems += [f"{where}: unknown fact {f!r}" for f in rec.facts if f not in facts]
            if rec.kind == "oos" and rec.facts:
                problems.append(f"{where}: out-of-scope records must not reference facts")
            qa.append(rec)

    # the held-out guarantee: a val/test phrasing must not be a training phrasing
    train_q = {(r.lang, _question_key(r.q)) for r in qa if r.split == "train"}
    for r in qa:
        if r.split != "train" and (r.lang, _question_key(r.q)) in train_q:
            problems.append(f"record {r.id!r} ({r.split}) repeats a training question: {r.q!r}")
    if problems:
        shown = "\n  ".join(problems[:25])
        more = f"\n  ... and {len(problems) - 25} more" if len(problems) > 25 else ""
        raise DataError(f"{root}: {len(problems)} problem(s) with the data contract:\n  {shown}{more}")
    meta_path = root / "corpus.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return Dataset(facts, topics, qa, str(root), meta)


def _question_key(q: str) -> str:
    return " ".join(strip_accents(q).lower().split()).rstrip("?!. ")


def tokenizer_texts(dataset: Dataset) -> list[str]:
    """Everything the tokenizer is trained on: the facts, topic names and the TRAINING questions/answers."""
    texts = [t for f in dataset.facts.values() for t in (f.pt, f.en)]
    texts += [t for topic in dataset.topics.values() for t in (topic.title_pt, topic.title_en, *topic.aliases)]
    for r in dataset.split("train"):
        texts += [r.q, r.a]
    return texts


# --------------------------------------------------------------------------------------- question noise
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def typo(s: str, rng: random.Random) -> str:
    """Swap, drop or double one inner letter of one word."""
    parts = s.split(" ")
    candidates = [i for i, w in enumerate(parts) if len(w) >= 4 and w.isalpha()]
    if not candidates:
        return s
    i = rng.choice(candidates)
    w = parts[i]
    j = rng.randrange(1, len(w) - 1)
    kind = rng.choice(("swap", "drop", "dup"))
    if kind == "swap":
        w = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    elif kind == "drop":
        w = w[:j] + w[j + 1 :]
    else:
        w = w[:j] + w[j] + w[j:]
    parts[i] = w
    return " ".join(parts)


def add_noise(q: str, rng: random.Random) -> str:
    """How people really type: lower case, no accents, no final punctuation, the odd typo."""
    for _ in range(8):
        out = q
        if rng.random() < 0.35:
            out = out.lower()
        if rng.random() < 0.35:
            out = strip_accents(out)
        if rng.random() < 0.35 and out and out[-1] in "?.!":
            out = out[:-1]
        if rng.random() < 0.25:
            out = typo(out, rng)
        if out != q and out.strip():
            return out
    return q.lower()


# --------------------------------------------------------------------------------------- prompt format
def build_prompt(tok, question: str, facts: list[str] | None, block_size: int,
                 cfg: PromptConfig | None = None, answer_reserve: int | None = None) -> tuple[list[int], int]:
    """Token ids of the prompt, ending in <|a|>, and the number of facts that fitted.

    facts=None (or empty) gives the closed-book format. Facts are added in the given order until the
    budget (block_size - answer_reserve - question) is used up. web/engine/prompt.js mirrors this."""
    cfg = cfg or PromptConfig()
    s = tok.special_ids
    q_ids = tok.encode(question)[: cfg.max_question_tokens]
    reserve = cfg.answer_reserve if answer_reserve is None else answer_reserve
    budget = block_size - reserve - len(q_ids) - 3  # <|bos|> <|q|> <|a|>
    ids, used = [s[BOS]], 0
    for text in (facts or [])[: cfg.max_facts]:
        f_ids = tok.encode(text)
        if len(f_ids) + 1 > budget:
            break
        ids.append(s[CTX] if used == 0 else s[SEP])
        ids.extend(f_ids)
        budget -= len(f_ids) + 1
        used += 1
    ids += [s[Q], *q_ids, s[A]]
    return ids, used


@dataclass
class Example:
    ids: list[int]  # the whole sequence, <|bos|> ... <|eos|>
    answer_start: int  # index of the first answer token; ids[answer_start:] is the span the loss covers
    record: str = ""  # QA record id
    open_book: bool = False
    gold_in_context: bool = False


def make_example(tok, question: str, answer: str, facts: list[str] | None, block_size: int,
                 cfg: PromptConfig | None = None) -> Example:
    """Prompt + answer + <|eos|>. The answer is truncated only if it cannot fit the context at all."""
    cfg = cfg or PromptConfig()
    a_ids = tok.encode(answer)
    reserve = max(cfg.answer_reserve, len(a_ids) + 1)
    prompt, _ = build_prompt(tok, question, facts, block_size, cfg, answer_reserve=reserve)
    a_ids = a_ids[: max(1, block_size - len(prompt) - 1)]
    return Example(prompt + a_ids + [tok.special_ids[EOS]], answer_start=len(prompt))


def training_context(record: QARecord, question: str, dataset: Dataset, retriever, rng: random.Random,
                     cfg: DataConfig) -> list[tuple[str, str]]:
    """(fact id, text) pairs for one open-book training example: retrieved facts plus augmentation."""
    k = cfg.prompt.max_facts
    ctx = [(h.id, h.text) for h in retriever.search(question, k)]
    gold = list(record.facts)
    r = rng.random()
    if gold and r < cfg.oracle_prob:  # the retriever found it (even if it really did not)
        for g in gold:
            if g not in {c[0] for c in ctx}:
                non_gold = [i for i, c in enumerate(ctx) if c[0] not in gold]
                if len(ctx) >= k and non_gold:
                    ctx.pop(non_gold[-1])
                ctx.append((g, dataset.facts[g].text(record.lang)))
    elif gold and r < cfg.oracle_prob + cfg.drop_gold_prob:  # the retriever missed: answer from memory
        ctx = [c for c in ctx if c[0] not in gold]
    if ctx and rng.random() < cfg.random_distractor_prob:
        non_gold = [i for i, c in enumerate(ctx) if c[0] not in gold]
        pool = [f for f in dataset.facts.values() if f.id not in gold and f.id not in {c[0] for c in ctx}]
        if non_gold and pool:
            f = rng.choice(pool)
            ctx[rng.choice(non_gold)] = (f.id, f.text(record.lang))
    if rng.random() < cfg.shuffle_prob:
        rng.shuffle(ctx)
    return ctx


# --------------------------------------------------------------------------------------- counterfactuals
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# code-like names: PyTorch, UFES, VGG16, ResNet34 (CamelCase, ALL CAPS, letters with digits)
_NAME = re.compile(r"\b(?:[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Z]{2,}[A-Za-z0-9-]*|[A-Za-z]+[0-9]+[A-Za-z0-9-]*)\b")


def _number_like(s: str, rng: random.Random) -> str:
    """Another number with the same shape: same digit count, same separators, no leading zero."""
    out, first = [], True
    for ch in s:
        if ch.isdigit():
            out.append(str(rng.randrange(1 if first else 0, 10)))
            first = False
        else:
            out.append(ch)
    return "".join(out)


def name_pool(dataset: Dataset) -> dict[str, list[str]]:
    """Every code-like name that occurs in the knowledge base, per language."""
    return {lang: sorted({m.group(0) for f in dataset.facts.values() for m in _NAME.finditer(f.text(lang))})
            for lang in LANGS}


def counterfactual(text: str, question: str, rng: random.Random, names: list[str]) -> str:
    """The same sentence with other numbers and, sometimes, one other name: false on purpose.

    Used only inside training contexts, with the answer rewritten in the same way. Since the values are
    different in every example, the only way to get the answer right is to copy it from the context, which
    is the behaviour wanted at inference time (where the context holds the true fact). Numbers and names
    that the question itself mentions are left alone, so the question still points at this fact, and the
    first word (usually the subject) is never replaced."""
    q = strip_accents(question).lower()
    out = _NUMBER.sub(lambda m: m.group(0) if m.group(0) in question else _number_like(m.group(0), rng), text)
    candidates = [m for m in _NAME.finditer(out) if m.start() > 0 and strip_accents(m.group(0)).lower() not in q]
    if candidates and names and (out == text or rng.random() < 0.5):
        m = rng.choice(candidates)
        other = rng.choice(names)
        if other != m.group(0):
            out = out[: m.start()] + other + out[m.end() :]
    return out


def build_examples(dataset: Dataset, split: str, tok, retriever, cfg: DataConfig, block_size: int,
                   seed: int = 0) -> tuple[list[Example], dict]:
    """Turn the QA records of a split into training examples. Deterministic for a given seed.

    Variant 0 of every record is the clean closed-book example and variant 1 the clean open-book example
    with the retriever's own context; the remaining variants are randomised."""
    examples: list[Example] = []
    open_with_gold = gold_hits = n_counterfactual = 0
    names = name_pool(dataset) if cfg.counterfactual_prob > 0 else {}
    for rec in dataset.split(split):
        for v in range(max(1, cfg.variants)):
            rng = random.Random(f"{seed}:{rec.id}:{v}")
            question = add_noise(rec.q, rng) if v >= 2 and rng.random() < cfg.noise_prob else rec.q
            open_book = retriever is not None and (v == 1 or (v >= 2 and rng.random() < cfg.open_book_prob))
            facts, gold_in = None, False
            if open_book:
                if v == 1:
                    ctx = [(h.id, h.text) for h in retriever.search(question, cfg.prompt.max_facts)]
                else:
                    ctx = training_context(rec, question, dataset, retriever, rng, cfg)
                facts = [text for _, text in ctx]
                gold_in = bool(rec.facts) and set(rec.facts) <= {i for i, _ in ctx}
                open_with_gold += bool(rec.facts)
                gold_hits += gold_in
            answer = rec.a
            if (v >= 2 and gold_in and len(rec.facts) == 1 and cfg.counterfactual_prob > 0
                    and rng.random() < cfg.counterfactual_prob):
                gold = rec.facts[0]
                original = dataset.facts[gold].text(rec.lang)
                changed = counterfactual(original, question, rng, names[rec.lang]) if rec.a == original else original
                if changed != original:
                    facts = [changed if i == gold else text for i, text in ctx]
                    answer = changed
                    n_counterfactual += 1
            ex = make_example(tok, question, answer, facts, block_size, cfg.prompt)
            ex.record, ex.open_book, ex.gold_in_context = rec.id, open_book, gold_in
            examples.append(ex)
    lengths = sorted(len(e.ids) for e in examples) or [0]
    n_open = sum(e.open_book for e in examples)
    stats = {
        "examples": len(examples),
        "open_book": n_open,
        "closed_book": len(examples) - n_open,
        "open_book_gold_in_context": round(gold_hits / max(1, open_with_gold), 4),
        "counterfactual": n_counterfactual,
        "tokens": sum(lengths),
        "answer_tokens": sum(len(e.ids) - e.answer_start for e in examples),
        "length_median": lengths[len(lengths) // 2],
        "length_p95": lengths[int(0.95 * (len(lengths) - 1))],
        "length_max": lengths[-1],
    }
    return examples, stats


# --------------------------------------------------------------------------------------- batches
@dataclass
class BatchSampler:
    """Token-budget batches of one example per row, bucketed by length.

    Every example starts at position 0, exactly as at inference time. Rows are padded to the width of
    their bucket and a batch holds batch_tokens / width rows, so short examples are not drowned in
    padding and every step costs about the same. A bucket is chosen with probability proportional to
    its token mass, which makes every example equally likely per step. batch(step) is a pure function of
    (seed, step): resuming a run replays exactly the batches it would have seen."""

    examples: list[Example]
    block_size: int
    batch_tokens: int
    pad_id: int
    seed: int = 0
    prompt_loss_weight: float = 0.0
    widths: tuple[int, ...] = (32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
    buckets: list[dict] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        widths = sorted({w for w in self.widths if w < self.block_size} | {self.block_size})
        groups: dict[int, list[Example]] = {}
        for ex in self.examples:
            n = len(ex.ids) - 1  # inputs are ids[:-1], targets ids[1:]
            if n > self.block_size:
                raise ValueError(f"example of {n + 1} tokens does not fit block_size {self.block_size}")
            groups.setdefault(next(w for w in widths if w >= n), []).append(ex)
        for width, group in sorted(groups.items()):
            ids = np.full((len(group), width + 1), self.pad_id, dtype=np.int64)
            weight = np.zeros((len(group), width + 1), dtype=np.float32)
            for i, ex in enumerate(group):
                ids[i, : len(ex.ids)] = ex.ids
                weight[i, : ex.answer_start] = self.prompt_loss_weight
                weight[i, ex.answer_start : len(ex.ids)] = 1.0
            self.buckets.append({"width": width, "ids": ids, "weight": weight,
                                 "rows": max(1, self.batch_tokens // width)})
        self._mass = [len(b["ids"]) * b["width"] for b in self.buckets]

    def batch(self, step: int):
        """Returns numpy arrays (x, y, w): inputs, targets (-100 where nothing is learned), weights
        (None when the loss is the plain mean over the answer spans)."""
        rng = random.Random(self.seed * 1_000_003 + step)
        b = rng.choices(self.buckets, weights=self._mass)[0]
        n = len(b["ids"])
        rows = rng.sample(range(n), b["rows"]) if b["rows"] <= n else rng.choices(range(n), k=b["rows"])
        ids, weight = b["ids"][rows], b["weight"][rows]
        x, y, w = ids[:, :-1], ids[:, 1:].copy(), weight[:, 1:]
        y[w == 0] = -100
        return x, y, (w if self.prompt_loss_weight > 0 else None)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Validate a data directory and print its statistics.")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    dataset = load_dataset(args.data_dir)
    print(f"{args.data_dir}: contract OK")
    print(json.dumps(dataset.stats(), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
