"""Evaluate a trained model by generating answers and scoring them.

    uv run python -m brasa.evaluate --run runs/brasa-12m
    uv run python -m brasa.evaluate --run runs/brasa-12m --model-card model-card.json     # the release card
    uv run python -m brasa.evaluate --run runs/brasa-12m --tune-prior                     # see tune_prior()

Every question is answered in several modes: open-book (facts from the retriever in the prompt, the way the
browser does it) and closed-book (question only, the answer has to come from the weights), each with free
greedy decoding and with decoding constrained to the knowledge base (the mode names ending in
"_constrained"; see constrain.py). Two more, on request (--modes):
  open_book_retrieved      open-book, and the answer may only be made of the facts that are IN THE CONTEXT
                           (or be one of the fixed answers): the model chooses among what the retriever found
  open_book_ranked         the same, and the retriever's ranking counts as a prior on that choice
                           (--prior-weight nats per rank, see constrain.ranking_priors): what the site does
  first_hit                no model at all: the answer is the retriever's first fact. The baseline every
                           open-book number has to beat; it can never decline a question
Splits: "train" is a sample of SEEN questions, "val" and "test" are held-out paraphrases plus unseen
out-of-scope questions; "final" is the untouched slice, evaluated once for the released model.

Strict metrics (per split and mode: overall, per language, per question kind, per topic):
  em                       normalised exact match (case, punctuation and spacing ignored)
  f1                       token-level F1 against the gold answer
  key_span_recall          share of the required names/numbers (key_spans) present in the answer
  refusal_precision/recall an answer counts as a refusal when it matches one of the refusal texts of the
                           data (token F1 >= 0.6); recall is measured on the out-of-scope questions
  refusal_rate             share of answers that are refusals (should be ~0 for in-scope groups)
  language_consistency     share of answers written in the language of the question
  unsupported_number_rate  share of the numbers in the answers that appear neither in the gold answer,
                           nor in the retrieved context, nor in the question (digits only: "nine" is
                           not checked)
  retrieval_hit            open-book only: share of questions whose gold facts were all retrieved; the
                           "by_retrieval" group scores the answers separately for the questions whose gold
                           facts were retrieved and for those where the retriever missed them

Metrics that describe what a visitor gets (an adjacent fact of the right subject is often a fine answer,
an invented sentence never is):
  verbatim_rate            share of answers made only of curated sentences: one fact, two or three facts
                           in a row, or one of the fixed answers. 1 - verbatim_rate is the share of
                           answers the model worded itself (blends, invented numbers, broken sentences)
  right_topic_rate         questions with gold facts: the answer is one to three curated facts and all of
                           them belong to a topic of the gold facts (exact matches included)
  gold_in_top1/2/4         open-book only, the retriever alone: all gold facts among its first k hits
                           (a question with two gold facts cannot score at k = 1)

Results go to <run>/eval.json, the individual answers to <run>/predictions.jsonl and a summary into the
model card (default <run>/model-card.json).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch

from .config import EOS, PAD, PromptConfig
from .constrain import PRIOR_KINDS, AnswerTrie, constrained_answers, ranking_priors
from .data import LANGS, Dataset, QARecord, build_prompt, load_dataset
from .model import GPT
from .retrieval import LAYOUTS, Bm25Retriever, kb_answers
from .tokenizer import BPETokenizer

MODES = ("open_book", "closed_book", "open_book_constrained", "closed_book_constrained")
EXTRA_MODES = ("open_book_retrieved", "open_book_ranked", "first_hit")
# weight 0 = no prior (open_book_ranked = open_book_retrieved). "score" keeps multi-fact answers better than
# "rank" at the same exact match (research log, entry 13), so it is the kind --tune-prior sweeps by default.
DEFAULT_PRIOR = {"kind": "score", "weight": 0.0, "fixed": 0.0}
TOP_KS = (1, 2, 4)

# --------------------------------------------------------------------------------------- text metrics
_WORD = re.compile(r"\w+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

_PT = set("o os um uma de da dos das em na nos nas é que com para por não ele ela você sobre mais como seu sua "
          "dele dela tem são foi está só sei pelo pela ou ao também já meu minha isso este esta esse essa muito "
          "quando onde qual quais quem trabalho perguntas responder tente perguntar e se sou pode".split())
_EN = set("the an of in on at is are was were he she his her him you your and or with for to about by from "
          "has have had it its this that these those can only answer questions not does did what who where "
          "when which how i my be been but if more most work try asking am ask".split())


def tokens(text: str) -> list[str]:
    """Lower-cased word tokens; punctuation and spacing are dropped, accents are kept."""
    return _WORD.findall(unicodedata.normalize("NFC", text).lower())


def exact_match(pred: str, gold: str) -> float:
    return float(tokens(pred) == tokens(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return float(p == g)
    remaining: dict[str, int] = defaultdict(int)
    for w in g:
        remaining[w] += 1
    common = 0
    for w in p:
        if remaining[w] > 0:
            common += 1
            remaining[w] -= 1
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def contains_span(pred: str, span: str) -> bool:
    """True when the tokens of span occur contiguously in pred ("40" does not match "140")."""
    p, s = tokens(pred), tokens(span)
    return bool(s) and any(p[i : i + len(s)] == s for i in range(len(p) - len(s) + 1))


def detect_language(text: str) -> str | None:
    """'pt', 'en' or None: counts function words of each language (and accented words for Portuguese)."""
    ws = tokens(text)
    pt = sum(w in _PT for w in ws) + sum(any(c in "ãõáéíóúâêôàç" for c in w) for w in ws)
    en = sum(w in _EN for w in ws)
    return None if pt == en else ("pt" if pt > en else "en")


def numbers(text: str) -> list[str]:
    return [n.replace(",", ".") for n in _NUMBER.findall(text)]


def is_refusal(pred: str, refusals: list[str], threshold: float = 0.6) -> bool:
    return any(f1_score(pred, r) >= threshold for r in refusals)


class FactMatcher:
    """Recognises curated sentences in an answer, with the same normalisation as exact match."""

    def __init__(self, dataset: Dataset):
        self.by_tokens: dict[tuple[str, ...], str] = {}
        for fact in dataset.facts.values():
            for lang in LANGS:
                self.by_tokens.setdefault(tuple(tokens(fact.text(lang))), fact.id)
        self.lengths = sorted({len(k) for k in self.by_tokens})
        self.canonical = {tuple(tokens(r.a)) for r in dataset.qa if not r.facts}

    def facts(self, pred: str, max_sentences: int = 3) -> list[str] | None:
        """The fact ids when pred is one to max_sentences curated facts in a row, else None."""
        p = tuple(tokens(pred))

        def walk(start: int, left: int) -> list[str] | None:
            if start == len(p):
                return []
            if left == 0:
                return None
            for n in self.lengths:
                if start + n > len(p):
                    break
                fact_id = self.by_tokens.get(p[start : start + n])
                rest = walk(start + n, left - 1) if fact_id is not None else None
                if rest is not None:
                    return [fact_id, *rest]
            return None

        return walk(0, max_sentences) if p else None

    def is_canonical(self, pred: str) -> bool:
        return tuple(tokens(pred)) in self.canonical


# --------------------------------------------------------------------------------------- generation
@torch.no_grad()
def generate_answers(model: GPT, tok: BPETokenizer, prompts: list[list[int]], max_new_tokens: int,
                     device: torch.device | str, max_tokens: int = 60000) -> list[str]:
    """Greedy answers for many prompts at once.

    Rows are right-padded and the whole batch is re-run at every step: causal attention makes the padding
    after a row's last token invisible to it, so no attention mask and no per-row cache are needed."""
    model.eval()
    pad, eos = tok.special_ids[PAD], tok.special_ids[EOS]
    block = model.cfg.block_size
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    answers = [""] * len(prompts)
    start = 0
    while start < len(order):
        longest = len(prompts[order[start]])
        end = start + 1
        while end < len(order):
            width = min(block, len(prompts[order[end]]) + max_new_tokens)
            if (end - start + 1) * width > max_tokens:
                break
            longest = len(prompts[order[end]])
            end += 1
        rows = order[start:end]
        width = min(block, longest + max_new_tokens)
        buf = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
        for r, i in enumerate(rows):
            buf[r, : len(prompts[i])] = torch.tensor(prompts[i], device=device)
        cur = torch.tensor([len(prompts[i]) for i in rows], device=device)
        done = cur >= width
        index = torch.arange(len(rows), device=device)
        for _ in range(max_new_tokens):
            logits, _, _ = model(buf[:, : int(cur.max())])
            nxt = logits[index, cur - 1].argmax(-1)
            active = ~done
            buf[index[active], cur[active]] = nxt[active]
            done = done | (nxt == eos)
            cur = torch.where(done, cur, cur + 1)
            done = done | (cur >= width)
            if bool(done.all()):
                break
        for r, i in enumerate(rows):
            gen = buf[r, len(prompts[i]) :].tolist()
            gen = gen[: gen.index(eos)] if eos in gen else [t for t in gen if t != pad]
            answers[i] = tok.decode(gen).strip()
        start = end
    return answers


# --------------------------------------------------------------------------------------- scoring
def score_record(rec: QARecord, pred: str, context: list[str], refusals: list[str], dataset: Dataset,
                 retrieved: list[str] | None, matcher: FactMatcher | None = None) -> dict:
    matcher = matcher or FactMatcher(dataset)
    answer_facts = matcher.facts(pred)
    gold_topics = {dataset.facts[f].topic for f in rec.facts}
    supported = set(numbers(rec.a)) | set(numbers(rec.q)) | {n for c in context for n in numbers(c)}
    nums = numbers(pred)
    row = {
        "id": rec.id, "lang": rec.lang, "kind": rec.kind, "split": rec.split, "topic": dataset.topic_of(rec),
        "q": rec.q, "gold": rec.a, "pred": pred,
        "em": exact_match(pred, rec.a), "f1": round(f1_score(pred, rec.a), 4),
        "spans_total": len(rec.key_spans), "spans_found": sum(contains_span(pred, s) for s in rec.key_spans),
        "refused": is_refusal(pred, refusals), "lang_detected": detect_language(pred),
        "numbers": len(nums), "unsupported_numbers": sum(n not in supported for n in nums),
        "answer_facts": answer_facts, "verbatim": answer_facts is not None or matcher.is_canonical(pred),
        "right_topic": (answer_facts is not None and all(dataset.facts[f].topic in gold_topics for f in answer_facts)
                        if rec.facts else None),
    }
    if retrieved is not None:
        row["retrieved"] = retrieved
        row["retrieval_hit"] = (set(rec.facts) <= set(retrieved)) if rec.facts else None
        for k in TOP_KS:
            row[f"gold_in_top{k}"] = (set(rec.facts) <= set(retrieved[:k])) if rec.facts else None
    return row


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def aggregate(rows: list[dict]) -> dict:
    """Metrics of one group of scored answers."""
    n = len(rows)
    detected = [r for r in rows if r["lang_detected"] is not None]
    oos = [r for r in rows if r["kind"] == "oos"]
    refused = [r for r in rows if r["refused"]]
    hits = [r["retrieval_hit"] for r in rows if r.get("retrieval_hit") is not None]
    with_gold = [r for r in rows if r.get("right_topic") is not None]
    out = {
        "n": n,
        "em": _ratio(sum(r["em"] for r in rows), n),
        "f1": _ratio(sum(r["f1"] for r in rows), n),
        "key_span_recall": _ratio(sum(r["spans_found"] for r in rows), sum(r["spans_total"] for r in rows)),
        "refusal_rate": _ratio(len(refused), n),
        "refusal_precision": _ratio(sum(r["kind"] == "oos" for r in refused), len(refused)),
        "refusal_recall": _ratio(sum(r["refused"] for r in oos), len(oos)),
        "language_consistency": _ratio(sum(r["lang_detected"] == r["lang"] for r in detected), len(detected)),
        "unsupported_number_rate": _ratio(sum(r["unsupported_numbers"] for r in rows), sum(r["numbers"] for r in rows)),
        "answers_with_unsupported_number": _ratio(sum(r["unsupported_numbers"] > 0 for r in rows), n),
        "verbatim_rate": _ratio(sum(bool(r.get("verbatim")) for r in rows), n),
        "right_topic_rate": _ratio(sum(r["right_topic"] for r in with_gold), len(with_gold)),
    }
    if hits:
        out["retrieval_hit"] = _ratio(sum(hits), len(hits))
        for k in TOP_KS:
            out[f"gold_in_top{k}"] = _ratio(sum(bool(r.get(f"gold_in_top{k}")) for r in rows
                                                if r.get("retrieval_hit") is not None), len(hits))
    return out


def report(rows: list[dict]) -> dict:
    def by(key: str) -> dict:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        return {k: aggregate(v) for k, v in sorted(groups.items())}

    out = {"overall": aggregate(rows), "by_lang": by("lang"), "by_kind": by("kind"), "by_topic": by("topic")}
    with_gold = [r for r in rows if r.get("retrieval_hit") is not None]
    if with_gold:  # open-book: separate what the retriever lost from what the model got wrong
        out["by_retrieval"] = {"gold_retrieved": aggregate([r for r in with_gold if r["retrieval_hit"]]),
                               "gold_missed": aggregate([r for r in with_gold if not r["retrieval_hit"]])}
    return out


def retriever_for(run: dict, lexical, model: GPT, tok: BPETokenizer, device, which: str = "auto",
                  encoder: str | Path | None = None):
    """(retriever, description) a run answers open-book questions with.

    "lexical" is the BM25 index alone. "auto" adds what the run has learned: a run trained with
    --retrieval-weight ranks the facts with its own hidden states too (BM25 and cosine fused with the weight
    chosen on validation, recorded in run.json). `encoder` fuses a stand-alone brasa.encoder run instead."""
    from .encoder import DenseRetriever, load_encoder

    def fused(net, net_tok, pooling, weight):
        return DenseRetriever(net, net_tok, lexical.docs, pooling, lexical, weight, device)

    if encoder:
        net, net_tok, enc_run = load_encoder(encoder, device)
        report = Path(encoder) / "eval-retrieval.json"
        weight = json.loads(report.read_text(encoding="utf-8")).get("best_weight", 1.0) if report.exists() else 1.0
        pooling = enc_run["checkpoint"]["encoder_config"]["pooling"]
        return fused(net, net_tok, pooling, weight), {"kind": "bm25+encoder", "encoder": str(encoder), "weight": weight}
    info = run.get("retrieval") or {}
    if which == "auto" and info.get("weight") is not None:
        return fused(model, tok, info["pooling"], info["weight"]), {"kind": "bm25+own", "weight": info["weight"]}
    return lexical, {"kind": "bm25"}


def answer_tries(dataset: Dataset, tok: BPETokenizer, retriever=None, langs=(*LANGS, None)) -> dict:
    """{lang: AnswerTrie} of what constrained decoding may write; the key None allows both languages.

    The sentences come from the retrieval index when it carries them (what the browser ships), otherwise
    from the dataset, which gives the same trie for an index built from that dataset."""
    index = getattr(retriever, "index", None)
    if not (index and "answers" in index):
        docs = [{"id": f.id, "lang": lang, "text": f.text(lang)} for f in dataset.facts.values() for lang in LANGS]
        index = {"docs": docs, "answers": kb_answers(dataset)}
    return {lang: AnswerTrie.from_index(index, tok, lang) for lang in langs}


def evaluate_model(model: GPT, tok: BPETokenizer, dataset: Dataset, retriever, device: torch.device | str,
                   prompt_cfg: PromptConfig | None = None, splits=("train", "val", "test"),
                   modes=MODES, seen_sample: int = 300, seed: int = 0, max_new_tokens: int | None = None,
                   beam: int = 1, constrain_lang: str = "any", prior: dict | None = None) -> tuple[dict, list[dict]]:
    """Returns (results[mode][split] -> report, scored rows).

    modes: "open_book" / "closed_book" decode freely; with the suffix "_constrained" the answer is
    restricted to the curated sentences (beam: 1 = greedy). constrain_lang: "any" lets the model choose
    between the Portuguese and the English sentences, as the browser does when it is not told the language;
    "question" only allows the sentences in the language of the question.
    prior: {"kind", "weight", "fixed"} of open_book_ranked: the retriever's ranking as a prior on the retrieved
    facts (constrain.ranking_priors) and the prior of the fixed answers; default: the one stored in the
    retrieval index ("answers" -> "prior"), else none."""
    prompt_cfg = prompt_cfg or PromptConfig()
    max_new = max_new_tokens or prompt_cfg.answer_reserve
    refusals = sorted({r.a for r in dataset.qa if r.kind == "oos"})
    matcher = FactMatcher(dataset)
    tries = answer_tries(dataset, tok, retriever) if any(m.endswith("_constrained") for m in modes) else {}
    results: dict = {}
    all_rows: list[dict] = []
    index = getattr(retriever, "index", None)
    prior = {**DEFAULT_PRIOR, **(((index or {}).get("answers") or {}).get("prior") or {}), **(prior or {})}
    for mode in modes:
        if mode not in MODES + EXTRA_MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES + EXTRA_MODES}")
        book = "closed_book" if mode.startswith("closed_book") else "open_book"
        if book == "open_book" and retriever is None:
            raise ValueError(f"mode {mode!r} needs a retriever")
        constrained = mode.endswith("_constrained")
        results[mode] = {}
        for split in splits:
            records = dataset.split(split)
            if split == "train" and len(records) > seen_sample:
                records = random.Random(seed).sample(records, seen_sample)
            if not records:
                continue
            prompts, contexts, retrieved, first, ranked = [], [], [], [], []
            for rec in records:
                hits = retriever.search(rec.q, prompt_cfg.max_facts) if book == "open_book" else []
                ids, used = build_prompt(tok, rec.q, [h.text for h in hits], model.cfg.block_size, prompt_cfg)
                prompts.append(ids)
                contexts.append([h.text for h in hits[:used]])
                retrieved.append([h.id for h in hits[:used]] if book == "open_book" else None)
                first.append(hits[0].text if hits else "")
                ranked.append(hits[:used])
            if mode == "first_hit":
                preds = first
            elif mode in ("open_book_retrieved", "open_book_ranked"):
                if not (index and "answers" in index):
                    raise ValueError(f"{mode} needs a retrieval index that carries the fixed answers")
                use_prior = mode == "open_book_ranked"
                per_prompt = [AnswerTrie.from_index(
                    index, tok, rec.lang if constrain_lang == "question" else None, only={h.id for h in hits},
                    priors=ranking_priors(hits, prior["weight"], prior["kind"]) if use_prior else None,
                    fixed_prior=prior["fixed"] if use_prior else 0.0)
                    for rec, hits in zip(records, ranked, strict=True)]
                preds = constrained_answers(model, tok, prompts, per_prompt, max_new, device, 1 if use_prior else beam)
            elif constrained:
                per_prompt = [tries[rec.lang if constrain_lang == "question" else None] for rec in records]
                preds = constrained_answers(model, tok, prompts, per_prompt, max_new, device, beam)
            else:
                preds = generate_answers(model, tok, prompts, max_new, device)
            rows = [score_record(rec, p, c, refusals, dataset, r, matcher)
                    for rec, p, c, r in zip(records, preds, contexts, retrieved, strict=True)]
            for row in rows:
                row["mode"] = mode
            results[mode][split] = report(rows)
            all_rows += rows
    return results, all_rows


PRIOR_WEIGHTS = {"score": (0.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0), "rank": (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)}
PRIOR_TOLERANCE = 0.005  # exact match: half a point


def tune_prior(model: GPT, tok: BPETokenizer, dataset: Dataset, retriever, device, prompt_cfg: PromptConfig,
               kind: str = "score", weights=None, fixed: float = 0.0, split: str = "val",
               tolerance: float = PRIOR_TOLERANCE) -> dict:
    """Choose how much the retriever's ranking weighs in open_book_ranked, on the VALIDATION questions.

    Every weight is evaluated (exact match over all question kinds) and the SMALLEST weight within
    `tolerance` of the best exact match is kept: the weakest prior that does the job, so that the model
    keeps deciding wherever the retriever is unsure. Returns {"prior": {"kind", "weight", "fixed"}, "split",
    "tolerance", "sweep": [{"weight", "em", "em_facts", "em_fixed", "refusal_precision", "refusal_recall"}]}."""
    weights = PRIOR_WEIGHTS[kind] if weights is None else weights
    sweep = []
    for weight in weights:
        results, _ = evaluate_model(model, tok, dataset, retriever, device, prompt_cfg, splits=(split,),
                                    modes=("open_book_ranked",),
                                    prior={"kind": kind, "weight": weight, "fixed": fixed})
        rep = results["open_book_ranked"][split]

        def em(kinds, rep=rep):
            groups = [v for k, v in rep["by_kind"].items() if k in kinds]
            return _ratio(sum(v["em"] * v["n"] for v in groups), sum(v["n"] for v in groups))

        sweep.append({"weight": weight, "em": rep["overall"]["em"], "em_facts": em(("single", "multi", "meta")),
                      "em_fixed": em(("chitchat", "oos")), "refusal_precision": rep["overall"]["refusal_precision"],
                      "refusal_recall": rep["overall"]["refusal_recall"]})
    best = max(row["em"] for row in sweep)
    chosen = min((row for row in sweep if row["em"] >= best - tolerance), key=lambda row: row["weight"])
    return {"prior": {"kind": kind, "weight": chosen["weight"], "fixed": fixed}, "split": split,
            "tolerance": tolerance, "sweep": sweep}


def headline(results: dict) -> dict:
    """The few numbers worth printing: seen vs held-out, per mode."""
    out: dict = {}
    for mode, per_split in results.items():
        out[mode] = {}
        for split, rep in per_split.items():
            o = rep["overall"]
            inscope = [v for k, v in rep["by_kind"].items() if k != "oos"]
            n_in = sum(v["n"] for v in inscope)
            out[mode]["seen" if split == "train" else split] = {
                "n": o["n"], "em": o["em"], "f1": o["f1"],
                "in_scope_em": _ratio(sum(v["em"] * v["n"] for v in inscope), n_in),
                "in_scope_f1": _ratio(sum(v["f1"] * v["n"] for v in inscope), n_in),
                "key_span_recall": o["key_span_recall"],
                "refusal_precision": o["refusal_precision"], "refusal_recall": o["refusal_recall"],
                "language_consistency": o["language_consistency"],
                "unsupported_number_rate": o["unsupported_number_rate"],
                "verbatim_rate": o["verbatim_rate"], "right_topic_rate": o["right_topic_rate"],
                **({k: o[k] for k in ("retrieval_hit", *(f"gold_in_top{k}" for k in TOP_KS)) if k in o}),
                **({f"em_when_{k}": v["em"] for k, v in rep["by_retrieval"].items()} if "by_retrieval" in rep else {}),
            }
    return out


# --------------------------------------------------------------------------------------- model card
def update_model_card(path: str | Path, sections: dict) -> dict:
    """Merge sections into the JSON model card (created when missing). Nested dicts are merged."""

    def merge(a: dict, b: dict) -> dict:
        for k, v in b.items():
            a[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
        return a

    path = Path(path)
    card = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema_version": 1}
    merge(card, sections)
    card["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(card, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return card


def card_sections_from_run(run_dir: Path) -> dict:
    """Model, tokenizer, training and data sections of the card, taken from <run>/run.json."""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    tc = run["train_config"]
    return {
        "name": run["preset"],
        "model": run["model"],
        "tokenizer": {"type": "byte-level BPE", **run["tokenizer"]},
        "training": {
            "steps": run["steps"], "batch_tokens": tc["batch_tokens"], "lr": run["lr"], "warmup": tc["warmup"],
            "min_lr_ratio": tc["min_lr_ratio"], "weight_decay": tc["weight_decay"], "dropout": tc["dropout"],
            "grad_clip": tc["grad_clip"], "betas": [tc["beta1"], tc["beta2"]], "seed": tc["seed"],
            "device": run["device"], "compile": run["compile"], "amp": run["amp"], "torch": run["torch"],
            "train_seconds": run["train_seconds"], "tokens_per_second": run["tokens_per_second"],
            "tokens_seen": run["tokens_seen"], "epochs_equivalent": run["epochs_equivalent"],
            "best_val_loss": run["best_val_loss"], "final_val": run["final_val"], "data_config": tc["data"],
            "index": tc.get("index"), "label_smoothing": tc.get("label_smoothing"),
            # what the run started from (a brasa.pretrain run, still from scratch) and how it ranks facts
            "init_from": run.get("init_from"), "contexts_from": run.get("contexts_from"),
            "retrieval_weight": tc.get("retrieval_weight"), "retrieval": _retrieval_summary(run.get("retrieval")),
        },
        "data": {**run["data"], "training_examples": run["examples"]},
    }


def _retrieval_summary(info: dict | None) -> dict | None:
    """The learned ranking of a --retrieval-weight run: how it is fused and what it finds on held-out questions."""
    if not info:
        return None
    out = {k: info.get(k) for k in ("pooling", "temperature", "weight")}
    out["recall"] = {"lexical": info.get("lexical"), "dense": info.get("dense"),
                     "fused": (info.get("fused") or {}).get(str(info.get("weight")))}
    return out


def load_run(run_dir: str | Path, checkpoint: str = "last", device: str | torch.device = "cpu"):
    """(model, tokenizer, retriever, run.json) of a run directory."""
    from .train import load_weights

    run_dir = Path(run_dir)
    model, _ = load_weights(run_dir / f"ckpt_{checkpoint}.pt", device)
    tok = BPETokenizer.load(run_dir / "tokenizer.json")
    retriever = Bm25Retriever.load(run_dir / "retrieval.json")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return model, tok, retriever, run


def main(argv=None) -> None:
    from .train import pick_device

    ap = argparse.ArgumentParser(description="Generate answers with a trained run and score them.")
    ap.add_argument("--run", required=True, help="run directory written by brasa.train")
    ap.add_argument("--data-dir", help="defaults to the data directory the run was trained on")
    ap.add_argument("--checkpoint", default="last", choices=["last", "best"])
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--modes", default=",".join(MODES),
                    help="open_book, closed_book (free decoding), the same with the suffix _constrained, "
                         "open_book_retrieved (only the retrieved facts may be written), open_book_ranked (the same "
                         "with the retriever's ranking as a prior) and first_hit (no model)")
    ap.add_argument("--prior", choices=PRIOR_KINDS, help="open_book_ranked: prior from the rank or from the scores")
    ap.add_argument("--prior-weight", type=float,
                    help="open_book_ranked: nats a fact loses per rank (rank) or at score 0 (score); default: the "
                         "value chosen for the run (run.json 'decoding'), else 0 = no prior")
    ap.add_argument("--fixed-prior", type=float, help="open_book_ranked: prior of the fixed answers, in nats")
    ap.add_argument("--tune-prior", action="store_true",
                    help="choose --prior-weight on the validation questions (the smallest weight within half a point "
                         "of the best exact match), record it in run.json as the run's decoding prior, and exit")
    ap.add_argument("--beam", type=int, default=1, help="constrained modes: 1 = greedy, more = beam search")
    ap.add_argument("--constrain-lang", default="any", choices=["any", "question"],
                    help="constrained modes: sentences of both languages, or only of the question's language")
    ap.add_argument("--index", choices=LAYOUTS,
                    help="rebuild the BM25 index from the data with this layout instead of using the run's own")
    ap.add_argument("--retriever", default="auto", choices=["auto", "lexical"],
                    help="auto: BM25 fused with the run's own learned ranking when it was trained with "
                         "--retrieval-weight; lexical: BM25 alone")
    ap.add_argument("--encoder", help="a brasa.encoder run whose ranking is fused with BM25 (instead of the run's own)")
    ap.add_argument("--tag", help="write eval-<tag>.json and predictions-<tag>.jsonl and leave the model card alone")
    ap.add_argument("--seen-sample", type=int, default=300, help="training questions sampled as the 'seen' set")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model-card", help="model card to update (default: <run>/model-card.json)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    device = pick_device(args.device)
    model, tok, retriever, run = load_run(run_dir, args.checkpoint, device)
    dataset = load_dataset(args.data_dir or run["train_config"]["data_dir"])
    prompt_cfg = PromptConfig(**run["train_config"]["data"]["prompt"])
    if args.index:
        retriever = Bm25Retriever.from_dataset(dataset, layout=args.index)
    retriever, retriever_info = retriever_for(run, retriever, model, tok, device, args.retriever, args.encoder)
    prior = {**DEFAULT_PRIOR, **((run.get("decoding") or {}).get("prior") or {})}
    for key, value in (("kind", args.prior), ("weight", args.prior_weight), ("fixed", args.fixed_prior)):
        if value is not None:
            prior[key] = value
    if args.tune_prior:
        tuned = tune_prior(model, tok, dataset, retriever, device, prompt_cfg, kind=prior["kind"], fixed=prior["fixed"])
        run["decoding"] = {**tuned, "checkpoint": args.checkpoint, "retriever": retriever_info}
        (run_dir / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
        for row in tuned["sweep"]:
            print(json.dumps(row))
        print(f"decoding prior chosen on {tuned['split']}: {json.dumps(tuned['prior'])} -> {run_dir / 'run.json'}")
        return
    t0 = time.time()
    results, rows = evaluate_model(model, tok, dataset, retriever, device, prompt_cfg,
                                   splits=args.splits.split(","), modes=args.modes.split(","),
                                   seen_sample=args.seen_sample, seed=run["train_config"]["seed"],
                                   beam=args.beam, constrain_lang=args.constrain_lang, prior=prior)
    seconds = round(time.time() - t0, 1)
    suffix = f"-{args.tag}" if args.tag else ""
    eval_path = run_dir / f"eval{suffix}.json"
    eval_path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(run_dir / f"predictions{suffix}.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(headline(results), indent=1))
    if args.tag:
        print(f"evaluated {len(rows)} answers in {seconds}s -> {eval_path} "
              f"(retriever: {json.dumps(retriever_info)}, prior: {json.dumps(prior)})")
        return
    card = args.model_card or run_dir / "model-card.json"
    decoding = {"constrained_beam": args.beam, "constrained_language": args.constrain_lang,
                "retriever": retriever_info, "prior": prior}
    update_model_card(card, {**card_sections_from_run(run_dir),
                             "evaluation": {"float32": {"checkpoint": args.checkpoint, "seconds": seconds,
                                                        "decoding": decoding,
                                                        "headline": headline(results), "full": results}}})
    print(f"evaluated {len(rows)} answers in {seconds}s -> {eval_path}, {card}")


if __name__ == "__main__":
    main()
