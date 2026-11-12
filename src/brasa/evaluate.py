"""Evaluate a trained model by generating answers and scoring them.

    uv run python -m brasa.evaluate --run runs/brasa-12m
    uv run python -m brasa.evaluate --run runs/brasa-12m --model-card model-card.json     # the release card

Every question is answered twice, greedily: open-book (facts from the retriever in the prompt, the way
the browser does it) and closed-book (question only, the answer has to come from the weights). Splits:
"train" is a sample of SEEN questions, "val" and "test" are held-out paraphrases plus unseen
out-of-scope questions.

Metrics (per split and mode: overall, per language, per question kind, per topic):
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
from .data import Dataset, QARecord, build_prompt, load_dataset
from .model import GPT
from .retrieval import Bm25Retriever
from .tokenizer import BPETokenizer

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
                 retrieved: list[str] | None) -> dict:
    supported = set(numbers(rec.a)) | set(numbers(rec.q)) | {n for c in context for n in numbers(c)}
    nums = numbers(pred)
    row = {
        "id": rec.id, "lang": rec.lang, "kind": rec.kind, "split": rec.split, "topic": dataset.topic_of(rec),
        "q": rec.q, "gold": rec.a, "pred": pred,
        "em": exact_match(pred, rec.a), "f1": round(f1_score(pred, rec.a), 4),
        "spans_total": len(rec.key_spans), "spans_found": sum(contains_span(pred, s) for s in rec.key_spans),
        "refused": is_refusal(pred, refusals), "lang_detected": detect_language(pred),
        "numbers": len(nums), "unsupported_numbers": sum(n not in supported for n in nums),
    }
    if retrieved is not None:
        row["retrieved"] = retrieved
        row["retrieval_hit"] = (set(rec.facts) <= set(retrieved)) if rec.facts else None
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
    }
    if hits:
        out["retrieval_hit"] = _ratio(sum(hits), len(hits))
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


def evaluate_model(model: GPT, tok: BPETokenizer, dataset: Dataset, retriever, device: torch.device | str,
                   prompt_cfg: PromptConfig | None = None, splits=("train", "val", "test"),
                   modes=("open_book", "closed_book"), seen_sample: int = 300, seed: int = 0,
                   max_new_tokens: int | None = None) -> tuple[dict, list[dict]]:
    """Returns (results[mode][split] -> report, scored rows)."""
    prompt_cfg = prompt_cfg or PromptConfig()
    max_new = max_new_tokens or prompt_cfg.answer_reserve
    refusals = sorted({r.a for r in dataset.qa if r.kind == "oos"})
    results: dict = {}
    all_rows: list[dict] = []
    for mode in modes:
        results[mode] = {}
        for split in splits:
            records = dataset.split(split)
            if split == "train" and len(records) > seen_sample:
                records = random.Random(seed).sample(records, seen_sample)
            if not records:
                continue
            prompts, contexts, retrieved = [], [], []
            for rec in records:
                hits = retriever.search(rec.q, prompt_cfg.max_facts) if mode == "open_book" else []
                ids, used = build_prompt(tok, rec.q, [h.text for h in hits], model.cfg.block_size, prompt_cfg)
                prompts.append(ids)
                contexts.append([h.text for h in hits[:used]])
                retrieved.append([h.id for h in hits[:used]] if mode == "open_book" else None)
            preds = generate_answers(model, tok, prompts, max_new, device)
            rows = [score_record(rec, p, c, refusals, dataset, r)
                    for rec, p, c, r in zip(records, preds, contexts, retrieved, strict=True)]
            for row in rows:
                row["mode"] = mode
            results[mode][split] = report(rows)
            all_rows += rows
    return results, all_rows


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
                **({"retrieval_hit": o["retrieval_hit"]} if "retrieval_hit" in o else {}),
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
        },
        "data": {**run["data"], "training_examples": run["examples"]},
    }


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
    ap.add_argument("--modes", default="open_book,closed_book")
    ap.add_argument("--seen-sample", type=int, default=300, help="training questions sampled as the 'seen' set")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model-card", help="model card to update (default: <run>/model-card.json)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    device = pick_device(args.device)
    model, tok, retriever, run = load_run(run_dir, args.checkpoint, device)
    dataset = load_dataset(args.data_dir or run["train_config"]["data_dir"])
    prompt_cfg = PromptConfig(**run["train_config"]["data"]["prompt"])
    t0 = time.time()
    results, rows = evaluate_model(model, tok, dataset, retriever, device, prompt_cfg,
                                   splits=args.splits.split(","), modes=args.modes.split(","),
                                   seen_sample=args.seen_sample, seed=run["train_config"]["seed"])
    seconds = round(time.time() - t0, 1)
    (run_dir / "eval.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    card = args.model_card or run_dir / "model-card.json"
    update_model_card(card, {**card_sections_from_run(run_dir),
                             "evaluation": {"float32": {"checkpoint": args.checkpoint, "seconds": seconds,
                                                        "headline": headline(results), "full": results}}})
    print(json.dumps(headline(results), indent=1))
    print(f"evaluated {len(rows)} answers in {seconds}s -> {run_dir / 'eval.json'}, {card}")


if __name__ == "__main__":
    main()
