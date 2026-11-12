"""Render the Hugging Face model card (Markdown) from model-card.json and MODEL_CARD.template.md.

    uv run python scripts/render_model_card.py model-card.json --out MODEL_CARD.md
    uv run python scripts/render_model_card.py runs/smoke/model-card.json          # prints to stdout

Every number in the output comes from the JSON card, which is written by brasa.evaluate and brasa.export.
A card measured on a corpus marked "fictional" gets a warning at the top instead of looking like a result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT_NAMES = {"seen": "seen (training)", "val": "held-out (validation)", "test": "held-out (test)"}
FILE_NOTES = {
    "model.i8.bin": "weights, int8 with one scale per row (what the site loads)",
    "model.f16.bin": "weights, float16",
    "tokenizer.json": "BPE merges and control tokens",
    "retrieval.json": "BM25 index over the knowledge base",
    "manifest.json": "configuration, sizes and checksums",
}


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def evaluation_table(evaluation: dict) -> str:
    head = ("| Weights | Mode | Questions | n | EM | F1 | EM in scope | Key spans | Refusal P | Refusal R | "
            "Language | Bad numbers |")
    lines = [head, "|" + " --- |" * 12]
    for label, entry in evaluation.items():
        for mode, per_split in entry.get("headline", {}).items():
            for split, m in per_split.items():
                cells = [label, mode.replace("_", "-"), SPLIT_NAMES.get(split, split), str(m["n"]), pct(m["em"]),
                         pct(m["f1"]), pct(m["in_scope_em"]), pct(m["key_span_recall"]), pct(m["refusal_precision"]),
                         pct(m["refusal_recall"]), pct(m["language_consistency"]), pct(m["unsupported_number_rate"])]
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def files_table(files: dict) -> str:
    lines = ["| File | Bytes | gzip | brotli | What |", "| --- | ---: | ---: | ---: | --- |"]
    for name, f in files.items():
        if isinstance(f, dict) and name in FILE_NOTES and f.get("bytes") is not None:
            brotli = "n/a" if f.get("brotli") is None else f"{f['brotli']:,}"
            lines.append(f"| `{name}` | {f['bytes']:,} | {f['gzip']:,} | {brotli} | {FILE_NOTES[name]} |")
    return "\n".join(lines)


def render(card: dict, template: str) -> str:
    model, training, data = card.get("model", {}), card.get("training", {}), card.get("data", {})
    evaluation = {k: v for k, v in card.get("evaluation", {}).items() if isinstance(v, dict)}
    corpus = data.get("corpus") or {}
    seconds = training.get("train_seconds")
    test = evaluation.get("float32", {}).get("headline", {}).get("open_book", {}).get("test", {})
    quant = card.get("quantisation") or {}

    values = {
        "name": card.get("name"), "params": f"{model['params']:,}" if model.get("params") else None,
        "n_layer": model.get("n_layer"), "d_model": model.get("d_model"), "n_head": model.get("n_head"),
        "block_size": model.get("block_size"), "vocab_size": model.get("vocab_size"),
        "steps": f"{training['steps']:,}" if training.get("steps") else None, "device": training.get("device"),
        "train_time": f"{seconds / 60:.0f} min" if seconds else None,
        "max_facts": training.get("data_config", {}).get("prompt", {}).get("max_facts"),
        "files_table": files_table(card.get("files", {})) if (card.get("files") or {}).get("directory") else None,
        "evaluation_table": evaluation_table(evaluation) if test else None,
        "corpus_warning": (
            "> **This card was produced from a FICTIONAL rehearsal corpus.** The person it describes does not "
            "exist and none of these numbers is a result of the released model."
            if corpus.get("fictional") else ""),
        "retrieval_note": (
            f"Open-book, held-out test questions: the retriever returned the gold facts for "
            f"{pct(test['retrieval_hit'])} of the in-scope questions; exact match was "
            f"{pct(test['em_when_gold_retrieved'])} when it did and {pct(test['em_when_gold_missed'])} when it "
            f"did not." if "em_when_gold_retrieved" in test else ""),
        "quantisation_note": "; ".join(
            f"{fmt}: relative weight error {q['mean_relative_weight_rmse']}, largest logit drift "
            f"{q['max_abs_logit_drift_vs_float32']}" for fmt, q in quant.items()
            if isinstance(q, dict) and q.get("mean_relative_weight_rmse") is not None)
        + "".join(f"; {label} answers identical to float32: {e['answers_identical_to_float32']}"
                  for label, e in evaluation.items() if e.get("answers_identical_to_float32"))
        if any(isinstance(q, dict) and q.get("mean_relative_weight_rmse") is not None for q in quant.values())
        else None,
        "data_note": (
            f"{data['facts']} facts in {data['topics']} topics, each in Portuguese and English; "
            f"{data['splits']['train']['n']} training, {data['splits']['val']['n']} validation and "
            f"{data['splits']['test']['n']} test question/answer records "
            f"({data['train_questions_per_fact_and_lang']['median']} training questions per fact and language, "
            f"median); {data['training_examples']['examples']:,} training examples of "
            f"{data['training_examples']['tokens']:,} tokens after open-book / closed-book expansion."
            if data.get("splits") and data["splits"]["train"].get("n") else None),
    }

    def fill(match: re.Match) -> str:
        value = values.get(match.group(1))
        return "TBD" if value is None else str(value)

    return re.sub(r"\{\{(\w+)\}\}", fill, template)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("card", help="model-card.json written by brasa.evaluate / brasa.export")
    ap.add_argument("--template", default=str(ROOT / "MODEL_CARD.template.md"))
    ap.add_argument("--out", help="write here instead of stdout")
    args = ap.parse_args()
    card = json.loads(Path(args.card).read_text(encoding="utf-8"))
    text = render(card, Path(args.template).read_text(encoding="utf-8"))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
