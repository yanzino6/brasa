"""Sweep the retriever prior of open_book_ranked on held-out questions (lab-notebook helper).

uv run python runs/scratch/prior_sweep.py runs/j1-4m-joint --kinds rank,score --weights 0,0.5,1,2,4,8 \
    [--fixed 0,-1,1] [--splits val] [--retriever auto|lexical] [--device cpu]
Prints one line per setting: exact match overall, on fact questions, per kind, refusal precision / recall.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from brasa.config import PromptConfig
from brasa.data import load_dataset
from brasa.evaluate import evaluate_model, load_run, retriever_for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--kinds", default="rank")
    ap.add_argument("--weights", default="0,0.5,1,2,4,8")
    ap.add_argument("--fixed", default="0")
    ap.add_argument("--splits", default="val")
    ap.add_argument("--retriever", default="auto")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    args = ap.parse_args()
    model, tok, lexical, run = load_run(args.run, "last", args.device)
    dataset = load_dataset(run["train_config"]["data_dir"])
    prompt_cfg = PromptConfig(**run["train_config"]["data"]["prompt"])
    retriever, info = retriever_for(run, lexical, model, tok, args.device, args.retriever)
    print(args.run, json.dumps(info))
    rows = []
    kinds = ("single", "multi", "meta", "chitchat", "oos")
    print(f"{'split':5s} {'kind':5s} {'weight':>6s} {'fixed':>5s} {'EM':>6s} {'facts':>6s} "
          + " ".join(f"{k:>8s}" for k in kinds) + f" {'refP':>6s} {'refR':>6s} {'topic':>6s}")
    grid = itertools.product(args.kinds.split(","), map(float, args.weights.split(",")), map(float, args.fixed.split(",")))
    for kind, weight, fixed in grid:
        results, _ = evaluate_model(model, tok, dataset, retriever, args.device, prompt_cfg,
                                    splits=args.splits.split(","), modes=["open_book_ranked"],
                                    prior={"kind": kind, "weight": weight, "fixed": fixed})
        for split, rep in results["open_book_ranked"].items():
            o, by = rep["overall"], rep["by_kind"]
            fact = [by[k] for k in ("single", "multi", "meta") if k in by]
            fact_em = sum(v["em"] * v["n"] for v in fact) / max(1, sum(v["n"] for v in fact))
            row = {"split": split, "kind": kind, "weight": weight, "fixed": fixed, "em": o["em"], "fact_em": fact_em,
                   **{f"em_{k}": by[k]["em"] for k in kinds if k in by},
                   "refusal_precision": o["refusal_precision"], "refusal_recall": o["refusal_recall"],
                   "right_topic_rate": o["right_topic_rate"]}
            rows.append(row)
            print(f"{split:5s} {kind:5s} {weight:6.2f} {fixed:5.1f} {100 * o['em']:6.1f} {100 * fact_em:6.1f} "
                  + " ".join(f"{100 * by[k]['em']:8.1f}" if k in by else "       -" for k in kinds)
                  + f" {100 * (o['refusal_precision'] or 0):6.1f} {100 * (o['refusal_recall'] or 0):6.1f}"
                  + f" {100 * o['right_topic_rate']:6.1f}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
