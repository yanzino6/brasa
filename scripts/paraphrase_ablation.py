"""How many question paraphrases does a fact need before held-out phrasings are answered?

    uv run python scripts/paraphrase_ablation.py                       # k = 1, 2, 4, 8, 11
    uv run python scripts/paraphrase_ablation.py --k 2,11 --steps 800 --preset brasa-12m

For every k the FICTIONAL rehearsal corpus is rebuilt with only k hand-written training questions per
fact and language (each one also appears once wrapped in a greeting, so the model sees 2k questions per
fact and language). The validation and test questions are the same for every k, and their phrasings are
never in training. A model is trained from scratch for the same number of steps and evaluated on the
test split, open-book and closed-book.

Output: one table on stdout and <out>/summary.json. Runs go to <out>/k<k>/ (ignored by git).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import make_rehearsal

from brasa.config import PromptConfig
from brasa.data import load_dataset
from brasa.evaluate import evaluate_model, headline, load_run
from brasa.retrieval import Bm25Retriever, recall_report
from brasa.train import TrainConfig, pick_device, train

ROOT = Path(__file__).resolve().parent.parent


def run_one(k: int, args) -> dict:
    out = Path(args.out) / f"k{k}"
    make_rehearsal.write(out / "data", k=k)
    cfg = TrainConfig(preset=args.preset, data_dir=str(out / "data"), out=str(out / "run"), steps=args.steps,
                      seed=args.seed, device=args.device, log_every=100, eval_every=200, ckpt_every=args.steps)
    info = train(cfg)
    device = pick_device(args.device)
    model, tok, retriever, _ = load_run(out / "run", "last", device)
    dataset = load_dataset(out / "data")
    results, _ = evaluate_model(model, tok, dataset, retriever, device, PromptConfig(**info["train_config"]["data"]
                                                                                    ["prompt"]), seed=args.seed)
    head = headline(results)
    recall = recall_report(dataset, Bm25Retriever.from_dataset(dataset), ks=(1, 4))
    row = {
        "k": k, "train_questions_per_fact_and_lang": info["data"]["train_questions_per_fact_and_lang"]["median"],
        "train_records": info["data"]["splits"]["train"]["n"], "training_examples": info["examples"]["examples"],
        "steps": info["steps"], "epochs": info["epochs_equivalent"], "train_seconds": info["train_seconds"],
        "final_val_loss": info["final_val"]["val_loss"],
        "retrieval_recall@1_test": recall["test"]["recall@1"], "retrieval_recall@4_test": recall["test"]["recall@4"],
    }
    for mode in ("open_book", "closed_book"):
        for split in ("seen", "test"):
            m = head[mode][split]
            for key in ("em", "f1", "in_scope_em", "in_scope_f1", "key_span_recall", "refusal_precision",
                        "refusal_recall", "em_when_gold_retrieved"):
                if key in m:
                    row[f"{mode}.{split}.{key}"] = m[key]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", default="1,2,4,8,11", help="comma separated numbers of training paraphrases (1..11)")
    ap.add_argument("--preset", default="brasa-4m")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ROOT / "runs" / "paraphrase-ablation"))
    args = ap.parse_args()

    rows = []
    for k in [int(x) for x in args.k.split(",")]:
        rows.append(run_one(k, args))
        Path(args.out, "summary.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")

    def pct(x) -> str:
        return "  n/a" if x is None else f"{100 * x:5.1f}"

    print(f"\n{args.preset}, {args.steps} steps per run, test split = held-out paraphrases (in-scope questions only)")
    print(f"{'k':>3s} {'q/fact/lang':>11s} {'recall@1':>8s} {'recall@4':>8s} | {'open EM':>7s} {'open F1':>7s} "
          f"{'EM|gold':>7s} | {'closed EM':>9s} {'closed F1':>9s} | {'refusal P':>9s} {'refusal R':>9s} | "
          f"{'seen EM':>7s}")
    for r in rows:
        print(f"{r['k']:>3d} {r['train_questions_per_fact_and_lang']:>11d} {pct(r['retrieval_recall@1_test']):>8s} "
              f"{pct(r['retrieval_recall@4_test']):>8s} | {pct(r['open_book.test.in_scope_em']):>7s} "
              f"{pct(r['open_book.test.in_scope_f1']):>7s} {pct(r.get('open_book.test.em_when_gold_retrieved')):>7s} | "
              f"{pct(r['closed_book.test.in_scope_em']):>9s} {pct(r['closed_book.test.in_scope_f1']):>9s} | "
              f"{pct(r['open_book.test.refusal_precision']):>9s} {pct(r['open_book.test.refusal_recall']):>9s} | "
              f"{pct(r['open_book.seen.em']):>7s}")
    print(f"written to {Path(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
