"""Accuracy as a function of how far a held-out question is from the training phrasings of its fact.

uv run python runs/scratch/novelty.py runs/f2-4m-fields/predictions-last.jsonl [--mode open_book] [--split test]
Distance = 1 - the highest word-set Jaccard similarity between the question and any TRAINING question of the
same gold fact (accents and case folded). Single-fact and meta questions only.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

from brasa.data import load_dataset
from brasa.retrieval import words


def main() -> None:
    args = sys.argv[1:]
    opts = {"--mode": "open_book", "--split": "test", "--data-dir": "runs/data-full2"}
    for key in list(opts):
        if key in args:
            i = args.index(key)
            opts[key] = args[i + 1]
            del args[i : i + 2]
    dataset = load_dataset(opts["--data-dir"])
    train = defaultdict(list)
    for r in dataset.split("train"):
        if len(r.facts) == 1:
            train[(r.facts[0], r.lang)].append(set(words(r.q)))
    gold = {r.id: r for r in dataset.split(opts["--split"]) if len(r.facts) == 1}
    edges = [(0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 1.01)]
    for path in args:
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        rows = [x for x in rows if x["mode"] == opts["--mode"] and x["split"] == opts["--split"] and x["id"] in gold]
        buckets = defaultdict(list)
        for x in rows:
            r = gold[x["id"]]
            q = set(words(r.q))
            sim = max((len(q & t) / max(1, len(q | t)) for t in train[(r.facts[0], r.lang)]), default=0.0)
            for lo, hi in edges:
                if lo <= sim < hi:
                    buckets[(lo, hi)].append(x)
        print(f"{path} [{opts['--mode']}, {opts['--split']}, n={len(rows)}]")
        print("  similarity to the nearest training phrasing |    n |   EM | right topic | gold@1 | gold@4")
        for lo, hi in edges:
            b = buckets[(lo, hi)]
            if not b:
                continue
            mean = lambda k: 100 * sum(float(x.get(k) or 0) for x in b) / len(b)  # noqa: E731
            print(f"  {lo:.2f} to {min(hi, 1.0):.2f}                               | {len(b):4d} | {mean('em'):4.1f} |"
                  f" {mean('right_topic'):11.1f} | {mean('gold_in_top1'):6.1f} | {mean('gold_in_top4'):6.1f}")


if __name__ == "__main__":
    main()
