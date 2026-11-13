"""Programme item 6: all facts against the important facts only, on the SAME held-out questions.

uv run python runs/scratch/kb_size.py runs/f3-4m-fields/predictions-last.jsonl \
    runs/k1-4m-imp4/predictions-last-full2.jsonl [--split test] [--mode open_book_retrieved]
The first file is the model trained on every fact, the second the model trained on the facts of importance
4 and 5, evaluated on the questions of the full corpus (evaluate --data-dir runs/data-full2). Questions are
split by whether the small knowledge base still holds their gold facts.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

KB = Path("runs/data-full2/kb/facts.jsonl")
THRESHOLD = 4


def main() -> None:
    args = sys.argv[1:]
    opts = {"--split": "test", "--mode": "open_book_retrieved"}
    for flag in list(opts):
        if flag in args:
            i = args.index(flag)
            opts[flag] = args[i + 1]
            del args[i : i + 2]
    importance = {}
    for line in KB.read_text(encoding="utf-8").splitlines():
        if line.strip():
            fact = json.loads(line)
            importance[fact["id"]] = float(fact.get("importance") or 1)
    kept = {f for f, imp in importance.items() if imp >= THRESHOLD}
    print(f"{len(importance)} facts, {len(kept)} of importance >= {THRESHOLD}; split {opts['--split']}, mode {opts['--mode']}")
    gold = {}
    for line in Path("runs/data-full2/qa").glob("*.jsonl"):
        for row in line.read_text(encoding="utf-8").splitlines():
            if row.strip():
                rec = json.loads(row)
                gold[rec["id"]] = rec.get("facts", [])
    for path in args:
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        rows = [r for r in rows if r["mode"] == opts["--mode"] and r["split"] == opts["--split"]]
        groups = defaultdict(list)
        for r in rows:
            facts = gold[r["id"]]
            if not facts:
                groups[f"no fact ({r['kind']})"].append(r)
            elif set(facts) <= kept:
                groups["facts kept in the small KB"].append(r)
            else:
                groups["facts dropped from the small KB"].append(r)
            groups["all"].append(r)
            if facts:
                groups["all fact questions"].append(r)
        print(path)
        for name in sorted(groups):
            g = groups[name]
            n = len(g)
            em = sum(r["em"] for r in g) / n
            refused = sum(bool(r["refused"]) for r in g) / n
            topic = [r["right_topic"] for r in g if r.get("right_topic") is not None]
            wrong_fact = sum(bool(r["answer_facts"]) and not r["em"] for r in g) / n
            hit = [r["retrieval_hit"] for r in g if r.get("retrieval_hit") is not None]
            print(f"  {name:34s} n={n:<5d} EM {100 * em:5.1f}  refused {100 * refused:5.1f}  another fact "
                  f"{100 * wrong_fact:5.1f}  right topic {100 * sum(topic) / len(topic) if topic else float('nan'):5.1f}"
                  f"  gold in top 4 {100 * sum(hit) / len(hit) if hit else float('nan'):5.1f}")


if __name__ == "__main__":
    main()
