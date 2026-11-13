"""What if the model only decided WHETHER to answer with a fact, and the retriever's first hit were the fact?

uv run python runs/scratch/hybrid.py runs/j1-4m-joint/predictions-last.jsonl [--split test]
Reads the open_book_retrieved and first_hit rows of a predictions file and scores three rules:
  model      the model's own answer (decoding restricted to the retrieved facts)
  first hit  always the retriever's first fact
  hybrid     the model's answer when it is a fixed answer (greeting, refusal) or a chain of facts,
             otherwise the retriever's first fact
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def main() -> None:
    args = sys.argv[1:]
    split = "test"
    if "--split" in args:
        i = args.index("--split")
        split = args[i + 1]
        del args[i : i + 2]
    for path in args:
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        model = {r["id"]: r for r in rows if r["mode"] == "open_book_retrieved" and r["split"] == split}
        first = {r["id"]: r for r in rows if r["mode"] == "first_hit" and r["split"] == split}
        score = defaultdict(lambda: defaultdict(float))
        count = defaultdict(int)
        for qid, m in model.items():
            f = first[qid]
            facts = m["answer_facts"]
            keep = facts is None or len(facts) != 1  # a fixed answer or a chain: the model's decision stands
            for kind in (m["kind"], "all", "facts" if m["kind"] in ("single", "multi", "meta") else "fixed"):
                count[kind] += 1
                score[kind]["model"] += m["em"]
                score[kind]["first hit"] += f["em"]
                score[kind]["hybrid"] += m["em"] if keep else f["em"]
                score[kind]["answers with one fact"] += not keep
        print(path, split)
        for kind in ("all", "facts", "fixed", "single", "multi", "meta", "chitchat", "oos"):
            if count[kind]:
                cells = "  ".join(f"{name} {100 * v / count[kind]:5.1f}" for name, v in score[kind].items())
                print(f"  {kind:9s} n={count[kind]:<5d} {cells}")


if __name__ == "__main__":
    main()
