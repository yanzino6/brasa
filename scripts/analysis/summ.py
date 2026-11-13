"""Compact one-line-per-mode summary of eval-<tag>.json files (lab-notebook helper).

uv run python runs/scratch/summ.py runs/f2-4m-fields/eval-last.json [more.json ...] [--split test]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KEYS = [
    ("em", "EM"),
    ("in_scope_em", "factEM"),
    ("f1", "F1"),
    ("right_topic_rate", "topic"),
    ("verbatim_rate", "verb"),
    ("em_when_gold_retrieved", "EM|hit"),
    ("em_when_gold_missed", "EM|miss"),
    ("unsupported_number_rate", "unsup"),
    ("refusal_precision", "refP"),
    ("refusal_recall", "refR"),
    ("gold_in_top1", "g@1"),
    ("gold_in_top2", "g@2"),
    ("gold_in_top4", "g@4"),
    ("em_single", "single"),
    ("em_multi", "multi"),
    ("em_meta", "meta"),
    ("em_chitchat", "chat"),
    ("em_oos", "oos"),
]
KINDS = ("single", "multi", "meta", "chitchat", "oos")


def main() -> None:
    args = sys.argv[1:]
    split = "test"
    if "--split" in args:
        i = args.index("--split")
        split = args[i + 1]
        del args[i : i + 2]
    print(f"{'file / mode':58s} n    " + " ".join(f"{label:>7s}" for _, label in KEYS))
    for path in args:
        report = json.loads(Path(path).read_text())
        for mode, by_split in report.items():
            if not isinstance(by_split, dict) or split not in by_split:
                continue
            rep = by_split[split]
            m = rep.get("overall", rep) if isinstance(rep, dict) else rep
            if not isinstance(m, dict) or "em" not in m:
                continue
            m = dict(m)
            if "by_kind" in rep:  # the derived numbers of evaluate.headline
                inscope = [v for k, v in rep["by_kind"].items() if k != "oos"]
                m["in_scope_em"] = sum(v["em"] * v["n"] for v in inscope) / max(1, sum(v["n"] for v in inscope))
                for kind in KINDS:
                    if kind in rep["by_kind"]:
                        m[f"em_{kind}"] = rep["by_kind"][kind]["em"]
            for k, v in rep.get("by_retrieval", {}).items():
                m[f"em_when_{k}"] = v["em"]
            cells = []
            for key, _ in KEYS:
                value = m.get(key)
                cells.append(f"{100 * value:7.1f}" if isinstance(value, (int, float)) else "      -")
            name = f"{Path(path).parent.name}/{Path(path).stem[5:]}:{mode}"
            print(f"{name:58s} {m.get('n', 0):<4d} " + " ".join(cells))


if __name__ == "__main__":
    main()
