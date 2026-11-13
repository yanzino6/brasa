"""Is a difference between two sets of answers to the SAME questions more than question-sampling noise?

uv run python scripts/analysis/paired.py runs/A/predictions-last.jsonl:open_book_retrieved \
    runs/B/predictions-last.jsonl:open_book_ranked [--split test] [--metric em] [--samples 10000]
Prints both means, the difference B - A, a 95% bootstrap interval over questions (paired resampling), the
discordant counts (A right and B wrong, A wrong and B right) and the exact two-sided sign test on them.
This measures the noise of the QUESTION sample only; two trainings of one setting on MPS also differ by 1 to
2 points (run-to-run noise), which only a second seed shows.
"""
from __future__ import annotations

import json
import math
import random
import sys


def load(spec: str, split: str, metric: str) -> dict[str, float]:
    path, mode = spec.rsplit(":", 1)
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["mode"] == mode and r["split"] == split:
                value = r[metric]
                out[r["id"]] = float(bool(value)) if value is not None else float("nan")
    return out


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial test of wins against losses with p = 1/2."""
    n, k = wins + losses, min(wins, losses)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> None:
    args = sys.argv[1:]
    opts = {"--split": "test", "--metric": "em", "--samples": "10000"}
    for flag in list(opts):
        if flag in args:
            i = args.index(flag)
            opts[flag] = args[i + 1]
            del args[i : i + 2]
    a, b = (load(spec, opts["--split"], opts["--metric"]) for spec in args[:2])
    ids = sorted(i for i in a if i in b and not math.isnan(a[i]) and not math.isnan(b[i]))
    if not ids:
        raise SystemExit("no questions in common")
    xa, xb = [a[i] for i in ids], [b[i] for i in ids]
    n = len(ids)
    diff = [y - x for x, y in zip(xa, xb, strict=True)]
    rng = random.Random(0)
    means = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(int(opts["--samples"])))
    lo, hi = means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]
    wins, losses = sum(d > 0 for d in diff), sum(d < 0 for d in diff)
    print(f"{opts['--metric']} on {opts['--split']}, n={n}")
    print(f"  A {100 * sum(xa) / n:5.1f}  {args[0]}")
    print(f"  B {100 * sum(xb) / n:5.1f}  {args[1]}")
    print(f"  B - A = {100 * sum(diff) / n:+.1f} points, 95% interval [{100 * lo:+.1f}, {100 * hi:+.1f}]; "
          f"B right where A is wrong: {wins}, A right where B is wrong: {losses}; sign test p = {sign_test(wins, losses):.4f}")


if __name__ == "__main__":
    main()
