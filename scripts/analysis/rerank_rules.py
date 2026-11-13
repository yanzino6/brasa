"""Decision rules over the candidate scores of rerank_scores.py.   uv run python runs/scratch/rerank_rules.py runs/f1-4m-fields/rerank-last.json"""
import json
import sys
from collections import defaultdict

from brasa.evaluate import exact_match

recs = json.load(open(sys.argv[1]))
for r in recs:
    for c in r["cands"]:
        c["em"] = exact_match(c["text"], r["gold"])
    r["top"] = max((c["bm25"] for c in r["cands"]), default=0.0) or 1.0


def pick(r, K, rule, lam=0.0, beta=0.0, norm=False, src="open", lenpow=0.0, samelang=False):
    best, best_s = None, -1e18
    nf = 0
    for c in r["cands"]:
        if c["fact"] is not None:
            nf += 1
            if nf > K:
                continue
        if samelang and c["lang"] != r["lang"]:
            continue
        if rule == "first":
            if c["fact"] is None:
                continue
            s = c["bm25"]
        else:
            m = c[src]
            if m is None:
                continue
            if lenpow:
                m = m / (c["n"] ** lenpow)
            b = c["bm25"] / r["top"] if norm else c["bm25"]
            s = m + lam * b + (beta if c["fact"] is None else 0.0)
        if s > best_s:
            best, best_s = c, s
    return best


def evaluate(split, **kw):
    g = defaultdict(lambda: [0, 0])
    for r in recs:
        if r["split"] != split:
            continue
        c = pick(r, **kw)
        ok = bool(c and c["em"])
        kind = "fact" if r["kind"] in ("single", "meta") else r["kind"]
        for k in (kind, "all"):
            g[k][0] += ok
            g[k][1] += 1
    return " ".join(f"{k}={100 * a / b:5.1f}" for k, (a, b) in sorted(g.items()))


for split in ("val", "test"):
    print("==", split, {k: sum(1 for r in recs if r["split"] == split and r["kind"] == k) for k in ("single", "meta", "multi", "chitchat", "oos")})
    print("first hit                     ", evaluate(split, K=16, rule="first"))
    for K in (4, 8, 16):
        for src in ("open", "closed"):
            print(f"model {src:6s} K={K:2d}            ", evaluate(split, K=K, rule="model", src=src))
    print("model open K=8 samelang       ", evaluate(split, K=8, rule="model", samelang=True))
    for K in (8, 16):
        for lam in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
            for beta in (0.0,):
                print(f"fused open K={K:2d} lam={lam:4.2f} raw   ", evaluate(split, K=K, rule="model", lam=lam, beta=beta))
        for lam in (5, 10, 20, 40):
            print(f"fused open K={K:2d} lam={lam:4.1f} norm  ", evaluate(split, K=K, rule="model", lam=lam, norm=True))
    for lam, beta in ((0.3, 2), (0.3, 4), (0.3, 6), (0.5, 4), (0.5, 8), (0.8, 8), (0.8, 12)):
        print(f"fused open K= 8 lam={lam} beta={beta}  ", evaluate(split, K=8, rule="model", lam=lam, beta=beta))
    for lam in (0.3, 0.5):
        print(f"fused closed K=8 lam={lam}        ", evaluate(split, K=8, rule="model", lam=lam, src="closed"))
