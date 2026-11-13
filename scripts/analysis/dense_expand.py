"""Document expansion in vector space: does a fact vector get better when it is moved towards the centroid
of the training questions that point to the fact?

uv run python runs/scratch/dense_expand.py runs/j2-4m-joint-pre [--device cpu]
new vector = normalise(fact vector + alpha * mean(vectors of its training questions, same language)).
Prints recall@1/2/4 of the dense ranking alone and fused with BM25 (the run's own weight), val and test.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from brasa.config import PAD
from brasa.data import load_dataset
from brasa.encoder import DenseRetriever, embed, question_ids, recall
from brasa.evaluate import load_run


def main() -> None:
    run_dir = sys.argv[1]
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
    model, tok, lexical, run = load_run(run_dir, "last", device)
    info = run["retrieval"]
    dataset = load_dataset(run["train_config"]["data_dir"])
    base = DenseRetriever(model, tok, lexical.docs, info["pooling"], device=device)
    by_doc: dict[tuple[str, str], list[str]] = {}
    for r in dataset.split("train"):
        for f in r.facts:
            by_doc.setdefault((f, r.lang), []).append(r.q)
    questions = sorted({q for qs in by_doc.values() for q in qs})
    with torch.no_grad():
        vectors = embed(model, [question_ids(tok, q) for q in questions], tok.special_ids[PAD], device, info["pooling"])
    at = {q: i for i, q in enumerate(questions)}
    centroid = torch.zeros_like(base.vectors)
    for i, d in enumerate(lexical.docs):
        qs = by_doc.get((d["id"], d["lang"]), [])
        if qs:
            centroid[i] = F.normalize(vectors[[at[q] for q in qs]].mean(0), dim=-1)
    print(f"{run_dir}: {len(questions)} training questions, fusion weight {info['weight']}")
    for alpha in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        new = F.normalize(base.vectors + alpha * centroid, dim=-1)
        dense = DenseRetriever(model, tok, lexical.docs, info["pooling"], device=device, vectors=new)
        dense._cache = base._cache
        fused = DenseRetriever(model, tok, lexical.docs, info["pooling"], lexical, info["weight"], device, new)
        fused._cache = base._cache
        cells = []
        for name, r in (("dense", dense), ("fused", fused)):
            for split in ("val", "test"):
                m = recall(dataset, r, split, ks=(1, 2, 4))
                cells.append(f"{name} {split} " + "/".join(f"{100 * m[f'recall@{k}']:.1f}" for k in (1, 2, 4)))
        print(f"alpha {alpha:4.2f}  " + "   ".join(cells), flush=True)


if __name__ == "__main__":
    main()
