"""Where does the retriever fail on held-out phrasings? CPU only. uv run python runs/scratch/retrieval_analysis.py"""
import collections
import sys

from brasa.data import load_dataset
from brasa.retrieval import Bm25Retriever

data = sys.argv[1] if len(sys.argv) > 1 else "runs/data-full1"
ds = load_dataset(data)

def recall(search, split, ks=(1, 2, 4, 8, 16)):
    recs = [r for r in ds.split(split) if len(r.facts) == 1]
    hits = collections.Counter()
    topic1 = 0
    for r in recs:
        got = search(r.q, max(ks))
        ids = [h.id for h in got]
        for k in ks:
            hits[k] += r.facts[0] in ids[:k]
        topic1 += bool(got) and got[0].topic == ds.facts[r.facts[0]].topic
    return {f"r@{k}": round(100 * hits[k] / len(recs), 1) for k in ks} | {"topic@1": round(100 * topic1 / len(recs), 1), "n": len(recs)}

for layout in ("flat", "fields"):
    r = Bm25Retriever.from_dataset(ds, layout=layout)
    for split in ("val", "test"):
        print(layout, split, recall(r.search, split))
