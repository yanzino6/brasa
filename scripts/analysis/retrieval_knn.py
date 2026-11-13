"""Per-question index (kNN over training questions) and topic routing, against the fields layout."""
import collections
import sys

from brasa.data import load_dataset
from brasa.retrieval import Bm25Retriever, build_index, words

ds = load_dataset(sys.argv[1] if len(sys.argv) > 1 else "runs/data-full1")
fields = Bm25Retriever.from_dataset(ds, layout="fields")
train = [r for r in ds.split("train") if len(r.facts) == 1]
qdocs = [{"id": r.facts[0], "lang": r.lang, "topic": ds.facts[r.facts[0]].topic, "text": r.q} for r in train]
def qindex(k1, cw):
    return Bm25Retriever(build_index(qdocs, [d["text"] for d in qdocs], {"k1": k1, "b": 0.75, "word_weight": 1.0, "char_weight": cw, "ngram": 3}))
fact_ids = list(ds.facts)
pos = {f: i for i, f in enumerate(fact_ids)}
doc_fact = [pos[d["id"]] for d in fields.docs]
topic_of = [ds.facts[f].topic for f in fact_ids]

def fact_scores_fields(q):
    s = fields.scores(q)
    out = [0.0] * len(fact_ids)
    for i, v in enumerate(s):
        if v > out[doc_fact[i]]:
            out[doc_fact[i]] = v
    return out

def knn_scores(qi, q, m):
    s = qi.scores(q)
    per = collections.defaultdict(list)
    for i, v in enumerate(s):
        if v > 0:
            per[pos[qi.docs[i]["id"]]].append(v)
    out = [0.0] * len(fact_ids)
    for f, vs in per.items():
        vs.sort(reverse=True)
        out[f] = sum(vs[:m])
    return out

# topic routing: aliases and titles found in the question
alias = {}
for t in ds.topics.values():
    for a in {t.title_pt, t.title_en, *t.aliases, t.slug}:
        key = " ".join(words(a))
        if key and key not in ("yan", "yan simmer"):
            alias.setdefault(key, set()).add(t.slug)
def topics_in(q):
    text = " " + " ".join(words(q)) + " "
    found = set()
    for a, slugs in alias.items():
        if f" {a} " in text:
            found |= slugs
    return found

def evaluate(score_fn, split, ks=(1, 2, 4, 8)):
    recs = [r for r in ds.split(split) if len(r.facts) == 1]
    hit = collections.Counter(); t1 = 0
    for r in recs:
        s = score_fn(r.q)
        order = sorted(range(len(s)), key=lambda i: (-s[i], i))[: max(ks)]
        g = pos[r.facts[0]]
        for k in ks:
            hit[k] += g in order[:k]
        t1 += topic_of[order[0]] == topic_of[g]
    return " ".join(f"r@{k}={100*hit[k]/len(recs):.1f}" for k in ks) + f" topic@1={100*t1/len(recs):.1f}"

print("fields            val ", evaluate(fact_scores_fields, "val"), "| test", evaluate(fact_scores_fields, "test"))
routed = sum(bool(topics_in(r.q)) for r in ds.split("val") if len(r.facts) == 1)
right = sum(ds.facts[r.facts[0]].topic in topics_in(r.q) for r in ds.split("val") if len(r.facts) == 1)
print(f"alias found in {routed} val questions; the gold topic among them in {right}")
def routed_fn(boost):
    def fn(q):
        s = fact_scores_fields(q); ts = topics_in(q)
        return [v * (boost if topic_of[i] in ts else 1.0) for i, v in enumerate(s)] if ts else s
    return fn
for boost in (1.25, 1.5, 2.0, 100.0):
    print(f"fields + alias x{boost:<5} val ", evaluate(routed_fn(boost), "val"), "| test", evaluate(routed_fn(boost), "test"))
for k1, cw in ((1.2, 0.5), (2.0, 0.25)):
    qi = qindex(k1, cw)
    for m in (1, 2, 3):
        print(f"knn k1={k1} cw={cw} m={m} val ", evaluate(lambda q: knn_scores(qi, q, m), "val"))
        for lam in (0.5, 1.0, 2.0):
            def fused(q, lam=lam, m=m):
                a, b = fact_scores_fields(q), knn_scores(qi, q, m)
                return [x + lam * y for x, y in zip(a, b, strict=False)]
            print(f"   fields + {lam}*knn(m={m}) val ", evaluate(fused, "val"), "| test", evaluate(fused, "test"))
