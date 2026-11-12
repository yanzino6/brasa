"""A tiny lexical retriever over the knowledge-base facts: BM25 on word tokens and character trigrams.

Why lexical: the knowledge base is a few hundred short facts, the index is a small JSON file, a query
costs microseconds, and the exact same ranking can be computed in the browser (web/engine/retrieval.js).
Words carry the meaning; character trigrams make the match tolerant to typos, missing accents and
inflection ("trabalha" / "trabalhou", "project" / "projects").

Every fact is indexed once per language: its text, the title and aliases of its topic, and the training
questions that point to it (see Bm25Retriever.from_dataset). A query is scored against all documents,
so the language does not have to be detected; a fact is returned at most once, in the language that
matched best.

Parity by construction: all logarithms and divisions happen here, at build time. The index stores one
finished weight per (feature, document) rounded to 4 decimals; searching is only lookups and float64
additions in a fixed order, which JavaScript reproduces bit for bit. tests/test_retrieval.py checks it.

The seam for a learned retriever is the `Retriever` protocol: anything with the same `search` signature
can be passed to data.py, evaluate.py and sample.py.

    uv run python -m brasa.retrieval --data-dir data/rehearsal --query "onde o teo mora"
    uv run python -m brasa.retrieval --data-dir data/rehearsal --eval          # recall@k per split
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_COMBINING = re.compile("[\u0300-\u036f]")
_WORD = re.compile("[a-z0-9]+")

DEFAULT_PARAMS = {"k1": 1.2, "b": 0.75, "word_weight": 1.0, "char_weight": 0.5, "ngram": 3}


@dataclass(frozen=True)
class Hit:
    id: str  # fact id
    lang: str  # language of the version that matched
    score: float
    text: str  # the fact text to show to the model
    topic: str
    doc: int  # index of the document inside the index


class Retriever(Protocol):
    def search(self, query: str, k: int = 4, lang: str | None = None) -> list[Hit]: ...


# --------------------------------------------------------------------------------------- features
def fold(text: str) -> str:
    """Lower-case and strip accents (NFD, drop U+0300..U+036F). Mirrored exactly in retrieval.js."""
    return _COMBINING.sub("", unicodedata.normalize("NFD", text)).lower()


def words(text: str) -> list[str]:
    return _WORD.findall(fold(text))


def char_ngrams(word: str, n: int = 3) -> list[str]:
    padded = f"#{word}#"
    return [padded[i : i + n] for i in range(max(1, len(padded) - n + 1))]


def features(text: str, n: int = 3) -> tuple[list[str], list[str]]:
    """All word features ("w:...") and character n-gram features ("c:...") of a text, with repeats."""
    ws = words(text)
    return [f"w:{w}" for w in ws], [f"c:{g}" for w in ws for g in char_ngrams(w, n)]


def query_features(text: str, n: int = 3) -> list[str]:
    """Unique features of a query in a fixed order: words first, then n-grams, by first appearance."""
    w, c = features(text, n)
    return list(dict.fromkeys(w)) + list(dict.fromkeys(c))


# --------------------------------------------------------------------------------------- index
def build_index(docs: list[dict], indexed_texts: list[str], params: dict | None = None) -> dict:
    """docs: [{"id", "lang", "topic", "text"}]; indexed_texts[i] is what document i is matched against.

    Returns the JSON-serialisable index that both implementations load."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    n_docs = len(docs)
    fields = [features(t, p["ngram"]) for t in indexed_texts]
    postings: dict[str, list] = {}
    for field, weight in ((0, p["word_weight"]), (1, p["char_weight"])):
        tfs = [Counter(f[field]) for f in fields]
        lengths = [len(f[field]) for f in fields]
        avg = sum(lengths) / max(1, n_docs) or 1.0
        df = Counter(term for tf in tfs for term in tf)
        for term in sorted(df):
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            entry = []
            for i, tf in enumerate(tfs):
                if term in tf:
                    norm = tf[term] + p["k1"] * (1.0 - p["b"] + p["b"] * lengths[i] / avg)
                    w = round(weight * idf * tf[term] * (p["k1"] + 1.0) / norm, 4)
                    if w > 0:
                        entry += [i, w]
            if entry:
                postings[term] = entry
    return {"version": 1, "type": "bm25", "params": p, "docs": docs, "postings": postings}


class Bm25Retriever:
    def __init__(self, index: dict):
        if index.get("type") != "bm25" or index.get("version") != 1:
            raise ValueError("unsupported retrieval index")
        self.index = index
        self.docs: list[dict] = index["docs"]
        self.postings: dict[str, list] = index["postings"]
        self.ngram: int = index["params"]["ngram"]

    @classmethod
    def from_dataset(cls, dataset, params: dict | None = None, expand_with_questions: bool = True) -> Bm25Retriever:
        """One document per (fact, language).

        A fact is matched against its own text, the title and aliases of its topic and, with
        expand_with_questions, the TRAINING questions that point to it (document expansion). Facts are
        written as statements and questions as questions, so they share few words; the questions people
        already asked about a fact are the best description of how the next person will ask. On the
        rehearsal corpus this lifts recall@4 on held-out paraphrases from 0.60 to 0.92. Validation and
        test questions are never indexed."""
        questions: dict[tuple[str, str], list[str]] = {}
        if expand_with_questions:
            for r in dataset.split("train"):
                for fact_id in r.facts:
                    questions.setdefault((fact_id, r.lang), []).append(r.q)
        docs, texts = [], []
        for fact in dataset.facts.values():
            topic = dataset.topics.get(fact.topic)
            for lang in ("pt", "en"):
                extra = [getattr(topic, f"title_{lang}"), *topic.aliases] if topic else []
                docs.append({"id": fact.id, "lang": lang, "topic": fact.topic, "text": fact.text(lang)})
                texts.append(" ".join([fact.text(lang), *extra, *questions.get((fact.id, lang), [])]))
        return cls(build_index(docs, texts, params))

    def scores(self, query: str) -> list[float]:
        scores = [0.0] * len(self.docs)
        for feat in query_features(query, self.ngram):
            entry = self.postings.get(feat)
            if entry:
                for j in range(0, len(entry), 2):
                    scores[entry[j]] += entry[j + 1]
        return scores

    def search(self, query: str, k: int = 4, lang: str | None = None) -> list[Hit]:
        """The k best facts (each fact once), best first. lang restricts the search to one language."""
        scores = self.scores(query)
        order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))
        hits, seen = [], set()
        for i in order:
            d = self.docs[i]
            if d["id"] in seen or (lang is not None and d["lang"] != lang):
                continue
            seen.add(d["id"])
            hits.append(Hit(d["id"], d["lang"], scores[i], d["text"], d["topic"], i))
            if len(hits) == k:
                break
        return hits

    def to_json(self) -> dict:
        return self.index

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Bm25Retriever:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def recall_report(dataset, retriever: Retriever, ks=(1, 2, 4, 8)) -> dict:
    """Share of questions whose gold facts are ALL among the top k, per split."""
    out: dict = {}
    for split in ("train", "val", "test"):
        records = [r for r in dataset.split(split) if r.facts]
        if not records:
            continue
        row = {"n": len(records)}
        for k in ks:
            hit = sum(set(r.facts) <= {h.id for h in retriever.search(r.q, k)} for r in records)
            row[f"recall@{k}"] = round(hit / len(records), 4)
        out[split] = row
    return out


def main() -> None:
    import argparse

    from .data import load_dataset

    ap = argparse.ArgumentParser(description="Query or evaluate the BM25 retriever of a data directory.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--query")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--lang", choices=["pt", "en"])
    ap.add_argument("--eval", action="store_true", help="recall@k of the gold facts for every split")
    ap.add_argument("--out", help="write the index JSON here")
    args = ap.parse_args()

    dataset = load_dataset(args.data_dir)
    retriever = Bm25Retriever.from_dataset(dataset)
    print(f"{len(retriever.docs)} documents, {len(retriever.postings)} features")
    if args.query:
        for h in retriever.search(args.query, args.k, args.lang):
            print(f"{h.score:7.3f}  {h.id} [{h.lang}]  {h.text}")
    if args.eval:
        print(json.dumps(recall_report(dataset, retriever), indent=1))
    if args.out:
        retriever.save(args.out)
        print("saved", args.out)


if __name__ == "__main__":
    main()
