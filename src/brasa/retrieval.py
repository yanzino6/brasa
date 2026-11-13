"""A tiny lexical retriever over the knowledge-base facts: BM25 on word tokens and character trigrams.

Why lexical: the knowledge base is a few hundred short facts, the index is a small JSON file, a query
costs microseconds, and the exact same ranking can be computed in the browser (web/engine/retrieval.js).
Words carry the meaning; character trigrams make the match tolerant to typos, missing accents and
inflection ("trabalha" / "trabalhou", "project" / "projects").

Every fact is indexed once per language: its text, the title and aliases of its topic, and the training
questions that point to it (see Bm25Retriever.from_dataset). A query is scored against all documents,
so the language does not have to be detected; a fact is returned at most once, in the language that
matched best.

Two layouts of the same index format. "flat" concatenates the three texts into one document. "fields"
scores the fact text, the topic names and the training questions as three BM25 fields, each with its own
length normalisation and document frequencies, and adds them with FIELD_WEIGHTS: a 40-word fact is no
longer drowned in 300 words of questions, and a word that is rare among facts but common in questions
keeps its weight. On the held-out questions of the real corpus this is worth about 12 points of
recall@4 (ml/runs/research-log.md, entry 4). The sum happens at build time: the file format and the
search code are the same for both layouts.

The index file is also the knowledge base as the browser sees it: next to the documents it carries the
fixed answers ("answers": greetings, the refusal texts), which together with the facts are the sentences
constrained decoding allows (constrain.py, AnswerTrie in web/engine/retrieval.js).

Parity by construction: all logarithms and divisions happen here, at build time. The index stores one
finished weight per (feature, document) rounded to 4 decimals; searching is only lookups and float64
additions in a fixed order, which JavaScript reproduces bit for bit. tests/test_retrieval.py checks it.

The seam for a learned retriever is the `Retriever` protocol: anything with the same `search` signature
can be passed to data.py, evaluate.py and sample.py.

The learned half (optional, "dense" in the index): one vector per document, produced by a network read as
an encoder (encoder.py), stored as int8. A search that is given the vector of the question adds the cosine
to the lexical score: bm25 / (best bm25 of the query) + weight * cosine. The question vector is rounded to
int16 first, so the dot products are integer sums: given the same question vector, Python and JavaScript
again return the same scores bit for bit. Without a vector the search is the lexical one (the fallback).

    uv run python -m brasa.retrieval --data-dir data/rehearsal --query "onde o teo mora"
    uv run python -m brasa.retrieval --data-dir data/rehearsal --eval          # recall@k per split
"""

from __future__ import annotations

import base64
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

_COMBINING = re.compile("[\u0300-\u036f]")
_WORD = re.compile("[a-z0-9]+")

DEFAULT_PARAMS = {"k1": 1.2, "b": 0.75, "word_weight": 1.0, "char_weight": 0.5, "ngram": 3}
# the "fields" layout: chosen on the validation questions of the real corpus (research log, entry 4)
FIELD_PARAMS = {"k1": 2.0, "b": 0.75, "word_weight": 1.0, "char_weight": 0.25, "ngram": 3}
FIELD_WEIGHTS = {"fact": 1.0, "topic": 1.0, "questions": 2.0}
LAYOUTS = ("flat", "fields")
QUERY_LEVELS = 32767  # a question vector is rounded to int16 before the dot products


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
def build_index(docs: list[dict], indexed_texts: list[str] | dict[str, list[str]], params: dict | None = None,
                field_weights: dict[str, float] | None = None) -> dict:
    """docs: [{"id", "lang", "topic", "text"}]; indexed_texts[i] is what document i is matched against.

    indexed_texts may also be several fields, {name: [text of every document]}: each field is scored by
    its own BM25 (own average length, own document frequencies) and the scores are added with
    field_weights[name] (default 1). Returns the JSON-serialisable index that both implementations load:
    one finished weight per (feature, document), whatever the number of fields."""
    fields = indexed_texts if isinstance(indexed_texts, dict) else {"text": indexed_texts}
    p = {**(FIELD_PARAMS if len(fields) > 1 else DEFAULT_PARAMS), **(params or {})}
    weights = {name: (field_weights or {}).get(name, 1.0) for name in fields}
    n_docs = len(docs)
    total: dict[str, dict[int, float]] = {}
    for name, texts in fields.items():
        feats = [features(t, p["ngram"]) for t in texts]
        for kind, kind_weight in ((0, p["word_weight"]), (1, p["char_weight"])):
            weight = weights[name] * kind_weight
            if weight <= 0:
                continue
            tfs = [Counter(f[kind]) for f in feats]
            lengths = [len(f[kind]) for f in feats]
            avg = sum(lengths) / max(1, n_docs) or 1.0
            df = Counter(term for tf in tfs for term in tf)
            for i, tf in enumerate(tfs):
                for term, count in tf.items():
                    idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                    norm = count + p["k1"] * (1.0 - p["b"] + p["b"] * lengths[i] / avg)
                    per_doc = total.setdefault(term, {})
                    per_doc[i] = per_doc.get(i, 0.0) + weight * idf * count * (p["k1"] + 1.0) / norm
    postings: dict[str, list] = {}
    for term in sorted(total):
        entry = [x for i, w in sorted(total[term].items()) if round(w, 4) > 0 for x in (i, round(w, 4))]
        if entry:
            postings[term] = entry
    if len(fields) > 1:
        p["fields"] = weights
    return {"version": 1, "type": "bm25", "params": p, "docs": docs, "postings": postings}


def kb_answers(dataset, max_sentences: int = 3) -> dict:
    """The fixed answers of a corpus (the answers of the TRAINING records that point to no fact: chit-chat
    and refusals) and the number of facts one answer may chain. Stored in the index as "answers"."""
    fixed = sorted({(r.lang, r.a) for r in dataset.split("train") if not r.facts})
    return {"max_sentences": max_sentences, "canonical": [{"lang": lang, "text": text} for lang, text in fixed]}


def pack_dense(vectors, weight: float, pooling: str = "mean", max_tokens: int = 128) -> dict:
    """The "dense" section of an index: unit vectors (one row per document) as int8 with one scale.

    weight is how much the cosine counts next to the normalised lexical score; pooling and max_tokens tell
    the engine how the question has to be embedded (encoder.question_ids, encoder.pool)."""
    v = np.asarray(vectors, dtype=np.float64)
    scale = float(np.abs(v).max()) / 127 or 1.0
    q = np.clip(np.floor(v / scale + 0.5), -127, 127).astype(np.int8)
    return {"version": 1, "dim": int(v.shape[1]), "pooling": pooling, "weight": float(weight),
            "max_tokens": int(max_tokens), "scale": scale, "vectors": base64.b64encode(q.tobytes()).decode("ascii")}


def quantize_query(vector) -> np.ndarray:
    """A question vector as integers: floor(v * 32767 + 0.5), the rounding retrieval.js uses."""
    v = np.asarray(vector, dtype=np.float32).astype(np.float64)
    return np.clip(np.floor(v * QUERY_LEVELS + 0.5), -QUERY_LEVELS, QUERY_LEVELS).astype(np.int64)


class Bm25Retriever:
    def __init__(self, index: dict):
        if index.get("type") != "bm25" or index.get("version") != 1:
            raise ValueError("unsupported retrieval index")
        self.index = index
        self.docs: list[dict] = index["docs"]
        self.postings: dict[str, list] = index["postings"]
        self.ngram: int = index["params"]["ngram"]
        self.dense: dict | None = None  # the learned half: {"matrix", "factor", "weight", "pooling", "max_tokens"}
        dense = index.get("dense")
        if dense:
            if dense.get("version") != 1:
                raise ValueError("unsupported dense section in the retrieval index")
            matrix = np.frombuffer(base64.b64decode(dense["vectors"]), dtype=np.int8).astype(np.int64)
            if matrix.size != len(self.docs) * dense["dim"]:
                raise ValueError("the dense section does not have one vector per document")
            self.dense = {"matrix": matrix.reshape(len(self.docs), dense["dim"]),
                          "factor": dense["scale"] / QUERY_LEVELS, "weight": dense["weight"],
                          "pooling": dense["pooling"], "max_tokens": dense["max_tokens"]}

    def with_dense(self, vectors, weight: float, pooling: str = "mean", max_tokens: int = 128) -> Bm25Retriever:
        """A retriever over the same documents whose index also carries their vectors (see pack_dense)."""
        return Bm25Retriever({**self.index, "dense": pack_dense(vectors, weight, pooling, max_tokens)})

    @classmethod
    def from_dataset(cls, dataset, params: dict | None = None, expand_with_questions: bool = True,
                     skip=None, layout: str = "flat") -> Bm25Retriever:
        """One document per (fact, language).

        A fact is matched against its own text, the title and aliases of its topic and, with
        expand_with_questions, the TRAINING questions that point to it (document expansion). Facts are
        written as statements and questions as questions, so they share few words; the questions people
        already asked about a fact are the best description of how the next person will ask. On the
        rehearsal corpus this lifts recall@4 on held-out paraphrases from 0.60 to 0.92. Validation and
        test questions are never indexed. skip(record) -> True leaves a training question out (used to
        build the cross-fitted indices of data.fold_retrievers). layout: "flat" (one document made of
        the three texts) or "fields" (three BM25 fields added with FIELD_WEIGHTS; see the module text)."""
        if layout not in LAYOUTS:
            raise ValueError(f"unknown index layout {layout!r}; choose from {LAYOUTS}")
        questions: dict[tuple[str, str], list[str]] = {}
        if expand_with_questions:
            for r in dataset.split("train"):
                if skip is not None and skip(r):
                    continue
                for fact_id in r.facts:
                    questions.setdefault((fact_id, r.lang), []).append(r.q)
        docs: list[dict] = []
        fields: dict[str, list[str]] = {"fact": [], "topic": [], "questions": []}
        for fact in dataset.facts.values():
            topic = dataset.topics.get(fact.topic)
            for lang in ("pt", "en"):
                docs.append({"id": fact.id, "lang": lang, "topic": fact.topic, "text": fact.text(lang)})
                fields["fact"].append(fact.text(lang))
                fields["topic"].append(" ".join([getattr(topic, f"title_{lang}"), *topic.aliases]) if topic else "")
                fields["questions"].append(" ".join(questions.get((fact.id, lang), [])))
        if layout == "fields":
            index = build_index(docs, fields, params, FIELD_WEIGHTS)
        else:
            index = build_index(docs, [" ".join(t for t in parts if t) for parts in zip(*fields.values(), strict=True)],
                                params)
        return cls({**index, "answers": kb_answers(dataset)})

    def scores(self, query: str, vector=None) -> list[float]:
        """One score per document: BM25, or with the vector of the question and a dense section the fusion
        bm25 / (best bm25 of the query) + weight * cosine."""
        scores = [0.0] * len(self.docs)
        for feat in query_features(query, self.ngram):
            entry = self.postings.get(feat)
            if entry:
                for j in range(0, len(entry), 2):
                    scores[entry[j]] += entry[j + 1]
        if vector is None or self.dense is None:
            return scores
        top = max(scores) or 1.0
        dots = self.dense["matrix"] @ quantize_query(vector)  # integers: exact, in any order
        factor, weight = self.dense["factor"], self.dense["weight"]
        return [s / top + weight * (d * factor) for s, d in zip(scores, dots.tolist(), strict=True)]

    def search(self, query: str, k: int = 4, lang: str | None = None, vector=None) -> list[Hit]:
        """The k best facts (each fact once), best first. lang restricts the search to one language.
        vector: the question as the encoder sees it; with it (and a dense section) the ranking is the fusion
        and every document is a candidate, without it only documents that share a feature with the query."""
        scores = self.scores(query, vector)
        fused = vector is not None and self.dense is not None
        order = sorted((i for i, s in enumerate(scores) if fused or s > 0), key=lambda i: (-scores[i], i))
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
    ap.add_argument("--layout", default="flat", choices=LAYOUTS, help="one document per fact, or three BM25 fields")
    ap.add_argument("--out", help="write the index JSON here")
    args = ap.parse_args()

    dataset = load_dataset(args.data_dir)
    retriever = Bm25Retriever.from_dataset(dataset, layout=args.layout)
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
