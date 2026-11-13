"""The BM25 retriever, and the proof that Python and JavaScript return the same ranking."""

from __future__ import annotations

import json
import subprocess

import pytest
from conftest import ENGINE, KERNEL, TINY, small_dataset

from brasa.data import load_dataset
from brasa.fixtures import feature_cases, retrieval_cases
from brasa.retrieval import Bm25Retriever, build_index, char_ngrams, fold, query_features, recall_report, words


def test_features():
    assert fold("Ação É Ímpar") == "acao e impar"
    assert words("Onde o Téo mora? 2028!") == ["onde", "o", "teo", "mora", "2028"]
    assert char_ngrams("teo") == ["#te", "teo", "eo#"]
    assert char_ngrams("a") == ["#a#"]
    assert query_features("teo teo") == ["w:teo", "c:#te", "c:teo", "c:eo#"], "unique, words first"


def test_finds_the_fact_in_both_languages_and_with_typos(retriever):
    assert retriever.search("onde o teo mora")[0].id == "location"
    assert retriever.search("Where does Téo live?")[0].id == "location"
    assert retriever.search("mora em curitba")[0].id == "location"  # typo: character n-grams still match
    assert retriever.search("em qual cidaed ele mroa")[0].id == "location"  # only the training questions help here


def test_each_fact_once_best_first_and_language_filter(retriever):
    hits = retriever.search("python pytorch", 8)
    assert 1 < len(hits) <= 8
    assert len({h.id for h in hits}) == len(hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert len(retriever.search("python pytorch", 1)) == 1
    assert {h.lang for h in retriever.search("contact email", 4, "pt")} == {"pt"}
    assert retriever.search("zzzz qqqq") == [] and retriever.search("") == []


def test_bm25_prefers_rare_words_and_short_documents():
    docs = [{"id": f"d{i}", "lang": "en", "topic": "t", "text": t} for i, t in enumerate(
        ["the cat sat", "the dog sat", "the the the the the the cat and a very long tail of other words here"])]
    r = Bm25Retriever(build_index(docs, [d["text"] for d in docs]))
    assert [h.id for h in r.search("the cat", 3)][:2] == ["d0", "d2"], "same words: the shorter document wins"
    assert r.search("dog", 3)[0].id == "d1"
    scores = r.scores("the cat")
    assert scores[0] > scores[1] > 0, "'cat' is rarer than 'the'"


def test_index_round_trips_through_json(retriever, tmp_path):
    path = tmp_path / "retrieval.json"
    retriever.save(path)
    again = Bm25Retriever.load(path)
    for q in ("onde o teo mora", "what did he build for fraud", "contato"):
        assert [(h.id, h.lang, h.score) for h in again.search(q, 6)] == [(h.id, h.lang, h.score) for h in
                                                                         retriever.search(q, 6)]
    with pytest.raises(ValueError):
        Bm25Retriever({"type": "dense", "version": 1})


def test_only_training_questions_are_indexed(tmp_path):
    dataset = load_dataset(small_dataset(tmp_path))
    index = Bm25Retriever.from_dataset(dataset).index
    assert "w:oi" in index["postings"], "the training prefix 'Oi!' was indexed with its fact"
    assert "w:diga" not in index["postings"], "'Me diga:' only occurs in test questions"
    assert "w:favor" not in index["postings"], "'Por favor:' only occurs in validation questions"
    plain = Bm25Retriever.from_dataset(dataset, expand_with_questions=False).index
    assert "w:oi" not in plain["postings"]


def test_question_expansion_improves_recall_on_held_out_paraphrases(dataset, retriever):
    with_questions = recall_report(dataset, retriever)
    without = recall_report(dataset, Bm25Retriever.from_dataset(dataset, expand_with_questions=False))
    assert with_questions["test"]["recall@4"] >= 0.85
    assert with_questions["test"]["recall@4"] > without["test"]["recall@4"] + 0.1
    assert with_questions["train"]["recall@4"] >= 0.99


def test_fields_layout_scores_each_field_on_its_own(dataset):
    """Three BM25 fields added at build time: same file format, same search code, better ranking."""
    docs = [{"id": "a", "lang": "pt", "topic": "t", "text": "x"}, {"id": "b", "lang": "pt", "topic": "t", "text": "y"}]
    fields = {"fact": ["gato preto", "cachorro"], "questions": ["onde mora o gato", "onde mora o cachorro grande"]}
    both = build_index(docs, fields, field_weights={"fact": 1.0, "questions": 2.0})
    assert both["params"]["fields"] == {"fact": 1.0, "questions": 2.0} and both["version"] == 1
    parts = {name: build_index(docs, {name: texts, "empty": ["", ""]}, field_weights={name: weight})
             for (name, texts), weight in zip(fields.items(), (1.0, 2.0), strict=True)}
    for term, entry in both["postings"].items():
        expected: dict[int, float] = {}
        for part in parts.values():
            for i, w in zip(*[iter(part["postings"].get(term, []))] * 2, strict=True):
                expected[i] = expected.get(i, 0.0) + w
        got = dict(zip(*[iter(entry)] * 2, strict=True))
        assert got == pytest.approx(expected, abs=1.1e-4), term
    # "gato" is in both fields of document a: the fields add up; a weight of 0 switches a field off
    assert Bm25Retriever(both).search("gato")[0].id == "a"
    assert "w:preto" not in build_index(docs, fields, field_weights={"fact": 0.0})["postings"]

    flat = Bm25Retriever.from_dataset(dataset)
    fielded = Bm25Retriever.from_dataset(dataset, layout="fields")
    assert "fields" not in flat.index["params"]
    assert set(fielded.index["params"]["fields"]) == {"fact", "topic", "questions"}
    assert [d["id"] for d in fielded.docs] == [d["id"] for d in flat.docs] and "answers" in fielded.index
    report = recall_report(dataset, fielded)
    assert report["train"]["recall@4"] >= 0.99 and report["test"]["recall@4"] >= 0.85
    with pytest.raises(ValueError):
        Bm25Retriever.from_dataset(dataset, layout="tree")


def test_python_reproduces_the_committed_fixtures():
    """The fixture file is what JavaScript is tested against, so it must be what Python returns today."""
    committed = json.loads((TINY / "parity.json").read_text(encoding="utf-8"))
    retriever = Bm25Retriever.load(TINY / "retrieval.json")
    queries = list(dict.fromkeys(c["q"] for c in committed["retrieval"]))
    assert retrieval_cases(retriever, queries) == committed["retrieval"]
    assert feature_cases([c["q"] for c in committed["features"]]) == committed["features"]
    assert len(committed["retrieval"]) > 100


def test_javascript_returns_the_same_ranking_and_scores(node):
    """Runs web/engine/retrieval.js over the same fixtures: identical ids, order and scores (bit for bit)."""
    out = subprocess.run([node, str(ENGINE / "tools" / "parity.mjs"), str(TINY), "--wasm", str(KERNEL), "--quiet"],
                         capture_output=True, text=True, timeout=120)
    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report["text"]["retrieval"] == {"cases": 204, "hits": report["text"]["retrieval"]["hits"],
                                           "rankingMismatches": 0, "scoreMismatches": 0}
    assert report["text"]["features"]["mismatches"] == 0
    assert report["text"]["retrieval"]["hits"] > 500


def test_the_dense_section_fuses_cosine_and_bm25_with_exact_integer_arithmetic(tmp_path):
    import base64

    import numpy as np

    from brasa.retrieval import QUERY_LEVELS, pack_dense, quantize_query

    ds = load_dataset(small_dataset(tmp_path))
    lexical = Bm25Retriever.from_dataset(ds, layout="fields")
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(lexical.docs), 16))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    dense = pack_dense(vectors, weight=1.5, pooling="mean")
    stored = np.frombuffer(base64.b64decode(dense["vectors"]), dtype=np.int8).reshape(len(lexical.docs), 16)
    assert dense["dim"] == 16 and np.abs(stored).max() == 127, "the largest component uses the whole int8 range"
    assert np.abs(stored * dense["scale"] - vectors).max() <= dense["scale"] / 2 + 1e-12

    fused = lexical.with_dense(vectors, weight=1.5)
    assert "dense" not in lexical.index and fused.index["docs"] is lexical.index["docs"]
    reloaded = Bm25Retriever(json.loads(json.dumps(fused.index)))  # survives the JSON file
    query, q = "Onde a Lia mora?", vectors[0].astype(np.float32)
    assert fused.scores(query) == lexical.scores(query), "no question vector: the lexical fallback"
    assert [h.id for h in fused.search(query, 4)] == [h.id for h in lexical.search(query, 4)]

    bm25 = lexical.scores(query)
    qi = [int(x) for x in quantize_query(q)]
    assert all(abs(x) <= QUERY_LEVELS for x in qi) and qi == [int(np.floor(float(x) * 32767 + 0.5)) for x in q]
    factor = dense["scale"] / QUERY_LEVELS
    expected = [s / max(bm25) + 1.5 * (sum(int(a) * b for a, b in zip(row, qi, strict=True)) * factor)
                for s, row in zip(bm25, stored.tolist(), strict=True)]
    assert fused.scores(query, q) == expected == reloaded.scores(query, q), "bit for bit, in plain integers"

    hits = fused.search(query, len(ds.facts), vector=q)
    assert len(hits) == len(ds.facts) and len({h.id for h in hits}) == len(hits), "every fact is a candidate"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert hits[0].id == lexical.docs[0]["id"], "cosine 1 with its own vector and the best lexical match"
    nothing = fused.search("zzzz qqqq", 2, vector=q)   # no lexical evidence at all: the cosine alone decides
    assert lexical.search("zzzz qqqq", 2) == [] and nothing[0].doc == 0
    assert all(h.lang == "en" for h in fused.search(query, 3, lang="en", vector=q))
    with pytest.raises(ValueError, match="one vector per document"):
        Bm25Retriever({**lexical.index, "dense": pack_dense(vectors[:3], 1.0)})
