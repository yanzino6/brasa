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
