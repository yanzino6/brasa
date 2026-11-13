"""brasa.sample: the terminal chat answers the way the browser session does."""

from __future__ import annotations

import json

import pytest
from conftest import TINY

from brasa.encoder import DenseRetriever
from brasa.retrieval import Bm25Retriever
from brasa.sample import answer, load_export, main

QUESTION = "Quais linguagens de programação ele usa?"


def test_an_exported_index_with_fact_vectors_ranks_with_the_model():
    model, tok, lexical, prompt_cfg = load_export(TINY, "i8")
    assert lexical.dense is not None
    fused = DenseRetriever.from_index(model, tok, lexical)
    assert fused.index is lexical.index and fused.weight == lexical.dense["weight"]
    hits = fused.search(QUESTION, 3)
    assert [h.score for h in hits] != [h.score for h in lexical.search(QUESTION, 3)]
    assert hits == lexical.search(QUESTION, 3, None, fused.vector(QUESTION).numpy()), "the index's own fusion"
    plain = Bm25Retriever({k: v for k, v in lexical.index.items() if k != "dense"})
    with pytest.raises(ValueError, match="no fact vectors"):
        DenseRetriever.from_index(model, tok, plain)

    # the tiny model may only write 16 tokens: its answer is a fixed answer or a retrieved fact, or the start of one
    out = answer(model, tok, fused, QUESTION, prompt_cfg, constrain="retrieved")
    allowed = {h.text for h in out["hits"]} | {c["text"] for c in lexical.index["answers"]["canonical"]}
    assert out["facts_used"] >= 1 and out["answer"]
    assert any(text.startswith(out["answer"]) or out["answer"].startswith(text) for text in allowed)
    whole = answer(model, tok, fused, QUESTION, prompt_cfg, constrain="kb", open_book=False)
    assert whole["hits"] == [] and whole["answer"]
    free = answer(model, tok, fused, QUESTION, prompt_cfg, max_new_tokens=5)
    assert len(free["tokens"]) <= 5
    with pytest.raises(ValueError, match="needs a retrieval index"):
        answer(model, tok, None, QUESTION, prompt_cfg, constrain="kb")


def test_command_line_reports_how_it_answered(capsys):
    main(["--export", str(TINY), "-q", QUESTION, "--json"])
    first = json.loads(capsys.readouterr().out)
    assert first["retriever"] == "bm25+dense" and first["constrain"] == "retrieved" and first["mode"] == "open"
    assert first["prior"] == {"kind": "score", "weight": 6.0, "fixed": 0.0}, "the prior the export ships"
    main(["--export", str(TINY), "-q", QUESTION, "--json", "--prior-weight", "0"])
    assert json.loads(capsys.readouterr().out)["prior"]["weight"] == 0.0
    main(["--export", str(TINY), "-q", QUESTION, "--json", "--lexical", "--constrain", "off", "--max-new-tokens", "4"])
    second = json.loads(capsys.readouterr().out)
    assert second["retriever"] == "bm25" and second["constrain"] is None and second["tokens"] <= 4
    assert second["prior"] is None
    main(["--export", str(TINY), "-q", QUESTION, "--json", "--closed-book"])
    third = json.loads(capsys.readouterr().out)
    assert third["retriever"] is None and third["constrain"] == "kb" and third["facts"] == []
    assert third["prior"] is None, "nothing was retrieved: no ranking to lean on"


def test_a_strong_prior_makes_the_terminal_answer_the_first_hit():
    model, tok, lexical, prompt_cfg = load_export(TINY, "i8")
    out = answer(model, tok, lexical, QUESTION, prompt_cfg, constrain="retrieved", max_new_tokens=100,
                 prior={"kind": "rank", "weight": 500.0, "fixed": -900.0})
    first = out["hits"][0]
    texts = [d["text"] for d in lexical.docs if d["id"] == first.id]
    assert any(out["answer"].startswith(text) for text in texts), "the first hit, in either language"
    alone = answer(model, tok, lexical, QUESTION, prompt_cfg, constrain="retrieved", prior={})
    assert alone["prior"] == {"kind": "score", "weight": 0.0, "fixed": 0.0}
