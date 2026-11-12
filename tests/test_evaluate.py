"""Evaluation metrics on hand-made examples, and the batched greedy decoder."""

from __future__ import annotations

import json

import pytest
import torch

from brasa.config import PromptConfig
from brasa.data import QARecord, build_prompt
from brasa.evaluate import (
    aggregate,
    contains_span,
    detect_language,
    evaluate_model,
    exact_match,
    f1_score,
    generate_answers,
    headline,
    is_refusal,
    numbers,
    report,
    score_record,
    update_model_card,
)

REFUSALS = ["I can only answer questions about Téo Arantes and his work.",
            "Só sei responder perguntas sobre o Téo Arantes e o trabalho dele."]


def test_exact_match_ignores_case_punctuation_and_spacing():
    assert exact_match("He lives in Curitiba.", "he lives in  Curitiba") == 1.0
    assert exact_match("He lives in Curitiba", "He lived in Curitiba") == 0.0
    assert exact_match("mora em São Paulo", "mora em Sao Paulo") == 0.0, "accents are part of the answer"
    assert exact_match("", "") == 1.0 and exact_match("", "x") == 0.0


def test_token_f1():
    assert f1_score("a b c", "a b c") == 1.0
    assert f1_score("x y", "a b") == 0.0
    assert f1_score("", "a") == 0.0 and f1_score("", "") == 1.0
    # 2 of 3 predicted tokens are right, 2 of 4 gold tokens are found
    assert f1_score("a b x", "a b c d") == pytest.approx(2 * (2 / 3) * (2 / 4) / (2 / 3 + 2 / 4))
    assert f1_score("a a a", "a b") == pytest.approx(2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2)), "repeats count once"


def test_key_spans_match_whole_tokens():
    assert contains_span("It cut fraud losses by 40 percent.", "40")
    assert not contains_span("It cut fraud losses by 140 percent.", "40")
    assert contains_span("He studied at the Federal University of Paraná.", "federal university of paraná")
    assert not contains_span("He studied at the University of Paraná.", "Federal University of Paraná")
    assert contains_span("Write to hello@lumenforge.example today", "hello@lumenforge.example")
    assert not contains_span("anything", "")


def test_language_detection():
    assert detect_language("He lives in Curitiba, in the south of Brazil.") == "en"
    assert detect_language("Ele mora em Curitiba, no sul do Brasil.") == "pt"
    assert detect_language("Só sei responder perguntas sobre o Téo.") == "pt"
    assert detect_language("Python, TypeScript, Rust") is None, "no evidence either way"


def test_numbers():
    assert numbers("Since 2019 it cut losses by 40% and costs R$ 1.234,56.") == ["2019", "40", "1.234.56"]
    assert numbers("nine years") == [], "spelled-out numbers are not checked"


def test_refusal_detection():
    assert is_refusal("I can only answer questions about Téo Arantes and his work.", REFUSALS)
    assert is_refusal("I can only answer questions about Téo and his work", REFUSALS), "near matches count"
    assert not is_refusal("Téo Arantes is a software developer.", REFUSALS)
    assert not is_refusal("", REFUSALS)


def _row(kind="single", lang="en", em=1.0, f1=1.0, spans=(1, 1), refused=False, detected="same", nums=(0, 0),
         hit=None):
    row = {"kind": kind, "lang": lang, "topic": "t", "em": em, "f1": f1, "spans_total": spans[0],
           "spans_found": spans[1], "refused": refused, "lang_detected": lang if detected == "same" else detected,
           "numbers": nums[0], "unsupported_numbers": nums[1]}
    if hit is not None:
        row["retrieval_hit"] = hit
    return row


def test_aggregate_on_hand_made_rows():
    rows = [
        _row(em=1, f1=1.0, spans=(2, 2), nums=(1, 0), hit=True),            # right
        _row(em=0, f1=0.5, spans=(2, 1), nums=(2, 1), hit=True),            # half right, one invented number
        _row(em=0, f1=0.0, spans=(1, 0), refused=True, hit=False),          # refused an in-scope question
        _row(kind="oos", em=1, f1=1.0, spans=(0, 0), refused=True),         # refused correctly
        _row(kind="oos", em=0, f1=0.1, spans=(0, 0), detected="pt"),        # answered out of scope, wrong language
        _row(kind="oos", lang="pt", em=1, f1=1.0, spans=(0, 0), refused=True, detected=None),
    ]
    m = aggregate(rows)
    near = lambda x: pytest.approx(x, abs=1e-4)  # noqa: E731  (aggregate rounds every ratio to 4 decimals)
    assert m["n"] == 6
    assert m["em"] == near(3 / 6)
    assert m["f1"] == near((1 + 0.5 + 0 + 1 + 0.1 + 1) / 6)
    assert m["key_span_recall"] == near(3 / 5)
    assert m["refusal_rate"] == near(3 / 6)
    assert m["refusal_precision"] == near(2 / 3), "3 refusals, 2 of them on out-of-scope questions"
    assert m["refusal_recall"] == near(2 / 3), "3 out-of-scope questions, 2 refused"
    assert m["language_consistency"] == near(4 / 5), "undetected answers are left out"
    assert m["unsupported_number_rate"] == near(1 / 3)
    assert m["answers_with_unsupported_number"] == near(1 / 6)
    assert m["retrieval_hit"] == near(2 / 3)


def test_aggregate_handles_empty_denominators():
    m = aggregate([_row(spans=(0, 0))])
    assert m["key_span_recall"] is None and m["refusal_precision"] is None and m["refusal_recall"] is None
    assert m["unsupported_number_rate"] is None and "retrieval_hit" not in m


def test_report_groups_by_language_kind_and_topic():
    rows = [_row(lang="en"), _row(lang="pt", em=0, f1=0.2), _row(kind="oos", lang="pt", refused=True, spans=(0, 0))]
    r = report(rows)
    assert set(r) == {"overall", "by_lang", "by_kind", "by_topic"}
    assert r["by_lang"]["pt"]["n"] == 2 and r["by_lang"]["en"]["em"] == 1.0
    assert r["by_kind"]["oos"]["refusal_recall"] == 1.0 and r["by_kind"]["single"]["refusal_rate"] == 0.0


def test_report_separates_retriever_misses_from_model_errors():
    rows = [_row(hit=True), _row(hit=True, em=0, f1=0.5), _row(hit=False, em=0, f1=0.1),
            _row(kind="oos", refused=True, spans=(0, 0))]  # no gold fact: in neither group
    r = report(rows)["by_retrieval"]
    assert r["gold_retrieved"]["n"] == 2 and r["gold_retrieved"]["em"] == 0.5
    assert r["gold_missed"]["n"] == 1 and r["gold_missed"]["em"] == 0.0
    h = headline({"open_book": {"test": report(rows)}})["open_book"]["test"]
    assert h["em_when_gold_retrieved"] == 0.5 and h["em_when_gold_missed"] == 0.0 and h["retrieval_hit"] == 0.6667


def test_score_record_unsupported_numbers(dataset):
    rec = QARecord("x", "en", "What did Sentinela achieve in 2022?", "It cut fraud losses by 40 percent.",
                   ("sentinela",), "single", "test", ("Sentinela", "40"))
    context = ["Sentinela processes 3 million events per day."]
    good = score_record(rec, "Sentinela cut fraud losses by 40 percent.", context, REFUSALS, dataset, ["sentinela"])
    assert (good["numbers"], good["unsupported_numbers"]) == (1, 0)
    assert (good["spans_total"], good["spans_found"]) == (2, 2)
    assert good["retrieval_hit"] is True and good["topic"] == "sentinela" and good["lang_detected"] == "en"

    from_context = score_record(rec, "It handles 3 million events since 2022.", context, REFUSALS, dataset, ["who"])
    assert (from_context["numbers"], from_context["unsupported_numbers"]) == (2, 0), "context and question count"
    assert from_context["retrieval_hit"] is False

    invented = score_record(rec, "It cut fraud by 75 percent in 2021.", context, REFUSALS, dataset, None)
    assert (invented["numbers"], invented["unsupported_numbers"]) == (2, 2)
    assert "retrieval_hit" not in invented and invented["em"] == 0.0 and 0 < invented["f1"] < 1


def test_headline_separates_seen_from_held_out():
    rows = [_row(), _row(kind="oos", refused=True, spans=(0, 0), em=0, f1=0.4)]
    results = {"open_book": {"train": report(rows), "test": report(rows[:1])}}
    h = headline(results)
    assert set(h["open_book"]) == {"seen", "test"}
    assert h["open_book"]["seen"]["em"] == 0.5 and h["open_book"]["seen"]["in_scope_em"] == 1.0
    assert h["open_book"]["test"]["refusal_recall"] is None


def test_batched_greedy_decoding_equals_one_at_a_time(tok):
    """Right-padded batches must give the answers of one-prompt-at-a-time generation with the KV cache."""
    from brasa.config import GPTConfig
    from brasa.fixtures import random_model

    model = random_model(GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=4, d_model=48))
    eos = tok.special_ids["<|eos|>"]
    torch.manual_seed(1)
    prompts = [torch.randint(0, 256, (n,)).tolist() for n in (3, 9, 9, 17, 30, 55)]
    batched = generate_answers(model, tok, prompts, max_new_tokens=12, device="cpu", max_tokens=200)
    for prompt, got in zip(prompts, batched, strict=True):
        budget = min(12, 64 - len(prompt))
        single = model.generate(torch.tensor([prompt]), budget, stop_ids=(eos,))[0].tolist()
        if eos in single:
            single = single[: single.index(eos)]
        assert got == tok.decode(single).strip()


def test_evaluate_model_runs_both_modes(tiny_model, tok, dataset, retriever):
    """Random weights: the scores are meaningless, the structure is what is checked."""
    from brasa.config import GPTConfig
    from brasa.fixtures import random_model

    model = random_model(GPTConfig(vocab_size=tok.vocab_size, block_size=192, n_layer=1, n_head=2, d_model=32))
    cfg = PromptConfig(answer_reserve=8, max_facts=2)
    results, rows = evaluate_model(model, tok, dataset, retriever, "cpu", cfg, splits=("val",), max_new_tokens=6)
    n = len(dataset.split("val"))
    assert set(results) == {"open_book", "closed_book"} and len(rows) == 2 * n
    assert results["open_book"]["val"]["overall"]["n"] == n
    assert "retrieval_hit" in results["open_book"]["val"]["overall"]
    assert "retrieval_hit" not in results["closed_book"]["val"]["overall"]
    assert set(results["open_book"]["val"]["by_kind"]) == {"single", "multi", "chitchat", "oos", "meta"}
    open_row = next(r for r in rows if r["mode"] == "open_book")
    ids, used = build_prompt(tok, open_row["q"], [h.text for h in retriever.search(open_row["q"], 2)], 192, cfg)
    assert len(open_row["retrieved"]) == used


def test_model_card_sections_are_merged(tmp_path):
    card = tmp_path / "model-card.json"
    update_model_card(card, {"name": "brasa-4m", "evaluation": {"float32": {"em": 0.5}}})
    update_model_card(card, {"evaluation": {"int8": {"em": 0.49}}, "files": {"model.i8.bin": {"bytes": 1}}})
    data = json.loads(card.read_text(encoding="utf-8"))
    assert data["evaluation"] == {"float32": {"em": 0.5}, "int8": {"em": 0.49}}
    assert data["name"] == "brasa-4m" and data["schema_version"] == 1 and "updated_at" in data
