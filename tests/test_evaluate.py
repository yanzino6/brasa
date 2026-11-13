"""Evaluation metrics on hand-made examples, and the batched greedy decoder."""

from __future__ import annotations

import json

import pytest
import torch

from brasa.config import PromptConfig
from brasa.data import QARecord, build_prompt
from brasa.evaluate import (
    MODES,
    FactMatcher,
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
    tune_prior,
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


def test_evaluate_model_runs_every_mode(tiny_model, tok, dataset, retriever):
    """Random weights: the scores are meaningless, the structure is what is checked."""
    from brasa.config import GPTConfig
    from brasa.fixtures import random_model

    model = random_model(GPTConfig(vocab_size=tok.vocab_size, block_size=192, n_layer=1, n_head=2, d_model=32))
    cfg = PromptConfig(answer_reserve=8, max_facts=2)
    results, rows = evaluate_model(model, tok, dataset, retriever, "cpu", cfg, splits=("val",), max_new_tokens=6)
    n = len(dataset.split("val"))
    assert set(results) == set(MODES) and len(rows) == 4 * n
    assert results["open_book"]["val"]["overall"]["n"] == n
    for mode in ("open_book", "open_book_constrained"):
        overall = results[mode]["val"]["overall"]
        assert {"retrieval_hit", "gold_in_top1", "gold_in_top2", "gold_in_top4"} <= set(overall)
        assert overall["gold_in_top1"] <= overall["gold_in_top2"] <= overall["retrieval_hit"]
    assert "retrieval_hit" not in results["closed_book"]["val"]["overall"]
    assert set(results["open_book"]["val"]["by_kind"]) == {"single", "multi", "chitchat", "oos", "meta"}
    open_row = next(r for r in rows if r["mode"] == "open_book")
    ids, used = build_prompt(tok, open_row["q"], [h.text for h in retriever.search(open_row["q"], 2)], 192, cfg)
    assert len(open_row["retrieved"]) == used
    with pytest.raises(ValueError):
        evaluate_model(model, tok, dataset, retriever, "cpu", cfg, splits=("val",), modes=("open",))


def test_constrained_modes_only_write_curated_sentences(tok, dataset, retriever):
    """Whatever the weights are, a constrained answer is a fact, a chain of facts or a fixed answer."""
    from brasa.config import GPTConfig
    from brasa.fixtures import random_model

    # a context long enough for three chained facts: an answer cut by the budget would not be a sentence
    model = random_model(GPTConfig(vocab_size=tok.vocab_size, block_size=1024, n_layer=1, n_head=2, d_model=32))
    modes = ("open_book_constrained", "closed_book_constrained")
    for beam, lang in ((1, "any"), (1, "question"), (2, "question")):
        results, rows = evaluate_model(model, tok, dataset, retriever, "cpu", splits=("test",), modes=modes,
                                       max_new_tokens=600, beam=beam, constrain_lang=lang)
        assert all(r["verbatim"] for r in rows)
        assert results["open_book_constrained"]["test"]["overall"]["verbatim_rate"] == 1.0
        assert results["closed_book_constrained"]["test"]["overall"]["unsupported_number_rate"] is not None
        if lang == "question":
            assert all(r["lang_detected"] in (r["lang"], None) for r in rows)
    free, _ = evaluate_model(model, tok, dataset, retriever, "cpu", splits=("test",), modes=("closed_book",),
                             max_new_tokens=8)
    assert free["closed_book"]["test"]["overall"]["verbatim_rate"] == 0.0, "random weights write no curated sentence"


def test_retrieved_mode_only_writes_what_the_retriever_found_and_first_hit_needs_no_model(tok, dataset, retriever):
    from brasa.config import GPTConfig
    from brasa.fixtures import random_model

    model = random_model(GPTConfig(vocab_size=tok.vocab_size, block_size=1024, n_layer=1, n_head=2, d_model=32))
    modes = ("open_book_retrieved", "first_hit")
    results, rows = evaluate_model(model, tok, dataset, retriever, "cpu", splits=("test",), modes=modes,
                                   max_new_tokens=600)
    restricted = [r for r in rows if r["mode"] == "open_book_retrieved"]
    assert restricted and all(r["verbatim"] for r in restricted)
    for r in restricted:  # a chain of retrieved facts, or a fixed answer: never a fact that was not in the prompt
        assert r["answer_facts"] is None or set(r["answer_facts"]) <= set(r["retrieved"]), r
    overall = results["open_book_retrieved"]["test"]["overall"]
    assert overall["unsupported_number_rate"] in (0.0, None), "every number written is in the context"

    first = [r for r in rows if r["mode"] == "first_hit"]
    hits = {r["id"]: retriever.search(r["q"], 1) for r in first}
    assert all(r["pred"] == (hits[r["id"]][0].text if hits[r["id"]] else "") for r in first)
    rep = results["first_hit"]["test"]
    in_scope = [r for r in first if r["kind"] in ("single", "meta")]
    assert sum(r["em"] for r in in_scope) > 0.5 * len(in_scope), "the rehearsal retriever is right more often than not"
    assert rep["by_kind"]["single"]["em"] == pytest.approx(rep["by_kind"]["single"]["gold_in_top1"], abs=0.05)
    assert rep["overall"]["refusal_recall"] == 0.0, "a retriever never declines"
    with pytest.raises(ValueError):
        evaluate_model(model, tok, dataset, None, "cpu", splits=("test",), modes=("first_hit",))


def test_ranked_mode_lets_the_retriever_weigh_in(tok, dataset, retriever):
    """open_book_ranked: no prior = open_book_retrieved; a prior strong enough turns a random model into the
    retriever's first hit (whenever it answers with a fact), and a prior on the fixed answers into a refuser."""
    from brasa.config import GPTConfig
    from brasa.fixtures import random_model
    from brasa.retrieval import Bm25Retriever

    model = random_model(GPTConfig(vocab_size=tok.vocab_size, block_size=1024, n_layer=1, n_head=2, d_model=32))

    def run(modes, used=retriever, **prior):
        _, rows = evaluate_model(model, tok, dataset, used, "cpu", splits=("test",), modes=modes, max_new_tokens=600,
                                 prior=prior or None)
        return {mode: [r for r in rows if r["mode"] == mode] for mode in modes}

    base = run(("open_book_retrieved", "open_book_ranked", "first_hit"))
    preds = {mode: [r["pred"] for r in rows] for mode, rows in base.items()}
    assert preds["open_book_ranked"] == preds["open_book_retrieved"], "weight 0: the model alone decides"

    def starts_with_the_first_hit(rows) -> bool:  # in either language, possibly followed by other facts
        return all(r["answer_facts"] and r["answer_facts"][0] == r["retrieved"][0] for r in rows if r["retrieved"])

    strong = run(("open_book_ranked",), kind="rank", weight=200.0, fixed=-400.0)["open_book_ranked"]
    assert starts_with_the_first_hit(strong) and not starts_with_the_first_hit(base["open_book_retrieved"])
    assert starts_with_the_first_hit(run(("open_book_ranked",), kind="score", weight=5000.0, fixed=-9000.0)
                                     ["open_book_ranked"])

    fixed = {c["text"] for c in retriever.index["answers"]["canonical"]}
    refuser = run(("open_book_ranked",), kind="rank", weight=1.0, fixed=500.0)["open_book_ranked"]
    assert all(r["pred"] in fixed for r in refuser), "a prior on the fixed answers makes every answer a fixed one"

    # tune_prior: every weight is tried on validation, the smallest within the tolerance of the best is kept
    tuned = tune_prior(model, tok, dataset, retriever, "cpu", PromptConfig(), kind="rank", weights=(0.0, 3.0, 300.0))
    ems = {row["weight"]: row["em"] for row in tuned["sweep"]}
    assert tuned["split"] == "val" and set(ems) == {0.0, 3.0, 300.0} and tuned["prior"]["kind"] == "rank"
    assert tuned["prior"]["weight"] == min(w for w, em in ems.items() if em >= max(ems.values()) - tuned["tolerance"])
    loose = tune_prior(model, tok, dataset, retriever, "cpu", PromptConfig(), kind="rank", weights=(0.0, 300.0),
                       tolerance=1.0)
    assert loose["prior"]["weight"] == 0.0, "inside the tolerance the weakest prior wins"

    # the prior stored in the index (what an export ships) is the default
    stored = {**retriever.index["answers"], "prior": {"kind": "rank", "weight": 200.0, "fixed": -400.0}}
    shipped = Bm25Retriever({**retriever.index, "answers": stored})
    assert [r["pred"] for r in run(("open_book_ranked",), shipped)["open_book_ranked"]] == [r["pred"] for r in strong]


def test_fact_matcher_recognises_curated_sentences(dataset):
    matcher = FactMatcher(dataset)
    a, b, c, d = list(dataset.facts.values())[:4]
    assert matcher.facts(a.pt) == [a.id] and matcher.facts(a.en.upper().rstrip(".")) == [a.id]
    assert matcher.facts(f"{a.pt} {b.pt}") == [a.id, b.id]
    assert matcher.facts(f"{c.en}  {a.en} {b.en}") == [c.id, a.id, b.id]
    assert matcher.facts(f"{a.pt} {b.pt} {c.pt} {d.pt}") is None, "more than three sentences"
    assert matcher.facts(a.pt + " e mais") is None and matcher.facts(a.pt[: len(a.pt) // 2]) is None
    assert matcher.facts("") is None and matcher.facts("something else entirely") is None
    refusal = next(r.a for r in dataset.qa if r.kind == "oos")
    assert matcher.is_canonical(refusal.lower()) and not matcher.is_canonical(a.pt)


def test_visitor_metrics(dataset):
    """verbatim_rate, right_topic_rate and gold_in_top_k on answers written by hand."""
    facts = list(dataset.facts.values())
    gold = facts[0]
    sibling = next(f for f in facts if f.topic == gold.topic and f.id != gold.id)
    other = next(f for f in facts if f.topic != gold.topic)
    refusal = next(r.a for r in dataset.qa if r.kind == "oos" and r.lang == "pt")
    rec = QARecord("x", "pt", "pergunta?", gold.pt, (gold.id,), "single", "test")
    oos = QARecord("y", "pt", "capital da França?", refusal, (), "oos", "test")

    def score(record, pred, retrieved):
        return score_record(record, pred, [], [refusal], dataset, retrieved)

    rows = [
        score(rec, gold.pt, [gold.id, other.id]),                      # exact
        score(rec, sibling.pt, [other.id, gold.id]),                    # another fact of the right topic
        score(rec, f"{gold.pt} {sibling.pt}", [other.id, sibling.id, gold.id]),   # two facts of the right topic
        score(rec, other.pt, [other.id]),                               # a true sentence about something else
        score(rec, gold.pt.replace(" ", " mesmo ", 1), [other.id]),     # a sentence nobody curated
        score(rec, refusal, [other.id]),                                # a fixed answer: verbatim, not the topic
        score(oos, refusal, [other.id]),                                # no gold fact: no topic to be right about
    ]
    assert [r["verbatim"] for r in rows] == [True, True, True, True, False, True, True]
    assert [r["right_topic"] for r in rows] == [True, True, True, False, False, False, None]
    assert [r["answer_facts"] for r in rows[:3]] == [[gold.id], [sibling.id], [gold.id, sibling.id]]
    assert [r["gold_in_top1"] for r in rows] == [True, False, False, False, False, False, None]
    assert [r["gold_in_top2"] for r in rows] == [True, True, False, False, False, False, None]
    assert [r["gold_in_top4"] for r in rows] == [True, True, True, False, False, False, None]
    agg = aggregate(rows)
    assert agg["verbatim_rate"] == pytest.approx(6 / 7, abs=1e-4)
    assert agg["right_topic_rate"] == pytest.approx(3 / 6, abs=1e-4), "over the questions that have gold facts"
    assert (agg["gold_in_top1"], agg["gold_in_top2"], agg["gold_in_top4"]) == (round(1 / 6, 4), round(2 / 6, 4), 0.5)
    assert agg["retrieval_hit"] == agg["gold_in_top4"]
    by_lang = report(rows)["by_lang"]["pt"]
    assert by_lang["verbatim_rate"] == agg["verbatim_rate"], "every group carries the visitor metrics"
    closed = aggregate([score(rec, gold.pt, None)])
    assert "gold_in_top1" not in closed and closed["right_topic_rate"] == 1.0

    two = QARecord("z", "pt", "duas coisas?", f"{gold.pt} {sibling.pt}", (gold.id, sibling.id), "multi", "test")
    row = score(two, f"{gold.pt} {sibling.pt}", [gold.id, sibling.id])
    assert (row["gold_in_top1"], row["gold_in_top2"]) == (False, True), "two gold facts cannot fit in one hit"


def test_model_card_sections_are_merged(tmp_path):
    card = tmp_path / "model-card.json"
    update_model_card(card, {"name": "brasa-4m", "evaluation": {"float32": {"em": 0.5}}})
    update_model_card(card, {"evaluation": {"int8": {"em": 0.49}}, "files": {"model.i8.bin": {"bytes": 1}}})
    data = json.loads(card.read_text(encoding="utf-8"))
    assert data["evaluation"] == {"float32": {"em": 0.5}, "int8": {"em": 0.49}}
    assert data["name"] == "brasa-4m" and data["schema_version"] == 1 and "updated_at" in data
