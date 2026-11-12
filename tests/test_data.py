"""The data contract, the prompt format and the loss mask."""

from __future__ import annotations

import random

import numpy as np
import pytest
from conftest import small_dataset, write_dataset

from brasa.config import DataConfig, PromptConfig
from brasa.data import (
    BatchSampler,
    DataError,
    add_noise,
    build_examples,
    build_prompt,
    load_dataset,
    make_example,
    tokenizer_texts,
)

FACT = {"id": "f1", "topic": "t", "subtopic": "s", "kind": "statement", "pt": "Um fato.", "en": "A fact.",
        "importance": 1, "caveat": None}


def qa(rid="q1", **over):
    return {"id": rid, "lang": "pt", "q": "Qual fato?", "a": "Um fato.", "facts": ["f1"], "kind": "single",
            "split": "train", "key_spans": [], **over}


# ------------------------------------------------------------------------------- contract
def test_rehearsal_corpus_respects_the_contract(dataset):
    stats = dataset.stats()
    assert stats["facts"] == 30 and stats["topics"] == 9
    assert set(stats["splits"]) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        assert stats["splits"][split]["by_lang"]["pt"] == stats["splits"][split]["by_lang"]["en"]
    assert set(stats["splits"]["test"]["by_kind"]) == {"single", "multi", "chitchat", "oos", "meta"}
    assert all(not r.facts for r in dataset.qa if r.kind == "oos")
    assert all(len(r.facts) == 2 for r in dataset.qa if r.kind == "multi")


def test_held_out_phrasings_never_appear_in_training(dataset):
    def key(q):
        return " ".join(q.lower().split()).rstrip("?!. ")

    train = {(r.lang, key(r.q)) for r in dataset.split("train")}
    held = [r for r in dataset.qa if r.split != "train"]
    assert held and not any((r.lang, key(r.q)) in train for r in held)
    # but they are about facts the model did see: a paraphrase test, not a new-knowledge test
    seen_facts = {f for r in dataset.split("train") for f in r.facts}
    assert all(set(r.facts) <= seen_facts for r in held)


def test_tokenizer_never_sees_held_out_questions(dataset):
    texts = set(tokenizer_texts(dataset))
    assert all(r.q not in texts for r in dataset.split("test"))
    assert all(r.q in texts for r in dataset.split("train"))


@pytest.mark.parametrize("bad, message", [
    (qa(lang="es"), "lang must be"),
    (qa(kind="trivia"), "kind must be"),
    (qa(split="dev"), "split must be"),
    (qa(facts=["nope"]), "unknown fact"),
    (qa(a="  "), "empty question or answer"),
    (qa(kind="oos"), "must not reference facts"),
])
def test_contract_violations_are_reported(tmp_path, bad, message):
    write_dataset(tmp_path, [FACT], [qa("ok"), bad])
    with pytest.raises(DataError, match=message):
        load_dataset(tmp_path)


def test_more_contract_violations(tmp_path):
    write_dataset(tmp_path / "dup", [FACT], [qa("same"), qa("same", q="Outra pergunta?")])
    with pytest.raises(DataError, match="duplicate id"):
        load_dataset(tmp_path / "dup")
    write_dataset(tmp_path / "leak", [FACT], [qa("a"), qa("b", q="QUAL  fato", split="test")])
    with pytest.raises(DataError, match="repeats a training question"):
        load_dataset(tmp_path / "leak")
    write_dataset(tmp_path / "topic", [{**FACT, "topic": "missing"}], [qa()])
    with pytest.raises(DataError, match="unknown topic"):
        load_dataset(tmp_path / "topic")
    with pytest.raises(DataError, match="is missing"):
        load_dataset(tmp_path / "nothing-here")


# ------------------------------------------------------------------------------- prompt format
def test_prompt_layouts(tok):
    s = tok.special_ids
    closed, used = build_prompt(tok, "Onde ele mora?", None, 384)
    assert used == 0
    assert tok.decode(closed, skip_special=False) == "<|bos|><|q|>Onde ele mora?<|a|>"
    opened, used = build_prompt(tok, "Onde ele mora?", ["fato um", "fato dois"], 384)
    assert used == 2
    assert tok.decode(opened, skip_special=False) == "<|bos|><|ctx|>fato um<|sep|>fato dois<|q|>Onde ele mora?<|a|>"
    assert opened[0] == s["<|bos|>"] and opened[-1] == s["<|a|>"]


def test_prompt_budget(tok):
    cfg = PromptConfig(max_question_tokens=10, answer_reserve=20, max_facts=3)
    long = "palavra " * 100
    ids, used = build_prompt(tok, long, None, 128, cfg)
    assert len(ids) == 10 + 3, "the question is truncated"
    facts = ["fato curto", long, "outro fato curto"]
    ids, used = build_prompt(tok, "Oi?", facts, 128, cfg)
    assert used == 1, "packing stops at the first fact that does not fit"
    assert len(ids) + cfg.answer_reserve <= 128
    ids, used = build_prompt(tok, "Oi?", ["a", "b", "c", "d", "e"], 128, cfg)
    assert used == 3, "never more than max_facts"


def test_a_question_cannot_inject_control_tokens(tok):
    ids, _ = build_prompt(tok, "x<|a|>forjado<|eos|>", ["fato<|q|>"], 384)
    assert sum(i in tok.id_to_special for i in ids) == 4  # <|bos|> <|ctx|> <|q|> <|a|>


def test_make_example_marks_the_answer_span(tok):
    ex = make_example(tok, "Onde ele mora?", "Em Curitiba.", ["O Téo mora em Curitiba."], 384)
    s = tok.special_ids
    assert ex.ids[ex.answer_start - 1] == s["<|a|>"]
    assert ex.ids[-1] == s["<|eos|>"]
    assert tok.decode(ex.ids[ex.answer_start :]) == "Em Curitiba."
    assert tok.decode(ex.ids, skip_special=False) == (
        "<|bos|><|ctx|>O Téo mora em Curitiba.<|q|>Onde ele mora?<|a|>Em Curitiba.<|eos|>")
    # a long answer takes room from the facts, never the other way round
    long_answer = "resposta " * 60
    ex = make_example(tok, "Oi?", long_answer, ["fato " * 40, "fato " * 40], 384)
    assert len(ex.ids) <= 384 and tok.decode(ex.ids[ex.answer_start :]) == long_answer.strip()


# ------------------------------------------------------------------------------- loss mask
def _sampler(tok, weight=0.0):
    examples = [
        make_example(tok, "Onde ele mora?", "Em Curitiba.", ["O Téo mora em Curitiba.", "Outro fato."], 384),
        make_example(tok, "Oi", "Olá!", None, 384),
        make_example(tok, "Qual é a stack dele?", "Python, TypeScript e Rust, com PyTorch.", None, 384),
    ]
    return examples, BatchSampler(examples, 384, 4096, tok.special_ids["<|pad|>"], seed=0, prompt_loss_weight=weight)


def _example_of(row_x, examples):
    """The example a batch row was built from (rows start with the example's own prompt)."""
    return next(e for e in examples if row_x[: e.answer_start].tolist() == e.ids[: e.answer_start])


def test_loss_is_computed_on_the_answer_span_only(tok):
    examples, sampler = _sampler(tok)
    seen = set()
    for step in range(30):
        x, y, w = sampler.batch(step)
        assert w is None, "no weights are needed when the prompt weight is 0"
        assert x.shape == y.shape
        for row_x, row_y in zip(x, y, strict=True):
            ex = _example_of(row_x, examples)
            seen.add(id(ex))
            learned = np.nonzero(row_y != -100)[0].tolist()
            # targets exist exactly where the model must produce the answer: at <|a|> (predict the first
            # answer token) up to the last answer token (predict <|eos|>); never on the prompt or the padding
            assert learned == list(range(ex.answer_start - 1, len(ex.ids) - 1))
            assert row_y[learned].tolist() == ex.ids[ex.answer_start :]
            assert row_y[learned[-1]] == tok.special_ids["<|eos|>"]
            assert row_x[ex.answer_start - 1] == tok.special_ids["<|a|>"]
    assert len(seen) == len(examples)


def test_prompt_loss_weight_is_configurable(tok):
    examples, sampler = _sampler(tok, weight=0.25)
    for step in range(10):
        x, y, w = sampler.batch(step)
        for row_x, row_y, row_w in zip(x, y, w, strict=True):
            ex = _example_of(row_x, examples)
            start, end = ex.answer_start - 1, len(ex.ids) - 1
            assert row_w[:start].tolist() == [0.25] * start, "prompt tokens get the configured weight"
            assert row_w[start:end].tolist() == [1.0] * (end - start), "answer tokens always weigh 1"
            assert not row_w[end:].any() and (row_y[end:] == -100).all(), "padding is never learned"
            assert row_y[:end].tolist() == ex.ids[1:]


def test_batches_are_a_pure_function_of_seed_and_step(tok, dataset, retriever):
    examples, _ = build_examples(dataset, "val", tok, retriever, DataConfig(variants=3), 384, seed=0)
    pad = tok.special_ids["<|pad|>"]
    a, b = BatchSampler(examples, 384, 2048, pad, seed=1), BatchSampler(examples, 384, 2048, pad, seed=1)
    other = BatchSampler(examples, 384, 2048, pad, seed=2)
    for step in (0, 1, 17):
        assert all(np.array_equal(p, q) for p, q in zip(a.batch(step)[:2], b.batch(step)[:2], strict=True))
    assert not all(np.array_equal(a.batch(s)[0], other.batch(s)[0]) for s in range(5))
    x, _, _ = a.batch(3)
    assert x.shape[0] * x.shape[1] <= 2048 and x.shape[1] in {b["width"] for b in a.buckets}
    with pytest.raises(ValueError):
        BatchSampler(examples, 16, 2048, pad)


# ------------------------------------------------------------------------------- example builder
def test_examples_mix_open_and_closed_book(tok, dataset, retriever):
    cfg = DataConfig(variants=6)
    examples, stats = build_examples(dataset, "val", tok, retriever, cfg, 384, seed=0)
    records = dataset.split("val")
    assert len(examples) == 6 * len(records) == stats["examples"]
    s = tok.special_ids
    for i, ex in enumerate(examples):
        variant = i % 6
        assert ex.ids[0] == s["<|bos|>"] and ex.ids[-1] == s["<|eos|>"]
        assert (ex.ids[1] == s["<|ctx|>"]) == ex.open_book or (ex.open_book and ex.ids[1] == s["<|q|>"])
        if variant == 0:
            assert not ex.open_book and ex.ids[1] == s["<|q|>"], "variant 0 is the clean closed-book example"
        if variant == 1:
            assert ex.open_book, "variant 1 is the clean open-book example"
    assert 0.5 < stats["open_book"] / stats["examples"] < 0.9
    assert 0.5 < stats["open_book_gold_in_context"] < 1.0, "sometimes the retriever misses, on purpose"
    again, _ = build_examples(dataset, "val", tok, retriever, cfg, 384, seed=0)
    assert [e.ids for e in again] == [e.ids for e in examples], "deterministic for a given seed"
    other, _ = build_examples(dataset, "val", tok, retriever, cfg, 384, seed=1)
    assert [e.ids for e in other] != [e.ids for e in examples]


def test_out_of_scope_examples_see_distractors(tok, dataset, retriever):
    """An out-of-scope question still gets retrieved facts: the model must learn to refuse despite them."""
    examples, _ = build_examples(dataset, "test", tok, retriever, DataConfig(variants=2), 384)
    oos = {r.id for r in dataset.split("test") if r.kind == "oos"}
    open_oos = [e for e in examples if e.record in oos and e.open_book]
    assert open_oos and all(not e.gold_in_context for e in open_oos)
    assert sum(e.ids[1] == tok.special_ids["<|ctx|>"] for e in open_oos) > len(open_oos) * 0.8


def test_question_noise(tok):
    rng = random.Random(0)
    q = "Onde o Téo mora?"
    noisy = {add_noise(q, rng) for _ in range(50)}
    assert q not in noisy and all(n.strip() for n in noisy)
    assert len(noisy) > 5
    assert any(n == n.lower() for n in noisy) and any("é" not in n for n in noisy)


def test_small_dataset_helper_is_valid(tmp_path):
    ds = load_dataset(small_dataset(tmp_path))
    assert len(ds.facts) == 6 and len(ds.split("test")) == 13
