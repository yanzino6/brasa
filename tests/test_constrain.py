"""Constrained decoding: the token trie of allowed answers and the greedy / beam search over it."""

from __future__ import annotations

import pytest
import torch

from brasa.config import EOS, GPTConfig
from brasa.constrain import AnswerTrie, constrained_answers, continuation_ids, ranking_priors
from brasa.data import build_prompt
from brasa.evaluate import answer_tries
from brasa.fixtures import random_model

PT = ["O Téo mora em Curitiba.", "O Téo trabalha com PyTorch desde 2019.", "O Téo mora em Curitiba desde 2015."]
EN = ["Téo lives in Curitiba.", "Téo has worked with PyTorch since 2019."]
FIXED = ["Isso eu não sei responder.", "I can't answer that."]


@pytest.fixture()
def trie(tok):
    sentences = [(t, "pt") for t in PT] + [(t, "en") for t in EN] + [(t, None) for t in FIXED]
    return AnswerTrie.build(tok, sentences, max_sentences=3)


@pytest.fixture()
def model(tok):
    return random_model(GPTConfig(vocab_size=tok.vocab_size, block_size=192, n_layer=1, n_head=2, d_model=32))


def test_a_following_sentence_has_the_ids_it_gets_inside_a_joined_answer(tok, dataset):
    facts = [f.text(lang) for f in list(dataset.facts.values())[:12] for lang in ("pt", "en")]
    for a, b in zip(facts, facts[1:], strict=False):
        assert tok.encode(f"{a} {b}") == tok.encode(a) + continuation_ids(tok, b)


def test_trie_accepts_exactly_the_allowed_answers(tok, trie):
    def ok(text: str) -> bool:
        return trie.accepts(tok.encode(text))

    assert all(ok(t) for t in PT + EN + FIXED)
    assert ok(f"{PT[0]} {PT[1]}") and ok(f"{PT[1]} {PT[0]} {PT[2]}") and ok(f"{EN[1]} {EN[0]}")
    assert not ok(f"{PT[0]} {EN[0]}"), "an answer does not change language"
    assert not ok(f"{PT[0]} {PT[1]} {PT[2]} {PT[0]}"), "at most three sentences"
    assert not ok(f"{FIXED[0]} {PT[0]}") and not ok(f"{PT[0]} {FIXED[0]}"), "a fixed answer stands alone"
    assert not ok("O Téo mora em") and not ok("O Téo mora em Curitiba. O Téo") and not ok("")
    assert not ok("O Téo mora em Recife."), "a sentence nobody curated"


def test_allowed_tokens_and_eos(tok, trie):
    eos = tok.special_ids[EOS]
    first = {tok.encode(t)[0] for t in PT + EN + FIXED}
    assert trie.allowed(trie.start()) == sorted(first) and eos not in first
    state = trie.start()
    for t in tok.encode(PT[1]):
        assert eos not in trie.allowed(state)
        state = trie.advance(state, t)
    following = {continuation_ids(tok, t)[0] for t in PT}
    assert trie.allowed(state) == sorted(following | {eos})
    assert trie.advance(state, eos) == () and trie.allowed(()) == []
    with pytest.raises(ValueError):
        trie.advance(trie.start(), eos)


def test_a_sentence_that_is_a_prefix_of_another_keeps_both_readings(tok):
    """ "A." followed by the sentence "B." is the same token sequence as the single sentence "A. B."."""
    a, b = "O Téo mora em Curitiba.", "Ele gosta de café."
    trie = AnswerTrie.build(tok, [(a, "pt"), (b, "pt"), (f"{a} {b}", "pt")], max_sentences=2)
    state = trie.start()
    for t in tok.encode(f"{a} {b}"):
        state = trie.advance(state, t)
    assert sorted(count for _, count in state) == [0, 1], "one sentence, or two"
    assert trie.accepts(tok.encode(f"{a} {b} {a}")), "the long sentence counts once, so one more fits"
    assert trie.accepts(tok.encode(f"{a} {b} {a} {b}")), "the long sentence twice"
    assert not trie.accepts(tok.encode(f"{a} {b} {a} {b} {a}"))


def test_max_sentences_one_allows_no_chain(tok):
    trie = AnswerTrie.build(tok, [(t, "pt") for t in PT], max_sentences=1)
    assert trie.accepts(tok.encode(PT[0])) and not trie.accepts(tok.encode(f"{PT[0]} {PT[1]}"))
    assert trie.describe()["sentences"] == 3


def _reference_greedy(model, prompt, trie, budget):
    """One forward pass per token, no shortcuts."""
    state, out = trie.start(), []
    while state and len(out) < budget:
        with torch.no_grad():
            logits, _, _ = model(torch.tensor([prompt + out]))
        options = trie.allowed(state)
        values = logits[0, -1].float().log_softmax(-1)[options].tolist()
        token = options[max(range(len(options)), key=lambda i: (values[i], -options[i]))]
        out.append(token)
        state = trie.advance(state, token)
    return out


def test_greedy_matches_a_step_by_step_reference_and_only_writes_allowed_answers(tok, trie, model):
    eos = tok.special_ids[EOS]
    questions = ["Onde o Téo mora?", "What does he work with?", "oi", "Qual o time dele?", "PyTorch?"]
    prompts = [build_prompt(tok, q, PT[:2] if i % 2 else None, 192)[0] for i, q in enumerate(questions)]
    answers, ids = constrained_answers(model, tok, prompts, trie, 96, "cpu", return_ids=True)
    for prompt, answer, got in zip(prompts, answers, ids, strict=True):
        ref = _reference_greedy(model, prompt, trie, 96)
        assert ref[-1] == eos and got == ref[:-1]
        assert trie.accepts(got) and answer == tok.decode(got)


def test_beam_search_finds_the_most_probable_allowed_answer(tok, model):
    """With a beam as wide as the trie the search is exhaustive: compare with teacher-forced scores."""
    sentences = PT + EN + FIXED
    trie = AnswerTrie.build(tok, [(t, None) for t in sentences])
    eos = tok.special_ids[EOS]
    for question in ("Onde o Téo mora?", "hello", "PyTorch desde quando?"):
        prompt = build_prompt(tok, question, None, 192)[0]
        scores = []
        for text in sentences:
            target = tok.encode(text) + [eos]
            with torch.no_grad():
                logits, _, _ = model(torch.tensor([prompt + target[:-1]]))
            logp = logits[0, len(prompt) - 1 :].float().log_softmax(-1)
            scores.append(float(sum(logp[i, t] for i, t in enumerate(target))))
        best = sentences[max(range(len(sentences)), key=scores.__getitem__)]
        assert constrained_answers(model, tok, [prompt], trie, 96, "cpu", beam=64) == [best]
        wide = constrained_answers(model, tok, [prompt], trie, 96, "cpu", beam=4)[0]
        assert wide in sentences


def test_budget_and_batching(tok, trie, model):
    prompts = [build_prompt(tok, q, None, 192)[0] for q in ("a", "Onde o Téo mora hoje em dia?", "b c d")]
    one_by_one = [constrained_answers(model, tok, [p], trie, 96, "cpu")[0] for p in prompts]
    assert constrained_answers(model, tok, prompts, [trie] * 3, 96, "cpu", max_tokens=40) == one_by_one
    short, ids = constrained_answers(model, tok, prompts, trie, 3, "cpu", return_ids=True)
    assert all(len(i) == 3 for i in ids), "the budget cuts the answer, it never fails"
    assert len(constrained_answers(model, tok, prompts, trie, 3, "cpu", beam=3, return_ids=True)[1][0]) == 3


def test_tries_of_a_dataset_match_its_retrieval_index(tok, dataset, retriever):
    from_index = answer_tries(dataset, tok, retriever)
    from_data = answer_tries(dataset, tok, None)
    assert set(from_index) == {"pt", "en", None}
    for lang in from_index:
        assert from_index[lang].describe() == from_data[lang].describe()
    fixed = {(r.lang, r.a) for r in dataset.split("train") if not r.facts}
    assert fixed and len(retriever.index["answers"]["canonical"]) == len(fixed)
    fact = next(iter(dataset.facts.values()))
    assert from_index["pt"].accepts(tok.encode(fact.pt)) and not from_index["pt"].accepts(tok.encode(fact.en))
    assert from_index[None].accepts(tok.encode(fact.en))
    for lang, text in fixed:
        assert from_index[lang].accepts(tok.encode(text))


# --------------------------------------------------------------------------------------- the retriever as a prior
def test_ranking_priors_from_ranks_and_from_scores():
    from brasa.retrieval import Hit

    hits = [Hit(f"f{i}", "pt", score, "", "t", i) for i, score in enumerate((8.0, 6.0, 2.0))]
    assert ranking_priors(hits, 1.5) == {"f0": -0.0, "f1": -1.5, "f2": -3.0}
    assert ranking_priors(hits, 4.0, "score") == {"f0": 0.0, "f1": -1.0, "f2": -3.0}, "distance to the best score"
    assert ranking_priors(hits, 0.0) == {} and ranking_priors([], 2.0, "score") == {}, "no weight, no prior"
    flat = [Hit("a", "pt", 0.0, "", "t", 0), Hit("b", "pt", -1.0, "", "t", 1)]
    assert ranking_priors(flat, 2.0, "score") == {"a": -0.0, "b": -2.0}, "no positive best score: ranks"
    with pytest.raises(ValueError, match="unknown prior"):
        ranking_priors(hits, 1.0, "magic")


def test_bias_is_the_best_prior_behind_each_token(tok):
    eos = tok.special_ids[EOS]
    plain = AnswerTrie.build(tok, [(t, "pt") for t in PT] + [(FIXED[0], None)])
    assert not plain.has_prior and set(plain.bias(plain.start())) == {0.0}, "no priors: nothing changes"

    # PT[0] and PT[2] share "O Téo mora em Curitiba" and part there; PT[1] parts earlier; the fixed answer at once
    trie = AnswerTrie.build(tok, [(PT[0], "pt", -1.0), (PT[1], "pt", -3.0), (PT[2], "pt", 0.0), (FIXED[0], None, -0.5)])
    assert trie.has_prior
    start = dict(zip(trie.allowed(trie.start()), trie.bias(trie.start()), strict=True))
    assert start == {tok.encode(PT[0])[0]: 0.0, tok.encode(FIXED[0])[0]: -0.5}, "the best sentence behind the token"
    state, seen = trie.start(), []
    for t in tok.encode(PT[0]):
        seen.append(dict(zip(trie.allowed(state), trie.bias(state), strict=True)))
        state = trie.advance(state, t)
    assert any(sorted(step.values()) == [-3.0, 0.0] for step in seen), "where PT[1] leaves the shared prefix"
    assert any(sorted(step.values()) == [-1.0, 0.0] for step in seen), "where PT[0] and PT[2] part"
    end = dict(zip(trie.allowed(state), trie.bias(state), strict=True))
    assert end[eos] == -1.0, "<|eos|> is worth the sentence that ends here"
    assert set(end.values()) == {-1.0}, "going on with another fact costs what stopping costs: the model decides"
    state = trie.advance(state, continuation_ids(tok, PT[1])[0])
    assert set(trie.bias(state)) == {0.0}, "the sentences that follow carry no prior"


def test_a_prior_moves_the_greedy_answer_and_matches_a_reference(tok, model):
    sentences = PT[:2] + EN[:1] + FIXED[:1]
    prompt = build_prompt(tok, "Onde o Téo mora?", None, 192)[0]
    plain = AnswerTrie.build(tok, [(t, None) for t in sentences])
    free = constrained_answers(model, tok, [prompt], plain, 96, "cpu")[0]
    for target in sentences:
        trie = AnswerTrie.build(tok, [(t, None, 0.0 if t == target else -50.0) for t in sentences])
        assert constrained_answers(model, tok, [prompt], trie, 96, "cpu") == [target], "a strong prior decides"
    mild = AnswerTrie.build(tok, [(t, None, -1e-6 * i) for i, t in enumerate(sentences)])
    assert constrained_answers(model, tok, [prompt], mild, 96, "cpu") == [free], "a negligible prior does not"

    # step by step, with the bias added to the log-probabilities of the allowed tokens
    trie = AnswerTrie.build(tok, [(t, "pt", -0.7 * i) for i, t in enumerate(PT)] + [(FIXED[0], None, -0.3)])
    state, out = trie.start(), []
    while state:
        with torch.no_grad():
            logits, _, _ = model(torch.tensor([prompt + out]))
        options = trie.allowed(state)
        values = logits[0, -1].float().log_softmax(-1)[options].tolist()
        values = [v + b for v, b in zip(values, trie.bias(state), strict=True)]
        out.append(options[max(range(len(options)), key=lambda i: (values[i], -options[i]))])
        state = trie.advance(state, out[-1])
    assert constrained_answers(model, tok, [prompt], trie, 96, "cpu", return_ids=True)[1] == [out[:-1]]


def test_from_index_gives_the_retrieved_facts_their_priors(tok, retriever):
    hits = retriever.search("Onde o Téo mora?", 3)
    priors = ranking_priors(hits, 2.0)
    trie = AnswerTrie.from_index(retriever.index, tok, only={h.id for h in hits}, priors=priors, fixed_prior=-1.0)
    assert trie.has_prior and max(trie.bias(trie.start())) == 0.0
    first = tok.encode(hits[0].text)
    assert dict(zip(trie.allowed(trie.start()), trie.bias(trie.start()), strict=True))[first[0]] == 0.0
    fixed = retriever.index["answers"]["canonical"][0]["text"]
    state = trie.start()
    for t in tok.encode(fixed):
        state = trie.advance(state, t)
    assert trie.bias(state) == [-1.0], "a fixed answer ends with its own prior"
    assert not AnswerTrie.from_index(retriever.index, tok, only={h.id for h in hits}).has_prior
