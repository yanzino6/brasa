"""The dual-encoder retriever (brasa.encoder): pooling, the contrastive loss, batches, training, fusion."""

from __future__ import annotations

import json

import pytest
import torch
from conftest import small_dataset

from brasa.config import PAD, preset
from brasa.data import load_dataset
from brasa.encoder import (
    DenseRetriever,
    EncoderConfig,
    contrastive_loss,
    embed,
    evaluate,
    fact_ids,
    load_encoder,
    make_batch,
    parse_args,
    pool,
    question_ids,
    recall,
    same_topic,
    train,
    training_pairs,
)
from brasa.model import GPT
from brasa.retrieval import Bm25Retriever
from brasa.tokenizer import BPETokenizer


def enc_config(tmp_path, **over) -> EncoderConfig:
    base = dict(preset="brasa-4m", data_dir=str(tmp_path / "data"), out=str(tmp_path / "enc"), steps=40, batch=8,
                lr=2e-3, warmup=4, dropout=0.0, vocab_size=320, device="cpu", amp="off", log_every=10, eval_every=20)
    return EncoderConfig(**{**base, **over})


@pytest.fixture()
def data(tmp_path):
    return load_dataset(small_dataset(tmp_path / "data"))


def test_pooling_ignores_the_padding():
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]], [[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]])
    lengths = torch.tensor([2, 3])
    mean = pool(hidden, lengths, "mean")
    assert torch.allclose(mean[0], torch.tensor([0.5, 0.5]) / (0.5**0.5))
    assert torch.allclose(mean.norm(dim=-1), torch.ones(2))
    last = pool(hidden, lengths, "last")
    assert torch.allclose(last, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))


def test_hidden_states_feed_the_output_head_and_embedding_is_batch_independent():
    torch.manual_seed(0)
    model = GPT(preset("brasa-4m", vocab_size=300, block_size=32)).eval()
    x = torch.randint(0, 290, (2, 9))
    assert torch.allclose(model.lm_head(model.hidden(x)), model(x)[0], atol=1e-5)
    seqs = [[1, 5, 7], [2, 9, 4, 4, 8, 3, 11], [6, 6]]
    for pooling in ("mean", "last"):
        together = embed(model, seqs, 299, "cpu", pooling)
        alone = torch.stack([embed(model, [s], 299, "cpu", pooling)[0] for s in seqs])
        assert torch.allclose(together, alone, atol=1e-5), "right-padding never changes a vector"


def test_contrastive_loss_masks_the_other_right_answers():
    q = torch.eye(3)
    d = torch.cat([torch.eye(3), torch.eye(3)[:1]])  # column 3 repeats the gold document of row 0
    positive = torch.arange(3)
    none = torch.zeros(3, 4, dtype=torch.bool)
    masked = none.clone()
    masked[0, 3] = True
    assert contrastive_loss(q, d, positive, masked, 0.05) < contrastive_loss(q, d, positive, none, 0.05)
    assert contrastive_loss(q, d, positive, masked, 0.05) < 1e-6
    assert contrastive_loss(q, d.roll(1, 0), positive, none, 0.05) > 5


def test_batches_are_deterministic_and_carry_same_topic_negatives(tmp_path, data):
    tok = BPETokenizer.train([f.pt for f in data.facts.values()], 300)
    cfg = enc_config(tmp_path, batch=4, noise_prob=0.0)
    pairs, siblings = training_pairs(data), same_topic(data)
    assert len(pairs) == sum(len(r.facts) for r in data.split("train")) and "city" not in siblings["city"]
    questions, docs, positive, false_negative = make_batch(pairs, data, siblings, tok, cfg, step=3)
    again = make_batch(pairs, data, siblings, tok, cfg, step=3)
    assert questions == again[0] and docs == again[1]
    assert make_batch(pairs, data, siblings, tok, cfg, step=4)[0] != questions
    assert len(questions) == 4 and len(docs) == 8 and positive.tolist() == [0, 1, 2, 3]
    assert not false_negative.diagonal().any(), "the positive itself is never masked"
    texts = {tuple(fact_ids(tok, f.text(lang))): f.id for f in data.facts.values() for lang in ("pt", "en")}
    gold = [texts[tuple(d)] for d in docs[:4]]
    for i in range(4):
        assert texts[tuple(docs[4 + i])] != gold[i], "a hard negative is another fact"
        for j in range(8):
            assert bool(false_negative[i, j]) == (texts[tuple(docs[j])] == gold[i] and j != i)
    assert question_ids(tok, "Oi")[0] == fact_ids(tok, "Oi")[0] and question_ids(tok, "Oi")[1] != fact_ids(tok, "Oi")[1]


def test_mined_negatives_are_the_retrievers_best_wrong_facts(tmp_path, data):
    from brasa.encoder import mine_negatives

    lexical = Bm25Retriever.from_dataset(data, layout="fields")
    mined = mine_negatives(data, lexical, top=3)
    train_records = [r for r in data.split("train") if r.facts]
    assert set(mined) == {(r.q, r.lang) for r in train_records}
    for r in train_records:
        wrong = mined[(r.q, r.lang)]
        assert 1 <= len(wrong) <= 3 and not set(wrong) & set(r.facts), "never a gold fact"
        ranked = [h.id for h in lexical.search(r.q, 3 + len(r.facts)) if h.id not in r.facts]
        assert wrong == ranked[:3], "in the retriever's order"

    tok = BPETokenizer.train([r.q for r in data.qa] + [f.pt for f in data.facts.values()], 320)
    cfg = enc_config(tmp_path, batch=4, hard_negatives=1, mined_negatives=2)
    pairs, siblings = training_pairs(data), same_topic(data)
    plain = make_batch(pairs, data, siblings, tok, cfg, step=3)
    with_mined = make_batch(pairs, data, siblings, tok, cfg, step=3, mined=mined)
    assert with_mined[0] == plain[0] and with_mined[1][: len(plain[1])] == plain[1], "the same batch, more documents"
    assert len(plain[1]) == 4 + 4 and len(with_mined[1]) == 4 + 4 + 8
    mask = with_mined[3]
    assert mask.shape == (4, 16)
    assert not any(mask[i, 8 + 2 * i + j] for i in range(4) for j in range(2)), "never gold for their own row"
    assert make_batch(pairs, data, siblings, tok, cfg, step=3, mined=mined)[1] == with_mined[1], "deterministic"


def test_training_learns_to_retrieve_and_the_fusion_falls_back_to_bm25(tmp_path, data):
    summary = train(enc_config(tmp_path))
    out = tmp_path / "enc"
    assert {"encoder_best.pt", "encoder_last.pt", "tokenizer.json", "run.json", "train_log.jsonl"} <= \
        {p.name for p in out.iterdir()}
    log = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    losses = [r["loss"] for r in log if "loss" in r]
    assert losses[-1] < 0.5 * losses[0]
    assert summary["best_val"]["recall@4"] >= 0.9, "six facts, questions that differ by a prefix: easy"

    model, tok, run = load_encoder(out)
    lexical = Bm25Retriever.from_dataset(data, layout="fields")
    dense = DenseRetriever(model, tok, lexical.docs)
    hit = dense.search("Onde a Lia mora?", 2)
    assert hit[0].id == "city" and len({h.id for h in hit}) == 2 and hit[0].score <= 1.0 + 1e-5
    assert all(h.lang == "en" for h in dense.search("Where does Lia live?", 3, lang="en"))
    assert recall(data, dense, "test")["recall@4"] >= 0.9
    zero = DenseRetriever(model, tok, lexical.docs, lexical=lexical, weight=0.0, vectors=dense.vectors)
    for q in ("Onde a Lia mora?", "What is her e-mail?"):
        assert [h.id for h in zero.search(q, 4)] == [h.id for h in lexical.search(q, 4)], "weight 0 is BM25"

    report = evaluate(out, weights=(0.5, 1.0))
    assert set(report) >= {"lexical", "dense", "fused", "best_weight"} and report["best_weight"] in (0.5, 1.0)
    assert report["dense"]["test"]["n"] == len([r for r in data.split("test") if r.facts])


def test_an_encoder_can_start_from_a_pretrained_language_model(tmp_path, data):
    tok = BPETokenizer.train([f.pt for f in data.facts.values()] + [r.q for r in data.split("train")], 320)
    pre = tmp_path / "pre"
    pre.mkdir()
    tok.save(pre / "tokenizer.json")
    lm = GPT(preset("brasa-4m", vocab_size=tok.vocab_size, block_size=384))
    torch.save({"model": lm.state_dict(), "config": lm.describe()}, pre / "ckpt_last.pt")
    train(enc_config(tmp_path, steps=1, lr=0.0, eval_every=1, init_from=str(pre)))
    model, tok2, _ = load_encoder(tmp_path / "enc", checkpoint="last")
    assert tok2.merges == tok.merges
    assert all(torch.equal(v, lm.state_dict()[k]) for k, v in model.state_dict().items())
    assert tok.special_ids[PAD] == tok2.special_ids[PAD]


def test_command_line():
    run, cfg = parse_args(["--preset", "brasa-4m", "--pooling", "last", "--hard-negatives", "2", "--steps", "7"])
    assert run is None and cfg.pooling == "last" and cfg.hard_negatives == 2 and cfg.steps == 7
    assert parse_args(["--eval", "runs/x"])[0] == "runs/x"
    with pytest.raises(ValueError):
        train(EncoderConfig(pooling="max"))


def test_one_network_can_learn_to_answer_and_to_retrieve(tmp_path, data):
    """brasa.train --retrieval-weight: the language model's own hidden states rank the facts."""
    from brasa.config import DataConfig, PromptConfig
    from brasa.evaluate import load_run, retriever_for
    from brasa.train import TrainConfig
    from brasa.train import train as train_lm

    def run(name, **over):
        cfg = TrainConfig(preset="brasa-4m", data_dir=str(tmp_path / "data"), out=str(tmp_path / name), steps=40,
                          batch_tokens=512, lr=3e-3, warmup=4, dropout=0.0, vocab_size=320, block_size=96,
                          device="cpu", compile="off", amp="off", log_every=10, eval_every=20, ckpt_every=20,
                          index="fields", data=DataConfig(variants=3, prompt=PromptConfig(max_question_tokens=32,
                                                                                         answer_reserve=24)), **over)
        return train_lm(cfg)

    joint = run("joint", retrieval_weight=1.0, retrieval_batch=8)
    log = [json.loads(line) for line in (tmp_path / "joint" / "train_log.jsonl").read_text().splitlines()]
    contrastive = [r["retrieval_loss"] for r in log if "retrieval_loss" in r]
    assert contrastive[-1] < 0.5 * contrastive[0], "the contrastive loss falls"
    assert [r for r in log if "val_dense_recall@4" in r], "dense recall is logged with the validation loss"
    info = joint["retrieval"]
    assert info["pooling"] == "mean" and info["weight"] in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
    assert info["dense"]["val"]["recall@4"] >= 0.9 and set(info) >= {"lexical", "dense", "fused"}
    assert json.loads((tmp_path / "joint" / "run.json").read_text())["retrieval"]["weight"] == info["weight"]

    model, tok, lexical, run_json = load_run(tmp_path / "joint")
    fused, described = retriever_for(run_json, lexical, model, tok, "cpu")
    assert described == {"kind": "bm25+own", "weight": info["weight"]}
    dense = fused.index["dense"]  # the index the browser would get: the same documents plus their vectors
    assert fused.index["docs"] is lexical.index["docs"] and "dense" not in lexical.index
    assert (dense["weight"], dense["pooling"], dense["dim"]) == (info["weight"], "mean", model.cfg.d_model)
    assert fused.search("Onde a Lia mora?", 2)[0].id == "city"
    plain, described = retriever_for(run_json, lexical, model, tok, "cpu", which="lexical")
    assert plain is lexical and described == {"kind": "bm25"}

    alone = run("alone")
    assert "retrieval" not in alone and "retrieval_loss" not in (tmp_path / "alone" / "train_log.jsonl").read_text()
    model, tok, lexical, run_json = load_run(tmp_path / "alone")
    assert retriever_for(run_json, lexical, model, tok, "cpu")[0] is lexical, "a plain run answers with BM25"

    train(enc_config(tmp_path))  # a stand-alone encoder can be fused with any run's index
    other, described = retriever_for(run_json, lexical, model, tok, "cpu", encoder=tmp_path / "enc")
    assert described["kind"] == "bm25+encoder" and other.search("What is Lia's job?", 1)[0].id == "job"

    # --contexts-from: the training contexts are ranked by the fused retriever of another run
    for source, kind in (("joint", "bm25+own"), ("enc", "bm25+encoder")):
        second = run(f"second-{source}", contexts_from=str(tmp_path / source), retrieval_weight=1.0, retrieval_batch=8)
        assert second["contexts_from"]["kind"] == kind and second["contexts_from"]["run"] == str(tmp_path / source)
        assert second["examples"]["open_book"] == joint["examples"]["open_book"], "the same variants, other contexts"
    with pytest.raises(ValueError, match="no learned ranking"):
        run("third", contexts_from=str(tmp_path / "alone"))
    with pytest.raises(ValueError, match="choose one"):
        train_lm(TrainConfig(preset="brasa-4m", data_dir=str(tmp_path / "data"), out=str(tmp_path / "fourth"), steps=1,
                             vocab_size=320, device="cpu", compile="off", amp="off",
                             contexts_from=str(tmp_path / "joint"), data=DataConfig(context_folds=2)))
