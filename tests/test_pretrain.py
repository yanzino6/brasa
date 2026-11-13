"""Pretraining on plain text and fine-tuning from it (brasa.pretrain, brasa.train --init-from). CPU only."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from conftest import small_dataset
from test_train import config

from brasa.config import BOS, EOS
from brasa.pretrain import (
    PretrainConfig,
    encode_files,
    parse_args,
    prepare_corpus,
    pretrain,
    read_head,
    window_batch,
)
from brasa.tokenizer import BPETokenizer
from brasa.train import train

SENTENCES = ["O rio corre para o mar e o mar devolve a chuva.",
             "The river runs to the sea and the sea returns the rain.",
             "A cidade acorda cedo e dorme tarde.", "The city wakes early and sleeps late."]


@pytest.fixture()
def text(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("".join(f"{SENTENCES[i % 4]} {SENTENCES[(i + 1) % 4]}\n" for i in range(120)), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"source": "test", "licence": "none"}), encoding="utf-8")
    return path


def pre_config(tmp_path, text, **over) -> PretrainConfig:
    base = dict(preset="brasa-4m", text=[str(text)], data_dir=str(small_dataset(tmp_path / "data")),
                out=str(tmp_path / "pre"), steps=12, batch_tokens=256, lr=3e-3, warmup=2, vocab_size=320, block_size=32,
                device="cpu", compile="off", amp="off", log_every=4, eval_every=6, ckpt_every=6, workers=1)
    return PretrainConfig(**{**base, **over})


def test_the_corpus_is_every_line_between_bos_and_eos_and_is_cached(tmp_path, text):
    cfg = pre_config(tmp_path, text)
    tok, ids, meta = prepare_corpus(cfg, tmp_path / "corpus")
    bos, eos = tok.special_ids[BOS], tok.special_ids[EOS]
    assert int((np.asarray(ids) == bos).sum()) == 120 == int((np.asarray(ids) == eos).sum())
    first = list(ids[: list(ids).index(eos) + 1])
    assert tok.decode(first) == f"{SENTENCES[0]} {SENTENCES[1]}" and first[0] == bos
    assert meta["tokens"] == len(ids) and meta["manifests"], "the provenance of the text travels with the corpus"
    # the tokenizer saw the text AND the knowledge base with its training questions
    assert tok.merges != BPETokenizer.train(read_head(text, 10**6), 320).merges
    assert len(tok.encode("river")) < len("river")
    tok2, ids2, meta2 = prepare_corpus(PretrainConfig(text=[]), tmp_path / "corpus")  # second call: from the cache
    assert tok2.merges == tok.merges and np.array_equal(ids, ids2) and meta2 == meta
    assert np.array_equal(encode_files([str(text)], tmp_path / "corpus" / "tokenizer.json", workers=2), ids)


def test_window_batches_are_a_pure_function_of_seed_and_step():
    ids = np.arange(1000, dtype=np.uint16)
    x, y = window_batch(ids, 900, 16, 4, seed=0, step=3)
    assert x.shape == (4, 16) and np.array_equal(y, x + 1) and x.max() < 900
    x2, _ = window_batch(ids, 900, 16, 4, seed=0, step=3)
    x3, _ = window_batch(ids, 900, 16, 4, seed=0, step=4)
    assert np.array_equal(x, x2) and not np.array_equal(x, x3)


def test_pretraining_learns_the_text_and_resumes(tmp_path, text):
    summary = pretrain(pre_config(tmp_path, text, steps=40))
    log = [json.loads(line) for line in (tmp_path / "pre" / "train_log.jsonl").read_text().splitlines()]
    losses = [r["loss"] for r in log if "loss" in r]
    assert losses[-1] < 0.6 * losses[0]
    assert summary["final_val"]["val_loss"] < 4.0 and summary["objective"].startswith("next-token")
    assert {"tokenizer.json", "ckpt_last.pt", "run.json"} <= {p.name for p in (tmp_path / "pre").iterdir()}

    shared = dict(corpus=str(tmp_path / "pre" / "corpus"), steps=12)
    whole = pretrain(pre_config(tmp_path, text, out=str(tmp_path / "a"), **shared))
    first = pretrain(pre_config(tmp_path, text, out=str(tmp_path / "b"), stop_at=6, **shared))
    resumed = pretrain(pre_config(tmp_path, text, out=str(tmp_path / "b"), resume=True, **shared))
    assert first["steps"] == 6 and resumed["resumed_from_step"] == 6 and resumed["steps"] == 12
    assert whole["final_val"] == resumed["final_val"], "CPU, no dropout: the split run is the same run"


def test_fine_tuning_starts_from_the_pretrained_weights_and_tokenizer(tmp_path, text):
    pretrain(pre_config(tmp_path, text))
    pre = torch.load(tmp_path / "pre" / "ckpt_last.pt", weights_only=False)["model"]
    with pytest.raises(RuntimeError):  # another model size: the weights do not fit
        train(config(tmp_path, "bad", steps=1, init_from=str(tmp_path / "pre"), preset="brasa-12m"))
    summary = train(config(tmp_path, "ft", steps=1, lr=0.0, init_from=str(tmp_path / "pre")))
    assert summary["init_from"]["dir"] == str(tmp_path / "pre") and summary["init_from"]["steps"] == 12
    assert BPETokenizer.load(tmp_path / "ft" / "tokenizer.json").merges == \
        BPETokenizer.load(tmp_path / "pre" / "tokenizer.json").merges
    after = torch.load(tmp_path / "ft" / "ckpt_last.pt", weights_only=False)["model"]
    assert all(torch.equal(pre[k], after[k]) for k in pre), "learning rate 0: exactly the pretrained weights"
    tuned = train(config(tmp_path, "ft2", steps=30, init_from=str(tmp_path / "pre")))
    assert tuned["final_val"]["val_loss"] < 6.0
    # the control of the experiment: the same vocabulary, random weights
    control = train(config(tmp_path, "tok", steps=1, tokenizer=str(tmp_path / "pre" / "tokenizer.json")))
    assert control["init_from"] is None and control["tokenizer"]["merges"] == summary["tokenizer"]["merges"]
    fresh = torch.load(tmp_path / "tok" / "ckpt_last.pt", weights_only=False)["model"]
    assert not torch.equal(fresh["tok_emb.weight"], pre["tok_emb.weight"])


def test_command_line():
    cfg = parse_args(["--preset", "brasa-4m", "--text", "a.txt", "b.txt", "--steps", "5", "--prepare-only"])
    assert cfg.text == ["a.txt", "b.txt"] and cfg.steps == 5 and cfg.prepare_only and cfg.dropout == 0.0
