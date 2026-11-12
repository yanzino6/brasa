"""The training loop on a toy dataset: schedule, logging, checkpoints and exact resume. CPU only."""

from __future__ import annotations

import json

import pytest
import torch
from conftest import small_dataset

from brasa.config import DataConfig, PromptConfig
from brasa.train import TrainConfig, load_weights, lr_at, parse_args, pick_device, train


def config(tmp_path, name, **over) -> TrainConfig:
    base = dict(preset="brasa-4m", data_dir=str(tmp_path / "data"), out=str(tmp_path / name), steps=8, batch_tokens=512,
                lr=3e-3, warmup=2, dropout=0.0, vocab_size=320, block_size=96, device="cpu", compile="off", amp="off",
                log_every=2, eval_every=4, ckpt_every=4,
                data=DataConfig(variants=3, prompt=PromptConfig(max_question_tokens=32, answer_reserve=24)))
    return TrainConfig(**{**base, **over})


@pytest.fixture()
def data(tmp_path):
    return small_dataset(tmp_path / "data")


def test_learning_rate_schedule():
    cfg = TrainConfig(steps=100, warmup=10, min_lr_ratio=0.1)
    assert lr_at(0, cfg, 1.0) == pytest.approx(0.1)
    assert lr_at(9, cfg, 1.0) == pytest.approx(1.0)
    assert lr_at(10, cfg, 1.0) == pytest.approx(1.0)
    assert lr_at(55, cfg, 1.0) == pytest.approx(0.55), "half-way down the cosine"
    assert lr_at(100, cfg, 1.0) == pytest.approx(0.1)
    values = [lr_at(s, cfg, 1.0) for s in range(10, 101)]
    assert values == sorted(values, reverse=True)


def test_training_reduces_the_loss_and_writes_the_run(tmp_path, data):
    summary = train(config(tmp_path, "run", steps=60, log_every=10, eval_every=30, ckpt_every=30))
    out = tmp_path / "run"
    assert {p.name for p in out.iterdir()} >= {"tokenizer.json", "retrieval.json", "train_log.jsonl", "ckpt_last.pt",
                                               "ckpt_best.pt", "run.json"}
    log = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert log[0]["event"] == "start" and log[0]["device"] == "cpu"
    steps = [r for r in log if "loss" in r]
    assert [r["step"] for r in steps] == [10, 20, 30, 40, 50, 60]
    assert all({"loss", "lr", "tok_s", "elapsed_s"} <= set(r) for r in steps)
    assert steps[-1]["loss"] < 0.75 * steps[0]["loss"], "the training loss goes down"
    vals = [r for r in log if "val_loss" in r]
    assert vals[-1]["val_loss"] < vals[0]["val_loss"], "and so does the loss on held-out paraphrases"
    assert [r["step"] for r in vals] == [30, 60] and all(r["val_loss_open"] and r["val_loss_closed"] for r in vals)

    assert summary["steps"] == 60 and summary["model"]["params"] > 3_000_000
    assert summary["best_val_loss"] == min(r["val_loss"] for r in vals)
    assert json.loads((out / "run.json").read_text())["train_config"]["data"]["variants"] == 3

    model, ckpt = load_weights(out / "ckpt_last.pt")
    assert ckpt["step"] == 60 and not model.training and model.cfg.dropout == 0.0
    assert model.cfg.vocab_size == summary["tokenizer"]["vocab_size"] <= 320


def test_resume_reproduces_an_uninterrupted_run_exactly(tmp_path, data):
    train(config(tmp_path, "straight"))

    first = train(config(tmp_path, "split", stop_at=4))
    assert first["steps"] == 4 and first["planned_steps"] == 8
    assert torch.load(tmp_path / "split" / "ckpt_last.pt", weights_only=False)["step"] == 4
    second = train(config(tmp_path, "split", resume=True))
    assert second["steps"] == 8 and second["resumed_from_step"] == 4

    a, _ = load_weights(tmp_path / "straight" / "ckpt_last.pt")
    b, _ = load_weights(tmp_path / "split" / "ckpt_last.pt")
    for (name, p), q in zip(a.state_dict().items(), b.state_dict().values(), strict=True):
        assert torch.equal(p, q), f"{name} differs after resuming"

    log = [json.loads(line) for line in (tmp_path / "split" / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log if "loss" in r] == [2, 4, 6, 8], "the log is appended to, not restarted"
    straight = [json.loads(line) for line in (tmp_path / "straight" / "train_log.jsonl").read_text().splitlines()]
    assert [r["loss"] for r in log if "loss" in r] == [r["loss"] for r in straight if "loss" in r]


def test_context_too_short_for_the_prompt_format_is_rejected(tmp_path, data):
    with pytest.raises(ValueError, match="block_size"):
        train(config(tmp_path, "short", block_size=48))


def test_resume_without_a_checkpoint_starts_from_scratch(tmp_path, data):
    summary = train(config(tmp_path, "fresh", steps=2, resume=True))
    assert summary["resumed_from_step"] is None and summary["steps"] == 2


def test_command_line_defaults():
    cfg = parse_args(["--preset", "brasa-4m", "--steps", "10", "--prompt-loss-weight", "0.2", "--max-facts", "3"])
    assert (cfg.preset, cfg.steps, cfg.device, cfg.compile, cfg.amp) == ("brasa-4m", 10, "auto", "auto", "auto")
    assert cfg.data.prompt_loss_weight == 0.2 and cfg.data.prompt.max_facts == 3
    assert cfg.block_size == 384 and cfg.vocab_size == 4096 and not cfg.resume and cfg.stop_at is None
    assert pick_device("cpu").type == "cpu" and pick_device("auto").type in ("mps", "cuda", "cpu")
