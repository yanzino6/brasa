"""The training loop on a toy dataset: schedule, logging, checkpoints and exact resume. CPU only."""

from __future__ import annotations

import json

import pytest
import torch
from conftest import small_dataset

from brasa.config import DataConfig, PromptConfig
from brasa.data import build_examples, load_dataset
from brasa.evaluate import exact_match, generate_answers
from brasa.retrieval import Bm25Retriever
from brasa.tokenizer import BPETokenizer
from brasa.train import (
    TrainConfig,
    answer_loss,
    answer_metrics,
    finite_state,
    load_weights,
    lr_at,
    make_scaler,
    optimizer_step,
    parse_args,
    pick_device,
    require_finite,
    train,
)


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


def test_validation_exact_match_is_what_greedy_decoding_gets_right(tmp_path, data):
    """Teacher-forced exact match (every answer token is the argmax) equals generating and comparing."""
    cfg = config(tmp_path, "run", steps=120, log_every=40, eval_every=40, ckpt_every=120, index="fields")
    summary = train(cfg)
    model, _ = load_weights(tmp_path / "run" / "ckpt_last.pt")
    tok = BPETokenizer.load(tmp_path / "run" / "tokenizer.json")
    dataset = load_dataset(data)
    retriever = Bm25Retriever.load(tmp_path / "run" / "retrieval.json")
    assert set(retriever.index["params"]["fields"]) == {"fact", "topic", "questions"}, "--index fields reached the run"
    examples, _ = build_examples(dataset, "val", tok, retriever, DataConfig(variants=2, prompt=cfg.data.prompt), 96)
    loss, em = answer_metrics(model, examples, tok.special_ids["<|pad|>"], "cpu")
    assert loss == pytest.approx(answer_loss(model, examples, tok.special_ids["<|pad|>"], "cpu"))
    records = [r for r in dataset.split("val") for _ in range(2)]
    preds = generate_answers(model, tok, [e.ids[: e.answer_start] for e in examples], 24, "cpu")
    generated = sum(exact_match(p, r.a) for p, r in zip(preds, records, strict=True)) / len(records)
    assert 0.0 < em < 1.0, "the toy run answers some held-out questions and not others"
    assert em == pytest.approx(generated)
    assert summary["final_val"]["val_em"] == pytest.approx(em, abs=1e-4)
    log = [json.loads(line) for line in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    assert all({"val_em", "val_em_open", "val_em_closed"} <= set(r) for r in log if "val_loss" in r)


def test_best_checkpoint_follows_the_selected_criterion(tmp_path, data):
    by_em = train(config(tmp_path, "em", steps=120, eval_every=20, ckpt_every=120))
    by_loss = train(config(tmp_path, "loss", steps=120, eval_every=20, ckpt_every=120, select="loss"))
    assert by_em["select"] == "em" and by_loss["select"] == "loss"
    history = [json.loads(line) for line in (tmp_path / "em" / "train_log.jsonl").read_text().splitlines()]
    history = [r for r in history if "val_em" in r]
    best_em = torch.load(tmp_path / "em" / "ckpt_best.pt", weights_only=False)
    best_loss = torch.load(tmp_path / "loss" / "ckpt_best.pt", weights_only=False)
    top = max(r["val_em"] for r in history)
    assert best_em["val_em"] == pytest.approx(top, abs=1e-4) and by_em["best_val_em"] == pytest.approx(top, abs=1e-4)
    assert best_em["step"] == min(r["step"] for r in history if r["val_em"] == top or abs(r["val_em"] - top) < 5e-5) \
        or best_em["step"] in {r["step"] for r in history if abs(r["val_em"] - top) < 5e-5}
    assert best_loss["best_val"] == pytest.approx(by_loss["best_val_loss"], abs=1e-4)
    with pytest.raises(ValueError):
        train(config(tmp_path, "bad", select="accuracy"))


def test_label_smoothing_and_new_flags_reach_the_run(tmp_path, data):
    plain = train(config(tmp_path, "plain", steps=4, eval_every=4))
    smooth = train(config(tmp_path, "smooth", steps=4, eval_every=4, label_smoothing=0.2))
    first = lambda name: json.loads((tmp_path / name / "train_log.jsonl").read_text().splitlines()[1])["loss"]  # noqa: E731
    assert first("smooth") != first("plain") and smooth["train_config"]["label_smoothing"] == 0.2
    assert plain["train_config"]["label_smoothing"] == 0.0
    cfg = parse_args(["--context-folds", "4", "--importance-power", "1.5", "--label-smoothing", "0.1", "--index",
                      "fields", "--select", "loss"])
    assert (cfg.data.context_folds, cfg.data.importance_power) == (4, 1.5)
    assert (cfg.label_smoothing, cfg.index, cfg.select) == (0.1, "fields", "loss")
    ranked = parse_args(["--oracle-prob", "0", "--drop-gold-prob", "0", "--shuffle-prob", "0",
                         "--random-distractor-prob", "0"]).data
    assert (ranked.oracle_prob, ranked.drop_gold_prob, ranked.shuffle_prob, ranked.random_distractor_prob) == (0,) * 4
    assert cfg.data.shuffle_prob == DataConfig().shuffle_prob
    folded = train(config(tmp_path, "folds", steps=2, eval_every=2,
                          data=DataConfig(variants=3, context_folds=2,
                                          prompt=PromptConfig(max_question_tokens=32, answer_reserve=24))))
    assert folded["examples"]["open_book_gold_in_context"] <= plain["examples"]["open_book_gold_in_context"]


def test_loss_scaling_is_on_for_float16_only_and_the_step_matches_a_plain_update():
    cpu = torch.device("cpu")
    assert not make_scaler(cpu, "off").is_enabled()
    assert not make_scaler(cpu, "bf16").is_enabled()
    if torch.backends.mps.is_available():
        assert make_scaler(torch.device("mps"), "fp16").is_enabled()

    def one_step(use_helper: bool) -> torch.Tensor:
        torch.manual_seed(0)
        layer = torch.nn.Linear(4, 3)
        opt = torch.optim.SGD(layer.parameters(), lr=0.1)
        loss = layer(torch.ones(2, 4)).pow(2).mean() * 50  # gradient norm well above the clip
        if use_helper:
            optimizer_step(layer, opt, make_scaler(cpu, "off"), loss, grad_clip=1.0)
        else:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
            opt.step()
        return layer.weight.detach().clone()

    assert torch.equal(one_step(True), one_step(False))


def test_a_diverged_run_stops_and_cannot_seed_another(tmp_path, data):
    require_finite(2.5, "the loss", 10)
    with pytest.raises(FloatingPointError, match="step 10"):
        require_finite(float("nan"), "the loss", 10)
    with pytest.raises(FloatingPointError):
        require_finite(float("inf"), "the loss", 10)
    good = {"w": torch.ones(2, 2), "steps": torch.tensor(3)}
    assert finite_state(good, "x") is good
    with pytest.raises(ValueError, match="not finite"):
        finite_state({"w": torch.tensor([1.0, float("nan")])}, "x")

    train(config(tmp_path, "pre", steps=4))
    ckpt = torch.load(tmp_path / "pre" / "ckpt_last.pt", weights_only=False)
    assert "scaler" in ckpt, "the loss-scaling state travels with the checkpoint"
    ckpt["model"] = {k: torch.full_like(v, float("nan")) if v.is_floating_point() else v
                     for k, v in ckpt["model"].items()}
    torch.save(ckpt, tmp_path / "pre" / "ckpt_last.pt")
    with pytest.raises(ValueError, match="not finite"):
        train(config(tmp_path, "tuned", steps=2, init_from=str(tmp_path / "pre")))
