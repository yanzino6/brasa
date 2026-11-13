"""Train Brasa on a data directory.

    uv run python -m brasa.train --preset brasa-12m --data-dir data --out runs/brasa-12m --steps 30000
    uv run python -m brasa.train --preset brasa-12m --data-dir data --out runs/brasa-12m --steps 30000 --resume

--resume continues from <out>/ckpt_last.pt (written every --ckpt-every steps, on Ctrl-C and at the end); a
run can also be split on purpose with --stop-at N.

What it does, in order: validate the data, train the BPE tokenizer, build the retrieval index, turn every
QA record into open-book and closed-book examples, then optimise with AdamW (cosine schedule with warm-up,
gradient clipping). The loss covers the answer span only unless --prompt-loss-weight says otherwise.

Files written to --out:
    tokenizer.json, retrieval.json   needed again by evaluate.py, export.py and sample.py
    train_log.jsonl                  one JSON object per log line: step, loss, lr, tok_s, val loss and exact match
    ckpt_last.pt                     everything needed to continue (--resume)
    ckpt_best.pt                     weights with the best validation score so far (--select: exact match by
                                     default; the validation LOSS bottoms out long before the answers stop
                                     improving, so it picks a worse model)
    run.json                         configuration, data statistics and timing of the run

Reproducibility: batch(step) is a pure function of (seed, step) and the optimiser state is checkpointed,
so with dropout 0 a resumed run reproduces an uninterrupted one exactly (tests/test_train.py). With
dropout the masks are re-seeded at the resume point.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_BLOCK_SIZE, DEFAULT_VOCAB_SIZE, PAD, PRESETS, DataConfig, GPTConfig, PromptConfig, preset
from .data import BatchSampler, Example, build_examples, fold_retrievers, load_dataset, tokenizer_texts
from .model import GPT
from .retrieval import LAYOUTS, Bm25Retriever
from .tokenizer import BPETokenizer

DEFAULT_LR = {"brasa-4m": 1e-3, "brasa-12m": 6e-4, "brasa-27m": 4e-4}


@dataclass
class TrainConfig:
    preset: str = "brasa-12m"
    data_dir: str = "data"
    out: str = "runs/brasa"
    steps: int = 30000
    batch_tokens: int = 64 * DEFAULT_BLOCK_SIZE  # positions per step (rows x width of the chosen bucket)
    lr: float | None = None  # default depends on the preset (DEFAULT_LR)
    min_lr_ratio: float = 0.1
    warmup: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dropout: float = 0.1
    label_smoothing: float = 0.0
    index: str = "flat"  # layout of the BM25 index: "flat" or "fields" (retrieval.py)
    select: str = "em"  # what ckpt_best.pt maximises: "em" (validation exact match) or "loss" (lowest loss)
    # a run directory of brasa.pretrain (or of this module): start from its weights and USE ITS TOKENIZER
    init_from: str | None = None
    tokenizer: str | None = None  # a tokenizer.json to use instead of training one on the data directory
    # The same network as a retriever (encoder.py): weight of a contrastive (question, fact) loss added to
    # every step (0 = off), the questions per contrastive batch, its temperature and how hidden states are pooled.
    retrieval_weight: float = 0.0
    retrieval_batch: int = 128
    retrieval_temperature: float = 0.05
    retrieval_pooling: str = "mean"
    retrieval_mined_negatives: int = 0  # per question, facts mined for it with the BM25 index (encoder.mine_negatives)
    # Training (and validation) contexts ranked by the fused retriever of ANOTHER run: a brasa.train run with
    # --retrieval-weight or a brasa.encoder run. A model that will be asked with facts ranked by BM25 + its own
    # hidden states should not be trained on BM25 contexts only: the facts that come with an out-of-scope
    # question look different (they are about the right subject), and it learns to decline on the wrong cue.
    contexts_from: str | None = None
    vocab_size: int = DEFAULT_VOCAB_SIZE
    block_size: int = DEFAULT_BLOCK_SIZE
    seed: int = 0
    device: str = "auto"  # auto | mps | cuda | cpu
    compile: str = "auto"  # auto | on | off      (auto: on for mps and cuda)
    amp: str = "auto"  # auto | off | fp16 | bf16  (auto: fp16 on mps and cuda)
    log_every: int = 50
    eval_every: int = 250
    ckpt_every: int = 500
    resume: bool = False
    stop_at: int | None = None  # stop (and checkpoint) at this step without changing the schedule of `steps`
    data: DataConfig = field(default_factory=DataConfig)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_at(step: int, cfg: TrainConfig, base_lr: float) -> float:
    """Linear warm-up, then a cosine from base_lr down to base_lr * min_lr_ratio at the last step."""
    if step < cfg.warmup:
        return base_lr * (step + 1) / cfg.warmup
    t = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    return base_lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def make_optimizer(model: GPT, cfg: TrainConfig, lr: float) -> torch.optim.AdamW:
    """AdamW with weight decay on the matrices only (not on the RMSNorm gains)."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(cfg.beta1, cfg.beta2))


@torch.no_grad()
def answer_metrics(model: GPT, examples: list[Example], pad_id: int, device: torch.device,
                   max_tokens: int = 32768) -> tuple[float, float]:
    """(mean cross-entropy per answer token, exact match) over a list of examples, teacher forced.

    Exact match is the share of examples whose every answer token, <|eos|> included, is the model's most
    likely token given the gold prefix. That is exactly the share greedy decoding gets right (it follows
    the gold answer for as long as each argmax is the gold token), at the price of one forward pass."""
    was_training = model.training
    model.eval()
    order = sorted(range(len(examples)), key=lambda i: len(examples[i].ids))
    total, count, exact, i = 0.0, 0, 0, 0
    while i < len(order):
        width = len(examples[order[i]].ids)
        rows = [order[i]]
        while i + len(rows) < len(order):
            nxt = len(examples[order[i + len(rows)]].ids)
            if (len(rows) + 1) * nxt > max_tokens:
                break
            rows.append(order[i + len(rows)])
            width = nxt
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        targets = torch.full((len(rows), width - 1), -100, dtype=torch.long)
        for r, j in enumerate(rows):
            ex = examples[j]
            ids[r, : len(ex.ids)] = torch.tensor(ex.ids)
            targets[r, ex.answer_start - 1 : len(ex.ids) - 1] = torch.tensor(ex.ids[ex.answer_start :])
        n = int((targets != -100).sum())
        targets = targets.to(device)
        logits, loss, _ = model(ids[:, :-1].to(device), targets)
        total += float(loss) * n
        count += n
        exact += int(((logits.argmax(-1) == targets) | (targets == -100)).all(dim=1).sum())
        i += len(rows)
    model.train(was_training)
    return total / max(1, count), exact / max(1, len(examples))


def answer_loss(model: GPT, examples: list[Example], pad_id: int, device: torch.device,
                max_tokens: int = 32768) -> float:
    """Mean cross-entropy per answer token (teacher forced), in float32, over a list of examples."""
    return answer_metrics(model, examples, pad_id, device, max_tokens)[0]


def make_scaler(device: torch.device, amp: str) -> torch.amp.GradScaler:
    """Loss scaling for float16 autocast (a pass-through for float32 and bfloat16).

    Under float16 the backward pass carries d loss / d logits = (softmax - one_hot) / n_tokens in half precision,
    whose smallest positive value is 6e-8. With thousands of tokens in a batch most of those terms are smaller
    and are flushed to zero: the model then trains on a fraction of its gradient, or not at all (plain text,
    where every token counts: the loss stays at ln(vocab) and ends in NaN). The scaler multiplies the loss by a
    large power of two before the backward pass and divides the float32 gradients back before the update."""
    return torch.amp.GradScaler(device.type, enabled=amp == "fp16")


def optimizer_step(model: GPT, opt: torch.optim.Optimizer, scaler: torch.amp.GradScaler, loss: torch.Tensor,
                   grad_clip: float) -> None:
    """One update: scaled backward pass, gradients unscaled and clipped, step (skipped if they overflowed)."""
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(opt)
    scaler.update()


def require_finite(value: float, what: str, step: int) -> None:
    """Stop a run whose loss is NaN or infinite instead of training on for hours."""
    if not math.isfinite(value):
        raise FloatingPointError(f"{what} is {value} at step {step}: the run has diverged; stopping")


def finite_state(state: dict, where: str | Path) -> dict:
    """The state dict, after checking that every weight is finite (a diverged run must not seed another)."""
    for name, tensor in state.items():
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{where}: weights are not finite ({name}); the run that wrote it diverged")
    return state


def save_checkpoint(path: Path, **payload) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: an interrupted save never corrupts the previous checkpoint


def load_weights(path: str | Path, device: str | torch.device = "cpu") -> tuple[GPT, dict]:
    """Load a checkpoint written by this module into a model in eval mode (dropout off)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in ckpt["config"].items() if k != "params"}
    model = GPT(GPTConfig(**{**cfg, "dropout": 0.0}))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def init_info(run_dir: str | Path) -> dict:
    """What a run started from (--init-from): the checkpoint and, for a brasa.pretrain run, its provenance."""
    run_dir = Path(run_dir)
    out: dict = {"dir": str(run_dir), "checkpoint": str(run_dir / "ckpt_last.pt")}
    run_json = run_dir / "run.json"
    if run_json.exists():
        run = json.loads(run_json.read_text(encoding="utf-8"))
        keys = ("objective", "steps", "tokens_seen", "epochs_equivalent", "train_seconds", "final_val", "corpus")
        out.update({k: run[k] for k in keys if k in run})
    return out


def context_retriever(source: str | Path, lexical: Bm25Retriever, device) -> tuple[object, dict]:
    """(retriever, description) for --contexts-from: the learned ranking of another run, fused with `lexical`
    (the BM25 index of THIS run, so the documents are the ones this run is trained on)."""
    from .evaluate import load_run, retriever_for

    source = Path(source)
    if (source / "encoder_best.pt").exists():  # a stand-alone dual encoder (brasa.encoder)
        retriever, info = retriever_for({}, lexical, None, None, device, encoder=source)
        return retriever, {**info, "run": str(source)}
    model, tok, _, run = load_run(source, "last", device)
    retriever, info = retriever_for(run, lexical, model, tok, device)
    if info["kind"] == "bm25":
        raise ValueError(f"{source} has no learned ranking: give a run trained with --retrieval-weight, "
                         "or a brasa.encoder run")
    return retriever, {**info, "run": str(source)}


def train(cfg: TrainConfig) -> dict:
    prompt = cfg.data.prompt
    if cfg.block_size < prompt.max_question_tokens + prompt.answer_reserve + 3:
        raise ValueError(f"block_size {cfg.block_size} cannot hold a question of {prompt.max_question_tokens} tokens, "
                         f"an answer of {prompt.answer_reserve} and the 3 control tokens: no fact would ever fit")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    base_lr = cfg.lr if cfg.lr is not None else DEFAULT_LR.get(cfg.preset, 6e-4)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ---- data, tokenizer, retriever
    t0 = time.time()
    dataset = load_dataset(cfg.data_dir)
    last = out / "ckpt_last.pt"
    resuming = cfg.resume and last.exists()
    if resuming and (out / "tokenizer.json").exists():
        tok = BPETokenizer.load(out / "tokenizer.json")
    elif cfg.init_from or cfg.tokenizer:  # pretrained weights only make sense with their own vocabulary
        tok = BPETokenizer.load(cfg.tokenizer or Path(cfg.init_from) / "tokenizer.json")
        tok.save(out / "tokenizer.json")
    else:
        tok = BPETokenizer.train(tokenizer_texts(dataset), cfg.vocab_size)
        tok.save(out / "tokenizer.json")
    retriever = Bm25Retriever.from_dataset(dataset, layout=cfg.index)
    retriever.save(out / "retrieval.json")
    folds = None
    if cfg.data.context_folds > 1:  # training contexts from indices that have not seen the question
        folds = fold_retrievers(dataset, cfg.data.context_folds,
                                lambda skip: Bm25Retriever.from_dataset(dataset, skip=skip, layout=cfg.index))
    contexts, contexts_info = retriever, None
    if cfg.contexts_from:
        if folds:
            raise ValueError("--contexts-from and --context-folds are two ways of building the contexts: choose one")
        contexts, contexts_info = context_retriever(cfg.contexts_from, retriever, device)
    examples, data_stats = build_examples(dataset, "train", tok, contexts, cfg.data, cfg.block_size, cfg.seed, folds)
    val_cfg = DataConfig(variants=2, prompt=cfg.data.prompt)  # clean closed-book + clean open-book
    val_examples, _ = build_examples(dataset, "val", tok, contexts, val_cfg, cfg.block_size, cfg.seed)
    del contexts  # the other run's network is not needed after this point
    val_open = [e for e in val_examples if e.open_book]
    val_closed = [e for e in val_examples if not e.open_book]
    pad_id = tok.special_ids[PAD]
    sampler = BatchSampler(examples, cfg.block_size, cfg.batch_tokens, pad_id, cfg.seed, cfg.data.prompt_loss_weight)
    dense = None
    if cfg.retrieval_weight > 0:  # imported here: encoder.py builds on this module
        from . import encoder as enc
        dense = enc.EncoderConfig(batch=cfg.retrieval_batch, seed=cfg.seed, temperature=cfg.retrieval_temperature,
                                  pooling=cfg.retrieval_pooling, mined_negatives=cfg.retrieval_mined_negatives)
        if dense.pooling not in enc.POOLINGS:
            raise ValueError(f"unknown --retrieval-pooling {dense.pooling!r}; choose from {enc.POOLINGS}")
        pairs, siblings = enc.training_pairs(dataset), enc.same_topic(dataset)
        mined = enc.mine_negatives(dataset, retriever, dense.mined_top) if dense.mined_negatives > 0 else None
    prep_seconds = time.time() - t0

    # ---- model, optimiser
    model = GPT(preset(cfg.preset, vocab_size=tok.vocab_size, block_size=cfg.block_size, dropout=cfg.dropout))
    model.to(device)
    opt = make_optimizer(model, cfg, base_lr)
    if cfg.select not in ("em", "loss"):
        raise ValueError(f"unknown --select {cfg.select!r}; choose em or loss")
    start_step, best_val, elapsed, tokens_seen = 0, float("inf"), 0.0, 0
    best_score = (-1.0, float("-inf"))  # (validation exact match, -loss); "loss" only compares the second
    init = None
    if cfg.init_from and not resuming:
        init = init_info(cfg.init_from)
        state = torch.load(init["checkpoint"], map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(finite_state(state, init["checkpoint"]))
    if resuming:
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"], ckpt["best_val"]
        best_score = tuple(ckpt.get("best_score", best_score))
        elapsed, tokens_seen = ckpt["elapsed"], ckpt["tokens_seen"]
        torch.manual_seed(cfg.seed + start_step)

    use_compile = cfg.compile == "on" or (cfg.compile == "auto" and device.type in ("mps", "cuda"))
    amp = cfg.amp if cfg.amp != "auto" else ("fp16" if device.type in ("mps", "cuda") else "off")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    forward = torch.compile(model) if use_compile else model  # evaluation always uses the plain module
    scaler = make_scaler(device, amp)
    if resuming and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])

    info = {
        "event": "start", "preset": cfg.preset, "model": model.describe(), "device": str(device),
        "compile": use_compile, "amp": amp, "lr": base_lr, "resumed_from_step": start_step if resuming else None,
        "init_from": init if init is not None else (cfg.init_from and init_info(cfg.init_from)),
        "contexts_from": contexts_info,
        "tokenizer": {"vocab_size": tok.vocab_size, "merges": len(tok.merges), "requested": cfg.vocab_size},
        "data": {"dir": cfg.data_dir, **dataset.stats()}, "examples": data_stats,
        "buckets": [{"width": b["width"], "examples": len(b["ids"]), "rows": b["rows"]} for b in sampler.buckets],
        "prep_seconds": round(prep_seconds, 1), "torch": torch.__version__,
    }
    log_path = out / "train_log.jsonl"
    log_file = open(log_path, "a" if resuming else "w", encoding="utf-8")  # noqa: SIM115

    def log(record: dict) -> None:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()
        print(json.dumps(record, ensure_ascii=False), flush=True)

    log(info)

    def checkpoint(step: int) -> None:
        save_checkpoint(last, model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                        step=step, best_val=best_val,
                        best_score=best_score, elapsed=elapsed, tokens_seen=tokens_seen, config=model.describe(),
                        train_config=asdict(cfg))

    def sync() -> None:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    # ---- loop
    model.train()
    window_loss, window_tokens, window_steps, window_retrieval = 0.0, 0, 0, 0.0
    t_window = time.time()
    val_history: list[dict] = []
    step = start_step
    end = min(cfg.steps, cfg.stop_at) if cfg.stop_at else cfg.steps
    try:
        for step in range(start_step, end):
            lr = lr_at(step, cfg, base_lr)
            for group in opt.param_groups:
                group["lr"] = lr
            x, y, w = sampler.batch(step)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            w = None if w is None else torch.from_numpy(w).to(device)
            contrast = enc.make_batch(pairs, dataset, siblings, tok, dense, step, mined) if dense else None

            def losses(x=x, y=y, w=w, contrast=contrast):
                """(what is optimised, the language-model loss, the contrastive loss or None)."""
                _, lm, _ = forward(x, y, loss_weights=w, label_smoothing=cfg.label_smoothing)
                if contrast is None:
                    return lm, lm, None
                questions, documents, positive, false_negative = contrast
                r = enc.contrastive_loss(enc.embed(model, questions, pad_id, device, dense.pooling),
                                         enc.embed(model, documents, pad_id, device, dense.pooling),
                                         positive.to(device), false_negative.to(device), dense.temperature)
                return lm + cfg.retrieval_weight * r, lm, r

            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    total, loss, r_loss = losses()
            else:
                total, loss, r_loss = losses()
            optimizer_step(model, opt, scaler, total, cfg.grad_clip)
            if r_loss is not None:
                window_retrieval += float(r_loss.detach())
            window_loss += float(loss.detach())
            window_tokens += x.numel()
            window_steps += 1
            tokens_seen += x.numel()

            done = step + 1
            if done % cfg.log_every == 0 or done == end:
                sync()
                dt = time.time() - t_window
                elapsed += dt
                require_finite(window_loss, "the training loss", done)
                rec = {"step": done, "loss": round(window_loss / window_steps, 4), "lr": round(lr, 7),
                       "tok_s": round(window_tokens / dt), "elapsed_s": round(elapsed, 1)}
                if dense:
                    require_finite(window_retrieval, "the contrastive loss", done)
                    rec["retrieval_loss"] = round(window_retrieval / window_steps, 4)
                log(rec)
                window_loss, window_tokens, window_steps, window_retrieval = 0.0, 0, 0, 0.0
                t_window = time.time()
            if done % cfg.eval_every == 0 or done == end:
                v_open, em_open = answer_metrics(model, val_open, pad_id, device) if val_open else (None, None)
                v_closed, em_closed = answer_metrics(model, val_closed, pad_id, device) if val_closed else (None, None)
                parts = [v for v in (v_open, v_closed) if v is not None]
                ems = [v for v in (em_open, em_closed) if v is not None]
                val = sum(parts) / len(parts) if parts else float("nan")
                val_em = sum(ems) / len(ems) if ems else 0.0
                rec = {"step": done, "val_loss": round(val, 4), "val_loss_open": v_open and round(v_open, 4),
                       "val_loss_closed": v_closed and round(v_closed, 4), "val_em": round(val_em, 4),
                       "val_em_open": em_open and round(em_open, 4), "val_em_closed": em_closed and round(em_closed, 4)}
                if dense:  # the network read as a retriever, alone: gold facts among its first hits
                    found = enc.recall(dataset, enc.DenseRetriever(model, tok, retriever.docs, dense.pooling,
                                                                   device=device), "val", ks=(1, 4))
                    model.train()
                    rec.update({"val_dense_recall@1": found["recall@1"], "val_dense_recall@4": found["recall@4"]})
                val_history.append(rec)
                log(rec)
                best_val = min(best_val, val)
                score = (val_em if cfg.select == "em" else 0.0, -val)
                if parts and score > best_score:
                    best_score = score
                    save_checkpoint(out / "ckpt_best.pt", model=model.state_dict(), step=done, best_val=val,
                                    val_em=val_em, config=model.describe(), train_config=asdict(cfg))
                t_window = time.time()
            if done % cfg.ckpt_every == 0 or done == end:
                checkpoint(done)
                t_window = time.time()
    except KeyboardInterrupt:
        checkpoint(step)
        log({"event": "interrupted", "step": step})
        raise
    finally:
        log_file.close()

    summary = {
        **{k: v for k, v in info.items() if k != "event"},
        "train_config": asdict(cfg), "steps": max(start_step, end), "planned_steps": cfg.steps,
        "train_seconds": round(elapsed, 1),
        "tokens_seen": tokens_seen, "tokens_per_second": round(tokens_seen / max(elapsed, 1e-9)),
        "epochs_equivalent": round(tokens_seen / max(1, data_stats["tokens"]), 1),
        "best_val_loss": None if math.isinf(best_val) else round(best_val, 4),
        "best_val_em": None if best_score[0] < 0 else round(best_score[0], 4), "select": cfg.select,
        "final_val": val_history[-1] if val_history else None,
    }
    if dense:  # how much the learned ranking weighs next to BM25: chosen on the validation questions
        summary["retrieval"] = {"pooling": dense.pooling, "temperature": dense.temperature,
                                **enc.fusion_report(model, tok, dataset, retriever, dense.pooling, device)}
    (out / "run.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args(argv=None) -> TrainConfig:
    d, dd, pd = TrainConfig(), DataConfig(), PromptConfig()
    ap = argparse.ArgumentParser(description="Train Brasa.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--preset", default=d.preset, choices=sorted(PRESETS))
    ap.add_argument("--data-dir", default=d.data_dir, help="directory with kb/ and qa/")
    ap.add_argument("--out", default=d.out, help="run directory")
    ap.add_argument("--steps", type=int, default=d.steps)
    ap.add_argument("--batch-tokens", type=int, default=d.batch_tokens, help="positions per optimisation step")
    ap.add_argument("--lr", type=float, default=None, help=f"peak learning rate (default per preset: {DEFAULT_LR})")
    ap.add_argument("--min-lr-ratio", type=float, default=d.min_lr_ratio)
    ap.add_argument("--warmup", type=int, default=d.warmup)
    ap.add_argument("--weight-decay", type=float, default=d.weight_decay)
    ap.add_argument("--grad-clip", type=float, default=d.grad_clip)
    ap.add_argument("--dropout", type=float, default=d.dropout)
    ap.add_argument("--label-smoothing", type=float, default=d.label_smoothing)
    ap.add_argument("--index", default=d.index, choices=LAYOUTS,
                    help="BM25 index layout: one document per fact, or fact / topic / questions as three fields")
    ap.add_argument("--select", default=d.select, choices=["em", "loss"],
                    help="ckpt_best.pt keeps the highest validation exact match, or the lowest validation loss")
    ap.add_argument("--init-from", default=None,
                    help="run directory of brasa.pretrain: start from its weights and use its tokenizer")
    ap.add_argument("--tokenizer", default=None,
                    help="use this tokenizer.json instead of training one (default with --init-from: its tokenizer)")
    ap.add_argument("--retrieval-weight", type=float, default=d.retrieval_weight,
                    help="weight of a contrastive (question, fact) loss on the pooled hidden states, added to every "
                         "step: the same network then also ranks the facts (0 = language model only)")
    ap.add_argument("--contexts-from", default=None,
                    help="rank the training contexts with the learned retriever of this run (brasa.train with "
                         "--retrieval-weight, or brasa.encoder) fused with BM25, instead of BM25 alone")
    ap.add_argument("--retrieval-batch", type=int, default=d.retrieval_batch,
                    help="questions per contrastive batch (each with its gold fact and a same-topic negative)")
    ap.add_argument("--retrieval-temperature", type=float, default=d.retrieval_temperature)
    ap.add_argument("--retrieval-mined-negatives", type=int, default=d.retrieval_mined_negatives,
                    help="per contrastive question, add this many of the BM25 index's best wrong facts for it")
    ap.add_argument("--retrieval-pooling", default=d.retrieval_pooling, choices=["mean", "last"])
    ap.add_argument("--vocab-size", type=int, default=d.vocab_size)
    ap.add_argument("--block-size", type=int, default=d.block_size, help="context length")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--device", default=d.device, choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--compile", default=d.compile, choices=["auto", "on", "off"], help="torch.compile")
    ap.add_argument("--amp", default=d.amp, choices=["auto", "off", "fp16", "bf16"], help="autocast dtype")
    ap.add_argument("--log-every", type=int, default=d.log_every)
    ap.add_argument("--eval-every", type=int, default=d.eval_every)
    ap.add_argument("--ckpt-every", type=int, default=d.ckpt_every)
    ap.add_argument("--resume", action="store_true", help="continue from --out/ckpt_last.pt if it exists")
    ap.add_argument("--stop-at", type=int, default=None,
                    help="stop and checkpoint at this step, keeping the schedule of --steps (continue with --resume)")
    ap.add_argument("--variants", type=int, default=dd.variants, help="training examples per QA record")
    ap.add_argument("--open-book-prob", type=float, default=dd.open_book_prob)
    ap.add_argument("--noise-prob", type=float, default=dd.noise_prob)
    ap.add_argument("--word-dropout-prob", type=float, default=dd.word_dropout_prob,
                    help="share of the training variants whose question loses some of its words")
    ap.add_argument("--word-dropout-rate", type=float, default=dd.word_dropout_rate,
                    help="probability of dropping each word of such a question")
    ap.add_argument("--prompt-loss-weight", type=float, default=dd.prompt_loss_weight,
                    help="loss weight of the non-answer tokens: 0 = answer span only, 1 = every token")
    ap.add_argument("--max-facts", type=int, default=pd.max_facts, help="facts shown in open-book mode")
    ap.add_argument("--oracle-prob", type=float, default=dd.oracle_prob,
                    help="open-book variants in which the gold facts are forced into the context")
    ap.add_argument("--drop-gold-prob", type=float, default=dd.drop_gold_prob,
                    help="open-book variants in which the gold facts are removed from the context")
    ap.add_argument("--shuffle-prob", type=float, default=dd.shuffle_prob,
                    help="open-book variants whose facts are shuffled (0 keeps the retriever's ranking)")
    ap.add_argument("--random-distractor-prob", type=float, default=dd.random_distractor_prob,
                    help="open-book variants in which one retrieved fact is swapped for a random one")
    ap.add_argument("--counterfactual-prob", type=float, default=dd.counterfactual_prob,
                    help="open-book examples whose gold fact gets other numbers and names, in context and answer")
    ap.add_argument("--context-folds", type=int, default=dd.context_folds,
                    help="cross-fitted training contexts: retrieve for a training question with an index built "
                         "without the questions of its fold (0 = one index that holds every training question)")
    ap.add_argument("--importance-power", type=float, default=dd.importance_power,
                    help="sample an example in proportion to (importance of its fact / 5) ** this; 0 = uniform")
    a = ap.parse_args(argv)
    data = DataConfig(variants=a.variants, open_book_prob=a.open_book_prob, noise_prob=a.noise_prob,
                      word_dropout_prob=a.word_dropout_prob, word_dropout_rate=a.word_dropout_rate,
                      oracle_prob=a.oracle_prob, drop_gold_prob=a.drop_gold_prob, shuffle_prob=a.shuffle_prob,
                      random_distractor_prob=a.random_distractor_prob, counterfactual_prob=a.counterfactual_prob,
                      context_folds=a.context_folds,
                      importance_power=a.importance_power,
                      prompt_loss_weight=a.prompt_loss_weight, prompt=PromptConfig(max_facts=a.max_facts))
    fields = {k: v for k, v in vars(a).items() if k in TrainConfig.__dataclass_fields__}
    return TrainConfig(**fields, data=data)


def main(argv=None) -> None:
    summary = train(parse_args(argv))
    print(json.dumps({"event": "end", "train_seconds": summary["train_seconds"],
                      "tokens_per_second": summary["tokens_per_second"], "final_val": summary["final_val"]}))


if __name__ == "__main__":
    main()
