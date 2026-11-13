"""Pretrain Brasa as a plain language model on public text, before it ever sees a question.

Still training from scratch: the weights start random and the only data is a text file on disk (for the
experiment in the README, a fixed slice of Portuguese and English Wikipedia; scripts/fetch_wiki.py). No
pretrained weights are involved anywhere. The question is whether general knowledge of the two languages
(which words are related, how sentences go) helps a model of this size answer phrasings it has never seen.

    uv run python -m brasa.pretrain --preset brasa-12m --data-dir data/yan --out runs/pre-12m --steps 14000 \
        --text data/cache/wiki/wiki.pt.txt data/cache/wiki/wiki.en.txt
    uv run python -m brasa.train --preset brasa-12m --data-dir data/yan --out runs/brasa-12m --init-from runs/pre-12m

What it does:

1. Tokenizer: the same byte-level BPE, trained on the first --tokenizer-mb megabytes of every text file plus
   what brasa.train would use (the facts, the topic names and the TRAINING questions of --data-dir; held-out
   questions never reach the vocabulary).
2. The text is encoded once into a flat array of token ids, every line (one article) wrapped in
   <|bos|> ... <|eos|>, and cached in --corpus (default <out>/corpus) so that two model sizes share it.
3. Next-token prediction on random windows of block_size tokens, with the optimiser and schedule of
   brasa.train. batch(step) is a pure function of (seed, step); --resume continues from ckpt_last.pt.

Files written to --out: tokenizer.json, ckpt_last.pt, train_log.jsonl, run.json (with the manifest of the
text when a manifest.json sits next to the text files). brasa.train --init-from <out> starts from them.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from .config import BOS, DEFAULT_BLOCK_SIZE, DEFAULT_VOCAB_SIZE, EOS, PRESETS, preset
from .data import load_dataset, tokenizer_texts
from .model import GPT
from .tokenizer import BPETokenizer
from .train import (
    DEFAULT_LR,
    lr_at,
    make_optimizer,
    make_scaler,
    optimizer_step,
    pick_device,
    require_finite,
    save_checkpoint,
)


@dataclass
class PretrainConfig:
    preset: str = "brasa-12m"
    text: list[str] = field(default_factory=list)
    data_dir: str | None = None  # its facts and training questions join the tokenizer's training text
    out: str = "runs/pretrain"
    corpus: str | None = None  # directory of the encoded text (default <out>/corpus)
    steps: int = 14000
    batch_tokens: int = 64 * DEFAULT_BLOCK_SIZE
    lr: float | None = None
    min_lr_ratio: float = 0.1
    warmup: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dropout: float = 0.0
    vocab_size: int = DEFAULT_VOCAB_SIZE
    block_size: int = DEFAULT_BLOCK_SIZE
    tokenizer_mb: float = 8.0
    val_fraction: float = 0.005
    seed: int = 0
    device: str = "auto"
    compile: str = "auto"
    amp: str = "auto"
    log_every: int = 100
    eval_every: int = 1000
    ckpt_every: int = 1000
    workers: int = 8
    resume: bool = False
    stop_at: int | None = None  # stop (and checkpoint) at this step, keeping the schedule of `steps`
    prepare_only: bool = False  # train the tokenizer, encode the text and stop


# --------------------------------------------------------------------------------------- corpus
def read_head(path: str | Path, max_bytes: int) -> list[str]:
    """The first lines of a text file, up to max_bytes."""
    lines, size = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            lines.append(line)
            size += len(line.encode("utf-8"))
            if size >= max_bytes:
                break
    return lines


_worker_tok: BPETokenizer | None = None


def _init_worker(tokenizer_path: str) -> None:
    global _worker_tok
    _worker_tok = BPETokenizer.load(tokenizer_path)


def _encode_lines(lines: list[str]) -> np.ndarray:
    tok = _worker_tok
    bos, eos = tok.special_ids[BOS], tok.special_ids[EOS]
    out: list[int] = []
    for line in lines:
        ids = tok.encode(line)
        if ids:
            out.append(bos)
            out.extend(ids)
            out.append(eos)
    return np.asarray(out, dtype=np.uint16)


def encode_files(paths: list[str], tokenizer_path: Path, workers: int = 8, lines_per_task: int = 200) -> np.ndarray:
    """Token ids of every line of every file, each line wrapped in <|bos|> ... <|eos|>, in file order."""
    def tasks():
        for path in paths:
            with open(path, encoding="utf-8") as f:
                chunk: list[str] = []
                for line in f:
                    chunk.append(line)
                    if len(chunk) == lines_per_task:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk

    if workers <= 1:
        _init_worker(str(tokenizer_path))
        parts = [_encode_lines(t) for t in tasks()]
    else:
        with Pool(workers, initializer=_init_worker, initargs=(str(tokenizer_path),)) as pool:
            parts = list(pool.imap(_encode_lines, tasks(), chunksize=1))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)


def prepare_corpus(cfg: PretrainConfig, corpus_dir: Path) -> tuple[BPETokenizer, np.ndarray, dict]:
    """(tokenizer, token array, description). Built once and cached in corpus_dir."""
    tok_path, ids_path, meta_path = corpus_dir / "tokenizer.json", corpus_dir / "tokens.npy", corpus_dir / "corpus.json"
    if tok_path.exists() and ids_path.exists() and meta_path.exists():
        return BPETokenizer.load(tok_path), np.load(ids_path, mmap_mode="r"), json.loads(meta_path.read_text())
    if not cfg.text:
        raise ValueError("no --text files and no prepared corpus")
    if cfg.vocab_size > 65536:
        raise ValueError("token ids are stored as uint16")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    texts: list[str] = []
    for path in cfg.text:
        texts += read_head(path, int(cfg.tokenizer_mb * 1e6))
    if cfg.data_dir:
        texts += tokenizer_texts(load_dataset(cfg.data_dir))
    tok = BPETokenizer.train(texts, cfg.vocab_size)
    tok.save(tok_path)
    t1 = time.time()
    ids = encode_files(cfg.text, tok_path, cfg.workers)
    np.save(ids_path, ids)
    manifests = {}
    for path in cfg.text:  # the provenance of the text, when the fetch script left it next to the file
        m = Path(path).parent / "manifest.json"
        if m.exists():
            manifests[str(m)] = json.loads(m.read_text(encoding="utf-8"))
    meta = {
        "text_files": [{"path": p, "bytes": Path(p).stat().st_size} for p in cfg.text],
        "tokenizer": {"vocab_size": tok.vocab_size, "merges": len(tok.merges),
                      "trained_on_mb_per_file": cfg.tokenizer_mb, "plus_data_dir": cfg.data_dir,
                      "seconds": round(t1 - t0, 1)},
        "tokens": int(len(ids)), "encode_seconds": round(time.time() - t1, 1), "manifests": manifests,
    }
    meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return tok, ids, meta


def window_batch(ids: np.ndarray, n_train: int, block: int, rows: int, seed: int, step: int):
    """(x, y) int64 arrays of `rows` random windows of the training part; a pure function of (seed, step)."""
    rng = np.random.default_rng([seed, step])
    starts = rng.integers(0, n_train - block - 1, size=rows)
    idx = starts[:, None] + np.arange(block + 1)[None, :]
    chunk = np.asarray(ids[idx], dtype=np.int64)
    return np.ascontiguousarray(chunk[:, :-1]), np.ascontiguousarray(chunk[:, 1:])


@torch.no_grad()
def lm_loss(model: GPT, ids: np.ndarray, start: int, block: int, device, rows: int = 32, batches: int = 8) -> float:
    """Mean next-token cross-entropy on consecutive windows of the held-out tail of the stream."""
    was_training = model.training
    model.eval()
    total, n, pos = 0.0, 0, start
    for _ in range(batches):
        if pos + rows * (block + 1) > len(ids):
            break
        chunk = np.asarray(ids[pos : pos + rows * (block + 1)], dtype=np.int64).reshape(rows, block + 1)
        x = torch.from_numpy(np.ascontiguousarray(chunk[:, :-1])).to(device)
        y = torch.from_numpy(np.ascontiguousarray(chunk[:, 1:])).to(device)
        _, loss, _ = model(x, y)
        total += float(loss)
        n += 1
        pos += rows * (block + 1)
    model.train(was_training)
    return total / max(1, n)


# --------------------------------------------------------------------------------------- training
def pretrain(cfg: PretrainConfig) -> dict:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    base_lr = cfg.lr if cfg.lr is not None else DEFAULT_LR.get(cfg.preset, 6e-4)
    torch.manual_seed(cfg.seed)
    tok, ids, corpus = prepare_corpus(cfg, Path(cfg.corpus) if cfg.corpus else out / "corpus")
    tok.save(out / "tokenizer.json")
    block = cfg.block_size
    n_val = max(block + 1, int(len(ids) * cfg.val_fraction))
    n_train = len(ids) - n_val
    if n_train <= block + 1:
        raise ValueError(f"the text is too short: {len(ids)} tokens for windows of {block}")
    rows = max(1, cfg.batch_tokens // block)

    model = GPT(preset(cfg.preset, vocab_size=tok.vocab_size, block_size=block, dropout=cfg.dropout)).to(device)
    opt = make_optimizer(model, cfg, base_lr)
    last = out / "ckpt_last.pt"
    start_step, elapsed = 0, 0.0
    resuming = cfg.resume and last.exists()
    if resuming:
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step, elapsed = ckpt["step"], ckpt["elapsed"]
    use_compile = cfg.compile == "on" or (cfg.compile == "auto" and device.type in ("mps", "cuda"))
    amp = cfg.amp if cfg.amp != "auto" else ("fp16" if device.type in ("mps", "cuda") else "off")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    forward = torch.compile(model) if use_compile else model
    scaler = make_scaler(device, amp)
    if resuming and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])

    info = {"event": "start", "objective": "next-token prediction on plain text (pretraining from scratch)",
            "preset": cfg.preset, "model": model.describe(), "device": str(device), "compile": use_compile, "amp": amp,
            "lr": base_lr, "rows": rows, "tokens_train": int(n_train), "tokens_val": int(n_val), "corpus": corpus,
            "resumed_from_step": start_step if resuming else None, "torch": torch.__version__}
    log_file = open(out / "train_log.jsonl", "a" if resuming else "w", encoding="utf-8")  # noqa: SIM115

    def log(record: dict) -> None:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def checkpoint(step: int) -> None:
        save_checkpoint(last, model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                        step=step, elapsed=elapsed,
                        config=model.describe(), pretrain_config=asdict(cfg))

    log(info)
    model.train()
    window_loss, window_steps, t_window = 0.0, 0, time.time()
    val_history: list[dict] = []
    step = start_step
    end = min(cfg.steps, cfg.stop_at) if cfg.stop_at else cfg.steps
    try:
        for step in range(start_step, end):
            lr = lr_at(step, cfg, base_lr)
            for group in opt.param_groups:
                group["lr"] = lr
            x, y = window_batch(ids, n_train, block, rows, cfg.seed, step)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    _, loss, _ = forward(x, y)
            else:
                _, loss, _ = forward(x, y)
            optimizer_step(model, opt, scaler, loss, cfg.grad_clip)
            window_loss += float(loss.detach())
            window_steps += 1
            done = step + 1
            if done % cfg.log_every == 0 or done == end:
                if device.type == "mps":
                    torch.mps.synchronize()
                dt = time.time() - t_window
                elapsed += dt
                require_finite(window_loss, "the pretraining loss", done)
                log({"step": done, "loss": round(window_loss / window_steps, 4), "lr": round(lr, 7),
                     "tok_s": round(window_steps * rows * block / dt), "elapsed_s": round(elapsed, 1)})
                window_loss, window_steps, t_window = 0.0, 0, time.time()
            if done % cfg.eval_every == 0 or done == end:
                val = lm_loss(model, ids, n_train, block, device)
                rec = {"step": done, "val_loss": round(val, 4), "val_ppl": round(math.exp(min(val, 20)), 1)}
                val_history.append(rec)
                log(rec)
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

    tokens_seen = max(start_step, end) * rows * block
    summary = {**{k: v for k, v in info.items() if k != "event"}, "pretrain_config": asdict(cfg),
               "steps": max(start_step, end), "planned_steps": cfg.steps,
               "train_seconds": round(elapsed, 1), "tokens_seen": tokens_seen,
               "epochs_equivalent": round(tokens_seen / max(1, n_train), 2),
               "final_val": val_history[-1] if val_history else None}
    (out / "run.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args(argv=None) -> PretrainConfig:
    d = PretrainConfig()
    ap = argparse.ArgumentParser(description="Pretrain Brasa on plain text (next-token prediction, from scratch).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--preset", default=d.preset, choices=sorted(PRESETS))
    ap.add_argument("--text", nargs="*", default=[], help="plain text files, one document per line")
    ap.add_argument("--data-dir", default=None, help="its facts and training questions also train the tokenizer")
    ap.add_argument("--out", default=d.out)
    ap.add_argument("--corpus", default=None, help="directory of the encoded text, shared between runs")
    ap.add_argument("--steps", type=int, default=d.steps)
    ap.add_argument("--batch-tokens", type=int, default=d.batch_tokens)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=d.warmup)
    ap.add_argument("--dropout", type=float, default=d.dropout)
    ap.add_argument("--vocab-size", type=int, default=d.vocab_size)
    ap.add_argument("--block-size", type=int, default=d.block_size)
    ap.add_argument("--tokenizer-mb", type=float, default=d.tokenizer_mb,
                    help="megabytes of every text file the tokenizer is trained on")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--device", default=d.device, choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--compile", default=d.compile, choices=["auto", "on", "off"])
    ap.add_argument("--amp", default=d.amp, choices=["auto", "off", "fp16", "bf16"])
    ap.add_argument("--log-every", type=int, default=d.log_every)
    ap.add_argument("--eval-every", type=int, default=d.eval_every)
    ap.add_argument("--ckpt-every", type=int, default=d.ckpt_every)
    ap.add_argument("--workers", type=int, default=d.workers, help="processes used to encode the text")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-at", type=int, default=None, help="stop and checkpoint at this step (then --resume)")
    ap.add_argument("--prepare-only", action="store_true", help="train the tokenizer, encode the text and stop")
    a = ap.parse_args(argv)
    return PretrainConfig(**{k: v for k, v in vars(a).items() if k in PretrainConfig.__dataclass_fields__})


def main(argv=None) -> None:
    cfg = parse_args(argv)
    if cfg.prepare_only:
        _, _, meta = prepare_corpus(cfg, Path(cfg.corpus) if cfg.corpus else Path(cfg.out) / "corpus")
        print(json.dumps(meta, indent=1))
        return
    summary = pretrain(cfg)
    print(json.dumps({"event": "end", "train_seconds": summary["train_seconds"], "final_val": summary["final_val"]}))


if __name__ == "__main__":
    main()
