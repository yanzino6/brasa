"""Export a trained run to the files the browser engine loads, and evaluate the weights as exported.

    uv run python -m brasa.export --run runs/brasa-12m                 # writes runs/brasa-12m/export/
    uv run python -m brasa.export --run runs/brasa-12m --out ../public/brasa --model-card model-card.json

Files written to --out:
    model.f16.bin     weights as float16
    model.i8.bin      weights as int8 with one scale per output row (the file the site ships)
    tokenizer.json    the BPE merges and special tokens (also embedded in the model header)
    retrieval.json    the BM25 index over the knowledge base
    testvec.<fmt>.json  prompts, logits, greedy continuations, attention and sampling fixtures computed by
                        PyTorch with the weights exactly as the engine sees them after de-quantisation
    parity.json       tokenizer, retrieval and prompt fixtures computed by the Python reference (fixtures.py)
    manifest.json     configuration, prompt format, sizes (raw / gzip / brotli) and SHA-256 of every file

The "TGPT" container (little endian):

    "TGPT" | u32 version = 1 | u32 header_bytes | header JSON, space padded to a multiple of 4 | tensor data

    header = {"format": "f16" | "i8", "name", "config", "prompt", "tokenizer",
              "tensors": [{"name", "shape", "dtype": "f32" | "f16" | "i8", "offset", "scalesOffset"?}]}

Offsets are relative to the start of the data section and 4-byte aligned. Matrices are stored in the
chosen format; the RMSNorm gains (1-D) always stay float32. An int8 matrix W of shape (rows, cols) is
stored as q = round(W / s) with s[row] = max|W[row]| / 127, followed by the float32 scales, and the engine
rebuilds W ~ q * s[row]. The tied output head is not stored: it is the embedding matrix.

Quantisation is not assumed to be harmless: every exported format is loaded back, de-quantised and
evaluated with the same protocol as the float32 checkpoint (see evaluate.py), and the scores go into the
model card next to the float32 ones.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .config import GPTConfig, PromptConfig
from .data import build_prompt
from .fixtures import build_parity
from .model import GPT
from .tokenizer import BPETokenizer

MAGIC = b"TGPT"
VERSION = 1
FORMATS = ("f16", "i8")
FORMAT_LABEL = {"f32": "float32", "f16": "float16", "i8": "int8"}


# --------------------------------------------------------------------------------------- quantisation
def quantize(w: np.ndarray, fmt: str) -> tuple[np.ndarray, str, np.ndarray | None, np.ndarray]:
    """Returns (stored array, dtype tag, per-row scales or None, the float32 array the engine rebuilds)."""
    if fmt == "f32" or w.ndim < 2:
        return w.astype("<f4"), "f32", None, w.astype(np.float32)
    if fmt == "f16":
        half = w.astype("<f2")
        return half, "f16", None, half.astype(np.float32)
    if fmt == "i8":
        scales = (np.abs(w).max(axis=1) / 127.0).astype(np.float32)
        scales[scales == 0] = 1.0
        q = np.clip(np.rint(w / scales[:, None]), -127, 127).astype(np.int8)
        return q, "i8", scales.astype("<f4"), q.astype(np.float32) * scales[:, None]
    raise ValueError(f"unknown format {fmt!r}; choose one of {('f32', *FORMATS)}")


def _stored_tensors(model: GPT) -> dict[str, np.ndarray]:
    """Parameters in a fixed order, without the tied head."""
    return {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items() if k != "lm_head.weight"}


def dequantized_model(model: GPT, fmt: str) -> tuple[GPT, float]:
    """A copy of the model holding the weights the engine will compute with, and the mean relative
    RMS error of the matrices (0 for f32)."""
    state, errors = {}, []
    for name, w in _stored_tensors(model).items():
        deq = quantize(w, fmt)[3]
        state[name] = torch.from_numpy(np.ascontiguousarray(deq))
        if w.ndim == 2:
            errors.append(float(np.sqrt(((deq - w) ** 2).mean()) / max(1e-12, np.sqrt((w**2).mean()))))
    state["lm_head.weight"] = state["tok_emb.weight"]
    copy = GPT(GPTConfig(**{k: v for k, v in model.cfg.to_dict().items()}))
    copy.load_state_dict(state)
    return copy.eval(), float(np.mean(errors)) if errors else 0.0


# --------------------------------------------------------------------------------------- container
def write_tgpt(model: GPT, fmt: str, path: str | Path, tokenizer: BPETokenizer | None = None,
               prompt: PromptConfig | None = None, name: str = "brasa") -> int:
    """Write one model file and return its size in bytes."""
    tensors, blobs, offset = [], [], 0

    def add(arr: np.ndarray) -> int:
        nonlocal offset
        at, raw = offset, arr.tobytes()
        raw += b"\0" * (-len(raw) % 4)
        blobs.append(raw)
        offset += len(raw)
        return at

    for tensor_name, w in _stored_tensors(model).items():
        stored, dtype, scales, _ = quantize(w, fmt)
        entry = {"name": tensor_name, "shape": list(w.shape), "dtype": dtype, "offset": add(stored)}
        if scales is not None:
            entry["scalesOffset"] = add(scales)
        tensors.append(entry)

    header = {
        "format": fmt, "name": name, "config": model.describe(),
        "prompt": (prompt or PromptConfig()).to_dict(),
        "tokenizer": tokenizer.to_json() if tokenizer is not None else None,
        "tensors": tensors,
    }
    head = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    head += b" " * (-len(head) % 4)
    path = Path(path)
    path.write_bytes(MAGIC + struct.pack("<II", VERSION, len(head)) + head + b"".join(blobs))
    return path.stat().st_size


def read_tgpt(path: str | Path) -> tuple[dict, GPT]:
    """Read a model file back: (header, model with de-quantised float32 weights). The mirror of what
    web/engine/engine.js does at load time; used by the tests and by anyone who only has the .bin."""
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError(f"{path} is not a TGPT model file")
    version, head_len = struct.unpack("<II", raw[4:12])
    if version != VERSION:
        raise ValueError(f"unsupported TGPT version {version}")
    header = json.loads(raw[12 : 12 + head_len].decode("utf-8"))
    data = memoryview(raw)[12 + head_len :]
    state = {}
    for t in header["tensors"]:
        n = int(np.prod(t["shape"]))
        if t["dtype"] == "f32":
            w = np.frombuffer(data, "<f4", n, t["offset"]).astype(np.float32)
        elif t["dtype"] == "f16":
            w = np.frombuffer(data, "<f2", n, t["offset"]).astype(np.float32)
        elif t["dtype"] == "i8":
            q = np.frombuffer(data, np.int8, n, t["offset"]).astype(np.float32).reshape(t["shape"])
            scales = np.frombuffer(data, "<f4", t["shape"][0], t["scalesOffset"])
            w = q * scales[:, None]
        else:
            raise ValueError(f"unknown dtype {t['dtype']!r}")
        state[t["name"]] = torch.from_numpy(np.ascontiguousarray(w.reshape(t["shape"])))
    state["lm_head.weight"] = state["tok_emb.weight"]
    cfg = {k: v for k, v in header["config"].items() if k != "params"}
    model = GPT(GPTConfig(**{**cfg, "dropout": 0.0}))
    model.load_state_dict(state)
    return header, model.eval()


# --------------------------------------------------------------------------------------- test vectors
SAMPLING_CASES = [
    {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0},
    {"temperature": 0.7, "top_k": 40, "top_p": 1.0, "repetition_penalty": 1.0},
    {"temperature": 0.8, "top_k": 0, "top_p": 0.9, "repetition_penalty": 1.0},
    {"temperature": 1.3, "top_k": 12, "top_p": 0.75, "repetition_penalty": 1.3},
    {"temperature": 0.5, "top_k": 5, "top_p": 0.5, "repetition_penalty": 1.1},
]


@torch.no_grad()
def test_vectors(model: GPT, prompts: list[list[int]], texts: list[str] | None = None, n_new: int = 24,
                 full_logits: int = 3) -> dict:
    """Reference outputs for the JavaScript engine (web/engine/test/parity.mjs).

    cases     every prompt: the greedy continuation; the first `full_logits` also carry the whole logit
              vector after the last prompt token
    inspect   first prompt: attention of every layer and head from the last position, residual norms at
              the last position, top-5 next-token probabilities
    sampling  the distribution GPT.filter_logits leaves for several settings (kept ids and their mass)"""
    model = model.eval().float().cpu()
    cases = []
    for i, prompt in enumerate(prompts):
        idx = torch.tensor([prompt])
        logits, _, _ = model(idx)
        budget = min(n_new, model.cfg.block_size - len(prompt))
        case = {"prompt": prompt, "greedy": model.generate(idx, budget)[0].tolist()}
        if texts:
            case["text"] = texts[i]
        if i < full_logits:
            case["last_logits"] = [round(float(v), 6) for v in logits[0, -1]]
        cases.append(case)

    idx = torch.tensor([prompts[0]])
    seen = model.inspect(idx)
    last = seen["logits"][0, -1]
    probs = torch.softmax(last, dim=-1)
    top = torch.topk(probs, min(5, probs.numel()))
    inspect = {
        "prompt": prompts[0],
        "attention": [[[round(float(v), 6) for v in head[-1]] for head in layer[0]] for layer in seen["attention"]],
        "resid_norms": [round(float(v), 5) for v in seen["resid_norms"][:, 0, -1]],
        "top": [{"id": int(i), "prob": round(float(p), 6)} for p, i in zip(top.values, top.indices, strict=True)],
    }

    previous = torch.tensor([prompts[0][-8:]])
    sampling = []
    for params in SAMPLING_CASES:
        kept = GPT.filter_logits(last[None, :], params["temperature"], params["top_k"] or None, params["top_p"],
                                 params["repetition_penalty"], previous)[0]
        p = torch.softmax(kept, dim=-1)
        ids = torch.nonzero(torch.isfinite(kept)).flatten().tolist()
        sampling.append({"params": params, "previous": previous[0].tolist(), "ids": ids,
                         "probs": [round(float(p[i]), 6) for i in ids]})
    return {"cases": cases, "inspect": inspect, "sampling": sampling}


def default_prompts(tok: BPETokenizer, dataset, retriever, block_size: int, prompt_cfg: PromptConfig,
                    n: int = 8) -> tuple[list[list[int]], list[str]]:
    """Real prompts for the test vectors: test questions, alternating open-book and closed-book."""
    records = [r for r in dataset.split("test")] or list(dataset.qa)
    step = max(1, len(records) // n)
    prompts, texts = [], []
    for i, rec in enumerate(records[::step][:n]):
        facts = [h.text for h in retriever.search(rec.q, prompt_cfg.max_facts)] if i % 2 == 0 else None
        ids, _ = build_prompt(tok, rec.q, facts, block_size, prompt_cfg)
        prompts.append(ids)
        texts.append(tok.decode(ids, skip_special=False))
    return prompts, texts


def random_prompts(vocab_size: int, lengths=(1, 5, 17, 40), seed: int = 1) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab_size, (n,), generator=g).tolist() for n in lengths]


# --------------------------------------------------------------------------------------- sizes
def _brotli_sizes(paths: list[Path]) -> dict[str, int | None]:
    """Brotli (quality 11) size of each file: the Python module if installed, else Node's zlib, else None."""
    try:
        import brotli  # type: ignore[import-not-found]

        return {p.name: len(brotli.compress(p.read_bytes(), quality=11)) for p in paths}
    except ImportError:
        pass
    node = shutil.which("node")
    if node:
        script = ("const fs=require('fs'),z=require('zlib');const o={};for(const p of process.argv.slice(1)){"
                  "o[require('path').basename(p)]=z.brotliCompressSync(fs.readFileSync(p),{params:{"
                  "[z.constants.BROTLI_PARAM_QUALITY]:11,[z.constants.BROTLI_PARAM_LGWIN]:24}}).length}"
                  "console.log(JSON.stringify(o))")
        try:
            out = subprocess.run([node, "-e", script, *map(str, paths)], capture_output=True, text=True, check=True,
                                 timeout=600)
            return json.loads(out.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError):
            pass
    return {p.name: None for p in paths}


def size_report(paths: list[Path], brotli: bool = True) -> dict[str, dict]:
    """bytes / gzip -9 / brotli -11 / sha256 for every file (what a CDN would actually transfer).

    Brotli at quality 11 takes about a second per megabyte; brotli=False leaves that column empty."""
    brotli_sizes = _brotli_sizes(paths) if brotli else {p.name: None for p in paths}
    report = {}
    for p in paths:
        raw = p.read_bytes()
        report[p.name] = {"bytes": len(raw), "gzip": len(gzip.compress(raw, 9)), "brotli": brotli_sizes[p.name],
                          "sha256": hashlib.sha256(raw).hexdigest()}
    return report


# --------------------------------------------------------------------------------------- the export
def export_model(model: GPT, tok: BPETokenizer, out: str | Path, prompts: list[list[int]],
                 texts: list[str] | None = None, prompt_cfg: PromptConfig | None = None, name: str = "brasa",
                 formats=FORMATS, retriever=None, dataset=None, brotli: bool = True) -> dict:
    """Write the model files, test vectors, tokenizer, retrieval index, parity fixtures and manifest.

    Returns the manifest. `dataset` only adds real questions to the parity fixtures."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prompt_cfg = prompt_cfg or PromptConfig()
    model = model.eval().float().cpu()

    reference = test_vectors(model, prompts, texts)
    ref_logits = np.array([c["last_logits"] for c in reference["cases"] if "last_logits" in c])
    parity, written = {}, []
    for fmt in formats:
        path = out / f"model.{fmt}.bin"
        write_tgpt(model, fmt, path, tok, prompt_cfg, name)
        _, loaded = read_tgpt(path)  # the vectors come from the file itself, not from the in-memory weights
        vec = test_vectors(loaded, prompts, texts)
        (out / f"testvec.{fmt}.json").write_text(json.dumps({"format": fmt, **vec}), encoding="utf-8")
        logits = np.array([c["last_logits"] for c in vec["cases"] if "last_logits" in c])
        same = sum(a["greedy"] == b["greedy"] for a, b in zip(vec["cases"], reference["cases"], strict=True))
        parity[fmt] = {
            "mean_relative_weight_rmse": round(dequantized_model(model, fmt)[1], 6),
            "max_abs_logit_drift_vs_float32": round(float(np.abs(logits - ref_logits).max()), 5),
            "greedy_identical_to_float32": f"{same}/{len(prompts)}",
        }
        written += [path, out / f"testvec.{fmt}.json"]

    tok.save(out / "tokenizer.json")
    written.append(out / "tokenizer.json")
    if retriever is not None:
        retriever.save(out / "retrieval.json")
        written.append(out / "retrieval.json")
    parity_fixtures = build_parity(tok, retriever, dataset, model.cfg.block_size, prompt_cfg)
    (out / "parity.json").write_text(json.dumps(parity_fixtures, ensure_ascii=False), encoding="utf-8")
    written.append(out / "parity.json")

    manifest = {
        "schema_version": 1, "name": name, "container": {"magic": "TGPT", "version": VERSION},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": model.describe(), "prompt": prompt_cfg.to_dict(),
        "special_tokens": tok.special_ids,
        "default_model": f"model.{'i8' if 'i8' in formats else formats[0]}.bin",
        "files": size_report(written, brotli), "quantisation": parity,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def evaluate_formats(model: GPT, tok: BPETokenizer, dataset, retriever, device, prompt_cfg: PromptConfig,
                     formats=FORMATS, splits=("val", "test"), seed: int = 0) -> dict:
    """Run the evaluation protocol on the de-quantised weights of every format.

    Returns {label: {"headline", "full", "answers_identical_to_float32", "changed_examples"}}."""
    from .evaluate import evaluate_model, headline

    def run(m: GPT):
        return evaluate_model(m.to(device), tok, dataset, retriever, device, prompt_cfg, splits=splits, seed=seed)

    _, ref_rows = run(model)
    out = {}
    for fmt in formats:
        qmodel, _ = dequantized_model(model.cpu(), fmt)
        results, rows = run(qmodel)
        same = sum(a["pred"] == b["pred"] for a, b in zip(rows, ref_rows, strict=True))
        changed = [{"q": a["q"], "mode": a["mode"], "float32": b["pred"], FORMAT_LABEL[fmt]: a["pred"]}
                   for a, b in zip(rows, ref_rows, strict=True) if a["pred"] != b["pred"]]
        out[FORMAT_LABEL[fmt]] = {
            "headline": headline(results), "full": results,
            "answers_identical_to_float32": f"{same}/{len(rows)}", "changed_examples": changed[:5],
        }
    return out


def main(argv=None) -> None:
    from .data import load_dataset
    from .evaluate import load_run, update_model_card
    from .train import pick_device

    ap = argparse.ArgumentParser(description="Export a run for the browser engine and evaluate the exported weights.")
    ap.add_argument("--run", required=True, help="run directory written by brasa.train")
    ap.add_argument("--out", help="output directory (default: <run>/export)")
    ap.add_argument("--checkpoint", default="last", choices=["last", "best"])
    ap.add_argument("--formats", default=",".join(FORMATS), help="comma separated: f16, i8, f32")
    ap.add_argument("--data-dir", help="defaults to the data directory the run was trained on")
    ap.add_argument("--no-eval", action="store_true", help="skip the evaluation of the quantised weights")
    ap.add_argument("--no-brotli", action="store_true", help="skip the (slow) brotli column of the size report")
    ap.add_argument("--eval-splits", default="val,test")
    ap.add_argument("--device", default="auto", help="device for the evaluation")
    ap.add_argument("--model-card", help="model card to update (default: <run>/model-card.json)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    out = Path(args.out) if args.out else run_dir / "export"
    formats = tuple(f for f in args.formats.split(",") if f)
    model, tok, retriever, run = load_run(run_dir, args.checkpoint, "cpu")
    dataset = load_dataset(args.data_dir or run["train_config"]["data_dir"])
    prompt_cfg = PromptConfig(**run["train_config"]["data"]["prompt"])

    prompts, texts = default_prompts(tok, dataset, retriever, model.cfg.block_size, prompt_cfg)
    manifest = export_model(model, tok, out, prompts, texts, prompt_cfg, run["preset"], formats, retriever, dataset,
                            brotli=not args.no_brotli)
    for name, f in manifest["files"].items():
        brotli = "-" if f["brotli"] is None else format(f["brotli"], ",")
        print(f"{name:18s} {f['bytes']:>11,} bytes   gzip {f['gzip']:>11,}   brotli {brotli:>11}")
    for fmt, q in manifest["quantisation"].items():
        print(f"{fmt}: relative weight RMSE {q['mean_relative_weight_rmse']}, max logit drift "
              f"{q['max_abs_logit_drift_vs_float32']}, greedy identical {q['greedy_identical_to_float32']}")

    sections: dict = {"files": {"directory": str(out), **manifest["files"]}, "quantisation": manifest["quantisation"]}
    if not args.no_eval:
        t0 = time.time()
        device = pick_device(args.device)
        evaluated = evaluate_formats(model, tok, dataset, retriever, device, prompt_cfg, formats,
                                     tuple(args.eval_splits.split(",")), run["train_config"]["seed"])
        sections["evaluation"] = evaluated
        for label, e in evaluated.items():
            test = {mode: per.get("test") for mode, per in e["headline"].items()}
            print(f"{label}: answers identical to float32 {e['answers_identical_to_float32']}; test "
                  + "; ".join(f"{mode} em {t['em']} f1 {t['f1']}" for mode, t in test.items() if t))
        print(f"evaluated the exported weights in {time.time() - t0:.1f}s")
    card = args.model_card or run_dir / "model-card.json"
    update_model_card(card, sections)
    print(f"export written to {out}; model card {card}")


if __name__ == "__main__":
    main()
