"""ONNX export, kept as an interoperability artefact for the model repository.

The site does not use ONNX: the browser runs the hand-written engine in web/engine. These files exist so
that the same weights can be loaded by any ONNX runtime (onnxruntime, onnxruntime-web, Transformers.js with
a custom loop, mobile runtimes) without this code base.

    uv sync --extra onnx
    uv run python -m brasa.export_onnx --run runs/brasa-12m          # writes runs/brasa-12m/export/onnx/

Two graphs are written, both float32:

    brasa.onnx          input_ids (batch, T) int64   ->  logits (batch, T, vocab)
    brasa.cached.onnx   input_ids (1, T) int64, past (L, 2, 1, H, P, head_dim)
                                                     ->  logits (1, T, vocab), present (L, 2, 1, H, P+T, head_dim)

The second one carries the key/value cache of every layer as ONE tensor, so a caller shuttles a single
buffer between steps: start with P = 0, then feed one token at a time together with the `present` of the
previous call.

Each graph is checked with onnx.checker and then run with onnxruntime against PyTorch (logits and greedy
continuations, including token-by-token decoding through the cache). The numbers go to onnx_report.json;
the command fails if a graph does not reproduce PyTorch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .export import random_prompts, size_report
from .model import GPT

LOGIT_TOLERANCE = 1e-3


class _Full(nn.Module):
    def __init__(self, model: GPT):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids)[0]


class _Cached(nn.Module):
    def __init__(self, model: GPT):
        super().__init__()
        self.model = model

    def forward(self, input_ids, past):
        cache = [(past[i, 0], past[i, 1]) for i in range(past.shape[0])]
        logits, _, new_cache = self.model(input_ids, cache=cache)
        return logits, torch.stack([torch.stack(kv) for kv in new_cache])


def _export(wrapper: nn.Module, example: tuple, path: Path, inputs: list[str], outputs: list[str],
            dynamic_axes: dict, dynamic_shapes: tuple, opset: int) -> dict:
    """Export with the current (dynamo) exporter, falling back to the TorchScript one. Returns what happened."""
    import onnx

    attempts = []
    wrapper.eval()
    for exporter in ("dynamo", "torchscript"):
        t0 = time.time()
        try:
            path.unlink(missing_ok=True)
            if exporter == "dynamo":
                torch.onnx.export(wrapper, example, str(path), input_names=inputs, output_names=outputs,
                                  opset_version=opset, dynamo=True, dynamic_shapes=dynamic_shapes,
                                  external_data=False)
            else:
                torch.onnx.export(wrapper, example, str(path), input_names=inputs, output_names=outputs,
                                  opset_version=opset, dynamo=False, dynamic_axes=dynamic_axes)
            graph = onnx.load(str(path))
            onnx.checker.check_model(graph)
            return {"exporter": exporter, "opset": graph.opset_import[0].version, "nodes": len(graph.graph.node),
                    "seconds": round(time.time() - t0, 1), "failed_attempts": attempts}
        except Exception as e:  # noqa: BLE001  (any exporter failure means: try the other exporter)
            attempts.append({"exporter": exporter, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    raise RuntimeError(f"could not export {path.name}: {attempts}")


def _session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


@torch.no_grad()
def check_parity(model: GPT, full_path: Path, cached_path: Path, prompts: list[list[int]], n_new: int = 16) -> dict:
    """Compare both graphs with PyTorch on every prompt: all logits, and a greedy continuation."""
    cfg = model.cfg
    full, cached = _session(full_path), _session(cached_path)
    empty = np.zeros((cfg.n_layer, 2, 1, cfg.n_head, 0, cfg.head_dim), dtype=np.float32)
    diff = {"full": 0.0, "cached_one_shot": 0.0, "cached_token_by_token": 0.0}
    greedy = {"full": 0, "cached": 0}
    for prompt in prompts:
        ids = np.array([prompt], dtype=np.int64)
        ref = model(torch.tensor([prompt]))[0].numpy()
        diff["full"] = max(diff["full"], float(np.abs(full.run(None, {"input_ids": ids})[0] - ref).max()))
        logits, _ = cached.run(None, {"input_ids": ids, "past": empty})
        diff["cached_one_shot"] = max(diff["cached_one_shot"], float(np.abs(logits - ref).max()))
        past = empty
        for t, token in enumerate(prompt):
            logits, past = cached.run(None, {"input_ids": np.array([[token]], dtype=np.int64), "past": past})
            step_diff = float(np.abs(logits[0, 0] - ref[0, t]).max())
            diff["cached_token_by_token"] = max(diff["cached_token_by_token"], step_diff)

        budget = min(n_new, cfg.block_size - len(prompt))
        want = model.generate(torch.tensor([prompt]), budget)[0].tolist()
        seq, got_full, got_cached = list(prompt), [], []
        for _ in range(budget):  # the plain graph: re-run the whole sequence
            nxt = int(full.run(None, {"input_ids": np.array([seq], dtype=np.int64)})[0][0, -1].argmax())
            got_full.append(nxt)
            seq.append(nxt)
        for _ in range(budget):  # the cached graph: continue from the cache built above
            nxt = int(logits[0, -1].argmax())
            got_cached.append(nxt)
            logits, past = cached.run(None, {"input_ids": np.array([[nxt]], dtype=np.int64), "past": past})
        greedy["full"] += got_full == want
        greedy["cached"] += got_cached == want
    ok = max(diff.values()) <= LOGIT_TOLERANCE and all(v == len(prompts) for v in greedy.values())
    return {"ok": ok, "prompts": len(prompts), "max_abs_logit_diff": {k: float(f"{v:.2e}") for k, v in diff.items()},
            "greedy_identical": {k: f"{v}/{len(prompts)}" for k, v in greedy.items()}}


def export_onnx(model: GPT, out: str | Path, prompts: list[list[int]] | None = None, opset: int = 18) -> dict:
    """Write brasa.onnx, brasa.cached.onnx and onnx_report.json into `out`; returns the report."""
    import onnx
    import onnxruntime as ort

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = model.eval().float().cpu()
    cfg = model.cfg
    batch = torch.export.Dim("batch", min=1, max=64)
    length = torch.export.Dim("T", min=1, max=cfg.block_size)
    cached_len = torch.export.Dim("P", min=0, max=cfg.block_size - 1)
    ids = torch.randint(0, cfg.vocab_size, (2, 7))
    past = torch.randn(cfg.n_layer, 2, 1, cfg.n_head, 5, cfg.head_dim)

    full_path, cached_path = out / "brasa.onnx", out / "brasa.cached.onnx"
    report = {
        "config": model.describe(), "torch": torch.__version__, "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "graphs": {
            full_path.name: _export(
                _Full(model), (ids,), full_path, ["input_ids"], ["logits"],
                {"input_ids": {0: "batch", 1: "T"}, "logits": {0: "batch", 1: "T"}}, ({0: batch, 1: length},), opset),
            cached_path.name: _export(
                _Cached(model), (ids[:1], past), cached_path, ["input_ids", "past"], ["logits", "present"],
                {"input_ids": {1: "T"}, "past": {4: "P"}, "logits": {1: "T"}, "present": {4: "PT"}},
                ({1: length}, {4: cached_len}), opset),
        },
    }
    prompts = prompts or random_prompts(cfg.vocab_size, lengths=(1, 5, 17, min(40, cfg.block_size - 16)))
    report["parity"] = check_parity(model, full_path, cached_path, prompts)
    report["files"] = size_report([full_path, cached_path], brotli=False)
    (out / "onnx_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Export a trained run to ONNX and verify it with onnxruntime.")
    ap.add_argument("--run", required=True, help="run directory written by brasa.train")
    ap.add_argument("--out", help="output directory (default: <run>/export/onnx)")
    ap.add_argument("--checkpoint", default="last", choices=["last", "best"])
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--model-card", help="model card to update (default: <run>/model-card.json)")
    args = ap.parse_args(argv)

    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"{e}. Install the optional dependencies first: uv sync --extra onnx") from e

    from .config import PromptConfig
    from .data import load_dataset
    from .evaluate import load_run, update_model_card
    from .export import default_prompts

    run_dir = Path(args.run)
    out = Path(args.out) if args.out else run_dir / "export" / "onnx"
    model, tok, retriever, run = load_run(run_dir, args.checkpoint, "cpu")
    prompt_cfg = PromptConfig(**run["train_config"]["data"]["prompt"])
    dataset = load_dataset(run["train_config"]["data_dir"])
    prompts, _ = default_prompts(tok, dataset, retriever, model.cfg.block_size, prompt_cfg, n=6)

    report = export_onnx(model, out, prompts, args.opset)
    for name, graph in report["graphs"].items():
        print(f"{name:18s} {report['files'][name]['bytes']:>11,} bytes   exporter {graph['exporter']}, "
              f"opset {graph['opset']}, {graph['nodes']} nodes")
    print("parity vs PyTorch:", json.dumps(report["parity"]))
    update_model_card(args.model_card or run_dir / "model-card.json",
                      {"onnx": {"directory": str(out), "files": report["files"], "parity": report["parity"],
                                "exporters": {k: v["exporter"] for k, v in report["graphs"].items()}}})
    if not report["parity"]["ok"]:
        raise SystemExit("the ONNX graphs do not reproduce PyTorch")
    print(f"ONNX export written to {out}")


if __name__ == "__main__":
    main()
