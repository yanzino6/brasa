"""Print the summary of a smoke run (scripts/smoke.sh) and write it to <run>/smoke_summary.json.

    uv run python scripts/smoke_summary.py runs/smoke

Reads what the pipeline wrote: run.json and train_log.jsonl (training), model-card.json (evaluation of the
float32, float16 and int8 weights, file sizes), parity_report.json and bench.jsonl (Node engine).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def loss_curve(log_path: Path) -> dict:
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train = [r for r in rows if "loss" in r]
    val = [r for r in rows if "val_loss" in r]
    last = train[-1]["step"]
    picks = []
    for fraction in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):  # the logged point closest to each fraction of the run
        row = min(train, key=lambda r: abs(r["step"] - fraction * last))
        if row["step"] not in [p["step"] for p in picks]:
            picks.append({"step": row["step"], "loss": row["loss"], "lr": row["lr"]})
    best = min(val, key=lambda r: r["val_loss"]) if val else None
    resumed = [r["resumed_from_step"] for r in rows if r.get("event") == "start" and r.get("resumed_from_step")]
    speeds = sorted(r["tok_s"] for r in train)
    return {"train": picks, "val": [{k: r[k] for k in ("step", "val_loss", "val_loss_open", "val_loss_closed")}
                                    for r in val], "best_val": best, "resumed_from_steps": resumed,
            "median_tokens_per_second": speeds[len(speeds) // 2]}


def pct(x) -> str:
    return "   n/a" if x is None else f"{100 * x:5.1f}%"


def main() -> None:
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/smoke")
    info, card = load(run / "run.json"), load(run / "model-card.json")
    if info is None or card is None:
        raise SystemExit(f"{run} has no run.json / model-card.json: run scripts/smoke.sh first")
    curve = loss_curve(run / "train_log.jsonl")
    parity = load(run / "parity_report.json")
    bench_path = run / "bench.jsonl"
    bench_lines = bench_path.read_text(encoding="utf-8").splitlines() if bench_path.exists() else []
    bench = [json.loads(line) for line in bench_lines if line.strip()]
    evaluation, files = card["evaluation"], card.get("files", {})

    summary = {
        "run": str(run), "preset": info["preset"], "params": info["model"]["params"], "model": info["model"],
        "tokenizer": info["tokenizer"], "data": info["data"], "training_examples": info["examples"],
        "training": {k: info[k] for k in ("steps", "device", "compile", "amp", "lr", "train_seconds",
                                          "tokens_per_second", "tokens_seen", "epochs_equivalent", "best_val_loss")},
        "loss_curve": curve,
        "evaluation": {label: e["headline"] for label, e in evaluation.items()},
        "answers_identical_to_float32": {label: e["answers_identical_to_float32"] for label, e in evaluation.items()
                                         if "answers_identical_to_float32" in e},
        "quantisation": card.get("quantisation"),
        "files": {k: v for k, v in files.items() if isinstance(v, dict)},
        "engine_parity": parity, "engine_speed": bench, "onnx": card.get("onnx", {}).get("parity"),
    }
    (run / "smoke_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    line = "-" * 100
    t = summary["training"]
    print(line)
    corpus = info["data"].get("corpus") or {}
    note = "FICTIONAL corpus: these numbers say nothing about the released model" if corpus.get("fictional") else ""
    print(f"BRASA SMOKE RUN  {run}   corpus: {corpus.get('name', info['data']['dir'])}   {note}")
    print(line)
    d = info["data"]
    print(f"model      {info['preset']}: {info['model']['params']:,} parameters, {info['model']['n_layer']} layers, "
          f"d_model {info['model']['d_model']}, context {info['model']['block_size']}, "
          f"vocabulary {info['tokenizer']['vocab_size']} (requested {info['tokenizer']['requested']})")
    print(f"data       {d['facts']} facts, {d['topics']} topics; QA records train/val/test = "
          f"{d['splits']['train']['n']}/{d['splits']['val']['n']}/{d['splits']['test']['n']}; "
          f"{info['examples']['examples']} training examples ({info['examples']['open_book']} open-book), "
          f"{info['examples']['tokens']:,} tokens")
    print(f"training   {t['steps']} steps on {t['device']} (compile={t['compile']}, amp={t['amp']}), lr {t['lr']}: "
          f"{t['train_seconds']}s, {t['tokens_per_second']:,} tokens/s, {t['epochs_equivalent']} epochs"
          + (f"; interrupted and resumed at step {', '.join(map(str, curve['resumed_from_steps']))}"
             if curve["resumed_from_steps"] else ""))
    print("loss       " + "  ".join(f"step {p['step']}: {p['loss']}" for p in curve["train"]))
    shown = curve["val"][:: max(1, len(curve["val"]) // 6)]
    best = curve["best_val"]
    print("val loss   " + "  ".join(f"step {v['step']}: {v['val_loss']}" for v in shown)
          + (f"   (best {best['val_loss']} at step {best['step']})" if best else ""))

    print(line)
    print(f"{'weights':8s} {'mode':11s} {'questions':24s} {'n':>4s} {'EM':>6s} {'F1':>6s} {'EM in-scope':>11s} "
          f"{'key spans':>9s} {'refusal P':>9s} {'refusal R':>9s} {'language':>8s} {'bad numbers':>11s}")
    names = {"seen": "seen (training)", "val": "held-out (validation)", "test": "held-out (test)"}
    for label, head in summary["evaluation"].items():
        for mode, per_split in head.items():
            for split, m in per_split.items():
                print(f"{label:8s} {mode:11s} {names.get(split, split):24s} {m['n']:>4d} {pct(m['em'])} "
                      f"{pct(m['f1'])} {pct(m['in_scope_em']):>11s} {pct(m['key_span_recall']):>9s} "
                      f"{pct(m['refusal_precision']):>9s} {pct(m['refusal_recall']):>9s} "
                      f"{pct(m['language_consistency']):>8s} {pct(m['unsupported_number_rate']):>11s}")
    test = summary["evaluation"]["float32"]["open_book"].get("test", {})
    if "em_when_gold_retrieved" in test:
        print(f"open-book test: gold facts retrieved for {pct(test['retrieval_hit']).strip()} of the in-scope "
              f"questions; EM {pct(test['em_when_gold_retrieved']).strip()} when retrieved, "
              f"{pct(test['em_when_gold_missed']).strip()} when missed")
    for label, same in summary["answers_identical_to_float32"].items():
        print(f"{label}: {same} answers identical to float32")

    print(line)
    for name, f in summary["files"].items():
        brotli = "-" if f.get("brotli") is None else f"{f['brotli']:,}"
        print(f"{name:18s} {f['bytes']:>11,} bytes   gzip {f['gzip']:>11,}   brotli {brotli:>11s}")
    print(line)
    if parity is None:
        print("engine parity: not run (node is not installed)")
    else:
        for name, r in parity["engine"].items():
            print(f"engine {name:13s} {'ok  ' if r['ok'] else 'FAIL'} max |logit diff| {r['maxAbsLogitDiff']:.1e} "
                  f"(max |logit| {r['maxAbsLogit']}), greedy {r['greedyIdentical']}, "
                  f"attention {r['maxAbsAttentionDiff']:.1e}, sampling {r['samplingIdentical']}")
        text = parity.get("text") or {}
        if text:
            print(f"tokenizer {text['tokenizer']['cases']} cases / {text['tokenizer']['mismatches']} mismatches;  "
                  f"retrieval {text['retrieval']['cases']} queries / {text['retrieval']['rankingMismatches']} ranking, "
                  f"{text['retrieval']['scoreMismatches']} score mismatches;  "
                  f"prompt {text['prompt']['cases']} cases / {text['prompt']['mismatches']} mismatches")
        print("ENGINE PARITY " + ("OK" if parity["ok"] else "FAILED"))
    for b in bench:
        print(f"engine speed ({b['kernel']}, {b['node']}): {b['decodeTokPerSec']} tokens/s decoding, "
              f"first token after {b['ttftMs']} ms, {b['memoryMB']} MB")
    if summary["onnx"]:
        print(f"onnx parity: {json.dumps(summary['onnx'])}")
    print(line)
    print(f"summary written to {run / 'smoke_summary.json'}")


if __name__ == "__main__":
    main()
