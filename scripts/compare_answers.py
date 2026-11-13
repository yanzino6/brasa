"""End-to-end check of an export: does the browser engine give the answers the Python reference gives?

    uv run python scripts/compare_answers.py runs/final/export --data-dir runs/data-full2 [--n 200] [--split test]
        [--format i8] [--wasm web/wasm/kernel.wasm] [--out answers_parity.json]

Takes n held-out questions (an even stride over the split), asks each of them twice with the SAME exported
weights and retrieval index: through web/engine/session.js in Node (tools/ask.mjs: what the site runs) and
through brasa.sample.answer in Python (the code the evaluation is built on), both with the default decoding
(open-book, constrained to the retrieved facts, the prior stored in the index). Reports how many answers and
how many retrieved fact lists are identical.

The pieces are compared exactly elsewhere (web/engine/tools/parity.mjs: logits, tokenizer, retrieval, trie).
This is the whole chain on real questions; a difference here is a rounding tie between two facts (the engine
computes in float32 with its own summation order), which the report shows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    from brasa.data import load_dataset
    from brasa.encoder import DenseRetriever
    from brasa.sample import answer, load_export

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("export", help="export directory (model.<fmt>.bin, retrieval.json)")
    ap.add_argument("--data-dir", required=True, help="where the questions come from")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--format", default="i8", choices=["i8", "f16"])
    ap.add_argument("--wasm", default=str(ROOT / "web" / "wasm" / "kernel.wasm"))
    ap.add_argument("--out", help="write the report here as JSON")
    args = ap.parse_args()

    records = load_dataset(args.data_dir).split(args.split)
    step = max(1, len(records) // args.n)
    questions = list(dict.fromkeys(r.q for r in records[::step]))[: args.n]

    model, tok, lexical, prompt_cfg = load_export(args.export, args.format)
    retriever = DenseRetriever.from_index(model, tok, lexical) if lexical.dense is not None else lexical
    python = []
    for q in questions:
        out = answer(model, tok, retriever, q, prompt_cfg, constrain="retrieved")
        python.append({"answer": out["answer"], "facts": [h.id for h in out["hits"]]})

    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
        f.write("\n".join(questions) + "\n")
    cmd = ["node", str(ROOT / "web" / "engine" / "tools" / "ask.mjs"), args.export, "--questions", f.name,
           "--format", args.format, "--wasm", args.wasm]
    done = subprocess.run(cmd, capture_output=True, text=True, check=True)
    Path(f.name).unlink()
    engine = [json.loads(line) for line in done.stdout.splitlines() if line.strip()]
    if len(engine) != len(questions):
        raise SystemExit(f"the engine answered {len(engine)} of {len(questions)} questions")

    same_answer = sum(e["answer"] == p["answer"] for e, p in zip(engine, python, strict=True))
    same_facts = sum(e["facts"] == p["facts"] for e, p in zip(engine, python, strict=True))
    differences = [{"question": q, "engine": e["answer"], "python": p["answer"], "same_facts": e["facts"] == p["facts"]}
                   for q, e, p in zip(questions, engine, python, strict=True) if e["answer"] != p["answer"]]
    report = {"export": args.export, "format": args.format, "split": args.split, "questions": len(questions),
              "identical_answers": same_answer, "identical_retrieved_facts": same_facts,
              "decoding": {"constraint": engine[0]["constraint"], "prior": engine[0]["prior"],
                           "retriever": engine[0]["retriever"]},
              "differences": differences[:10]}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
