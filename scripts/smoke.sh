#!/usr/bin/env bash
# The whole pipeline, end to end, in one command and a few minutes:
#
#   scripts/smoke.sh
#
#   1. rebuild the FICTIONAL rehearsal corpus (data/rehearsal) and validate the data contract
#   2. train brasa-4m on it, interrupting half-way on purpose and resuming from the checkpoint
#   3. evaluate the float32 checkpoint (seen questions, held-out paraphrases, unseen out-of-scope questions)
#   4. export (float16 + int8 TGPT files, tokenizer, retrieval index, test vectors) and evaluate the
#      quantised weights
#   5. run the JavaScript engine in Node against the PyTorch test vectors (both kernels) and time it
#   6. answer a few questions with the exported int8 weights
#   7. print a summary (also written to <run>/smoke_summary.json)
#
# Environment variables: RUN (default runs/smoke), STEPS (1200), PRESET (brasa-4m), DATA (data/rehearsal),
# DEVICE (auto), ONNX=1 to also export and verify the ONNX graphs (needs the optional dependencies).
#
# "Téo Arantes" does not exist: the rehearsal corpus only proves that the code works. Nothing measured here
# is a result of the real model.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${RUN:-runs/smoke}"
STEPS="${STEPS:-1200}"
PRESET="${PRESET:-brasa-4m}"
DATA="${DATA:-data/rehearsal}"
DEVICE="${DEVICE:-auto}"
step() { printf '\n==> %s\n' "$*"; }

step "environment"
uv sync --quiet --inexact   # --inexact: keep optional extras (ONNX) if they are installed
uv run python -c "import torch, sys; print('python', sys.version.split()[0], '| torch', torch.__version__, '| mps', torch.backends.mps.is_available(), '| cuda', torch.cuda.is_available())"

step "1/7 rehearsal corpus (fictional) and data contract"
if [ "$DATA" = "data/rehearsal" ]; then uv run python scripts/make_rehearsal.py; fi
uv run python -m brasa.data --data-dir "$DATA" | sed -n 1p
uv run python -m brasa.retrieval --data-dir "$DATA" --eval | tr -d '\n ' | sed 's/^/retrieval: /'; echo

step "2/7 train $PRESET for $STEPS steps (stop at $((STEPS / 2)), then resume)"
rm -rf "$RUN"
TRAIN=(uv run python -m brasa.train --preset "$PRESET" --data-dir "$DATA" --out "$RUN" --steps "$STEPS"
       --device "$DEVICE" --log-every 100 --eval-every 200 --ckpt-every 200)
"${TRAIN[@]}" --stop-at $((STEPS / 2)) | grep --line-buffered -v '"event": "start"'
"${TRAIN[@]}" --resume | grep --line-buffered -v '"event": "start"'

step "3/7 evaluate the float32 checkpoint"
uv run python -m brasa.evaluate --run "$RUN" | tail -1

step "4/7 export and evaluate the quantised weights"
uv run python -m brasa.export --run "$RUN"

step "5/7 JavaScript engine against PyTorch (Node)"
if command -v node >/dev/null 2>&1; then
  node web/engine/tools/parity.mjs "$RUN/export" --wasm web/wasm/kernel.wasm | tee "$RUN/parity.log" | grep -v '^{'
  tail -1 "$RUN/parity.log" > "$RUN/parity_report.json"
  node web/engine/tools/bench.mjs "$RUN/export/model.i8.bin" --wasm web/wasm/kernel.wasm > "$RUN/bench.jsonl"
else
  echo "node is not installed: skipping the engine parity check and the benchmark"
fi

if [ "${ONNX:-0}" = "1" ]; then
  step "ONNX export (interoperability artefact)"
  uv run --extra onnx python -m brasa.export_onnx --run "$RUN" 2>/dev/null | tail -4
fi

step "6/7 a few questions, answered by the exported int8 weights"
uv run python -m brasa.sample --export "$RUN/export" --format i8 --show-context \
  -q "Onde o Téo mora?" -q "What did he build for fraud detection?" -q "qual o email dele" \
  -q "What is the capital of France?"

step "7/7 summary"
uv run python scripts/smoke_summary.py "$RUN"
