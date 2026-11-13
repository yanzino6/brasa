#!/usr/bin/env bash
# The release recipe after training: evaluate, export, check the browser engine against PyTorch, render the
# model card and collect what the site needs.
#
#   scripts/release.sh runs/<run> [runs/final]
#
#   0. choose the decoding prior (how much the retriever's ranking weighs in the answer) on the VALIDATION
#      questions and record it in <run>/run.json; skipped when the run already has one
#   1. evaluate the float32 checkpoint on seen and held-out questions, in every decoding mode
#   2. export (float16 + int8 TGPT files, tokenizer, retrieval index, test vectors, manifest) and evaluate the
#      de-quantised weights with the same protocol
#   3. run the JavaScript engine in Node against the PyTorch test vectors (both kernels), time it, and ask
#      200 held-out questions through the browser session and through Python: the answers have to agree
#   4. render MODEL_CARD.md from model-card.json
#   5. copy the export directory, model-card.json and MODEL_CARD.md into the release directory, with RELEASE.json
#      (run, checkpoint, commands, date) so that a release can always be traced back to its run
#
# Environment variables:
#   CKPT     last | best (default last)
#   SPLITS   default train,val,test. Add "final" ONLY for the released model: that slice is read once.
#   DEVICE   auto | mps | cpu (default auto)
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${1:?usage: scripts/release.sh runs/<run> [release dir]}"
OUT="${2:-runs/final}"
CKPT="${CKPT:-last}"
SPLITS="${SPLITS:-train,val,test}"
DEVICE="${DEVICE:-auto}"
MODES=open_book,closed_book,open_book_constrained,closed_book_constrained,open_book_retrieved,open_book_ranked,first_hit
HELD_OUT="$(echo "$SPLITS" | tr ',' '\n' | grep -v '^train$' | paste -sd, -)"
step() { printf '\n==> %s\n' "$*"; }

TUNE=(uv run python -m brasa.evaluate --run "$RUN" --checkpoint "$CKPT" --tune-prior --device "$DEVICE")
if grep -q '"decoding"' "$RUN/run.json"; then
  step "0/5 decoding prior: already chosen (run.json)"
else
  step "0/5 choose the decoding prior on the validation questions"
  "${TUNE[@]}" | tail -1
fi

step "1/5 evaluate $RUN ($CKPT checkpoint; splits $SPLITS)"
EVALUATE=(uv run python -m brasa.evaluate --run "$RUN" --checkpoint "$CKPT" --splits "$SPLITS" --modes "$MODES" --device "$DEVICE")
"${EVALUATE[@]}" | tail -1

step "2/5 export and evaluate the quantised weights (splits $HELD_OUT)"
EXPORT=(uv run python -m brasa.export --run "$RUN" --checkpoint "$CKPT" --eval-splits "$HELD_OUT" --eval-modes "$MODES" --device "$DEVICE")
"${EXPORT[@]}"

step "3/5 JavaScript engine against PyTorch (Node)"
node web/engine/tools/parity.mjs "$RUN/export" --wasm web/wasm/kernel.wasm | tee "$RUN/parity.log" | grep -v '^{'
tail -1 "$RUN/parity.log" > "$RUN/parity_report.json"
node web/engine/tools/bench.mjs "$RUN/export/model.i8.bin" --wasm web/wasm/kernel.wasm > "$RUN/bench.jsonl"
cat "$RUN/bench.jsonl"
DATA_DIR="$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['train_config']['data_dir'])" "$RUN/run.json")"
uv run python scripts/compare_answers.py "$RUN/export" --data-dir "$DATA_DIR" --n 200 --out "$RUN/answers_parity.json" \
  | grep -E '"(questions|identical_answers|identical_retrieved_facts)"'

step "4/5 model card"
uv run python scripts/render_model_card.py "$RUN/model-card.json" --out "$RUN/MODEL_CARD.md"

step "5/5 collect $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"
cp -R "$RUN/export" "$OUT/export"
cp "$RUN/model-card.json" "$RUN/MODEL_CARD.md" "$RUN/parity_report.json" "$RUN/bench.jsonl" "$RUN/answers_parity.json" "$OUT/"
[ -f "$RUN/predictions.jsonl" ] && cp "$RUN/predictions.jsonl" "$OUT/"
uv run python - "$RUN" "$OUT" "$CKPT" "$SPLITS" "${EVALUATE[*]}" "${EXPORT[*]}" "${TUNE[*]}" <<'PY'
import json, sys, time
from pathlib import Path
run, out, ckpt, splits, evaluate, export, tune = sys.argv[1:8]
info = json.loads((Path(run) / "run.json").read_text(encoding="utf-8"))
cfg = info["train_config"]
train = ["uv run python -m brasa.train", f"--preset {cfg['preset']}", f"--data-dir {cfg['data_dir']}", f"--out {run}",
         f"--steps {cfg['steps']}", f"--index {cfg['index']}", f"--seed {cfg['seed']}"]
train += [f"--init-from {cfg['init_from']}"] if cfg.get("init_from") else []
train += [f"--retrieval-weight {cfg['retrieval_weight']}"] if cfg.get("retrieval_weight") else []
train += [f"--contexts-from {cfg['contexts_from']}"] if cfg.get("contexts_from") else []
pretrain = (info.get("init_from") or {})
release = {
    "run": run, "checkpoint": ckpt, "preset": info["preset"], "params": info["model"]["params"],
    "data_dir": cfg["data_dir"], "evaluated_splits": splits.split(","),
    "final_split_read": "final" in splits.split(","),
    "released_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "decoding_prior": (info.get("decoding") or {}).get("prior"),
    "started_from": ({"run": pretrain.get("dir"), "objective": pretrain.get("objective"), "steps": pretrain.get("steps"),
                      "tokens_seen": pretrain.get("tokens_seen")} if pretrain else None),
    "commands": {"train (main flags; the full configuration is in model-card.json)": " ".join(train),
                 "choose the decoding prior (validation questions)": tune,
                 "evaluate": evaluate, "export": export,
                 "sync to the site (from the repository root)":
                     f"node scripts/sync-brasa.mjs ml/{out}/export ml/{out}/model-card.json"},
}
(Path(out) / "RELEASE.json").write_text(json.dumps(release, indent=1) + "\n", encoding="utf-8")
print(json.dumps(release["commands"], indent=1))
PY
ls -la "$OUT" "$OUT/export"
