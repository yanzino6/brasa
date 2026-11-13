"""Export: the TGPT container, quantisation, and the proof that the JavaScript engine computes what
PyTorch computes. The tests that need Node are skipped cleanly when it is not installed."""

from __future__ import annotations

import json
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch
from conftest import ENGINE, KERNEL, ML_ROOT, REHEARSAL, TINY, small_dataset

from brasa.config import DataConfig, GPTConfig, PromptConfig
from brasa.evaluate import MODES
from brasa.export import (
    SAMPLING_CASES,
    dequantized_model,
    export_model,
    quantize,
    random_prompts,
    read_tgpt,
    size_report,
    test_vectors,
    write_tgpt,
)
from brasa.export import main as export_main
from brasa.fixtures import build_parity, random_model
from brasa.retrieval import Bm25Retriever
from brasa.tokenizer import BPETokenizer
from brasa.train import TrainConfig, train

test_vectors.__test__ = False  # a library function whose name starts with "test", not a test


def run_parity(node: str, export_dir, wasm: bool = True) -> tuple[int, dict]:
    cmd = [node, str(ENGINE / "tools" / "parity.mjs"), str(export_dir), "--quiet"]
    if wasm:
        cmd += ["--wasm", str(KERNEL)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert out.stdout.strip(), f"parity.mjs printed nothing; stderr: {out.stderr[-2000:]}"
    return out.returncode, json.loads(out.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------------------- quantisation
def test_int8_uses_one_scale_per_row():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(6, 40)).astype(np.float32) * np.array([[0.01], [0.1], [1], [10], [100], [0]], np.float32)
    stored, dtype, scales, rebuilt = quantize(w, "i8")
    assert dtype == "i8" and stored.dtype == np.int8 and scales.shape == (6,)
    assert np.abs(stored).max() == 127 and stored.min() >= -127, "the largest value of a row maps to +-127"
    np.testing.assert_allclose(scales[:5], np.abs(w[:5]).max(axis=1) / 127, rtol=1e-6)
    assert scales[5] == 1.0 and not rebuilt[5].any(), "an all-zero row stays zero and does not divide by zero"
    # rounding error is at most half a quantisation step of ITS row, however different the rows are
    assert (np.abs(rebuilt - w) <= scales[:, None] / 2 + 1e-7).all()
    # a single global scale would wipe out the small rows; per-row scales keep their relative error small
    rel = np.abs(rebuilt[:5] - w[:5]).max(axis=1) / np.abs(w[:5]).max(axis=1)
    assert rel.max() < 0.005


def test_float16_and_gains():
    w = np.random.default_rng(1).normal(size=(4, 9)).astype(np.float32)
    stored, dtype, scales, rebuilt = quantize(w, "f16")
    assert dtype == "f16" and stored.dtype == np.dtype("<f2") and scales is None
    np.testing.assert_array_equal(rebuilt, w.astype(np.float16).astype(np.float32))
    gain = np.linspace(0.5, 1.5, 7, dtype=np.float32)
    for fmt in ("f16", "i8"):
        stored, dtype, scales, rebuilt = quantize(gain, fmt)
        assert dtype == "f32" and scales is None, "1-D tensors (RMSNorm gains) always stay float32"
        np.testing.assert_array_equal(rebuilt, gain)
    with pytest.raises(ValueError):
        quantize(w, "i4")


# --------------------------------------------------------------------------------------- container
def test_container_layout(tiny_model, tok, tmp_path):
    size = write_tgpt(tiny_model, "i8", tmp_path / "m.bin", tok, PromptConfig(max_facts=3), name="unit")
    raw = (tmp_path / "m.bin").read_bytes()
    assert len(raw) == size and raw[:4] == b"TGPT"
    version, head_len = struct.unpack("<II", raw[4:12])
    assert version == 1 and head_len % 4 == 0
    header = json.loads(raw[12 : 12 + head_len])
    assert header["format"] == "i8" and header["name"] == "unit" and header["prompt"]["max_facts"] == 3
    assert header["config"]["params"] == tiny_model.num_params()
    assert header["tokenizer"]["merges"] == [list(m) for m in tok.merges]

    names = [t["name"] for t in header["tensors"]]
    assert "tok_emb.weight" in names and "lm_head.weight" not in names, "the tied head is stored once"
    assert all(t["offset"] % 4 == 0 for t in header["tensors"]), "tensors are 4-byte aligned"
    end = 0
    for t in header["tensors"]:  # no overlap, no gap larger than the alignment padding
        n = int(np.prod(t["shape"]))
        assert 0 <= t["offset"] - end < 4
        end = t["offset"] + (n if t["dtype"] == "i8" else 4 * n)
        if t["dtype"] == "i8":
            assert len(t["shape"]) == 2 and 0 <= t["scalesOffset"] - end < 4
            end = t["scalesOffset"] + 4 * t["shape"][0]
        else:
            assert t["dtype"] == "f32" and len(t["shape"]) == 1
    assert len(raw) - 12 - head_len - end in range(4)
    assert size < 1.2 * tiny_model.num_params(), "about one byte per parameter"


@pytest.mark.parametrize("fmt", ["f32", "f16", "i8"])
def test_container_round_trip(tiny_model, fmt, tmp_path):
    write_tgpt(tiny_model, fmt, tmp_path / "m.bin")
    header, loaded = read_tgpt(tmp_path / "m.bin")
    assert header["tokenizer"] is None and not loaded.training
    assert loaded.lm_head.weight is loaded.tok_emb.weight, "the head is tied again after loading"
    expected, _ = dequantized_model(tiny_model, fmt)
    for (name, a), b in zip(loaded.state_dict().items(), expected.state_dict().values(), strict=True):
        assert torch.equal(a, b), f"{name}: the file does not hold what dequantized_model() predicts"
    if fmt == "f32":
        for (name, a), b in zip(loaded.state_dict().items(), tiny_model.state_dict().values(), strict=True):
            assert torch.equal(a, b), name

    x = torch.randint(0, 300, (1, 20))
    drift = (loaded(x)[0] - tiny_model(x)[0]).abs().max().item()
    assert drift <= {"f32": 0.0, "f16": 0.02, "i8": 0.5}[fmt]


def test_quantisation_error_is_ordered(tiny_model):
    errors = {fmt: dequantized_model(tiny_model, fmt)[1] for fmt in ("f32", "f16", "i8")}
    assert errors["f32"] == 0.0 < errors["f16"] < errors["i8"] < 0.02


def test_bad_files_are_rejected(tiny_model, tmp_path):
    (tmp_path / "junk.bin").write_bytes(b"NOPE" + bytes(64))
    with pytest.raises(ValueError, match="not a TGPT"):
        read_tgpt(tmp_path / "junk.bin")
    write_tgpt(tiny_model, "f16", tmp_path / "m.bin")
    raw = bytearray((tmp_path / "m.bin").read_bytes())
    raw[4:8] = struct.pack("<I", 99)
    (tmp_path / "future.bin").write_bytes(raw)
    with pytest.raises(ValueError, match="version"):
        read_tgpt(tmp_path / "future.bin")


# --------------------------------------------------------------------------------------- test vectors
def test_reference_vectors(tiny_model):
    prompts = random_prompts(300, lengths=(1, 5, 17, 60))
    vec = test_vectors(tiny_model, prompts, n_new=10, full_logits=2)
    assert [len(c["greedy"]) for c in vec["cases"]] == [10, 10, 10, 4], "generation stops at the context length"
    assert ["last_logits" in c for c in vec["cases"]] == [True, True, False, False]
    assert len(vec["cases"][0]["last_logits"]) == 300

    ins = vec["inspect"]
    cfg = tiny_model.cfg
    assert len(ins["attention"]) == cfg.n_layer and len(ins["attention"][0]) == cfg.n_head
    assert all(abs(sum(head) - 1) < 1e-4 for layer in ins["attention"] for head in layer), "rows are distributions"
    assert len(ins["resid_norms"]) == cfg.n_layer + 1
    assert ins["top"][0]["id"] == int(np.argmax(vec["cases"][0]["last_logits"]))
    assert [t["prob"] for t in ins["top"]] == sorted((t["prob"] for t in ins["top"]), reverse=True)

    assert [s["params"] for s in vec["sampling"]] == SAMPLING_CASES
    assert len(vec["sampling"][0]["ids"]) == 300, "temperature only: nothing is removed"
    assert len(vec["sampling"][1]["ids"]) == 40, "top-k 40"
    assert all(abs(sum(s["probs"]) - 1) < 1e-3 for s in vec["sampling"])


def test_size_report(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"k": ["value"] * 400}), encoding="utf-8")
    report = size_report([tmp_path / "a.json"])["a.json"]
    assert report["bytes"] > report["gzip"] > 0 and len(report["sha256"]) == 64
    assert report["brotli"] is None or 0 < report["brotli"] <= report["gzip"]


# --------------------------------------------------------------------------------------- committed fixture
def test_committed_fixture_is_what_python_produces_today(dataset):
    """tests/fixtures/tiny is what the JavaScript tests read. If the reference code changes, the fixture
    must be regenerated (uv run python scripts/make_fixtures.py); this test says so."""
    sys.path.insert(0, str(ML_ROOT / "scripts"))
    try:
        import make_fixtures
    finally:
        sys.path.pop(0)

    from brasa.encoder import embed, fact_ids, question_ids

    tok = BPETokenizer.load(TINY / "tokenizer.json")
    retriever = Bm25Retriever.load(TINY / "retrieval.json")
    committed = json.loads((TINY / "parity.json").read_text(encoding="utf-8"))
    # the fixture index carries fact vectors computed with the int8 weights: the same weights embed the questions
    _, shipped = read_tgpt(TINY / "model.i8.bin")
    pad = tok.special_ids["<|pad|>"]
    assert retriever.dense is not None and retriever.index["dense"]["weights"] == "i8"

    def embedder(question: str):
        with torch.no_grad():
            return embed(shipped, [question_ids(tok, question)], pad, "cpu", retriever.dense["pooling"])[0].numpy()

    rebuilt = build_parity(tok, retriever, dataset, make_fixtures.MODEL["block_size"], make_fixtures.PROMPT,
                           embedder=embedder)
    rebuilt, dense, committed_dense = json.loads(json.dumps(rebuilt)), None, committed.pop("dense")
    dense = rebuilt.pop("dense")
    assert rebuilt == committed, "parity.json is stale: regenerate the fixtures"
    # the fused searches depend on a forward pass, so they are compared with a tolerance instead of exactly
    assert [c["q"] for c in dense] == [c["q"] for c in committed_dense] and len(dense) >= 20
    for old, new in zip(committed_dense, dense, strict=True):
        np.testing.assert_allclose(new["vector"], old["vector"], atol=1e-5)
        for a, b in zip(old["searches"], new["searches"], strict=True):
            assert [h["doc"] for h in a["hits"]] == [h["doc"] for h in b["hits"]]
            np.testing.assert_allclose([h["score"] for h in a["hits"]], [h["score"] for h in b["hits"]], atol=1e-4)
    with torch.no_grad():
        vectors = embed(shipped, [fact_ids(tok, d["text"]) for d in retriever.docs], pad, "cpu", "mean").numpy()
    assert retriever.with_dense(vectors, 1.0).index["dense"]["vectors"] == retriever.index["dense"]["vectors"], \
        "the fact vectors of the fixture are what the int8 weights produce"

    for fmt in ("f16", "i8"):
        header, model = read_tgpt(TINY / f"model.{fmt}.bin")
        assert header["tokenizer"] == tok.to_json() and header["prompt"] == make_fixtures.PROMPT.to_dict()
        vec = json.loads((TINY / f"testvec.{fmt}.json").read_text(encoding="utf-8"))
        for case in vec["embed"]:
            with torch.no_grad():
                v = embed(model, [case["tokens"]], pad, "cpu", case["pooling"])[0]
            np.testing.assert_allclose(v.numpy(), case["vector"], atol=1e-5)
            assert abs(float(v.norm()) - 1) < 1e-5
        assert {c["pooling"] for c in vec["embed"]} == {"mean", "last"}
        again = test_vectors(model, [c["prompt"] for c in vec["cases"]])
        for old, new in zip(vec["cases"], again["cases"], strict=True):
            assert old["greedy"] == new["greedy"]
            if "last_logits" in old:
                np.testing.assert_allclose(new["last_logits"], old["last_logits"], atol=2e-4)
        np.testing.assert_allclose(again["inspect"]["attention"], vec["inspect"]["attention"], atol=1e-5)


# --------------------------------------------------------------------------------------- Node parity
def test_export_then_node_engine_parity(node, tok, retriever, dataset, tmp_path):
    """A fresh export of a model the JavaScript side has never seen, run through the engine with both
    kernels: logits, greedy continuations, attention, residual norms and sampling filters must match."""
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=96, n_layer=3, n_head=2, d_model=64)
    model = random_model(cfg, seed=7)
    prompt_cfg = PromptConfig(max_question_tokens=20, answer_reserve=12, max_facts=2)
    prompts = random_prompts(tok.vocab_size, lengths=(1, 2, 9, 33, 80), seed=3)
    manifest = export_model(model, tok, tmp_path, prompts, None, prompt_cfg, "unit", ("f32", "f16", "i8"),
                            retriever, dataset)
    assert set(manifest["files"]) >= {"model.f32.bin", "model.f16.bin", "model.i8.bin", "tokenizer.json",
                                      "retrieval.json", "parity.json", "testvec.i8.json"}
    assert manifest["quantisation"]["f32"]["max_abs_logit_drift_vs_float32"] == 0.0

    code, report = run_parity(node, tmp_path)
    assert code == 0 and report["ok"], report
    kernels = ("js", "wasm-simd")
    assert set(report["engine"]) == {f"{fmt}.{kernel}" for fmt in ("f32", "f16", "i8") for kernel in kernels}
    for name, r in report["engine"].items():
        assert r["ok"], f"{name}: {r}"
        assert r["maxAbsLogitDiff"] < 1e-4 * max(1.0, r["maxAbsLogit"]), f"{name}: {r}"
        assert r["greedyIdentical"] == "5/5"
        assert r["samplingIdentical"] == f"{len(SAMPLING_CASES)}/{len(SAMPLING_CASES)}"
        assert r["attentionShape"] == [3, 2, len(prompts[0])]
    text = report["text"]
    assert text["tokenizer"]["mismatches"] == 0 and text["prompt"]["mismatches"] == 0
    assert text["retrieval"]["rankingMismatches"] == 0 and text["retrieval"]["scoreMismatches"] == 0
    assert "dense" not in text and all("denseSearches" not in r for r in report["engine"].values()), \
        "a plain export has no fact vectors"
    assert all(r["maxAbsEmbedDiff"] < 1e-5 for r in report["engine"].values()), "but the engine can still embed"

    # the same model exported as its own retriever: fact vectors in the index, fused searches in the fixtures
    dense_dir = tmp_path / "dense"
    manifest = export_model(model, tok, dense_dir, prompts, None, prompt_cfg, "unit", ("f16", "i8"), retriever,
                            dataset, dense={"weight": 0.5, "pooling": "last"})
    assert manifest["dense_retrieval"] == {"version": 1, "dim": 64, "pooling": "last", "weight": 0.5, "max_tokens": 128,
                                           "scale": manifest["dense_retrieval"]["scale"], "weights": "i8"}
    index = json.loads((dense_dir / "retrieval.json").read_text(encoding="utf-8"))
    assert index["dense"]["dim"] == 64 and index["docs"] == retriever.index["docs"]
    assert "dense" not in retriever.index, "the retriever that was passed in is not changed"
    code, report = run_parity(node, dense_dir)
    assert code == 0 and report["ok"], report
    dense = report["text"]["dense"]
    assert dense["searches"] >= 40 and dense["rankingMismatches"] == 0 and dense["scoreMismatches"] == 0
    for kernel in kernels:   # end to end with the engine's own question vector, for the weights that made the facts'
        r = report["engine"][f"i8.{kernel}"]
        assert r["denseScoreMismatches"] == 0 and r["maxDenseScoreDiff"] < 1e-4, r
        assert "denseSearches" not in report["engine"][f"f16.{kernel}"]


def test_node_parity_detects_a_wrong_weight(node, tmp_path):
    """The parity check must be able to fail: corrupt one stored value and it has to notice."""
    model = random_model(GPTConfig(vocab_size=300, block_size=64, n_layer=2, n_head=4, d_model=48), seed=0)
    tok = BPETokenizer([])  # bytes and special tokens only
    export_model(model, tok, tmp_path, random_prompts(256, lengths=(4, 12)), None, PromptConfig(answer_reserve=8),
                 formats=("f16",))
    code, report = run_parity(node, tmp_path, wasm=False)
    assert code == 0 and report["ok"]

    path = tmp_path / "model.f16.bin"
    raw = bytearray(path.read_bytes())
    head_len = struct.unpack("<I", raw[8:12])[0]
    header = json.loads(raw[12 : 12 + head_len])
    proj = next(t for t in header["tensors"] if t["name"] == "blocks.0.attn.proj.weight")
    at = 12 + head_len + proj["offset"]
    raw[at : at + 2 * 48] = np.full(48, 3.0, "<f2").tobytes()  # overwrite the first output row
    path.write_bytes(raw)
    code, report = run_parity(node, tmp_path, wasm=False)
    assert code == 1 and not report["ok"] and not report["engine"]["f16.js"]["ok"]


def test_train_evaluate_export_end_to_end(tmp_path, request):
    """The whole pipeline on a toy corpus through the command-line entry points: train for a few steps,
    export, evaluate the quantised weights, write the model card; then Node parity if Node is installed."""
    data = small_dataset(tmp_path / "data")
    run = tmp_path / "run"
    prompt = PromptConfig(max_question_tokens=32, answer_reserve=20, max_facts=3)
    train(TrainConfig(preset="brasa-4m", data_dir=str(data), out=str(run), steps=6, batch_tokens=512, lr=3e-3,
                      warmup=2, dropout=0.0, vocab_size=320, block_size=128, device="cpu", compile="off", amp="off",
                      log_every=3, eval_every=6, ckpt_every=6, data=DataConfig(variants=3, prompt=prompt)))
    export_main(["--run", str(run), "--device", "cpu", "--no-brotli"])

    out = run / "export"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "brasa-4m" and manifest["default_model"] == "model.i8.bin"
    assert manifest["config"]["block_size"] == 128 and manifest["prompt"] == prompt.to_dict()
    assert manifest["special_tokens"]["<|a|>"] == manifest["config"]["vocab_size"] - 1
    files = manifest["files"]
    assert files["model.i8.bin"]["bytes"] < 0.55 * files["model.f16.bin"]["bytes"], "int8 is half of float16"
    assert all((out / name).stat().st_size == f["bytes"] for name, f in files.items())

    card = json.loads((run / "model-card.json").read_text(encoding="utf-8"))
    assert set(card["evaluation"]) == {"float16", "int8"}, "the quantised weights are evaluated, not assumed"
    for label in ("float16", "int8"):
        e = card["evaluation"][label]
        assert set(e["headline"]) == set(MODES) and "test" in e["headline"]["open_book"]
        assert e["answers_identical_to_float32"].endswith(f"/{len(MODES) * (12 + 13)}")
    assert card["files"]["model.i8.bin"]["sha256"] == files["model.i8.bin"]["sha256"]
    assert set(card["quantisation"]) == {"f16", "i8"}

    header, loaded = read_tgpt(out / "model.i8.bin")
    assert header["config"]["params"] == loaded.num_params() and header["tokenizer"]["type"] == "bpe"

    node = request.getfixturevalue("node")  # skips here, after the Python checks, when Node is missing
    code, report = run_parity(node, out)
    assert code == 0 and report["ok"], report
    assert set(report["engine"]) == {"f16.js", "f16.wasm-simd", "i8.js", "i8.wasm-simd"}


def test_a_run_that_also_retrieves_is_exported_with_its_fact_vectors(tmp_path, request):
    """train --retrieval-weight, then the command-line export: the index carries the vectors of the int8
    weights, the quantised weights are evaluated with their own fused ranking, and Node agrees."""
    from brasa.evaluate import main as evaluate_main

    data = small_dataset(tmp_path / "data")
    run = tmp_path / "run"
    prompt = PromptConfig(max_question_tokens=32, answer_reserve=20, max_facts=3)
    summary = train(TrainConfig(preset="brasa-4m", data_dir=str(data), out=str(run), steps=6, batch_tokens=512,
                                lr=3e-3, warmup=2, dropout=0.0, vocab_size=320, block_size=128, device="cpu",
                                compile="off", amp="off", log_every=3, eval_every=6, ckpt_every=6, index="fields",
                                retrieval_weight=1.0, retrieval_batch=8,
                                data=DataConfig(variants=3, prompt=prompt)))
    weight = summary["retrieval"]["weight"]
    evaluate_main(["--run", str(run), "--device", "cpu", "--splits", "test",
                   "--modes", "open_book_retrieved,first_hit"])
    export_main(["--run", str(run), "--device", "cpu", "--no-brotli", "--eval-splits", "test",
                 "--eval-modes", "open_book_retrieved"])

    out = run / "export"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dense_retrieval"]["weight"] == weight and manifest["dense_retrieval"]["weights"] == "i8"
    index = json.loads((out / "retrieval.json").read_text(encoding="utf-8"))
    assert index["dense"]["dim"] == manifest["config"]["d_model"] and index["dense"]["pooling"] == "mean"
    kept = json.loads((run / "retrieval.json").read_text(encoding="utf-8"))
    assert "dense" not in kept, "the run keeps its BM25 index"

    card = json.loads((run / "model-card.json").read_text(encoding="utf-8"))
    assert card["evaluation"]["float32"]["decoding"]["retriever"] == {"kind": "bm25+own", "weight": weight}
    assert set(card["evaluation"]["int8"]["headline"]) == {"open_book_retrieved"}

    export_main(["--run", str(run), "--out", str(tmp_path / "lexical"), "--device", "cpu", "--no-brotli", "--no-eval",
                 "--retriever", "lexical"])
    assert "dense" not in json.loads((tmp_path / "lexical" / "retrieval.json").read_text(encoding="utf-8"))

    node = request.getfixturevalue("node")
    code, report = run_parity(node, out)
    assert code == 0 and report["ok"], report
    assert report["text"]["dense"]["scoreMismatches"] == 0 and report["engine"]["i8.js"]["denseScoreMismatches"] == 0


def test_node_test_suite(node):
    """The engine's own tests (web/engine/test/*.test.mjs) pass on the committed fixture."""
    files = sorted(str(p) for p in (ENGINE / "test").glob("*.test.mjs"))
    assert len(files) >= 5
    out = subprocess.run([node, "--test", "--test-timeout=120000", *files], cwd=ML_ROOT / "web",
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]


def test_the_session_in_node_gives_the_answers_python_gives(node, tmp_path):
    """scripts/compare_answers.py on the committed fixture: the whole chain (retrieve, embed the question,
    build the prompt, constrained decoding with the stored prior) through the browser session and in Python."""
    out = tmp_path / "answers_parity.json"
    done = subprocess.run([sys.executable, str(ML_ROOT / "scripts" / "compare_answers.py"), str(TINY),
                           "--data-dir", str(REHEARSAL), "--n", "16", "--out", str(out)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
    report = json.loads(out.read_text())
    assert report["questions"] == 16 and report["identical_answers"] == 16 and report["identical_retrieved_facts"] == 16
    assert report["decoding"] == {"constraint": "retrieved", "prior": {"kind": "score", "weight": 6, "fixed": 0},
                                  "retriever": "bm25+dense"}
    assert report["differences"] == []

    # the tool behind it, with other decodings
    questions = tmp_path / "q.txt"
    questions.write_text("Onde o Téo mora?\nWhat is his job?\n", encoding="utf-8")
    for flags, expect in ((["--no-prior"], {"constraint": "retrieved", "prior": None}),
                          (["--free", "--lexical"], {"constraint": None, "retriever": "bm25"}),
                          (["--closed"], {"mode": "closed", "constraint": "kb", "facts": []})):
        asked = subprocess.run([node, str(ENGINE / "tools" / "ask.mjs"), str(TINY), "--questions", str(questions),
                                "--wasm", str(KERNEL), *flags], capture_output=True, text=True, timeout=120)
        assert asked.returncode == 0, asked.stderr[-2000:]
        rows = [json.loads(line) for line in asked.stdout.splitlines()]
        assert [r["question"] for r in rows] == ["Onde o Téo mora?", "What is his job?"]
        assert all(r[k] == v for r in rows for k, v in expect.items()), rows


# --------------------------------------------------------------------------------------- ONNX (optional)
def test_onnx_graphs_reproduce_pytorch(tiny_model, tmp_path):
    """The interoperability export, checked with onnxruntime. Skipped without `uv sync --extra onnx`."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from brasa.export_onnx import export_onnx

    report = export_onnx(tiny_model, tmp_path, random_prompts(300, lengths=(1, 6, 30)))
    assert set(report["graphs"]) == {"brasa.onnx", "brasa.cached.onnx"} == set(report["files"])
    parity = report["parity"]
    assert parity["ok"] and max(parity["max_abs_logit_diff"].values()) < 1e-4
    assert parity["greedy_identical"] == {"full": "3/3", "cached": "3/3"}
    assert json.loads((tmp_path / "onnx_report.json").read_text())["parity"] == parity

    import onnxruntime as ort

    session = ort.InferenceSession(str(tmp_path / "brasa.onnx"), providers=["CPUExecutionProvider"])
    ids = torch.randint(0, 300, (3, 11))  # a batch size and a length the exporter never saw
    logits = session.run(None, {"input_ids": ids.numpy()})[0]
    np.testing.assert_allclose(logits, tiny_model(ids)[0].detach().numpy(), atol=1e-4)
