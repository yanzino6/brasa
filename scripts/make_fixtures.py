"""Regenerate tests/fixtures/tiny: a tiny random model exported the normal way.

    uv run python scripts/make_fixtures.py

The directory is a complete export (model.f16.bin, model.i8.bin, tokenizer.json, retrieval.json,
testvec.*.json, parity.json, manifest.json) small enough to commit (about 0.6 MB). The JavaScript tests in
web/engine/test read it, so they run without Python; tests/test_fixtures.py checks that the committed
files still match what the Python code produces.

The weights are random: this fixture is about numerical agreement between PyTorch and the JavaScript
engine, not about answering questions.
"""

from __future__ import annotations

from pathlib import Path

from brasa.config import GPTConfig, PromptConfig
from brasa.data import load_dataset, tokenizer_texts
from brasa.export import default_prompts, export_model
from brasa.fixtures import random_model
from brasa.retrieval import Bm25Retriever
from brasa.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "tiny"

VOCAB_SIZE = 512
MODEL = dict(block_size=128, n_layer=2, n_head=4, d_model=48)
PROMPT = PromptConfig(max_question_tokens=24, answer_reserve=16, max_facts=3)


def build(out: Path = OUT) -> dict:
    dataset = load_dataset(ROOT / "data" / "rehearsal")
    tok = BPETokenizer.train(tokenizer_texts(dataset), VOCAB_SIZE)
    retriever = Bm25Retriever.from_dataset(dataset, expand_with_questions=False)  # keeps the index small
    model = random_model(GPTConfig(vocab_size=tok.vocab_size, **MODEL), seed=0)
    prompts, texts = default_prompts(tok, dataset, retriever, model.cfg.block_size, PROMPT, n=6)
    return export_model(model, tok, out, prompts, texts, PROMPT, "brasa-tiny", ("f16", "i8"), retriever, dataset)


if __name__ == "__main__":
    manifest = build()
    for name, f in manifest["files"].items():
        print(f"{name:18s} {f['bytes']:>9,} bytes")
    print(f"wrote {OUT}")
