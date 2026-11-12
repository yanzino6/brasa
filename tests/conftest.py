"""Shared fixtures. Everything runs on the CPU and takes a few seconds in total."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from brasa.config import GPTConfig
from brasa.data import load_dataset, tokenizer_texts
from brasa.retrieval import Bm25Retriever
from brasa.tokenizer import BPETokenizer

ML_ROOT = Path(__file__).resolve().parent.parent
REHEARSAL = ML_ROOT / "data" / "rehearsal"
TINY = ML_ROOT / "tests" / "fixtures" / "tiny"
ENGINE = ML_ROOT / "web" / "engine"
KERNEL = ML_ROOT / "web" / "wasm" / "kernel.wasm"


@pytest.fixture(scope="session")
def dataset():
    return load_dataset(REHEARSAL)


@pytest.fixture(scope="session")
def tok(dataset):
    """A small tokenizer trained on the rehearsal corpus (the same recipe as the committed fixture)."""
    return BPETokenizer.train(tokenizer_texts(dataset), 512)


@pytest.fixture(scope="session")
def retriever(dataset):
    return Bm25Retriever.from_dataset(dataset)


@pytest.fixture()
def tiny_model():
    from brasa.fixtures import random_model

    return random_model(GPTConfig(vocab_size=300, block_size=64, n_layer=2, n_head=4, d_model=48), seed=0)


@pytest.fixture(scope="session")
def node():
    """Path of the node binary; tests that need it are skipped when it is not installed."""
    path = shutil.which("node")
    if path is None:
        pytest.skip("node is not installed")
    return path


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)


def write_dataset(root: Path, facts: list[dict], qa: list[dict], topics: list[dict] | None = None) -> Path:
    """Write a data directory in the contract format (used to build small or deliberately broken ones)."""
    (root / "kb").mkdir(parents=True, exist_ok=True)
    (root / "qa").mkdir(parents=True, exist_ok=True)
    topics = topics or [{"slug": "t", "type": "person", "title_pt": "Tema", "title_en": "Topic", "aliases": []}]
    (root / "kb" / "topics.json").write_text(json.dumps(topics), encoding="utf-8")
    (root / "kb" / "facts.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts), encoding="utf-8")
    (root / "qa" / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in qa), encoding="utf-8")
    return root


def small_dataset(root: Path) -> Path:
    """Six facts, a few paraphrases each, in both languages: enough to train for a handful of steps."""
    things = [("city", "Lia mora em Vitória.", "Lia lives in Vitória.", "Onde a Lia mora?", "Where does Lia live?"),
              ("job", "Lia é engenheira de dados.", "Lia is a data engineer.", "Qual é o trabalho da Lia?",
               "What is Lia's job?"),
              ("lang", "Ela programa em Python e Go.", "She codes in Python and Go.", "Em que ela programa?",
               "What does she code in?"),
              ("pet", "Ela tem dois gatos.", "She has two cats.", "Ela tem animais?", "Does she have pets?"),
              ("study", "Ela estudou na UFES.", "She studied at UFES.", "Onde ela estudou?", "Where did she study?"),
              ("mail", "O e-mail dela é lia@example.org.", "Her e-mail is lia@example.org.", "Qual é o e-mail dela?",
               "What is her e-mail?")]
    facts, qa = [], []
    for fid, pt, en, q_pt, q_en in things:
        facts.append({"id": fid, "topic": "t", "subtopic": fid, "kind": "statement", "pt": pt, "en": en,
                      "importance": 1, "caveat": None})
        for lang, q, a in (("pt", q_pt, pt), ("en", q_en, en)):
            for n, (prefix, split) in enumerate((("", "train"), ("Oi! ", "train"), ("Por favor: ", "val"),
                                                 ("Me diga: ", "test"))):
                qa.append({"id": f"{fid}-{lang}-{n}", "lang": lang, "q": prefix + q, "a": a, "facts": [fid],
                           "kind": "single", "split": split, "key_spans": []})
    for lang, q, a, split in (("pt", "Qual é a capital da França?", "Só sei falar sobre a Lia.", "train"),
                              ("en", "What is the capital of France?", "I can only talk about Lia.", "train"),
                              ("en", "Who won the World Cup?", "I can only talk about Lia.", "test")):
        qa.append({"id": f"oos-{len(qa)}", "lang": lang, "q": q, "a": a, "facts": [], "kind": "oos", "split": split,
                   "key_spans": []})
    return write_dataset(root, facts, qa)
