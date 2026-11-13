"""scripts/build_corpus.py: the split of the teacher's questions and the knowledge-base size ablation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_corpus.py"
spec = importlib.util.spec_from_file_location("build_corpus", SCRIPT)
build_corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_corpus)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture()
def source(tmp_path) -> Path:
    """Three facts of importance 3, 4 and 5; 14 questions each per language; one two-fact question; one refusal."""
    root = tmp_path / "src"
    facts = [{"id": f"f{i}", "topic": "t", "subtopic": "s", "kind": "fact", "importance": imp,
              "pt": f"O fato {i} usa n8n e tem {i}0 itens.", "en": f"Fact {i} uses n8n and has {i}0 items."}
             for i, imp in ((1, 3), (2, 4), (3, 5))]
    write_jsonl(root / "kb" / "facts.jsonl", facts)
    topics = [{"slug": "t", "title_pt": "T", "title_en": "T", "aliases": []}]
    (root / "kb" / "topics.json").write_text(json.dumps(topics))
    write_jsonl(root / "qa-src" / "facts.jsonl", [
        {"f": f["id"], "pt": [f"pergunta {n} sobre {f['id']}?" for n in range(14)],
         "en": [f"question {n} about {f['id']}?" for n in range(14)]} for f in facts])
    write_jsonl(root / "qa-src" / "multi.jsonl", [
        {"id": "m1", "facts": ["f1", "f2"], "pt": [f"junta {n}?" for n in range(8)],
         "en": [f"both {n}?" for n in range(8)]},
        {"id": "m2", "facts": ["f2", "f3"], "pt": [f"soma {n}?" for n in range(8)],
         "en": [f"sum {n}?" for n in range(8)]}])
    write_jsonl(root / "qa-src" / "oos.jsonl", [
        {"cat": "oos", **{lang: {"train": [f"{lang} off {n}" for n in range(3)], "val": [f"{lang} off v"],
                                 "test": [f"{lang} off t1", f"{lang} off t2"]} for lang in ("pt", "en")}}])
    write_jsonl(root / "qa-src" / "chitchat.jsonl", [
        {"cat": "greeting", **{lang: {"train": [f"{lang} hi {n}" for n in range(3)], "val": [f"{lang} hi v"],
                                      "test": [f"{lang} hi t"], "final": [f"{lang} hi f"]} for lang in ("pt", "en")}}])
    (root / "qa-src" / "canonical.json").write_text(json.dumps(
        {"greeting": {"pt": "Oi!", "en": "Hi!"}, "oos": {"pt": "Não sei.", "en": "I do not know."}}))
    return root


def test_positions_decide_the_split_and_held_out_phrasings_are_new(source):
    files, stats = build_corpus.build(source, 11, seed=1)
    facts = files["facts"]
    by_split = {s: [r for r in facts if r["split"] == s] for s in ("train", "val", "test", "final")}
    assert len(by_split["val"]) == len(by_split["test"]) == len(by_split["final"]) == 3 * 2
    plain = [r for r in by_split["train"] if not r["id"].endswith("-w")]
    assert len(plain) == 3 * 2 * 11 and len(by_split["train"]) == 2 * len(plain), "every training question once wrapped"
    assert {r["id"][-2:] for r in by_split["final"]} == {"13"} and {r["id"][-2:] for r in by_split["test"]} == {"12"}
    train_q = {build_corpus.norm(r["q"]) for r in by_split["train"]}
    assert not any(build_corpus.norm(r["q"]) in train_q for s in ("val", "test", "final") for r in by_split[s])
    assert all(r["a"] in (r["q"], *[f"O fato {i} usa n8n e tem {i}0 itens." for i in (1, 2, 3)],
                          *[f"Fact {i} uses n8n and has {i}0 items." for i in (1, 2, 3)]) for r in facts)
    # the out-of-scope source has no "final" list: every second test question becomes the untouched slice
    oos = files["oos"]
    assert sorted(r["q"] for r in oos if r["split"] == "test") == ["en off t1", "pt off t1"]
    assert sorted(r["q"] for r in oos if r["split"] == "final") == ["en off t2", "pt off t2"]
    assert stats["facts_in_kb"] == 3 and stats["min_importance"] == 0


def test_key_spans_keep_names_with_digits_whole():
    assert build_corpus.key_spans("O fato usa n8n e tem 10 itens.") == ["10", "n8n"]
    assert "8" not in build_corpus.key_spans("Automations run on n8n.")


def test_a_smaller_knowledge_base_drops_the_facts_and_every_question_that_needs_them(source, tmp_path, monkeypatch):
    files, stats = build_corpus.build(source, 11, seed=1, min_importance=4)
    assert stats["facts_in_kb"] == 2 and stats["min_importance"] == 4
    assert {fid for r in files["facts"] for fid in r["facts"]} == {"f2", "f3"}
    assert {r["id"].split("-")[0] for r in files["multi"]} == {"m2"}, "m1 needs the dropped fact f1"
    assert len(files["oos"]) == len(build_corpus.build(source, 11, seed=1)[0]["oos"]), "refusals do not depend on it"

    out = tmp_path / "small"
    monkeypatch.setattr(sys, "argv", ["build_corpus.py", "--data-dir", str(source), "--min-importance", "4",
                                      "--out", str(out)])
    build_corpus.main()
    kept = [json.loads(line) for line in (out / "kb" / "facts.jsonl").read_text().splitlines()]
    assert [f["id"] for f in kept] == ["f2", "f3"]
    assert json.loads((out / "corpus.json").read_text())["name"] == "yan-imp4"
    monkeypatch.setattr(sys, "argv", ["build_corpus.py", "--data-dir", str(source), "--min-importance", "4"])
    with pytest.raises(SystemExit):
        build_corpus.main()  # refuses to shrink the knowledge base in place
