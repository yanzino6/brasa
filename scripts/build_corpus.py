#!/usr/bin/env python3
"""Turn the question sources of the real knowledge base into the data contract of brasa.data.

    uv run python scripts/build_corpus.py --data-dir data/yan            # writes data/yan/qa/*.jsonl
    uv run python scripts/build_corpus.py --data-dir data/yan --k 4 --out runs/yan-k4    # ablation: 4 paraphrases

Inputs (see scripts/write_questions.py, which produces the qa-src files with an LLM teacher):

    kb/facts.jsonl, kb/topics.json     the curated public knowledge base
    qa-src/facts.jsonl                 {"f": fact id, "pt": [14 questions], "en": [14 questions]}
    qa-src/multi.jsonl                 {"id", "facts": [ids], "pt": [8], "en": [8]}
    qa-src/chitchat.jsonl, oos.jsonl   {"cat", "pt": {"train","val","test"}, "en": {...}}
    qa-src/canonical.json              the fixed answers for chit-chat and for declined questions

Rules:

* The answer to a fact question is the fact sentence itself, in the language of the question; the answer to
  a multi-fact question is its fact sentences joined in order.
* Per fact and language the first 11 questions are training paraphrases (the first --k are used), the 12th
  is the validation question and the last two are test questions. Their phrasings never occur in training:
  any held-out question that also exists as a training question is dropped and counted.
* Every training question also appears once wrapped in a greeting or a polite ending.
* A question that maps to two different answers is ambiguous: it is removed everywhere and counted.
* key_spans (the numbers and names an answer must contain) are extracted from the answer text.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

N_TRAIN, VAL_AT = 11, 11          # positions 0..10 train, 11 validation, 12..13 test
MULTI_TRAIN, MULTI_VAL_AT = 6, 6  # positions 0..5 train, 6 validation, 7 test

PREFIXES = {
    "pt": ["Oi, ", "Olá! ", "Oi, tudo bem? ", "Bom dia. ", "Boa tarde, ", "E aí, ", "Por favor, ", "Uma dúvida: ",
           "Me diz uma coisa: ", "Queria saber: ", "Rapidinho: ", "Pergunta: "],
    "en": ["Hi, ", "Hello! ", "Hey, ", "Good morning. ", "Quick question: ", "Please, ", "One thing: ",
           "I was wondering: ", "Tell me: ", "Question: ", "Hi there, ", "So, "],
}
SUFFIXES = {
    "pt": [" Obrigado!", " Valeu!", " Pode me dizer?", " Por favor.", " Sabe me dizer?", " Obrigada."],
    "en": [" Thanks!", " Thank you.", " Can you tell me?", " Please.", " Do you know?", " Cheers."],
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s).split())


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.exists() else []


NUM = re.compile(r"\d+(?:[.,]\d+)*%?")
NAME = re.compile(r"\b(?:[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Z]{2,}[A-Za-z0-9-]*|[A-Za-z]+[0-9]+[A-Za-z0-9-]*|[a-z]+\.[a-z]{2,4})\b")


def key_spans(answer: str, limit: int = 5) -> list[str]:
    """Numbers and code-like names (PyTorch, UFES, VGG16, seer.app) that the answer must reproduce."""
    out: list[str] = []
    for m in list(NUM.finditer(answer)) + list(NAME.finditer(answer)):
        s = m.group(0).rstrip(".,")
        if s and s not in out and s.lower() not in ("yan",):
            out.append(s)
    return out[:limit]


def wrap(q: str, lang: str, rng: random.Random) -> str:
    if rng.random() < 0.6:
        pre = rng.choice(PREFIXES[lang])
        return pre + (q if pre.rstrip().endswith((":", ".", "!", "?")) else q[:1].lower() + q[1:])
    return q + rng.choice(SUFFIXES[lang])


def build(data_dir: Path, k: int, seed: int) -> tuple[dict[str, list[dict]], dict]:
    facts = {f["id"]: f for f in read_jsonl(data_dir / "kb/facts.jsonl")}
    src = data_dir / "qa-src"
    canon = json.loads((src / "canonical.json").read_text(encoding="utf-8"))
    rows: list[dict] = []  # file, id, lang, q, a, facts, kind, split

    def add(file: str, rid: str, lang: str, q: str, a: str, fids: list[str], kind: str, split: str) -> None:
        rows.append({"file": file, "id": rid, "lang": lang, "q": q.strip(), "a": a, "facts": fids, "kind": kind,
                     "split": split})

    missing = [fid for fid in facts if fid not in {it["f"] for it in read_jsonl(src / "facts.jsonl")}]
    for it in read_jsonl(src / "facts.jsonl"):
        f = facts.get(it["f"])
        if f is None:
            continue
        kind = "meta" if f["topic"] == "brasa" else "single"
        for lang in ("pt", "en"):
            for n, q in enumerate(it[lang]):
                split = "train" if n < N_TRAIN else "val" if n == VAL_AT else "test"
                if split == "train" and n >= k:
                    continue
                add("facts", f"{f['id']}-{lang}-{n:02d}", lang, q, f[lang], [f["id"]], kind, split)
    for it in read_jsonl(src / "multi.jsonl"):
        if not all(fid in facts for fid in it["facts"]):
            continue
        for lang in ("pt", "en"):
            answer = " ".join(facts[fid][lang] for fid in it["facts"])
            for n, q in enumerate(it[lang]):
                split = "train" if n < MULTI_TRAIN else "val" if n == MULTI_VAL_AT else "test"
                add("multi", f"{it['id']}-{lang}-{n:02d}", lang, q, answer, list(it["facts"]), "multi", split)
    for file, kind in (("chitchat", "chitchat"), ("oos", "oos")):
        for c in read_jsonl(src / f"{file}.jsonl"):
            for lang in ("pt", "en"):
                for split in ("train", "val", "test"):
                    for n, q in enumerate(c[lang].get(split, [])):
                        add(file, f"{c['cat']}-{lang}-{split}-{n:03d}", lang, q, canon[c["cat"]][lang], [], kind, split)

    # one question, one answer: ambiguous questions go; held-out questions never repeat a training phrasing
    answers = defaultdict(set)
    for r in rows:
        answers[(r["lang"], norm(r["q"]))].add(r["a"])
    ambiguous = {key for key, a in answers.items() if len(a) > 1}
    train_keys = {(r["lang"], norm(r["q"])) for r in rows if r["split"] == "train"}
    kept, seen, dropped = [], set(), Counter()
    for r in rows:
        key = (r["lang"], norm(r["q"]))
        if key in ambiguous:
            dropped["ambiguous"] += 1
        elif r["split"] != "train" and key in train_keys:
            dropped["held_out_seen_in_train"] += 1
        elif key in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(key)
            kept.append(r)

    rng = random.Random(seed)
    files: dict[str, list[dict]] = defaultdict(list)
    for r in kept:
        rec = {"id": r["id"], "lang": r["lang"], "q": r["q"], "a": r["a"], "facts": r["facts"], "kind": r["kind"],
               "split": r["split"], "key_spans": key_spans(r["a"]) if r["facts"] else []}
        files[r["file"]].append(rec)
        if r["split"] == "train":
            w = wrap(r["q"], r["lang"], rng)
            if (r["lang"], norm(w)) not in seen:
                seen.add((r["lang"], norm(w)))
                files[r["file"]].append({**rec, "id": r["id"] + "-w", "q": w})
    stats = {
        "facts_in_kb": len(facts), "facts_without_questions": len(missing), "first_missing": missing[:10],
        "k": k, "dropped": dict(dropped),
        "records": {f: len(v) for f, v in files.items()},
        "splits": dict(Counter(r["split"] for v in files.values() for r in v)),
        "by_kind": dict(Counter(r["kind"] for v in files.values() for r in v)),
    }
    return files, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="data/yan")
    ap.add_argument("--out", default=None, help="write a complete data directory here instead of <data-dir>/qa")
    ap.add_argument("--k", type=int, default=N_TRAIN, help="training paraphrases per fact and language (1 to 11)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    files, stats = build(data_dir, max(1, min(N_TRAIN, args.k)), args.seed)
    out = Path(args.out) if args.out else data_dir
    if args.out:
        (out / "kb").mkdir(parents=True, exist_ok=True)
        for name in ("facts.jsonl", "topics.json"):
            shutil.copy(data_dir / "kb" / name, out / "kb" / name)
    qa = out / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    for old in qa.glob("*.jsonl"):
        old.unlink()
    for name, recs in files.items():
        (qa / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    corpus = {
        "name": "yan" if args.k >= N_TRAIN else f"yan-k{args.k}", "fictional": False,
        "description": "Questions about Yan Simmer written by an LLM teacher from a curated public knowledge base; "
                       "answers are the curated facts. Held-out validation and test questions use phrasings that "
                       "never occur in training.",
        "training_paraphrases_per_fact_and_language": min(N_TRAIN, args.k), "build": stats,
    }
    (out / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
