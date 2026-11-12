#!/usr/bin/env python3
"""Write the question corpus for the real knowledge base with an LLM "teacher".

Brasa is trained on (question, answer) pairs whose answers are curated facts. The questions are written
by a much larger language model, one schema-constrained call per small job, through the Codex CLI
(`codex exec --output-schema`). This script is the whole recipe: prompts, validation, retries, merge.

    uv run python scripts/write_questions.py --data-dir data/yan --workers 4      # resumable
    uv run python scripts/write_questions.py --data-dir data/yan --merge-only

Jobs (each writes one JSON file under <data-dir>/qa-src/chunks/, so an interrupted run resumes):

    facts/<topic>-NN     14 questions per fact and language (11 train, 1 validation, 2 test)
    multi/<topic|cross>  questions that need two or three facts read together
    chitchat/<category>  greetings, thanks, goodbyes, "what can I ask?", "who are you?"
    oos/<area>           questions that are not about the person: the model must decline
    unknown/<area>       questions about the person that the knowledge base does not answer

Held-out questions (validation and test) are requested with different structure and key words, and
validated mechanically (they may not share their first three words with a training question).
The merge step writes qa-src/{facts,multi,chitchat,oos}.jsonl; scripts/build_corpus.py turns those
into the qa/*.jsonl files of the data contract.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path

N_Q, N_TRAIN = 14, 11  # per fact and language: positions 0..10 train, 11 validation, 12..13 test
N_MULTI, N_MULTI_TRAIN = 8, 6
PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with PRINT_LOCK:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def first3(s: str) -> str:
    return " ".join(norm(s).split()[:3])


# ----------------------------------------------------------------------------------------- prompts
INTRO = """You write TRAINING DATA for "Brasa", a very small language model (about 12 million parameters) trained from
scratch to answer questions, in Portuguese or English, about one person: Yan Simmer, a Brazilian Computer
Science student at UFES and AI developer. The model runs in the browser of visitors to Yan's portfolio site.
Visitors are recruiters, engineers and potential clients who type whatever they like. The questions you write
decide whether the model understands them. Do not run any commands and do not read any files: everything you
need is in this message. Answer only with the JSON object requested.
"""

RULES = """RULES
1. Specific: each question must be one that its own fact, and no sibling fact, answers best. When two facts are
   close, put the distinguishing detail in the question.
2. Self-sufficient: the model sees one question at a time with no conversation history. Name the subject in the
   question (the project, the tool, "Yan", "o Brasa"). No "ele", "isso", "esse projeto", "he", "it", "this project"
   without a named antecedent in the same question.
3. Real visitor voices, mixed: formal and informal; long and short; direct questions, indirect ones ("queria
   saber...", "I was wondering..."), imperatives ("me conta...", "explain..."), keyword style ("stack do seer.app?").
   Portuguese is natural Brazilian Portuguese, sometimes colloquial ("qual é", "tem", "pra", "vc"). English is
   written natively, not translated question by question. In about 2 of every 14, address the model as if it
   spoke for Yan ("qual a sua stack no seer.app?", "where did you study?"); for facts of the topic "brasa",
   "você/you" means the model itself.
4. Varied vocabulary and structure: do not reuse the same opening or key verb more than twice per fact.
5. Do not give the answer away: no number, date or name that IS the answer may appear in the question. At most
   two yes/no questions per fact and language, only with a true premise.
6. Plain text: 3 to 20 words, correct spelling and accents (typos are added later by a program), no quotation
   marks inside a question, no line breaks, no emoji.
7. Honest scope: ask only what the fact states. Never introduce names, clients, numbers or claims that are not
   in the facts given here.
8. HELD-OUT questions must be phrased in ways that are NOT among the training questions of the same item: a
   different sentence structure AND different key words (if training used "stack", "tecnologias" and
   "ferramentas", the held-out ones say "feito com o quê", "construído em cima de quê"). They may not start with
   the same three words as any training question of that item. They are how the model is graded on phrasings it
   has never seen: natural, not harder than a real visitor would be.
"""


def facts_prompt(topic: dict, chunk: list[dict], siblings: list[dict]) -> str:
    sib = "\n".join(f"- {f['id']}: {f['pt']}" for f in siblings) or "(none)"
    todo = "\n".join(f"- id {f['id']}\n  pt: {f['pt']}\n  en: {f['en']}" for f in chunk)
    return f"""{INTRO}
TOPIC: {topic.get('title_pt', topic['slug'])} / {topic.get('title_en', topic['slug'])} (slug "{topic['slug']}").

For EACH of the facts listed under WRITE FOR, write exactly {N_Q} questions in Portuguese and {N_Q} in English whose
answer is that fact. Positions 1 to {N_TRAIN} are TRAINING questions, position {N_TRAIN + 1} is the VALIDATION question,
positions {N_TRAIN + 2} to {N_Q} are TEST questions (validation and test are the held-out ones, see rule 8).

{RULES}
OTHER FACTS OF THE SAME TOPIC (context only, so that your questions do not fit them better):
{sib}

WRITE FOR:
{todo}

Return {{"items": [{{"f": "<fact id>", "pt": [{N_Q} strings], "en": [{N_Q} strings]}}, ...]}} with one item per fact above,
in the same order.
"""


def multi_prompt(scope: str, facts: list[dict], n_items: int, hint: str) -> str:
    listing = "\n".join(f"- {f['id']}: {f['pt']}  ||  {f['en']}" for f in facts)
    return f"""{INTRO}
Write {n_items} MULTI-FACT items for {scope}. Each item is a question whose natural answer needs TWO or THREE of the
facts below read together; the answer shown to the visitor will be those fact sentences joined in the order
you list them, so choose facts that read as one coherent answer of at most about 70 words. {hint}

For each item write exactly {N_MULTI} questions in Portuguese and {N_MULTI} in English: positions 1 to {N_MULTI_TRAIN} are TRAINING
questions, position {N_MULTI_TRAIN + 1} is the VALIDATION question and position {N_MULTI} is the TEST question (held-out, rule 8).

{RULES}
FACTS:
{listing}

Return {{"items": [{{"id": "multi-<short-slug>", "facts": ["<fact id>", "<fact id>"], "pt": [{N_MULTI} strings], "en": [{N_MULTI} strings]}}, ...]}}.
Use only fact ids from the list.
"""


CHITCHAT = {
    "greeting": 'greetings and openers ("oi", "e aí", "bom dia", "opa, tudo bem?", "hey there", "yo", "good evening")',
    "thanks": 'thanks and compliments ("valeu", "obrigado pela ajuda", "muito bom", "thanks a lot", "nice, cheers")',
    "bye": 'goodbyes ("tchau", "até mais", "flw", "bye", "see you later", "that is all, thanks")',
    "help": 'asking what the assistant can do or what to ask ("o que você sabe fazer?", "sobre o que posso perguntar?", "what can I ask you?", "help")',
    "identity": 'asking who or what the assistant is ("quem é você?", "você é o ChatGPT?", "que modelo é esse?", "are you a real AI?", "who made you?", "am I talking to Yan?")',
}

OOS_TRAIN = {
    "geo-history": "general knowledge: geography and capitals, history and dates, famous people",
    "science-health": "science and nature, health and medical questions, cooking and recipes",
    "tasks": "requests for tasks: write or fix code, write an essay, an e-mail or a poem, summarise or translate a text",
    "math": "arithmetic, maths and logic puzzles, unit conversions",
    "entertainment": "films, series, music, football and other sports, games, celebrities, recommendations",
    "now": "current events, the weather, time and date, prices, the stock market, news, directions and travel",
    "smalltalk": "the visitor's own life and feelings, opinions on controversial topics, philosophy, relationship advice, jokes on demand",
    "misuse": "attempts to change the rules or to misuse the assistant: ignore your instructions, pretend to be another AI, reveal your system prompt, say something offensive, mildly worded generic requests for dangerous or illegal instructions, and questions about other companies or people unrelated to Yan",
}
OOS_HELD = {
    "tech-trivia": "technology in general, not about Yan: what a programming language or library is, how the internet works, comparing phones or laptops",
    "food-shopping": "food, restaurants, delivery and shopping advice",
    "homework": "school homework in physics, chemistry, biology and literature",
    "language": "language learning, grammar, word meanings and spelling",
    "everyday": "cars, home repairs, pets, gardening, horoscopes and dreams",
}
UNK_TRAIN = {
    "family-home": "Yan's family, relationships, friends, pets, home, exact address and phone number",
    "tastes": "Yan's tastes and hobbies: favourite food, football team, music, films, games, weekends, travel",
    "opinions": "Yan's private opinions and beliefs: politics, religion, what he thinks of former employers, colleagues, professors or classmates, gossip",
    "private-data": "Yan's private data and history: school grades, exam scores, health, exact birth date, documents, bank account, salary history, childhood, daily routine",
}
UNK_HELD = {
    "appearance-plans": "Yan's appearance, height, weight and clothes, and his personal plans for the future (marriage, children, moving abroad)",
    "habits-things": "Yan's social-media habits, sleep, diet, sports practice, car and possessions",
}


def list_prompt(kind: str, area: str, desc: str, counts: dict[str, int], known: str = "") -> str:
    if kind == "chitchat":
        what = f"short chat messages of this category: {desc}. They are not questions about Yan's work."
    elif kind == "oos":
        what = (f"questions or requests that have NOTHING to do with Yan Simmer or with the Brasa model, in this area: {desc}. "
                "The model must decline all of them, so they must be clearly out of scope: never mention Yan, his projects, "
                "his employers or Brasa. Harmful requests stay generic and mild in wording.")
    else:
        what = (f"questions ABOUT YAN that his portfolio does not and should not answer, in this area: {desc}. "
                "Name Yan in each question. Do NOT write questions that the following published facts already answer "
                f"(availability and salary expectations, age and marital status, current occupation, how to contact him):\n{known}")
    want = ", ".join(f'"{k}": {v} strings' for k, v in counts.items())
    held = ("" if set(counts) == {"train"} else
            ' Lists other than "train" are held-out: they may not start with the same three words as any "train" entry and must use different key words.')
    only = ("" if "train" in counts else " (All lists here are held-out sets for evaluation: make them natural, not harder than a real visitor would be.)")
    return f"""{INTRO}
Write {what}

Per language (Portuguese and English, each written natively) return these lists: {want}.{held}{only}
Real variety: formal and informal, very short and longer, with and without punctuation style differences, different
openings. Plain text, 1 to 22 words each, correct spelling and accents, no quotation marks inside, no line breaks,
no emoji, no duplicates.

Return {{"pt": {{{", ".join(f'"{k}": [...]' for k in counts)}}}, "en": {{{", ".join(f'"{k}": [...]' for k in counts)}}}}}.
"""


# ----------------------------------------------------------------------------------------- schemas
STR_LIST = {"type": "array", "items": {"type": "string"}}
ITEMS_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["f", "pt", "en"],
        "properties": {"f": {"type": "string"}, "pt": STR_LIST, "en": STR_LIST}}}},
}
MULTI_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "facts", "pt", "en"],
        "properties": {"id": {"type": "string"}, "facts": STR_LIST, "pt": STR_LIST, "en": STR_LIST}}}},
}


def list_schema(keys: list[str]) -> dict:
    lang = {"type": "object", "additionalProperties": False, "required": keys, "properties": {k: STR_LIST for k in keys}}
    return {"type": "object", "additionalProperties": False, "required": ["pt", "en"], "properties": {"pt": lang, "en": lang}}


# ----------------------------------------------------------------------------------------- validation
def check_questions(qs: list[str], n: int, n_train: int, lo: int, hi: int, where: str) -> list[str]:
    errs: list[str] = []
    if len(qs) != n:
        return [f"{where}: expected exactly {n} questions, got {len(qs)}"]
    seen: set[str] = set()
    for i, q in enumerate(qs, 1):
        w = len(q.split())
        if not (lo <= w <= hi):
            errs.append(f"{where} #{i}: {w} words (allowed {lo} to {hi}): {q!r}")
        if '"' in q or "\n" in q:
            errs.append(f"{where} #{i}: quotation mark or line break: {q!r}")
        k = norm(q)
        if k in seen:
            errs.append(f"{where} #{i}: duplicate of another question of the same item: {q!r}")
        seen.add(k)
    heads = {first3(q) for q in qs[:n_train]}
    for i, q in enumerate(qs[n_train:], n_train + 1):
        if first3(q) in heads:
            errs.append(f"{where} #{i} (held-out) starts with the same three words as a training question: {q!r}")
    return errs


def validate_facts(obj: dict, chunk: list[dict]) -> list[str]:
    errs: list[str] = []
    got = {it.get("f"): it for it in obj.get("items", [])}
    for f in chunk:
        it = got.get(f["id"])
        if it is None:
            errs.append(f"missing item for fact {f['id']}")
            continue
        for lang in ("pt", "en"):
            errs += check_questions(it.get(lang, []), N_Q, N_TRAIN, 3, 20, f"{f['id']}.{lang}")
    extra = set(got) - {f["id"] for f in chunk}
    if extra:
        errs.append(f"unexpected fact ids: {sorted(extra)}")
    return errs


def validate_multi(obj: dict, valid_ids: set[str], n_items: int) -> list[str]:
    errs: list[str] = []
    items = obj.get("items", [])
    if not (max(1, n_items - 2) <= len(items) <= n_items + 2):
        errs.append(f"expected about {n_items} items, got {len(items)}")
    ids: set[str] = set()
    for it in items:
        iid = it.get("id", "?")
        if iid in ids:
            errs.append(f"duplicate item id {iid}")
        ids.add(iid)
        fs = it.get("facts", [])
        if not (2 <= len(fs) <= 3) or len(set(fs)) != len(fs):
            errs.append(f"{iid}: needs two or three distinct fact ids, got {fs}")
        for f in fs:
            if f not in valid_ids:
                errs.append(f"{iid}: unknown fact id {f}")
        for lang in ("pt", "en"):
            errs += check_questions(it.get(lang, []), N_MULTI, N_MULTI_TRAIN, 3, 22, f"{iid}.{lang}")
    return errs


def validate_lists(obj: dict, counts: dict[str, int]) -> list[str]:
    errs: list[str] = []
    for lang in ("pt", "en"):
        seen: set[str] = set()
        block = obj.get(lang, {})
        heads = {first3(q) for q in block.get("train", [])}
        for key, n in counts.items():
            qs = block.get(key, [])
            if not (n <= len(qs) <= n + 5):
                errs.append(f"{lang}.{key}: expected {n} entries, got {len(qs)}")
            for q in qs:
                w = len(q.split())
                if not (1 <= w <= 22):
                    errs.append(f"{lang}.{key}: {w} words: {q!r}")
                if '"' in q or "\n" in q:
                    errs.append(f"{lang}.{key}: quotation mark or line break: {q!r}")
                k = norm(q)
                if k in seen:
                    errs.append(f"{lang}.{key}: duplicate: {q!r}")
                seen.add(k)
                if key != "train" and "train" in counts and len(q.split()) >= 3 and first3(q) in heads:
                    errs.append(f"{lang}.{key} (held-out) starts like a training entry: {q!r}")
    return errs


# ----------------------------------------------------------------------------------------- codex
def run_codex(prompt: str, schema: dict, effort: str, model: str | None, timeout: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="brasa-q-") as tmp:
        t = Path(tmp)
        (t / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
        cmd = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "-C", tmp,
               "-c", f'model_reasoning_effort="{effort}"', "--output-schema", str(t / "schema.json"),
               "-o", str(t / "out.json"), "-"]
        if model:
            cmd[2:2] = ["-m", model]
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        out = t / "out.json"
        if p.returncode != 0 or not out.exists():
            raise RuntimeError(f"codex exited {p.returncode}: {(p.stderr or p.stdout)[-400:]}")
        return json.loads(out.read_text(encoding="utf-8"))


def do_job(job: dict, args) -> tuple[str, bool, str]:
    out: Path = job["out"]
    if out.exists():
        return job["name"], True, "cached"
    prompt, last = job["prompt"], ""
    for attempt in range(1, args.retries + 2):
        try:
            obj = run_codex(prompt, job["schema"], args.effort, args.model, args.timeout)
        except Exception as e:  # noqa: BLE001 - any failure of the external tool is retried
            last = f"call failed: {e}"
            time.sleep(min(60, 10 * attempt))
            continue
        errs = job["validate"](obj)
        if not errs:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
            return job["name"], True, f"ok (attempt {attempt})"
        last = f"{len(errs)} validation problems, e.g. {errs[0]}"
        prompt = (job["prompt"] + "\n\nYour previous answer was rejected by the validator. Fix ALL of these problems and "
                  "return the complete JSON again:\n" + "\n".join("- " + e for e in errs[:40]))
    return job["name"], False, last


# ----------------------------------------------------------------------------------------- jobs
def load(data_dir: Path):
    facts = [json.loads(line) for line in (data_dir / "kb/facts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    topics = {t["slug"]: t for t in json.loads((data_dir / "kb/topics.json").read_text(encoding="utf-8"))}
    return facts, topics


def existing_fact_items(src: Path) -> dict[str, dict]:
    """Fact items already written (by earlier runs or by hand), keyed by fact id; first valid one wins."""
    done: dict[str, dict] = {}
    files = sorted(src.glob("facts-*.jsonl")) + sorted((src / "chunks/facts").glob("*.json"))
    for p in files:
        if p.suffix == ".jsonl":
            items = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            items = json.loads(p.read_text(encoding="utf-8")).get("items", [])
        for it in items:
            fid = it.get("f")
            if fid and fid not in done:
                ok = all(not check_questions(it.get(lang, []), N_Q, N_TRAIN, 3, 20, fid) for lang in ("pt", "en"))
                if ok:
                    done[fid] = it
    return done


def build_jobs(data_dir: Path, args) -> list[dict]:
    facts, topics = load(data_dir)
    src = data_dir / "qa-src"
    chunks = src / "chunks"
    by_topic: dict[str, list[dict]] = {}
    for f in facts:
        by_topic.setdefault(f["topic"], []).append(f)
    ids = {f["id"] for f in facts}
    done = existing_fact_items(src)
    jobs: list[dict] = []

    def want(kind: str) -> bool:
        return not args.only or kind in args.only

    if want("facts"):
        for slug, fs in by_topic.items():
            todo = [f for f in fs if f["id"] not in done]
            for i in range(0, len(todo), args.chunk):
                chunk = todo[i:i + args.chunk]
                tag = chunk[0]["id"].rsplit("-", 1)[-1]
                jobs.append({"name": f"facts/{slug}-{tag}", "out": chunks / "facts" / f"{slug}-{tag}.json",
                             "prompt": facts_prompt(topics.get(slug, {"slug": slug}), chunk, [f for f in fs if f not in chunk]),
                             "schema": ITEMS_SCHEMA, "validate": (lambda o, c=chunk: validate_facts(o, c))})
    if want("multi"):
        for slug, fs in by_topic.items():
            n = 4 if len(fs) >= 20 else 3 if len(fs) >= 10 else 2
            hint = "The first item is an overview of the topic (what it is, plus Yan's role or the main result)."
            jobs.append({"name": f"multi/{slug}", "out": chunks / "multi" / f"{slug}.json",
                         "prompt": multi_prompt(f'the topic "{slug}"', fs, n, hint), "schema": MULTI_SCHEMA,
                         "validate": (lambda o, n=n: validate_multi(o, ids, n))})
        cross = {
            "cross-ml": (["breakhis", "pad", "ml-coursework", "brasa", "skills"],
                         "Cross-topic questions a technical interviewer would ask: which machine-learning projects Yan has and with which models, how he avoids data leakage across projects, what Brasa is and how it was trained and is evaluated, a metric together with its caveat."),
            "cross-work": (["experience", "seer", "crm", "prospecting", "isabela", "nina", "ct-junior-clients", "services"],
                           "Cross-topic questions a recruiter would ask: where Yan worked and when, which AI agents he built, which products he contributed to and in which role, the stack of one project versus another."),
            "cross-about": (["profile", "studies", "contact", "process", "boundaries", "simmer", "gjdb", "projeto-piloto", "armv7a", "algorithms-c", "operating-systems", "esp32-cam-car", "c-coursework"],
                            "Cross-topic questions: who Yan is and what he studies and when he graduates, how to contact him, his own projects, his computing fundamentals, how he works."),
        }
        for name, (slugs, hint) in cross.items():
            fs = [f for s in slugs for f in by_topic.get(s, []) if f["importance"] >= 4][:140]
            jobs.append({"name": f"multi/{name}", "out": chunks / "multi" / f"{name}.json",
                         "prompt": multi_prompt("several topics at once", fs, 8, hint + " Each item must combine facts from at least two different topics when it makes sense."),
                         "schema": MULTI_SCHEMA, "validate": (lambda o: validate_multi(o, ids, 8))})
    known = "\n".join(f"- {f['pt']}" for s in ("boundaries", "contact") for f in by_topic.get(s, []))

    def add_list(kind: str, area: str, desc: str, counts: dict[str, int]):
        jobs.append({"name": f"{kind}/{area}", "out": chunks / kind / f"{area}.json",
                     "prompt": list_prompt(kind, area, desc, counts, known), "schema": list_schema(list(counts)),
                     "validate": (lambda o, c=counts: validate_lists(o, c))})

    if want("chitchat"):
        for cat, desc in CHITCHAT.items():
            add_list("chitchat", cat, desc, {"train": 40, "val": 5, "test": 10})
    if want("oos"):
        for area, desc in OOS_TRAIN.items():
            add_list("oos", area, desc, {"train": 40})
        for area, desc in OOS_HELD.items():
            add_list("oos", "held-" + area, desc, {"val": 5, "test": 14})
    if want("unknown"):
        for area, desc in UNK_TRAIN.items():
            add_list("unknown", area, desc, {"train": 40})
        for area, desc in UNK_HELD.items():
            add_list("unknown", "held-" + area, desc, {"val": 8, "test": 20})
    return jobs


# ----------------------------------------------------------------------------------------- merge
def merge(data_dir: Path) -> dict:
    facts, _ = load(data_dir)
    src = data_dir / "qa-src"
    done = existing_fact_items(src)
    order = [f["id"] for f in facts]
    missing = [i for i in order if i not in done]
    with (src / "facts.jsonl").open("w", encoding="utf-8") as fh:
        for i in order:
            if i in done:
                it = done[i]
                fh.write(json.dumps({"f": i, "pt": it["pt"], "en": it["en"]}, ensure_ascii=False) + "\n")
    multi, seen = [], set()
    old = src / "multi-claude.jsonl"
    sources = ([json.loads(line) for line in old.read_text(encoding="utf-8").splitlines() if line.strip()] if old.exists() else [])
    for p in sorted((src / "chunks/multi").glob("*.json")):
        sources += json.loads(p.read_text(encoding="utf-8")).get("items", [])
    ids = set(order)
    for it in sources:
        iid = it.get("id", "")
        if not iid.startswith("multi-"):
            iid = "multi-" + iid
        base, k = iid, 2
        while iid in seen:
            iid, k = f"{base}-{k}", k + 1
        if all(f in ids for f in it.get("facts", [])) and not validate_multi({"items": [it]}, ids, 1):
            seen.add(iid)
            multi.append({"id": iid, "facts": it["facts"], "pt": it["pt"], "en": it["en"]})
    with (src / "multi.jsonl").open("w", encoding="utf-8") as fh:
        for it in multi:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    def gather(kind: str) -> dict[str, dict]:
        cats: dict[str, dict] = {}
        for p in sorted((src / "chunks" / kind).glob("*.json")):
            obj = json.loads(p.read_text(encoding="utf-8"))
            cat = p.stem if kind == "chitchat" else kind
            slot = cats.setdefault(cat, {"cat": cat, "pt": {"train": [], "val": [], "test": []}, "en": {"train": [], "val": [], "test": []}})
            for lang in ("pt", "en"):
                for split, qs in obj.get(lang, {}).items():
                    have = {norm(q) for s in slot[lang].values() for q in s}
                    slot[lang][split] += [q for q in qs if norm(q) not in have]
        return cats

    counts = {"facts": len(done), "facts_missing": len(missing), "multi": len(multi)}
    for kind, fname in (("chitchat", "chitchat.jsonl"), ("oos", "oos.jsonl")):
        cats = gather(kind)
        if kind == "oos":
            cats.update(gather("unknown"))
        with (src / fname).open("w", encoding="utf-8") as fh:
            for c in cats.values():
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
                counts[c["cat"]] = {lang: {s: len(v) for s, v in c[lang].items()} for lang in ("pt", "en")}
    if missing:
        counts["first_missing"] = missing[:12]
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/yan")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=10, help="facts per call")
    ap.add_argument("--effort", default="high", help="reasoning effort passed to the teacher model")
    ap.add_argument("--model", default=None, help="teacher model (default: the Codex CLI default)")
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--only", nargs="*", help="facts multi chitchat oos unknown")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    if not args.merge_only:
        jobs = build_jobs(data_dir, args)
        pending = [j for j in jobs if not j["out"].exists()]
        log(f"{len(jobs)} jobs, {len(pending)} to run with {args.workers} workers (effort {args.effort})")
        failed = 0
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for name, ok, msg in ex.map(lambda j: do_job(j, args), pending):
                failed += 0 if ok else 1
                log(f"{'done ' if ok else 'FAIL '} {name}: {msg}")
        log(f"finished: {len(pending) - failed} ok, {failed} failed")
    print(json.dumps(merge(data_dir), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
