"""Render the Hugging Face model card (Markdown) from model-card.json and MODEL_CARD.template.md.

    uv run python scripts/render_model_card.py model-card.json --out MODEL_CARD.md
    uv run python scripts/render_model_card.py runs/smoke/model-card.json          # prints to stdout

Every number in the output comes from the JSON card, which is written by brasa.evaluate and brasa.export.
A card measured on a corpus marked "fictional" gets a warning at the top instead of looking like a result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT_NAMES = {"seen": "seen (training)", "val": "held-out (validation)", "test": "held-out (test)",
               "final": "held-out (final, read once)"}
MODE_NAMES = {
    "open_book": "open-book, free", "closed_book": "closed-book, free",
    "open_book_constrained": "open-book, any curated sentence",
    "closed_book_constrained": "closed-book, any curated sentence",
    "open_book_retrieved": "open-book, retrieved facts only",
    "open_book_ranked": "open-book, retrieved facts + ranking prior (site default)",
    "first_hit": "retriever's first fact (no model)",
}
# open-book decodings, from the one the site uses to the least constrained
SITE_MODES = ("open_book_ranked", "open_book_retrieved", "open_book_constrained", "open_book")
KIND_NAMES = {"single": "one fact", "multi": "several facts", "meta": "about the model", "chitchat": "small talk",
              "oos": "out of scope"}
FILE_NOTES = {
    "model.i8.bin": "weights, int8 with one scale per row (what the site loads)",
    "model.f16.bin": "weights, float16",
    "tokenizer.json": "BPE merges and control tokens",
    "retrieval.json": "BM25 index over the knowledge base",
    "manifest.json": "configuration, sizes and checksums",
}


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def evaluation_table(evaluation: dict) -> str:
    head = ("| Weights | Mode | Questions | n | EM | F1 | EM in scope | Key spans | Refusal P | Refusal R | "
            "Language | Bad numbers |")
    lines = [head, "|" + " --- |" * 12]
    for label, entry in evaluation.items():
        for mode, per_split in entry.get("headline", {}).items():
            for split, m in per_split.items():
                cells = [label, MODE_NAMES.get(mode, mode), SPLIT_NAMES.get(split, split), str(m["n"]), pct(m["em"]),
                         pct(m["f1"]), pct(m["in_scope_em"]), pct(m["key_span_recall"]), pct(m["refusal_precision"]),
                         pct(m["refusal_recall"]), pct(m["language_consistency"]), pct(m["unsupported_number_rate"])]
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def visitor_table(evaluation: dict) -> str:
    """What a visitor experiences on held-out questions, next to the strict numbers."""
    head = ("| Weights | Mode | Questions | Exact | Curated sentence | Right topic | Gold fact in top 1 / 2 / 4 | "
            "Exact when retrieved / when missed |")
    lines = [head, "|" + " --- |" * 8]
    for label, entry in evaluation.items():
        for mode, per_split in entry.get("headline", {}).items():
            for split, m in per_split.items():
                if split == "seen" or "verbatim_rate" not in m:
                    continue
                top = " / ".join(pct(m.get(f"gold_in_top{k}")) for k in (1, 2, 4)) if "gold_in_top1" in m else "n/a"
                when = (f"{pct(m.get('em_when_gold_retrieved'))} / {pct(m.get('em_when_gold_missed'))}"
                        if "em_when_gold_retrieved" in m else "n/a")
                lines.append("| " + " | ".join([label, MODE_NAMES.get(mode, mode), SPLIT_NAMES.get(split, split),
                                                pct(m["em"]), pct(m["verbatim_rate"]), pct(m["right_topic_rate"]),
                                                top, when]) + " |")
    return "\n".join(lines) if len(lines) > 2 else ""


def breakdown_tables(evaluation: dict) -> str:
    """Per language, per kind of question and per topic, for the decoding the site uses, on the strictest
    held-out split available (final, else test)."""
    full = evaluation.get("float32", {}).get("full", {})
    mode = next((m for m in SITE_MODES if m in full), None)
    if mode is None:
        return ""
    split = next((s for s in ("final", "test", "val") if s in full[mode]), None)
    if split is None:
        return ""
    rep = full[mode][split]
    out = [f"Breakdown of {MODE_NAMES.get(mode, mode)}, float32, {SPLIT_NAMES.get(split, split)}:", ""]

    def table(title: str, groups: dict, names: dict | None = None) -> None:
        out.extend([f"| {title} | n | Exact | F1 | Curated sentence | Right topic |", "|" + " --- |" * 6])
        for key, m in groups.items():
            out.append(f"| {(names or {}).get(key, key)} | {m['n']} | {pct(m['em'])} | {pct(m['f1'])} | "
                       f"{pct(m.get('verbatim_rate'))} | {pct(m.get('right_topic_rate'))} |")
        out.append("")

    table("Language", rep.get("by_lang", {}))
    table("Kind of question", rep.get("by_kind", {}), KIND_NAMES)
    table("Topic", dict(sorted(rep.get("by_topic", {}).items(), key=lambda kv: -kv[1]["n"])))
    return "\n".join(out).rstrip()


def files_table(files: dict) -> str:
    lines = ["| File | Bytes | gzip | brotli | What |", "| --- | ---: | ---: | ---: | --- |"]
    for name, f in files.items():
        if isinstance(f, dict) and name in FILE_NOTES and f.get("bytes") is not None:
            brotli = "n/a" if f.get("brotli") is None else f"{f['brotli']:,}"
            lines.append(f"| `{name}` | {f['bytes']:,} | {f['gzip']:,} | {brotli} | {FILE_NOTES[name]} |")
    return "\n".join(lines)


def decoding_note(decoding: dict) -> str:
    retriever = decoding.get("retriever") or {}
    kind = retriever.get("kind", "bm25")
    if kind == "bm25+own":
        how = (f"BM25 fused with the model's own ranking of the facts (cosine of its pooled hidden states, "
               f"weight {retriever.get('weight')})")
    elif kind == "bm25+encoder":
        how = f"BM25 fused with a separate learned encoder (weight {retriever.get('weight')})"
    else:
        how = "BM25 over the facts, their topic names and the training questions"
    prior = decoding.get("prior") or {}
    if prior.get("weight") or prior.get("fixed"):
        form = ("a fact loses `weight x (1 - its score / the best score)` nats" if prior.get("kind") == "score"
                else "a fact loses `weight` nats per position in the ranking")
        fixed = f"; fixed answers {prior['fixed']:+} nats." if prior.get("fixed") else "."
        ranked = (f" Ranking prior of the site default: {form}, weight {prior.get('weight')}, chosen on the "
                  f"validation questions{fixed}")
    else:
        ranked = " No ranking prior was chosen for this model: the two 'retrieved facts' rows are the same decoding."
    return f"Retriever used in the open-book rows: {how}.{ranked}"


def pretraining_note(training: dict) -> str:
    """Where the weights started: random, or a language-model pretraining run of the same code (from scratch)."""
    init = training.get("init_from") or {}
    if not init:
        return "The weights start random and the only training data is the question corpus described below."
    corpus = init.get("corpus") or {}
    manifests = list((corpus.get("manifests") or {}).values())
    source = ""
    if manifests:
        m = manifests[0]
        langs = ", ".join(f"{lang}: {v['articles']:,} articles, {v['bytes'] / 1e6:.0f} MB (`{v['shard']}`)"
                          for lang, v in (m.get("languages") or {}).items())
        source = (f" Text: {m.get('source')} at revision `{str(m.get('revision'))[:8]}`, dump {m.get('dump')}, "
                  f"licence {m.get('licence')}; {langs}.")
    seen = init.get("tokens_seen")
    tokens = f"{seen / 1e6:.0f}M tokens seen" if seen else "an unknown number of tokens"
    ppl = (init.get("final_val") or {}).get("val_ppl")
    minutes = f"{init['train_seconds'] / 60:.0f} minutes" if init.get("train_seconds") else "an unrecorded time"
    return ("Still from scratch: the weights start random and are first trained by this repository's own code as a "
            f"plain language model on public text ({init.get('steps'):,} steps, {tokens}, {minutes} on the same "
            f"laptop" + (f", validation perplexity {ppl}" if ppl else "") + "), then on the question corpus. No "
            f"pretrained weights or tokenizers from anyone else are involved.{source}")


def retriever_note(training: dict) -> str:
    info = training.get("retrieval") or {}
    if not info:
        return ("Facts come from a BM25 retriever over the knowledge base (`retrieval.json`: words and character "
                "trigrams of the facts, their topic names and the training questions).")
    fused = ((info.get("recall") or {}).get("fused") or {}).get("test") or {}
    lexical = ((info.get("recall") or {}).get("lexical") or {}).get("test") or {}
    found = ""
    if fused and lexical:
        found = (f" On held-out test questions the gold fact is the first hit for {pct(fused.get('recall@1'))} "
                 f"(BM25 alone: {pct(lexical.get('recall@1'))}) and among the first four for "
                 f"{pct(fused.get('recall@4'))} ({pct(lexical.get('recall@4'))}).")
    return ("Facts are ranked by two signals added together: BM25 over the knowledge base (`retrieval.json`: words "
            "and character trigrams of the facts, their topic names and the training questions) and the model's "
            "own opinion. The same network that writes the answer is also trained, with a contrastive loss, to put "
            "a question next to the fact that answers it: the browser runs it once over the question, pools its "
            "final hidden states into one vector and compares it with the precomputed vectors of the facts "
            f"(int8, inside `retrieval.json`; fusion weight {info.get('weight')}).{found}")


def render(card: dict, template: str) -> str:
    model, training, data = card.get("model", {}), card.get("training", {}), card.get("data", {})
    evaluation = {k: v for k, v in card.get("evaluation", {}).items() if isinstance(v, dict)}
    corpus = data.get("corpus") or {}
    seconds = training.get("train_seconds")
    headline = evaluation.get("float32", {}).get("headline", {})
    test = next((headline[m]["test"] for m in ("open_book", *SITE_MODES) if "test" in headline.get(m, {})), {})
    quant = card.get("quantisation") or {}

    values = {
        "name": card.get("name"), "params": f"{model['params']:,}" if model.get("params") else None,
        "n_layer": model.get("n_layer"), "d_model": model.get("d_model"), "n_head": model.get("n_head"),
        "block_size": model.get("block_size"), "vocab_size": model.get("vocab_size"),
        "steps": f"{training['steps']:,}" if training.get("steps") else None, "device": training.get("device"),
        "train_time": f"{seconds / 60:.0f} min" if seconds else None,
        "max_facts": training.get("data_config", {}).get("prompt", {}).get("max_facts"),
        "files_table": files_table(card.get("files", {})) if (card.get("files") or {}).get("directory") else None,
        "evaluation_table": evaluation_table(evaluation) if test else None,
        "visitor_table": visitor_table(evaluation) if test else None,
        "breakdown_tables": breakdown_tables(evaluation) if test else None,
        "decoding_note": decoding_note(evaluation.get("float32", {}).get("decoding") or {}),
        "pretraining_note": pretraining_note(training),
        "retriever_note": retriever_note(training),
        "corpus_warning": (
            "> **This card was produced from a FICTIONAL rehearsal corpus.** The person it describes does not "
            "exist and none of these numbers is a result of the released model."
            if corpus.get("fictional") else ""),
        "retrieval_note": (
            f"Open-book, held-out test questions: the retriever returned the gold facts for "
            f"{pct(test['retrieval_hit'])} of the in-scope questions; exact match was "
            f"{pct(test['em_when_gold_retrieved'])} when it did and {pct(test['em_when_gold_missed'])} when it "
            f"did not." if "em_when_gold_retrieved" in test else ""),
        "quantisation_note": "; ".join(
            f"{fmt}: relative weight error {q['mean_relative_weight_rmse']}, largest logit drift "
            f"{q['max_abs_logit_drift_vs_float32']}" for fmt, q in quant.items()
            if isinstance(q, dict) and q.get("mean_relative_weight_rmse") is not None)
        + "".join(f"; {label} answers identical to float32: {e['answers_identical_to_float32']}"
                  for label, e in evaluation.items() if e.get("answers_identical_to_float32"))
        if any(isinstance(q, dict) and q.get("mean_relative_weight_rmse") is not None for q in quant.values())
        else None,
        "data_note": (
            f"{data['facts']} facts in {data['topics']} topics, each in Portuguese and English; "
            f"{data['splits']['train']['n']} training, {data['splits']['val']['n']} validation and "
            f"{data['splits']['test']['n']} test question/answer records "
            f"({data['train_questions_per_fact_and_lang']['median']} training questions per fact and language, "
            f"median); {data['training_examples']['examples']:,} training examples of "
            f"{data['training_examples']['tokens']:,} tokens after open-book / closed-book expansion."
            if data.get("splits") and data["splits"]["train"].get("n") else None),
    }

    def fill(match: re.Match) -> str:
        value = values.get(match.group(1))
        return "TBD" if value is None else str(value)

    return re.sub(r"\{\{(\w+)\}\}", fill, template)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("card", help="model-card.json written by brasa.evaluate / brasa.export")
    ap.add_argument("--template", default=str(ROOT / "MODEL_CARD.template.md"))
    ap.add_argument("--out", help="write here instead of stdout")
    args = ap.parse_args()
    card = json.loads(Path(args.card).read_text(encoding="utf-8"))
    text = render(card, Path(args.template).read_text(encoding="utf-8"))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
