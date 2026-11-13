"""scripts/render_model_card.py: the Markdown card shows the strict numbers and what a visitor experiences."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("render_model_card", ROOT / "scripts" / "render_model_card.py")
card_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(card_script)


def metrics(n: int, em: float, **extra) -> dict:
    return {"n": n, "em": em, "f1": em + 0.1, "in_scope_em": em, "in_scope_f1": em, "key_span_recall": 0.5,
            "refusal_precision": 0.9, "refusal_recall": 0.8, "language_consistency": 1.0,
            "unsupported_number_rate": 0.0, "verbatim_rate": 1.0, "right_topic_rate": 0.8, **extra}


def card() -> dict:
    retrieval = {"retrieval_hit": 0.6, "gold_in_top1": 0.3, "gold_in_top2": 0.45, "gold_in_top4": 0.6,
                 "em_when_gold_retrieved": 0.5, "em_when_gold_missed": 0.05}
    group = {"n": 10, "em": 0.4, "f1": 0.5, "verbatim_rate": 1.0, "right_topic_rate": 0.9}
    full = {"open_book_retrieved": {"test": {"overall": group, "by_lang": {"pt": group, "en": group},
                                             "by_kind": {"single": group, "oos": {**group, "right_topic_rate": None}},
                                             "by_topic": {"small": {**group, "n": 2}, "big": {**group, "n": 8}}}}}
    headline = {"open_book_retrieved": {"seen": metrics(300, 1.0, **retrieval),
                                        "test": metrics(1278, 0.35, **retrieval)},
                "closed_book": {"test": metrics(1278, 0.25)},
                "first_hit": {"test": metrics(1278, 0.26, **retrieval)}}
    return {"name": "brasa-test", "model": {"params": 4188416, "n_layer": 4, "d_model": 256, "n_head": 4,
                                           "block_size": 384, "vocab_size": 4096},
            "training": {"steps": 6000, "device": "mps", "train_seconds": 600},
            "data": {"corpus": {"fictional": False}},
            "evaluation": {"float32": {"headline": headline, "full": full,
                                       "decoding": {"retriever": {"kind": "bm25+own", "weight": 1.5}}}}}


def test_the_card_has_the_visitor_table_and_the_breakdowns():
    template = (ROOT / "MODEL_CARD.template.md").read_text(encoding="utf-8")
    text = card_script.render(card(), template)
    assert "| float32 | open-book, retrieved facts only | held-out (test) | 1278 | 35.0% |" in text
    assert "| float32 | retriever's first fact (no model) | held-out (test) | 26.0% | 100.0% | 80.0% | "\
           "30.0% / 45.0% / 60.0% | 50.0% / 5.0% |" in text
    assert "| float32 | closed-book, free | held-out (test) | 25.0% | 100.0% | 80.0% | n/a | n/a |" in text
    assert "seen (training) | 100.0% | 100.0%" not in text, "the visitor table is about held-out questions"
    assert "Breakdown of open-book, retrieved facts only, float32, held-out (test):" in text
    assert "| pt | 10 | 40.0% | 50.0% | 100.0% | 90.0% |" in text and "| out of scope | 10 |" in text
    assert text.index("| big | 8 |") < text.index("| small | 2 |"), "topics sorted by size"
    assert "BM25 fused with the model's own ranking of the facts" in text and "weight 1.5" in text
    assert "{{" not in text


def test_final_rows_and_a_lexical_retriever_are_described():
    c = card()
    c["evaluation"]["float32"]["headline"]["open_book_retrieved"]["final"] = metrics(1210, 0.3)
    c["evaluation"]["float32"]["full"]["open_book_retrieved"]["final"] = \
        c["evaluation"]["float32"]["full"]["open_book_retrieved"]["test"]
    c["evaluation"]["float32"]["decoding"] = {}
    text = card_script.render(c, (ROOT / "MODEL_CARD.template.md").read_text(encoding="utf-8"))
    assert "held-out (final, read once) | 1210 | 30.0%" in text
    assert "Breakdown of open-book, retrieved facts only, float32, held-out (final, read once):" in text
    assert "Retriever used in the open-book rows: BM25 over the facts" in text
    assert "No ranking prior was chosen" in text
    assert "The weights start random and the only training data is the question corpus" in text
    assert "Facts come from a BM25 retriever over the knowledge base" in text


def test_ranked_decoding_pretraining_and_the_learned_ranking_are_described():
    c = card()
    held_out = c["evaluation"]["float32"]
    held_out["headline"]["open_book_ranked"] = {"test": metrics(1278, 0.41, gold_in_top1=0.44)}
    held_out["full"]["open_book_ranked"] = held_out["full"]["open_book_retrieved"]
    held_out["decoding"]["prior"] = {"kind": "score", "weight": 16.0, "fixed": 0.0}
    recall = {"test": {"recall@1": 0.445, "recall@4": 0.751}}
    c["training"].update(
        init_from={"steps": 12000, "tokens_seen": 294912000, "train_seconds": 1475.2, "final_val": {"val_ppl": 25.0},
                   "corpus": {"manifests": {"m.json": {
                       "source": "https://huggingface.co/datasets/wikimedia/wikipedia", "revision": "b04c8d1ceb2f",
                       "dump": "2023-11-01", "licence": "CC BY-SA 3.0 and GFDL (Wikipedia text)",
                       "languages": {"pt": {"articles": 22631, "bytes": 160066479, "shard": "pt/a.parquet"}}}}}},
        retrieval={"weight": 2.0, "pooling": "mean",
                   "recall": {"lexical": {"test": {"recall@1": 0.298, "recall@4": 0.573}}, "fused": recall}})
    text = card_script.render(c, (ROOT / "MODEL_CARD.template.md").read_text(encoding="utf-8"))
    assert ("| float32 | open-book, retrieved facts + ranking prior (site default) | held-out (test) | 1278 | 41.0% |"
            in text)
    assert "Breakdown of open-book, retrieved facts + ranking prior (site default), float32" in text
    assert "loses `weight x (1 - its score / the best score)` nats, weight 16.0, chosen on the validation" in text
    assert "Still from scratch" in text and "12,000 steps, 295M tokens seen, 25 minutes" in text
    assert "validation perplexity 25.0" in text and "revision `b04c8d1c`" in text and "CC BY-SA 3.0" in text
    assert "pt: 22,631 articles, 160 MB (`pt/a.parquet`)" in text
    assert "The same network that writes the answer is also trained, with a contrastive loss" in text
    assert "first hit for 44.5% (BM25 alone: 29.8%) and among the first four for 75.1% (57.3%)" in text
    assert "{{" not in text


def test_the_readme_block_is_written_from_a_release_directory(tmp_path):
    import json

    spec2 = importlib.util.spec_from_file_location("readme_results", ROOT / "scripts" / "readme_results.py")
    readme_results = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(readme_results)

    c = card()
    float32 = c["evaluation"]["float32"]
    float32["headline"]["open_book_ranked"] = {"seen": metrics(300, 1.0), "test": metrics(1278, 0.41, gold_in_top1=0.44,
                                                                                         gold_in_top2=0.6,
                                                                                         gold_in_top4=0.75),
                                               "final": metrics(1210, 0.39)}
    float32["full"]["open_book_ranked"] = {"final": float32["full"]["open_book_retrieved"]["test"]}
    float32["decoding"]["prior"] = {"kind": "score", "weight": 24.0, "fixed": 0.0}
    c["evaluation"]["int8"] = {"headline": {"open_book_ranked": {"test": metrics(1278, 0.405)}},
                               "answers_identical_to_float32": "17000/17892"}
    c["training"]["retrieval"] = {"weight": 1.5}
    c["quantisation"] = {"i8": {"mean_relative_weight_rmse": 0.007, "max_abs_logit_drift_vs_float32": 0.17}}
    c["files"] = {"model.i8.bin": {"bytes": 4296864, "brotli": 3934058}}
    release = tmp_path / "final"
    release.mkdir()
    (release / "model-card.json").write_text(json.dumps(c))
    (release / "RELEASE.json").write_text(json.dumps({"run": "runs/x", "checkpoint": "last", "final_split_read": True,
                                                      "released_at": "2026-11-13 09:00:00 -0300"}))
    (release / "bench.jsonl").write_text(json.dumps({"kernel": "wasm-simd", "decodeTokPerSec": 2999.5, "node": "v24"}))
    block = readme_results.release_block(release)
    assert "**`brasa-test`** (4,188,416 parameters; run `runs/x`, last checkpoint, released 2026-11-13 09:00)" in block
    assert "| held-out test | site default: retrieved facts + ranking prior | 1278 | 41.0 |" in block
    assert "| held-out test | site default, **int8 weights** (what the browser runs) | 1278 | 40.5 |" in block
    assert "| final slice (read once) | site default: retrieved facts + ranking prior | 1210 | 39.0 |" in block
    assert "| held-out test | the retriever's first fact, no language model | 1278 | 26.0 |" in block
    assert "(final slice (read once), float32, exact match): one fact 40.0 (n=10)" in block
    assert "gold fact first for 44.0%, in the top 2 for 60.0%, in the top 4 for 75.0%" in block
    assert "Ranking prior: `score`, weight 24.0" in block and "3,000 tokens/s (wasm-simd)" in block
    assert "The final slice has not been read" not in block

    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{readme_results.BEGIN}\nSTALE\n{readme_results.END}\nafter\n")
    import sys
    argv, sys.argv = sys.argv, ["readme_results.py", str(release), "--readme", str(readme)]
    try:
        readme_results.main()
    finally:
        sys.argv = argv
    text = readme.read_text()
    assert text.startswith("before\n") and text.endswith("after\n") and "STALE" not in text and "41.0" in text
