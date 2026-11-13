#!/usr/bin/env python3
"""Download a fixed slice of Portuguese and English Wikipedia as plain text for the pretraining experiment.

    uv run --with pyarrow python scripts/fetch_wiki.py                 # about 160 MB of text per language
    uv run --with pyarrow python scripts/fetch_wiki.py --mb 40         # a smaller slice

Source: the `wikimedia/wikipedia` dataset on the Hugging Face hub (the 2023-11-01 dumps, already stripped of
markup), pinned to one revision. Licence of the text: CC BY-SA 3.0 and GFDL (Wikipedia). The slice is exact
and reproducible: the first shard of each language, articles in file order, paragraphs of at least
--min-chars characters, until --mb megabytes (UTF-8) per language are written.

Output (data/cache/ is ignored by git; nothing downloaded is ever committed):

    data/cache/wiki/<shard>.parquet     the downloaded shards
    data/cache/wiki/wiki.pt.txt, wiki.en.txt   one article per line (its paragraphs joined by a space)
    data/cache/wiki/manifest.json       source, revision, licence, files, article and byte counts, SHA-256

pyarrow is only needed here (to read the parquet shards), which is why it is not a project dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

DATASET = "wikimedia/wikipedia"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SHARDS = {"pt": "20231101.pt/train-00000-of-00006.parquet", "en": "20231101.en/train-00000-of-00041.parquet"}
LICENCE = "CC BY-SA 3.0 and GFDL (Wikipedia text)"


def shard_path(out: Path, lang: str) -> Path:
    return out / SHARDS[lang].replace("/", "-")


def download(out: Path, lang: str) -> Path:
    path = shard_path(out, lang)
    if not path.exists():
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SHARDS[lang]}"
        print("downloading", url)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (fixed https URL)
        tmp.rename(path)
    return path


def paragraphs(text: str, min_chars: int) -> list[str]:
    """Prose paragraphs of an article: headings, list items and other short lines are dropped."""
    return [" ".join(p.split()) for p in text.split("\n") if len(p.strip()) >= min_chars]


def extract(parquet: Path, target: Path, max_bytes: int, min_chars: int) -> dict:
    import pyarrow.parquet as pq

    articles = written = 0
    digest = hashlib.sha256()
    with open(target, "w", encoding="utf-8") as f:
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=512, columns=["text"]):
            for text in batch["text"].to_pylist():
                paras = paragraphs(text, min_chars)
                if not paras:
                    continue
                line = (" ".join(paras) + "\n").encode("utf-8")
                f.write(line.decode("utf-8"))
                digest.update(line)
                written += len(line)
                articles += 1
                if written >= max_bytes:
                    break
            if written >= max_bytes:
                break
    return {"file": target.name, "articles": articles, "bytes": written, "sha256": digest.hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/cache/wiki")
    ap.add_argument("--mb", type=float, default=160.0, help="megabytes of text per language")
    ap.add_argument("--min-chars", type=int, default=120, help="shorter lines (headings, lists) are dropped")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": f"https://huggingface.co/datasets/{DATASET}", "revision": REVISION, "licence": LICENCE,
                "dump": "2023-11-01", "min_paragraph_chars": args.min_chars, "languages": {}}
    for lang in SHARDS:
        parquet = download(out, lang)
        stats = extract(parquet, out / f"wiki.{lang}.txt", int(args.mb * 1e6), args.min_chars)
        manifest["languages"][lang] = {"shard": SHARDS[lang], "shard_bytes": parquet.stat().st_size, **stats}
        print(lang, stats)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print("wrote", out / "manifest.json")


if __name__ == "__main__":
    main()
