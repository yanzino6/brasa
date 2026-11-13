"""scripts/publish_hf.py: the Hugging Face release folder is built from a release directory, reloads to the same
logits, carries a README whose snippet runs, and the forbidden-string scan reports counts and file names only."""

from __future__ import annotations

import importlib.util
import json
import shutil

import pytest
import torch
from conftest import ML_ROOT, TINY

pytest.importorskip("safetensors")

from brasa.export import read_tgpt  # noqa: E402

spec = importlib.util.spec_from_file_location("publish_hf", ML_ROOT / "scripts" / "publish_hf.py")
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)

CARD = """---
license: mit
---

<!-- Rendered by scripts/render_model_card.py from model-card.json. -->

# tiny



Intro.

## Files

| File | Bytes |
| --- | --- |
| `model.f16.bin` | 1 |

## How to use

old text

```js
await client.load({ model: 'model.i8.bin', retrieval: 'retrieval.json' });
```

## Limitations

None.
"""


@pytest.fixture()
def release(tmp_path):
    """A release directory (and the run it points at) made from the committed tiny export."""
    header, model = read_tgpt(TINY / "model.i8.bin")
    run = tmp_path / "run"
    run.mkdir()
    torch.save({"model": model.state_dict(), "config": header["config"]}, run / "ckpt_last.pt")
    shutil.copyfile(TINY / "tokenizer.json", run / "tokenizer.json")
    rel = tmp_path / "release"
    shutil.copytree(TINY, rel / "export")
    manifest = json.loads((TINY / "manifest.json").read_text(encoding="utf-8"))
    (rel / "RELEASE.json").write_text(json.dumps({"run": "run", "checkpoint": "last", "params": model.num_params()}))
    final = {"em": 0.5, "f1": 0.75}
    card = {"name": "tiny", "evaluation": {"float32": {"headline": {"open_book_ranked": {"final": final}}}}}
    (rel / "model-card.json").write_text(json.dumps(card))
    (rel / "MODEL_CARD.md").write_text(CARD, encoding="utf-8")
    assert manifest["config"]["params"] == model.num_params()
    return rel


def test_the_folder_reloads_to_the_same_logits_and_the_readme_snippet_runs(release, tmp_path, capsys):
    out = tmp_path / "hf"
    publish.build(release, out, tmp_path, repo_id="user/tiny", demo_url=None)
    assert sorted(p for p, _ in publish.listing(out)) == [
        "README.md", "config.json", "model-card.json", "model.safetensors", "retrieval.json", "tokenizer.json",
        "web/manifest.json", "web/model.f16.bin", "web/model.i8.bin"]
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\nlicense: mit\n") and "Rendered by" not in readme
    assert "value: 50.0" in readme and "value: 75.0" in readme, "model-index comes from model-card.json"
    assert "Live demo" not in readme and "old text" not in readme
    assert "| `model.f16.bin` | 1 |" not in readme and "| `web/model.f16.bin` |" in readme
    assert "'web/model.i8.bin'" in readme and readme.count("## How to use") == 1
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["context_length"] == config["model"]["block_size"] and "<|a|>" in config["special_tokens"]

    result = publish.verify(out, release, tmp_path, ("Quais linguagens de programação ele usa?",),
                            repo_id="user/tiny")
    # the snippet asks its own fixed question: it only has to run and agree with the verified model
    assert result["max_abs_logit_diff"] == 0.0 and len(result["answers"]) == 1
    assert "README snippet: runs" in capsys.readouterr().out


def test_a_demo_url_adds_one_line(release, tmp_path):
    publish.build(release, tmp_path / "hf", tmp_path, repo_id="user/tiny", demo_url="https://example.org/demo")
    assert "**Live demo:** https://example.org/demo" in (tmp_path / "hf" / "README.md").read_text(encoding="utf-8")


def test_the_forbidden_scan_is_case_and_accent_insensitive_and_skips_vector_blobs(tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "a.md").write_text("Nada aqui. Informação sensível.", encoding="utf-8")
    (folder / "b.json").write_text(json.dumps({"x": "Fine", "blob": "abc" * 100, "k": "SÃO paulo"}))
    (folder / "c.bin").write_bytes(b"secret")
    words = tmp_path / "words.txt"
    words.write_text("\ninformacao SENSIVEL\nsao paulo\nabc\nsecret\nabsent\n", encoding="utf-8")
    count, names = publish.scan_forbidden(folder, words)
    assert (count, names) == (2, ["a.md", "b.json"]), "binary files are not text; a base64-like blob is not scanned"
