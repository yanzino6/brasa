"""Prepare (and, only on request, upload) the Hugging Face release of a Brasa model.

    uv run --with safetensors --with huggingface_hub python scripts/publish_hf.py
    uv run --with safetensors --with huggingface_hub python scripts/publish_hf.py --forbidden words.txt
    uv run --with safetensors --with huggingface_hub python scripts/publish_hf.py --upload --yes

The default action builds a local folder (runs/hf-release/brasa-12m) from a release directory
(runs/final-12m, written by scripts/release.sh) and touches no network:

    model.safetensors   float32 weights of the released checkpoint (no optimizer state; the output
                        matrix is tied to the token embedding, so lm_head.weight is not stored twice)
    config.json         what is needed to rebuild the module, plus prompt format, special tokens, context length
    tokenizer.json      the byte-level BPE tokenizer
    retrieval.json      the BM25 index and the curated facts the model answers from (with fact vectors)
    web/model.i8.bin    the int8 bundle the browser engine loads
    web/model.f16.bin   the same weights as float16
    web/manifest.json   its manifest (configuration, sizes and checksums)
    model-card.json     the machine-readable evaluation
    README.md           front matter + the released model card (every number comes from model-card.json)

Then --verify (on by default): reload model.safetensors + config.json into a fresh model, compare its logits
with the original checkpoint, answer two questions end to end and run the README's own Python snippet.
--forbidden FILE aborts (exit 2) when a line of FILE occurs, case-insensitively, in any text file of the
folder; only the count and the file names are printed. --upload (needs --yes) pushes the folder with
huggingface_hub; nothing here ever runs it on its own.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # lets the script run from a checkout without installing the package

DEFAULT_RELEASE = "runs/final-12m"
DEFAULT_OUT = "runs/hf-release/brasa-12m"
DEFAULT_REPO = "Yanzino6/brasa-12m"
GITHUB_URL = "https://github.com/yanzino6/brasa"
QUESTIONS = ("Onde o Yan estuda?", "What is seer.app?")
SNIPPET_QUESTION = QUESTIONS[0]
TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml"}
LOGIT_TOLERANCE = 1e-6  # float32 round trip through safetensors: expected to be exactly 0
QUANT_DRIFT_BOUND = 1.0  # sanity bound on |int8 bundle - float32 checkpoint| logits: another checkpoint is far off

FILE_NOTES = {
    "model.safetensors": "float32 weights of the released checkpoint, tied output matrix stored once (rows labelled "
                         "float32 in the evaluation)",
    "config.json": "model configuration, prompt format, special tokens, context length",
    "tokenizer.json": "BPE merges and control tokens",
    "retrieval.json": "BM25 index, fact vectors and the curated facts the model answers from",
    "web/model.f16.bin": "float16 weights in the TGPT container (rows labelled float16 in the evaluation)",
    "web/model.i8.bin": "int8 weights, one scale per row: the bundle the browser engine loads (rows labelled int8)",
    "web/manifest.json": "manifest of the browser bundle (configuration, sizes, checksums)",
    "model-card.json": "machine-readable evaluation (every number of this card comes from it)",
}


def fail(message: str, code: int = 1) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def need(module: str) -> None:
    try:
        __import__(module)
    except ModuleNotFoundError:
        fail(f"the Python module '{module}' is missing; run this script as: "
             "uv run --with safetensors --with huggingface_hub python scripts/publish_hf.py ...")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def listing(folder: Path) -> list[tuple[str, int]]:
    return sorted((p.relative_to(folder).as_posix(), p.stat().st_size) for p in folder.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------------------------- model files
def load_model(folder: Path):
    """(model, config.json) of a release folder: model.safetensors + config.json into a fresh GPT, eval mode."""
    from safetensors.torch import load_file

    from brasa.config import GPTConfig
    from brasa.model import GPT

    config = read_json(folder / "config.json")
    model = GPT(GPTConfig(**{k: v for k, v in config["model"].items() if k != "params"}))
    state = load_file(folder / "model.safetensors")
    state["lm_head.weight"] = state["tok_emb.weight"]  # tied embeddings
    model.load_state_dict(state)
    return model.eval(), config


def checkpoint_path(info: dict, root: Path) -> Path:
    return root / info["run"] / f"ckpt_{info['checkpoint']}.pt"


def build(release: Path, out: Path, root: Path, *, repo_id: str, demo_url: str | None) -> dict:
    from safetensors.torch import save_file

    from brasa.config import GPTConfig
    from brasa.train import load_weights

    info = read_json(release / "RELEASE.json")
    card = read_json(release / "model-card.json")
    export = release / "export"
    manifest = read_json(export / "manifest.json")
    ckpt_file = checkpoint_path(info, root)
    if not ckpt_file.exists():
        fail(f"the checkpoint {ckpt_file} named by RELEASE.json does not exist")
    for name in ("model.i8.bin", "model.f16.bin", "manifest.json", "tokenizer.json", "retrieval.json"):
        if not (export / name).exists():
            fail(f"{export / name} is missing")
    run_tokenizer = ckpt_file.parent / "tokenizer.json"
    if run_tokenizer.exists() and run_tokenizer.read_bytes() != (export / "tokenizer.json").read_bytes():
        fail("the tokenizer of the training run differs from the one in the release export")

    model, ckpt = load_weights(ckpt_file)
    params = model.num_params()
    cfg = GPTConfig(**{**{k: v for k, v in ckpt["config"].items() if k != "params"}, "dropout": 0.0}).to_dict()
    cfg["params"] = params
    if params != info["params"] or cfg != manifest["config"]:
        fail("the checkpoint does not match RELEASE.json / the export manifest (parameters or configuration)")

    if out.exists():
        if any(out.iterdir()) and not ((out / "config.json").exists() and (out / "model.safetensors").exists()):
            fail(f"{out} exists and is not a previous build of this script; choose another --out")
        shutil.rmtree(out)
    (out / "web").mkdir(parents=True)

    assert model.lm_head.weight is model.tok_emb.weight
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt", "name": manifest["name"]})
    config = {
        "model_type": "brasa-gpt",
        "name": manifest["name"],
        "architecture": "decoder-only transformer: RMSNorm (pre-norm), rotary position embeddings, SwiGLU, "
                        "tied embeddings, no biases",
        "model": cfg,
        "context_length": cfg["block_size"],
        "tied_embeddings": True,
        "weights": {"file": "model.safetensors", "dtype": "float32",
                    "note": "lm_head.weight is tied to tok_emb.weight and not stored twice"},
        "prompt": manifest["prompt"],
        "special_tokens": manifest["special_tokens"],
        "decoding": {"constrain": "retrieved", "prior": manifest.get("decoding_prior")},
        "retrieval": {"file": "retrieval.json", "dense": manifest.get("dense_retrieval")},
    }
    (out / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copyfile(export / "tokenizer.json", out / "tokenizer.json")
    shutil.copyfile(export / "retrieval.json", out / "retrieval.json")
    shutil.copyfile(export / "model.i8.bin", out / "web" / "model.i8.bin")
    shutil.copyfile(export / "model.f16.bin", out / "web" / "model.f16.bin")
    shutil.copyfile(export / "manifest.json", out / "web" / "manifest.json")
    shutil.copyfile(release / "model-card.json", out / "model-card.json")
    readme = render_readme((release / "MODEL_CARD.md").read_text(encoding="utf-8"), card, out, repo_id, demo_url)
    (out / "README.md").write_text(readme, encoding="utf-8")
    return {"params": params}


# ---------------------------------------------------------------------------------------------- README
def yaml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)  # a JSON string is a valid YAML scalar


def model_index(card: dict) -> list[str]:
    """YAML lines of a model-index block, filled from model-card.json; [] when the card has no such numbers."""
    name = card.get("name")
    split = ((((card.get("evaluation") or {}).get("float32") or {}).get("headline") or {})
             .get("open_book_ranked") or {}).get("final")
    if not name or not split or split.get("em") is None:
        return []
    lines = ["model-index:", f"  - name: {yaml_str(name)}", "    results:",
             "      - task:", "          type: question-answering", "          name: Question answering (open-book)",
             "        dataset:", f"          name: {yaml_str('Brasa held-out questions, final split (read once)')}",
             "          type: brasa-qa-final", "        metrics:"]
    for key, label in (("em", "exact_match"), ("f1", "f1")):
        if split.get(key) is not None:
            name = f"{label} (%, float32, open-book, retrieved facts + ranking prior)"
            lines += [f"          - type: {label}", f"            value: {round(100 * split[key], 1)}",
                      f"            name: {yaml_str(name)}"]
    return lines


def front_matter(card: dict) -> str:
    tags = ["from-scratch", "gpt", "small-language-model", "question-answering", "retrieval-augmented",
            "retrieval-augmented-generation", "webassembly"]
    lines = ["---", "license: mit", "language:", "  - pt", "  - en", "library_name: pytorch",
             "pipeline_tag: text-generation", "tags:", *[f"  - {t}" for t in tags], *model_index(card), "---"]
    return "\n".join(lines) + "\n"


def replace_section(md: str, heading: str, body: str) -> str:
    """Replace the '## heading' section (up to the next '## ' heading) with body; append it when absent."""
    pattern = re.compile(rf"^## {re.escape(heading)}\n.*?(?=^## |\Z)", re.S | re.M)
    if pattern.search(md):
        return pattern.sub(lambda _: body.rstrip("\n") + "\n\n", md, count=1)
    return md.rstrip("\n") + "\n\n" + body.rstrip("\n") + "\n"


def usage_snippet(repo_id: str) -> str:
    return f'''import json

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from brasa.config import GPTConfig, PromptConfig
from brasa.encoder import DenseRetriever
from brasa.model import GPT
from brasa.retrieval import Bm25Retriever
from brasa.sample import answer
from brasa.tokenizer import BPETokenizer

folder = snapshot_download("{repo_id}")  # or the path of a local copy of this repository
config = json.load(open(f"{{folder}}/config.json"))

model = GPT(GPTConfig(**{{k: v for k, v in config["model"].items() if k != "params"}}))
state = load_file(f"{{folder}}/model.safetensors")
state["lm_head.weight"] = state["tok_emb.weight"]  # tied embeddings
model.load_state_dict(state)
model.eval()

tok = BPETokenizer.load(f"{{folder}}/tokenizer.json")
index = Bm25Retriever.load(f"{{folder}}/retrieval.json")
retriever = DenseRetriever.from_index(model, tok, index)  # BM25 fused with the model's own ranking, as on the site

# Retrieval, open-book prompt, then greedy decoding restricted to the retrieved facts and the fixed answers,
# with the retriever's ranking as a prior (the prior stored in retrieval.json): the site's decoding.
out = answer(model, tok, retriever, "{SNIPPET_QUESTION}", PromptConfig(**config["prompt"]), constrain="retrieved")
print(out["answer"])
for hit in out["hits"]:
    print(f"  [{{hit.score:.2f}}] {{hit.text}}")
'''


def how_to_use(card_md: str, repo_id: str) -> str:
    js = re.search(r"```js\n.*?```", card_md, re.S)
    browser = ""
    if js:
        browser = (
            "\nIn the browser, with the engine from the code repository (no dependencies, WebAssembly SIMD with a "
            "plain JavaScript fallback); the int8 bundle is `web/model.i8.bin`:\n\n"
            + js.group(0).replace("'model.i8.bin'", "'web/model.i8.bin'") + "\n"
        )
    return f"""## How to use

In Python, with the `brasa` package from the code repository ({GITHUB_URL}) and PyTorch:

```sh
pip install "git+{GITHUB_URL}" safetensors huggingface_hub
```

```python
{usage_snippet(repo_id).rstrip()}
```
{browser}"""


def files_section(folder: Path) -> str:
    rows = ["| File | Bytes | What |", "| --- | ---: | --- |"]
    rows += [f"| `{name}` | {size:,} | {FILE_NOTES[name]} |" for name, size in listing(folder) if name in FILE_NOTES]
    return ("## Files\n\n" + "\n".join(rows) + "\n\nThe evaluation tables below have a Weights column: `float32` "
            "is `model.safetensors`, `float16` is `web/model.f16.bin` and `int8` is `web/model.i8.bin`, each "
            "evaluated as stored (the quantised files de-quantised to float32 for scoring). The two `web/*.bin` "
            "files use the single-file \"TGPT\" container described in the code repository (JSON header with the "
            "configuration, the prompt format and the tokenizer, followed by the tensors); int8 stores every matrix "
            "with one scale per output row.\n")


def render_readme(card_md: str, card: dict, folder: Path, repo_id: str, demo_url: str | None) -> str:
    body = re.sub(r"\A---\n.*?\n---\n", "", card_md, count=1, flags=re.S)  # the card's own front matter
    body = re.sub(r"<!-- Rendered by .*?-->\n*", "", body, flags=re.S)
    links = [f"**Code:** {GITHUB_URL}"]
    if demo_url:
        links.append(f"**Live demo:** {demo_url}")
    body = re.sub(r"^(# [^\n]*\n)\n*", lambda m: m.group(1) + "\n" + "  \n".join(links) + "\n\n", body.lstrip("\n"),
                  count=1, flags=re.M)
    body = replace_section(body, "How to use", how_to_use(card_md, repo_id))
    body = replace_section(body, "Files", files_section(folder))
    return front_matter(card) + "\n" + body


# ---------------------------------------------------------------------------------------------- forbidden
def fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


BLOB = re.compile(r"[A-Za-z0-9+/=_-]{256,}")


def json_strings(node) -> list[str]:
    """Every key and string value of a JSON document, except long base64-looking blobs."""
    if isinstance(node, str):
        return [] if BLOB.fullmatch(node) else [node]
    if isinstance(node, dict):
        return [s for k, v in node.items() for s in (*json_strings(str(k)), *json_strings(v))]
    if isinstance(node, list):
        return [s for v in node for s in json_strings(v)]
    return []


def scan_forbidden(folder: Path, forbidden_file: Path) -> tuple[int, list[str]]:
    """(number of forbidden strings found, names of the files they occur in). Never returns the strings."""
    words = [w.strip() for w in forbidden_file.read_text(encoding="utf-8").splitlines() if w.strip()]
    found: set[int] = set()
    files: list[str] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() == ".json":
            # Look at the decoded text (json escapes non-ASCII letters) and leave out the base64 blobs of
            # vectors and weights: a short string would match inside them by chance.
            with contextlib.suppress(ValueError):
                text = "\n".join(json_strings(json.loads(text)))
        views = [text.lower(), fold(text)]
        hit = False
        for i, word in enumerate(words):
            needles = {word.lower(), fold(word)}
            if any(n in v for n in needles for v in views):
                found.add(i)
                hit = True
        if hit:
            files.append(path.relative_to(folder).as_posix())
    return len(found), files


# ---------------------------------------------------------------------------------------------- verify
def fused_retriever(model, tok, folder: Path):
    from brasa.encoder import DenseRetriever
    from brasa.retrieval import Bm25Retriever

    return DenseRetriever.from_index(model, tok, Bm25Retriever.load(folder / "retrieval.json"))


def verify(out: Path, release: Path, root: Path, questions=QUESTIONS, *, repo_id: str = DEFAULT_REPO) -> dict:
    import torch

    from brasa.config import PromptConfig
    from brasa.data import build_prompt
    from brasa.export import read_tgpt
    from brasa.sample import answer
    from brasa.tokenizer import BPETokenizer
    from brasa.train import load_weights

    original, _ = load_weights(checkpoint_path(read_json(release / "RELEASE.json"), root))
    model, config = load_model(out)
    tok = BPETokenizer.load(out / "tokenizer.json")
    prompt_cfg = PromptConfig(**config["prompt"])
    retriever, reference = fused_retriever(model, tok, out), fused_retriever(original, tok, out)

    hits = retriever.search(questions[0], prompt_cfg.max_facts)
    ids, _ = build_prompt(tok, questions[0], [h.text for h in hits], model.cfg.block_size, prompt_cfg)
    with torch.no_grad():
        theirs = original(torch.tensor([ids]))[0]
        diff = float((model(torch.tensor([ids]))[0] - theirs).abs().max())
        drift = float((read_tgpt(out / "web" / "model.i8.bin")[1](torch.tensor([ids]))[0] - theirs).abs().max())
        drift16 = float((read_tgpt(out / "web" / "model.f16.bin")[1](torch.tensor([ids]))[0] - theirs).abs().max())
    if diff > LOGIT_TOLERANCE:
        fail(f"logits of the reloaded model differ from the checkpoint by {diff:.3g}")
    if drift > QUANT_DRIFT_BOUND:
        fail(f"web/model.i8.bin is {drift:.3g} logits away from the checkpoint: is it from another checkpoint?")
    print(f"logits on a fixed prompt ({len(ids)} tokens): max |reloaded - checkpoint| = {diff:.3g}"
          f" (tolerance {LOGIT_TOLERANCE:g})")
    print(f"max |web/model.f16.bin - checkpoint| on the same prompt = {drift16:.4f}")
    print(f"max |web/model.i8.bin - checkpoint| on the same prompt = {drift:.4f} "
          f"(quantisation; bound {QUANT_DRIFT_BOUND})")

    answers = {}
    for q in questions:
        a = answer(model, tok, retriever, q, prompt_cfg, constrain="retrieved")["answer"]
        if a != answer(original, tok, reference, q, prompt_cfg, constrain="retrieved")["answer"]:
            fail(f"the reloaded model answers {q!r} differently from the checkpoint")
        print(f"> {q}\n  {a}")
        answers[q] = a

    block = re.search(r"## How to use\n.*?```python\n(.*?)```", (out / "README.md").read_text(encoding="utf-8"), re.S)
    if not block:
        fail("README.md has no Python snippet under 'How to use'")
    code = block.group(1).replace(f'snapshot_download("{repo_id}")', repr(str(out)))
    code = code.replace("from huggingface_hub import snapshot_download\n", "")
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(code, "README.md:snippet", "exec"), {"__name__": "__readme__"})  # noqa: S102
    expected = answer(model, tok, retriever, SNIPPET_QUESTION, prompt_cfg, constrain="retrieved")["answer"]
    if captured.getvalue().splitlines()[0] != expected:
        fail("the README snippet prints a different answer than the verified one")
    print("README snippet: runs and prints the same answer as above")
    return {"max_abs_logit_diff": diff, "int8_drift": drift, "f16_drift": drift16, "answers": answers}


# ---------------------------------------------------------------------------------------------- upload
def upload(folder: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model", commit_message="Release brasa-12m")
    print(f"uploaded {folder} to https://huggingface.co/{repo_id}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build (and optionally upload) the Hugging Face release folder.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--release", default=DEFAULT_RELEASE, help="release directory written by scripts/release.sh")
    ap.add_argument("--out", default=DEFAULT_OUT, help="folder to build")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--demo-url", help="adds a 'Live demo' line to the README")
    ap.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                    help="reload the folder and compare it with the checkpoint, answer two questions")
    ap.add_argument("--forbidden", help="text file, one string per line: abort if any occurs in the folder")
    ap.add_argument("--no-build", action="store_true", help="use the existing --out folder as it is")
    ap.add_argument("--upload", action="store_true", help="push the folder to the Hub (needs --yes)")
    ap.add_argument("--yes", action="store_true", help="confirm --upload")
    args = ap.parse_args(argv)

    need("safetensors")
    if args.upload:
        need("huggingface_hub")
        if not args.yes:
            fail("--upload needs --yes: it creates the repository if needed and publishes the folder", 2)
    release, out = (Path(p) if Path(p).is_absolute() else ROOT / p for p in (args.release, args.out))
    if args.forbidden and not Path(args.forbidden).exists():
        fail(f"--forbidden file {args.forbidden} not found")
    if not release.is_dir():
        fail(f"release directory {release} not found")

    if args.no_build:
        if not out.is_dir():
            fail(f"{out} does not exist")
    else:
        result = build(release, out, ROOT, repo_id=args.repo_id, demo_url=args.demo_url)
        print(f"built {out} ({result['params']:,} parameters)")
    for name, size in listing(out):
        print(f"  {size:>12,}  {name}")
    if args.forbidden:
        count, names = scan_forbidden(out, Path(args.forbidden))
        if count:
            print(f"forbidden scan: {count} forbidden string(s) found in: {', '.join(names)}")
            return 2
        print("forbidden scan: clean")
    if args.verify:
        verify(out, release, ROOT, repo_id=args.repo_id)
    if args.upload:
        upload(out, args.repo_id)
    else:
        print("not uploaded (pass --upload --yes to publish)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
