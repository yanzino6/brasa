"""A learned retriever: a dual encoder trained from scratch on (training question, fact) pairs.

The lexical retriever only finds a fact when the question shares words or character trigrams with it, its
topic names or the questions already asked about it. This module learns the match instead: one small
transformer (the SAME code as the language model, model.GPT, read at its last layer instead of its output
head) turns a question and a fact into two vectors, and is trained so that a question lands next to the
fact that answers it.

    question   <|bos|><|q|> question <|a|>        fact   <|bos|><|ctx|> fact <|sep|>
    vector = mean over positions of the final hidden states (or the last one), L2-normalised

Training (InfoNCE): every question of a batch is contrasted against the gold facts of the other questions
(in-batch negatives) and against one HARD negative per question, another fact of the same topic, which is
the mistake the lexical retriever makes most. Facts that are also gold for the row are masked out.
Optionally (mined_negatives) a question also gets facts MINED for it: the wrong facts the lexical retriever
ranks highest for that very question (mine_negatives), i.e. the ones a fused ranking has to tell apart.
Only training questions are used; the best checkpoint is chosen on the validation questions (recall@4).

At inference time the fact vectors are precomputed and a question costs one forward pass of a few dozen
tokens. DenseRetriever implements the Retriever protocol, alone or FUSED with BM25 (the lexical ranking
stays available as the fallback): score = bm25 / (best bm25 of the query) + weight * cosine.

    uv run python -m brasa.encoder --preset brasa-4m --data-dir data/yan --out runs/enc-4m --steps 3000
    uv run python -m brasa.encoder --eval runs/enc-4m          # recall@1/2/4 alone and fused, val and test
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import BOS, CTX, DEFAULT_VOCAB_SIZE, PAD, PRESETS, SEP, A, GPTConfig, Q, preset
from .data import Dataset, add_noise, load_dataset, tokenizer_texts
from .model import GPT
from .retrieval import LAYOUTS, Bm25Retriever, Hit
from .tokenizer import BPETokenizer
from .train import (
    DEFAULT_LR,
    finite_state,
    lr_at,
    make_optimizer,
    make_scaler,
    optimizer_step,
    pick_device,
    require_finite,
    save_checkpoint,
)

POOLINGS = ("mean", "last")
MAX_TOKENS = 128  # questions and facts longer than this are truncated (the longest fact has about 90 tokens)


@dataclass
class EncoderConfig:
    preset: str = "brasa-4m"
    data_dir: str = "data"
    out: str = "runs/encoder"
    steps: int = 3000
    batch: int = 256  # questions per step (each brings its gold fact and one hard negative)
    lr: float | None = None
    min_lr_ratio: float = 0.1
    warmup: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dropout: float = 0.1
    temperature: float = 0.05
    pooling: str = "mean"
    hard_negatives: int = 1  # same-topic facts added per question (0 = in-batch negatives only)
    mined_negatives: int = 0  # facts mined for the question itself (the retriever's best wrong hits) added per question
    mined_top: int = 8  # how many of the retriever's wrong hits are kept as candidates per question
    noise_prob: float = 0.5  # share of the questions that get surface noise (case, accents, typos)
    index: str = "fields"  # layout of the BM25 index used in the fused evaluation
    vocab_size: int = DEFAULT_VOCAB_SIZE
    seed: int = 0
    device: str = "auto"
    amp: str = "auto"
    log_every: int = 50
    eval_every: int = 250
    init_from: str | None = None  # a brasa.pretrain run: start from its weights and use its tokenizer
    tokenizer: str | None = None


# --------------------------------------------------------------------------------------- the encoder
def question_ids(tok: BPETokenizer, text: str) -> list[int]:
    s = tok.special_ids
    return [s[BOS], s[Q], *tok.encode(text)[: MAX_TOKENS - 3], s[A]]


def fact_ids(tok: BPETokenizer, text: str) -> list[int]:
    s = tok.special_ids
    return [s[BOS], s[CTX], *tok.encode(text)[: MAX_TOKENS - 3], s[SEP]]


def pool(hidden: torch.Tensor, lengths: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
    """(B, T, C) final hidden states of right-padded rows -> (B, C) unit vectors."""
    if pooling == "last":
        v = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths - 1]
    else:
        mask = (torch.arange(hidden.size(1), device=hidden.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        v = (hidden * mask).sum(1) / lengths[:, None]
    return F.normalize(v.float(), dim=-1)


def embed(model: GPT, sequences: list[list[int]], pad: int, device, pooling: str = "mean",
          max_tokens: int = 32768) -> torch.Tensor:
    """Unit vectors of many token sequences, (N, d_model), in the order given (gradients flow when training)."""
    out: list[torch.Tensor | None] = [None] * len(sequences)
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and (end - start + 1) * len(sequences[order[end]]) <= max_tokens:
            end += 1
        rows = order[start:end]
        width = len(sequences[rows[-1]])
        buf = torch.full((len(rows), width), pad, dtype=torch.long)
        for r, i in enumerate(rows):
            buf[r, : len(sequences[i])] = torch.tensor(sequences[i])
        lengths = torch.tensor([len(sequences[i]) for i in rows], device=device)
        vectors = pool(model.hidden(buf.to(device)), lengths, pooling)
        for r, i in enumerate(rows):
            out[i] = vectors[r]
        start = end
    return torch.stack(out)  # type: ignore[arg-type]


def contrastive_loss(q: torch.Tensor, d: torch.Tensor, positive: torch.Tensor, false_negative: torch.Tensor,
                     temperature: float) -> torch.Tensor:
    """InfoNCE. q: (B, C) questions, d: (M, C) documents, positive[i] = the column of row i's gold document,
    false_negative: (B, M) bool, True where a column is also a right answer for the row (never the positive)."""
    logits = (q @ d.T) / temperature
    logits = logits.masked_fill(false_negative, float("-inf"))
    return F.cross_entropy(logits, positive)


# --------------------------------------------------------------------------------------- data
def training_pairs(dataset: Dataset) -> list[tuple[str, str, tuple[str, ...], str]]:
    """(question, gold fact id, all gold fact ids, language) for every training question with facts."""
    return [(r.q, f, r.facts, r.lang) for r in dataset.split("train") for f in r.facts]


def same_topic(dataset: Dataset) -> dict[str, list[str]]:
    by_topic: dict[str, list[str]] = {}
    for f in dataset.facts.values():
        by_topic.setdefault(f.topic, []).append(f.id)
    return {f.id: [g for g in by_topic[f.topic] if g != f.id] for f in dataset.facts.values()}


def mine_negatives(dataset: Dataset, retriever, top: int = 8) -> dict[tuple[str, str], list[str]]:
    """{(question, language): the `top` best facts of `retriever` for that training question that are NOT gold}.

    With the lexical index, which holds the training questions, the gold fact comes first and what follows is
    what shares the most wording with the question: the confusions a learned ranking is there to resolve."""
    mined: dict[tuple[str, str], list[str]] = {}
    for r in dataset.split("train"):
        if r.facts and (r.q, r.lang) not in mined:
            hits = retriever.search(r.q, top + len(r.facts))
            mined[(r.q, r.lang)] = [h.id for h in hits if h.id not in r.facts][:top]
    return mined


def make_batch(pairs, dataset: Dataset, siblings: dict[str, list[str]], tok: BPETokenizer, cfg: EncoderConfig,
               step: int, mined: dict[tuple[str, str], list[str]] | None = None):
    """(question ids, document ids, positive column per row, false-negative mask): pure function of (seed, step).
    mined (mine_negatives) adds cfg.mined_negatives of each question's own hard negatives to the documents."""
    rng = random.Random(cfg.seed * 1_000_003 + step)
    rows = rng.sample(pairs, min(cfg.batch, len(pairs)))
    questions, docs, doc_fact = [], [], []
    for q, fact, _, lang in rows:
        questions.append(question_ids(tok, add_noise(q, rng) if rng.random() < cfg.noise_prob else q))
        docs.append(fact_ids(tok, dataset.facts[fact].text(lang)))
        doc_fact.append(fact)
    for _, fact, gold, lang in rows:
        pool_ = [f for f in siblings[fact] if f not in gold]
        for neg in rng.sample(pool_, min(cfg.hard_negatives, len(pool_))):
            docs.append(fact_ids(tok, dataset.facts[neg].text(lang)))
            doc_fact.append(neg)
    if mined and cfg.mined_negatives > 0:
        for q, _, gold, lang in rows:
            pool_ = [f for f in mined.get((q, lang), []) if f not in gold]
            for neg in rng.sample(pool_, min(cfg.mined_negatives, len(pool_))):
                docs.append(fact_ids(tok, dataset.facts[neg].text(lang)))
                doc_fact.append(neg)
    positive = torch.arange(len(rows))
    false_negative = torch.tensor([[doc_fact[j] in gold and j != i for j in range(len(docs))]
                                   for i, (_, _, gold, _) in enumerate(rows)])
    return questions, docs, positive, false_negative


# --------------------------------------------------------------------------------------- retrieval
class DenseRetriever:
    """Retriever protocol on top of a network read as an encoder.

    Alone (weight=None) it ranks by cosine. With `lexical` (a Bm25Retriever over the same documents) and a
    weight, the document vectors are packed into the index exactly as they are shipped (int8, see
    retrieval.pack_dense) and every search is the index's own fusion, bm25 / (best bm25) + weight * cosine:
    what is measured here is what the browser computes. `index` is that index, vectors included."""

    def __init__(self, model: GPT, tok: BPETokenizer, docs: list[dict], pooling: str = "mean",
                 lexical: Bm25Retriever | None = None, weight: float | None = None, device="cpu",
                 vectors: torch.Tensor | None = None):
        self.model, self.tok, self.docs, self.pooling, self.device = model.eval(), tok, docs, pooling, device
        self.weight = weight
        self.pad = tok.special_ids[PAD]
        if vectors is None:
            with torch.no_grad():
                vectors = embed(model, [fact_ids(tok, d["text"]) for d in docs], self.pad, device, pooling)
        self.vectors = vectors.to(device)
        self.fused = None
        if lexical is not None and weight is not None:
            self.fused = lexical.with_dense(self.vectors.float().cpu().numpy(), weight, pooling, MAX_TOKENS)
        self.index = self.fused.index if self.fused else getattr(lexical, "index", None)
        self._cache: dict[str, torch.Tensor] = {}

    @classmethod
    def from_index(cls, model: GPT, tok: BPETokenizer, lexical: Bm25Retriever, device="cpu") -> DenseRetriever:
        """The retriever of an EXPORTED index that carries fact vectors (retrieval.json "dense"): its stored
        vectors, pooling and weight are used as they are, and `model` only embeds the questions, which is
        what the browser does."""
        if lexical.dense is None:
            raise ValueError("the retrieval index has no fact vectors")
        self = cls.__new__(cls)
        self.model, self.tok, self.docs, self.device = model.eval(), tok, lexical.docs, device
        self.pooling, self.weight = lexical.dense["pooling"], lexical.dense["weight"]
        self.pad = tok.special_ids[PAD]
        self.vectors, self.fused, self.index, self._cache = None, lexical, lexical.index, {}
        return self

    @torch.no_grad()
    def encode_queries(self, queries: list[str]) -> None:
        """Embed many queries at once (an evaluation asks thousands); search() then finds them cached."""
        todo = [q for q in dict.fromkeys(queries) if q not in self._cache]
        if todo:
            vectors = embed(self.model, [question_ids(self.tok, q) for q in todo], self.pad, self.device, self.pooling)
            self._cache.update(zip(todo, vectors, strict=True))

    def vector(self, query: str) -> torch.Tensor:
        self.encode_queries([query])
        return self._cache[query]

    def scores(self, query: str) -> list[float]:
        if self.fused is None:
            return (self.vectors @ self.vector(query)).tolist()
        return self.fused.scores(query, self.vector(query).float().cpu().numpy())

    def search(self, query: str, k: int = 4, lang: str | None = None) -> list[Hit]:
        if self.fused is not None:
            return self.fused.search(query, k, lang, self.vector(query).float().cpu().numpy())
        scores = self.scores(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        hits, seen = [], set()
        for i in order:
            d = self.docs[i]
            if d["id"] in seen or (lang is not None and d["lang"] != lang):
                continue
            seen.add(d["id"])
            hits.append(Hit(d["id"], d["lang"], scores[i], d["text"], d["topic"], i))
            if len(hits) == k:
                break
        return hits


def recall(dataset: Dataset, retriever, split: str, ks=(1, 2, 4, 8)) -> dict:
    """Share of the questions of a split whose gold facts are ALL among the first k hits."""
    records = [r for r in dataset.split(split) if r.facts]
    if hasattr(retriever, "encode_queries"):
        retriever.encode_queries([r.q for r in records])
    hits = {k: 0 for k in ks}
    for r in records:
        found = [h.id for h in retriever.search(r.q, max(ks))]
        for k in ks:
            hits[k] += set(r.facts) <= set(found[:k])
    return {"n": len(records), **{f"recall@{k}": round(hits[k] / max(1, len(records)), 4) for k in ks}}


# --------------------------------------------------------------------------------------- training
def load_encoder(run_dir: str | Path, device="cpu", checkpoint: str = "best"):
    """(model, tokenizer, run.json contents) of an encoder run."""
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / f"encoder_{checkpoint}.pt", map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in ckpt["config"].items() if k != "params"}
    model = GPT(GPTConfig(**{**cfg, "dropout": 0.0}))
    model.load_state_dict(ckpt["model"])
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if (run_dir / "run.json").exists() else {}
    return model.to(device).eval(), BPETokenizer.load(run_dir / "tokenizer.json"), {**run, "checkpoint": ckpt}


def train(cfg: EncoderConfig) -> dict:
    if cfg.pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {cfg.pooling!r}; choose from {POOLINGS}")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    base_lr = cfg.lr if cfg.lr is not None else DEFAULT_LR.get(cfg.preset, 6e-4)
    torch.manual_seed(cfg.seed)
    dataset = load_dataset(cfg.data_dir)
    if cfg.init_from or cfg.tokenizer:
        tok = BPETokenizer.load(cfg.tokenizer or Path(cfg.init_from) / "tokenizer.json")
    else:
        tok = BPETokenizer.train(tokenizer_texts(dataset), cfg.vocab_size)
    tok.save(out / "tokenizer.json")
    pad = tok.special_ids[PAD]
    model = GPT(preset(cfg.preset, vocab_size=tok.vocab_size, block_size=MAX_TOKENS, dropout=cfg.dropout))
    if cfg.init_from:  # a language model pretrained on text: same blocks; its positions are relative (RoPE)
        state = torch.load(Path(cfg.init_from) / "ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(finite_state(state, cfg.init_from))
    model.to(device)
    opt = make_optimizer(model, cfg, base_lr)
    amp = cfg.amp if cfg.amp != "auto" else ("fp16" if device.type in ("mps", "cuda") else "off")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    scaler = make_scaler(device, amp)

    pairs = training_pairs(dataset)
    siblings = same_topic(dataset)
    mined = (mine_negatives(dataset, Bm25Retriever.from_dataset(dataset, layout=cfg.index), cfg.mined_top)
             if cfg.mined_negatives > 0 else None)
    docs = [{"id": f.id, "lang": lang, "topic": f.topic, "text": f.text(lang)}
            for f in dataset.facts.values() for lang in ("pt", "en")]
    info = {"event": "start", "preset": cfg.preset, "model": model.describe(), "device": str(device), "amp": amp,
            "lr": base_lr, "pairs": len(pairs), "documents": len(docs), "data": cfg.data_dir,
            "init_from": cfg.init_from, "torch": torch.__version__}
    log_file = open(out / "train_log.jsonl", "w", encoding="utf-8")  # noqa: SIM115

    def log(record: dict) -> None:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()
        print(json.dumps(record, ensure_ascii=False), flush=True)

    log(info)
    best, history, window, t0 = (-1.0, -1.0), [], [], time.time()
    model.train()
    try:
        for step in range(cfg.steps):
            lr = lr_at(step, cfg, base_lr)
            for group in opt.param_groups:
                group["lr"] = lr
            questions, documents, positive, false_negative = make_batch(pairs, dataset, siblings, tok, cfg, step,
                                                                        mined)

            def loss_of(questions=questions, documents=documents, positive=positive, false_negative=false_negative):
                q = embed(model, questions, pad, device, cfg.pooling)
                d = embed(model, documents, pad, device, cfg.pooling)
                return contrastive_loss(q, d, positive.to(device), false_negative.to(device), cfg.temperature)

            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    loss = loss_of()
            else:
                loss = loss_of()
            optimizer_step(model, opt, scaler, loss, cfg.grad_clip)
            window.append(float(loss.detach()))
            require_finite(window[-1], "the contrastive loss", step + 1)
            done = step + 1
            if done % cfg.log_every == 0 or done == cfg.steps:
                log({"step": done, "loss": round(sum(window) / len(window), 4), "lr": round(lr, 7),
                     "elapsed_s": round(time.time() - t0, 1)})
                window = []
            if done % cfg.eval_every == 0 or done == cfg.steps:
                model.eval()
                val = recall(dataset, DenseRetriever(model, tok, docs, cfg.pooling, device=device), "val")
                model.train()
                rec = {"step": done, **{f"val_{k}": v for k, v in val.items()}}
                history.append(rec)
                log(rec)
                score = (val["recall@4"], val["recall@1"])
                state = dict(model=model.state_dict(), step=done, config=model.describe(), encoder_config=asdict(cfg),
                             val=val)
                if score > best:
                    best = score
                    save_checkpoint(out / "encoder_best.pt", **state)
                save_checkpoint(out / "encoder_last.pt", **state)
    finally:
        log_file.close()
    summary = {**{k: v for k, v in info.items() if k != "event"}, "encoder_config": asdict(cfg), "steps": cfg.steps,
               "train_seconds": round(time.time() - t0, 1), "best_val": {"recall@4": best[0], "recall@1": best[1]},
               "final_val": history[-1] if history else None}
    (out / "run.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


FUSION_WEIGHTS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


def fusion_report(model: GPT, tok: BPETokenizer, dataset: Dataset, lexical: Bm25Retriever, pooling: str = "mean",
                  device="cpu", weights=FUSION_WEIGHTS, splits=("val", "test")) -> dict:
    """Recall of the lexical index, of the network read as an encoder, and of their fusion, on held-out
    questions. "weight" is the fusion weight with the best validation recall@4 (then recall@1); test numbers
    are reported for every weight but never used to choose."""
    was_training = model.training
    dense = DenseRetriever(model, tok, lexical.docs, pooling, device=device)
    out: dict = {"lexical": {}, "dense": {}, "fused": {}}
    for split in splits:
        out["lexical"][split] = recall(dataset, lexical, split)
        out["dense"][split] = recall(dataset, dense, split)
    for w in weights:
        fused = DenseRetriever(model, tok, lexical.docs, pooling, lexical, w, device, dense.vectors)
        fused._cache = dense._cache
        out["fused"][str(w)] = {split: recall(dataset, fused, split) for split in splits}
    if "val" in splits:
        out["weight"] = max(weights, key=lambda w: (out["fused"][str(w)]["val"]["recall@4"],
                                                    out["fused"][str(w)]["val"]["recall@1"]))
    model.train(was_training)
    return out


def evaluate(run_dir: str | Path, device="cpu", weights=FUSION_WEIGHTS, splits=("val", "test"),
             checkpoint: str = "best") -> dict:
    """fusion_report of a stand-alone encoder run (its best checkpoint) against the BM25 index of its data."""
    model, tok, run = load_encoder(run_dir, device, checkpoint)
    cfg = run["checkpoint"]["encoder_config"]
    dataset = load_dataset(cfg["data_dir"])
    lexical = Bm25Retriever.from_dataset(dataset, layout=cfg["index"])
    report = fusion_report(model, tok, dataset, lexical, cfg["pooling"], device, weights, splits)
    return {"checkpoint_step": run["checkpoint"]["step"], **report, "best_weight": report.get("weight")}


def parse_args(argv=None):
    d = EncoderConfig()
    ap = argparse.ArgumentParser(description="Train or evaluate the dual-encoder retriever.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--eval", metavar="RUN", help="evaluate a trained encoder run and exit")
    ap.add_argument("--preset", default=d.preset, choices=sorted(PRESETS))
    ap.add_argument("--data-dir", default=d.data_dir)
    ap.add_argument("--out", default=d.out)
    ap.add_argument("--steps", type=int, default=d.steps)
    ap.add_argument("--batch", type=int, default=d.batch)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=d.warmup)
    ap.add_argument("--dropout", type=float, default=d.dropout)
    ap.add_argument("--temperature", type=float, default=d.temperature)
    ap.add_argument("--pooling", default=d.pooling, choices=POOLINGS)
    ap.add_argument("--hard-negatives", type=int, default=d.hard_negatives)
    ap.add_argument("--mined-negatives", type=int, default=d.mined_negatives,
                    help="per question, this many of the lexical retriever's best wrong facts for it are added")
    ap.add_argument("--noise-prob", type=float, default=d.noise_prob)
    ap.add_argument("--index", default=d.index, choices=LAYOUTS)
    ap.add_argument("--vocab-size", type=int, default=d.vocab_size)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--device", default=d.device, choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--amp", default=d.amp, choices=["auto", "off", "fp16", "bf16"])
    ap.add_argument("--log-every", type=int, default=d.log_every)
    ap.add_argument("--eval-every", type=int, default=d.eval_every)
    ap.add_argument("--init-from", default=None, help="a brasa.pretrain run: its weights and its tokenizer")
    ap.add_argument("--tokenizer", default=None, help="use this tokenizer.json instead of training one")
    a = ap.parse_args(argv)
    cfg = EncoderConfig(**{k: v for k, v in vars(a).items() if k in EncoderConfig.__dataclass_fields__})
    return a.eval, cfg


def main(argv=None) -> None:
    eval_run, cfg = parse_args(argv)
    if eval_run:
        result = evaluate(eval_run, pick_device(cfg.device))
        (Path(eval_run) / "eval-retrieval.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps(result, indent=1))
        return
    summary = train(cfg)
    print(json.dumps({"event": "end", "train_seconds": summary["train_seconds"], "best_val": summary["best_val"]}))


if __name__ == "__main__":
    main()
