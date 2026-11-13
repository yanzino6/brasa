"""Cross-language fixtures: what the Python reference returns for a fixed set of inputs.

The tokenizer, the retriever and the prompt builder exist twice, in Python (training, evaluation) and in
JavaScript (the browser). They must agree exactly, otherwise the model would be shown prompts it was not
trained on. This module records the Python answers; web/engine/tools/parity.mjs replays the same inputs
through the JavaScript ports and compares.

    parity.json = {"tokenizer": [...], "features": [...], "retrieval": [...], "prompt": [...], "trie": [...],
                   "dense": [...]}      (the last one only for an index that carries fact vectors)

It is written into every export directory (export.py) and into tests/fixtures/tiny (scripts/make_fixtures.py).
"""

from __future__ import annotations

from .config import PromptConfig
from .constrain import AnswerTrie
from .data import add_noise, build_prompt
from .retrieval import query_features
from .tokenizer import BPETokenizer

# Inputs chosen to hit the places where two implementations usually drift apart.
TRICKY_TEXTS = [
    "Olá, você fala inglês?",
    "São Paulo é MAIOR que Vitória!",
    "ação AÇÃO Ação aÇãO",
    "naïve café déjà-vu, coração, pingüim",
    "emoji 🙂🔥 e bandeira 🇧🇷",
    "família 👨\u200d👩\u200d👧 com ZWJ",
    "  espaços   no começo,\ttabs\ne quebras de linha  ",
    "e\u0301 decomposto vira é",  # NFD input must be composed
    "<|q|>isto não é especial<|a|>",
    "<|bos|><|ctx|>fato<|sep|>outro<|q|>pergunta<|a|>resposta<|eos|>",
    "preço: R$ 1.234,56 (40%) em 2028-12",
    "日本語のテキスト and Ελληνικά",
    "a",
    "",
    "   ",
    "non\u00a0breaking\u2003wide\u3000spaces",
    "vertical\x0btab, form\x0cfeed, fs\x1csep",
    "MiXeD cAsE with UPPER and lower",
    "snake_case, kebab-case, camelCase, dotted.name@example.com",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "x²+y²=z² × ÷ ± ≠ …",
    "What's his e-mail? Is it hello@lumenforge.example?",
    "O Téo trabalhou com PyTorch, scikit-learn e ClickHouse?",
    "<|unknown|> <| q |> <|Q|>",
]

TRICKY_QUERIES = [
    "onde o teo mora",
    "Where does Téo live?",
    "ONDE O TÉO MORA???",
    "qual a linguaguem de programacao dele",  # typo and missing accents
    "pytorch",
    "PyTorch scikit-learn",
    "contato email",
    "how can i contact him",
    "quais projetos",
    "what did he build for fraud",
    "notas fiscais OCR",
    "zzzz qqqq",  # nothing matches
    "",
    "a",
    "kubernetes docker aws",
    "Sentinela Sentinela Sentinela",  # repeated words count once
    "é",
    "trabalha trabalhou trabalhando",
    "quantos anos de experiência ele tem?",
    "東京",
]


def tokenizer_cases(tok: BPETokenizer, texts: list[str]) -> list[dict]:
    cases = []
    for text in texts:
        for allow_special in (False, True):
            norm, tokens = tok.tokenize(text, allow_special=allow_special)
            ids = tok.encode(text, allow_special=allow_special)
            assert ids == [t["id"] for t in tokens]
            cases.append({
                "text": text, "allow_special": allow_special, "norm": norm, "ids": ids,
                "bounds": [[t["start"], t["end"]] for t in tokens],
                "decoded": tok.decode(ids), "decoded_with_specials": tok.decode(ids, skip_special=False),
            })
    return cases


def feature_cases(queries: list[str], ngram: int = 3) -> list[dict]:
    return [{"q": q, "features": query_features(q, ngram)} for q in queries]


def retrieval_cases(retriever, queries: list[str]) -> list[dict]:
    cases = []
    for i, q in enumerate(queries):
        for k, lang in ((4, None), (1, None), (8, ("pt", "en")[i % 2])):
            hits = retriever.search(q, k, lang)
            cases.append({"q": q, "k": k, "lang": lang,
                          "hits": [{"id": h.id, "lang": h.lang, "score": h.score, "doc": h.doc} for h in hits]})
    return cases


def dense_cases(retriever, queries: list[str], embedder) -> list[dict]:
    """Fused searches of an index with fact vectors: the vector of each question, as the Python encoder
    computes it, and what the index returns for it. Given that vector the JavaScript port has to return
    the same documents with the same scores, bit for bit (the dot products are integer sums)."""
    cases = []
    for i, q in enumerate(queries):
        vector = [float(v) for v in embedder(q)]
        searches = []
        for k, lang in ((4, None), (8, ("pt", "en")[i % 2])):
            hits = retriever.search(q, k, lang, vector)
            searches.append({"k": k, "lang": lang,
                             "hits": [{"id": h.id, "lang": h.lang, "score": h.score, "doc": h.doc} for h in hits]})
        cases.append({"q": q, "vector": vector, "searches": searches})
    return cases


def prompt_cases(tok: BPETokenizer, retriever, questions: list[str], block_size: int, cfg: PromptConfig) -> list[dict]:
    cases = []
    for i, q in enumerate(questions):
        facts = [h.text for h in retriever.search(q, cfg.max_facts + 2)] if retriever is not None and i % 3 else None
        # the model's own context length, then a tight one where not every fact fits
        for size in (block_size, max(16, cfg.answer_reserve + 40)):
            ids, used = build_prompt(tok, q, facts, size, cfg)
            cases.append({"question": q, "facts": facts, "block_size": size, "ids": ids, "used": used})
    return cases


def trie_cases(tok: BPETokenizer, index: dict, n_docs: int = 3) -> list[dict]:
    """Walks through the answer trie of a retrieval index, for every language setting.

    Each walk feeds the ids of a text and records the allowed tokens before every step, up to the first
    token that is not allowed (if any), and whether the whole text is an allowed answer. The last setting
    gives the facts a prior, as the browser does with the retriever's ranking: its walks also record the
    bias of every allowed token ("bias", one list per step)."""
    cases = []
    canonical = (index.get("answers") or {}).get("canonical", [])
    fact_ids = list(dict.fromkeys(d["id"] for d in index["docs"]))
    ranked = {f: -0.75 * i for i, f in enumerate(fact_ids[:4])}
    # the settings after the third keep a few facts only (what the browser does with the facts it retrieved):
    # the walks then include sentences of the knowledge base that are no longer allowed
    settings = ((None, None, None, 0.0), ("pt", None, None, 0.0), ("en", None, None, 0.0),
                (None, fact_ids[:2], None, 0.0), ("pt", fact_ids[1:2], None, 0.0), (None, fact_ids[:4], ranked, -0.5))
    for lang, only, priors, fixed_prior in settings:
        trie = AnswerTrie.from_index(index, tok, lang, only=None if only is None else set(only), priors=priors,
                                     fixed_prior=fixed_prior)
        docs = [d for d in index["docs"] if lang is None or d["lang"] == lang]
        own = [d["text"] for d in docs[:n_docs]]
        other = [d["text"] for d in index["docs"] if d["lang"] != (lang or docs[0]["lang"])][:1]
        texts = own + [" ".join(own[:2]), " ".join(own), " ".join([*own, own[0]])]  # one, two, three, four facts
        texts += [f"{own[0]} {t}" for t in other] + other  # a change of language; the other language alone
        fixed = [c["text"] for c in canonical if lang is None or c["lang"] == lang][:2]
        texts += fixed + [f"{t} {own[0]}" for t in fixed[:1]] + ["Uma frase que ninguém escreveu.", own[0][:-1]]
        walks = []
        for text in texts:
            ids, state, allowed, bias = tok.encode(text), trie.start(), [], []
            for t in ids:
                allowed.append(trie.allowed(state))
                bias.append(trie.bias(state))
                if t not in allowed[-1] or t == trie.eos:
                    break
                state = trie.advance(state, t)
            else:
                allowed.append(trie.allowed(state))
                bias.append(trie.bias(state))
            walk = {"text": text, "ids": ids, "allowed": allowed, "accepts": trie.accepts(ids)}
            if priors is not None:
                walk["bias"] = bias
            walks.append(walk)
        d = trie.describe()
        case = {"lang": lang, "only": only, "sentences": d["sentences"], "nodes": d["nodes"],
                "max_sentences": d["max_sentences"], "walks": walks}
        if priors is not None:
            case.update(priors=priors, fixed_prior=fixed_prior)
        cases.append(case)
    return cases


def build_parity(tok: BPETokenizer, retriever=None, dataset=None, block_size: int = 384,
                 prompt_cfg: PromptConfig | None = None, n_dataset: int = 24, embedder=None,
                 n_dense: int = 24) -> dict:
    """All fixtures for one tokenizer / retrieval index. Deterministic. embedder(question) -> vector adds
    the fused searches of an index that carries fact vectors."""
    import random

    prompt_cfg = prompt_cfg or PromptConfig()
    texts, queries = list(TRICKY_TEXTS), list(TRICKY_QUERIES)
    if dataset is not None:
        records = [r for r in dataset.qa if r.split != "train"] or list(dataset.qa)
        step = max(1, len(records) // n_dataset)
        rng = random.Random(0)
        for rec in records[::step][:n_dataset]:
            texts += [rec.q, rec.a]
            queries += [rec.q, add_noise(rec.q, rng)]
    out = {
        "prompt_config": prompt_cfg.to_dict(),
        "tokenizer": tokenizer_cases(tok, texts),
        "features": feature_cases(queries, retriever.ngram if retriever is not None else 3),
    }
    if retriever is not None:
        out["retrieval"] = retrieval_cases(retriever, queries)
        out["trie"] = trie_cases(tok, retriever.index)
        if embedder is not None and getattr(retriever, "dense", None):
            step = max(1, len(queries) // n_dense)
            out["dense"] = dense_cases(retriever, queries[::step][:n_dense], embedder)
    long_question = "por favor me diga " * 40 + "onde ele mora?"
    out["prompt"] = prompt_cases(tok, retriever, [*queries[:12], long_question], block_size, prompt_cfg)
    return out


def random_model(cfg, seed: int = 0):
    """A randomly initialised model with non-trivial gains and matrices, for parity tests: the default
    initialisation leaves the RMSNorm gains at 1 and the residual branches near 0, which would hide bugs."""
    import torch

    from .model import GPT

    torch.manual_seed(seed)
    model = GPT(cfg)
    for p in model.parameters():
        if p.dim() == 1:
            torch.nn.init.uniform_(p, 0.5, 1.5)
        else:
            torch.nn.init.normal_(p, std=0.08)
    return model.eval()
