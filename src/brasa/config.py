"""Configuration objects shared by every module: model sizes, special tokens, prompt format.

Nothing here imports torch, so the data and evaluation code can be used without a GPU stack loaded.

    uv run python -m brasa.config                       # every preset and the default prompt / data settings
    uv run python -m brasa.config --preset brasa-12m    # one preset
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# --------------------------------------------------------------------------------------- special tokens
# Order matters: the ids are assigned after the byte tokens and the BPE merges, in this order.
PAD, BOS, EOS, CTX, SEP, Q, A = "<|pad|>", "<|bos|>", "<|eos|>", "<|ctx|>", "<|sep|>", "<|q|>", "<|a|>"
SPECIAL_TOKENS: list[str] = [PAD, BOS, EOS, CTX, SEP, Q, A]

DEFAULT_VOCAB_SIZE = 4096
DEFAULT_BLOCK_SIZE = 384


# --------------------------------------------------------------------------------------- model
@dataclass
class GPTConfig:
    """Hyper-parameters of the decoder-only transformer (see model.py)."""

    vocab_size: int = DEFAULT_VOCAB_SIZE
    block_size: int = DEFAULT_BLOCK_SIZE  # context length: the largest number of positions
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    mlp: str = "swiglu"  # "swiglu" | "gelu"
    mlp_hidden: int | None = None  # default: 8/3 * d rounded to a multiple of 8 (swiglu) or 4 * d (gelu)
    dropout: float = 0.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if (self.d_model // self.n_head) % 2 != 0:
            raise ValueError("the head dimension must be even for rotary embeddings")
        if self.mlp not in ("swiglu", "gelu"):
            raise ValueError(f"unknown mlp kind: {self.mlp}")
        if self.mlp_hidden is None:
            self.mlp_hidden = int(round(8 * self.d_model / 3 / 8)) * 8 if self.mlp == "swiglu" else 4 * self.d_model

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


# name -> (layers, heads, width). brasa-12m is the main model; brasa-27m exists only as an ablation.
PRESETS: dict[str, dict[str, int]] = {
    "brasa-4m": dict(n_layer=4, n_head=4, d_model=256),
    "brasa-12m": dict(n_layer=6, n_head=6, d_model=384),
    "brasa-27m": dict(n_layer=8, n_head=8, d_model=512),
}


def preset(name: str, **overrides) -> GPTConfig:
    """Build the configuration of a named preset, e.g. preset("brasa-12m", dropout=0.1)."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose one of {sorted(PRESETS)}")
    return GPTConfig(**{**PRESETS[name], **overrides})


# --------------------------------------------------------------------------------------- prompt format
@dataclass
class PromptConfig:
    """How a question (and optional retrieved facts) becomes a token sequence.

        open-book    <|bos|><|ctx|>fact<|sep|>fact<|sep|>fact<|q|>question<|a|>answer<|eos|>
        closed-book  <|bos|><|q|>question<|a|>answer<|eos|>

    The same numbers are written into the exported model header so that the JavaScript side
    (web/engine/prompt.js) builds byte-identical prompts.
    """

    max_question_tokens: int = 96  # longer questions are truncated
    answer_reserve: int = 96  # tokens kept free for the answer when facts are packed into the context
    max_facts: int = 4  # facts shown in open-book mode (the retriever's k)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------- training data
@dataclass
class DataConfig:
    """How QA records are turned into training examples (see data.py)."""

    variants: int = 6  # training examples generated per QA record
    open_book_prob: float = 0.7  # share of the variants that carry a retrieved context
    noise_prob: float = 0.5  # share of the variants whose question gets surface noise (case, accents, typos)
    # share of the variants whose question loses some of its words (each with probability word_dropout_rate):
    # the fact has to be recognised from part of the wording, as with a phrasing never seen in training
    word_dropout_prob: float = 0.0
    word_dropout_rate: float = 0.25
    oracle_prob: float = 0.35  # open-book: make sure the gold facts are in the context
    drop_gold_prob: float = 0.15  # open-book: remove the gold facts (the retriever missed)
    shuffle_prob: float = 0.5  # open-book: shuffle the order of the facts
    random_distractor_prob: float = 0.25  # open-book: swap one non-gold fact for a random one
    # open-book, gold fact in the context: rewrite its numbers and one of its names, in the context AND in
    # the answer, so that the answer can only be produced by copying from the context (never from memory)
    counterfactual_prob: float = 0.0
    # cross-fitted contexts: with n > 1 folds, the context of a training question comes from an index that
    # does not hold the questions of its fold, so that retrieval is as hard in training as it is for a new
    # phrasing (with 0 the gold fact is nearly always retrieved first and the context teaches nothing)
    context_folds: int = 0
    # sampling weight of an example: (importance of its fact / 5) ** importance_power; 0 = uniform
    importance_power: float = 0.0
    prompt_loss_weight: float = 0.0  # loss weight of the non-answer tokens (0 = answer span only, 1 = all)
    prompt: PromptConfig = field(default_factory=PromptConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Print the model presets and the default prompt and data settings.")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="only this preset")
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    args = ap.parse_args()
    names = [args.preset] if args.preset else list(PRESETS)
    out = {
        "presets": {n: preset(n, vocab_size=args.vocab_size, block_size=args.block_size).to_dict() for n in names},
        "special_tokens": SPECIAL_TOKENS,
        "data": DataConfig().to_dict(),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
