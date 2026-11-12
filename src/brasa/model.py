"""Brasa: a small GPT-style language model, written from scratch in PyTorch.

    tokens -> embedding
           -> N x [ x + Attention(RMSNorm(x)) ; x + MLP(RMSNorm(x)) ]     (pre-norm residual blocks)
           -> RMSNorm -> linear head (weights tied to the embedding) -> logits

Design choices (nothing is imported from nn.Transformer or Hugging Face):
  * RMSNorm instead of LayerNorm, applied before each sub-layer (pre-norm).
  * Rotary position embeddings (RoPE) on queries and keys; there is no learned position table.
  * Causal multi-head self-attention through F.scaled_dot_product_attention.
  * SwiGLU feed-forward (GELU is available for ablations).
  * No biases anywhere; the embedding and the output head share one matrix (weight tying).
  * generate() uses a key/value cache, so each new token costs a single-position forward pass.

This file is the reference implementation: web/engine/engine.js reproduces it operation by operation
and is tested against it (tests/test_export_parity.py).

    uv run python -m brasa.model          # parameter counts of the presets and a cache self-test
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PRESETS, GPTConfig, preset

# ----------------------------------------------------------------------------- building blocks


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * g      (no mean subtraction, no bias)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()  # keep the mean of squares in float32 even under float16 autocast
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_tables(head_dim: int, max_pos: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_pos, head_dim/2). Pair i of a head rotates by pos * base^(-2i/head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x (..., T, head_dim). Pairs are (x[i], x[i + head_dim/2]); cos/sin are (T, head_dim/2).

    Rotating queries and keys by an angle proportional to their position makes the dot product q.k
    depend only on the distance between the two positions (tests/test_model.py checks this).
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, past=None, need_weights: bool = False):
        """x: (B, T, C). past: None, or (k, v) each (B, H, P, head_dim) from earlier positions.

        Returns (y, present, weights): present is the updated (k, v) when a cache is in use; weights is
        the (B, H, T, P+T) attention matrix when need_weights is set (slow path, used for inspection)."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        present, weights, mask = None, None, None
        if past is not None:  # incremental path: T new positions attend to P cached ones and themselves
            P = past[0].shape[2]
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
            present = (k, v)
            new_pos = torch.arange(T, device=x.device).unsqueeze(1) + P  # (T, 1) absolute positions
            mask = torch.arange(P + T, device=x.device).unsqueeze(0) <= new_pos  # (T, P+T)
        if need_weights:  # explicit softmax so that the attention matrix can be returned
            if mask is None:
                mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            y = weights @ v
        elif mask is None:  # training / full-context path
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y), present, weights


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.kind = cfg.mlp
        if cfg.mlp == "swiglu":  # down( silu(gate(x)) * up(x) )
            self.gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        if self.kind == "swiglu":
            return self.down(F.silu(self.gate(x)) * self.up(x))
        return self.down(F.gelu(self.up(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, past=None, need_weights: bool = False):
        a, present, weights = self.attn(self.norm1(x), cos, sin, past, need_weights)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, present, weights


# ----------------------------------------------------------------------------- the model


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        cos, sin = rope_tables(cfg.head_dim, cfg.block_size, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():  # residual-branch outputs start small
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear | nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # the tied matrix is counted once

    def describe(self) -> dict:
        return {**self.cfg.to_dict(), "params": self.num_params()}

    def empty_cache(self, batch_size: int = 1, device=None, dtype=None):
        """A zero-length KV cache: one (k, v) pair per layer, each (B, H, 0, head_dim)."""
        device = device or self.tok_emb.weight.device
        dtype = dtype or self.tok_emb.weight.dtype
        shape = (batch_size, self.cfg.n_head, 0, self.cfg.head_dim)
        return [(torch.zeros(shape, device=device, dtype=dtype),) * 2 for _ in self.blocks]

    def forward(self, idx, targets=None, cache=None, loss_weights=None):
        """idx: (B, T) token ids.

        cache=None  -> plain causal forward over the whole sequence (training).
        cache=[...] -> idx holds only the NEW tokens; positions continue after the cached ones.
        targets     -> (B, T) next-token ids, -100 where there is nothing to predict.
        loss_weights-> optional (B, T) per-position weights; the loss is their weighted mean.
        Returns (logits, loss, new_cache)."""
        B, T = idx.shape
        P = 0 if cache is None else cache[0][0].shape[2]
        if P + T > self.cfg.block_size:
            raise ValueError(f"sequence of {P + T} tokens exceeds block_size {self.cfg.block_size}")
        cos, sin = self.rope_cos[P : P + T], self.rope_sin[P : P + T]

        x = self.drop(self.tok_emb(idx))
        new_cache = None if cache is None else []
        for i, block in enumerate(self.blocks):
            x, present, _ = block(x, cos, sin, None if cache is None else cache[i])
            if cache is not None:
                new_cache.append(present)
        logits = self.lm_head(self.norm_f(x))

        loss = None
        if targets is not None:
            flat = logits.view(-1, logits.size(-1)).float()
            if loss_weights is None:
                loss = F.cross_entropy(flat, targets.view(-1), ignore_index=-100)
            else:
                per_token = F.cross_entropy(flat, targets.view(-1), ignore_index=-100, reduction="none")
                w = loss_weights.view(-1).float()
                loss = (per_token * w).sum() / w.sum().clamp_min(1e-8)
        return logits, loss, new_cache

    @torch.no_grad()
    def inspect(self, idx) -> dict:
        """Full forward pass that also returns what the browser visualisation shows.

        Returns {"logits": (B, T, V), "attention": (L, B, H, T, T), "resid_norms": (L+1, B, T)} where
        resid_norms[0] is the norm of the embedding and resid_norms[l+1] the norm after block l."""
        T = idx.shape[1]
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.tok_emb(idx)
        attention, norms = [], [x.norm(dim=-1)]
        for block in self.blocks:
            x, _, w = block(x, cos, sin, need_weights=True)
            attention.append(w)
            norms.append(x.norm(dim=-1))
        logits = self.lm_head(self.norm_f(x))
        return {"logits": logits, "attention": torch.stack(attention), "resid_norms": torch.stack(norms)}

    # ------------------------------------------------------------------ sampling

    @staticmethod
    def filter_logits(logits, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, previous=None):
        """The sampling distribution as log-weights: (B, V) -> (B, V) with -inf on removed tokens.

        Order of operations (identical in web/engine/engine.js):
          1. repetition penalty on tokens in `previous` (B, n): positive logits are divided, negative multiplied
          2. temperature
          3. top-k: keep the k largest
          4. top-p: keep the smallest prefix of the sorted distribution whose mass reaches top_p"""
        logits = logits.float()
        if repetition_penalty != 1.0 and previous is not None and previous.numel():
            seen = torch.gather(logits, -1, previous)
            seen = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
            logits = logits.scatter(-1, previous, seen)
        logits = logits / max(temperature, 1e-8)
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None and top_p < 1.0:
            sorted_logits, order = torch.sort(logits, descending=True, dim=-1, stable=True)  # ties: lower id first
            probs = torch.softmax(sorted_logits, dim=-1)
            drop = probs.cumsum(dim=-1) - probs >= top_p  # the token that crosses top_p is kept
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
        return logits

    @classmethod
    def sample_next(cls, logits, temperature=0.0, top_k=None, top_p=None, repetition_penalty=1.0, previous=None,
                    generator=None):
        """logits: (B, V) -> next token ids (B, 1). temperature == 0 means greedy."""
        if temperature <= 0.0:
            logits = cls.filter_logits(logits, 1.0, None, None, repetition_penalty, previous)
            return logits.argmax(dim=-1, keepdim=True)
        logits = cls.filter_logits(logits, temperature, top_k, top_p, repetition_penalty, previous)
        return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1, generator=generator)

    @torch.no_grad()
    def stream(self, idx, max_new_tokens, temperature=0.0, top_k=None, top_p=None, repetition_penalty=1.0,
               stop_ids=(), use_cache=True, generator=None):
        """Yield new tokens one step at a time as (B, 1) tensors. idx: (B, T) prompt.

        With use_cache the prompt is run once and every later step feeds a single token."""
        self.eval()
        B = idx.size(0)
        max_new_tokens = min(max_new_tokens, self.cfg.block_size - idx.size(1))
        done = torch.zeros(B, dtype=torch.bool, device=idx.device)
        stop = torch.tensor(list(stop_ids), device=idx.device, dtype=torch.long)
        cache = self.empty_cache(B, idx.device) if use_cache else None
        step_input, generated = idx, idx[:, :0]
        for _ in range(max_new_tokens):
            if use_cache:
                logits, _, cache = self(step_input, cache=cache)
            else:
                logits, _, _ = self(idx)
            nxt = self.sample_next(logits[:, -1, :], temperature, top_k, top_p, repetition_penalty, generated,
                                   generator)
            yield nxt
            if stop.numel():
                done |= torch.isin(nxt.squeeze(1), stop)
                if bool(done.all()):
                    return
            idx = torch.cat((idx, nxt), dim=1)
            generated = torch.cat((generated, nxt), dim=1)
            step_input = nxt

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, **kw):
        """Return (B, n_new) generated token ids (the prompt is not included)."""
        out = list(self.stream(idx, max_new_tokens, **kw))
        return torch.cat(out, dim=1) if out else idx[:, :0]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Print the size of each preset and self-test the KV cache.")
    ap.add_argument("--vocab-size", type=int, default=GPTConfig.vocab_size)
    ap.add_argument("--block-size", type=int, default=GPTConfig.block_size)
    args = ap.parse_args()
    torch.manual_seed(0)
    for name in PRESETS:
        m = GPT(preset(name, vocab_size=args.vocab_size, block_size=args.block_size))
        c = m.cfg
        print(f"{name:10s} {m.num_params():>11,} params   layers={c.n_layer} d_model={c.d_model} "
              f"heads={c.n_head} mlp_hidden={c.mlp_hidden}")
    m = GPT(preset("brasa-4m", vocab_size=512, block_size=64)).eval()
    x = torch.randint(0, 512, (2, 40))
    full, _, _ = m(x)
    cache, parts = m.empty_cache(2), []
    for chunk in (x[:, :7], x[:, 7:8], x[:, 8:25], x[:, 25:]):
        lg, _, cache = m(chunk, cache=cache)
        parts.append(lg)
    print("max |full - chunked cache| logits:", f"{(full - torch.cat(parts, dim=1)).abs().max().item():.2e}")
    a, b = m.generate(x[:, :5], 30, use_cache=True), m.generate(x[:, :5], 30, use_cache=False)
    print("greedy cached == uncached:", bool((a == b).all()))


if __name__ == "__main__":
    main()
