"""The transformer: shapes, parameter counts, the KV cache, rotary embeddings, the loss mask, sampling."""

from __future__ import annotations

import math

import pytest
import torch

from brasa.config import PRESETS, GPTConfig, preset
from brasa.model import GPT, apply_rope, rope_tables

# vocabulary 4096: the tied embedding is counted once
EXPECTED_PARAMS = {"brasa-4m": 4_188_416, "brasa-12m": 12_194_688, "brasa-27m": 27_304_448}


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_shapes_and_parameter_counts(name):
    model = GPT(preset(name))
    cfg = model.cfg
    assert model.num_params() == EXPECTED_PARAMS[name]
    assert (cfg.vocab_size, cfg.block_size) == (4096, 384)

    # the count, derived by hand: embedding + per layer (2 gains, qkv, proj, 3 MLP matrices) + final gain
    d, h = cfg.d_model, cfg.mlp_hidden
    per_layer = 2 * d + 3 * d * d + d * d + 3 * d * h
    assert model.num_params() == cfg.vocab_size * d + cfg.n_layer * per_layer + d
    assert h % 8 == 0 and abs(h - 8 * d / 3) < 8

    x = torch.randint(0, cfg.vocab_size, (2, 9))
    logits, loss, cache = model(x, targets=torch.randint(0, cfg.vocab_size, (2, 9)))
    assert logits.shape == (2, 9, cfg.vocab_size)
    assert loss.shape == () and cache is None
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5, "an untrained model is close to uniform"


def test_no_biases_and_tied_head():
    model = GPT(preset("brasa-4m"))
    assert model.lm_head.weight is model.tok_emb.weight
    assert all("bias" not in name for name, _ in model.named_parameters())
    assert not any(name.startswith("rope") for name in model.state_dict()), "rotary tables are not parameters"


def test_config_validation():
    with pytest.raises(ValueError):
        GPTConfig(d_model=100, n_head=3)
    with pytest.raises(ValueError):
        GPTConfig(d_model=36, n_head=4)  # head dimension 9 is odd
    with pytest.raises(KeyError):
        preset("brasa-1b")
    assert preset("brasa-12m", dropout=0.1).dropout == 0.1


def test_gelu_variant():
    model = GPT(GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=4, d_model=48, mlp="gelu"))
    assert model.cfg.mlp_hidden == 4 * 48
    assert not hasattr(model.blocks[0].mlp, "gate")
    assert model(torch.randint(0, 300, (1, 5)))[0].shape == (1, 5, 300)


def test_kv_cache_equals_full_context_forward(tiny_model):
    x = torch.randint(0, 300, (2, 40))
    full, _, _ = tiny_model(x)
    cache, parts = tiny_model.empty_cache(2), []
    for chunk in (x[:, :7], x[:, 7:8], x[:, 8:25], x[:, 25:]):  # a prompt, single tokens, another block
        logits, _, cache = tiny_model(chunk, cache=cache)
        parts.append(logits)
    assert torch.allclose(torch.cat(parts, dim=1), full, atol=1e-5)
    assert cache[0][0].shape == (2, 4, 40, 12)


def test_generation_with_cache_equals_generation_without(tiny_model):
    prompt = torch.randint(0, 300, (3, 6))
    cached = tiny_model.generate(prompt, 30, use_cache=True)
    uncached = tiny_model.generate(prompt, 30, use_cache=False)
    assert torch.equal(cached, uncached)
    # and both equal the definition: argmax of a full forward pass, one token at a time
    seq = prompt.clone()
    for _ in range(30):
        seq = torch.cat((seq, tiny_model(seq)[0][:, -1].argmax(-1, keepdim=True)), dim=1)
    assert torch.equal(cached, seq[:, 6:])


def test_generation_respects_the_context_length_and_stop_tokens(tiny_model):
    prompt = torch.randint(0, 300, (1, 60))
    assert tiny_model.generate(prompt, 100).shape == (1, 4)  # block_size 64
    with pytest.raises(ValueError):
        tiny_model(torch.randint(0, 300, (1, 65)))
    free = tiny_model.generate(prompt[:, :5], 20)
    stopped = tiny_model.generate(prompt[:, :5], 20, stop_ids=(int(free[0, 3]),))
    first = free[0].tolist().index(int(free[0, 3]))
    assert stopped[0].tolist() == free[0, : first + 1].tolist()


def test_causality(tiny_model):
    x = torch.randint(0, 300, (1, 20))
    y = x.clone()
    y[0, 12] = (y[0, 12] + 1) % 300
    a, b = tiny_model(x)[0], tiny_model(y)[0]
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-6), "a later token must not change earlier outputs"
    assert not torch.allclose(a[:, 12:], b[:, 12:], atol=1e-6)


def test_rope_depends_only_on_relative_position():
    head_dim = 16
    cos, sin = rope_tables(head_dim, 128)
    q, k = torch.randn(head_dim), torch.randn(head_dim)

    def score(m: int, n: int) -> float:
        return float(apply_rope(q, cos[m], sin[m]) @ apply_rope(k, cos[n], sin[n]))

    for m, n in ((3, 1), (10, 10), (0, 7), (40, 25)):
        for shift in (1, 17, 60):
            assert abs(score(m, n) - score(m + shift, n + shift)) < 1e-4
    assert abs(score(5, 2) - score(5, 3)) > 1e-3, "different distances give different scores"
    assert abs(score(0, 0) - float(q @ k)) < 1e-5, "position 0 is the identity"
    rotated = apply_rope(q, cos[9], sin[9])
    assert abs(float(rotated.norm()) - float(q.norm())) < 1e-5, "a rotation preserves the norm"


def test_rope_first_frequency_is_one_radian_per_position():
    cos, sin = rope_tables(8, 4)
    assert cos.shape == (4, 4)
    assert torch.allclose(cos[:, 0], torch.cos(torch.arange(4.0)))
    assert torch.allclose(sin[1], torch.sin(10000.0 ** (-torch.arange(0, 8, 2) / 8)))


def test_loss_covers_only_unmasked_targets(tiny_model):
    x = torch.randint(0, 300, (2, 10))
    targets = torch.full((2, 10), -100)
    targets[0, 6:] = x[0, 6:]
    logits, loss, _ = tiny_model(x, targets=targets)
    manual = torch.nn.functional.cross_entropy(logits[0, 6:], x[0, 6:])
    assert torch.allclose(loss, manual, atol=1e-6)

    other = targets.clone()
    other[1, :3] = 5  # learning something else on other positions changes the loss
    assert not torch.allclose(tiny_model(x, targets=other)[1], loss)


def test_loss_weights_give_a_weighted_mean(tiny_model):
    x = torch.randint(0, 300, (1, 8))
    y = torch.randint(0, 300, (1, 8))
    w = torch.tensor([[0.0, 0.0, 0.25, 0.25, 1.0, 1.0, 1.0, 0.0]])
    logits, loss, _ = tiny_model(x, targets=y, loss_weights=w)
    per_token = torch.nn.functional.cross_entropy(logits[0], y[0], reduction="none")
    assert torch.allclose(loss, (per_token * w[0]).sum() / w.sum(), atol=1e-6)


def test_inspect_returns_attention_and_residual_norms(tiny_model):
    x = torch.randint(0, 300, (2, 11))
    out = tiny_model.inspect(x)
    assert out["attention"].shape == (2, 2, 4, 11, 11)  # layers, batch, heads, query, key
    assert out["resid_norms"].shape == (3, 2, 11)
    assert torch.allclose(out["attention"].sum(-1), torch.ones(2, 2, 4, 11), atol=1e-5)
    assert float(out["attention"].triu(1).abs().max()) == 0.0, "no attention to the future"
    assert torch.allclose(out["logits"], tiny_model(x)[0], atol=1e-5)
    assert torch.allclose(out["resid_norms"][0], tiny_model.tok_emb(x).norm(dim=-1))


def test_filter_logits():
    logits = torch.tensor([[2.0, 1.0, 1.0, 0.0, -1.0]])
    kept = lambda t: torch.nonzero(torch.isfinite(t[0])).flatten().tolist()  # noqa: E731
    assert kept(GPT.filter_logits(logits)) == [0, 1, 2, 3, 4]
    assert kept(GPT.filter_logits(logits, top_k=2)) == [0, 1, 2], "ties with the k-th logit are kept"
    assert kept(GPT.filter_logits(logits, top_k=1)) == [0]
    assert kept(GPT.filter_logits(logits, top_p=0.5)) == [0], "the token that crosses top-p is kept"
    assert kept(GPT.filter_logits(logits, top_p=0.6)) == [0, 1]
    sharp = torch.softmax(GPT.filter_logits(logits, temperature=0.1), -1)
    flat = torch.softmax(GPT.filter_logits(logits, temperature=10.0), -1)
    assert sharp[0, 0] > 0.99 and flat[0, 0] < 0.3
    penalised = GPT.filter_logits(logits, repetition_penalty=4.0, previous=torch.tensor([[0, 4, 4]]))
    assert penalised[0].tolist() == [0.5, 1.0, 1.0, 0.0, -4.0]


def test_sampling_is_reproducible_and_greedy_at_temperature_zero(tiny_model):
    prompt = torch.randint(0, 300, (1, 5))

    def run(seed):
        g = torch.Generator().manual_seed(seed)
        return tiny_model.generate(prompt, 20, temperature=1.5, top_k=50, top_p=0.95, generator=g)

    assert torch.equal(run(1), run(1))
    assert not torch.equal(run(1), run(2))
    greedy = tiny_model.generate(prompt, 20)
    assert torch.equal(greedy, tiny_model.generate(prompt, 20, temperature=0.0, top_k=5))
    assert torch.equal(greedy, tiny_model.generate(prompt, 20, temperature=1.0, top_k=1))


def test_dropout_is_inactive_in_eval_mode():
    model = GPT(GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=4, d_model=48, dropout=0.5))
    x = torch.randint(0, 300, (1, 10))
    model.eval()
    assert torch.equal(model(x)[0], model(x)[0])
    model.train()
    assert not torch.equal(model(x)[0], model(x)[0])
