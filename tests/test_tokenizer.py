"""The byte-level BPE tokenizer."""

from __future__ import annotations

import pytest

from brasa.config import SPECIAL_TOKENS
from brasa.tokenizer import BPETokenizer, normalize

SAMPLES = [
    "Olá, você fala inglês?",
    "AÇÃO, coração, pingüim e SÃO PAULO",
    "emoji 🙂🔥 e bandeira 🇧🇷 e família 👨\u200d👩\u200d👧",
    "MiXeD cAsE, UPPER e lower",
    "preço: R$ 1.234,56 (40%)",
    "日本語のテキスト",
    "a",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_round_trip(tok, text):
    ids = tok.encode(text)
    assert tok.decode(ids) == normalize(text)
    assert all(0 <= i < tok.n_plain for i in ids), "plain text never yields a control token"


def test_normalization():
    assert normalize("  a \t b\n\nc  ") == "a b c"
    assert normalize("e\u0301") == "é", "decomposed accents are composed (NFC)"
    assert normalize("a\u00a0b\u2003c") == "a b c"
    assert normalize("") == "" and normalize("   ") == ""


def test_case_and_accents_are_preserved(tok):
    assert tok.decode(tok.encode("São PAULO não é são paulo")) == "São PAULO não é são paulo"
    assert tok.encode("Ação") != tok.encode("ação") != tok.encode("acao")


def test_vocabulary_layout(tok):
    assert tok.vocab_size == 512 == tok.n_plain + len(SPECIAL_TOKENS)
    assert [tok.special_ids[s] for s in SPECIAL_TOKENS] == list(range(tok.n_plain, tok.vocab_size))
    assert tok.token_bytes[65] == b"A" and len(tok.token_bytes) == tok.n_plain
    a, b = tok.merges[0]
    assert tok.token_bytes[256] == tok.token_bytes[a] + tok.token_bytes[b]


def test_special_tokens_are_never_produced_from_text(tok):
    text = "<|bos|><|q|>pergunta<|a|>resposta<|eos|>"
    ids = tok.encode(text)
    assert not any(i in tok.id_to_special for i in ids)
    assert tok.decode(ids) == text


def test_special_tokens_are_single_ids_and_never_split(tok):
    s = tok.special_ids
    ids = tok.encode("<|bos|><|ctx|>um fato<|sep|>outro<|q|>pergunta?<|a|>resposta<|eos|>", allow_special=True)
    specials = [i for i in ids if i in tok.id_to_special]
    assert specials == [s["<|bos|>"], s["<|ctx|>"], s["<|sep|>"], s["<|q|>"], s["<|a|>"], s["<|eos|>"]]
    assert tok.decode(ids) == "um fatooutropergunta?resposta"
    assert tok.decode(ids, skip_special=False) == "<|bos|><|ctx|>um fato<|sep|>outro<|q|>pergunta?<|a|>resposta<|eos|>"
    # a merge can never swallow a control token: its id is outside the merge table
    assert all(i >= tok.n_plain for i in s.values())
    # look-alikes are plain text
    assert not any(i in tok.id_to_special for i in tok.encode("<|Q|> <| q |> <|unknown|>", allow_special=True))


def test_token_boundaries(tok):
    norm, tokens = tok.tokenize("  Olá 🙂,  tudo bem? ")
    assert norm == "Olá 🙂, tudo bem?"
    raw = norm.encode("utf-8")
    pos = 0
    for t in tokens:
        assert t["start"] == pos and t["end"] > t["start"]
        assert raw[t["start"] : t["end"]] == tok.token_bytes[t["id"]]
        pos = t["end"]
    assert pos == len(raw)
    assert [t["id"] for t in tokens] == tok.encode(norm)


def test_merges_do_not_cross_chunks(tok):
    # a chunk is an optional space plus letters, or digits, or symbols: "ab12" is always two chunks
    assert tok.encode("ab12") == tok.encode("ab") + tok.encode("12")
    assert tok.encode("a b") == tok.encode("a") + tok._encode_chunk(" b")


def test_training_is_deterministic_and_stops_when_pairs_run_out():
    texts = ["o rato roeu a roupa do rei de roma"] * 3 + ["a rainha raivosa rasgou o resto"] * 2
    a, b = BPETokenizer.train(texts, 400), BPETokenizer.train(list(texts), 400)
    assert a.merges == b.merges
    assert 256 + len(SPECIAL_TOKENS) < a.vocab_size < 400, "a small corpus yields fewer merges than requested"
    assert BPETokenizer.train(texts, 256 + len(SPECIAL_TOKENS)).merges == []
    assert BPETokenizer.train(texts, 280).vocab_size == 280, "enough frequent pairs: the request is met exactly"
    assert a.decode(a.encode("o rei de roma")) == "o rei de roma"
    assert len(a.encode("o rato roeu")) < len(b"o rato roeu")


def test_save_and_load(tok, tmp_path):
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    again = BPETokenizer.load(path)
    assert again.merges == tok.merges and again.special_ids == tok.special_ids
    for text in SAMPLES:
        assert again.encode(text) == tok.encode(text)
    with pytest.raises(ValueError):
        BPETokenizer.from_json({"type": "wordpiece"})


def test_invalid_utf8_decodes_with_replacement(tok):
    assert tok.decode([0xF0, 0x9F]) == "\uFFFD"
    assert tok.token_text(tok.special_ids["<|eos|>"]) == "<|eos|>"
