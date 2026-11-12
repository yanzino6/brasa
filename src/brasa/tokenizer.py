"""A byte-level BPE tokenizer written from scratch (no dependencies).

Text is normalised first (Unicode NFC, every whitespace run collapsed to one space), split into chunks
(optional leading space + letters | digits | other symbols) and each chunk is encoded as UTF-8 bytes that
are then merged pair by pair in the order learned during training. Because the base alphabet is the 256
byte values, any string can be encoded and decoding always returns the normalised input: accents, emoji
and scripts never seen in training simply fall back to shorter tokens.

Id layout:   0..255 raw bytes | 256..256+M-1 merges (in training order) | then the special tokens.

Special tokens (<|bos|>, <|q|>, ...) are control symbols. encode() never produces them from text, so a
visitor typing "<|a|>" cannot forge the prompt structure; encode(..., allow_special=True) parses them as
single ids and is meant for trusted strings (tests, the tokenizer playground).

web/engine/tokenizer.js is a line-by-line port; both are checked against shared fixtures.

    uv run python -m brasa.tokenizer --data-dir data/rehearsal --vocab-size 4096 --out runs/tokenizer.json
    uv run python -m brasa.tokenizer --load runs/tokenizer.json --encode "Olá, você fala inglês?"
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from .config import DEFAULT_VOCAB_SIZE, SPECIAL_TOKENS

# A chunk is: optional single space + (letters | digits | other non-space symbols), or a whitespace run.
# Merges never cross chunk boundaries, which keeps training fast and lets encoding be cached per chunk.
_LETTER = "A-Za-zÀ-ɏ"
PRETOK = re.compile(rf" ?[{_LETTER}]+| ?[0-9]+| ?[^\s{_LETTER}0-9]+|\s+")


def normalize(text: str) -> str:
    """NFC, then collapse every whitespace run (including newlines) to one space and trim the ends."""
    return " ".join(unicodedata.normalize("NFC", text).split())


class BPETokenizer:
    def __init__(self, merges: Iterable[tuple[int, int]], specials: Iterable[str] = SPECIAL_TOKENS):
        self.merges = [tuple(m) for m in merges]
        self.specials = list(specials)
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        self.token_bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        self.n_plain = 256 + len(self.merges)
        self.special_ids = {s: self.n_plain + i for i, s in enumerate(self.specials)}
        self.id_to_special = {i: s for s, i in self.special_ids.items()}
        self.vocab_size = self.n_plain + len(self.specials)
        self._special_re = (
            re.compile("(" + "|".join(re.escape(s) for s in self.specials) + ")") if self.specials else None
        )
        self._cache: dict[str, list[int]] = {}

    # -- training -----------------------------------------------------------------------------
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = DEFAULT_VOCAB_SIZE,
              specials: Iterable[str] = SPECIAL_TOKENS, min_freq: int = 2) -> BPETokenizer:
        """Learn up to vocab_size - 256 - len(specials) merges. Stops early when no pair occurs min_freq
        times, so a small corpus yields a smaller vocabulary than requested (check .vocab_size)."""
        specials = list(specials)
        words: Counter = Counter()
        for t in texts:
            words.update(PRETOK.findall(normalize(t)))
        seqs = [list(w.encode("utf-8")) for w in words]
        freqs = list(words.values())
        pair_count: Counter = Counter()
        pair_words = defaultdict(set)
        for wi, s in enumerate(seqs):
            for pair in zip(s, s[1:], strict=False):
                pair_count[pair] += freqs[wi]
                pair_words[pair].add(wi)

        merges: list[tuple[int, int]] = []
        for _ in range(max(0, vocab_size - 256 - len(specials))):
            if not pair_count:
                break
            # most frequent pair; ties are broken by the smaller ids so that training is deterministic
            pair, cnt = max(pair_count.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if cnt < min_freq:
                break
            new_id = 256 + len(merges)
            merges.append(pair)
            for wi in list(pair_words[pair]):
                s, f = seqs[wi], freqs[wi]
                for old in zip(s, s[1:], strict=False):
                    pair_count[old] -= f
                    if pair_count[old] <= 0:
                        del pair_count[old]
                    pair_words[old].discard(wi)
                merged = _merge(s, pair, new_id)
                seqs[wi] = merged
                for new in zip(merged, merged[1:], strict=False):
                    pair_count[new] += f
                    pair_words[new].add(wi)
        return cls(merges, specials)

    # -- encode -------------------------------------------------------------------------------
    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = self._cache.get(chunk)
        if ids is None:
            ids = list(chunk.encode("utf-8"))
            while len(ids) >= 2:  # always apply the earliest-learned merge that is present
                best = min(zip(ids, ids[1:], strict=False), key=lambda p: self.rank.get(p, float("inf")))
                if best not in self.rank:
                    break
                ids = _merge(ids, best, 256 + self.rank[best])
            self._cache[chunk] = ids
        return ids

    def _segments(self, norm: str, allow_special: bool) -> list[tuple[str, bool]]:
        """Split normalised text into (segment, is_special) pieces."""
        if not allow_special or self._special_re is None:
            return [(norm, False)]
        return [(seg, i % 2 == 1) for i, seg in enumerate(self._special_re.split(norm)) if seg]

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        out: list[int] = []
        for seg, special in self._segments(normalize(text), allow_special):
            if special:
                out.append(self.special_ids[seg])
            else:
                for chunk in PRETOK.findall(seg):
                    out.extend(self._encode_chunk(chunk))
        return out

    def tokenize(self, text: str, allow_special: bool = False) -> tuple[str, list[dict]]:
        """Encode and keep the boundaries: returns (normalised text, [{"id", "start", "end"}, ...]) where
        start/end are byte offsets into the UTF-8 encoding of the normalised text."""
        norm = normalize(text)
        tokens, pos = [], 0
        for seg, special in self._segments(norm, allow_special):
            if special:
                n = len(seg.encode("utf-8"))
                tokens.append({"id": self.special_ids[seg], "start": pos, "end": pos + n})
                pos += n
                continue
            for chunk in PRETOK.findall(seg):
                for i in self._encode_chunk(chunk):
                    n = len(self.token_bytes[i])
                    tokens.append({"id": i, "start": pos, "end": pos + n})
                    pos += n
        return norm, tokens

    # -- decode -------------------------------------------------------------------------------
    def decode_bytes(self, ids: Iterable[int], skip_special: bool = True) -> bytes:
        parts = []
        for i in ids:
            if i < self.n_plain:
                parts.append(self.token_bytes[i])
            elif not skip_special and i in self.id_to_special:
                parts.append(self.id_to_special[i].encode("utf-8"))
        return b"".join(parts)

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        return self.decode_bytes(ids, skip_special).decode("utf-8", errors="replace")

    def token_text(self, i: int) -> str:
        """Printable form of one token (a partial UTF-8 sequence shows as the replacement character)."""
        return self.id_to_special.get(i) or self.token_bytes[i].decode("utf-8", errors="replace")

    # -- persistence --------------------------------------------------------------------------
    def to_json(self) -> dict:
        return {"type": "bpe", "version": 1, "merges": [list(m) for m in self.merges], "specials": self.specials,
                "vocab_size": self.vocab_size}

    @classmethod
    def from_json(cls, spec: dict) -> BPETokenizer:
        if spec.get("type") != "bpe":
            raise ValueError(f"unsupported tokenizer type: {spec.get('type')!r}")
        return cls([tuple(m) for m in spec["merges"]], spec["specials"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i, n = [], 0, len(ids)
    while i < n:
        if i + 1 < n and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train the BPE tokenizer on a data directory, or encode a string.")
    ap.add_argument("--data-dir", default="data", help="directory with kb/ and qa/ (see data.py)")
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    ap.add_argument("--out", help="where to write tokenizer.json")
    ap.add_argument("--load", help="use an existing tokenizer.json instead of training")
    ap.add_argument("--encode", help="text to encode and show token by token")
    args = ap.parse_args()

    if args.load:
        tok = BPETokenizer.load(args.load)
    else:
        from .data import load_dataset, tokenizer_texts

        tok = BPETokenizer.train(tokenizer_texts(load_dataset(args.data_dir)), args.vocab_size)
        print(f"trained {len(tok.merges)} merges -> vocabulary of {tok.vocab_size} (requested {args.vocab_size})")
        if args.out:
            tok.save(args.out)
            print("saved", args.out)
    if args.encode:
        norm, tokens = tok.tokenize(args.encode, allow_special=True)
        print(repr(norm))
        print(" ".join(f"[{t['id']}:{tok.token_text(t['id'])!r}]" for t in tokens))
        print(f"{len(tokens)} tokens for {len(norm)} characters")


if __name__ == "__main__":
    main()
