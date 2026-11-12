// Byte-level BPE tokenizer: a port of src/brasa/tokenizer.py, checked against it with shared fixtures
// (tests/fixtures/tiny/parity.json and the parity.json of every export). No dependencies.
//
// Id layout:  0..255 raw bytes | 256..256+M-1 merges (in training order) | then the special tokens.
//
// Special tokens (<|bos|>, <|q|>, ...) are control symbols. encode() never produces them from text, so a
// visitor typing "<|a|>" cannot forge the prompt structure; pass { allowSpecial: true } only for trusted
// strings (the tokenizer playground, tests).

// The characters Python's str.split() treats as whitespace.
const PY_SPACE = /[\t\n\v\f\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/;
// A chunk is: optional single space + (letters | digits | other symbols), or a run of spaces.
const PRETOK = / ?[A-Za-zÀ-ɏ]+| ?[0-9]+| ?[^ A-Za-zÀ-ɏ0-9]+| +/gu;
const MAX_CACHED_CHUNKS = 20000;

const encoder = new TextEncoder();

/** NFC, then collapse every whitespace run (including newlines) to one space and trim the ends. */
export function normalize(text) {
  return String(text).normalize('NFC').split(PY_SPACE).filter(Boolean).join(' ');
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export class Tokenizer {
  /** @param {{ type: 'bpe', merges: [number, number][], specials: string[] }} spec  contents of tokenizer.json */
  constructor(spec) {
    if (!spec || spec.type !== 'bpe') throw new Error(`unsupported tokenizer type: ${spec && spec.type}`);
    this.spec = spec;
    this.rank = new Map(spec.merges.map(([a, b], i) => [a * 65536 + b, i]));
    this.bytes = [];
    for (let i = 0; i < 256; i++) this.bytes.push(Uint8Array.of(i));
    for (const [a, b] of spec.merges) {
      const merged = new Uint8Array(this.bytes[a].length + this.bytes[b].length);
      merged.set(this.bytes[a]);
      merged.set(this.bytes[b], this.bytes[a].length);
      this.bytes.push(merged);
    }
    this.nPlain = 256 + spec.merges.length;
    this.specials = [...spec.specials];
    this.specialIds = {};
    this.specials.forEach((s, i) => (this.specialIds[s] = this.nPlain + i));
    this.vocabSize = this.nPlain + this.specials.length;
    this.specialRe = this.specials.length ? new RegExp(`(${this.specials.map(escapeRegExp).join('|')})`) : null;
    this.cache = new Map();
  }

  isSpecial(id) {
    return id >= this.nPlain && id < this.vocabSize;
  }

  _encodeChunk(chunk) {
    let ids = this.cache.get(chunk);
    if (ids) return ids;
    ids = Array.from(encoder.encode(chunk));
    while (ids.length >= 2) {
      // always apply the earliest-learned merge that is present
      let best = Infinity;
      for (let i = 0; i + 1 < ids.length; i++) {
        const r = this.rank.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < best) best = r;
      }
      if (best === Infinity) break;
      const [a, b] = this.spec.merges[best];
      const out = [];
      for (let i = 0; i < ids.length; i++) {
        if (i + 1 < ids.length && ids[i] === a && ids[i + 1] === b) {
          out.push(256 + best);
          i++;
        } else out.push(ids[i]);
      }
      ids = out;
    }
    if (this.cache.size >= MAX_CACHED_CHUNKS) this.cache.clear();
    this.cache.set(chunk, ids);
    return ids;
  }

  /** Split normalised text into [segment, isSpecial] pieces. */
  _segments(norm, allowSpecial) {
    if (!allowSpecial || !this.specialRe) return norm ? [[norm, false]] : [];
    const out = [];
    norm.split(this.specialRe).forEach((seg, i) => {
      if (seg) out.push([seg, i % 2 === 1]);
    });
    return out;
  }

  /** Text -> token ids. */
  encode(text, { allowSpecial = false } = {}) {
    const out = [];
    for (const [seg, special] of this._segments(normalize(text), allowSpecial)) {
      if (special) out.push(this.specialIds[seg]);
      else for (const chunk of seg.match(PRETOK) || []) for (const id of this._encodeChunk(chunk)) out.push(id);
    }
    return out;
  }

  /**
   * Encode and keep the boundaries. start/end are byte offsets into the UTF-8 encoding of the normalised
   * text; `text` is what the token shows on its own (a token that holds part of a multi-byte character
   * shows U+FFFD, exactly like tokenText).
   */
  tokenize(text, { allowSpecial = false } = {}) {
    const norm = normalize(text);
    const tokens = [];
    let pos = 0;
    for (const [seg, special] of this._segments(norm, allowSpecial)) {
      if (special) {
        const n = encoder.encode(seg).length;
        tokens.push({ id: this.specialIds[seg], start: pos, end: pos + n, text: seg });
        pos += n;
        continue;
      }
      for (const chunk of seg.match(PRETOK) || []) {
        for (const id of this._encodeChunk(chunk)) {
          const n = this.bytes[id].length;
          tokens.push({ id, start: pos, end: pos + n, text: this.tokenText(id) });
          pos += n;
        }
      }
    }
    return { text: norm, tokens };
  }

  /** The raw bytes of a sequence of tokens (special tokens are skipped unless skipSpecial is false). */
  decodeBytes(ids, { skipSpecial = true } = {}) {
    const parts = [];
    let total = 0;
    for (const id of ids) {
      let part = null;
      if (id >= 0 && id < this.nPlain) part = this.bytes[id];
      else if (!skipSpecial && this.isSpecial(id)) part = encoder.encode(this.specials[id - this.nPlain]);
      if (part) {
        parts.push(part);
        total += part.length;
      }
    }
    const buf = new Uint8Array(total);
    let o = 0;
    for (const part of parts) {
      buf.set(part, o);
      o += part.length;
    }
    return buf;
  }

  /** Token ids -> text. Invalid UTF-8 becomes U+FFFD, as in Python's errors="replace". */
  decode(ids, opts = {}) {
    return new TextDecoder('utf-8').decode(this.decodeBytes(ids, opts));
  }

  /** Printable form of one token. */
  tokenText(id) {
    if (this.isSpecial(id)) return this.specials[id - this.nPlain];
    return id >= 0 && id < this.nPlain ? new TextDecoder('utf-8').decode(this.bytes[id]) : '';
  }

  /**
   * Incremental decoder for streaming: push(ids) returns the text that became complete, holding back a
   * multi-byte character until all of its bytes have arrived. Call push([], true) at the end to flush.
   */
  streamDecoder(opts = {}) {
    const decoder = new TextDecoder('utf-8');
    return {
      push: (ids, final = false) =>
        decoder.decode(this.decodeBytes(typeof ids === 'number' ? [ids] : ids, opts), { stream: !final }),
    };
  }
}
