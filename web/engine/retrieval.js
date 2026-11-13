// BM25 retrieval over the knowledge-base facts: a port of the search half of src/brasa/retrieval.py.
// No dependencies.
//
// The index (retrieval.json) is built in Python and already holds one finished weight per
// (feature, document): every logarithm and division happened at build time. Searching is lookups and
// float64 additions in a fixed order, so Python and JavaScript return the same ranking and the same
// scores bit for bit. tests/fixtures/tiny/parity.json and the parity.json of every export prove it.
//
// The learned half (optional, index.dense): one int8 vector per document, produced by the model read as
// an encoder (Engine.embed, src/brasa/encoder.py). A search that is given the vector of the question adds
// the cosine to the lexical score: bm25 / (best bm25 of the query) + weight * cosine. The question vector
// is rounded to int16 first, so the dot products are integer sums and, given the same question vector,
// Python and JavaScript still agree bit for bit. Without a vector the search is the lexical one.
//
// The index is also the knowledge base as the browser sees it: its documents are the facts and
// index.answers holds the fixed answers (greetings, refusals). AnswerTrie, at the end of this file, turns
// them into the token trie that constrained decoding walks: a port of src/brasa/constrain.py.

const COMBINING = /[\u0300-\u036f]/g;
const WORD = /[a-z0-9]+/g;

/** Lower-case and strip accents (NFD, drop U+0300..U+036F). */
export function fold(text) {
  return String(text).normalize('NFD').replace(COMBINING, '').toLowerCase();
}

export function words(text) {
  return fold(text).match(WORD) || [];
}

export function charNgrams(word, n = 3) {
  const padded = `#${word}#`;
  const out = [];
  for (let i = 0; i < Math.max(1, padded.length - n + 1); i++) out.push(padded.slice(i, i + n));
  return out;
}

/** Unique features of a query in a fixed order: words first, then n-grams, by first appearance. */
export function queryFeatures(text, n = 3) {
  const ws = words(text);
  const seen = new Set();
  const out = [];
  const add = (f) => {
    if (!seen.has(f)) {
      seen.add(f);
      out.push(f);
    }
  };
  for (const w of ws) add(`w:${w}`);
  for (const w of ws) for (const g of charNgrams(w, n)) add(`c:${g}`);
  return out;
}

const QUERY_LEVELS = 32767;   // a question vector is rounded to int16 before the dot products

function base64Bytes(text) {
  if (typeof atob === 'function') {
    const binary = atob(text), out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
    return out;
  }
  return Uint8Array.from(Buffer.from(text, 'base64'));   // very old Node
}

/** The question vector as integers: floor(v * 32767 + 0.5), the rounding of retrieval.quantize_query. */
export function quantizeQuery(vector) {
  const out = new Int32Array(vector.length);
  for (let j = 0; j < vector.length; j++) {
    const q = Math.floor(vector[j] * QUERY_LEVELS + 0.5);
    out[j] = q > QUERY_LEVELS ? QUERY_LEVELS : q < -QUERY_LEVELS ? -QUERY_LEVELS : q;
  }
  return out;
}

export class Retriever {
  /** @param {object} index  contents of retrieval.json */
  constructor(index) {
    if (!index || index.type !== 'bm25' || index.version !== 1) throw new Error('unsupported retrieval index');
    this.index = index;
    this.docs = index.docs;
    this.postings = index.postings;
    this.ngram = index.params.ngram;
    /** The learned half of the index, or null: { dim, pooling, weight, maxTokens, factor, vectors: Int8Array }. */
    this.dense = null;
    if (index.dense) {
      const d = index.dense;
      if (d.version !== 1) throw new Error('unsupported dense section in the retrieval index');
      const bytes = base64Bytes(d.vectors);
      if (bytes.length !== this.docs.length * d.dim) throw new Error('the dense section does not have one vector per document');
      this.dense = {
        dim: d.dim, pooling: d.pooling, weight: d.weight, maxTokens: d.max_tokens, factor: d.scale / QUERY_LEVELS,
        vectors: new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length),
      };
    }
  }

  /**
   * One score per document of the index: BM25, or, with the vector of the question and a dense section, the
   * fusion bm25 / (best bm25 of the query) + weight * cosine.
   * @param {string} query
   * @param {ArrayLike<number>|null} [vector=null]  the question as the encoder sees it (Engine.embed)
   */
  scores(query, vector = null) {
    const scores = new Float64Array(this.docs.length);
    for (const feature of queryFeatures(query, this.ngram)) {
      const entry = Object.prototype.hasOwnProperty.call(this.postings, feature) ? this.postings[feature] : null;
      if (entry) for (let j = 0; j < entry.length; j += 2) scores[entry[j]] += entry[j + 1];
    }
    if (!vector || !this.dense) return scores;
    const { dim, vectors, factor, weight } = this.dense;
    if (vector.length !== dim) throw new Error(`the question vector has ${vector.length} values, the index expects ${dim}`);
    let top = 0;
    for (let i = 0; i < scores.length; i++) if (scores[i] > top) top = scores[i];
    if (!(top > 0)) top = 1;
    const q = quantizeQuery(vector);
    for (let i = 0, o = 0; i < scores.length; i++, o += dim) {
      let dot = 0;                       // integers: exact, whatever the order
      for (let j = 0; j < dim; j++) dot += vectors[o + j] * q[j];
      scores[i] = scores[i] / top + weight * (dot * factor);
    }
    return scores;
  }

  /**
   * The k best facts (each fact once, in the language that matched best), best first.
   * @param {string} query
   * @param {number} [k=4]
   * @param {'pt'|'en'|null} [lang=null]  restrict the search to one language
   * @param {ArrayLike<number>|null} [vector=null]  the question vector: with it (and a dense section) the
   *        ranking is the fusion and every document is a candidate; without it, the lexical ranking
   * @returns {{ id: string, lang: string, score: number, text: string, topic: string, doc: number }[]}
   */
  search(query, k = 4, lang = null, vector = null) {
    const scores = this.scores(query, vector);
    const fused = !!(vector && this.dense);
    const order = [];
    for (let i = 0; i < scores.length; i++) if (fused || scores[i] > 0) order.push(i);
    order.sort((a, b) => scores[b] - scores[a] || a - b);
    const hits = [];
    const seen = new Set();
    for (const i of order) {
      const d = this.docs[i];
      if (seen.has(d.id) || (lang != null && d.lang !== lang)) continue;
      seen.add(d.id);
      hits.push({ id: d.id, lang: d.lang, score: scores[i], text: d.text, topic: d.topic, doc: i });
      if (hits.length === k) break;
    }
    return hits;
  }
}

// ------------------------------------------------------------------------- constrained decoding

/**
 * Token ids of `text` when it follows another sentence after one space. The tokenizer trims its input
 * and never merges across a space, so the ids of "A B" are the ids of "A" followed by these ids of "B".
 */
export function continuationIds(tok, text) {
  return tok.encode(`x ${text}`).slice(tok.encode('x').length);
}

export const PRIOR_KINDS = ['rank', 'score'];

/**
 * The retriever's ranking as a prior on the facts of the prompt (best first): Map of fact id -> nats added to
 * the model's log-probability of that sentence. A port of ranking_priors() in src/brasa/constrain.py.
 *   'rank'   -weight * position (0 for the first hit)
 *   'score'  -weight * (1 - score / best score); falls back to 'rank' when the best score is not positive
 * weight 0 gives an empty Map: the model alone decides.
 * @param {{ id: string, score: number }[]} hits
 */
export function rankingPriors(hits, weight, kind = 'rank') {
  if (!PRIOR_KINDS.includes(kind)) throw new Error(`unknown prior: ${kind}`);
  const out = new Map();
  if (!weight || !hits.length) return out;
  const top = hits[0].score;
  hits.forEach((h, i) => out.set(h.id, kind === 'score' && top > 0 ? -weight * (1 - h.score / top) : -weight * i));
  return out;
}

/**
 * Token trie of the answers the model is allowed to write (see src/brasa/constrain.py):
 *   a fact, two or three facts of the same language joined by a space, or one fixed answer alone.
 * A state is a sorted array of numbers node * maxSentences + sentencesFinished; several entries mean
 * that the tokens so far can be read in more than one way. [] is the state after <|eos|>.
 *
 * A sentence may carry a prior (nats added to the model's log-probability of that sentence, see
 * rankingPriors); bias(state) turns the priors into one number per allowed token.
 */
export class AnswerTrie {
  /** @param {number} eos  id of <|eos|>  @param {number} [maxSentences=3] */
  constructor(eos, maxSentences = 3) {
    this.eos = eos;
    this.maxSentences = Math.max(1, maxSentences);
    this.children = [new Map()];   // node 0: the start of an answer
    this.ends = new Map();         // node -> Set of the groups of the sentences that end there
    this.nextRoot = new Map();     // group -> node where a following sentence of the group starts
    this.sentences = 0;
    this.hasPrior = false;         // true once a sentence was added with a prior other than 0
    this.best = [-Infinity];       // node -> the best prior among the sentences that pass through it
    this.endPrior = new Map();     // node -> the best prior among the sentences that end there
    this._allowed = new Map();
    this._bias = new Map();
  }

  _insert(node, ids, group, prior) {
    if (prior > this.best[node]) this.best[node] = prior;
    for (const t of ids) {
      let next = this.children[node].get(t);
      if (next === undefined) {
        next = this.children.length;
        this.children.push(new Map());
        this.best.push(-Infinity);
        this.children[node].set(t, next);
      }
      node = next;
      if (prior > this.best[node]) this.best[node] = prior;
    }
    if (!this.ends.has(node)) this.ends.set(node, new Set());
    this.ends.get(node).add(group);
    if (!(this.endPrior.get(node) >= prior)) this.endPrior.set(node, prior);
  }

  /**
   * group null: a fixed answer, alone. Otherwise the sentence may chain with others of its group.
   * prior: nats added to the model's log-probability of this sentence when it comes first (see bias).
   */
  add(tok, text, group = null, prior = 0) {
    const ids = tok.encode(text);
    if (!ids.length) return;
    this._allowed.clear();
    this._bias.clear();
    this.sentences += 1;
    if (prior !== 0) this.hasPrior = true;
    this._insert(0, ids, group, prior);
    if (group !== null && this.maxSentences > 1) {
      if (!this.nextRoot.has(group)) {
        this.nextRoot.set(group, this.children.length);
        this.children.push(new Map());
        this.best.push(-Infinity);
      }
      this._insert(this.nextRoot.get(group), continuationIds(tok, text), group, 0);   // see bias
    }
  }

  /**
   * The trie of a retrieval index: its facts, chained per language, and its fixed answers.
   * @param {object} index  contents of retrieval.json
   * @param {import('./tokenizer.js').Tokenizer} tok
   * @param {'pt'|'en'|null} [lang=null]  keep one language; null lets the model choose
   * @param {number|null} [maxSentences=null]  facts one answer may chain (default: the index's own)
   * @param {Iterable<string>|null} [only=null]  keep these fact ids (and the fixed answers): the facts a
   *        question retrieved, so that the model can only say what is in front of it
   * @param {Map<string, number>|Record<string, number>|null} [priors=null]  fact id -> prior in nats (see
   *        rankingPriors); a fact that is not named gets 0
   * @param {number} [fixedPrior=0]  the prior of every fixed answer
   */
  static fromIndex(index, tok, lang = null, maxSentences = null, only = null, priors = null, fixedPrior = 0) {
    const answers = index.answers || {};
    const trie = new AnswerTrie(tok.specialIds['<|eos|>'], maxSentences || answers.max_sentences || 3);
    const keep = only === null ? null : new Set(only);
    const table = priors instanceof Map ? priors : new Map(Object.entries(priors || {}));
    for (const d of index.docs) {
      if ((lang === null || d.lang === lang) && (keep === null || keep.has(d.id))) trie.add(tok, d.text, d.lang, table.get(d.id) ?? 0);
    }
    for (const c of answers.canonical || []) if (lang === null || c.lang === lang) trie.add(tok, c.text, null, fixedPrior);
    return trie;
  }

  start() { return [0]; }

  _moves(state, visit) {
    const m = this.maxSentences;
    for (const key of state) {
      const node = Math.floor(key / m), count = key % m;
      for (const [t, child] of this.children[node]) visit(t, child * m + count);
      const groups = this.ends.get(node);
      if (!groups || count + 1 >= m) continue;
      for (const group of groups) {
        if (group === null) continue;
        for (const [t, child] of this.children[this.nextRoot.get(group)]) visit(t, child * m + count + 1);
      }
    }
  }

  _atEnd(state) {
    const m = this.maxSentences;
    return state.some((key) => this.ends.has(Math.floor(key / m)));
  }

  /** Token ids that may come next, ascending. <|eos|> is allowed where a sentence ends. */
  allowed(state) {
    const memo = state.join(',');
    let out = this._allowed.get(memo);
    if (!out) {
      const tokens = new Set();
      this._moves(state, (t) => tokens.add(t));
      if (this._atEnd(state)) tokens.add(this.eos);
      out = Int32Array.from(tokens).sort();
      this._allowed.set(memo, out);
    }
    return out;
  }

  /**
   * What the priors add to the log-probability of each token of allowed(state), in the same order
   * (Float64Array). The prior is about the FIRST sentence of an answer: a token inside it is worth the best
   * prior among the sentences it still leads to, <|eos|> is worth the prior of the sentence that ends here,
   * and so is a token that starts a following sentence (whether to go on, and with which fact, stays the
   * model's decision). Only differences between the tokens of one step matter.
   */
  bias(state) {
    const memo = state.join(',');
    let out = this._bias.get(memo);
    if (!out) {
      const m = this.maxSentences, worth = new Map();
      const raise = (t, value) => { if (!(worth.get(t) >= value)) worth.set(t, value); };
      for (const key of state) {
        const node = Math.floor(key / m), count = key % m;
        for (const [t, child] of this.children[node]) raise(t, this.best[child]);
        const groups = this.ends.get(node);
        if (!groups) continue;
        const done = this.endPrior.get(node);
        raise(this.eos, done);
        if (count + 1 >= m) continue;
        for (const group of groups) {
          if (group === null) continue;
          for (const t of this.children[this.nextRoot.get(group)].keys()) raise(t, done);
        }
      }
      out = Float64Array.from(this.allowed(state), (t) => worth.get(t));
      this._bias.set(memo, out);
    }
    return out;
  }

  /** The state after `token`; [] after <|eos|>. Throws when the token is not allowed. */
  advance(state, token) {
    if (token === this.eos && this._atEnd(state)) return [];
    const next = new Set();
    this._moves(state, (t, key) => { if (t === token) next.add(key); });
    if (!next.size) throw new RangeError(`token ${token} is not allowed here`);
    return Array.from(next).sort((a, b) => a - b);
  }

  /** True when ids (without <|eos|>) is exactly one allowed answer. */
  accepts(ids) {
    let state = this.start();
    for (const t of ids) {
      if (t === this.eos || !this.allowed(state).includes(t)) return false;
      state = this.advance(state, t);
    }
    return this._atEnd(state);
  }

  describe() {
    return { sentences: this.sentences, nodes: this.children.length, maxSentences: this.maxSentences, groups: [...this.nextRoot.keys()].sort() };
  }
}
