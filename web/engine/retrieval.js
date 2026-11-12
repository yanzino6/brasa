// BM25 retrieval over the knowledge-base facts: a port of the search half of src/brasa/retrieval.py.
// No dependencies.
//
// The index (retrieval.json) is built in Python and already holds one finished weight per
// (feature, document): every logarithm and division happened at build time. Searching is lookups and
// float64 additions in a fixed order, so Python and JavaScript return the same ranking and the same
// scores bit for bit. tests/fixtures/tiny/parity.json and the parity.json of every export prove it.
//
// A learned retriever only has to offer the same search(query, k, lang) method.

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

export class Retriever {
  /** @param {object} index  contents of retrieval.json */
  constructor(index) {
    if (!index || index.type !== 'bm25' || index.version !== 1) throw new Error('unsupported retrieval index');
    this.index = index;
    this.docs = index.docs;
    this.postings = index.postings;
    this.ngram = index.params.ngram;
  }

  /** One score per document of the index. */
  scores(query) {
    const scores = new Float64Array(this.docs.length);
    for (const feature of queryFeatures(query, this.ngram)) {
      const entry = Object.prototype.hasOwnProperty.call(this.postings, feature) ? this.postings[feature] : null;
      if (entry) for (let j = 0; j < entry.length; j += 2) scores[entry[j]] += entry[j + 1];
    }
    return scores;
  }

  /**
   * The k best facts (each fact once, in the language that matched best), best first.
   * @param {string} query
   * @param {number} [k=4]
   * @param {'pt'|'en'|null} [lang=null]  restrict the search to one language
   * @returns {{ id: string, lang: string, score: number, text: string, topic: string, doc: number }[]}
   */
  search(query, k = 4, lang = null) {
    const scores = this.scores(query);
    const order = [];
    for (let i = 0; i < scores.length; i++) if (scores[i] > 0) order.push(i);
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
