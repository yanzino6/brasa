import test from 'node:test';
import assert from 'node:assert/strict';
import { AnswerTrie, Retriever, continuationIds, fold, words, charNgrams, queryFeatures, quantizeQuery, rankingPriors } from '../retrieval.js';
import { Tokenizer } from '../tokenizer.js';
import { readJson } from './helpers.mjs';

const retriever = new Retriever(readJson('retrieval.json'));

test('fold strips accents and case', () => {
  assert.equal(fold('Ação É Ímpar'), 'acao e impar');
  assert.deepEqual(words('Onde o Téo mora? 2028!'), ['onde', 'o', 'teo', 'mora', '2028']);
});

test('character n-grams are padded', () => {
  assert.deepEqual(charNgrams('teo'), ['#te', 'teo', 'eo#']);
  assert.deepEqual(charNgrams('a'), ['#a#']);
});

test('query features are unique, words first', () => {
  assert.deepEqual(queryFeatures('teo teo'), ['w:teo', 'c:#te', 'c:teo', 'c:eo#']);
});

test('finds the right fact in both languages, with typos and without accents', () => {
  assert.equal(retriever.search('onde o teo mora')[0].id, 'location');
  assert.equal(retriever.search('Where does Téo live?')[0].id, 'location');
  assert.equal(retriever.search('mora em curitba', 4)[0].id, 'location');   // typo: character n-grams still match
});

test('each fact appears once and k is respected', () => {
  const hits = retriever.search('python pytorch', 8);
  assert.ok(hits.length > 1 && hits.length <= 8);
  assert.equal(new Set(hits.map((h) => h.id)).size, hits.length);
  for (let i = 1; i < hits.length; i++) assert.ok(hits[i - 1].score >= hits[i].score);
  assert.equal(retriever.search('python pytorch', 1).length, 1);
});

test('the language filter restricts the documents', () => {
  for (const h of retriever.search('contact email', 4, 'pt')) assert.equal(h.lang, 'pt');
  for (const h of retriever.search('contato', 4, 'en')) assert.equal(h.lang, 'en');
});

test('nothing matches: no hits', () => {
  assert.deepEqual(retriever.search('zzzz qqqq'), []);
  assert.deepEqual(retriever.search(''), []);
});

test('rejects an unknown index', () => {
  assert.throws(() => new Retriever({ type: 'dense', version: 1 }), /unsupported/);
});

// ---------------------------------------------------------------------------------- answer trie

const tok = new Tokenizer(readJson('tokenizer.json'));
const EOS = tok.specialIds['<|eos|>'];
const PT = ['O Téo mora em Curitiba.', 'O Téo trabalha com PyTorch desde 2019.', 'O Téo mora em Curitiba desde 2015.'];
const EN = ['Téo lives in Curitiba.', 'Téo has worked with PyTorch since 2019.'];
const FIXED = ['Isso eu não sei responder.', "I can't answer that."];

function smallTrie(maxSentences = 3) {
  const trie = new AnswerTrie(EOS, maxSentences);
  for (const t of PT) trie.add(tok, t, 'pt');
  for (const t of EN) trie.add(tok, t, 'en');
  for (const t of FIXED) trie.add(tok, t, null);
  return trie;
}

test('a following sentence has the ids it gets inside a joined answer', () => {
  for (const [a, b] of [[PT[0], PT[1]], [EN[1], EN[0]], [PT[2], FIXED[1]]]) {
    assert.deepEqual(tok.encode(`${a} ${b}`), [...tok.encode(a), ...continuationIds(tok, b)]);
  }
});

test('the trie accepts exactly the allowed answers', () => {
  const trie = smallTrie();
  const ok = (text) => trie.accepts(tok.encode(text));
  for (const t of [...PT, ...EN, ...FIXED]) assert.ok(ok(t), t);
  assert.ok(ok(`${PT[0]} ${PT[1]}`) && ok(`${PT[1]} ${PT[0]} ${PT[2]}`) && ok(`${EN[1]} ${EN[0]}`));
  assert.ok(!ok(`${PT[0]} ${EN[0]}`), 'an answer does not change language');
  assert.ok(!ok(`${PT[0]} ${PT[1]} ${PT[2]} ${PT[0]}`), 'at most three sentences');
  assert.ok(!ok(`${FIXED[0]} ${PT[0]}`) && !ok(`${PT[0]} ${FIXED[0]}`), 'a fixed answer stands alone');
  assert.ok(!ok('O Téo mora em') && !ok('') && !ok('O Téo mora em Recife.'));
  assert.ok(!smallTrie(1).accepts(tok.encode(`${PT[0]} ${PT[1]}`)));
  assert.deepEqual(trie.describe().groups, ['en', 'pt']);
  assert.equal(trie.describe().sentences, 7);
});

test('allowed tokens: sorted, <|eos|> only where a sentence ends, both readings of a prefix kept', () => {
  const trie = smallTrie();
  const first = [...new Set([...PT, ...EN, ...FIXED].map((t) => tok.encode(t)[0]))].sort((a, b) => a - b);
  assert.deepEqual(Array.from(trie.allowed(trie.start())), first);
  let state = trie.start();
  for (const t of tok.encode(PT[1])) {
    assert.ok(!trie.allowed(state).includes(EOS));
    state = trie.advance(state, t);
  }
  const following = [...new Set(PT.map((t) => continuationIds(tok, t)[0])), EOS].sort((a, b) => a - b);
  assert.deepEqual(Array.from(trie.allowed(state)), following);
  assert.deepEqual(trie.advance(state, EOS), []);
  assert.equal(trie.allowed([]).length, 0);
  assert.throws(() => trie.advance(trie.start(), EOS), RangeError);

  // "A." + the sentence "B." is the same token sequence as the single sentence "A. B."
  const [a, b] = ['O Téo mora em Curitiba.', 'Ele gosta de café.'];
  const twice = new AnswerTrie(EOS, 2);
  for (const t of [a, b, `${a} ${b}`]) twice.add(tok, t, 'pt');
  state = twice.start();
  for (const t of tok.encode(`${a} ${b}`)) state = twice.advance(state, t);
  assert.deepEqual(state.map((key) => key % 2).sort(), [0, 1], 'one sentence, or two');
  assert.ok(twice.accepts(tok.encode(`${a} ${b} ${a} ${b}`)) && !twice.accepts(tok.encode(`${a} ${b} ${a} ${b} ${a}`)));
});

test('the trie of an index holds its facts per language and its fixed answers', () => {
  const index = readJson('retrieval.json');
  const any = AnswerTrie.fromIndex(index, tok), pt = AnswerTrie.fromIndex(index, tok, 'pt');
  assert.equal(any.describe().sentences, index.docs.length + index.answers.canonical.length);
  assert.equal(any.maxSentences, index.answers.max_sentences);
  const ptDoc = index.docs.find((d) => d.lang === 'pt'), enDoc = index.docs.find((d) => d.lang === 'en');
  assert.ok(pt.accepts(tok.encode(ptDoc.text)) && !pt.accepts(tok.encode(enDoc.text)));
  assert.ok(any.accepts(tok.encode(enDoc.text)));
  for (const c of index.answers.canonical) assert.ok(any.accepts(tok.encode(c.text)), c.text);

  // only: the facts a question retrieved (in both languages) and the fixed answers, nothing else
  const ids = [...new Set(index.docs.map((d) => d.id))];
  const few = AnswerTrie.fromIndex(index, tok, null, null, ids.slice(0, 2));
  assert.equal(few.describe().sentences, 4 + index.answers.canonical.length);
  const text = (id, lang) => index.docs.find((d) => d.id === id && d.lang === lang).text;
  assert.ok(few.accepts(tok.encode(text(ids[0], 'pt'))) && few.accepts(tok.encode(text(ids[1], 'en'))));
  assert.ok(few.accepts(tok.encode(`${text(ids[1], 'pt')} ${text(ids[0], 'pt')}`)), 'retrieved facts still chain');
  assert.ok(!few.accepts(tok.encode(text(ids[2], 'pt'))) && any.accepts(tok.encode(text(ids[2], 'pt'))));
  assert.ok(!few.accepts(tok.encode(`${text(ids[0], 'pt')} ${text(ids[2], 'pt')}`)));
  for (const c of index.answers.canonical) assert.ok(few.accepts(tok.encode(c.text)), c.text);
  const none = AnswerTrie.fromIndex(index, tok, 'en', null, []);
  assert.equal(none.describe().sentences, index.answers.canonical.filter((c) => c.lang === 'en').length);
});

test('rankingPriors: nats per rank, or the distance to the best score', () => {
  const hits = [{ id: 'a', score: 8 }, { id: 'b', score: 6 }, { id: 'c', score: 2 }];
  assert.deepEqual([...rankingPriors(hits, 1.5)], [['a', -0], ['b', -1.5], ['c', -3]]);
  assert.deepEqual([...rankingPriors(hits, 4, 'score')], [['a', -0], ['b', -1], ['c', -3]]);
  assert.equal(rankingPriors(hits, 0).size, 0, 'no weight, no prior');
  assert.equal(rankingPriors([], 2, 'score').size, 0);
  assert.deepEqual([...rankingPriors([{ id: 'a', score: 0 }, { id: 'b', score: -1 }], 2, 'score')], [['a', -0], ['b', -2]], 'no positive best score: ranks');
  assert.throws(() => rankingPriors(hits, 1, 'magic'), /unknown prior/);
});

test('bias: the best prior behind each allowed token; the sentences that follow carry none', () => {
  const plain = smallTrie();
  assert.equal(plain.hasPrior, false);
  assert.ok(plain.bias(plain.start()).every((b) => b === 0), 'no priors: nothing changes');

  const trie = new AnswerTrie(EOS, 3);
  trie.add(tok, PT[0], 'pt', -1);
  trie.add(tok, PT[1], 'pt', -3);
  trie.add(tok, PT[2], 'pt', 0);
  trie.add(tok, FIXED[0], null, -0.5);
  assert.equal(trie.hasPrior, true);
  const worth = (state) => Object.fromEntries(Array.from(trie.allowed(state), (t, i) => [t, trie.bias(state)[i]]));
  assert.deepEqual(worth(trie.start()), { [tok.encode(PT[0])[0]]: 0, [tok.encode(FIXED[0])[0]]: -0.5 });
  let state = trie.start();
  const seen = [];
  for (const t of tok.encode(PT[0])) {
    seen.push(Object.values(worth(state)).sort((a, b) => a - b));
    state = trie.advance(state, t);
  }
  assert.ok(seen.some((v) => v.length === 2 && v[0] === -3 && v[1] === 0), 'where PT[1] leaves the shared prefix');
  assert.ok(seen.some((v) => v.length === 2 && v[0] === -1 && v[1] === 0), 'where PT[0] and PT[2] part');
  const end = worth(state);
  assert.equal(end[EOS], -1, '<|eos|> is worth the sentence that ends here');
  assert.ok(Object.values(end).every((b) => b === -1), 'going on costs what stopping costs: the model decides');
  state = trie.advance(state, continuationIds(tok, PT[1])[0]);
  assert.ok(trie.bias(state).every((b) => b === 0), 'the sentences that follow carry no prior');

  // fromIndex: priors by fact id (a Map or a plain object), one prior for the fixed answers
  const index = readJson('retrieval.json');
  const ids = [...new Set(index.docs.map((d) => d.id))].slice(0, 3);
  const priors = rankingPriors(ids.map((id, i) => ({ id, score: 3 - i })), 2);
  for (const table of [priors, Object.fromEntries(priors)]) {
    const ranked = AnswerTrie.fromIndex(index, tok, null, null, ids, table, -1);
    assert.equal(ranked.hasPrior, true);
    assert.ok(Math.max(...ranked.bias(ranked.start())) === 0, 'the first hit loses nothing');
    let at = ranked.start();
    for (const t of tok.encode(index.answers.canonical[0].text)) at = ranked.advance(at, t);
    assert.deepEqual(Array.from(ranked.bias(at)), [-1], 'a fixed answer ends with its own prior');
  }
  assert.equal(AnswerTrie.fromIndex(index, tok, null, null, ids).hasPrior, false);
});

test('the dense half of the index: fused search with a question vector, lexical search without', () => {
  const index = readJson('retrieval.json'), fixtures = readJson('parity.json');
  assert.ok(retriever.dense, 'the fixture index carries fact vectors');
  assert.equal(retriever.dense.vectors.length, retriever.docs.length * retriever.dense.dim);
  assert.equal(retriever.dense.pooling, 'mean');
  assert.deepEqual(Array.from(quantizeQuery([0, 1, -1, 0.5, -0.5, 2, 1e-9])), [0, 32767, -32767, 16384, -16383, 32767, 0]);

  let searches = 0;
  for (const c of fixtures.dense) {
    const vector = Float32Array.from(c.vector);
    for (const s of c.searches) {
      const got = retriever.search(c.q, s.k, s.lang, vector);
      assert.deepEqual(got.map((h) => [h.id, h.lang, h.doc, h.score]), s.hits.map((h) => [h.id, h.lang, h.doc, h.score]), c.q);
      searches++;
    }
  }
  assert.ok(searches >= 20);

  // every document is a candidate once the cosine counts, even with no lexical evidence at all
  const vector = Float32Array.from(fixtures.dense[0].vector);
  assert.equal(retriever.search('zzzz qqqq', 4).length, 0);
  const hits = retriever.search('zzzz qqqq', 4, null, vector);
  assert.equal(hits.length, 4);
  assert.equal(new Set(hits.map((h) => h.id)).size, 4);
  assert.ok(hits.every((h, i) => i === 0 || h.score <= hits[i - 1].score));
  assert.ok(retriever.search('onde o teo mora', 8, 'en', vector).every((h) => h.lang === 'en'));

  // no vector, or no dense section: exactly the lexical search
  const plain = new Retriever({ ...index, dense: undefined });
  assert.equal(plain.dense, null);
  assert.deepEqual(retriever.search('onde o teo mora', 4), plain.search('onde o teo mora', 4));
  assert.deepEqual(plain.search('onde o teo mora', 4, null, vector), plain.search('onde o teo mora', 4));

  assert.throws(() => retriever.search('x', 4, null, new Float32Array(3)), /question vector has 3 values/);
  assert.throws(() => new Retriever({ ...index, dense: { ...index.dense, dim: index.dense.dim + 1 } }), /one vector per document/);
  assert.throws(() => new Retriever({ ...index, dense: { ...index.dense, version: 2 } }), /unsupported dense section/);
});
