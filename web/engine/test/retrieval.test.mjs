import test from 'node:test';
import assert from 'node:assert/strict';
import { Retriever, fold, words, charNgrams, queryFeatures } from '../retrieval.js';
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
