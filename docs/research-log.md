# Brasa research log

Lab notebook of the experiments on the real corpus. Newest entry last. Every entry: hypothesis, exact
command, run directory, numbers, conclusion, next step. All numbers were measured in this repository;
"test" is the held-out test split unless stated otherwise. Times are local (UTC-3).

Conventions
- Partial corpus `runs/data-partial1`: 230 of 525 facts with questions, 10,312 train / 476 val / 936 test
  records, no chit-chat and no out-of-scope data. Numbers on it are NOT the final numbers.
- One run per setting has a standard error of about 1.3 points of exact match on 936 questions
  (sqrt(0.2*0.8/936)), plus the MPS run-to-run noise. Differences below about 3 points are noise until
  repeated with another seed.
- The held-out test split is used to choose between settings (validation is small), so the final numbers
  are optimistic. A final untouched slice is planned for the full corpus (see entry 0, plan).

---

## Entry 0 — 2026-11-12 09:20 — state found, plan

No log existed: this is the first session. Read README, src/brasa/*, web/engine/*, scripts, exp1.

State
- `runs/exp1.sh` is running in the background (started 08:56): p1-4m-base, p1-4m-cf50 done; p1-4m-cf50-do,
  p1-12m-base, p1-12m-cf50-do to come (about 10 min per 4m run with its two evaluations).
- NOTE: `TrainConfig.dropout` defaults to 0.1, so `--dropout 0.1` in exp1.sh changes nothing:
  p1-4m-cf50 and p1-4m-cf50-do are two runs of the SAME setting (a free estimate of MPS noise), and
  p1-4m-base already has dropout 0.1.
- `scripts/sync-brasa.mjs` (outside ml/, not mine) exists. It copies a FIXED list: the default model
  file, tokenizer.json, retrieval.json, manifest.json, model-card.json, the engine files engine.js,
  tokenizer.js, retrieval.js, prompt.js, session.js, worker.js, client.js, index.js, kernel.wasm and all
  *.d.ts. Consequences for my work: new JavaScript must live in those files (a new module would not be
  copied), and new data must travel inside retrieval.json or the model file. It calls
  `session.handle({type:'ask', question, mode:'open', lang:null, inspect})`.

Experiment 1 so far (brasa-4m, 3000 steps, seed 0, partial corpus; test n=936)

| run | ckpt | open EM | open F1 | EM gold retrieved | EM gold missed | unsupported numbers | closed EM | closed F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p1-4m-base | best (step 500) | 15.7 | 38.7 | 30.0 | 2.8 | 16.1 | 8.2 | 30.9 |
| p1-4m-base | last (3000) | 21.2 | 41.7 | 35.2 | 8.5 | 24.8 | 14.7 | 35.9 |
| p1-4m-cf50 | best (step 500) | 13.8 | 37.2 | 24.4 | 4.3 | 20.0 | 9.8 | 32.3 |
| p1-4m-cf50 | last (3000) | 20.7 | 42.1 | 36.3 | 6.7 | 21.2 | 15.6 | 36.2 |

Retriever (BM25 + training questions): gold fact in top 4 for 47.3% of the test questions (48.3% val).

Observations
1. The checkpoint with the lowest validation LOSS (step 500) is clearly worse on exact match than the
   last one (15.7 vs 21.2 open, 8.2 vs 14.7 closed). Validation loss is the wrong selection criterion:
   the loss rises because the model becomes confident, while its argmax keeps improving. -> select on
   validation exact match (programme item 4).
2. Counterfactual augmentation at 0.5: no visible effect at brasa-4m (20.7 vs 21.2 open EM; inside the
   noise). Wait for the replicate (p1-4m-cf50-do) and the 12m pair.
3. Retrieval is the first bottleneck: 47% recall@4, and EM is 35% when the gold fact is there against
   8% when it is not.

Plan (programme order, adapted to what can run while the GPU is busy)
- A. Now, CPU only: constrained decoding (token trie) in Python + visitor metrics in evaluate.py, tests.
  Evaluate on the existing checkpoints (no retraining).
- B. Retrieval analysis on held-out phrasings (CPU): where BM25 fails; topic routing; lexical features.
- C. When the GPU frees up (about 10:00): dual encoder; reading-vs-memorising levers; then pretraining.
- D. Full corpus after 12:10: rebuild, split off a final untouched test slice (second test question per
  fact), re-run the chosen levers, final recipe.
- E. JavaScript constrained decoding with parity fixtures (after the Python design is settled).

---

## Entry 1 — 09:55 — constrained decoding and visitor metrics (Python), experiment C1

Implemented (CPU work while exp1 trains):
- `src/brasa/constrain.py`: `AnswerTrie` (token trie of the allowed answers: every fact, chains of up to
  three facts of one language, the fixed chit-chat/refusal answers alone) and `constrained_answers`
  (greedy, where forced tokens cost no forward pass; beam search on the sum of log-probabilities).
- `brasa.evaluate`: modes `open_book_constrained`, `closed_book_constrained` next to the free ones
  (`--modes`, `--beam`, `--constrain-lang any|question`, `--tag` to write eval-<tag>.json without touching
  the card). New metrics in every group: `verbatim_rate`, `right_topic_rate`, `gold_in_top1/2/4`.
- The retrieval index now carries `"answers": {"canonical": [...], "max_sentences": 3}` so the browser
  can build the same trie from retrieval.json (the only data file the sync script copies).
- Tests: tests/test_constrain.py (9), new tests in tests/test_evaluate.py. `uv run ruff check` clean,
  `uv run pytest` 116 passed.
- Found: `key_spans` in scripts/build_corpus.py extracts "8" from "n8n" (360 gold answers of the partial
  corpus do not contain their own key span) -> key-span recall is capped at 0.93 on seen questions.
  To fix in build_corpus.py before the full corpus is rebuilt.

C1: decoding rule, no retraining. Checkpoint runs/p1-4m-base (last), test n=936.
Command: `uv run python -m brasa.evaluate --run runs/p1-4m-base --checkpoint last --splits val,test --device cpu --tag c1-any`
and `runs/c1-decoding.sh` (beam 4, beam 16, `--constrain-lang question`).

| decoding | open EM | open F1 | open verbatim | open right topic | open unsupported numbers | closed EM | closed verbatim | closed right topic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| free greedy | 21.1 | 41.7 | 96.3 | 66.7 | 24.8 | 14.7 | 94.0 | 59.5 |
| constrained greedy, both languages | 21.4 | 41.4 | 100.0 | 67.8 | 27.8 | 14.8 | 100.0 | 62.5 |
| constrained beam 4, both languages | 22.1 | 42.2 | 100.0 | 69.1 | 23.2 | 15.8 | 100.0 | 63.3 |
| constrained beam 16 / language of the question | pending | | | | | | | |

Retriever alone (open-book rows): gold in top 1 / 2 / 4 = 22.3 / 33.6 / 47.3 % (test).

Conclusions
1. Diagnosis (2) is mostly wrong for this model: free decoding already writes a curated sentence 96% of
   the time (94% closed-book). It does not blend or invent much; it picks the WRONG REAL fact. Constrained
   decoding turns the remaining 4-6% into curated sentences (verbatim 100% by construction) and leaves
   exact match unchanged (21.1 -> 21.4). Its value is the guarantee, not accuracy.
2. The "unsupported number rate" (25% open, 58% closed) is therefore not a hallucination rate: those are
   numbers of other real facts. Under constrained decoding every number is a curated one and the metric
   stays at 28%. The model card must say so.
3. Beam search buys about +1 point (inside the noise): the search rule is not the lever either.
4. Right-topic rate: 67% of the open-book answers are a true sentence about the right subject, against
   21% exact. Diagnosis (3) holds: exact match alone undersells what a visitor sees by a factor of three.

Where the open-book answers come from (single-fact test questions, n=920, free decoding):
- gold retrieved (47%): exact 35%, ANOTHER RETRIEVED fact 29%, a fact that was NOT in the context 31%.
- gold missed (53%): exact 8%, another retrieved fact 36%, non-retrieved fact 52%.
- exact match by the position of the gold fact in the context: 1st 51%, 2nd 27%, 3rd 17%, 4th 15%.
So the model does not treat the context as the place to look: half of its answers are not in the context
at all, and when the gold fact is there it is only found reliably in first position.
Reason (hypothesis H-xfit): in training, the retriever has the training question itself in its index
(document expansion), so the gold fact is nearly always retrieved first (seen-question retrieval hit
100%); the context carries no information the question does not already give, and memorising the
question -> answer map is enough. At test time the retriever misses half of the time.

Retrieval analysis (BM25 + training questions, single-fact test questions, n=920)
- recall@1/2/4/8/16/32/64 = 22.7 / 33.7 / 47.4 / 63.0 / 78.6 / 90.9 / 95.9 %.
- without the training questions in the index: recall@1/4/16 = 10.9 / 22.9 / 49.8 %.
- restricting to the language of the question changes nothing (22.6 / 47.3).
- the top hit has the gold TOPIC 67.6% of the time; gold topic among the top 4: 87.4%.
- with an oracle topic filter: recall@1/2/4/8 = 32.9 / 46.4 / 64.9 / 83.7 %.
So topic routing has a ceiling of 65% recall@4: the hard part is choosing among sibling facts of the
right topic, with held-out questions that were written to avoid the training key words ("Walk me through
the software lineup powering Isabela" for the fact that lists Isabela's tech stack).

Next
- GPU (after exp1): H-xfit = cross-fitted retrieval for the training contexts (the retriever that builds
  the context of a training question does not have that question in its index), checkpoint selection on
  validation exact match (teacher-forced, free), then label smoothing / open-book share / importance
  sampling; dual encoder; pretraining.
- CPU: JavaScript port of the trie and constrained greedy decoding with parity fixtures.

---

## Entry 2 — 09:45 — exp1 4m result, levers implemented, exp2 queued

exp1, brasa-4m, last checkpoint, test n=936 (three runs; cf50 and cf50-do are the same setting):

| run | open EM | open F1 | EM gold retrieved | EM gold missed | unsupported numbers | closed EM | closed F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p1-4m-base | 21.1 | 41.7 | 35.2 | 8.5 | 24.8 | 14.7 | 35.9 |
| p1-4m-cf50 | 20.7 | 42.1 | 36.3 | 6.7 | 21.2 | 15.6 | 36.2 |
| p1-4m-cf50-do (= cf50 again) | 21.7 | 42.1 | 36.6 | 8.3 | 22.7 | 17.9 | 37.9 |

Conclusion: counterfactual augmentation at 0.5 does nothing measurable at brasa-4m when the training
contexts come from the full index (20.7 and 21.7 against 21.1). The two identical runs differ by 1.0
point open-book and 2.3 points closed-book: that is the MPS noise floor of a single run.

C1 remainder: the beam-16 and `--constrain-lang question` evaluations were stopped (they shared the GPU
with exp1 and halved its speed); beam 4 already answered the question. To redo on the CPU if needed.

Implemented for exp2 (all tested by the existing suite; new tests to add with the JS work):
- `--context-folds N` (DataConfig.context_folds, data.fold_retrievers, Bm25Retriever.from_dataset(skip=)):
  cross-fitted training contexts. Measured on 1500 training questions of the partial corpus: with the
  full index the gold fact is retrieved first for 99.3% (recall@4 99.9%): the context is redundant. With
  4 folds: recall@1/2/4 = 58.0 / 70.1 / 82.0 % (held-out test: 22.7 / 33.7 / 47.4).
- `--select em|loss`: ckpt_best.pt now keeps the highest validation exact match (teacher-forced: every
  answer token is the argmax, which equals greedy exact match, for the price of the validation loss).
  train_log.jsonl has val_em, val_em_open, val_em_closed.
- `--label-smoothing`, `--importance-power` (sampling weight (importance/5)^p).

Queue: `runs/queue.sh` reads `runs/queue.txt` (one run per line, restartable, waits for exp1 to end and
never overlaps another training); log in runs/queue.log; each run is evaluated with
`brasa.evaluate --checkpoint last|best --splits val,test --tag last|best` -> eval-last.json, eval-best.json.
exp1 is left to finish (p1-12m-base, p1-12m-cf50-do; expected about 10:15).

exp2 (brasa-4m, 3000 steps, partial corpus, baseline p1-4m-base): p2-4m-xfit, p2-4m-xfit-cf50,
p2-4m-base-s1 (seed 1), p2-4m-xfit-s1, p2-4m-ob90, p2-4m-ls10, p2-4m-do0, p2-4m-do20, p2-4m-imp1.
Hypothesis H-xfit: open-book EM when the gold fact is retrieved rises well above 35%, and the share of
answers that are not in the context falls.

Meanwhile (CPU): JavaScript trie + constrained decoding + parity fixtures.

---

## Entry 3 — 09:47 (clock check: entries 1 and 2 were written at about 09:30 and 09:40) — constrained decoding in the JavaScript engine

Implemented, inside the files the sync script copies (no new module):
- `web/engine/retrieval.js`: `AnswerTrie` (port of constrain.py; `AnswerTrie.fromIndex(index, tok, lang)`
  builds it from retrieval.json: the facts per language plus `index.answers.canonical`), `continuationIds`.
- `web/engine/engine.js`: `stream()/generate()` take `constraint`; each step carries `choices` (number of
  allowed tokens; 1 = forced). Forced tokens are fed WITHOUT the output head unless `inspect` is on, so
  constrained decoding does less work than free decoding; the free path is the old code path
  (`constraint` null). `chooseAllowed()` = greedy among allowed ids (ties to the lower id, as in Python)
  or sampling restricted to them.
- `web/engine/session.js`: `ask` option `constrain` (default true when a retrieval index is loaded;
  `false` = the model's free generation; `lang` restricts retrieval and the allowed sentences to one
  language, `lang: null` allows both). `context.constrained`, `token.choices`, `done.stats.constrained`,
  `ready.info.constrained`. Tries are built lazily per language and cached.
- Declarations updated (engine.d.ts DecodingConstraint, retrieval.d.ts AnswerTrie, worker.d.ts).
- Parity fixtures: `parity.json` gets `trie` (39 walks, 1604 steps on the tiny fixture: allowed-token sets
  before every step, accept/reject, for both / pt / en), `testvec.<fmt>.json` gets `constrained` (Python's
  constrained greedy tokens for every test prompt, alternating language settings and budgets).
  `web/engine/tools/parity.mjs` replays both.
- tests/fixtures/tiny regenerated (`uv run python scripts/make_fixtures.py`).

Checks: `node --test engine/test/*.test.mjs` 53 passed (45 before), `tsc -p engine/test/tsconfig.json` clean,
`node web/engine/tools/parity.mjs tests/fixtures/tiny --wasm web/wasm/kernel.wasm` PARITY OK with
constrained 6/6 on f16.js, f16.wasm-simd, i8.js, i8.wasm-simd and 0 trie mismatches; `uv run pytest` 116
passed, ruff clean.

Decision: beam search stays a Python evaluation option only (about +1 point in C1, inside the noise; a
beam in the browser would need one KV cache per hypothesis or re-feeding the diverging suffix).

Next: retrieval on the CPU (failure analysis is in entry 1): topic routing by aliases, lexical features;
then the dual encoder when the GPU is free.

---

## Entry 4 — 13:48 — session 2 starts: what session 1 left unlogged, exp1 (12m) and exp2 results

Session 1 was cut at about 09:52, after entry 3 and before logging the `--index fields` work; its queue
kept running until 11:00. Reconstructed from the tree, the queue log and the eval files.

State found
- Code in the tree but not in the log: `Bm25Retriever.from_dataset(layout="fields")` (fact text, topic
  names and training questions as three BM25 fields with their own length normalisation and document
  frequencies, added at build time with FIELD_WEIGHTS fact 1 / topic 1 / questions 2, k1 2.0, trigram
  weight 0.25; same file format and same search code, so the JavaScript side is unchanged),
  `brasa.train --index flat|fields`, `brasa.evaluate --index`. tests/fixtures/tiny uses the fields layout.
- `uv run pytest` 122 passed, `node --test` 53 passed; ruff had one long line in tests (fixed).
- The automatic rebuild of data/yan/qa happened at 12:57: **457 of 525 facts have questions (68 missing,
  the first is esp32-cam-car-001), 8 multi-fact items, and NO chit-chat and NO out-of-scope questions**
  (qa-src/chitchat.jsonl and oos.jsonl are empty: the teacher hit its usage limit, "try again at 5:10 PM",
  runs/write_questions2.log). 20,300 train / 930 val / 1,844 test records.
- `runs/queue.sh` is still alive and idle (it reads runs/queue.txt; new lines are picked up).

exp1, brasa-12m (partial corpus, 3000 steps, seed 0, last checkpoint, test n=936)

| run | open EM | open F1 | EM gold retrieved | EM gold missed | right topic (open) | closed EM | closed F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p1-4m-base | 21.1 | 41.7 | 35.2 | 8.5 | 66.7 | 14.7 | 35.9 |
| p1-12m-base | 24.1 | 43.5 | 40.9 | 9.1 | 68.4 | 18.1 | 38.0 |
| p1-12m-cf50-do (counterfactual 0.5) | 23.8 | 44.3 | 39.5 | 9.7 | 72.4 | 19.0 | 39.6 |

- brasa-12m is about 3 points better than brasa-4m in both modes (one run each: 2 standard errors).
- Counterfactual augmentation: no effect at 12m either (23.8 vs 24.1 open, 19.0 vs 18.1 closed).
  Conclusion for programme item 4: `--counterfactual-prob` does not help on this corpus; dropped.
- p1-12m-base "best" (lowest validation loss, step 500) scores 18.4 open EM against 24.1 for the last
  checkpoint: confirms that validation loss is the wrong selection rule (now `--select em`).

exp2 (brasa-4m, 3000 steps, seed 0, partial corpus, last checkpoint, test n=936; `runs/queue.txt`)

| run | index | training contexts | gold in top 1 / 2 / 4 | open EM | EM gold retrieved | EM gold missed | right topic | closed EM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p1-4m-base | flat | full index | 22.3 / 33.6 / 47.3 | 21.1 | 35.2 | 8.5 | 66.7 | 14.7 |
| p1-4m-base, evaluated with the fields index (eval-fields.json) | fields | (flat) | 29.1 / 44.3 / 58.7 | 23.2 | 35.5 | 5.7 | 69.4 | 14.7 |
| p2-4m-fields | fields | full index | 29.1 / 44.3 / 58.7 | 23.7 | 36.6 | 5.4 | 69.2 | 17.7 |
| p2-4m-xfit | flat | 4 folds | 22.3 / 33.6 / 47.3 | 19.8 | 30.0 | 10.6 | 65.0 | 15.9 |
| p2-4m-fields-xfit | fields | 4 folds | 29.1 / 44.3 / 58.7 | 22.1 | 31.9 | 8.3 | 70.4 | 17.8 |
| p2-4m-fields-xfit-cf50 | fields | 4 folds + counterfactual 0.5 | 29.1 / 44.3 / 58.7 | 21.6 | 32.4 | 6.2 | 68.6 | 16.2 |

Conclusions
1. The fields layout is a real gain for the retriever: gold in top 4 from 47.3 to 58.7% (+11.4), top 1
   from 22.3 to 29.1%. The model's exact match follows only partly (21.1 -> 23.2 / 23.7).
2. H-xfit is REFUTED. Cross-fitted training contexts make the context harder in training (gold in the
   context 83% instead of 89%) but the model does not read better: exact match when the gold fact is
   retrieved FALLS (36.6 -> 31.9 with fields, 35.2 -> 30.0 flat), and open EM is 1.5 points lower in both
   pairs. With counterfactuals on top: no change.
3. The uncomfortable baseline: "answer with the retriever's first hit" scores exact match = gold in
   top 1 = **29.1%** with the fields index. Every trained model so far is BELOW that (21 to 24%): in
   exact-match terms the generator loses information the retriever had. Selecting one fact among four
   from an unseen phrasing is what the model cannot do (35% when the gold fact is there; the rank prior
   alone would give 29.1 / 58.7 = 50%).

Decisions
- Counterfactual augmentation and cross-fitted contexts are not part of the recipe.
- New question (exp3): can the model at least inherit the retriever's ranking? Training contexts keep the
  rank order (no shuffle, no forced/dropped gold, no random distractor) and come from cross-fitted indices,
  so "the first fact is the answer 58% of the time" is learnable. Flags added:
  `--oracle-prob --drop-gold-prob --shuffle-prob --random-distractor-prob`.
  Queued 13:47: `p3-4m-fields-xfit-ranked` (fields, 4 folds, all four probabilities 0) and
  `p3-4m-fields-s1` (p2-4m-fields with seed 1: the noise of a single run).
- Add a "copy the first hit" row to every table (a decoding mode of evaluate.py, no model needed).

Plan for this session (in order; GPU queue kept busy while the CPU work goes on)
1. Frozen snapshot of the full corpus `runs/data-full1` with: the key-span fix (n8n), a FINAL split (the
   second test question of every fact, never looked at before the final evaluation), and fallback
   chit-chat / out-of-scope questions written here (templates), since the teacher's did not arrive.
2. CPU: retrieval on the full corpus (first-hit baseline, per-question index, topic routing), decoding
   restricted to the retrieved candidates and retriever + model score fusion on existing checkpoints.
3. GPU: baseline on the full corpus, then the dual encoder, then pretraining on Wikipedia text.
4. A provisional final recipe (brasa-12m + export + card + ml/runs/final) as soon as the first decisions
   are in, refreshed when a later experiment changes the recipe.

---

## Entry 5 — 18:57 — session 3 starts: what session 2 left unlogged, the full corpus has arrived

Session 2 was cut at about 14:25 (its last log entry is 13:48). Reconstructed from the tree and the queue log.

State found
- Unlogged code of session 2, all in the tree and passing: `scripts/build_corpus.py` (the key-span fix for
  "n8n"-like names; the FINAL split = the 14th question of every fact, never evaluated before the release;
  `qa-fallback/` used only while the teacher's chit-chat / out-of-scope files are empty),
  `scripts/fallback_questions.py` (hand-written frames for chit-chat and out-of-scope),
  `data.SPLITS` with "final", `runs/data-full1` (snapshot at 13:52: 457 facts with questions, fallback
  chit-chat / out-of-scope, 22,217 train / 1,008 val / 1,065 test / 1,045 final),
  `runs/scratch/retrieval_{analysis,knn}.py` (CPU retrieval studies; results were not logged and are redone below).
- One test was left red (tests/test_data.py expected three splits); fixed. `uv run pytest` 122 passed, ruff clean,
  `node --test engine/test/*.test.mjs` 53 passed.
- **The full corpus arrived at 17:43** (runs/write_questions3.log "finished"): all 525 facts have questions, 34
  multi-fact items, chit-chat and out-of-scope from the teacher. `data/yan/qa` was rebuilt by the other process
  with the current build_corpus.py. Frozen copy for the experiments (identical files, checked with shasum):
  `uv run python scripts/build_corpus.py --data-dir data/yan --out runs/data-full2`
  -> 26,236 train / 1,250 val / 1,278 test / 1,210 final records; kinds: single 25,150, meta 1,100, multi 952,
  chitchat 550, oos 2,222. The final split stays unread until the release evaluation.

Unlogged results of session 2 (brasa-4m, seed 0 unless stated, last checkpoint, held-out test)

| run | corpus | steps | training contexts | open EM | EM gold retrieved | EM gold missed | right topic | closed EM |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| p2-4m-fields | partial1 (n=936) | 3000 | full index, shuffled/augmented | 23.7 | 36.6 | 5.4 | 69.2 | 17.7 |
| p3-4m-fields-s1 (same, seed 1) | partial1 | 3000 | same | 22.9 | 34.1 | 7.0 | 70.4 | 17.2 |
| p3-4m-fields-xfit-ranked | partial1 | 3000 | 4 folds, rank order kept, no oracle/drop/distractor | 26.0 | 40.3 | 5.7 | 73.5 | 17.5 |
| f1-4m-fields | full1 (n=1065; 930 fact questions) | 6000 | full index, shuffled/augmented | 31.5 (facts 26.2) | 40.4 | 7.1 | 81.7 | 25.5 (facts 18.4) |

- exp3: keeping the retriever's rank order in cross-fitted training contexts gives 26.0 against 23.7 / 22.9
  for the two seeds of the baseline: +2.7 points, about two standard errors, and the first lever that moved
  "EM when the gold fact is retrieved" (40.3 against 36.6 / 34.1). Kept as a candidate; repeated on the full corpus.
- f1-4m-fields: gold in top 1 / 2 / 4 = 30.1 / 43.6 / 57.5; refusal precision 0.85, recall 0.72 on the
  FALLBACK out-of-scope questions (hand-written frames; an upper bound). Its "best" checkpoint (validation EM,
  step 5750) gives 32.8 open / 26.1 closed: selection on validation EM is no longer harmful, and worth about a point.
- The retriever's first hit is still better than the model on fact questions: 30.1 against 26.2.

Plan for this session
1. GPU now (queue): `f2-4m-fields` (baseline on the full corpus, 6000 steps) and `f2-4m-ranked`.
2. CPU meanwhile: (a) decoding restricted to the RETRIEVED candidates, and retriever + model score fusion
   (the model as a re-ranker of the top k), on the existing f1 checkpoint; (b) "copy the first hit" as a
   model-free baseline row; (c) the learned dual encoder; (d) download of the Wikipedia sample for item 5.
3. Provisional final (brasa-12m) as soon as f2 says which contexts to use; export, card, runs/final.

---

## Entry 6 — 19:05 — pretraining pipeline (programme item 5) implemented and queued

Hypothesis H-pre: a model that has first learned Portuguese and English from public text (still from scratch:
random weights, no pretrained anything) generalises better to unseen phrasings than a model that has only
ever seen the question corpus, because it knows which words are related before it meets the questions.

Implemented
- `scripts/fetch_wiki.py` (`uv run --with pyarrow python scripts/fetch_wiki.py`): the first parquet shard of
  the 2023-11-01 Portuguese and English Wikipedia dumps from the Hugging Face dataset `wikimedia/wikipedia`,
  pinned to revision b04c8d1c; licence CC BY-SA 3.0 and GFDL. Paragraphs of at least 120 characters, articles
  in file order, 160 MB per language: 22,631 pt articles and 32,049 en articles. Everything under
  `data/cache/wiki/` (ignored by git), with `manifest.json` (shards, byte counts, SHA-256 of both text files).
- `src/brasa/pretrain.py`: tokenizer (same BPE, 4096) trained on the first 8 MB of each text file + the facts +
  the TRAINING questions of the data directory (42 s); the text encoded once (104,310,555 tokens, 12.5 s with
  10 processes) and cached in `data/cache/wiki/prep-full2/`; next-token prediction on random 384-token windows,
  same optimiser and schedule as brasa.train; `--resume`, `--stop-at`.
- `brasa.train --init-from <pretrain run>` (weights + its tokenizer; recorded in run.json as "init_from" with
  the provenance of the text) and `--tokenizer <tokenizer.json>` (the control: same vocabulary, random weights).
- tests/test_pretrain.py (5 tests). ruff clean.
- `runs/queue2.sh` + `runs/queue2.txt`: generic GPU queue ("name | command", done markers in runs/logs/),
  started 19:04; it waits for the first queue (f2-4m-fields, f2-4m-ranked) to finish.

Commands queued (runs/queue2.txt)
- `pre-4m`: `uv run python -m brasa.pretrain --preset brasa-4m --corpus data/cache/wiki/prep-full2 --out runs/pre-4m --steps 12000`
  (about 295M tokens seen = 2.8 passes over the text; about 20 minutes)
- `w1-4m-pre`: `brasa.train --preset brasa-4m --steps 6000 --index fields --data-dir runs/data-full2 --init-from runs/pre-4m`
- `w1-4m-tok`: the same without the weights (`--tokenizer data/cache/wiki/prep-full2/tokenizer.json`)
- `pre-12m`: 14,000 steps (about an hour), for the 12m confirmation.
Baseline for both: f2-4m-fields (running).

---

## Entry 7 — 19:20 — decoding restricted to the RETRIEVED facts (experiment C2), the model as a re-ranker

Hypothesis H-cand: half of the model's open-book answers are facts that were not in its context (entry 1).
If the answer may only be made of the facts in the context (or be a fixed answer), those errors turn into a
choice among four candidates, one of which is right 58% of the time.

Implemented
- `brasa.evaluate --modes open_book_retrieved` (AnswerTrie.from_index(..., only=retrieved ids): one small trie
  per question; chains of retrieved facts and the fixed answers stay allowed) and `--modes first_hit` (no
  model: the answer is the retriever's first fact; the baseline every open-book number has to beat).
  Tests in tests/test_evaluate.py. New: "final" documented as a split; open-book modes without a retriever
  raise a clear error.
- JavaScript: `AnswerTrie.fromIndex(index, tok, lang, maxSentences, only)`; `ask` option
  `constrain: true | 'retrieved' | 'kb' | false`. `true` (the default with an index) now means 'retrieved'
  in open-book mode and 'kb' closed-book; `context.constraint` and `done.stats.constraint` report which one.
  Parity fixtures: every constrained test vector now exists twice (whole knowledge base / four facts only:
  "only"), the trie walks have two subset cases. `uv run python scripts/make_fixtures.py` regenerated;
  `node web/engine/tools/parity.mjs tests/fixtures/tiny --wasm web/wasm/kernel.wasm`: PARITY OK, constrained
  12/12 on the four engine variants, trie 65 walks / 2222 steps / 0 mismatches. `node --test` 54 passed,
  `uv run pytest` 134 passed (before the two new evaluate tests), ruff clean.

C2, no retraining: checkpoint runs/f1-4m-fields (last), corpus data-full1, held-out test n=1065 (930 fact
questions, 39 chit-chat, 96 out-of-scope from the FALLBACK frames).
Command: `uv run python -m brasa.evaluate --run runs/f1-4m-fields --checkpoint last --splits val,test --modes open_book_retrieved,first_hit --device cpu --tag r1`

| decoding | EM all | EM fact questions | right topic | EM gold retrieved | unsupported numbers | refusal precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| free (open_book) | 31.5 | 26.2 | 81.7 | 40.4 | 13.6 | 0.85 / 0.72 |
| constrained to the knowledge base | 31.5 | 26.2 | 82.2 | 40.4 | 13.1 | 0.85 / 0.72 |
| constrained to the retrieved facts | **34.6** | **29.7** | 83.7 | 51.6 | **0.0** | 0.85 / 0.74 |
| first hit, no model | 26.2 | 30.0 | 83.4 | 52.2 | 0.0 | - / 0.00 |

Conclusions
1. H-cand holds: +3.1 points of exact match overall and +3.5 on fact questions, for free (no retraining, and
   LESS work in the browser: a smaller trie). Every number in an answer is now a number of a fact shown to
   the visitor (unsupported-number rate 0 by construction), which is the honest version of "grounded".
2. On fact questions the model with this rule equals the retriever's first hit (29.7 against 30.0): it has
   stopped losing what the retriever knew, but adds nothing to the ranking. What it adds is the decision to
   answer, to greet or to decline (first hit: 26.2 overall, it can never decline).

Re-ranking study (runs/scratch/rerank_scores.py + rerank_rules.py, same checkpoint; the model's exact
log-probability of every candidate = top-K retrieved facts + the 14 fixed answers, teacher forced):
- model alone over the top 4 / 8 / 16 candidates: fact EM 31.1 / 29.9 / 29.1 (test), 27.2 / 26.4 / 25.6 (val).
  MORE candidates make it worse: the model's own preference among facts is weaker evidence than the
  retriever's rank.
- fusion log p + 0.3 x BM25 score over the top 8: fact EM 34.1 test / 31.4 val (first hit: 30.5 / 27.7), but
  the fixed answers then lose (out-of-scope refusals fall from 79% to 24%) unless they get a bias. A
  two-stage rule (the model decides fact-or-fixed-answer, the fusion picks the fact) would be worth about +3
  points on fact questions over 'retrieved' greedy decoding, for K extra forward passes in the browser.
  NOT implemented yet: first see what pretraining and the dual encoder do to the ranking itself.

Decision: 'retrieved' is the default constrained decoding of the engine; 'kb' and free decoding stay
available (the site shows both). To confirm on the full-corpus runs (f2) and on brasa-12m.

---

## Entry 8 — 23:55 — session 4 starts: what session 3 left unlogged; the pretraining run was BROKEN

Session 3 was cut at about 19:30 (last entry 19:20). Reconstructed from the tree, queue logs and eval files.

State found
- `runs/queue.sh` finished (f2-4m-fields 19:20, f2-4m-ranked 19:41). `runs/queue2.sh` was still alive.
- **`runs/pre-4m` (pretraining, entry 6) never learned anything and then diverged**: training loss 8.5 to 8.6
  and validation loss 8.2 to 9.7 for the first 8,400 steps (a uniform guess over 4,096 tokens is ln 4096 =
  8.32, so the model was AT CHANCE), NaN from step 8,500 on. The queue then started `w1-4m-pre` from those NaN
  weights at 20:20: NaN loss from the first step, validation EM 0; the laptop slept, the job was still "running"
  at 23:49 (step 600). Killed both the queue and the job. Moved to `runs/broken/pre-4m-nan` and
  `runs/broken/w1-4m-pre-nan` (kept as evidence, not results). No conclusion about H-pre can be drawn yet.
- `runs/queue2.txt` had `enc-4m` inserted as its first line AFTER the queue started; the queue counts lines, so
  it would never have run it. `src/brasa/encoder.py` + tests/test_encoder.py exist (the dual encoder, unlogged;
  only a 6-step smoke run in runs/scratch/enc-smoke). To review before use.
- Unlogged CPU studies (runs/logs/retrieval-knn-full2.log, data-full2, single-fact questions n=1050 per split):
  fields index recall@1/2/4/8 = 27.8/42.1/57.1/73.0 val, 31.7/44.2/59.0/72.9 test; topic of the first hit right
  83.6%. Alias routing (boost facts whose topic alias occurs in the question) x1.25: 28.3/43.8/58.9 val,
  31.9/44.4/59.0 test (+0.5 to +1.8 val, nothing on test; stronger boosts hurt). A nearest-training-question
  vote (kNN over the questions as their own BM25 index) fused with the fields score: at best 29.3/43.6/58.7 val,
  32.6/45.7/59.5 test (+1 point). **Conclusion: the cheap lexical candidates are exhausted at about 28-32%
  recall@1 and 57-60% recall@4**; the topic is already right 84-87% of the time, the miss is among siblings.

Unlogged results: exp4 on the FULL corpus (data-full2, brasa-4m, 6000 steps, seed 0, test n=1278 = 1006 single,
68 multi, 44 meta, 50 chit-chat, 110 out-of-scope; gold in top 1/2/4 = 29.8/42.6/57.3)

| run | ckpt | training contexts | open EM | open F1 | right topic | unsupported numbers | refusal P / R | closed EM |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| f2-4m-fields | last | full index, shuffled/augmented | 32.8 | 51.7 | 83.0 | 15.3 | 0.71 / 0.87 | 25.7 |
| f2-4m-fields | best (val EM) | same | 31.9 | 50.9 | 81.4 | 15.0 | 0.67 / 0.85 | 25.1 |
| f2-4m-ranked | last (= best) | 4 folds, rank order kept | 33.4 | 51.8 | 81.8 | 9.5 | 0.72 / 0.89 | 25.7 |

- "ranked" contexts on the full corpus: 33.4 against 32.8, inside the noise (it was +2.7 on the partial corpus).
  It does lower the unsupported-number rate (9.5 against 15.3: the model answers from the context more often).
  Not a clear win for exact match; it becomes irrelevant under 'retrieved' decoding (to check).
- Refusals with the TEACHER's out-of-scope questions (110 in test): precision 0.71, recall 0.87 open-book.
- The helper `runs/scratch/summ.py` prints eval-*.json files as one line per mode.

Next (this session)
1. Find the pretraining bug (loss at chance from step 0 is a bug, not a hyper-parameter), fix, test, rerun.
2. Review the encoder code; train `enc-4m`; recall@1/2/4 against the fields index; fusion.
3. C2 decoding ('retrieved') on f2; then the 12m provisional final with what is known, so that runs/final
   exists early; refresh if the encoder or pretraining changes the recipe.

---

## Entry 9 — 2026-11-13 00:00 — the pretraining bug: float16 gradients underflow (affects EVERY earlier run)

Diagnosis. Pretraining with `--amp off` for 800 steps: loss 7.56 -> 3.93, validation perplexity 60 (the broken
run was at 8.45 at step 800). So float16 autocast was the cause. Mechanism: the loss is computed in float32 but
the backward pass carries d loss / d logits = (softmax - one_hot) / n_tokens through float16 activations, whose
smallest positive value is 6e-8. With 24,576 tokens per batch (plain text, every token counts) a wrong token
of probability 1/4096 contributes 1e-8: flushed to zero. Only the one-hot terms survive, so the model gets a
biased gradient, drifts to worse than uniform as the learning rate warms up, then NaN.

Fix (src/brasa/train.py, used by train, pretrain and encoder): `make_scaler` (torch.amp.GradScaler when amp is
fp16; works on MPS with torch 2.15), `optimizer_step` (scaled backward, unscale, clip, step), the scaler state
in ckpt_last.pt; `require_finite` stops a run whose loss is NaN/inf at the next log line; `finite_state`
refuses `--init-from` weights that are not finite.
Check: 800 pretraining steps with fp16 + scaling: loss 3.9286, val ppl 60.2 at 222k tokens/s, against
3.9288 / 60.2 at 198k tokens/s in float32. Same result, full speed.

**Consequence for everything above this entry**: all question-corpus runs (p1, p2, p3, f1, f2) were trained in
float16 WITHOUT loss scaling. There the loss only counts answer tokens (a few thousand per batch), so the
one-hot part of the gradient survived (the runs reach 100% on training questions), but the push against wrong
tokens of low probability was quantised or lost. The comparisons between those runs are still like for like;
their absolute level may be too low. exp8 measures it.

exp8, queued 23:57 with the new restart-safe queue (`runs/queue3.sh`, jobs in `runs/queue3.txt`, log
`runs/queue3.log`, markers runs/logs/<name>.done|.failed; evaluation modes
open_book, closed_book, open_book_retrieved, first_hit on val,test for the last and best checkpoints):
- `f3-4m-fields` = f2-4m-fields again (brasa-4m, 6000 steps, fields, data-full2, seed 0) with the fix.
  Hypothesis: equal or better than f2 (32.8 open / 25.7 closed EM, test n=1278).
- then `pre-4m` (12,000 steps, about 22 min), `w2-4m-pre` (fine-tune from it), `w2-4m-tok` (control: the
  Wikipedia tokenizer, random weights).

Meanwhile (CPU): tests for the scaler helpers, review of encoder.py, then the dual encoder goes in the queue.

---

## Entry 10 — 00:09 — CPU work while f3 trains: novelty analysis, knowledge-base variant, one network for both jobs

1. How hard is the held-out test? `runs/scratch/novelty.py runs/f2-4m-fields/predictions-last.jsonl` (single-fact
   test questions, n=1050, open-book free decoding): similarity = highest word-set Jaccard between the test
   question and any TRAINING question of the same fact.

   | similarity to the nearest training phrasing | n | EM | right topic | gold in top 1 | gold in top 4 |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | 0.00 to 0.20 | 204 | 16.2 | 73.5 | 18.1 | 39.2 |
   | 0.20 to 0.35 | 686 | 26.1 | 87.5 | 29.9 | 59.6 |
   | 0.35 to 0.50 | 149 | 49.0 | 90.6 | 57.0 | 81.2 |
   | 0.50 to 1.00 | 11 | 63.6 | 90.9 | 54.5 | 81.8 |

   85% of the test questions share less than 35% of their words with every training phrasing of their fact: the
   teacher wrote them to be different. Accuracy climbs steeply with similarity (16 -> 26 -> 49 -> 64). The test
   split measures the hard end of what a visitor can ask; to be said next to every headline number.

2. Programme item 6 prepared: `scripts/build_corpus.py --min-importance N --out DIR` (drops the facts below N
   and every question that needs one; refuses to shrink data/yan in place; tests/test_build_corpus.py, also the
   first tests of the split rules and the key-span fix). Without the flag the output is byte-identical to
   runs/data-full2 (checked with cmp). `runs/data-imp4`: 362 facts of importance 4 and 5,
   18,944 train / 914 val / 942 test / 884 final. Queued as `k1-4m-imp4`.

3. exp10, hypothesis H-joint: ONE network can be the reader and the retriever. `brasa.train --retrieval-weight w`
   adds, at every step, w x the contrastive loss of encoder.py (128 questions, their gold facts, one same-topic
   negative each; mean-pooled final hidden states, temperature 0.05) to the language-model loss. The run logs
   `retrieval_loss`, `val_dense_recall@1/4`, and at the end chooses the fusion weight (BM25 / best BM25 + weight
   x cosine) on the validation questions -> run.json "retrieval". `brasa.evaluate` then retrieves with the fused
   ranking for such a run (`--retriever lexical` switches it off; `--encoder RUN` fuses a stand-alone encoder).
   Why it matters for the site: scripts/sync-brasa.mjs copies ONE model file, so a second network would have
   to be smuggled into it; a shared network costs no extra download, only precomputed fact vectors in
   retrieval.json and one short forward pass for the question.
   Tests: tests/test_encoder.py::test_one_network_can_learn_to_answer_and_to_retrieve. pytest 141 passed, ruff clean.

Queue now (runs/queue3.txt): f3-4m-fields (running, step 3750) -> enc-4m (stand-alone encoder, from scratch)
-> j1-4m-joint (`--retrieval-weight 1.0`, evaluated with the fused ranking and with `--retriever lexical`)
-> pre-4m -> w2-4m-pre -> enc-4m-pre -> w2-4m-tok -> k1-4m-imp4.
`runs/q.sh tr|ev` = helpers for queue lines (external file, editable while the queue runs; evaluates the last
checkpoint only, 4 modes on val,test).

---

## Entry 11 — 00:30 — exp8 result (the float16 fix does not change the question-corpus runs); dense retrieval in the engine

exp8: `runs/f3-4m-fields` = f2-4m-fields with loss scaling (brasa-4m, 6000 steps, fields, data-full2, seed 0).
Held-out test n=1278 (1006 single, 68 multi, 44 meta, 50 chit-chat, 110 out-of-scope); gold in top 1/2/4 = 29.8/42.6/57.3.

| run | ckpt | decoding | EM all | EM fact questions | F1 | right topic | unsupported numbers | refusal P / R |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| f2-4m-fields (no loss scaling) | last | open-book free | 32.8 | - | 51.7 | 83.0 | 15.3 | 0.71 / 0.87 |
| f2-4m-fields (no loss scaling) | last | closed-book free | 25.7 | - | 45.1 | 76.6 | 53.4 | 0.64 / 0.86 |
| f3-4m-fields | last | open-book free | 33.3 | 28.6 | 51.9 | 80.4 | 11.8 | 0.69 / 0.88 |
| f3-4m-fields | last | closed-book free | 25.4 | 19.6 | 45.0 | 74.0 | 46.6 | 0.63 / 0.86 |
| f3-4m-fields | last | open-book, retrieved facts only | **34.5** | 29.8 | 52.9 | 81.7 | 0.0 | 0.68 / 0.90 |
| f3-4m-fields | last | first hit, no model | 26.0 | 29.7 | 45.0 | 81.7 | 0.0 | - / 0.00 |
| f3-4m-fields | best (val EM) | open-book, retrieved facts only | 33.8 | 29.2 | 52.7 | 82.6 | 0.0 | 0.71 / 0.86 |

Conclusions
1. The float16 underflow did NOT hurt the question-corpus runs: 33.3 / 25.4 against 32.8 / 25.7 (inside the
   noise), validation EM 0.256 against 0.266. Everything measured before entry 9 stands. The fix only matters
   where every token carries loss (pretraining), and there it is the difference between learning and not.
2. C2 confirmed on the full corpus with the teacher's out-of-scope questions: decoding restricted to the
   retrieved facts gives 34.5 against 33.3 free (+1.2), with zero unsupported numbers. On fact questions the
   model equals the retriever's first hit (29.8 against 29.7): the ranking IS the answer. The model's own
   contribution is the decision to decline (out-of-scope EM 90%), to greet, and to chain facts (multi 20.6
   against 0 for the first hit).
3. "best" (validation EM) is not better than "last" (33.8 against 34.5): keep the last checkpoint.
4. Chit-chat is weak on held-out phrasings: 18% exact (n=50). The held-out greetings are unusual on purpose
   ("Adieu", "Despeço-me cordialmente") and are mostly answered with the refusal (23 of 50), which is a
   tolerable failure; only 6 of 50 get a fact. 80 training phrasings per category is what there is.

Provisional release pipeline: `scripts/release.sh <run> [runs/final]` (evaluate in six modes -> export and
evaluate float16/int8 -> Node parity + benchmark -> MODEL_CARD.md -> runs/final with RELEASE.json). SPLITS
defaults to train,val,test: the final split is added by hand only for the real release. Running on the CPU
for f3-4m-fields (log runs/logs/release-f3.log) so that runs/final holds a working model early.
Model card: scripts/render_model_card.py + MODEL_CARD.template.md now have the visitor table (curated
sentence, right topic, gold in top 1/2/4, EM when retrieved / missed), per-language / per-kind / per-topic
breakdowns, the decoding modes, the retriever used, and the selection caveat. tests/test_model_card.py.

Dense retrieval, end to end (programme item 3, the export half; built before its numbers because both designs,
shared network or separate encoder, need the same engine work):
- Index format: `retrieval.json` may carry `"dense"`: one int8 vector per document (scale = max |v| / 127),
  the fusion weight, the pooling. `Bm25Retriever.search(query, k, lang, vector)` fuses
  bm25 / (best bm25) + weight x cosine; the question vector is rounded to int16 first, so the dot products are
  integer sums and Python and JavaScript agree BIT FOR BIT given the same vector. Without a vector: the
  lexical search, unchanged (the fallback).
- `encoder.DenseRetriever` (fused) now goes through that index arithmetic: what evaluation measures is what
  the browser computes, int8 vectors included.
- `brasa.export`: for a run with run.json "retrieval" (train --retrieval-weight) the fact vectors are computed
  with the weights of the default model file (int8) and written into retrieval.json; `manifest.dense_retrieval`;
  testvec.<fmt>.json gets `embed` (pooled vectors of questions and facts), parity.json gets `dense` (24
  questions, their vectors, fused hits). The quantised weights are evaluated with their own fused ranking
  and the same modes as the float32 checkpoint (`--eval-modes`, `--retriever`).
- JavaScript: `Engine.embed(tokens, pooling)` (pooled final hidden states, unit vector; resets the context),
  `Retriever` decodes `index.dense`, `search(query, k, lang, vector)`, `quantizeQuery`; `Session`: ask and
  retrieve embed the question first when the index has vectors (`dense: false` = lexical only),
  `context.retriever` / `hits.retriever` = 'bm25+dense' | 'bm25', `ready.info.dense`. Declarations updated.
- Parity on tests/fixtures/tiny (regenerated, 760 KB): embed max |diff| 2.6e-7 on the four engine variants;
  fused search with Python's vector: 48 searches, 0 ranking and 0 score mismatches; with the engine's own
  vector (int8 weights): 48/48 identical rankings, largest score difference 7.6e-6.
- Tests: `node --test` 58 passed (54 before), tsc clean, pytest green (counts at the next entry).

enc-4m (stand-alone encoder from scratch) is training: validation recall@1/4/8 after 750 steps =
25.8 / 52.8 / 66.2 for the dense ranking ALONE (BM25 fields on the same questions: about 28 / 57 / 73).

---

## Entry 12 — 2026-11-13 04:55 — session 5 starts: the queue of session 4 finished; its results (exp6, exp7, exp9, exp10, exp11)

Session 4 was cut at about 00:32 (last entry 00:30). The GPU queue (`runs/queue3.sh`) ran every job to the end
(03:07) and is still alive and idle. Reconstructed from the queue log, run directories and eval files.

State found
- One test was left red (tests/test_sample.py: the tiny fixture model may write 16 tokens, so its constrained
  answer is the START of an allowed sentence; the assertion now says so) and two lint errors in
  src/brasa/sample.py (the terminal chat now answers like the browser: `--constrain auto|retrieved|kb|off`,
  `--lexical`, fused ranking from an exported index; unlogged work of session 4). Fixed:
  `uv run ruff check .` clean, `uv run pytest` 148 passed, `node --test engine/test/*.test.mjs` 58 passed.
- `runs/final` holds the provisional release of f3-4m-fields (00:56; brasa-4m baseline, no learned ranking).
- README.md has NOT been updated with real results yet (last change 2026-11-12 08:48).
- New helper: `runs/scratch/summ.py` now prints fact-question EM, EM when the gold fact was retrieved / missed
  and EM per question kind.

All runs below: brasa-4m, 6000 steps, seed 0, fields index, data-full2, LAST checkpoint, held-out test n=1278
(1006 single, 68 multi, 44 meta, 50 chit-chat, 110 out-of-scope). "retrieved" = decoding restricted to the
retrieved facts and the fixed answers (the site's default).

| run | what changed | gold in top 1 / 2 / 4 | free open EM | retrieved EM | single | multi | meta | chit-chat | out-of-scope | closed EM | val: retrieved EM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f3-4m-fields | baseline | 29.8 / 42.6 / 57.3 | 33.3 | 34.5 | 30.1 | 20.6 | 36.4 | 18.0 | 90.0 | 25.4 | 29.5 |
| j1-4m-joint | + contrastive loss, own ranking fused with BM25 (weight 1.0) | **39.0 / 55.2 / 70.3** | 35.4 | **37.6** | 34.8 | 22.1 | 40.9 | 20.0 | 80.0 | **29.7** | 34.6 |
| j1-4m-joint, `--retriever lexical` | the same weights, BM25 contexts | 29.8 / 42.6 / 57.3 | 34.0 | 34.2 | 29.9 | 16.2 | 31.8 | 18.0 | 93.6 | 29.7 | 31.5 |
| d1-4m-drop50 | word dropout in half of the training questions | 29.8 / 42.6 / 57.3 | 31.7 | 32.2 | 29.4 | 19.1 | 20.4 | 14.0 | 78.2 | 22.3 | 27.8 |
| w2-4m-tok | Wikipedia tokenizer, random weights (control) | 29.8 / 42.6 / 57.3 | 30.4 | 31.5 | 28.2 | 17.6 | 20.4 | 14.0 | 82.7 | 25.9 | 31.3 |
| w2-4m-pre | Wikipedia tokenizer + pretrained weights | 29.8 / 42.6 / 57.3 | 33.2 | 33.5 | 29.8 | 19.1 | 34.1 | 14.0 | 84.5 | 26.2 | 29.4 |
| k1-4m-imp4 (its own test set, n=942) | only the 362 facts of importance 4-5 | 31.8 / 46.2 / 61.8 | 35.1 | 34.9 | 28.1 | 22.4 | 25.0 | 24.0 | 92.7 | 29.2 | 30.2 |

Stand-alone dual encoder (programme item 3), single- and multi-fact questions, n=1118 per split, recall@1/2/4:

| retriever | val | test |
| --- | --- | --- |
| BM25 fields (lexical) | 26.1 / 40.8 / 55.6 | 29.8 / 42.6 / 57.3 |
| enc-4m, dense alone (from scratch, 3000 steps) | 31.0 / 43.5 / 56.4 | 32.7 / 46.0 / 59.3 |
| enc-4m fused with BM25 (weight 1.0, chosen on val) | 37.0 / 53.3 / 67.9 | 39.3 / 55.0 / 69.1 |
| j1-4m-joint, its own ranking alone | 31.0 / 45.6 / 58.6 | 34.5 / 48.2 / 62.1 |
| j1-4m-joint fused with BM25 (weight 1.0) | 35.9 / 53.4 / 68.7 | 39.0 / 55.2 / 70.3 |
| enc-4m-pre, dense alone (initialised from the Wikipedia pretraining) | 37.3 / 53.8 / 65.7 | 41.2 / 55.7 / 69.1 |
| enc-4m-pre fused with BM25 (weight 2.0, chosen on val) | **41.8 / 57.3 / 71.7** | **44.5 / 59.0 / 75.1** |

Pretraining run (exp6, the rerun with the float16 fix): `runs/pre-4m`, 12,000 steps, 1,475 s, 295M tokens seen,
training loss 2.99, validation perplexity 25.0 (the broken run sat at chance).

Conclusions
1. **H-joint holds (exp10).** One network that answers and ranks: gold in top 1 from 29.8 to 39.0%, in top 4 from
   57.3 to 70.3% (the same as a separate encoder of the same size: 39.3 / 69.1), 'retrieved' EM 34.5 -> 37.6 on
   test and 29.5 -> 34.6 on validation, and closed-book EM 25.4 -> 29.7: the contrastive loss also makes the
   weights a better memory. No extra download. This is the first lever that beats the lexical first hit.
2. Two problems the joint model brings, both visible with `runs/scratch/hybrid.py`:
   (a) out-of-scope refusals fall from 90 to 80% (validation: 89 -> 68). With `--retriever lexical` the same
   weights refuse 94%: the model was trained with BM25 contexts and is asked with fused contexts, in which
   an out-of-scope question now gets four facts that look related. A train/test mismatch, fixable.
   (b) on single-fact questions the reader is now BEHIND its own ranking: 34.8 against 41.5 for the first hit
   (validation 33.4 against 38.7). Rule "the model decides whether to answer with a fact, a chain or a fixed
   answer; a single fact is replaced by the first hit": 41.9 overall against 37.6 (validation 37.4 against
   34.6). The reader needs the rank as a prior.
3. **Pretraining (exp6/exp7): it helps the encoder, not the reader.** As a reader, the pretrained model equals
   the baseline (33.5 against 34.5; against its proper control with the same tokenizer 31.5: +2.0, one
   standard error and a half). As an encoder the gain is large and the same on validation and test: dense
   alone 32.7 -> 41.2 recall@1, fused 39.3 -> 44.5 recall@1 and 69.1 -> 75.1 recall@4. Knowing which words are
   related is exactly what matching an unseen phrasing needs. Also: the Wikipedia tokenizer alone costs about
   3 points (31.5 against 34.5), which the pretrained weights win back.
4. Word dropout (exp11) hurts everywhere (32.2 against 34.5; closed 22.3 against 25.4). Dropped.
5. Knowledge-base size (exp9), first look: the smaller knowledge base is NOT easier per question (single-fact
   EM 28.1 against 30.1 on different question sets). A like-for-like comparison on the shared questions follows.

Next (this session; GPU queue first, CPU work in parallel)
- GPU now: `j2-4m-joint-pre` = j1 initialised from runs/pre-4m (does the encoder gain of pretraining survive in
  the shared network?), then `pre-12m` (10,000 steps, about an hour; needed for the 12m confirmation).
- CPU: (1) rank prior in constrained decoding (the retriever's order as a bias on the allowed tokens: no extra
  forward pass), Python then JavaScript with parity; (2) `--contexts-from RUN`: training contexts ranked by a
  fused retriever, so that training and asking see the same kind of context (problem a);
  (3) like-for-like knowledge-base comparison; (4) README results section.
- Then the 12m recipe: pretrained init + joint loss + fused training contexts + rank prior, with a 12m
  baseline and a 12m joint-from-scratch run as controls; release with `scripts/release.sh`.

---

## Entry 13 — 05:20 — the retriever's ranking as a prior in constrained decoding (experiment C3); `--contexts-from`

Hypothesis H-prior (from entry 12, conclusion 2b): the reader loses against its own ranking on single-fact
questions; adding the retriever's opinion to the model's log-probabilities of the allowed sentences recovers
that, and costs nothing at inference time.

Implemented
- `constrain.py`: every sentence of an `AnswerTrie` may carry a prior in nats (`add(..., prior=)`,
  `from_index(..., priors={fact id: nats}, fixed_prior=)`); `bias(state)` gives each allowed token the best
  prior among the sentences it still leads to (<|eos|>: the prior of the sentence that ends there); greedy
  decoding adds it to the log-probabilities at every branch. The prior is about the FIRST sentence only: a
  following sentence is worth what stopping is worth, so the stop-or-go-on decision of a multi-fact answer
  stays the model's (a first version penalised every continuation: multi-fact EM fell from 23.5 to 5.9 on
  validation). `ranking_priors(hits, weight, kind)`: "rank" = -weight x position, "score" = -weight x
  (1 - score / best score). No extra forward pass; with all priors at 0 the code path is the old one.
- `brasa.evaluate`: mode `open_book_ranked` (= open_book_retrieved + the prior), `--prior rank|score`,
  `--prior-weight`, `--fixed-prior`, and `--tune-prior`: sweeps the weight on the VALIDATION questions, keeps
  the smallest weight within half a point of the best exact match and records it in run.json ("decoding").
- `brasa.export`: the chosen prior travels in retrieval.json ("answers" -> "prior") and manifest.json
  ("decoding_prior"); the quantised weights are evaluated with it.
- JavaScript (inside the files the sync script copies): `rankingPriors`, `AnswerTrie.add/fromIndex` with
  priors, `AnswerTrie.bias`, `chooseAllowed(..., bias)`, `Engine.stream` adds the bias when the constraint has
  one; `Session.ask` option `prior` (omitted = the index's own; false / 0 = the model alone; a number = that
  weight; an object = merged), `context.prior`, `ready.info.prior`. Declarations updated.
- Parity fixtures: every constrained test vector now exists three times (whole knowledge base / four facts /
  four facts with priors, built so that the prior changes the answer: it does in 2 of 6), the trie walks have
  a setting with priors that records the bias of every allowed token.
  `node web/engine/tools/parity.mjs tests/fixtures/tiny --wasm web/wasm/kernel.wasm`: PARITY OK, constrained
  18/18 (6 with a prior) on the four engine variants, trie 78 walks / 2721 steps (499 with priors), 0 mismatches.
- Tests: tests/test_constrain.py (4 new), tests/test_evaluate.py (ranked mode, tune_prior), Node: retrieval,
  engine, session, parity. `uv run pytest` 154 passed, ruff clean, `node --test` 62 passed, `tsc` clean.

C3, no retraining: runs/j1-4m-joint (last), fused retriever, `runs/scratch/prior_sweep.py` on the CPU
(logs runs/logs/prior-sweep-j1*.json). EM in %, validation n=1250, test n=1278.

| prior | weight | val EM | val single | val multi | val out-of-scope | test EM | test single | test multi | test out-of-scope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none (open_book_retrieved) | 0 | 34.6 | 33.4 | 23.5 | 68.3 | 37.6 | 34.8 | 22.1 | 80.0 |
| rank | 1 | 36.2 | 35.3 | 22.1 | 70.7 | | | | |
| rank | 4 | 37.2 | 37.3 | 11.8 | 72.0 | 40.3 | 38.3 | 13.2 | 85.5 |
| rank | 16 | 36.5 | 36.5 | 11.8 | 72.0 | 41.0 | 39.1 | 13.2 | 85.5 |
| score | 8 | 36.6 | 35.6 | 22.1 | 72.0 | | | | |
| score | 16 | 37.0 | 36.3 | 20.6 | 72.0 | 40.8 | 38.6 | 20.6 | 83.6 |
| score | 32 | 37.2 | 36.9 | 17.6 | 72.0 | 40.5 | 38.3 | 17.6 | 84.5 |
| score | 48 | 37.4 | 37.2 | 16.2 | 72.0 | | | | |
| first hit, no model | | 32.1 | 38.7 | 0 | 0 | 34.0 | 41.5 | 0 | 0 |

Conclusions
1. H-prior holds: +2.4 to +2.8 points on validation and about +3 on test, at no cost. Out-of-scope refusals
   also improve (a lower-ranked fact now has to beat the fixed answers by more).
2. "score" is the better form: it reaches the same exact match as "rank" and keeps the multi-fact chains
   (20.6 against 11.8 on validation at the comparable setting), because a near tie between two facts costs
   nothing and the model still chooses the order. Default kind: score; weight chosen per model by --tune-prior.
3. The reader is still not better than the first hit on single-fact questions (36-37 against 38.7 on
   validation): what the language model adds over its own ranking is the decision to decline, to greet and
   to chain facts. Worth saying plainly on the site.

`brasa.train --contexts-from RUN` (problem 2a of entry 12): the training and validation contexts are ranked
by the fused retriever of another run (a `--retrieval-weight` run or a `brasa.encoder` run) over this run's
own index; `build_examples` embeds all open-book questions in one batch first. Recorded in run.json
("contexts_from"). Test in tests/test_encoder.py.

Queue (runs/queue3.txt; `runs/q2.sh` = q.sh + prior tuning + the open_book_ranked mode):
j2-4m-joint-pre (running, step 4800) -> j2-ranked -> `j3-4m-joint-pre-ctx` (exp13: j2 + `--contexts-from
runs/j2-4m-joint-pre`; hypothesis: out-of-scope refusals return to about 90% without losing the ranking) ->
pre-12m.

---

## Entry 14 — 05:32 — exp12: pretraining + the shared network (the best model so far); knowledge-base size (item 6); a negative result

exp12 `runs/j2-4m-joint-pre` = j1-4m-joint initialised from runs/pre-4m (brasa-4m, 6000 steps, seed 0, fields,
data-full2, 1,720 s of training). Command in runs/queue3.txt. Held-out test n=1278, LAST checkpoint.

| run | gold in top 1 / 2 / 4 (fused) | dense alone, recall@1 / 4 | free open EM | retrieved EM | single | multi | meta | out-of-scope | refusal P / R | right topic | closed EM | val EM (teacher-forced) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| f3-4m-fields (baseline) | 29.8 / 42.6 / 57.3 | - | 33.3 | 34.5 | 30.1 | 20.6 | 36.4 | 90.0 | 0.68 / 0.90 | 81.7 | 25.4 | 25.6 |
| w2-4m-pre (pretrained, reader only) | 29.8 / 42.6 / 57.3 | - | 33.2 | 33.5 | 29.8 | 19.1 | 34.1 | 84.5 | 0.72 / 0.85 | 82.0 | 26.2 | 28.2 |
| j1-4m-joint (shared network) | 39.0 / 55.2 / 70.3 | 34.5 / 62.1 | 35.4 | 37.6 | 34.8 | 22.1 | 40.9 | 80.0 | 0.67 / 0.81 | 85.9 | 29.7 | 27.7 |
| **j2-4m-joint-pre (both)** | **45.0 / 60.6 / 74.8** | 41.4 / 69.0 | **42.6** | **43.8** | 41.2 | 32.4 | 47.7 | 84.5 | 0.75 / 0.86 | 87.0 | **34.7** | 33.3 |

(fusion weight 1.5, chosen on validation: val recall@1/4 = 41.1 / 70.8.)

Conclusions
1. **Pretraining and the contrastive objective are complementary: +9.3 points of exact match over the baseline**
   (34.5 -> 43.8 with the site's decoding, 25.4 -> 34.7 closed-book), where each alone gave -1.0 and +3.1. Read
   together with entry 12: general language knowledge only pays once the network is asked to MATCH questions
   to facts (the contrastive loss); next-token training on 26k questions alone washes it out. Every question
   kind gains (single 30.1 -> 41.2, multi 20.6 -> 32.4, meta 36.4 -> 47.7); refusal precision rises (0.68 -> 0.75).
2. The shared network keeps the whole gain of the separate pretrained encoder (enc-4m-pre fused 44.5 / 59.0 /
   75.1 against 45.0 / 60.6 / 74.8 here). Programme item 5 answered at brasa-4m: yes, it helps, by 6 points over
   the same recipe without pretraining (37.6 -> 43.8), one run each; the validation gap is the same size
   (teacher-forced EM 27.7 -> 33.3).
3. Out-of-scope recall is 84.5% (baseline 90.0): the mismatch of entry 12 is smaller with the better encoder
   but still there. exp13 (j3, `--contexts-from`) is queued to close it.
4. The validation curves still rise at step 6000 (j2: 31.5 at 4500, 33.6 at 5500, 33.3 at 6000), mostly the
   cosine schedule; a longer schedule is a cheap lever to try after the 12m run.

Programme item 6, knowledge-base size: `runs/scratch/kb_size.py`. k1-4m-imp4 (trained on the 362 facts of
importance 4-5) evaluated on the questions of the FULL corpus
(`brasa.evaluate --run runs/k1-4m-imp4 --data-dir runs/data-full2 --tag last-full2 --device cpu`), against
f3-4m-fields (all 525 facts): the same 1278 test questions, 'retrieved' decoding, seed 0 each.

| questions (test) | n | all facts: EM | all facts: gold in top 4 | important facts only: EM | gold in top 4 | refused | answered with another fact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| about a fact the small knowledge base keeps | 782 | 28.9 | 54.9 | 27.5 | 61.8 | 3.2 | 69.2 |
| about a fact it drops | 336 | 31.8 | 63.1 | 0.0 | 0.0 | 7.1 | 92.9 |
| out of scope | 110 | 90.0 | | 92.7 | | 92.7 | 6.4 |
| all | 1278 | 34.5 | | 25.7 | | | |

Validation agrees: kept facts 24.3 (all) against 25.3 (important only); closed-book test, kept facts: 19.1 against 21.6.
Conclusion: **a smaller knowledge base is not a more accurate one.** Retrieval gets easier (gold in top 4 from
55 to 62% on the facts it keeps) but exact match on those questions does not move (-1.4 test, +1.0 validation:
noise; +2.5 closed-book), and the 30% of fact questions that fall outside are answered with a neighbouring
fact 93% of the time (right topic 70%), not declined: the model does not know what was removed. All 525 facts
stay.

Negative result (CPU, `runs/scratch/dense_expand.py runs/j2-4m-joint-pre`): moving each fact vector towards the
centroid of the vectors of its training questions (document expansion in vector space) changes nothing in
the fused ranking (alpha 0 / 1 / 4: test 45.0 / 44.5 / 44.3 recall@1, 74.8 / 75.3 / 75.2 recall@4) and at most +1.5
points for the dense ranking alone. The contrastive loss has already put the fact where its questions are.

Decision: the final recipe starts from a pretraining run and uses the shared network. Queue: j2-ranked (prior
tuning for j2, running) -> j3-4m-joint-pre-ctx -> pre-12m -> the 12m model.

---

## Entry 15 — 05:45 — the ranking prior on the best model (C3 on j2); provisional release of j2

`runs/j2-4m-joint-pre` (last), fused retriever, `runs/scratch/prior_sweep.py` (CPU; runs/logs/prior-sweep-j2-*.json).
Validation n=1250, test n=1278. The weight is chosen on validation only.

| prior | weight | val EM | val single | val multi | test EM | test single | test multi | test meta | test out-of-scope | test refusal P / R | test right topic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| none (open_book_retrieved) | 0 | 39.4 | 38.6 | 25.0 | 43.8 | 41.2 | 32.4 | 47.7 | 84.5 | 0.75 / 0.86 | 87.0 |
| score | 12 | 41.8 | 41.5 | 22.1 | 46.4 | 44.5 | 27.9 | 50.0 | 85.5 | 0.74 / 0.87 | 87.8 |
| **score** | **24** | 41.9 | 41.9 | 19.1 | **47.6** | 46.0 | 23.5 | 56.8 | 85.5 | 0.74 / 0.87 | 88.0 |
| score | 48 | 42.4 | 43.0 | 13.2 | 47.7 | 46.5 | 20.6 | 54.5 | 85.5 | 0.72 / 0.87 | 87.6 |
| rank | 2 | 42.0 | 41.9 | 19.1 | 46.7 | 45.1 | 23.5 | 52.3 | 85.5 | 0.73 / 0.87 | 88.0 |
| rank | 4 | 42.6 | 43.2 | 13.2 | 46.9 | 45.7 | 19.1 | 52.3 | 85.5 | 0.72 / 0.87 | 87.1 |
| first hit, no model | | | | | 39.3 | 47.5 | 0 | 54.5 | 0 | | 88.6 |

- `--tune-prior` (score, smallest weight within half a point of the best validation EM) chooses **score, weight
  24**: validation 39.4 -> 41.9, test 43.8 -> 47.6. Recorded in runs/j2-4m-joint-pre/run.json ("decoding").
  (The first automatic tuning in the queue used kind "rank" by mistake of a default and chose rank 4; the
  default kind is now "score" everywhere; eval-last.json of j2 still holds the rank-4 row: test 46.9.)
- The cost is on multi-fact questions (68 of 1278): 32.4 -> 23.5 on test. With a prior the answer starts more
  often with the first hit, and a chain that starts with the wrong fact is not continued the way the gold
  answer is (one-fact answers to multi-fact questions: 44% -> 59% with rank 4; the set of facts matches gold as
  often as the exact text does, so this is not an artefact of the order). It is the price of +5 points on the
  79% of questions that have one fact. The "score" form pays less of it than "rank" at the same total.
- brasa-4m today, with the site's decoding: **47.6% exact / 88.0% right topic** on held-out test phrasings,
  against 33.3% / 80.4% for the first full-corpus baseline with free decoding (f3).

Provisional release (CPU, in the background, log runs/logs/release-j2.log):
`DEVICE=cpu SPLITS=train,val,test scripts/release.sh runs/j2-4m-joint-pre runs/final`. release.sh now tunes the
prior first when the run has none, evaluates seven modes (open_book_ranked added) and records the prior and the
pretraining run in RELEASE.json. The model card (scripts/render_model_card.py, MODEL_CARD.template.md) names
the site's decoding, the prior, the learned ranking with its recall, and where the weights started
("still from scratch": the pretraining run, its text, licence and slice). tests/test_model_card.py covers it.
`brasa.sample` answers with the same prior as the browser (`--prior-weight`).
Checks: `uv run pytest` 156 passed, ruff clean, `node --test` 62 passed, tsc clean.

Running: j3-4m-joint-pre-ctx (GPU, contexts from j2's fused retriever: training gold-in-context 89.1%, 209 s of
preparation), then pre-12m.

---

## Entry 16 — 06:27 — exp13 result (`--contexts-from`: not worth a second generation); the fixed-answer prior; the 12m run is queued

exp13 `runs/j3-4m-joint-pre-ctx` = j2 with the training and validation contexts ranked by j2's fused retriever
(brasa-4m, 6000 steps, seed 0, 1,923 s). Prior chosen on validation for each run (j2: score 24; j3: score 48).

| run | split | gold in top 1 / 4 | retrieved EM | ranked EM | single | multi | small talk | out-of-scope (ranked) | refusal P / R (ranked) | closed EM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| j2-4m-joint-pre | val | 41.1 / 70.8 | 39.4 | 41.9 | 41.9 | 19.1 | 22.0 | 80.5 | 0.65 / 0.80 | 30.7 |
| j3-4m-joint-pre-ctx | val | 42.4 / 71.9 | 40.1 | 43.1 | 42.8 | 14.7 | 32.0 | 84.2 | 0.63 / 0.85 | 29.8 |
| j2-4m-joint-pre | test | 45.0 / 74.8 | 43.8 | 47.6 | 46.0 | 23.5 | 20.0 | 85.5 | 0.74 / 0.87 | 34.7 |
| j3-4m-joint-pre-ctx | test | 43.1 / 74.8 | 43.2 | 46.7 | 45.3 | 25.0 | 18.0 | 87.3 | 0.71 / 0.90 | 33.7 |

Conclusion: the hypothesis is half right. Out-of-scope refusals do come back a little (80.5 -> 84.2 on validation,
85.5 -> 87.3 on test; the lexical baseline had 89 / 90), but exact match does not move (+1.2 validation, -0.9 test)
and the cost is a second training generation that depends on the first. Not part of the recipe. The flag and
its test stay in the code as a documented negative result.

The prior of the fixed answers (`--fixed-prior`, `ask({ prior: { fixed } })`), j2, score 24, CPU
(runs/logs/prior-sweep-j2-fixed.json):

| fixed prior (nats) | val EM | val refusal P / R | test EM | test refusal P / R |
| ---: | ---: | --- | ---: | --- |
| -3.0 | 41.8 | 0.76 / 0.76 | 47.5 | 0.78 / 0.79 |
| -1.5 | 41.6 | 0.67 / 0.77 | 47.5 | 0.76 / 0.82 |
| 0 | 41.9 | 0.65 / 0.81 | 47.6 | 0.74 / 0.87 |
| +1.5 | 41.2 | 0.58 / 0.82 | 47.3 | 0.71 / 0.92 |
| +3.0 | 40.6 | 0.55 / 0.84 | 46.2 | 0.65 / 0.95 |

Exact match is flat between -3 and +1.5: this is a pure precision / recall knob for declining (at +1.5 the
model declines 92% of the out-of-scope test questions and 71% of its refusals are deserved). Left at 0 in the
release; the site can expose it.

Also in this entry
- `scripts/readme_results.py <release dir> [...]`: writes the "released model" block of README.md from
  model-card.json, RELEASE.json, bench.jsonl and parity_report.json (test in tests/test_model_card.py).
- README.md rewritten where it was stale: retrieval (fields, learned half), constrained decoding, evaluation
  modes and visitor metrics, training levers, pretraining provenance, export files, reproduce recipe, results
  ladder and negative results. `docs/research-log.md` is a copy of this notebook that git tracks (ml/runs is
  ignored by the repository's root .gitignore, which is outside ml/).
- Root .gitignore note: runs/final is not tracked either; the site gets the model through scripts/sync-brasa.mjs.

GPU queue from here (runs/queue3.txt): pre-12m (running since 06:22, 87k tokens/s, about 47 min) ->
`j4-12m-joint-pre` (exp14: the j2 recipe at brasa-12m) -> `rel-12m` (scripts/release.sh into runs/final-12m,
WITHOUT the final slice) -> j2-4m-joint-pre-s1 (seed 1) -> s1-4m-ls10 (label smoothing 0.1) -> s2-4m-12k (12,000
steps) -> j4-12m-joint (no pretraining, the 12m control of item 5) -> s3-4m-ob90, s4-4m-do0.
CPU: the provisional release of j2 into runs/final is at step 2/5 (evaluating the quantised weights).

What the next session must do if this one is cut: read the eval-last.json of the queued runs
(`uv run python runs/scratch/summ.py runs/<run>/eval-last.json`), log them, pick the release (12m if its
validation numbers beat j2), run the release WITH the final slice once
(`SPLITS=train,val,test,final scripts/release.sh runs/<run> runs/final`), keep the 4m release as
runs/final-4m, run `uv run python scripts/readme_results.py runs/final runs/final-4m`, finish README
"Limitations" and "Status".

---

## Entry 17 — 06:50 — provisional release: brasa-4m (j2) is in runs/final; end-to-end check of the engine

`DEVICE=cpu SPLITS=train,val,test scripts/release.sh runs/j2-4m-joint-pre runs/final` finished at 06:41
(log runs/logs/release-j2.log; 958 s for the float32 evaluation, 2,596 s for the two quantised formats, on the
CPU while the GPU trained). runs/final now holds: export/ (model.i8.bin 4,296,144 bytes, model.f16.bin,
tokenizer.json, retrieval.json 6,746,959 bytes with the fact vectors and the prior, manifest.json, testvec,
parity.json), model-card.json, MODEL_CARD.md, RELEASE.json, bench.jsonl, parity_report.json,
answers_parity.json, predictions.jsonl. The final slice was NOT read.

Held-out test, n=1278 (float32 unless stated):

| decoding | EM | F1 | right topic | refusal P / R | unsupported numbers |
| --- | ---: | ---: | ---: | --- | ---: |
| site default (retrieved facts + prior score 24) | 47.6 | 62.7 | 88.0 | 0.74 / 0.87 | 0.0 |
| site default, int8 weights | 47.4 | 62.6 | 87.9 | 0.74 / 0.87 | 0.0 |
| retrieved facts only | 43.8 | 59.8 | 87.0 | 0.75 / 0.86 | 0.0 |
| free, open-book | 42.6 | 59.2 | 86.1 | 0.76 / 0.86 | 4.2 |
| free, closed-book | 34.7 | 53.4 | 81.0 | 0.77 / 0.87 | 46.8 |
| first hit, no model | 39.3 | 55.5 | 88.6 | - / 0 | 0.0 |

- Quantisation: int8 answers identical to float32 for 17,418 of 17,696 (98.4%), float16 17,681; int8 logit
  drift 0.21. The int8 model loses 0.2 points.
- Node parity on the real export: PARITY OK; logits within 1.8e-5; greedy 8/8; constrained 24/24 (8 with a
  prior) on float16 and int8 with both kernels; fused search with the engine's own question vector 48/48.
  Engine speed with the int8 file: 305 tokens/s (JavaScript), 2,865 tokens/s (WASM SIMD).
- **End to end** (new: `scripts/compare_answers.py`, `web/engine/tools/ask.mjs`): 200 held-out test questions
  asked through the browser session in Node (WASM kernel, int8 file, default decoding) and through
  `brasa.sample.answer` in Python: 200 identical answers, 200 identical lists of retrieved facts. Now step 3
  of scripts/release.sh (answers_parity.json) and a pytest on the committed fixture.
- `uv run python scripts/readme_results.py runs/final` wrote the "released model" block of README.md.
- The analysis helpers of this notebook (runs/scratch/*.py) are copied to scripts/analysis/ (tracked by git,
  with a README); model-card.template.json documents the new card sections.
- Checks: `uv run ruff check .` clean, `uv run pytest` 157 passed, `node --test` 62 passed, tsc clean.

Site command for this provisional model (from the repository root):
`node scripts/sync-brasa.mjs ml/runs/final/export ml/runs/final/model-card.json`

GPU: pre-12m is at step 4,000 of 10,000 (validation perplexity 45.7 at step 1000); then j4-12m-joint-pre and
its automatic release into runs/final-12m.

---

## Entry 18 — 06:55 — while the 12m pretraining runs: mined hard negatives (exp20 queued), a gated final release

- `brasa.encoder` / `brasa.train --retrieval-mined-negatives N` (`encoder.mine_negatives`, `make_batch(..., mined)`):
  per contrastive question, N of the facts the BM25 index ranks highest for that very question without
  being gold are added to the documents of the batch, next to the in-batch negatives and the random
  same-topic negative. Hypothesis H-mined: the fused ranking fails among lexically close siblings, so
  negatives chosen by the lexical ranking teach the encoder exactly the distinctions the fusion needs:
  recall@1 above j2's 45.0 (test) / 41.1 (validation). Test in tests/test_encoder.py. Queued as `h1-4m-mined`
  (the j2 recipe + 2 mined negatives per question), after the seed-1 repeat of j2.
- `runs/q2.sh relfinal <run> <dir> <baseline run>`: runs scripts/release.sh with the FINAL slice only when
  <run> beats <baseline> on validation exact match with the site's decoding (eval-last.json of both); the
  decision is written to runs/logs/release-<run>.decision.log. The queue line `rel-12m` uses it with j2 as the
  baseline and writes runs/final-12m. Dry-run with a stub: j3 against j2 -> "splits: train,val,test,final";
  a run without the ranked mode -> no final slice.
- runs/j2-4m-joint-pre/eval-last.json is being recomputed on the CPU with the prior the run records
  (score 24); until then its open_book_ranked row is the rank-4 one (validation 42.6, test 46.9).
- Checks: ruff clean, pytest 158 passed, node 62 passed.

Queue order now: pre-12m (step 5400 of 10000, validation perplexity 25.5 at step 4000) -> j4-12m-joint-pre ->
rel-12m -> j2-4m-joint-pre-s1 -> h1-4m-mined -> s1-4m-ls10 -> s2-4m-12k -> j4-12m-joint -> s3-4m-ob90 -> s4-4m-do0.

---

## Entry 19 — 07:15 — are the gains more than question-sampling noise? (paired tests); smoke run of the pipeline

`scripts/analysis/paired.py A.jsonl:mode B.jsonl:mode --split val|test`: the same questions answered by two
systems; difference of exact match (B - A) in points, 95% bootstrap interval over questions (10,000 paired
resamples), discordant pairs and the exact sign test. Validation n=1250, test n=1278.

| comparison (A -> B) | val: B - A [95%] | sign test | test: B - A [95%] | sign test |
| --- | --- | ---: | --- | ---: |
| f3 free -> f3 restricted to the retrieved facts | +1.2 [-0.2, +2.6] | 0.14 | +1.2 [-0.2, +2.6] | 0.13 |
| f3 -> j1 (the network also ranks), retrieved | +5.1 [+2.5, +7.8] | 0.0002 | +3.1 [+0.3, +5.9] | 0.03 |
| j1 -> j2 (+ pretraining), retrieved | +4.8 [+1.9, +7.7] | 0.0015 | +6.2 [+3.4, +9.1] | < 0.0001 |
| j2 retrieved -> j2 ranked (prior score 24) | +2.5 [+1.0, +3.9] | 0.001 | +3.8 [+2.3, +5.2] | < 0.0001 |
| f3 free -> j2 ranked (the whole ladder) | +13.6 [+10.8, +16.6] | < 0.0001 | +14.2 [+11.3, +17.1] | < 0.0001 |
| j2 first hit (no model) -> j2 ranked | +5.2 [+3.0, +7.4] | < 0.0001 | +8.3 [+6.2, +10.5] | < 0.0001 |
| w2-tok -> w2-pre (pretraining without the contrastive loss), retrieved | -1.9 [-4.3, +0.3] | 0.12 | +2.0 [-0.3, +4.3] | 0.10 |
| j2 -> j3 (`--contexts-from`), ranked | +1.2 [-1.1, +3.4] | 0.34 | -0.9 [-3.0, +1.3] | 0.46 |

Conclusions
1. The three steps of the recipe each clear the question-sampling noise on BOTH splits: the shared network,
   pretraining under it, the ranking prior. The total is +14 points with an interval of about +-3.
2. Restricting decoding to the retrieved facts is worth +1.2 on both splits but is not significant alone
   (p about 0.13): its value is the guarantee (every statement is one of the sources shown), as said in entry 7.
3. Pretraining without the contrastive loss, and `--contexts-from`, change sign between the two splits: no effect.
4. The model with the prior beats "answer with the first hit" by 5 to 8 points overall: that is what the
   language model adds (declining, greeting, chaining), since on single-fact questions alone it does not win.
5. This is the noise of the question sample only. The run-to-run noise of training (1 to 2 points in two
   identical runs, entry 2) is not in these intervals; `j2-4m-joint-pre-s1` (queued) measures it for the best recipe.

Pipeline check: `DEVICE=cpu STEPS=200 RUN=runs/smoke-s5 scripts/smoke.sh` (the fictional rehearsal corpus, on
the CPU so as not to share the GPU): exit 0, ENGINE PARITY OK, float16 1216/1216 and int8 1208/1216 answers
identical to float32. The run directory was removed afterwards.

---

## Entry 20 — 07:22 — pre-12m finished; the 12m run (exp14) is training

- `runs/pre-12m`: brasa-12m, 10,000 steps of next-token prediction on the Wikipedia slice, 246M tokens seen
  (2.4 passes), 3,299 s (55 min), training loss 2.76, validation perplexity 20.4 (brasa-4m after 12,000 steps:
  25.0). Command: `uv run python -m brasa.pretrain --preset brasa-12m --corpus data/cache/wiki/prep-full2 --out runs/pre-12m --steps 10000 --resume`.
- `runs/j4-12m-joint-pre` started at 07:18 (`brasa.train --preset brasa-12m --steps 6000 --index fields
  --data-dir runs/data-full2 --retrieval-weight 1.0 --init-from runs/pre-12m`): 38.5k tokens/s, about 75 minutes
  with its validation passes. Hypothesis: above j2 on validation (39.4 retrieved / 41.9 ranked) by about the
  3 points brasa-12m had over brasa-4m in experiment 1, and a better ranking (j2: 41.1 / 70.8 recall@1 / @4 on
  validation). Then `rel-12m` (final slice only if it beats j2 on validation) and `rel-4m`.
- runs/j2-4m-joint-pre/eval-last.json now holds the open_book_ranked row with the recorded prior (score 24):
  validation 41.9, test 47.6 (CPU evaluation, 1,124 s).

---

## Entry 21 — 07:36 — the small variant is released with the final slice (started; CPU)

Decision: `runs/j2-4m-joint-pre` is the brasa-4m of this release (the small variant). Its release with the
FINAL slice, read here for the first time and once for this model, runs on the CPU while the 12m model trains:
`DEVICE=cpu SPLITS=train,val,test,final scripts/release.sh runs/j2-4m-joint-pre runs/final-4m`
(log runs/logs/release-j2-final.log). No decision of any kind will be taken from the final rows. When it
succeeds, `runs/logs/rel-4m.done` is created so that the queue does not repeat it. runs/final keeps the
provisional copy without the final rows until the 12m release is decided.
j4-12m-joint-pre: step 1000 of 6000 at 07:34 (validation EM 17.9, dense recall@1 34.2; j2 at the same step: 15.5 / 32).
