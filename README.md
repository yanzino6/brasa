# Brasa

> **Resumo (pt-BR).** Brasa é um pequeno modelo de linguagem no estilo GPT, escrito e treinado do zero em
> PyTorch: o transformer, o tokenizador BPE, o laço de treino, o buscador lexical e o motor de inferência
> que roda no navegador (JavaScript + WebAssembly SIMD) foram escritos à mão, sem pesos pré-treinados e
> sem bibliotecas de modelos. Ele responde, em português ou inglês, perguntas sobre uma pessoa a partir de
> uma pequena base de conhecimento curada, e é pequeno o bastante (4 a 12 milhões de parâmetros, 4 a 12 MB)
> para rodar inteiro no navegador do visitante, com as probabilidades, a atenção e os tokens à mostra.
> Não é um assistente geral: fora da base de conhecimento ele deve dizer que não sabe. Este repositório
> contém o código, os testes, um corpus de ensaio **fictício** e o protocolo de avaliação; os resultados
> do modelo real ainda não existem e estão marcados como pendentes abaixo.

Brasa ("ember" in Portuguese) is a small GPT-style language model written and trained from scratch, plus
everything around it that is usually borrowed from a library:

| Piece | Where | Written from scratch |
| --- | --- | --- |
| Decoder-only transformer | `src/brasa/model.py` | RMSNorm, rotary positions, SwiGLU, KV cache, sampling |
| Byte-level BPE tokenizer | `src/brasa/tokenizer.py` | training, encoding, token boundaries |
| Lexical retriever (BM25) | `src/brasa/retrieval.py` | index builder and search |
| Data builder and training loop | `src/brasa/data.py`, `train.py` | open-book / closed-book examples, AdamW loop, resume |
| Evaluation | `src/brasa/evaluate.py` | generation and seven metrics |
| Export | `src/brasa/export.py` | single-file binary format, int8 quantisation |
| Browser inference engine | `web/engine/`, `web/wasm/` | JavaScript engine, 1.5 kB WebAssembly SIMD kernel, Web Worker |

Runtime dependencies: `torch` and `numpy` for Python, nothing for JavaScript.

## What it is, and what it is not

Brasa answers questions about one person from a knowledge base of a few hundred short facts, in
Portuguese or English. A question is matched against the facts by a tiny lexical retriever, the best
facts are put in front of the question, and the model writes the answer token by token. All of it runs
in the visitor's browser: there is no server and no API call, and the page can show what the model is
doing (token probabilities, attention, residual stream) because the engine exposes it.

It is **not** a general assistant and it is not competitive with one. It has 4 to 12 million parameters
and a 384-token context. It does not reason, calculate, remember earlier turns or follow instructions.
It knows nothing outside its knowledge base and is trained to refuse instead of guessing; that refusal
is a learned behaviour and it fails sometimes (see [Rehearsal results](#rehearsal-results-fictional-corpus)).
The point of the project is the engineering: every layer between raw text and tokens appearing in a
browser is in this repository, small enough to read and tested against a reference.

## Status

| | |
| --- | --- |
| Code, tests, export, browser engine | done; `uv run pytest` (102 tests) and `node --test` (45 tests) pass |
| End-to-end rehearsal on a fictional corpus | done; `scripts/smoke.sh`, numbers below |
| Real knowledge base and training questions | **pending** (produced by a later stage of the project) |
| Released model (`brasa-12m`) and its metrics | **pending**; see the placeholder in [Results](#results-on-the-real-corpus) |

## Architecture

```
                         question typed by the visitor
                                      |
              +-----------------------+------------------------+
              |                                                |
   BM25 over word tokens and                          byte-level BPE
   character trigrams of the facts                    (vocabulary 4096)
              | top 4 facts                                    |
              +-----------------------+------------------------+
                                      v
   <|bos|><|ctx|> fact <|sep|> fact <|sep|> fact <|q|> question <|a|>         (prompt, at most 384 tokens)
                                      |
                                      v
   token embedding  (vocab x d_model, shared with the output head)
                                      |
        +-----------------------------+------------------------------+
        |  x = x + Attention(RMSNorm(x))                             |
        |        queries and keys rotated by position (RoPE),        |   x N layers
        |        causal mask, keys and values cached per layer       |
        |  x = x + SwiGLU(RMSNorm(x))                                |
        |        down( silu(gate(x)) * up(x) ),  hidden = 8/3 d      |
        +-----------------------------+------------------------------+
                                      |
                                   RMSNorm
                                      |
             logits = embedding matrix (transposed) . x      ->  next token  ->  ...  ->  <|eos|>
```

| Preset | Layers | Width | Heads | MLP hidden | Parameters (vocabulary 4096) | Role |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `brasa-4m` | 4 | 256 | 4 | 680 | 4,188,416 | smoke runs, fastest in the browser |
| `brasa-12m` | 6 | 384 | 6 | 1024 | 12,194,688 | the main model |
| `brasa-27m` | 8 | 512 | 8 | 1368 | 27,304,448 | ablation only |

Design choices, all in `src/brasa/model.py` (about 300 lines, nothing imported from `nn.Transformer`):

- **Pre-norm residual blocks with RMSNorm.** No mean subtraction and no bias; the mean of squares is
  computed in float32 even under float16 autocast.
- **Rotary position embeddings.** Queries and keys are rotated by an angle proportional to their
  position, so attention scores depend only on the distance between tokens. There is no position table:
  the parameter count does not depend on the context length.
- **SwiGLU feed-forward** with hidden size 8/3 of the width (GELU is available for ablations).
- **No biases anywhere, tied input and output embeddings.** The tied matrix is stored once.
- **KV cache.** `forward(idx, cache=...)` takes only the new tokens; generation costs one position per
  step. The test suite checks that cached and uncached generation are identical.
- **Inspection path.** `GPT.inspect()` returns the attention of every layer and head and the residual
  norms. It is the reference the browser visualisation is tested against.

## Tokenizer

A byte-level BPE trained from scratch (`src/brasa/tokenizer.py`, no dependencies).

- Text is normalised (Unicode NFC, whitespace runs collapsed to one space), split into chunks (optional
  leading space, then letters, digits or other symbols) and each chunk is merged pair by pair in the
  order learned in training. The base alphabet is the 256 byte values, so any string encodes and decodes
  back exactly: accents, emoji and unseen scripts fall back to shorter tokens.
- Vocabulary layout: `0..255` bytes, then the merges in training order, then seven control tokens
  (`<|pad|> <|bos|> <|eos|> <|ctx|> <|sep|> <|q|> <|a|>`). The requested size is 4096; training stops
  early when no pair occurs twice, so a small corpus gives a smaller vocabulary (3200 on the rehearsal
  corpus).
- **Control tokens cannot be forged.** `encode()` never produces them from text: a visitor who types
  `<|a|>` gets the seven characters, not the token. Only the prompt builder inserts control tokens.
- `tokenize()` returns byte offsets for every token, which the site uses to draw token boundaries.
- It is trained on the facts and on the *training* questions and answers only; held-out questions never
  influence the vocabulary (tested).

`web/engine/tokenizer.js` is a port; both implementations are compared on 144 fixture cases (accents,
emoji with ZWJ sequences, decomposed Unicode, exotic whitespace, control-token look-alikes).

## Data

A data directory follows this contract (`brasa.data.load_dataset` validates it and reports every
violation):

```
kb/facts.jsonl   {"id", "topic", "subtopic", "kind", "pt", "en", "importance", "caveat"}
kb/topics.json   [{"slug", "type", "title_pt", "title_en", "aliases": [...]}]
qa/*.jsonl       {"id", "lang": "pt"|"en", "q", "a", "facts": ["fact-id", ...],
                  "kind": "single"|"multi"|"chitchat"|"oos"|"meta",
                  "split": "train"|"val"|"test", "key_spans": ["..."]}
corpus.json      optional: {"name", "fictional", "description"}, copied into runs and model cards
```

The **test split holds questions whose phrasings never appear in training** (held-out paraphrases of
the same facts) and out-of-scope questions never seen. The loader refuses a data directory in which a
validation or test question repeats a training question, ignoring case, accents and final punctuation.

Each question/answer record becomes six training examples in two prompt formats:

```
open-book    <|bos|><|ctx|>fact<|sep|>fact<|sep|>fact<|q|>question<|a|>answer<|eos|>
closed-book  <|bos|><|q|>question<|a|>answer<|eos|>
```

- Variant 0 is the clean closed-book example, variant 1 the clean open-book example with exactly what
  the retriever returns for that question; the other four are randomised (70% open-book).
- **Open-book contexts come from the real retriever**, so the model trains on the distractors it will
  meet in the browser. Augmentations: the gold fact is forced into the context (35%), removed from it
  so that the answer has to come from the weights (15%), a random fact is swapped in (25%), the order
  is shuffled (50%).
- **Question noise** (half of the randomised variants): lower case, missing accents, missing question
  mark, one typo.
- **Out-of-scope questions** are answered with a short refusal in the language of the question; they
  also get retrieved facts in their context, so the model learns that a context does not make a
  question answerable.
- **The loss covers the answer span only** by default (`--prompt-loss-weight 0`); any weight between 0
  and 1 can be given to the prompt tokens.
- Batches are length-bucketed with a token budget (24,576 positions per step by default) and are a pure
  function of `(seed, step)`, which is what makes an interrupted run resume exactly.

## Retrieval

`src/brasa/retrieval.py` is a BM25 scorer over two kinds of features of every fact: word tokens (weight
1.0) and character trigrams (weight 0.5). Words carry the meaning; trigrams make the match tolerant to
typos, missing accents and inflection. Every fact is indexed once per language, together with the title
and aliases of its topic and the **training questions that point to it** (document expansion: facts are
statements, questions are questions, and they share few words). On the rehearsal corpus the expansion
lifts recall@4 on held-out paraphrases from 0.60 to 0.92.

The retriever is implemented twice, in Python (to build training examples and to evaluate) and in
JavaScript (`web/engine/retrieval.js`, for the browser). Parity is by construction: all logarithms and
divisions happen when the index is built, the index stores one finished weight per (feature, document),
and a search is only table lookups and float64 additions in a fixed order. Fixtures check that both
return the same facts, in the same order, with bit-identical scores (204 queries).

A learned retriever is a later stretch goal. The seam is the `Retriever` protocol (`search(query, k,
lang) -> hits`) in Python and `RetrieverLike` in the TypeScript declarations: the data builder, the
evaluation and the worker only depend on that method.

## Training

```sh
uv run python -m brasa.train --preset brasa-12m --data-dir data --out runs/brasa-12m --steps 30000
uv run python -m brasa.train --preset brasa-12m --data-dir data --out runs/brasa-12m --steps 30000 --resume
```

| Setting | Default |
| --- | --- |
| Optimiser | AdamW, betas (0.9, 0.95), weight decay 0.1 on the matrices only |
| Learning rate | 1e-3 (`brasa-4m`), 6e-4 (`brasa-12m`), 4e-4 (`brasa-27m`); 200 warm-up steps, cosine to 10% |
| Batch | 24,576 positions per step, length-bucketed |
| Regularisation | dropout 0.1, gradient clipping at 1.0 |
| Device | auto: `mps`, then `cuda`, then `cpu` |
| Speed switches | `--compile auto` (`torch.compile` on mps and cuda), `--amp auto` (float16 autocast on mps and cuda) |
| Checkpoints | `ckpt_last.pt` every 500 steps, at the end and on Ctrl-C; `ckpt_best.pt` at the lowest validation loss |
| Log | `train_log.jsonl`: step, loss, learning rate, tokens/s; validation loss (open-book and closed-book) |

`--resume` continues from `ckpt_last.pt`, including the optimiser state and the position in the batch
stream. On the CPU with dropout 0 a resumed run reproduces an uninterrupted one bit for bit (tested). On
MPS the kernels are not deterministic, so two runs with the same seed differ slightly: in two identical
smoke runs the final validation loss was 0.268 and 0.304.

## Evaluation

`src/brasa/evaluate.py` generates an answer for every question (greedy decoding) and scores it. Every
question is answered twice: **open-book**, with retrieved facts in the prompt as in the browser, and
**closed-book**, where the answer has to come from the weights.

| Metric | Definition |
| --- | --- |
| Exact match | prediction equals the gold answer after lower-casing and dropping punctuation and spacing |
| Token F1 | F1 over word tokens against the gold answer |
| Key-span recall | share of the required names and numbers (`key_spans`) present in the answer |
| Refusal precision / recall | an answer is a refusal when it matches one of the refusal texts of the data (token F1 at least 0.6); recall is measured on the out-of-scope questions |
| Language consistency | share of answers written in the language of the question |
| Unsupported-number rate | share of the numbers in the answers that appear neither in the gold answer, nor in the retrieved context, nor in the question |
| Retrieval hit | open-book only: share of questions whose gold facts were all retrieved, and exact match separately for hits and misses |

Results are reported for a sample of **seen** training questions and for the **held-out** validation and
test splits; overall, per language, per question kind, per topic and (open-book) per retrieval outcome.
They are written to `<run>/eval.json`, the individual answers to `<run>/predictions.jsonl`, and a summary
into the model card together with the configuration, the parameter count, the training settings, the
data statistics and the file sizes. `model-card.template.json` documents that file;
`scripts/render_model_card.py` renders it as the Hugging Face model card.

The exported weights are evaluated too: `brasa.export` loads each file back, de-quantises it and runs
the same protocol, so the int8 model the site ships has its own measured scores instead of an assumption.

## Export format and browser engine

```sh
uv run python -m brasa.export --run runs/brasa-12m       # writes runs/brasa-12m/export/
```

| File | What |
| --- | --- |
| `model.i8.bin` | weights as int8 with one scale per output row (what the site loads) |
| `model.f16.bin` | weights as float16 |
| `tokenizer.json`, `retrieval.json` | BPE merges; BM25 index |
| `testvec.<fmt>.json` | logits, greedy continuations, attention, residual norms and sampling distributions computed by PyTorch from the exported file |
| `parity.json` | tokenizer, retrieval and prompt fixtures computed by the Python reference |
| `manifest.json` | configuration, prompt format, sizes (raw, gzip, brotli) and SHA-256 of every file |

The model file is a single "TGPT" container, little endian:

```
"TGPT" | u32 version = 1 | u32 header_bytes | header JSON (space padded to 4 bytes) | tensor data
header = {"format", "name", "config", "prompt", "tokenizer",
          "tensors": [{"name", "shape", "dtype": "f32"|"f16"|"i8", "offset", "scalesOffset"?}]}
```

Matrices are stored in the chosen format; the RMSNorm gains stay float32; the tied output head is not
stored. An int8 matrix `W` is `q = round(W / s)` with one scale per row, `s[row] = max|W[row]| / 127`,
and the engine rebuilds `W = q * s[row]` at load time. Rows of the same matrix differ in magnitude by
orders of magnitude, which is why one scale per matrix is not enough.

**The engine** (`web/engine/engine.js`) reproduces `model.py` operation by operation on one
`Float32Array` heap that holds the weights, the KV cache and the scratch vectors. The two heavy loops
(matrix times vector, weighted sum of cached value rows) exist twice: in plain JavaScript and in a
1.5 kB WebAssembly module with hand-written SIMD intrinsics (`web/wasm/kernel.c`) that works on the same
memory. If WebAssembly SIMD is not available the JavaScript kernel is used. Nothing is allocated while
generating.

```js
import { BrasaClient } from './engine/index.js';

const worker = new Worker(new URL('./engine/worker.js', import.meta.url), { type: 'module' });
const brasa = new BrasaClient(worker);
await brasa.load({ model: '/brasa/model.i8.bin', wasm: '/brasa/kernel.wasm', retrieval: '/brasa/retrieval.json' });

const run = brasa.ask('Onde ele mora?', { inspect: { top: 5, attention: true, norms: true } }, {
  onContext: ({ hits, used }) => showFacts(hits.slice(0, used)),     // what the retriever found
  onToken: ({ delta, top, attention, norms }) => render(delta),      // one message per generated token
});
const { text, stats } = await run.done;                               // run.abort() stops it early
```

What a visualisation can ask for, none of which costs anything when it is not requested:

- the `top` k next-token probabilities of every step, and the probability of the chosen token;
- the attention weights of every layer and head for the current position;
- the residual-stream norm after the embedding and after every block;
- `tokenize(text)`: token ids with byte boundaries; the exact prompt tokens of every question;
- streaming generation with temperature, top-k, top-p, repetition penalty and a seed;
- `generate(tokens)`: continue any token sequence, reusing the cached prefix, which is how a page steps
  one token at a time or explores "what if this token instead".

TypeScript declarations cover the engine, the tokenizer, the retriever, the prompt builder and the
worker protocol (`web/engine/*.d.ts`). ONNX is not used in the browser; `brasa.export_onnx` writes two
ONNX graphs (plain and with a KV cache) as an interoperability artefact and verifies them with
onnxruntime (`uv sync --extra onnx`).

**How parity is established.** The test vectors are computed by PyTorch from the exported file itself.
`web/engine/tools/parity.mjs` loads the same file in Node with both kernels and compares the logits, the
greedy continuations, the attention of every layer and head, the residual norms and the sampling
distributions; it also replays the tokenizer, retriever and prompt-builder fixtures. A test corrupts one
weight row and checks that the comparison fails.

## Benchmarks

Measured on an Apple M3 Max (48 GB) while other jobs were running; context 256, vocabulary 4096, batch
64. These are speed measurements of the code, not results of a trained model.

Training throughput on MPS (tokens per second):

| Model | Parameters | `torch.compile` + float16 autocast | 30,000 steps | Plain float32 |
| --- | ---: | ---: | ---: | ---: |
| `brasa-4m` | 4.2M | 258,000 | about 32 min | 152,000 |
| `brasa-12m` | 12.2M | 102,000 | about 80 min | 64,000 |
| `brasa-27m` | 27.3M | 49,000 | about 168 min | 31,000 |

The CPU is 4 to 7 times slower.

Inference with the hand-written engine, int8-per-row weights, Node 24 (the same V8 as Chrome), tokens
per second while decoding with the KV cache:

| Model | Plain JavaScript | WebAssembly SIMD | File (int8) | File (float16) |
| --- | ---: | ---: | ---: | ---: |
| `brasa-4m` | 300 | 2,940 | 4.3 MB | 8.6 MB |
| `brasa-12m` | 105 | 1,080 | 12.3 MB | 24.6 MB |
| `brasa-27m` | 47 | 500 | 27.5 MB | 55 MB |

The engine's logits match PyTorch to about 4e-6. Reproduce the inference numbers with
`node web/engine/tools/bench.mjs <model.bin> --wasm web/wasm/kernel.wasm`.

## Rehearsal results (fictional corpus)

Until the real knowledge base exists, the pipeline is rehearsed on an invented persona
(`data/rehearsal/`, 30 facts, clearly marked fictional). **These numbers describe the code and the
method, not the released model.** They come from one run of `scripts/smoke.sh` on the machine above:
`brasa-4m` (3,959,040 parameters with the 3200-token vocabulary of this corpus), 1200 steps, 138 seconds
of training at 214,000 tokens/s on MPS, interrupted at step 600 and resumed.

Training loss on the answer tokens: 4.84 at step 100, 0.028 at step 300, 0.0056 at step 600, 0.0002 at
step 1200. Validation loss (held-out paraphrases): 0.18 at step 200, its minimum, then rising to 0.30.

| Questions | Mode | n | Exact match | Token F1 | Key-span recall | Refusal precision | Refusal recall | Language | Unsupported numbers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seen (training sample) | open-book | 300 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| Seen (training sample) | closed-book | 300 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| Held-out test | open-book | 216 | 45.8% | 55.4% | 59.1% | 84.0% | 53.8% | 98.6% | 8.3% |
| Held-out test | closed-book | 216 | 42.6% | 52.3% | 57.3% | 82.3% | 53.8% | 95.8% | 46.7% |

- Quantisation is harmless here: the float16 weights give 608 of 608 identical answers on the held-out
  splits and the int8 weights 606 of 608, with the same scores to the first decimal. Largest logit drift:
  0.002 (float16) and 0.041 (int8).
- Engine parity in Node: largest logit difference 5.5e-6 (float16) and 1.1e-5 (int8) on logits up to 18.9,
  greedy continuations identical for 8 of 8 prompts with both kernels; tokenizer 144 of 144, retrieval
  204 of 204 queries with bit-identical scores, prompt builder 26 of 26.
- Engine speed for this model: 313 tokens/s in plain JavaScript and 3,102 tokens/s with the WebAssembly
  kernel.
- Files: `model.i8.bin` 4,054,804 bytes (3,767,279 with brotli), `model.f16.bin` 7,956,044 bytes,
  `retrieval.json` 199,537 bytes (55,122 with brotli), `tokenizer.json` 30,570 bytes (9,028 with brotli).

**What the rehearsal shows.** The model memorises every training question and answers a little under
half of the held-out phrasings. The errors are not garbled text: of the 74 wrong answers to in-scope
test questions (open-book), 64 are a fluent, verbatim answer *to a different question*, 8 are refusals
and 2 are something else. Of the 78 out-of-scope test questions, 42 are refused and 36 get a memorised
in-scope answer. The retriever returned the gold fact for 91.5% of the in-scope test questions, but
exact match was only 51.3% in those cases: with 30 facts and one fixed answer per fact, the model can
solve the training set by mapping question wording to a memorised sentence, and it learns to do that
instead of reading the context. Training longer does not help (the validation loss is lowest at step
200), and the open-book advantage is small (45.8% against 42.6%), except that numbers are much better
supported.

How the number of training paraphrases changes this (`scripts/paraphrase_ablation.py`: same model, 600
steps, same held-out test questions, only the number of training questions per fact changes):

| Hand-written training questions per fact and language | With the wrapped copy | Retriever recall@1 / recall@4 | Open-book EM | Open-book EM when the gold fact was retrieved | Closed-book EM | Refusal precision | Refusal recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 46.9% / 68.5% | 8.7% | 11.2% | 8.0% | 58.2% | 82.0% |
| 2 | 4 | 48.5% / 73.1% | 14.5% | 17.9% | 10.9% | 66.3% | 83.3% |
| 4 | 8 | 53.8% / 80.8% | 20.3% | 23.8% | 20.3% | 72.2% | 66.7% |
| 8 | 16 | 61.5% / 86.2% | 44.2% | 50.9% | 47.1% | 93.6% | 56.4% |
| 11 | 22 | 63.8% / 91.5% | 46.4% | 52.1% | 42.0% | 84.9% | 57.7% |

Exact match is over the 138 in-scope held-out test questions; every model reaches 100% on its own
training questions. One run per row, so the standard error of an exact-match figure is about 4 points:
the difference between 8 and 11 is noise. The 328 out-of-scope training records are the same in every
row, which is why the smallest corpora refuse the most.

Three things follow for the real corpus:

1. **Below about eight distinct questions per fact and language the model is useless on new
   phrasings** (20% or less). The jump comes between four and eight.
2. **More paraphrases of the same kind stop paying off** at around 45 to 50%. The held-out questions
   that still fail approach a fact from an angle no training question takes ("Has he passed any cloud
   exams?" for his certification, "Which companies has he worked for?" for his previous job). Coverage has
   to grow in *kinds* of question (yes/no, indirect, keyword-only, by synonym, by consequence), not in
   the number of rewordings.
3. **Refusal generalises across fillers, not across frames.** On the test split the main run refuses
   76% of the out-of-scope questions that reuse a training frame with a new filler (32 of 42), 17% of
   the questions with a frame never seen (4 of 24) and 50% of the questions about personal details that
   are not in the knowledge base (6 of 12). Out-of-scope training data needs many different frames.

## Results on the real corpus

> **PLACEHOLDER: these results do not exist yet.** The real knowledge base and its training questions
> are produced by a later stage of the project. When `brasa-12m` has been trained on them, this section
> will hold the same table as above (seen, held-out test, open-book and closed-book, float32 and int8),
> the per-language and per-topic breakdown, the file sizes and the measured browser speed, taken from
> `model-card.json`.

| Questions | Mode | n | Exact match | Token F1 | Key-span recall | Refusal precision | Refusal recall | Language | Unsupported numbers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seen | open-book | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Held-out test | open-book | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Held-out test | closed-book | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Held-out test, int8 weights | open-book | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and the dependencies into
`.venv`), and Node 20 or newer for the engine tests.

```sh
uv sync                        # add --extra onnx for the ONNX export and its test
uv run ruff check
uv run pytest                  # 102 tests; the ones that need Node or ONNX are skipped when missing
(cd web && npm test)           # 45 engine tests on the committed fixture, no dependencies to install
scripts/smoke.sh               # the whole pipeline on the fictional corpus, about 3.5 minutes on an M3 Max
```

The release recipe, once `data/` holds the real corpus:

```sh
uv run python -m brasa.data --data-dir data                                   # validate the contract
uv run python -m brasa.train --preset brasa-12m --data-dir data --out runs/brasa-12m --steps 30000
uv run python -m brasa.evaluate --run runs/brasa-12m --model-card model-card.json
uv run python -m brasa.export --run runs/brasa-12m --model-card model-card.json
node web/engine/tools/parity.mjs runs/brasa-12m/export --wasm web/wasm/kernel.wasm
uv run python -m brasa.sample --export runs/brasa-12m/export --format i8      # chat in the terminal
uv run python scripts/render_model_card.py model-card.json --out MODEL_CARD.md
```

Every module has `--help`. Other entry points: `brasa.model` (parameter counts, cache self-test),
`brasa.tokenizer` (train or inspect a tokenizer), `brasa.retrieval` (query the index, recall@k per
split), `brasa.export_onnx`, `scripts/make_rehearsal.py`, `scripts/make_fixtures.py`,
`scripts/paraphrase_ablation.py`, `web/wasm/build.sh` (rebuilds `kernel.wasm` with `zig cc` from a PyPI
wheel; the committed binary is reproduced byte for byte).

## Repository layout

```
src/brasa/        config, model, tokenizer, retrieval, data, train, evaluate, export, export_onnx, sample, fixtures
tests/            pytest suite; tests/fixtures/tiny is a tiny random model exported the normal way
web/engine/       engine.js, tokenizer.js, retrieval.js, prompt.js, session.js, worker.js, client.js, *.d.ts
web/engine/test/  Node tests;  web/engine/tools/  parity.mjs, bench.mjs
web/wasm/         kernel.c, build.sh, kernel.wasm
scripts/          smoke.sh, smoke_summary.py, make_rehearsal.py, paraphrase_ablation.py, render_model_card.py
data/rehearsal/   the fictional corpus in the contract format
runs/             training runs and exports (ignored by git)
```

## What the tests cover

- Model: shapes and parameter counts of every preset; KV-cache forward equals the full-context forward;
  cached generation equals uncached generation; causality; the RoPE relative-position property; loss
  masking and weighting; sampling filters.
- Tokenizer: round trips on Portuguese accents, emoji and mixed case; normalisation; control tokens are
  never produced from text and never split; token boundaries; deterministic training.
- Retrieval: ranking behaviour; only training questions are indexed; Python and JavaScript return the
  same ranking with identical scores.
- Data: the contract and its violations; held-out phrasings never reach training or the tokenizer; the
  loss mask covers exactly the answer span; batches are a pure function of seed and step.
- Training: schedule, logging, checkpoints, exact resume.
- Evaluation: every metric on hand-made examples.
- Export: quantisation error bounds, container layout and round trip, a fresh export run through the
  Node engine with both kernels, a corrupted weight is detected, the committed fixture is current, the
  ONNX graphs reproduce PyTorch.

## Limitations

- **Generalisation to new phrasings is the weak point**, as the rehearsal shows. The model is a
  memoriser with some tolerance to surface variation; coverage has to come from the training questions.
- **Wrong answers are fluent.** A miss is usually a correct-sounding sentence about something else, not
  an obvious failure. The site should show the retrieved facts next to the answer so that a visitor can
  check it.
- **Refusal is learned, not enforced.** In the rehearsal 46% of the unseen out-of-scope questions got
  an in-scope answer.
- **The retriever is lexical.** A question that shares no words or trigrams with a fact, its topic
  aliases or its training questions retrieves nothing useful.
- **Language detection in the evaluation is a word-list heuristic**, and the unsupported-number metric
  only sees digits ("nine" is not checked).
- **One question at a time.** There is no dialogue state; a follow-up such as "and before that?" is
  answered as if it were the first message.
- **Reproducibility is exact on the CPU only.** MPS runs with the same seed differ slightly.
- The evaluation measures similarity to one reference answer per question. A correct answer worded
  differently scores below 1.

## License

MIT, see [LICENSE](LICENSE). The rehearsal corpus is fictional; any resemblance to real people or
companies is accidental.
