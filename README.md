# Brasa

> **Resumo (pt-BR).** Brasa é um pequeno modelo de linguagem no estilo GPT, escrito e treinado do zero em
> PyTorch: o transformer, o tokenizador BPE, o laço de treino, o buscador (lexical e aprendido), a
> decodificação restrita à base de conhecimento e o motor de inferência que roda no navegador (JavaScript +
> WebAssembly SIMD) foram escritos à mão, sem pesos pré-treinados e sem bibliotecas de modelos. Ele
> responde, em português ou inglês, perguntas sobre uma pessoa a partir de uma base de conhecimento curada
> de 525 fatos, e é pequeno o bastante (4 a 12 milhões de parâmetros, 4 a 12 MB) para rodar inteiro no
> navegador do visitante, com as probabilidades, a atenção e os tokens à mostra. Não é um assistente
> geral: fora da base de conhecimento ele deve dizer que não sabe. Este repositório contém o código, os
> testes, um corpus de ensaio **fictício**, o protocolo de avaliação e os resultados medidos no corpus
> real, com o caderno de laboratório de todos os experimentos (`docs/research-log.md`). Os números são
> modestos e estão ditos como são: em perguntas escritas de propósito com palavras diferentes das do
> treino, o modelo acerta exatamente o fato pedido em quase metade das vezes e fala do assunto certo em
> quase 9 de cada 10 (tabela em [Results](#results-on-the-real-corpus)).

Brasa ("ember" in Portuguese) is a small GPT-style language model written and trained from scratch, plus
everything around it that is usually borrowed from a library:

| Piece | Where | Written from scratch |
| --- | --- | --- |
| Decoder-only transformer | `src/brasa/model.py` | RMSNorm, rotary positions, SwiGLU, KV cache, sampling |
| Byte-level BPE tokenizer | `src/brasa/tokenizer.py` | training, encoding, token boundaries |
| Lexical retriever (BM25) | `src/brasa/retrieval.py` | index builder and search, three weighted fields |
| Learned retriever | `src/brasa/encoder.py`, `train.py --retrieval-weight` | the same network read as an encoder, contrastive loss, int8 fact vectors |
| Constrained decoding | `src/brasa/constrain.py` | token trie of the allowed answers, greedy and beam search, the retriever as a prior |
| Data builder and training loop | `src/brasa/data.py`, `train.py` | open-book / closed-book examples, AdamW loop, resume |
| Language-model pretraining | `src/brasa/pretrain.py`, `scripts/fetch_wiki.py` | next-token prediction on a pinned slice of Wikipedia (from random weights) |
| Evaluation | `src/brasa/evaluate.py` | seven decoding modes, strict and visitor metrics |
| Export | `src/brasa/export.py` | single-file binary format, int8 quantisation |
| Browser inference engine | `web/engine/`, `web/wasm/` | JavaScript engine, 1.5 kB WebAssembly SIMD kernel, Web Worker |

Runtime dependencies: `torch` and `numpy` for Python, nothing for JavaScript.

## What it is, and what it is not

Brasa answers questions about one person from a knowledge base of a few hundred short facts, in
Portuguese or English. A question is matched against the facts by a small retriever (BM25 plus the
model's own ranking), the best facts are put in front of the question, and the model answers token by
token, by default choosing among the curated sentences it was shown instead of writing freely. All of
it runs in the visitor's browser: there is no server and no API call, and the page can show what the
model is doing (token probabilities, attention, residual stream) because the engine exposes it.

It is **not** a general assistant and it is not competitive with one. It has 4 to 12 million parameters
and a 384-token context. It does not reason, calculate, remember earlier turns or follow instructions.
It knows nothing outside its knowledge base and is trained to refuse instead of guessing; that refusal
is a learned behaviour and it fails sometimes (see [Results](#results-on-the-real-corpus)).
The point of the project is the engineering: every layer between raw text and tokens appearing in a
browser is in this repository, small enough to read and tested against a reference.

## Status

| | |
| --- | --- |
| Code, tests, export, browser engine | done; `uv run pytest` (158 tests) and `node --test` (62 tests) pass |
| End-to-end rehearsal on a fictional corpus | done; `scripts/smoke.sh`, numbers below |
| Real knowledge base and questions | done: 525 facts, 29,974 question records (`data/yan`, frozen copy `runs/data-full2`) |
| Experiments on the real corpus | lab notebook in `docs/research-log.md`; summary in [Results](#results-on-the-real-corpus) |
| Release in `runs/final` | <!-- STATUS-RELEASE -->provisional: `brasa-4m` (`runs/j2-4m-joint-pre`); the `brasa-12m` run with the same recipe is training |

## Architecture

```
                         question typed by the visitor
                                      |
              +-----------------------+------------------------+
              |                                                |
   BM25 over word tokens and                          byte-level BPE
   character trigrams of the facts                    (vocabulary 4096)
   + cosine between the question and the                       |
     facts as THIS model embeds them                           |
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
                                      |
        answer trie: only tokens that continue one of the retrieved facts or a fixed answer are
        allowed, and the retriever's ranking is added to their log-probabilities as a prior
```

The network is used twice per question: once as an encoder (mean of the final hidden states of
`<|bos|><|q|> question <|a|>`, to rank the facts) and once as the language model that answers.

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
                  "split": "train"|"val"|"test"|"final", "key_spans": ["..."]}
corpus.json      optional: {"name", "fictional", "description"}, copied into runs and model cards
```

The **validation, test and final splits hold questions whose phrasings never appear in training**
(held-out paraphrases of the same facts) and out-of-scope questions never seen. The loader refuses a data
directory in which a held-out question repeats a training question, ignoring case, accents and final
punctuation. "final" is optional: a slice that is evaluated once, for the released model, and never used
to choose between experiments (`scripts/build_corpus.py` puts the 14th question of every fact there).

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

**The lexical half.** `src/brasa/retrieval.py` is a BM25 scorer over two kinds of features of every
fact: word tokens and character trigrams. Words carry the meaning; trigrams make the match tolerant to
typos, missing accents and inflection. Every fact is indexed once per language, together with the title
and aliases of its topic and the **training questions that point to it** (document expansion: facts are
statements, questions are questions, and they share few words). Two layouts of the same file format:
`flat` concatenates the three texts into one document; `fields` (`--index fields`, the one the released
model uses) scores the fact, the topic names and the questions as three BM25 fields with their own length
normalisation and document frequencies and adds them with weights 1 / 1 / 2, so that a 40-word fact is
not drowned in 300 words of questions. On the real corpus that is worth 11 points of recall@4.

**The learned half.** Lexical matching stops at about 30% recall@1 on held-out phrasings (the topic is
right 84% of the time; the miss is among sibling facts). The same transformer is therefore also trained
as an encoder: `brasa.train --retrieval-weight 1` adds, at every step, a contrastive loss on its pooled
final hidden states (a batch of training questions, their gold facts, one same-topic fact each as a hard
negative; InfoNCE, temperature 0.05). One network, one download: the fact vectors are precomputed at
export time (int8, one scale, inside `retrieval.json`) and a question costs one extra forward pass of a
few dozen tokens. The two rankings are added, `bm25 / (best bm25 of the query) + weight * cosine`, with
the weight chosen on validation questions. `brasa.encoder` trains the same thing as a stand-alone dual
encoder, which is how the idea was first measured. The lexical path remains the fallback (`dense: false`
in the browser, `--retriever lexical` in the evaluation).

The retriever is implemented twice, in Python (to build training examples and to evaluate) and in
JavaScript (`web/engine/retrieval.js`, for the browser). Parity is by construction: all logarithms and
divisions happen when the index is built, the index stores one finished weight per (feature, document),
and a search is only table lookups and float64 additions in a fixed order. The question vector is
rounded to int16 before the dot products, so the fused scores are integer sums too. Fixtures check that
both return the same facts, in the same order, with bit-identical scores (204 lexical queries, 48 fused
searches given the same question vector).

The seam for other retrievers is the `Retriever` protocol (`search(query, k, lang) -> hits`) in Python
and `RetrieverLike` in the TypeScript declarations.

## Constrained decoding

Free generation lets a model of a few million parameters write a sentence nobody curated. Here the
answer is, by default, restricted with a **token trie of the allowed sentences** (`src/brasa/constrain.py`,
`AnswerTrie` in `web/engine/retrieval.js`): at every step only the tokens that keep the output a prefix
of an allowed answer may be produced. Allowed answers are a fact, two or three facts of the same
language in a row (the format of multi-fact answers), or one of the fixed answers (greetings, the
refusal) alone. Positions with a single allowed token are not decisions: they cost no forward pass in
Python and no output head in the browser, so constrained decoding is cheaper than free decoding.

| Mode (`brasa.evaluate --modes`, `ask({ constrain })` in the browser) | The answer may be |
| --- | --- |
| `open_book`, `closed_book` (`constrain: false`) | anything: the model's free generation |
| `open_book_constrained`, `closed_book_constrained` (`'kb'`) | any curated sentence of the knowledge base |
| `open_book_retrieved` (`'retrieved'`, `prior: false`) | only the facts **in the prompt**, or a fixed answer |
| `open_book_ranked` (`'retrieved'`, the browser's default) | the same, with the retriever's ranking as a prior |
| `first_hit` | no model: the retriever's first fact (the baseline; it can never decline) |

**The retriever as a prior.** On the real corpus the reader, left alone, is worse at choosing among the
four retrieved facts than the retriever's own order. `open_book_ranked` adds the retriever's opinion to
the model's log-probabilities: every retrieved fact gets a prior of `-weight * (1 - score / best score)`
nats (or `-weight` per rank), and a token is worth the best prior among the sentences it still leads
to. It needs no extra forward pass. The prior applies to the first sentence of an answer only, so
whether to go on with a second fact stays the model's decision; the weight is chosen on validation
questions (`brasa.evaluate --tune-prior`) and shipped inside `retrieval.json`. In the browser it is one
knob: `prior: 0` lets the model decide alone, a large value turns the answer into the first hit whenever
the model answers with a fact.

Beam search exists in Python (`--beam`) as an analysis tool; it was worth about one point and is not
ported. Free decoding stays available everywhere: the site shows both.

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
| Checkpoints | `ckpt_last.pt` every 500 steps, at the end and on Ctrl-C; `ckpt_best.pt` at the highest validation exact match (`--select em`; the validation loss bottoms out long before the answers stop improving) |
| Log | `train_log.jsonl`: step, loss, learning rate, tokens/s; validation loss and exact match (open-book and closed-book), contrastive loss and dense recall when the network also ranks |
| Loss scaling | float16 autocast always runs with a gradient scaler: without it small gradients underflow and a plain-text run sits at chance (research log, entry 9) |

Optional levers, all measured on the real corpus (see [Results](#results-on-the-real-corpus)):
`--index fields`, `--retrieval-weight` (the network also ranks the facts), `--init-from <pretrain run>`
(start from a language model pretrained by `brasa.pretrain`), `--contexts-from <run>` (training contexts
ranked by a fused retriever instead of BM25 alone), `--context-folds`, `--counterfactual-prob`,
`--word-dropout-prob`, `--label-smoothing`, `--importance-power`, `--oracle-prob`, `--drop-gold-prob`,
`--shuffle-prob`, `--random-distractor-prob`.

**Pretraining, still from scratch.** `scripts/fetch_wiki.py` downloads one parquet shard each of the
Portuguese and the English Wikipedia (Hugging Face dataset `wikimedia/wikipedia`, dump 2023-11-01,
revision `b04c8d1c`, licence CC BY-SA 3.0 and GFDL), keeps paragraphs of at least 120 characters and
writes 160 MB of text per language (22,631 and 32,049 articles) with a manifest of byte counts and
SHA-256 under `data/cache/wiki/` (not in git). `brasa.pretrain` trains the tokenizer on the first 8 MB of
each file plus the knowledge base and the training questions, encodes the text once (104 million tokens)
and trains the same architecture on next-token prediction. The weights start random; nobody else's
weights or tokenizer are involved.

`--resume` continues from `ckpt_last.pt`, including the optimiser state and the position in the batch
stream. On the CPU with dropout 0 a resumed run reproduces an uninterrupted one bit for bit (tested). On
MPS the kernels are not deterministic, so two runs with the same seed differ slightly: in two identical
smoke runs the final validation loss was 0.268 and 0.304.

## Evaluation

`src/brasa/evaluate.py` generates an answer for every question (greedy decoding) and scores it. Every
question is answered **open-book**, with retrieved facts in the prompt as in the browser, and
**closed-book**, where the answer has to come from the weights, in the decoding modes listed under
[Constrained decoding](#constrained-decoding).

| Metric | Definition |
| --- | --- |
| Exact match | prediction equals the gold answer after lower-casing and dropping punctuation and spacing |
| Token F1 | F1 over word tokens against the gold answer |
| Key-span recall | share of the required names and numbers (`key_spans`) present in the answer |
| Refusal precision / recall | an answer is a refusal when it matches one of the refusal texts of the data (token F1 at least 0.6); recall is measured on the out-of-scope questions |
| Language consistency | share of answers written in the language of the question |
| Unsupported-number rate | share of the numbers in the answers that appear neither in the gold answer, nor in the retrieved context, nor in the question |
| Retrieval hit | open-book only: share of questions whose gold facts were all retrieved, and exact match separately for hits and misses |
| Curated-sentence rate (`verbatim_rate`) | share of answers made only of curated sentences: one fact, two or three facts in a row, or a fixed answer. One minus this is what the model worded itself |
| Right-topic rate | questions with gold facts: the answer is one to three curated facts, all of a topic of the gold facts (exact matches included) |
| Gold in top 1 / 2 / 4 | open-book only, the retriever alone: all gold facts among its first k hits |

Exact match is strict: with dozens of sibling facts per project, an adjacent fact of the right topic is
often a fine answer for a visitor, and an invented sentence never is. The last three metrics describe that
experience next to the strict ones.

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
| `tokenizer.json` | BPE merges and control tokens |
| `retrieval.json` | the knowledge base as the browser sees it: BM25 index, fixed answers, one int8 vector per fact (`dense`, when the model also ranks) and the decoding prior chosen on validation (`answers.prior`) |
| `testvec.<fmt>.json` | logits, greedy continuations (free, constrained, constrained with a prior), pooled text vectors, attention, residual norms and sampling distributions computed by PyTorch from the exported file |
| `parity.json` | tokenizer, retrieval (lexical and fused), answer-trie and prompt fixtures computed by the Python reference |
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

brasa.ask('Onde ele mora?', { constrain: false });   // the model's free generation
brasa.ask('Onde ele mora?', { prior: false });       // retrieved facts only, the model alone chooses
brasa.ask('Onde ele mora?', { prior: 40 });          // lean on the retriever's ranking
brasa.ask('Onde ele mora?', { dense: false });       // rank the facts with BM25 alone
```

What a visualisation can ask for, none of which costs anything when it is not requested:

- the `top` k next-token probabilities of every step, and the probability of the chosen token;
- the attention weights of every layer and head for the current position;
- the residual-stream norm after the embedding and after every block;
- for a constrained answer, how many tokens were allowed at every step (`choices`: 1 means the token was
  forced), which sentences were allowed (`context.constraint`) and how much the ranking weighed
  (`context.prior`);
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
greedy continuations (free, constrained to the knowledge base, to four facts, and to four facts with a
prior), the pooled text vectors, the attention of every layer and head, the residual norms and the
sampling distributions; it also replays the tokenizer, retriever (lexical and fused), answer-trie and
prompt-builder fixtures. A test corrupts one weight row and checks that the comparison fails.

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
- Run-to-run noise: training on MPS is not bit-reproducible, and a second run of the same command on the
  same code gave 47.7% (open-book) and 46.8% (closed-book) exact match on the held-out test split,
  refusal precision 83.9%, refusal recall 66.7% and 605 of 608 int8 answers identical. Read differences
  of a few points as noise; refusal recall is the least stable figure (53.8% against 66.7%).

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

Every number in this section was measured in this repository on the real corpus; the lab notebook with
the hypothesis, command, run directory and numbers of every experiment is `docs/research-log.md` (a copy
of `runs/research-log.md`, which lives next to the runs and, like them, is not in git). "Entry" numbers
below refer to it.

**The corpus.** 525 curated facts about one person in 26 topics, each in Portuguese and English
(`data/yan/kb/`). An LLM teacher (`scripts/write_questions.py`) wrote 14 questions per fact and language:
11 for training, 1 for validation, 1 for test and 1 for a **final** slice that no experiment has read. With
multi-fact, small-talk and out-of-scope questions that is 26,236 training, 1,250 validation, 1,278 test
and 1,210 final records (`scripts/build_corpus.py`; frozen copy `runs/data-full2`). Test split: 1,006
single-fact, 68 multi-fact, 44 about the model itself, 50 small-talk and 110 out-of-scope questions.

**How to read the numbers.**

- *Held-out means differently worded.* The teacher wrote the held-out questions to avoid the training
  wording: 85% of the single-fact test questions share less than 35% of their words with every training
  phrasing of their fact, and accuracy depends on that distance more than on anything else (table at the
  end of this section). The test split is the hard end of what a visitor can ask.
- *Exact match is strict.* It asks for the one curated sentence of the gold fact. With up to 56 facts
  about one project, the neighbouring fact is often a fine answer; "right topic" counts those.
- *One run per setting.* The standard error of an exact-match figure on 1,278 questions is about 1.3
  points and two runs of the same setting on MPS differ by 1 to 2 points: differences under 3 points are
  noise unless they repeat on the validation split.
- *The test split chose the recipe.* Validation is small, so held-out **test** numbers guided the
  decisions between experiments; that makes the test numbers of the final model optimistic by selection.
  Only the decoding prior and the fusion weight were chosen on validation alone. The "final" rows of the
  released model, when present, come from the slice that was read once.

### The released model

<!-- RELEASE-TABLE:BEGIN (generated by scripts/readme_results.py; do not edit by hand) -->
**`brasa-4m`** (4,188,416 parameters; run `runs/j2-4m-joint-pre`, last checkpoint, released 2026-11-13 06:41). Greedy decoding; exact match and the other columns in %.

| Questions | Decoding | n | Exact match | Token F1 | Right topic | Curated sentence | Refusal P / R | Unsupported numbers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| seen in training (sample) | site default | 300 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 / 100.0 | 0.0 |
| held-out test | site default: retrieved facts + ranking prior | 1278 | 47.6 | 62.7 | 88.0 | 99.8 | 73.9 / 87.3 | 0.0 |
| held-out test | site default, **int8 weights** (what the browser runs) | 1278 | 47.4 | 62.6 | 87.9 | 99.8 | 73.9 / 87.3 | 0.0 |
| held-out test | retrieved facts only (no prior) | 1278 | 43.8 | 59.8 | 87.0 | 99.8 | 75.4 / 86.4 | 0.0 |
| held-out test | free generation, open-book | 1278 | 42.6 | 59.2 | 86.1 | 98.4 | 75.8 / 85.5 | 4.2 |
| held-out test | free generation, closed-book (no facts in the prompt) | 1278 | 34.7 | 53.4 | 81.0 | 96.2 | 77.4 / 87.3 | 46.8 |
| held-out test | the retriever's first fact, no language model | 1278 | 39.3 | 55.5 | 88.6 | 100.0 | - / 0 | 0.0 |

Site default by kind of question and language (held-out test, float32, exact match): one fact 46.0 (n=1006); several facts 23.5 (n=68); about the model 56.8 (n=44); small talk 20.0 (n=50); out of scope 85.5 (n=110); en 47.7; pt 47.4.

Retriever (BM25 fused with the model's own ranking, weight 1.5), held-out test: gold fact first for 45.0%, in the top 2 for 60.6%, in the top 4 for 74.8%; exact match 60.3% when it was retrieved and 0.0% when it was not.

Ranking prior: `score`, weight 24.0, chosen on the validation questions.

Started from `runs/pre-4m`: 12,000 steps of next-token prediction on the Wikipedia slice (295M tokens seen, 25 min, validation perplexity 25.0), from random weights; then 6,000 steps on the questions (29 min).

Quantisation: int8 relative weight error 0.007224, largest logit drift 0.21453; answers identical to float32: 17418/17696 (int8), 17681/17696 (float16).

Files the site loads: `model.i8.bin` 4,296,144 bytes (3,950,399 with brotli), `retrieval.json` 6,746,959 bytes (1,919,490 with brotli), `tokenizer.json` 38,945 bytes (12,289 with brotli).

Engine speed in Node v24.22.2 with the int8 file: 305 tokens/s (js), 2,865 tokens/s (wasm-simd).

Engine parity against PyTorch (ok): largest logit difference 1.8e-05 over 4 format x kernel combinations; constrained greedy answers identical: 24/24.

End to end: of 200 held-out test questions asked through the browser session in Node and through Python with the i8 file, 200 got the same answer and 200 the same retrieved facts.

The final slice has not been read for this release.
<!-- RELEASE-TABLE:END -->

The release is produced by `scripts/release.sh` and lives in `runs/final/` (export directory,
`model-card.json`, `MODEL_CARD.md`, `RELEASE.json` with the exact commands). See
[Status](#status) for which run it currently holds.

### What the held-out questions look like

Four test questions and the answers of the released `brasa-4m` with the site's decoding (from
`runs/final/predictions.jsonl`), one of each outcome:

| Question (never seen in training) | Answer | Outcome |
| --- | --- | --- |
| When reviewing results person by person, which seer.app development comes from Yan? | Yan developed the seer.app sales dashboard, including a performance-by-salesperson chart. | exact |
| To pick up a sales discussion, what prior material does Isabela consult? | Isabela's value lies in turning initial service into a conversation that considers the physician's interest, the campaign context and the information needed to move forward to a consultant. *(gold: "Isabela receives the campaign context and retrieves the participation history before replying ...")* | a true sentence about the right project, not the one asked for |
| The garage door squeaks whenever it opens. What should I check? | I can't answer that. I am a small model trained only to talk about Yan Simmer: his projects, experience, studies and skills. | out of scope, declined |
| How does a search engine find websites? | Nina's vector search uses OpenAI embeddings in a Supabase knowledge base and filters content by type and audience ... | out of scope, answered with a fact that shares its words |

Everyday questions ("Onde o Yan estuda?", "How can I contact him?", "o que é a brasa") are much closer to
the training wording than these and are usually answered correctly; the test split was written to be the
hard case.

### What each idea was worth

One ladder, `brasa-4m`, 6,000 steps, seed 0, held-out test (n = 1,278). Each row changes one thing with
respect to the row above it; "site decoding" is what a visitor gets with that model.

| Step | Run | Gold fact in top 1 / 4 | Exact match | Right topic | Refusal P / R | Closed-book EM |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Baseline: BM25 fields index, free decoding | `f3-4m-fields` | 29.8 / 57.3 | 33.3 | 80.4 | 0.69 / 0.88 | 25.4 |
| Decoding restricted to any curated sentence | same weights | 29.8 / 57.3 | 33.5 | 81.7 | 0.69 / 0.88 | 25.7 |
| Decoding restricted to the retrieved facts | same weights | 29.8 / 57.3 | 34.5 | 81.7 | 0.68 / 0.90 | |
| The network also ranks the facts (contrastive loss) | `j1-4m-joint` | 39.0 / 70.3 | 37.6 | 85.9 | 0.67 / 0.81 | 29.7 |
| ... and starts from the Wikipedia pretraining | `j2-4m-joint-pre` | 45.0 / 74.8 | 43.8 | 87.0 | 0.75 / 0.86 | 34.7 |
| ... and the ranking is a prior on the answer (score, weight 24) | same weights | 45.0 / 74.8 | 47.6 | 88.0 | 0.74 / 0.87 | |
| The retriever's first fact, no language model | | 45.0 / 74.8 | 39.3 | 88.6 | - / 0 | |

Are the steps more than noise? The same questions answered by two systems, difference in exact match
with a 95% bootstrap interval over questions (`scripts/analysis/paired.py`); validation and test agree:

| Step | Validation (n = 1,250) | Test (n = 1,278) |
| --- | --- | --- |
| Free decoding -> retrieved facts only | +1.2 [-0.2, +2.6] | +1.2 [-0.2, +2.6] |
| -> the network also ranks the facts | +5.1 [+2.5, +7.8] | +3.1 [+0.3, +5.9] |
| -> it starts from the pretraining | +4.8 [+1.9, +7.7] | +6.2 [+3.4, +9.1] |
| -> the ranking is a prior on the answer | +2.5 [+1.0, +3.9] | +3.8 [+2.3, +5.2] |
| Baseline -> best `brasa-4m` | +13.6 [+10.8, +16.6] | +14.2 [+11.3, +17.1] |
| The retriever's first fact -> best `brasa-4m` | +5.2 [+3.0, +7.4] | +8.3 [+6.2, +10.5] |

These intervals cover the sampling of questions, not the run-to-run noise of training (1 to 2 points).

Earlier steps, measured before this ladder: the `fields` layout of the BM25 index (gold in top 4 from 47.3
to 58.7% on the partial corpus), selecting the checkpoint on validation exact match instead of loss
(the lowest-loss checkpoint scored 15.7% where the last one scored 21.2%), and loss scaling under float16
(without it the pretraining run sat at chance; the question-corpus runs were unaffected).

What the ladder says:

1. **Free generation was never the problem.** The free model already writes a curated sentence 97% of the
   time; it picks the wrong real fact. Constraining the output to the knowledge base changes the
   guarantee, not the accuracy. Constraining it to the *retrieved* facts adds a point and makes every
   number in an answer one the visitor can see in the sources.
2. **Selecting the fact from an unseen phrasing is the bottleneck, and matching is a different skill from
   answering.** Lexical retrieval stops at 30% recall@1; alias routing, a nearest-training-question vote
   and other lexical features moved it by a point. Training the same network with a contrastive loss to
   put a question next to its fact gives 39%, at no cost in download.
3. **General language knowledge helps only through that matching objective.** Pretrained on 160 MB of
   Wikipedia per language (25 minutes, from random weights), the model is no better as a reader (33.5%
   against 34.5%), but as an encoder it is much better (recall@1 39 -> 45%), and the two together are
   worth 9 points of exact match and 9 points closed-book.
4. **The reader is not better than its own ranking.** On single-fact questions the retriever's first hit
   is right 47.5% of the time and the language model choosing among the four retrieved facts 41.2%.
   Adding the ranking to the model's log-probabilities as a prior recovers most of that (46.0%) for
   free. What the language model adds over the retriever is the decision to decline (84.5% of
   out-of-scope questions; a retriever alone declines none), small talk, and multi-fact answers.
5. **A smaller knowledge base is not a more accurate one.** Trained on the 362 most important facts only,
   the model answers the questions about those facts no better (27.5% against 28.9%), and answers 93%
   of the questions about the dropped facts with a neighbouring fact instead of declining.

### What did not help

| Idea | Result | Entry |
| --- | --- | ---: |
| Counterfactual contexts (numbers and names of the gold fact rewritten in context and answer, so that it must be copied) | 20.7 and 21.7 against 21.1 (`brasa-4m`), 23.8 against 24.1 (`brasa-12m`), partial corpus | 2, 4 |
| Cross-fitted training contexts (the retriever of a training question has never seen it) | 22.1 against 23.7; exact match when the gold fact is retrieved fell from 36.6 to 31.9 | 4 |
| Cross-fitted contexts that keep the retriever's order | +2.7 on the partial corpus, +0.6 on the full one | 5, 8 |
| Word dropout in the training questions | 32.2 against 34.5; closed-book 22.3 against 25.4 | 12 |
| Beam search over the allowed answers (beam 4) | 22.1 against 21.4 | 1 |
| Re-ranking more candidates with the model's own log-probability (top 8, top 16) | worse than top 4: the model's preference among facts is weaker evidence than the retriever's rank | 7 |
| Alias routing; nearest-training-question vote (lexical) | recall@1 +0.5 to +1.8 on validation, nothing on test | 8 |
| Fact vectors moved towards the centroid of their training questions | fused recall@1 45.0 -> 44.5 | 14 |
| Training on the important facts only | see point 5 above | 14 |
| Pretraining alone (without the contrastive objective) | 33.5 against 34.5; +2.0 against a control with the same tokenizer | 12 |

### How far a question is from the training wording

Single-fact test questions (n = 1,050), grouped by the highest word overlap (Jaccard, accents and case
folded) between the question and any training question of the same fact
(`scripts/analysis/novelty.py`). First the full-corpus baseline with free decoding, then the best
`brasa-4m` with the site's decoding:

| Similarity to the nearest training phrasing | n | Baseline: exact | right topic | gold fact first | Best `brasa-4m`: exact | right topic | gold fact first |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 to 0.20 | 204 | 18.1 | 71.1 | 18.1 | 30.9 | 78.9 | 32.8 |
| 0.20 to 0.35 | 686 | 27.4 | 84.7 | 29.9 | 47.1 | 90.7 | 47.4 |
| 0.35 to 0.50 | 149 | 46.3 | 87.9 | 57.0 | 62.4 | 96.0 | 68.5 |
| 0.50 to 1.00 | 11 | 45.5 | 81.8 | 54.5 | 81.8 | 100.0 | 81.8 |

A question that shares a third of its words with something the model was trained on is answered exactly
about six times in ten; one that shares almost none, three times in ten. The work of this project moved
every row, but the slope is still there: what the model has is tolerance to rewording, not understanding.

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and the dependencies into
`.venv`), and Node 20 or newer for the engine tests.

```sh
uv sync                        # add --extra onnx for the ONNX export and its test
uv run ruff check
uv run pytest                  # 158 tests; the ones that need Node or ONNX are skipped when missing
(cd web && npm test)           # 62 engine tests on the committed fixture, no dependencies to install
scripts/smoke.sh               # the whole pipeline on the fictional corpus, about 3.5 minutes on an M3 Max
```

The release recipe on the real corpus (the exact commands of the current release are in
`runs/final/RELEASE.json`; `<size>` is `4m` or `12m`):

```sh
uv run python scripts/build_corpus.py --data-dir data/yan --out runs/data-full2  # questions -> train/val/test/final
uv run python -m brasa.data --data-dir runs/data-full2                           # validate the contract

# 1. pretraining, from random weights, on a pinned slice of Wikipedia (kept out of git)
uv run --with pyarrow python scripts/fetch_wiki.py                               # data/cache/wiki/, with a manifest
uv run python -m brasa.pretrain --preset brasa-<size> --data-dir runs/data-full2 --out runs/pre-<size> \
    --text data/cache/wiki/wiki.pt.txt data/cache/wiki/wiki.en.txt --corpus data/cache/wiki/prep-full2 \
    --steps 12000                                                                # 10000 for brasa-12m (about an hour)

# 2. the question corpus: one network that answers and ranks the facts
uv run python -m brasa.train --preset brasa-<size> --data-dir runs/data-full2 --out runs/brasa-<size> \
    --steps 6000 --index fields --retrieval-weight 1.0 --init-from runs/pre-<size>

# 3. choose the decoding prior on validation, evaluate (float32, float16, int8), export, check the
#    JavaScript engine against PyTorch, render the model card, collect runs/final
SPLITS=train,val,test,final scripts/release.sh runs/brasa-<size> runs/final      # "final" only once, for the release

uv run python -m brasa.sample --export runs/final/export --format i8             # chat in the terminal
node ../scripts/sync-brasa.mjs runs/final/export runs/final/model-card.json      # publish to the site (from ml/)
```

Every module has `--help`. Other entry points: `brasa.model` (parameter counts, cache self-test),
`brasa.tokenizer` (train or inspect a tokenizer), `brasa.retrieval` (query the index, recall@k per
split), `brasa.encoder` (the stand-alone dual encoder and its recall report), `brasa.evaluate
--tune-prior`, `brasa.export_onnx`, `scripts/make_rehearsal.py`, `scripts/make_fixtures.py`,
`scripts/paraphrase_ablation.py`, `web/wasm/build.sh` (rebuilds `kernel.wasm` with `zig cc` from a PyPI
wheel; the committed binary is reproduced byte for byte).

## Repository layout

```
src/brasa/        config, model, tokenizer, retrieval, encoder, constrain, data, train, pretrain, evaluate,
                  export, export_onnx, sample, fixtures
tests/            pytest suite; tests/fixtures/tiny is a tiny random model exported the normal way
web/engine/       engine.js, tokenizer.js, retrieval.js (retriever and answer trie), prompt.js, session.js,
                  worker.js, client.js, *.d.ts
web/engine/test/  Node tests;  web/engine/tools/  parity.mjs, bench.mjs
web/wasm/         kernel.c, build.sh, kernel.wasm
scripts/          release.sh, build_corpus.py, write_questions.py, fetch_wiki.py, render_model_card.py, smoke.sh,
                  smoke_summary.py, make_rehearsal.py, make_fixtures.py, paraphrase_ablation.py
data/yan/         the real knowledge base (kb/), the teacher's questions (qa-src/) and the built corpus (qa/)
data/rehearsal/   the fictional corpus in the contract format
data/cache/       downloaded text for pretraining (ignored by git)
docs/             research-log.md: the lab notebook of every experiment on the real corpus
runs/             training runs, exports, runs/final (the release); ignored by git
```

## What the tests cover

- Model: shapes and parameter counts of every preset; KV-cache forward equals the full-context forward;
  cached generation equals uncached generation; causality; the RoPE relative-position property; loss
  masking and weighting; sampling filters.
- Tokenizer: round trips on Portuguese accents, emoji and mixed case; normalisation; control tokens are
  never produced from text and never split; token boundaries; deterministic training.
- Retrieval: ranking behaviour; only training questions are indexed; Python and JavaScript return the
  same ranking with identical scores, lexical and fused; the dual encoder learns to retrieve on a toy
  corpus, alone and inside the language model.
- Constrained decoding: the trie accepts exactly the allowed answers; greedy decoding equals a
  step-by-step reference with and without priors; the JavaScript trie and engine reproduce Python token
  by token.
- Data: the contract and its violations; held-out phrasings never reach training or the tokenizer; the
  loss mask covers exactly the answer span; batches are a pure function of seed and step.
- Training: schedule, logging, checkpoints, exact resume, loss scaling, pretraining and `--init-from`.
- Evaluation: every metric on hand-made examples; every decoding mode on random weights; the prior
  tuning rule; the model card renderer.
- Export: quantisation error bounds, container layout and round trip, a fresh export run through the
  Node engine with both kernels, a corrupted weight is detected, the committed fixture is current, the
  ONNX graphs reproduce PyTorch.

## Limitations

Numbers in this list are those of [the released model](#the-released-model) on held-out test questions.

- **Generalisation to new phrasings is still the weak point.** The model gives exactly the requested fact
  for a little under half of the held-out phrasings and a fact of the right topic for almost nine in ten.
  The questions that fail approach a fact with words no training question used.
- **Wrong answers are fluent, and now also true.** With the default decoding a wrong answer is a real
  curated sentence about something else, usually a sibling fact, and it is one of the sources shown next
  to the answer. It cannot be an invented number; it can be beside the point.
- **The language model does not out-read its own retriever.** On single-fact questions the first hit of
  the ranking is as good as the model's choice among the four retrieved facts. The model earns its place
  by declining, greeting and chaining facts, and by being the encoder that produces the ranking.
- **Multi-fact questions and small talk are weak**: about a quarter of the multi-fact answers and a fifth
  of the (deliberately unusual) held-out greetings are exact.
- **Refusal is learned, not enforced.** About 86% of the unseen out-of-scope questions are declined, and
  about a quarter of the refusals answer a question that had an answer. `prior.fixed` moves that
  trade-off.
- **The questions were written by an LLM teacher, not by visitors.** The held-out phrasings are hard on
  purpose; real questions may be closer to the training wording, or further in ways the teacher did not
  imagine. No visitor data has been used.
- **The numbers are optimistic by selection** (see "How to read the numbers") and come from one run per
  setting; training on MPS is not bit-reproducible.
- **`retrieval.json` is larger than the small model** (6.7 MB, 1.9 MB with brotli): the BM25 postings of
  the training questions dominate it.
- **Language detection in the evaluation is a word-list heuristic**, and the unsupported-number metric
  only sees digits ("nine" is not checked).
- **One question at a time.** There is no dialogue state; a follow-up such as "and before that?" is
  answered as if it were the first message.
- The evaluation measures agreement with one reference answer per question. "Right topic" is the only
  credit an adjacent correct fact gets.

## License

MIT, see [LICENSE](LICENSE). The rehearsal corpus is fictional; any resemblance to real people or
companies is accidental.
