---
license: mit
language:
  - pt
  - en
library_name: pytorch
pipeline_tag: text-generation
tags:
  - from-scratch
  - gpt
  - small-language-model
  - question-answering
  - retrieval-augmented-generation
  - webassembly
  - onnx
---

<!-- Rendered by scripts/render_model_card.py from model-card.json. Edit MODEL_CARD.template.md, not the output. -->

# {{name}}

{{corpus_warning}}

Brasa is a small GPT-style language model **written and trained from scratch** in PyTorch: the
transformer, the byte-level BPE tokenizer, the training loop, the retriever and the browser
inference engine are all hand-written, with no pretrained weights and no model libraries. It answers
questions about one person, in Portuguese or English, from a small curated knowledge base, and it is
small enough to run entirely in a web browser.

{{pretraining_note}}

It is a narrow model on purpose. It is **not** a general assistant: it knows nothing outside its
knowledge base and is trained to say so.

| | |
| --- | --- |
| Parameters | {{params}} |
| Architecture | decoder-only transformer, {{n_layer}} layers, width {{d_model}}, {{n_head}} heads, RMSNorm (pre-norm), rotary position embeddings, SwiGLU, tied embeddings, no biases |
| Context length | {{block_size}} tokens |
| Tokenizer | byte-level BPE trained from scratch, {{vocab_size}} tokens (7 control tokens) |
| Languages | Portuguese, English |
| Training | {{steps}} steps, AdamW, cosine schedule, {{train_time}} on `{{device}}` |
| License | MIT |

## Files

{{files_table}}

`model.i8.bin` and `model.f16.bin` use the single-file "TGPT" container described in the code
repository (JSON header with the configuration, the prompt format and the tokenizer, followed by the
tensors). `model.i8.bin` stores every matrix as int8 with one scale per output row.

## How to use

In the browser, with the engine from the code repository (no dependencies, WebAssembly SIMD with a plain
JavaScript fallback):

```js
import { BrasaClient } from './engine/index.js';

const client = new BrasaClient(new Worker(new URL('./engine/worker.js', import.meta.url), { type: 'module' }));
await client.load({ model: 'model.i8.bin', wasm: 'kernel.wasm', retrieval: 'retrieval.json' });
const run = client.ask('Onde ele mora?', {}, { onToken: (t) => console.log(t.delta) });
console.log((await run.done).text);
```

In Python, with the `brasa` package from the code repository:

```sh
uv run python -m brasa.sample --export . --format i8 -q "Where does he live?"
```

## Prompt format

```
open-book    <|bos|><|ctx|>fact<|sep|>fact<|sep|>fact<|q|>question<|a|>answer<|eos|>
closed-book  <|bos|><|q|>question<|a|>answer<|eos|>
```

{{retriever_note}} Up to {{max_facts}} facts are put in the prompt. Out-of-scope questions are answered
with a short refusal in the language of the question.

## Evaluation

Greedy decoding. "Seen" are training questions; "held-out" are questions whose wording never occurs in
training (written to be phrased differently: most share less than a third of their words with every
training phrasing of the same fact), plus small talk and out-of-scope questions never seen.

Decoding modes. *Free*: the model writes whatever it wants. *Any curated sentence*: every step may only
continue one of the curated sentences of the knowledge base (or a fixed greeting / refusal), so the
answer is always a real fact, possibly the wrong one. *Retrieved facts only*: the same, restricted to
the facts the retriever put in the prompt; every statement and number in the answer is then one the
visitor can see in the sources. *Retrieved facts + ranking prior* (the site's default): the same, and the
retriever's ranking counts as a prior on the model's choice, so a fact the retriever put lower has to be
that much more likely for the model to win; it costs no extra computation. *Retriever's first fact* uses
no model: it is the baseline the open-book rows have to beat, and it can never decline a question.

EM is normalised exact match against the one gold answer, F1 is token-level, "key spans" is the share of
required names and numbers present in the answer, "bad numbers" is the share of numbers in the answers
that appear neither in the gold answer, nor in the retrieved facts, nor in the question (with constrained
decoding these are numbers of other real facts, not inventions).

{{evaluation_table}}

Exact match is strict: with dozens of closely related facts per project, a neighbouring fact of the right
topic is often a reasonable answer. The next table adds what a visitor experiences: whether the answer is
a curated sentence at all, whether it is about the right topic, and how often the retriever had the gold
fact among its first hits.

{{visitor_table}}

{{decoding_note}} {{retrieval_note}}

{{breakdown_tables}}

The held-out test questions were also used to choose between training settings, so the test rows are
optimistic by selection; the "final" rows, when present, come from a slice of questions that was read once,
for this card.

Quantisation: {{quantisation_note}}

## Training data

{{data_note}}

## Limitations

- The model only knows the facts of its knowledge base, and only in the phrasings it can generalise to
  from its training questions. Unusual wordings can retrieve the wrong fact or get a wrong answer that
  sounds confident.
- Answers are short and close to the wording of the knowledge base. It does not reason, calculate,
  hold a conversation across turns or follow instructions.
- With {{params}} parameters and a {{block_size}}-token context it is not comparable to general language
  models and is not meant to be.
- The refusal behaviour is learned from examples, not guaranteed: an out-of-scope question can still
  get an in-scope-looking answer.

## Reproduce

See the code repository: `uv sync`, `scripts/smoke.sh` for the end-to-end rehearsal, and the README
section "Reproduce" for the release recipe.
