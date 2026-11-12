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
transformer, the byte-level BPE tokenizer, the training loop, the lexical retriever and the browser
inference engine are all hand-written, with no pretrained weights and no model libraries. It answers
questions about one person, in Portuguese or English, from a small curated knowledge base, and it is
small enough to run entirely in a web browser.

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

Facts come from a BM25 retriever over the knowledge base (`retrieval.json`), up to {{max_facts}} per
question. Out-of-scope questions are answered with a short refusal in the language of the question.

## Evaluation

Greedy decoding. "Seen" are training questions; "held-out" are paraphrases whose wording never occurs in
training, plus out-of-scope questions never seen. EM is normalised exact match, F1 is token-level,
"key spans" is the share of required names and numbers present in the answer, "bad numbers" is the share
of numbers in the answers that appear neither in the gold answer, nor in the retrieved facts, nor in the
question.

{{evaluation_table}}

{{retrieval_note}}

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
