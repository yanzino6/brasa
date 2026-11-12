# Rehearsal corpus (FICTIONAL)

**"Téo Arantes" does not exist.** Neither do Lumen Forge, Brisa Logistics, Sentinela, NotaClara, Tessela
or the e-mail address `hello@lumenforge.example`. Every fact in this directory was invented.

The corpus exists for two reasons:

1. to rehearse the whole pipeline (tokenizer, retrieval, training, evaluation, export, browser engine)
   before the real knowledge base is written, and to keep the tests and `scripts/smoke.sh` runnable by
   anyone who clones the repository;
2. to measure, on a corpus where the ground truth is known by construction, how many question
   paraphrases a fact needs before the model answers phrasings it has never seen.

Nothing measured on this corpus is a result of the released model.

## Contents

It follows the data contract checked by `brasa.data.load_dataset` (see the main README):

| File | Records | What |
| --- | --- | --- |
| `corpus.json` | 1 | Name and description of the corpus, with `"fictional": true` (copied into every run and model card) |
| `kb/facts.jsonl` | 30 | One fact per line, in Portuguese and English |
| `kb/topics.json` | 9 | Topics with titles and aliases |
| `qa/facts.jsonl` | 1500 | Single-fact questions (kinds `single` and `meta`) |
| `qa/multi.jsonl` | 100 | Questions that need two facts |
| `qa/chitchat.jsonl` | 80 | Greetings, thanks, goodbyes, "what can I ask?" |
| `qa/oos.jsonl` | 416 | Out-of-scope questions and questions about personal details that are not in the knowledge base, answered with a refusal |

Splits: 1792 train, 88 validation, 216 test, half Portuguese and half English.

* Per fact and language there are 14 hand-written questions. Two are held out for **test** and one for
  **validation**; their phrasings never occur in training. The other 11 are training paraphrases, and
  each one also appears once wrapped in a greeting or a polite ending (22 training questions per fact
  and language).
* Out-of-scope **test** questions are of two kinds: frames seen in training with fillers never seen
  ("What is the capital of *Portugal*?"), and frames never seen at all.

## Regenerate

```sh
uv run python scripts/make_rehearsal.py              # this directory, byte for byte
uv run python scripts/make_rehearsal.py --k 4 --out runs/rehearsal-k4     # only 4 training paraphrases per fact
```

The source of the persona is `scripts/rehearsal_persona.py` and `scripts/rehearsal_persona_extra.py`.
