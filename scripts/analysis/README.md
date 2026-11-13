# Analyses behind the research log

One-off scripts that produced tables of `docs/research-log.md`. They read run directories under `runs/`
(not in git) and print to the terminal; each docstring gives the command that was run. The log refers to them
by their working path `runs/scratch/<name>.py`: these are the same files.

| Script | What it answers | Log entry |
| --- | --- | ---: |
| `summ.py` | one line per decoding mode of an `eval-*.json` file | 8, 12 |
| `retrieval_analysis.py`, `retrieval_knn.py` | where BM25 fails on held-out phrasings; alias routing; nearest-training-question vote | 1, 8 |
| `rerank_scores.py`, `rerank_rules.py` | the model as a re-ranker of the top k retrieved facts | 7 |
| `novelty.py` | accuracy against the distance of a question from the training wording of its fact | 10 |
| `hybrid.py` | "the model decides whether to answer, the first hit is the fact" | 12 |
| `prior_sweep.py` | the ranking prior and the fixed-answer prior, per weight | 13, 15, 16 |
| `kb_size.py` | all facts against the important facts only, on the same questions | 14 |
| `dense_expand.py` | fact vectors moved towards their training questions | 14 |

Run them from `ml/`, for example `uv run python scripts/analysis/novelty.py runs/<run>/predictions.jsonl --mode open_book_ranked`.
