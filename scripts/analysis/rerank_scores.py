"""Score every candidate answer of every held-out question with the model (teacher forced), and save.

    uv run python runs/scratch/rerank_scores.py runs/f1-4m-fields last 16 cpu

For each val/test question: the top-K retrieved facts (text in the language of the matching document) and
the fixed answers; for each, log p(answer + <eos> | open-book prompt) and the same for the closed-book
prompt. Output: runs/<run>/rerank-<ckpt>.json, read by runs/scratch/rerank_rules.py.
"""
import json
import sys
import time

import torch

from brasa.config import EOS, PAD, PromptConfig
from brasa.data import build_prompt, load_dataset
from brasa.evaluate import load_run

run_dir, ckpt, K, device = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
splits = sys.argv[5].split(",") if len(sys.argv) > 5 else ["val", "test"]
model, tok, retriever, run = load_run(run_dir, ckpt, device)
ds = load_dataset(run["train_config"]["data_dir"])
pc = PromptConfig(**run["train_config"]["data"]["prompt"])
pad, eos = tok.special_ids[PAD], tok.special_ids[EOS]
canon = retriever.index["answers"]["canonical"]
enc = {}


def ids_of(text):
    if text not in enc:
        enc[text] = tok.encode(text) + [eos]
    return enc[text]


@torch.no_grad()
def score(seqs, starts, max_tokens=40000):
    """sum of log p over seq[start:] for each sequence."""
    out = [0.0] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    i = 0
    while i < len(order):
        rows = [order[i]]
        while i + len(rows) < len(order) and (len(rows) + 1) * len(seqs[order[i + len(rows)]]) <= max_tokens:
            rows.append(order[i + len(rows)])
        width = len(seqs[rows[-1]])
        buf = torch.full((len(rows), width), pad, dtype=torch.long)
        for r, j in enumerate(rows):
            buf[r, : len(seqs[j])] = torch.tensor(seqs[j])
        logits, _, _ = model(buf[:, :-1].to(device))
        logp = logits.float().log_softmax(-1).cpu()
        for r, j in enumerate(rows):
            s, n = starts[j], len(seqs[j])
            tgt = buf[r, s:n]
            out[j] = float(logp[r, s - 1 : n - 1].gather(-1, tgt[:, None]).sum())
        i += len(rows)
    return out


t0 = time.time()
records, seqs, starts, owner = [], [], [], []
for split in splits:
    for rec in ds.split(split):
        hits = retriever.search(rec.q, K)
        open_ids, used = build_prompt(tok, rec.q, [h.text for h in hits[: pc.max_facts]], model.cfg.block_size, pc)
        closed_ids, _ = build_prompt(tok, rec.q, None, model.cfg.block_size, pc)
        cands = [{"fact": h.id, "lang": h.lang, "text": h.text, "bm25": h.score, "topic": h.topic} for h in hits]
        cands += [{"fact": None, "lang": c["lang"], "text": c["text"], "bm25": 0.0, "topic": None} for c in canon]
        r = {"id": rec.id, "split": split, "lang": rec.lang, "kind": rec.kind, "q": rec.q, "gold": rec.a,
             "facts": list(rec.facts), "used": used, "cands": cands}
        records.append(r)
        for c in cands:
            a = ids_of(c["text"])
            for name, prompt in (("open", open_ids), ("closed", closed_ids)):
                if len(prompt) + len(a) > model.cfg.block_size:
                    c[name] = None
                    continue
                seqs.append(prompt + a)
                starts.append(len(prompt))
                owner.append((c, name))
            c["n"] = len(a)
print(len(records), "records", len(seqs), "sequences", sum(map(len, seqs)), "tokens")
for (c, name), v in zip(owner, score(seqs, starts), strict=False):
    c[name] = round(v, 4)
out = f"{run_dir}/rerank-{ckpt}.json"
json.dump(records, open(out, "w"), ensure_ascii=False)
print("wrote", out, round(time.time() - t0, 1), "s")
