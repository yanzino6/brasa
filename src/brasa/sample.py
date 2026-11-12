"""Chat with Brasa from the command line.

    uv run python -m brasa.sample --run runs/brasa-12m                          # interactive
    uv run python -m brasa.sample --run runs/brasa-12m -q "Onde ele mora?"      # one question, then exit
    uv run python -m brasa.sample --export runs/brasa-12m/export --format i8    # the exported int8 weights
    uv run python -m brasa.sample --run runs/smoke --closed-book --temperature 0.7 --top-p 0.9

A question goes through the same steps as in the browser: retrieve facts from the knowledge base, build
the prompt, generate until <|eos|> with the KV cache. --show-context prints what the retriever found and
--show-prompt the exact token sequence the model receives.

Inside the chat: /open and /closed switch the mode, /context toggles the retrieved facts, /quit leaves.
"""

from __future__ import annotations

import argparse
import codecs
import json
import sys
import time
from pathlib import Path

import torch

from .config import EOS, PromptConfig
from .data import build_prompt
from .model import GPT
from .retrieval import Bm25Retriever
from .tokenizer import BPETokenizer


def load_export(export_dir: str | Path, fmt: str = "i8"):
    """(model, tokenizer, retriever or None, prompt config) from an export directory."""
    from .export import read_tgpt

    export_dir = Path(export_dir)
    header, model = read_tgpt(export_dir / f"model.{fmt}.bin")
    if not header.get("tokenizer"):
        raise ValueError("the model file does not embed a tokenizer")
    tok = BPETokenizer.from_json(header["tokenizer"])
    index = export_dir / "retrieval.json"
    retriever = Bm25Retriever.load(index) if index.exists() else None
    return model, tok, retriever, PromptConfig(**header["prompt"])


@torch.no_grad()
def answer(model: GPT, tok: BPETokenizer, retriever, question: str, prompt_cfg: PromptConfig, *,
           open_book: bool = True, max_new_tokens: int | None = None, temperature: float = 0.0,
           top_k: int | None = None, top_p: float | None = None, repetition_penalty: float = 1.0,
           generator: torch.Generator | None = None, on_text=None) -> dict:
    """Answer one question. on_text(piece) is called with each piece of text as soon as it is complete.

    Returns {"answer", "hits", "facts_used", "prompt_ids", "tokens", "seconds", "tokens_per_second"}."""
    device = next(model.parameters()).device
    hits = retriever.search(question, prompt_cfg.max_facts) if open_book and retriever is not None else []
    ids, used = build_prompt(tok, question, [h.text for h in hits], model.cfg.block_size, prompt_cfg)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    eos = tok.special_ids[EOS]
    pieces, tokens = [], []
    t0 = time.perf_counter()
    stream = model.stream(torch.tensor([ids], device=device), max_new_tokens or prompt_cfg.answer_reserve,
                          temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                          stop_ids=(eos,), generator=generator)
    for nxt in stream:
        token = int(nxt)
        tokens.append(token)
        if token == eos:
            break
        piece = decoder.decode(tok.decode_bytes([token]))
        if piece:
            pieces.append(piece)
            if on_text:
                on_text(piece)
    tail = decoder.decode(b"", final=True)
    if tail:
        pieces.append(tail)
        if on_text:
            on_text(tail)
    seconds = time.perf_counter() - t0
    return {"answer": "".join(pieces).strip(), "hits": hits[:used], "facts_used": used, "prompt_ids": ids,
            "tokens": tokens, "seconds": seconds, "tokens_per_second": len(tokens) / max(seconds, 1e-9)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Chat with a trained Brasa model.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="run directory written by brasa.train (float32 checkpoint)")
    source.add_argument("--export", help="export directory written by brasa.export (quantised weights)")
    ap.add_argument("--checkpoint", default="last", choices=["last", "best"], help="with --run")
    ap.add_argument("--format", default="i8", choices=["i8", "f16", "f32"], help="with --export")
    ap.add_argument("-q", "--question", action="append", help="ask this and exit (repeatable)")
    ap.add_argument("--closed-book", action="store_true", help="do not retrieve facts into the prompt")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default: the answer reserve of the model")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=0, help="0 = off")
    ap.add_argument("--top-p", type=float, default=1.0, help="1 = off")
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu is fast enough for one question at a time")
    ap.add_argument("--show-context", action="store_true", help="print the retrieved facts")
    ap.add_argument("--show-prompt", action="store_true", help="print the exact prompt given to the model")
    ap.add_argument("--json", action="store_true", help="with -q: print one JSON object per question")
    args = ap.parse_args(argv)

    if args.run:
        from .evaluate import load_run

        model, tok, retriever, run = load_run(args.run, args.checkpoint, args.device)
        prompt_cfg = PromptConfig(**run["train_config"]["data"]["prompt"])
    else:
        model, tok, retriever, prompt_cfg = load_export(args.export, args.format)
        model.to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    state = {"open": not args.closed_book and retriever is not None, "context": args.show_context}

    def ask(question: str) -> None:
        out = answer(
            model, tok, retriever, question, prompt_cfg, open_book=state["open"], max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k or None, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty, generator=generator,
            on_text=None if args.json else lambda piece: (sys.stdout.write(piece), sys.stdout.flush()),
        )
        if args.json:
            print(json.dumps({"question": question, "answer": out["answer"],
                              "mode": "open" if state["open"] else "closed",
                              "facts": [h.id for h in out["hits"]], "tokens": len(out["tokens"]),
                              "tokens_per_second": round(out["tokens_per_second"], 1)}, ensure_ascii=False))
            return
        print()
        if state["context"]:
            for h in out["hits"]:
                print(f"    [{h.score:6.2f}] {h.id} ({h.lang}): {h.text}")
        if args.show_prompt:
            print("    prompt:", tok.decode(out["prompt_ids"], skip_special=False))
        print(f"    ({len(out['prompt_ids'])} prompt tokens, {len(out['tokens'])} new, "
              f"{out['tokens_per_second']:.0f} tokens/s, {'open' if state['open'] else 'closed'}-book)")

    if args.question:
        for q in args.question:
            if not args.json:
                print(f"> {q}")
            ask(q)
        return

    print(f"Brasa ({model.num_params():,} parameters). Ask in Portuguese or English; /quit to leave.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            return
        if line in ("/open", "/closed"):
            state["open"] = line == "/open" and retriever is not None
            print(f"    mode: {'open' if state['open'] else 'closed'}-book")
        elif line == "/context":
            state["context"] = not state["context"]
            print(f"    retrieved facts: {'shown' if state['context'] else 'hidden'}")
        else:
            ask(line)


if __name__ == "__main__":
    main()
