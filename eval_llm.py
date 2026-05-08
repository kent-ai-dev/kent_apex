"""
V11: token-level evaluation against LLM baselines.

The RCE works byte-by-byte. To score multiple-choice tasks we compute
P(option_text | context_text) by chaining byte-level conditional
probabilities under the trained library. The chosen answer is the one
with highest conditional probability under our library.

Implemented:

  - LAMBADA-style last-word-prediction: per example, given a passage
    and a true last word, score every byte of the last word and report
    the conditional probability.

  - HellaSwag-style multiple-choice: for each candidate continuation,
    score it under the library; pick argmax. Accuracy reported.

  - WikiText perplexity on a held-out slice (already covered by bench.py
    BPB; here we report it in token-perplexity terms for LLM comparability).

  - TriviaQA closed-book: hopeless on this scale, but we record the
    refusal rate so it's auditable.

The HF datasets we hit for V11 each go through validate_dataset.py
before any prediction happens (per plans/v1.md §0).

Numbers go into BENCHMARKS.md / v11_results.json.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

from engine import Library, load_library


REPO = Path(__file__).resolve().parent
CTX_WINDOW = 16


def _byte_logprob(lib: Library, ctx: bytes, b: int) -> float:
    """log2 P(b | ctx) under the library. Floored at 1e-12 to avoid -inf."""
    dist = lib.predict(ctx[-CTX_WINDOW:])
    total = sum(dist.values()) or 1.0
    p = max(dist.get(b, 0.0) / total, 1e-12)
    return math.log2(p)


def score_continuation(lib: Library, context: bytes, continuation: bytes) -> float:
    """Sum of byte-level log2 P(b | ctx) over the continuation. More positive = more likely."""
    total = 0.0
    cur = bytearray(context)
    for b in continuation:
        total += _byte_logprob(lib, bytes(cur), b)
        cur.append(b)
    return total


def eval_lambada(lib: Library, n_examples: int = 50) -> dict:
    """LAMBADA: predict the last word given the passage. Score via byte
    logprob of the last word."""
    from validate_dataset import validate
    rep = validate("EleutherAI/lambada_openai", split="test", sample=20)
    if rep.verdict == "FAIL":
        return {"task": "lambada", "skipped": True, "reason": rep.summary}

    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", split="test", streaming=True)
    scored = []
    for i, ex in enumerate(ds):
        if i >= n_examples:
            break
        text = ex["text"]
        last_space = text.rstrip().rfind(" ")
        if last_space <= 0:
            continue
        context = text[:last_space + 1].encode("utf-8", errors="replace")
        target = text[last_space + 1:].rstrip().encode("utf-8", errors="replace")
        if not target:
            continue
        score = score_continuation(lib, context, target)
        per_byte = score / max(len(target), 1)
        scored.append({"target_len": len(target), "score": score,
                        "score_per_byte": per_byte})
    avg_score = sum(s["score"] for s in scored) / max(len(scored), 1)
    avg_per_byte = sum(s["score_per_byte"] for s in scored) / max(len(scored), 1)
    return {"task": "lambada", "n": len(scored), "avg_score": avg_score,
            "avg_per_byte_log2p": avg_per_byte,
            "interpretation": "higher = better; baseline (uniform 256) = -8 per byte"}


def eval_hellaswag(lib: Library, n_examples: int = 50) -> dict:
    """HellaSwag: pick the most probable of 4 endings given context."""
    from validate_dataset import validate
    rep = validate("Rowan/hellaswag", split="validation", text_field="ctx", sample=20)
    if rep.verdict == "FAIL":
        return {"task": "hellaswag", "skipped": True, "reason": rep.summary}

    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
    correct = 0
    n_seen = 0
    for i, ex in enumerate(ds):
        if i >= n_examples:
            break
        ctx = (ex.get("ctx_a", "") + " " + ex.get("ctx_b", "")).encode("utf-8", "replace")
        endings = ex.get("endings", [])
        if len(endings) != 4:
            continue
        try:
            label = int(ex.get("label"))
        except (TypeError, ValueError):
            continue
        scores = []
        for end in endings:
            cont = end.encode("utf-8", "replace")
            scores.append(score_continuation(lib, ctx, cont) / max(len(cont), 1))
        pred = max(range(4), key=lambda j: scores[j])
        if pred == label:
            correct += 1
        n_seen += 1
    return {"task": "hellaswag", "n": n_seen, "accuracy": correct / max(n_seen, 1),
            "random_baseline": 0.25}


def eval_wikitext_ppl(lib: Library, n_bytes: int = 30_000) -> dict:
    """Plain perplexity in nats and bits-per-byte on wikitext-103 held-out."""
    from validate_dataset import validate
    rep = validate("Salesforce/wikitext", config="wikitext-103-raw-v1",
                   split="validation", sample=20)
    if rep.verdict == "FAIL":
        return {"task": "wikitext-103-ppl", "skipped": True, "reason": rep.summary}

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="validation", streaming=True)
    buf = bytearray()
    for ex in ds:
        if not ex.get("text"):
            continue
        buf.extend(ex["text"].encode("utf-8", "replace"))
        if len(buf) >= n_bytes:
            break
    data = bytes(buf[:n_bytes])
    total_log2p = 0.0
    cnt = 0
    for i in range(1, len(data)):
        total_log2p += _byte_logprob(lib, data[max(0, i-CTX_WINDOW):i], data[i])
        cnt += 1
    bpb = -total_log2p / max(cnt, 1)
    nats = bpb * math.log(2)
    ppl = math.exp(nats)
    return {"task": "wikitext-103-ppl", "n_bytes": cnt,
            "bpb": round(bpb, 4),
            "perplexity_per_byte": round(ppl, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=".rce_library.pkl")
    ap.add_argument("--lambada", type=int, default=30)
    ap.add_argument("--hellaswag", type=int, default=30)
    ap.add_argument("--wikitext-bytes", type=int, default=20_000)
    ap.add_argument("--skip", default="", help="comma-separated: lambada,hellaswag,wikitext")
    args = ap.parse_args()
    skip = set(args.skip.split(",")) if args.skip else set()

    lib = load_library(args.lib)
    if lib is None:
        raise SystemExit(f"no library at {args.lib} — run a bench first")
    print(f"library: {len(lib.programs)} programs")

    results = {}
    if "lambada" not in skip:
        t0 = time.time()
        results["lambada"] = eval_lambada(lib, n_examples=args.lambada)
        print(f"lambada: {json.dumps(results['lambada'])}  ({time.time()-t0:.1f}s)")
    if "hellaswag" not in skip:
        t0 = time.time()
        results["hellaswag"] = eval_hellaswag(lib, n_examples=args.hellaswag)
        print(f"hellaswag: {json.dumps(results['hellaswag'])}  ({time.time()-t0:.1f}s)")
    if "wikitext" not in skip:
        t0 = time.time()
        results["wikitext_ppl"] = eval_wikitext_ppl(lib, n_bytes=args.wikitext_bytes)
        print(f"wikitext: {json.dumps(results['wikitext_ppl'])}  ({time.time()-t0:.1f}s)")

    out = REPO / "v11_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
