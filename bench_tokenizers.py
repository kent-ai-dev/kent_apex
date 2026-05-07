"""
V2 tokenization sweep — baseline n-gram BPB across {bytes, BPE-4K, word}.

The V2 measurement is *which tokenization gives the best bits-per-input-byte
under a fair n-gram baseline*. We are not yet adapting the full RCE engine
to multi-vocab predictors — that is V3 work. Here we just need a clean,
apples-to-apples comparison between tokenizations.

For each tokenization:
  1. train the tokenizer on wikitext-2 train (no-op for bytes)
  2. tokenize train + held-out
  3. fit a 5-gram model with add-k smoothing (k=0.01) on train tokens
  4. compute log-loss per token on held-out
  5. divide by *input byte count of held-out* to get BPB

Output: BPB per tokenizer + the resulting decision. The V2 gate is "pick
the tokenization with best BPB" — we record the winner in BENCHMARKS.md
and `state.json["tokenization"]`.

This script is independent of engine.py so V2 does not block on the
vocab-agnostic refactor. V3+ will use the winner.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from tokenize_rce import BytesTokenizer, BPETokenizer, WordTokenizer

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
WIKITEXT_TRAIN = DATA / "wikitext2_train.txt"
WIKITEXT_HELDOUT = DATA / "wikitext2_heldout.txt"
TOKEN_DIR = REPO / "tokenizers_trained"
TOKEN_DIR.mkdir(exist_ok=True)


# ---------- n-gram model with add-k smoothing ----------

class NGramModel:
    def __init__(self, n: int, vocab_size: int, k: float = 0.01):
        self.n = n
        self.vocab_size = vocab_size
        self.k = k
        # tables[m]: dict[tuple[int,...], Counter] for m-gram contexts (m = n-1, n-2, ..., 0)
        self.tables: list[dict] = [defaultdict(Counter) for _ in range(n)]

    def fit(self, tokens: list[int]):
        for i in range(len(tokens)):
            for m in range(self.n):
                if i < m:
                    continue
                ctx = tuple(tokens[i - m:i])  # m-token context
                self.tables[m][ctx][tokens[i]] += 1

    def logprob(self, ctx: tuple[int, ...], actual: int) -> float:
        """log2 P(actual | ctx). Backs off through shorter contexts on OOV."""
        # try longest available context first
        for m in range(min(len(ctx), self.n - 1), -1, -1):
            sub = ctx[-m:] if m > 0 else ()
            t = self.tables[m].get(sub)
            if t and sum(t.values()) > 0:
                count = t.get(actual, 0)
                total = sum(t.values())
                p = (count + self.k) / (total + self.k * self.vocab_size)
                return math.log2(p)
        # nothing seen: uniform
        return -math.log2(self.vocab_size)


def evaluate(name: str, tok, train_bytes: bytes, heldout_bytes: bytes,
             n: int = 5, k: float = 0.01) -> dict:
    print(f"\n--- {name} ---")
    t0 = time.time()
    tok.train(train_bytes)
    print(f"  trained tokenizer: vocab_size={tok.vocab_size}, t={time.time()-t0:.1f}s")

    train_ids = tok.encode(train_bytes)
    heldout_ids = tok.encode(heldout_bytes)
    print(f"  train tokens: {len(train_ids):,}; heldout tokens: {len(heldout_ids):,}")
    print(f"  bytes/token train: {len(train_bytes)/max(len(train_ids),1):.2f};  "
          f"heldout: {len(heldout_bytes)/max(len(heldout_ids),1):.2f}")

    t1 = time.time()
    model = NGramModel(n=n, vocab_size=tok.vocab_size, k=k)
    model.fit(train_ids)
    print(f"  fit n-gram-{n}: t={time.time()-t1:.1f}s")

    # evaluate
    t2 = time.time()
    log_loss = 0.0
    for i in range(1, len(heldout_ids)):
        ctx = tuple(heldout_ids[max(0, i - (n - 1)):i])
        actual = heldout_ids[i]
        log_loss += -model.logprob(ctx, actual)
    bits_per_token = log_loss / max(len(heldout_ids) - 1, 1)
    bpb = log_loss / len(heldout_bytes)
    print(f"  eval: t={time.time()-t2:.1f}s  bits/token={bits_per_token:.4f}  BPB={bpb:.4f}")

    return {
        "tokenizer": name,
        "vocab_size": tok.vocab_size,
        "n_train_tokens": len(train_ids),
        "n_heldout_tokens": len(heldout_ids),
        "bytes_per_token_heldout": round(len(heldout_bytes) / max(len(heldout_ids), 1), 4),
        "bits_per_token": round(bits_per_token, 4),
        "bpb": round(bpb, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--k", type=float, default=0.01)
    ap.add_argument("--bpe-vocab", type=int, default=4096)
    args = ap.parse_args()

    train_bytes = WIKITEXT_TRAIN.read_bytes()
    heldout_bytes = WIKITEXT_HELDOUT.read_bytes()
    print(f"train: {len(train_bytes):,} bytes; heldout: {len(heldout_bytes):,} bytes")
    print(f"n-gram order: {args.n};  add-k smoothing: k={args.k}")

    results = []
    results.append(evaluate("bytes", BytesTokenizer(),
                            train_bytes, heldout_bytes, n=args.n, k=args.k))
    results.append(evaluate("bpe", BPETokenizer(vocab_size=args.bpe_vocab),
                            train_bytes, heldout_bytes, n=args.n, k=args.k))
    results.append(evaluate("word", WordTokenizer(),
                            train_bytes, heldout_bytes, n=args.n, k=args.k))

    print("\n=== summary ===")
    print(f"{'tokenizer':10s}  {'vocab':>6s}  {'BPB':>8s}  {'bits/token':>11s}  bytes/token")
    for r in results:
        print(f"{r['tokenizer']:10s}  {r['vocab_size']:>6d}  "
              f"{r['bpb']:>8.4f}  {r['bits_per_token']:>11.4f}  "
              f"{r['bytes_per_token_heldout']:.2f}")

    winner = min(results, key=lambda r: r["bpb"])
    print(f"\nwinner: {winner['tokenizer']} (BPB={winner['bpb']})")

    # write summary file for downstream consumers
    out = REPO / "v2_tokenization_sweep.json"
    out.write_text(json.dumps({"results": results, "winner": winner}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
