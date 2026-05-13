"""
V24 — SCAN compositional-generalisation benchmark.

SCAN (Lake & Baroni 2018) tests whether a model can generalise to
longer/novel commands than seen in training. Symbolic systems hit 99%+;
transformers historically collapse to <20% on the length-split.

Our library does byte-level next-byte prediction, so we frame SCAN as:
  - train: byte stream of `IN: <command> OUT: <action sequence>\\n`
  - test:  score `log P(gold_target | input)` byte-by-byte against
           K random-shuffled distractors. Rank = % where gold scores
           higher than all distractors.

This is NOT the canonical "exact-match accuracy" metric (which requires
generation, and our generator is incoherent per V20). It's a calibration
metric: does the trained library assign higher probability to the *right*
action sequence than to scrambled alternatives? If yes, the architecture
has learned the SCAN mapping in a measurable way.

Datasets fetched into ./data/scan/ from the canonical GitHub repo.

Run:
    python eval_scan.py --split length --train --bayes-train 30000 \\
                         --max-test 200 --save-path .rce_library_v24_scan.pkl
    python eval_scan.py --split length --lib .rce_library_v24_scan.pkl \\
                         --max-test 200
"""
from __future__ import annotations
import argparse
import math
import random
import time
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent
SCAN_DIR = REPO / "data" / "scan"


def parse_scan_line(line: str) -> tuple[str, str] | None:
    if "OUT:" not in line:
        return None
    head, _, tail = line.partition("OUT:")
    cmd = head.replace("IN:", "").strip()
    act = tail.strip()
    if not cmd or not act:
        return None
    return cmd, act


def load_split(split: str, subset: str) -> list[tuple[str, str]]:
    path = SCAN_DIR / f"{split}_{subset}.txt"
    if not path.exists():
        raise SystemExit(f"missing {path}; fetch with the curl commands in LOG.md V24 setup")
    out: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        ex = parse_scan_line(line)
        if ex is not None:
            out.append(ex)
    return out


def encode_example(cmd: str, act: str) -> bytes:
    return f"IN: {cmd} OUT: {act}\n".encode("utf-8", errors="replace")


def score_target(lib, input_prefix: bytes, target: bytes,
                 ctx_window: int = 16) -> float:
    """Sum of byte-level log2 P(b | prefix...preceding) over target bytes.
    More positive (less negative) = higher probability."""
    total = 0.0
    cur = bytearray(input_prefix)
    for b in target:
        ctx = bytes(cur[-ctx_window:])
        dist = lib.predict(ctx)
        tot = sum(dist.values()) or 1.0
        p = max(dist.get(b, 0.0) / tot, 1e-12)
        total += math.log2(p)
        cur.append(b)
    return total


def shuffle_actions(act: str, rng: random.Random) -> str:
    toks = act.split()
    if len(toks) <= 1:
        return act
    shuffled = toks[:]
    rng.shuffle(shuffled)
    return " ".join(shuffled)


def evaluate(lib, examples: list[tuple[str, str]], n_distractors: int = 4,
             max_test: int = 200, seed: int = 0) -> dict:
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    test_idx = indices[:max_test]

    gold_scores_per_byte: list[float] = []
    gold_rank1 = 0
    n_scored = 0
    target_lens: list[int] = []
    for i in test_idx:
        cmd, act = examples[i]
        input_prefix = f"IN: {cmd} OUT: ".encode("utf-8", errors="replace")
        gold = (act + "\n").encode("utf-8", errors="replace")
        gold_logp = score_target(lib, input_prefix, gold)
        gold_per_byte = gold_logp / max(len(gold), 1)
        gold_scores_per_byte.append(gold_per_byte)
        target_lens.append(len(gold))

        # K distractors: shuffled-action sequences
        best_distractor = float("-inf")
        for _ in range(n_distractors):
            d_act = shuffle_actions(act, rng)
            if d_act == act:
                continue
            d_bytes = (d_act + "\n").encode("utf-8", errors="replace")
            d_logp = score_target(lib, input_prefix, d_bytes)
            d_per_byte = d_logp / max(len(d_bytes), 1)
            if d_per_byte > best_distractor:
                best_distractor = d_per_byte
        if gold_per_byte > best_distractor:
            gold_rank1 += 1
        n_scored += 1

    return {
        "n_examples": n_scored,
        "n_distractors": n_distractors,
        "rank1_accuracy": round(gold_rank1 / max(n_scored, 1), 4),
        "random_baseline_rank1": round(1.0 / (n_distractors + 1), 4),
        "mean_gold_log2p_per_byte": round(sum(gold_scores_per_byte) / max(n_scored, 1), 4),
        "uniform_baseline_log2p_per_byte": round(-math.log2(256), 4),  # = -8.0
        "mean_target_bytes": round(sum(target_lens) / max(n_scored, 1), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["length", "addprim"], default="length")
    ap.add_argument("--train", action="store_true",
                    help="train a fresh library on SCAN train split (else load --lib)")
    ap.add_argument("--lib", default=None)
    ap.add_argument("--save-path", default=None)
    ap.add_argument("--bayes-train", type=int, default=30_000)
    ap.add_argument("--train-bytes", type=int, default=200_000,
                    help="cap on training-byte stream length")
    ap.add_argument("--max-test", type=int, default=200)
    ap.add_argument("--n-distractors", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_exs = load_split(args.split, "train")
    test_exs = load_split(args.split, "test")
    print(f"SCAN {args.split}: train={len(train_exs)} test={len(test_exs)}")
    sample = train_exs[0]
    print(f"sample train: {sample[0]!r} -> {sample[1][:60]!r}{'...' if len(sample[1])>60 else ''}")

    if args.train or args.lib is None:
        train_buf = bytearray()
        for cmd, act in train_exs:
            train_buf.extend(encode_example(cmd, act))
            if len(train_buf) >= args.train_bytes:
                break
        train_bytes = bytes(train_buf[:args.train_bytes])
        print(f"training on {len(train_bytes):,} bytes...")

        from engine import fresh_library, save_library
        from bench import fit_primitives, bayes_train
        lib = fresh_library()
        t = time.time()
        fit_primitives(lib, train_bytes)
        print(f"  fit primitives: {time.time()-t:.1f}s")
        t = time.time()
        bayes_train(lib, train_bytes[:args.bayes_train],
                    decay_every_steps=500, decay_factor=0.99,
                    replay_buffer_size=2000, replay_sample=128,
                    temperature=64.0, max_delta=0.3,  # V22 best params
                    progress_every=2000)
        print(f"  bayes-train: {time.time()-t:.1f}s; lib={len(lib.programs)}")
        save_path = args.save_path or f".rce_library_v24_scan_{args.split}.pkl"
        save_library(lib, save_path)
        print(f"  saved {save_path}")
    else:
        from engine import load_library
        lib = load_library(args.lib)
        if lib is None:
            raise SystemExit(f"could not load {args.lib}")
        print(f"loaded library: {len(lib.programs)} programs")

    print(f"\nEvaluating on {args.max_test} test examples with {args.n_distractors} distractors each...")
    t = time.time()
    res = evaluate(lib, test_exs, n_distractors=args.n_distractors,
                   max_test=args.max_test, seed=args.seed)
    print(f"  done in {time.time()-t:.1f}s")
    import json
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
