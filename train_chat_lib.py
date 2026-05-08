"""
Focused training script for the chat library.

Trains a longer, higher-quality library than `python rce.py train` does
by default and saves to `.rce_library.pkl` so `python rce.py chat` picks
it up.  Skips the eval phase that makes `bench.py` slow.

Defaults are tuned to land under ~10 minutes wall clock on the dev
machine while producing a library with diversified posterior weight
(i.e., breaking the kn-5 mode collapse documented in V4 / addressed
in V7).

Usage:
    python3 train_chat_lib.py
    python3 train_chat_lib.py --bayes-train 50000
"""
from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

from engine import fresh_library, save_library
from bench import fit_primitives, bayes_train

REPO = Path(__file__).resolve().parent
TRAIN = REPO / "data" / "wikitext2_train.txt"
MODEL_PATH = REPO / ".rce_library.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bayes-train", type=int, default=30_000,
                    help="number of train bytes to run Bayesian update over")
    ap.add_argument("--fit-bytes", type=int, default=300_000,
                    help="bytes used to fit n-gram / KN / skip-gram primitives "
                         "(more = better stats but slower fit)")
    ap.add_argument("--decay-every", type=int, default=500,
                    help="V7 posterior decay step (0 = off; 500 breaks kn-5 collapse)")
    ap.add_argument("--decay-factor", type=float, default=0.97)
    ap.add_argument("--replay-buffer", type=int, default=2000,
                    help="V7 replay buffer size (0 = off)")
    ap.add_argument("--replay-sample", type=int, default=256)
    ap.add_argument("--max-lib-size", type=int, default=200)
    args = ap.parse_args()

    if not TRAIN.exists():
        raise SystemExit(f"missing {TRAIN}; run python3 fetch_data.py")

    t0 = time.time()
    lib = fresh_library()
    train_bytes = TRAIN.read_bytes()
    fit_slice = train_bytes[: args.fit_bytes]
    print(f"fitting primitives on {len(fit_slice):,} bytes...")
    fit_primitives(lib, fit_slice)
    print(f"  primitives fit in {time.time() - t0:.1f}s; |lib|={len(lib.programs)}")

    t1 = time.time()
    bayes_slice = train_bytes[: args.bayes_train]
    print(f"bayes-train over {len(bayes_slice):,} bytes "
          f"(decay_every={args.decay_every}, replay_buf={args.replay_buffer})...")
    steps = bayes_train(
        lib, bayes_slice,
        progress_every=2000,
        max_lib_size=args.max_lib_size,
        decay_every_steps=args.decay_every,
        decay_factor=args.decay_factor,
        replay_buffer_size=args.replay_buffer,
        replay_sample=args.replay_sample,
    )
    print(f"  done: {steps} update steps in {time.time() - t1:.1f}s; "
          f"|lib|={len(lib.programs)}")

    save_library(lib, str(MODEL_PATH))
    print(f"saved to {MODEL_PATH}")
    print()
    print("top 10 programs by posterior weight:")
    post = lib.posterior()
    for p in lib.top_programs(k=10):
        print(f"  {post.get(p.name, 0):.4f}  {p.name}")
    print()
    print(f"total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
