"""
RCE benchmark harness — V1.

Three metrics, all computed against the *current* library by feeding it
byte-by-byte and asking for the next-byte distribution:

  1. BPB    — bits per byte: -mean(log2(P(actual_byte | ctx)))
  2. ECE    — expected calibration error: bin predictions by stated
              top-prediction probability, measure accuracy in each bin,
              compute weighted L1 between confidence and accuracy.
  3. REFUSE — refusal rate on out-of-distribution input: fraction of
              positions where library entropy > 7.0 bits.

Datasets live under ./data/ and are fetched once via fetch_data.py.

Usage:
    python bench.py                  # run all three on the held-out slice
    python bench.py --train          # fit n-grams on the train slice first
    python bench.py --metric bpb     # one metric only

Determinism: this script seeds Python's `random`. The library itself is
deterministic given a fixed input, so two runs on the same library and
slice should be byte-identical. The V1 gate ("reproducible within ±2%")
is therefore really a check that the data load and the eval loop are
order-stable.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

from engine import (
    Library, NGramPrimitive,
    fresh_library, load_library, save_library,
)


REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
MODEL_PATH = REPO / ".rce_library.pkl"

WIKITEXT_TRAIN = DATA / "wikitext2_train.txt"
WIKITEXT_HELDOUT = DATA / "wikitext2_heldout.txt"
OOD_BYTES = DATA / "ood_random.bin"

CTX_WINDOW = 16
HELDOUT_BUDGET_BYTES = 100_000   # V1 spec: 100KB held-out
OOD_BUDGET_BYTES = 20_000


def _ensure_data():
    missing = [p for p in (WIKITEXT_TRAIN, WIKITEXT_HELDOUT, OOD_BYTES) if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing data files: {missing}\n"
            f"run: python fetch_data.py"
        )


def _byte_dist_to_probs(dist) -> list[float]:
    """Convert a Counter-like {byte: weight} into a length-256 probability list."""
    total = sum(dist.values()) or 1.0
    return [(dist.get(b, 0.0) + 1e-9) / (total + 256e-9) for b in range(256)]


def bpb(lib: Library, data: bytes, ctx_window: int = CTX_WINDOW) -> float:
    """Bits per byte over `data`. Predicts byte i from bytes [i-ctx_window:i]."""
    if len(data) < 2:
        return float("nan")
    log_loss = 0.0
    n = 0
    for i in range(1, len(data)):
        ctx = data[max(0, i - ctx_window):i]
        actual = data[i]
        probs = _byte_dist_to_probs(lib.predict(ctx))
        p = max(probs[actual], 1e-12)
        log_loss += -math.log2(p)
        n += 1
    return log_loss / max(n, 1)


def ece(lib: Library, data: bytes, n_bins: int = 10,
        ctx_window: int = CTX_WINDOW) -> tuple[float, list[dict]]:
    """Expected Calibration Error. Bin by top-prediction confidence; per bin
    measure (a) mean stated confidence and (b) actual accuracy of the top pick.
    Return weighted-mean |conf - acc| and the per-bin breakdown."""
    bins: list[dict] = [{"n": 0, "conf_sum": 0.0, "correct": 0} for _ in range(n_bins)]
    for i in range(1, len(data)):
        ctx = data[max(0, i - ctx_window):i]
        actual = data[i]
        probs = _byte_dist_to_probs(lib.predict(ctx))
        top_b = max(range(256), key=lambda b: probs[b])
        top_p = probs[top_b]
        idx = min(int(top_p * n_bins), n_bins - 1)
        bins[idx]["n"] += 1
        bins[idx]["conf_sum"] += top_p
        if top_b == actual:
            bins[idx]["correct"] += 1
    total = sum(b["n"] for b in bins) or 1
    weighted_err = 0.0
    breakdown = []
    for k, b in enumerate(bins):
        if b["n"] == 0:
            breakdown.append({"bin": k, "n": 0, "conf": None, "acc": None})
            continue
        conf = b["conf_sum"] / b["n"]
        acc = b["correct"] / b["n"]
        weighted_err += (b["n"] / total) * abs(conf - acc)
        breakdown.append({"bin": k, "n": b["n"],
                          "conf": round(conf, 4), "acc": round(acc, 4)})
    return weighted_err, breakdown


def refusal_rate(lib: Library, data: bytes,
                 entropy_thresh: float = 7.0,
                 ctx_window: int = CTX_WINDOW) -> float:
    """Fraction of positions where the library's entropy on the prefix
    exceeds the threshold (i.e., the library would refuse to answer)."""
    refused = 0
    n = 0
    for i in range(1, len(data)):
        ctx = data[max(0, i - ctx_window):i]
        h = lib.entropy(ctx)
        n += 1
        if h > entropy_thresh:
            refused += 1
    return refused / max(n, 1)


def fit_primitives(lib: Library, train: bytes):
    """Fit any primitive that exposes a fit() method on the train bytes.
    NGramPrimitive, KneserNeyNGram, and SkipGramPredictor all qualify."""
    for p in lib.programs:
        fit = getattr(p, "fit", None)
        if callable(fit):
            try:
                fit(train)
            except TypeError:
                # programs without a (data,) fit signature — skip
                pass


def bayes_train(lib: Library, train: bytes, ctx_window: int = CTX_WINDOW,
                update_every: int = 4, grow_every_steps: int = 200,
                max_lib_size: int = 150, n_children: int = 4,
                progress_every: int | None = None,
                abstract_every_grows: int = 5,
                decay_every_steps: int = 0, decay_factor: float = 0.99,
                replay_buffer_size: int = 0,
                replay_every_grows: int = 1, replay_sample: int = 256) -> int:
    """Run the Bayesian-update + grow + prune + abstract phase.

    `abstract_every_grows`: every Nth grow phase, also run
    `lib.abstract_phase()` to lift recurring sub-programs into Memoized
    primitives (V5 wake-sleep step). Set to 0 to disable.
    """
    import random as _r
    step = 0
    grow_phase = 0
    abstractions: list[str] = []
    replay_buffer: list[tuple[bytes, int]] = []
    for i in range(1, len(train)):
        if i % update_every != 0:
            continue
        ctx = train[max(0, i - ctx_window):i]
        actual = train[i]
        lib.update(ctx, actual)
        step += 1

        # V7: maintain a reservoir-style replay buffer
        if replay_buffer_size > 0:
            if len(replay_buffer) < replay_buffer_size:
                replay_buffer.append((ctx, actual))
            else:
                # uniform-random replacement preserves a uniform sample of past
                j = _r.randint(0, step - 1)
                if j < replay_buffer_size:
                    replay_buffer[j] = (ctx, actual)

        # V7: posterior decay every N steps
        if decay_every_steps > 0 and step % decay_every_steps == 0:
            lib.decay(factor=decay_factor)

        if step % grow_every_steps == 0:
            lib.grow(n_children=n_children)
            grow_phase += 1
            # V5 abstraction
            if abstract_every_grows and grow_phase % abstract_every_grows == 0:
                lifted = lib.abstract_phase(scan_top=50, min_count=3, max_lift=2)
                abstractions.extend(p.name for p in lifted)
            # V7 replay
            if replay_buffer and replay_every_grows and grow_phase % replay_every_grows == 0:
                sample = _r.sample(replay_buffer,
                                    min(replay_sample, len(replay_buffer)))
                lib.replay(sample)
            lib.prune(max_size=max_lib_size)
            if progress_every and step % progress_every == 0:
                print(f"  bayes-train step {step}: |lib|={len(lib.programs)}"
                      + (f"  +abstract={len(abstractions)}" if abstractions else "")
                      + (f"  buf={len(replay_buffer)}" if replay_buffer else ""))
    lib.grow(n_children=n_children * 2)
    if abstract_every_grows:
        lib.abstract_phase(scan_top=50, min_count=3, max_lift=3)
    lib.prune(max_size=max_lib_size)
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true",
                    help="fit n-grams on wikitext2_train.txt before evaluating")
    ap.add_argument("--bayes-train", type=int, default=0, metavar="BYTES",
                    help="after fitting n-grams, run Bayesian-update + grow/prune "
                         "over the first BYTES of train data (mirrors rce.py train)")
    ap.add_argument("--decay-every", type=int, default=0,
                    help="V7: shrink log-weights by --decay-factor every N steps")
    ap.add_argument("--decay-factor", type=float, default=0.99,
                    help="multiplicative decay on log-weights (default 0.99)")
    ap.add_argument("--replay-buffer", type=int, default=0,
                    help="V7: maintain a reservoir replay buffer of N past (ctx,actual)")
    ap.add_argument("--replay-sample", type=int, default=256,
                    help="V7: sample size from the replay buffer per grow phase")
    ap.add_argument("--metric", choices=["bpb", "ece", "refuse", "all"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true",
                    help="persist the fitted library to .rce_library.pkl")
    args = ap.parse_args()

    random.seed(args.seed)
    _ensure_data()

    if args.train or args.bayes_train > 0 or not MODEL_PATH.exists():
        lib = fresh_library()
        train_bytes = WIKITEXT_TRAIN.read_bytes()
        print(f"fitting primitives on {len(train_bytes):,} train bytes...")
        fit_primitives(lib, train_bytes)
        if args.bayes_train > 0:
            slice_bytes = train_bytes[:args.bayes_train]
            print(f"bayes-train over {len(slice_bytes):,} bytes "
                  f"(decay_every={args.decay_every}, replay_buf={args.replay_buffer})...")
            steps = bayes_train(
                lib, slice_bytes, progress_every=2000,
                decay_every_steps=args.decay_every,
                decay_factor=args.decay_factor,
                replay_buffer_size=args.replay_buffer,
                replay_sample=args.replay_sample,
            )
            print(f"  done: {steps} update steps; |lib|={len(lib.programs)}")
        if args.save:
            save_library(lib, str(MODEL_PATH))
    else:
        lib = load_library(str(MODEL_PATH)) or fresh_library()

    print(f"library: {len(lib.programs)} programs")

    heldout = WIKITEXT_HELDOUT.read_bytes()[:HELDOUT_BUDGET_BYTES]
    ood = OOD_BYTES.read_bytes()[:OOD_BUDGET_BYTES]
    print(f"heldout: {len(heldout):,} bytes; ood: {len(ood):,} bytes")

    results: dict = {}
    if args.metric in ("bpb", "all"):
        v = bpb(lib, heldout)
        results["bpb"] = round(v, 4)
        print(f"  BPB  (wikitext-2 heldout) = {v:.4f}")
    if args.metric in ("ece", "all"):
        v, breakdown = ece(lib, heldout)
        results["ece"] = round(v, 4)
        results["ece_breakdown"] = breakdown
        print(f"  ECE  (wikitext-2 heldout) = {v:.4f}")
    if args.metric in ("refuse", "all"):
        v = refusal_rate(lib, ood)
        results["refusal_rate_ood"] = round(v, 4)
        print(f"  REFUSE (ood random)       = {v:.4f}")

    print(json.dumps({k: v for k, v in results.items() if k != "ece_breakdown"}))
    return results


if __name__ == "__main__":
    main()
