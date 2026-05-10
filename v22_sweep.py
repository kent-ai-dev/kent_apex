"""
V22 temperature sweep — for each T in the sweep, train a fresh library
on the same training data, measure BPB on held-out + ESS + (V5
abstraction lifts during training).

Pick T that minimises BPB while keeping ESS ≥ 10 (the V22 gate).

Output: v22_sweep.json with per-T metrics + the chosen T.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
TRAIN = REPO / "data" / "wikitext2_train.txt"
HELDOUT = REPO / "data" / "wikitext2_heldout.txt"


def run_one(T: float, train_bytes: bytes, heldout_bytes: bytes,
            bayes_steps: int = 6000, max_delta: float | None = None) -> dict:
    from engine import fresh_library, save_library
    from bench import fit_primitives, bayes_train, bpb

    lib = fresh_library()
    fit_primitives(lib, train_bytes)
    abstractions_before = sum(1 for p in lib.programs
                              if p.name.startswith("abstracted("))
    bayes_train(lib, train_bytes[:bayes_steps], temperature=T,
                decay_every_steps=500, decay_factor=0.99,
                replay_buffer_size=2000, replay_sample=128,
                max_delta=max_delta,
                progress_every=2000)
    abstractions_after = sum(1 for p in lib.programs
                             if p.name.startswith("abstracted("))
    val_bpb = bpb(lib, heldout_bytes)
    ess = lib.effective_sample_size()
    # save the lib for the chosen T to keep momentum
    save_library(lib, str(REPO / f".rce_library_v22_T{T}.pkl"))
    return {
        "T": T,
        "bpb_heldout": round(val_bpb, 4),
        "ess": round(ess, 3),
        "n_programs": len(lib.programs),
        "abstractions_lifted": abstractions_after - abstractions_before,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, nargs="+",
                    default=[16.0, 32.0, 64.0, 128.0])
    ap.add_argument("--max-delta", type=float, default=None,
                    help="V22 augment: per-step delta cap (e.g. 0.3 nats)")
    ap.add_argument("--bayes-steps", type=int, default=6000)
    ap.add_argument("--heldout-bytes", type=int, default=30_000)
    ap.add_argument("--out", default="v22_sweep.json")
    args = ap.parse_args()

    train = TRAIN.read_bytes()
    heldout = HELDOUT.read_bytes()[:args.heldout_bytes]
    print(f"V22 sweep: T values = {args.T}")
    print(f"  bayes_steps = {args.bayes_steps}, heldout = {len(heldout):,} bytes")

    results = []
    for T in args.T:
        print(f"\n--- T = {T} ---")
        t0 = time.time()
        r = run_one(T, train, heldout, bayes_steps=args.bayes_steps,
                    max_delta=args.max_delta)
        r["wall_sec"] = round(time.time() - t0, 1)
        print(f"  BPB={r['bpb_heldout']}  ESS={r['ess']}  "
              f"|lib|={r['n_programs']}  abstract_lifts={r['abstractions_lifted']}  "
              f"({r['wall_sec']}s)")
        results.append(r)

    # pick T: lowest BPB AMONG those with ESS >= 10
    qualifying = [r for r in results if r["ess"] >= 10.0]
    if qualifying:
        chosen = min(qualifying, key=lambda r: r["bpb_heldout"])
        chosen_reason = "lowest BPB with ESS≥10"
    else:
        # gate fails — pick lowest BPB anyway and flag
        chosen = min(results, key=lambda r: r["bpb_heldout"])
        chosen_reason = f"NO T qualifies for ESS≥10 (max ESS = {max(r['ess'] for r in results):.2f})"

    out = {
        "results": results,
        "chosen_T": chosen["T"],
        "chosen_bpb": chosen["bpb_heldout"],
        "chosen_ess": chosen["ess"],
        "chosen_reason": chosen_reason,
    }
    print(f"\n=== chosen: T = {chosen['T']} ({chosen_reason}) ===")
    print(f"  BPB = {chosen['bpb_heldout']}, ESS = {chosen['ess']}")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
