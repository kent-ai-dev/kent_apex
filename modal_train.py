"""
V8: Modal-based distributed training for the RCE.

Architecture (per plans/storage.md Tier 1):
  - N workers, each holds a copy of the program library
  - Each worker trains on its own corpus shard
  - Master periodically merges weights (sum of log-weights, prior-corrected)
  - Programs (combinator structures) deduplicated by name on merge
  - Final library is union of programs with merged posterior

Merge math (this is the spec's correction — without it the merge
double-counts the Solomonoff prior):
    For program P appearing in N_p worker libraries:
        merged_log_weight(P) = sum_i(worker_i_log_weight(P))
                                 - (N_p - 1) * (-P.length * ln 2)
        # the prior is encoded in the seed log-weight and gets summed
        # alongside the data-driven updates; subtract (N_p - 1) copies
        # so it remains the prior of one library, not N.

Run:
    modal token set-id ak-... --token-secret as-...
    modal run modal_train.py::run_distributed --shards 4 --bytes-per-shard 100000

Local dry-run (no Modal — exercises the merge math only):
    python modal_train.py --dry-run --shards 4 --bytes-per-shard 30000

This file is a structural placeholder for V8: the actual cloud run is gated
on an explicit human go-ahead because of Modal compute spend. The dry-run
path validates the merge logic without spending cloud minutes.
"""
from __future__ import annotations
import argparse
import math
import os
import pickle
import time
from pathlib import Path

# Modal is optional at import-time so the dry-run path works without it.
try:
    import modal
except ImportError:
    modal = None


REPO = Path(__file__).resolve().parent

# ---------- Modal app (only realised if modal is importable) ----------

if modal is not None:
    app = modal.App("rce-train")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("datasets", "tokenizers", "huggingface-hub")
        .add_local_dir(str(REPO), remote_path="/repo", copy=True)
    )
    volume = modal.Volume.from_name("rce-libraries", create_if_missing=True)

    @app.function(image=image, cpu=4, memory=8192,
                  volumes={"/lib": volume}, timeout=3600)
    def worker_train(shard_id: int, shard_bytes: bytes,
                     bayes_train_bytes: int = 50_000) -> bytes:
        """Train one shard. Returns the pickled library log_weights."""
        import sys
        sys.path.insert(0, "/repo")
        from engine import fresh_library
        from bench import fit_primitives, bayes_train

        lib = fresh_library()
        fit_primitives(lib, shard_bytes)
        bayes_train(lib, shard_bytes[:bayes_train_bytes],
                    decay_every_steps=500, replay_buffer_size=2000)

        return pickle.dumps({
            "shard_id": shard_id,
            "log_weights": lib.log_weights,
            "lengths": {p.name: p.length for p in lib.programs},
        })


# ---------- merge logic (pure, runs anywhere) ----------

def merge_libraries(shard_payloads: list[dict]) -> dict:
    """Merge N worker shards into one library's log_weights.

    Returns merged log_weights dict, and the names of programs that
    appeared in ≥2 shards (for diagnostics).
    """
    LN2 = math.log(2.0)
    # tally appearances and accumulate weights
    sum_log_weights: dict[str, float] = {}
    appearances: dict[str, int] = {}
    lengths: dict[str, float] = {}
    for payload in shard_payloads:
        for name, w in payload["log_weights"].items():
            sum_log_weights[name] = sum_log_weights.get(name, 0.0) + w
            appearances[name] = appearances.get(name, 0) + 1
            if name not in lengths:
                lengths[name] = payload["lengths"][name]

    # apply the prior-correction
    merged = {}
    for name, w in sum_log_weights.items():
        N = appearances[name]
        prior = -lengths[name] * LN2
        merged[name] = w - (N - 1) * prior

    # renormalize so max weight is 0 (numerical stability)
    if merged:
        m = max(merged.values())
        merged = {k: v - m for k, v in merged.items()}

    return {
        "log_weights": merged,
        "appearances": appearances,
        "n_programs": len(merged),
        "n_shared": sum(1 for c in appearances.values() if c >= 2),
    }


# ---------- dry-run path (no Modal needed) ----------

def dry_run(shards: int, bytes_per_shard: int):
    """Run N shards locally in series (single-process), then merge.
    Validates the merge math without spending Modal compute.
    """
    from engine import fresh_library
    from bench import fit_primitives, bayes_train

    train_bytes = (REPO / "data" / "wikitext2_train.txt").read_bytes()
    if len(train_bytes) < shards * bytes_per_shard:
        raise SystemExit(
            f"need at least {shards * bytes_per_shard:,} bytes; "
            f"data/wikitext2_train.txt has {len(train_bytes):,}"
        )

    payloads = []
    for i in range(shards):
        offset = i * bytes_per_shard
        shard = train_bytes[offset:offset + bytes_per_shard]
        print(f"\n--- shard {i}: bytes [{offset:,}, {offset+len(shard):,}) ---")
        lib = fresh_library()
        t0 = time.time()
        fit_primitives(lib, shard)
        # short bayes-train on a fraction of the shard
        bayes_train(lib, shard[:max(bytes_per_shard // 2, 5000)],
                    decay_every_steps=500, replay_buffer_size=2000,
                    progress_every=2000)
        print(f"  shard {i} trained in {time.time()-t0:.1f}s; lib={len(lib.programs)}")
        payloads.append({
            "shard_id": i,
            "log_weights": lib.log_weights,
            "lengths": {p.name: p.length for p in lib.programs},
        })

    print(f"\n=== merging {shards} shards ===")
    merged = merge_libraries(payloads)
    print(f"  merged: {merged['n_programs']} unique programs; "
          f"{merged['n_shared']} appeared in ≥2 shards")
    out = REPO / "v8_merge_result.json"
    import json
    out.write_text(json.dumps({
        "n_programs": merged["n_programs"],
        "n_shared": merged["n_shared"],
        "top_shared": sorted(
            ((n, c) for n, c in merged["appearances"].items() if c >= 2),
            key=lambda x: -x[1])[:20],
    }, indent=2))
    print(f"  wrote {out}")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--bytes-per-shard", type=int, default=30_000)
    ap.add_argument("--dry-run", action="store_true",
                    help="run shards locally; no Modal calls")
    args = ap.parse_args()

    if args.dry_run or modal is None:
        if modal is None and not args.dry_run:
            print("modal not installed; falling back to --dry-run")
        dry_run(args.shards, args.bytes_per_shard)
        return

    # real Modal run path — left for explicit human-driven invocation
    raise SystemExit(
        "Real Modal run requires `modal run modal_train.py::worker_train`; "
        "this entry point only supports --dry-run for safety."
    )


if __name__ == "__main__":
    main()
