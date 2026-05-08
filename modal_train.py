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
        .pip_install("datasets", "tokenizers", "huggingface-hub", "zstandard")
        .add_local_dir(str(REPO), remote_path="/repo", copy=True)
    )
    volume = modal.Volume.from_name("rce-libraries", create_if_missing=True)

    @app.function(image=image, cpu=4, memory=8192,
                  volumes={"/vol": volume}, timeout=3600)
    def worker_train(shard_id: int, shard_bytes: bytes,
                     bayes_train_bytes: int = 50_000) -> bytes:
        """Train one shard. Returns the pickled library log_weights + lengths."""
        import sys
        sys.path.insert(0, "/repo")
        from engine import fresh_library
        from bench import fit_primitives, bayes_train

        lib = fresh_library()
        fit_primitives(lib, shard_bytes)
        bayes_train(lib, shard_bytes[:bayes_train_bytes],
                    decay_every_steps=500, decay_factor=0.99,
                    replay_buffer_size=2000, replay_sample=256)

        return pickle.dumps({
            "shard_id": shard_id,
            "log_weights": lib.log_weights,
            "lengths": {p.name: p.length for p in lib.programs},
            "n_programs": len(lib.programs),
        })

    @app.function(image=image, cpu=4, memory=16384,
                  volumes={"/vol": volume}, timeout=7200)
    def worker_train_hf(shard_id: int, dataset_name: str,
                        config: str | None,
                        skip_bytes: int, take_bytes: int,
                        bayes_train_bytes: int = 200_000) -> bytes:
        """V9: fetch a slice of an HF dataset INSIDE the worker (no 1GB wire
        transfer), train, return weights. Each worker pulls bytes
        [skip_bytes, skip_bytes+take_bytes) from the streaming dataset.
        """
        import sys
        sys.path.insert(0, "/repo")
        from engine import fresh_library
        from bench import fit_primitives, bayes_train
        from datasets import load_dataset

        ds = load_dataset(dataset_name, config, split="train", streaming=True)
        buf = bytearray()
        skipped = 0
        for row in ds:
            text_field = next((k for k in ("text", "content", "sentence")
                               if k in row), None)
            if text_field is None:
                continue
            v = row.get(text_field)
            if not v:
                continue
            b = str(v).encode("utf-8", errors="replace")
            if skipped + len(b) <= skip_bytes:
                skipped += len(b)
                continue
            # we may straddle the skip boundary on first useful chunk
            if skipped < skip_bytes:
                b = b[skip_bytes - skipped:]
                skipped = skip_bytes
            buf.extend(b)
            if len(buf) >= take_bytes:
                break
        shard_bytes = bytes(buf[:take_bytes])

        lib = fresh_library()
        fit_primitives(lib, shard_bytes)
        bayes_train(lib, shard_bytes[:bayes_train_bytes],
                    decay_every_steps=500, decay_factor=0.99,
                    replay_buffer_size=2000, replay_sample=256)

        return pickle.dumps({
            "shard_id": shard_id,
            "n_bytes_trained": len(shard_bytes),
            "log_weights": lib.log_weights,
            "lengths": {p.name: p.length for p in lib.programs},
            "n_programs": len(lib.programs),
        })

    @app.local_entrypoint()
    def run_v9(dataset: str = "monology/pile-uncopyrighted",
               config: str | None = None,
               shards: int = 16,
               bytes_per_shard: int = 30_000_000,
               bayes_train_bytes: int = 200_000,
               save_path: str = "v9_pile_merged.pkl"):
        """V9 first-serious-training-run orchestrator: each Modal worker
        pulls a non-overlapping slice of an HF streaming dataset (default
        pile-uncopyrighted), trains, and ships back log_weights. Master
        merges with prior-correction.

        Defaults: 16 × 30MB ≈ 480MB; bump --shards to 32 and/or
        --bytes-per-shard for the full 1GB target. v1.md V9 gate is
        BPB ≤ 2.5 on wikitext-103 held-out — measured locally after merge.
        """
        import time as _t
        shard_args = [
            (i, dataset, config, i * bytes_per_shard, bytes_per_shard,
             bayes_train_bytes)
            for i in range(shards)
        ]
        print(f"V9: {shards} workers × {bytes_per_shard:,} bytes each "
              f"({shards * bytes_per_shard:,} total) from {dataset}")
        t0 = _t.time()
        payload_blobs = list(worker_train_hf.starmap(shard_args))
        elapsed = _t.time() - t0
        print(f"all {shards} V9 workers done in {elapsed:.1f}s wall-clock")
        payloads = [pickle.loads(b) for b in payload_blobs]
        for p in payloads:
            print(f"  shard {p['shard_id']}: {p['n_bytes_trained']:,} bytes -> "
                  f"{p['n_programs']} programs")
        merged = merge_libraries(payloads)
        out = REPO / save_path
        with out.open("wb") as f:
            pickle.dump(merged, f)
        print(f"V9 merged: {merged['n_programs']} unique programs; "
              f"{merged['n_shared']} appeared in ≥2 shards")
        print(f"saved {out}")
        return merged

    @app.local_entrypoint()
    def run_distributed(shards: int = 4,
                        bytes_per_shard: int = 30000,
                        bayes_train_bytes: int = 50000,
                        save_path: str = "merged_library.pkl"):
        """V8 orchestrator: split local wikitext-2 train into N shards,
        ship to N parallel Modal workers via .map(), collect, merge, save.

        For V9-scale runs: increase --shards (e.g. 32) and --bytes-per-shard
        (e.g. 30_000_000 = 30MB; 32×30MB ≈ 1GB total) — same code path.
        """
        import time as _t
        train_bytes = (REPO / "data" / "wikitext2_train.txt").read_bytes()
        if len(train_bytes) < shards * bytes_per_shard:
            raise SystemExit(
                f"need ≥{shards * bytes_per_shard:,} bytes; "
                f"data/wikitext2_train.txt has {len(train_bytes):,}. "
                f"For V9-scale, swap to a streamed pile-uncopyrighted source."
            )
        shard_args = [
            (i, train_bytes[i * bytes_per_shard:(i + 1) * bytes_per_shard],
             bayes_train_bytes)
            for i in range(shards)
        ]
        print(f"shipping {shards} shards × {bytes_per_shard:,} bytes to Modal "
              f"(bayes_train_bytes={bayes_train_bytes:,} per worker)")

        t0 = _t.time()
        # starmap to expand the (shard_id, shard_bytes, bayes_train_bytes) tuple
        payload_blobs = list(worker_train.starmap(shard_args))
        elapsed = _t.time() - t0
        print(f"all {shards} workers done in {elapsed:.1f}s wall-clock")

        payloads = [pickle.loads(b) for b in payload_blobs]
        for p in payloads:
            print(f"  shard {p['shard_id']}: {p['n_programs']} programs")

        merged = merge_libraries(payloads)
        out = REPO / save_path
        with out.open("wb") as f:
            pickle.dump(merged, f)
        print(f"merged: {merged['n_programs']} unique programs; "
              f"{merged['n_shared']} appeared in ≥2 shards")
        print(f"saved {out}")
        return merged


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
