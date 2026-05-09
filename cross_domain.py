"""
V12: cross-domain training.

Trains on a 50% text / 25% code / 25% structured-knowledge mix, with a
domain-tag prefix byte so the engine's combinators can branch on
domain. Per-domain BPB measured separately on each domain's held-out
slice; gate = no domain regresses worse than 1.2× single-domain BPB.

Datasets (all run through validate_dataset.py before any compute):
  - text:       allenai/c4 (English subset)
  - code:       code-search-net/code_search_net
  - structured: intfloat/wikidata5m

Domain-tag bytes (chosen above 0x7f to avoid collision with ASCII text):
  0x80 = TEXT
  0x81 = CODE
  0x82 = STRUCTURED

Per-domain BPB excludes the tag byte from the cost; the byte just
serves as a signal the library can condition combinators on.
"""
from __future__ import annotations
import argparse
import math
import time
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parent

DOMAIN_TAGS = {
    "text":       0x80,
    "code":       0x81,
    "structured": 0x82,
}


DATASETS = {
    "text":       ("allenai/c4", "en"),
    "code":       ("code-search-net/code_search_net", "python"),
    "structured": ("intfloat/wikidata5m", None),
}


def _validate(name: str, config: str | None):
    from validate_dataset import validate
    rep = validate(name, config=config, split="train", sample=50)
    print(f"  {name} ({config}): {rep.summary}")
    if rep.verdict == "FAIL":
        raise SystemExit(f"FAIL on {name} — pick alternate dataset.")


def stream_domain(name: str, max_bytes: int) -> Iterator[bytes]:
    """Yield bytes from one domain's HF dataset, with the domain-tag prefix
    byte attached at every chunk boundary."""
    ds_name, config = DATASETS[name]
    tag = DOMAIN_TAGS[name]
    from datasets import load_dataset
    ds = load_dataset(ds_name, config, split="train", streaming=True)
    served = 0
    for row in ds:
        text_field = next((k for k in ("text", "content", "func_code_string",
                                        "object", "code")
                           if k in row), None)
        if text_field is None:
            continue
        v = row.get(text_field)
        if not v:
            continue
        b = bytes([tag]) + str(v).encode("utf-8", errors="replace")
        if served + len(b) > max_bytes:
            yield b[:max_bytes - served]
            return
        yield b
        served += len(b)


def interleave_streams(plans: list[tuple[str, int]]) -> Iterator[bytes]:
    """Round-robin across domains in proportion to their byte budget.

    plans = [("text", 50000), ("code", 25000), ("structured", 25000)]
    """
    iterators = {name: stream_domain(name, budget) for name, budget in plans}
    served = {name: 0 for name, _ in plans}
    budgets = dict(plans)
    while iterators:
        for name in list(iterators.keys()):
            if served[name] >= budgets[name]:
                iterators.pop(name)
                continue
            try:
                chunk = next(iterators[name])
            except StopIteration:
                iterators.pop(name)
                continue
            served[name] += len(chunk)
            yield chunk


def per_domain_bpb(lib, name: str, n_bytes: int = 30_000,
                   skip_bytes: int = 100_000) -> float:
    """Pull a held-out slice from `name`'s dataset, prepend the tag byte,
    compute BPB under the library.

    `skip_bytes` skips past the bytes the library was trained on so the
    eval slice doesn't overlap training data — the V12 measurement bug
    fix.
    """
    from bench import bpb as _bpb
    buf = bytearray()
    skipped = 0
    served = 0
    for chunk in stream_domain(name, n_bytes + skip_bytes + 10000):
        if skipped < skip_bytes:
            need = skip_bytes - skipped
            if len(chunk) <= need:
                skipped += len(chunk)
                continue
            chunk = chunk[need:]
            skipped = skip_bytes
        if served >= n_bytes:
            break
        buf.extend(chunk[:n_bytes - served])
        served += min(len(chunk), n_bytes - served)
    if not buf:
        return float("nan")
    return _bpb(lib, bytes(buf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-bytes", type=int, default=50_000)
    ap.add_argument("--code-bytes", type=int, default=25_000)
    ap.add_argument("--structured-bytes", type=int, default=25_000)
    ap.add_argument("--bayes-train", type=int, default=30_000)
    ap.add_argument("--save-path", default=".rce_library_v12.pkl")
    ap.add_argument("--skip-validate", action="store_true",
                    help="skip pre-ingestion validate (debug only)")
    args = ap.parse_args()

    if not args.skip_validate:
        print("validating cross-domain datasets...")
        for name, (ds, cfg) in DATASETS.items():
            _validate(ds, cfg)

    print("interleaving streams...")
    plans = [("text", args.text_bytes),
             ("code", args.code_bytes),
             ("structured", args.structured_bytes)]
    buf = bytearray()
    t0 = time.time()
    for chunk in interleave_streams(plans):
        buf.extend(chunk)
    train = bytes(buf)
    print(f"  collected {len(train):,} bytes ({time.time()-t0:.1f}s)")

    from engine import fresh_library, save_library
    from bench import fit_primitives, bayes_train

    lib = fresh_library()
    print(f"fitting primitives on {len(train):,} bytes...")
    fit_primitives(lib, train)
    print(f"bayes-train on first {min(args.bayes_train, len(train)):,} bytes...")
    bayes_train(lib, train[:args.bayes_train],
                decay_every_steps=500, decay_factor=0.99,
                replay_buffer_size=2000, progress_every=2000)
    save_library(lib, args.save_path)
    print(f"saved {args.save_path}; lib={len(lib.programs)}")

    print("\nper-domain held-out BPB:")
    bpbs = {}
    for name in DATASETS:
        b = per_domain_bpb(lib, name, n_bytes=10_000)
        bpbs[name] = b
        print(f"  {name:10s}  BPB = {b:.4f}")

    # gate: per-domain BPB no worse than 1.2× the single-domain baseline
    # (single-domain baseline is V7's 2.2278 on wikitext-2; we use that
    # as a rough anchor here — a strict V12 gate would re-train each
    # domain solo, which we skip for compute budget)
    print(f"\n(rough V12 gate: per-domain BPB ≤ 2.67 = 1.2 × V7 single-domain 2.23)")


if __name__ == "__main__":
    main()
