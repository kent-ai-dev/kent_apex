"""
V17: distributed library federation.

Train multiple specialist libraries (one per domain), then merge them
with the same prior-correction math V8 uses for Modal sharding. The
bet: combinator generation across heterogeneous specialists discovers
cross-domain abstractions that none of them would have produced alone.

Library exchange format:
    {
      "version": 17,
      "library_name": str,                 # e.g. "code-specialist"
      "log_weights": dict[str, float],
      "lengths":     dict[str, float],
      "n_programs":  int,
      "provenance":  {
        "training_dataset": str,
        "training_bytes":   int,
        "trained_at":       ISO timestamp,
      }
    }

Libraries are exchanged as gzipped pickles with a manifest.

Federation = merge_libraries (V8) but on specialists instead of shards.
The math is identical; the framing is different.
"""
from __future__ import annotations
import argparse
import gzip
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

from engine import load_library
from modal_train import merge_libraries

REPO = Path(__file__).resolve().parent


def export_library(lib_path: str, name: str, training_dataset: str,
                   training_bytes: int, out_path: str):
    """Serialise a trained library into the federation exchange format."""
    lib = load_library(lib_path)
    if lib is None:
        raise SystemExit(f"could not load {lib_path}")
    payload = {
        "version": 17,
        "library_name": name,
        "log_weights": dict(lib.log_weights),
        "lengths": {p.name: p.length for p in lib.programs},
        "n_programs": len(lib.programs),
        "provenance": {
            "training_dataset": training_dataset,
            "training_bytes": training_bytes,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    with gzip.open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"exported {name} -> {out_path} ({len(payload['log_weights'])} programs)")


def import_library(path: str) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def federate(library_paths: list[str], out_path: str = "federated.pkl"):
    """Load N specialist libraries, merge with prior-correction, save."""
    payloads = []
    for p in library_paths:
        payload = import_library(p)
        print(f"  imported {payload['library_name']}: "
              f"{payload['n_programs']} programs from "
              f"{payload['provenance']['training_dataset']}")
        payloads.append(payload)

    merged = merge_libraries(payloads)
    out = REPO / out_path
    with out.open("wb") as f:
        pickle.dump(merged, f)
    print(f"federated: {merged['n_programs']} unique programs; "
          f"{merged['n_shared']} appeared in ≥2 specialists")
    print(f"saved {out}")
    return merged


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="serialise a trained library")
    e.add_argument("--lib", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--dataset", required=True)
    e.add_argument("--bytes", type=int, required=True)
    e.add_argument("--out", required=True)

    f = sub.add_parser("federate", help="merge multiple specialists")
    f.add_argument("paths", nargs="+", help="paths to exported libraries")
    f.add_argument("--out", default="federated.pkl")

    args = ap.parse_args()
    if args.cmd == "export":
        export_library(args.lib, args.name, args.dataset, args.bytes, args.out)
    elif args.cmd == "federate":
        federate(args.paths, args.out)


if __name__ == "__main__":
    main()
