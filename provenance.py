"""
V16: append-only provenance store.

When a program achieves high posterior weight, we want to know *why*.
This module writes one JSONL line per program-weight-bump event so we
can later answer "when asked about X, programs P1, P2, P3 activate, and
they were strengthened primarily by data from sources S1, S2."

Persistence: `provenance/<program_name>.jsonl`. Append-only. Programs
that get pruned still keep their provenance file (per storage.md).

Each line is one event:
  {
    "ts": "2026-05-07T18:30:00Z",
    "shard": "wikitext-2",
    "ctx_hash": "ab3f...",
    "ctx_preview": "the cat sat on",
    "delta_log_weight": +0.014,
    "rank_after": 8,
  }

The store is sampled — we only log when the program's log-weight
crossed a threshold or moved by more than a noise floor — to keep
volume manageable.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent
PROV_DIR = REPO / "provenance"
PROV_DIR.mkdir(exist_ok=True)


def _safe(name: str) -> str:
    """Make a program name safe to use as a filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:200]


class ProvenanceStore:
    """Sampled append-only logger for per-program weight events.

    Each call to record() may or may not write — we only persist events
    that meet `delta_threshold` so we don't drown in noise.
    """
    def __init__(self, shard: str = "default",
                 delta_threshold: float = 0.005,
                 ctx_preview_len: int = 32):
        self.shard = shard
        self.delta_threshold = delta_threshold
        self.ctx_preview_len = ctx_preview_len

    def record(self, program_name: str, ctx: bytes, delta_log_weight: float,
               rank_after: int | None = None):
        if abs(delta_log_weight) < self.delta_threshold:
            return
        path = PROV_DIR / f"{_safe(program_name)}.jsonl"
        h = hashlib.sha256(ctx).hexdigest()[:12]
        try:
            preview = ctx[-self.ctx_preview_len:].decode("utf-8", errors="replace")
        except Exception:
            preview = repr(ctx[-self.ctx_preview_len:])
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "shard": self.shard,
            "ctx_hash": h,
            "ctx_preview": preview,
            "delta_log_weight": round(delta_log_weight, 6),
            "rank_after": rank_after,
        }
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def query(self, program_name: str, limit: int = 50) -> list[dict]:
        path = PROV_DIR / f"{_safe(program_name)}.jsonl"
        if not path.exists():
            return []
        out = []
        with path.open() as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out[-limit:]

    def summary(self, program_name: str) -> dict:
        rows = self.query(program_name, limit=10_000)
        if not rows:
            return {"program": program_name, "events": 0}
        shards = {}
        for r in rows:
            shards[r["shard"]] = shards.get(r["shard"], 0) + 1
        total_delta = sum(r["delta_log_weight"] for r in rows)
        return {
            "program": program_name,
            "events": len(rows),
            "first_ts": rows[0]["ts"],
            "last_ts": rows[-1]["ts"],
            "total_delta_log_weight": round(total_delta, 4),
            "by_shard": shards,
            "recent_contexts": [r["ctx_preview"] for r in rows[-5:]],
        }


# A simple module-level instance for common use
default_store: ProvenanceStore | None = None


def init_default(shard: str, delta_threshold: float = 0.005):
    global default_store
    default_store = ProvenanceStore(shard=shard, delta_threshold=delta_threshold)


def record(program_name: str, ctx: bytes, delta: float, rank: int | None = None):
    if default_store is not None:
        default_store.record(program_name, ctx, delta, rank)


# CLI: provenance <query>
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="program name to query (or 'list' to enumerate)")
    args = ap.parse_args()
    if args.name == "list":
        for p in sorted(PROV_DIR.glob("*.jsonl")):
            print(p.stem)
        return
    store = ProvenanceStore()
    s = store.summary(args.name)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
