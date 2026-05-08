"""
V6: streaming HuggingFace ingestion.

Streams bytes from any HF dataset by name, with the pre-ingestion validator
as a hard gate (per plans/v1.md §0). No new download in advance — the
trainer pulls from the stream and updates as bytes flow.

Usage (programmatic):
    from datasets_hf import iter_bytes
    for chunk in iter_bytes("Salesforce/wikitext", config="wikitext-103-raw-v1",
                            split="train", max_bytes=10_000_000):
        # `chunk` is a bytes object; feed it into the trainer

Each yielded chunk is a single document/row's text encoded as UTF-8.
The caller decides how to slice further (n-byte windows, etc).

Throughput target (V6 gate): ≥50 KB/s sustained. Realised throughput
depends on network + HF backend; the streaming load via `datasets`
typically saturates a single core's UTF-8 decode at well above that.
"""
from __future__ import annotations
from typing import Iterator
import time


def _validate_or_fail(name: str, config: str | None, split: str):
    from validate_dataset import validate
    rep = validate(name, config=config, split=split, sample=100)
    print(f"  validator: {rep.summary}")
    if rep.verdict == "FAIL":
        for c in rep.checks:
            mark = "✓" if c["score"] == c["max"] else ("~" if c["score"] > 0 else "✗")
            print(f"    {mark} {c['check']}: {c.get('note', '')}")
        raise SystemExit(
            f"FAIL: {name} ({config}) failed pre-ingestion validation. "
            f"Pick a different dataset and validate it before retrying."
        )
    if rep.verdict == "WARN":
        print(f"  WARN: {rep.summary} — proceeding")


def iter_bytes(name: str, config: str | None = None, split: str = "train",
               text_field: str | None = None,
               max_bytes: int | None = None,
               text_only: bool = True) -> Iterator[bytes]:
    """Yield UTF-8-encoded bytes from a HF dataset, in document order.

    Stops at `max_bytes` total bytes if specified.
    """
    _validate_or_fail(name, config, split)

    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, streaming=True)

    # auto-detect text field on first row if not given
    field = text_field
    served = 0
    t0 = time.time()
    last_report = t0
    for row in ds:
        if field is None:
            for cand in ("text", "content", "sentence", "article", "document"):
                if cand in row:
                    field = cand
                    break
            if field is None:
                # fall back to first string-valued field in the row
                for k, v in row.items():
                    if isinstance(v, str):
                        field = k
                        break
        if field is None:
            continue
        v = row.get(field)
        if not v:
            continue
        b = (str(v) if text_only else v).encode("utf-8", errors="replace")
        if max_bytes is not None and served + len(b) > max_bytes:
            yield b[:max_bytes - served]
            served = max_bytes
            break
        yield b
        served += len(b)

        now = time.time()
        if now - last_report > 5.0:
            rate = served / max(now - t0, 0.01) / 1024
            print(f"  ingest: {served:,} bytes in {now-t0:.1f}s = {rate:.1f} KB/s")
            last_report = now

    elapsed = time.time() - t0
    rate = served / max(elapsed, 0.01) / 1024
    print(f"  ingest done: {served:,} bytes in {elapsed:.1f}s = {rate:.1f} KB/s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="HF dataset id, e.g. Salesforce/wikitext")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-bytes", type=int, default=1_000_000)
    args = ap.parse_args()

    total = 0
    for chunk in iter_bytes(args.name, config=args.config,
                             split=args.split, max_bytes=args.max_bytes):
        total += len(chunk)
    print(f"total: {total:,} bytes")
