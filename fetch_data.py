"""
Fetch the V1 evaluation data into ./data/ once.

  - wikitext2_train.txt    : train slice from Salesforce/wikitext, wikitext-2-raw-v1
  - wikitext2_heldout.txt  : 100KB held-out slice (validation split)
  - ood_random.bin         : 20KB of /dev/urandom — out-of-distribution bytes
                             for the V1 refusal-rate metric
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
DATA.mkdir(exist_ok=True)

WIKITEXT_TRAIN = DATA / "wikitext2_train.txt"
WIKITEXT_HELDOUT = DATA / "wikitext2_heldout.txt"
OOD_BYTES = DATA / "ood_random.bin"

TRAIN_BUDGET_BYTES = 1_000_000   # 1MB train slice — enough for the n-gram fit
HELDOUT_BUDGET_BYTES = 100_000   # 100KB held-out (V1 spec)
OOD_BUDGET_BYTES = 20_000


def _validate_or_fail(name: str, config: str | None):
    from validate_dataset import validate
    rep = validate(name, config=config, split="train", sample=100)
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
        print(f"  WARN: {rep.summary} — proceeding but flagging in LOG.md")


def fetch_wikitext():
    from datasets import load_dataset

    if WIKITEXT_TRAIN.exists() and WIKITEXT_HELDOUT.exists():
        print("wikitext: already present")
        return

    print("validating Salesforce/wikitext wikitext-2-raw-v1...")
    _validate_or_fail("Salesforce/wikitext", "wikitext-2-raw-v1")

    print("downloading Salesforce/wikitext wikitext-2-raw-v1...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    if not WIKITEXT_TRAIN.exists():
        with WIKITEXT_TRAIN.open("wb") as f:
            written = 0
            for row in ds["train"]:
                line = row["text"]
                if not line:
                    continue
                b = line.encode("utf-8", errors="replace")
                if written + len(b) > TRAIN_BUDGET_BYTES:
                    f.write(b[:TRAIN_BUDGET_BYTES - written])
                    break
                f.write(b)
                written += len(b)
        print(f"  wrote {WIKITEXT_TRAIN} ({WIKITEXT_TRAIN.stat().st_size:,} bytes)")

    if not WIKITEXT_HELDOUT.exists():
        with WIKITEXT_HELDOUT.open("wb") as f:
            written = 0
            for row in ds["validation"]:
                line = row["text"]
                if not line:
                    continue
                b = line.encode("utf-8", errors="replace")
                if written + len(b) > HELDOUT_BUDGET_BYTES:
                    f.write(b[:HELDOUT_BUDGET_BYTES - written])
                    break
                f.write(b)
                written += len(b)
        print(f"  wrote {WIKITEXT_HELDOUT} ({WIKITEXT_HELDOUT.stat().st_size:,} bytes)")


def fetch_ood():
    if OOD_BYTES.exists():
        print("ood: already present")
        return
    # Use os.urandom for cross-platform; equivalent to /dev/urandom.
    OOD_BYTES.write_bytes(os.urandom(OOD_BUDGET_BYTES))
    print(f"  wrote {OOD_BYTES} ({OOD_BYTES.stat().st_size:,} bytes)")


def main():
    fetch_wikitext()
    fetch_ood()
    print("done.")


if __name__ == "__main__":
    main()
