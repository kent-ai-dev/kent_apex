"""
V15: conversational fine-grain.

Train on multi-turn dialogue corpora with a `<dialog>` framing primitive
so the engine learns turn structure. Stop generation on turn-boundary
markers. Continue mode maintains chat context across turns.

Datasets (each validated before ingestion):
  - daily_dialog       : everyday two-party dialog
  - persona_chat       : persona-conditioned multi-turn
  - oasst1             : assistant-style conversations

Turn marker bytes (avoids ASCII collision):
  0x83 = USER turn start
  0x84 = ASSISTANT turn start
  0x85 = TURN end (used as a hard stop in generation)
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parent

USER_TAG = 0x83
ASSISTANT_TAG = 0x84
TURN_END = 0x85


DATASETS = {
    "daily_dialog": ("li2017dailydialog/daily_dialog", None,
                     "dialog"),
    "oasst1":       ("OpenAssistant/oasst1", None, None),
    "persona_chat": ("bavard/personachat_truecased", None, None),
}


def _validate(name: str, config: str | None):
    from validate_dataset import validate
    rep = validate(name, config=config, split="train", sample=30)
    print(f"  {name} ({config}): {rep.summary}")
    return rep


def stream_daily_dialog(max_bytes: int) -> Iterator[bytes]:
    from datasets import load_dataset
    ds = load_dataset("li2017dailydialog/daily_dialog", split="train", streaming=True)
    served = 0
    for row in ds:
        turns = row.get("dialog") or []
        for i, turn in enumerate(turns):
            tag = USER_TAG if (i % 2 == 0) else ASSISTANT_TAG
            b = bytes([tag]) + str(turn).encode("utf-8", errors="replace") + bytes([TURN_END])
            if served + len(b) > max_bytes:
                yield b[:max_bytes - served]
                return
            yield b
            served += len(b)


def stream_oasst(max_bytes: int) -> Iterator[bytes]:
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    served = 0
    for row in ds:
        role = row.get("role")
        text = row.get("text", "")
        if not text:
            continue
        tag = USER_TAG if role == "prompter" else ASSISTANT_TAG
        b = bytes([tag]) + str(text).encode("utf-8", errors="replace") + bytes([TURN_END])
        if served + len(b) > max_bytes:
            yield b[:max_bytes - served]
            return
        yield b
        served += len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-bytes", type=int, default=20_000)
    ap.add_argument("--oasst-bytes", type=int, default=20_000)
    ap.add_argument("--bayes-train", type=int, default=15_000)
    ap.add_argument("--save-path", default=".rce_library_v15.pkl")
    args = ap.parse_args()

    print("validating conversational datasets...")
    rep_dd = _validate("li2017dailydialog/daily_dialog", None)
    rep_oa = _validate("OpenAssistant/oasst1", None)

    streams = []
    if rep_dd.verdict != "FAIL":
        streams.append(("daily", stream_daily_dialog(args.daily_bytes)))
    if rep_oa.verdict != "FAIL":
        streams.append(("oasst", stream_oasst(args.oasst_bytes)))

    if not streams:
        raise SystemExit("no validated dialogue datasets — abort")

    print(f"streaming from {[n for n, _ in streams]}")
    buf = bytearray()
    t0 = time.time()
    for name, stream in streams:
        for chunk in stream:
            buf.extend(chunk)
    train = bytes(buf)
    print(f"  collected {len(train):,} bytes ({time.time()-t0:.1f}s)")

    from engine import fresh_library, save_library
    from bench import fit_primitives, bayes_train

    lib = fresh_library()
    print(f"fitting on {len(train):,} bytes...")
    fit_primitives(lib, train)
    print(f"bayes-train on first {min(args.bayes_train, len(train)):,} bytes...")
    bayes_train(lib, train[:args.bayes_train],
                decay_every_steps=500, replay_buffer_size=2000,
                progress_every=2000)
    save_library(lib, args.save_path)
    print(f"saved {args.save_path}; lib={len(lib.programs)}")


if __name__ == "__main__":
    main()
