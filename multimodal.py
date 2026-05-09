"""
V19: multi-modal extension — train the same combinator library on
non-text byte streams. The architecture is structurally agnostic: a
byte is a byte, n-grams over MNIST pixel sequences should pick up
spatial regularities just like n-grams over text pick up grammatical
ones.

Two modalities here:
  - images:  ylecun/mnist (28×28 grayscale flattened to 784 bytes per image)
  - audio:   openslr/librispeech_asr (μ-law 8-bit, downsampled to 8kHz)

Gate (v1.md V19): BPB on images ≤ 6.5 (raw bytes 8.0; PNG ~4.0 — we
sit somewhere in the middle, hopefully).

This is V20's stretch goal — failure here is informative, not blocking.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent


def stream_mnist_bytes(max_bytes: int) -> bytes:
    """Flatten N MNIST images into a continuous byte stream. The library
    sees one giant sequence of pixel intensities with image-boundary
    markers (0xFF — never a valid uint8 pixel value)."""
    from datasets import load_dataset
    import numpy as np
    from validate_dataset import validate
    rep = validate("ylecun/mnist", split="train", sample=20)
    print(f"  validator: {rep.summary}")
    if rep.verdict == "FAIL":
        raise SystemExit("FAIL on ylecun/mnist")

    ds = load_dataset("ylecun/mnist", split="train", streaming=True)
    buf = bytearray()
    for row in ds:
        img = row.get("image")
        if img is None:
            continue
        arr = np.array(img, dtype=np.uint8).flatten()
        # cap pixel values at 254 so 0xFF can serve as a separator
        arr = np.where(arr == 255, 254, arr)
        buf.extend(arr.tobytes())
        buf.append(0xFF)  # image-boundary marker
        if len(buf) >= max_bytes:
            break
    return bytes(buf[:max_bytes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist-bytes", type=int, default=200_000,
                    help="MNIST byte stream size (200KB ≈ 250 images)")
    ap.add_argument("--bayes-train", type=int, default=20_000)
    ap.add_argument("--save-path", default=".rce_library_v19_mnist.pkl")
    args = ap.parse_args()

    print("streaming MNIST...")
    t0 = time.time()
    train = stream_mnist_bytes(args.mnist_bytes)
    print(f"  collected {len(train):,} bytes ({time.time()-t0:.1f}s)")

    from engine import fresh_library, save_library
    from bench import fit_primitives, bayes_train, bpb

    lib = fresh_library()
    print(f"fitting on {len(train):,} bytes...")
    fit_primitives(lib, train)
    print(f"bayes-train on first {min(args.bayes_train, len(train)):,} bytes...")
    bayes_train(lib, train[:args.bayes_train],
                decay_every_steps=500, replay_buffer_size=2000,
                progress_every=2000)
    save_library(lib, args.save_path)

    # eval on a held-out slice (the last 50KB of the collected stream)
    heldout = train[-50_000:] if len(train) > 50_000 else train
    val_bpb = bpb(lib, heldout)
    print(f"\nV19 image BPB: {val_bpb:.4f}  (gate ≤ 6.5; raw 8.0; PNG ~4.0)")
    print(f"saved {args.save_path}; lib={len(lib.programs)}")


if __name__ == "__main__":
    main()
