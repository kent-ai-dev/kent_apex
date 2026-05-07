"""
Recurrent Compression Engine - CLI.

Usage:
    python rce.py train <text-file>     # train on a text file
    python rce.py train                 # train on stdin
    python rce.py chat                  # chat with the latest model
    python rce.py status                # inspect the library
    python rce.py reset                 # wipe and start over
"""

import sys
import os
import random
from collections import Counter

from engine import (
    Library, NGramPrimitive,
    fresh_library, load_library, save_library,
)


MODEL_PATH = os.path.expanduser("~/.rce_library.pkl")


def cmd_train(source: str | None):
    """Train the library on text. Streaming, byte-level."""
    lib = load_library(MODEL_PATH) or fresh_library()

    # read input
    if source is None:
        print("[reading from stdin; Ctrl-D when done]")
        data = sys.stdin.read().encode("utf-8", errors="replace")
    else:
        if not os.path.exists(source):
            print(f"file not found: {source}")
            return
        with open(source, "rb") as f:
            data = f.read()

    if not data:
        print("no data to train on")
        return

    print(f"training on {len(data)} bytes...")

    # first: fit the n-gram primitives directly (they are fast statistical learners)
    for p in lib.programs:
        if isinstance(p, NGramPrimitive):
            p.fit(data)

    # second: stream through the data, doing Bayesian updates and growth
    ctx_window = 16
    grow_every = 200
    # update on every byte for small corpora, subsample for big ones
    update_every = 1 if len(data) < 50_000 else 4
    step = 0
    for i in range(1, len(data)):
        if i % update_every != 0:
            continue
        ctx = data[max(0, i - ctx_window):i]
        actual = data[i]
        lib.update(ctx, actual)
        step += 1
        if step % grow_every == 0:
            new = lib.grow(n_children=4)
            lib.prune(max_size=150)
            if step % (grow_every * 5) == 0:
                top = lib.top_programs(k=3)
                names = ", ".join(p.name[:30] for p in top)
                print(f"  step {step}: |lib|={len(lib.programs)}  top: {names}")

    # final growth pass
    lib.grow(n_children=8)
    lib.prune(max_size=150)

    save_library(lib, MODEL_PATH)
    print(f"saved. library size: {len(lib.programs)}")
    print("top programs:")
    post = lib.posterior()
    for p in lib.top_programs(k=10):
        w = post.get(p.name, 0.0)
        print(f"  {w:.4f}  {p.name}")


def sample_from(dist: Counter, temperature: float = 0.8) -> int:
    """Sample a byte from a distribution, with temperature."""
    if not dist:
        return random.randint(32, 126)
    items = list(dist.items())
    total = sum(v for _, v in items) or 1.0
    if temperature <= 0:
        return max(items, key=lambda x: x[1])[0]
    # apply temperature
    weights = [(k, (v / total) ** (1.0 / temperature)) for k, v in items]
    z = sum(w for _, w in weights) or 1.0
    r = random.random() * z
    acc = 0.0
    for k, w in weights:
        acc += w
        if acc >= r:
            return k
    return weights[-1][0]


def generate(lib: Library, prompt: bytes, max_bytes: int = 200,
             temperature: float = 0.8, ctx_window: int = 16) -> bytes:
    """Generate a continuation of the prompt."""
    out = bytearray(prompt)
    for _ in range(max_bytes):
        ctx = bytes(out[-ctx_window:])
        dist = lib.predict(ctx)
        nxt = sample_from(dist, temperature=temperature)
        out.append(nxt)
        # stop on newline if we have at least some output
        if nxt == ord("\n") and len(out) - len(prompt) > 4:
            break
    return bytes(out[len(prompt):])


def cmd_chat():
    lib = load_library(MODEL_PATH)
    if lib is None:
        print("no trained model. run: python rce.py train <file>")
        return
    print(f"loaded library: {len(lib.programs)} programs")
    print("type to chat. /quit to exit, /entropy to see uncertainty, /top to see active programs")
    print()
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/entropy":
            ctx = line.encode("utf-8")[-8:]
            print(f"  entropy on empty ctx: {lib.entropy(b''):.2f} bits")
            print(f"  (max possible = 8.0; high = system is uncertain)")
            continue
        if line == "/top":
            post = lib.posterior()
            for p in lib.top_programs(k=8):
                print(f"  {post.get(p.name, 0):.4f}  {p.name}")
            continue

        prompt = (line + "\n").encode("utf-8", errors="replace")
        # measure uncertainty before generating
        h = lib.entropy(prompt[-8:])
        if h > 7.5:
            print("rce> [I don't have enough learned structure to respond meaningfully.]")
            continue
        reply_bytes = generate(lib, prompt, max_bytes=160, temperature=0.7)
        try:
            reply = reply_bytes.decode("utf-8", errors="replace").strip()
        except Exception:
            reply = repr(reply_bytes)
        if not reply:
            reply = "[silence]"
        print(f"rce> {reply}")


def cmd_status():
    lib = load_library(MODEL_PATH)
    if lib is None:
        print("no trained model")
        return
    print(f"library size: {len(lib.programs)}")
    by_kind = {}
    for p in lib.programs:
        by_kind[p.combinator] = by_kind.get(p.combinator, 0) + 1
    print("by combinator:")
    for k, v in sorted(by_kind.items()):
        print(f"  {k}: {v}")
    print("\ntop 15 programs by posterior weight:")
    post = lib.posterior()
    for p in lib.top_programs(k=15):
        print(f"  {post.get(p.name, 0):.4f}  len={p.length:5.1f}  {p.name}")


def cmd_reset():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print("library wiped.")
    else:
        print("no library to wipe.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "train":
        source = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_train(source)
    elif cmd == "chat":
        cmd_chat()
    elif cmd == "status":
        cmd_status()
    elif cmd == "reset":
        cmd_reset()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()