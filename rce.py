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
    Library, NGramPrimitive, Toplevel,
    fresh_library, load_library, save_library,
)


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "RCE_LIBRARY",
    os.path.join(REPO_DIR, ".rce_library.pkl"),
)


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
             temperature: float = 0.8, ctx_window: int = 16,
             strict_threshold: float = 0.0,
             explain: bool = False) -> bytes:
    """Generate a continuation of the prompt.

    V10 additions:
    - `strict_threshold` > 0: refuse to emit a byte if its (normalised)
      confidence is below the threshold. Returns whatever was emitted
      so far + a refusal marker.
    - `explain`: after each byte, print top-3 programs that contributed
      most to selecting it.
    """
    out = bytearray(prompt)
    for _ in range(max_bytes):
        ctx = bytes(out[-ctx_window:])
        dist = lib.predict(ctx)
        total = sum(dist.values()) or 1.0
        nxt = sample_from(dist, temperature=temperature)
        conf = dist.get(nxt, 0.0) / total

        # Print the explain line BEFORE the abstain check so the user
        # can see which programs voted at the position we actually
        # abstained on (previously the explain printed only for
        # accepted bytes, making the abstain message look detached
        # from the voters shown above it).
        if explain:
            top3 = _top_voting_programs(lib, ctx, nxt, k=3)
            mark = " [ABSTAIN]" if (strict_threshold > 0 and conf < strict_threshold) else ""
            print(f"  byte={chr(nxt) if 32 <= nxt < 127 else f'\\x{nxt:02x}'} "
                  f"conf={conf:.3f}{mark}", end=" ")
            print("top voters:",
                  ", ".join(f"{n}:{w:.3f}" for n, w in top3))

        if strict_threshold > 0 and conf < strict_threshold:
            out.extend(b"[abstain: confidence " + f"{conf:.2f}".encode() + b"]")
            break

        out.append(nxt)
        if nxt == ord("\n") and len(out) - len(prompt) > 4:
            break
    return bytes(out[len(prompt):])


def _top_voting_programs(lib: Library, ctx: bytes, target_byte: int, k: int = 3):
    """Return the top-k (program_name, weighted_contribution) for `target_byte`.
    Decomposes lib.predict to attribute the chosen byte's mass to programs.
    """
    post = lib.posterior()
    contribs: list[tuple[str, float]] = []
    for p in lib.programs:
        w = post.get(p.name, 0.0)
        if w < 1e-6:
            continue
        d = p.predict(ctx)
        tot = sum(d.values()) or 1.0
        contribs.append((p.name, w * (d.get(target_byte, 0.0) / tot)))
    contribs.sort(key=lambda x: -x[1])
    return contribs[:k]


def cmd_chat(strict_threshold: float = 0.0, explain: bool = False,
             toplevel: bool = False, refusal_tau: float = 0.5):
    lib = load_library(MODEL_PATH)
    if lib is None:
        print("no trained model. run: python rce.py train <file>")
        return
    if toplevel and not isinstance(lib, Toplevel):
        # V10: wrap a saved bare library in Toplevel for the chat session
        lib = Toplevel(lib, vocab_size=256, evidence_window=32)
        print(f"wrapped library in Toplevel (V10 architectural refusal)")
    elif isinstance(lib, Toplevel):
        print(f"library is already Toplevel-wrapped (V10)")
    print(f"loaded library: {len(lib.programs)} programs")
    if strict_threshold > 0:
        print(f"--strict mode: abstain when byte-level confidence < {strict_threshold:.2f}")
        print("  (note: byte-level top-prediction confidence rarely exceeds 0.3"
              " outside very predictable contexts like spaces after words;"
              " 0.10-0.15 is a more typical abstain threshold)")
    if explain:
        print("--explain mode: per-byte top voters printed")
    print("type to chat. /quit, /entropy, /top, /strict <threshold>, /explain on|off")
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
            print(f"  entropy on empty ctx: {lib.entropy(b''):.2f} bits")
            print(f"  (max possible = 8.0; high = system is uncertain)")
            continue
        if line == "/top":
            post = lib.posterior()
            for p in lib.top_programs(k=8):
                print(f"  {post.get(p.name, 0):.4f}  {p.name}")
            continue
        if line.startswith("/strict"):
            parts = line.split()
            # Byte-level top-prediction confidence is bounded by the
            # vocab structure; even for "obvious" continuations like
            # the space after a word, conf rarely exceeds ~0.3, so a
            # 0.30 default is effectively "always abstain". 0.10 is a
            # more honest "I really have no idea" gate. Pass an
            # explicit value to override.
            strict_threshold = float(parts[1]) if len(parts) > 1 else 0.10
            print(f"  strict threshold = {strict_threshold:.2f}")
            continue
        if line.startswith("/explain"):
            parts = line.split()
            explain = (len(parts) > 1 and parts[1] == "on")
            print(f"  explain = {explain}")
            continue

        prompt = (line + "\n").encode("utf-8", errors="replace")
        # V10 refusal: if Toplevel-wrapped, P(Background|prompt) > τ blocks
        # generation BEFORE we sample. This is the architectural invariant
        # — not the V1 entropy heuristic which the V3 KN regression broke.
        if isinstance(lib, Toplevel):
            r = lib.refusal_score(prompt)
            if r > refusal_tau:
                print(f"rce> [refused: P(Background|prompt) = {r:.3f} > "
                      f"τ = {refusal_tau:.2f}]")
                continue
        else:
            # back-compat: V1 entropy gate when no Toplevel wrapper
            h = lib.entropy(prompt[-8:])
            if h > 7.5:
                print("rce> [I don't have enough learned structure to respond meaningfully.]")
                continue
        reply_bytes = generate(lib, prompt, max_bytes=160, temperature=0.7,
                                strict_threshold=strict_threshold,
                                explain=explain)
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
        # parse --strict T, --explain, --toplevel, --tau τ flags
        rest = sys.argv[2:]
        strict = 0.0
        explain = False
        toplevel = False
        tau = 0.5
        for i, a in enumerate(rest):
            if a == "--strict" and i + 1 < len(rest):
                strict = float(rest[i + 1])
            if a == "--explain":
                explain = True
            if a == "--toplevel":
                toplevel = True
            if a == "--tau" and i + 1 < len(rest):
                tau = float(rest[i + 1])
        cmd_chat(strict_threshold=strict, explain=explain,
                 toplevel=toplevel, refusal_tau=tau)
    elif cmd == "status":
        cmd_status()
    elif cmd == "reset":
        cmd_reset()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()