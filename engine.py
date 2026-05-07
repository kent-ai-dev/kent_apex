"""
Recurrent Compression Engine - core.

The system is a library of small programs that predict the next byte
given recent context. Programs compete by Bayesian posterior weighted
by a Solomonoff-style length prior. New programs are *generated* (not
searched) by applying four combinators to existing high-weight programs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import math
import random
import pickle
import os


# ---------- Programs ----------
# A "program" here is a predictor: given recent context (a tuple of bytes),
# it returns a probability distribution over the next byte (a Counter).
# We represent programs as small dataclasses with a `predict` method and
# a `length` (description length, in bits, for the Solomonoff prior).

@dataclass
class Program:
    name: str
    length: float           # description length in bits
    parents: tuple = ()     # lineage: which programs generated this one
    combinator: str = "primitive"

    def predict(self, ctx: bytes) -> Counter:
        raise NotImplementedError

    def __hash__(self):
        return hash(self.name)


# ---------- Primitives ----------
# These are the seed programs. Everything grows from these by combinator.

class UniformPrimitive(Program):
    """Total ignorance: every byte equally likely."""
    def __init__(self):
        super().__init__(name="uniform", length=8.0)

    def predict(self, ctx: bytes) -> Counter:
        return Counter({b: 1.0 for b in range(256)})


class NGramPrimitive(Program):
    """Order-n Markov predictor learned from data."""
    def __init__(self, n: int):
        super().__init__(name=f"ngram-{n}", length=8.0 + 4.0 * n)
        self.n = n
        self.table: dict = {}   # plain dict, picklable
        self.fitted = False

    def fit(self, data: bytes):
        for i in range(self.n, len(data)):
            ctx = data[i - self.n:i]
            if ctx not in self.table:
                self.table[ctx] = Counter()
            self.table[ctx][data[i]] += 1
        self.fitted = True

    def predict(self, ctx: bytes) -> Counter:
        if not self.fitted or self.n == 0 or len(ctx) < self.n:
            return Counter({b: 1.0 for b in range(256)})
        # Look up the full n-context. Backoff to shorter contexts is handled
        # by lower-order NGramPrimitives in the same library (each n is its own
        # primitive); doing it inside one primitive devolves into an O(table)
        # scan per byte, which made eval pathologically slow.
        key = ctx[-self.n:]
        c = self.table.get(key)
        if c:
            return Counter({b: c.get(b, 0) + 0.01 for b in range(256)})
        return Counter({b: 1.0 for b in range(256)})


class RepeatPrimitive(Program):
    """Predict that the next byte equals the most recent byte."""
    def __init__(self):
        super().__init__(name="repeat", length=10.0)

    def predict(self, ctx: bytes) -> Counter:
        if not ctx:
            return Counter({b: 1.0 for b in range(256)})
        c = Counter({b: 0.1 for b in range(256)})
        c[ctx[-1]] += 10.0
        return c


# ---------- Combinators ----------
# These take parent programs and produce a child. This is the recurrence.
# New = combinator(parent_a, parent_b). No search.

class Composed(Program):
    """Mixture: blend two parents' predictions geometrically."""
    def __init__(self, a: Program, b: Program, alpha: float = 0.5):
        name = f"compose({a.name},{b.name})"
        length = a.length + b.length + 4.0
        super().__init__(name=name, length=length,
                         parents=(a, b), combinator="compose")
        self.a, self.b, self.alpha = a, b, alpha

    def predict(self, ctx: bytes) -> Counter:
        pa = self.a.predict(ctx)
        pb = self.b.predict(ctx)
        out = Counter()
        for k in range(256):
            out[k] = self.alpha * pa.get(k, 0.01) + (1 - self.alpha) * pb.get(k, 0.01)
        return out


class Branched(Program):
    """Context-conditional: pick parent A or B based on a learned trigger byte."""
    def __init__(self, a: Program, b: Program, trigger: int):
        name = f"branch({a.name},{b.name}|{trigger})"
        length = a.length + b.length + 8.0
        super().__init__(name=name, length=length,
                         parents=(a, b), combinator="branch")
        self.a, self.b, self.trigger = a, b, trigger

    def predict(self, ctx: bytes) -> Counter:
        if ctx and ctx[-1] == self.trigger:
            return self.a.predict(ctx)
        return self.b.predict(ctx)


class Abstracted(Program):
    """Lift a recurring pattern: predict from longer context if both parents agree."""
    def __init__(self, a: Program, b: Program):
        name = f"abstract({a.name},{b.name})"
        length = a.length + b.length + 6.0
        super().__init__(name=name, length=length,
                         parents=(a, b), combinator="abstract")
        self.a, self.b = a, b

    def predict(self, ctx: bytes) -> Counter:
        pa = self.a.predict(ctx)
        pb = self.b.predict(ctx)
        # consensus weighting: bytes both parents agree on get boosted
        out = Counter()
        for k in range(256):
            va = pa.get(k, 0.01)
            vb = pb.get(k, 0.01)
            out[k] = math.sqrt(va * vb)  # geometric mean = consensus
        return out


class Recurred(Program):
    """Self-application: predict the byte after what parent A predicts."""
    def __init__(self, a: Program):
        name = f"recur({a.name})"
        length = a.length + 5.0
        super().__init__(name=name, length=length,
                         parents=(a,), combinator="recur")
        self.a = a

    def predict(self, ctx: bytes) -> Counter:
        # apply parent, take its top prediction, append to context, predict again
        first = self.a.predict(ctx)
        if not first:
            return Counter({b: 1.0 for b in range(256)})
        top = max(first.items(), key=lambda x: x[1])[0]
        new_ctx = ctx + bytes([top])
        return self.a.predict(new_ctx)


COMBINATORS = ["compose", "branch", "abstract", "recur"]


# ---------- The Library ----------

@dataclass
class Library:
    programs: list = field(default_factory=list)
    log_weights: dict = field(default_factory=dict)  # name -> log posterior weight

    def add(self, p: Program):
        if p.name in self.log_weights:
            return
        self.programs.append(p)
        # initial log weight = -length (Solomonoff prior, log base 2)
        self.log_weights[p.name] = -p.length * math.log(2)

    def update(self, ctx: bytes, actual: int, lr: float = 0.03):
        """Bayesian update: programs that predicted `actual` better gain weight.
        lr < 1 softens the posterior so a few programs don't dominate everything.
        """
        for p in self.programs:
            dist = p.predict(ctx)
            total = sum(dist.values()) or 1.0
            prob = (dist.get(actual, 0.01) + 1e-9) / total
            self.log_weights[p.name] += lr * math.log(prob)
        self._renormalize()

    def _renormalize(self):
        # subtract max for numerical stability; this doesn't change the posterior
        m = max(self.log_weights.values())
        for k in self.log_weights:
            self.log_weights[k] -= m

    def posterior(self) -> dict:
        """Return normalized posterior weights (sum to 1)."""
        m = max(self.log_weights.values())
        exps = {k: math.exp(v - m) for k, v in self.log_weights.items()}
        z = sum(exps.values()) or 1.0
        return {k: v / z for k, v in exps.items()}

    def predict(self, ctx: bytes) -> Counter:
        """Posterior predictive: weighted ensemble over all programs."""
        post = self.posterior()
        out = Counter()
        for p in self.programs:
            w = post.get(p.name, 0.0)
            if w < 1e-6:
                continue
            dist = p.predict(ctx)
            total = sum(dist.values()) or 1.0
            for k, v in dist.items():
                out[k] += w * (v / total)
        return out

    def entropy(self, ctx: bytes) -> float:
        """How uncertain the library is. High = 'I don't know'."""
        dist = self.predict(ctx)
        total = sum(dist.values()) or 1.0
        h = 0.0
        for v in dist.values():
            p = v / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    def top_programs(self, k: int = 8) -> list:
        post = self.posterior()
        ranked = sorted(self.programs,
                        key=lambda p: post.get(p.name, 0.0),
                        reverse=True)
        return ranked[:k]

    def grow(self, n_children: int = 4):
        """The recurrence step. Generate new programs from top parents.
        This is the Fibonacci move: child = combinator(parent_i, parent_j)
        where i, j are recent high-weight programs.
        """
        parents = self.top_programs(k=8)
        if len(parents) < 2:
            return []
        added = []
        for _ in range(n_children):
            a, b = random.sample(parents, 2)
            combinator = random.choice(COMBINATORS)
            try:
                if combinator == "compose":
                    child = Composed(a, b, alpha=random.uniform(0.3, 0.7))
                elif combinator == "branch":
                    child = Branched(a, b, trigger=random.randint(0, 255))
                elif combinator == "abstract":
                    child = Abstracted(a, b)
                else:  # recur
                    child = Recurred(a)
            except Exception:
                continue
            if child.name not in self.log_weights:
                self.add(child)
                added.append(child)
        return added

    def prune(self, max_size: int = 200):
        """Cull low-weight programs. Keep primitives always."""
        if len(self.programs) <= max_size:
            return
        post = self.posterior()
        keep = []
        for p in self.programs:
            if p.combinator == "primitive":
                keep.append(p)
            elif post.get(p.name, 0) > 1e-5:
                keep.append(p)
        # then take top by weight up to max_size
        keep.sort(key=lambda p: post.get(p.name, 0.0), reverse=True)
        keep = keep[:max_size]
        kept_names = {p.name for p in keep}
        self.programs = keep
        self.log_weights = {k: v for k, v in self.log_weights.items()
                            if k in kept_names}


# ---------- Persistence ----------

def save_library(lib: Library, path: str):
    with open(path, "wb") as f:
        pickle.dump(lib, f)


def load_library(path: str) -> Library | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def fresh_library() -> Library:
    """Seed library with primitives."""
    lib = Library()
    lib.add(UniformPrimitive())
    lib.add(RepeatPrimitive())
    for n in (1, 2, 3, 4, 5, 6):
        lib.add(NGramPrimitive(n))
    return lib