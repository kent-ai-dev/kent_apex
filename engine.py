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


class KneserNeyNGram(Program):
    """N-gram predictor with Kneser-Ney smoothing. Stronger than add-k:
    instead of "how often did this byte follow this context," uses
    "in how many distinct contexts has this byte appeared as a continuation."
    Backs off recursively through lower orders.
    """
    def __init__(self, n: int, d: float = 0.75):
        super().__init__(name=f"kn-{n}", length=12.0 + 4.5 * n)
        self.n = n
        self.d = d
        # tables[m]: dict[tuple[bytes], Counter[next_byte]] for m-byte contexts
        self.tables: list[dict] = [{} for _ in range(n + 1)]
        # continuation[m][b] = how many distinct (m-byte) contexts have b as their continuation
        self.continuation: list[Counter] = [Counter() for _ in range(n + 1)]
        self.total_bigrams = 0
        self.fitted = False

    def fit(self, data: bytes):
        for i in range(self.n, len(data)):
            for m in range(self.n + 1):
                ctx = bytes(data[i - m:i])
                if ctx not in self.tables[m]:
                    self.tables[m][ctx] = Counter()
                # for continuation counts at order m, increment the FIRST time
                # we see (ctx, data[i]) — this measures distinct contexts
                if self.tables[m][ctx][data[i]] == 0:
                    self.continuation[m][data[i]] += 1
                self.tables[m][ctx][data[i]] += 1
        self.total_bigrams = sum(self.continuation[1].values())
        self.fitted = True

    def _kn_prob(self, ctx: bytes, b: int, m: int) -> float:
        """Recursive Kneser-Ney probability of byte `b` after `ctx` at order m."""
        if m == 0:
            # base case: continuation distribution
            total_cont = self.total_bigrams or 1
            return self.continuation[1][b] / total_cont
        sub = ctx[-m:] if len(ctx) >= m else ctx
        t = self.tables[m].get(sub)
        if not t:
            return self._kn_prob(ctx, b, m - 1)
        total = sum(t.values())
        if total == 0:
            return self._kn_prob(ctx, b, m - 1)
        # discount + interpolation with lower order
        c = t.get(b, 0)
        prob_main = max(c - self.d, 0.0) / total
        n_distinct = sum(1 for v in t.values() if v > 0)
        lambda_w = (self.d * n_distinct) / total
        return prob_main + lambda_w * self._kn_prob(ctx, b, m - 1)

    def predict(self, ctx: bytes) -> Counter:
        if not self.fitted:
            return Counter({b: 1.0 for b in range(256)})
        return Counter({b: max(self._kn_prob(ctx, b, self.n), 1e-9)
                        for b in range(256)})


class SkipGramPredictor(Program):
    """Predict from non-adjacent context: bytes at offsets -k, -2k, -3k.
    Captures dependencies that strict n-grams miss (e.g., long-range
    grammatical agreement).
    """
    def __init__(self, k: int = 2, depth: int = 3):
        super().__init__(name=f"skip-{k}x{depth}", length=14.0 + 3.0 * depth)
        self.k = k
        self.depth = depth
        # table maps tuple of skip-bytes -> Counter over next byte
        self.table: dict = {}
        self.fitted = False

    def _skip_key(self, ctx: bytes) -> tuple | None:
        offs = [-i * self.k for i in range(1, self.depth + 1)]
        if any(-o > len(ctx) for o in offs):
            return None
        return tuple(ctx[o] for o in offs)

    def fit(self, data: bytes):
        max_off = self.k * self.depth
        for i in range(max_off, len(data)):
            ctx = data[max(0, i - max_off):i]
            key = self._skip_key(ctx)
            if key is None:
                continue
            if key not in self.table:
                self.table[key] = Counter()
            self.table[key][data[i]] += 1
        self.fitted = True

    def predict(self, ctx: bytes) -> Counter:
        if not self.fitted:
            return Counter({b: 1.0 for b in range(256)})
        key = self._skip_key(ctx)
        if key is None:
            return Counter({b: 1.0 for b in range(256)})
        c = self.table.get(key)
        if not c:
            return Counter({b: 1.0 for b in range(256)})
        return Counter({b: c.get(b, 0) + 0.05 for b in range(256)})


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


def fresh_library(rich: bool = True) -> Library:
    """Seed library with primitives.

    With rich=True (default since V3), also adds Kneser-Ney n-grams and a
    skip-gram predictor — primitives that capture stronger statistical
    structure than plain add-k n-grams. Set rich=False for the V1/V2
    baseline behaviour.
    """
    lib = Library()
    lib.add(UniformPrimitive())
    lib.add(RepeatPrimitive())
    for n in (1, 2, 3, 4, 5, 6):
        lib.add(NGramPrimitive(n))
    if rich:
        for n in (3, 4, 5):
            lib.add(KneserNeyNGram(n))
        lib.add(SkipGramPredictor(k=2, depth=3))
        lib.add(SkipGramPredictor(k=3, depth=3))
    return lib