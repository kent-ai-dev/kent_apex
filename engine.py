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


class Gated(Program):
    """Multi-trigger context-conditional: use parent A if last byte is in
    `trigger_set`, otherwise B. Generalizes Branched (single-byte trigger).
    """
    def __init__(self, a: Program, b: Program, trigger_set: frozenset[int]):
        triggers = sorted(trigger_set)[:5]   # cap label length
        tag = "+".join(str(t) for t in triggers) + (
            f"+{len(trigger_set)-5}more" if len(trigger_set) > 5 else "")
        name = f"gate({a.name},{b.name}|{{{tag}}})"
        length = a.length + b.length + 8.0 + 0.5 * len(trigger_set)
        super().__init__(name=name, length=length,
                         parents=(a, b), combinator="gate")
        self.a, self.b, self.trigger_set = a, b, trigger_set

    def predict(self, ctx: bytes) -> Counter:
        if ctx and ctx[-1] in self.trigger_set:
            return self.a.predict(ctx)
        return self.b.predict(ctx)


class Memoized(Program):
    """LRU-cache wrapper: parent A's predictions are cached by context tail.
    Doesn't change semantics — only performance — but length is heavier so
    Memoized children only survive if they're getting hit often.
    """
    def __init__(self, a: Program, cache_size: int = 256, key_len: int = 8):
        name = f"memo({a.name})"
        length = a.length + 3.0
        super().__init__(name=name, length=length,
                         parents=(a,), combinator="memo")
        self.a = a
        self.cache_size = cache_size
        self.key_len = key_len
        self._cache: dict = {}
        self._order: list = []

    def predict(self, ctx: bytes) -> Counter:
        key = ctx[-self.key_len:]
        if key in self._cache:
            return self._cache[key]
        result = self.a.predict(ctx)
        self._cache[key] = result
        self._order.append(key)
        if len(self._order) > self.cache_size:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return result


class Mixed(Program):
    """N-way mixture: weighted sum of arbitrary parent predictions. Generalizes
    binary Composed. Weights default to uniform; sum is normalised at predict.
    """
    def __init__(self, parents: tuple[Program, ...], weights: tuple[float, ...] | None = None):
        if len(parents) < 2:
            raise ValueError("Mixed needs ≥2 parents")
        if weights is None:
            weights = tuple(1.0 / len(parents) for _ in parents)
        if len(weights) != len(parents):
            raise ValueError("weights/parents length mismatch")
        name = f"mix({','.join(p.name[:12] for p in parents)})"
        length = sum(p.length for p in parents) + 5.0 + 0.5 * len(parents)
        super().__init__(name=name, length=length,
                         parents=tuple(parents), combinator="mix")
        self.ps = parents
        self.ws = weights

    def predict(self, ctx: bytes) -> Counter:
        out = Counter()
        for p, w in zip(self.ps, self.ws):
            d = p.predict(ctx)
            tot = sum(d.values()) or 1.0
            for k in range(256):
                out[k] += w * (d.get(k, 0.01) / tot)
        return out


COMBINATORS = ["compose", "branch", "abstract", "recur", "gate", "memo", "mix"]


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
                elif combinator == "recur":
                    child = Recurred(a)
                elif combinator == "gate":
                    triggers = random.sample(range(256),
                                              k=random.randint(2, 8))
                    child = Gated(a, b, trigger_set=frozenset(triggers))
                elif combinator == "memo":
                    child = Memoized(a)
                elif combinator == "mix":
                    n_ps = random.randint(3, min(5, len(parents)))
                    ps = tuple(random.sample(parents, n_ps))
                    child = Mixed(ps)
                else:
                    continue
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

    def abstract_phase(self, scan_top: int = 50, min_count: int = 3,
                       max_lift: int = 3):
        """V5 wake-sleep: scan the top `scan_top` programs and lift any
        sub-program (i.e., a parent in their lineage) that appears in ≥
        `min_count` of them into a Memoized primitive.

        Equivalence is by program name (which is deterministic from
        structure + parameters), so two `compose(a,b,alpha=0.5)` programs
        are the same sub-tree but `compose(a,b,alpha=0.7)` is different.
        Caps lifts per call so we don't pour the whole library back in
        as primitives in one pass.

        Returns a list of newly-lifted Memoized primitives.
        """
        from collections import Counter as _C
        post = self.posterior()
        ranked = sorted(self.programs,
                        key=lambda p: post.get(p.name, 0.0), reverse=True)[:scan_top]
        # Count how many top-N programs reference each ancestor by name.
        # Walking the lineage means: for each ranked program, walk its parents,
        # grandparents, etc. and tally each unique ancestor once per ranked program.
        usage = _C()
        for prog in ranked:
            seen = set()
            stack = list(prog.parents)
            while stack:
                anc = stack.pop()
                if anc.name in seen:
                    continue
                seen.add(anc.name)
                usage[anc.name] += 1
                stack.extend(anc.parents)
        # Candidates: appear in ≥ min_count ranked programs, NOT already a primitive.
        prog_by_name = {p.name: p for p in self.programs}
        candidates = [(name, count) for name, count in usage.most_common()
                      if count >= min_count
                      and name in prog_by_name
                      and prog_by_name[name].combinator != "primitive"
                      and not name.startswith("memo(")]
        lifted: list[Program] = []
        for name, _ in candidates[:max_lift]:
            target = prog_by_name[name]
            wrapped = Memoized(target)
            # mark the wrapper as a primitive so prune() keeps it forever
            wrapped.combinator = "primitive"
            wrapped.length = max(target.length - 4.0, 6.0)  # cheaper to use, by construction
            wrapped.name = f"abstracted({name})"
            if wrapped.name not in self.log_weights:
                self.add(wrapped)
                lifted.append(wrapped)
        return lifted


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