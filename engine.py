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
    """Total ignorance: every token equally likely.

    `vocab_size` parameterises the alphabet so the same primitive can serve
    as the architectural Background in V10's Toplevel mixture for any
    tokenisation (bytes, BPE, word). Default 256 = bytes.
    """
    def __init__(self, vocab_size: int = 256):
        super().__init__(name=f"uniform-{vocab_size}" if vocab_size != 256 else "uniform",
                         length=8.0)
        self.vocab_size = vocab_size

    def predict(self, ctx: bytes) -> Counter:
        # back-compat: pickles saved before vocab_size was a field
        # (V1-V9) deserialise without the attribute. Default to 256.
        vs = getattr(self, "vocab_size", 256)
        return Counter({b: 1.0 for b in range(vs)})


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


class Verified(Program):
    """V26 reasoning combinator: runs N parent programs on the same context.
    If ≥`threshold` of them agree on the top predicted byte, emit a
    sharpened distribution over that byte. If they disagree, emit a
    flat uniform — feeding into V10's Toplevel Bayesian mixture, which
    will then weight Background higher (cross-program disagreement is
    itself an OOD signal). Couples Verify to the refusal invariant.
    """
    def __init__(self, parents: tuple[Program, ...], threshold: int = 2):
        if len(parents) < 2:
            raise ValueError("Verified needs ≥2 parents")
        threshold = max(2, min(threshold, len(parents)))
        name = f"verify({','.join(p.name[:10] for p in parents)}|{threshold})"
        length = sum(p.length for p in parents) + 4.0 + 0.5 * len(parents)
        super().__init__(name=name, length=length,
                         parents=tuple(parents), combinator="verify")
        self.ps = parents
        self.threshold = threshold

    def predict(self, ctx: bytes) -> Counter:
        tops: list[int] = []
        for p in self.ps:
            d = p.predict(ctx)
            if not d:
                continue
            tops.append(max(d.items(), key=lambda x: x[1])[0])
        if not tops:
            return Counter({b: 1.0 for b in range(256)})
        # find any byte that ≥threshold parents agree on
        from collections import Counter as _C
        votes = _C(tops)
        best_b, best_n = votes.most_common(1)[0]
        if best_n >= self.threshold:
            # confident emission: sharp distribution over best_b
            out = Counter({b: 0.01 for b in range(256)})
            out[best_b] = 10.0
            return out
        # disagreement: return flat distribution. Toplevel mixture sees
        # this as low information (~uniform) and Background dominates.
        return Counter({b: 1.0 for b in range(256)})


class Searched(Program):
    """V26 reasoning combinator: beam search over k continuations from
    parent `p`, scored by likelihood under `scorer`. Returns the
    distribution at the best-scoring continuation's last position.
    For look-ahead reasoning without transformer attention.

    Beam is small by default (k=4) — Search is *expensive* and meant
    to be rare. Latency budget per v2.md V26: ≤5× direct prediction.
    """
    def __init__(self, p: Program, scorer: Program, k: int = 4,
                 horizon: int = 3):
        name = f"search({p.name[:12]}|k={k},h={horizon})"
        length = p.length + scorer.length + 6.0
        super().__init__(name=name, length=length,
                         parents=(p, scorer), combinator="search")
        self.p, self.scorer = p, scorer
        self.k, self.horizon = k, horizon

    def predict(self, ctx: bytes) -> Counter:
        d0 = self.p.predict(ctx)
        if not d0:
            return Counter({b: 1.0 for b in range(256)})
        # beam: take top-k starting bytes
        ranked = sorted(d0.items(), key=lambda x: -x[1])[:self.k]
        best_score = float("-inf")
        best_dist: Counter | None = None
        import math as _m
        for first_b, _w in ranked:
            cur = bytearray(ctx)
            cur.append(first_b)
            score = 0.0
            for _ in range(self.horizon - 1):
                sd = self.scorer.predict(bytes(cur))
                stot = sum(sd.values()) or 1.0
                top_b = max(sd.items(), key=lambda x: x[1])[0]
                score += _m.log(max(sd.get(top_b, 1e-9) / stot, 1e-12))
                cur.append(top_b)
            # final distribution at this beam's last position
            final_d = self.scorer.predict(bytes(cur))
            if score > best_score:
                best_score = score
                best_dist = final_d
        return best_dist or d0


COMBINATORS = ["compose", "branch", "abstract", "recur", "gate", "memo",
               "mix", "verify"]


class LLMPrimitive(Program):
    """V25 — the one authorised neural intrusion (per plans/v2.md §3 V25).

    A small local language model exposed as one Program in the library.
    Receives a context (bytes), returns a byte distribution. Its weight
    is updated by the standard Bayesian rule; if the LLM consistently
    predicts well, it dominates; if not, it gets pruned. Refusal still
    comes from V10's Toplevel mixture — the LLM is *one voice in the
    choir, not the conductor* (per spec).

    Constraints (these justify the §0 prohibition exception):
      - local inference only; no remote API calls
      - small model (default gpt2 = 124M); document the choice
      - deterministic top-k truncation for inspectability
      - cache by context-tail for repeated lookups (LRU bounded)

    Lazy initialisation: the heavyweight model isn't loaded until the
    first predict() call, so import of engine.py stays cheap.
    """
    def __init__(self, model_name: str = "gpt2",
                 ctx_window: int = 64, top_k: int = 64,
                 cache_size: int = 1024,
                 length: float = 8.0):
        # Solomonoff prior is -length*ln2. A 124M-param transformer has
        # an astronomical Kolmogorov complexity in theory, but in
        # practice the *interface* it presents to the library is a
        # single predict(ctx) function — same as any primitive. Length
        # here is the description length of "use the LLM at this URL",
        # not the model weights themselves. Default 8.0 matches the
        # other primitives so the LLM can win on likelihood, not lose
        # by prior. Configurable for ablations.
        super().__init__(name=f"llm({model_name})", length=length)
        self.model_name = model_name
        self.ctx_window = ctx_window
        self.top_k = top_k
        self.cache_size = cache_size
        self._cache: dict = {}
        self._cache_order: list = []
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.eval()
        self._torch = torch

    def predict(self, ctx: bytes) -> Counter:
        # cache check first (cheap)
        key = ctx[-self.ctx_window:] if ctx else b""
        if key in self._cache:
            return self._cache[key]
        self._ensure_loaded()

        torch = self._torch
        # decode bytes → string for tokenizer; fall back gracefully on
        # invalid UTF-8 (which is common in byte-level training)
        text = key.decode("utf-8", errors="replace") if key else ""
        if not text:
            # nothing to condition on — return a softened uniform
            return Counter({b: 1.0 for b in range(256)})

        with torch.no_grad():
            enc = self._tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=512)
            out = self._model(**enc)
            logits = out.logits[0, -1, :]   # last-position distribution
            # top-k for inspectability
            top = torch.topk(logits, k=min(self.top_k, logits.shape[-1]))
            top_tokens = top.indices.tolist()
            top_probs = torch.softmax(top.values, dim=-1).tolist()

        # marginalise tokens → bytes. each token's text is decoded; the
        # FIRST byte of each token gets that token's probability mass.
        # This is approximate but consistent across the library (every
        # other program also emits byte-level distributions).
        byte_dist: dict[int, float] = {b: 1e-6 for b in range(256)}
        for tok_id, p in zip(top_tokens, top_probs):
            try:
                tok_str = self._tokenizer.decode([tok_id])
                tok_bytes = tok_str.encode("utf-8", errors="replace")
                if tok_bytes:
                    byte_dist[tok_bytes[0]] = byte_dist.get(tok_bytes[0], 0.0) + p
            except Exception:
                continue
        result = Counter(byte_dist)

        # LRU insert
        self._cache[key] = result
        self._cache_order.append(key)
        if len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return result

    def __getstate__(self):
        # don't pickle the model — re-load on demand after unpickling
        st = self.__dict__.copy()
        st["_model"] = None
        st["_tokenizer"] = None
        st["_torch"] = None
        st["_cache"] = {}
        st["_cache_order"] = []
        return st


class BlendedInference:
    """V25 finding workaround: the Bayesian competition under-weights the
    LLMPrimitive because byte-level n-gram-1 wins per-byte likelihood at
    our train scale. The architecture doesn't grant the LLM enough
    posterior weight to influence generation.

    BlendedInference is an *inference-time* hybrid that mixes the
    Bayesian library's predict() with an explicit LLM's predict() at a
    fixed blend ratio α. This is NOT a Bayesian update — it's an
    engineered generation path. Refusal, compression, and audit trail
    still come from the underlying Bayesian library (the LLM is bypassed
    for those). Generation goes through the blend.

    Audit trail per call: the byte distribution is `α * LLM_dist + (1-α)
    * library_dist`, both inspectable. The LLM's contribution is bounded
    and reportable; the symbolic library still constrains outputs.

    Use case: making the architecture *generate* coherent text after
    V20+V25's gate-fail honest finding that the Bayesian competition
    doesn't surface the LLM. The product positioning stays "calibrated
    refusal" (Toplevel) — this is for when generation actually matters
    (V18 endpoint, V30 narrow-domain).
    """
    def __init__(self, library, llm: "LLMPrimitive", alpha: float = 0.5,
                 vocab_size: int = 256):
        self.library = library
        self.llm = llm
        self.alpha = alpha
        self.vocab_size = vocab_size

    @property
    def programs(self):
        # surface what's behind us so inspection tools work
        return list(self.library.programs) + [self.llm]

    def predict(self, ctx: bytes) -> Counter:
        lib_d = self.library.predict(ctx)
        llm_d = self.llm.predict(ctx)
        lib_tot = sum(lib_d.values()) or 1.0
        llm_tot = sum(llm_d.values()) or 1.0
        out = Counter()
        for b in range(self.vocab_size):
            out[b] = (self.alpha * (llm_d.get(b, 0.0) / llm_tot)
                      + (1.0 - self.alpha) * (lib_d.get(b, 0.0) / lib_tot))
        return out

    def entropy(self, ctx: bytes) -> float:
        d = self.predict(ctx)
        tot = sum(d.values()) or 1.0
        import math as _m
        h = 0.0
        for v in d.values():
            p = v / tot
            if p > 0:
                h -= p * _m.log2(p)
        return h

    def refusal_score(self, ctx: bytes) -> float:
        # delegate refusal to underlying Toplevel if present
        return getattr(self.library, "refusal_score", lambda c: 0.0)(ctx)


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

    def update(self, ctx: bytes, actual: int, lr: float = 0.03,
               temperature: float | None = None,
               max_delta: float | None = None):
        """Bayesian update: programs that predicted `actual` better gain weight.

        V22 (when `temperature` is set): replaces the incremental learning-rate
        heuristic with a PAC-Bayes tempered posterior step. The accumulated
        log-weight tracks `(1/T) * cumulative_log_likelihood` plus the
        Solomonoff prior baked in by `add()`. T=1 sharpens the posterior;
        larger T (higher temperature) flattens it. The `lr` path remains
        the back-compat default for V1-V21.

        Mathematical equivalence: setting `temperature = 1/lr` reproduces
        the lr-based behaviour exactly (so `lr=0.03` ≈ `temperature ≈ 33`).

        V16: provenance.default_store records non-trivial deltas if init'd.
        """
        try:
            from provenance import default_store as _prov_store
        except Exception:
            _prov_store = None

        coef = (1.0 / temperature) if temperature is not None else lr
        for p in self.programs:
            dist = p.predict(ctx)
            total = sum(dist.values()) or 1.0
            prob = (dist.get(actual, 0.01) + 1e-9) / total
            delta = coef * math.log(prob)
            # V22 augment: per-step delta cap. Tempering alone can't keep
            # ESS≥10 in our regime — accumulated log-likelihood gap over
            # thousands of bayes-steps becomes astronomical. Capping the
            # per-step magnitude is a structural anti-runaway: no single
            # update can shift any program's log-weight by more than
            # `max_delta` nats. Documented in LOG.md as a v2.md V22
            # extension beyond the spec.
            if max_delta is not None:
                if delta > max_delta:
                    delta = max_delta
                elif delta < -max_delta:
                    delta = -max_delta
            self.log_weights[p.name] += delta
            if _prov_store is not None:
                _prov_store.record(p.name, ctx, delta)
        self._renormalize()

    def effective_sample_size(self) -> float:
        """V22 ESS = exp(entropy(posterior)). Counts how many programs
        effectively contribute to predictions. Mode-collapse → ESS ≈ 1.
        Healthy diverse posterior over an N-program library → ESS up to N.
        """
        post = self.posterior()
        h = 0.0
        for w in post.values():
            if w > 1e-12:
                h -= w * math.log(w)
        return math.exp(h)

    def top_k_dpp(self, k: int = 8, n_probes: int = 16,
                  diversity_weight: float = 0.5) -> list:
        """V22 DPP-style diverse top-k selection. Greedy approximation:

          1. Compute each program's prediction signature on `n_probes` fixed
             canonical contexts (the same byte sequences used every call so
             signatures are comparable across calls).
          2. Pick the highest-posterior-weight program first.
          3. Iteratively pick the program that maximises
                 weight(p) * (1 - max_cosine_sim(p, already_picked))
             — penalising near-duplicate predictors. `diversity_weight`
             scales the diversity term.

        Falls back to plain top-k if `len(self.programs) < k`.
        Replaces `top_programs` for diverse-ensemble inference; the original
        `top_programs` (pure argsort) is kept for comparison.
        """
        if len(self.programs) <= k:
            return self.top_programs(k=k)

        post = self.posterior()
        # canonical probes — fixed across calls, deterministic
        probes = [b"the ", b" of ", b" and", b"\n\n", b"\nT", b". I", b" 19",
                  b" 20", b"a", b" t", b"in ", b" co", b" pr", b"th", b" h",
                  b"#! "][:n_probes]
        # build signatures
        import math as _m
        sigs: dict[str, list[float]] = {}
        for p in self.programs:
            s: list[float] = []
            for probe in probes:
                d = p.predict(probe)
                tot = sum(d.values()) or 1.0
                # compress 256-dim to 16 bins for cheaper cosine
                bins = [0.0] * 16
                for b, v in d.items():
                    bins[b // 16] += v / tot
                s.extend(bins)
            sigs[p.name] = s

        def cos_sim(a: list[float], b: list[float]) -> float:
            num = sum(x * y for x, y in zip(a, b))
            da = _m.sqrt(sum(x * x for x in a)) or 1e-9
            db = _m.sqrt(sum(y * y for y in b)) or 1e-9
            return num / (da * db)

        # greedy DPP-style selection
        ranked = sorted(self.programs,
                        key=lambda p: post.get(p.name, 0.0), reverse=True)
        if not ranked:
            return []
        picked = [ranked[0]]
        candidates = ranked[1:]
        while len(picked) < k and candidates:
            best_score = -1.0
            best_idx = 0
            for i, c in enumerate(candidates):
                w = post.get(c.name, 0.0)
                max_sim = max(cos_sim(sigs[c.name], sigs[p.name])
                              for p in picked)
                score = w * (1.0 - diversity_weight * max_sim)
                if score > best_score:
                    best_score = score
                    best_idx = i
            picked.append(candidates.pop(best_idx))
        return picked

    def decay(self, factor: float = 0.99):
        """V7: shrink all log-weights toward zero so the posterior stays
        responsive to non-stationary data and doesn't astronomically
        concentrate on one program. Call every ~N bytes during training.
        """
        for k in self.log_weights:
            self.log_weights[k] *= factor

    def replay(self, buffer: list[tuple[bytes, int]], lr: float = 0.03):
        """V7: re-run Bayesian updates on a sample of past (ctx, actual)
        pairs. Counters catastrophic forgetting when training on shifting
        distributions and gives early-formed programs a chance to recover.
        """
        for ctx, actual in buffer:
            self.update(ctx, actual, lr=lr)

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
                elif combinator == "verify":
                    n_ps = random.randint(3, min(5, len(parents)))
                    ps = tuple(random.sample(parents, n_ps))
                    threshold = max(2, n_ps // 2 + 1)  # simple majority
                    child = Verified(ps, threshold=threshold)
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


class Toplevel:
    """V10 architectural refusal invariant: a Bayesian mixture between the
    trained library (in-distribution) and a literal `UniformPrimitive`
    (the Background reference). `refusal_score(ctx)` is the posterior
    probability that ctx came from Background — a measured quantity, not
    a hand-tuned threshold.

    Stateless in the sense that the mixture posterior is computed fresh
    per call from a sliding evidence window over the last
    `evidence_window` bytes of ctx. The inner `Library` carries all the
    long-term training; Toplevel only mediates "is this input in the
    distribution we trained on?"

    KN's continuation-distribution base case (the V3 regression) doesn't
    matter here: on OOD, KN gives confidently-wrong predictions with
    low likelihood on the actual byte. Background's `1/vocab_size`
    becomes higher than KN's likelihood for OOD bytes, so the posterior
    shifts to Background as evidence accumulates.
    """
    def __init__(self, in_dist: "Library", vocab_size: int = 256,
                 evidence_window: int = 32, prior_log_in: float = 0.0,
                 prior_log_bg: float = 0.0):
        self.in_dist = in_dist
        self.vocab_size = vocab_size
        self.evidence_window = evidence_window
        self.background = UniformPrimitive(vocab_size)
        self.prior_log_in = prior_log_in
        self.prior_log_bg = prior_log_bg

    @property
    def programs(self):
        return self.in_dist.programs

    @property
    def log_weights(self):
        return self.in_dist.log_weights

    def _accumulate_evidence(self, ctx: bytes) -> tuple[float, float]:
        """Walk the last `evidence_window` bytes, accumulating log-likelihood
        of each observed byte under in_dist vs Background. Returns
        (log_w_in, log_w_bg) including the priors.
        """
        window = ctx[-self.evidence_window:] if ctx else b""
        log_w_in = self.prior_log_in
        log_w_bg = self.prior_log_bg
        bg_prob = 1.0 / max(self.vocab_size, 1)
        log_bg_per = math.log(bg_prob)
        for i in range(1, len(window)):
            sub = window[:i]
            actual = window[i]
            dist = self.in_dist.predict(sub)
            total = sum(dist.values()) or 1.0
            p_in = max(dist.get(actual, 0.0) / total, 1e-12)
            log_w_in += math.log(p_in)
            log_w_bg += log_bg_per
        return log_w_in, log_w_bg

    def _mixture_weights(self, ctx: bytes) -> tuple[float, float]:
        log_w_in, log_w_bg = self._accumulate_evidence(ctx)
        m = max(log_w_in, log_w_bg)
        e_in = math.exp(log_w_in - m)
        e_bg = math.exp(log_w_bg - m)
        z = e_in + e_bg
        return e_in / z, e_bg / z

    def predict(self, ctx: bytes) -> Counter:
        w_in, w_bg = self._mixture_weights(ctx)
        p_in = self.in_dist.predict(ctx)
        total_in = sum(p_in.values()) or 1.0
        bg_per = 1.0 / max(self.vocab_size, 1)
        out = Counter()
        for k in range(self.vocab_size):
            out[k] = w_in * (p_in.get(k, 0.01) / total_in) + w_bg * bg_per
        return out

    def refusal_score(self, ctx: bytes) -> float:
        """P(Background | ctx). High = "this input doesn't look like training";
        used as the architectural refusal probability. Default τ = 0.5.
        """
        _, w_bg = self._mixture_weights(ctx)
        return w_bg

    def update(self, ctx: bytes, actual: int, lr: float = 0.03):
        """Delegate per-program Bayesian update to the inner Library. The
        Toplevel itself is stateless across training — the mixture is
        re-computed per prediction from sliding-window evidence.
        """
        self.in_dist.update(ctx, actual, lr=lr)

    def decay(self, factor: float = 0.99):
        self.in_dist.decay(factor=factor)

    def replay(self, buffer, lr: float = 0.03):
        self.in_dist.replay(buffer, lr=lr)

    def grow(self, n_children: int = 4):
        return self.in_dist.grow(n_children=n_children)

    def prune(self, max_size: int = 200):
        return self.in_dist.prune(max_size=max_size)

    def abstract_phase(self, **kwargs):
        return self.in_dist.abstract_phase(**kwargs)

    def top_programs(self, k: int = 8):
        return self.in_dist.top_programs(k=k)

    def posterior(self):
        return self.in_dist.posterior()

    def entropy(self, ctx: bytes) -> float:
        return self.in_dist.entropy(ctx)


def fresh_library(rich: bool = True, toplevel: bool = False,
                  vocab_size: int = 256):
    """Seed library with primitives.

    With rich=True (default since V3), also adds Kneser-Ney n-grams and a
    skip-gram predictor — primitives that capture stronger statistical
    structure than plain add-k n-grams. Set rich=False for the V1/V2
    baseline behaviour.

    With toplevel=True (V10+), wraps the Library in a Toplevel Bayesian
    mixture against a UniformPrimitive Background; refusal becomes a
    measured posterior probability, not an entropy threshold.
    """
    lib = Library()
    lib.add(UniformPrimitive(vocab_size=vocab_size))
    lib.add(RepeatPrimitive())
    for n in (1, 2, 3, 4, 5, 6):
        lib.add(NGramPrimitive(n))
    if rich:
        for n in (3, 4, 5):
            lib.add(KneserNeyNGram(n))
        lib.add(SkipGramPredictor(k=2, depth=3))
        lib.add(SkipGramPredictor(k=3, depth=3))
    if toplevel:
        return Toplevel(lib, vocab_size=vocab_size)
    return lib