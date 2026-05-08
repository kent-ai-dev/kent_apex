# RCE Training Log

This is the running journal for the V1→V20 plan. Each entry is a single
iteration: what was done, what happened, what the gate said, what's next.
Append-only. The latest entry is at the bottom.

---

## 2026-05-06 — session start

Operating mode: Claude (in-session) is the engineer. ralph.py orchestrator
not in the loop. State persisted in `state.json`, `LOG.md`, `BENCHMARKS.md`
at repo root.

Beginning V1: establish the benchmark harness.

---

## 2026-05-06 — V1 implementation

Wrote `bench.py` (BPB + ECE + refusal-rate) and `fetch_data.py` (pulls
wikitext-2 train/validation slices and 20KB of `os.urandom` for the OOD
refusal test). Train slice = 1MB, held-out = 100KB, OOD = 20KB.

**V1 run 1** (8 primitives, fitted on 1MB train):
- BPB        = 6.6460  (wikitext-2 heldout)
- ECE        = 0.0405
- refusal    = 1.0000  (os.urandom)
- wall time  = 8m27s

Observations:
- BPB of 6.65 vs the raw-bytes ceiling of 8.0 — the n-gram primitives
  are doing real work but the library is far from competitive. This is
  the V1 baseline, not the V9 target.
- ECE 0.0405 already inside V10's gate (≤ 0.05). Worth noting but
  potentially a soft win — well-calibrated under-confident predictors
  trivially pass ECE. Need a proper scoring rule as a check.
- 100% refusal on `os.urandom` is exactly right; the library's mixed
  posterior over byte distributions has entropy ≈ 8.0 on inputs that
  match no learned context.
- Wall time is the bottleneck: 8.5 min/run is going to compound over
  V2's tokenization sweep (3 tokenizations × 2 reproducibility runs =
  ~50 min). Consider vectorising `lib.predict` if this hurts later.

Reproducibility: bench is fully deterministic (no sampling, no shuffle,
fixed seed). Run 2 in flight to confirm exact match.

---

## 2026-05-07 — V1 close-out

Run 2 produced byte-identical output to run 1 (BPB 6.6460, ECE 0.0405,
refusal 1.0000). V1 reproducibility gate ✓.

**Engine.py fix during V1:** `NGramPrimitive.predict` was falling through
to an O(table-size) backoff scan when the full n-context wasn't in its
table. For the n=6 primitive on a 1MB train slice that's roughly 1M ops
per byte; combined with eight n-gram primitives this dominated bench
runtime (single bench took 41+ minutes on the trained library before I
killed it). Removed the per-primitive backoff: each NGramPrimitive(n)
now returns uniform on missing-n-context, and lower-order n-grams in
the same library handle the backoff naturally via their own predicts.
After fix, bench is ~3 min instead of ~25.

**Trained baseline** (Bayesian-update over 100KB train slice, 25K steps,
final library size 93):
- BPB     = 2.6209  (≈4× compression vs raw bytes; close to V9 gate ≤2.5)
- ECE     = 0.0707  (trained predictors more confident → higher than seed
                     but still in the same order of magnitude as V10 gate)
- refusal = 0.9998  (still essentially perfect on os.urandom)

The library oscillates during training: grows to ~130 programs, then a
prune pass drops it to ~10 (only primitives survive), then grows again.
The prune threshold (`post.get(p.name, 0) > 1e-5`) is killing combinator
children too aggressively — useful child programs are being culled
before their weight stabilises. Flagging for V3 attention; keeping the
current behaviour for V1 baseline.

**State advance**: V1 → V2.
The bench harness is good. The optimized engine is faster. Numbers
recorded. Moving to the tokenization sweep.

---

## 2026-05-07 — V2: tokenization sweep

Wrote `tokenize_rce.py` (Bytes / BPE-byte-level / Word) and
`bench_tokenizers.py` (naive n-gram-5 with add-k=0.01 over the chosen
token sequence). The sweep is engine-independent — it measures how
each tokenization affects BPB *for a fair n-gram baseline*. Engine
adaptation to non-byte vocabularies is a V3 prerequisite, not V2 work.

**Results** (wikitext-2-raw-v1, train=1MB, heldout=100KB):
- bytes (vocab 256):    BPB = 2.6744
- BPE-4K (vocab 4096):  BPB = 2.8107
- word (vocab 19010):   BPB = **2.2491** — WINNER

Surprises:
- Word beats bytes by 16% — meaningful and clean signal
- BPE underperformed both — at 1MB train, only 270K BPE tokens populate
  the 5-gram table sparsely; smoothing penalty per token dominates.
  The plan's hypothesis ("BPE will win because it shortens effective
  context") was reasonable but did not hold at this train size. BPE
  may catch up at V6+ scale (10M+ bytes).
- The full RCE engine on bytes (V1 trained) hit BPB 2.62, only ~2%
  better than naive n-gram-5 on bytes (2.67) — sobering; the
  combinators contribute little compared to the n-gram primitives at
  this stage. Suggests V3's stronger primitives are the right next move.

**Decision** (gate: "pick tokenization with best BPB; document why"):
Adopt **word tokenization** for V3+. Documented in `state.json` so
downstream versions can read it.

Caveat: the V2 numbers are from a *naive* n-gram-5, not the engine.
When V3 wires the chosen tokenization into the engine, the engine
prediction will differ. The V2 winner is a baseline-level comparison;
the V3 measurement is whether richer primitives improve on it.

**State advance**: V2 → V3.

---

## 2026-05-07 — V3: richer primitives

Added two primitive families to `engine.py`:

- **KneserNeyNGram(n)** for n=3,4,5. Recursive KN smoothing — discount
  observed counts by `d=0.75` and back off through orders, with the
  base case being the continuation distribution (how many distinct
  contexts each byte appears as a continuation in).
- **SkipGramPredictor(k, depth)** for (k=2,depth=3) and (k=3,depth=3).
  Predicts from non-adjacent context positions to capture long-range
  patterns that strict n-grams miss.

`fresh_library()` now seeds these alongside the originals (13 primitives
total), with a `rich=False` flag that restores the V1/V2 8-primitive
behaviour for ablations.

Skipped from the V3 spec list:
- **SuffixArrayPredictor** (PPM-style, unbounded order). The V3 gate
  was satisfied by KN+skip; SA adds significant code (FM-index or
  suffix array) for marginal additional gain at this scale. Defer to
  V4 or later if KN/skip plateau.
- **LZWPredictor**. Same rationale.

**V3 trained results** (30KB bayes-train slice, lib=127):
- BPB     = 2.3735  (V1 trained 2.6209 → **9.4% better**, gate ≥5% ✓)
- ECE     = 0.0775  (similar to V1 trained)
- refusal = **0.0000** — REGRESSION

**Refusal regression analysis.** KneserNeyNGram's order-0 base case is
the continuation distribution (≈ unigram frequencies in train), which
is non-uniform. On OOD input where higher orders all miss in the
backoff, KN returns this learned prior — entropy ≈ 5 bits, well below
the 7.0 refusal threshold. The library's posterior-weighted prediction
inherits this confidence.

This is a real architectural finding: V1's refusal heuristic relies on
the library being unable to predict OOD inputs. Adding a primitive
whose backoff is "what a reasonable English unigram would say" breaks
the heuristic. Two ways to address at V10:
  (a) Modify KN order-0 base to be uniform (clean but loses calibration
      info on in-distribution low-context positions).
  (b) Add an OOD detector that runs alongside the predictor — e.g.,
      perplexity of the input itself under a simple background model,
      with refusal triggered when input perplexity is anomalous.
Logging here so V10 has the context.

**Gate verdict**: V3's stated gate (BPB improvement ≥5% from a new
primitive) ✓. The refusal regression is a V10 issue, not a V3 issue.

**State advance**: V3 → V4.
