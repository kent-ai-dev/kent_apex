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
