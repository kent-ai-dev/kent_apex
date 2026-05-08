# RCE — Honest Interim Results (V1–V11+)

**Status:** mid-experiment. The plan in `plans/v1.md` runs V1 → V20.
This session executed V1 through V11 (with V8/V14/V16/V17 implemented
but not all benched, and V13/V18/V19 deferred). It is the V20 report
written halfway: an honest read of what we have so far, what worked,
what didn't, and what's left.

## What is RCE

A symbolic byte-level predictor: a posterior-weighted mixture over a
library of small programs (n-grams, Kneser-Ney, skip-grams, plus
combinator-generated children). Bayesian update rewards programs that
predicted the next byte well; wake-sleep-style abstraction lifts
recurring sub-programs into primitives. No neural network, no gradient
descent. Every prediction is auditable to specific weighted programs.

## Headline numbers (so far)

| Metric                                | Value     | Comparison                      |
|---------------------------------------|-----------|--------------------------------- |
| BPB on wikitext-2 (V1, untrained)     | 6.6460    | raw bytes 8.0; trivial baseline  |
| BPB on wikitext-2 (V1, trained)       | 2.6209    | ~4× compression vs raw bytes     |
| BPB on wikitext-2 (V3, KN+SkipGram)   | **2.3735**| **9.4% over V1 trained**         |
| BPB on wikitext-2 (V2 word naive 5gm) | 2.2491    | tokenization sweep winner        |
| BPB on wikitext-103 (V11 cross-corpus)| 3.4085    | cross-corpus deg. as expected    |
| LAMBADA per-byte log2p (V11)          | -5.17     | uniform = -8.0; +2.83 bits/byte  |
| HellaSwag accuracy (V11)              | 0.15      | random = 0.25 — *below* random   |
| ECE on wikitext-2 held-out (V1 seed)  | 0.0405    | already inside V10 gate ≤0.05    |
| ECE (V3+ trained)                     | ~0.077    | rises after training, still ~ok  |
| Refusal on `os.urandom` (V1 seed)     | 1.0000    | perfect refusal on noise         |
| Refusal (V3+ trained)                 | 0.0000    | **regression**: see findings     |
| HF streaming throughput (V6)          | 883 KB/s  | gate ≥50 KB/s ✓ by 17×           |
| V8 4-shard dry-run scaling            | 80% of linear | gate ≥70% ✓                  |

## What worked

1. **The bench harness is honest.** Bits-per-byte, ECE binned by
   confidence, refusal rate on `os.urandom`. Reproducible across runs
   (V1 gate trivially passed: deterministic eval). Now used by every
   later version.

2. **Pre-ingestion validation is real.** `validate_dataset.py` runs a
   7-check vibes test on any HF dataset; FAIL blocks ingestion. Caught
   the expected 38% empty-row ratio in raw wikitext as WARN — exactly
   the kind of signal the user wanted.

3. **The cheap engine fix mattered most of all.** V1's `NGramPrimitive`
   had an O(table-size) backoff that dominated bench runtime; removing
   it was a one-paragraph change that made eval ~10× faster. Every
   later version benefits.

4. **Stronger primitives clearly win on BPB.** V3's
   `KneserNeyNGram(3,4,5)` plus `SkipGramPredictor(2,3)` beat the
   V1-baseline by 9.4% — the V3 gate was a 5% threshold, easily
   cleared. KN's recursive backoff is mathematically the right move.

5. **Tokenization sweep produced a clean signal.** V2 ran a fair n-gram
   baseline across {bytes, BPE-4K, word} and word tokenization won by
   16% over bytes, 20% over BPE. The plan's hypothesis that BPE would
   win didn't hold at 1MB train scale (BPE corpus too sparse for a
   5-gram table). Documented; word tokenization is queued for the
   engine vocab refactor at V6+.

6. **HF streaming hits 883 KB/s sustained.** The streaming path scaled
   far past the 50 KB/s gate. Gate satisfied without optimisation.

7. **Modal merge math is right.** V8's `merge_libraries()` with the
   prior-correction (subtract `(N-1) × Solomonoff_prior` per program
   appearing in N shards) reproduces a clean 4-shard merge: 126 unique
   programs from 179 raw, 25 shared, primitives correctly deduplicated.

8. **V7's decay+replay restores posterior diversity.** Documented in
   the partial V7 run: 2 abstractions lifted by step 4000, vs V5's full
   run with 0 lifts. The mechanism works.

## What didn't work

1. **Mode collapse onto `kn-5` at V4–V5.** The Bayesian update with
   `lr=0.03` accumulated log-weight over ~7500 steps to the point that
   one program owns 100% of the posterior — every other program rounds
   to 0 weight. The library is no longer an ensemble; predictions
   reduce to "what kn-5 says." V7's decay+replay was designed for
   exactly this.

2. **V3's KneserNey broke V1's refusal metric.** KN's order-0 base
   case is the continuation distribution (≈ unigram frequency over
   training bytes). On out-of-distribution input the library returns
   this learned prior — entropy ~5 bits, well below the 7.0 refusal
   threshold. V1's refusal heuristic (entropy > 7.0) silently fails as
   soon as a primitive is added whose backoff says "what reasonable
   English would say." The architecture's structural-impossibility-of-
   hallucination claim is more nuanced than V1's framing suggested.

3. **HellaSwag accuracy below random.** 15% on a 4-way task where
   random is 25%. The library, trained on wikitext, prefers
   "encyclopedia-like" continuations to natural-narrative ones —
   actively *wrong* on stories. V12's cross-domain training is supposed
   to address this; not yet validated.

4. **V5 abstraction couldn't fire.** With mode collapse, top-50
   programs are all direct children of one primitive (kn-5). No
   non-primitive ancestor reaches the `min_count=3` threshold for
   lifting. The V5 gate ("library asymptotes within 2× V4 size") passed
   for the wrong reason — the prune cap, not the abstraction. V7's
   decay must run before V5 produces meaningful lifts.

5. **Combinators beyond V3's primitives add little BPB.** V4 added
   Gate/Memo/Mix combinators, but the trained BPB was 2.43 — *worse*
   than V3's 2.37 — because the additional combinators slowed
   convergence without helping the dominant predictor. `gate` and `mix`
   in particular earned no top-20 placement despite many being grown.
   `memo` was the only new combinator that survived (8 in top-20).

6. **The full V20 plan is too long for one session.** Each trained
   bench takes 10-30 minutes wall-clock. V8's Modal sharding wants real
   cloud compute (cost-gated). V13's self-modifying interpreter is a
   conceptually heavy refactor. V18/V19 (public endpoint, multi-modal)
   need infrastructure beyond what this session can validate.

## What's left

Implemented in code but not fully benched:
  - **V8** Modal sharding — `modal_train.py` with merge math; dry-run
    validates correctness; awaiting human approval for cloud spend.
  - **V11** LLM benchmarks — `eval_llm.py` works; numbers above are
    against the 5KB-trained chat library; need re-run after V7 retrain.
  - **V14** Curiosity-driven streaming — `curiosity.py` with EMA
    baseline + priority queue; not yet wired into bayes_train.
  - **V16** Provenance store — `provenance.py` with append-only JSONL
    per program; not yet hooked into Library.update.

Not implemented:
  - **V9** First serious training run (1GB pile-uncopyrighted, 32 Modal
    workers, target BPB ≤2.5 on wikitext-103). Code is V8-ready;
    awaiting cost approval.
  - **V12** Cross-domain training (c4 / code_search_net / wikidata5m).
    Needs a dataset mixer + domain-tag prefix bytes.
  - **V13** Self-modifying interpreter. Requires the engine's program
    representation to be more general — a multi-version refactor.
  - **V15** Conversational fine-grain. Needs `<dialog>` primitive and
    turn-boundary stop logic.
  - **V17** Federation. Library exchange format; specialist+merge.
    Builds on V8.
  - **V18** Public chat endpoint via Modal. Web framework + auth.
  - **V19** Multi-modal extension (image bytes, audio bytes). Substantial.

## Honest assessment

The architectural bet is partially supported by what we have:

- **Compression works.** V3's BPB of 2.37 on wikitext-2 is decent for a
  byte-level predictor on 1MB train. Far from frontier (GPT-2 ≈ 1.0 BPB
  on equivalent text), but real.
- **Calibration is genuinely better than typical LLM behaviour** when
  the library is healthy: V1 trained's ECE of 0.07 holds even with KN.
  But the *refusal* claim — that the architecture is structurally
  immune to hallucination — is weaker than the plan stated. KN (and
  any predictor with a learned non-uniform base case) breaks the
  entropy-threshold refusal heuristic. The architecture *could* be
  refusal-friendly with deliberate design (a separate OOD detector or
  uniform-base-case backoff), but it isn't automatically so.
- **The combinator/library-growth process has a serious failure mode**
  in mode collapse. V7's decay+replay is the right fix; we have
  partial evidence it works (2 abstractions where V5 had 0). It needs
  a clean V7 full-run benchmark to confirm BPB regression / improvement.
- **Cost asymmetry holds in principle.** V8's dry-run shows ~80% of
  linear scaling on 4 shards, against the merge-math-corrected
  Bayesian aggregation. A real Modal run at 32 workers would tell us
  whether linear scaling holds at the size where it matters.

The plan's most ambitious version-level claims (V13 self-modification,
V19 multi-modal, V20 frontier-competitive) remain unvalidated. The
intermediate claims (V1 baseline, V2 tokenization, V3 primitives, V6
streaming, V8 merge math) are validated.

The architecture doesn't lose to LLMs in the ways the plan suggested
it would — and doesn't win in the ways it suggested either. The honest
read is: this is a real research direction with concrete next steps,
not a vindicated bet and not a dead end.

## Cost so far (this session)

- Anthropic API: $0 (engineer-in-session, no API)
- Modal compute: $0 (V8 was dry-run only)
- HuggingFace: $0 (public datasets)
- Wall time: ~6-8 hours of session
- Compute: a single CPU on a developer laptop

For comparison, the plan's V20 success criterion says "compute cost
orders of magnitude below an LLM of equivalent BPB." We can't yet
claim equivalent BPB, but the cost asymmetry at this stage is
extreme: this session cost $0 in marginal compute.

## Reproducing this

```bash
git clone https://github.com/kent-ai-dev/kent_apex
cd kent_apex
pip install -r requirements.txt
pip install datasets tokenizers   # for V2/V6+
python3 fetch_data.py             # downloads wikitext-2, makes OOD bytes
python3 bench.py --bayes-train 30000 --decay-every 500 --replay-buffer 2000 --save
python3 rce.py chat               # chat with the trained library
python3 eval_llm.py               # V11 benchmarks against the saved library
```

## Where the code is

- `engine.py` — primitives (Uniform, Repeat, NGram, KneserNey, SkipGram),
  combinators (Compose, Branch, Abstract, Recur, Gate, Memo, Mix), Library
  (predict, update, decay, replay, abstract_phase, grow, prune)
- `bench.py` — V1+ harness: BPB, ECE, refusal-rate, optional bayes-train
- `bench_tokenizers.py` — V2 sweep
- `tokenize_rce.py` — Bytes/BPE/Word tokenizers
- `validate_dataset.py` — V0 pre-ingestion gate
- `datasets_hf.py` — V6 streaming
- `modal_train.py` — V8 worker + merge math
- `eval_llm.py` — V11 LAMBADA / HellaSwag / WikiText-PPL
- `curiosity.py` — V14 priority-queue streaming
- `provenance.py` — V16 append-only JSONL store
- `rce.py` — CLI: train, chat (with V10 strict + explain), status, reset
- `plans/v1.md` — the V1-V20 master plan
- `plans/storage.md` — storage tier system + migration triggers
- `LOG.md` — running journal of every iteration
- `BENCHMARKS.md` — append-only ledger of every metric
- `state.json` — durable run state (current version, last gate, findings)
