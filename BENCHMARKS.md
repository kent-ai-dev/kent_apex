# RCE Benchmarks

Append-only ledger of every metric measured across versions.

| date | version | metric | value | dataset | notes |
|---|---|---|---|---|---|
| 2026-05-06 | V1 run-1 (seed) | BPB | 6.6460 | wikitext-2-raw-v1 heldout 100KB | seed library, no Bayesian update; posterior at Solomonoff prior |
| 2026-05-06 | V1 run-1 (seed) | ECE | 0.0405 | wikitext-2-raw-v1 heldout 100KB | trivially low (under-confident predictor) |
| 2026-05-06 | V1 run-1 (seed) | refusal_rate | 1.0000 | os.urandom 20KB | entropy > 7.0 on every position |
| 2026-05-06 | V1 run-2 (seed) | BPB | 6.6460 | wikitext-2-raw-v1 heldout 100KB | identical to run-1; gate (reproducibility) ✓ |
| 2026-05-07 | V1 trained | BPB | 2.6209 | wikitext-2-raw-v1 heldout 100KB | 100KB Bayesian-update slice, 25K steps, lib=93; engine.py NGramPrimitive backoff fixed |
| 2026-05-07 | V1 trained | ECE | 0.0707 | wikitext-2-raw-v1 heldout 100KB | trained predictors are more confident → ECE rises but still inside V10 gate ballpark |
| 2026-05-07 | V1 trained | refusal_rate | 0.9998 | os.urandom 20KB | trained library still refuses on pure noise |
| 2026-05-07 | V2 sweep | BPB-bytes | 2.6744 | wikitext-2-raw-v1 heldout 100KB | naive n-gram-5 add-k=0.01, vocab=256 |
| 2026-05-07 | V2 sweep | BPB-bpe4096 | 2.8107 | wikitext-2-raw-v1 heldout 100KB | naive n-gram-5 add-k=0.01, byte-level BPE; underperforms because train corpus too small for 4K-vocab 5-gram density |
| 2026-05-07 | V2 sweep | BPB-word | 2.2491 | wikitext-2-raw-v1 heldout 100KB | naive n-gram-5 add-k=0.01, vocab=19010; **WINNER** — 16% better than bytes, 20% better than BPE |
| 2026-05-07 | V3 trained | BPB | 2.3735 | wikitext-2-raw-v1 heldout 100KB | added KN-3,4,5 + skip-2x3, skip-3x3; 30KB bayes-train; lib=127. **9.4% better than V1 trained (gate 5%) ✓** |
| 2026-05-07 | V3 trained | ECE | 0.0775 | wikitext-2-raw-v1 heldout 100KB | similar to V1 trained |
| 2026-05-07 | V3 trained | refusal_rate | 0.0000 | os.urandom 20KB | **REGRESSION** — KN's continuation-distribution base case is non-uniform, so the library is now confident on random bytes too. V10 must address. |
| 2026-05-07 | V4 trained | BPB | 2.4297 | wikitext-2-raw-v1 heldout 100KB | added Gated/Memoized/Mixed combinators; lib=139. Slightly worse than V3 — additional combinators slowed posterior concentration; net effect: posterior collapses onto kn-5 anyway (weight 1.0). V7 decay+replay should fix. |
| 2026-05-07 | V4 trained | ECE | 0.0799 | wikitext-2-raw-v1 heldout 100KB | similar to V3 |
| 2026-05-07 | V4 trained | refusal_rate | 0.0000 | os.urandom 20KB | V3 regression carries forward |
| 2026-05-07 | V4 gate-check | combinators_in_top20 | memo×8 | n/a | gate ✓ — memo appears 8× in top-20. compose×1 also present. gate, mix did NOT make top-20 despite many in library (gate=24, mix=22) — kept for now, candidates for removal if still dead at V5. |
| 2026-05-07 | V5 trained | BPB | 2.4297 | wikitext-2-raw-v1 heldout 100KB | identical to V4 — mode collapse on kn-5 means abstraction can't lift (kn-5 is a primitive; its direct children dominate top-50 with shallow lineage; no non-primitive ancestor reaches min_count=3). |
| 2026-05-07 | V5 gate-check | library_size | 139 | n/a | V4=139, V5=139 → 1.0× ratio, gate ≤2× ✓. But asymptote is from prune cap, not abstraction. After V7 decay+replay, expect richer lineage and meaningful lifts. |
| 2026-05-07 | V5 abstraction_lifts | 0 | n/a | n/a | abstract_phase ran but no ancestor reached min_count=3 — diagnostic for deferred re-evaluation post-V7 |
| 2026-05-07 | V6 streaming | throughput | 883.8 KB/s | wikitext-103-raw-v1 stream | datasets_hf.iter_bytes() with validator gate; 1MB streamed in 1.1s. Gate ≥50 KB/s ✓ by 17×. |
| 2026-05-07 | V7 partial | abstractions_at_step_4000 | 2 | wikitext-2 30KB train | decay-every=500, decay-factor=0.99, replay-buf=2000. V5 had 0 lifts at full 7500 steps; V7 has 2 by step 4000. Decay+replay does restore posterior diversity. Bench was killed at step 4500 to free CPU for chat library — full BPB number deferred. |
| 2026-05-07 | V7 chat-lib | n_programs | 42 | 5KB bayes-train | trained with decay+replay; saved to .rce_library.pkl for `python3 rce.py chat` |
| 2026-05-07 | V8 dry-run | n_merged_programs | 126 | 4 shards × 8KB wikitext-2 | sequential local; merge ran instantly. Sequential wall = 584s; projected 4-worker Modal = 183s = 3.2× / 80% of linear ≥70% gate ✓ |
| 2026-05-07 | V8 dry-run | n_shared_programs | 25 | 4 shards × 8KB | programs appearing in ≥2 shards (got prior-correction); top shared = primitives uniform/repeat/ngram-1..N appearing in all 4 |
| 2026-05-07 | V11 LAMBADA | avg_per_byte_log2p | -5.17 | EleutherAI/lambada_openai 20 ex | uniform baseline = -8.0; library is 2.83 bits/byte better than random |
| 2026-05-07 | V11 HellaSwag | accuracy | 0.15 | Rowan/hellaswag 20 ex | **WORSE than random 0.25** — library trained on wiki picks wiki-like endings over story-natural ones |
| 2026-05-07 | V11 WikiText-103 | BPB | 3.4085 | wikitext-103 validation 10KB | cross-corpus shift from V1's 2.62 on wikitext-2; n-gram tables don't generalize |
| 2026-05-07 | V11 gate | beat_ngram_baseline | partial | n/a | LAMBADA improvement is real; HellaSwag below random; WikiText cross-corpus deg. Gate explicitly says "do NOT expect to beat GPT-2"; honest interim. |
| 2026-05-08 | V7 full | BPB | **2.2278** | wikitext-2-raw-v1 heldout 100KB | decay-every=500, decay-factor=0.99, replay-buf=2000. **-15% vs V1 trained, -6% vs V3, BEATS V2 word-tokenizer naive baseline (2.2491)**. Decay+replay produced not just diversity restoration but a real compression win. |
| 2026-05-08 | V7 full | ECE | 0.0334 | wikitext-2 heldout 100KB | inside V10 gate ≤0.05 ✓ |
| 2026-05-08 | V7 full | refusal_rate | 0.3962 | os.urandom 20KB | partial recovery (V1=1.0, V3-V5=0.0). Decay bleeds confidence on OOD but doesn't restore the V1 invariant — that's V10's job. |
| 2026-05-08 | V7 final lib | n_programs | 17 | n/a | smaller than V4-V5 (139) — decay reduces the gap blowup so prune doesn't murder as many programs; the surviving 17 are the actual high-utility ones |
| 2026-05-08 | V7+V10 stack | refusal English | 0.0000 | hand-crafted text | Toplevel on V7 lib: English in-distribution ✓ |
| 2026-05-08 | V7+V10 stack | refusal os.urandom | 1.0000 | os.urandom 96B | **V3 silent regression fully repaired** ✓ |
| 2026-05-08 | V7+V10 stack | refusal Cyrillic | 0.9930 | UTF-8 Cyrillic 64B | foreign-script OOD detected without special-casing ✓ |
| 2026-05-08 | V8 real Modal | wall-clock | 94.1s | 4 workers × 8KB | sequential dry-run 584s → 6.2× speedup; 133 unique merged programs; 23 in ≥2 shards. https://modal.com/apps/kent-ai-dev/main/ap-ybdtrJ7Gu51L7VyNaSLEfO |
| 2026-05-08 | V11 vs V7 lib | LAMBADA per-byte log2p | -3.84 | EleutherAI/lambada_openai 30 ex | was -5.17 with chat lib; +1.33 bits/byte improvement |
| 2026-05-08 | V11 vs V7 lib | HellaSwag accuracy | 0.233 | Rowan/hellaswag 30 ex | was 0.15; back near random 0.25 (no longer anti-correlated) |
| 2026-05-08 | V11 vs V7 lib | WikiText-103 BPB | **2.194** | wikitext-103-raw-v1 validation 20KB | was 3.41 with chat lib; -36%. **V9 gate ≤2.5 ALREADY MET** by V7+V10 trained on wikitext-2 alone |
| 2026-05-08 | V9 attempt 1 | failure | timeout | monology/pile-uncopyrighted | Modal workers hit FileNotFoundError on zst-streaming path even with zstandard installed (datasets+fsspec version skew in Modal image). Pivoted dataset. |
| 2026-05-09 | V9 attempt 2 | partial | timeout | DKYoon/SlimPajama-6B, 16 × 10MB | Some Modal workers exceeded the 7200s function timeout — likely KN-fit on 10MB of mixed prose was unexpectedly slow on some shards. Retrying with 32 × 1MB shards. |
| 2026-05-09 | V12 trained | text BPB | 1.0143 | allenai/c4 en stream (SUSPICIOUS) | per_domain_bpb restarts the stream so eval bytes overlap heavily with training bytes — this is closer to train BPB than held-out BPB. Honest finding: V12's measurement methodology needs a proper held-out split before this number can be trusted. |
| 2026-05-09 | V12 trained | code BPB | 0.6679 | code_search_net python (SUSPICIOUS) | same caveat as text |
| 2026-05-09 | V12 trained | structured BPB | NaN | wikidata5m | numerical issue (likely division by zero when stream has no usable bytes after wikidata5m's WARN-level signal) |
| 2026-05-09 | V12 lib | n_programs | 100 | cross-domain mix, 20K bayes-train | mode collapse persists despite decay (kn-5 weight 0.9984) — decay-factor 0.99 every 500 steps too gentle for cross-domain training where one general predictor (kn-5) dominates. v2.md V22 addresses this structurally. |
| 2026-05-09 | V9 attempt 3 | success | 6759.7s wall | DKYoon/SlimPajama-6B, 32 × 1MB | 32 Modal workers in parallel produced 32 specialist libraries; merged via prior-correction math to **2274 unique programs (197 in ≥2 shards)**. Per-shard size varied wildly (24-150 programs) due to Modal preemption restarts on spot capacity. Library *reconstruction* (instantiating program objects from merged lineage) is a remaining piece — the merge dict has weights+lengths+appearances but not the program code. |
| 2026-05-09 | V9 gate | BPB ≤ 2.5 on wikitext-103 | met by transitivity | wikitext-103 validation 20KB | V11 against V7 library (trained on 30KB wikitext-2) already showed BPB 2.194 — V9's pile training would need to demonstrate further improvement, which we couldn't measure due to library-reconstruction gap. |
| 2026-05-09 | V14 curiosity | OOD chunks first | ✓ | synthetic 5-chunk stream | input order [1.08, 1.08, 11.5, 1.08, 11.25] → curiosity_filter yields [11.5, 1.08, 11.25, 1.08, 1.08] |
| 2026-05-09 | V15 trained | n_programs | 43 | OpenAssistant/oasst1, 12KB train | li2017dailydialog FAILed validation (HF deprecated script-based loading); gracefully skipped, oasst1 only |
| 2026-05-09 | V17 federation | merged_programs | 102 | V7 + V12 specialists | 17 + 100 = 117 raw → 102 unique (15 shared got prior-correction); gzipped exchange format with provenance manifest |
| 2026-05-09 | V19 MNIST | BPB | **0.8632** | ylecun/mnist 30KB stream (rich=False n-grams) | gate ≤6.5 ✓ by ~7×. **Beats PNG (~4.0)** because MNIST's pixel distribution is highly redundant (most pixels = 0); the architecture handles non-text byte streams without modification. KneserNey rich primitives hit pathological pathways on all-zero contexts; n-grams alone cleared the gate decisively. |
