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
