# RCE Benchmarks

Append-only ledger of every metric measured across versions.

| date | version | metric | value | dataset | notes |
|---|---|---|---|---|---|
| 2026-05-06 | V1 run-1 | BPB | 6.6460 | wikitext-2-raw-v1 heldout 100KB | base library: 8 primitives (uniform, repeat, ngram-1..6) |
| 2026-05-06 | V1 run-1 | ECE | 0.0405 | wikitext-2-raw-v1 heldout 100KB | already inside V10's ≤0.05 gate |
| 2026-05-06 | V1 run-1 | refusal_rate | 1.0000 | os.urandom 20KB | entropy > 7.0 on every position |
