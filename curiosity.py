"""
V14: curiosity-driven streaming.

Wraps any byte-stream iterator with a priority queue: chunks scored
by *current* prediction error under the library are pulled first.
This is the Schmidhuber compression-progress drive — train where the
library is most surprised, not on the easiest data.

Uses a sliding-window EMA of recent BPB as the "expected" baseline;
chunks with BPB exceeding the baseline by `threshold` are prioritized.

Usage:
    from datasets_hf import iter_bytes
    from curiosity import curiosity_filter
    stream = iter_bytes("Salesforce/wikitext", "wikitext-103-raw-v1")
    for chunk in curiosity_filter(stream, lib, baseline_window=64):
        # high-curiosity chunk first; low-curiosity chunks deprioritized
        train_on(chunk)
"""
from __future__ import annotations
import heapq
import math
from typing import Iterator
from collections import deque

from engine import Library


def _chunk_bpb(lib: Library, chunk: bytes, ctx_window: int = 16) -> float:
    if len(chunk) < 2:
        return 0.0
    log_loss = 0.0
    for i in range(1, len(chunk)):
        ctx = chunk[max(0, i - ctx_window):i]
        actual = chunk[i]
        dist = lib.predict(ctx)
        total = sum(dist.values()) or 1.0
        p = max(dist.get(actual, 0.0) / total, 1e-12)
        log_loss += -math.log2(p)
    return log_loss / max(len(chunk) - 1, 1)


def curiosity_filter(stream: Iterator[bytes], lib: Library,
                     baseline_window: int = 64,
                     queue_high_water: int = 256,
                     priority_threshold: float = 1.2) -> Iterator[bytes]:
    """Yield chunks reordered by curiosity score.

    Implementation: maintain an EMA of recent BPB. For each incoming chunk:
      - score it
      - if score > baseline * threshold → yield IMMEDIATELY (high curiosity)
      - else → push into a heap; yield from heap when stream pauses
    Periodically flush low-priority chunks so they're not starved forever.
    """
    ema = None
    alpha = 2.0 / (baseline_window + 1)
    backlog: list[tuple[float, int, bytes]] = []  # negative score for max-heap
    counter = 0  # tie-breaker for heap stability

    for chunk in stream:
        if not chunk:
            continue
        score = _chunk_bpb(lib, chunk)
        if ema is None:
            ema = score
        else:
            ema = alpha * score + (1 - alpha) * ema

        if ema and score >= ema * priority_threshold:
            yield chunk    # high curiosity — train now
        else:
            heapq.heappush(backlog, (-score, counter, chunk))
            counter += 1
            if len(backlog) > queue_high_water:
                # flush the highest-scoring backlog item
                _, _, c = heapq.heappop(backlog)
                yield c

    # drain
    while backlog:
        _, _, c = heapq.heappop(backlog)
        yield c
