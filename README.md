# Recurrent Compression Engine (RCE)

A tiny, self-growing, self-modifying predictive system. No neural networks.
No LLMs. Built on a Fibonacci-style recurrence over a library of small programs.

## The idea

The system holds a **library of programs**, each one a hypothesis about how
the world generates the bytes it sees. Every program is weighted by Bayesian
posterior with a Solomonoff length prior — short programs that predict well
dominate.

New programs are **generated, not searched**. Four combinators
(`compose`, `branch`, `abstract`, `recur`) take two existing programs and
produce a child. The library grows like a Fibonacci spiral: each generation
builds on the previous, and complexity emerges from iteration of a fixed
local rule. No exponential search.

The posterior tells the system how confident it is. When entropy is high,
it knows it doesn't know — and says so. **No hallucination by construction.**

## Files

- `engine.py` — programs, combinators, the library, Bayesian update logic
- `rce.py` — CLI: `train`, `chat`, `status`, `reset`
- `corpus.txt` / `corpus2.txt` — small test corpora
- `PLAN.md` — V1 → V20 experiment plan (the spec for the Ralph loop)
- `ralph.py` — the Ralph loop orchestrator that drives PLAN.md autonomously

## Quick start (manual)

```bash
python rce.py train corpus.txt    # train on a file
python rce.py train               # train on stdin
python rce.py chat                # chat with the latest model
python rce.py status              # see what the library knows
python rce.py reset               # wipe and start over
```

The library persists at `~/.rce_library.pkl`. Each `train` run continues
from where the last left off — knowledge accretes.

## Inside the chat

- `/top` — show top-weighted programs
- `/entropy` — show uncertainty
- `/quit` — exit

## Running the Ralph loop (autonomous training)

`PLAN.md` defines V1 → V20: a sequence of experiments designed to push the
RCE toward frontier-model competitiveness while preserving its structural
advantages (auditability, calibration, no-hallucination). `ralph.py` is the
orchestrator that drives this plan by repeatedly calling Claude as the
engineer, applying the changes Claude proposes, running the experiments,
and advancing the version when gates pass.

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...

mkdir -p /workspace/rce
cp engine.py rce.py PLAN.md /workspace/rce/

# test one iteration first
python ralph.py --workspace /workspace/rce --once

# then go autonomous
python ralph.py --workspace /workspace/rce --interval 30m --max-iters 100
```

The loop pauses for human review every 10 iterations and halts on any
`ESCALATE` token in `LOG.md`. State persists across runs in `state.json`.

## Training at scale (Modal)

`PLAN.md` V8+ describes how to scale beyond a single laptop using Modal
(modal.com). The RCE doesn't need GPUs — it needs many CPUs running many
program evaluations in parallel. Modal's `@app.function(cpu=N)` is the
right primitive. The plan also covers HuggingFace dataset streaming
(V6+), so `wikitext`, `the_pile`, `slimpajama`, etc. are all training
inputs.

## What to expect

This is a **seed**. Trained on small text it will produce coherent
phrases drawn from corpus vocabulary, and will refuse to answer (high
entropy) on inputs unlike anything it has seen. It will not write essays.
It will not pass a Turing test.

What it *does* do, that statistical models cannot:
- Tell you exactly which programs are responsible for any prediction
- Refuse to fabricate when it doesn't know
- Grow its own representational capacity through combinator recurrence
- Save and resume training without forgetting

The bet of this architecture is that scaling laws emerge from the recurrence
dynamics — the dominant eigenvalue of combinator-driven growth — rather than
from gradient descent on giant tensors. Whether that bet is correct at scale
is what `PLAN.md` is designed to test, experimentally, V1 through V20.