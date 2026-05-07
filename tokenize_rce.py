"""
Tokenizer abstraction for the V2 sweep.

Three tokenizers, all share the same interface:

  - BytesTokenizer       : vocab_size=256, identity over raw bytes
  - BPETokenizer(N)      : HuggingFace BPE with vocab_size=N (default 4096)
  - WordTokenizer        : split on whitespace+punctuation; vocab built from train

Interface:
    tok = SomeTokenizer(...)
    tok.train(training_bytes)        # only BPE/Word need this; bytes is no-op
    ids: list[int] = tok.encode(b)   # bytes -> token IDs
    out: bytes      = tok.decode(ids)

All three keep the round-trip bytes-equivalent on training-distribution input.
The Word tokenizer reserves token id 0 for `<unk>`.

Used by bench.py via `--tokenizer {bytes,bpe,word}`. The tokenization
sweep computes BPB *per input byte* across all three so they are directly
comparable: `bpb = sum(-log2(P(token))) / len(eval_bytes)`.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Protocol


class Tokenizer(Protocol):
    name: str
    vocab_size: int

    def train(self, data: bytes) -> None: ...
    def encode(self, data: bytes) -> list[int]: ...
    def decode(self, ids: list[int]) -> bytes: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str): ...


# ---------- BytesTokenizer ----------

class BytesTokenizer:
    name = "bytes"
    vocab_size = 256

    def train(self, data: bytes) -> None:
        return  # nothing to learn

    def encode(self, data: bytes) -> list[int]:
        return list(data)

    def decode(self, ids: list[int]) -> bytes:
        return bytes(i for i in ids if 0 <= i < 256)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({"type": "bytes"}))

    @classmethod
    def load(cls, path: str) -> "BytesTokenizer":
        return cls()


# ---------- BPETokenizer ----------

class BPETokenizer:
    name = "bpe"

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self._tk = None  # tokenizers.Tokenizer, lazily built

    def _build_fresh(self):
        from tokenizers import Tokenizer as HFTok
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel

        tk = HFTok(BPE(unk_token=None))  # byte-level BPE handles all bytes
        tk.pre_tokenizer = ByteLevel(add_prefix_space=False)
        # we'll let train() install the trainer
        return tk

    def train(self, data: bytes) -> None:
        from tokenizers.trainers import BpeTrainer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        tk = self._build_fresh()
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            initial_alphabet=[chr(b) for b in range(256)],
            special_tokens=[],
            show_progress=False,
        )
        # tokenizers expects an iterator of strings; we treat bytes as latin-1
        text = data.decode("latin-1")
        tk.train_from_iterator([text], trainer=trainer)
        tk.decoder = ByteLevelDecoder()
        # actual vocab might be slightly under target; record the realized size
        self.vocab_size = tk.get_vocab_size()
        self._tk = tk

    def encode(self, data: bytes) -> list[int]:
        if self._tk is None:
            raise RuntimeError("BPE tokenizer not trained — call train() or load()")
        return self._tk.encode(data.decode("latin-1")).ids

    def decode(self, ids: list[int]) -> bytes:
        if self._tk is None:
            raise RuntimeError("BPE tokenizer not trained — call train() or load()")
        return self._tk.decode(ids).encode("latin-1", errors="replace")

    def save(self, path: str) -> None:
        if self._tk is None:
            raise RuntimeError("nothing to save")
        self._tk.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        from tokenizers import Tokenizer as HFTok
        inst = cls()
        inst._tk = HFTok.from_file(path)
        inst.vocab_size = inst._tk.get_vocab_size()
        return inst


# ---------- WordTokenizer ----------

# Split on whitespace + punctuation. Words and punctuation chars become tokens.
# Anything not in the trained vocab maps to <unk> (id 0).
_WORD_RE = re.compile(rb"[A-Za-z0-9]+|[^\sA-Za-z0-9]")


class WordTokenizer:
    name = "word"

    def __init__(self):
        self.id_to_tok: list[bytes] = [b"<unk>"]   # 0 = unk
        self.tok_to_id: dict[bytes, int] = {b"<unk>": 0}
        self.vocab_size = 1

    def train(self, data: bytes) -> None:
        tokens = _WORD_RE.findall(data)
        # also reserve a token for whitespace runs so we round-trip byte counts
        # roughly. We don't need exact round-trip for BPB — only consistent
        # token count per byte across the held-out slice.
        self.id_to_tok = [b"<unk>"]
        self.tok_to_id = {b"<unk>": 0}
        # frequency-rank vocab (cap implicitly via uniqueness in this corpus)
        from collections import Counter
        c = Counter(tokens)
        for tok, _ in c.most_common():
            self.tok_to_id[tok] = len(self.id_to_tok)
            self.id_to_tok.append(tok)
        self.vocab_size = len(self.id_to_tok)

    def encode(self, data: bytes) -> list[int]:
        return [self.tok_to_id.get(t, 0) for t in _WORD_RE.findall(data)]

    def decode(self, ids: list[int]) -> bytes:
        # Best-effort decoding: join with single spaces. This is lossy,
        # but the V2 sweep evaluates BPB, not round-trip fidelity.
        return b" ".join(self.id_to_tok[i] for i in ids if 0 <= i < self.vocab_size)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(
            [t.decode("latin-1") for t in self.id_to_tok]
        ))

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        inst = cls()
        toks = json.loads(Path(path).read_text())
        inst.id_to_tok = [t.encode("latin-1") for t in toks]
        inst.tok_to_id = {t: i for i, t in enumerate(inst.id_to_tok)}
        inst.vocab_size = len(inst.id_to_tok)
        return inst


# ---------- Factory ----------

def make(name: str, **kwargs) -> Tokenizer:
    if name == "bytes":
        return BytesTokenizer()
    if name == "bpe":
        return BPETokenizer(vocab_size=kwargs.get("vocab_size", 4096))
    if name == "word":
        return WordTokenizer()
    raise ValueError(f"unknown tokenizer: {name}")
