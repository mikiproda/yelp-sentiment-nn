"""LSTM-path tokenizer and capped vocabulary.

Built from the **train split only** (never val/test) to avoid leakage. The same
vocabulary is reused by both LSTM and Bi-LSTM so their only difference stays the
architecture. Persisted to JSON so a Kaggle run reloads the identical mapping.

Reserved ids: ``<pad>=0``, ``<unk>=1``. Real tokens start at id 2.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase + simple word/number tokenization (no external deps)."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Vocab:
    """Token<->id mapping with a fixed cap. ``stoi`` includes pad/unk."""

    stoi: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, max_len: int) -> tuple[list[int], int]:
        """Return (padded ids of length ``max_len``, true length before padding).

        The true length (capped at ``max_len``, min 1) lets the model pack the
        sequence so trailing pad tokens don't pollute the final hidden state.
        """
        ids = [self.stoi.get(tok, UNK_ID) for tok in tokenize(text)][:max_len]
        if not ids:  # empty/whitespace review -> single <unk> so length>=1
            ids = [UNK_ID]
        length = len(ids)
        if length < max_len:
            ids = ids + [PAD_ID] * (max_len - length)
        return ids, length

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(stoi=data["stoi"])


def build_vocab(texts: Iterable[str], max_size: int) -> Vocab:
    """Build a capped vocabulary from training texts.

    Keeps the ``max_size - 2`` most frequent tokens (2 slots reserved for
    pad/unk). Ties are broken by token string for determinism.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    # Deterministic order: by descending count, then alphabetically.
    most_common = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    keep = most_common[: max_size - 2]

    stoi = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for tok, _count in keep:
        stoi[tok] = len(stoi)
    return Vocab(stoi=stoi)
