"""Torch datasets for the two preprocessing paths.

Both consume the *same* persisted splits (DataFrames with ``text``/``label``);
only the encoding differs:

  * ``RNNDataset``  -> integer ids via our capped vocab (LSTM / Bi-LSTM).
  * ``make_hf_dataset`` -> model tokenizer ids + attention mask (BERT / RoBERTa).

torch / transformers are imported lazily so the data-prep environment (M1) need
not have them installed.
"""

from __future__ import annotations

import pandas as pd

from .vocab import Vocab


class RNNDataset:
    """Yields ``(input_ids[max_len] LongTensor, length LongTensor, label LongTensor)``."""

    def __init__(self, df: pd.DataFrame, vocab: Vocab, max_len: int):
        import torch

        self._torch = torch
        self.vocab = vocab
        self.max_len = max_len
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        torch = self._torch
        ids, length = self.vocab.encode(self.texts[idx], self.max_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def make_hf_dataset(df: pd.DataFrame, tokenizer, max_len: int):
    """Tokenize a split DataFrame for a transformer model.

    Returns a ``datasets.Dataset`` with ``input_ids``, ``attention_mask``,
    ``labels`` — directly consumable by the HF ``Trainer``.
    """
    from datasets import Dataset

    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

    def _tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        enc["labels"] = batch["label"]
        return enc

    ds = ds.map(_tok, batched=True, remove_columns=["text", "label"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds
