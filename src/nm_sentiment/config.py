"""Single source of truth for all configuration.

Frozen dataclasses keep every run explicit, logged, and JSON-serializable next to
its results. The fairness invariant of the project is encoded here:

  * Bi-LSTM is ``replace(RNNConfig(), bidirectional=True)`` — nothing else changes.
  * RoBERTa is ``replace(HFConfig(), model_name="roberta-base")`` — nothing else
    changes.

So the *only* factor that varies between compared models is the architecture.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DataConfig:
    """Dataset, subset, and split configuration."""

    dataset_name: str = "Yelp/yelp_review_full"
    num_classes: int = 5
    # Total reviews drawn from the 650k train pool, split 80/10/10 below.
    subset_size: int = 50_000
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42
    split_dir: str = "data/splits"

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test fractions must sum to 1.0, got {total}")
        if self.subset_size % self.num_classes != 0:
            raise ValueError(
                f"subset_size ({self.subset_size}) must be divisible by "
                f"num_classes ({self.num_classes}) for a balanced stratified draw"
            )


@dataclass(frozen=True)
class RNNConfig:
    """LSTM / Bi-LSTM configuration. Bi-LSTM flips only ``bidirectional``."""

    max_len: int = 256
    vocab_size: int = 25_000  # cap; <pad>=0 and <unk>=1 are reserved inside it
    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1  # optionally 2 if time allows
    dropout: float = 0.3
    bidirectional: bool = False
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 15
    early_stop_patience: int = 3  # on validation loss
    seed: int = 42

    @property
    def name(self) -> str:
        return "bilstm" if self.bidirectional else "lstm"


@dataclass(frozen=True)
class HFConfig:
    """BERT / RoBERTa fine-tuning configuration. RoBERTa flips only ``model_name``."""

    model_name: str = "bert-base-uncased"  # or "roberta-base"
    max_len: int = 256
    batch_size: int = 16
    lr: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    seed: int = 42

    @property
    def name(self) -> str:
        # "bert-base-uncased" -> "bert", "roberta-base" -> "roberta"
        return self.model_name.split("-")[0].split("/")[-1]


def config_hash(cfg) -> str:
    """Short deterministic hash of a config, used to detect split/config drift."""
    payload = json.dumps(asdict(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]
