"""Data pipeline: load Yelp -> stratified subset -> stratified 80/10/10 split.

The resulting split is persisted to disk (parquet + meta.json) and is the single
artifact every model loads. This is the fairness backbone of the project: the
subset and the splits are computed once from a fixed seed, then reused verbatim.

We persist the *texts* (not just indices) so a Kaggle session can load the 50k
split directly without re-downloading and re-subsetting the 650k pool, and so the
split can be attached as a Kaggle Dataset for guaranteed identical data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig, config_hash

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitArtifact:
    """In-memory representation of the persisted split: one DataFrame per split.

    Each DataFrame has columns ``text`` (str) and ``label`` (int 0-4).
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    meta: dict

    def frame(self, name: str) -> pd.DataFrame:
        return getattr(self, name)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def load_pool(cfg: DataConfig):
    """Load the 650k Yelp training pool. The official test split is never touched."""
    from datasets import load_dataset

    return load_dataset(cfg.dataset_name, split="train")


def stratified_subset_indices(labels: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Return indices for an exactly class-balanced subset of ``subset_size`` rows.

    Equal draw of ``subset_size // num_classes`` per class, seeded for reproducibility.
    """
    rng = np.random.default_rng(cfg.seed)
    per_class = cfg.subset_size // cfg.num_classes
    chosen = []
    for c in range(cfg.num_classes):
        idx_c = np.flatnonzero(labels == c)
        if len(idx_c) < per_class:
            raise ValueError(
                f"class {c} has only {len(idx_c)} samples in the pool, "
                f"need {per_class}"
            )
        chosen.append(rng.choice(idx_c, size=per_class, replace=False))
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


def make_splits(cfg: DataConfig) -> SplitArtifact:
    """Build the stratified subset and split it into train/val/test."""
    pool = load_pool(cfg)
    labels_all = np.asarray(pool["label"])

    sub_idx = stratified_subset_indices(labels_all, cfg)
    subset = pool.select(sub_idx.tolist())
    df = subset.to_pandas()[["text", "label"]].reset_index(drop=True)

    # First carve the untouched test split, then split the remainder into train/val.
    test_size = cfg.test_frac
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=cfg.seed,
        shuffle=True,
    )
    # val fraction relative to the remaining (train+val) portion
    val_relative = cfg.val_frac / (cfg.train_frac + cfg.val_frac)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_relative,
        stratify=train_df["label"],
        random_state=cfg.seed,
        shuffle=True,
    )

    for d in (train_df, val_df, test_df):
        d.reset_index(drop=True, inplace=True)

    meta = _build_meta(cfg, train_df, val_df, test_df)
    return SplitArtifact(train=train_df, val=val_df, test=test_df, meta=meta)


def _class_counts(df: pd.DataFrame, num_classes: int) -> dict[int, int]:
    counts = df["label"].value_counts().to_dict()
    return {int(c): int(counts.get(c, 0)) for c in range(num_classes)}


def _build_meta(cfg, train_df, val_df, test_df) -> dict:
    return {
        "config": cfg.__dict__,
        "config_hash": config_hash(cfg),
        "sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "class_counts": {
            "train": _class_counts(train_df, cfg.num_classes),
            "val": _class_counts(val_df, cfg.num_classes),
            "test": _class_counts(test_df, cfg.num_classes),
        },
    }


# --------------------------------------------------------------------------- #
# Persist / load
# --------------------------------------------------------------------------- #
def persist(artifact: SplitArtifact, split_dir: str) -> None:
    out = Path(split_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        artifact.frame(name).to_parquet(out / f"{name}.parquet", index=False)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(artifact.meta, f, indent=2)


def load_splits(split_dir: str) -> SplitArtifact:
    out = Path(split_dir)
    meta_path = out / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No split found in {out}. Run scripts/prepare_data.py first."
        )
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    frames = {name: pd.read_parquet(out / f"{name}.parquet") for name in SPLIT_NAMES}
    return SplitArtifact(meta=meta, **frames)


def prepare_and_persist(cfg: DataConfig) -> SplitArtifact:
    """End-to-end M1 entry point: build the split and write it to ``cfg.split_dir``."""
    artifact = make_splits(cfg)
    persist(artifact, cfg.split_dir)
    return artifact
