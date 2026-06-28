"""CLI: build and persist the stratified train/val/test split, then print stats.

Usage:
    python scripts/prepare_data.py [--subset-size N] [--split-dir PATH]

This is milestone M1. It downloads the Yelp pool once, draws a class-balanced
subset, splits 80/10/10 (stratified), writes parquet + meta.json, and prints a
class-balance report for review.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

# Make src/ importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nm_sentiment.config import DataConfig  # noqa: E402
from nm_sentiment.data import prepare_and_persist  # noqa: E402
from nm_sentiment.seeding import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build + persist the Yelp split (M1).")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Total reviews drawn from the pool (default: config).")
    parser.add_argument("--split-dir", type=str, default=None,
                        help="Where to write the split (default: config).")
    args = parser.parse_args()

    cfg = DataConfig()
    overrides = {}
    if args.subset_size is not None:
        overrides["subset_size"] = args.subset_size
    if args.split_dir is not None:
        overrides["split_dir"] = args.split_dir
    if overrides:
        cfg = replace(cfg, **overrides)

    set_seed(cfg.seed)
    print(f"[M1] Building split: subset_size={cfg.subset_size}, "
          f"split={cfg.train_frac}/{cfg.val_frac}/{cfg.test_frac}, seed={cfg.seed}")
    print(f"[M1] Loading pool '{cfg.dataset_name}' (first run downloads ~200MB)...")

    artifact = prepare_and_persist(cfg)

    print(f"[M1] Wrote split to {cfg.split_dir}/  (config_hash={artifact.meta['config_hash']})")
    print("\n=== Split stats ===")
    print(json.dumps(artifact.meta["sizes"], indent=2))
    print("\nPer-class counts:")
    for split in ("train", "val", "test"):
        counts = artifact.meta["class_counts"][split]
        total = sum(counts.values())
        pcts = {k: f"{100*v/total:.1f}%" for k, v in counts.items()}
        print(f"  {split:5s} (n={total}): counts={counts}  pct={pcts}")


if __name__ == "__main__":
    main()
