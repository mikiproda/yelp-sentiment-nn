"""CLI: aggregate per-model results into the comparison table + report artifacts.

Usage:
    python scripts/build_report.py

Reads results/metrics/*.json and writes results/comparison.csv +
results/per_class_f1.csv, printing markdown tables and a benchmark sanity check.
Confusion-matrix and training-curve figures are produced per model at train time
(results/figures/); this step assembles the cross-model comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nm_sentiment.compare import build  # noqa: E402

if __name__ == "__main__":
    build()
