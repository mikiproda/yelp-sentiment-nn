"""Aggregate the four per-model results into the comparison deliverables.

Produces, from ``results/metrics/*.json``:
  * ``results/comparison.csv``      — the 4x4 table (accuracy, macro P/R/F1)
  * ``results/per_class_f1.csv``    — per-class F1 (1..5 stars) per model
  * a markdown table + a benchmark sanity check printed to stdout

The benchmark ranges come from IMPLEMENTATION_PLAN.md section 8.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Display order + pretty names.
MODEL_ORDER = ["lstm", "bilstm", "bert", "roberta"]
PRETTY = {"lstm": "LSTM", "bilstm": "Bi-LSTM", "bert": "BERT", "roberta": "RoBERTa"}

# (flag-below accuracy, human-readable expected range) from section 8.
BENCHMARKS = {
    "lstm": (0.45, "~0.50-0.55"),
    "bilstm": (0.45, "~0.50-0.56"),
    "bert": (0.60, "~0.65-0.70"),
    "roberta": (0.63, "~0.70-0.73"),
}

METRIC_COLS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


def _to_markdown(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavored markdown table (avoids the optional 'tabulate' dep)."""
    idx = df.index.name or ""
    header = [idx, *[str(c) for c in df.columns]]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for name, row in df.iterrows():
        cells = [str(name), *[f"{v}" for v in row.tolist()]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_all(metrics_dir: str) -> dict[str, dict]:
    out = {}
    for key in MODEL_ORDER:
        path = Path(metrics_dir) / f"{key}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                out[key] = json.load(f)
    return out


def comparison_table(data: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key in MODEL_ORDER:
        if key not in data:
            continue
        tm = data[key]["test_metrics"]
        rows.append({"model": PRETTY[key], **{c: tm[c] for c in METRIC_COLS}})
    return pd.DataFrame(rows).set_index("model")


def per_class_f1_table(data: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key in MODEL_ORDER:
        if key not in data:
            continue
        rep = data[key]["classification_report"]
        rows.append({"model": PRETTY[key],
                     **{f"{c+1}star": rep[str(c)]["f1-score"] for c in range(5)}})
    return pd.DataFrame(rows).set_index("model")


def adjacent_error_share(data: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key in MODEL_ORDER:
        if key not in data:
            continue
        cm = np.array(data[key]["confusion_matrix"])
        total_err = cm.sum() - np.trace(cm)
        adj = sum(cm[i, j] for i in range(cm.shape[0]) for j in range(cm.shape[1])
                  if abs(i - j) == 1)
        rows.append({"model": PRETTY[key],
                     "off_by_1_pct": 100 * adj / total_err,
                     "off_by_2plus_pct": 100 * (1 - adj / total_err)})
    return pd.DataFrame(rows).set_index("model")


def sanity_check(data: dict[str, dict]) -> list[str]:
    lines = []
    for key in MODEL_ORDER:
        if key not in data:
            continue
        acc = data[key]["test_metrics"]["accuracy"]
        floor, expected = BENCHMARKS[key]
        flag = "OK " if acc >= floor else "LOW"
        lines.append(f"  [{flag}] {PRETTY[key]:8s} acc={acc:.4f}  "
                     f"expected {expected} (flag if < {floor:.2f})")
    return lines


def build(metrics_dir: str = "results/metrics", out_dir: str = "results") -> pd.DataFrame:
    data = load_all(metrics_dir)
    if not data:
        raise FileNotFoundError(f"No metrics JSON found in {metrics_dir}")

    table = comparison_table(data)
    per_class = per_class_f1_table(data)
    adjacent = adjacent_error_share(data)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison.csv")
    per_class.to_csv(out / "per_class_f1.csv")

    print("=== Comparison (test set) ===")
    print(_to_markdown(table.round(4)))
    print("\n=== Per-class F1 (1..5 stars) ===")
    print(_to_markdown(per_class.round(3)))
    print("\n=== Error locality (adjacent vs distant) ===")
    print(_to_markdown(adjacent.round(1)))
    print("\n=== Benchmark sanity check (section 8) ===")
    print("\n".join(sanity_check(data)))
    print(f"\nSaved {out/'comparison.csv'} and {out/'per_class_f1.csv'}")
    return table


if __name__ == "__main__":
    build()
