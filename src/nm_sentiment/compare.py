"""Aggregate per-model, per-seed results into the comparison deliverables.

Reads every ``results/metrics/<model>_seed<seed>.json`` and groups by model, so
runs across multiple training seeds are averaged (mean +/- std). With a single
seed present, std is 0.

Produces:
  * ``results/comparison.csv``      — 4x4 table with per-metric mean and std
  * ``results/per_class_f1.csv``    — per-class F1 (1..5 stars), canonical seed
  * markdown tables + a benchmark sanity check printed to stdout

Benchmark ranges come from IMPLEMENTATION_PLAN.md section 8.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ORDER = ["lstm", "bilstm", "bert", "roberta"]
PRETTY = {"lstm": "LSTM", "bilstm": "Bi-LSTM", "bert": "BERT", "roberta": "RoBERTa"}
CANONICAL_SEED = 42

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


def load_runs(metrics_dir: str) -> dict[str, dict[int, dict]]:
    """Return {model_key: {seed: results_dict}} for every metrics JSON found."""
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in Path(metrics_dir).glob("*.json"):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        runs[r["model"]][int(r["seed"])] = r
    return runs


def comparison_table(runs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (display table with 'mean +/- std' strings, numeric mean/std table)."""
    disp_rows, num_rows = [], []
    for key in MODEL_ORDER:
        if key not in runs:
            continue
        seeds = sorted(runs[key])
        disp = {"model": PRETTY[key], "seeds": len(seeds)}
        num = {"model": PRETTY[key], "n_seeds": len(seeds)}
        for c in METRIC_COLS:
            vals = np.array([runs[key][s]["test_metrics"][c] for s in seeds])
            mean, std = float(vals.mean()), float(vals.std(ddof=0))
            disp[c] = f"{mean:.4f} ± {std:.4f}"
            num[f"{c}_mean"] = round(mean, 4)
            num[f"{c}_std"] = round(std, 4)
        disp_rows.append(disp)
        num_rows.append(num)
    return (pd.DataFrame(disp_rows).set_index("model"),
            pd.DataFrame(num_rows).set_index("model"))


def _canonical(runs, key):
    """The canonical-seed run if present, else the lowest available seed."""
    seeds = runs[key]
    return seeds.get(CANONICAL_SEED, seeds[min(seeds)])


def per_class_f1_table(runs) -> pd.DataFrame:
    rows = []
    for key in MODEL_ORDER:
        if key not in runs:
            continue
        rep = _canonical(runs, key)["classification_report"]
        rows.append({"model": PRETTY[key],
                     **{f"{c+1}star": round(rep[str(c)]["f1-score"], 3) for c in range(5)}})
    return pd.DataFrame(rows).set_index("model")


def adjacent_error_share(runs) -> pd.DataFrame:
    rows = []
    for key in MODEL_ORDER:
        if key not in runs:
            continue
        cm = np.array(_canonical(runs, key)["confusion_matrix"])
        total_err = cm.sum() - np.trace(cm)
        adj = sum(cm[i, j] for i in range(cm.shape[0]) for j in range(cm.shape[1])
                  if abs(i - j) == 1)
        rows.append({"model": PRETTY[key],
                     "off_by_1_pct": round(100 * adj / total_err, 1),
                     "off_by_2plus_pct": round(100 * (1 - adj / total_err), 1)})
    return pd.DataFrame(rows).set_index("model")


def sanity_check(runs) -> list[str]:
    lines = []
    for key in MODEL_ORDER:
        if key not in runs:
            continue
        seeds = sorted(runs[key])
        accs = np.array([runs[key][s]["test_metrics"]["accuracy"] for s in seeds])
        floor, expected = BENCHMARKS[key]
        flag = "OK " if accs.mean() >= floor else "LOW"
        lines.append(f"  [{flag}] {PRETTY[key]:8s} acc={accs.mean():.4f} "
                     f"(n={len(seeds)} seed{'s' if len(seeds) != 1 else ''}: "
                     f"{', '.join(str(s) for s in seeds)})  "
                     f"expected {expected} (flag if < {floor:.2f})")
    return lines


def build(metrics_dir: str = "results/metrics", out_dir: str = "results") -> pd.DataFrame:
    runs = load_runs(metrics_dir)
    if not runs:
        raise FileNotFoundError(f"No metrics JSON found in {metrics_dir}")

    disp, numeric = comparison_table(runs)
    per_class = per_class_f1_table(runs)
    adjacent = adjacent_error_share(runs)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numeric.to_csv(out / "comparison.csv")
    per_class.to_csv(out / "per_class_f1.csv")

    print("=== Comparison (test set, mean +/- std across seeds) ===")
    print(_to_markdown(disp))
    print("\n=== Per-class F1 (1..5 stars, canonical seed) ===")
    print(_to_markdown(per_class))
    print("\n=== Error locality (adjacent vs distant, canonical seed) ===")
    print(_to_markdown(adjacent))
    print("\n=== Benchmark sanity check (section 8) ===")
    print("\n".join(sanity_check(runs)))
    print(f"\nSaved {out/'comparison.csv'} and {out/'per_class_f1.csv'}")
    return numeric


if __name__ == "__main__":
    build()
