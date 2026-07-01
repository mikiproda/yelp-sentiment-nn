"""Per-model figures: training/validation curves and the confusion matrix.

Uses a non-interactive matplotlib backend so it runs headless on Kaggle.
Star labels (1-5) are shown instead of raw class ids (0-4) for readability.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

STAR_LABELS = ["1", "2", "3", "4", "5"]


def _series(history, xkey, ykey):
    """(x, y) pairs where y is present and non-None (HF omits train_acc)."""
    xs, ys = [], []
    for h in history:
        if h.get(ykey) is not None:
            xs.append(h[xkey]); ys.append(h[ykey])
    return xs, ys


def plot_curves(history: list[dict], model_name: str, out_dir: str) -> Path:
    """Two-panel loss & accuracy curves over epochs. Skips absent series."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    for key, style, label in [("train_loss", "o-", "train"), ("val_loss", "s-", "val")]:
        xs, ys = _series(history, "epoch", key)
        if xs:
            ax_loss.plot(xs, ys, style, label=label)
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_loss.set_title(f"{model_name} — loss"); ax_loss.legend(); ax_loss.grid(alpha=0.3)

    for key, style, label in [("train_acc", "o-", "train"), ("val_acc", "s-", "val")]:
        xs, ys = _series(history, "epoch", key)
        if xs:
            ax_acc.plot(xs, ys, style, label=label)
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy")
    ax_acc.set_title(f"{model_name} — accuracy"); ax_acc.legend(); ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{model_name}_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(cm, model_name: str, out_dir: str, normalize: bool = True) -> Path:
    """Confusion-matrix heatmap (row-normalized by default)."""
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        data = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        fmt, title = ".2f", f"{model_name} — confusion matrix (row-normalized)"
    else:
        data, fmt, title = cm, ".0f", f"{model_name} — confusion matrix"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", cbar=True,
                xticklabels=STAR_LABELS, yticklabels=STAR_LABELS, ax=ax,
                vmin=0, vmax=1 if normalize else None)
    ax.set_xlabel("predicted stars"); ax.set_ylabel("true stars"); ax.set_title(title)

    fig.tight_layout()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{model_name}_confusion.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_all_figures(results: dict, figures_dir: str) -> list[Path]:
    """Convenience: curves + confusion from a results dict."""
    paths = []
    if results.get("history"):
        paths.append(plot_curves(results["history"], results["model"], figures_dir))
    paths.append(plot_confusion(results["confusion_matrix"], results["model"], figures_dir))
    return paths
