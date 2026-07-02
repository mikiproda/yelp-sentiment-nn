"""Evaluation: metrics, confusion matrix, and the uniform results JSON.

Every model — RNN or transformer — produces a results dict with the same schema,
so ``compare.py`` and ``plots.py`` treat all four identically.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, num_classes: int) -> dict:
    """Accuracy + macro precision/recall/F1, plus per-class report and confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "test_metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        },
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def predict_rnn(model, loader, device):
    """Return (y_true, y_pred) numpy arrays over a dataloader of (ids, lengths, labels)."""
    import torch

    model.eval()
    model.to(device)
    y_true, y_pred = [], []
    with torch.no_grad():
        for input_ids, lengths, labels in loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            logits = model(input_ids, lengths)
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
    return np.array(y_true), np.array(y_pred)


def _cfg_to_dict(cfg) -> dict:
    return asdict(cfg) if is_dataclass(cfg) else dict(cfg)


def build_results(model_name, cfg, history, metrics, *, seed, device, extra=None) -> dict:
    """Assemble the uniform per-model results dict."""
    results = {
        "model": model_name,
        "seed": seed,
        "device": str(device),
        "config": _cfg_to_dict(cfg),
        "history": history,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,  # test_metrics, classification_report, confusion_matrix
    }
    if extra:
        results.update(extra)
    return results


def save_results(results: dict, metrics_dir: str) -> Path:
    """Write ``<model>_seed<seed>.json`` so multiple seeds coexist for aggregation."""
    out = Path(metrics_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{results['model']}_seed{results['seed']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return path
