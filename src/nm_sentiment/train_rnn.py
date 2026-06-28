"""Training loop for the LSTM / Bi-LSTM models.

Cross-entropy + Adam, early stopping on validation loss, best-state restore.
Produces a per-epoch ``history`` in the same schema the HF path emits, so plots
and the comparison table are uniform across all four models.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import RNNConfig


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    """One pass. If ``optimizer`` is given, trains; else evaluates. Returns (loss, acc)."""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for input_ids, lengths, labels in loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += bs

    return total_loss / total, total_correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: RNNConfig,
    device: torch.device,
    verbose: bool = True,
) -> list[dict]:
    """Train with early stopping on val loss; restores best weights in place.

    Returns the per-epoch history list of
    ``{epoch, train_loss, train_acc, val_loss, val_acc}``.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: list[dict] = []
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        if verbose:
            print(f"  epoch {epoch:2d} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
                  f"| val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch} "
                          f"(no val-loss improvement for {cfg.early_stop_patience} epochs)")
                break

    model.load_state_dict(best_state)
    return history
