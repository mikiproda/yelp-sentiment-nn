"""CLI: train + evaluate one model on the persisted split, save metrics + figures.

Usage:
    python scripts/train_model.py --model lstm   [--smoke 2000]
    python scripts/train_model.py --model bilstm
    python scripts/train_model.py --model bert     (M4)
    python scripts/train_model.py --model roberta  (M5)

Loads the shared split, asserts its config hash matches the current DataConfig
(fairness guard), trains, evaluates on the untouched test split, and writes
results/metrics/<model>.json + results/figures/<model>_{curves,confusion}.png.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nm_sentiment.config import DataConfig, RNNConfig, config_hash  # noqa: E402
from nm_sentiment.data import load_splits  # noqa: E402
from nm_sentiment.seeding import set_seed  # noqa: E402

RNN_MODELS = {"lstm": False, "bilstm": True}
HF_MODELS = {"bert": "bert-base-uncased", "roberta": "roberta-base"}


def _maybe_smoke(df, n, seed):
    """Take a stratified-ish subset for a fast smoke test (keeps class balance)."""
    if not n or n >= len(df):
        return df
    per_class = n // df["label"].nunique()
    parts = [g.sample(min(per_class, len(g)), random_state=seed)
             for _, g in df.groupby("label")]
    import pandas as pd
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def run_rnn(model_key: str, args) -> None:
    import torch
    from torch.utils.data import DataLoader

    from nm_sentiment.datasets import RNNDataset
    from nm_sentiment.evaluate import (
        build_results, compute_metrics, predict_rnn, save_results,
    )
    from nm_sentiment.models import LSTMClassifier
    from nm_sentiment.plots import make_all_figures
    from nm_sentiment.train_rnn import get_device, train_model
    from nm_sentiment.vocab import build_vocab

    data_cfg = DataConfig()
    cfg = replace(RNNConfig(), bidirectional=RNN_MODELS[model_key])
    set_seed(cfg.seed)

    splits = load_splits(data_cfg.split_dir)
    assert splits.meta["config_hash"] == config_hash(data_cfg), (
        "Split on disk does not match current DataConfig — rebuild with prepare_data.py"
    )

    train_df = _maybe_smoke(splits.train, args.smoke, cfg.seed)
    val_df = _maybe_smoke(splits.val, args.smoke // 8 if args.smoke else 0, cfg.seed)
    test_df = splits.test

    print(f"[{model_key}] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"smoke={'on' if args.smoke else 'off'}")

    # Vocab from TRAIN ONLY (shared by lstm/bilstm via the same split + seed).
    vocab = build_vocab(train_df["text"], cfg.vocab_size)
    Path(data_cfg.split_dir).mkdir(parents=True, exist_ok=True)
    vocab.save(Path(data_cfg.split_dir) / "vocab.json")
    print(f"[{model_key}] vocab size = {vocab.size}")

    def loader(df, shuffle):
        ds = RNNDataset(df, vocab, cfg.max_len)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)

    device = get_device(args.device)
    print(f"[{model_key}] device = {device}")

    model = LSTMClassifier(cfg, vocab_size=vocab.size, num_classes=data_cfg.num_classes)
    history = train_model(model, loader(train_df, True), loader(val_df, False), cfg, device)

    y_true, y_pred = predict_rnn(model, loader(test_df, False), device)
    metrics = compute_metrics(y_true, y_pred, data_cfg.num_classes)
    results = build_results(
        model_key, cfg, history, metrics,
        seed=cfg.seed, device=device,
        extra={"smoke_n": args.smoke or None, "train_size": len(train_df)},
    )

    out = save_results(results, "results/metrics")
    figs = make_all_figures(results, "results/figures")
    tm = metrics["test_metrics"]
    print(f"\n[{model_key}] TEST  acc={tm['accuracy']:.4f}  "
          f"P={tm['precision_macro']:.4f}  R={tm['recall_macro']:.4f}  F1={tm['f1_macro']:.4f}")
    print(f"[{model_key}] saved {out} and {[str(f) for f in figs]}")


def run_hf(model_key: str, args) -> None:
    from dataclasses import replace as _replace

    from nm_sentiment.config import HFConfig
    from nm_sentiment.evaluate import save_results
    from nm_sentiment.plots import make_all_figures
    from nm_sentiment.train_hf import train_hf

    data_cfg = DataConfig()
    cfg = _replace(HFConfig(), model_name=HF_MODELS[model_key])
    set_seed(cfg.seed)

    splits = load_splits(data_cfg.split_dir)
    assert splits.meta["config_hash"] == config_hash(data_cfg), (
        "Split on disk does not match current DataConfig — rebuild with prepare_data.py"
    )

    if args.smoke:
        splits.train = _maybe_smoke(splits.train, args.smoke, cfg.seed)
        splits.val = _maybe_smoke(splits.val, max(args.smoke // 8, data_cfg.num_classes), cfg.seed)

    print(f"[{model_key}] model={cfg.model_name} train={len(splits.train)} "
          f"val={len(splits.val)} test={len(splits.test)} max_len={cfg.max_len}")

    results = train_hf(cfg, data_cfg, splits, output_dir=f"results/checkpoints/{model_key}")

    out = save_results(results, "results/metrics")
    figs = make_all_figures(results, "results/figures")
    tm = results["test_metrics"]
    print(f"\n[{model_key}] TEST  acc={tm['accuracy']:.4f}  "
          f"P={tm['precision_macro']:.4f}  R={tm['recall_macro']:.4f}  F1={tm['f1_macro']:.4f}")
    print(f"[{model_key}] saved {out} and {[str(f) for f in figs]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate one model.")
    parser.add_argument("--model", required=True, choices=[*RNN_MODELS, *HF_MODELS])
    parser.add_argument("--smoke", type=int, default=0,
                        help="If >0, train on a small balanced subset to validate the loop.")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto).")
    args = parser.parse_args()

    if args.model in RNN_MODELS:
        run_rnn(args.model, args)
    else:
        run_hf(args.model, args)


if __name__ == "__main__":
    main()
