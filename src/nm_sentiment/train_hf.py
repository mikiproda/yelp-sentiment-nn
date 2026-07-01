"""Fine-tuning for BERT / RoBERTa via the Hugging Face Trainer.

Same pipeline for both — only ``HFConfig.model_name`` differs — so BERT and
RoBERTa stay directly comparable, and comparable to the LSTMs through the shared
split and the uniform results schema emitted by ``evaluate.build_results``.

Per-epoch metrics are pulled from ``trainer.state.log_history`` and normalized
into the same ``{epoch, train_loss, train_acc, val_loss, val_acc}`` schema the
RNN path uses (``train_acc`` is ``None`` — the Trainer does not evaluate on the
train set by default).
"""

from __future__ import annotations

import numpy as np

from .config import DataConfig, HFConfig
from .datasets import make_hf_dataset
from .evaluate import build_results, compute_metrics


def _compute_metrics_builder(num_classes: int):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def _fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=list(range(num_classes)),
            average="macro", zero_division=0,
        )
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision_macro": p,
            "recall_macro": r,
            "f1_macro": f1,
        }

    return _fn


def _history_from_logs(log_history: list[dict]) -> list[dict]:
    """Normalize Trainer logs into the uniform per-epoch history schema."""
    train_loss, evals = {}, {}
    for entry in log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        ep = int(round(epoch))
        if "loss" in entry:  # training log
            train_loss[ep] = entry["loss"]
        if "eval_loss" in entry:  # evaluation log
            evals[ep] = entry

    history = []
    for ep in sorted(set(train_loss) | set(evals)):
        ev = evals.get(ep, {})
        history.append({
            "epoch": ep,
            "train_loss": train_loss.get(ep),
            "train_acc": None,  # Trainer does not eval on train by default
            "val_loss": ev.get("eval_loss"),
            "val_acc": ev.get("eval_accuracy"),
        })
    return history


def train_hf(hf_cfg: HFConfig, data_cfg: DataConfig, splits, output_dir: str) -> dict:
    """Fine-tune one transformer and return the uniform results dict."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(hf_cfg.model_name)
    train_ds = make_hf_dataset(splits.train, tokenizer, hf_cfg.max_len)
    val_ds = make_hf_dataset(splits.val, tokenizer, hf_cfg.max_len)
    test_ds = make_hf_dataset(splits.test, tokenizer, hf_cfg.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_cfg.model_name, num_labels=data_cfg.num_classes
    )

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        per_device_train_batch_size=hf_cfg.batch_size,
        per_device_eval_batch_size=hf_cfg.batch_size,
        learning_rate=hf_cfg.lr,
        num_train_epochs=hf_cfg.epochs,
        weight_decay=hf_cfg.weight_decay,
        warmup_ratio=hf_cfg.warmup_ratio,
        fp16=hf_cfg.fp16 and use_cuda,
        seed=hf_cfg.seed,
        report_to="none",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics_builder(data_cfg.num_classes),
    )
    trainer.train()

    history = _history_from_logs(trainer.state.log_history)

    pred_out = trainer.predict(test_ds)
    y_pred = np.argmax(pred_out.predictions, axis=-1)
    y_true = np.asarray(splits.test["label"])
    metrics = compute_metrics(y_true, y_pred, data_cfg.num_classes)

    return build_results(
        hf_cfg.name, hf_cfg, history, metrics,
        seed=hf_cfg.seed, device=device,
        extra={"train_size": len(splits.train)},
    )
