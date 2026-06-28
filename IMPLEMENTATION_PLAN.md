# Implementation Plan — Deep Learning Sentiment Analysis on Yelp Review Full

> Comparative study of LSTM, Bi-LSTM, BERT, RoBERTa on 5-class Yelp star-rating
> prediction. Individual NN course project (ETF Belgrade). This document is the
> execution reference; code comes after, in milestone order.

---

## 0. Decisions locked in

| Decision | Value | Rationale |
|---|---|---|
| Compute | **Kaggle Notebooks** (free, P100 16GB / 2×T4, 30 GPU-h/week) | Guaranteed quota + stable 12h sessions beat Colab Free's disconnects for multi-hour fine-tuning |
| Code organization | **`src/` package + thin notebook runner** | Reusable, reproducible, prevents the four pipelines drifting |
| Pacing | **Incremental**, with a checkpoint after each milestone | Lower risk; validate the loop before scaling |
| Working subset | **50k reviews**, stratified → 40k/5k/5k (80/10/10); configurable to 100k | Fits weekly GPU quota for two transformer runs |
| Test set | **Own carved stratified 5k**, untouched until final eval | Controlled class balance; §8 benchmarks remain ballpark refs |
| Max sequence length | **256** for all four models | Fairness + speed/quality tradeoff |
| Seed | **42**, logged everywhere | Reproducibility |
| Language | Code/comments **English**; final ETRAN report **Serbian** (separate later step) | Per brief |

**Fairness invariant (the spine of the project):** the stratified subset and the
train/val/test split are computed **once** from a fixed seed and **persisted to
disk**. All four models load the exact same split artifact. Same data, same
splits, same `max_len`, same test set, same metrics — any deviation invalidates
the comparison.

---

## 1. Repository structure

```
NM/
├── IMPLEMENTATION_PLAN.md      # this file
├── README.md                   # how to run (local + Kaggle)
├── requirements.txt
├── src/nm_sentiment/
│   ├── __init__.py
│   ├── config.py               # single source of truth (dataclasses)
│   ├── seeding.py              # set_seed() across random/numpy/torch
│   ├── data.py                 # load Yelp → stratified subset → split → persist
│   ├── vocab.py                # LSTM-path tokenizer + capped vocabulary
│   ├── datasets.py             # torch Datasets for both preprocessing paths
│   ├── models/
│   │   ├── __init__.py
│   │   └── lstm.py             # LSTMClassifier (bidirectional flag covers Bi-LSTM)
│   ├── train_rnn.py            # LSTM/Bi-LSTM training loop
│   ├── train_hf.py             # BERT/RoBERTa via HF Trainer
│   ├── evaluate.py             # metrics + confusion matrix + classification_report
│   ├── plots.py                # confusion matrix PNG + train/val curve PNGs
│   └── compare.py              # assemble 4×4 summary table from per-model results
├── notebooks/
│   └── run_kaggle.ipynb        # thin runner: clone → configure → train → eval
├── scripts/
│   ├── prepare_data.py         # CLI: build + persist the split
│   ├── train_model.py          # CLI: train one model by name
│   └── build_report.py         # CLI: aggregate results into table + figures
├── data/splits/                # persisted split (indices + texts/labels), gitignored
├── results/
│   ├── metrics/                # <model>.json (all metrics + per-epoch history)
│   ├── figures/                # <model>_confusion.png, <model>_curves.png
│   └── comparison.csv          # final 4×4 table
└── .gitignore
```

---

## 2. Environment & dependencies

**`requirements.txt`** (pin majors loosely; Kaggle already ships most):
```
torch
transformers>=4.40
datasets>=2.19
scikit-learn
numpy
pandas
matplotlib
seaborn
tqdm
```

Notes:
- On Kaggle, `torch`, `transformers`, `datasets` are preinstalled with CUDA; we
  `pip install -q -U` only what's outdated.
- **Local caveat:** machine has no NVIDIA GPU and Python 3.14.4 (torch wheels lag).
  Local use is limited to authoring code + tiny-subset LSTM smoke tests on CPU.
  All real training happens on Kaggle. Document a 3.11/3.12 venv if local runs
  are needed.
- Hugging Face dataset auth: `Yelp/yelp_review_full` is public, no token needed.

---

## 3. `config.py` — single source of truth

Use frozen dataclasses so configs are explicit, logged, and serializable to JSON
alongside results.

```python
@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "Yelp/yelp_review_full"
    num_classes: int = 5
    subset_size: int = 50_000          # total reviews drawn from the 650k train pool
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1             # carved, untouched until final eval
    seed: int = 42
    split_dir: str = "data/splits"

@dataclass(frozen=True)
class RNNConfig:
    max_len: int = 256
    vocab_size: int = 25_000           # cap; <pad>=0, <unk>=1 reserved
    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1                # optionally 2 (second layer 64) if time allows
    dropout: float = 0.3
    bidirectional: bool = False        # Bi-LSTM flips this only
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 15
    early_stop_patience: int = 3       # on val loss
    seed: int = 42

@dataclass(frozen=True)
class HFConfig:
    model_name: str = "bert-base-uncased"   # or "roberta-base"
    max_len: int = 256
    batch_size: int = 16
    lr: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
```

Bi-LSTM = `replace(RNNConfig(), bidirectional=True)`; everything else identical to
LSTM. RoBERTa = `replace(HFConfig(), model_name="roberta-base")`; config otherwise
identical to BERT. This guarantees the only varying factor is the architecture.

---

## 4. Data pipeline

### 4.1 `seeding.py`
- `set_seed(seed)`: seeds `random`, `numpy`, `torch`, `torch.cuda`, sets
  `torch.backends.cudnn.deterministic=True`, `benchmark=False`, and HF
  `transformers.set_seed`. Called at the top of every entry point.

### 4.2 `data.py`
Functions:
- `load_pool(cfg) -> Dataset`: `load_dataset(cfg.dataset_name, split="train")`
  (the 650k pool). We never touch the official test split.
- `stratified_subset(pool, size, seed) -> indices`: per-class equal draw
  (size/5 each) using `sklearn`-style stratification on `label`. Assert resulting
  class counts are balanced.
- `make_splits(cfg) -> SplitArtifact`: from the subset, stratified split into
  train/val/test by `train_frac/val_frac/test_frac`. Returns texts + labels per
  split (not just indices — texts persisted so Kaggle runs are pool-independent).
- `persist(split_artifact, split_dir)`: write `train.parquet`, `val.parquet`,
  `test.parquet` + `meta.json` (seed, sizes, per-class counts, config hash).
- `load_splits(split_dir) -> SplitArtifact`: reload; raises if missing.

**Acceptance:** `meta.json` shows each split ~balanced across 5 classes; sizes are
40k/5k/5k; reloading yields identical data (hash check).

### 4.3 `vocab.py` (LSTM path only)
- Tokenizer: lowercase, regex word tokenization (`\b\w+\b`), no external deps.
- `build_vocab(train_texts, max_size)`: count tokens on **train split only** (no
  leakage), keep top `max_size-2`, reserve `<pad>=0`, `<unk>=1`.
- `encode(text, vocab, max_len)`: tokens→ids, truncate/pad to `max_len`, return ids.
- Persist vocab to `data/splits/vocab.json` so LSTM and Bi-LSTM share the exact
  same vocabulary.

### 4.4 `datasets.py`
- `RNNDataset(texts, labels, vocab, max_len)`: returns `(input_ids[L], label)`.
- `HFDataset`: wrap HF tokenizer output (`input_ids`, `attention_mask`, `label`)
  via `AutoTokenizer(...).__call__(truncation=True, max_length, padding="max_length")`.
  Implemented as `datasets.map(tokenize, batched=True)` for speed.
- Both consume the **same** persisted splits — only the encoding differs.

---

## 5. Models — `models/lstm.py`

```python
class LSTMClassifier(nn.Module):
    # embedding(vocab, embed_dim, padding_idx=0)
    # LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
    #      bidirectional=cfg.bidirectional, dropout=... if num_layers>1)
    # dropout → Linear(hidden_dim * (2 if bidirectional else 1), num_classes)
    # forward: take final hidden state(s); if bidirectional, concat last-layer
    #          forward & backward h_n; many-to-one.
```

- One class serves both LSTM and Bi-LSTM via `cfg.bidirectional`. No separate file.
- Use final hidden state (not mean-pool) for the many-to-one classification head,
  per brief.

BERT/RoBERTa need no custom model code — `AutoModelForSequenceClassification
.from_pretrained(name, num_labels=5)`.

---

## 6. Training

### 6.1 `train_rnn.py`
- Standard PyTorch loop: `CrossEntropyLoss`, `Adam(lr=cfg.lr)`.
- Per epoch: train pass → val pass; record `{train_loss, train_acc, val_loss,
  val_acc}`.
- **Early stopping** on val loss with `patience`; keep best-val checkpoint
  (`results/checkpoints/<model>.pt` or in-memory best state).
- Returns `history` (list of per-epoch dicts) + best model. Saves history into the
  model's metrics JSON so `plots.py` can draw curves.
- Device: `cuda` if available else `cpu`. Log which.

### 6.2 `train_hf.py`
- `AutoModelForSequenceClassification` + HF `Trainer` with `TrainingArguments`:
  `eval_strategy="epoch"`, `save_strategy="epoch"`,
  `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`,
  `per_device_train_batch_size=cfg.batch_size`, `learning_rate=cfg.lr`,
  `num_train_epochs=cfg.epochs`, `weight_decay`, `warmup_ratio`,
  `fp16=True` (P100/T4), `report_to="none"`, `seed=cfg.seed`.
- `compute_metrics` hook returns acc + macro P/R/F1 each epoch so the HF log
  history feeds the same curve plots.
- Extract `trainer.state.log_history` → normalize into the same `history` schema
  as the RNN path so plotting/aggregation is uniform.

**Memory guard:** if P100 16GB OOMs at batch 16/`max_len` 256, fall back to batch
8 + `gradient_accumulation_steps=2` (effective batch 16, keeps comparison fair).

---

## 7. Evaluation & reporting

### 7.1 `evaluate.py`
- `evaluate(model, test_loader_or_dataset) -> dict`: predictions on the **held-out
  test split** only. Compute via sklearn:
  - `accuracy_score`
  - `precision_recall_fscore_support(..., average="macro")`
  - full `classification_report(output_dict=True)` (per-class)
  - `confusion_matrix` (5×5, raw counts + normalized)
- Write `results/metrics/<model>.json`: config used, metrics, per-class report,
  confusion matrix, training history, seed, timestamp.

### 7.2 `plots.py`
- `plot_confusion(cm, model_name)` → `results/figures/<model>_confusion.png`
  (seaborn heatmap, normalized, labeled 1–5 stars).
- `plot_curves(history, model_name)` → `results/figures/<model>_curves.png`
  (train/val loss + train/val accuracy, two subplots).

### 7.3 `compare.py`
- Read all four `results/metrics/*.json`, build the 4×4 table
  (rows = models, cols = accuracy, macro precision, macro recall, macro F1).
- Save `results/comparison.csv` + a markdown version printed for the report.
- Print a sanity line comparing each model to §8 benchmark ranges; warn on outliers.

---

## 8. Sanity-check targets (correctness oracle)

| Model | Expected 5-class accuracy | Flag if |
|---|---|---|
| LSTM | ~50–55% | < 45% |
| Bi-LSTM | ~50–56% (may *not* beat LSTM — that's a valid finding) | < 45% |
| BERT | ~65–70% | < 60% |
| RoBERTa | ~70–73% | < 63% |

Expected qualitative result: **confusion concentrated in adjacent mid classes
(2–4 stars)**; extremes (1, 5) easiest. Confirm in confusion matrices and discuss
in the report.

---

## 9. Milestones & acceptance criteria

| # | Milestone | Deliverable | Acceptance | Compute |
|---|---|---|---|---|
| **M1** | Pipeline + fairness layer | `config.py`, `seeding.py`, `data.py`, persisted split + `meta.json` | Splits 40k/5k/5k, balanced classes, reload hash-stable | Local (CPU) |
| **M2** | LSTM end-to-end | `vocab.py`, `datasets.py`, `models/lstm.py`, `train_rnn.py`, `evaluate.py`, `plots.py` | Loop runs on 2k smoke subset; then full run lands in target range; curves + CM produced | Kaggle |
| **M3** | Bi-LSTM | `bidirectional=True` run | Full run, metrics + figures saved | Kaggle |
| **M4** | BERT | `train_hf.py` | Fine-tune 3 epochs; accuracy in range; figures saved | Kaggle |
| **M5** | RoBERTa | `roberta-base` config | Same pipeline; figures saved | Kaggle |
| **M6** | Comparison + figures | `compare.py`, `comparison.csv`, all figures, README | 4×4 table complete; all results in `results/`; sanity checks pass | Local |

Checkpoint with the user after **M1** (show split stats) and after **M2** (first
real model result) before scaling to the transformers.

---

## 10. Kaggle workflow (documented in README + `run_kaggle.ipynb`)

1. Push repo to GitHub (or upload `src/` as a Kaggle Dataset).
2. New Kaggle Notebook → enable **GPU P100** + Internet.
3. Runner cell: `!git clone <repo>` → `pip install -q -U -r requirements.txt`.
4. `from nm_sentiment import ...`; `set_seed(42)`.
5. M1 prepare/persist split to `/kaggle/working/data/splits` (or attach a prebuilt
   split Dataset so every model run reuses identical data).
6. Train one model per cell/session (mind the 30h/week + 12h/session limits);
   save `results/` artifacts and **download / save notebook version** after each.
7. Final session: run `compare.py` to assemble the table + figures.

**Quota budgeting (rough, P100):** LSTM/Bi-LSTM ~15–30 min each; BERT/RoBERTa
~3–5h each at 50k/256/3ep. Comfortably under 30h/week even with reruns. If tight,
drop to 2 epochs or batch 8 before reducing subset (subset must stay fixed across
models).

---

## 11. Reproducibility & fairness checklist (verify before reporting)

- [ ] Same persisted split loaded by all four models (assert via `meta.json` hash).
- [ ] Same `max_len=256` everywhere.
- [ ] Vocab built on **train split only**; shared by LSTM + Bi-LSTM.
- [ ] LSTM vs Bi-LSTM differ **only** in `bidirectional`.
- [ ] BERT vs RoBERTa differ **only** in `model_name`.
- [ ] Seed 42 logged in every metrics JSON.
- [ ] Test split untouched until final `evaluate.py`.
- [ ] Macro-averaged P/R/F1 reported (not just accuracy).
- [ ] Each result JSON embeds the exact config used.

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Kaggle session timeout mid-fine-tune | `save_strategy="epoch"` + `load_best_model_at_end`; resume from checkpoint; save notebook version per model |
| P100 OOM at batch 16 | Batch 8 + grad-accum 2 (effective batch unchanged) |
| Single-seed variance on macro-F1 | Log seed; note as limitation; optional second seed if quota allows |
| Data leakage via vocab/tokenization | Vocab from train only; splits stratified + persisted before any model sees data |
| Local Python 3.14 / no GPU blocks dev | Author + smoke-test LSTM on CPU tiny subset; all real runs on Kaggle |
| Bi-LSTM underperforms LSTM | Expected per Belaroussi 2025 — report as a finding, not a bug |

---

## 13. Out of scope (explicitly)

- ETRAN-format Serbian report writing (separate later step).
- State-of-the-art hyperparameter tuning ("don't over-engineer").
- Char-level CNN / additional architectures beyond the committed four.
- Using review metadata / multi-task setups (mentioned in refs, not required).
```
