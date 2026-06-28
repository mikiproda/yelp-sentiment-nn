# Deep Learning Sentiment Analysis on Yelp Review Full

Comparative study of four architectures — **LSTM, Bi-LSTM, BERT, RoBERTa** — on
the 5-class Yelp Review Full star-rating task. Individual Neural Networks course
project (ETF Belgrade). See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for
the full design and milestones.

All four models train and are evaluated on an **identical persisted stratified
split** (built once, fixed seed) so the comparison is fair. Metrics: accuracy,
macro precision/recall/F1, plus a confusion matrix and training curves per model.

## Project layout

```
src/nm_sentiment/   # library: config, data, vocab, datasets, models, train, eval, plots
scripts/            # CLIs: prepare_data.py, train_model.py
notebooks/          # run_kaggle.ipynb — the GPU runner
data/splits/        # persisted split (parquet, gitignored; rebuilt from code)
results/            # metrics JSON + figures (generated)
```

## Running on Kaggle (recommended — needs a GPU)

Transformer fine-tuning needs a GPU; the LSTMs are far faster on one too. Use a
free **Kaggle Notebook** with GPU.

1. Push this repo to GitHub.
2. On Kaggle: **Create → New Notebook**, then in the right panel set
   **Accelerator → GPU** and **Internet → On** (both require one-time phone
   verification on your Kaggle account).
3. Upload / open [`notebooks/run_kaggle.ipynb`](notebooks/run_kaggle.ipynb),
   edit `REPO_URL` to your repo, and **Run All**.
4. The notebook clones the code, rebuilds the split (`prepare_data.py`), trains
   each model, and zips `results/` to `/kaggle/working/results.zip` for download.

GPU budget: 30 h/week free — ample for all four models at the 50k subset.

## Running locally

Local use is limited to authoring + small smoke tests (no NVIDIA GPU assumed).

```bash
pip install -r requirements.txt
python scripts/prepare_data.py                    # build + persist the split (M1)
python scripts/train_model.py --model lstm --smoke 2000   # quick loop check
python scripts/train_model.py --model lstm        # full LSTM (slow on CPU)
```

`--model` accepts `lstm`, `bilstm`, `bert`, `roberta`. `bert`/`roberta` require
the transformer training code (milestones M4/M5).

## Reproducibility

- Fixed seed (42) across Python/NumPy/PyTorch/Transformers.
- The split carries a `config_hash`; every training run asserts the on-disk split
  matches the current `DataConfig` before training (fairness guard).
- Each run writes `results/metrics/<model>.json` with the exact config, per-epoch
  history, test metrics, per-class report, and confusion matrix.
