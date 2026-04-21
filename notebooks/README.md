# `notebooks/`

| Notebook | Purpose |
| -------- | ------- |
| `Geo-BTC-LSTM_Model.ipynb` | **Primary research notebook** — data preparation, baseline/tuned/final LSTM training, residual analysis, exports for Phase 2 |
| `pedictionlive.ipynb` | Exploratory live-price experiments using `yfinance`. Optional; not required to reproduce the paper |

## Relationship with `src/` and `scripts/`

The logic of `Geo-BTC-LSTM_Model.ipynb` has been refactored into the
`src/` package and split across the CLI scripts in `scripts/` for a
programmatic, test-covered reproduction. Sectional correspondence:

| Notebook section | Script |
| ---------------- | ------ |
| §2 Data loading & preprocessing | `scripts/run_preprocessing.py` |
| §3 Baseline LSTM | `scripts/run_train_baseline.py` |
| §4 Fine-tuned LSTM | `scripts/run_train_tuned.py` |
| §5 Grid search | `scripts/run_grid_search.py` |
| §6 Final model + 2015 inference | `scripts/run_train_final.py` |
| §7 Outlier detection & exports | `scripts/run_residual_analysis.py` |

The notebook is kept as the canonical narrative of the paper; the
scripts are the engineering surface for reproducibility and CI.
