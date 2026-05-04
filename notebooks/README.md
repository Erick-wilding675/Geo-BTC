# `notebooks/`

| Notebook | Purpose |
| -------- | ------- |
| `Geo-BTC-LSTM_Model.ipynb` | **Primary research notebook (Phase 1)** — data preparation, baseline / tuned / final LSTM training, residual analysis, exports for Phase 2 |
| `pedictionlive.ipynb` | Exploratory live-price experiments using `yfinance`. Optional; not required to reproduce the paper |

## Relationship with `src/` and `scripts/`

The logic of `Geo-BTC-LSTM_Model.ipynb` was refactored into the
`src/` package and split across the CLI scripts in `scripts/` for
programmatic, test-covered reproduction. The Phase 2 logic (merge,
Explaining Ratio, Plotly storyboard) was authored directly under
`src/inference/` and has no notebook counterpart.

| Phase / notebook section | Script |
| ------------------------ | ------ |
| §2 Data loading & preprocessing | `scripts/run_preprocessing.py` |
| §3 Baseline LSTM | `scripts/run_train_baseline.py` |
| §4 Fine-tuned LSTM | `scripts/run_train_tuned.py` |
| §5 Grid search | `scripts/run_grid_search.py` |
| §6 Final model + 2015 inference | `scripts/run_train_final.py` |
| §7 Outlier detection & exports | `scripts/run_residual_analysis.py` |
| Phase 2 — merge LSTM ↔ qualitative DB | `scripts/run_inference_merge.py` |
| Phase 2 — Explaining Ratio + per-category MAE | `scripts/run_explaining_ratio.py` |
| Phase 2 — Plotly inference storyboard | `scripts/run_inference_plot.py` |

The Phase-1 notebook is the canonical narrative of the paper; the
scripts are the engineering surface for reproducibility and CI.
