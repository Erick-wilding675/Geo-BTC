# Reproducibility

This document describes **exactly** how to reproduce every figure and
table in the Geo-BTC paper from a clean machine.

## Supported environments

| Component | Version |
| --------- | ------- |
| Python | 3.10 – 3.12 |
| TensorFlow | 2.17.0 |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| scikit-learn | 1.5.1 |

The pinned versions live in `requirements.txt`, `environment.yml` and
`pyproject.toml`.

## Installation

Two routes are supported:

### A. `pip` (virtualenv)

```bash
git clone https://github.com/Erick-wilding675/Geo-BTC.git
cd Geo-BTC
git lfs install        # fetch model weights / figures
git lfs pull

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install                 # pip install -e ".[dev,notebook]"
```

### B. `conda`

```bash
git clone https://github.com/Erick-wilding675/Geo-BTC.git
cd Geo-BTC
git lfs pull

make env                     # conda env create -f environment.yml
conda activate geo-btc
pip install -e .
```

## Reproducing the paper

```bash
make pipeline                # run all 6 stages in order
```

To run a subset:

```bash
make train-baseline
make train-tuned
make grid-search
make train-final
make residuals
```

Or invoke scripts directly:

```bash
python scripts/run_pipeline.py --config configs/config.yaml
python scripts/run_pipeline.py --skip-baseline --skip-tuned
```

## Produced artefacts

| Path | Produced by |
| ---- | ----------- |
| `models/baseline_lstm.keras` | `run_train_baseline.py` |
| `models/best_btc_model.keras` | `run_train_tuned.py` |
| `models/final_lstm.keras` | `run_train_final.py` |
| `results/grid_search_results.csv` | `run_grid_search.py` |
| `results/analysis_residuals_2015.csv` | `run_residual_analysis.py` |
| `results/investigation_periods.csv` | `run_residual_analysis.py` |
| `reports/figures/*.png` | training + analysis scripts |

## Determinism

`src.utils.seed.set_global_seed(42)` is invoked at the top of every
script. On CPU this yields bit-for-bit reproducible runs. On GPU,
additionally export:

```bash
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1
```

## Running the test suite

```bash
make test            # pytest + coverage
make lint            # ruff
```

## Questions?

Open an issue on GitHub or email the corresponding author
(`cmendeserick@gmail.com`).
