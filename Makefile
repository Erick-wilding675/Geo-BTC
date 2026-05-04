# ==============================================================================
#  Geo-BTC — Reproducibility Makefile
# ==============================================================================
#  Usage:
#      make help          List available targets
#      make install       Install package + dependencies (editable)
#      make env           Create conda environment
#      make pipeline      Reproduce the whole paper (Phase 1 + Phase 2)
#      make test          Run unit tests
#      make lint          Run ruff lint checks
#      make format        Apply ruff formatting
#      make clean         Remove caches, compiled artefacts, __pycache__
# ==============================================================================

SHELL := /bin/bash
PYTHON ?= python3
CONFIG ?= configs/config.yaml

.PHONY: help install env preprocess train-baseline train-tuned grid-search \
        train-final residuals merge er plot phase2 pipeline test lint format \
        clean notebook

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode + dev dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,notebook]"

env: ## Create conda environment from environment.yml
	conda env create -f environment.yml

# ── Phase 1: quantitative ─────────────────────────────────────────────────────
preprocess: ## Generate the processed CSV from raw
	$(PYTHON) scripts/run_preprocessing.py --config $(CONFIG)

train-baseline: ## Train baseline single-layer LSTM
	$(PYTHON) scripts/run_train_baseline.py --config $(CONFIG)

train-tuned: ## Train fine-tuned stacked LSTM
	$(PYTHON) scripts/run_train_tuned.py --config $(CONFIG)

grid-search: ## Run hyper-parameter grid search
	$(PYTHON) scripts/run_grid_search.py --config $(CONFIG)

train-final: ## Train final model on merged 2012-2014 data
	$(PYTHON) scripts/run_train_final.py --config $(CONFIG)

residuals: ## Run 2015 inference + outlier detection
	$(PYTHON) scripts/run_residual_analysis.py --config $(CONFIG)

# ── Phase 2: causal inference ─────────────────────────────────────────────────
merge: ## Merge LSTM outliers with the qualitative event database
	$(PYTHON) scripts/run_inference_merge.py --config $(CONFIG)

er: ## Compute the Explaining Ratio (ER) and per-category table
	$(PYTHON) scripts/run_explaining_ratio.py --config $(CONFIG)

plot: ## Build the Plotly inference storyboard
	$(PYTHON) scripts/run_inference_plot.py --config $(CONFIG)

phase2: merge er plot ## Reproduce Phase 2 only

# ── Full reproduction ─────────────────────────────────────────────────────────
pipeline: ## Reproduce the entire paper end-to-end
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG)

# ── Quality gates ─────────────────────────────────────────────────────────────
test: ## Run the test suite with coverage
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

lint: ## Lint source code with ruff
	$(PYTHON) -m ruff check src scripts tests

format: ## Auto-format source code with ruff
	$(PYTHON) -m ruff format src scripts tests
	$(PYTHON) -m ruff check --fix src scripts tests

notebook: ## Launch JupyterLab
	$(PYTHON) -m jupyter lab

clean: ## Remove caches and compiled Python artefacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info .coverage htmlcov
