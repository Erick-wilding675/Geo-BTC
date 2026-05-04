"""Canonical project paths.

All modules and scripts should import from this file instead of hard-coding
paths — this keeps the pipeline portable when cloned to a different host.
"""

from __future__ import annotations

from pathlib import Path

# ── Project root (the repository's top-level directory) ───────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"

RAW_DATA_PATH: Path = RAW_DIR / "bitstamp_cleaned.csv"
PROCESSED_DATA_PATH: Path = PROCESSED_DIR / "bitstamp_daily_final.csv"
WIKIPEDIA_EDITS_PATH: Path = EXTERNAL_DIR / "wikipedia_edits.csv"
QUALITATIVE_CSV_PATH: Path = EXTERNAL_DIR / "qualitative_analysis.csv"
QUALITATIVE_XLSX_PATH: Path = EXTERNAL_DIR / "qualitative_analysis.xlsx"

# ── Model artefacts ───────────────────────────────────────────────────────────
MODELS_DIR: Path = BASE_DIR / "models"
BASELINE_MODEL_PATH: Path = MODELS_DIR / "baseline_lstm.keras"
TUNED_MODEL_PATH: Path = MODELS_DIR / "best_btc_model.keras"
FINAL_MODEL_PATH: Path = MODELS_DIR / "final_lstm.keras"

# ── Results & reports ─────────────────────────────────────────────────────────
RESULTS_DIR: Path = BASE_DIR / "results"
REPORTS_DIR: Path = BASE_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# ── Phase 1 outputs ───────────────────────────────────────────────────────────
RESIDUALS_CSV_PATH: Path = RESULTS_DIR / "analysis_residuals_2015.csv"
INVESTIGATION_CSV_PATH: Path = RESULTS_DIR / "investigation_periods.csv"
GRID_SEARCH_CSV_PATH: Path = RESULTS_DIR / "grid_search_results.csv"

# ── Phase 2 outputs ───────────────────────────────────────────────────────────
INFERENCE_TABLE_CSV_PATH: Path = RESULTS_DIR / "inference_table.csv"
EXPLAINING_RATIO_CSV_PATH: Path = RESULTS_DIR / "explaining_ratio.csv"
INFERENCE_PLOT_HTML_PATH: Path = FIGURES_DIR / "inference_storyboard_2015.html"

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_DIR: Path = BASE_DIR / "configs"
DEFAULT_CONFIG_PATH: Path = CONFIG_DIR / "config.yaml"
