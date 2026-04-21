#!/usr/bin/env python3
"""Step 6 — Run 2015 inference, detect outliers, export Phase-2 tables.

Reproduces Sections 6 and 7 of the paper. Produces:

    results/analysis_residuals_2015.csv
    results/investigation_periods.csv
    reports/figures/residual_panels_2015.png

Usage
-----
    python scripts/run_residual_analysis.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.paths import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    FIGURES_DIR,
    FINAL_MODEL_PATH,
    INVESTIGATION_CSV_PATH,
    PROCESSED_DATA_PATH,
    RESIDUALS_CSV_PATH,
    RESULTS_DIR,
)
from src.analysis import (  # noqa: E402
    build_investigation_periods,
    build_residual_dataframe,
    detect_outliers_2sigma,
)
from src.data import load_bitcoin_data, temporal_split  # noqa: E402
from src.evaluation import compute_regression_metrics, invert_predictions  # noqa: E402
from src.features import create_dataset, reshape_lstm, scale_data  # noqa: E402
from src.utils import get_logger, load_config, set_global_seed  # noqa: E402
from src.visualization import plot_residual_panels  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.get("seed", 42))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found at {FINAL_MODEL_PATH}. "
            "Run `python scripts/run_train_final.py` first."
        )

    target_col = cfg["data"]["target_col"]
    look_back = cfg["features"]["look_back"]
    top_n = cfg["analysis"]["top_n_errors"]
    pre = cfg["analysis"]["investigation_window_pre"]
    post = cfg["analysis"]["investigation_window_post"]

    df = load_bitcoin_data(PROCESSED_DATA_PATH)
    train_df, test_df, val_df = temporal_split(df)
    _, _, val_scaled, scaler = scale_data(train_df, test_df, val_df, target_col)

    X_val, y_val = create_dataset(val_scaled, look_back)
    X_val = reshape_lstm(X_val)

    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    preds = model.predict(X_val, verbose=0)
    preds_real, y_val_real = invert_predictions(preds, y_val, scaler)

    metrics = compute_regression_metrics(y_val_real, preds_real)
    logger.info(
        "FINAL (2015 out-of-sample)  MAE=$%.2f  RMSE=$%.2f",
        metrics.mae,
        metrics.rmse,
    )

    dates_2015 = val_df.index[look_back:]
    residuals_df = build_residual_dataframe(dates_2015, y_val_real, preds_real)
    flagged_df, thr = detect_outliers_2sigma(residuals_df)
    flagged_df.to_csv(RESIDUALS_CSV_PATH, index=False)
    logger.info(
        "Outlier thresholds: mean=%.4f  std=%.4f  upper=%.4f  lower=%.4f",
        thr.mean,
        thr.std,
        thr.upper,
        thr.lower,
    )
    logger.info("Exported: %s", RESIDUALS_CSV_PATH)

    investigation_df = build_investigation_periods(flagged_df, pre_days=pre, post_days=post)
    investigation_df.to_csv(INVESTIGATION_CSV_PATH, index=False)
    logger.info(
        "Exported: %s  (%d outlier events)",
        INVESTIGATION_CSV_PATH,
        len(investigation_df),
    )

    top_errors = (
        flagged_df.sort_values(by="Abs_Error", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    logger.info(
        "TOP-%d dates for geopolitical investigation:\n%s",
        top_n,
        top_errors[["Date", "Actual_Price", "Predicted_Price", "Abs_Error", "Pct_Error"]].to_string(
            index=False
        ),
    )

    plot_residual_panels(
        dates=dates_2015,
        actual=y_val_real.flatten(),
        predicted=preds_real.flatten(),
        abs_error=flagged_df["Abs_Error"].values,
        mae=metrics.mae,
        output=FIGURES_DIR / "residual_panels_2015.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
