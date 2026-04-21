#!/usr/bin/env python3
"""Step 2 — Train and evaluate the baseline single-layer LSTM.

Reproduces Section 3 of the paper. Saves the trained model to
``models/baseline_lstm.keras`` and the learning-curve / forecast figures
to ``reports/figures/``.

Usage
-----
    python scripts/run_train_baseline.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tensorflow.keras.callbacks import EarlyStopping  # noqa: E402

from configs.paths import (  # noqa: E402
    BASELINE_MODEL_PATH,
    DEFAULT_CONFIG_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
)
from src.data import load_bitcoin_data, temporal_split  # noqa: E402
from src.evaluation import compute_regression_metrics, invert_predictions  # noqa: E402
from src.features import create_dataset, reshape_lstm, scale_data  # noqa: E402
from src.models import build_baseline  # noqa: E402
from src.utils import get_logger, load_config, set_global_seed  # noqa: E402
from src.visualization import plot_forecast_vs_actual, plot_learning_curve  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.get("seed", 42))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    target_col = cfg["data"]["target_col"]
    look_back = cfg["features"]["look_back"]

    df = load_bitcoin_data(PROCESSED_DATA_PATH)
    train_df, test_df, val_df = temporal_split(df)
    train_scaled, test_scaled, _, scaler = scale_data(train_df, test_df, val_df, target_col)

    X_train, y_train = create_dataset(train_scaled, look_back)
    X_test, y_test = create_dataset(test_scaled, look_back)
    X_train, X_test = reshape_lstm(X_train), reshape_lstm(X_test)

    model = build_baseline((X_train.shape[1], 1))
    model.summary(print_fn=logger.info)

    es = EarlyStopping(
        monitor="val_loss",
        patience=cfg["training"]["baseline"]["early_stopping_patience"],
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=cfg["training"]["baseline"]["epochs"],
        batch_size=cfg["training"]["baseline"]["batch_size"],
        validation_data=(X_test, y_test),
        callbacks=[es],
        verbose=2,
    )

    preds = model.predict(X_test, verbose=0)
    preds_real, y_test_real = invert_predictions(preds, y_test, scaler)
    metrics = compute_regression_metrics(y_test_real, preds_real)
    logger.info(
        "BASELINE  MAE=$%.2f (%.2f%%)  RMSE=$%.2f (%.2f%%)",
        metrics.mae,
        metrics.mae_pct,
        metrics.rmse,
        metrics.rmse_pct,
    )

    model.save(BASELINE_MODEL_PATH)
    logger.info("Saved model to %s", BASELINE_MODEL_PATH)

    plot_learning_curve(
        history.history,
        title="Baseline — Learning Curve (MSE)",
        output=FIGURES_DIR / "baseline_learning_curve.png",
    )
    plot_forecast_vs_actual(
        actual=y_test_real.flatten(),
        predicted=preds_real.flatten(),
        title="Baseline LSTM — BTC/USD Forecast vs Actual (Oct–Dec 2014)",
        pred_color="tomato",
        output=FIGURES_DIR / "baseline_forecast_vs_actual.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
