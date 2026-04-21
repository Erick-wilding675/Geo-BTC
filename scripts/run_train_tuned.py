#!/usr/bin/env python3
"""Step 3 — Train and evaluate the fine-tuned stacked LSTM (Section 4).

The best epoch (lowest ``val_loss``) is checkpointed to
``models/best_btc_model.keras``.

Usage
-----
    python scripts/run_train_tuned.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402

from configs.paths import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    TUNED_MODEL_PATH,
)
from src.data import load_bitcoin_data, temporal_split  # noqa: E402
from src.evaluation import compute_regression_metrics, invert_predictions  # noqa: E402
from src.features import create_dataset, reshape_lstm, scale_data  # noqa: E402
from src.models import build_tuned_model  # noqa: E402
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
    tuned = cfg["training"]["tuned"]

    df = load_bitcoin_data(PROCESSED_DATA_PATH)
    train_df, test_df, val_df = temporal_split(df)
    train_scaled, test_scaled, _, scaler = scale_data(train_df, test_df, val_df, target_col)

    X_train, y_train = create_dataset(train_scaled, look_back)
    X_test, y_test = create_dataset(test_scaled, look_back)
    X_train, X_test = reshape_lstm(X_train), reshape_lstm(X_test)

    model = build_tuned_model((X_train.shape[1], 1), learning_rate=tuned["learning_rate"])
    model.summary(print_fn=logger.info)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=tuned["early_stopping_patience"],
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            str(TUNED_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=tuned["epochs"],
        batch_size=tuned["batch_size"],
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=2,
    )

    best = tf.keras.models.load_model(TUNED_MODEL_PATH)
    preds = best.predict(X_test, verbose=0)
    preds_real, y_test_real = invert_predictions(preds, y_test, scaler)
    metrics = compute_regression_metrics(y_test_real, preds_real)
    logger.info(
        "TUNED  MAE=$%.2f  RMSE=$%.2f  (best val_loss checkpoint loaded)",
        metrics.mae,
        metrics.rmse,
    )

    plot_learning_curve(
        history.history,
        title="Fine-Tuned Model — Learning Curve (MSE)",
        output=FIGURES_DIR / "tuned_learning_curve.png",
    )
    plot_forecast_vs_actual(
        actual=y_test_real.flatten(),
        predicted=preds_real.flatten(),
        title="Fine-Tuned LSTM — BTC/USD Forecast vs Actual (Oct–Dec 2014)",
        pred_color="green",
        output=FIGURES_DIR / "tuned_forecast_vs_actual.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
