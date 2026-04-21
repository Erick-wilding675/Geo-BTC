#!/usr/bin/env python3
"""Step 5 — Train the final LSTM on merged 2012-2014 data (Section 6).

The final model is trained on (train + test) and used downstream to
perform out-of-sample inference for all 2015 trading days.

Usage
-----
    python scripts/run_train_final.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.paths import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    FINAL_MODEL_PATH,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
)
from src.data import load_bitcoin_data, temporal_split  # noqa: E402
from src.features import create_dataset, reshape_lstm, scale_data  # noqa: E402
from src.models import build_final_model  # noqa: E402
from src.utils import get_logger, load_config, set_global_seed  # noqa: E402

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

    target_col = cfg["data"]["target_col"]
    look_back = cfg["features"]["look_back"]
    final = cfg["training"]["final"]

    df = load_bitcoin_data(PROCESSED_DATA_PATH)
    train_df, test_df, val_df = temporal_split(df)
    train_scaled, test_scaled, val_scaled, scaler = scale_data(
        train_df, test_df, val_df, target_col
    )

    X_train, y_train = create_dataset(train_scaled, look_back)
    X_test, y_test = create_dataset(test_scaled, look_back)

    X_final_train = np.concatenate((X_train, X_test), axis=0)
    y_final_train = np.concatenate((y_train, y_test), axis=0)
    X_final_train = reshape_lstm(X_final_train)

    logger.info(
        "Final training data: %d samples (merged train+test, 2012-2014)",
        X_final_train.shape[0],
    )

    model = build_final_model((X_final_train.shape[1], 1), units=final["units"])
    model.summary(print_fn=logger.info)

    model.fit(
        X_final_train,
        y_final_train,
        epochs=final["epochs"],
        batch_size=final["batch_size"],
        verbose=2,
    )

    model.save(FINAL_MODEL_PATH)
    logger.info("Saved final model to %s", FINAL_MODEL_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
