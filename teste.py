"""Smoke-test: sanity-check the data pipeline (no model training).

DEPRECATED — this file was the original scratchpad used while extracting
the notebook into the :mod:`src` package. The production pipeline lives
under ``scripts/``; see ``scripts/run_pipeline.py``.

This script is kept as a lightweight smoke test that runs in seconds and
validates that the data → split → scale → window chain is wired up
correctly on the current machine.

Usage
-----
    python teste.py
"""

from __future__ import annotations

from configs.paths import PROCESSED_DATA_PATH
from src.data import load_bitcoin_data, temporal_split
from src.features import create_dataset, reshape_lstm, scale_data

TARGET_COL = "Close"
LOOK_BACK = 90


def main() -> int:
    df = load_bitcoin_data(PROCESSED_DATA_PATH)

    train_df, test_df, val_df = temporal_split(df)
    train_scaled, test_scaled, val_scaled, _ = scale_data(
        train_df, test_df, val_df, TARGET_COL
    )

    X_train, _ = create_dataset(train_scaled, LOOK_BACK)
    X_test, _ = create_dataset(test_scaled, LOOK_BACK)
    X_val, _ = create_dataset(val_scaled, LOOK_BACK)

    X_train = reshape_lstm(X_train)
    X_test = reshape_lstm(X_test)
    X_val = reshape_lstm(X_val)

    print("Shapes:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)
    print("  X_val  :", X_val.shape)
    print("\nOK — pipeline wired up correctly.")
    print("To reproduce the full paper run: python scripts/run_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
