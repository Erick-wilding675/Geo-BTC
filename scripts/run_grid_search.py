#!/usr/bin/env python3
"""Step 4 — Hyper-parameter grid search (Section 5).

Exports the results to ``results/grid_search_results.csv`` sorted by MAE.

Usage
-----
    python scripts/run_grid_search.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.paths import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    GRID_SEARCH_CSV_PATH,
    PROCESSED_DATA_PATH,
    RESULTS_DIR,
)
from src.data import load_bitcoin_data, temporal_split  # noqa: E402
from src.features import scale_data  # noqa: E402
from src.models.grid_search import grid_search_lstm  # noqa: E402
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    target_col = cfg["data"]["target_col"]
    gs = cfg["grid_search"]

    df = load_bitcoin_data(PROCESSED_DATA_PATH)
    train_df, test_df, val_df = temporal_split(df)
    train_scaled, test_scaled, _, scaler = scale_data(train_df, test_df, val_df, target_col)

    results = grid_search_lstm(
        train_scaled=train_scaled,
        test_scaled=test_scaled,
        scaler=scaler,
        look_back_options=gs["look_back_options"],
        neurons_options=gs["neurons_options"],
        epochs=gs["epochs"],
        batch_size=gs["batch_size"],
        patience=gs["patience"],
        verbose=0,
    )

    results.to_csv(GRID_SEARCH_CSV_PATH, index=False)
    logger.info("Grid-search results (sorted by MAE):\n%s", results.to_string(index=False))
    best = results.iloc[0]
    logger.info(
        "Best configuration: %s  (MAE=$%.2f, look_back=%d, neurons=%d)",
        best["config"],
        best["mae"],
        best["look_back"],
        best["neurons"],
    )
    logger.info("Exported: %s", GRID_SEARCH_CSV_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
