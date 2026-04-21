#!/usr/bin/env python3
"""Step 1 — Load the processed dataset and report summary statistics.

This script does **not** regenerate the processed CSV from raw (that step
was performed once, offline, from the Bitstamp tick data). Rather it
validates the on-disk artefact and produces a short diagnostics report
consumed by the downstream training scripts.

Usage
-----
    python scripts/run_preprocessing.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.paths import DEFAULT_CONFIG_PATH, PROCESSED_DATA_PATH  # noqa: E402
from src.data import load_bitcoin_data, temporal_split  # noqa: E402
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

    logger.info("Loading processed dataset from %s", PROCESSED_DATA_PATH)
    df = load_bitcoin_data(PROCESSED_DATA_PATH)
    logger.info("Loaded %d rows spanning %s -> %s", len(df), df.index.min(), df.index.max())

    train_df, test_df, val_df = temporal_split(df)
    logger.info(
        "Temporal split -> train: %d | test: %d | val: %d",
        len(train_df),
        len(test_df),
        len(val_df),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
