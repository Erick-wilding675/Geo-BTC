#!/usr/bin/env python3
"""Phase 2 — Step 2: compute the Explaining Ratio (ER) and per-category table.

Usage
-----
    python scripts/run_explaining_ratio.py --config configs/config.yaml
    python scripts/run_explaining_ratio.py --threshold 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.paths import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    EXPLAINING_RATIO_CSV_PATH,
    INFERENCE_TABLE_CSV_PATH,
    RESULTS_DIR,
)
from src.inference import compute_explaining_ratio, summarise_by_category  # noqa: E402
from src.inference.metrics import DEFAULT_ERROR_THRESHOLD  # noqa: E402
from src.utils import get_logger, load_config, set_global_seed  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--inference", type=Path, default=INFERENCE_TABLE_CSV_PATH)
    p.add_argument("--out", type=Path, default=EXPLAINING_RATIO_CSV_PATH)
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_ERROR_THRESHOLD,
        help="Absolute-error threshold in USD (Rule of 20 by default).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.get("seed", 42))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.inference.exists():
        raise FileNotFoundError(
            f"inference_table.csv not found at {args.inference}. "
            "Run scripts/run_inference_merge.py first."
        )

    df = pd.read_csv(args.inference, parse_dates=["Event_Date"])

    er = compute_explaining_ratio(df, threshold_usd=args.threshold)
    logger.info(
        "Explaining Ratio (theta=$%.0f USD): %d / %d explained outliers (ER = %.2f%%)",
        er.threshold_usd,
        er.explained_outliers,
        er.total_outliers,
        er.ratio_pct,
    )

    summary = summarise_by_category(df, threshold_usd=args.threshold)
    if summary.empty:
        logger.warning("No outliers above the threshold — nothing to summarise.")
    else:
        logger.info("Per-category summary:\n%s", summary.to_string(index=False))

    summary.to_csv(args.out, index=False)
    logger.info("Exported: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
