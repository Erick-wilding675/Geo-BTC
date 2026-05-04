#!/usr/bin/env python3
"""Phase 2 — Step 1: merge LSTM outliers with the qualitative event database.

Usage
-----
    python scripts/run_inference_merge.py --config configs/config.yaml
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
    INVESTIGATION_CSV_PATH,
    QUALITATIVE_CSV_PATH,
    RESULTS_DIR,
    INFERENCE_TABLE_CSV_PATH,
)
from src.inference import build_inference_table  # noqa: E402
from src.utils import get_logger, load_config, set_global_seed  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--investigation", type=Path, default=INVESTIGATION_CSV_PATH)
    p.add_argument("--qualitative", type=Path, default=QUALITATIVE_CSV_PATH)
    p.add_argument("--out", type=Path, default=INFERENCE_TABLE_CSV_PATH)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.get("seed", 42))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.investigation.exists():
        raise FileNotFoundError(
            f"investigation_periods.csv not found at {args.investigation}. "
            "Run scripts/run_residual_analysis.py first."
        )
    if not args.qualitative.exists():
        raise FileNotFoundError(f"Qualitative CSV not found at {args.qualitative}")

    invest_df = pd.read_csv(args.investigation, parse_dates=["Event_Date", "Start_Date", "End_Date"])
    qual_df = pd.read_csv(args.qualitative)

    logger.info(
        "Merging %d LSTM outliers with %d qualitative windows…",
        len(invest_df),
        len(qual_df),
    )
    merged = build_inference_table(invest_df, qual_df)

    matched = merged["Category"].notna().sum()
    logger.info("Matched %d / %d outliers to a qualitative window.", matched, len(merged))

    merged.to_csv(args.out, index=False)
    logger.info("Exported: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
