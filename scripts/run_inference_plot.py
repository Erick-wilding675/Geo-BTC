#!/usr/bin/env python3
"""Phase 2 — Step 3: build the Plotly inference storyboard.

Reads the residual time series (Real vs Predicted) and the merged
inference table, then renders the final 2015 storyboard with event
flags for every outlier above the configured USD threshold.

Usage
-----
    python scripts/run_inference_plot.py --config configs/config.yaml
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
    FIGURES_DIR,
    INFERENCE_PLOT_HTML_PATH,
    INFERENCE_TABLE_CSV_PATH,
    RESIDUALS_CSV_PATH,
)
from src.inference import compute_explaining_ratio  # noqa: E402
from src.inference.metrics import DEFAULT_ERROR_THRESHOLD  # noqa: E402
from src.inference.plot import build_inference_figure, export_figure  # noqa: E402
from src.utils import get_logger, load_config, set_global_seed  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--residuals", type=Path, default=RESIDUALS_CSV_PATH)
    p.add_argument("--inference", type=Path, default=INFERENCE_TABLE_CSV_PATH)
    p.add_argument("--out", type=Path, default=INFERENCE_PLOT_HTML_PATH)
    p.add_argument("--threshold", type=float, default=DEFAULT_ERROR_THRESHOLD)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg.get("seed", 42))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not args.residuals.exists():
        raise FileNotFoundError(
            f"residuals CSV not found at {args.residuals}. "
            "Run scripts/run_residual_analysis.py first."
        )
    if not args.inference.exists():
        raise FileNotFoundError(
            f"inference_table.csv not found at {args.inference}. "
            "Run scripts/run_inference_merge.py first."
        )

    residuals = pd.read_csv(args.residuals, parse_dates=["Date"])
    inference = pd.read_csv(args.inference, parse_dates=["Event_Date"])

    er = compute_explaining_ratio(inference, threshold_usd=args.threshold)
    title = (
        f"Geo-BTC — 2015 Causal-Inference Storyboard  "
        f"(ER = {er.ratio_pct:.1f}%, theta = ${er.threshold_usd:.0f})"
    )

    fig = build_inference_figure(
        residuals_df=residuals,
        inference_df=inference,
        threshold_usd=args.threshold,
        title=title,
    )
    export_figure(fig, args.out)
    logger.info("Exported: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
