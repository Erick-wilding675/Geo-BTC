#!/usr/bin/env python3
"""End-to-end reproduction of the Geo-BTC paper.

Orchestrates the full pipeline:

    Phase 1 — quantitative
        1. scripts/run_preprocessing.py
        2. scripts/run_train_baseline.py
        3. scripts/run_train_tuned.py
        4. scripts/run_grid_search.py
        5. scripts/run_train_final.py
        6. scripts/run_residual_analysis.py

    Phase 2 — causal inference
        7. scripts/run_inference_merge.py
        8. scripts/run_explaining_ratio.py
        9. scripts/run_inference_plot.py

Each stage can be skipped via CLI flags, which is useful when iterating
on downstream analysis without re-training the network.

Usage
-----
    python scripts/run_pipeline.py --config configs/config.yaml
    python scripts/run_pipeline.py --skip-baseline --skip-tuned --skip-grid-search
    python scripts/run_pipeline.py --only-phase2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.paths import DEFAULT_CONFIG_PATH  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

PHASE1 = [
    ("preprocess", "scripts/run_preprocessing.py"),
    ("baseline", "scripts/run_train_baseline.py"),
    ("tuned", "scripts/run_train_tuned.py"),
    ("grid-search", "scripts/run_grid_search.py"),
    ("final", "scripts/run_train_final.py"),
    ("residuals", "scripts/run_residual_analysis.py"),
]
PHASE2 = [
    ("merge", "scripts/run_inference_merge.py"),
    ("er", "scripts/run_explaining_ratio.py"),
    ("plot", "scripts/run_inference_plot.py"),
]
STAGES = PHASE1 + PHASE2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument(
        "--only-phase2",
        action="store_true",
        help="Skip every Phase 1 stage and run only the inference pipeline.",
    )
    for name, _ in STAGES:
        p.add_argument(
            f"--skip-{name}",
            action="store_true",
            help=f"Skip the '{name}' stage.",
        )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    skipped_phase1 = bool(args.only_phase2)

    for name, script in STAGES:
        if skipped_phase1 and (name, script) in PHASE1:
            logger.info("SKIP   %s  (--only-phase2)", name)
            continue
        if getattr(args, f"skip_{name.replace('-', '_')}"):
            logger.info("SKIP   %s", name)
            continue
        logger.info("START  %s  (%s)", name, script)
        result = subprocess.run(
            [sys.executable, str(ROOT / script), "--config", str(args.config)],
            cwd=ROOT,
            check=False,
        )
        if result.returncode != 0:
            logger.error("FAILED %s (exit %d)", name, result.returncode)
            return result.returncode
        logger.info("DONE   %s", name)

    logger.info("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
