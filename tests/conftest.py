"""Shared pytest fixtures and project-root setup."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def synthetic_btc_df() -> pd.DataFrame:
    """Synthetic OHLCV dataset covering 2012-01-01 → 2015-12-31."""
    rng = np.random.default_rng(seed=42)
    idx = pd.date_range("2012-01-01", "2015-12-31", freq="D", name="DateTime")
    close = 10 + np.cumsum(rng.normal(0, 0.5, size=len(idx)))
    close = np.clip(close, 1.0, None)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume_(BTC)": rng.uniform(10, 100, size=len(idx)),
        },
        index=idx,
    )


@pytest.fixture()
def synthetic_csv(tmp_path, synthetic_btc_df) -> Path:
    """Persist the synthetic dataframe as a CSV for load-round-trip tests."""
    out = tmp_path / "synthetic.csv"
    synthetic_btc_df.reset_index().to_csv(out, index=False)
    return out
