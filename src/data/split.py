"""Strictly-temporal train/test/validation splitting to prevent data leakage."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ── Default temporal boundaries (see paper Section 2.3) ────────────────────────
TRAIN_END: str = "2014-09-30"
TEST_START: str = "2014-10-01"
TEST_END: str = "2014-12-31"
VAL_START: str = "2015-01-01"
VAL_END: str = "2015-12-31"


@dataclass(frozen=True)
class SplitBounds:
    """Immutable boundaries for the temporal split."""

    train_end: str = TRAIN_END
    test_start: str = TEST_START
    test_end: str = TEST_END
    val_start: str = VAL_START
    val_end: str = VAL_END


def temporal_split(
    df: pd.DataFrame,
    bounds: SplitBounds | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame into train / test / validation sets.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by ``DateTime`` (or any datetime-like index).
    bounds : SplitBounds, optional
        Custom temporal boundaries. Defaults to the bounds used in the paper:
        train (... -> 2014-09-30), test (2014-10-01 -> 2014-12-31),
        validation (2015-01-01 -> 2015-12-31).

    Returns
    -------
    tuple
        ``(train_df, test_df, val_df)`` — three independent copies.
    """
    b = bounds or SplitBounds()

    train_df = df.loc[: b.train_end].copy()
    test_df = df.loc[b.test_start : b.test_end].copy()
    val_df = df.loc[b.val_start : b.val_end].copy()

    return train_df, test_df, val_df
