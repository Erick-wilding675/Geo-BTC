"""Tests for src.data.load_data and src.data.split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import load_bitcoin_data, temporal_split
from src.data.split import SplitBounds


class TestLoadBitcoinData:
    def test_returns_dataframe_indexed_by_datetime(self, synthetic_csv):
        df = load_bitcoin_data(synthetic_csv)
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "DateTime"
        assert df.index.is_monotonic_increasing

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_bitcoin_data(tmp_path / "nope.csv")

    def test_raises_without_datetime_column(self, tmp_path: Path):
        bad = tmp_path / "bad.csv"
        bad.write_text("Open,Close\n1,2\n")
        with pytest.raises(KeyError):
            load_bitcoin_data(bad)


class TestTemporalSplit:
    def test_partition_sizes_are_consistent(self, synthetic_btc_df):
        train, test, val = temporal_split(synthetic_btc_df)
        total = len(train) + len(test) + len(val)
        # The last day of 2011-12-31 is not in any partition (train starts 2012-01-01);
        # here our synthetic dataset starts 2012-01-01 so we expect full coverage.
        assert total == len(synthetic_btc_df)

    def test_bounds_are_strictly_monotonic(self, synthetic_btc_df):
        train, test, val = temporal_split(synthetic_btc_df)
        assert train.index.max() < test.index.min()
        assert test.index.max() < val.index.min()

    def test_custom_bounds_respected(self, synthetic_btc_df):
        custom = SplitBounds(
            train_end="2013-12-31",
            test_start="2014-01-01",
            test_end="2014-06-30",
            val_start="2014-07-01",
            val_end="2014-12-31",
        )
        train, test, val = temporal_split(synthetic_btc_df, bounds=custom)
        assert train.index.max() == pd.Timestamp("2013-12-31")
        assert test.index.min() == pd.Timestamp("2014-01-01")
        assert val.index.max() == pd.Timestamp("2014-12-31")
