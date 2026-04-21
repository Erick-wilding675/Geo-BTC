"""Tests for src.analysis.residuals and src.analysis.outliers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    build_investigation_periods,
    build_residual_dataframe,
    detect_outliers_2sigma,
)


@pytest.fixture()
def residual_df() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=20, freq="D")
    y_actual = np.linspace(100, 200, 20)
    # Make day 10 a huge outlier — roughly +1000 USD swing.
    y_predicted = y_actual + 0.5
    y_predicted[10] -= 1000
    return build_residual_dataframe(dates, y_actual, y_predicted)


class TestBuildResidualDataframe:
    def test_columns_and_positivity(self, residual_df):
        expected = {
            "Date",
            "Actual_Price",
            "Predicted_Price",
            "Abs_Error",
            "Pct_Error",
            "Directional_Error",
        }
        assert expected.issubset(residual_df.columns)
        assert (residual_df["Abs_Error"] >= 0).all()

    def test_length_mismatch_raises(self):
        dates = pd.date_range("2015-01-01", periods=5, freq="D")
        with pytest.raises(ValueError):
            build_residual_dataframe(dates, np.zeros(5), np.zeros(4))


class TestDetectOutliers2Sigma:
    def test_flags_the_constructed_outlier(self, residual_df):
        flagged, thr = detect_outliers_2sigma(residual_df)
        assert flagged["Is_Outlier"].sum() >= 1
        assert flagged.loc[10, "Is_Outlier"]
        assert thr.upper > thr.lower


class TestBuildInvestigationPeriods:
    def test_window_bounds_are_asymmetric(self, residual_df):
        flagged, _ = detect_outliers_2sigma(residual_df)
        periods = build_investigation_periods(flagged, pre_days=3, post_days=7)
        if len(periods) > 0:
            row = periods.iloc[0]
            assert (row["Event_Date"] - row["Start_Date"]).days == 3
            assert (row["End_Date"] - row["Event_Date"]).days == 7
