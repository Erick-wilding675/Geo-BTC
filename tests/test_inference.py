"""Tests for the Phase-2 causal-inference pipeline (src.inference)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.inference import (
    build_inference_table,
    compute_explaining_ratio,
    parse_period_string,
    summarise_by_category,
)


class TestParsePeriodString:
    def test_same_month_dash(self):
        p = parse_period_string("15-25 Mar 2015")
        assert p.start == pd.Timestamp("2015-03-15")
        assert p.end == pd.Timestamp("2015-03-25")

    def test_same_month_endash(self):
        p = parse_period_string("15–25 Mar 2015")
        assert p.start == pd.Timestamp("2015-03-15")
        assert p.end == pd.Timestamp("2015-03-25")

    def test_cross_month_endash(self):
        p = parse_period_string("23 Nov – 03 Dec 2015")
        assert p.start == pd.Timestamp("2015-11-23")
        assert p.end == pd.Timestamp("2015-12-03")

    def test_portuguese_months(self):
        p = parse_period_string("15-25 Ago 2015")
        assert p.start == pd.Timestamp("2015-08-15")
        assert p.end == pd.Timestamp("2015-08-25")

    def test_default_year(self):
        p = parse_period_string("05-15 Dec", default_year=2015)
        assert p.start == pd.Timestamp("2015-12-05")
        assert p.end == pd.Timestamp("2015-12-15")

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            parse_period_string("this is not a date")


@pytest.fixture()
def qualitative_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": "15–25 Mar 2015",
                "Fluctuation": "Down",
                "Category": "Ecosystem Vulnerability",
                "Event (Macro)": "Hot-wallet hacks reignite Mt. Gox trauma.",
                "Breakdown (Micro)": "Coinapult and AllCrypt hacks.",
                "Price Movement": "US$ 286 → US$ 245 (-14%)",
            },
            {
                "Date": "27 Oct – 06 Nov 2015",
                "Fluctuation": "Up",
                "Category": "Chinese Market Demand Surge",
                "Event (Macro)": "Volume on Chinese exchanges explodes.",
                "Breakdown (Micro)": "Huobi + OKCoin = 78% of global volume.",
                "Price Movement": "US$ 280 → US$ 380 (+30 to +35%)",
            },
        ]
    )


@pytest.fixture()
def investigation_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Event_Date": pd.Timestamp("2015-03-19"),
                "Actual_Price_USD": 250.0,
                "Error_Value_USD": 35.5,
                "Type": "Drop",
                "Start_Date": pd.Timestamp("2015-03-16"),
                "End_Date": pd.Timestamp("2015-03-26"),
            },
            {
                "Event_Date": pd.Timestamp("2015-11-02"),
                "Actual_Price_USD": 380.0,
                "Error_Value_USD": 90.0,
                "Type": "Peak",
                "Start_Date": pd.Timestamp("2015-10-30"),
                "End_Date": pd.Timestamp("2015-11-09"),
            },
            {
                "Event_Date": pd.Timestamp("2015-06-15"),
                "Actual_Price_USD": 240.0,
                "Error_Value_USD": 8.0,
                "Type": "Drop",
                "Start_Date": pd.Timestamp("2015-06-12"),
                "End_Date": pd.Timestamp("2015-06-22"),
            },
        ]
    )


class TestBuildInferenceTable:
    def test_columns_present(self, qualitative_df, investigation_df):
        out = build_inference_table(investigation_df, qualitative_df)
        for col in (
            "Event_Date",
            "Real_Price",
            "Absolute_Error",
            "Category",
            "Event_Macro",
        ):
            assert col in out.columns

    def test_outlier_inside_window_is_matched(self, qualitative_df, investigation_df):
        out = build_inference_table(investigation_df, qualitative_df)
        mar = out[out["Event_Date"] == pd.Timestamp("2015-03-19")].iloc[0]
        assert mar["Category"] == "Ecosystem Vulnerability"

    def test_outlier_outside_any_window_has_null_category(
        self, qualitative_df, investigation_df
    ):
        out = build_inference_table(investigation_df, qualitative_df)
        june = out[out["Event_Date"] == pd.Timestamp("2015-06-15")].iloc[0]
        assert pd.isna(june["Category"])


class TestExplainingRatio:
    def test_basic_ratio(self, qualitative_df, investigation_df):
        merged = build_inference_table(investigation_df, qualitative_df)
        er = compute_explaining_ratio(merged, threshold_usd=20.0)
        # Two outliers above $20: Mar (matched) + Nov (matched). June (8.0) excluded.
        assert er.total_outliers == 2
        assert er.explained_outliers == 2
        assert er.ratio_pct == 100.0

    def test_summary_table_structure(self, qualitative_df, investigation_df):
        merged = build_inference_table(investigation_df, qualitative_df)
        summary = summarise_by_category(merged, threshold_usd=20.0)
        assert set(summary.columns) == {
            "Event Category",
            "Frequence",
            "Mean Absolute Error (MAE)",
            "Explaining Ratio (ER)",
        }
        assert summary["Frequence"].sum() == 2
