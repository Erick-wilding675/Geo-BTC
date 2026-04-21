"""Two-sigma outlier detection and investigation-window generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OutlierThresholds:
    """Thresholds used by the ±2σ rule on directional residuals."""

    mean: float
    std: float
    upper: float
    lower: float


def detect_outliers_2sigma(
    analysis_df: pd.DataFrame,
    directional_col: str = "Directional_Error",
) -> tuple[pd.DataFrame, OutlierThresholds]:
    """Flag days whose directional residual lies outside the μ ± 2σ band.

    Adds two columns to a **copy** of ``analysis_df``:

    - ``Is_Outlier`` — boolean flag.
    - ``Outlier_Type`` — ``"Positive Spike"``, ``"Negative Drop"``
      or ``"Normal"``.

    Parameters
    ----------
    analysis_df : pandas.DataFrame
        Output of :func:`src.analysis.residuals.build_residual_dataframe`.
    directional_col : str
        Name of the signed-residual column. Default ``"Directional_Error"``.

    Returns
    -------
    tuple
        ``(flagged_df, thresholds)``.
    """
    df = analysis_df.copy()
    errors = df[directional_col].values
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))

    upper = mean_err + 2.0 * std_err
    lower = mean_err - 2.0 * std_err

    df["Is_Outlier"] = (df[directional_col] > upper) | (df[directional_col] < lower)
    df["Outlier_Type"] = "Normal"
    df.loc[df[directional_col] > upper, "Outlier_Type"] = "Positive Spike"
    df.loc[df[directional_col] < lower, "Outlier_Type"] = "Negative Drop"

    return df, OutlierThresholds(mean=mean_err, std=std_err, upper=upper, lower=lower)


def build_investigation_periods(
    flagged_df: pd.DataFrame,
    pre_days: int = 3,
    post_days: int = 7,
) -> pd.DataFrame:
    """Expand each outlier day into an asymmetric ``[-pre, +post]`` window.

    The default ``[-3, +7]`` accounts for the typical lag between a
    geopolitical event and its observable market effect (see paper
    Section 7.2).

    Parameters
    ----------
    flagged_df : pandas.DataFrame
        Output of :func:`detect_outliers_2sigma` — must contain the columns
        ``Date, Actual_Price, Abs_Error, Directional_Error``.
    pre_days, post_days : int
        Size of the asymmetric investigation window.

    Returns
    -------
    pandas.DataFrame
        Table of investigation windows, one row per detected outlier.
    """
    # Use the ±2σ rule on **absolute** residuals to preserve the paper's
    # export procedure (Section 7.2).
    mean_abs = flagged_df["Abs_Error"].mean()
    std_abs = flagged_df["Abs_Error"].std()
    threshold = mean_abs + 2.0 * std_abs

    outliers = flagged_df[flagged_df["Abs_Error"] > threshold].copy()
    outliers["Type"] = np.where(outliers["Directional_Error"] > 0, "Peak", "Drop")
    outliers["Start_Date"] = outliers["Date"] - pd.Timedelta(days=pre_days)
    outliers["End_Date"] = outliers["Date"] + pd.Timedelta(days=post_days)

    return (
        outliers[
            [
                "Date",
                "Actual_Price",
                "Abs_Error",
                "Type",
                "Start_Date",
                "End_Date",
            ]
        ]
        .rename(
            columns={
                "Date": "Event_Date",
                "Actual_Price": "Actual_Price_USD",
                "Abs_Error": "Error_Value_USD",
            }
        )
        .round({"Actual_Price_USD": 2, "Error_Value_USD": 2})
        .reset_index(drop=True)
    )
