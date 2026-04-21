"""Build the residual dataframe used by the outlier detection pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_residual_dataframe(
    dates: pd.DatetimeIndex,
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
) -> pd.DataFrame:
    """Assemble the per-day residual dataframe for 2015 inference.

    The output has one row per inference day and contains:

    - ``Date`` — calendar day of the target price.
    - ``Actual_Price`` — ground-truth BTC/USD close.
    - ``Predicted_Price`` — model output after inverting the scaler.
    - ``Abs_Error`` — ``|actual − predicted|`` (always non-negative).
    - ``Pct_Error`` — absolute error as a percentage of the actual price.
    - ``Directional_Error`` — signed residual ``actual − predicted``
      (positive → actual beat prediction; negative → model over-shot).

    Parameters
    ----------
    dates : pandas.DatetimeIndex
        Dates aligned with ``y_actual`` and ``y_predicted`` (typically
        ``val_df.index[LOOK_BACK:]``).
    y_actual, y_predicted : numpy.ndarray
        Ground-truth and predicted BTC/USD prices in the original USD domain,
        i.e. after inverting the MinMax scaling.
    """
    y_actual_flat = np.asarray(y_actual).flatten()
    y_pred_flat = np.asarray(y_predicted).flatten()

    if len(dates) != len(y_actual_flat) or len(dates) != len(y_pred_flat):
        raise ValueError(
            "Length mismatch: dates={}, y_actual={}, y_predicted={}.".format(
                len(dates), len(y_actual_flat), len(y_pred_flat)
            )
        )

    abs_error = np.abs(y_actual_flat - y_pred_flat)
    pct_error = abs_error / y_actual_flat * 100.0
    directional_error = y_actual_flat - y_pred_flat

    return pd.DataFrame(
        {
            "Date": dates,
            "Actual_Price": y_actual_flat,
            "Predicted_Price": y_pred_flat,
            "Abs_Error": abs_error,
            "Pct_Error": pct_error,
            "Directional_Error": directional_error,
        }
    )
