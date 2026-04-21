"""Min-max scaling fitted exclusively on the training partition."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale a single target column to ``feature_range`` via MinMax.

    The scaler is fitted **only** on the training partition to avoid
    information leakage from future data into the scaling parameters.

    Parameters
    ----------
    train_df, test_df, val_df : pandas.DataFrame
        The three temporal partitions produced by
        :func:`src.data.split.temporal_split`.
    target_col : str
        Name of the column to scale (typically ``"Close"``).
    feature_range : tuple of float, optional
        Target range for the MinMax transform. Default ``(0.0, 1.0)``.

    Returns
    -------
    tuple
        ``(train_scaled, test_scaled, val_scaled, scaler)`` — scaled arrays
        plus the fitted :class:`~sklearn.preprocessing.MinMaxScaler` instance
        (used later to invert predictions back to USD).
    """
    scaler = MinMaxScaler(feature_range=feature_range)

    train_scaled = scaler.fit_transform(train_df[[target_col]])
    test_scaled = scaler.transform(test_df[[target_col]])
    val_scaled = scaler.transform(val_df[[target_col]])

    return train_scaled, test_scaled, val_scaled, scaler
