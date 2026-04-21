"""Regression evaluation metrics in the original (un-scaled) price domain."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


@dataclass
class RegressionMetrics:
    """Aggregate regression metrics."""

    mae: float
    rmse: float
    mean_price: float

    @property
    def mae_pct(self) -> float:
        """MAE as a percentage of the mean actual price."""
        return self.mae / self.mean_price * 100.0

    @property
    def rmse_pct(self) -> float:
        """RMSE as a percentage of the mean actual price."""
        return self.rmse / self.mean_price * 100.0


def invert_predictions(
    predictions: np.ndarray,
    y_true: np.ndarray,
    scaler: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert MinMax scaling on model predictions and ground truth.

    Parameters
    ----------
    predictions : numpy.ndarray
        Scaled model output of shape ``(n, 1)`` or ``(n,)``.
    y_true : numpy.ndarray
        Scaled ground-truth labels of shape ``(n,)`` or ``(n, 1)``.
    scaler : MinMaxScaler
        Fitted scaler produced by :func:`src.features.scaling.scale_data`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(predictions_real, y_true_real)`` both of shape ``(n, 1)``.
    """
    preds_real = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_real = scaler.inverse_transform(y_true.reshape(-1, 1))
    return preds_real, y_real


def compute_regression_metrics(
    y_true_real: np.ndarray,
    y_pred_real: np.ndarray,
) -> RegressionMetrics:
    """Compute MAE, RMSE and the mean of ``y_true_real``."""
    mae = mean_absolute_error(y_true_real, y_pred_real)
    rmse = math.sqrt(mean_squared_error(y_true_real, y_pred_real))
    mean_price = float(np.mean(y_true_real))
    return RegressionMetrics(mae=mae, rmse=rmse, mean_price=mean_price)
