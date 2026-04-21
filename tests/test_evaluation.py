"""Tests for src.evaluation.metrics."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.evaluation import compute_regression_metrics, invert_predictions


class TestComputeRegressionMetrics:
    def test_perfect_predictions_give_zero_error(self):
        y = np.array([[100.0], [200.0], [300.0]])
        m = compute_regression_metrics(y, y.copy())
        assert m.mae == 0.0
        assert m.rmse == 0.0

    def test_mean_price_reported(self):
        y = np.array([[100.0], [200.0], [300.0]])
        m = compute_regression_metrics(y, y.copy())
        assert m.mean_price == 200.0
        assert m.mae_pct == 0.0

    def test_known_error_values(self):
        y_true = np.array([[10.0], [20.0], [30.0]])
        y_pred = np.array([[11.0], [22.0], [28.0]])
        m = compute_regression_metrics(y_true, y_pred)
        # absolute errors: 1, 2, 2 → MAE = 5/3
        np.testing.assert_allclose(m.mae, 5.0 / 3.0, rtol=1e-6)
        # squared errors: 1, 4, 4 → MSE = 3 → RMSE = sqrt(3)
        np.testing.assert_allclose(m.rmse, np.sqrt(3.0), rtol=1e-6)


class TestInvertPredictions:
    def test_roundtrip(self):
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        scaler = MinMaxScaler().fit(x)
        scaled = scaler.transform(x)
        preds_real, y_real = invert_predictions(scaled.copy(), scaled.copy(), scaler)
        np.testing.assert_allclose(preds_real, x, rtol=1e-6)
        np.testing.assert_allclose(y_real, x, rtol=1e-6)
