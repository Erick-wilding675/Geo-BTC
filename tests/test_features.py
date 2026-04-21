"""Tests for src.features.scaling and src.features.windowing."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import temporal_split
from src.features import create_dataset, reshape_lstm, scale_data


class TestScaleData:
    def test_train_values_fall_in_target_range(self, synthetic_btc_df):
        train, test, val = temporal_split(synthetic_btc_df)
        train_s, _, _, scaler = scale_data(train, test, val, "Close")
        assert train_s.min() >= 0.0 - 1e-9
        assert train_s.max() <= 1.0 + 1e-9
        assert scaler is not None

    def test_invert_roundtrips(self, synthetic_btc_df):
        train, test, val = temporal_split(synthetic_btc_df)
        train_s, _, _, scaler = scale_data(train, test, val, "Close")
        inverted = scaler.inverse_transform(train_s)
        np.testing.assert_allclose(inverted.ravel(), train["Close"].values, rtol=1e-6)


class TestCreateDataset:
    def test_shapes(self):
        data = np.arange(100, dtype=float).reshape(-1, 1)
        X, y = create_dataset(data, look_back=10)
        assert X.shape == (90, 10)
        assert y.shape == (90,)

    def test_alignment(self):
        data = np.arange(20, dtype=float).reshape(-1, 1)
        X, y = create_dataset(data, look_back=3)
        # Window at position 0 must equal [0,1,2] with target 3
        np.testing.assert_array_equal(X[0], [0, 1, 2])
        assert y[0] == 3

    def test_raises_on_bad_look_back(self):
        with pytest.raises(ValueError):
            create_dataset(np.zeros((10, 1)), look_back=0)
        with pytest.raises(ValueError):
            create_dataset(np.zeros((5, 1)), look_back=5)


class TestReshapeLstm:
    def test_output_shape(self):
        X = np.zeros((7, 4))
        reshaped = reshape_lstm(X)
        assert reshaped.shape == (7, 4, 1)
