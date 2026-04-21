"""Baseline single-layer LSTM (Section 3 of the paper)."""

from __future__ import annotations

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


def build_baseline(input_shape: tuple[int, int]) -> Sequential:
    """Lightweight single-layer LSTM baseline.

    Architecture::

        LSTM(50) → Dropout(0.2) → Dense(1)

    Optimiser: Adam (default LR).  Loss: mean-squared error.

    Parameters
    ----------
    input_shape : tuple of int
        ``(look_back, n_features)`` — typically ``(LOOK_BACK, 1)``.

    Returns
    -------
    tensorflow.keras.models.Sequential
        Compiled, ready-to-fit model.
    """
    model = Sequential(
        [
            LSTM(units=50, return_sequences=False, input_shape=input_shape),
            Dropout(0.2),
            Dense(units=1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
