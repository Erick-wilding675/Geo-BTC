"""Final LSTM trained on merged 2012-2014 data (Section 6 of the paper)."""

from __future__ import annotations

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


def build_final_model(input_shape: tuple[int, int], units: int = 100) -> Sequential:
    """Champion LSTM architecture selected from the grid search.

    The default ``(LOOK_BACK=90, units=100)`` corresponds to the best
    configuration reported in the paper. Both values can be overridden here
    and in ``configs/config.yaml``.

    Architecture::

        LSTM(units) → Dropout(0.2) → Dense(1)

    Parameters
    ----------
    input_shape : tuple of int
        ``(look_back, n_features)``.
    units : int, optional
        Number of LSTM units. Default ``100``.
    """
    model = Sequential(
        [
            LSTM(units=units, return_sequences=False, input_shape=input_shape),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
