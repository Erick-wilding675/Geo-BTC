"""Fine-tuned stacked LSTM (Section 4 of the paper)."""

from __future__ import annotations

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_tuned_model(
    input_shape: tuple[int, int],
    learning_rate: float = 1e-3,
) -> Sequential:
    """Stacked two-layer LSTM with intermediate Dense refinement.

    Architecture::

        LSTM(128, return_seq=True) → Dropout(0.2)
        → LSTM(64)                 → Dropout(0.2)
        → Dense(25)                → Dense(1)

    Optimiser: Adam with explicit ``learning_rate``.  Loss: MSE.

    Parameters
    ----------
    input_shape : tuple of int
        ``(look_back, n_features)``.
    learning_rate : float, optional
        Adam learning rate. Default ``1e-3``.
    """
    model = Sequential(
        [
            LSTM(units=128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=64, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error")
    return model
