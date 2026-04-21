"""Hyper-parameter grid search (Section 5 of the paper)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from src.features.windowing import create_dataset


@dataclass
class GridResult:
    """One row of the grid-search report."""

    config: str
    look_back: int
    neurons: int
    mae: float


def grid_search_lstm(
    train_scaled: np.ndarray,
    test_scaled: np.ndarray,
    scaler: MinMaxScaler,
    look_back_options: Iterable[int] = (60, 90),
    neurons_options: Iterable[int] = (50, 100),
    epochs: int = 20,
    batch_size: int = 32,
    patience: int = 5,
    verbose: int = 0,
) -> pd.DataFrame:
    """Exhaustive grid search over ``look_back`` × ``neurons``.

    For each combination a single-layer LSTM is trained on the merged
    train+test block with early stopping (``patience``), and MAE is
    computed on the test slice after inverting the MinMax scaling.

    Parameters
    ----------
    train_scaled, test_scaled : numpy.ndarray
        Scaled target arrays produced by :func:`src.features.scaling.scale_data`.
    scaler : MinMaxScaler
        Fitted scaler used to invert predictions for MAE computation.
    look_back_options, neurons_options : iterable of int
        Hyper-parameter grids.
    epochs, batch_size, patience : int
        Training configuration.
    verbose : int
        Keras verbosity.

    Returns
    -------
    pandas.DataFrame
        Results sorted ascending by MAE.
    """
    scaled_data = np.concatenate((train_scaled, test_scaled), axis=0)
    results: list[GridResult] = []

    for lb in look_back_options:
        for neurons in neurons_options:
            X_all, y_all = create_dataset(scaled_data, lb)
            X_all = X_all.reshape(X_all.shape[0], X_all.shape[1], 1)

            split_idx = max(len(train_scaled) - lb, 0)
            X_tr, y_tr = X_all[:split_idx], y_all[:split_idx]
            X_te, y_te = X_all[split_idx:], y_all[split_idx:]

            model = Sequential(
                [
                    LSTM(units=neurons, return_sequences=False, input_shape=(lb, 1)),
                    Dropout(0.2),
                    Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mean_squared_error")

            es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
            model.fit(
                X_tr,
                y_tr,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_te, y_te),
                callbacks=[es],
                verbose=verbose,
            )

            preds = scaler.inverse_transform(model.predict(X_te, verbose=0))
            y_te_real = scaler.inverse_transform(y_te.reshape(-1, 1))
            mae = mean_absolute_error(y_te_real, preds)
            assert not math.isnan(mae)

            results.append(
                GridResult(config=f"Win{lb}_Neur{neurons}", look_back=lb, neurons=neurons, mae=mae)
            )

    df = (
        pd.DataFrame([r.__dict__ for r in results])
        .sort_values(by="mae")
        .reset_index(drop=True)
    )
    return df
