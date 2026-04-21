"""Sliding-window transformation for supervised sequence learning."""

from __future__ import annotations

import numpy as np


def create_dataset(dataset: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 1-D scaled array into (X, y) supervised sequences.

    For each index ``i``, ``X[i]`` is a window of ``look_back`` consecutive
    values and ``y[i]`` is the very next value — the standard "next-step"
    supervised formulation for time-series forecasting.

    Parameters
    ----------
    dataset : numpy.ndarray
        A 2-D array of shape ``(n_timesteps, 1)``.
    look_back : int
        Size of the sliding window in time steps (days).

    Returns
    -------
    tuple of numpy.ndarray
        ``(X, y)`` where ``X`` has shape ``(n_samples, look_back)`` and
        ``y`` has shape ``(n_samples,)``.
    """
    if look_back <= 0:
        raise ValueError(f"`look_back` must be positive, got {look_back}.")
    if len(dataset) <= look_back:
        raise ValueError(
            f"`dataset` length ({len(dataset)}) must be strictly greater "
            f"than `look_back` ({look_back})."
        )

    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i : i + look_back, 0])
        y.append(dataset[i + look_back, 0])

    return np.asarray(X), np.asarray(y)


def reshape_lstm(X: np.ndarray) -> np.ndarray:
    """Reshape ``(samples, timesteps)`` into ``(samples, timesteps, 1)``.

    LSTM layers expect a 3-D tensor of shape
    ``(samples, timesteps, features)``.
    """
    return X.reshape(X.shape[0], X.shape[1], 1)
