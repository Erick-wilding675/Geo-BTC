"""Matplotlib figures used throughout the paper."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _maybe_save(fig: plt.Figure, output: str | Path | None) -> None:
    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")


def plot_learning_curve(
    history: Mapping[str, list[float]],
    title: str,
    output: str | Path | None = None,
) -> plt.Figure:
    """Plot train vs validation loss across epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["loss"], label="Train Loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Validation Loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _maybe_save(fig, output)
    return fig


def plot_forecast_vs_actual(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str,
    x_label: str = "Days (test window)",
    actual_color: str = "black",
    pred_color: str = "#00CC96",
    output: str | Path | None = None,
) -> plt.Figure:
    """Overlay predicted vs actual BTC prices on a single axis."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual, color=actual_color, linewidth=1.5, label="Actual Price")
    ax.plot(predicted, color=pred_color, linewidth=1.5, linestyle="--", label="Predicted Price")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _maybe_save(fig, output)
    return fig


def plot_residual_panels(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    abs_error: np.ndarray,
    mae: float,
    output: str | Path | None = None,
) -> plt.Figure:
    """Two-panel figure: forecast overlay + residual bars (Fig. 6 of the paper)."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    axes[0].plot(dates, actual, color="black", linewidth=1.5, label="Actual (BTC/USD)")
    axes[0].plot(
        dates,
        predicted,
        color="#00CC96",
        linewidth=1.5,
        linestyle="--",
        label="Predicted (LSTM)",
    )
    axes[0].set_title(f"Bitcoin Price Forecast — 2015  (MAE: ${mae:.2f})", fontsize=14)
    axes[0].set_ylabel("Price (USD)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))

    axes[1].bar(
        dates,
        abs_error,
        color="#EF553B",
        alpha=0.7,
        label="Prediction Error (Residual)",
    )
    axes[1].set_title(
        "Residual Magnitude — Candidates for External Event Investigation",
        fontsize=13,
    )
    axes[1].set_ylabel("Absolute Error (USD)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))

    fig.tight_layout()
    _maybe_save(fig, output)
    return fig
