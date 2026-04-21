"""Residual analysis and outlier detection for the 2015 inference window."""

from src.analysis.outliers import (
    build_investigation_periods,
    detect_outliers_2sigma,
)
from src.analysis.residuals import build_residual_dataframe

__all__ = [
    "build_residual_dataframe",
    "detect_outliers_2sigma",
    "build_investigation_periods",
]
