"""Phase 2 — causal inference: merge LSTM residuals with qualitative events."""

from src.inference.merge import (
    build_inference_table,
    parse_period_string,
)
from src.inference.metrics import (
    compute_explaining_ratio,
    summarise_by_category,
)

__all__ = [
    "parse_period_string",
    "build_inference_table",
    "compute_explaining_ratio",
    "summarise_by_category",
]
