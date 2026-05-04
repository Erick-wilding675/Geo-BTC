"""Phase 2 — Explaining Ratio (ER) and per-category summary statistics.

Definitions
-----------
Let ``T = {t : Abs_Error_t > theta}`` be the set of LSTM outliers
identified by the ``theta`` rule (Rule of 20: ``theta = 20 USD``,
or any user-supplied threshold). Let ``E ⊆ T`` be the subset of
outliers that the qualitative phase **explained** — i.e. for which the
inference table has a non-null ``Category``. The Explaining Ratio is

    ER = |E| / |T| × 100

The category-level summary refines this to:

    ER_c = |E_c| / |T| × 100   (share of total outliers explained by category c)
    MAE_c = mean(Abs_Error of explained outliers in category c)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ── "Rule of 20" — the default error threshold (in USD) used in the paper ─────
DEFAULT_ERROR_THRESHOLD: float = 20.0


@dataclass(frozen=True)
class ExplainingRatio:
    """Top-level Explaining Ratio result."""

    explained_outliers: int
    total_outliers: int
    threshold_usd: float

    @property
    def ratio_pct(self) -> float:
        if self.total_outliers == 0:
            return 0.0
        return self.explained_outliers / self.total_outliers * 100.0


def compute_explaining_ratio(
    inference_df: pd.DataFrame,
    threshold_usd: float = DEFAULT_ERROR_THRESHOLD,
) -> ExplainingRatio:
    """Compute the global Explaining Ratio over the inference table.

    Parameters
    ----------
    inference_df : pandas.DataFrame
        Output of :func:`src.inference.merge.build_inference_table`.
    threshold_usd : float
        Absolute-error threshold (USD). Default: 20.0 (Rule of 20).
    """
    above = inference_df[inference_df["Absolute_Error"] > threshold_usd]
    explained = above[above["Category"].notna()]
    return ExplainingRatio(
        explained_outliers=int(len(explained)),
        total_outliers=int(len(above)),
        threshold_usd=float(threshold_usd),
    )


def summarise_by_category(
    inference_df: pd.DataFrame,
    threshold_usd: float = DEFAULT_ERROR_THRESHOLD,
) -> pd.DataFrame:
    """Per-category summary table for the paper.

    Returns a DataFrame with the columns:

        Event Category, Frequence, Mean Absolute Error (MAE), Explaining Ratio (ER)

    The ER column is the **share of the total identified outliers**
    explained by that category — so summing the column reproduces the
    overall Explaining Ratio.

    Parameters
    ----------
    inference_df : pandas.DataFrame
        Output of :func:`src.inference.merge.build_inference_table`.
    threshold_usd : float
        Absolute-error threshold (USD). Default: 20.0.
    """
    above = inference_df[inference_df["Absolute_Error"] > threshold_usd].copy()
    total = len(above)
    if total == 0:
        return pd.DataFrame(
            columns=[
                "Event Category",
                "Frequence",
                "Mean Absolute Error (MAE)",
                "Explaining Ratio (ER)",
            ]
        )

    grouped = (
        above.dropna(subset=["Category"])
        .groupby("Category", as_index=False)
        .agg(
            Frequence=("Absolute_Error", "size"),
            MAE=("Absolute_Error", "mean"),
        )
    )
    grouped["Explaining Ratio (ER)"] = grouped["Frequence"] / total * 100.0
    grouped = grouped.rename(
        columns={
            "Category": "Event Category",
            "MAE": "Mean Absolute Error (MAE)",
        }
    )
    grouped["Mean Absolute Error (MAE)"] = grouped["Mean Absolute Error (MAE)"].round(2)
    grouped["Explaining Ratio (ER)"] = grouped["Explaining Ratio (ER)"].round(2)
    return grouped.sort_values("Frequence", ascending=False).reset_index(drop=True)
