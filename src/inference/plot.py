"""Phase 2 — Plotly inference storyboard (Real vs Predicted with event flags)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.inference.metrics import DEFAULT_ERROR_THRESHOLD


def _truncate(text: str | None, max_len: int = 60) -> str:
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def build_inference_figure(
    residuals_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    threshold_usd: float = DEFAULT_ERROR_THRESHOLD,
    title: str = "Geo-BTC — 2015 Causal-Inference Storyboard",
) -> go.Figure:
    """Return a Plotly figure with Real vs Predicted price + event flags.

    Parameters
    ----------
    residuals_df : pandas.DataFrame
        Output of ``scripts/run_residual_analysis.py``
        (``analysis_residuals_2015.csv``). Required columns:
        ``Date, Actual_Price, Predicted_Price, Abs_Error``.
    inference_df : pandas.DataFrame
        Output of :func:`src.inference.merge.build_inference_table`.
        Each row whose ``Absolute_Error > threshold_usd`` becomes a
        labelled flag annotation.
    threshold_usd : float
        Absolute-error threshold (USD) for flagging events. Default 20.
    title : str
        Figure title.
    """
    res = residuals_df.copy()
    res["Date"] = pd.to_datetime(res["Date"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=res["Date"],
            y=res["Actual_Price"],
            mode="lines",
            name="Real Price (BTC/USD)",
            line=dict(color="black", width=1.6),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Real: $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=res["Date"],
            y=res["Predicted_Price"],
            mode="lines",
            name="Predicted Price (LSTM)",
            line=dict(color="#00CC96", width=1.4, dash="dash"),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Predicted: $%{y:.2f}<extra></extra>",
        )
    )

    flagged = inference_df[inference_df["Absolute_Error"] > threshold_usd].copy()
    flagged["Event_Date"] = pd.to_datetime(flagged["Event_Date"])

    if not flagged.empty:
        flag_x = flagged["Event_Date"]
        flag_y = flagged["Real_Price"]
        flag_text = flagged.apply(
            lambda r: (
                f"<b>{_truncate(r['Event_Macro'], 90)}</b><br>"
                f"Category: {r.get('Category') or '—'}<br>"
                f"Abs. error: ${r['Absolute_Error']:.2f}"
            ),
            axis=1,
        )

        fig.add_trace(
            go.Scatter(
                x=flag_x,
                y=flag_y,
                mode="markers",
                name=f"Outlier flag (Error > ${threshold_usd:.0f})",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="#EF553B",
                    line=dict(width=1, color="black"),
                ),
                text=flag_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

        # Stagger annotations vertically so they don't overlap.
        for i, row in enumerate(flagged.itertuples(index=False)):
            ay = -40 - (i % 4) * 18
            fig.add_annotation(
                x=row.Event_Date,
                y=row.Real_Price,
                text=_truncate(row.Event_Macro, 38),
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=ay,
                font=dict(size=10, color="#1F4E79"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#EF553B",
                borderwidth=1,
                borderpad=2,
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date (2015)",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=1.07, x=0.5, xanchor="center"),
        margin=dict(l=60, r=40, t=80, b=60),
    )
    return fig


def export_figure(fig: go.Figure, output_path: str | Path) -> None:
    """Write the figure to disk as an interactive HTML file (and PNG if kaleido)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    try:
        png = out.with_suffix(".png")
        fig.write_image(str(png), width=1600, height=720, scale=2)
    except Exception:
        # kaleido is optional; skip silently when unavailable.
        pass
