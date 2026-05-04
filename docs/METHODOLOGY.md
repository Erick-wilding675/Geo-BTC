# Methodology

This document describes the scientific methodology implemented by the
Geo-BTC pipeline. It is the authoritative reference for the design
choices made in `src/` and `scripts/`.

## Research question

Can large prediction errors of a univariate LSTM forecast of daily
Bitcoin closing prices act as empirical markers of geopolitically
significant events?

The hypothesis is operationalised in two phases:

1. **Quantitative phase.** Fit an LSTM on 2012-2014 BTC/USD Bitstamp
   closing prices, perform out-of-sample inference for all 2015 trading
   days, and flag days whose residuals lie outside a ±2σ band.
2. **Qualitative + causal-inference phase.** For each flagged day,
   expand an asymmetric ``[-3, +7]``-day window, merge with a curated
   qualitative event database (`data/external/qualitative_analysis.csv`)
   and compute the **Explaining Ratio (ER)** — the share of LSTM
   outliers explained by a contemporaneous geopolitical or
   institutional event.

## Temporal partitioning

A strictly chronological split prevents data leakage from future into
past:

| Subset | Period | Purpose |
| ------ | ------ | ------- |
| Train | …  → 2014-09-30 | Learn LSTM parameters |
| Test | 2014-10-01 → 2014-12-31 | Monitor overfit during fit |
| Validation | 2015-01-01 → 2015-12-31 | Out-of-sample inference |

The 2012-2015 window intentionally covers Bitcoin's pre-maturity phase
and coincides with several geopolitical events under investigation.

## Feature engineering

- **Target.** `Close` price (univariate).
- **Scaling.** `MinMaxScaler` fitted exclusively on the training
  partition, then applied to test/validation. This is the main
  anti-leakage safeguard.
- **Sliding window.** `LOOK_BACK = 90` days (champion from the grid
  search). Each training sample is a window of 90 consecutive closing
  prices; the label is the 91st day.

## Model architectures

| Model | Section | Architecture |
| ----- | ------- | ------------ |
| Baseline | 3 | `LSTM(50) → Dropout(0.2) → Dense(1)` |
| Tuned | 4 | `LSTM(128, ret=True) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(25) → Dense(1)` |
| Final | 6 | `LSTM(100) → Dropout(0.2) → Dense(1)`  (merged 2012-2014) |

All models use mean-squared error loss; baseline and final use the
default `Adam`, and the tuned model uses `Adam(lr=1e-3)`.

## Grid search

A light grid (`look_back ∈ {60, 90}`, `neurons ∈ {50, 100}`) with
early stopping (`patience=5`) is evaluated on the test slice. MAE in
the USD domain is the selection criterion.

## Outlier detection

Let ``e_t = Actual_t − Predicted_t`` be the directional residual for
day ``t ∈ 2015``. Define μ and σ as the sample mean and standard
deviation of ``{e_t}``. Day ``t`` is flagged as an **outlier** iff

    e_t > μ + 2σ  (Positive Spike)
    e_t < μ − 2σ  (Negative Drop)

A second ±2σ rule on **absolute** residuals is applied to export the
`investigation_periods.csv` table (for Phase 2).

## Investigation window

Each outlier day ``t`` is expanded into an asymmetric window
``[t − 3, t + 7]``. The longer post-event tail accounts for the typical
lag between a geopolitical event and its observable market effect.

## Phase 2 — causal inference

Phase 2 lives under [`src/inference/`](../src/inference/) and is
documented in detail in [`reports/TR-04.md`](../reports/TR-04.md).
The three building blocks are:

1. **Period parser** (`src/inference/merge.parse_period_string`) — a
   bilingual (PT-BR / EN) parser that converts free-form qualitative
   strings such as `"15-25 Mar 2015"` or `"23 Nov - 03 Dec 2015"` into
   closed `(start, end)` `pandas.Timestamp` intervals.
2. **Containment-based merge** (`src/inference/merge.build_inference_table`)
   — each LSTM outlier `Event_Date` is joined to the qualitative
   window that contains it; ties broken by minimum distance to the
   window midpoint. Outliers without a matching window keep
   `Category = NaN` and contribute to the un-explained residual share.
3. **Explaining Ratio** (`src/inference/metrics`):

   ER = |E| / |T| × 100

   where `T` is the set of outliers above the **Rule of 20**
   (`Abs_Error > 20 USD` by default) and `E ⊆ T` the subset whose
   `Category` is non-null. The same module emits a per-category
   summary with `Frequence`, `Mean Absolute Error (MAE)` and
   `Explaining Ratio (ER)` columns; summing the ER column reproduces
   the global ratio.

The Phase-2 storyboard is rendered with Plotly
(`src/inference/plot`) as an interactive HTML file. A static PNG
companion is generated automatically when the optional `kaleido`
extra is installed.

## Metrics

For every evaluation window we report:

- **MAE** — mean absolute error in USD.
- **RMSE** — root mean squared error in USD.
- **MAE (% of mean price)** — scale-normalised MAE.
- **ER** — Explaining Ratio (Phase 2).

See :class:`src.evaluation.metrics.RegressionMetrics` and
:class:`src.inference.metrics.ExplainingRatio`.

## Reproducibility

All stochastic components are seeded via
:func:`src.utils.seed.set_global_seed` (Python `random`, NumPy,
TensorFlow, `PYTHONHASHSEED`). The default seed is `42`.
