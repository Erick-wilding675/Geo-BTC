# Methodology

This document describes the scientific methodology implemented by the
Geo-BTC pipeline. It is the authoritative reference for the design
choices made in `src/` and `scripts/`.

## Research question

Can large prediction errors of a univariate LSTM forecast of daily
Bitcoin closing prices act as empirical markers of geopolitically
significant events?

The hypothesis is operationalised in two phases:

1. **Quantitative phase (this repository).** Fit an LSTM on 2012-2014
   BTC/USD Bitstamp closing prices, perform out-of-sample inference for
   all 2015 trading days, and flag days whose residuals lie outside a
   ±2σ band.
2. **Qualitative phase.** For each flagged day, expand an asymmetric
   ``[-3, +7]``-day window and investigate news of that period.

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

## Metrics

For every evaluation window we report:

- **MAE** — mean absolute error in USD.
- **RMSE** — root mean squared error in USD.
- **MAE (% of mean price)** — scale-normalised MAE.

See :class:`src.evaluation.metrics.RegressionMetrics`.

## Reproducibility

All stochastic components are seeded via
:func:`src.utils.seed.set_global_seed` (Python `random`, NumPy,
TensorFlow, `PYTHONHASHSEED`). The default seed is `42`.
