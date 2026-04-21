# Data

## Data sources

| File | Location | Origin |
| ---- | -------- | ------ |
| `bitstamp_cleaned.csv` | `data/raw/` | Bitstamp exchange tick data, cleaned (duplicate rows removed, DateTime coerced) |
| `bitstamp_daily_final.csv` | `data/processed/` | Resampled to **daily** OHLCV with `DateTime` index |
| `wikipedia_edits.csv` | `data/external/` | Daily Wikipedia edit counts on the `Bitcoin` article, with sentiment and negative-sentiment scores — used as a contextual signal in Phase 2 |

The raw Bitstamp file is an intermediate snapshot derived from
the publicly available Kaggle dataset
"Bitcoin Historical Data". The original minute-level file is not
redistributed; the cleaned CSV is kept under Git LFS.

## Schemas

### `data/raw/bitstamp_cleaned.csv`

| Column | Type | Description |
| ------ | ---- | ----------- |
| `Unix_Timestamp` | int | Minute-bar timestamp (seconds since epoch) |
| `DateTime` | string | ISO-style timestamp (`MM/DD/YYYY HH:MM:SS`) |
| `Open`, `High`, `Low`, `Close` | float | OHLC in USD |
| `Volume_(BTC)` | float | Traded BTC volume |
| `Volume_(Currency)` | float | Traded USD volume |
| `Weighted_Price` | float | VWAP for the bar |

### `data/processed/bitstamp_daily_final.csv`

| Column | Type | Description |
| ------ | ---- | ----------- |
| `DateTime` | date | Calendar day |
| `Open`, `High`, `Low`, `Close` | float | Daily OHLC in USD |
| `Volume_(BTC)` | float | Daily BTC volume |

### `data/external/wikipedia_edits.csv`

| Column | Type | Description |
| ------ | ---- | ----------- |
| *(index)* | date | Calendar day |
| `edit_count` | float | Number of edits to the Bitcoin article |
| `sentiment` | float | Mean sentiment polarity of edits |
| `neg_sentiment` | float | Negative-sentiment share |

## Regenerating the processed CSV

The daily CSV was produced once, offline, by resampling the Bitstamp
tick data to a daily OHLCV panel. The process is intentionally kept
out of the automated pipeline because the raw file is bulky (> 100 MB
at minute resolution) and is tracked with Git LFS.

A minimal recipe that reproduces the processed file from the raw one:

```python
import pandas as pd

raw = pd.read_csv("data/raw/bitstamp_cleaned.csv")
raw["DateTime"] = pd.to_datetime(raw["DateTime"])
raw = raw.set_index("DateTime").sort_index()

daily = raw.resample("1D").agg({
    "Open":        "first",
    "High":        "max",
    "Low":         "min",
    "Close":       "last",
    "Volume_(BTC)":"sum",
})
daily = daily.dropna().reset_index()
daily.to_csv("data/processed/bitstamp_daily_final.csv", index=False)
```

## Licence

The Bitstamp tick data is released under the CC0 public-domain
dedication by its original Kaggle publisher.
