# `data/`

This directory holds every dataset consumed or produced by the
quantitative pipeline.

| Subfolder | Contents | Git LFS? |
| --------- | -------- | -------- |
| `raw/` | Immutable, as-received data snapshots (`bitstamp_cleaned.csv`) | yes |
| `processed/` | Cleaned, analysis-ready CSVs (`bitstamp_daily_final.csv`) | no |
| `external/` | Auxiliary signals from third-party sources (`wikipedia_edits.csv`) | yes |

See [`docs/DATA.md`](../docs/DATA.md) for full schemas, provenance and
regeneration recipes.
