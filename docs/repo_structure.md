# Repository Structure Standard

This document defines the branch layout we want to keep consistent.

## Top-Level Rule

Keep each concern in one place:

- diagnosis work in `diagnosis/`
- reusable code in `src/`
- run scripts in `scripts/`
- configs in `configs/`
- contracts in `contracts/`
- generated forecast outputs in `forecasts/`
- generated diagnosis outputs inside the relevant diagnosis package
- local derived inputs in `data/interim/`

## Current Standard Layout

```text
.
├── configs/
├── contracts/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── diagnosis/
│   ├── system_level_analysis/
│   └── station_level_analysis/
├── docs/
├── forecasts/
│   └── system_level/
├── scripts/
│   ├── system_level/
│   └── station_level/
├── src/
│   ├── forecasting_diagnostics/
│   └── metro_bike_share_forecasting/
├── sql/
└── tests/
```

## What Goes Where

### `scripts/system_level/`

Use this for runnable system-level entrypoints only.

- `scripts/system_level/diagnosis/`
- `scripts/system_level/forecasting/`

### `scripts/station_level/`

Use this for runnable station-level entrypoints only.

- `scripts/station_level/diagnosis/`
- `scripts/station_level/forecasting/`

### `diagnosis/system_level_analysis/`

Use this for:

- system-level diagnostic notes
- diagnosis figures, tables, and markdown reports

Do not use this for:

- live Python implementation modules
- production forecast outputs
- forecasting model registry files

### `diagnosis/station_level_analysis/`

Use this for:

- station summary feature generation
- rule-based station categorization
- station clustering for diagnosis
- diagnosis reports and inspection tables

Do not use this for:

- live Python implementation modules
- forecasting models
- station forecast outputs

### `src/metro_bike_share_forecasting/system_level/forecasting/`

Use this for:

- system-level forecasting logic
- feature engineering
- models
- backtesting
- calibrated intervals
- evaluation helpers

### `src/metro_bike_share_forecasting/system_level/diagnosis/`

Use this for:

- system-level diagnosis-facing wrappers
- level-specific diagnosis helpers built on the reusable diagnostics package

### `src/metro_bike_share_forecasting/station_level/diagnosis/`

Use this for:

- reusable station-level diagnosis logic
- summary feature generation
- categorization rules
- clustering helpers
- diagnosis report builders

### `forecasts/system_level/`

Use this for:

- backtest artifacts
- future forecasts
- interval outputs
- model comparison tables
- forecast reports
- model metadata
- forecast figures

### `data/interim/`

Use this for:

- local derived inputs needed by diagnosis or forecasting
- examples: `data/interim/station_level/station_daily.csv`

These files are intentionally not committed by default.

## Naming Rules

- always keep `system_level` and `station_level` explicit in paths
- diagnosis outputs stay under the owning diagnosis folder
- forecasting outputs stay under `forecasts/system_level/`
- avoid loose CSV, JSON, CSS, or helper scripts at the repo root

## Current Branch Convention

Today the branch is intentionally asymmetric:

- system-level: diagnosis + forecasting
- station-level: diagnosis only

That is acceptable as long as the folders make the scope obvious, which is why diagnosis and forecasting are split instead of mixed together.
