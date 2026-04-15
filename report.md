# Metro Bike Share Adaptive Forecasting System

Production-style Python forecasting project for Metro Bike Share trip demand. The system reads raw CSV trip files, reuses cleaning rules from the legacy SQL project in this repository, standardizes and validates the data, persists curated datasets to PostgreSQL, builds multi-granularity time series, runs pandemic-aware diagnostics, backtests multiple models, selects champions, and stores probabilistic forecasts with monitoring outputs.

## What this project does

The first working version implements:

- CSV ingestion across multiple quarterly files
- reuse of legacy SQL cleaning logic from [sql/legacy/foundation/metro_bike_share.sql](/Users/morteza/Metro-Bike-Share/sql/legacy/foundation/metro_bike_share.sql) and [sql/legacy/staging](/Users/morteza/Metro-Bike-Share/sql/legacy/staging)
- canonical cleaned trip data with source lineage and data quality logging
- aggregation to hourly, daily, weekly, monthly, and quarterly demand series
- grouped aggregates by start station, plus a system-total training series
- time-series diagnostics with seasonality and structural-break analysis
- pandemic-aware regime features and changepoint-assisted break detection
- strict train / validation / final test splits
- rolling-origin backtesting inside the development window
- champion-challenger model selection on validation only
- final held-out test evaluation after promotion
- probabilistic forecasts with 50%, 80%, and 95% intervals
- station-level direct modeling plus coherent share-allocation / reconciliation for the wider station set
- PostgreSQL persistence for curated data, model metadata, forecasts, and monitoring outputs
- a unified Streamlit studio for running the pipeline and reviewing outputs

## Model stack in the first solid version

- `seasonal_naive`
- `count_glm`
  Uses a count-aware GLM and automatically chooses Poisson or Negative Binomial behavior based on dispersion.
- `sarimax_fourier`
  Uses SARIMAX with Fourier terms and regime-aware exogenous signals.
- `weighted_ensemble`
  Combines top challengers using inverse backtest error weights.

This order is intentional. It gives a strong first production baseline before later additions such as TBATS, gradient boosting, deep probabilistic models, and fleet optimization.

## How legacy cleaning logic is reused

This project does not ignore the base repository work.

The Python cleaner explicitly reuses the intent of these SQL rules:

- `check_id`
- `Check_date`
- `check_duration`
- `check_station`
- `check_b_ids`
- `check_lat_lon`
- `check_plan_duration`
- `check_text`

Those rules live in [sql/legacy/foundation/metro_bike_share.sql](sql/legacy/foundation/metro_bike_share.sql). The quarter-level staging filters live in [sql/legacy/staging](sql/legacy/staging). The Python implementation mirrors those rules for CSV ingestion and documents the reuse in `outputs/reports/legacy_reuse_summary.json` when the pipeline runs.

If future source files drift further from the historical schema, the Python cleaner serves as the operational fallback while still keeping the original SQL heuristics visible and reusable.

## Data flow

1. Read raw CSV files from `data/raw/trips/`
2. Standardize column names and parse timestamps safely
3. Apply legacy-rule cleaning, validation, filtering, and deduplication
4. Persist canonical cleaned trip data to PostgreSQL
5. Build aggregated demand tables for:
   - hourly
   - daily
   - weekly
   - monthly
   - quarterly
6. Build feature stores with:
   - calendar features
   - lag and rolling features
   - holiday flags
   - pandemic phase flags
   - interaction features
   - Fourier terms
7. Run diagnostics and save plots/reports
8. Create strict train / validation / final test windows per frequency and segment
9. Run rolling-origin backtests inside the training span
10. Select champion models from validation only
11. Check promoted models on the final held-out test window
12. Generate probabilistic forecasts and persist them
13. Build coherent station forecasts and reconciliation diagnostics
14. Write monitoring outputs for data quality and drift

## Pandemic-aware design

The pipeline treats the pandemic as a structural break, not ordinary seasonality.

It creates these phases:

- `pre_pandemic`
- `pandemic_shock`
- `recovery`
- `post_pandemic`

It combines:

- known business-rule anchors
- changepoint detection when available

It also creates:

- `is_lockdown`
- `is_reopening`
- `is_post_pandemic`
- `days_since_lockdown_start`
- `days_since_reopening_start`
- interaction features such as hour/day/month crossed with pandemic phase

## Project structure

```text
.
├── data
│   ├── raw
│   ├── interim
│   └── processed
├── docs
├── metro_bike_share_studio.py
├── notebooks
├── outputs
│   ├── reports
│   ├── figures
│   └── forecasts
├── scripts
├── sql
│   ├── legacy
│   ├── forecasting
│   └── warehouse
├── src/metro_bike_share_forecasting
│   ├── cleaning
│   ├── config
│   ├── database
│   ├── diagnostics
│   ├── evaluation
│   ├── features
│   ├── forecasting
│   ├── ingestion
│   ├── monitoring
│   ├── orchestration
│   ├── selection
│   ├── utils
│   └── validation
└── tests
```

## PostgreSQL tables

The schema in [sql/forecasting/001_create_forecasting_schema.sql](sql/forecasting/001_create_forecasting_schema.sql) creates:

- `raw_ingestion_log`
- `cleaned_trip_data`
- `aggregated_hourly`
- `aggregated_daily`
- `aggregated_weekly`
- `aggregated_monthly`
- `aggregated_quarterly`
- `feature_store_hourly`
- `feature_store_daily`
- `feature_store_weekly`
- `feature_store_monthly`
- `model_registry`
- `backtest_results`
- `forecast_outputs`
- `forecast_intervals`
- `drift_monitoring`
- `data_quality_monitoring`
- `pipeline_run_log`
- `champion_model_registry`

## Setup

The easiest setup path is the bootstrap helper:

```bash
python3 scripts/bootstrap.py --prepare
```

That creates `.venv` and installs the Python dependencies used by the pipeline and dashboard.

Manual setup still works:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `POSTGRES_URL` if you want PostgreSQL persistence.

Important note: the bootstrap helper prepares the Python environment. It does not try to force-install OS-level packages, which keeps setup safer and more portable across macOS, Linux, and Windows.

## Environment variables

See [.env.example](.env.example).

Key settings:

- `POSTGRES_URL`
- `POSTGRES_SCHEMA`
- `RAW_TIMEZONE`
- `FREQUENCIES`
- `HORIZON_DAILY`, `HORIZON_WEEKLY`, etc.
- `VALIDATION_WINDOW_DAILY`, `TEST_WINDOW_DAILY`, etc.
- `MAX_BACKTEST_FOLDS`
- `STATION_LEVEL_TOP_N`
- `STATION_LEVEL_FREQUENCIES`

## How to run

Inspect the reused base-project logic:

```bash
PYTHONPATH=src python -m metro_bike_share_forecasting.cli inspect-base-logic
```

Run a fast repo-local daily version:

```bash
make run-daily-fast
```

Run the full first-version pipeline:

```bash
PYTHONPATH=src python -m metro_bike_share_forecasting.cli run-full-pipeline
```

Launch the unified studio:

```bash
python3 scripts/bootstrap.py --dashboard
```

Or from an activated environment:

```bash
PYTHONPATH=src python -m streamlit run metro_bike_share_studio.py
```

Or use Make targets:

```bash
make bootstrap
make inspect-base
make run-daily-fast
make run-full
make studio
make test
```

## Studio and communication layer

The project now includes a simple communication layer so the work is easier to demo and understand.

- audience:
  operations managers, fleet planners, and analytics leads
- story:
  evaluation is separated from future forecasting, so users can see how models performed on held-out history before looking at the production forecast
- features:
  latest champion decision, held-out backtest views, future forecast comparisons, diagnostics story, segment explorer, and artifact browser

The entrypoint is [metro_bike_share_studio.py](/Users/morteza/Metro-Bike-Share/metro_bike_share_studio.py), and the dashboard logic lives in [dashboard.py](/Users/morteza/Metro-Bike-Share/src/metro_bike_share_forecasting/dashboard.py).

## How evaluation works

- strict time order only
- each segment gets:
  - train window
  - validation window
  - final test window
- rolling backtests run only inside the training span
- champion selection uses validation only
- the final test window is reserved for post-selection evaluation
- ensemble weights are learned from validation performance, not final test
- no random split
- frequency-specific horizons
- multiple folds
- metrics stored per model, fold, horizon step, and regime
- evaluation tab:
  shows held-out predictions versus actuals
- forecast tab:
  shows future forecasts after the selected models are refit on the full history

A fold is one rolling time-based evaluation window.

Example for daily demand:

- `initial_window = 365`
- `horizon = 28`
- `step = 7`
- `max_backtest_folds = 8`

That means:

- train on the first 365 days
- predict the next 28 days
- move forward by 7 days
- repeat for up to 8 evaluation windows

Metrics include:

- MAE
- RMSE
- MAPE
- sMAPE
- MASE
- pinball loss
- interval coverage
- interval width
- bias

## Champion selection

Champion selection is explainable and practical.

- challenger models are backtested first
- backtest summaries are aggregated per frequency
- a composite score ranks models
- the champion is the model with the best rolling backtest profile
- ensemble weights are derived from challenger performance

The selected champion is stored in `champion_model_registry`.

## Forecast outputs

Each forecast includes:

- point forecast
- lower and upper bounds for 50%
- lower and upper bounds for 80%
- lower and upper bounds for 95%
- model name
- frequency
- horizon
- generation timestamp
- training window metadata

Forecasts are persisted into:

- `forecast_outputs`
- `forecast_intervals`

They are also exported locally into `outputs/forecasts/` for easier review when PostgreSQL is not configured.

## Local artifacts you should expect

After a successful run, you should see:

- diagnostics plots in `outputs/figures/<frequency>/`
- run summaries, champion CSVs, backtest summaries, backtest predictions, and fold schedules in `outputs/reports/`
- forecast CSVs in `outputs/forecasts/`
- cleaned and aggregated datasets in `data/processed/`

## Monitoring

The first version stores:

- ingestion counts
- schema drift flags
- missingness checks
- duplicate removal summaries
- recent-vs-historical demand drift metrics

These are written to:

- `data_quality_monitoring`
- `drift_monitoring`

## Current limitations

- PostgreSQL writes require installing `sqlalchemy` and `psycopg2-binary`
- the first version trains on the system-total series while preserving station-level aggregates for later fleet-distribution work
- some historical enrichment assets in the legacy SQL project still require normalization before they are folded into the forecasting layer
- holiday and changepoint features degrade gracefully if optional dependencies are not installed

## Next steps

- add TBATS
- add tree-based lag models
- add deeper probabilistic global models
- train grouped station-level models for rebalancing use cases
- add downstream fleet management and inventory optimization
