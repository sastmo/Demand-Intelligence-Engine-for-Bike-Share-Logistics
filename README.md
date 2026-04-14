# Metro Bike Share Forecasting

This branch is organized around two separate workstreams:

- `system_level`: aggregate daily demand diagnosis and forecasting
- `station_level`: station diagnosis first, forecasting later

The main cleanup goal is consistency. Diagnosis code lives under `diagnosis/`, reusable forecasting code lives under `src/`, runnable entrypoints live under `scripts/`, and generated artifacts live under either `diagnosis/*/outputs/` or `forecasts/system_level/`.

## Current Status

| Area | Current stage | Main location |
|------|---------------|---------------|
| System-level diagnosis | Implemented | `diagnosis/system_level_analysis/` |
| System-level forecasting | Implemented | `src/metro_bike_share_forecasting/system_level/forecasting/` |
| Station-level diagnosis | Implemented | `diagnosis/station_level_analysis/` |
| Station-level forecasting | Not started yet | reserved for later |

## Repository Standard

- `diagnosis/`: diagnosis-only workflows and their local outputs
- `src/`: reusable Python packages
- `scripts/`: operational entrypoints
- `configs/`: runtime configuration
- `contracts/`: forecasting contracts and scope documents
- `docs/`: project docs, architecture notes, and decision records
- `forecasts/system_level/`: system-level forecasting artifacts
- `data/interim/`: derived local working data, not committed by default

Diagnosis outputs and forecasting outputs are intentionally separate:

- `diagnosis/system_level_analysis/outputs/`: system-level diagnosis artifacts
- `diagnosis/station_level_analysis/outputs/`: station-level diagnosis artifacts
- `forecasts/system_level/`: system-level backtests, forecasts, metrics, and reports

## Repository Layout

```text
.
├── configs/
│   └── system_level/
├── contracts/
│   └── system_level/
├── data/
│   ├── raw/
│   ├── interim/
│   │   └── station_level/
│   └── processed/
├── diagnosis/
│   ├── system_level_analysis/
│   └── station_level_analysis/
├── docs/
│   └── system_level/
├── forecasts/
│   └── system_level/
├── scripts/
│   ├── system_level/
│   └── station_level/
├── src/
│   ├── forecasting_diagnostics/
│   └── metro_bike_share_forecasting/
│       ├── system_level/
│       └── station_level/
├── sql/
│   ├── forecasting/
│   ├── legacy/
│   └── warehouse/
└── tests/
```

More detail is in [docs/repo_structure.md](docs/repo_structure.md) and [docs/architecture.md](docs/architecture.md).

## Main Commands

### System-level diagnosis

Synthetic smoke run:

```bash
source .venv/bin/activate
python3 scripts/system_level/diagnosis/run_diagnostics.py --synthetic-demo --target-col value --frequency daily
```

Run on a real daily series:

```bash
source .venv/bin/activate
python3 scripts/system_level/diagnosis/run_diagnostics.py data/processed/daily_aggregate.csv.gz \
  --target-col trip_count \
  --time-col bucket_start \
  --frequency daily \
  --segment-type system_total \
  --segment-id all \
  --seasonal-periods 7 30 365
```

Outputs go to:

- `diagnosis/system_level_analysis/outputs/figures/`
- `diagnosis/system_level_analysis/outputs/tables/`
- `diagnosis/system_level_analysis/outputs/reports/`

### Station-level diagnosis

The station diagnosis stage is summary + categorization + clustering only. It does not train forecasting models.

```bash
source .venv/bin/activate
python3 scripts/station_level/diagnosis/build_station_summary.py \
  --input data/interim/station_level/station_daily.csv \
  --date-col date \
  --station-col station_id \
  --target-col target \
  --n-clusters 6
```

Outputs go to:

- `diagnosis/station_level_analysis/outputs/tables/`
- `diagnosis/station_level_analysis/outputs/diagnostics/`
- `diagnosis/station_level_analysis/outputs/reports/`

### System-level forecasting

```bash
source .venv/bin/activate
.venv/bin/python scripts/system_level/forecasting/run_system_level_pipeline.py --config configs/system_level/config.yaml
```

Outputs go to:

- `forecasts/system_level/forecasts/`
- `forecasts/system_level/metrics/`
- `forecasts/system_level/backtests/`
- `forecasts/system_level/reports/`
- `forecasts/system_level/models/`
- `forecasts/system_level/feature_artifacts/`
- `forecasts/system_level/figures/`

## Data Conventions

- raw source files stay in `data/raw/`
- derived local analysis inputs go in `data/interim/`
- processed shared artifacts go in `data/processed/`
- station diagnosis input is currently expected at `data/interim/station_level/station_daily.csv`

## Forecasting Scope

### System level

Implemented now:

- daily total-demand feature generation
- rolling backtests
- point forecasts
- calibrated intervals
- system-level reports and figures

Main code:

- [docs/system_level/README_system_level.md](docs/system_level/README_system_level.md)
- [src/metro_bike_share_forecasting/system_level/forecasting/](src/metro_bike_share_forecasting/system_level/forecasting)

### Station level

Implemented now:

- one-row-per-station summary table
- rule-based categories
- clustering on station summary features
- diagnosis report

Not implemented yet:

- station-level forecasting models
- station-level probabilistic forecasts
- station-level production outputs

## Legacy SQL

The original SQL project is preserved under `sql/legacy/`. It is still useful as historical business logic and source cleaning context, but the active Python diagnosis and forecasting work on this branch is organized around the structure above.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

If you want the older integrated Streamlit studio, it is still available:

```bash
python3 scripts/bootstrap.py --dashboard
```

That studio reflects the older integrated pipeline. The branch standard going forward is the clearer split between `diagnosis/` and `system_level` forecasting paths described above.
