# System-Level Forecasting Module

This module is only for the aggregated daily total demand across the full network.

Diagnosis and forecasting are intentionally separated on this branch:

- diagnosis lives in `diagnosis/system_level_analysis/`
- forecasting code lives in `src/metro_bike_share_forecasting/system_level/forecasting/`
- forecasting outputs live in `forecasts/system_level/`

## What It Does
- builds the system-level daily target
- builds calendar, holiday, lag, rolling, and optional external features
- runs rolling backtests
- compares baseline, classical, and ML models
- writes outputs under `forecasts/system_level/`

## What It Does Not Do
- no station-level forecasting
- no deep multi-series models
- no fleet optimization logic

## Main Entry Script
```bash
python3 scripts/system_level/forecasting/run_system_level_pipeline.py --config configs/system_level/config.yaml
```

## Related Diagnosis Entry Script
```bash
python3 scripts/system_level/diagnosis/run_diagnostics.py --synthetic-demo --target-col value --frequency daily
```

## Output Areas
- `forecasts/system_level/forecasts/`
- `forecasts/system_level/metrics/`
- `forecasts/system_level/backtests/`
- `forecasts/system_level/reports/`
- `forecasts/system_level/models/`
- `forecasts/system_level/feature_artifacts/`
- `forecasts/system_level/figures/`
