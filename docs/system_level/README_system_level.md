# System-Level Forecasting Module

This module is only for the aggregated daily total demand across the full network.

## What It Does
- builds the system-level daily target
- builds calendar, holiday, lag, rolling, and optional external features
- runs rolling backtests
- compares baseline, classical, and ML models
- writes outputs under `outputs/system_level/`

## What It Does Not Do
- no station-level forecasting
- no deep multi-series models
- no fleet optimization logic

## Main Entry Script
```bash
python3 scripts/system_level/run_system_level_pipeline.py --config configs/system_level/config.yaml
```

## Output Areas
- `outputs/system_level/forecasts/`
- `outputs/system_level/metrics/`
- `outputs/system_level/backtests/`
- `outputs/system_level/reports/`
- `outputs/system_level/models/`
- `outputs/system_level/feature_artifacts/`
- `outputs/system_level/figures/`
