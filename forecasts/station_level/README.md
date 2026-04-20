# Station-Level Demand Forecasting

![Forecasting](https://img.shields.io/badge/Forecasting-Station%20Level-69BE45?style=flat-square)
![Panel](https://img.shields.io/badge/Panel-Service%20Aware-2F80C9?style=flat-square)
![Horizon](https://img.shields.io/badge/Horizon-Multi--Horizon%20%287%2C%2030%29-2F80C9?style=flat-square)
![Models](https://img.shields.io/badge/Models-DeepAR%20%7C%20LGBM%20%7C%20XGBoost-2F80C9?style=flat-square)
![Baselines](https://img.shields.io/badge/Baselines-Naive%20%7C%20Seasonal%20Naive%20%287%29-2F80C9?style=flat-square)
![Validation](https://img.shields.io/badge/Validation-Rolling%20Backtest%20%28MASE%29-E88A45?style=flat-square)
![Metrics](https://img.shields.io/badge/Metrics-MAE%20%7C%20RMSE%20%7C%20MASE%20%7C%20Bias-2F80C9?style=flat-square)

This module examines forecasting at the **station-day level**, where the target is daily rides for each station while it is actually in service. Unlike the system-level view, the goal here is not just to forecast total network demand, but to preserve **local operational visibility** across the station network.

The workflow uses **one global station-day forecasting pipeline** across an intentionally **unbalanced panel**, then compares baselines, pooled tree models, and DeepAR within the same rolling backtest framework. Performance is reviewed across **7-day and 30-day horizons**, with results reported both overall and by operational slices such as maturity, sparsity, category, and cluster.

![Station-level model comparison](figures/station_level_model_comparison.png)

## Workflow

1. Build a station-day panel using only days when each station is actually in service.
2. Train one global forecasting pipeline across the full station network.
3. Compare baselines, LightGBM, XGBoost, and DeepAR using time-based rolling backtests.
4. Report results overall and by slice to separate healthy-core performance from short-history and weak-signal stations.

## What This Shows

- the station-level forecasting setup
- the shared backtest framework across 7-day and 30-day horizons
- the model comparison view for pooled baselines, ML, and DeepAR
- the shift from network-level forecasting to local station-level visibility

## Key Takeaways

- Station-level forecasting should be built as **one global station-day pipeline**, not as many separate first-stage station models.
- The panel is intentionally **unbalanced** because stations open, mature, and leave service at different times.
- **XGBoost and LightGBM** are the strongest performers in the current backtest comparison.
- **DeepAR** is included as a heavier benchmark, but it is not the leading option in the current run.
- Forecast quality should be interpreted by **horizon and slice**, not by one pooled average alone.
- Slice-based reporting is important because station behavior is heterogeneous across maturity, sparsity, category, and cluster.

## Outcome

The main outcome is a working **station-level forecasting framework** that keeps training, validation, metrics, and reporting inside one comparable pipeline while preserving local demand detail. This creates a practical foundation for station-level benchmarking, operational planning, and targeted refinement where the residuals show real value.
