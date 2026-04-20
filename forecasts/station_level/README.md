# Station-Level Demand Forecasting

<div style="display:flex;flex-wrap:wrap;gap:6px 8px;margin:6px 0 14px 0;">
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Forecasting</span>
    <span style="background:#69BE45;color:#FFFFFF;padding:5px 9px;">Station Level</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Panel Design</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">Service-Aware</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Methods</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">Baseline + ML + DeepAR</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Validation</span>
    <span style="background:#E88A45;color:#FFFFFF;padding:5px 9px;">Rolling Backtest</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Metrics</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">MAE RMSE MASE Bias</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
  </span>
</div>

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