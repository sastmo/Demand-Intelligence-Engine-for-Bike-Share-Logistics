# Aggregate Network Demand Forecasting

<div style="display:flex;flex-wrap:wrap;gap:6px 8px;margin:6px 0 14px 0;">
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Forecasting</span>
    <span style="background:#69BE45;color:#FFFFFF;padding:5px 9px;">System Level</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Horizon Design</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">Multi-Horizon</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Methods</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">Time Series + ML</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Validation</span>
    <span style="background:#E88A45;color:#FFFFFF;padding:5px 9px;">Walk-Forward</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Metrics</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">MAE RMSE MASE</span>
  </span>
  <span style="display:inline-flex;border-radius:6px;overflow:hidden;font-family:Arial,sans-serif;font-size:11px;line-height:1;">
    <span style="background:#555555;color:#FFFFFF;padding:5px 9px;">Uncertainty</span>
    <span style="background:#2F80C9;color:#FFFFFF;padding:5px 9px;">Confidence Intervals</span>
  </span>
</div>

This module examines forecasting at the **aggregate network level**, where the target is total daily demand across the full system. The goal is to build a reliable system-level view that supports planning, monitoring, and decision-making before moving to more granular forecasting layers.

The workflow compares multiple forecasting methods across **7-day, 30-day, and 90-day horizons**, combining **time series** and **machine learning** approaches within a shared evaluation framework. Performance is assessed with **walk-forward validation**, practical error metrics, and **confidence intervals** to make the results more useful for real planning decisions.

![System-level model comparison](figures/system_level_model_comparison.png)

## Workflow

1. Aggregate the network into a single daily demand series.
2. Train and compare multiple forecasting methods on the same system-level target.
3. Evaluate performance with walk-forward validation to preserve time order.
4. Review results by forecast horizon using both point forecasts and confidence intervals.

## What This Shows

- the system-level forecasting setup
- the multi-horizon comparison framework
- the model performance view across horizons
- the practical planning takeaway from the current baseline

## Key Takeaways

- Aggregate demand is forecastable and strong enough to support practical planning.
- The forecasting pipeline is functioning end to end.
- Both time series and machine learning methods can be compared within the same framework.
- Forecast quality should be interpreted by horizon rather than by a single summary score.
- Confidence intervals add useful context for planning, inventory, and redistribution decisions.
- The 7-day horizon is mainly operational, the 30-day horizon is tactical, and the 90-day horizon is best treated as directional sensitivity.

## Outcome

The main outcome is a working **system-level forecasting baseline** with horizon-aware model comparison. This creates a practical foundation for aggregate demand planning, network monitoring, and better-informed operational decisions.