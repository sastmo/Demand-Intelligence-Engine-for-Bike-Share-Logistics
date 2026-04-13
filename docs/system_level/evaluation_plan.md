# System-Level Evaluation Plan

## Validation Design
- Use rolling-origin / walk-forward backtesting only.
- Do not use random train/test splits.
- Evaluate at least 7-day and 30-day horizons.

## Baseline-First Philosophy
The first question is not whether a complex model can fit the series.  
The first question is whether it can beat a strong seasonal baseline on a fair time-based evaluation.

## Metrics
Primary metrics:
- MAE for plain forecast error scale
- RMSE for penalizing larger misses
- MASE for scale-free comparison against a seasonal naive reference

Secondary metric:
- Bias / mean error to understand systematic over- or under-forecasting

Optional later metrics:
- Coverage
- Mean interval width

MAPE is not the main metric because the series is nonnegative but not well described by percentage error alone.

## Horizon-Based Evaluation
Evaluation should be reported separately for:
- 7-day horizon
- 30-day horizon

This matters because the best short-horizon model may not be the best monthly-horizon model.

## Handling Anomalies In Evaluation
- Keep anomalies in the test windows because they are part of the business problem.
- Do not hide model weakness by removing difficult periods from evaluation.
- Track bias and RMSE alongside MAE so spike handling is visible.

## Optional Interval Evaluation
If intervals are added later:
- track empirical coverage
- track average interval width
- compare interval behavior by horizon

The first version should get the point forecast evaluation right before expanding interval logic.
