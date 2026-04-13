# Station-Level Diagnosis Summary

This is station-level diagnosis only. It is not system-level analysis, forecasting, or model training.

- Number of stations: 381
- Date range covered: 2019-01-01 to 2024-12-31
- Dominant category: mixed_profile

## Station Categories
- `mixed_profile`: 102 stations
- `weekend_leisure`: 91 stations
- `sparse_intermittent`: 91 stations
- `busy_stable`: 64 stations
- `anomaly_heavy`: 26 stations
- `seasonal_commuter`: 7 stations

## Top 5 Busiest Stations
- `4407`: avg_demand=150.000
- `4670`: avg_demand=128.000
- `4633`: avg_demand=100.000
- `4646`: avg_demand=94.000
- `4402`: avg_demand=90.000

## Top 5 Sparsest Stations
- `4629`: zero_rate=0.997
- `4625`: zero_rate=0.997
- `4403`: zero_rate=0.997
- `4634`: zero_rate=0.995
- `4395`: zero_rate=0.980

## Top 5 Most Volatile Stations
- `4403`: coefficient_of_variation=22.875
- `4625`: coefficient_of_variation=19.598
- `4629`: coefficient_of_variation=19.193
- `4634`: coefficient_of_variation=13.610
- `4395`: coefficient_of_variation=8.811

## Top 5 Anomaly-Heavy Stations
- `4626`: outlier_rate=0.333
- `4667`: outlier_rate=0.333
- `4675`: outlier_rate=0.333
- `3030`: outlier_rate=0.139
- `4440`: outlier_rate=0.125

## Interpretation
- Stations look heterogeneous rather than uniform.
- Sparse or intermittent behavior appears in about 24% of stations.
- Clear weekday commuter structure appears in about 2% of stations.
- One global model may be too blunt on its own; sparse, commuter, and anomaly-heavy stations likely need differentiated treatment later.
- The summary table should be reviewed before deciding whether later forecasting should use one global model, special sparse-station handling, or a more formal grouping strategy.