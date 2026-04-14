# Station-Level Diagnosis Summary

This is station-level diagnosis only. It is not system-level analysis, forecasting, or model training.

## Station Universe Validation
- Expected station count: 340
- Observed unique stations: 381
- Count gap: +41
- Short-history stations: 77
- Nearly always-zero stations: 6
- Not recently active stations: 158
- The higher-than-expected station count is likely explained by temporary, retired, or nearly empty stations that still appear in the station inventory.

## Date Range and Maturity
- Date range covered: 2019-01-01 to 2024-12-31
- `newborn`: 46 stations
- `young`: 31 stations
- `mature`: 304 stations

## Category Counts
- `mixed_profile`: 123 stations
- `short_history`: 77 stations
- `weekend_leisure`: 74 stations
- `busy_stable`: 50 stations
- `sparse_intermittent`: 44 stations
- `anomaly_heavy`: 7 stations
- `seasonal_commuter`: 6 stations

## Cluster Counts
- `cluster_1`: 32 stations
- `cluster_2`: 139 stations
- `cluster_3`: 43 stations
- `cluster_4`: 90 stations
- `not_clustered_short_history`: 77 stations

## Cluster Assessment
- Overall cluster strength: meaningful
- Clustering looks meaningful as a later refinement because the selected solution separates mature stations reasonably well without tiny unstable clusters.
- Selected clustering setup: k=4, silhouette=0.226
- Selection reason: best_silhouette_without_tiny_clusters

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
- `4331`: coefficient_of_variation=3.879
- `4135`: coefficient_of_variation=2.952
- `4457`: coefficient_of_variation=2.458
- `4134`: coefficient_of_variation=1.832
- `4602`: coefficient_of_variation=1.781

## Top 5 Anomaly-Heavy Stations
- `4539`: outlier_rate=0.147
- `3030`: outlier_rate=0.141
- `3014`: outlier_rate=0.126
- `4669`: outlier_rate=0.116
- `4614`: outlier_rate=0.100

## Interpretation
- Station behavior looks heterogeneous rather than homogeneous.
- About 53% of stations are active on at least 75% of days, while about 12% are clearly sparse or intermittent.
- Strong weekday commuter structure appears in about 2% of stations.
- Sparse stations should not be allowed to dominate the interpretation because their ratios and autocorrelation measures become less reliable.
- A single global model still looks like a reasonable first benchmark, but it should be paired with explicit sparse-station handling if sparsity is material.
- Clustering looks meaningful as a later refinement because the selected solution separates mature stations reasonably well without tiny unstable clusters.
- DeepAR or another global probabilistic model remains a strong later candidate because the station universe contains many related series with shared calendar structure and meaningful cross-station variation.

## Final Recommendation
- Recommended next step: **global model plus cluster-based refinement**
- Use station-day as the primary forecasting unit in the next stage.
- Keep seasonal naive per station as the first baseline benchmark.
- Add a pooled global tree-based benchmark with lag and calendar features before moving to deeper models.
- Keep DeepAR or another global probabilistic sequence model as a strong next candidate because the project has many related station series and interval forecasting will matter later.
- Newborn and young stations should be handled carefully because short history can distort station-specific feature quality.
- Cluster-based refinement looks justifiable later, but only after a strong global baseline is established.