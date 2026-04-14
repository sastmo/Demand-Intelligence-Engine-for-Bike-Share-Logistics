from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StationDiagnosisConfig:
    """Configuration for station-level diagnosis, categorization, clustering, and visuals."""

    output_root: Path = field(default_factory=lambda: Path("diagnosis") / "station_level_analysis" / "outputs")
    expected_station_count: int = 340
    top_n: int = 5
    n_clusters: int = 6
    cluster_k_values: tuple[int, ...] = (4, 5, 6, 7)
    cluster_mature_only: bool = True
    random_state: int = 42

    newborn_history_days: int = 90
    mature_history_days: int = 365
    recent_activity_window_days: int = 90
    min_recent_active_days: int = 3
    zero_almost_always_threshold: float = 0.95

    outlier_z_threshold: float = 3.5
    anomaly_outlier_rate_threshold: float = 0.08
    sparse_zero_rate_threshold: float = 0.60
    sparse_active_day_rate_threshold: float = 0.40
    low_mean_demand_threshold: float = 1.0
    max_coefficient_of_variation: float = 10.0
    min_history_for_autocorr: int = 30
    min_history_days_for_trend: int = 60
    min_active_days_for_pattern_metrics: int = 14
    min_active_days_for_outlier_metrics: int = 20
    rolling_window_days: int = 30

    busy_avg_demand_quantile: float = 0.75
    stable_cv_threshold: float = 0.85
    volatile_cv_threshold: float = 1.35
    commuter_weekday_effect_threshold: float = 0.18
    commuter_weekend_ratio_threshold: float = 0.85
    leisure_weekend_ratio_threshold: float = 1.20

    min_cluster_size_abs: int = 8
    min_cluster_size_ratio: float = 0.02

    representative_samples_per_category: int = 3
    representative_min_category_size: int = 3

    cluster_feature_columns: tuple[str, ...] = (
        "avg_demand",
        "coefficient_of_variation",
        "zero_rate",
        "lag1_autocorr",
        "lag7_autocorr",
        "weekday_effect_strength",
        "month_effect_strength",
        "weekend_ratio",
        "trend_slope",
        "outlier_rate",
        "correlation_with_system_total",
        "demand_share_of_system",
    )
