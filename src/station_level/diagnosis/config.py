from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StationDiagnosisConfig:
    """Configuration for station-level diagnosis, categorization, clustering, validation, and visuals."""

    output_root: Path = field(default_factory=lambda: Path("diagnosis") / "station_level" / "outputs")
    expected_station_count: int = 340
    top_n: int = 5
    random_state: int = 42

    recent_activity_window_days: int = 90
    newborn_observed_days: int = 30
    young_observed_days: int = 120
    mature_observed_days: int = 240
    mature_in_service_days: int = 300
    sparse_mature_span_days: int = 365
    stale_recent_observed_days_threshold: int = 3
    low_coverage_ratio_threshold: float = 0.60
    mature_coverage_ratio_threshold: float = 0.75
    recent_availability_ratio_threshold: float = 0.60
    recent_window_min_service_days: int = 30
    service_gap_days_for_inactive: int = 21
    post_service_gap_days: int = 30
    reliable_recent_days_window: int = 30

    zero_almost_always_threshold: float = 0.95
    sparse_zero_rate_threshold: float = 0.60
    sparse_active_day_rate_threshold: float = 0.40
    low_mean_demand_threshold: float = 1.0
    max_coefficient_of_variation: float = 10.0
    min_observed_days_for_autocorr: int = 30
    min_valid_pairs_for_autocorr: int = 20
    min_observed_days_for_trend: int = 45
    min_observed_days_for_pattern_metrics: int = 21
    min_positive_days_for_outlier_metrics: int = 20
    min_coverage_for_temporal_metrics: float = 0.65
    rolling_window_days: int = 30
    outlier_z_threshold: float = 3.5
    anomaly_outlier_rate_threshold: float = 0.08

    busy_avg_demand_quantile: float = 0.75
    stable_cv_threshold: float = 0.85
    volatile_cv_threshold: float = 1.35
    commuter_weekday_effect_threshold: float = 0.18
    commuter_weekend_ratio_threshold: float = 0.85
    leisure_weekend_ratio_threshold: float = 1.20
    min_behavior_coverage_ratio: float = 0.55
    min_behavior_observed_days: int = 28
    min_behavior_recent_observed_days: int = 10

    n_clusters: int = 6
    cluster_k_values: tuple[int, ...] = (4, 5, 6, 7)
    cluster_random_seeds: tuple[int, ...] = (11, 19, 29, 37, 43)
    cluster_bootstrap_iterations: int = 8
    cluster_bootstrap_fraction: float = 0.80
    cluster_mature_only: bool = True
    min_cluster_size_abs: int = 8
    min_cluster_size_ratio: float = 0.02
    min_cluster_eligible_stations: int = 12
    min_cluster_coverage_ratio: float = 0.70
    min_cluster_observed_days: int = 120
    cluster_stability_good_threshold: float = 0.75
    cluster_stability_ok_threshold: float = 0.55

    representative_samples_per_category: int = 3
    representative_min_category_size: int = 3

    validation_spike_quantile: float = 0.999
    validation_spike_multiplier: float = 8.0
    warning_min_usable_days: int = 14
    warning_long_missing_streak_days: int = 14

    cluster_feature_columns: tuple[str, ...] = (
        "avg_demand_observed",
        "coefficient_of_variation",
        "zero_rate_observed",
        "active_day_rate_observed",
        "coverage_ratio",
        "lag1_autocorr",
        "lag7_autocorr",
        "weekday_effect_strength",
        "month_effect_strength",
        "weekend_ratio",
        "trend_slope",
        "outlier_rate",
        "correlation_with_system_excl_self",
        "demand_share_of_system_observed_window",
        "fraction_of_demand_recent_window",
    )
