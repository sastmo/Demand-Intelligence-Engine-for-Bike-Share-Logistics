from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StationDiagnosisConfig:
    """Configuration for station-level diagnosis, categorization, and clustering."""

    output_root: Path = field(default_factory=lambda: Path("diagnosis") / "station_level_analysis" / "outputs")
    top_n: int = 5
    n_clusters: int = 6
    random_state: int = 42
    outlier_z_threshold: float = 3.5
    sparse_zero_rate_threshold: float = 0.45
    sparse_active_day_rate_threshold: float = 0.55
    anomaly_outlier_rate_threshold: float = 0.05
    busy_avg_demand_quantile: float = 0.75
    stable_cv_threshold: float = 0.80
    volatile_cv_threshold: float = 1.20
    commuter_weekday_effect_threshold: float = 0.18
    commuter_weekend_ratio_threshold: float = 0.85
    leisure_weekend_ratio_threshold: float = 1.15
    min_history_for_autocorr: int = 10
    rolling_window_days: int = 30
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
