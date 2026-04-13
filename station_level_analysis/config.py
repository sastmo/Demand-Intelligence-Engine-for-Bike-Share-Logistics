from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StationDiagnosisConfig:
    """Configuration for rule-based station diagnosis."""

    output_dir: Path = field(default_factory=lambda: Path("station_level_analysis") / "outputs")
    top_n: int = 5
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

