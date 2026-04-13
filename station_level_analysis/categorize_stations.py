from __future__ import annotations

import pandas as pd

from station_level_analysis.config import StationDiagnosisConfig


def assign_station_categories(
    station_summary: pd.DataFrame,
    config: StationDiagnosisConfig,
) -> pd.DataFrame:
    """Assign interpretable rule-based categories to stations."""

    if station_summary.empty:
        return station_summary.copy()

    categorized = station_summary.copy()
    busy_threshold = float(categorized["avg_demand"].quantile(config.busy_avg_demand_quantile))

    def categorize(row: pd.Series) -> str:
        zero_rate = float(row.get("zero_rate", float("nan")))
        active_day_rate = float(row.get("active_day_rate", float("nan")))
        outlier_rate = float(row.get("outlier_rate", float("nan")))
        avg_demand = float(row.get("avg_demand", float("nan")))
        cv = float(row.get("coefficient_of_variation", float("nan")))
        weekday_effect = float(row.get("weekday_effect_strength", float("nan")))
        weekend_ratio = float(row.get("weekend_ratio", float("nan")))

        if zero_rate >= config.sparse_zero_rate_threshold or active_day_rate <= config.sparse_active_day_rate_threshold:
            return "sparse_intermittent"
        if outlier_rate >= config.anomaly_outlier_rate_threshold:
            return "anomaly_heavy"
        if avg_demand >= busy_threshold and cv <= config.stable_cv_threshold:
            return "busy_stable"
        if avg_demand >= busy_threshold and cv >= config.volatile_cv_threshold:
            return "busy_volatile"
        if weekday_effect >= config.commuter_weekday_effect_threshold and weekend_ratio <= config.commuter_weekend_ratio_threshold:
            return "seasonal_commuter"
        if weekend_ratio >= config.leisure_weekend_ratio_threshold:
            return "weekend_leisure"
        return "mixed_profile"

    categorized["station_category"] = categorized.apply(categorize, axis=1)
    return categorized
