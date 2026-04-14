from __future__ import annotations

import pandas as pd

from metro_bike_share_forecasting.station_level.diagnosis.config import StationDiagnosisConfig


def assign_station_categories(station_summary: pd.DataFrame, config: StationDiagnosisConfig) -> pd.DataFrame:
    """Assign interpretable rule-based categories to stations."""

    if station_summary.empty:
        return station_summary.copy()

    categorized = station_summary.copy()
    reference_frame = categorized.loc[categorized["history_group"] == "mature", ["avg_demand"]].dropna()
    if reference_frame.empty:
        reference_frame = categorized[["avg_demand"]].dropna()
    busy_threshold = float(reference_frame["avg_demand"].quantile(config.busy_avg_demand_quantile)) if not reference_frame.empty else 0.0

    def categorize(row: pd.Series) -> str:
        history_group = str(row.get("history_group", ""))
        zero_rate = float(row.get("zero_rate", float("nan")))
        active_day_rate = float(row.get("active_day_rate", float("nan")))
        outlier_rate = float(row.get("outlier_rate", float("nan")))
        avg_demand = float(row.get("avg_demand", float("nan")))
        cv = float(row.get("coefficient_of_variation", float("nan")))
        weekday_effect = float(row.get("weekday_effect_strength", float("nan")))
        weekend_ratio = float(row.get("weekend_ratio", float("nan")))

        if history_group in {"newborn", "young"}:
            return "short_history"
        if zero_rate >= config.sparse_zero_rate_threshold or active_day_rate <= config.sparse_active_day_rate_threshold:
            return "sparse_intermittent"
        if pd.notna(outlier_rate) and outlier_rate >= config.anomaly_outlier_rate_threshold:
            return "anomaly_heavy"
        if avg_demand >= busy_threshold and pd.notna(cv) and cv <= config.stable_cv_threshold:
            return "busy_stable"
        if avg_demand >= busy_threshold and pd.notna(cv) and cv >= config.volatile_cv_threshold:
            return "busy_volatile"
        if pd.notna(weekday_effect) and pd.notna(weekend_ratio):
            if weekday_effect >= config.commuter_weekday_effect_threshold and weekend_ratio <= config.commuter_weekend_ratio_threshold:
                return "seasonal_commuter"
            if weekend_ratio >= config.leisure_weekend_ratio_threshold:
                return "weekend_leisure"
        return "mixed_profile"

    categorized["station_category"] = categorized.apply(categorize, axis=1)
    return categorized


def build_station_category_summary(categorized: pd.DataFrame) -> pd.DataFrame:
    if categorized.empty:
        return pd.DataFrame()
    return (
        categorized.groupby("station_category", as_index=False)
        .agg(
            station_count=("station_id", "nunique"),
            avg_demand_mean=("avg_demand", "mean"),
            zero_rate_mean=("zero_rate", "mean"),
            outlier_rate_mean=("outlier_rate", "mean"),
            mature_share=("history_group", lambda values: float((pd.Series(values) == "mature").mean())),
        )
        .sort_values(["station_count", "avg_demand_mean"], ascending=[False, False])
        .reset_index(drop=True)
    )
