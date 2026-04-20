from __future__ import annotations

import pandas as pd

from station_level.diagnosis.config import StationDiagnosisConfig


def _confidence_from_row(row: pd.Series) -> tuple[str, str]:
    coverage = float(row.get("coverage_ratio", float("nan")))
    observed_days = int(row.get("observed_days", 0) or 0)
    history_group = str(row.get("history_group", ""))
    reliable_behavior = bool(row.get("metric_reliable_behavior", False))

    reasons: list[str] = []
    if not reliable_behavior:
        reasons.append("behavior metrics do not meet minimum coverage thresholds")
    if history_group in {"newborn", "young"}:
        reasons.append("station maturity is limited")
    if pd.notna(coverage) and coverage < 0.70:
        reasons.append("coverage remains below the strong-confidence threshold")
    if observed_days < 120:
        reasons.append("observed history is modest")

    if reliable_behavior and history_group == "mature" and pd.notna(coverage) and coverage >= 0.80 and observed_days >= 180:
        return "high", "High-confidence behavior label based on mature history and strong coverage."
    if reliable_behavior:
        return "medium", "; ".join(reasons) if reasons else "Behavior label is usable but not high-confidence."
    return "low", "; ".join(reasons) if reasons else "Behavior label is unreliable due to weak data support."


def assign_station_categories(station_summary: pd.DataFrame, config: StationDiagnosisConfig) -> pd.DataFrame:
    """Assign separate maturity and behavioral station labels with confidence metadata."""

    if station_summary.empty:
        return station_summary.copy()

    categorized = station_summary.copy()
    reference_mask = categorized["history_group"].eq("mature") & categorized["metric_reliable_behavior"].fillna(False)
    reference_frame = categorized.loc[reference_mask, ["avg_demand_observed"]].dropna()
    if reference_frame.empty:
        reference_frame = categorized.loc[categorized["metric_reliable_behavior"].fillna(False), ["avg_demand_observed"]].dropna()
    busy_threshold = (
        float(reference_frame["avg_demand_observed"].quantile(config.busy_avg_demand_quantile))
        if not reference_frame.empty
        else config.low_mean_demand_threshold
    )

    def categorize(row: pd.Series) -> str:
        zero_rate = float(row.get("zero_rate_observed", float("nan")))
        active_day_rate = float(row.get("active_day_rate_observed", float("nan")))
        outlier_rate = float(row.get("outlier_rate", float("nan")))
        avg_demand = float(row.get("avg_demand_observed", float("nan")))
        cv = float(row.get("coefficient_of_variation", float("nan")))
        weekday_effect = float(row.get("weekday_effect_strength", float("nan")))
        weekend_ratio = float(row.get("weekend_ratio", float("nan")))
        reliable = bool(row.get("metric_reliable_behavior", False))
        coverage_ratio = float(row.get("coverage_ratio", float("nan")))

        if not reliable:
            return "unclassified_due_to_low_coverage"
        if pd.notna(zero_rate) and pd.notna(active_day_rate):
            if zero_rate >= config.sparse_zero_rate_threshold or active_day_rate <= config.sparse_active_day_rate_threshold:
                return "sparse_intermittent"
        if pd.notna(outlier_rate) and outlier_rate >= config.anomaly_outlier_rate_threshold:
            return "anomaly_prone"
        if pd.notna(avg_demand) and avg_demand >= busy_threshold and pd.notna(cv):
            if cv <= config.stable_cv_threshold:
                return "busy_stable"
            if cv >= config.volatile_cv_threshold:
                return "busy_volatile"
        if pd.notna(weekday_effect) and pd.notna(weekend_ratio):
            if weekday_effect >= config.commuter_weekday_effect_threshold and weekend_ratio <= config.commuter_weekend_ratio_threshold:
                return "commuter_pattern"
            if weekend_ratio >= config.leisure_weekend_ratio_threshold:
                return "weekend_leisure"
        if pd.notna(coverage_ratio) and coverage_ratio < config.min_behavior_coverage_ratio:
            return "unclassified_due_to_low_coverage"
        return "mixed_profile"

    categorized["behavior_label"] = categorized.apply(categorize, axis=1)
    confidence = categorized.apply(_confidence_from_row, axis=1, result_type="expand")
    confidence.columns = ["behavior_label_confidence", "behavior_label_reliability_reason"]
    categorized = pd.concat([categorized, confidence], axis=1)
    categorized["station_category"] = categorized["behavior_label"]
    return categorized


def build_station_category_summary(categorized: pd.DataFrame) -> pd.DataFrame:
    if categorized.empty:
        return pd.DataFrame()
    return (
        categorized.groupby("behavior_label", as_index=False)
        .agg(
            station_count=("station_id", "nunique"),
            avg_demand_mean=("avg_demand_observed", "mean"),
            zero_rate_mean=("zero_rate_observed", "mean"),
            outlier_rate_mean=("outlier_rate", "mean"),
            mature_share=("history_group", lambda values: float((pd.Series(values) == "mature").mean())),
            high_confidence_share=("behavior_label_confidence", lambda values: float((pd.Series(values) == "high").mean())),
            low_confidence_share=("behavior_label_confidence", lambda values: float((pd.Series(values) == "low").mean())),
        )
        .sort_values(["station_count", "avg_demand_mean"], ascending=[False, False])
        .reset_index(drop=True)
    )
