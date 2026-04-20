from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from station_level.diagnosis.config import StationDiagnosisConfig


OBSERVED_POSITIVE = "observed_positive"
OBSERVED_ZERO = "observed_zero"
MISSING_IN_SERVICE = "missing_in_service"
OUT_OF_SERVICE = "out_of_service"
PRE_SERVICE = "pre_service"
POST_SERVICE = "post_service"
UNKNOWN_AMBIGUOUS = "unknown_ambiguous"


@dataclass(frozen=True)
class HistoryGroupDecision:
    label: str
    reason: str


def _safe_ratio(numerator: float, denominator: float, min_denominator: float = 0.0) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or abs(float(denominator)) <= min_denominator:
        return float("nan")
    return float(numerator) / float(denominator)


def _longest_true_run(series: pd.Series) -> int:
    if series.empty:
        return 0
    values = series.fillna(False).astype(bool).to_numpy()
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _run_lengths(mask: pd.Series) -> pd.Series:
    if mask.empty:
        return pd.Series(dtype=int)
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    lengths = mask.groupby(groups).transform("sum")
    return lengths.astype(int)


def _median_inter_arrival_days(dates: pd.Series) -> float:
    ordered = pd.to_datetime(dates).sort_values().drop_duplicates()
    if len(ordered) < 2:
        return float("nan")
    diffs = ordered.diff().dt.days.dropna()
    return float(diffs.median()) if not diffs.empty else float("nan")


def _robust_cv(mean_value: float, std_value: float, config: StationDiagnosisConfig) -> float:
    if not np.isfinite(mean_value) or mean_value < config.low_mean_demand_threshold:
        return float("nan")
    cv = std_value / mean_value if mean_value else float("nan")
    if not np.isfinite(cv):
        return float("nan")
    return float(min(cv, config.max_coefficient_of_variation))


def _calendar_autocorr(
    target: pd.Series,
    lag: int,
    config: StationDiagnosisConfig,
) -> tuple[float, bool]:
    series = target.astype(float)
    shifted = series.shift(lag)
    valid_mask = series.notna() & shifted.notna()
    valid_pairs = int(valid_mask.sum())
    observed_days = int(series.notna().sum())
    if observed_days < config.min_observed_days_for_autocorr:
        return float("nan"), False
    if valid_pairs < max(config.min_valid_pairs_for_autocorr, lag + 2):
        return float("nan"), False
    left = series.loc[valid_mask]
    right = shifted.loc[valid_mask]
    if float(left.std(ddof=0)) == 0.0 or float(right.std(ddof=0)) == 0.0:
        return float("nan"), False
    return float(left.corr(right)), True


def _effect_strength(
    observed_frame: pd.DataFrame,
    group_col: str,
    config: StationDiagnosisConfig,
) -> tuple[float, bool]:
    if len(observed_frame) < config.min_observed_days_for_pattern_metrics:
        return float("nan"), False
    if float(observed_frame["target"].mean()) < config.low_mean_demand_threshold:
        return float("nan"), False
    grouped_means = observed_frame.groupby(group_col)["target"].mean()
    if grouped_means.empty or len(grouped_means) < 2:
        return float("nan"), False
    overall = float(observed_frame["target"].mean())
    if overall == 0.0:
        return float("nan"), False
    return float(grouped_means.std(ddof=0) / abs(overall)), True


def _weekend_ratio(observed_frame: pd.DataFrame, config: StationDiagnosisConfig) -> tuple[float, bool]:
    if len(observed_frame) < config.min_observed_days_for_pattern_metrics:
        return float("nan"), False
    if float(observed_frame["target"].mean()) < config.low_mean_demand_threshold:
        return float("nan"), False
    weekend_mask = observed_frame["weekday"].isin([5, 6])
    if weekend_mask.nunique() < 2:
        return float("nan"), False
    weekday_mean = float(observed_frame.loc[~weekend_mask, "target"].mean()) if (~weekend_mask).any() else float("nan")
    weekend_mean = float(observed_frame.loc[weekend_mask, "target"].mean()) if weekend_mask.any() else float("nan")
    return _safe_ratio(weekend_mean, weekday_mean, min_denominator=config.low_mean_demand_threshold / 2.0), True


def _trend_slope(observed_frame: pd.DataFrame, config: StationDiagnosisConfig) -> tuple[float, bool]:
    if len(observed_frame) < config.min_observed_days_for_trend:
        return float("nan"), False
    x = (observed_frame["date"] - observed_frame["date"].min()).dt.days.astype(float).to_numpy()
    y = observed_frame["target"].astype(float).to_numpy()
    if not np.isfinite(y).all() or float(np.std(y, ddof=0)) == 0.0:
        return float("nan"), False
    return float(np.polyfit(x, y, 1)[0]), True


def _rolling_shift(observed_target: pd.Series, window: int, reducer: str) -> tuple[float, bool]:
    series = observed_target.astype(float).dropna()
    if len(series) < window * 2:
        return float("nan"), False
    first = series.iloc[:window]
    last = series.iloc[-window:]
    if reducer == "mean":
        return float(last.mean() - first.mean()), True
    return float(last.std(ddof=0) - first.std(ddof=0)), True


def _outlier_rate(observed_target: pd.Series, config: StationDiagnosisConfig) -> tuple[float, bool]:
    positives = observed_target.astype(float).loc[observed_target.astype(float) > 0]
    if len(positives) < config.min_positive_days_for_outlier_metrics:
        return float("nan"), False
    median = float(positives.median())
    mad = float((positives - median).abs().median())
    if mad == 0.0:
        std = float(positives.std(ddof=0))
        if std == 0.0:
            return 0.0, True
        z_scores = (positives - float(positives.mean())) / std
    else:
        z_scores = 0.6745 * (positives - median) / mad
    return float((z_scores.abs() > config.outlier_z_threshold).mean()), True


def _system_minus_self_correlation(
    observed_frame: pd.DataFrame,
    system_total: pd.DataFrame,
    config: StationDiagnosisConfig,
) -> tuple[float, bool]:
    if len(observed_frame) < config.min_observed_days_for_autocorr:
        return float("nan"), False
    aligned = observed_frame[["date", "target"]].merge(system_total, on="date", how="left")
    aligned["system_total_excl_self"] = aligned["system_total"] - aligned["target"]
    valid = aligned[["target", "system_total_excl_self"]].dropna()
    if len(valid) < config.min_valid_pairs_for_autocorr:
        return float("nan"), False
    if float(valid["target"].std(ddof=0)) == 0.0 or float(valid["system_total_excl_self"].std(ddof=0)) == 0.0:
        return float("nan"), False
    return float(valid["target"].corr(valid["system_total_excl_self"])), True


def _monthly_share_cv(observed_frame: pd.DataFrame, system_total: pd.DataFrame) -> float:
    if observed_frame.empty:
        return float("nan")
    merged = observed_frame[["date", "target"]].merge(system_total, on="date", how="left")
    merged["system_total"] = merged["system_total"].astype(float)
    monthly = (
        merged.assign(month=merged["date"].dt.to_period("M"))
        .groupby("month", as_index=False)
        .agg(station_total=("target", "sum"), system_total=("system_total", "sum"))
    )
    if len(monthly) < 3:
        return float("nan")
    monthly["share"] = monthly["station_total"] / monthly["system_total"].replace(0.0, np.nan)
    shares = monthly["share"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(shares) < 3 or float(shares.mean()) == 0.0:
        return float("nan")
    return float(shares.std(ddof=0) / abs(shares.mean()))


def classify_history_group(row: pd.Series, config: StationDiagnosisConfig) -> HistoryGroupDecision:
    observed_days = int(row.get("observed_days", 0) or 0)
    in_service_days = int(row.get("in_service_days", 0) or 0)
    history_span_days = int(row.get("history_span_days", 0) or 0)
    coverage_ratio = float(row.get("coverage_ratio", float("nan")))
    recent_observed_days = int(row.get("recent_observed_days", 0) or 0)
    recent_active_days = int(row.get("recent_active_days", 0) or 0)
    recent_availability_ratio = float(row.get("recent_availability_ratio", float("nan")))
    days_since_last_observation = int(row.get("days_since_last_observation", 0) or 0)

    if days_since_last_observation > config.recent_activity_window_days and recent_observed_days <= config.stale_recent_observed_days_threshold:
        return HistoryGroupDecision(
            "stale_inactive",
            "No recent reliable observations inside the recent activity window.",
        )
    if observed_days < config.newborn_observed_days or in_service_days < config.newborn_observed_days:
        return HistoryGroupDecision(
            "newborn",
            "Too few observed or in-service days for stable station diagnosis.",
        )
    if history_span_days >= config.sparse_mature_span_days and observed_days >= config.young_observed_days and coverage_ratio < config.low_coverage_ratio_threshold:
        return HistoryGroupDecision(
            "sparse_mature",
            "Long calendar span but weak coverage inside the apparent service window.",
        )
    if (
        observed_days >= config.mature_observed_days
        and in_service_days >= config.mature_in_service_days
        and coverage_ratio >= config.mature_coverage_ratio_threshold
        and recent_availability_ratio >= config.recent_availability_ratio_threshold
    ):
        return HistoryGroupDecision(
            "mature",
            "Sufficient observed history, service continuity, and recent availability.",
        )
    if recent_active_days == 0 and days_since_last_observation > config.recent_activity_window_days // 2:
        return HistoryGroupDecision(
            "stale_inactive",
            "Observed history exists but recent activity and recent coverage are both weak.",
        )
    return HistoryGroupDecision(
        "young",
        "Observed history is usable but not yet strong enough for mature classification.",
    )


def build_station_analysis_panel(
    station_daily: pd.DataFrame,
    config: StationDiagnosisConfig,
) -> pd.DataFrame:
    """Build an explicit station-day panel with observed, service, and missingness states.

    Assumptions:
    - Observed rows represent reliable station-day observations, including true zero demand.
    - Missing days inside the apparent service window are not imputed to zero.
    - Long missing streaks inside the apparent service window are marked as likely out-of-service.
    - Days after the last observation are only marked post-service when the trailing gap is long enough;
      otherwise they remain unknown/ambiguous.
    """

    if station_daily.empty:
        return pd.DataFrame()

    global_start = pd.to_datetime(station_daily["date"]).min()
    global_end = pd.to_datetime(station_daily["date"]).max()
    full_dates = pd.date_range(global_start, global_end, freq="D")

    rows: list[pd.DataFrame] = []
    for station_id, station_frame in station_daily.groupby("station_id", sort=True):
        observed = station_frame.sort_values("date").copy()
        first_observed = observed["date"].min()
        last_observed = observed["date"].max()
        panel = pd.DataFrame({"date": full_dates})
        panel["station_id"] = str(station_id)
        panel = panel.merge(observed[["date", "target"]], on="date", how="left")

        panel["is_observed"] = panel["target"].notna()
        panel["within_service_window"] = panel["date"].between(first_observed, last_observed, inclusive="both")
        panel["missing_inside_service_window"] = (~panel["is_observed"]) & panel["within_service_window"]
        missing_run_lengths = _run_lengths(panel["missing_inside_service_window"])
        panel["likely_out_of_service_gap"] = panel["missing_inside_service_window"] & (
            missing_run_lengths >= config.service_gap_days_for_inactive
        )

        trailing_gap_days = int((global_end - last_observed).days)
        panel["day_state"] = np.select(
            condlist=[
                panel["is_observed"] & panel["target"].gt(0),
                panel["is_observed"] & panel["target"].eq(0),
                panel["date"].lt(first_observed),
                panel["date"].gt(last_observed) & (trailing_gap_days >= config.post_service_gap_days),
                panel["likely_out_of_service_gap"],
                panel["missing_inside_service_window"],
            ],
            choicelist=[
                OBSERVED_POSITIVE,
                OBSERVED_ZERO,
                PRE_SERVICE,
                POST_SERVICE,
                OUT_OF_SERVICE,
                MISSING_IN_SERVICE,
            ],
            default=UNKNOWN_AMBIGUOUS,
        )

        panel["observed_target"] = panel["target"]
        panel["is_observed_positive"] = panel["day_state"].eq(OBSERVED_POSITIVE)
        panel["is_observed_zero"] = panel["day_state"].eq(OBSERVED_ZERO)
        panel["is_observed_in_service"] = panel["day_state"].isin([OBSERVED_POSITIVE, OBSERVED_ZERO])
        panel["is_missing_in_service"] = panel["day_state"].eq(MISSING_IN_SERVICE)
        panel["is_out_of_service"] = panel["day_state"].eq(OUT_OF_SERVICE)
        panel["is_pre_service"] = panel["day_state"].eq(PRE_SERVICE)
        panel["is_post_service"] = panel["day_state"].eq(POST_SERVICE)
        panel["is_unknown_ambiguous"] = panel["day_state"].eq(UNKNOWN_AMBIGUOUS)
        panel["is_in_service"] = panel["day_state"].isin([OBSERVED_POSITIVE, OBSERVED_ZERO, MISSING_IN_SERVICE])
        panel["analysis_target"] = np.where(panel["is_observed_in_service"], panel["observed_target"], np.nan)
        panel["service_window_target"] = np.where(panel["within_service_window"], panel["observed_target"], np.nan)
        panel["first_observed_date"] = first_observed
        panel["last_observed_date"] = last_observed
        rows.append(panel)

    return pd.concat(rows, ignore_index=True).sort_values(["station_id", "date"]).reset_index(drop=True)


def build_station_inventory(
    station_daily: pd.DataFrame,
    config: StationDiagnosisConfig,
    analysis_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a station inventory table with explicit coverage and maturity diagnostics."""

    if station_daily.empty:
        return pd.DataFrame()
    if analysis_panel is None:
        analysis_panel = build_station_analysis_panel(station_daily, config)

    global_end = pd.to_datetime(station_daily["date"]).max()
    recent_start = global_end - pd.Timedelta(days=config.recent_activity_window_days - 1)

    rows: list[dict[str, object]] = []
    for station_id, panel in analysis_panel.groupby("station_id", sort=True):
        panel = panel.sort_values("date").copy()
        observed = panel.loc[panel["is_observed_in_service"]].copy()
        observed_positive = panel.loc[panel["is_observed_positive"]].copy()
        recent_panel = panel.loc[panel["date"] >= recent_start].copy()

        first_observed = observed["date"].min() if not observed.empty else pd.NaT
        last_observed = observed["date"].max() if not observed.empty else pd.NaT
        history_span_days = int((last_observed - first_observed).days) + 1 if pd.notna(first_observed) and pd.notna(last_observed) else 0
        observed_days = int(len(observed))
        observed_zero_days = int(panel["is_observed_zero"].sum())
        observed_positive_days = int(panel["is_observed_positive"].sum())
        missing_in_service_days = int(panel["is_missing_in_service"].sum())
        out_of_service_days = int(panel["is_out_of_service"].sum())
        pre_service_days = int(panel["is_pre_service"].sum())
        post_service_days = int(panel["is_post_service"].sum())
        unknown_days = int(panel["is_unknown_ambiguous"].sum())
        in_service_days = int(panel["is_in_service"].sum())
        recent_in_service_days = int(recent_panel["is_in_service"].sum())
        recent_observed_days = int(recent_panel["is_observed_in_service"].sum())
        recent_active_days = int(recent_panel["is_observed_positive"].sum())
        longest_missing_streak = _longest_true_run(panel["is_missing_in_service"])
        longest_active_streak = _longest_true_run(panel["is_observed_positive"])
        coverage_ratio = _safe_ratio(observed_days, in_service_days)
        recent_availability_ratio = _safe_ratio(recent_observed_days, recent_in_service_days)
        intermittent_gap_ratio = _safe_ratio(missing_in_service_days, in_service_days)
        service_continuity_score = float(max(0.0, 1.0 - _safe_ratio(longest_missing_streak, max(in_service_days, 1), min_denominator=0.0))) if in_service_days > 0 else float("nan")
        days_since_first_observation = int((global_end - first_observed).days) if pd.notna(first_observed) else np.nan
        days_since_last_observation = int((global_end - last_observed).days) if pd.notna(last_observed) else np.nan
        decision = classify_history_group(
            pd.Series(
                {
                    "observed_days": observed_days,
                    "in_service_days": in_service_days,
                    "history_span_days": history_span_days,
                    "coverage_ratio": coverage_ratio,
                    "recent_observed_days": recent_observed_days,
                    "recent_active_days": recent_active_days,
                    "recent_availability_ratio": recent_availability_ratio,
                    "days_since_last_observation": days_since_last_observation,
                }
            ),
            config,
        )

        rows.append(
            {
                "station_id": str(station_id),
                "first_observed_date": first_observed,
                "last_observed_date": last_observed,
                "history_span_days": history_span_days,
                "history_days": history_span_days,
                "observed_days": observed_days,
                "n_observed_days": observed_days,
                "observed_zero_days": observed_zero_days,
                "observed_positive_days": observed_positive_days,
                "missing_in_service_days": missing_in_service_days,
                "n_missing_days": missing_in_service_days,
                "out_of_service_days": out_of_service_days,
                "pre_service_days": pre_service_days,
                "post_service_days": post_service_days,
                "unknown_days": unknown_days,
                "in_service_days": in_service_days,
                "coverage_ratio": coverage_ratio,
                "observed_share": _safe_ratio(observed_days, history_span_days),
                "recent_in_service_days": recent_in_service_days,
                "recent_observed_days": recent_observed_days,
                "recent_active_days": recent_active_days,
                "recent_availability_ratio": recent_availability_ratio,
                "longest_missing_streak": longest_missing_streak,
                "longest_active_streak": longest_active_streak,
                "days_since_first_observation": days_since_first_observation,
                "days_since_last_observation": days_since_last_observation,
                "intermittent_gap_ratio": intermittent_gap_ratio,
                "service_continuity_score": service_continuity_score,
                "median_inter_arrival_active_days": _median_inter_arrival_days(observed_positive["date"]),
                "history_group": decision.label,
                "history_group_reason": decision.reason,
                "is_short_history": bool(decision.label in {"newborn", "young"}),
                "appears_active_recently": bool(recent_active_days >= 1),
                "is_zero_almost_always": bool(_safe_ratio(observed_zero_days, observed_days) >= config.zero_almost_always_threshold) if observed_days > 0 else False,
            }
        )

    return pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)


def build_station_summary_table(
    station_daily: pd.DataFrame,
    inventory: pd.DataFrame,
    config: StationDiagnosisConfig,
    analysis_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a robust one-row-per-station summary table for downstream diagnosis and forecasting."""

    if station_daily.empty:
        return pd.DataFrame()
    if analysis_panel is None:
        analysis_panel = build_station_analysis_panel(station_daily, config)

    inventory_lookup = inventory.set_index("station_id") if not inventory.empty else pd.DataFrame()
    system_total = station_daily.groupby("date", as_index=False)["target"].sum().rename(columns={"target": "system_total"})
    system_total_sum = float(system_total["system_total"].sum())
    global_end = pd.to_datetime(station_daily["date"]).max()
    recent_start = global_end - pd.Timedelta(days=config.recent_activity_window_days - 1)

    rows: list[dict[str, object]] = []
    for station_id, panel in analysis_panel.groupby("station_id", sort=True):
        panel = panel.sort_values("date").copy()
        observed = panel.loc[panel["is_observed_in_service"], ["date", "analysis_target"]].rename(columns={"analysis_target": "target"}).copy()
        observed["weekday"] = observed["date"].dt.dayofweek
        observed["month"] = observed["date"].dt.month
        calendar_series = panel.loc[panel["within_service_window"], ["date", "analysis_target"]].set_index("date")["analysis_target"]
        inventory_row = inventory_lookup.loc[str(station_id)] if not inventory.empty else pd.Series(dtype=object)

        mean_value = float(observed["target"].mean()) if not observed.empty else float("nan")
        std_value = float(observed["target"].std(ddof=0)) if not observed.empty else float("nan")
        total_demand = float(observed["target"].sum()) if not observed.empty else 0.0
        recent_observed = observed.loc[observed["date"] >= recent_start]
        early_mean_shift, mean_shift_reliable = _rolling_shift(observed["target"], config.rolling_window_days, "mean") if not observed.empty else (float("nan"), False)
        std_shift, std_shift_reliable = _rolling_shift(observed["target"], config.rolling_window_days, "std") if not observed.empty else (float("nan"), False)
        lag1_autocorr, lag1_reliable = _calendar_autocorr(calendar_series, 1, config)
        lag7_autocorr, lag7_reliable = _calendar_autocorr(calendar_series, 7, config)
        weekday_effect, weekday_reliable = _effect_strength(observed, "weekday", config) if not observed.empty else (float("nan"), False)
        month_effect, month_reliable = _effect_strength(observed, "month", config) if not observed.empty else (float("nan"), False)
        weekend_ratio, weekend_reliable = _weekend_ratio(observed, config) if not observed.empty else (float("nan"), False)
        trend_slope, trend_reliable = _trend_slope(observed, config) if not observed.empty else (float("nan"), False)
        outlier_rate, outlier_reliable = _outlier_rate(observed["target"], config) if not observed.empty else (float("nan"), False)
        corr_ex_self, corr_reliable = _system_minus_self_correlation(observed, system_total, config) if not observed.empty else (float("nan"), False)
        monthly_share_cv = _monthly_share_cv(observed, system_total)
        station_contribution_stability = float(1.0 / (1.0 + monthly_share_cv)) if pd.notna(monthly_share_cv) and monthly_share_cv >= 0 else float("nan")

        observed_dates = observed["date"].drop_duplicates()
        observed_window_system_total = float(system_total.loc[system_total["date"].isin(observed_dates), "system_total"].sum()) if not observed.empty else float("nan")
        demand_share_observed_window = _safe_ratio(total_demand, observed_window_system_total, min_denominator=config.low_mean_demand_threshold)
        fraction_recent = _safe_ratio(
            float(recent_observed["target"].sum()) if not recent_observed.empty else 0.0,
            total_demand,
            min_denominator=config.low_mean_demand_threshold,
        )

        coverage_ratio = float(inventory_row.get("coverage_ratio", float("nan")))
        missing_rate_in_service = _safe_ratio(
            float(inventory_row.get("missing_in_service_days", np.nan)),
            float(inventory_row.get("in_service_days", np.nan)),
        )
        activity_rate_in_service = _safe_ratio(
            float(inventory_row.get("observed_positive_days", np.nan)),
            float(inventory_row.get("in_service_days", np.nan)),
        )
        temporal_reliable = bool(pd.notna(coverage_ratio) and coverage_ratio >= config.min_coverage_for_temporal_metrics)

        rows.append(
            {
                "station_id": str(station_id),
                "start_date": inventory_row.get("first_observed_date", pd.NaT),
                "end_date": inventory_row.get("last_observed_date", pd.NaT),
                "n_days": int(inventory_row.get("observed_days", 0) or 0),
                "history_span_days": int(inventory_row.get("history_span_days", 0) or 0),
                "history_days": int(inventory_row.get("history_days", 0) or 0),
                "observed_days": int(inventory_row.get("observed_days", 0) or 0),
                "in_service_days": int(inventory_row.get("in_service_days", 0) or 0),
                "coverage_ratio": coverage_ratio,
                "recent_observed_days": int(inventory_row.get("recent_observed_days", 0) or 0),
                "recent_active_days": int(inventory_row.get("recent_active_days", 0) or 0),
                "recent_availability_ratio": float(inventory_row.get("recent_availability_ratio", float("nan"))),
                "missing_rate_in_service": missing_rate_in_service,
                "missing_rate": missing_rate_in_service,
                "activity_rate_in_service": activity_rate_in_service,
                "history_group": inventory_row.get("history_group", "young"),
                "history_group_reason": inventory_row.get("history_group_reason", ""),
                "total_demand": total_demand,
                "avg_demand_observed": mean_value,
                "avg_demand": mean_value,
                "median_demand_observed": float(observed["target"].median()) if not observed.empty else float("nan"),
                "median_demand": float(observed["target"].median()) if not observed.empty else float("nan"),
                "std_demand_observed": std_value,
                "std_demand": std_value,
                "coefficient_of_variation": _robust_cv(mean_value, std_value, config),
                "zero_rate_observed": float((observed["target"] == 0).mean()) if not observed.empty else float("nan"),
                "zero_rate": float((observed["target"] == 0).mean()) if not observed.empty else float("nan"),
                "active_day_rate_observed": float((observed["target"] > 0).mean()) if not observed.empty else float("nan"),
                "active_day_rate": float((observed["target"] > 0).mean()) if not observed.empty else float("nan"),
                "lag1_autocorr": lag1_autocorr if temporal_reliable else float("nan"),
                "lag7_autocorr": lag7_autocorr if temporal_reliable else float("nan"),
                "weekday_effect_strength": weekday_effect if temporal_reliable else float("nan"),
                "month_effect_strength": month_effect if temporal_reliable else float("nan"),
                "weekend_ratio": weekend_ratio if temporal_reliable else float("nan"),
                "trend_slope": trend_slope if temporal_reliable else float("nan"),
                "rolling_mean_shift_30obs": early_mean_shift,
                "rolling_std_shift_30obs": std_shift,
                "outlier_rate": outlier_rate,
                "max_value": float(observed["target"].max()) if not observed.empty else float("nan"),
                "p95_value": float(observed["target"].quantile(0.95)) if not observed.empty else float("nan"),
                "correlation_with_system_excl_self": corr_ex_self if temporal_reliable else float("nan"),
                "correlation_with_system_total": corr_ex_self if temporal_reliable else float("nan"),
                "demand_share_of_system_observed_window": demand_share_observed_window,
                "demand_share_of_system": demand_share_observed_window,
                "fraction_of_demand_recent_window": fraction_recent,
                "monthly_demand_share_cv": monthly_share_cv,
                "station_contribution_stability": station_contribution_stability,
                "longest_missing_streak": int(inventory_row.get("longest_missing_streak", 0) or 0),
                "longest_active_streak": int(inventory_row.get("longest_active_streak", 0) or 0),
                "intermittent_gap_ratio": float(inventory_row.get("intermittent_gap_ratio", float("nan"))),
                "median_inter_arrival_active_days": float(inventory_row.get("median_inter_arrival_active_days", float("nan"))),
                "service_continuity_score": float(inventory_row.get("service_continuity_score", float("nan"))),
                "days_since_first_observation": int(inventory_row.get("days_since_first_observation", 0) or 0),
                "days_since_last_observation": int(inventory_row.get("days_since_last_observation", 0) or 0),
                "is_short_history": bool(inventory_row.get("is_short_history", False)),
                "is_zero_almost_always": bool(inventory_row.get("is_zero_almost_always", False)),
                "appears_active_recently": bool(inventory_row.get("appears_active_recently", False)),
                "metric_reliable_autocorr": bool(temporal_reliable and lag1_reliable and lag7_reliable),
                "metric_reliable_pattern": bool(temporal_reliable and weekday_reliable and month_reliable and weekend_reliable),
                "metric_reliable_trend": bool(temporal_reliable and trend_reliable),
                "metric_reliable_outlier": bool(outlier_reliable),
                "metric_reliable_system_correlation": bool(temporal_reliable and corr_reliable),
                "metric_reliable_rolling_shift": bool(mean_shift_reliable and std_shift_reliable),
                "metric_reliable_behavior": bool(
                    pd.notna(coverage_ratio)
                    and coverage_ratio >= config.min_behavior_coverage_ratio
                    and int(inventory_row.get("observed_days", 0) or 0) >= config.min_behavior_observed_days
                    and int(inventory_row.get("recent_observed_days", 0) or 0) >= config.min_behavior_recent_observed_days
                ),
                "metrics_limited_by_sparsity": bool(
                    pd.isna(coverage_ratio)
                    or coverage_ratio < config.min_behavior_coverage_ratio
                    or mean_value < config.low_mean_demand_threshold
                ),
                "summary_metric_population": "Observed in-service days for demand and pattern metrics; calendar service window with NaN-preserved gaps for autocorrelation.",
                "summary_missingness_policy": "Missing inside service window remains missing; no zero imputation is used for behavioral metrics.",
            }
        )

    return pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
