from __future__ import annotations

import numpy as np
import pandas as pd

from metro_bike_share_forecasting.station_level.diagnosis.config import StationDiagnosisConfig


def classify_history_group(history_days: int, config: StationDiagnosisConfig) -> str:
    if history_days < config.newborn_history_days:
        return "newborn"
    if history_days < config.mature_history_days:
        return "young"
    return "mature"


def build_complete_station_grid(station_frame: pd.DataFrame) -> pd.DataFrame:
    observed = station_frame.sort_values("date").copy()
    full_dates = pd.date_range(observed["date"].min(), observed["date"].max(), freq="D")
    grid = pd.DataFrame({"date": full_dates}).merge(observed[["date", "target"]], on="date", how="left")
    grid["missing_flag"] = grid["target"].isna().astype(int)
    grid["target"] = grid["target"].fillna(0.0)
    return grid


def _safe_ratio(numerator: float, denominator: float, min_denominator: float = 0.0) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or abs(float(denominator)) <= min_denominator:
        return float("nan")
    return float(numerator) / float(denominator)


def _safe_autocorr(values: pd.Series, lag: int, min_history: int) -> float:
    series = values.astype(float)
    if len(series) < max(lag + 2, min_history) or float(series.std(ddof=0)) == 0.0:
        return float("nan")
    return float(series.autocorr(lag=lag))


def _effect_strength(
    target: pd.Series,
    groups: pd.Series,
    *,
    min_active_days: int,
    min_mean_demand: float,
) -> float:
    series = target.astype(float)
    if int((series > 0).sum()) < min_active_days or float(series.mean()) < min_mean_demand:
        return float("nan")
    frame = pd.DataFrame({"target": series, "group": groups})
    overall = float(frame["target"].mean())
    if overall == 0.0:
        return float("nan")
    grouped_means = frame.groupby("group")["target"].mean()
    if grouped_means.empty or len(grouped_means) < 2:
        return float("nan")
    return float(grouped_means.std(ddof=0) / abs(overall))


def _weekend_ratio(target: pd.Series, weekday: pd.Series, config: StationDiagnosisConfig) -> float:
    series = target.astype(float)
    if int((series > 0).sum()) < config.min_active_days_for_pattern_metrics or float(series.mean()) < config.low_mean_demand_threshold:
        return float("nan")
    weekend_mask = weekday.isin([5, 6])
    weekday_mean = float(series.loc[~weekend_mask].mean()) if (~weekend_mask).any() else float("nan")
    weekend_mean = float(series.loc[weekend_mask].mean()) if weekend_mask.any() else float("nan")
    return _safe_ratio(weekend_mean, weekday_mean, min_denominator=config.low_mean_demand_threshold / 2.0)


def _robust_cv(mean_value: float, std_value: float, config: StationDiagnosisConfig) -> float:
    if not np.isfinite(mean_value) or mean_value < config.low_mean_demand_threshold:
        return float("nan")
    cv = std_value / mean_value if mean_value else float("nan")
    if not np.isfinite(cv):
        return float("nan")
    return float(min(cv, config.max_coefficient_of_variation))


def _trend_slope(target: pd.Series, config: StationDiagnosisConfig) -> float:
    if len(target) < config.min_history_days_for_trend:
        return float("nan")
    x = np.arange(len(target), dtype=float)
    y = target.astype(float).to_numpy()
    if not np.isfinite(y).all() or float(np.std(y, ddof=0)) == 0.0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _rolling_shift(target: pd.Series, window: int, reducer: str) -> float:
    if len(target) < window * 2:
        return float("nan")
    first = target.iloc[:window]
    last = target.iloc[-window:]
    if reducer == "mean":
        return float(last.mean() - first.mean())
    return float(last.std(ddof=0) - first.std(ddof=0))


def _outlier_rate(target: pd.Series, config: StationDiagnosisConfig) -> float:
    series = target.astype(float)
    positives = series.loc[series > 0]
    if len(positives) < config.min_active_days_for_outlier_metrics:
        return float("nan")
    median = float(positives.median())
    mad = float((positives - median).abs().median())
    if mad == 0.0:
        std = float(positives.std(ddof=0))
        if std == 0.0:
            return 0.0
        z_scores = (positives - float(positives.mean())) / std
    else:
        z_scores = 0.6745 * (positives - median) / mad
    return float((z_scores.abs() > config.outlier_z_threshold).mean())


def build_station_inventory(station_daily: pd.DataFrame, config: StationDiagnosisConfig) -> pd.DataFrame:
    """Build a station inventory table that validates the apparent station universe."""

    if station_daily.empty:
        return pd.DataFrame()

    global_end = pd.to_datetime(station_daily["date"]).max()
    recent_start = global_end - pd.Timedelta(days=config.recent_activity_window_days - 1)

    rows: list[dict[str, object]] = []
    for station_id, station_frame in station_daily.groupby("station_id", sort=True):
        observed = station_frame.sort_values("date").drop_duplicates(subset=["date"], keep="last").copy()
        grid = build_complete_station_grid(observed)
        history_days = int(len(grid))
        n_observed_days = int(observed["date"].nunique())
        n_missing_days = int(history_days - n_observed_days)
        zero_rate = float((grid["target"] == 0).mean())
        recent_active_days = int(grid.loc[grid["date"] >= recent_start, "target"].gt(0).sum())

        rows.append(
            {
                "station_id": str(station_id),
                "first_date": observed["date"].min(),
                "last_date": observed["date"].max(),
                "n_observed_days": n_observed_days,
                "n_missing_days": n_missing_days,
                "history_days": history_days,
                "total_demand": float(grid["target"].sum()),
                "history_group": classify_history_group(history_days, config),
                "is_short_history": bool(history_days < config.mature_history_days),
                "is_zero_almost_always": bool(zero_rate >= config.zero_almost_always_threshold),
                "appears_active_recently": bool(recent_active_days >= config.min_recent_active_days),
            }
        )

    return pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)


def build_station_summary_table(
    station_daily: pd.DataFrame,
    inventory: pd.DataFrame,
    config: StationDiagnosisConfig,
) -> pd.DataFrame:
    """Build a one-row-per-station summary table for robust diagnosis."""

    if station_daily.empty:
        return pd.DataFrame()

    inventory_lookup = inventory.set_index("station_id") if not inventory.empty else pd.DataFrame()
    grouped = station_daily.groupby("station_id", sort=True)
    system_total = station_daily.groupby("date", as_index=False)["target"].sum().rename(columns={"target": "system_total"})
    system_total_sum = float(system_total["system_total"].sum())

    rows: list[dict[str, object]] = []
    for station_id, station_frame in grouped:
        observed = station_frame.sort_values("date").drop_duplicates(subset=["date"], keep="last").copy()
        grid = build_complete_station_grid(observed)
        target = grid["target"].astype(float)
        weekday = grid["date"].dt.dayofweek
        month = grid["date"].dt.month

        system_aligned = grid[["date"]].merge(system_total, on="date", how="left")["system_total"].fillna(0.0)
        if len(target) < 2 or float(target.std(ddof=0)) == 0.0 or float(system_aligned.std(ddof=0)) == 0.0:
            correlation = float("nan")
        else:
            correlation = float(pd.Series(target).corr(system_aligned))

        mean_value = float(target.mean())
        std_value = float(target.std(ddof=0))
        total_demand = float(target.sum())
        inventory_row = inventory_lookup.loc[str(station_id)] if not inventory.empty else {}

        rows.append(
            {
                "station_id": str(station_id),
                "start_date": observed["date"].min(),
                "end_date": observed["date"].max(),
                "n_days": int(observed["date"].nunique()),
                "history_days": int(inventory_row["history_days"]) if inventory_row is not None else int(len(grid)),
                "missing_rate": float(grid["missing_flag"].mean()),
                "history_group": inventory_row["history_group"] if inventory_row is not None else classify_history_group(len(grid), config),
                "total_demand": total_demand,
                "avg_demand": mean_value,
                "median_demand": float(target.median()),
                "std_demand": std_value,
                "coefficient_of_variation": _robust_cv(mean_value, std_value, config),
                "zero_rate": float((target == 0).mean()),
                "active_day_rate": float((target > 0).mean()),
                "lag1_autocorr": _safe_autocorr(target, 1, config.min_history_for_autocorr),
                "lag7_autocorr": _safe_autocorr(target, 7, config.min_history_for_autocorr),
                "weekday_effect_strength": _effect_strength(
                    target,
                    weekday,
                    min_active_days=config.min_active_days_for_pattern_metrics,
                    min_mean_demand=config.low_mean_demand_threshold,
                ),
                "month_effect_strength": _effect_strength(
                    target,
                    month,
                    min_active_days=config.min_active_days_for_pattern_metrics,
                    min_mean_demand=config.low_mean_demand_threshold,
                ),
                "weekend_ratio": _weekend_ratio(target, weekday, config),
                "trend_slope": _trend_slope(target, config),
                "rolling_mean_shift_30d": _rolling_shift(target, config.rolling_window_days, "mean"),
                "rolling_std_shift_30d": _rolling_shift(target, config.rolling_window_days, "std"),
                "outlier_rate": _outlier_rate(target, config),
                "max_value": float(target.max()),
                "p95_value": float(target.quantile(0.95)),
                "correlation_with_system_total": correlation,
                "demand_share_of_system": _safe_ratio(total_demand, system_total_sum, min_denominator=config.low_mean_demand_threshold),
                "is_short_history": bool(inventory_row["is_short_history"]) if inventory_row is not None else bool(len(grid) < config.mature_history_days),
                "is_zero_almost_always": bool(inventory_row["is_zero_almost_always"]) if inventory_row is not None else bool((target == 0).mean() >= config.zero_almost_always_threshold),
                "appears_active_recently": bool(inventory_row["appears_active_recently"]) if inventory_row is not None else True,
                "metrics_limited_by_sparsity": bool(mean_value < config.low_mean_demand_threshold or float((target > 0).mean()) <= config.sparse_active_day_rate_threshold),
            }
        )

    return pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
