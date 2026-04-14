from __future__ import annotations

import numpy as np
import pandas as pd

from metro_bike_share_forecasting.station_level.diagnosis.config import StationDiagnosisConfig


def _safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _safe_autocorr(values: pd.Series, lag: int, min_history: int) -> float:
    series = values.astype(float)
    if len(series) < max(lag + 2, min_history) or float(series.std(ddof=0)) == 0.0:
        return float("nan")
    return float(series.autocorr(lag=lag))


def _effect_strength(target: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"target": target.astype(float), "group": groups})
    overall = float(frame["target"].mean())
    if overall == 0.0:
        return float("nan")
    grouped_means = frame.groupby("group")["target"].mean()
    if grouped_means.empty:
        return float("nan")
    return float(grouped_means.std(ddof=0) / abs(overall))


def _trend_slope(target: pd.Series) -> float:
    if len(target) < 2:
        return float("nan")
    x = np.arange(len(target), dtype=float)
    y = target.astype(float).to_numpy()
    return float(np.polyfit(x, y, 1)[0])


def _rolling_shift(target: pd.Series, window: int, reducer: str) -> float:
    if len(target) < window * 2:
        return float("nan")
    first = target.iloc[:window]
    last = target.iloc[-window:]
    if reducer == "mean":
        return float(last.mean() - first.mean())
    return float(last.std(ddof=0) - first.std(ddof=0))


def _outlier_rate(target: pd.Series, threshold: float) -> float:
    series = target.astype(float)
    median = float(series.median())
    mad = float((series - median).abs().median())
    if mad == 0.0:
        std = float(series.std(ddof=0))
        if std == 0.0:
            return 0.0
        z_scores = (series - float(series.mean())) / std
    else:
        z_scores = 0.6745 * (series - median) / mad
    return float((z_scores.abs() > threshold).mean())


def _build_station_grid(station_frame: pd.DataFrame) -> pd.DataFrame:
    observed = station_frame.sort_values("date").copy()
    full_dates = pd.date_range(observed["date"].min(), observed["date"].max(), freq="D")
    grid = pd.DataFrame({"date": full_dates}).merge(observed[["date", "target"]], on="date", how="left")
    grid["missing_flag"] = grid["target"].isna().astype(int)
    grid["target"] = grid["target"].fillna(0.0)
    return grid


def build_station_summary_table(station_daily: pd.DataFrame, config: StationDiagnosisConfig) -> pd.DataFrame:
    """Build a one-row-per-station summary table for diagnosis."""

    if station_daily.empty:
        return pd.DataFrame()

    grouped = station_daily.groupby("station_id", sort=True)
    system_total = station_daily.groupby("date", as_index=False)["target"].sum().rename(columns={"target": "system_total"})
    system_total_sum = float(system_total["system_total"].sum())

    rows: list[dict[str, object]] = []
    for station_id, station_frame in grouped:
        observed = station_frame.sort_values("date").copy()
        grid = _build_station_grid(observed)
        target = grid["target"].astype(float)
        weekday = grid["date"].dt.dayofweek
        month = grid["date"].dt.month

        weekend_mask = weekday.isin([5, 6])
        weekday_mean = float(target.loc[~weekend_mask].mean()) if (~weekend_mask).any() else float("nan")
        weekend_mean = float(target.loc[weekend_mask].mean()) if weekend_mask.any() else float("nan")

        system_aligned = grid[["date"]].merge(system_total, on="date", how="left")["system_total"].fillna(0.0)
        if len(target) < 2 or float(target.std(ddof=0)) == 0.0 or float(system_aligned.std(ddof=0)) == 0.0:
            correlation = float("nan")
        else:
            correlation = float(pd.Series(target).corr(system_aligned))

        mean_value = float(target.mean())
        std_value = float(target.std(ddof=0))
        total_demand = float(target.sum())

        rows.append(
            {
                "station_id": station_id,
                "start_date": observed["date"].min(),
                "end_date": observed["date"].max(),
                "n_days": int(len(grid)),
                "missing_rate": float(grid["missing_flag"].mean()),
                "total_demand": total_demand,
                "avg_demand": mean_value,
                "median_demand": float(target.median()),
                "std_demand": std_value,
                "coefficient_of_variation": _safe_ratio(std_value, mean_value),
                "zero_rate": float((target == 0).mean()),
                "active_day_rate": float((target > 0).mean()),
                "lag1_autocorr": _safe_autocorr(target, 1, config.min_history_for_autocorr),
                "lag7_autocorr": _safe_autocorr(target, 7, config.min_history_for_autocorr),
                "weekday_effect_strength": _effect_strength(target, weekday),
                "month_effect_strength": _effect_strength(target, month),
                "weekend_ratio": _safe_ratio(weekend_mean, weekday_mean),
                "trend_slope": _trend_slope(target),
                "rolling_mean_shift_30d": _rolling_shift(target, config.rolling_window_days, "mean"),
                "rolling_std_shift_30d": _rolling_shift(target, config.rolling_window_days, "std"),
                "outlier_rate": _outlier_rate(target, config.outlier_z_threshold),
                "max_value": float(target.max()),
                "p95_value": float(target.quantile(0.95)),
                "correlation_with_system_total": correlation,
                "demand_share_of_system": _safe_ratio(total_demand, system_total_sum),
            }
        )

    return pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
