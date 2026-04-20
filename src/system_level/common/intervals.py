from __future__ import annotations

import numpy as np
import pandas as pd


INTERVAL_OUTPUT_COLUMNS = [
    "point_forecast",
    "lower_80",
    "upper_80",
    "lower_95",
    "upper_95",
    "interval_calibration_method",
    "interval_source_horizon",
    "interval_source_step_start",
    "interval_source_step_end",
    "interval_n_residuals",
]


def ensure_horizon_step(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return forecasts.copy()
    enriched = forecasts.copy()
    if "horizon_step" in enriched.columns:
        return enriched
    group_keys = ["model_name", "horizon"]
    if "fold_id" in enriched.columns:
        group_keys = ["model_name", "horizon", "fold_id"]
    enriched = enriched.sort_values(group_keys + ["date"]).reset_index(drop=True)
    enriched["horizon_step"] = enriched.groupby(group_keys).cumcount() + 1
    return enriched


def collect_backtest_residuals(backtest_forecasts: pd.DataFrame) -> pd.DataFrame:
    if backtest_forecasts.empty:
        return pd.DataFrame()
    residuals = ensure_horizon_step(backtest_forecasts)
    residuals = residuals.rename(columns={"date": "target_date", "prediction": "point_forecast"}).copy()
    residuals["point_forecast"] = pd.to_numeric(residuals["point_forecast"], errors="coerce")
    residuals["actual"] = pd.to_numeric(residuals["actual"], errors="coerce")
    residuals["residual"] = residuals["actual"] - residuals["point_forecast"]
    residuals["abs_error"] = (residuals["actual"] - residuals["point_forecast"]).abs()
    columns = [
        column
        for column in [
            "model_name",
            "fold_id",
            "horizon",
            "horizon_step",
            "target_date",
            "point_forecast",
            "actual",
            "residual",
            "abs_error",
            "fit_success",
            "fallback_triggered",
            "fallback_reason",
            "selected_spec",
        ]
        if column in residuals.columns
    ]
    return residuals[columns].copy()


def _pool_residuals_for_step(
    residuals: pd.DataFrame,
    horizon: int,
    horizon_step: int,
    min_samples: int,
) -> tuple[pd.DataFrame, str, int, int]:
    exact = residuals.loc[residuals["horizon_step"] == horizon_step].copy()
    if len(exact) >= min_samples:
        return exact, "step_specific", horizon_step, horizon_step

    for radius in range(1, horizon):
        start = max(1, horizon_step - radius)
        end = min(horizon, horizon_step + radius)
        pooled = residuals.loc[residuals["horizon_step"].between(start, end)].copy()
        if len(pooled) >= min_samples:
            return pooled, "pooled_nearby_steps", start, end

    return residuals.copy(), "pooled_full_horizon", int(residuals["horizon_step"].min()), int(residuals["horizon_step"].max())


def fit_interval_calibration(backtest_residuals: pd.DataFrame, min_samples: int = 24) -> pd.DataFrame:
    if backtest_residuals.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    grouped = backtest_residuals.groupby(["model_name", "horizon"], as_index=False)
    for (model_name, horizon), group in grouped:
        ordered = group.sort_values(["horizon_step", "target_date"]).copy()
        for horizon_step in sorted(ordered["horizon_step"].unique()):
            pooled, method, source_start, source_end = _pool_residuals_for_step(ordered, int(horizon), int(horizon_step), min_samples)
            quantiles = pooled["residual"].quantile([0.025, 0.10, 0.90, 0.975]).to_dict()
            rows.append(
                {
                    "model_name": model_name,
                    "horizon": int(horizon),
                    "horizon_step": int(horizon_step),
                    "calibration_method": method,
                    "source_horizon": int(horizon),
                    "source_step_start": int(source_start),
                    "source_step_end": int(source_end),
                    "n_residuals": int(len(pooled)),
                    "residual_q025": float(quantiles.get(0.025, np.nan)),
                    "residual_q10": float(quantiles.get(0.10, np.nan)),
                    "residual_q90": float(quantiles.get(0.90, np.nan)),
                    "residual_q975": float(quantiles.get(0.975, np.nan)),
                }
            )
    return pd.DataFrame(rows).sort_values(["model_name", "horizon", "horizon_step"]).reset_index(drop=True)


def _resolve_calibration_row(row: pd.Series, calibration: pd.DataFrame) -> dict[str, object] | None:
    exact = calibration.loc[
        (calibration["model_name"] == row["model_name"])
        & (calibration["horizon"] == row["horizon"])
        & (calibration["horizon_step"] == row["horizon_step"])
    ]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    model_rows = calibration.loc[calibration["model_name"] == row["model_name"]].copy()
    if model_rows.empty:
        return None

    available_horizons = sorted(model_rows["horizon"].unique())
    source_horizon = max([value for value in available_horizons if value <= int(row["horizon"])] or available_horizons)
    source_step = min(int(row["horizon_step"]), source_horizon)
    proxy = model_rows.loc[
        (model_rows["horizon"] == source_horizon)
        & (model_rows["horizon_step"] == source_step)
    ]
    if proxy.empty:
        proxy = model_rows.loc[model_rows["horizon"] == source_horizon].sort_values("horizon_step").tail(1)
    if proxy.empty:
        return None

    resolved = proxy.iloc[0].to_dict()
    resolved["calibration_method"] = f"{resolved['calibration_method']}+proxy_from_h{source_horizon}"
    resolved["source_horizon"] = int(source_horizon)
    return resolved


def apply_calibrated_intervals(
    forecasts: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    if forecasts.empty:
        return forecasts.copy()

    intervalized = ensure_horizon_step(forecasts).copy()
    intervalized["point_forecast"] = pd.to_numeric(intervalized["prediction"], errors="coerce")
    lower_80: list[float] = []
    upper_80: list[float] = []
    lower_95: list[float] = []
    upper_95: list[float] = []
    methods: list[str | None] = []
    source_horizons: list[int | None] = []
    source_step_starts: list[int | None] = []
    source_step_ends: list[int | None] = []
    residual_counts: list[int | None] = []

    for _, row in intervalized.iterrows():
        resolved = _resolve_calibration_row(row, calibration) if not calibration.empty else None
        if resolved is None:
            lower_80.append(np.nan)
            upper_80.append(np.nan)
            lower_95.append(np.nan)
            upper_95.append(np.nan)
            methods.append(None)
            source_horizons.append(None)
            source_step_starts.append(None)
            source_step_ends.append(None)
            residual_counts.append(None)
            continue

        point_forecast = float(row["point_forecast"])
        lower_80.append(max(0.0, point_forecast + float(resolved["residual_q10"])))
        upper_80.append(point_forecast + float(resolved["residual_q90"]))
        lower_95.append(max(0.0, point_forecast + float(resolved["residual_q025"])))
        upper_95.append(point_forecast + float(resolved["residual_q975"]))
        methods.append(str(resolved["calibration_method"]))
        source_horizons.append(int(resolved["source_horizon"]))
        source_step_starts.append(int(resolved["source_step_start"]))
        source_step_ends.append(int(resolved["source_step_end"]))
        residual_counts.append(int(resolved["n_residuals"]))

    intervalized["lower_80"] = lower_80
    intervalized["upper_80"] = upper_80
    intervalized["lower_95"] = lower_95
    intervalized["upper_95"] = upper_95
    intervalized["interval_calibration_method"] = methods
    intervalized["interval_source_horizon"] = source_horizons
    intervalized["interval_source_step_start"] = source_step_starts
    intervalized["interval_source_step_end"] = source_step_ends
    intervalized["interval_n_residuals"] = residual_counts
    return intervalized


def _winkler_score(actual: pd.Series, lower: pd.Series, upper: pd.Series, alpha: float) -> pd.Series:
    width = upper - lower
    below = actual < lower
    above = actual > upper
    score = width.copy()
    score = score.where(~below, width + (2.0 / alpha) * (lower - actual))
    score = score.where(~above, width + (2.0 / alpha) * (actual - upper))
    return score


def evaluate_interval_quality(intervalized_backtests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if intervalized_backtests.empty or "actual" not in intervalized_backtests.columns:
        return pd.DataFrame(), pd.DataFrame()

    evaluated = intervalized_backtests.copy()
    evaluated["covered_80"] = (
        (evaluated["actual"] >= evaluated["lower_80"]) & (evaluated["actual"] <= evaluated["upper_80"])
    ).astype(float)
    evaluated["covered_95"] = (
        (evaluated["actual"] >= evaluated["lower_95"]) & (evaluated["actual"] <= evaluated["upper_95"])
    ).astype(float)
    evaluated["width_80"] = evaluated["upper_80"] - evaluated["lower_80"]
    evaluated["width_95"] = evaluated["upper_95"] - evaluated["lower_95"]
    evaluated["winkler_80"] = _winkler_score(evaluated["actual"], evaluated["lower_80"], evaluated["upper_80"], 0.20)
    evaluated["winkler_95"] = _winkler_score(evaluated["actual"], evaluated["lower_95"], evaluated["upper_95"], 0.05)

    summary = (
        evaluated.groupby(["model_name", "horizon"], as_index=False)
        .agg(
            n=("actual", "size"),
            coverage_80=("covered_80", "mean"),
            coverage_95=("covered_95", "mean"),
            avg_width_80=("width_80", "mean"),
            avg_width_95=("width_95", "mean"),
            mean_winkler_80=("winkler_80", "mean"),
            mean_winkler_95=("winkler_95", "mean"),
        )
        .sort_values(["horizon", "model_name"])
        .reset_index(drop=True)
    )

    by_step = (
        evaluated.groupby(["model_name", "horizon", "horizon_step"], as_index=False)
        .agg(
            n=("actual", "size"),
            coverage_80=("covered_80", "mean"),
            coverage_95=("covered_95", "mean"),
            avg_width_80=("width_80", "mean"),
            avg_width_95=("width_95", "mean"),
        )
        .sort_values(["model_name", "horizon", "horizon_step"])
        .reset_index(drop=True)
    )

    return summary, by_step
