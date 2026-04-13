from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from metro_bike_share_forecasting.system_level.config import SystemLevelConfig
from metro_bike_share_forecasting.system_level.models import MODEL_DIAGNOSTIC_COLUMNS


def summarize_backtest_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    summary = (
        metrics.groupby(["model_name", "horizon"], as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mase=("mase", "mean"),
            mean_bias=("bias", "mean"),
            folds=("fold_id", "nunique"),
        )
        .sort_values(["horizon", "mean_mase", "mean_mae", "mean_rmse"])
        .reset_index(drop=True)
    )
    return summary


def build_recommendation_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return (
        summary.sort_values(["horizon", "mean_mase", "mean_mae"])
        .groupby("horizon", as_index=False)
        .first()
        .rename(columns={"model_name": "recommended_model"})
    )


def build_fit_diagnostics_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame()
    columns = [column for column in ["model_name", "horizon", *MODEL_DIAGNOSTIC_COLUMNS] if column in forecasts.columns]
    if not columns:
        return pd.DataFrame()
    return forecasts[columns].drop_duplicates().reset_index(drop=True)


def plot_model_comparison(summary: pd.DataFrame, path: Path) -> Path | None:
    if summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = summary.pivot(index="model_name", columns="horizon", values="mean_mase").sort_index()
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("System-level backtest comparison (MASE)")
    ax.set_ylabel("Mean MASE")
    ax.set_xlabel("Model")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_production_forecasts(forecasts: pd.DataFrame, path: Path) -> Path | None:
    if forecasts.empty:
        return None
    horizons = sorted(forecasts["horizon"].unique().tolist())
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 4 * len(horizons)), squeeze=False)
    for ax, horizon in zip(axes.flatten(), horizons):
        subset = forecasts.loc[forecasts["horizon"] == horizon].sort_values(["model_name", "date"])
        for model_name, model_frame in subset.groupby("model_name"):
            ax.plot(model_frame["date"], model_frame["prediction"], marker="o", linewidth=1.5, label=model_name)
        ax.set_title(f"System-level production forecasts (h={horizon})")
        ax.set_ylabel("Forecast")
        ax.legend(loc="upper left", fontsize=8)
    axes.flatten()[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_system_level_summary(
    config: SystemLevelConfig,
    recommendation_table: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    lines = [
        "# System-Level Forecasting Summary",
        "",
        "## Problem",
        "This module forecasts the daily total demand across the full network. It is built for aggregate planning, network-wide monitoring, and system-level resource planning.",
        "",
        "## Contract Summary",
        f"- Frequency: {config.frequency}",
        f"- Main horizons: {', '.join(str(value) for value in config.forecast_horizons)} days",
        f"- Extended sensitivity horizon: {config.extended_horizon} days",
        "- Scope: aggregated system-level total only",
        "- Constraint: forecasts should remain nonnegative",
        "",
        "## Diagnostic Shortlist",
        "- The aggregate series is persistent and clearly forecastable, but it is not stable around one fixed mean.",
        "- Weekly effects are strong enough that seasonal baselines matter.",
        "- The daily total shows changing variance and visible anomalies, so evaluation must stay time-aware.",
        "- Calendar and holiday signals are likely useful because the aggregate series reflects broad network behavior.",
        "",
        "## Model Shortlist",
    ]
    for _, row in recommendation_table.iterrows():
        lines.append(
            f"- Horizon {int(row['horizon'])}: recommend `{row['recommended_model']}` first, based on mean MASE {row['mean_mase']:.3f}."
        )

    lines.extend(
        [
            "",
            "## Metric Summary",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['model_name']}` at horizon {int(row['horizon'])}: "
            f"MAE={row['mean_mae']:.2f}, RMSE={row['mean_rmse']:.2f}, MASE={row['mean_mase']:.3f}, Bias={row['mean_bias']:.2f}."
        )

    lines.extend(
        [
            "",
            "## Recommended Next Step",
            "Promote the strongest baseline and the strongest non-baseline model into a deeper review, then decide whether richer external drivers materially improve the aggregate forecast.",
        ]
    )
    output_path.write_text("\n".join(lines))
    return output_path


def write_sarimax_review(
    summary: pd.DataFrame,
    fit_diagnostics: pd.DataFrame,
    output_path: Path,
) -> Path:
    sarimax_summary = summary.loc[summary["model_name"] == "sarimax_dynamic"].copy()
    sarimax_fit = fit_diagnostics.loc[fit_diagnostics["model_name"] == "sarimax_dynamic"].copy() if not fit_diagnostics.empty else pd.DataFrame()

    lines = [
        "# SARIMAX Review",
        "",
        "## What Was Wrong",
        "- The earlier SARIMAX path used one fixed specification and minimal exogenous preprocessing.",
        "- That made the model brittle for a nonstationary daily series with weekly seasonality and changing variance.",
        "",
        "## What Changed",
        "- Added a small conservative SARIMAX candidate search.",
        "- Added exogenous cleanup: constant-column drop, train-only scaling, and collinearity pruning.",
        "- Added forecast guardrails so unstable paths are rejected instead of silently accepted.",
        "",
    ]

    if sarimax_summary.empty:
        lines.extend(
            [
                "## Current Result",
                "SARIMAX did not produce a scored result in the latest run.",
            ]
        )
    else:
        lines.extend(["## Current Result"])
        for _, row in sarimax_summary.iterrows():
            lines.append(
                f"- Horizon {int(row['horizon'])}: MAE={row['mean_mae']:.2f}, RMSE={row['mean_rmse']:.2f}, "
                f"MASE={row['mean_mase']:.3f}, Bias={row['mean_bias']:.2f}."
            )

    if not sarimax_fit.empty:
        lines.extend(["", "## Latest Fit Diagnostics"])
        sample = sarimax_fit.sort_values(["horizon"]).drop_duplicates(subset=["horizon"], keep="last")
        for _, row in sample.iterrows():
            lines.append(
                f"- Horizon {int(row['horizon'])}: fit_success={row.get('fit_success')}, "
                f"fallback_triggered={row.get('fallback_triggered')}, selected_spec={row.get('selected_spec')}, "
                f"n_exog={row.get('n_exog')}, warning_count={row.get('warning_count')}."
            )

    lines.extend(
        [
            "",
            "## Recommendation",
            "Keep SARIMAX as a candidate model only if it remains stable and materially improves h=7 or h=30. If it continues to lose to ETS or Fourier regression, keep it enabled for review but do not promote it as the preferred production model.",
        ]
    )
    output_path.write_text("\n".join(lines))
    return output_path


def write_interval_summary_report(
    config: SystemLevelConfig,
    point_summary: pd.DataFrame,
    interval_summary: pd.DataFrame,
    interval_sample: pd.DataFrame,
    output_path: Path,
) -> Path:
    preferred_models = {
        7: "ets",
        30: "fourier_dynamic_regression",
    }
    lines = [
        "# System-Level Interval Summary",
        "",
        "## Approach",
        "- Point models were kept intact and intervals were added on top.",
        "- Calibration uses only rolling backtest residuals, never in-sample residuals.",
        "- Residual quantiles are asymmetric: 10/90 for 80% and 2.5/97.5 for 95%.",
        f"- Main validated horizons remain {', '.join(str(value) for value in config.forecast_horizons)} days.",
        f"- The {config.extended_horizon}-day horizon is still sensitivity-only and uses proxy calibration when needed.",
        "",
        "## Point Forecast Baseline",
    ]
    for horizon, model_name in preferred_models.items():
        subset = point_summary.loc[(point_summary["horizon"] == horizon) & (point_summary["model_name"] == model_name)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.append(
            f"- `{model_name}` at h={horizon}: MAE={row['mean_mae']:.2f}, RMSE={row['mean_rmse']:.2f}, "
            f"MASE={row['mean_mase']:.3f}, Bias={row['mean_bias']:.2f}."
        )

    lines.extend(["", "## Interval Coverage"])
    for horizon, model_name in preferred_models.items():
        subset = interval_summary.loc[(interval_summary["horizon"] == horizon) & (interval_summary["model_name"] == model_name)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.append(
            f"- `{model_name}` at h={horizon}: coverage_80={row['coverage_80']:.3f}, coverage_95={row['coverage_95']:.3f}, "
            f"avg_width_80={row['avg_width_80']:.2f}, avg_width_95={row['avg_width_95']:.2f}."
        )

    lines.extend(["", "## Sample Output"])
    if interval_sample.empty:
        lines.append("No interval sample rows were available in the latest run.")
    else:
        sample_columns = [
            "date",
            "horizon",
            "horizon_step",
            "model_name",
            "point_forecast",
            "lower_80",
            "upper_80",
            "lower_95",
            "upper_95",
        ]
        sample = interval_sample[sample_columns].copy()
        sample["date"] = pd.to_datetime(sample["date"]).dt.strftime("%Y-%m-%d")
        lines.append(_frame_to_markdown_table(sample.head(10)))

    lines.extend(
        [
            "",
            "## Notes",
            "- Lower bounds are clipped at zero to respect the nonnegative demand constraint.",
            "- If a horizon-step cell has too few residuals, calibration falls back to pooled nearby steps and logs that choice.",
            "- These intervals are practical empirical intervals, not full distributional forecasts.",
        ]
    )
    output_path.write_text("\n".join(lines))
    return output_path


def _frame_to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.2f}")
    header = "| " + " | ".join(str(column) for column in display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.astype(object).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])
