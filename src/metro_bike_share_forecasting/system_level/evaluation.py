from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from metro_bike_share_forecasting.system_level.config import SystemLevelConfig


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
