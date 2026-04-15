from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def summarize_backtest_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return (
        metrics.groupby(["model_name", "horizon"], as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mase=("mase", "mean"),
            mean_bias=("bias", "mean"),
            folds=("fold_id", "nunique"),
            mean_rows=("rows", "mean"),
            mean_stations=("stations", "mean"),
        )
        .sort_values(["horizon", "mean_mase", "mean_mae"])
        .reset_index(drop=True)
    )


def build_slice_metrics(scored_forecasts: pd.DataFrame) -> pd.DataFrame:
    if scored_forecasts.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []

    all_slice = scored_forecasts.copy()
    all_slice["slice_type"] = "all"
    all_slice["slice_value"] = "all_stations"
    rows.append(all_slice)

    if "history_group" in scored_forecasts.columns:
        mature = scored_forecasts.loc[scored_forecasts["history_group"] == "mature"].copy()
        mature["slice_type"] = "history_group"
        mature["slice_value"] = "mature"
        rows.append(mature)

        short_history = scored_forecasts.loc[scored_forecasts["history_group"].isin(["newborn", "young"])].copy()
        short_history["slice_type"] = "history_group"
        short_history["slice_value"] = "short_history"
        rows.append(short_history)

    if "station_category" in scored_forecasts.columns:
        sparse = scored_forecasts.loc[scored_forecasts["station_category"] == "sparse_intermittent"].copy()
        sparse["slice_type"] = "station_group"
        sparse["slice_value"] = "sparse_intermittent"
        rows.append(sparse)

        category = scored_forecasts.copy()
        category["slice_type"] = "category"
        category["slice_value"] = category["station_category"].fillna("unknown")
        rows.append(category)

    if "cluster_label" in scored_forecasts.columns:
        cluster = scored_forecasts.copy()
        cluster["slice_type"] = "cluster"
        cluster["slice_value"] = cluster["cluster_label"].fillna("unknown")
        rows.append(cluster)

    combined = pd.concat([frame for frame in rows if not frame.empty], ignore_index=True)
    interval_columns = {
        "coverage_80": ("covered_80", "mean"),
        "avg_width_80": ("width_80", "mean"),
    } if {"covered_80", "width_80"}.issubset(combined.columns) else {}

    return (
        combined.groupby(["model_name", "horizon", "slice_type", "slice_value"], as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda values: float(values.mean() ** 0.5)),
            mase=("scaled_abs_error", "mean"),
            bias=("bias_error", "mean"),
            rows=("actual", "size"),
            stations=("station_id", "nunique"),
            **interval_columns,
        )
        .sort_values(["horizon", "slice_type", "slice_value", "mae"])
        .reset_index(drop=True)
    )


def plot_model_comparison(summary: pd.DataFrame, path: Path) -> Path | None:
    if summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = summary.pivot(index="model_name", columns="horizon", values="mean_mase").sort_index()
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Station-level backtest comparison (MASE)")
    ax.set_ylabel("Mean MASE")
    ax.set_xlabel("Model")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
