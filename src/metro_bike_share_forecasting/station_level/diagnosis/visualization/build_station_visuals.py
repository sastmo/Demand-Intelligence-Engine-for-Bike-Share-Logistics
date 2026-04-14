from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metro_bike_share_forecasting.station_level.diagnosis.config import StationDiagnosisConfig
from metro_bike_share_forecasting.station_level.diagnosis.features.summary_features import build_complete_station_grid


def _write_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _value_counts_bar(series: pd.Series, title: str, xlabel: str, path: Path) -> Path:
    counts = series.fillna("missing").value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index.astype(str), counts.values, color="#3b6ea5")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Station count")
    ax.tick_params(axis="x", rotation=30)
    return _write_figure(fig, path)


def _histogram(series: pd.Series, title: str, xlabel: str, path: Path) -> Path | None:
    data = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=min(30, max(10, int(math.sqrt(len(data))))), color="#6baed6", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    return _write_figure(fig, path)


def _scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    color_by: str,
    title: str,
    path: Path,
) -> Path | None:
    plot_frame = frame[[x, y, color_by]].replace([np.inf, -np.inf], np.nan).dropna()
    if plot_frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = plot_frame[color_by].astype(str).unique().tolist()
    cmap = plt.get_cmap("tab20", max(len(categories), 1))
    for index, category in enumerate(categories):
        subset = plot_frame.loc[plot_frame[color_by].astype(str) == category]
        ax.scatter(subset[x], subset[y], s=30, alpha=0.75, color=cmap(index), label=category)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(loc="best", fontsize=8)
    return _write_figure(fig, path)


def _boxplot(frame: pd.DataFrame, by: str, value: str, title: str, path: Path) -> Path | None:
    plot_frame = frame[[by, value]].replace([np.inf, -np.inf], np.nan).dropna()
    if plot_frame.empty:
        return None
    groups = [subset[value].to_numpy() for _, subset in plot_frame.groupby(by)]
    labels = [str(label) for label, _ in plot_frame.groupby(by)]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(groups, tick_labels=labels, vert=True, patch_artist=True)
    ax.set_title(title)
    ax.set_xlabel(by)
    ax.set_ylabel(value)
    ax.tick_params(axis="x", rotation=30)
    return _write_figure(fig, path)


def _cluster_profile_heatmap(cluster_profile: pd.DataFrame, path: Path) -> Path | None:
    if cluster_profile.empty:
        return None
    metric_columns = [column for column in cluster_profile.columns if column.endswith("_mean")]
    if not metric_columns:
        return None
    matrix = cluster_profile.set_index("cluster_label")[metric_columns].copy()
    matrix = matrix.apply(lambda column: (column - column.mean()) / (column.std(ddof=0) or 1.0), axis=0).fillna(0.0)
    fig, ax = plt.subplots(figsize=(max(8, len(metric_columns) * 0.7), max(4, len(matrix) * 0.6)))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(metric_columns)))
    ax.set_xticklabels(metric_columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_title("Normalized cluster profile heatmap")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.04)
    return _write_figure(fig, path)


def _select_representative_stations(summary: pd.DataFrame, config: StationDiagnosisConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for category, subset in summary.groupby("station_category"):
        if len(subset) < config.representative_min_category_size:
            continue
        category_medians = subset[["avg_demand", "zero_rate"]].median()
        scored = subset.assign(
            _representative_distance=(subset["avg_demand"] - category_medians["avg_demand"]).abs()
            + (subset["zero_rate"] - category_medians["zero_rate"]).abs()
        )
        rows.append(scored.sort_values(["_representative_distance", "station_id"]).head(config.representative_samples_per_category))
    if not rows:
        return pd.DataFrame(columns=summary.columns)
    representative = pd.concat(rows, ignore_index=True)
    return representative.drop(columns=["_representative_distance"], errors="ignore")


def _sample_station_grid_map(station_daily: pd.DataFrame, sampled_stations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if sampled_stations.empty:
        return {}
    grids: dict[str, pd.DataFrame] = {}
    for station_id in sampled_stations["station_id"].tolist():
        observed = station_daily.loc[station_daily["station_id"] == station_id, ["date", "target"]].copy()
        if observed.empty:
            continue
        grids[str(station_id)] = build_complete_station_grid(observed)
    return grids


def _representative_time_series(
    station_daily: pd.DataFrame,
    sampled_stations: pd.DataFrame,
    path: Path,
) -> Path | None:
    grid_map = _sample_station_grid_map(station_daily, sampled_stations)
    if not grid_map:
        return None
    stations = sampled_stations[["station_id", "station_category"]].drop_duplicates().reset_index(drop=True)
    n = len(stations)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")
    for ax, (_, row) in zip(axes.flatten(), stations.iterrows()):
        ax.axis("on")
        grid = grid_map.get(str(row["station_id"]))
        if grid is None:
            continue
        ax.plot(grid["date"], grid["target"], color="#3b6ea5", linewidth=1.2)
        ax.set_title(f"{row['station_category']}: {row['station_id']}", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("Representative station time-series samples", fontsize=14)
    return _write_figure(fig, path)


def _profile_figure(
    station_daily: pd.DataFrame,
    sampled_stations: pd.DataFrame,
    by: str,
    title: str,
    xlabel: str,
    path: Path,
) -> Path | None:
    if sampled_stations.empty:
        return None
    categories = sampled_stations["station_category"].dropna().unique().tolist()
    if not categories:
        return None
    fig, axes = plt.subplots(len(categories), 1, figsize=(9, max(4, len(categories) * 3)), squeeze=False)
    for ax, category in zip(axes.flatten(), categories):
        sample_ids = sampled_stations.loc[sampled_stations["station_category"] == category, "station_id"].tolist()
        for station_id in sample_ids:
            observed = station_daily.loc[station_daily["station_id"] == station_id, ["date", "target"]].copy()
            if observed.empty:
                continue
            grid = build_complete_station_grid(observed)
            if by == "weekday":
                profile = grid.groupby(grid["date"].dt.dayofweek)["target"].mean().reindex(range(7))
            else:
                profile = grid.groupby(grid["date"].dt.month)["target"].mean().reindex(range(1, 13))
            ax.plot(profile.index, profile.values, marker="o", linewidth=1.2, label=str(station_id))
        ax.set_title(str(category))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean demand")
        ax.legend(loc="best", fontsize=7)
    fig.suptitle(title, fontsize=14)
    return _write_figure(fig, path)


def build_station_visuals(
    station_daily: pd.DataFrame,
    summary_with_clusters: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    config: StationDiagnosisConfig,
    output_dir: Path,
) -> dict[str, str]:
    """Build aggregate and representative station-level diagnosis visuals."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    plot_specs = [
        ("category_count_bar_chart", _value_counts_bar(summary_with_clusters["station_category"], "Station categories", "station_category", output_dir / "category_counts.png")),
        ("cluster_count_bar_chart", _value_counts_bar(summary_with_clusters["cluster_label"], "Cluster labels", "cluster_label", output_dir / "cluster_counts.png")),
    ]
    for label, path in plot_specs:
        written[label] = str(path)

    histogram_specs = [
        ("history_days_histogram", "history_days", "History days distribution"),
        ("zero_rate_histogram", "zero_rate", "Zero-rate distribution"),
        ("avg_demand_histogram", "avg_demand", "Average demand distribution"),
        ("coefficient_of_variation_histogram", "coefficient_of_variation", "Coefficient of variation distribution"),
    ]
    for label, column, title in histogram_specs:
        path = _histogram(summary_with_clusters[column], title, column, output_dir / f"{label}.png")
        if path is not None:
            written[label] = str(path)

    scatter_specs = [
        ("avg_demand_vs_zero_rate_by_category", "avg_demand", "zero_rate", "station_category", "Average demand vs zero rate by category"),
        ("avg_demand_vs_cv_by_cluster", "avg_demand", "coefficient_of_variation", "cluster_label", "Average demand vs coefficient of variation by cluster"),
        ("lag7_vs_weekday_effect_by_category", "lag7_autocorr", "weekday_effect_strength", "station_category", "Lag-7 autocorrelation vs weekday effect by category"),
    ]
    for label, x, y, color_by, title in scatter_specs:
        path = _scatter(summary_with_clusters, x, y, color_by, title, output_dir / f"{label}.png")
        if path is not None:
            written[label] = str(path)

    box_specs = [
        ("avg_demand_by_category_boxplot", "station_category", "avg_demand", "Average demand by category"),
        ("zero_rate_by_category_boxplot", "station_category", "zero_rate", "Zero rate by category"),
    ]
    for label, by, value, title in box_specs:
        path = _boxplot(summary_with_clusters, by, value, title, output_dir / f"{label}.png")
        if path is not None:
            written[label] = str(path)

    heatmap_path = _cluster_profile_heatmap(cluster_profile, output_dir / "cluster_profile_heatmap.png")
    if heatmap_path is not None:
        written["cluster_profile_heatmap"] = str(heatmap_path)

    history_group_path = _value_counts_bar(
        summary_with_clusters["history_group"],
        "Station counts by history group",
        "history_group",
        output_dir / "history_group_counts.png",
    )
    written["history_group_counts"] = str(history_group_path)

    sampled_stations = _select_representative_stations(summary_with_clusters, config)
    sampled_time_series = _representative_time_series(station_daily, sampled_stations, output_dir / "representative_station_timeseries.png")
    if sampled_time_series is not None:
        written["representative_station_timeseries"] = str(sampled_time_series)

    weekday_profiles = _profile_figure(
        station_daily,
        sampled_stations,
        by="weekday",
        title="Representative weekday profiles by category",
        xlabel="Day of week",
        path=output_dir / "representative_weekday_profiles.png",
    )
    if weekday_profiles is not None:
        written["representative_weekday_profiles"] = str(weekday_profiles)

    monthly_profiles = _profile_figure(
        station_daily,
        sampled_stations,
        by="month",
        title="Representative monthly profiles by category",
        xlabel="Month",
        path=output_dir / "representative_monthly_profiles.png",
    )
    if monthly_profiles is not None:
        written["representative_monthly_profiles"] = str(monthly_profiles)

    return written
