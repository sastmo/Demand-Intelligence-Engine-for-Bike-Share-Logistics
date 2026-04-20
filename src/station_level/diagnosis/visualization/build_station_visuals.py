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

from station_level.diagnosis.config import StationDiagnosisConfig


def _write_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _value_counts_bar(series: pd.Series, title: str, xlabel: str, path: Path) -> Path:
    counts = series.fillna("missing").value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index.astype(str), counts.values)
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
    ax.hist(data, bins=min(30, max(10, int(math.sqrt(len(data))))), edgecolor="white")
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


def _heatmap_from_crosstab(crosstab: pd.DataFrame, title: str, path: Path) -> Path | None:
    if crosstab.empty:
        return None
    matrix = crosstab.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, len(crosstab.columns) * 1.1), max(4, len(crosstab.index) * 0.8)))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(crosstab.columns)))
    ax.set_xticklabels(crosstab.columns.astype(str), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(crosstab.index)))
    ax.set_yticklabels(crosstab.index.astype(str))
    ax.set_title(title)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, f"{matrix[row_index, col_index]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.04)
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


def _cluster_stability_plot(selection: pd.DataFrame, path: Path) -> Path | None:
    if selection.empty:
        return None
    plot_frame = selection.sort_values("candidate_k")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_frame["candidate_k"], plot_frame["silhouette_score"], marker="o", label="silhouette")
    ax.plot(plot_frame["candidate_k"], plot_frame["seed_stability_ari"], marker="o", label="seed stability")
    ax.plot(plot_frame["candidate_k"], plot_frame["bootstrap_stability_ari"], marker="o", label="bootstrap stability")
    selected = plot_frame.loc[plot_frame["selected"]]
    if not selected.empty:
        ax.axvline(float(selected["candidate_k"].iloc[0]), linestyle="--", linewidth=1.0)
    ax.set_title("Cluster selection diagnostics")
    ax.set_xlabel("candidate_k")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    return _write_figure(fig, path)


def _select_representative_stations(summary: pd.DataFrame, config: StationDiagnosisConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for category, subset in summary.groupby("behavior_label"):
        if len(subset) < config.representative_min_category_size:
            continue
        usable = subset.loc[subset["behavior_label"] != "unclassified_due_to_low_coverage"].copy()
        if usable.empty:
            usable = subset.copy()
        medians = usable[["avg_demand_observed", "coverage_ratio"]].median()
        scored = usable.assign(
            _representative_distance=(usable["avg_demand_observed"] - medians["avg_demand_observed"]).abs()
            + (usable["coverage_ratio"] - medians["coverage_ratio"]).abs()
        )
        rows.append(scored.sort_values(["_representative_distance", "station_id"]).head(config.representative_samples_per_category))
    if not rows:
        return pd.DataFrame(columns=summary.columns)
    representative = pd.concat(rows, ignore_index=True)
    return representative.drop(columns=["_representative_distance"], errors="ignore")


def _representative_time_series(
    analysis_panel: pd.DataFrame,
    sampled_stations: pd.DataFrame,
    path: Path,
) -> Path | None:
    if sampled_stations.empty:
        return None
    stations = sampled_stations[["station_id", "behavior_label"]].drop_duplicates().reset_index(drop=True)
    n = len(stations)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")
    for ax, (_, row) in zip(axes.flatten(), stations.iterrows()):
        ax.axis("on")
        observed = analysis_panel.loc[
            (analysis_panel["station_id"] == row["station_id"]) & analysis_panel["is_observed_in_service"],
            ["date", "observed_target"],
        ].copy()
        if observed.empty:
            continue
        ax.plot(observed["date"], observed["observed_target"], linewidth=1.0, marker="o", markersize=2)
        ax.set_title(f"{row['behavior_label']}: {row['station_id']}\nObserved in-service days only", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("Representative observed-demand station profiles", fontsize=14)
    return _write_figure(fig, path)


def _profile_figure(
    analysis_panel: pd.DataFrame,
    sampled_stations: pd.DataFrame,
    by: str,
    title: str,
    xlabel: str,
    path: Path,
) -> Path | None:
    if sampled_stations.empty:
        return None
    categories = sampled_stations["behavior_label"].dropna().unique().tolist()
    if not categories:
        return None
    fig, axes = plt.subplots(len(categories), 1, figsize=(9, max(4, len(categories) * 3)), squeeze=False)
    for ax, category in zip(axes.flatten(), categories):
        sample_ids = sampled_stations.loc[sampled_stations["behavior_label"] == category, "station_id"].tolist()
        for station_id in sample_ids:
            observed = analysis_panel.loc[
                (analysis_panel["station_id"] == station_id) & analysis_panel["is_observed_in_service"],
                ["date", "observed_target"],
            ].copy()
            if observed.empty:
                continue
            if by == "weekday":
                profile = observed.groupby(observed["date"].dt.dayofweek)["observed_target"].mean().reindex(range(7))
            else:
                profile = observed.groupby(observed["date"].dt.month)["observed_target"].mean().reindex(range(1, 13))
            ax.plot(profile.index, profile.values, marker="o", linewidth=1.2, label=str(station_id))
        ax.set_title(f"{category} (observed in-service days only)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean demand")
        ax.legend(loc="best", fontsize=7)
    fig.suptitle(title, fontsize=14)
    return _write_figure(fig, path)


def build_station_visuals(
    analysis_panel: pd.DataFrame,
    summary_with_clusters: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    cluster_selection: pd.DataFrame,
    config: StationDiagnosisConfig,
    output_dir: Path,
) -> dict[str, str]:
    """Build honest station-level diagnosis visuals using observed-only behavior data."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    plot_specs = [
        ("behavior_count_bar_chart", _value_counts_bar(summary_with_clusters["behavior_label"], "Behavior labels", "behavior_label", output_dir / "behavior_counts.png")),
        ("history_group_count_bar_chart", _value_counts_bar(summary_with_clusters["history_group"], "Maturity labels", "history_group", output_dir / "history_group_counts.png")),
        ("cluster_count_bar_chart", _value_counts_bar(summary_with_clusters["cluster_label"], "Cluster labels", "cluster_label", output_dir / "cluster_counts.png")),
    ]
    for label, path in plot_specs:
        written[label] = str(path)

    histogram_specs = [
        ("coverage_ratio_histogram", "coverage_ratio", "Coverage ratio inside apparent service window"),
        ("history_span_days_histogram", "history_span_days", "History span distribution"),
        ("missing_streak_histogram", "longest_missing_streak", "Longest missing-streak distribution"),
        ("avg_demand_histogram", "avg_demand_observed", "Average demand on observed in-service days"),
    ]
    for label, column, title in histogram_specs:
        path = _histogram(summary_with_clusters[column], title, column, output_dir / f"{label}.png")
        if path is not None:
            written[label] = str(path)

    scatter_specs = [
        (
            "observed_days_vs_history_span_by_maturity",
            "observed_days",
            "history_span_days",
            "history_group",
            "Observed days vs history span by maturity",
        ),
        (
            "zero_rate_vs_missing_rate_by_behavior",
            "zero_rate_observed",
            "missing_rate_in_service",
            "behavior_label",
            "Observed zero rate vs missing rate in service window",
        ),
        (
            "recent_activity_vs_coverage_by_maturity",
            "recent_active_days",
            "coverage_ratio",
            "history_group",
            "Recent active days vs coverage ratio",
        ),
    ]
    for label, x, y, color_by, title in scatter_specs:
        path = _scatter(summary_with_clusters, x, y, color_by, title, output_dir / f"{label}.png")
        if path is not None:
            written[label] = str(path)

    crosstab = pd.crosstab(summary_with_clusters["history_group"], summary_with_clusters["behavior_label"])
    crosstab_path = _heatmap_from_crosstab(crosstab, "Maturity vs behavior cross-tab", output_dir / "maturity_behavior_crosstab.png")
    if crosstab_path is not None:
        written["maturity_behavior_crosstab"] = str(crosstab_path)

    heatmap_path = _cluster_profile_heatmap(cluster_profile, output_dir / "cluster_profile_heatmap.png")
    if heatmap_path is not None:
        written["cluster_profile_heatmap"] = str(heatmap_path)

    stability_path = _cluster_stability_plot(cluster_selection, output_dir / "cluster_selection_diagnostics.png")
    if stability_path is not None:
        written["cluster_selection_diagnostics"] = str(stability_path)

    sampled_stations = _select_representative_stations(summary_with_clusters, config)
    sampled_time_series = _representative_time_series(analysis_panel, sampled_stations, output_dir / "representative_station_timeseries.png")
    if sampled_time_series is not None:
        written["representative_station_timeseries"] = str(sampled_time_series)

    weekday_profiles = _profile_figure(
        analysis_panel,
        sampled_stations,
        by="weekday",
        title="Representative weekday profiles by behavior label",
        xlabel="Day of week",
        path=output_dir / "representative_weekday_profiles.png",
    )
    if weekday_profiles is not None:
        written["representative_weekday_profiles"] = str(weekday_profiles)

    monthly_profiles = _profile_figure(
        analysis_panel,
        sampled_stations,
        by="month",
        title="Representative monthly profiles by behavior label",
        xlabel="Month",
        path=output_dir / "representative_monthly_profiles.png",
    )
    if monthly_profiles is not None:
        written["representative_monthly_profiles"] = str(monthly_profiles)

    return written
