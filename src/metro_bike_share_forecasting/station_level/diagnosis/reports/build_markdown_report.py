from __future__ import annotations

from pathlib import Path

import pandas as pd

from metro_bike_share_forecasting.station_level.diagnosis.config import StationDiagnosisConfig


def _top_table(frame: pd.DataFrame, column: str, top_n: int, ascending: bool = False) -> pd.DataFrame:
    return frame.sort_values(column, ascending=ascending).head(top_n).reset_index(drop=True)


def _clustering_signal_text(summary: pd.DataFrame, cluster_profile: pd.DataFrame) -> str:
    if cluster_profile.empty or len(cluster_profile) < 2:
        return "Clustering is too weak to interpret yet because only one usable cluster was formed."
    avg_demand_range = float(cluster_profile["avg_demand_mean"].max() - cluster_profile["avg_demand_mean"].min())
    zero_rate_range = float(cluster_profile["zero_rate_mean"].max() - cluster_profile["zero_rate_mean"].min())
    overall_avg = float(summary["avg_demand"].mean()) if not summary.empty else 0.0
    if avg_demand_range > max(2.0, overall_avg * 0.5) or zero_rate_range > 0.25:
        return "Clustering looks meaningful as an extra diagnostic lens because clusters differ clearly in demand level or sparsity."
    return "Clustering looks weak to moderate so far; rule-based categories may still be the clearer diagnostic lens."


def build_station_markdown_report(
    summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    config: StationDiagnosisConfig,
    output_path: Path,
) -> Path:
    """Write the station-level diagnosis markdown report."""

    if summary.empty:
        output_path.write_text("# Station-Level Diagnosis Summary\n\nNo station rows were available.\n")
        return output_path

    start_date = pd.to_datetime(summary["start_date"]).min().date()
    end_date = pd.to_datetime(summary["end_date"]).max().date()
    busiest = _top_table(summary, "avg_demand", config.top_n, ascending=False)
    sparsest = _top_table(summary, "zero_rate", config.top_n, ascending=False)
    volatile = _top_table(summary, "coefficient_of_variation", config.top_n, ascending=False)
    anomaly_heavy = _top_table(summary, "outlier_rate", config.top_n, ascending=False)
    sparse_share = float((summary["station_category"] == "sparse_intermittent").mean())
    commuter_share = float((summary["station_category"] == "seasonal_commuter").mean())
    heterogeneity = "heterogeneous" if summary["station_category"].nunique() >= 4 or summary["cluster_label"].nunique() >= 4 else "moderately mixed"
    global_model_view = (
        "One global model might be reasonable for many stations, but sparse or anomaly-heavy stations will likely need special handling later."
        if sparse_share < 0.30 and summary["cluster_label"].nunique() <= 4
        else "A single global model may be too blunt on its own; sparse, anomaly-heavy, or strongly behavioral subgroups likely need differentiated treatment later."
    )
    cluster_signal = _clustering_signal_text(summary, cluster_profile)

    lines = [
        "# Station-Level Diagnosis Summary",
        "",
        "This is station-level diagnosis only. It is not system-level analysis, forecasting, or model training.",
        "",
        f"- Number of stations: {summary['station_id'].nunique()}",
        f"- Date range covered: {start_date} to {end_date}",
        f"- Number of clusters: {summary['cluster_label'].nunique()}",
        "",
        "## Category Counts",
    ]
    for _, row in category_summary.iterrows():
        lines.append(f"- `{row['station_category']}`: {int(row['station_count'])} stations")

    lines.extend(["", "## Cluster Counts"])
    cluster_counts = summary["cluster_label"].value_counts().sort_index()
    for cluster_label, count in cluster_counts.items():
        lines.append(f"- `{cluster_label}`: {int(count)} stations")

    def add_top_section(title: str, frame: pd.DataFrame, metric: str) -> None:
        lines.extend(["", f"## {title}"])
        for _, row in frame.iterrows():
            lines.append(f"- `{row['station_id']}`: {metric}={row[metric]:.3f}")

    add_top_section("Top 5 Busiest Stations", busiest, "avg_demand")
    add_top_section("Top 5 Sparsest Stations", sparsest, "zero_rate")
    add_top_section("Top 5 Most Volatile Stations", volatile, "coefficient_of_variation")
    add_top_section("Top 5 Anomaly-Heavy Stations", anomaly_heavy, "outlier_rate")

    lines.extend(
        [
            "",
            "## Interpretation",
            f"- Stations look {heterogeneity} rather than homogeneous.",
            f"- Sparse or intermittent behavior appears in about {sparse_share:.0%} of stations.",
            f"- Clear weekday commuter structure appears in about {commuter_share:.0%} of stations.",
            f"- {global_model_view}",
            f"- {cluster_signal}",
            "- This diagnosis layer should be reviewed before deciding whether later forecasting should use one global model, special sparse-station handling, cluster-based modeling, or a deeper global model such as DeepAR.",
        ]
    )
    output_path.write_text("\n".join(lines))
    return output_path
