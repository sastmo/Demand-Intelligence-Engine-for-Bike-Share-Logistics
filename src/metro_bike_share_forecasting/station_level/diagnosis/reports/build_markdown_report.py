from __future__ import annotations

from pathlib import Path

import pandas as pd

from metro_bike_share_forecasting.station_level.diagnosis.config import StationDiagnosisConfig


def _top_table(frame: pd.DataFrame, column: str, top_n: int, ascending: bool = False) -> pd.DataFrame:
    clean = frame.copy()
    clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.replace([float("inf"), float("-inf")], pd.NA)
    return clean.sort_values(column, ascending=ascending, na_position="last").head(top_n).reset_index(drop=True)


def _cluster_strength_text(cluster_selection: pd.DataFrame, cluster_profile: pd.DataFrame) -> tuple[str, str]:
    if cluster_selection.empty or cluster_profile.empty:
        return (
            "weak",
            "Clustering is weak right now because there were not enough mature stations or the selected clustering setup was not stable enough to trust.",
        )

    selected = cluster_selection.loc[cluster_selection["selected"]].head(1)
    if selected.empty:
        return (
            "weak",
            "Clustering is weak right now because no selected clustering configuration was available.",
        )

    row = selected.iloc[0]
    silhouette = pd.to_numeric(pd.Series([row.get("silhouette_score")]), errors="coerce").iloc[0]
    tiny_clusters = int(pd.to_numeric(pd.Series([row.get("tiny_cluster_count")]), errors="coerce").fillna(0).iloc[0])

    if pd.notna(silhouette) and silhouette >= 0.20 and tiny_clusters == 0:
        return (
            "meaningful",
            "Clustering looks meaningful as a later refinement because the selected solution separates mature stations reasonably well without tiny unstable clusters.",
        )
    if pd.notna(silhouette) and silhouette >= 0.12 and tiny_clusters <= 1:
        return (
            "moderate",
            "Clustering looks moderately useful as an extra lens, but it should be treated as a later refinement rather than the first modeling split.",
        )
    return (
        "weak",
        "Clustering looks weak or unstable because separation is limited or the selected solution still relies on tiny clusters.",
    )


def _recommendation_text(
    summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    cluster_strength: str,
) -> tuple[str, list[str]]:
    sparse_share = float((summary["station_category"] == "sparse_intermittent").mean()) if not summary.empty else 0.0
    short_history_share = float(summary["history_group"].isin(["newborn", "young"]).mean()) if not summary.empty else 0.0
    heterogeneity_score = int(summary["station_category"].nunique()) + int(
        summary.loc[summary["cluster_label"].str.startswith("cluster_", na=False), "cluster_label"].nunique()
    )

    if sparse_share >= 0.20:
        recommendation = "global model plus sparse handling"
    elif cluster_strength == "meaningful" and heterogeneity_score >= 7:
        recommendation = "global model plus cluster-based refinement"
    else:
        recommendation = "global model first"

    notes = [
        "Use station-day as the primary forecasting unit in the next stage.",
        "Keep seasonal naive per station as the first baseline benchmark.",
        "Add a pooled global tree-based benchmark with lag and calendar features before moving to deeper models.",
        "Keep DeepAR or another global probabilistic sequence model as a strong next candidate because the project has many related station series and interval forecasting will matter later.",
    ]
    if sparse_share >= 0.20:
        notes.append("Sparse and intermittent stations should get explicit fallback handling instead of relying on one shared model behavior.")
    if short_history_share >= 0.15:
        notes.append("Newborn and young stations should be handled carefully because short history can distort station-specific feature quality.")
    if cluster_strength == "meaningful":
        notes.append("Cluster-based refinement looks justifiable later, but only after a strong global baseline is established.")
    else:
        notes.append("Cluster-based refinement should stay optional for now because the diagnosis does not show strong enough clustering separation yet.")
    return recommendation, notes


def _format_top_section(lines: list[str], title: str, frame: pd.DataFrame, metric: str) -> None:
    lines.extend(["", f"## {title}"])
    if frame.empty:
        lines.append("- No stations available.")
        return
    for _, row in frame.iterrows():
        value = pd.to_numeric(pd.Series([row[metric]]), errors="coerce").iloc[0]
        if pd.isna(value):
            lines.append(f"- `{row['station_id']}`: {metric}=NA")
        else:
            lines.append(f"- `{row['station_id']}`: {metric}={float(value):.3f}")


def build_station_markdown_report(
    inventory: pd.DataFrame,
    summary_with_clusters: pd.DataFrame,
    category_summary: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    cluster_selection: pd.DataFrame,
    config: StationDiagnosisConfig,
    output_path: Path,
) -> Path:
    """Write the final station-level diagnosis report."""

    if summary_with_clusters.empty:
        output_path.write_text("# Station-Level Diagnosis Summary\n\nNo station rows were available.\n")
        return output_path

    start_date = pd.to_datetime(summary_with_clusters["start_date"]).min().date()
    end_date = pd.to_datetime(summary_with_clusters["end_date"]).max().date()
    actual_station_count = int(summary_with_clusters["station_id"].nunique())
    expected_station_count = int(config.expected_station_count)
    count_gap = actual_station_count - expected_station_count

    short_history_count = int(inventory["is_short_history"].sum()) if not inventory.empty else 0
    nearly_zero_count = int(inventory["is_zero_almost_always"].sum()) if not inventory.empty else 0
    inactive_recent_count = int((~inventory["appears_active_recently"]).sum()) if not inventory.empty else 0

    maturity_counts = summary_with_clusters["history_group"].value_counts().reindex(["newborn", "young", "mature"], fill_value=0)
    cluster_counts = summary_with_clusters["cluster_label"].value_counts().sort_index()
    cluster_strength, cluster_signal = _cluster_strength_text(cluster_selection, cluster_profile)
    recommendation, recommendation_notes = _recommendation_text(summary_with_clusters, category_summary, cluster_strength)

    sparse_share = float((summary_with_clusters["station_category"] == "sparse_intermittent").mean())
    commuter_share = float((summary_with_clusters["station_category"] == "seasonal_commuter").mean())
    active_share = float((summary_with_clusters["active_day_rate"] >= 0.75).mean())
    heterogeneity = (
        "heterogeneous"
        if summary_with_clusters["station_category"].nunique() >= 5
        or summary_with_clusters.loc[summary_with_clusters["cluster_label"].str.startswith("cluster_", na=False), "cluster_label"].nunique() >= 4
        else "moderately mixed"
    )

    if count_gap > 0 and (short_history_count + nearly_zero_count + inactive_recent_count) > max(10, count_gap // 2):
        count_explanation = (
            "The higher-than-expected station count is likely explained by temporary, retired, or nearly empty stations that still appear in the station inventory."
        )
    elif count_gap > 0:
        count_explanation = (
            "The higher-than-expected station count looks real in the data snapshot, so the expected station count may be stale or narrower than the current inventory definition."
        )
    elif count_gap < 0:
        count_explanation = "The observed station count is below the expected count, which suggests some stations may be absent from the current station-day extract."
    else:
        count_explanation = "The observed station count aligns with the expected station universe."

    busiest = _top_table(summary_with_clusters, "avg_demand", config.top_n, ascending=False)
    sparsest = _top_table(summary_with_clusters, "zero_rate", config.top_n, ascending=False)
    volatile = _top_table(summary_with_clusters, "coefficient_of_variation", config.top_n, ascending=False)
    anomaly_heavy = _top_table(summary_with_clusters, "outlier_rate", config.top_n, ascending=False)

    lines = [
        "# Station-Level Diagnosis Summary",
        "",
        "This is station-level diagnosis only. It is not system-level analysis, forecasting, or model training.",
        "",
        "## Station Universe Validation",
        f"- Expected station count: {expected_station_count}",
        f"- Observed unique stations: {actual_station_count}",
        f"- Count gap: {count_gap:+d}",
        f"- Short-history stations: {short_history_count}",
        f"- Nearly always-zero stations: {nearly_zero_count}",
        f"- Not recently active stations: {inactive_recent_count}",
        f"- {count_explanation}",
        "",
        "## Date Range and Maturity",
        f"- Date range covered: {start_date} to {end_date}",
        f"- `newborn`: {int(maturity_counts['newborn'])} stations",
        f"- `young`: {int(maturity_counts['young'])} stations",
        f"- `mature`: {int(maturity_counts['mature'])} stations",
        "",
        "## Category Counts",
    ]
    for _, row in category_summary.iterrows():
        lines.append(f"- `{row['station_category']}`: {int(row['station_count'])} stations")

    lines.extend(["", "## Cluster Counts"])
    for cluster_label, count in cluster_counts.items():
        lines.append(f"- `{cluster_label}`: {int(count)} stations")

    lines.extend(
        [
            "",
            "## Cluster Assessment",
            f"- Overall cluster strength: {cluster_strength}",
            f"- {cluster_signal}",
        ]
    )
    if not cluster_selection.empty:
        selected = cluster_selection.loc[cluster_selection["selected"]].head(1)
        if not selected.empty:
            row = selected.iloc[0]
            silhouette = pd.to_numeric(pd.Series([row.get("silhouette_score")]), errors="coerce").iloc[0]
            lines.append(
                f"- Selected clustering setup: k={int(row['candidate_k'])}, silhouette={float(silhouette):.3f}" if pd.notna(silhouette) else f"- Selected clustering setup: k={int(row['candidate_k'])}, silhouette=NA"
            )
            lines.append(f"- Selection reason: {row.get('selection_reason', '')}")

    _format_top_section(lines, f"Top {config.top_n} Busiest Stations", busiest, "avg_demand")
    _format_top_section(lines, f"Top {config.top_n} Sparsest Stations", sparsest, "zero_rate")
    _format_top_section(lines, f"Top {config.top_n} Most Volatile Stations", volatile, "coefficient_of_variation")
    _format_top_section(lines, f"Top {config.top_n} Anomaly-Heavy Stations", anomaly_heavy, "outlier_rate")

    lines.extend(
        [
            "",
            "## Interpretation",
            f"- Station behavior looks {heterogeneity} rather than homogeneous.",
            f"- About {active_share:.0%} of stations are active on at least 75% of days, while about {sparse_share:.0%} are clearly sparse or intermittent.",
            f"- Strong weekday commuter structure appears in about {commuter_share:.0%} of stations.",
            "- Sparse stations should not be allowed to dominate the interpretation because their ratios and autocorrelation measures become less reliable.",
            "- A single global model still looks like a reasonable first benchmark, but it should be paired with explicit sparse-station handling if sparsity is material.",
            f"- {cluster_signal}",
            "- DeepAR or another global probabilistic model remains a strong later candidate because the station universe contains many related series with shared calendar structure and meaningful cross-station variation.",
            "",
            "## Final Recommendation",
            f"- Recommended next step: **{recommendation}**",
        ]
    )
    for note in recommendation_notes:
        lines.append(f"- {note}")

    output_path.write_text("\n".join(lines))
    return output_path
