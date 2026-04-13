from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":  # pragma: no cover - script entrypoint
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from station_level_analysis.categorize_stations import assign_station_categories
from station_level_analysis.config import StationDiagnosisConfig
from station_level_analysis.summary_features import build_station_summary_table
from station_level_analysis.utils import ensure_analysis_directories, load_station_daily_data, write_dataframe


def _top_table(frame: pd.DataFrame, column: str, top_n: int, ascending: bool = False) -> pd.DataFrame:
    return frame.sort_values(column, ascending=ascending).head(top_n).reset_index(drop=True)


def _build_category_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("station_category", as_index=False)
        .agg(
            station_count=("station_id", "nunique"),
            avg_demand_mean=("avg_demand", "mean"),
            zero_rate_mean=("zero_rate", "mean"),
            outlier_rate_mean=("outlier_rate", "mean"),
        )
        .sort_values(["station_count", "avg_demand_mean"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _write_markdown_report(
    summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    output_path: Path,
    top_n: int,
) -> Path:
    if summary.empty:
        output_path.write_text("# Station-Level Diagnosis Summary\n\nNo station rows were available.\n")
        return output_path

    start_date = pd.to_datetime(summary["start_date"]).min().date()
    end_date = pd.to_datetime(summary["end_date"]).max().date()

    busiest = _top_table(summary, "avg_demand", top_n, ascending=False)
    sparsest = _top_table(summary, "zero_rate", top_n, ascending=False)
    volatile = _top_table(summary, "coefficient_of_variation", top_n, ascending=False)
    anomaly_heavy = _top_table(summary, "outlier_rate", top_n, ascending=False)

    dominant_category = category_summary.iloc[0]["station_category"] if not category_summary.empty else "unknown"
    sparse_share = float((summary["station_category"] == "sparse_intermittent").mean())
    commuter_share = float((summary["station_category"] == "seasonal_commuter").mean())
    heterogeneity = "heterogeneous" if summary["station_category"].nunique() >= 4 else "moderately mixed"
    global_model_view = (
        "A single global model looks plausible for the majority of stations, but sparse or anomaly-heavy stations will likely need special handling later."
        if summary["station_category"].nunique() <= 4 and sparse_share < 0.35
        else "One global model may be too blunt on its own; sparse, commuter, and anomaly-heavy stations likely need differentiated treatment later."
    )

    lines = [
        "# Station-Level Diagnosis Summary",
        "",
        "This is station-level diagnosis only. It is not system-level analysis, forecasting, or model training.",
        "",
        f"- Number of stations: {summary['station_id'].nunique()}",
        f"- Date range covered: {start_date} to {end_date}",
        f"- Dominant category: {dominant_category}",
        "",
        "## Station Categories",
    ]
    for _, row in category_summary.iterrows():
        lines.append(f"- `{row['station_category']}`: {int(row['station_count'])} stations")

    def add_top_section(title: str, frame: pd.DataFrame, metric: str) -> None:
        lines.extend(["", f"## {title}"])
        for _, row in frame.iterrows():
            value = row[metric]
            formatted = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
            lines.append(f"- `{row['station_id']}`: {metric}={formatted}")

    add_top_section("Top 5 Busiest Stations", busiest, "avg_demand")
    add_top_section("Top 5 Sparsest Stations", sparsest, "zero_rate")
    add_top_section("Top 5 Most Volatile Stations", volatile, "coefficient_of_variation")
    add_top_section("Top 5 Anomaly-Heavy Stations", anomaly_heavy, "outlier_rate")

    lines.extend(
        [
            "",
            "## Interpretation",
            f"- Stations look {heterogeneity} rather than uniform.",
            f"- Sparse or intermittent behavior appears in about {sparse_share:.0%} of stations.",
            f"- Clear weekday commuter structure appears in about {commuter_share:.0%} of stations.",
            f"- {global_model_view}",
            "- The summary table should be reviewed before deciding whether later forecasting should use one global model, special sparse-station handling, or a more formal grouping strategy.",
        ]
    )

    output_path.write_text("\n".join(lines))
    return output_path


def build_station_level_diagnosis(
    input_path: str | Path,
    date_col: str,
    station_col: str,
    target_col: str,
    config: StationDiagnosisConfig | None = None,
) -> dict[str, str]:
    config = config or StationDiagnosisConfig()
    paths = ensure_analysis_directories(config)
    station_daily = load_station_daily_data(input_path, date_col, station_col, target_col)
    summary = build_station_summary_table(station_daily, config)
    categorized = assign_station_categories(summary, config)
    category_summary = _build_category_summary(categorized)

    output_dir = paths["station_level_output"]
    written = {
        "station_summary_csv": str(write_dataframe(categorized, output_dir / "station_summary_table.csv")),
        "station_summary_parquet": str(write_dataframe(categorized, output_dir / "station_summary_table.parquet")),
        "station_category_summary": str(write_dataframe(category_summary, output_dir / "station_category_summary.csv")),
        "top_avg_demand": str(write_dataframe(_top_table(categorized, "avg_demand", config.top_n), output_dir / "top_stations_by_avg_demand.csv")),
        "top_zero_rate": str(write_dataframe(_top_table(categorized, "zero_rate", config.top_n), output_dir / "top_stations_by_zero_rate.csv")),
        "top_outlier_rate": str(write_dataframe(_top_table(categorized, "outlier_rate", config.top_n), output_dir / "top_stations_by_outlier_rate.csv")),
        "top_cv": str(write_dataframe(_top_table(categorized, "coefficient_of_variation", config.top_n), output_dir / "top_stations_by_coefficient_of_variation.csv")),
    }
    report_path = _write_markdown_report(categorized, category_summary, output_dir / "station_level_diagnosis_summary.md", config.top_n)
    written["report"] = str(report_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build station-level diagnosis summaries from daily station data.")
    parser.add_argument("--input", required=True, help="Path to the station-level daily CSV or parquet file.")
    parser.add_argument("--date-col", required=True, help="Date column name.")
    parser.add_argument("--station-col", required=True, help="Station identifier column name.")
    parser.add_argument("--target-col", required=True, help="Daily target column name.")
    args = parser.parse_args()

    written = build_station_level_diagnosis(
        input_path=args.input,
        date_col=args.date_col,
        station_col=args.station_col,
        target_col=args.target_col,
    )
    print("Station-level diagnosis outputs saved:")
    for label, path in written.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
