from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from metro_bike_share_forecasting.station_level.diagnosis.config import StationDiagnosisConfig
from metro_bike_share_forecasting.station_level.diagnosis.pipeline import build_station_level_diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(description="Build station-level diagnosis summaries from daily station data.")
    parser.add_argument("--input", required=True, help="Path to the station-level daily CSV or parquet file.")
    parser.add_argument("--date-col", required=True, help="Date column name.")
    parser.add_argument("--station-col", required=True, help="Station identifier column name.")
    parser.add_argument("--target-col", required=True, help="Daily target column name.")
    parser.add_argument("--n-clusters", type=int, default=6, help="Number of KMeans clusters for diagnostic grouping.")
    args = parser.parse_args()

    written = build_station_level_diagnosis(
        input_path=args.input,
        date_col=args.date_col,
        station_col=args.station_col,
        target_col=args.target_col,
        config=StationDiagnosisConfig(n_clusters=args.n_clusters),
    )
    print("Station-level diagnosis outputs saved:")
    for label, path in written.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()

