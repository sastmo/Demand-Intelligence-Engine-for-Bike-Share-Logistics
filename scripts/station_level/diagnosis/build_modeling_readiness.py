from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from metro_bike_share_forecasting.station_level.diagnosis.model_readiness import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    build_station_modeling_readiness_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the station-level modeling-readiness package from finalized diagnosis outputs.")
    parser.add_argument(
        "--outputs-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Path to diagnosis/station_level_analysis/outputs.",
    )
    args = parser.parse_args()

    written = build_station_modeling_readiness_package(args.outputs_root)
    print("Station-level modeling-readiness outputs saved:")
    for label, path in written.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
