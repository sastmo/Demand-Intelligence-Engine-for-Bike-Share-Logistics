from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from metro_bike_share_forecasting.system_level.config import load_system_level_config
from metro_bike_share_forecasting.system_level.data import ensure_output_directories, write_dataframe
from metro_bike_share_forecasting.system_level.pipeline import build_external_feature_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Build optional known-future external features for the system-level series.")
    parser.add_argument("--config", default="configs/system_level/config.yaml")
    args = parser.parse_args()

    config = load_system_level_config(args.config)
    directories = ensure_output_directories(config)
    frame = build_external_feature_artifact(config)
    output_path = directories["feature_artifacts"] / "system_level_external_features.csv"
    write_dataframe(frame, output_path)
    print(f"Wrote system-level external features to {output_path}")


if __name__ == "__main__":
    main()
