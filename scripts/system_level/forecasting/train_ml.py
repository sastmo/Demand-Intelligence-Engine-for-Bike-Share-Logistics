from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from metro_bike_share_forecasting.system_level.forecasting.config import load_system_level_config
from metro_bike_share_forecasting.system_level.forecasting.data import ensure_output_directories, write_dataframe
from metro_bike_share_forecasting.system_level.forecasting.pipeline import (
    build_external_feature_artifact,
    build_time_index_and_target_artifact,
    run_family_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and forecast ML system-level models.")
    parser.add_argument("--config", default="configs/system_level/config.yaml")
    args = parser.parse_args()

    config = load_system_level_config(args.config)
    directories = ensure_output_directories(config)
    target = build_time_index_and_target_artifact(config)
    external = build_external_feature_artifact(config)
    forecasts = run_family_training(target, external, config, "ml")
    output_path = directories["forecasts"] / "system_level_ml_forecasts.csv"
    write_dataframe(forecasts, output_path)
    print(f"Wrote system-level ML forecasts to {output_path}")


if __name__ == "__main__":
    main()
