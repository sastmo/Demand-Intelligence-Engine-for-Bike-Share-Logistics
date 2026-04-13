from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metro_bike_share_forecasting.system_level.backtesting import run_backtests
from metro_bike_share_forecasting.system_level.config import load_system_level_config
from metro_bike_share_forecasting.system_level.data import ensure_output_directories, write_dataframe
from metro_bike_share_forecasting.system_level.pipeline import (
    build_external_feature_artifact,
    build_time_index_and_target_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling-origin backtests for system-level forecasting.")
    parser.add_argument("--config", default="configs/system_level/config.yaml")
    args = parser.parse_args()

    config = load_system_level_config(args.config)
    directories = ensure_output_directories(config)
    target = build_time_index_and_target_artifact(config)
    external = build_external_feature_artifact(config)
    metrics, forecasts, windows = run_backtests(target, external, config, config.enabled_model_keys)
    write_dataframe(metrics, directories["backtests"] / "system_level_fold_metrics.csv")
    write_dataframe(forecasts, directories["backtests"] / "system_level_fold_forecasts.csv")
    write_dataframe(windows, directories["backtests"] / "system_level_backtest_windows.csv")
    print(f"Wrote system-level backtest artifacts to {directories['backtests']}")


if __name__ == "__main__":
    main()
