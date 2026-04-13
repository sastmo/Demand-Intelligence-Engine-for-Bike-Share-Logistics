from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metro_bike_share_forecasting.system_level import load_system_level_config, run_system_level_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full system-level daily forecasting pipeline.")
    parser.add_argument("--config", default="configs/system_level/config.yaml")
    args = parser.parse_args()

    config = load_system_level_config(args.config)
    summary = run_system_level_pipeline(config)
    print("System-level forecasting pipeline completed.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
