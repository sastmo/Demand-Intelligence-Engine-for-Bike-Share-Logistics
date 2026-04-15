from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from metro_bike_share_forecasting.system_level.forecasting.config import load_system_level_config
from metro_bike_share_forecasting.system_level.forecasting.data import ensure_output_directories, write_dataframe
from metro_bike_share_forecasting.system_level.forecasting.evaluation import (
    plot_model_comparison,
    summarize_backtest_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate system-level backtests and produce tables and figures.")
    parser.add_argument("--config", default="configs/system_level/config.yaml")
    args = parser.parse_args()

    config = load_system_level_config(args.config)
    directories = ensure_output_directories(config)
    metrics_path = directories["backtests"] / "system_level_fold_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected backtest metrics at {metrics_path}. Run backtest.py first.")
    metrics = pd.read_csv(metrics_path, parse_dates=["train_start", "train_end", "test_start", "test_end"])
    summary = summarize_backtest_metrics(metrics)
    write_dataframe(summary, directories["metrics"] / "system_level_model_comparison.csv")
    plot_model_comparison(summary, directories["figures"] / "system_level_model_comparison.png")
    print(f"Wrote system-level evaluation tables and figures to {config.output_root}")


if __name__ == "__main__":
    main()
