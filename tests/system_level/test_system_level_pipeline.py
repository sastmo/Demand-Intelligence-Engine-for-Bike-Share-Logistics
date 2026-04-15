from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from metro_bike_share_forecasting.system_level import load_system_level_config, run_system_level_pipeline


class SystemLevelPipelineTests(unittest.TestCase):
    def test_system_level_pipeline_runs_on_synthetic_daily_total(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            aggregate_path = root / "daily_aggregate.csv"
            output_root = root / "forecasts" / "system_level"

            periods = 220
            dates = pd.date_range("2022-01-01", periods=periods, freq="D")
            index = np.arange(periods)
            values = 500 + 0.4 * index + 35 * np.sin(2 * np.pi * index / 7) + 8 * np.cos(2 * np.pi * index / 30)
            aggregate = pd.DataFrame(
                {
                    "bucket_start": dates,
                    "trip_count": values,
                    "segment_type": "system_total",
                    "segment_id": "all",
                }
            )
            aggregate.to_csv(aggregate_path, index=False)

            config_path = root / "config.yaml"
            config_payload = {
                "input": {
                    "daily_aggregate_path": str(aggregate_path),
                    "cleaned_trip_path": str(root / "missing_cleaned.csv.gz"),
                    "target_column": "trip_count",
                    "date_column": "bucket_start",
                    "frequency": "daily",
                    "segment_type": "system_total",
                    "segment_id": "all",
                },
                "forecast": {"main_horizons": [7, 30], "extended_horizon": 90},
                "features": {
                    "lags": [1, 2, 3, 7, 14, 21, 28, 30],
                    "rolling_windows": [7, 14, 28],
                    "holiday_country": "US",
                    "include_weekly_fourier": True,
                    "include_yearly_fourier": True,
                    "weekly_fourier_order": 2,
                    "yearly_fourier_order": 2,
                    "external_features_path": "",
                    "external_date_column": "date",
                },
                "backtest": {"initial_train_size": 120, "step_size": 14, "max_folds": 3},
                "models": {
                    "baselines": {"naive": True, "seasonal_naive_7": True, "seasonal_naive_30": False},
                    "classical": {
                        "ets": True,
                        "sarimax_dynamic": False,
                        "fourier_dynamic_regression": True,
                        "unobserved_components": False,
                    },
                    "ml": {"tree_boosting": True},
                },
                "output": {"root": str(output_root)},
            }
            config_path.write_text(yaml.safe_dump(config_payload))

            config = load_system_level_config(config_path)
            summary = run_system_level_pipeline(config)

            self.assertGreater(summary["target_rows"], 0)
            self.assertGreater(summary["backtest_metric_rows"], 0)
            self.assertGreater(summary["forecast_rows"], 0)
            self.assertTrue((output_root / "metrics" / "system_level_model_comparison.csv").exists())
            self.assertTrue((output_root / "forecasts" / "system_level_future_forecasts.csv").exists())
            self.assertTrue((output_root / "backtests" / "system_level_fit_diagnostics.csv").exists())
            self.assertTrue((output_root / "backtests" / "system_level_backtest_residuals.csv").exists())
            self.assertTrue((output_root / "metrics" / "system_level_interval_calibration.csv").exists())
            self.assertTrue((output_root / "metrics" / "system_level_interval_coverage.csv").exists())

            future_forecasts = pd.read_csv(output_root / "forecasts" / "system_level_future_forecasts.csv")
            self.assertTrue(
                {"point_forecast", "lower_80", "upper_80", "lower_95", "upper_95", "horizon_step"}.issubset(
                    future_forecasts.columns
                )
            )
            self.assertTrue((future_forecasts["lower_80"].fillna(0.0) >= 0.0).all())
            self.assertTrue((future_forecasts["lower_95"].fillna(0.0) >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
