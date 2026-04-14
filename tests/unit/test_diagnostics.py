from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forecasting_diagnostics import DiagnosticConfig, run_forecasting_diagnostics
from metro_bike_share_forecasting.diagnostics.time_series import (
    TimeSeriesDiagnosticsConfig,
    run_diagnostics,
    run_time_series_diagnostics,
)
from metro_bike_share_forecasting.features.regime import RegimeDefinition


class DiagnosticsTests(unittest.TestCase):
    def test_run_forecasting_diagnostics_writes_structured_outputs_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "daily_diagnostics"
            timestamps = pd.date_range("2021-01-01", periods=180, freq="D")
            index = np.arange(len(timestamps))
            values = 120 + 0.25 * index + 18 * np.sin(2 * np.pi * index / 7) + 3 * np.cos(2 * np.pi * index / 30)
            frame = pd.DataFrame({"timestamp": timestamps, "value": values})

            result = run_forecasting_diagnostics(
                frame,
                DiagnosticConfig(
                    output_root=output_dir,
                    series_name="daily_test_series",
                    target_col="value",
                    time_col="timestamp",
                    frequency="daily",
                    clean_output=True,
                ),
            )
            summary = result.summary

            self.assertEqual(summary["frequency"], "daily")
            self.assertTrue(summary["recommended_model_families"])
            self.assertTrue(summary["recommendations"])
            self.assertIsNotNone(summary["trend_strength"])
            self.assertIsNotNone(summary["seasonal_strength"])
            self.assertLess(summary["seasonal_naive_mae"], summary["naive_mae"])

            expected_files = [
                output_dir / "figures" / "series.png",
                output_dir / "figures" / "time_index_gaps.png",
                output_dir / "figures" / "acf.png",
                output_dir / "figures" / "pacf.png",
                output_dir / "figures" / "stl.png",
                output_dir / "figures" / "periodogram.png",
                output_dir / "figures" / "distribution.png",
                output_dir / "figures" / "rolling_stats.png",
                output_dir / "figures" / "outliers.png",
                output_dir / "figures" / "seasonal_profile.png",
                output_dir / "tables" / "weekday_profile.csv",
                output_dir / "tables" / "monthly_profile.csv",
                output_dir / "tables" / "diagnostics_summary.json",
                output_dir / "tables" / "diagnostics_summary.csv",
                output_dir / "reports" / "diagnostics_report.md",
            ]
            for path in expected_files:
                self.assertTrue(path.exists(), str(path))

    def test_run_forecasting_diagnostics_flags_multiple_seasonalities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "hourly_diagnostics"
            timestamps = pd.date_range("2021-01-01", periods=24 * 28, freq="h")
            index = np.arange(len(timestamps))
            values = (
                70
                + 12 * np.sin(2 * np.pi * index / 24)
                + 7 * np.sin(2 * np.pi * index / 168)
                + 0.5 * np.cos(2 * np.pi * index / 12)
            )
            frame = pd.DataFrame({"timestamp": timestamps, "value": values})

            summary = run_forecasting_diagnostics(
                frame,
                DiagnosticConfig(
                    output_root=output_dir,
                    series_name="hourly_multi_seasonal",
                    target_col="value",
                    time_col="timestamp",
                    frequency="hourly",
                    clean_output=True,
                ),
            ).summary

            self.assertTrue(summary["multiple_seasonalities_detected"])
            self.assertIn("TBATS / multi-seasonal state space", summary["recommended_model_families"])
            self.assertIn("Fourier-based regression or dynamic harmonic regression", summary["recommended_model_families"])

    def test_legacy_wrapper_still_writes_flat_compatibility_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "legacy_compat"
            timestamps = pd.date_range("2021-01-01", periods=120, freq="D")
            values = 100 + 15 * np.sin(2 * np.pi * np.arange(120) / 7)
            frame = pd.DataFrame({"timestamp": timestamps, "value": values})

            summary = run_time_series_diagnostics(
                frame,
                TimeSeriesDiagnosticsConfig(
                    output_dir=output_dir,
                    series_name="legacy_series",
                    frequency="daily",
                ),
            )

            self.assertEqual(summary["frequency"], "daily")
            self.assertTrue((output_dir / "series.png").exists())
            self.assertTrue((output_dir / "acf.png").exists())
            self.assertTrue((output_dir / "diagnostics_summary.json").exists())
            self.assertTrue((output_dir / "diagnostics_report.md").exists())

    def test_run_diagnostics_wrapper_handles_composite_frequency_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            frame = pd.DataFrame(
                {
                    "bucket_start": pd.date_range("2022-01-01", periods=90, freq="D"),
                    "trip_count": 200 + 10 * np.sin(2 * np.pi * np.arange(90) / 7),
                }
            )
            regime_definition = RegimeDefinition(
                pandemic_shock_start=pd.Timestamp("2020-03-15"),
                recovery_start=pd.Timestamp("2021-06-15"),
                post_pandemic_start=pd.Timestamp("2022-06-15"),
                detected_breakpoints=[],
                detection_method="known_dates_only",
            )

            summary = run_diagnostics(
                frame=frame,
                frequency="daily__system_total__all",
                output_root=output_root,
                regime_definition=regime_definition,
            )

            target_dir = output_root / "daily__system_total__all"
            self.assertEqual(summary["frequency"], "daily")
            self.assertEqual(summary["series_key"], "daily__system_total__all")
            self.assertTrue((target_dir / "diagnostics_summary.json").exists())
            self.assertTrue((target_dir / "diagnostics_report.md").exists())


if __name__ == "__main__":
    unittest.main()
