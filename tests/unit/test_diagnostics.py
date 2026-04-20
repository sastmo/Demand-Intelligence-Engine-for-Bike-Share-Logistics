from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from system_level.diagnosis import DiagnosticConfig, run_forecasting_diagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_run_forecasting_diagnostics_writes_structured_outputs(self) -> None:
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
            self.assertIsNotNone(summary["trend_strength"])
            self.assertIsNotNone(summary["seasonal_strength"])
            self.assertLess(summary["baseline_screening_seasonal_naive_mae"], summary["baseline_screening_naive_mae"])
            self.assertEqual(summary["baseline_screening_scope"], "in_sample_screening_only")
            self.assertTrue((output_dir / "report" / "diagnostics_report.md").exists())

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
                output_dir / "tables" / "diagnostics_summary.csv",
                output_dir / "tables" / "diagnostics_summary.json",
                output_dir / "tables" / "prepared_time_index.csv",
                output_dir / "tables" / "dependency_status.csv",
                output_dir / "tables" / "diagnostic_series_usage.csv",
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
            self.assertEqual(summary["frequency_domain_status"], "ok")
            self.assertTrue((output_dir / "tables" / "diagnostics_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
