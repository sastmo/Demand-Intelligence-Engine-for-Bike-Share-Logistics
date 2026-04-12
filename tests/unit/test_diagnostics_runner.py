from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DiagnosticsRunnerTests(unittest.TestCase):
    def test_runner_executes_end_to_end_on_synthetic_series(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "diagnostics_tmp"
            command = [
                sys.executable,
                str(ROOT / "run_diagnostics.py"),
                "--synthetic-demo",
                "--target-col",
                "value",
                "--frequency",
                "daily",
                "--output-root",
                str(output_root),
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Forecasting diagnostics completed.", result.stdout)
            self.assertTrue((output_root / "figures" / "series.png").exists())
            self.assertTrue((output_root / "tables" / "diagnostics_summary.json").exists())
            self.assertTrue((output_root / "report" / "diagnostics_report.md").exists())

    def test_runner_filters_segment_and_date_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "aggregate.csv"
            output_root = root / "diagnostics_tmp"
            rows = [
                ["bucket_start", "trip_count", "segment_type", "segment_id"],
                ["2019-01-01", "100", "system_total", "all"],
                ["2019-01-02", "120", "system_total", "all"],
                ["2019-01-03", "140", "system_total", "all"],
                ["2019-01-04", "25", "start_station", "3005"],
            ]
            with dataset.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(rows)

            command = [
                sys.executable,
                str(ROOT / "run_diagnostics.py"),
                str(dataset),
                "--target-col",
                "trip_count",
                "--time-col",
                "bucket_start",
                "--frequency",
                "daily",
                "--segment-type",
                "system_total",
                "--segment-id",
                "all",
                "--start-date",
                "2019-01-02",
                "--end-date",
                "2019-01-03",
                "--output-root",
                str(output_root),
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output_root / "tables" / "diagnostics_summary.json").exists())
            self.assertTrue((output_root / "figures" / "series.png").exists())


if __name__ == "__main__":
    unittest.main()
