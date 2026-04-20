from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from system_level.common.cli_utils import discover_project_root
from system_level.diagnosis import DiagnosticConfig, DiagnosticResult, run_forecasting_diagnostics
from system_level.diagnosis.anomalies import detect_anomalies
from system_level.diagnosis.baselines import compute_baseline_diagnostics
from system_level.diagnosis.stationarity import run_stationarity_checks
from system_level.diagnosis.time_index import validate_time_index
from system_level.diagnosis.trend import analyze_trend_and_decomposition, detect_level_shifts


ROOT = discover_project_root(__file__)


class DiagnosisRefactorTests(unittest.TestCase):
    def test_validate_time_index_accepts_datetime_index(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=6, freq="D")
        frame = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6]}, index=timestamps)
        config = DiagnosticConfig(series_name="indexed", target_col="value", frequency="daily")

        prepared, summary, tables = validate_time_index(frame, config)

        self.assertEqual(len(prepared), 6)
        self.assertEqual(summary["observed_points"], 6)
        self.assertEqual(summary["imputed_points"], 0)
        self.assertIn("prepared_time_index", tables)

    def test_validate_time_index_aggregates_duplicates_and_tracks_imputation(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-03"]),
                "value": [1.0, 2.0, 5.0],
            }
        )
        config = DiagnosticConfig(series_name="dupes", target_col="value", time_col="timestamp", frequency="daily")

        prepared, summary, tables = validate_time_index(frame, config)

        self.assertEqual(summary["duplicate_timestamps"], 1)
        self.assertEqual(summary["missing_periods"], 1)
        self.assertEqual(summary["imputed_points"], 1)
        self.assertEqual(prepared.loc[prepared["timestamp"] == pd.Timestamp("2024-01-01"), "observed_value"].iloc[0], 3.0)
        self.assertTrue({"imputed_value", "imputed_flag", "value_source"}.issubset(tables["prepared_time_index"].columns))

    def test_detect_anomalies_reports_method_and_suppresses_imputed_points(self) -> None:
        values = pd.Series([10.0, 10.5, 11.0, 35.0, 11.5, 12.0, 12.5, 13.0, 13.5])
        imputed = pd.Series([False, False, False, True, False, False, False, False, False])

        detail, summary = detect_anomalies(
            values,
            window=5,
            threshold=2.5,
            method="retrospective_centered_mad",
            is_imputed=imputed,
        )

        self.assertEqual(summary["anomaly_method"], "retrospective_centered_mad")
        self.assertEqual(summary["anomaly_scope"], "retrospective_offline")
        self.assertEqual(summary["suppressed_imputed_anomaly_count"], 1)
        self.assertFalse(bool(detail.loc[3, "outlier_flag"]))
        self.assertTrue(bool(detail.loc[3, "suppressed_imputed_anomaly_flag"]))

    def test_baseline_diagnostics_are_labeled_as_screening(self) -> None:
        summary, table = compute_baseline_diagnostics(pd.Series([10, 11, 9, 10, 11, 9, 10]), primary_period=3)
        self.assertEqual(summary["baseline_screening_scope"], "in_sample_screening_only")
        self.assertIn("descriptive screening diagnostics", summary["baseline_screening_note"])
        self.assertTrue((table["diagnostic_scope"] == "screening_only_in_sample").all())

    def test_stationarity_checks_report_short_history_and_normal_case(self) -> None:
        short = run_stationarity_checks(pd.Series([1.0, 2.0, 3.0]))
        self.assertEqual(short["stationarity_test_status"], "insufficient_history")

        rng = np.random.default_rng(42)
        long = run_stationarity_checks(pd.Series(rng.normal(0, 1, 80)))
        self.assertIn(long["stationarity_test_status"], {"ok", "partial", "unavailable"})
        self.assertIn(
            long["stationarity_assessment"],
            {"screening_suggests_stationary", "screening_suggests_nonstationary", "mixed_or_uncertain", "unknown"},
        )

    def test_trend_decomposition_reports_status_for_short_and_long_history(self) -> None:
        short_summary, short_decomposition = analyze_trend_and_decomposition(pd.Series(range(10)), (7, 30), 7)
        self.assertEqual(short_summary["decomposition_status"], "insufficient_history")
        self.assertIsNone(short_decomposition)

        periods = 200
        index = np.arange(periods)
        values = pd.Series(100 + 0.1 * index + 5 * np.sin(2 * np.pi * index / 7) + 2 * np.cos(2 * np.pi * index / 30))
        long_summary, _ = analyze_trend_and_decomposition(values, (7, 30, 365), 7)
        self.assertIn(long_summary["decomposition_status"], {"ok", "not_run", "failed"})
        self.assertIn("mstl_available", long_summary)

    def test_level_shift_detection_reports_optional_dependency_visibility(self) -> None:
        values = pd.Series(np.linspace(10, 20, 60))
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=60, freq="D"))
        with mock.patch("system_level.diagnosis.trend._load_ruptures", return_value=None):
            shifts, summary = detect_level_shifts(values, timestamps)
        self.assertEqual(shifts, [])
        self.assertEqual(summary["level_shift_detection_status"], "unavailable_optional_dependency")
        self.assertFalse(summary["level_shift_detection_available"])

    def test_diagnostic_result_as_dict_serializes_report_and_warnings(self) -> None:
        result = DiagnosticResult(
            summary={"series_name": "demo"},
            output_root=Path("/tmp/out"),
            figures_dir=Path("/tmp/out/figures"),
            tables_dir=Path("/tmp/out/tables"),
            report_dir=Path("/tmp/out/report"),
            figures={"series": Path("/tmp/out/figures/series.png")},
            tables={"summary": Path("/tmp/out/tables/diagnostics_summary.csv")},
            report_path=Path("/tmp/out/report/diagnostics_report.md"),
            warnings=["example warning"],
        )
        payload = result.as_dict()
        self.assertEqual(payload["report_path"], "/tmp/out/report/diagnostics_report.md")
        self.assertEqual(payload["warnings"], ["example warning"])

    def test_package_cli_runs_without_sys_path_hacks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "diagnostics_cli"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "system_level.diagnosis.cli",
                    "--synthetic-demo",
                    "--target-col",
                    "value",
                    "--frequency",
                    "daily",
                    "--output-root",
                    str(output_root),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output_root / "tables" / "diagnostics_summary.json").exists())
            self.assertTrue((output_root / "report" / "diagnostics_report.md").exists())

    def test_pipeline_end_to_end_writes_explicit_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "daily_diagnostics"
            timestamps = pd.date_range("2021-01-01", periods=120, freq="D")
            values = pd.Series(120 + np.sin(np.arange(len(timestamps)) * 2 * np.pi / 7))
            frame = pd.DataFrame({"timestamp": timestamps.delete([10, 11]), "value": values.drop(index=[10, 11]).to_numpy()})

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

            self.assertGreater(result.summary["imputed_points"], 0)
            self.assertIn("baseline_screening", result.summary["diagnostics_using_imputed_series"])
            self.assertIn("distribution_summary", result.summary["diagnostics_using_observed_only"])
            self.assertTrue((output_dir / "tables" / "prepared_time_index.csv").exists())
            self.assertTrue((output_dir / "tables" / "dependency_status.csv").exists())
            self.assertTrue((output_dir / "report" / "diagnostics_report.md").exists())


if __name__ == "__main__":
    unittest.main()
