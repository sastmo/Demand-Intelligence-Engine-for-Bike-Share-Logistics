from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml

from system_level.cli import build_parser as build_root_cli_parser
from system_level.common.cli_utils import discover_project_root
from system_level.common.cli_utils import make_progress_logger
from system_level.common.intervals import (
    apply_calibrated_intervals,
    collect_backtest_residuals,
    fit_interval_calibration,
)
from system_level.common.metrics import default_mase_season_length
from station_level.forecasting.backtesting import build_station_backtest_windows
from station_level.forecasting.pipeline import run_station_level_pipeline
from station_level.forecasting.models import (
    DeepARArtifact,
    TreeModelArtifact,
    predict_with_deepar,
    predict_with_tree,
    station_model_runtime_report,
)
from system_level.forecasting.backtesting import build_rolling_windows
from system_level.forecasting.config import load_system_level_config
from system_level.forecasting.data import _finalize_target_frame
from system_level.forecasting.features import build_system_level_features
from system_level.forecasting.models import system_model_runtime_report


ROOT = discover_project_root(__file__)


class _LagPlusOneEstimator:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["lag_1"].fillna(0.0).to_numpy(dtype=float) + 1.0


class _ConstantDeepARNetwork(torch.nn.Module):
    def __init__(self, mean_value: float, scale_value: float) -> None:
        super().__init__()
        self.mean_value = float(mean_value)
        self.scale_value = float(scale_value)

    def forward(self, station_codes: torch.Tensor, numeric_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = station_codes.shape[0]
        return (
            torch.full((rows,), self.mean_value, dtype=torch.float32),
            torch.full((rows,), self.scale_value, dtype=torch.float32),
        )


class ForecastingRefactorTests(unittest.TestCase):
    def test_config_loader_resolves_paths_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src" / "dashboard").mkdir(parents=True)
            (root / "src" / "system_level").mkdir(parents=True)
            (root / "src" / "station_level").mkdir(parents=True)
            (root / "configs" / "system_level").mkdir(parents=True)
            aggregate_path = root / "data" / "aggregate.csv"
            aggregate_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "bucket_start": pd.date_range("2024-01-01", periods=10, freq="D"),
                    "trip_count": np.arange(10),
                    "segment_type": "system_total",
                    "segment_id": "all",
                }
            ).to_csv(aggregate_path, index=False)

            config_path = root / "configs" / "system_level" / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "input": {
                            "daily_aggregate_path": "data/aggregate.csv",
                            "cleaned_trip_path": "data/missing.csv.gz",
                            "target_column": "trip_count",
                            "date_column": "bucket_start",
                            "frequency": "weekly",
                            "segment_type": "system_total",
                            "segment_id": "all",
                        },
                        "forecast": {"main_horizons": [2], "extended_horizon": 4},
                        "features": {
                            "lags": [1],
                            "rolling_windows": [2],
                            "holiday_country": "US",
                            "external_features_path": "",
                        },
                        "backtest": {"initial_train_size": 5, "step_size": 1, "max_folds": 2},
                        "models": {
                            "baselines": {"naive": True},
                            "classical": {"ets": False},
                            "ml": {"tree_boosting": False},
                        },
                        "output": {"root": "forecasts/system_level"},
                    }
                )
            )

            config = load_system_level_config(config_path)
            self.assertEqual(config.project_root, root.resolve())
            self.assertEqual(config.daily_aggregate_path, aggregate_path.resolve())
            self.assertEqual(config.output_root, (root / "forecasts" / "system_level").resolve())
            self.assertEqual(config.mase_season_length, 52)

    def test_time_index_construction_zero_fills_with_missing_flag(self) -> None:
        frame = pd.DataFrame(
            {
                "bucket_start": pd.to_datetime(["2024-01-01", "2024-01-03"]),
                "trip_count": [5, 7],
            }
        )
        completed = _finalize_target_frame(frame, "bucket_start", "trip_count", "zero_fill_with_flag")
        self.assertEqual(completed["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-01", "2024-01-02", "2024-01-03"])
        self.assertEqual(completed["target"].tolist(), [5.0, 0.0, 7.0])
        self.assertEqual(completed["missing_period_flag"].tolist(), [0, 1, 0])

    def test_system_level_features_use_shifted_history_only(self) -> None:
        target = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5, freq="D"),
                "target": [10.0, 20.0, 30.0, 40.0, 50.0],
                "missing_period_flag": 0,
                "series_scope": "system_level",
            }
        )
        config = SimpleNamespace(
            holiday_country="US",
            include_weekly_fourier=False,
            include_yearly_fourier=False,
            weekly_fourier_order=0,
            yearly_fourier_order=0,
            lags=(1,),
            rolling_windows=(3,),
        )
        features = build_system_level_features(target, config, pd.DataFrame(columns=["date"]))
        self.assertEqual(features.loc[2, "lag_1"], 20.0)
        self.assertAlmostEqual(features.loc[3, "rolling_mean_3"], 20.0)
        self.assertNotEqual(features.loc[3, "rolling_mean_3"], (20.0 + 30.0 + 40.0) / 3.0)

    def test_rolling_windows_are_expanding_and_time_ordered(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=12, freq="D"),
                "target": np.arange(12, dtype=float),
            }
        )
        config = SimpleNamespace(initial_train_size=5, step_size=2, max_folds=3)
        windows = build_rolling_windows(frame, config, horizon=2)
        self.assertEqual(len(windows), 3)
        self.assertTrue(all(window.train_end < window.test_start for window in windows))
        self.assertEqual([len(window.train_frame) for window in windows], [5, 7, 9])

    def test_station_backtest_windows_respect_frequency_and_horizon(self) -> None:
        panel = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=20, freq="D"),
                "station_id": ["A"] * 20,
                "target": np.arange(20, dtype=float),
                "in_service": True,
            }
        )
        config = SimpleNamespace(initial_train_size=8, step_size=4, max_folds=2, max_backtest_horizon=3)
        windows = build_station_backtest_windows(panel, config)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].forecast_dates[0], pd.Timestamp("2024-01-13"))
        self.assertEqual(windows[1].forecast_dates[-1], pd.Timestamp("2024-01-19"))

    def test_default_mase_scaling_changes_with_frequency(self) -> None:
        self.assertEqual(default_mase_season_length("daily"), 7)
        self.assertEqual(default_mase_season_length("weekly"), 52)
        self.assertEqual(default_mase_season_length("monthly"), 12)

    def test_recursive_tree_forecast_updates_future_lag_features(self) -> None:
        train_panel = pd.DataFrame(
            {
                "station_id": ["A"] * 5,
                "date": pd.date_range("2024-01-01", periods=5, freq="D"),
                "target": [1.0, 2.0, 3.0, 4.0, 5.0],
                "in_service": True,
            }
        )
        artifact = TreeModelArtifact(
            model_name="lgbm",
            implementation="test-double",
            point_estimator=_LagPlusOneEstimator(),
            lower_estimator=_LagPlusOneEstimator(),
            upper_estimator=_LagPlusOneEstimator(),
            feature_columns=["station_id", "lag_1"],
            encoder_metadata={"categorical_maps": {"station_id": {"A": 1}}, "columns": ["station_id", "lag_1"]},
            selected_params={"mode": "lag_plus_one"},
            tuned=False,
        )
        config = SimpleNamespace(
            holiday_country="US",
            include_category_feature=False,
            include_cluster_feature=False,
        )
        forecast_dates = pd.date_range("2024-01-06", periods=2, freq="D")
        predictions = predict_with_tree(artifact, train_panel, forecast_dates, ["A"], config, None)
        self.assertEqual(predictions["prediction"].tolist(), [6.0, 7.0])

    def test_tree_forecast_without_direct_quantiles_omits_interval_columns(self) -> None:
        train_panel = pd.DataFrame(
            {
                "station_id": ["A"] * 3,
                "date": pd.date_range("2024-01-01", periods=3, freq="D"),
                "target": [2.0, 3.0, 4.0],
                "in_service": True,
            }
        )
        artifact = TreeModelArtifact(
            model_name="xgboost",
            implementation="xgboost",
            point_estimator=_LagPlusOneEstimator(),
            lower_estimator=None,
            upper_estimator=None,
            feature_columns=["station_id", "lag_1"],
            encoder_metadata={"categorical_maps": {"station_id": {"A": 1}}, "columns": ["station_id", "lag_1"]},
            selected_params={"mode": "point_only"},
            tuned=False,
        )
        config = SimpleNamespace(
            holiday_country="US",
            include_category_feature=False,
            include_cluster_feature=False,
        )
        predictions = predict_with_tree(artifact, train_panel, pd.date_range("2024-01-04", periods=1, freq="D"), ["A"], config, None)
        self.assertNotIn("lower_80", predictions.columns)
        self.assertNotIn("upper_80", predictions.columns)

    def test_deepar_predictions_are_rescaled_per_station(self) -> None:
        train_panel = pd.DataFrame(
            {
                "station_id": ["A", "B"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "target": [10.0, 2.0],
                "in_service": [True, True],
            }
        )
        artifact = DeepARArtifact(
            model_name="deepar",
            network=_ConstantDeepARNetwork(mean_value=1.5, scale_value=0.5),
            feature_columns=["station_id", "lag_1"],
            encoder_metadata={"categorical_maps": {"station_id": {"A": 1, "B": 2}}, "columns": ["station_id", "lag_1"]},
            numeric_columns=["lag_1"],
            numeric_means={"lag_1": 0.0},
            numeric_stds={"lag_1": 1.0},
            station_scales={"A": 10.0, "B": 2.0},
            tuned=False,
            selected_params={"mode": "constant"},
        )
        config = SimpleNamespace(
            holiday_country="US",
            include_category_feature=False,
            include_cluster_feature=False,
        )
        predictions = predict_with_deepar(
            artifact,
            train_panel,
            pd.date_range("2024-01-02", periods=1, freq="D"),
            ["A", "B"],
            config,
            None,
        ).sort_values("station_id")
        self.assertEqual(predictions["prediction"].round(3).tolist(), [15.0, 3.0])

    def test_interval_calibration_is_based_on_backtest_residuals(self) -> None:
        backtest_forecasts = pd.DataFrame(
            {
                "model_name": ["naive", "naive", "naive", "naive"],
                "fold_id": [1, 1, 2, 2],
                "horizon": [2, 2, 2, 2],
                "date": pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-15", "2024-01-16"]),
                "prediction": [10.0, 11.0, 9.0, 8.0],
                "actual": [12.0, 10.0, 11.0, 9.0],
                "horizon_step": [1, 2, 1, 2],
            }
        )
        residuals = collect_backtest_residuals(backtest_forecasts)
        calibration = fit_interval_calibration(residuals, min_samples=1)
        intervalized = apply_calibrated_intervals(backtest_forecasts, calibration)
        self.assertEqual(len(residuals), 4)
        self.assertTrue({"residual_q10", "residual_q90"}.issubset(calibration.columns))
        self.assertEqual(intervalized["interval_n_residuals"].dropna().astype(int).tolist(), [2, 2, 2, 2])

    def test_python_module_cli_build_target_runs_without_sys_path_hacks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            aggregate_path = root / "daily_aggregate.csv"
            output_root = root / "forecasts" / "system_level"
            pd.DataFrame(
                {
                    "bucket_start": pd.date_range("2024-01-01", periods=20, freq="D"),
                    "trip_count": np.arange(20, dtype=float),
                    "segment_type": "system_total",
                    "segment_id": "all",
                }
            ).to_csv(aggregate_path, index=False)

            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "input": {
                            "daily_aggregate_path": str(aggregate_path),
                            "cleaned_trip_path": str(root / "missing.csv.gz"),
                            "target_column": "trip_count",
                            "date_column": "bucket_start",
                            "frequency": "daily",
                            "segment_type": "system_total",
                            "segment_id": "all",
                        },
                        "forecast": {"main_horizons": [2], "extended_horizon": 3},
                        "features": {
                            "lags": [1],
                            "rolling_windows": [2],
                            "holiday_country": "US",
                            "external_features_path": "",
                        },
                        "backtest": {"initial_train_size": 10, "step_size": 2, "max_folds": 2},
                        "models": {
                            "baselines": {"naive": True},
                            "classical": {"ets": False},
                            "ml": {"tree_boosting": False},
                        },
                        "output": {"root": str(output_root)},
                    }
                )
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "system_level.forecasting.cli",
                    "build-target",
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output_root / "feature_artifacts" / "system_level_target.csv").exists())

    def test_package_imports_work_without_test_bootstrap(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import system_level.forecasting.cli;"
                    "import station_level.forecasting.cli"
                ),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_root_cli_forecast_parser_accepts_verbose(self) -> None:
        parser = build_root_cli_parser()
        args = parser.parse_args(
            [
                "forecast",
                "--level",
                "station",
                "--config",
                "configs/station_level/config.yaml",
                "--verbose",
            ]
        )
        self.assertTrue(args.verbose)

    def test_root_cli_doctor_parser_accepts_station_scope(self) -> None:
        parser = build_root_cli_parser()
        args = parser.parse_args(
            [
                "doctor",
                "--level",
                "station",
                "--config",
                "configs/station_level/config.yaml",
            ]
        )
        self.assertEqual(args.task, "doctor")
        self.assertEqual(args.level, "station")

    def test_station_model_runtime_report_marks_experimental_deep_model(self) -> None:
        config = SimpleNamespace(
            baselines_enabled={"naive": True, "seasonal_naive_7": True},
            tree_enabled={"lgbm": True, "xgboost": True},
            deepar_enabled=True,
            tune_enabled=False,
        )
        report = station_model_runtime_report(config)
        deepar_row = report.loc[report["model_name"] == "deepar"].iloc[0]
        self.assertEqual(deepar_row["implementation"], "global_neural_mlp")
        self.assertTrue(bool(deepar_row["experimental"]))

    def test_system_model_runtime_report_marks_tree_tuning_as_fixed_defaults(self) -> None:
        config = SimpleNamespace(
            baselines_enabled={"naive": True, "seasonal_naive_7": False, "seasonal_naive_30": False},
            classical_enabled={"ets": False, "sarimax_dynamic": False, "fourier_dynamic_regression": False, "unobserved_components": False},
            ml_enabled={"tree_boosting": True},
        )
        report = system_model_runtime_report(config)
        row = report.loc[report["model_name"] == "tree_boosting"].iloc[0]
        self.assertEqual(row["tuning_strategy"], "fixed_defaults")

    def test_progress_logger_prints_timestamped_messages(self) -> None:
        logger = make_progress_logger(True, prefix="station-forecast")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            logger("Starting backtests.")
        output = buffer.getvalue()
        self.assertIn("station-forecast Starting backtests.", output)
        self.assertIn("[", output)

    def test_station_pipeline_emits_progress_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directories = {
                "feature_artifacts": root / "feature_artifacts",
                "backtests": root / "backtests",
                "metrics": root / "metrics",
                "models": root / "models",
                "forecasts": root / "forecasts",
                "figures": root / "figures",
            }
            for path in directories.values():
                path.mkdir(parents=True, exist_ok=True)

            artifacts = {
                "panel": pd.DataFrame({"station_id": ["A"], "date": pd.to_datetime(["2024-01-01"]), "target": [1.0], "in_service": [True]}),
                "slice_lookup": pd.DataFrame({"station_id": ["A"]}),
                "feature_frame": pd.DataFrame({"station_id": ["A"], "date": pd.to_datetime(["2024-01-01"]), "target": [1.0]}),
                "observed_daily": pd.DataFrame({"station_id": ["A"], "date": pd.to_datetime(["2024-01-01"]), "target": [1.0]}),
            }
            backtest_outputs = {
                "interval_calibration": pd.DataFrame({"model_name": ["naive"], "horizon": [7], "residual_q10": [-1.0], "residual_q90": [1.0]}),
                "enabled_model_keys": ["naive"],
                "tuning_enabled": False,
                "backtest_metrics": pd.DataFrame({"model_name": ["naive"], "fold_id": [1], "horizon": [7], "mae": [1.0], "rmse": [1.0], "mase": [1.0], "bias": [0.0], "rows": [1], "stations": [1]}),
                "backtest_summary": pd.DataFrame({"model_name": ["naive"], "horizon": [7], "mae": [1.0]}),
            }
            production_outputs = {
                "future_forecasts": pd.DataFrame({"station_id": ["A"], "date": pd.to_datetime(["2024-01-02"]), "prediction": [1.0], "model_name": ["naive"]}),
            }
            config = SimpleNamespace(output_root=root)
            progress_messages: list[str] = []

            with (
                mock.patch("station_level.forecasting.pipeline.ensure_output_directories", return_value=directories),
                mock.patch("station_level.forecasting.pipeline.build_station_level_artifacts", return_value=artifacts),
                mock.patch("station_level.forecasting.pipeline.write_station_level_feature_artifacts"),
                mock.patch("station_level.forecasting.pipeline.write_station_level_runtime_outputs"),
                mock.patch("station_level.forecasting.pipeline.run_station_level_backtest_stage", return_value=backtest_outputs),
                mock.patch("station_level.forecasting.pipeline.write_station_level_backtest_outputs"),
                mock.patch("station_level.forecasting.pipeline.run_station_level_production_stage", return_value=production_outputs),
                mock.patch("station_level.forecasting.pipeline.write_station_level_production_outputs"),
                mock.patch("station_level.forecasting.pipeline.plot_model_comparison", return_value=directories["figures"] / "station_level_model_comparison.png"),
            ):
                summary = run_station_level_pipeline(config, model="all", tune=False, progress=progress_messages.append)

            self.assertEqual(summary["forecast_rows"], 1)
            self.assertTrue(any("Writing runtime diagnostics." in message for message in progress_messages))
            self.assertTrue(any("Preparing station-level artifacts." in message for message in progress_messages))
            self.assertTrue(any("Writing backtest outputs." in message for message in progress_messages))
            self.assertTrue(any("Station-level forecasting pipeline complete." in message for message in progress_messages))


if __name__ == "__main__":
    unittest.main()
