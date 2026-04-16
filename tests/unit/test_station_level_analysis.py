from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from metro_bike_share_forecasting.cli import main as cli_main
from metro_bike_share_forecasting.station_level.diagnosis.config import StationDiagnosisConfig
from metro_bike_share_forecasting.station_level.diagnosis.pipeline import build_station_level_diagnosis
from metro_bike_share_forecasting.station_level.forecasting.config import StationLevelForecastConfig
from metro_bike_share_forecasting.station_level.forecasting.pipeline import run_station_level_pipeline


class StationLevelDiagnosisTests(unittest.TestCase):
    def _build_station_frame(self) -> pd.DataFrame:
        dates = pd.date_range("2023-01-01", periods=420, freq="D")
        rng = np.random.default_rng(42)
        rows: list[dict[str, object]] = []

        profiles = {
            "A": lambda i, d: 42 + 6 * np.sin(2 * np.pi * (d.dayofweek / 7)) + (6 if d.dayofweek < 5 else -4),
            "B": lambda i, d: 28 + 12 * np.sin(2 * np.pi * (i / 30)),
            "C": lambda i, d: 5 if d.dayofweek < 5 else 14,
            "D": lambda i, d: 0 if i % 3 else 3,
            "E": lambda i, d: 20 + (18 if i % 45 == 0 else 0),
            "F": lambda i, d: 9 + 0.02 * i,
            "G": lambda i, d: 16 + 4 * np.cos(2 * np.pi * (i / 14)),
            "H": lambda i, d: 12 + (7 if d.month in {6, 7, 8} else 0),
        }

        for station_id, generator in profiles.items():
            for i, date_value in enumerate(dates):
                level = max(0.0, float(generator(i, date_value)) + float(rng.normal(0, 1.25)))
                rows.append(
                    {
                        "date": date_value,
                        "station_id": station_id,
                        "target": round(level, 2),
                    }
                )

        frame = pd.DataFrame(rows)
        short_history_dates = pd.date_range("2024-01-01", periods=45, freq="D")
        short_history = pd.DataFrame(
            {
                "date": short_history_dates,
                "station_id": "Z",
                "target": [0, 0, 1, 0, 0] * 9,
            }
        )
        return pd.concat([frame, short_history], ignore_index=True)

    def test_station_level_diagnosis_builds_revised_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "station_daily.csv"
            output_dir = root / "station_level_analysis" / "outputs"

            frame = self._build_station_frame()
            frame.to_csv(input_path, index=False)

            written = build_station_level_diagnosis(
                input_path=input_path,
                date_col="date",
                station_col="station_id",
                target_col="target",
                config=StationDiagnosisConfig(
                    output_root=output_dir,
                    top_n=3,
                    n_clusters=3,
                    cluster_k_values=(2, 3),
                    expected_station_count=8,
                ),
            )

            required_keys = {
                "station_inventory",
                "station_summary_csv",
                "station_summary_parquet",
                "station_category_summary",
                "station_cluster_profile",
                "cluster_model_selection",
                "summary_with_clusters",
                "category_count_bar_chart",
                "history_days_histogram",
                "cluster_profile_heatmap",
            }
            self.assertTrue(required_keys.issubset(written.keys()))
            for key in required_keys:
                self.assertTrue(Path(written[key]).exists(), msg=f"Missing output for {key}")

            summary = pd.read_csv(written["station_summary_csv"])
            self.assertIn("station_category", summary.columns)
            self.assertIn("history_group", summary.columns)
            self.assertNotIn("cluster_label", summary.columns)

            summary_with_clusters = pd.read_csv(written["summary_with_clusters"])
            self.assertIn("cluster_label", summary_with_clusters.columns)
            self.assertIn("metrics_limited_by_sparsity", summary_with_clusters.columns)
            self.assertEqual(summary_with_clusters["station_id"].nunique(), 9)

            inventory = pd.read_csv(written["station_inventory"])
            self.assertIn("is_zero_almost_always", inventory.columns)
            self.assertIn("appears_active_recently", inventory.columns)

            cluster_selection = pd.read_csv(written["cluster_model_selection"])
            self.assertIn("selected", cluster_selection.columns)
            self.assertGreaterEqual(len(cluster_selection), 1)

    def test_station_forecast_cli_requires_all_models(self) -> None:
        with self.assertRaises(SystemExit) as captured:
            cli_main(["forecast", "--level", "station", "--model", "baseline"])
        self.assertEqual(str(captured.exception), "Station forecast CLI currently supports only --model all.")

    def test_station_level_forecast_runs_all_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            aggregate_path = root / "daily_aggregate.csv"
            output_root = root / "forecasts" / "station_level"
            diagnosis_path = root / "station_summary_with_clusters.csv"

            dates = pd.date_range("2023-01-01", periods=170, freq="D")
            stations = ["3001", "3002", "3003", "3004"]
            rows: list[dict[str, object]] = []
            for station_index, station_id in enumerate(stations):
                for day_index, date_value in enumerate(dates):
                    rows.append(
                        {
                            "bucket_start": date_value,
                            "segment_id": station_id,
                            "trip_count": max(
                                0.0,
                                12
                                + 1.5 * station_index
                                + 2.5 * np.sin(2 * np.pi * (day_index / 7))
                                + (2.0 if date_value.dayofweek < 5 else -1.0),
                            ),
                            "segment_type": "start_station",
                            "is_observed": True,
                        }
                    )
            pd.DataFrame(rows).to_csv(aggregate_path, index=False)

            pd.DataFrame(
                {
                    "station_id": stations,
                    "history_group": ["mature"] * 4,
                    "station_category": ["mixed_profile"] * 4,
                    "cluster_label": ["cluster_1", "cluster_1", "cluster_2", "cluster_2"],
                    "is_short_history": [False] * 4,
                    "is_zero_almost_always": [False] * 4,
                    "appears_active_recently": [True] * 4,
                }
            ).to_csv(diagnosis_path, index=False)

            config = StationLevelForecastConfig(
                project_root=root,
                daily_aggregate_path=aggregate_path,
                diagnosis_summary_path=diagnosis_path,
                date_column="bucket_start",
                station_column="segment_id",
                target_column="trip_count",
                segment_type="start_station",
                in_service_column="is_observed",
                forecast_horizons=(7, 30),
                extended_horizon=90,
                lags=(1, 7, 14),
                rolling_windows=(7, 28),
                holiday_country="US",
                include_category_feature=False,
                include_cluster_feature=False,
                initial_train_size=90,
                step_size=20,
                max_folds=2,
                recent_activity_window_days=60,
                min_recent_service_days=3,
                baselines_enabled={"naive": True, "seasonal_naive_7": True},
                tree_enabled={"lgbm": True, "xgboost": True},
                deepar_enabled=True,
                random_state=42,
                tune_enabled=False,
                deepar_context_length=28,
                deepar_hidden_size=8,
                deepar_embedding_dim=4,
                deepar_batch_size=256,
                deepar_epochs=1,
                deepar_learning_rate=0.001,
                output_root=output_root,
            )

            summary = run_station_level_pipeline(config, model="all", tune=False)
            self.assertGreater(summary["forecast_rows"], 0)

            comparison = pd.read_csv(output_root / "metrics" / "station_level_model_comparison.csv")
            self.assertTrue({"naive", "seasonal_naive_7", "lgbm", "xgboost", "deepar"}.issubset(set(comparison["model_name"])))

            future = pd.read_csv(output_root / "forecasts" / "station_level_future_forecasts.csv")
            self.assertTrue({"naive", "seasonal_naive_7", "lgbm", "xgboost", "deepar"}.issubset(set(future["model_name"])))

            manifest = pd.read_json(output_root / "models" / "station_level_run_manifest.json", typ="series")
            self.assertEqual(manifest["requested_model"], "all")
            self.assertTrue((output_root / "models" / "station_level_model_registry.csv").exists())


if __name__ == "__main__":
    unittest.main()
