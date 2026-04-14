from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from metro_bike_share_forecasting.station_level.diagnosis.model_readiness import (  # noqa: E402
    build_station_modeling_readiness_package,
)


class StationLevelModelingReadinessTests(unittest.TestCase):
    def test_modeling_readiness_builds_outputs_from_diagnosis_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "diagnosis" / "station_level_analysis" / "outputs"
            tables = root / "tables"
            reports = root / "reports"
            tables.mkdir(parents=True, exist_ok=True)
            reports.mkdir(parents=True, exist_ok=True)

            summary = pd.DataFrame(
                {
                    "station_id": ["A", "B", "C", "D", "E", "F"],
                    "history_group": ["mature", "mature", "mature", "young", "mature", "mature"],
                    "station_category": [
                        "busy_stable",
                        "sparse_intermittent",
                        "mixed_profile",
                        "short_history",
                        "anomaly_heavy",
                        "weekend_leisure",
                    ],
                    "cluster_label": ["cluster_1", "cluster_2", "cluster_1", "not_clustered_short_history", "cluster_2", "cluster_3"],
                    "avg_demand": [40.0, 2.0, 18.0, 1.0, 12.0, 15.0],
                    "zero_rate": [0.05, 0.82, 0.20, 0.60, 0.10, 0.12],
                    "active_day_rate": [0.95, 0.18, 0.80, 0.40, 0.90, 0.88],
                    "outlier_rate": [0.01, 0.02, 0.03, 0.00, 0.11, 0.02],
                }
            )
            summary.to_csv(tables / "station_summary_with_clusters.csv", index=False)

            pd.DataFrame(
                {
                    "station_category": [
                        "busy_stable",
                        "sparse_intermittent",
                        "mixed_profile",
                        "short_history",
                        "anomaly_heavy",
                        "weekend_leisure",
                    ],
                    "station_count": [1, 1, 1, 1, 1, 1],
                }
            ).to_csv(tables / "station_category_summary.csv", index=False)

            pd.DataFrame(
                {
                    "cluster_label": ["cluster_1", "cluster_2", "cluster_3"],
                    "station_count": [2, 2, 1],
                    "avg_demand_mean": [29.0, 7.0, 15.0],
                }
            ).to_csv(tables / "station_cluster_profile.csv", index=False)

            pd.DataFrame(
                {
                    "candidate_k": [4, 5],
                    "silhouette_score": [0.21, 0.16],
                    "tiny_cluster_count": [0, 1],
                    "selected": [True, False],
                }
            ).to_csv(tables / "cluster_model_selection.csv", index=False)

            (reports / "station_level_diagnosis_summary.md").write_text("# diagnosis\n")

            written = build_station_modeling_readiness_package(root)

            recommendations_path = Path(written["modeling_group_recommendations"])
            report_path = Path(written["modeling_readiness_summary"])

            self.assertTrue(recommendations_path.exists())
            self.assertTrue(report_path.exists())

            recommendations = pd.read_csv(recommendations_path)
            self.assertIn("group_name", recommendations.columns)
            self.assertIn("recommended_modeling_treatment", recommendations.columns)
            self.assertIn("baseline_recommendation", recommendations.columns)
            self.assertIn("all_station_day_panel", recommendations["group_name"].tolist())

            report_text = report_path.read_text()
            self.assertIn("station-day", report_text)
            self.assertIn("DeepAR", report_text)
            self.assertIn("one global model", report_text.lower())


if __name__ == "__main__":
    unittest.main()
