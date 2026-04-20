from __future__ import annotations

import unittest
from pathlib import Path


class DashboardPagesTests(unittest.TestCase):
    def _source(self, relative_path: str) -> str:
        root = Path(__file__).resolve().parents[2]
        return (root / relative_path).read_text(encoding="utf-8")

    def test_system_diagnosis_page_omits_internal_metric_cards(self) -> None:
        source = self._source("src/dashboard/pages/system_diagnosis.py")
        self.assertNotIn("Primary period", source)
        self.assertNotIn("Trend strength", source)
        self.assertNotIn("Seasonal strength", source)
        self.assertNotIn('"Shift windows"', source)
        self.assertNotIn("metric_cards(", source)

    def test_station_forecast_appendix_guidance_is_removed(self) -> None:
        source = self._source("src/dashboard/pages/station_forecast.py")
        self.assertNotIn("Appendix guidance", source)
        self.assertNotIn("Keep diagnosis visuals off this page except where a bridge sentence is needed.", source)
        self.assertNotIn("Move category discovery, cluster heatmaps, and all-model forecast dumps to appendix or diagnosis pages.", source)
        self.assertNotIn("The final production decision should focus on overall ranking, slice metrics, representative examples, and the recommendation block.", source)

    def test_dashboard_html_blocks_are_dedented_before_rendering(self) -> None:
        components_source = self._source("src/dashboard/components.py")
        actions_source = self._source("src/dashboard/actions.py")
        app_source = self._source("src/dashboard/app.py")
        self.assertIn("from textwrap import dedent", components_source)
        self.assertIn("_render_html(", components_source)
        self.assertIn("from textwrap import dedent", actions_source)
        self.assertIn("from textwrap import dedent", app_source)


if __name__ == "__main__":
    unittest.main()
