from __future__ import annotations

import unittest

from dashboard.actions import action_for_page


class DashboardActionsTests(unittest.TestCase):
    def test_system_forecast_action_uses_canonical_cli(self) -> None:
        action = action_for_page("system_forecast")
        command = list(action.commands[0])
        self.assertIn("system_level.cli", command)
        self.assertIn("forecast", command)
        self.assertIn("system", command)

    def test_overview_action_runs_all_four_workflows(self) -> None:
        action = action_for_page("overview")
        self.assertEqual(len(action.commands), 4)


if __name__ == "__main__":
    unittest.main()
