from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.core.daily_step import FORBIDDEN, choose_daily_step

LIVE_ERA = [
    {
        "name": "kvmmn",
        "era_mean_mmc": 0.00231,
        "recent_mean_mmc": -0.00779,
    },
    {
        "name": "kvmmn_te",
        "era_mean_mmc": -0.00143,
        "recent_mean_mmc": -0.00485,
    },
    {
        "name": "kvmmn_fn",
        "era_mean_mmc": 0.00391,
        "recent_mean_mmc": -0.00195,
    },
]


class DailyStepTests(unittest.TestCase):
    def test_live_era_bars_against_fn_and_forbids_stake(self):
        step = choose_daily_step(LIVE_ERA)

        self.assertTrue(step["ok"])
        self.assertEqual(step["step"], "compare_challenger")
        self.assertEqual(step["bar_slot"], "kvmmn_fn")
        self.assertEqual(step["weak_slot"], "kvmmn_te")
        self.assertEqual(step["experiment"], "run_seed_neutralization_sweep")
        self.assertEqual(step["regime"], "mmc_negative")
        for action in ("stake", "full-auto", "mcp-submit", "agent-submit"):
            self.assertIn(action, step["forbidden"])
        self.assertEqual(tuple(step["forbidden"]), FORBIDDEN)

    def test_missing_scores_do_not_invent_a_slot(self):
        step = choose_daily_step([])

        self.assertFalse(step["ok"])
        self.assertEqual(step["status"], "DAILY_STEP_NEEDS_SCORES")
        self.assertIsNone(step["bar_slot"])
        self.assertIn("stake", step["forbidden"])

    def test_runner_reads_a_score_file_without_training(self):
        from automation.daily_numerai_run import run_mode
        import argparse

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.json"
            path.write_text(json.dumps({"slots": LIVE_ERA}))
            result = run_mode(
                "daily-step",
                argparse.Namespace(scores_path=str(path)),
            )

        self.assertEqual(result["bar_slot"], "kvmmn_fn")
        self.assertNotIn("submission_id", result)
