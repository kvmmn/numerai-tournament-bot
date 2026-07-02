from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.core.performance_listener import PerformanceListener


class FakeNumerai:
    def __init__(self):
        self.rows = [
            {
                "roundNumber": 100,
                "roundResolved": True,
                "atRisk": "1",
                "submissionScores": [
                    {"displayName": "canon_corr", "value": 0.02},
                    {"displayName": "canon_mmc", "value": 0.01},
                ],
            }
        ]

    def get_models(self):
        return {"model": "model-id"}

    def round_model_performances_v2(self, model_id):
        return list(self.rows)


class PerformanceListenerTests(unittest.TestCase):
    def test_initializes_then_emits_new_outcome_and_postmortem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            napi = FakeNumerai()
            listener = PerformanceListener(temp_dir, napi=napi)
            initial = listener.poll()
            self.assertEqual(initial["status"], "INITIALIZED")
            self.assertEqual(initial["events"], [])

            napi.rows.append(
                {
                    "roundNumber": 101,
                    "roundResolved": True,
                    "atRisk": "1",
                    "submissionScores": [
                        {"displayName": "canon_corr", "value": -0.03},
                        {"displayName": "canon_mmc", "value": -0.01},
                    ],
                }
            )
            second = listener.poll()
            self.assertEqual(second["status"], "POSTMORTEM_REQUIRED")
            self.assertEqual(len(second["events"]), 1)
            self.assertTrue(second["postmortem_reasons"])
            self.assertTrue(second["postmortem_created"])
            self.assertTrue(second["postmortem_path"])
            self.assertTrue(Path(second["postmortem_path"]).exists())
            self.assertTrue(second["read_only"])

            repeated = listener.poll()
            self.assertEqual(repeated["status"], "NO_NEW_OUTCOMES")
            self.assertFalse(repeated["postmortem_created"])


if __name__ == "__main__":
    unittest.main()
