from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.core.alerts import AlertDispatcher


class AlertDispatcherTests(unittest.TestCase):
    @staticmethod
    def write(path: Path, payload: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def test_actionable_reports_are_delivered_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            reports = root / "reports"
            competition = state / "competition" / "latest.json"
            self.write(
                competition,
                {
                    "status": "COMPETITION_ACTION_REQUIRED",
                    "generated_at": "2026-07-02T15:20:00+00:00",
                    "current_round": 1302,
                    "current_round_coverage": {
                        "missing_models": ["kvmmn", "kvmmn_fn"],
                    },
                },
            )
            self.write(
                reports / "20260702_151000_stake-status.json",
                {
                    "status": "STAKE_POLICY_ACTION_REQUIRED",
                    "violations": [
                        {
                            "model_name": "kvmmn",
                            "code": "SHADOW_MODEL_HAS_STAKE",
                        }
                    ],
                },
            )
            notifications = []
            dispatcher = AlertDispatcher(
                state,
                reports,
                notifier=lambda title, message: notifications.append(
                    (title, message)
                ),
            )
            first = dispatcher.dispatch()
            self.assertEqual(first["status"], "ALERTS_DELIVERED")
            self.assertEqual(len(first["delivered"]), 2)
            self.assertEqual(len(notifications), 2)

            second = dispatcher.dispatch()
            self.assertEqual(second["status"], "NO_NEW_ALERTS")
            self.assertEqual(len(notifications), 2)

            payload = json.loads(competition.read_text())
            payload["generated_at"] = "2026-07-03T15:20:00+00:00"
            competition.write_text(json.dumps(payload))
            third = dispatcher.dispatch()
            self.assertEqual(third["status"], "ALERTS_DELIVERED")
            self.assertEqual(len(third["delivered"]), 1)
            self.assertEqual(len(notifications), 3)

    def test_failed_delivery_is_recorded_and_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            reports = root / "reports"
            self.write(
                state / "competition" / "latest.json",
                {
                    "status": "COMPETITION_ACTION_REQUIRED",
                    "generated_at": "2026-07-02T15:20:00+00:00",
                    "current_round": 1302,
                    "current_round_coverage": {
                        "missing_models": ["kvmmn"],
                    },
                },
            )
            attempts = []

            def fail(title, message):
                attempts.append((title, message))
                raise RuntimeError("notifications unavailable")

            dispatcher = AlertDispatcher(state, reports, notifier=fail)
            first = dispatcher.dispatch()
            self.assertFalse(first["ok"])
            self.assertEqual(first["status"], "ALERT_DELIVERY_FAILED")
            second = dispatcher.dispatch()
            self.assertEqual(second["status"], "ALERT_DELIVERY_FAILED")
            self.assertEqual(len(attempts), 2)
            ledger = json.loads(
                (state / "alerts" / "ledger.json").read_text()
            )
            self.assertEqual(ledger["delivered"], {})
            self.assertEqual(len(ledger["failures"]), 2)


if __name__ == "__main__":
    unittest.main()
