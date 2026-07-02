from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.core.competition import CompetitionTracker


NOW = datetime(2026, 7, 2, 16, 0, tzinfo=timezone.utc)


class FakeCompetitionApi:
    def __init__(self, *, account_rank_found: bool = True):
        self.tournament_id = 8
        self.account_rank_found = account_rank_found
        self.models = {
            "kvmmn": "model-a",
            "kvmmn_fn": "model-b",
            "kvmmn_te": "model-c",
        }

    def get_account(self):
        return {"username": "kvmmn"}

    def get_models(self):
        return dict(self.models)

    def get_current_round(self):
        return 1302

    def check_round_open(self):
        return True

    def get_account_leaderboard(self, *, limit, offset):
        rows = [
            {
                "username": "leader",
                "rank": 1,
                "nmrStaked": 100,
            }
        ]
        if self.account_rank_found and offset == 0:
            rows.append(
                {
                    "username": "kvmmn",
                    "rank": 42,
                    "nmrStaked": 1.2,
                    "v2Corr20": 0.01,
                    "mmc": 0.002,
                }
            )
        return rows

    def daily_model_performances(self, model_name):
        return [
            {
                "date": datetime(2025, 1, 1, tzinfo=timezone.utc),
                "corrRank": 500,
                "corrRep": 0.001,
                "mmcRank": 600,
                "mmcRep": 0.0,
                "fncV3Rank": 700,
                "fncV3Rep": 0.001,
                "tcRank": 800,
                "tcRep": 0.001,
            },
            {
                "date": datetime(2026, 6, 30, tzinfo=timezone.utc),
                "corrRank": 100,
                "corrRep": 0.01,
                "mmcRank": 120,
                "mmcRep": 0.003,
                "fncV3Rank": 90,
                "fncV3Rep": 0.012,
                "tcRank": 140,
                "tcRep": 0.004,
            },
        ]

    def round_model_performances_v2(self, model_id):
        rows = {
            "model-a": [
                {
                    "roundNumber": 1290,
                    "roundOpenTime": datetime(
                        2026, 6, 1, 13, 0, tzinfo=timezone.utc
                    ),
                    "roundResolved": True,
                    "atRisk": 0.6,
                    "submissionScores": [],
                }
            ],
            "model-b": [
                {
                    "roundNumber": 1290,
                    "roundOpenTime": datetime(
                        2026, 6, 1, 13, 0, tzinfo=timezone.utc
                    ),
                    "roundResolved": True,
                    "atRisk": 0.5,
                    "submissionScores": [],
                }
            ],
            "model-c": [
                {
                    "roundNumber": 1291,
                    "roundOpenTime": datetime(
                        2026, 6, 2, 13, 0, tzinfo=timezone.utc
                    ),
                    "roundResolved": True,
                    "atRisk": 0.4,
                    "submissionScores": [],
                }
            ],
        }
        return rows[model_id]


class CompetitionTrackerTests(unittest.TestCase):
    @staticmethod
    def write_portfolio(registry: Path):
        path = registry / "portfolio" / "current.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "assignments": [
                        {
                            "model_name": "kvmmn",
                            "deployment_tier": "shadow",
                            "stake_eligible": False,
                        },
                        {
                            "model_name": "kvmmn_fn",
                            "deployment_tier": "shadow",
                            "stake_eligible": False,
                        },
                        {
                            "model_name": "kvmmn_te",
                            "deployment_tier": "production",
                            "stake_eligible": False,
                        },
                    ]
                }
            )
        )

    @staticmethod
    def write_ledger(state: Path, model_ids: list[str]):
        submissions = {}
        for model_id in model_ids:
            rounds = [1300, 1301, 1302] if model_id == "model-c" else [1302]
            for round_number in rounds:
                submissions[f"{round_number}:{model_id}"] = {
                    "round_number": round_number,
                    "model_id": model_id,
                    "submission_id": f"submission-{round_number}-{model_id}",
                    "run_id": f"run-{round_number}-{model_id}",
                    "verified": True,
                }
        path = state / "submission_ledger.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"submissions": submissions}))

    def test_reports_current_coverage_streak_rank_and_season_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            registry = root / "registry"
            self.write_portfolio(registry)
            self.write_ledger(state, ["model-c"])
            result = CompetitionTracker(
                state,
                registry,
                napi=FakeCompetitionApi(),
                now=NOW,
            ).snapshot()
            self.assertEqual(result["status"], "COMPETITION_ACTION_REQUIRED")
            self.assertEqual(
                result["current_round_coverage"]["submitted_verified"],
                1,
            )
            self.assertEqual(
                result["current_round_coverage"]["missing_models"],
                ["kvmmn", "kvmmn_fn"],
            )
            self.assertEqual(
                result["season_qualification"]["qualified_rounds"],
                [1290],
            )
            self.assertEqual(
                result["season_qualification"]["qualified_round_count"],
                1,
            )
            self.assertEqual(
                result["season_qualification"]["rounds_remaining"],
                19,
            )
            self.assertEqual(result["account_leaderboard"]["rank"], 42)
            production = next(
                row for row in result["models"] if row["model_name"] == "kvmmn_te"
            )
            self.assertEqual(production["local_submission_streak"], 3)
            self.assertEqual(
                production["leaderboard"]["as_of"],
                "2026-06-30T00:00:00+00:00",
            )
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertTrue(Path(result["latest_path"]).exists())

    def test_complete_coverage_and_rank_lower_bound_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            registry = root / "registry"
            self.write_portfolio(registry)
            self.write_ledger(state, ["model-a", "model-b", "model-c"])
            result = CompetitionTracker(
                state,
                registry,
                napi=FakeCompetitionApi(account_rank_found=False),
                now=NOW,
                account_rank_scan_limit=500,
            ).snapshot()
            self.assertEqual(
                result["status"],
                "COMPETITION_CURRENT_ROUND_COMPLETE",
            )
            self.assertTrue(result["current_round_coverage"]["complete"])
            self.assertFalse(result["account_leaderboard"]["found"])
            self.assertEqual(
                result["account_leaderboard"]["rank_lower_bound"],
                result["account_leaderboard"]["scanned"] + 1,
            )
            self.assertIn(
                "ACCOUNT_NOT_IN_SCANNED_LEADERBOARD",
                result["strategic_gaps"],
            )


if __name__ == "__main__":
    unittest.main()
