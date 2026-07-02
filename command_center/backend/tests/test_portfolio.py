from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app.core.portfolio import (
    PortfolioControlPlane,
    PortfolioError,
    activate_portfolio,
    approve_portfolio_proposal,
    create_portfolio_proposal,
)


class _FakeApi:
    def get_current_round(self):
        return 1302

    def get_models(self):
        return {"kvmmn": "model-a", "kvmmn_fn": "model-b"}


class PortfolioGovernanceTests(unittest.TestCase):
    def _artifact(self, root: Path, name: str, content: bytes) -> Path:
        path = root / name
        path.write_bytes(content)
        return path

    def test_rejects_duplicate_artifacts_across_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = self._artifact(root, "model.pkl", b"same")
            with self.assertRaisesRegex(PortfolioError, "distinct artifacts"):
                create_portfolio_proposal(
                    root,
                    [
                        {"model_name": "kvmmn", "artifact_path": artifact},
                        {"model_name": "kvmmn_fn", "artifact_path": artifact},
                    ],
                )

    def test_approval_and_activation_are_challenge_and_checksum_bound(self):
        now = datetime(2026, 7, 2, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = self._artifact(root, "model.pkl", b"candidate")
            proposal, proposal_path = create_portfolio_proposal(
                root,
                [{"model_name": "kvmmn", "artifact_path": artifact, "role": "core"}],
                now=now,
            )
            with self.assertRaisesRegex(PortfolioError, "challenge"):
                approve_portfolio_proposal(
                    proposal_path,
                    challenge="WRONG",
                    actor="operator",
                    now=now,
                )
            approve_portfolio_proposal(
                proposal_path,
                challenge=proposal["approval_challenge"],
                actor="operator",
                now=now,
            )
            current_path = activate_portfolio(root, proposal_path, now=now)
            current = json.loads(current_path.read_text())
            self.assertEqual(current["approved_by"], "operator")
            self.assertEqual(
                current["assignments"][0]["artifact"]["sha256"],
                hashlib.sha256(b"candidate").hexdigest(),
            )

    def test_prepare_all_skips_verified_round_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry"
            state = root / "state"
            artifact = self._artifact(root, "model.pkl", b"candidate")
            current = {
                "schema_version": 1,
                "assignments": [
                    {
                        "model_name": "kvmmn",
                        "role": "core",
                        "artifact": {
                            "path": str(artifact),
                            "sha256": hashlib.sha256(b"candidate").hexdigest(),
                            "size_bytes": len(b"candidate"),
                        },
                        "source_bundle_id": None,
                    }
                ],
            }
            current_path = registry / "portfolio" / "current.json"
            current_path.parent.mkdir(parents=True)
            current_path.write_text(json.dumps(current))
            ledger_path = state / "submission_ledger.json"
            ledger_path.parent.mkdir(parents=True)
            ledger_path.write_text(
                json.dumps(
                    {
                        "submissions": {
                            "1302:model-a": {
                                "round_number": 1302,
                                "model_id": "model-a",
                                "verified": True,
                            }
                        }
                    }
                )
            )
            with patch(
                "app.core.portfolio.AgenticControlPlane.prepare"
            ) as prepare:
                result = PortfolioControlPlane(
                    registry_dir=registry,
                    state_dir=state,
                    napi=_FakeApi(),
                ).prepare_all()
            self.assertTrue(result["ok"])
            self.assertEqual(result["status"], "PORTFOLIO_PARTIAL_COVERAGE")
            self.assertEqual(result["unassigned_count"], 1)
            self.assertFalse(result["coverage_complete"])
            self.assertEqual(result["results"][1]["status"], "UNASSIGNED")
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
