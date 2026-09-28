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
                        "deployment_tier": "production",
                        "stake_eligible": False,
                        "artifact": {
                            "path": str(artifact),
                            "sha256": hashlib.sha256(b"candidate").hexdigest(),
                            "size_bytes": len(b"candidate"),
                        },
                        "evidence": None,
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
                plane = PortfolioControlPlane(
                    registry_dir=registry,
                    state_dir=state,
                    napi=_FakeApi(),
                )
                result = plane.prepare_all()
            self.assertTrue(result["ok"])
            self.assertEqual(result["status"], "PORTFOLIO_PARTIAL_COVERAGE")
            self.assertEqual(result["unassigned_count"], 1)
            self.assertFalse(result["coverage_complete"])
            self.assertEqual(result["results"][1]["status"], "UNASSIGNED")
            prepare.assert_not_called()
            inspected = plane.inspect()
            self.assertTrue(inspected["read_only"])
            self.assertEqual(
                inspected["status"],
                "DEADLINE_GUARD_PARTIAL_COVERAGE",
            )
            self.assertEqual(
                [slot["status"] for slot in inspected["slots"]],
                ["SUBMITTED_VERIFIED", "UNASSIGNED"],
            )

    def test_prepare_all_syncs_round_data_once_for_multiple_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry"
            state = root / "state"
            assignments = []
            for model_name, content in (
                ("kvmmn", b"candidate-a"),
                ("kvmmn_fn", b"candidate-b"),
            ):
                artifact = self._artifact(root, f"{model_name}.pkl", content)
                assignments.append(
                    {
                        "model_name": model_name,
                        "role": "core",
                        "deployment_tier": "production",
                        "stake_eligible": False,
                        "artifact": {
                            "path": str(artifact),
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "size_bytes": len(content),
                        },
                        "evidence": None,
                        "source_bundle_id": None,
                    }
                )
            current_path = registry / "portfolio" / "current.json"
            current_path.parent.mkdir(parents=True)
            current_path.write_text(
                json.dumps({"schema_version": 1, "assignments": assignments})
            )
            shared_report = {"round_number": 1302, "live_rows": 7_164}
            with (
                patch(
                    "app.core.portfolio.sync_datasets",
                    return_value=shared_report,
                ) as sync,
                patch(
                    "app.core.portfolio.AgenticControlPlane.prepare",
                    side_effect=[
                        {"ok": True, "status": "AWAITING_HUMAN_APPROVAL"},
                        {"ok": True, "status": "AWAITING_HUMAN_APPROVAL"},
                    ],
                ) as prepare,
            ):
                result = PortfolioControlPlane(
                    registry_dir=registry,
                    state_dir=state,
                    napi=_FakeApi(),
                ).prepare_all()
            self.assertTrue(result["ok"])
            sync.assert_called_once()
            self.assertEqual(prepare.call_count, 2)
            self.assertEqual(
                [row.kwargs["data_report"] for row in prepare.call_args_list],
                [shared_report, shared_report],
            )

    def test_auto_submit_requires_explicit_enablement(self):
        with patch("app.core.portfolio.settings.AUTO_SUBMIT_PORTFOLIO", False):
            with self.assertRaisesRegex(PortfolioError, "disabled"):
                PortfolioControlPlane(napi=_FakeApi()).auto_submit_all()

    def test_auto_submit_all_prepares_approves_and_submits_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry"
            state = root / "state"
            artifact = self._artifact(root, "model.pkl", b"candidate")
            current_path = registry / "portfolio" / "current.json"
            current_path.parent.mkdir(parents=True)
            current_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "assignments": [
                            {
                                "model_name": "kvmmn",
                                "role": "core",
                                "deployment_tier": "production",
                                "stake_eligible": False,
                                "artifact": {
                                    "path": str(artifact),
                                    "sha256": hashlib.sha256(b"candidate").hexdigest(),
                                    "size_bytes": len(b"candidate"),
                                },
                                "evidence": None,
                                "source_bundle_id": None,
                            }
                        ],
                    }
                )
            )
            with (
                patch("app.core.portfolio.settings.AUTO_SUBMIT_PORTFOLIO", True),
                patch(
                    "app.core.portfolio.sync_datasets",
                    return_value={"downloaded": ["live.parquet"]},
                ),
                patch(
                    "app.core.portfolio.AgenticControlPlane.prepare",
                    return_value={
                        "ok": True,
                        "status": "AWAITING_HUMAN_APPROVAL",
                        "run_id": "run-a",
                        "approval_challenge": "CHALLENGE",
                        "validation": {"stake_eligible": False},
                    },
                ) as prepare,
                patch(
                    "app.core.portfolio.AgenticControlPlane.approve",
                    return_value={"ok": True, "status": "APPROVED"},
                ) as approve,
                patch(
                    "app.core.portfolio.AgenticControlPlane.submit",
                    return_value={
                        "ok": True,
                        "status": "SUBMITTED_VERIFIED",
                        "submission_id": "submission-a",
                        "verified": True,
                    },
                ) as submit,
            ):
                result = PortfolioControlPlane(
                    registry_dir=registry,
                    state_dir=state,
                    napi=_FakeApi(),
                ).auto_submit_all(actor="automation")
            self.assertTrue(result["ok"])
            self.assertEqual(result["status"], "PORTFOLIO_AUTO_SUBMIT_PARTIAL_COVERAGE")
            self.assertFalse(result["stake_mutations_enabled"])
            prepare.assert_called_once()
            approve.assert_called_once_with(
                run_id="run-a",
                challenge="CHALLENGE",
                actor="automation",
            )
            submit.assert_called_once_with(run_id="run-a")

    def test_auto_submit_refuses_changed_or_stake_eligible_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry"
            state = root / "state"
            artifact = self._artifact(root, "model.pkl", b"candidate")
            current_path = registry / "portfolio" / "current.json"
            current_path.parent.mkdir(parents=True)
            assignment = {
                "model_name": "kvmmn",
                "role": "core",
                "deployment_tier": "production",
                "stake_eligible": False,
                "artifact": {
                    "path": str(artifact),
                    "sha256": hashlib.sha256(b"candidate").hexdigest(),
                    "size_bytes": len(b"candidate"),
                },
                "evidence": None,
                "source_bundle_id": None,
            }
            current_path.write_text(
                json.dumps({"schema_version": 1, "assignments": [assignment]})
            )
            artifact.write_bytes(b"tampered")
            with (
                patch("app.core.portfolio.settings.AUTO_SUBMIT_PORTFOLIO", True),
                patch("app.core.portfolio.sync_datasets") as sync,
            ):
                with self.assertRaisesRegex(PortfolioError, "checksum changed"):
                    PortfolioControlPlane(
                        registry_dir=registry,
                        state_dir=state,
                        napi=_FakeApi(),
                    ).auto_submit_all()
            sync.assert_not_called()
            artifact.write_bytes(b"candidate")
            assignment["stake_eligible"] = True
            current_path.write_text(
                json.dumps({"schema_version": 1, "assignments": [assignment]})
            )
            with patch("app.core.portfolio.settings.AUTO_SUBMIT_PORTFOLIO", True):
                with self.assertRaisesRegex(PortfolioError, "stake-eligible"):
                    PortfolioControlPlane(
                        registry_dir=registry,
                        state_dir=state,
                        napi=_FakeApi(),
                    ).auto_submit_all()

    def test_shadow_assignment_requires_evidence_and_cannot_stake(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = self._artifact(root, "shadow.pkl", b"shadow")
            with self.assertRaisesRegex(PortfolioError, "frozen evidence"):
                create_portfolio_proposal(
                    root,
                    [
                        {
                            "model_name": "kvmmn_fn",
                            "artifact_path": artifact,
                            "deployment_tier": "shadow",
                        }
                    ],
                )
            evidence = self._artifact(
                root,
                "evidence.json",
                json.dumps(
                    {
                        "decision": "DEPLOY_SHADOW",
                        "stake_eligible": False,
                    }
                ).encode(),
            )
            with self.assertRaisesRegex(PortfolioError, "never be stake eligible"):
                create_portfolio_proposal(
                    root,
                    [
                        {
                            "model_name": "kvmmn_fn",
                            "artifact_path": artifact,
                            "deployment_tier": "shadow",
                            "stake_eligible": True,
                            "evidence_path": evidence,
                        }
                    ],
                )


if __name__ == "__main__":
    unittest.main()
