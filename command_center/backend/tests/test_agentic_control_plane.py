from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.core.agentic_control_plane import AgenticControlPlane
from app.core.config import settings
from app.core.submission_guard import SubmissionGuardError


class FakeNumerai:
    def __init__(self):
        self.upload_calls = 0

    def get_current_round(self):
        return 1300

    def check_round_open(self):
        return True

    def get_models(self):
        return {"kvmmn_te": "model-123"}

    def upload_predictions(self, *, df, model_id, timeout):
        self.upload_calls += 1
        self.uploaded = (df.copy(), model_id, timeout)
        return "submission-456"

    def submission_ids(self, model_id):
        return [{"id": "submission-456"}]


class AgenticControlPlaneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifact = self.root / "model.pkl"
        self.artifact.write_bytes(b"model")
        self.frame = pd.DataFrame(
            {
                "id": [f"id-{index}" for index in range(100)],
                "prediction": [(index + 1) / 101 for index in range(100)],
            }
        )
        self.napi = FakeNumerai()

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self):
        plane = AgenticControlPlane(self.root / "state", napi=self.napi)
        with (
            patch(
                "app.core.agentic_control_plane.sync_datasets",
                return_value={"downloaded": ["live.parquet"]},
            ),
            patch(
                "app.core.agentic_control_plane.build_submission_dataframe",
                return_value=self.frame,
            ),
            patch(
                "app.core.agentic_control_plane.evaluate_artifact_robustness",
                return_value={"packet_sha256": "packet"},
            ),
            patch(
                "app.core.agentic_control_plane.promotion_recommendation",
                return_value={"decision": "PROMOTE", "failures": []},
            ),
        ):
            result = plane.prepare(
                target_model="KVMMN_TE",
                artifact_path=self.artifact,
            )
        return plane, result

    def test_prepare_delegates_and_emits_readiness_packet(self):
        _, result = self.prepare()
        self.assertEqual(result["status"], "AWAITING_HUMAN_APPROVAL")
        self.assertEqual(result["target_model"], "kvmmn_te")
        self.assertTrue(Path(result["packet_path"]).exists())
        self.assertEqual(
            result["delegated_agents"],
            [
                "platform_scout",
                "data_steward",
                "prediction_specialist",
                "risk_judge",
                "governance_guard",
            ],
        )

    def test_submit_requires_approval_then_verifies_and_is_idempotent(self):
        plane, prepared = self.prepare()
        with self.assertRaises(SubmissionGuardError):
            plane.submit(run_id=prepared["run_id"])
        plane.approve(
            run_id=prepared["run_id"],
            challenge=prepared["approval_challenge"],
            actor="operator",
        )
        result = plane.submit(run_id=prepared["run_id"])
        self.assertEqual(result["status"], "SUBMITTED_VERIFIED")
        self.assertEqual(self.napi.upload_calls, 1)
        with self.assertRaises(SubmissionGuardError):
            plane.submit(run_id=prepared["run_id"])
        self.assertEqual(self.napi.upload_calls, 1)

    def test_default_artifact_resolves_frozen_champion_member(self):
        registry = self.root / "registry"
        manifest = registry / "candidates" / "bundle" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "members": [
                        {"artifact": {"path": str(self.artifact)}}
                    ]
                }
            )
        )
        current = registry / "champion" / "current.json"
        current.parent.mkdir(parents=True)
        current.write_text(json.dumps({"manifest_path": str(manifest)}))
        with patch.object(settings, "MODEL_REGISTRY_DIR", str(registry)):
            self.assertEqual(
                AgenticControlPlane._default_artifact(),
                self.artifact,
            )

    def test_shadow_prepare_uses_separate_zero_stake_policy(self):
        evidence = self.root / "shadow_evidence.json"
        evidence.write_text(
            json.dumps(
                {
                    "decision": "DEPLOY_SHADOW",
                    "stake_eligible": False,
                    "maximum_observed_portfolio_correlation": 0.7,
                }
            )
        )
        plane = AgenticControlPlane(self.root / "state", napi=self.napi)
        with (
            patch(
                "app.core.agentic_control_plane.sync_datasets",
                return_value={"downloaded": ["live.parquet"]},
            ),
            patch(
                "app.core.agentic_control_plane.build_submission_dataframe",
                return_value=self.frame,
            ),
            patch(
                "app.core.agentic_control_plane.evaluate_artifact_robustness",
                return_value={"packet_sha256": "packet"},
            ),
            patch(
                "app.core.agentic_control_plane.shadow_recommendation",
                return_value={"decision": "DEPLOY_SHADOW", "failures": []},
            ) as shadow_policy,
            patch(
                "app.core.agentic_control_plane.promotion_recommendation"
            ) as production_policy,
        ):
            result = plane.prepare(
                target_model="kvmmn_te",
                artifact_path=self.artifact,
                deployment_tier="shadow",
                shadow_evidence_path=evidence,
            )
        self.assertEqual(result["validation"]["deployment_tier"], "shadow")
        self.assertFalse(result["validation"]["stake_eligible"])
        shadow_policy.assert_called_once()
        production_policy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
