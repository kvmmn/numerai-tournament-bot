from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.core.model_registry import (
    ModelRegistryError,
    approve_candidate_bundle,
    create_candidate_bundle,
    create_shadow_bundle,
    promote_candidate_bundle,
)


NOW = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)


class ModelRegistryTests(unittest.TestCase):
    def fixtures(self, root: Path, decision: str = "PROMOTE"):
        model = root / "model.pkl"
        model.write_bytes(b"model-v1")
        data = root / "features.json"
        data.write_text('{"feature_sets":{"small":["a"]}}')
        evaluation = root / "evaluation.json"
        evaluation.write_text('{"packet_sha256":"evaluation"}')
        recommendation = root / "recommendation.json"
        recommendation.write_text(json.dumps({"decision": decision}))
        return model, data, evaluation, recommendation

    def create(self, root: Path, decision: str = "PROMOTE"):
        model, data, evaluation, recommendation = self.fixtures(root, decision)
        return create_candidate_bundle(
            root / "registry",
            name="candidate",
            members=[{"name": "model", "path": model, "weight": 1.0}],
            evaluation_packet_path=evaluation,
            recommendation_path=recommendation,
            data_files=[data],
            configuration={"feature_set": "small"},
            code_revision="abc123",
            now=NOW,
        )

    def test_rejects_non_promote_and_invalid_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ModelRegistryError, "PROMOTE"):
                self.create(root, decision="REJECT")
            model, data, evaluation, recommendation = self.fixtures(root)
            with self.assertRaisesRegex(ModelRegistryError, "sum to 1.0"):
                create_candidate_bundle(
                    root / "registry",
                    name="bad",
                    members=[{"name": "model", "path": model, "weight": 0.5}],
                    evaluation_packet_path=evaluation,
                    recommendation_path=recommendation,
                    data_files=[data],
                    configuration={},
                    code_revision=None,
                )

    def test_promotion_requires_matching_approval_and_frozen_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, manifest_path = self.create(root)
            with self.assertRaisesRegex(ModelRegistryError, "explicit human approval"):
                promote_candidate_bundle(root / "registry", manifest_path, now=NOW)
            with self.assertRaisesRegex(ModelRegistryError, "does not match"):
                approve_candidate_bundle(
                    manifest_path,
                    challenge="WRONG",
                    actor="operator",
                    now=NOW,
                )
            approve_candidate_bundle(
                manifest_path,
                challenge=manifest["approval_challenge"],
                actor="operator",
                now=NOW,
            )
            current_path = promote_candidate_bundle(
                root / "registry",
                manifest_path,
                now=NOW,
            )
            current = json.loads(current_path.read_text())
            self.assertEqual(current["bundle_id"], manifest["bundle_id"])
            self.assertEqual(current["approved_by"], "operator")

            frozen = Path(manifest["members"][0]["artifact"]["path"])
            frozen.write_bytes(b"tampered")
            with self.assertRaisesRegex(ModelRegistryError, "changed"):
                promote_candidate_bundle(root / "registry", manifest_path, now=NOW)

    def test_shadow_bundle_requires_shadow_evidence_and_disables_staking(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model, data, evaluation, recommendation = self.fixtures(
                root,
                decision="DEPLOY_SHADOW",
            )
            manifest, manifest_path = create_shadow_bundle(
                root / "registry",
                name="shadow",
                artifact_path=model,
                evaluation_packet_path=evaluation,
                recommendation_path=recommendation,
                data_files=[data],
                configuration={"tier": "shadow"},
                code_revision="abc123",
                now=NOW,
            )
            self.assertTrue(manifest_path.exists())
            self.assertEqual(manifest["deployment_tier"], "shadow")
            self.assertFalse(manifest["stake_eligible"])


if __name__ == "__main__":
    unittest.main()
