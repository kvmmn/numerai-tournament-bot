from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.core.submission_guard import (
    SubmissionGuardError,
    SubmissionLedger,
    approve_readiness_packet,
    create_readiness_packet,
    resolve_single_model,
    validate_approval,
    validate_raw_prediction_series,
    validate_submission_dataframe,
)


NOW = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)


class ResolveSingleModelTests(unittest.TestCase):
    def test_resolves_configured_model_case_insensitively(self) -> None:
        resolved = resolve_single_model(
            {"KVMMN": "model-1", "Research": "model-2"},
            "kvmmn",
        )

        self.assertEqual(resolved, ("KVMMN", "model-1"))

    def test_allows_implicit_selection_when_account_has_one_model(self) -> None:
        self.assertEqual(
            resolve_single_model({"OnlyModel": "model-1"}, None),
            ("OnlyModel", "model-1"),
        )

    def test_fails_closed_when_multiple_models_have_no_explicit_target(self) -> None:
        with self.assertRaisesRegex(
            SubmissionGuardError,
            "Multiple Numerai models exist",
        ):
            resolve_single_model(
                {"KVMMN": "model-1", "KVMMN_FN": "model-2"},
                None,
            )

    def test_fails_closed_when_casefolded_name_is_ambiguous(self) -> None:
        with self.assertRaisesRegex(SubmissionGuardError, "does not uniquely match"):
            resolve_single_model(
                {"Alpha": "model-1", "ALPHA": "model-2"},
                "alpha",
            )


class SubmissionDataframeValidationTests(unittest.TestCase):
    @staticmethod
    def valid_frame(size: int = 100) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": [f"id-{index}" for index in range(size)],
                "prediction": [(index + 0.5) / size for index in range(size)],
            }
        )

    def test_accepts_complete_finite_diverse_predictions(self) -> None:
        frame = self.valid_frame()

        result = validate_submission_dataframe(
            frame,
            expected_ids=pd.Index(reversed(frame["id"].tolist())),
        )

        self.assertEqual(result["row_count"], 100)
        self.assertEqual(result["unique_ids"], 100)
        self.assertEqual(result["unique_predictions"], 100)
        self.assertGreaterEqual(result["prediction_min"], 0.0)
        self.assertLessEqual(result["prediction_max"], 1.0)

    def test_rejects_incorrect_columns(self) -> None:
        frame = self.valid_frame().rename(columns={"prediction": "score"})

        with self.assertRaisesRegex(SubmissionGuardError, "columns must be exactly"):
            validate_submission_dataframe(frame)

    def test_rejects_duplicate_or_missing_ids(self) -> None:
        duplicate = self.valid_frame()
        duplicate.loc[1, "id"] = duplicate.loc[0, "id"]
        missing = self.valid_frame()
        missing.loc[0, "id"] = None

        for frame in (duplicate, missing):
            with self.subTest(frame=frame.head(2).to_dict()):
                with self.assertRaisesRegex(SubmissionGuardError, "present and unique"):
                    validate_submission_dataframe(frame)

    def test_rejects_non_finite_and_out_of_range_predictions(self) -> None:
        cases = (float("nan"), float("inf"), -0.01, 1.01)

        for value in cases:
            with self.subTest(value=value):
                frame = self.valid_frame()
                frame.loc[0, "prediction"] = value
                with self.assertRaises(SubmissionGuardError):
                    validate_submission_dataframe(frame)

    def test_rejects_low_prediction_diversity(self) -> None:
        frame = self.valid_frame()
        frame["prediction"] = 0.5

        with self.assertRaisesRegex(SubmissionGuardError, "unique values"):
            validate_submission_dataframe(frame)

    def test_rejects_live_id_set_mismatch(self) -> None:
        frame = self.valid_frame()
        expected_ids = pd.Index(frame["id"].tolist()[:-1] + ["different-id"])

        with self.assertRaisesRegex(SubmissionGuardError, "id mismatch"):
            validate_submission_dataframe(frame, expected_ids=expected_ids)

    def test_raw_constant_predictions_are_rejected_before_ranking(self) -> None:
        with self.assertRaisesRegex(SubmissionGuardError, "before ranking"):
            validate_raw_prediction_series(pd.Series([0.5] * 100))


class SubmissionLedgerTests(unittest.TestCase):
    def test_round_and_model_key_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger_path = Path(temp_dir) / "submission-ledger.json"
            ledger = SubmissionLedger(ledger_path)

            self.assertFalse(ledger.contains(718, "model-1"))
            ledger.record(
                round_number=718,
                model_id="model-1",
                submission_id="submission-1",
                run_id="run-1",
                verified=True,
            )
            self.assertTrue(ledger.contains(718, "model-1"))

            with self.assertRaisesRegex(
                SubmissionGuardError,
                "already has a recorded submission",
            ):
                ledger.record(
                    round_number=718,
                    model_id="model-1",
                    submission_id="submission-2",
                    run_id="run-2",
                    verified=True,
                )

            ledger.record(
                round_number=719,
                model_id="model-1",
                submission_id="submission-3",
                run_id="run-3",
                verified=True,
            )
            ledger.record(
                round_number=718,
                model_id="model-2",
                submission_id="submission-4",
                run_id="run-4",
                verified=False,
            )
            payload = json.loads(ledger_path.read_text())
            self.assertEqual(len(payload["submissions"]), 3)
            self.assertFalse(payload["submissions"]["718:model-2"]["verified"])


class ReadinessApprovalTests(unittest.TestCase):
    def create_packet(
        self,
        temp_dir: str,
        *,
        ttl_minutes: int = 15,
    ) -> tuple[object, Path, Path]:
        submission_path = Path(temp_dir) / "submission.csv"
        submission_path.write_text("id,prediction\nid-1,0.5\n")
        packet, packet_path = create_readiness_packet(
            temp_dir,
            round_number=718,
            target_model="KVMMN",
            target_model_id="model-1",
            submission_path=submission_path,
            validation={"row_count": 1},
            approval_ttl_minutes=ttl_minutes,
            now=NOW,
        )
        return packet, packet_path, submission_path

    def test_matching_unexpired_approval_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet, packet_path, _ = self.create_packet(temp_dir)
            approve_readiness_packet(
                packet_path,
                challenge=packet.approval_challenge.lower(),
                actor="operator",
                now=NOW + timedelta(minutes=1),
            )

            validated = validate_approval(
                packet_path,
                now=NOW + timedelta(minutes=2),
            )

            self.assertEqual(validated["run_id"], packet.run_id)
            self.assertEqual(validated["target_model_id"], "model-1")

    def test_wrong_challenge_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, packet_path, _ = self.create_packet(temp_dir)

            with self.assertRaisesRegex(SubmissionGuardError, "does not match"):
                approve_readiness_packet(
                    packet_path,
                    challenge="WRONG",
                    actor="operator",
                    now=NOW,
                )

    def test_revoked_packet_cannot_be_approved_or_submitted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet, packet_path, _ = self.create_packet(temp_dir)
            packet_path.with_name("revocation.json").write_text('{"status":"REVOKED"}')
            with self.assertRaisesRegex(SubmissionGuardError, "revoked"):
                approve_readiness_packet(
                    packet_path,
                    challenge=packet.approval_challenge,
                    actor="operator",
                    now=NOW,
                )

    def test_expired_packet_cannot_be_approved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet, packet_path, _ = self.create_packet(temp_dir, ttl_minutes=5)

            with self.assertRaisesRegex(SubmissionGuardError, "expired"):
                approve_readiness_packet(
                    packet_path,
                    challenge=packet.approval_challenge,
                    actor="operator",
                    now=NOW + timedelta(minutes=6),
                )

    def test_expired_grant_is_rejected_at_validation_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet, packet_path, _ = self.create_packet(temp_dir, ttl_minutes=5)
            approve_readiness_packet(
                packet_path,
                challenge=packet.approval_challenge,
                actor="operator",
                now=NOW + timedelta(minutes=1),
            )

            with self.assertRaisesRegex(SubmissionGuardError, "expired"):
                validate_approval(
                    packet_path,
                    now=NOW + timedelta(minutes=6),
                )

    def test_grant_must_match_packet_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet, packet_path, _ = self.create_packet(temp_dir)
            approval_path = approve_readiness_packet(
                packet_path,
                challenge=packet.approval_challenge,
                actor="operator",
                now=NOW,
            )
            approval = json.loads(approval_path.read_text())
            approval["target_model_id"] = "different-model"
            approval_path.write_text(json.dumps(approval))

            with self.assertRaisesRegex(
                SubmissionGuardError,
                "target_model_id",
            ):
                validate_approval(packet_path, now=NOW + timedelta(minutes=1))

    def test_approved_submission_artifact_cannot_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            packet, packet_path, submission_path = self.create_packet(temp_dir)
            approve_readiness_packet(
                packet_path,
                challenge=packet.approval_challenge,
                actor="operator",
                now=NOW,
            )
            submission_path.write_text("id,prediction\nid-1,0.9\n")

            with self.assertRaisesRegex(SubmissionGuardError, "artifact changed"):
                validate_approval(packet_path, now=NOW + timedelta(minutes=1))


if __name__ == "__main__":
    unittest.main()
