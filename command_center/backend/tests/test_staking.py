from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

from app.core.staking import (
    StakePolicy,
    StakeSizingPolicy,
    StakingPolicyError,
    approve_stake_proposal,
    create_stake_proposal,
    execute_approved_stake_change,
    recommend_stake_increase,
    summarize_live_performance,
)


NOW = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
POLICY = StakePolicy(
    max_total_nmr=100.0,
    max_per_model_nmr=40.0,
    max_change_nmr=10.0,
)


class StakePolicyCapTests(unittest.TestCase):
    def create_proposal(self, temp_dir: str, **overrides):
        arguments = {
            "model_name": "KVMMN",
            "model_id": "model-1",
            "action": "increase",
            "amount_nmr": 5.0,
            "current_model_stake_nmr": 20.0,
            "current_total_stake_nmr": 50.0,
            "rationale": "Controlled allocation adjustment.",
            "policy": POLICY,
            "ttl_minutes": 15,
            "now": NOW,
        }
        arguments.update(overrides)
        return create_stake_proposal(temp_dir, **arguments)

    def test_proposal_records_projected_stakes_within_caps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal, path = self.create_proposal(temp_dir)

            self.assertEqual(proposal.projected_model_stake_nmr, 25.0)
            self.assertEqual(proposal.projected_total_stake_nmr, 55.0)
            self.assertEqual(proposal.status, "AWAITING_HUMAN_APPROVAL")
            self.assertTrue(path.exists())

    def test_rejects_nonpositive_or_over_limit_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for amount in (0.0, -1.0, 10.01):
                with self.subTest(amount=amount):
                    with self.assertRaisesRegex(
                        StakingPolicyError,
                        "per-change limit",
                    ):
                        self.create_proposal(temp_dir, amount_nmr=amount)

    def test_rejects_projected_per_model_cap_breach(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(StakingPolicyError, "model cap"):
                self.create_proposal(
                    temp_dir,
                    amount_nmr=5.0,
                    current_model_stake_nmr=39.0,
                )

    def test_rejects_projected_total_cap_breach(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(StakingPolicyError, "portfolio cap"):
                self.create_proposal(
                    temp_dir,
                    amount_nmr=5.0,
                    current_total_stake_nmr=99.0,
                )

    def test_rejects_decrease_below_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(StakingPolicyError, "exceeds the current stake"):
                self.create_proposal(
                    temp_dir,
                    action="decrease",
                    amount_nmr=5.0,
                    current_model_stake_nmr=2.0,
                )


class StakeSizingTests(unittest.TestCase):
    @staticmethod
    def rows(count: int, corr: float = 0.02, mmc: float = 0.01):
        return [
            {
                "roundNumber": 100 + index,
                "roundResolved": True,
                "atRisk": "1.0",
                "submissionScores": [
                    {"displayName": "canon_corr", "value": corr},
                    {"displayName": "canon_mmc", "value": mmc},
                ],
            }
            for index in range(count)
        ]

    def test_summarizes_resolved_live_metrics(self):
        summary = summarize_live_performance(self.rows(25), recent_window=20)
        self.assertEqual(summary["resolved_rounds"], 25)
        self.assertEqual(summary["recent_rounds"], 20)
        self.assertAlmostEqual(summary["mean_corr"], 0.02)
        self.assertAlmostEqual(summary["mean_mmc"], 0.01)

    def test_new_model_is_ineligible_until_live_evidence_exists(self):
        result = recommend_stake_increase(
            [],
            current_model_stake_nmr=0,
            current_total_stake_nmr=0,
            caps=POLICY,
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(result["recommended_increase_nmr"], 0.0)
        self.assertIn("insufficient_resolved_rounds", result["failures"])

    def test_caps_must_be_enabled_and_performance_positive(self):
        disabled = recommend_stake_increase(
            self.rows(25),
            current_model_stake_nmr=0,
            current_total_stake_nmr=0,
            caps=StakePolicy(0, 0, 0),
            deployment_round=100,
        )
        self.assertIn("staking_caps_disabled", disabled["failures"])
        weak = recommend_stake_increase(
            self.rows(25, corr=-0.01, mmc=-0.01),
            current_model_stake_nmr=0,
            current_total_stake_nmr=0,
            caps=POLICY,
            deployment_round=100,
        )
        self.assertFalse(weak["eligible"])
        self.assertIn("mean_corr_below_floor", weak["failures"])

    def test_recommends_only_small_initial_capped_allocation(self):
        result = recommend_stake_increase(
            self.rows(25),
            current_model_stake_nmr=0,
            current_total_stake_nmr=10,
            caps=POLICY,
            sizing=StakeSizingPolicy(maximum_initial_allocation_fraction=0.05),
            deployment_round=100,
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(result["recommended_increase_nmr"], 5.0)


class ManualStakeApprovalTests(unittest.TestCase):
    def create_proposal(self, temp_dir: str, *, ttl_minutes: int = 15):
        return create_stake_proposal(
            temp_dir,
            model_name="KVMMN",
            model_id="model-1",
            action="increase",
            amount_nmr=5.0,
            current_model_stake_nmr=20.0,
            current_total_stake_nmr=50.0,
            rationale="Controlled allocation adjustment.",
            policy=POLICY,
            ttl_minutes=ttl_minutes,
            now=NOW,
        )

    def test_execution_without_manual_approval_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, proposal_path = self.create_proposal(temp_dir)
            napi = Mock()

            with self.assertRaisesRegex(StakingPolicyError, "explicit human approval"):
                execute_approved_stake_change(
                    napi,
                    proposal_path,
                    confirmation="INCREASE 5.0 NMR ON KVMMN",
                    now=NOW + timedelta(minutes=1),
                )

            napi.stake_increase.assert_not_called()

    def test_wrong_or_expired_approval_challenge_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal, proposal_path = self.create_proposal(temp_dir, ttl_minutes=5)

            with self.assertRaisesRegex(StakingPolicyError, "does not match"):
                approve_stake_proposal(
                    proposal_path,
                    challenge="WRONG",
                    actor="operator",
                    now=NOW,
                )
            with self.assertRaisesRegex(StakingPolicyError, "expired"):
                approve_stake_proposal(
                    proposal_path,
                    challenge=proposal.approval_challenge,
                    actor="operator",
                    now=NOW + timedelta(minutes=6),
                )

    def test_confirmation_must_match_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal, proposal_path = self.create_proposal(temp_dir)
            approve_stake_proposal(
                proposal_path,
                challenge=proposal.approval_challenge,
                actor="operator",
                now=NOW,
            )
            napi = Mock()

            with self.assertRaisesRegex(StakingPolicyError, "exactly match"):
                execute_approved_stake_change(
                    napi,
                    proposal_path,
                    confirmation="increase 5 NMR",
                    now=NOW + timedelta(minutes=1),
                )

            napi.stake_increase.assert_not_called()

    def test_approved_increase_calls_api_once_and_records_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal, proposal_path = self.create_proposal(temp_dir)
            approve_stake_proposal(
                proposal_path,
                challenge=proposal.approval_challenge,
                actor="operator",
                now=NOW,
            )
            napi = Mock()
            napi.stake_increase.return_value = {"transaction_id": "tx-1"}

            result = execute_approved_stake_change(
                napi,
                proposal_path,
                confirmation="INCREASE 5.0 NMR ON KVMMN",
                now=NOW + timedelta(minutes=1),
            )

            self.assertEqual(result, {"transaction_id": "tx-1"})
            napi.stake_increase.assert_called_once_with(5.0, "model-1")
            execution = json.loads(
                proposal_path.with_name("execution.json").read_text()
            )
            self.assertEqual(execution["proposal_id"], proposal.proposal_id)
            self.assertEqual(execution["result"], result)

    def test_tampered_approval_cannot_execute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal, proposal_path = self.create_proposal(temp_dir)
            approval_path = approve_stake_proposal(
                proposal_path,
                challenge=proposal.approval_challenge,
                actor="operator",
                now=NOW,
            )
            approval = json.loads(approval_path.read_text())
            approval["amount_nmr"] = 7.0
            approval_path.write_text(json.dumps(approval))
            napi = Mock()

            with self.assertRaisesRegex(StakingPolicyError, "amount_nmr"):
                execute_approved_stake_change(
                    napi,
                    proposal_path,
                    confirmation="INCREASE 5.0 NMR ON KVMMN",
                    now=NOW + timedelta(minutes=1),
                )

            napi.stake_increase.assert_not_called()

    def test_expired_approval_cannot_execute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal, proposal_path = self.create_proposal(temp_dir, ttl_minutes=5)
            approve_stake_proposal(
                proposal_path,
                challenge=proposal.approval_challenge,
                actor="operator",
                now=NOW,
            )
            napi = Mock()

            with self.assertRaisesRegex(StakingPolicyError, "expired"):
                execute_approved_stake_change(
                    napi,
                    proposal_path,
                    confirmation="INCREASE 5.0 NMR ON KVMMN",
                    now=NOW + timedelta(minutes=6),
                )

            napi.stake_increase.assert_not_called()


if __name__ == "__main__":
    unittest.main()
