from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.staking import (
    StakeControlPlane,
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


class FakeStakeApi:
    def __init__(
        self,
        *,
        stakes: dict[str, float] | None = None,
        available_nmr: float = 10.0,
    ):
        self.models = {
            "KVMMN": "model-1",
            "OTHER": "model-2",
        }
        self.stakes = stakes or {"KVMMN": 20.0, "OTHER": 30.0}
        self.available_nmr = available_nmr
        self.increase_calls = []
        self.decrease_calls = []
        self.performance_rows = StakeSizingTests.rows(25)

    def get_models(self):
        return dict(self.models)

    def get_account(self):
        return {
            "availableNmr": self.available_nmr,
            "models": [
                {"id": model_id, "v2Stake": {"status": ""}}
                for model_id in self.models.values()
            ],
        }

    def stake_get(self, model_name):
        return self.stakes.get(model_name, 0.0)

    def stake_increase(self, amount_nmr, model_id):
        self.increase_calls.append((amount_nmr, model_id))
        return {"transaction_id": "tx-increase"}

    def stake_decrease(self, amount_nmr, model_id):
        self.decrease_calls.append((amount_nmr, model_id))
        return {"transaction_id": "tx-decrease"}

    def round_model_performances_v2(self, model_id):
        return list(self.performance_rows)


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
            napi = FakeStakeApi()

            with self.assertRaisesRegex(StakingPolicyError, "explicit human approval"):
                execute_approved_stake_change(
                    napi,
                    proposal_path,
                    confirmation="INCREASE 5.0 NMR ON KVMMN",
                    policy=POLICY,
                    now=NOW + timedelta(minutes=1),
                )

            self.assertEqual(napi.increase_calls, [])

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
            napi = FakeStakeApi()

            with self.assertRaisesRegex(StakingPolicyError, "exactly match"):
                execute_approved_stake_change(
                    napi,
                    proposal_path,
                    confirmation="increase 5 NMR",
                    policy=POLICY,
                    now=NOW + timedelta(minutes=1),
                )

            self.assertEqual(napi.increase_calls, [])

    def test_approved_increase_calls_api_once_and_records_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal, proposal_path = self.create_proposal(temp_dir)
            approve_stake_proposal(
                proposal_path,
                challenge=proposal.approval_challenge,
                actor="operator",
                now=NOW,
            )
            napi = FakeStakeApi()

            result = execute_approved_stake_change(
                napi,
                proposal_path,
                confirmation="INCREASE 5.0 NMR ON KVMMN",
                policy=POLICY,
                now=NOW + timedelta(minutes=1),
            )

            self.assertEqual(result, {"transaction_id": "tx-increase"})
            self.assertEqual(napi.increase_calls, [(5.0, "model-1")])
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
            napi = FakeStakeApi()

            with self.assertRaisesRegex(StakingPolicyError, "amount_nmr"):
                execute_approved_stake_change(
                    napi,
                    proposal_path,
                    confirmation="INCREASE 5.0 NMR ON KVMMN",
                    policy=POLICY,
                    now=NOW + timedelta(minutes=1),
                )

            self.assertEqual(napi.increase_calls, [])

    def test_expired_approval_cannot_execute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal, proposal_path = self.create_proposal(temp_dir, ttl_minutes=5)
            approve_stake_proposal(
                proposal_path,
                challenge=proposal.approval_challenge,
                actor="operator",
                now=NOW,
            )
            napi = FakeStakeApi()

            with self.assertRaisesRegex(StakingPolicyError, "expired"):
                execute_approved_stake_change(
                    napi,
                    proposal_path,
                    confirmation="INCREASE 5.0 NMR ON KVMMN",
                    policy=POLICY,
                    now=NOW + timedelta(minutes=6),
                )

    def test_changed_proposal_is_rejected_after_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal, proposal_path = self.create_proposal(temp_dir)
            approve_stake_proposal(
                proposal_path,
                challenge=proposal.approval_challenge,
                actor="operator",
                now=NOW,
            )
            payload = json.loads(proposal_path.read_text())
            payload["rationale"] = "tampered"
            proposal_path.write_text(json.dumps(payload))
            napi = FakeStakeApi()

            with self.assertRaisesRegex(StakingPolicyError, "changed after approval"):
                execute_approved_stake_change(
                    napi,
                    proposal_path,
                    confirmation="INCREASE 5.0 NMR ON KVMMN",
                    policy=POLICY,
                    now=NOW + timedelta(minutes=1),
                )
            self.assertEqual(napi.increase_calls, [])

    def test_stale_live_balance_and_duplicate_execution_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal, proposal_path = self.create_proposal(temp_dir)
            approve_stake_proposal(
                proposal_path,
                challenge=proposal.approval_challenge,
                actor="operator",
                now=NOW,
            )
            stale = FakeStakeApi(stakes={"KVMMN": 21.0, "OTHER": 30.0})
            with self.assertRaisesRegex(StakingPolicyError, "Live stake changed"):
                execute_approved_stake_change(
                    stale,
                    proposal_path,
                    confirmation="INCREASE 5.0 NMR ON KVMMN",
                    policy=POLICY,
                    now=NOW + timedelta(minutes=1),
                )

            napi = FakeStakeApi()
            execute_approved_stake_change(
                napi,
                proposal_path,
                confirmation="INCREASE 5.0 NMR ON KVMMN",
                policy=POLICY,
                now=NOW + timedelta(minutes=1),
            )
            with self.assertRaisesRegex(StakingPolicyError, "already been executed"):
                execute_approved_stake_change(
                    napi,
                    proposal_path,
                    confirmation="INCREASE 5.0 NMR ON KVMMN",
                    policy=POLICY,
                    now=NOW + timedelta(minutes=2),
                )
            self.assertEqual(napi.increase_calls, [(5.0, "model-1")])

    def test_uncertain_api_failure_writes_intent_and_blocks_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal, proposal_path = self.create_proposal(temp_dir)
            approve_stake_proposal(
                proposal_path,
                challenge=proposal.approval_challenge,
                actor="operator",
                now=NOW,
            )
            napi = FakeStakeApi()

            def fail_after_request(amount_nmr, model_id):
                napi.increase_calls.append((amount_nmr, model_id))
                raise RuntimeError("connection lost")

            napi.stake_increase = fail_after_request
            with self.assertRaisesRegex(RuntimeError, "connection lost"):
                execute_approved_stake_change(
                    napi,
                    proposal_path,
                    confirmation="INCREASE 5.0 NMR ON KVMMN",
                    policy=POLICY,
                    now=NOW + timedelta(minutes=1),
                )
            self.assertTrue(
                proposal_path.with_name("execution_intent.json").exists()
            )
            self.assertTrue(
                proposal_path.with_name("execution_error.json").exists()
            )

            with self.assertRaisesRegex(StakingPolicyError, "Unresolved"):
                execute_approved_stake_change(
                    napi,
                    proposal_path,
                    confirmation="INCREASE 5.0 NMR ON KVMMN",
                    policy=POLICY,
                    now=NOW + timedelta(minutes=2),
                )
            self.assertEqual(napi.increase_calls, [(5.0, "model-1")])


class StakeControlPlaneTests(unittest.TestCase):
    @staticmethod
    def write_portfolio(
        registry: Path,
        *,
        deployment_tier: str,
        stake_eligible: bool,
        artifact_sha256: str = "artifact-sha",
    ) -> None:
        path = registry / "portfolio" / "current.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "assignments": [
                        {
                            "model_name": "KVMMN",
                            "deployment_tier": deployment_tier,
                            "stake_eligible": stake_eligible,
                            "artifact": {
                                "path": "/tmp/model.pkl",
                                "sha256": artifact_sha256,
                            },
                        }
                    ]
                }
            )
        )

    def test_live_audit_flags_stake_on_shadow_assignment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_portfolio(
                root / "registry",
                deployment_tier="shadow",
                stake_eligible=False,
            )
            napi = FakeStakeApi(stakes={"KVMMN": 0.136, "OTHER": 0.0})
            result = StakeControlPlane(
                root / "state",
                root / "registry",
                napi=napi,
                policy=StakePolicy(0, 0, 0),
            ).inspect()
            self.assertTrue(result["read_only"])
            self.assertEqual(result["status"], "STAKE_POLICY_ACTION_REQUIRED")
            codes = {row["code"] for row in result["violations"]}
            self.assertIn("SHADOW_MODEL_HAS_STAKE", codes)
            self.assertIn("STAKE_INELIGIBLE_MODEL_HAS_STAKE", codes)

    def test_decrease_can_be_proposed_with_increase_caps_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_portfolio(
                root / "registry",
                deployment_tier="shadow",
                stake_eligible=False,
            )
            result = StakeControlPlane(
                root / "state",
                root / "registry",
                napi=FakeStakeApi(stakes={"KVMMN": 0.136, "OTHER": 0.0}),
                policy=StakePolicy(0, 0, 0.1),
            ).propose(
                target_model="kvmmn",
                action="decrease",
                amount_nmr=0.05,
                rationale="Reduce stake on a shadow assignment.",
            )
            self.assertEqual(result["status"], "STAKE_AWAITING_HUMAN_APPROVAL")
            self.assertAlmostEqual(result["projected_model_stake_nmr"], 0.086)

    def test_increase_requires_verified_active_artifact_and_live_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = root / "registry"
            state = root / "state"
            self.write_portfolio(
                registry,
                deployment_tier="production",
                stake_eligible=True,
            )
            ledger = state / "submission_ledger.json"
            ledger.parent.mkdir(parents=True)
            ledger.write_text(
                json.dumps(
                    {
                        "submissions": {
                            "100:model-1": {
                                "round_number": 100,
                                "model_id": "model-1",
                                "run_id": "run-1",
                                "submission_id": "submission-1",
                                "verified": True,
                            }
                        }
                    }
                )
            )
            packet = state / "runs" / "run-1" / "readiness.json"
            packet.parent.mkdir(parents=True)
            packet.write_text(
                json.dumps(
                    {
                        "validation": {
                            "model_artifact_sha256": "artifact-sha",
                        }
                    }
                )
            )
            result = StakeControlPlane(
                state,
                registry,
                napi=FakeStakeApi(stakes={"KVMMN": 0.0, "OTHER": 0.0}),
                policy=StakePolicy(100, 40, 10),
            ).propose(
                target_model="kvmmn",
                action="increase",
                amount_nmr=5.0,
                rationale="Initial allocation after resolved live evidence.",
                deployment_round=100,
            )
            self.assertEqual(result["status"], "STAKE_AWAITING_HUMAN_APPROVAL")


if __name__ == "__main__":
    unittest.main()
