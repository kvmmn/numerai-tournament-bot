from __future__ import annotations

import argparse
import asyncio
import unittest
from unittest.mock import patch

from automation.daily_numerai_run import run_mode
from app.core.numerai_mcp_client import NumeraiMCPError
from app.core.numerai_mcp_workflow import NumeraiMCPWorkflowRunner


class GovernedRunnerTests(unittest.TestCase):
    def test_stake_status_is_read_only_dispatch(self):
        with patch(
            "automation.daily_numerai_run.StakeControlPlane.inspect",
            return_value={"ok": True, "status": "STAKE_POLICY_OK"},
        ) as inspect:
            result = run_mode("stake-status")
        self.assertEqual(result["status"], "STAKE_POLICY_OK")
        inspect.assert_called_once_with()

    def test_stake_mutations_require_explicit_arguments(self):
        args = argparse.Namespace(
            target_model="kvmmn",
            stake_action="decrease",
            amount_nmr=0.05,
            rationale="Reduce ineligible stake.",
            deployment_round=None,
            stake_proposal_path="/tmp/proposal.json",
            challenge="CHALLENGE",
            actor="operator",
            confirmation="DECREASE 0.05 NMR ON kvmmn",
        )
        with patch(
            "automation.daily_numerai_run.StakeControlPlane.propose",
            return_value={"ok": True, "status": "STAKE_AWAITING_HUMAN_APPROVAL"},
        ) as propose:
            run_mode("stake-propose", args)
        propose.assert_called_once_with(
            target_model="kvmmn",
            action="decrease",
            amount_nmr=0.05,
            rationale="Reduce ineligible stake.",
            deployment_round=None,
        )

        with patch(
            "automation.daily_numerai_run.StakeControlPlane.approve",
            return_value={"ok": True, "status": "STAKE_APPROVED"},
        ) as approve:
            run_mode("stake-approve", args)
        approve.assert_called_once_with(
            proposal_path="/tmp/proposal.json",
            challenge="CHALLENGE",
            actor="operator",
        )

        with patch(
            "automation.daily_numerai_run.StakeControlPlane.execute",
            return_value={"ok": True, "status": "STAKE_CHANGE_REQUESTED"},
        ) as execute:
            run_mode("stake-execute", args)
        execute.assert_called_once_with(
            proposal_path="/tmp/proposal.json",
            confirmation="DECREASE 0.05 NMR ON kvmmn",
        )

    def test_legacy_submission_modes_remain_disabled(self):
        for mode in ("mcp-submit", "numerapi-submit", "full-auto"):
            with self.subTest(mode=mode):
                result = run_mode(mode)
                self.assertFalse(result["ok"])
                self.assertEqual(result["status"], "unsafe_legacy_mode_disabled")

    def test_mcp_runner_rejects_submission_even_when_called_directly(self):
        with self.assertRaisesRegex(NumeraiMCPError, "MCP submission is disabled"):
            asyncio.run(
                NumeraiMCPWorkflowRunner().run_cycle(
                    approve_submission=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
