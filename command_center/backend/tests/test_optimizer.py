from __future__ import annotations

import unittest

from app.core.optimizer import OptimizationError, select_best_candidate


class OptimizerSelectionTests(unittest.TestCase):
    def test_prefers_eligible_candidate_over_higher_rejected_score(self):
        results = [
            {
                "candidate_name": "rejected",
                "development_objective": 1.0,
                "development_recommendation": {"decision": "REJECT"},
            },
            {
                "candidate_name": "eligible",
                "development_objective": 0.5,
                "development_recommendation": {"decision": "PROMOTE"},
            },
        ]
        self.assertEqual(
            select_best_candidate(results)["candidate_name"],
            "eligible",
        )

    def test_uses_best_rejected_candidate_for_diagnostics_if_none_eligible(self):
        results = [
            {
                "candidate_name": "a",
                "development_objective": 0.1,
                "development_recommendation": {"decision": "REJECT"},
            },
            {
                "candidate_name": "b",
                "development_objective": 0.2,
                "development_recommendation": {"decision": "REJECT"},
            },
        ]
        self.assertEqual(select_best_candidate(results)["candidate_name"], "b")
        with self.assertRaises(OptimizationError):
            select_best_candidate([])


if __name__ == "__main__":
    unittest.main()
