from __future__ import annotations

import unittest

from app.core.portfolio_research import shadow_recommendation


def packet() -> dict:
    return {
        "packet_sha256": "packet",
        "feature_exposure": 0.08,
        "bootstrap_mean_correlation": {"lower": 0.008},
        "overall": {
            "mean_correlation": 0.0095,
            "sharpe": 0.7,
            "max_drawdown": 0.15,
        },
        "recent": {
            "25": {"mean_correlation": 0.003},
            "50": {"mean_correlation": -0.0005},
            "100": {"mean_correlation": 0.004},
        },
    }


class ShadowPolicyTests(unittest.TestCase):
    def test_accepts_diverse_zero_stake_forward_candidate(self):
        result = shadow_recommendation(
            packet(),
            maximum_observed_portfolio_correlation=0.7,
        )
        self.assertEqual(result["decision"], "DEPLOY_SHADOW")
        self.assertFalse(result["stake_eligible"])

    def test_rejects_duplicate_or_recently_weak_candidate(self):
        weak = packet()
        weak["recent"]["50"]["mean_correlation"] = -0.01
        result = shadow_recommendation(
            weak,
            maximum_observed_portfolio_correlation=0.99,
        )
        self.assertEqual(result["decision"], "REJECT")
        self.assertEqual(
            result["failures"],
            ["recent_50", "portfolio_diversity"],
        )


if __name__ == "__main__":
    unittest.main()
