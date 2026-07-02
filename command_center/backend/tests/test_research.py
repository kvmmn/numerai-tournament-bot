from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.core.research import (
    RobustnessPolicy,
    bootstrap_mean_interval,
    build_walk_forward_folds,
    build_robustness_packet,
    per_era_correlation,
    promotion_recommendation,
    summarize_era_scores,
)


class ResearchMetricsTests(unittest.TestCase):
    def test_per_era_correlation_and_summary(self):
        targets = pd.Series([0.0, 1.0, 0.0, 1.0])
        predictions = pd.Series([0.1, 0.9, 0.8, 0.2])
        eras = pd.Series(["a", "a", "b", "b"])
        scores = per_era_correlation(targets, predictions, eras)
        self.assertAlmostEqual(scores.iloc[0], 1.0)
        self.assertAlmostEqual(scores.iloc[1], -1.0)
        summary = summarize_era_scores(scores)
        self.assertEqual(summary["era_count"], 2)
        self.assertAlmostEqual(summary["mean_correlation"], 0.0)

    def test_bootstrap_is_deterministic(self):
        scores = pd.Series([0.01, 0.02, 0.03, -0.01])
        first = bootstrap_mean_interval(scores, samples=100, seed=7)
        second = bootstrap_mean_interval(scores, samples=100, seed=7)
        self.assertEqual(first, second)

    def test_walk_forward_folds_preserve_time_embargo_and_lockbox(self):
        eras = [f"era{index:04d}" for index in range(400)]
        result = build_walk_forward_folds(
            list(reversed(eras)),
            minimum_train_eras=200,
            validation_eras=50,
            embargo_eras=5,
            lockbox_eras=50,
        )
        self.assertEqual(result["lockbox_eras"], eras[-50:])
        self.assertGreaterEqual(len(result["folds"]), 1)
        for fold in result["folds"]:
            train = fold["train_eras"]
            embargo = fold["embargo_eras"]
            validation = fold["validation_eras"]
            self.assertTrue(set(train).isdisjoint(embargo))
            self.assertTrue(set(train).isdisjoint(validation))
            self.assertTrue(set(embargo).isdisjoint(validation))
            self.assertLess(train[-1], embargo[0])
            self.assertLess(embargo[-1], validation[0])
            self.assertTrue(set(validation).isdisjoint(result["lockbox_eras"]))


class PromotionPolicyTests(unittest.TestCase):
    def packet(self, temp_dir: str, scores: list[float], exposure: float):
        artifact = Path(temp_dir) / "model.pkl"
        artifact.write_bytes(b"model")
        return build_robustness_packet(
            pd.Series(scores),
            feature_exposure=exposure,
            artifact_path=artifact,
            model_name="candidate",
        )

    def test_promotes_robust_candidate_without_champion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = self.packet(
                temp_dir,
                [0.015, 0.025] * 60,
                0.05,
            )
            recommendation = promotion_recommendation(candidate, champion=None)
            self.assertEqual(recommendation["decision"], "PROMOTE")

    def test_rejects_exposure_and_recent_regression(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = self.packet(
                temp_dir,
                [0.03] * 20 + [-0.01] * 100,
                0.2,
            )
            recommendation = promotion_recommendation(
                candidate,
                champion=None,
                policy=RobustnessPolicy(minimum_mean_correlation=-1.0),
            )
            self.assertEqual(recommendation["decision"], "REJECT")
            self.assertIn("feature_exposure", recommendation["failures"])
            self.assertIn("recent_50", recommendation["failures"])
            self.assertIn("recent_100", recommendation["failures"])

    def test_requires_champion_improvement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            champion = self.packet(temp_dir, [0.02] * 120, 0.05)
            candidate = self.packet(temp_dir, [0.015] * 120, 0.05)
            recommendation = promotion_recommendation(
                candidate,
                champion=champion,
            )
            self.assertEqual(recommendation["decision"], "REJECT")
            self.assertIn(
                "does_not_improve_champion_mean",
                recommendation["failures"],
            )


if __name__ == "__main__":
    unittest.main()
