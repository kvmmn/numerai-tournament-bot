from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.numerai_ops import (
    NumeraiOpsError,
    _prepare_training_frame,
    predictor_features,
)


class _FeatureFamilyLike:
    members = {
        "small": (object(), ["feature_b", "feature_a"]),
        "serenity": (object(), ["feature_c", "feature_b"]),
    }
    neutralization_features = ["feature_a", "feature_d"]


class _TargetEnsembleLike:
    features = ["feature_z", "feature_y"]


class TrainingDataValidationTests(unittest.TestCase):
    def frame(self):
        rows = [
            {
                "era": f"era{index:04d}",
                "target": 0.5,
                "feature_a": float(index % 5) / 4,
            }
            for index in reversed(range(120))
        ]
        rows.append({"era": "era0120", "target": np.nan, "feature_a": 0.5})
        return pd.DataFrame(rows)

    def test_drops_unresolved_targets_and_sorts_eras_before_stride(self):
        with (
            patch.object(settings, "TRAIN_ERA_STRIDE", 2),
            patch.object(settings, "MIN_TRAIN_ERAS", 50),
        ):
            prepared, report = _prepare_training_frame(
                self.frame(),
                ["feature_a"],
            )
        self.assertEqual(report["dropped_target_rows"], 1)
        self.assertEqual(report["selected_era_count"], 60)
        self.assertEqual(report["selected_first_era"], "era0000")
        self.assertEqual(report["selected_last_era"], "era0118")
        self.assertFalse(prepared["target"].isna().any())

    def test_rejects_nonfinite_features(self):
        frame = self.frame()
        frame.loc[0, "feature_a"] = np.inf
        with self.assertRaisesRegex(NumeraiOpsError, "non-finite"):
            _prepare_training_frame(frame, ["feature_a"])


class PredictorFeatureTests(unittest.TestCase):
    def test_collects_feature_family_member_and_neutralization_union(self):
        self.assertEqual(
            predictor_features(_FeatureFamilyLike()),
            ["feature_a", "feature_b", "feature_c", "feature_d"],
        )

    def test_uses_explicit_predictor_features(self):
        self.assertEqual(
            predictor_features(_TargetEnsembleLike()),
            ["feature_y", "feature_z"],
        )

    def test_falls_back_to_configured_feature_set(self):
        with patch(
            "app.core.numerai_ops.load_features",
            return_value=["feature_default"],
        ) as mocked:
            self.assertEqual(
                predictor_features(lambda *_: None, "small"),
                ["feature_default"],
            )
        mocked.assert_called_once_with("small")


if __name__ == "__main__":
    unittest.main()
