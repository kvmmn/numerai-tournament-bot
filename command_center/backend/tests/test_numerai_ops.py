from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.numerai_ops import NumeraiOpsError, _prepare_training_frame


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


if __name__ == "__main__":
    unittest.main()
