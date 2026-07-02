from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from app.core.platform_health import PlatformHealthMonitor, REQUIRED_DATASETS


NOW = datetime(2026, 7, 2, 9, 45, tzinfo=timezone.utc)


class FakePlatformApi:
    def __init__(
        self,
        *,
        datasets: list[str] | None = None,
        stake_error: bool = False,
    ):
        self.datasets = datasets or [
            f"v5.2/{name}" for name in REQUIRED_DATASETS
        ]
        self.stake_error = stake_error

    def get_account(self):
        return {
            "username": "kvmmn",
            "availableNmr": 0,
        }

    def get_models(self):
        return {
            "kvmmn": "model-a",
            "kvmmn_fn": "model-b",
        }

    def get_current_round(self):
        return 1302

    def list_datasets(self):
        return list(self.datasets)

    def stake_get(self, model_name):
        if self.stake_error:
            raise RuntimeError("stake query contract changed")
        return 0.1 if model_name == "kvmmn" else None


class PlatformHealthMonitorTests(unittest.TestCase):
    @staticmethod
    def write_fixture(root: Path) -> tuple[Path, Path, Path]:
        state = root / "state"
        registry = root / "registry"
        data = root / "data"
        portfolio = registry / "portfolio" / "current.json"
        portfolio.parent.mkdir(parents=True)
        portfolio.write_text(
            json.dumps(
                {
                    "assignments": [
                        {"model_name": "kvmmn"},
                        {"model_name": "kvmmn_fn"},
                    ]
                }
            )
        )

        version_dir = data / "v5.2"
        version_dir.mkdir(parents=True)
        (version_dir / "features.json").write_text(
            json.dumps(
                {
                    "feature_sets": {
                        "small": ["feature_a"],
                    }
                }
            )
        )
        training = pa.table(
            {
                "era": ["era1", "era2"],
                "feature_a": [0.1, 0.2],
                "target": [0.4, 0.6],
            }
        )
        live = pa.table(
            {
                "era": ["eraX", "eraX"],
                "feature_a": [0.3, 0.7],
            }
        )
        pq.write_table(training, version_dir / "train.parquet")
        pq.write_table(training, version_dir / "validation.parquet")
        pq.write_table(live, version_dir / "live.parquet")
        return state, registry, data

    @staticmethod
    def window(_napi, round_number):
        return {
            "number": round_number,
            "openTime": "2026-07-02T08:00:00Z",
            "closeTime": "2026-07-03T12:00:00Z",
            "closeStakingTime": "2026-07-02T13:00:00Z",
            "accepting_submissions": True,
        }

    def monitor(self, root: Path, napi: FakePlatformApi) -> PlatformHealthMonitor:
        state, registry, data = self.write_fixture(root)
        return PlatformHealthMonitor(
            state,
            registry,
            data,
            data_version="v5.2",
            feature_set="small",
            napi=napi,
            now=NOW,
            round_window_resolver=self.window,
        )

    def test_current_platform_contract_passes_and_is_persisted(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.monitor(
                Path(temporary),
                FakePlatformApi(),
            ).snapshot()
            self.assertTrue(result["ok"])
            self.assertEqual(result["status"], "PLATFORM_COMPATIBLE")
            self.assertEqual(result["failures"], [])
            self.assertEqual(result["warnings"], [])
            self.assertEqual(len(result["checks"]), 7)
            self.assertTrue(Path(result["report_path"]).exists())
            latest = json.loads(Path(result["latest_path"]).read_text())
            self.assertEqual(latest["status"], "PLATFORM_COMPATIBLE")

    def test_newer_data_version_is_a_review_warning_not_a_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            datasets = [
                *(f"v5.2/{name}" for name in REQUIRED_DATASETS),
                *(f"v5.3/{name}" for name in REQUIRED_DATASETS),
            ]
            result = self.monitor(
                Path(temporary),
                FakePlatformApi(datasets=datasets),
            ).snapshot()
            self.assertTrue(result["ok"])
            self.assertEqual(result["status"], "PLATFORM_MIGRATION_REVIEW")
            self.assertEqual(result["warnings"], ["REMOTE_DATA_VERSION"])

    def test_missing_remote_data_and_broken_stake_read_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.monitor(
                Path(temporary),
                FakePlatformApi(
                    datasets=["v5.2/features.json"],
                    stake_error=True,
                ),
            ).snapshot()
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "PLATFORM_CONTRACT_BROKEN")
            self.assertIn("REMOTE_DATA_VERSION", result["failures"])
            self.assertIn("STAKE_READ_API", result["failures"])


if __name__ == "__main__":
    unittest.main()
