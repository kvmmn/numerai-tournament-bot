from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.metrics_ledger import MetricsLedger, read_metrics_history


NOW = datetime(2026, 11, 17, 18, 5, tzinfo=timezone.utc)


def performance(round_number, corr, mmc, *, resolved=True):
    return {
        "roundNumber": round_number,
        "roundResolved": resolved,
        "atRisk": 0,
        "submissionScores": [
            {"displayName": "canon_corr", "value": corr},
            {"displayName": "canon_mmc", "value": mmc},
        ],
    }


class FakeNumerai:
    def get_models(self):
        return {"alpha": "model-a", "beta": "model-b"}

    def get_current_round(self):
        return 1305

    def round_model_performances_v2(self, model_id):
        if model_id == "model-a":
            return [
                performance(1301, 0.9, 0.9),
                performance(1302, 0.01, 0.001),
                performance(1303, 0.04, 0.005),
                performance(1304, -0.01, 0.002),
            ]
        return [
            performance(1302, 0.03, 0.003),
            performance(1304, 0.02, 0.004),
        ]

    def submission_ids(self, model_id):
        if model_id == "model-a":
            return [
                {"insertedAt": "2026-07-01T23:59:59Z", "id": "old"},
                {"insertedAt": "2026-07-02T00:00:00Z", "id": "a1"},
                {"insertedAt": "2026-07-02T00:00:01Z", "id": "a2"},
            ]
        return [{"insertedAt": "2026-07-03T00:00:00Z", "id": "b1"}]


class MetricsLedgerTests(unittest.TestCase):
    def fixture(self, root: Path):
        backend = root / "backend"
        state = root / "state"
        data = root / "data"
        version = data / "v5.2"
        version.mkdir(parents=True)
        for name, content in (
            ("train.parquet", b"train-data"),
            ("validation.parquet", b"validation-data"),
            ("live.parquet", b"live-data"),
        ):
            (version / name).write_bytes(content)
        (version / "features.json").write_text(
            json.dumps({"feature_sets": {"small": ["f1", "f2"]}})
        )
        reports = backend / "automation" / "reports"
        reports.mkdir(parents=True)
        for suffix in ("json", "md"):
            (reports / f"20261117_180000_portfolio-status.{suffix}").write_text("{}")
        (reports / "20261117_180100_score-listen.json").write_text("{}")
        logs = backend / "automation" / "logs"
        logs.mkdir(parents=True)
        (logs / "daily.log").write_text("ok")
        (backend / "automation" / "com.numerai.test.plist").write_text("<plist/>")
        (backend / "app").mkdir()
        (backend / "app" / "example.py").write_text("x = 1\n")
        tests = backend / "tests"
        tests.mkdir()
        (tests / "test_example.py").write_text(
            "def test_one():\n    assert True\n"
        )
        state.mkdir()
        (state / "submission_ledger.json").write_text(
            json.dumps({
                "1302:model-a": {"verified": True},
                "1304:model-b": {"verified": False},
            })
        )
        return backend, state, data

    @staticmethod
    def parquet_reader(path: Path):
        rows = {"train.parquet": 10, "validation.parquet": 20, "live.parquet": 3}
        return {"rows": rows[path.name], "columns": 5, "row_groups": 1}

    def ledger(self, backend, state, data, *, now=NOW, napi=None):
        return MetricsLedger(
            backend_root=backend,
            state_dir=state,
            data_dir=data,
            napi=napi or FakeNumerai(),
            now=now,
            start_round=1302,
            recent_rounds=2,
            parquet_reader=self.parquet_reader,
        )

    def test_snapshot_keeps_uploads_records_and_portfolio_rounds_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend, state, data = self.fixture(Path(temporary))
            result = self.ledger(backend, state, data).snapshot()

            self.assertTrue(result["ok"])
            metrics = result["metrics"]
            self.assertEqual(metrics["numerai.models"]["value"], 2)
            self.assertEqual(metrics["numerai.performance_records"]["value"], 5)
            self.assertEqual(metrics["numerai.distinct_rounds"]["value"], 3)
            self.assertEqual(metrics["numerai.raw_upload_records"]["value"], 3)
            portfolio = result["details"]["numerai"]["portfolio_recent"]
            self.assertEqual(portfolio["round_numbers"], [1302, 1304])
            self.assertAlmostEqual(portfolio["mean_corr"], 0.0125)
            self.assertAlmostEqual(portfolio["mean_mmc"], 0.0025)
            self.assertEqual(portfolio["positive_corr_rate"], 1.0)
            self.assertEqual(metrics["dataset.total_rows"]["value"], 33)
            self.assertEqual(metrics["operations.paired_report_runs"]["value"], 1)
            self.assertEqual(metrics["operations.report_artifacts"]["value"], 3)
            self.assertEqual(metrics["quality.test_functions_defined"]["value"], 1)
            for row in metrics.values():
                for key in ("value", "unit", "source", "measured_at", "scope", "definition"):
                    self.assertIn(key, row)
            for name in ("latest.json", "latest.md", "latest_successful.json", "index.json"):
                self.assertTrue((state / "metrics" / name).is_file())

    def test_partial_snapshot_preserves_last_successful(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend, state, data = self.fixture(Path(temporary))
            first = self.ledger(backend, state, data).snapshot()
            (data / "v5.2" / "live.parquet").unlink()
            second = self.ledger(
                backend, state, data, now=NOW + timedelta(minutes=1)
            ).snapshot()

            self.assertFalse(second["ok"])
            self.assertEqual(second["status"], "METRICS_SNAPSHOT_PARTIAL")
            self.assertFalse(second["sources"]["dataset"]["ok"])
            latest = json.loads((state / "metrics" / "latest.json").read_text())
            successful = json.loads(
                (state / "metrics" / "latest_successful.json").read_text()
            )
            self.assertEqual(latest["snapshot_id"], second["snapshot_id"])
            self.assertEqual(successful["snapshot_id"], first["snapshot_id"])

    def test_unclassifiable_upload_history_is_partial_not_zero(self):
        class ChangedUploadSchema(FakeNumerai):
            def submission_ids(self, model_id):
                return [{"id": f"unknown-{model_id}"}]

        with tempfile.TemporaryDirectory() as temporary:
            backend, state, data = self.fixture(Path(temporary))
            result = self.ledger(
                backend, state, data, napi=ChangedUploadSchema()
            ).snapshot()
            self.assertFalse(result["ok"])
            self.assertFalse(result["sources"]["numerai"]["ok"])
            self.assertNotIn("numerai.raw_upload_records", result["metrics"])
            self.assertTrue(result["details"]["numerai"]["raw_upload_errors"])

    def test_history_is_newest_first_and_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend, state, data = self.fixture(Path(temporary))
            first = self.ledger(backend, state, data).snapshot()
            second = self.ledger(
                backend, state, data, now=NOW + timedelta(minutes=1)
            ).snapshot()
            history = read_metrics_history(state, limit=1)
            self.assertEqual(history["total_indexed"], 2)
            self.assertEqual(history["snapshots"][0]["snapshot_id"], second["snapshot_id"])
            self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])


if __name__ == "__main__":
    unittest.main()
