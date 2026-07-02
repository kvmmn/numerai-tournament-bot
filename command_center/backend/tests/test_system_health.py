from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.system_health import (
    SystemHealthMonitor,
    inspect_launchd_service,
)


NOW = datetime(2026, 7, 2, 17, 10, tzinfo=timezone.utc)


class SystemHealthMonitorTests(unittest.TestCase):
    @staticmethod
    def write(path: Path, payload: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if payload is None:
            path.write_bytes(b"archive")
        else:
            path.write_text(json.dumps(payload))
        os.utime(path, (NOW.timestamp(), NOW.timestamp()))

    def fixture(self, root: Path) -> tuple[Path, Path, Path]:
        state = root / "state"
        reports = root / "reports"
        backup = root / "backups"
        self.write(
            state / "platform" / "latest.json",
            {"ok": True, "status": "PLATFORM_COMPATIBLE"},
        )
        self.write(
            state / "competition" / "latest.json",
            {"ok": True, "status": "COMPETITION_ACTION_REQUIRED"},
        )
        self.write(
            state / "alerts" / "ledger.json",
            {"updated_at": NOW.isoformat(), "delivered": {}},
        )
        for mode, status in (
            ("portfolio-status", "DEADLINE_GUARD_ACTION_REQUIRED"),
            ("portfolio-prepare", "PORTFOLIO_AWAITING_HUMAN_APPROVAL"),
            ("score-listen", "NO_NEW_OUTCOMES"),
            ("stake-status", "STAKE_POLICY_ACTION_REQUIRED"),
            ("research-evaluate", "RESEARCH_PROMOTE"),
        ):
            self.write(
                reports / f"20260702_170000_{mode}.json",
                {"ok": True, "status": status},
            )
        self.write(backup / "numerai-state-20260702T170000Z.tar.gz")
        return state, reports, backup

    @staticmethod
    def healthy_service(_label: str):
        return {
            "loaded": True,
            "state": "not running",
            "runs": 1,
            "last_exit_code": 0,
            "error": None,
        }

    def test_all_loaded_jobs_with_fresh_evidence_are_healthy(self):
        with tempfile.TemporaryDirectory() as temporary:
            state, reports, backup = self.fixture(Path(temporary))
            result = SystemHealthMonitor(
                state,
                reports,
                backup,
                now=NOW,
                service_inspector=self.healthy_service,
            ).snapshot()
            self.assertTrue(result["ok"])
            self.assertEqual(result["status"], "SYSTEM_HEALTHY")
            self.assertEqual(result["healthy_job_count"], 9)
            self.assertEqual(result["expected_job_count"], 9)
            self.assertTrue(Path(result["latest_path"]).exists())

    def test_stale_evidence_and_nonzero_exit_are_actionable(self):
        with tempfile.TemporaryDirectory() as temporary:
            state, reports, backup = self.fixture(Path(temporary))
            stale = reports / "20260702_170000_portfolio-status.json"
            timestamp = (NOW - timedelta(hours=31)).timestamp()
            os.utime(stale, (timestamp, timestamp))

            def service(label: str):
                payload = dict(self.healthy_service(label))
                if label == "com.numerai.staking":
                    payload["last_exit_code"] = 1
                return payload

            result = SystemHealthMonitor(
                state,
                reports,
                backup,
                now=NOW,
                service_inspector=service,
            ).snapshot()
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "SYSTEM_HEALTH_ACTION_REQUIRED")
            issues = {(row["label"], row["code"]) for row in result["issues"]}
            self.assertIn(
                ("com.numerai.deadline", "EVIDENCE_STALE"),
                issues,
            )
            self.assertIn(
                ("com.numerai.staking", "LAST_EXIT_NONZERO"),
                issues,
            )

    def test_failed_latest_report_is_not_treated_as_fresh_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            state, reports, backup = self.fixture(Path(temporary))
            self.write(
                reports / "20260702_171000_score-listen.json",
                {"ok": False, "status": "failed"},
            )
            result = SystemHealthMonitor(
                state,
                reports,
                backup,
                now=NOW,
                service_inspector=self.healthy_service,
            ).snapshot()
            outcomes = next(
                row
                for row in result["jobs"]
                if row["label"] == "com.numerai.outcomes"
            )
            self.assertIn("LAST_REPORT_FAILED", outcomes["issues"])
            self.assertTrue(
                outcomes["evidence"]["path"].endswith(
                    "20260702_171000_score-listen.json"
                )
            )

    def test_launchd_parser_reads_top_level_status(self):
        stdout = """
gui/501/com.numerai.example = {
\tstate = not running
\truns = 3
\tlast exit code = 0
\tspawn type = daemon (3)
\t\tstate = active
}
"""

        def runner(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

        result = inspect_launchd_service(
            "com.numerai.example",
            uid=501,
            runner=runner,
        )
        self.assertTrue(result["loaded"])
        self.assertEqual(result["state"], "not running")
        self.assertEqual(result["runs"], 3)
        self.assertEqual(result["last_exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
