from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.core.runtime_manager import (
    RuntimeManager,
    RuntimeManagerError,
    sha256_file,
)


class RuntimeManagerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.runtime = self.root / "runtime" / "backend"
        self.backups = self.root / "backups"
        for relative, content in {
            "README.md": "source readme\n",
            "requirements.txt": "numpy\n",
            "requirements-test.txt": "numpy\n",
            "app/core/example.py": "VALUE = 'source-v1'\n",
            "automation/daily.py": "print('daily')\n",
            "automation/com.example.plist": "<plist/>\n",
            "automation/prompts/watch.md": "# watch\n",
            "docs/guide.md": "# guide\n",
            "tests/test_example.py": "def test_placeholder(): pass\n",
        }.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        mutable_source = self.source / "automation/state/not-source.json"
        mutable_source.parent.mkdir(parents=True, exist_ok=True)
        mutable_source.write_text("{}")

        for relative, content in {
            ".env": "SECRET=value\n",
            "app/core/example.py": "VALUE = 'runtime-old'\n",
            "automation/state/submission_ledger.json": '{"submissions": {}}\n',
            "automation/model_registry/model.pkl": "model-bytes",
            "automation/reports/report.json": '{"ok": true}\n',
        }.items():
            path = self.runtime / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.manager = RuntimeManager(
            source_root=self.source,
            runtime_root=self.runtime,
            backup_dir=self.backups,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_deploy_preserves_mutable_state_and_supports_guarded_rollback(self):
        before = self.manager.audit()
        self.assertEqual(before["status"], "RUNTIME_DRIFT")
        first = self.manager.deploy(run_tests=False)
        self.assertEqual(first["status"], "RUNTIME_DEPLOYED")
        self.assertEqual(self.manager.audit()["status"], "RUNTIME_IN_SYNC")
        self.assertEqual(
            (self.runtime / "app/core/example.py").read_text(),
            "VALUE = 'source-v1'\n",
        )
        self.assertEqual((self.runtime / ".env").read_text(), "SECRET=value\n")
        self.assertTrue(
            (self.runtime / "automation/state/submission_ledger.json").exists()
        )
        self.assertFalse(
            (self.runtime / "automation/state/not-source.json").exists()
        )

        (self.source / "app/core/example.py").write_text(
            "VALUE = 'source-v2'\n"
        )
        second = self.manager.deploy(run_tests=False)
        backup = Path(second["backup_path"])
        self.assertTrue(backup.exists())
        with self.assertRaisesRegex(RuntimeManagerError, "exactly match"):
            self.manager.rollback_code(backup, confirmation="WRONG")
        confirmation = f"ROLLBACK {sha256_file(backup)[:12]}"
        rollback = self.manager.rollback_code(
            backup,
            confirmation=confirmation,
        )
        self.assertEqual(rollback["status"], "RUNTIME_CODE_ROLLED_BACK")
        self.assertEqual(
            (self.runtime / "app/core/example.py").read_text(),
            "VALUE = 'source-v1'\n",
        )
        self.assertEqual((self.runtime / ".env").read_text(), "SECRET=value\n")

    def test_state_backup_verifies_and_restores_without_credentials(self):
        result = self.manager.create_state_backup()
        backup = Path(result["backup_path"])
        self.assertEqual(result["status"], "STATE_BACKUP_CREATED")
        self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
        self.assertEqual(
            Path(result["sidecar_path"]).stat().st_mode & 0o777,
            0o600,
        )
        verified = self.manager.verify_state_backup(backup)
        self.assertEqual(verified["status"], "STATE_BACKUP_VERIFIED")
        self.assertFalse(verified["credential_files_included"])

        destination = self.root / "restored"
        restored = self.manager.restore_state_backup(
            backup,
            destination=destination,
        )
        self.assertFalse(restored["live_restore"])
        self.assertTrue(
            (destination / "automation/state/submission_ledger.json").exists()
        )
        self.assertTrue(
            (destination / "automation/model_registry/model.pkl").exists()
        )
        self.assertFalse((destination / ".env").exists())

        with backup.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(RuntimeManagerError, "checksum"):
            self.manager.verify_state_backup(backup)

    def test_live_restore_requires_archive_bound_confirmation(self):
        result = self.manager.create_state_backup()
        backup = Path(result["backup_path"])
        with self.assertRaisesRegex(RuntimeManagerError, "exactly match"):
            self.manager.restore_state_backup(
                backup,
                destination=self.runtime,
                confirmation="WRONG",
            )
        confirmation = f"RESTORE {result['archive_sha256'][:12]}"
        restored = self.manager.restore_state_backup(
            backup,
            destination=self.runtime,
            confirmation=confirmation,
        )
        self.assertTrue(restored["live_restore"])
        self.assertTrue(Path(restored["pre_restore_backup"]).exists())

    def test_backup_retention_prunes_oldest_verified_archives(self):
        first = self.manager.create_state_backup(keep=2)
        second = self.manager.create_state_backup(keep=2)
        third = self.manager.create_state_backup(keep=2)
        archives = list((self.backups / "state").glob("*.tar.gz"))
        self.assertEqual(len(archives), 2)
        self.assertFalse(Path(first["backup_path"]).exists())
        self.assertTrue(Path(second["backup_path"]).exists())
        self.assertTrue(Path(third["backup_path"]).exists())
        self.assertEqual(third["retention_keep"], 2)


if __name__ == "__main__":
    unittest.main()
