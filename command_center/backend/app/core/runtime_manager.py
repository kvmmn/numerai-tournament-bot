from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


class RuntimeManagerError(RuntimeError):
    pass


ROOT_FILES = {
    ".env.example",
    "README.md",
    "numerai_mcp_codex_config.toml.example",
    "requirements-test.txt",
    "requirements.txt",
}
MUTABLE_PATHS = (
    Path("automation/state"),
    Path("automation/model_registry"),
    Path("automation/reports"),
)
BACKUP_MANIFEST_NAME = "BACKUP_MANIFEST.json"
DEPLOYMENT_MANIFEST_NAME = ".runtime_deployment.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def _safe_relative(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeManagerError(f"Unsafe relative path: {path}")
    return path


def _is_managed_relative(path: Path) -> bool:
    path = _safe_relative(path)
    if len(path.parts) == 1:
        return path.name in ROOT_FILES or path.name == DEPLOYMENT_MANIFEST_NAME
    if path.parts[0] == "app":
        return path.suffix == ".py" and "__pycache__" not in path.parts
    if path.parts[0] == "docs":
        return path.suffix == ".md"
    if path.parts[0] == "tests":
        return path.suffix == ".py" and "__pycache__" not in path.parts
    if path.parts[0] == "automation":
        if len(path.parts) == 2:
            return path.suffix in {".html", ".md", ".plist", ".py"}
        return (
            len(path.parts) == 3
            and path.parts[1] == "prompts"
            and path.suffix == ".md"
        )
    return False


def managed_source_files(source_root: str | Path) -> list[Path]:
    source_root = Path(source_root).resolve()
    files: set[Path] = set()
    for name in ROOT_FILES:
        path = source_root / name
        if path.is_file():
            files.add(path.relative_to(source_root))
    for path in (source_root / "app").rglob("*.py"):
        if path.is_file() and "__pycache__" not in path.parts:
            files.add(path.relative_to(source_root))
    automation = source_root / "automation"
    if automation.exists():
        for path in automation.iterdir():
            if path.is_file() and path.suffix in {
                ".html",
                ".md",
                ".plist",
                ".py",
            }:
                files.add(path.relative_to(source_root))
        prompts = automation / "prompts"
        if prompts.exists():
            for path in prompts.glob("*.md"):
                files.add(path.relative_to(source_root))
    for directory, suffixes in (
        ("docs", {".md"}),
        ("tests", {".py"}),
    ):
        root = source_root / directory
        if root.exists():
            for path in root.rglob("*"):
                if (
                    path.is_file()
                    and path.suffix in suffixes
                    and "__pycache__" not in path.parts
                ):
                    files.add(path.relative_to(source_root))
    return sorted(files, key=str)


def _manifest_for_files(root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    files = {}
    for relative in paths:
        relative = _safe_relative(relative)
        path = root / relative
        if not path.is_file():
            raise RuntimeManagerError(f"Managed file is missing: {path}")
        files[relative.as_posix()] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "mode": path.stat().st_mode & 0o777,
        }
    return {"files": files}


def _source_revision(source_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class RuntimeManager:
    def __init__(
        self,
        *,
        source_root: str | Path,
        runtime_root: str | Path,
        backup_dir: str | Path | None = None,
    ):
        self.source_root = Path(source_root).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.backup_dir = Path(
            backup_dir
            or self.runtime_root.parents[1] / "backups"
        ).resolve()
        self.deployment_manifest_path = (
            self.runtime_root / DEPLOYMENT_MANIFEST_NAME
        )

    def _require_distinct_source(self) -> None:
        if self.source_root == self.runtime_root:
            raise RuntimeManagerError(
                "Source and runtime roots must be different for code operations."
            )

    def audit(self) -> dict[str, Any]:
        self._require_distinct_source()
        paths = managed_source_files(self.source_root)
        source_manifest = _manifest_for_files(self.source_root, paths)
        missing = []
        changed = []
        matched = []
        for name, expected in source_manifest["files"].items():
            target = self.runtime_root / name
            if not target.is_file():
                missing.append(name)
            elif sha256_file(target) != expected["sha256"]:
                changed.append(name)
            else:
                matched.append(name)
        deployed = {}
        if self.deployment_manifest_path.exists():
            deployed = json.loads(self.deployment_manifest_path.read_text())
        stale = sorted(
            set(deployed.get("files", {})) - set(source_manifest["files"])
        )
        in_sync = not missing and not changed and not stale
        return {
            "ok": True,
            "status": "RUNTIME_IN_SYNC" if in_sync else "RUNTIME_DRIFT",
            "read_only": True,
            "source_root": str(self.source_root),
            "runtime_root": str(self.runtime_root),
            "source_revision": _source_revision(self.source_root),
            "managed_file_count": len(paths),
            "matched_count": len(matched),
            "missing": missing,
            "changed": changed,
            "stale": stale,
        }

    def _run_source_tests(self, python_path: Path | None = None) -> None:
        python_path = python_path or (
            self.runtime_root.parent / ".venv" / "bin" / "python"
        )
        if not python_path.is_file():
            raise RuntimeManagerError(f"Runtime Python is missing: {python_path}")
        result = subprocess.run(
            [
                str(python_path),
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
            ],
            cwd=self.source_root,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeManagerError(
                "Source tests failed before deployment:\n"
                + result.stdout
                + result.stderr
            )

    def _code_backup(self, paths: Iterable[Path], deployment_id: str) -> Path:
        destination = self.backup_dir / "code" / f"{deployment_id}.tar.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination, "w:gz") as archive:
            for relative in paths:
                target = self.runtime_root / relative
                if target.is_file():
                    archive.add(target, arcname=relative.as_posix(), recursive=False)
            if self.deployment_manifest_path.is_file():
                archive.add(
                    self.deployment_manifest_path,
                    arcname=DEPLOYMENT_MANIFEST_NAME,
                    recursive=False,
                )
        os.chmod(destination, 0o600)
        sidecar = destination.with_suffix(destination.suffix + ".sha256.json")
        _atomic_json(
            sidecar,
            {
                "archive": destination.name,
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            },
        )
        os.chmod(sidecar, 0o600)
        return destination

    def _restore_code_archive(
        self,
        backup_path: Path,
        *,
        remove_paths: Iterable[Path],
    ) -> int:
        backup_path = backup_path.resolve()
        sidecar = backup_path.with_suffix(backup_path.suffix + ".sha256.json")
        if sidecar.exists():
            expected = json.loads(sidecar.read_text())
            if expected.get("sha256") != sha256_file(backup_path):
                raise RuntimeManagerError(
                    "Code backup archive checksum does not match sidecar."
                )
        with tarfile.open(backup_path, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                relative = self._validate_member_name(member.name)
                if not member.isfile() or not _is_managed_relative(relative):
                    raise RuntimeManagerError(
                        f"Code backup contains unsupported member: {member.name}"
                    )
            paths_to_remove = set(remove_paths) | {
                Path(DEPLOYMENT_MANIFEST_NAME)
            }
            for relative in paths_to_remove:
                relative = _safe_relative(relative)
                if not _is_managed_relative(relative):
                    raise RuntimeManagerError(
                        f"Refusing to remove unmanaged runtime path: {relative}"
                    )
                target = self.runtime_root / relative
                if target.is_file():
                    target.unlink()
            restored = 0
            for member in members:
                relative = self._validate_member_name(member.name)
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeManagerError(
                        f"Code backup member cannot be read: {member.name}"
                    )
                target = self.runtime_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=target.parent,
                    prefix=f".{target.name}.rollback-",
                    delete=False,
                ) as temporary:
                    shutil.copyfileobj(handle, temporary)
                    temporary_path = Path(temporary.name)
                os.chmod(temporary_path, member.mode & 0o777)
                os.replace(temporary_path, target)
                restored += 1
        return restored

    def deploy(
        self,
        *,
        run_tests: bool = True,
        python_path: str | Path | None = None,
    ) -> dict[str, Any]:
        self._require_distinct_source()
        if run_tests:
            self._run_source_tests(
                Path(python_path).resolve() if python_path else None
            )
        paths = managed_source_files(self.source_root)
        source_manifest = _manifest_for_files(self.source_root, paths)
        deployment_id = _utc_now().strftime("%Y%m%dT%H%M%S%fZ")
        old_manifest = {}
        if self.deployment_manifest_path.exists():
            old_manifest = json.loads(self.deployment_manifest_path.read_text())
        stale = sorted(
            set(old_manifest.get("files", {})) - set(source_manifest["files"])
        )
        backup_paths = sorted(
            set(paths) | {Path(name) for name in stale},
            key=str,
        )
        backup_path = self._code_backup(backup_paths, deployment_id)

        staging_root = (
            self.runtime_root.parent
            / ".deploy-staging"
            / deployment_id
            / "backend"
        )
        try:
            for relative in paths:
                source = self.source_root / relative
                staged = staging_root / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, staged)
            staged_manifest = _manifest_for_files(staging_root, paths)
            if staged_manifest["files"] != source_manifest["files"]:
                raise RuntimeManagerError("Staged deployment checksum mismatch.")

            for relative in paths:
                staged = staging_root / relative
                target = self.runtime_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".deploying")
                shutil.copy2(staged, temporary)
                os.replace(temporary, target)
            for name in stale:
                relative = _safe_relative(Path(name))
                target = self.runtime_root / relative
                if target.is_file():
                    target.unlink()

            deployment_manifest = {
                "schema_version": 1,
                "deployment_id": deployment_id,
                "deployed_at": _utc_now().isoformat(),
                "source_root": str(self.source_root),
                "source_revision": _source_revision(self.source_root),
                "backup_path": str(backup_path),
                "files": source_manifest["files"],
            }
            _atomic_json(self.deployment_manifest_path, deployment_manifest)
        except Exception:
            self._restore_code_archive(
                backup_path,
                remove_paths=backup_paths,
            )
            raise
        finally:
            shutil.rmtree(staging_root.parent, ignore_errors=True)

        audit = self.audit()
        if audit["status"] != "RUNTIME_IN_SYNC":
            raise RuntimeManagerError(
                f"Runtime verification failed after deployment: {audit}"
            )
        return {
            "ok": True,
            "status": "RUNTIME_DEPLOYED",
            "deployment_id": deployment_id,
            "backup_path": str(backup_path),
            "source_revision": deployment_manifest["source_revision"],
            "managed_file_count": len(paths),
            "audit": audit,
        }

    def rollback_code(
        self,
        backup_path: str | Path,
        *,
        confirmation: str,
    ) -> dict[str, Any]:
        self._require_distinct_source()
        backup_path = Path(backup_path).resolve()
        if not backup_path.is_file():
            raise RuntimeManagerError(f"Code backup is missing: {backup_path}")
        archive_sha256 = sha256_file(backup_path)
        expected_confirmation = f"ROLLBACK {archive_sha256[:12]}"
        if confirmation != expected_confirmation:
            raise RuntimeManagerError(
                f"Code rollback confirmation must exactly match: "
                f"{expected_confirmation}"
            )
        current_manifest = {}
        if self.deployment_manifest_path.exists():
            current_manifest = json.loads(self.deployment_manifest_path.read_text())
        current_paths = [
            Path(name) for name in current_manifest.get("files", {})
        ]
        rollback_id = _utc_now().strftime("%Y%m%dT%H%M%S%fZ")
        pre_rollback_backup = self._code_backup(current_paths, rollback_id)
        restored = self._restore_code_archive(
            backup_path,
            remove_paths=current_paths,
        )
        return {
            "ok": True,
            "status": "RUNTIME_CODE_ROLLED_BACK",
            "backup_path": str(backup_path),
            "archive_sha256": archive_sha256,
            "pre_rollback_backup": str(pre_rollback_backup),
            "restored_file_count": restored,
        }

    def _mutable_files(self) -> list[Path]:
        paths = []
        for relative_root in MUTABLE_PATHS:
            root = self.runtime_root / relative_root
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise RuntimeManagerError(
                        f"Mutable runtime backup rejects symlinks: {path}"
                    )
                if path.is_file():
                    paths.append(path.relative_to(self.runtime_root))
        if self.deployment_manifest_path.is_file():
            paths.append(Path(DEPLOYMENT_MANIFEST_NAME))
        return sorted(set(paths), key=str)

    def create_state_backup(self, *, keep: int = 7) -> dict[str, Any]:
        if keep < 1:
            raise RuntimeManagerError("Backup retention must keep at least one archive.")
        paths = self._mutable_files()
        if not paths:
            raise RuntimeManagerError("No mutable runtime files were found.")
        stamp = _utc_now().strftime("%Y%m%dT%H%M%S%fZ")
        destination = self.backup_dir / "state" / f"numerai-state-{stamp}.tar.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        files_manifest = _manifest_for_files(self.runtime_root, paths)["files"]
        manifest = {
            "schema_version": 1,
            "created_at": _utc_now().isoformat(),
            "runtime_root": str(self.runtime_root),
            "credential_files_included": False,
            "files": files_manifest,
        }
        with tarfile.open(destination, "w:gz") as archive:
            for relative in paths:
                archive.add(
                    self.runtime_root / relative,
                    arcname=relative.as_posix(),
                    recursive=False,
                )
            encoded = (
                json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
            )
            info = tarfile.TarInfo(BACKUP_MANIFEST_NAME)
            info.size = len(encoded)
            info.mtime = int(_utc_now().timestamp())
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(encoded))
        os.chmod(destination, 0o600)
        archive_sha256 = sha256_file(destination)
        sidecar = destination.with_suffix(destination.suffix + ".sha256.json")
        _atomic_json(
            sidecar,
            {
                "archive": destination.name,
                "sha256": archive_sha256,
                "size_bytes": destination.stat().st_size,
            },
        )
        os.chmod(sidecar, 0o600)
        verification = self.verify_state_backup(destination)
        archives = sorted(
            (self.backup_dir / "state").glob("numerai-state-*.tar.gz"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        pruned = []
        for old in archives[keep:]:
            sidecar = old.with_suffix(old.suffix + ".sha256.json")
            old.unlink()
            if sidecar.exists():
                sidecar.unlink()
            pruned.append(str(old))
        return {
            "ok": True,
            "status": "STATE_BACKUP_CREATED",
            "backup_path": str(destination),
            "sidecar_path": str(sidecar),
            "archive_sha256": archive_sha256,
            "file_count": len(paths),
            "retention_keep": keep,
            "pruned_backups": pruned,
            "verification": verification,
        }

    @staticmethod
    def _validate_member_name(name: str) -> Path:
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise RuntimeManagerError(f"Unsafe backup member: {name}")
        relative = Path(*pure.parts)
        if relative.name == ".env" or any(
            part.casefold() in {"credentials", "secrets"} for part in relative.parts
        ):
            raise RuntimeManagerError(
                f"Credential-like file is forbidden in backup: {name}"
            )
        return relative

    def verify_state_backup(self, backup_path: str | Path) -> dict[str, Any]:
        backup_path = Path(backup_path).resolve()
        if not backup_path.is_file():
            raise RuntimeManagerError(f"Backup is missing: {backup_path}")
        archive_sha256 = sha256_file(backup_path)
        sidecar = backup_path.with_suffix(backup_path.suffix + ".sha256.json")
        if sidecar.exists():
            expected = json.loads(sidecar.read_text())
            if expected.get("sha256") != archive_sha256:
                raise RuntimeManagerError("Backup archive checksum does not match sidecar.")
        with tarfile.open(backup_path, "r:gz") as archive:
            members = {member.name: member for member in archive.getmembers()}
            for member in members.values():
                self._validate_member_name(member.name)
                if member.issym() or member.islnk() or not member.isfile():
                    raise RuntimeManagerError(
                        f"Backup contains unsupported member: {member.name}"
                    )
            manifest_member = members.get(BACKUP_MANIFEST_NAME)
            if manifest_member is None:
                raise RuntimeManagerError("Backup manifest is missing.")
            manifest_handle = archive.extractfile(manifest_member)
            if manifest_handle is None:
                raise RuntimeManagerError("Backup manifest cannot be read.")
            manifest = json.loads(manifest_handle.read())
            expected_files = manifest.get("files", {})
            archived_files = set(members) - {BACKUP_MANIFEST_NAME}
            if archived_files != set(expected_files):
                raise RuntimeManagerError("Backup file list does not match manifest.")
            for name, expected in expected_files.items():
                handle = archive.extractfile(members[name])
                if handle is None:
                    raise RuntimeManagerError(f"Backup member cannot be read: {name}")
                digest = _sha256_stream(handle)
                if digest != expected["sha256"]:
                    raise RuntimeManagerError(
                        f"Backup member checksum mismatch: {name}"
                    )
        return {
            "ok": True,
            "status": "STATE_BACKUP_VERIFIED",
            "backup_path": str(backup_path),
            "archive_sha256": archive_sha256,
            "file_count": len(expected_files),
            "credential_files_included": False,
        }

    def restore_state_backup(
        self,
        backup_path: str | Path,
        *,
        destination: str | Path,
        confirmation: str | None = None,
    ) -> dict[str, Any]:
        backup_path = Path(backup_path).resolve()
        verification = self.verify_state_backup(backup_path)
        destination = Path(destination).resolve()
        live_restore = destination == self.runtime_root
        expected_confirmation = (
            f"RESTORE {verification['archive_sha256'][:12]}"
        )
        pre_restore_backup = None
        if live_restore:
            if confirmation != expected_confirmation:
                raise RuntimeManagerError(
                    f"Live restore confirmation must exactly match: "
                    f"{expected_confirmation}"
                )
            pre_restore_backup = self.create_state_backup()["backup_path"]
        destination.mkdir(parents=True, exist_ok=True)
        restored = []
        with tarfile.open(backup_path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.name == BACKUP_MANIFEST_NAME:
                    continue
                relative = self._validate_member_name(member.name)
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeManagerError(
                        f"Backup member cannot be read: {member.name}"
                    )
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=target.parent,
                    prefix=f".{target.name}.restore-",
                    delete=False,
                ) as temporary:
                    shutil.copyfileobj(handle, temporary)
                    temporary_path = Path(temporary.name)
                os.chmod(temporary_path, member.mode & 0o777)
                os.replace(temporary_path, target)
                restored.append(relative.as_posix())
        return {
            "ok": True,
            "status": "STATE_RESTORED",
            "destination": str(destination),
            "live_restore": live_restore,
            "pre_restore_backup": pre_restore_backup,
            "restored_file_count": len(restored),
            "verification": verification,
        }
