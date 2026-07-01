from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class ModelRegistryError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": int(stat.st_size),
        "modified_ns": int(stat.st_mtime_ns),
    }


def create_candidate_bundle(
    registry_dir: str | Path,
    *,
    name: str,
    members: list[dict[str, Any]],
    evaluation_packet_path: str | Path,
    recommendation_path: str | Path,
    data_files: list[str | Path],
    configuration: dict[str, Any],
    code_revision: str | None,
    approval_ttl_hours: int = 24,
    now: datetime | None = None,
) -> tuple[dict[str, Any], Path]:
    now = now or datetime.now(timezone.utc)
    if not members:
        raise ModelRegistryError("A candidate bundle requires at least one model member.")
    weight_sum = sum(float(member["weight"]) for member in members)
    if abs(weight_sum - 1.0) > 1e-9:
        raise ModelRegistryError("Candidate ensemble weights must sum to 1.0.")

    evaluation_packet_path = Path(evaluation_packet_path)
    recommendation_path = Path(recommendation_path)
    if not evaluation_packet_path.exists() or not recommendation_path.exists():
        raise ModelRegistryError("Evaluation packet and recommendation must both exist.")
    recommendation = json.loads(recommendation_path.read_text())
    if recommendation.get("decision") != "PROMOTE":
        raise ModelRegistryError("Only a PROMOTE recommendation can enter approval.")

    source_records = []
    for member in members:
        source = Path(member["path"])
        if not source.exists():
            raise ModelRegistryError(f"Candidate model member is missing: {source}")
        source_records.append(
            {
                "name": member["name"],
                "weight": float(member["weight"]),
                "source": _file_record(source),
            }
        )
    identity_payload = {
        "name": name,
        "members": source_records,
        "evaluation_sha256": _sha256(evaluation_packet_path),
        "recommendation_sha256": _sha256(recommendation_path),
        "configuration": configuration,
        "code_revision": code_revision,
    }
    bundle_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:24]
    bundle_dir = Path(registry_dir) / "candidates" / bundle_id
    models_dir = bundle_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    frozen_members = []
    for member in source_records:
        source = Path(member["source"]["path"])
        destination = models_dir / f"{member['name']}{source.suffix}"
        if not destination.exists():
            shutil.copy2(source, destination)
        copied = _file_record(destination)
        if copied["sha256"] != member["source"]["sha256"]:
            raise ModelRegistryError(f"Copied model checksum mismatch: {member['name']}")
        frozen_members.append(
            {
                "name": member["name"],
                "weight": member["weight"],
                "artifact": copied,
            }
        )

    frozen_evaluation = bundle_dir / "evaluation_packet.json"
    frozen_recommendation = bundle_dir / "promotion_recommendation.json"
    if not frozen_evaluation.exists():
        shutil.copy2(evaluation_packet_path, frozen_evaluation)
    if not frozen_recommendation.exists():
        shutil.copy2(recommendation_path, frozen_recommendation)
    data_snapshot = [_file_record(Path(path)) for path in data_files]
    manifest = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "name": name,
        "created_at": now.isoformat(),
        "approval_expires_at": (now + timedelta(hours=approval_ttl_hours)).isoformat(),
        "members": frozen_members,
        "evaluation_packet": _file_record(frozen_evaluation),
        "promotion_recommendation": _file_record(frozen_recommendation),
        "data_snapshot": data_snapshot,
        "configuration": configuration,
        "code_revision": code_revision,
        "status": "AWAITING_HUMAN_PROMOTION_APPROVAL",
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=str).encode()
    ).hexdigest()
    manifest["manifest_sha256"] = manifest_sha256
    manifest["approval_challenge"] = hashlib.sha256(
        f"PROMOTE:{bundle_id}:{manifest_sha256}".encode()
    ).hexdigest()[:12].upper()
    manifest_path = bundle_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest, manifest_path


def approve_candidate_bundle(
    manifest_path: str | Path,
    *,
    challenge: str,
    actor: str,
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now(timezone.utc)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if challenge.strip().upper() != manifest["approval_challenge"]:
        raise ModelRegistryError("Promotion approval challenge does not match.")
    if now > datetime.fromisoformat(manifest["approval_expires_at"]):
        raise ModelRegistryError("Promotion approval has expired.")
    approval = {
        "bundle_id": manifest["bundle_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "actor": actor,
        "approved_at": now.isoformat(),
        "expires_at": manifest["approval_expires_at"],
    }
    path = manifest_path.with_name("promotion_approval.json")
    _atomic_json(path, approval)
    return path


def promote_candidate_bundle(
    registry_dir: str | Path,
    manifest_path: str | Path,
    *,
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now(timezone.utc)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    approval_path = manifest_path.with_name("promotion_approval.json")
    if not approval_path.exists():
        raise ModelRegistryError("Promotion requires explicit human approval.")
    approval = json.loads(approval_path.read_text())
    if approval.get("bundle_id") != manifest["bundle_id"]:
        raise ModelRegistryError("Promotion approval bundle mismatch.")
    if approval.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ModelRegistryError("Promotion approval manifest mismatch.")
    if now > datetime.fromisoformat(approval["expires_at"]):
        raise ModelRegistryError("Promotion approval has expired.")
    for member in manifest["members"]:
        artifact = Path(member["artifact"]["path"])
        if not artifact.exists() or _sha256(artifact) != member["artifact"]["sha256"]:
            raise ModelRegistryError(f"Frozen model artifact changed: {member['name']}")
    current = {
        "bundle_id": manifest["bundle_id"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "promoted_at": now.isoformat(),
        "approved_by": approval["actor"],
    }
    current_path = Path(registry_dir) / "champion" / "current.json"
    _atomic_json(current_path, current)
    return current_path
