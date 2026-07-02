from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .agentic_control_plane import AgenticControlPlane
from .config import settings
from .submission_guard import resolve_single_model


class PortfolioError(RuntimeError):
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _artifact_record(path: str | Path) -> dict[str, Any]:
    artifact = Path(path).resolve()
    if not artifact.exists():
        raise PortfolioError(f"Portfolio artifact is missing: {artifact}")
    return {
        "path": str(artifact),
        "sha256": _sha256(artifact),
        "size_bytes": int(artifact.stat().st_size),
    }


def _normalize_assignments(assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not assignments:
        raise PortfolioError("A portfolio requires at least one model assignment.")
    normalized = []
    model_keys: set[str] = set()
    artifact_hashes: set[str] = set()
    for assignment in assignments:
        model_name = str(assignment["model_name"]).strip()
        model_key = model_name.casefold()
        if not model_name or model_key in model_keys:
            raise PortfolioError("Portfolio model assignments must be non-empty and unique.")
        artifact = _artifact_record(assignment["artifact_path"])
        if artifact["sha256"] in artifact_hashes:
            raise PortfolioError(
                "Portfolio assignments must use distinct artifacts; duplicate predictions "
                "belong in one slot, not several."
            )
        model_keys.add(model_key)
        artifact_hashes.add(artifact["sha256"])
        deployment_tier = str(assignment.get("deployment_tier") or "production")
        if deployment_tier not in {"production", "shadow"}:
            raise PortfolioError("Portfolio deployment tier must be production or shadow.")
        stake_eligible = bool(assignment.get("stake_eligible", False))
        if deployment_tier == "shadow" and stake_eligible:
            raise PortfolioError("Shadow assignments can never be stake eligible.")
        evidence = None
        evidence_path = assignment.get("evidence_path")
        if deployment_tier == "shadow":
            if not evidence_path:
                raise PortfolioError("Shadow assignments require frozen evidence.")
            evidence = _artifact_record(evidence_path)
            evidence_payload = json.loads(Path(evidence["path"]).read_text())
            if (
                evidence_payload.get("decision") != "DEPLOY_SHADOW"
                or evidence_payload.get("stake_eligible") is not False
            ):
                raise PortfolioError("Shadow evidence must authorize zero-stake deployment.")
        normalized.append(
            {
                "model_name": model_name,
                "role": str(assignment.get("role") or "unlabeled"),
                "deployment_tier": deployment_tier,
                "stake_eligible": stake_eligible,
                "artifact": artifact,
                "evidence": evidence,
                "source_bundle_id": assignment.get("source_bundle_id"),
            }
        )
    return normalized


def create_portfolio_proposal(
    registry_dir: str | Path,
    assignments: list[dict[str, Any]],
    *,
    approval_ttl_hours: int = 24,
    now: datetime | None = None,
) -> tuple[dict[str, Any], Path]:
    now = now or datetime.now(timezone.utc)
    normalized = _normalize_assignments(assignments)
    identity = json.dumps(normalized, sort_keys=True).encode()
    proposal_id = hashlib.sha256(identity).hexdigest()[:24]
    proposal = {
        "schema_version": 1,
        "proposal_id": proposal_id,
        "created_at": now.isoformat(),
        "approval_expires_at": (now + timedelta(hours=approval_ttl_hours)).isoformat(),
        "status": "AWAITING_HUMAN_PORTFOLIO_APPROVAL",
        "assignments": normalized,
    }
    proposal_sha256 = hashlib.sha256(
        json.dumps(proposal, sort_keys=True).encode()
    ).hexdigest()
    proposal["proposal_sha256"] = proposal_sha256
    proposal["approval_challenge"] = hashlib.sha256(
        f"PORTFOLIO:{proposal_id}:{proposal_sha256}".encode()
    ).hexdigest()[:12].upper()
    path = Path(registry_dir) / "portfolio" / "proposals" / f"{proposal_id}.json"
    _atomic_json(path, proposal)
    return proposal, path


def approve_portfolio_proposal(
    proposal_path: str | Path,
    *,
    challenge: str,
    actor: str,
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now(timezone.utc)
    proposal_path = Path(proposal_path)
    proposal = json.loads(proposal_path.read_text())
    if challenge.strip().upper() != proposal["approval_challenge"]:
        raise PortfolioError("Portfolio approval challenge does not match.")
    if now > datetime.fromisoformat(proposal["approval_expires_at"]):
        raise PortfolioError("Portfolio approval has expired.")
    approval = {
        "proposal_id": proposal["proposal_id"],
        "proposal_sha256": proposal["proposal_sha256"],
        "actor": actor,
        "approved_at": now.isoformat(),
        "expires_at": proposal["approval_expires_at"],
    }
    path = proposal_path.with_name(f"{proposal['proposal_id']}.approval.json")
    _atomic_json(path, approval)
    return path


def activate_portfolio(
    registry_dir: str | Path,
    proposal_path: str | Path,
    *,
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now(timezone.utc)
    proposal_path = Path(proposal_path)
    proposal = json.loads(proposal_path.read_text())
    approval_path = proposal_path.with_name(f"{proposal['proposal_id']}.approval.json")
    if not approval_path.exists():
        raise PortfolioError("Portfolio activation requires explicit human approval.")
    approval = json.loads(approval_path.read_text())
    if approval.get("proposal_sha256") != proposal.get("proposal_sha256"):
        raise PortfolioError("Portfolio approval does not match this proposal.")
    if now > datetime.fromisoformat(approval["expires_at"]):
        raise PortfolioError("Portfolio approval has expired.")
    assignments = _normalize_assignments(
        [
            {
                "model_name": row["model_name"],
                "role": row["role"],
                "deployment_tier": row.get("deployment_tier"),
                "stake_eligible": row.get("stake_eligible", False),
                "artifact_path": row["artifact"]["path"],
                "evidence_path": (
                    row["evidence"]["path"] if row.get("evidence") else None
                ),
                "source_bundle_id": row.get("source_bundle_id"),
            }
            for row in proposal["assignments"]
        ]
    )
    for expected, actual in zip(proposal["assignments"], assignments, strict=True):
        if expected["artifact"]["sha256"] != actual["artifact"]["sha256"]:
            raise PortfolioError("A portfolio artifact changed after approval.")
    current = {
        "schema_version": 1,
        "proposal_id": proposal["proposal_id"],
        "proposal_sha256": proposal["proposal_sha256"],
        "activated_at": now.isoformat(),
        "approved_by": approval["actor"],
        "assignments": assignments,
    }
    path = Path(registry_dir) / "portfolio" / "current.json"
    _atomic_json(path, current)
    return path


def bootstrap_portfolio_from_verified_submission(
    registry_dir: str | Path,
    state_dir: str | Path,
    *,
    run_id: str,
) -> Path:
    """Migrate an already approved and verified submission into one assignment."""
    state_dir = Path(state_dir)
    readiness_path = state_dir / "runs" / run_id / "readiness.json"
    approval_path = readiness_path.with_name("approval.json")
    ledger_path = state_dir / "submission_ledger.json"
    if not readiness_path.exists() or not approval_path.exists() or not ledger_path.exists():
        raise PortfolioError("Verified submission evidence is incomplete.")
    readiness = json.loads(readiness_path.read_text())
    ledger = json.loads(ledger_path.read_text()).get("submissions", {})
    key = f"{readiness['round_number']}:{readiness['target_model_id']}"
    record = ledger.get(key)
    if not record or record.get("run_id") != run_id or not record.get("verified"):
        raise PortfolioError("The source run is not a verified ledger submission.")
    artifact_path = readiness["validation"]["model_artifact"]
    artifact = _artifact_record(artifact_path)
    if artifact["sha256"] != readiness["validation"]["model_artifact_sha256"]:
        raise PortfolioError("The verified run artifact checksum no longer matches.")
    source_bundle_id = None
    champion_path = Path(registry_dir) / "champion" / "current.json"
    if champion_path.exists():
        champion = json.loads(champion_path.read_text())
        manifest = json.loads(Path(champion["manifest_path"]).read_text())
        if any(
            member["artifact"]["sha256"] == artifact["sha256"]
            for member in manifest.get("members", [])
        ):
            source_bundle_id = champion.get("bundle_id")
    current = {
        "schema_version": 1,
        "proposal_id": None,
        "proposal_sha256": None,
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": json.loads(approval_path.read_text())["actor"],
        "activation_basis": {
            "type": "verified_submission",
            "run_id": run_id,
            "submission_id": record["submission_id"],
        },
        "assignments": [
            {
                "model_name": readiness["target_model"],
                "role": "feature-family champion",
                "deployment_tier": "production",
                "stake_eligible": False,
                "artifact": artifact,
                "evidence": None,
                "source_bundle_id": source_bundle_id,
            }
        ],
    }
    path = Path(registry_dir) / "portfolio" / "current.json"
    _atomic_json(path, current)
    return path


def load_current_portfolio(registry_dir: str | Path) -> dict[str, Any]:
    path = Path(registry_dir) / "portfolio" / "current.json"
    if not path.exists():
        raise PortfolioError("No active model portfolio exists.")
    portfolio = json.loads(path.read_text())
    normalized = _normalize_assignments(
        [
            {
                "model_name": row["model_name"],
                "role": row["role"],
                "deployment_tier": row.get("deployment_tier"),
                "stake_eligible": row.get("stake_eligible", False),
                "artifact_path": row["artifact"]["path"],
                "evidence_path": (
                    row["evidence"]["path"] if row.get("evidence") else None
                ),
                "source_bundle_id": row.get("source_bundle_id"),
            }
            for row in portfolio["assignments"]
        ]
    )
    for expected, actual in zip(portfolio["assignments"], normalized, strict=True):
        if expected["artifact"]["sha256"] != actual["artifact"]["sha256"]:
            raise PortfolioError("An active portfolio artifact checksum changed.")
    portfolio["assignments"] = normalized
    return portfolio


class PortfolioControlPlane:
    def __init__(
        self,
        *,
        registry_dir: str | Path | None = None,
        state_dir: str | Path | None = None,
        napi: Any | None = None,
    ):
        self.registry_dir = Path(registry_dir or settings.MODEL_REGISTRY_DIR)
        self.state_dir = Path(state_dir or settings.CONTROL_PLANE_DIR)
        self.napi = napi

    def prepare_all(self) -> dict[str, Any]:
        portfolio = load_current_portfolio(self.registry_dir)
        control = AgenticControlPlane(self.state_dir, napi=self.napi)
        napi = control._api()
        round_number = int(napi.get_current_round())
        models = napi.get_models()
        results = []
        assigned_model_keys: set[str] = set()
        for assignment in portfolio["assignments"]:
            model_name, model_id = resolve_single_model(models, assignment["model_name"])
            assigned_model_keys.add(model_name.casefold())
            if control.ledger.contains(round_number, model_id):
                results.append(
                    {
                        "ok": True,
                        "status": "ALREADY_SUBMITTED",
                        "round_number": round_number,
                        "target_model": model_name,
                    }
                )
                continue
            try:
                result = control.prepare(
                    target_model=model_name,
                    artifact_path=assignment["artifact"]["path"],
                    deployment_tier=assignment["deployment_tier"],
                    shadow_evidence_path=(
                        assignment["evidence"]["path"]
                        if assignment.get("evidence")
                        else None
                    ),
                )
            except Exception as exc:
                result = {
                    "ok": False,
                    "status": "PREPARATION_FAILED",
                    "target_model": model_name,
                    "error": str(exc),
                }
            results.append(result)
        assigned_results = list(results)
        for model_name in models:
            if model_name.casefold() not in assigned_model_keys:
                results.append(
                    {
                        "ok": True,
                        "status": "UNASSIGNED",
                        "round_number": round_number,
                        "target_model": model_name,
                    }
                )
        ok = all(result["ok"] for result in assigned_results)
        assigned_statuses = {result["status"] for result in assigned_results}
        unassigned_count = len(models) - len(assigned_model_keys)
        if not ok:
            status = "PORTFOLIO_PARTIAL_FAILURE"
        elif unassigned_count:
            status = "PORTFOLIO_PARTIAL_COVERAGE"
        elif assigned_statuses == {"ALREADY_SUBMITTED"}:
            status = "PORTFOLIO_ALREADY_SUBMITTED"
        else:
            status = "PORTFOLIO_AWAITING_HUMAN_APPROVAL"
        return {
            "ok": ok,
            "status": status,
            "round_number": round_number,
            "assignment_count": len(portfolio["assignments"]),
            "account_model_count": len(models),
            "unassigned_count": unassigned_count,
            "coverage_complete": unassigned_count == 0,
            "results": results,
        }
