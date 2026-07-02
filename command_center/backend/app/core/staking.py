from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import settings
from .numerai_ops import build_napi
from .submission_guard import resolve_single_model


class StakingPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class StakePolicy:
    max_total_nmr: float
    max_per_model_nmr: float
    max_change_nmr: float


@dataclass(frozen=True)
class StakeSizingPolicy:
    minimum_resolved_rounds: int = 20
    recent_window: int = 20
    minimum_mean_corr: float = 0.0
    minimum_mean_mmc: float = 0.0
    minimum_positive_corr_rate: float = 0.55
    maximum_initial_allocation_fraction: float = 0.10


@dataclass(frozen=True)
class StakeProposal:
    proposal_id: str
    model_name: str
    model_id: str
    action: str
    amount_nmr: float
    current_model_stake_nmr: float
    current_total_stake_nmr: float
    projected_model_stake_nmr: float
    projected_total_stake_nmr: float
    rationale: str
    created_at: str
    expires_at: str
    approval_challenge: str
    evidence: dict[str, Any] | None = None
    status: str = "AWAITING_HUMAN_APPROVAL"


def _score_value(round_row: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    scores = round_row.get("submissionScores") or []
    by_name = {
        str(score.get("displayName", "")).casefold(): score.get("value")
        for score in scores
    }
    for name in names:
        value = by_name.get(name.casefold())
        if value is not None:
            return float(value)
    return None


def summarize_live_performance(
    performance_rows: list[Mapping[str, Any]],
    *,
    recent_window: int = 20,
    deployment_round: int | None = None,
) -> dict[str, Any]:
    resolved = [row for row in performance_rows if row.get("roundResolved")]
    if deployment_round is not None:
        resolved = [
            row
            for row in resolved
            if int(row.get("roundNumber", -1)) >= int(deployment_round)
        ]
    resolved.sort(key=lambda row: int(row.get("roundNumber", -1)))
    observations = []
    for row in resolved:
        corr = _score_value(row, ("canon_corr", "corr", "v2_corr20"))
        mmc = _score_value(row, ("canon_mmc", "mmc"))
        if corr is None:
            continue
        observations.append(
            {
                "round_number": int(row["roundNumber"]),
                "corr": corr,
                "mmc": mmc,
                "at_risk_nmr": float(row.get("atRisk") or 0.0),
            }
        )
    recent = observations[-recent_window:]
    corr_values = np.asarray([row["corr"] for row in recent], dtype=float)
    mmc_values = np.asarray(
        [row["mmc"] for row in recent if row["mmc"] is not None],
        dtype=float,
    )
    return {
        "resolved_rounds": len(observations),
        "deployment_round": deployment_round,
        "recent_rounds": len(recent),
        "latest_round": recent[-1]["round_number"] if recent else None,
        "mean_corr": float(corr_values.mean()) if len(corr_values) else None,
        "corr_std": float(corr_values.std(ddof=1)) if len(corr_values) > 1 else None,
        "positive_corr_rate": float((corr_values > 0).mean()) if len(corr_values) else None,
        "mean_mmc": float(mmc_values.mean()) if len(mmc_values) else None,
        "observations": recent,
    }


def recommend_stake_increase(
    performance_rows: list[Mapping[str, Any]],
    *,
    current_model_stake_nmr: float,
    current_total_stake_nmr: float,
    caps: StakePolicy,
    sizing: StakeSizingPolicy | None = None,
    deployment_round: int | None = None,
) -> dict[str, Any]:
    sizing = sizing or StakeSizingPolicy()
    evidence = summarize_live_performance(
        performance_rows,
        recent_window=sizing.recent_window,
        deployment_round=deployment_round,
    )
    failures = []
    if deployment_round is None:
        failures.append("deployment_round_required")
    if caps.max_total_nmr <= 0 or caps.max_per_model_nmr <= 0 or caps.max_change_nmr <= 0:
        failures.append("staking_caps_disabled")
    if evidence["resolved_rounds"] < sizing.minimum_resolved_rounds:
        failures.append("insufficient_resolved_rounds")
    if evidence["recent_rounds"] < min(sizing.recent_window, sizing.minimum_resolved_rounds):
        failures.append("insufficient_recent_rounds")
    if evidence["mean_corr"] is None or evidence["mean_corr"] < sizing.minimum_mean_corr:
        failures.append("mean_corr_below_floor")
    if evidence["mean_mmc"] is None or evidence["mean_mmc"] < sizing.minimum_mean_mmc:
        failures.append("mean_mmc_below_floor")
    if (
        evidence["positive_corr_rate"] is None
        or evidence["positive_corr_rate"] < sizing.minimum_positive_corr_rate
    ):
        failures.append("positive_corr_rate_below_floor")
    if failures:
        return {
            "eligible": False,
            "recommended_increase_nmr": 0.0,
            "failures": failures,
            "evidence": evidence,
            "sizing_policy": asdict(sizing),
        }
    initial_cap = caps.max_total_nmr * sizing.maximum_initial_allocation_fraction
    capacity = min(
        caps.max_change_nmr,
        caps.max_per_model_nmr - current_model_stake_nmr,
        caps.max_total_nmr - current_total_stake_nmr,
        initial_cap - current_model_stake_nmr,
    )
    recommended = max(0.0, float(capacity))
    return {
        "eligible": recommended > 0,
        "recommended_increase_nmr": recommended,
        "failures": [] if recommended > 0 else ["no_remaining_policy_capacity"],
        "evidence": evidence,
        "sizing_policy": asdict(sizing),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_stake_proposal(
    state_dir: str | Path,
    *,
    model_name: str,
    model_id: str,
    action: str,
    amount_nmr: float,
    current_model_stake_nmr: float,
    current_total_stake_nmr: float,
    rationale: str,
    policy: StakePolicy,
    evidence: dict[str, Any] | None = None,
    ttl_minutes: int = 60,
    now: datetime | None = None,
) -> tuple[StakeProposal, Path]:
    now = now or datetime.now(timezone.utc)
    action = action.lower()
    amount_nmr = float(amount_nmr)
    if action not in {"increase", "decrease"}:
        raise StakingPolicyError("Stake action must be increase or decrease.")
    if amount_nmr <= 0 or amount_nmr > policy.max_change_nmr:
        raise StakingPolicyError("Stake amount is outside the configured per-change limit.")
    sign = 1.0 if action == "increase" else -1.0
    projected_model = current_model_stake_nmr + sign * amount_nmr
    projected_total = current_total_stake_nmr + sign * amount_nmr
    if projected_model < 0 or projected_total < 0:
        raise StakingPolicyError("Stake decrease exceeds the current stake.")
    if action == "increase":
        if policy.max_total_nmr <= 0 or policy.max_per_model_nmr <= 0:
            raise StakingPolicyError("Stake increases are disabled by configured caps.")
        if projected_model > policy.max_per_model_nmr:
            raise StakingPolicyError(
                "Projected model stake exceeds the configured model cap."
            )
        if projected_total > policy.max_total_nmr:
            raise StakingPolicyError(
                "Projected total stake exceeds the configured portfolio cap."
            )
    identity = (
        f"{model_id}:{action}:{amount_nmr:.12f}:{current_model_stake_nmr:.12f}:"
        f"{current_total_stake_nmr:.12f}:{now.isoformat()}"
    )
    proposal_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
    challenge = hashlib.sha256(f"STAKE:{proposal_id}".encode()).hexdigest()[:12].upper()
    proposal = StakeProposal(
        proposal_id=proposal_id,
        model_name=model_name,
        model_id=model_id,
        action=action,
        amount_nmr=amount_nmr,
        current_model_stake_nmr=float(current_model_stake_nmr),
        current_total_stake_nmr=float(current_total_stake_nmr),
        projected_model_stake_nmr=projected_model,
        projected_total_stake_nmr=projected_total,
        rationale=rationale,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=ttl_minutes)).isoformat(),
        approval_challenge=challenge,
        evidence=evidence,
    )
    path = Path(state_dir) / "staking" / proposal_id / "proposal.json"
    _atomic_json(path, asdict(proposal))
    return proposal, path


def approve_stake_proposal(
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
        raise StakingPolicyError("Stake approval challenge does not match.")
    if now > datetime.fromisoformat(proposal["expires_at"]):
        raise StakingPolicyError("Stake proposal has expired.")
    proposal_sha256 = _sha256_file(proposal_path)
    approval = {
        "proposal_id": proposal["proposal_id"],
        "model_id": proposal["model_id"],
        "action": proposal["action"],
        "amount_nmr": proposal["amount_nmr"],
        "proposal_sha256": proposal_sha256,
        "actor": actor,
        "approved_at": now.isoformat(),
        "expires_at": proposal["expires_at"],
    }
    path = proposal_path.with_name("approval.json")
    _atomic_json(path, approval)
    return path


def execute_approved_stake_change(
    napi: Any,
    proposal_path: str | Path,
    *,
    confirmation: str,
    policy: StakePolicy | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    proposal_path = Path(proposal_path)
    proposal = json.loads(proposal_path.read_text())
    execution_path = proposal_path.with_name("execution.json")
    intent_path = proposal_path.with_name("execution_intent.json")
    if execution_path.exists():
        raise StakingPolicyError("Stake proposal has already been executed.")
    if intent_path.exists():
        raise StakingPolicyError(
            "Unresolved stake execution intent exists; reconcile it before retrying."
        )
    approval_path = proposal_path.with_name("approval.json")
    if not approval_path.exists():
        raise StakingPolicyError("Stake execution requires explicit human approval.")
    approval = json.loads(approval_path.read_text())
    for key in (
        "proposal_id",
        "model_id",
        "action",
        "amount_nmr",
    ):
        if approval.get(key) != proposal.get(key):
            raise StakingPolicyError(f"Stake approval mismatch: {key}.")
    if approval.get("proposal_sha256") != _sha256_file(proposal_path):
        raise StakingPolicyError("Stake proposal changed after approval.")
    if now > datetime.fromisoformat(approval["expires_at"]):
        raise StakingPolicyError("Stake approval has expired.")
    expected_confirmation = (
        f"{proposal['action'].upper()} {proposal['amount_nmr']} NMR "
        f"ON {proposal['model_name']}"
    )
    if confirmation != expected_confirmation:
        raise StakingPolicyError(
            f"Confirmation must exactly match: {expected_confirmation}"
        )

    model_name, model_id = resolve_single_model(
        napi.get_models(),
        proposal["model_name"],
    )
    if model_id != proposal["model_id"]:
        raise StakingPolicyError("Numerai model mapping changed after stake approval.")
    current_stakes = {
        name: float(napi.stake_get(name) or 0.0)
        for name in napi.get_models()
    }
    current_model_stake = current_stakes[model_name]
    current_total_stake = sum(current_stakes.values())
    tolerance = 1e-9
    if (
        abs(current_model_stake - float(proposal["current_model_stake_nmr"]))
        > tolerance
        or abs(current_total_stake - float(proposal["current_total_stake_nmr"]))
        > tolerance
    ):
        raise StakingPolicyError(
            "Live stake changed after proposal; create and approve a fresh proposal."
        )

    if policy is None:
        policy_data = (proposal.get("evidence") or {}).get("policy")
        if not policy_data:
            raise StakingPolicyError(
                "Stake execution requires the current configured policy."
            )
        policy = StakePolicy(
            max_total_nmr=float(policy_data["max_total_nmr"]),
            max_per_model_nmr=float(policy_data["max_per_model_nmr"]),
            max_change_nmr=float(policy_data["max_change_nmr"]),
        )
    amount_nmr = float(proposal["amount_nmr"])
    if policy.max_change_nmr <= 0 or amount_nmr > policy.max_change_nmr:
        raise StakingPolicyError("Stake amount exceeds the current per-change cap.")
    if proposal["action"] == "increase":
        projected_model = current_model_stake + amount_nmr
        projected_total = current_total_stake + amount_nmr
        if (
            policy.max_total_nmr <= 0
            or policy.max_per_model_nmr <= 0
            or projected_model > policy.max_per_model_nmr
            or projected_total > policy.max_total_nmr
        ):
            raise StakingPolicyError("Current stake caps no longer allow this increase.")
        available_nmr = float(napi.get_account().get("availableNmr") or 0.0)
        if available_nmr + tolerance < amount_nmr:
            raise StakingPolicyError("Available NMR is below the approved increase.")

    _atomic_json(
        intent_path,
        {
            "proposal_id": proposal["proposal_id"],
            "requested_at": now.isoformat(),
            "model_id": proposal["model_id"],
            "action": proposal["action"],
            "amount_nmr": amount_nmr,
        },
    )
    try:
        if proposal["action"] == "increase":
            result = napi.stake_increase(amount_nmr, proposal["model_id"])
        else:
            result = napi.stake_decrease(amount_nmr, proposal["model_id"])
    except Exception as exc:
        _atomic_json(
            proposal_path.with_name("execution_error.json"),
            {
                "proposal_id": proposal["proposal_id"],
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "requires_manual_reconciliation": True,
            },
        )
        raise
    _atomic_json(
        execution_path,
        {
            "proposal_id": proposal["proposal_id"],
            "requested_at": now.isoformat(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
        },
    )
    return result


class StakeControlPlane:
    """Live stake reconciliation plus separately governed stake mutations."""

    def __init__(
        self,
        state_dir: str | Path | None = None,
        registry_dir: str | Path | None = None,
        *,
        napi: Any | None = None,
        policy: StakePolicy | None = None,
    ):
        self.state_dir = Path(state_dir or settings.CONTROL_PLANE_DIR)
        self.registry_dir = Path(registry_dir or settings.MODEL_REGISTRY_DIR)
        self.napi = napi
        self.policy = policy or StakePolicy(
            max_total_nmr=float(settings.MAX_TOTAL_STAKE_NMR),
            max_per_model_nmr=float(settings.MAX_MODEL_STAKE_NMR),
            max_change_nmr=float(settings.MAX_STAKE_CHANGE_NMR),
        )

    def _api(self):
        if self.napi is None:
            self.napi = build_napi()
        return self.napi

    def _assignments(self) -> dict[str, dict[str, Any]]:
        current_path = self.registry_dir / "portfolio" / "current.json"
        if not current_path.exists():
            return {}
        current = json.loads(current_path.read_text())
        return {
            assignment["model_name"].casefold(): assignment
            for assignment in current.get("assignments", [])
        }

    def inspect(self) -> dict[str, Any]:
        napi = self._api()
        models = napi.get_models()
        account = napi.get_account()
        account_models = {
            str(row.get("id")): row for row in account.get("models", [])
        }
        assignments = self._assignments()
        slots = []
        violations = []
        total_stake_nmr = 0.0
        for model_name, model_id in models.items():
            stake_nmr = float(napi.stake_get(model_name) or 0.0)
            total_stake_nmr += stake_nmr
            assignment = assignments.get(model_name.casefold())
            deployment_tier = (
                assignment.get("deployment_tier") if assignment else None
            )
            stake_eligible = bool(
                assignment and assignment.get("stake_eligible", False)
            )
            slot_violations = []
            if stake_nmr > 1e-12 and assignment is None:
                slot_violations.append("UNASSIGNED_MODEL_HAS_STAKE")
            if stake_nmr > 1e-12 and deployment_tier == "shadow":
                slot_violations.append("SHADOW_MODEL_HAS_STAKE")
            if stake_nmr > 1e-12 and assignment and not stake_eligible:
                slot_violations.append("STAKE_INELIGIBLE_MODEL_HAS_STAKE")
            for code in slot_violations:
                violations.append(
                    {
                        "code": code,
                        "model_name": model_name,
                        "model_id": model_id,
                        "stake_nmr": stake_nmr,
                    }
                )
            stake_state = account_models.get(model_id, {}).get("v2Stake") or {}
            slots.append(
                {
                    "model_name": model_name,
                    "model_id": model_id,
                    "deployment_tier": deployment_tier,
                    "stake_eligible": stake_eligible,
                    "stake_nmr": stake_nmr,
                    "pending_stake_status": stake_state.get("status"),
                    "violations": slot_violations,
                }
            )
        caps = asdict(self.policy)
        caps_enabled = all(value > 0 for value in caps.values())
        if violations:
            status = "STAKE_POLICY_ACTION_REQUIRED"
        elif not caps_enabled:
            status = "STAKE_INCREASES_DISABLED"
        else:
            status = "STAKE_POLICY_OK"
        return {
            "ok": True,
            "status": status,
            "read_only": True,
            "available_nmr": float(account.get("availableNmr") or 0.0),
            "total_stake_nmr": total_stake_nmr,
            "caps": caps,
            "caps_enabled": caps_enabled,
            "slots": slots,
            "violations": violations,
        }

    def _verified_deployment(
        self,
        *,
        model_id: str,
        deployment_round: int,
        artifact_sha256: str,
    ) -> dict[str, Any]:
        ledger_path = self.state_dir / "submission_ledger.json"
        if not ledger_path.exists():
            raise StakingPolicyError("Verified submission ledger is missing.")
        ledger = json.loads(ledger_path.read_text()).get("submissions", {})
        record = ledger.get(f"{int(deployment_round)}:{model_id}")
        if not record or not record.get("verified"):
            raise StakingPolicyError(
                "Deployment round has no verified local submission record."
            )
        packet_path = self.state_dir / "runs" / record["run_id"] / "readiness.json"
        if not packet_path.exists():
            raise StakingPolicyError("Deployment readiness packet is missing.")
        packet = json.loads(packet_path.read_text())
        if packet.get("validation", {}).get("model_artifact_sha256") != artifact_sha256:
            raise StakingPolicyError(
                "Deployment round was not submitted with the active artifact."
            )
        return {
            "deployment_round": int(deployment_round),
            "submission_id": record.get("submission_id"),
            "run_id": record.get("run_id"),
            "artifact_sha256": artifact_sha256,
        }

    def propose(
        self,
        *,
        target_model: str,
        action: str,
        amount_nmr: float,
        rationale: str,
        deployment_round: int | None = None,
    ) -> dict[str, Any]:
        napi = self._api()
        model_name, model_id = resolve_single_model(
            napi.get_models(),
            target_model,
        )
        snapshot = self.inspect()
        slot = next(row for row in snapshot["slots"] if row["model_id"] == model_id)
        assignment = self._assignments().get(model_name.casefold())
        action = action.casefold()
        recommendation = None
        deployment = None
        if action == "increase":
            if (
                assignment is None
                or assignment.get("deployment_tier") != "production"
                or not assignment.get("stake_eligible", False)
            ):
                raise StakingPolicyError(
                    "Stake increases require an active stake-eligible production assignment."
                )
            if deployment_round is None:
                raise StakingPolicyError(
                    "Stake increase requires the active artifact deployment round."
                )
            deployment = self._verified_deployment(
                model_id=model_id,
                deployment_round=deployment_round,
                artifact_sha256=assignment["artifact"]["sha256"],
            )
            recommendation = recommend_stake_increase(
                napi.round_model_performances_v2(model_id),
                current_model_stake_nmr=slot["stake_nmr"],
                current_total_stake_nmr=snapshot["total_stake_nmr"],
                caps=self.policy,
                deployment_round=deployment_round,
            )
            if not recommendation["eligible"]:
                raise StakingPolicyError(
                    "Live evidence does not permit a stake increase: "
                    + ", ".join(recommendation["failures"])
                )
            if float(amount_nmr) > float(
                recommendation["recommended_increase_nmr"]
            ) + 1e-12:
                raise StakingPolicyError(
                    "Requested increase exceeds the evidence-based recommendation."
                )
            if snapshot["available_nmr"] + 1e-12 < float(amount_nmr):
                raise StakingPolicyError("Available NMR is below the requested increase.")
        elif action != "decrease":
            raise StakingPolicyError("Stake action must be increase or decrease.")

        evidence = {
            "policy": asdict(self.policy),
            "portfolio_assignment": assignment,
            "deployment": deployment,
            "recommendation": recommendation,
            "live_snapshot": {
                "available_nmr": snapshot["available_nmr"],
                "total_stake_nmr": snapshot["total_stake_nmr"],
                "slot": slot,
            },
        }
        proposal, proposal_path = create_stake_proposal(
            self.state_dir,
            model_name=model_name,
            model_id=model_id,
            action=action,
            amount_nmr=amount_nmr,
            current_model_stake_nmr=slot["stake_nmr"],
            current_total_stake_nmr=snapshot["total_stake_nmr"],
            rationale=rationale,
            policy=self.policy,
            evidence=evidence,
        )
        return {
            "ok": True,
            "status": "STAKE_AWAITING_HUMAN_APPROVAL",
            "proposal_id": proposal.proposal_id,
            "proposal_path": str(proposal_path),
            "approval_challenge": proposal.approval_challenge,
            "expires_at": proposal.expires_at,
            "required_confirmation": (
                f"{proposal.action.upper()} {proposal.amount_nmr} NMR "
                f"ON {proposal.model_name}"
            ),
            "projected_model_stake_nmr": proposal.projected_model_stake_nmr,
            "projected_total_stake_nmr": proposal.projected_total_stake_nmr,
        }

    def approve(
        self,
        *,
        proposal_path: str | Path,
        challenge: str,
        actor: str,
    ) -> dict[str, Any]:
        approval_path = approve_stake_proposal(
            proposal_path,
            challenge=challenge,
            actor=actor,
        )
        return {
            "ok": True,
            "status": "STAKE_APPROVED",
            "approval_path": str(approval_path),
        }

    def execute(
        self,
        *,
        proposal_path: str | Path,
        confirmation: str,
    ) -> dict[str, Any]:
        proposal = json.loads(Path(proposal_path).read_text())
        if proposal["action"] == "increase":
            assignment = self._assignments().get(
                str(proposal["model_name"]).casefold()
            )
            if (
                assignment is None
                or assignment.get("deployment_tier") != "production"
                or not assignment.get("stake_eligible", False)
            ):
                raise StakingPolicyError(
                    "Active portfolio no longer permits this stake increase."
                )
        result = execute_approved_stake_change(
            self._api(),
            proposal_path,
            confirmation=confirmation,
            policy=self.policy,
        )
        return {
            "ok": True,
            "status": "STAKE_CHANGE_REQUESTED",
            "proposal_id": proposal["proposal_id"],
            "result": result,
        }
