from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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
    if projected_model > policy.max_per_model_nmr:
        raise StakingPolicyError("Projected model stake exceeds the configured model cap.")
    if projected_total > policy.max_total_nmr:
        raise StakingPolicyError("Projected total stake exceeds the configured portfolio cap.")
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
    approval = {
        "proposal_id": proposal["proposal_id"],
        "model_id": proposal["model_id"],
        "action": proposal["action"],
        "amount_nmr": proposal["amount_nmr"],
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
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    proposal_path = Path(proposal_path)
    proposal = json.loads(proposal_path.read_text())
    approval_path = proposal_path.with_name("approval.json")
    if not approval_path.exists():
        raise StakingPolicyError("Stake execution requires explicit human approval.")
    approval = json.loads(approval_path.read_text())
    for key in ("proposal_id", "model_id", "action", "amount_nmr"):
        if approval.get(key) != proposal.get(key):
            raise StakingPolicyError(f"Stake approval mismatch: {key}.")
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
    if proposal["action"] == "increase":
        result = napi.stake_increase(proposal["amount_nmr"], proposal["model_id"])
    else:
        result = napi.stake_decrease(proposal["amount_nmr"], proposal["model_id"])
    _atomic_json(
        proposal_path.with_name("execution.json"),
        {
            "proposal_id": proposal["proposal_id"],
            "executed_at": now.isoformat(),
            "result": result,
        },
    )
    return result
