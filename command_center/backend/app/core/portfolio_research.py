from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShadowPolicy:
    minimum_mean_correlation: float = 0.009
    minimum_sharpe: float = 0.3
    maximum_feature_exposure: float = 0.1
    maximum_drawdown: float = 0.2
    minimum_recent_25_correlation: float = 0.0
    minimum_recent_50_correlation: float = -0.001
    minimum_recent_100_correlation: float = 0.0
    minimum_bootstrap_correlation: float = 0.0
    maximum_portfolio_correlation: float = 0.85


def shadow_recommendation(
    packet: dict[str, Any],
    *,
    maximum_observed_portfolio_correlation: float,
    policy: ShadowPolicy | None = None,
) -> dict[str, Any]:
    """Decide whether a zero-stake candidate merits forward shadow scoring."""
    policy = policy or ShadowPolicy()
    checks = {
        "mean_correlation": (
            packet["overall"]["mean_correlation"] >= policy.minimum_mean_correlation
        ),
        "sharpe": packet["overall"]["sharpe"] >= policy.minimum_sharpe,
        "feature_exposure": (
            packet["feature_exposure"] <= policy.maximum_feature_exposure
        ),
        "max_drawdown": (
            packet["overall"]["max_drawdown"] <= policy.maximum_drawdown
        ),
        "recent_25": (
            packet["recent"]["25"]["mean_correlation"]
            >= policy.minimum_recent_25_correlation
        ),
        "recent_50": (
            packet["recent"]["50"]["mean_correlation"]
            >= policy.minimum_recent_50_correlation
        ),
        "recent_100": (
            packet["recent"]["100"]["mean_correlation"]
            >= policy.minimum_recent_100_correlation
        ),
        "bootstrap_lower": (
            packet["bootstrap_mean_correlation"]["lower"]
            >= policy.minimum_bootstrap_correlation
        ),
        "portfolio_diversity": (
            maximum_observed_portfolio_correlation
            <= policy.maximum_portfolio_correlation
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "decision": "DEPLOY_SHADOW" if not failures else "REJECT",
        "deployment_tier": "shadow",
        "stake_eligible": False,
        "checks": checks,
        "failures": failures,
        "maximum_observed_portfolio_correlation": float(
            maximum_observed_portfolio_correlation
        ),
        "policy": asdict(policy),
        "candidate_packet_sha256": packet.get("packet_sha256"),
    }


def write_shadow_evidence(
    output_dir: str | Path,
    *,
    name: str,
    packet: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / f"{name}_evaluation_packet.json"
    recommendation_path = output_dir / f"{name}_shadow_recommendation.json"
    for path, payload in (
        (packet_path, packet),
        (recommendation_path, recommendation),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        )
        os.replace(temporary, path)
    return {
        "evaluation_packet": str(packet_path),
        "shadow_recommendation": str(recommendation_path),
        "evidence_sha256": hashlib.sha256(
            (
                packet_path.read_text()
                + recommendation_path.read_text()
            ).encode()
        ).hexdigest(),
    }
