"""Choose the single research step for one day.

The result is a plan. It does not train, submit, promote, or stake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

FORBIDDEN = (
    "full-auto",
    "mcp-submit",
    "numerapi-submit",
    "stake",
    "agent-submit",
)
EXPERIMENT = "run_seed_neutralization_sweep"


def choose_daily_step(slots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank slots by era MMC and name one research-only comparison."""
    if not slots:
        return {
            "ok": False,
            "status": "DAILY_STEP_NEEDS_SCORES",
            "step": None,
            "experiment": None,
            "bar_slot": None,
            "weak_slot": None,
            "regime": None,
            "forbidden": list(FORBIDDEN),
            "summary": "No slot scores were provided.",
        }

    ranked = sorted(slots, key=_era_mmc, reverse=True)
    bar = ranked[0]
    weak = ranked[-1]
    recent = [_recent_mmc(slot) for slot in slots]
    regime = "mmc_negative" if all(value < 0 for value in recent) else "mixed"
    experiment = EXPERIMENT
    summary = (
        f"Compare {experiment} against {bar['name']}. "
        f"{weak['name']} has the weaker era MMC. "
        "Judge the result on walk-forward MMC and do not submit or stake."
    )
    return {
        "ok": True,
        "status": "DAILY_STEP_READY",
        "step": "compare_challenger",
        "experiment": experiment,
        "bar_slot": bar["name"],
        "weak_slot": weak["name"],
        "regime": regime,
        "forbidden": list(FORBIDDEN),
        "summary": summary,
    }


def experiment_preflight(
    experiment: str | None,
    data_dir: Path,
    *,
    credentials_present: bool,
) -> dict[str, Any]:
    """Say whether the named sweep can run. Never trains."""
    required = ("train.parquet", "validation.parquet")
    missing = [name for name in required if not (Path(data_dir) / name).is_file()]
    if not credentials_present:
        missing.append("numerai_credentials")
    blocked = bool(missing) or not experiment
    return {
        "ok": not blocked,
        "status": "RESEARCH_BLOCKED" if blocked else "RESEARCH_READY",
        "experiment": experiment,
        "trained": False,
        "blocker": missing if experiment else ["experiment"],
        "summary": (
            "Named experiment was not trained. Missing " + ", ".join(missing) + "."
            if missing
            else "Training files are present. This preflight does not train."
        ),
    }


def _era_mmc(slot: Mapping[str, Any]) -> float:
    value = slot.get("era_mean_mmc")
    if value is None:
        raise ValueError(f"{slot.get('name', 'slot')} is missing era_mean_mmc")
    return float(value)


def _recent_mmc(slot: Mapping[str, Any]) -> float:
    value = slot.get("recent_mean_mmc")
    if value is None:
        raise ValueError(f"{slot.get('name', 'slot')} is missing recent_mean_mmc")
    return float(value)
