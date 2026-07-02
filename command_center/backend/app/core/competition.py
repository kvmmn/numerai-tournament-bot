from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .agentic_control_plane import AgenticControlPlane
from .config import settings
from .numerai_ops import build_napi


class CompetitionStatusError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _consecutive_streak(rounds: list[int]) -> int:
    ordered = sorted(set(int(value) for value in rounds), reverse=True)
    if not ordered:
        return 0
    streak = 1
    for previous, current in zip(ordered, ordered[1:]):
        if previous - current != 1:
            break
        streak += 1
    return streak


class CompetitionTracker:
    """Read-only tournament participation, season, rank, and coverage status."""

    def __init__(
        self,
        state_dir: str | Path | None = None,
        registry_dir: str | Path | None = None,
        *,
        napi: Any | None = None,
        now: datetime | None = None,
        account_rank_scan_limit: int | None = None,
        qualifying_round_target: int | None = None,
        qualifying_stake_nmr: float | None = None,
    ):
        self.state_dir = Path(state_dir or settings.CONTROL_PLANE_DIR)
        self.registry_dir = Path(registry_dir or settings.MODEL_REGISTRY_DIR)
        self.napi = napi
        self.now = now or datetime.now(timezone.utc)
        self.account_rank_scan_limit = int(
            account_rank_scan_limit
            if account_rank_scan_limit is not None
            else settings.ACCOUNT_RANK_SCAN_LIMIT
        )
        self.qualifying_round_target = int(
            qualifying_round_target
            if qualifying_round_target is not None
            else settings.SEASON_QUALIFYING_ROUNDS
        )
        self.qualifying_stake_nmr = float(
            qualifying_stake_nmr
            if qualifying_stake_nmr is not None
            else settings.SEASON_MIN_AT_RISK_NMR
        )

    def _api(self):
        if self.napi is None:
            self.napi = build_napi()
        return self.napi

    def _portfolio(self) -> dict[str, dict[str, Any]]:
        path = self.registry_dir / "portfolio" / "current.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text())
        return {
            row["model_name"].casefold(): row
            for row in payload.get("assignments", [])
        }

    def _ledger(self) -> dict[str, dict[str, Any]]:
        path = self.state_dir / "submission_ledger.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text()).get("submissions", {})

    def _account_rank(
        self,
        napi: Any,
        username: str,
    ) -> dict[str, Any]:
        page_size = 500
        scanned = 0
        while scanned < self.account_rank_scan_limit:
            limit = min(page_size, self.account_rank_scan_limit - scanned)
            rows = napi.get_account_leaderboard(limit=limit, offset=scanned)
            for row in rows:
                if str(row.get("username", "")).casefold() == username.casefold():
                    return {
                        "found": True,
                        "rank": row.get("rank"),
                        "scanned": scanned + len(rows),
                        "entry": {
                            key: (
                                _as_float(value)
                                if isinstance(value, Decimal)
                                else value
                            )
                            for key, value in row.items()
                        },
                    }
            scanned += len(rows)
            if len(rows) < limit:
                break
        return {
            "found": False,
            "rank": None,
            "scanned": scanned,
            "rank_lower_bound": scanned + 1,
            "entry": None,
        }

    def _latest_model_rank(self, napi: Any, model_name: str) -> dict[str, Any]:
        rows = napi.daily_model_performances(model_name)
        dated = [
            (date, row)
            for row in rows
            if (date := _as_datetime(row.get("date"))) is not None
        ]
        if not dated:
            return {
                "available": False,
                "as_of": None,
                "stale_days": None,
            }
        date, row = max(dated, key=lambda item: item[0])
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        stale_days = max(0, (self.now - date).days)
        return {
            "available": True,
            "as_of": date.isoformat(),
            "stale_days": stale_days,
            "corr_rank": row.get("corrRank"),
            "corr_reputation": row.get("corrRep"),
            "mmc_rank": row.get("mmcRank"),
            "mmc_reputation": row.get("mmcRep"),
            "fnc_v3_rank": row.get("fncV3Rank"),
            "fnc_v3_reputation": row.get("fncV3Rep"),
            "tc_rank": row.get("tcRank"),
            "tc_reputation": row.get("tcRep"),
        }

    def snapshot(self) -> dict[str, Any]:
        napi = self._api()
        account = napi.get_account()
        username = str(account["username"])
        models = napi.get_models()
        current_round = int(napi.get_current_round())
        round_window = AgenticControlPlane._submission_window(
            napi,
            current_round,
        )
        portfolio = self._portfolio()
        ledger = self._ledger()
        season_year = self.now.year
        at_risk_by_round: dict[int, float] = {}
        participating_rounds: set[int] = set()
        slots = []
        current_round_submitted = 0
        verified_rounds_by_model: dict[str, list[int]] = {
            model_id: [] for model_id in models.values()
        }
        for record in ledger.values():
            model_id = str(record.get("model_id", ""))
            if record.get("verified") and model_id in verified_rounds_by_model:
                verified_rounds_by_model[model_id].append(
                    int(record["round_number"])
                )

        for model_name, model_id in models.items():
            assignment = portfolio.get(model_name.casefold())
            ledger_record = ledger.get(f"{current_round}:{model_id}")
            submitted_verified = bool(
                ledger_record and ledger_record.get("verified")
            )
            if submitted_verified:
                current_round_submitted += 1
            performance_rows = napi.round_model_performances_v2(model_id)
            model_season_rounds = []
            for row in performance_rows:
                opened = _as_datetime(row.get("roundOpenTime"))
                if opened is None or opened.year != season_year:
                    continue
                round_number = int(row["roundNumber"])
                model_season_rounds.append(round_number)
                participating_rounds.add(round_number)
                at_risk_by_round[round_number] = (
                    at_risk_by_round.get(round_number, 0.0)
                    + _as_float(row.get("atRisk"))
                )
            rank = self._latest_model_rank(napi, model_name)
            slots.append(
                {
                    "model_name": model_name,
                    "model_id": model_id,
                    "deployment_tier": (
                        assignment.get("deployment_tier")
                        if assignment
                        else None
                    ),
                    "stake_eligible": bool(
                        assignment and assignment.get("stake_eligible", False)
                    ),
                    "current_round_status": (
                        "SUBMITTED_VERIFIED"
                        if submitted_verified
                        else "MISSING_CURRENT_ROUND"
                    ),
                    "submission_id": (
                        ledger_record.get("submission_id")
                        if ledger_record
                        else None
                    ),
                    "local_verified_rounds": sorted(
                        verified_rounds_by_model[model_id]
                    ),
                    "local_submission_streak": _consecutive_streak(
                        verified_rounds_by_model[model_id]
                    ),
                    "season_participation_rounds": sorted(
                        set(model_season_rounds)
                    ),
                    "leaderboard": rank,
                }
            )

        qualified_rounds = sorted(
            round_number
            for round_number, total_at_risk in at_risk_by_round.items()
            if total_at_risk + 1e-12 >= self.qualifying_stake_nmr
        )
        missing_slots = [
            row["model_name"]
            for row in slots
            if row["current_round_status"] != "SUBMITTED_VERIFIED"
        ]
        accepting = bool(round_window.get("accepting_submissions"))
        if missing_slots and accepting:
            status = "COMPETITION_ACTION_REQUIRED"
        elif missing_slots:
            status = "COMPETITION_ROUND_INCOMPLETE"
        else:
            status = "COMPETITION_CURRENT_ROUND_COMPLETE"
        strategic_gaps = []
        if len(qualified_rounds) < self.qualifying_round_target:
            strategic_gaps.append("SEASON_NOT_QUALIFIED")
        if not any(row["leaderboard"]["available"] for row in slots):
            strategic_gaps.append("NO_CURRENT_MODEL_REPUTATION")
        elif all(
            (row["leaderboard"].get("stale_days") or 0) > 30
            for row in slots
            if row["leaderboard"]["available"]
        ):
            strategic_gaps.append("MODEL_RANK_DATA_STALE")
        account_rank = self._account_rank(napi, username)
        if not account_rank["found"]:
            strategic_gaps.append("ACCOUNT_NOT_IN_SCANNED_LEADERBOARD")

        generated_at = self.now.astimezone(timezone.utc)
        report = {
            "ok": True,
            "status": status,
            "read_only": True,
            "generated_at": generated_at.isoformat(),
            "season_year": season_year,
            "current_round": current_round,
            "round_window": round_window,
            "current_round_coverage": {
                "submitted_verified": current_round_submitted,
                "account_models": len(models),
                "complete": current_round_submitted == len(models),
                "missing_models": missing_slots,
            },
            "season_qualification": {
                "qualified_rounds": qualified_rounds,
                "qualified_round_count": len(qualified_rounds),
                "target_round_count": self.qualifying_round_target,
                "rounds_remaining": max(
                    0,
                    self.qualifying_round_target - len(qualified_rounds),
                ),
                "minimum_total_at_risk_nmr": self.qualifying_stake_nmr,
                "participating_rounds": sorted(participating_rounds),
                "at_risk_by_round": {
                    str(round_number): total
                    for round_number, total in sorted(at_risk_by_round.items())
                },
            },
            "account_leaderboard": account_rank,
            "models": slots,
            "strategic_gaps": sorted(set(strategic_gaps)),
            "methodology": {
                "current_round_submission_source": "verified_local_ledger",
                "season_participation_source": "numerai_round_model_performances_v2",
                "qualification_rule": (
                    f"{self.qualifying_round_target} distinct on-time rounds "
                    f"with >= {self.qualifying_stake_nmr} NMR total at risk"
                ),
                "account_rank_scan_limit": self.account_rank_scan_limit,
            },
        }
        reports_dir = self.state_dir / "competition"
        history_path = (
            reports_dir / f"{generated_at.strftime('%Y%m%d_%H%M%S')}.json"
        )
        latest_path = reports_dir / "latest.json"
        _atomic_json(history_path, report)
        _atomic_json(latest_path, report)
        report["report_path"] = str(history_path)
        report["latest_path"] = str(latest_path)
        return report
