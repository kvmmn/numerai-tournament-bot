from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class AlertDeliveryError(RuntimeError):
    pass


Notifier = Callable[[str, str], None]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def _escape_applescript(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def macos_notification(title: str, message: str) -> None:
    script = (
        f'display notification "{_escape_applescript(message)}" '
        f'with title "{_escape_applescript(title)}"'
    )
    result = subprocess.run(
        ["/usr/bin/osascript", "-e", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AlertDeliveryError(
            result.stderr.strip() or "macOS notification command failed."
        )


class AlertDispatcher:
    """Deduplicated native notifications from durable control-plane reports."""

    def __init__(
        self,
        state_dir: str | Path,
        reports_dir: str | Path | None = None,
        *,
        notifier: Notifier | None = None,
    ):
        self.state_dir = Path(state_dir)
        self.reports_dir = Path(reports_dir or self.state_dir.parent / "reports")
        self.notifier = notifier or macos_notification
        self.ledger_path = self.state_dir / "alerts" / "ledger.json"

    def _ledger(self) -> dict[str, Any]:
        if not self.ledger_path.exists():
            return {"delivered": {}, "failures": []}
        return json.loads(self.ledger_path.read_text())

    @staticmethod
    def _latest(directory: Path, pattern: str) -> Path | None:
        matches = list(directory.glob(pattern))
        return max(matches, key=lambda path: path.stat().st_mtime_ns) if matches else None

    @staticmethod
    def _load(path: Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _candidate(
        *,
        category: str,
        title: str,
        message: str,
        source: Path,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        identity = {
            "category": category,
            "source": str(source),
            "status": payload.get("status"),
            "generated_at": payload.get("generated_at"),
            "round": payload.get("round_number") or payload.get("current_round"),
            "message": message,
        }
        alert_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode()
        ).hexdigest()[:20]
        return {
            "alert_id": alert_id,
            "category": category,
            "title": title,
            "message": message[:220],
            "source": str(source),
            "source_status": payload.get("status"),
        }

    def collect(self) -> list[dict[str, Any]]:
        candidates = []
        platform_path = self.state_dir / "platform" / "latest.json"
        platform = self._load(platform_path)
        if platform and platform.get("status") in {
            "PLATFORM_CONTRACT_BROKEN",
            "PLATFORM_MIGRATION_REVIEW",
        }:
            issue_codes = [
                *platform.get("failures", []),
                *platform.get("warnings", []),
            ]
            candidates.append(
                self._candidate(
                    category="platform",
                    title="Numerai platform compatibility",
                    message=(
                        ", ".join(str(code) for code in issue_codes)
                        or str(platform.get("status"))
                    ),
                    source=platform_path,
                    payload=platform,
                )
            )

        competition_path = self.state_dir / "competition" / "latest.json"
        competition = self._load(competition_path)
        if competition and competition.get("status") in {
            "COMPETITION_ACTION_REQUIRED",
            "COMPETITION_ROUND_INCOMPLETE",
        }:
            coverage = competition.get("current_round_coverage", {})
            missing = ", ".join(coverage.get("missing_models", [])) or "unknown slots"
            candidates.append(
                self._candidate(
                    category="competition",
                    title=f"Numerai round {competition.get('current_round')} needs action",
                    message=f"Missing verified submissions: {missing}.",
                    source=competition_path,
                    payload=competition,
                )
            )

        portfolio_path = self._latest(
            self.reports_dir,
            "*_portfolio-prepare.json",
        )
        portfolio = self._load(portfolio_path)
        if portfolio_path and portfolio and portfolio.get("status") in {
            "PORTFOLIO_AWAITING_HUMAN_APPROVAL",
            "PORTFOLIO_PREPARATION_FAILED",
            "PORTFOLIO_PARTIAL_COVERAGE",
        }:
            waiting = [
                str(row.get("target_model"))
                for row in portfolio.get("results", [])
                if row.get("status") == "AWAITING_HUMAN_APPROVAL"
            ]
            detail = (
                f"Approval required for {', '.join(waiting)}."
                if waiting
                else f"Portfolio status: {portfolio.get('status')}."
            )
            candidates.append(
                self._candidate(
                    category="readiness",
                    title="Numerai readiness requires review",
                    message=detail,
                    source=portfolio_path,
                    payload=portfolio,
                )
            )

        deadline_path = self._latest(
            self.reports_dir,
            "*_portfolio-status.json",
        )
        deadline = self._load(deadline_path)
        if deadline_path and deadline and deadline.get("status") in {
            "DEADLINE_GUARD_ACTION_REQUIRED",
            "DEADLINE_GUARD_PARTIAL_COVERAGE",
        }:
            slots = [
                f"{row.get('model_name')}={row.get('status')}"
                for row in deadline.get("slots", [])
                if row.get("status") != "SUBMITTED_VERIFIED"
            ]
            candidates.append(
                self._candidate(
                    category="deadline",
                    title="Numerai deadline guard",
                    message=", ".join(slots) or str(deadline.get("status")),
                    source=deadline_path,
                    payload=deadline,
                )
            )

        stake_path = self._latest(self.reports_dir, "*_stake-status.json")
        stake = self._load(stake_path)
        if stake_path and stake and stake.get("status") == "STAKE_POLICY_ACTION_REQUIRED":
            violations = [
                f"{row.get('model_name')}:{row.get('code')}"
                for row in stake.get("violations", [])
            ]
            candidates.append(
                self._candidate(
                    category="stake",
                    title="Numerai stake policy action",
                    message=", ".join(violations) or "Review live stake policy.",
                    source=stake_path,
                    payload=stake,
                )
            )

        outcome_path = self._latest(self.reports_dir, "*_score-listen.json")
        outcome = self._load(outcome_path)
        if outcome_path and outcome and outcome.get("status") == "POSTMORTEM_REQUIRED":
            reasons = outcome.get("postmortem_reasons", [])
            candidates.append(
                self._candidate(
                    category="postmortem",
                    title="Numerai performance postmortem",
                    message="; ".join(str(reason) for reason in reasons)
                    or "Review adverse live performance.",
                    source=outcome_path,
                    payload=outcome,
                )
            )
        return candidates

    def dispatch(self) -> dict[str, Any]:
        ledger = self._ledger()
        delivered = ledger.setdefault("delivered", {})
        candidates = self.collect()
        pending = [
            candidate
            for candidate in candidates
            if candidate["alert_id"] not in delivered
        ]
        sent = []
        failures = []
        for candidate in pending:
            try:
                self.notifier(candidate["title"], candidate["message"])
            except Exception as exc:
                failures.append(
                    {
                        **candidate,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            delivered[candidate["alert_id"]] = {
                **candidate,
                "delivered_at": datetime.now(timezone.utc).isoformat(),
            }
            sent.append(candidate)
        if failures:
            ledger.setdefault("failures", []).extend(
                {
                    **failure,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }
                for failure in failures
            )
            ledger["failures"] = ledger["failures"][-100:]
        if len(delivered) > 500:
            newest = sorted(
                delivered.items(),
                key=lambda item: item[1]["delivered_at"],
                reverse=True,
            )[:500]
            ledger["delivered"] = dict(newest)
        ledger["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self.ledger_path, ledger)
        return {
            "ok": not failures,
            "status": (
                "ALERT_DELIVERY_FAILED"
                if failures
                else "ALERTS_DELIVERED"
                if sent
                else "NO_NEW_ALERTS"
            ),
            "candidate_count": len(candidates),
            "pending_count": len(pending),
            "delivered": sent,
            "failures": failures,
            "ledger_path": str(self.ledger_path),
        }
