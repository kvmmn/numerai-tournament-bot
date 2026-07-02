from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import BACKEND_ROOT, settings


ServiceInspector = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True)
class EvidenceSpec:
    label: str
    name: str
    directory: Path
    pattern: str
    max_age_hours: float
    json_report: bool = True


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def inspect_launchd_service(
    label: str,
    *,
    uid: int | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Return the top-level launchd state without changing the service."""
    domain = f"gui/{uid if uid is not None else os.getuid()}/{label}"
    result = runner(
        ["/bin/launchctl", "print", domain],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {
            "loaded": False,
            "state": None,
            "runs": None,
            "last_exit_code": None,
            "error": (result.stderr or result.stdout).strip()[:400],
        }

    def first(pattern: str) -> str | None:
        match = re.search(pattern, result.stdout, flags=re.MULTILINE)
        return match.group(1).strip() if match else None

    state = first(r"^\s*state = (.+)$")
    runs_text = first(r"^\s*runs = (\d+)$")
    exit_text = first(r"^\s*last exit code = (.+)$")
    exit_code = None
    if exit_text and exit_text.lstrip("-").isdigit():
        exit_code = int(exit_text)
    return {
        "loaded": True,
        "state": state,
        "runs": int(runs_text) if runs_text is not None else None,
        "last_exit_code": exit_code,
        "error": None,
    }


class SystemHealthMonitor:
    """Supervise native jobs and the freshness of their durable evidence."""

    def __init__(
        self,
        state_dir: str | Path | None = None,
        reports_dir: str | Path | None = None,
        backup_dir: str | Path | None = None,
        *,
        now: datetime | None = None,
        service_inspector: ServiceInspector | None = None,
        evidence_specs: Sequence[EvidenceSpec] | None = None,
    ):
        self.state_dir = Path(state_dir or settings.CONTROL_PLANE_DIR)
        self.reports_dir = Path(
            reports_dir or BACKEND_ROOT / "automation" / "reports"
        )
        runtime_root = BACKEND_ROOT.parents[1]
        self.backup_dir = Path(
            backup_dir or runtime_root / "backups" / "state"
        )
        self.now = now or datetime.now(timezone.utc)
        self.service_inspector = service_inspector or inspect_launchd_service
        self.evidence_specs = list(
            evidence_specs or self._default_specs()
        )

    def _default_specs(self) -> list[EvidenceSpec]:
        return [
            EvidenceSpec(
                "com.numerai.platform",
                "platform compatibility",
                self.state_dir / "platform",
                "latest.json",
                30,
            ),
            EvidenceSpec(
                "com.numerai.deadline",
                "deadline guard",
                self.reports_dir,
                "*_portfolio-status.json",
                30,
            ),
            EvidenceSpec(
                "com.numerai.daily",
                "portfolio readiness",
                self.reports_dir,
                "*_portfolio-prepare.json",
                30,
            ),
            EvidenceSpec(
                "com.numerai.competition",
                "competition status",
                self.state_dir / "competition",
                "latest.json",
                30,
            ),
            EvidenceSpec(
                "com.numerai.outcomes",
                "outcome listener",
                self.reports_dir,
                "*_score-listen.json",
                30,
            ),
            EvidenceSpec(
                "com.numerai.staking",
                "stake audit",
                self.reports_dir,
                "*_stake-status.json",
                30,
            ),
            EvidenceSpec(
                "com.numerai.alerts",
                "alert dispatcher",
                self.state_dir / "alerts",
                "ledger.json",
                30,
            ),
            EvidenceSpec(
                "com.numerai.backup",
                "state backup",
                self.backup_dir,
                "numerai-state-*.tar.gz",
                30,
                json_report=False,
            ),
            EvidenceSpec(
                "com.numerai.research",
                "research review",
                self.reports_dir,
                "*_research-evaluate.json",
                192,
            ),
        ]

    @staticmethod
    def _latest(spec: EvidenceSpec) -> Path | None:
        matches = [
            path for path in spec.directory.glob(spec.pattern) if path.is_file()
        ]
        return (
            max(
                matches,
                key=lambda path: (path.stat().st_mtime_ns, path.name),
            )
            if matches
            else None
        )

    @staticmethod
    def _load_report(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text())
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def snapshot(self) -> dict[str, Any]:
        generated_at = self.now.astimezone(timezone.utc)
        jobs = []
        issues = []
        now_timestamp = generated_at.timestamp()

        for spec in self.evidence_specs:
            job_issues = []
            try:
                service = dict(self.service_inspector(spec.label))
            except Exception as exc:
                service = {
                    "loaded": False,
                    "state": None,
                    "runs": None,
                    "last_exit_code": None,
                    "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                }
            if not service.get("loaded"):
                job_issues.append("SERVICE_NOT_LOADED")
            exit_code = service.get("last_exit_code")
            if exit_code is not None and int(exit_code) != 0:
                job_issues.append("LAST_EXIT_NONZERO")

            evidence_path = self._latest(spec)
            age_hours = None
            report_ok = None
            report_status = None
            if evidence_path is None:
                job_issues.append("EVIDENCE_MISSING")
            else:
                age_hours = max(
                    0.0,
                    (now_timestamp - evidence_path.stat().st_mtime) / 3600,
                )
                if age_hours > spec.max_age_hours:
                    job_issues.append("EVIDENCE_STALE")
                if spec.json_report:
                    payload = self._load_report(evidence_path)
                    if not payload:
                        job_issues.append("EVIDENCE_INVALID")
                    else:
                        report_ok = payload.get("ok")
                        report_status = payload.get("status")
                        if report_ok is False:
                            job_issues.append("LAST_REPORT_FAILED")

            job = {
                "label": spec.label,
                "name": spec.name,
                "healthy": not job_issues,
                "service": service,
                "evidence": {
                    "path": str(evidence_path) if evidence_path else None,
                    "age_hours": (
                        round(age_hours, 3)
                        if age_hours is not None
                        else None
                    ),
                    "max_age_hours": spec.max_age_hours,
                    "report_ok": report_ok,
                    "report_status": report_status,
                },
                "issues": job_issues,
            }
            jobs.append(job)
            issues.extend(
                {
                    "label": spec.label,
                    "name": spec.name,
                    "code": code,
                }
                for code in job_issues
            )

        status = (
            "SYSTEM_HEALTH_ACTION_REQUIRED"
            if issues
            else "SYSTEM_HEALTHY"
        )
        report: dict[str, Any] = {
            "ok": not issues,
            "status": status,
            "read_only": True,
            "generated_at": generated_at.isoformat(),
            "healthy_job_count": sum(job["healthy"] for job in jobs),
            "expected_job_count": len(jobs),
            "jobs": jobs,
            "issues": issues,
            "summary": (
                f"{sum(job['healthy'] for job in jobs)}/{len(jobs)} "
                f"native jobs have loaded services and fresh evidence."
            ),
        }
        reports_dir = self.state_dir / "system_health"
        history_path = (
            reports_dir / f"{generated_at.strftime('%Y%m%d_%H%M%S')}.json"
        )
        latest_path = reports_dir / "latest.json"
        _atomic_json(history_path, report)
        _atomic_json(latest_path, report)
        report["report_path"] = str(history_path)
        report["latest_path"] = str(latest_path)
        return report
