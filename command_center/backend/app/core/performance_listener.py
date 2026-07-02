from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .numerai_ops import build_napi
from .staking import _score_value, summarize_live_performance


class PerformanceListenerError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


class PerformanceListener:
    def __init__(self, state_dir: str | Path, napi: Any | None = None):
        self.state_dir = Path(state_dir)
        self.napi = napi
        self.cursor_path = self.state_dir / "performance_cursor.json"

    def _api(self):
        if self.napi is None:
            self.napi = build_napi()
        return self.napi

    def _cursor(self) -> dict[str, Any]:
        if not self.cursor_path.exists():
            return {"models": {}}
        return json.loads(self.cursor_path.read_text())

    def poll(self) -> dict[str, Any]:
        napi = self._api()
        models = napi.get_models()
        cursor = self._cursor()
        first_poll = not self.cursor_path.exists()
        events = []
        snapshots = {}
        postmortem_reasons = []

        for model_name, model_id in models.items():
            rows = napi.round_model_performances_v2(model_id)
            resolved = [row for row in rows if row.get("roundResolved")]
            resolved.sort(key=lambda row: int(row.get("roundNumber", -1)))
            latest_round = int(resolved[-1]["roundNumber"]) if resolved else None
            previous = cursor.get("models", {}).get(model_id, {}).get("latest_round")
            new_rows = (
                []
                if first_poll
                else [
                    row
                    for row in resolved
                    if previous is None or int(row["roundNumber"]) > int(previous)
                ]
            )
            for row in new_rows:
                corr = _score_value(row, ("canon_corr", "corr", "v2_corr20"))
                mmc = _score_value(row, ("canon_mmc", "mmc"))
                event = {
                    "model_name": model_name,
                    "model_id": model_id,
                    "round_number": int(row["roundNumber"]),
                    "corr": corr,
                    "mmc": mmc,
                    "at_risk_nmr": float(row.get("atRisk") or 0.0),
                }
                events.append(event)
                if corr is not None and corr < -0.02:
                    postmortem_reasons.append(
                        f"{model_name} round {row['roundNumber']} corr={corr:.5f}"
                    )
            snapshot = summarize_live_performance(resolved, recent_window=20)
            snapshots[model_name] = {
                "model_id": model_id,
                "latest_round": latest_round,
                "performance": snapshot,
            }
            observations = snapshot["observations"]
            if new_rows and len(observations) >= 5:
                rolling_corr = sum(row["corr"] for row in observations[-5:]) / 5
                if rolling_corr < 0:
                    postmortem_reasons.append(
                        f"{model_name} latest-5 mean corr={rolling_corr:.5f}"
                    )

        generated_at = datetime.now(timezone.utc)
        postmortem_path = None
        postmortem_created = False
        if postmortem_reasons:
            incident_identity = {
                "reasons": sorted(set(postmortem_reasons)),
                "latest_rounds": {
                    name: snapshot["latest_round"]
                    for name, snapshot in snapshots.items()
                },
            }
            incident_id = hashlib.sha256(
                json.dumps(incident_identity, sort_keys=True).encode()
            ).hexdigest()[:20]
            postmortem_path = (
                self.state_dir / "postmortems" / incident_id / "incident.json"
            )
            if not postmortem_path.exists():
                _atomic_json(
                    postmortem_path,
                    {
                        "incident_id": incident_id,
                        "status": "OPEN",
                        "created_at": generated_at.isoformat(),
                        "reasons": incident_identity["reasons"],
                        "events": events,
                        "model_snapshots": snapshots,
                        "required_actions": [
                            "review_live_scores",
                            "pause_stake_increases",
                            "open_bounded_research_experiment",
                        ],
                    },
                )
                postmortem_created = True
        report = {
            "ok": True,
            "status": (
                "POSTMORTEM_REQUIRED"
                if postmortem_reasons
                else "NEW_OUTCOMES"
                if events
                else "INITIALIZED"
                if first_poll
                else "NO_NEW_OUTCOMES"
            ),
            "generated_at": generated_at.isoformat(),
            "events": events,
            "models": snapshots,
            "postmortem_reasons": sorted(set(postmortem_reasons)),
            "postmortem_path": (
                str(postmortem_path) if postmortem_path else None
            ),
            "postmortem_created": postmortem_created,
            "read_only": True,
        }
        reports_dir = self.state_dir / "outcomes"
        report_path = reports_dir / f"{generated_at.strftime('%Y%m%d_%H%M%S')}.json"
        _atomic_json(report_path, report)
        _atomic_json(
            self.cursor_path,
            {
                "updated_at": generated_at.isoformat(),
                "models": {
                    snapshot["model_id"]: {
                        "model_name": name,
                        "latest_round": snapshot["latest_round"],
                    }
                    for name, snapshot in snapshots.items()
                },
            },
        )
        report["report_path"] = str(report_path)
        return report
