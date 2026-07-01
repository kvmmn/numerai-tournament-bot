from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import settings
from .numerai_ops import (
    build_napi,
    build_submission_dataframe,
    model_path,
    sync_datasets,
)
from .submission_guard import (
    SubmissionGuardError,
    SubmissionLedger,
    approve_readiness_packet,
    create_readiness_packet,
    resolve_single_model,
    validate_approval,
    validate_submission_dataframe,
)
from .research import evaluate_artifact_robustness, promotion_recommendation


class AgenticControlPlane:
    """Durable, fail-closed coordinator for daily Numerai operations.

    Each specialist emits an auditable artifact. Only an explicit approval
    grant can activate the execution specialist.
    """

    AGENTS = (
        "platform_scout",
        "data_steward",
        "prediction_specialist",
        "risk_judge",
        "governance_guard",
        "submission_executor",
        "verification_listener",
    )

    def __init__(self, state_dir: str | Path | None = None, napi: Any | None = None):
        self.state_dir = Path(state_dir or settings.CONTROL_PLANE_DIR)
        self.napi = napi
        self.audit_path = self.state_dir / "audit.jsonl"
        self.ledger = SubmissionLedger(self.state_dir / "submission_ledger.json")

    def _api(self):
        if self.napi is None:
            self.napi = build_napi()
        return self.napi

    @staticmethod
    def _default_artifact() -> Path:
        current_path = Path(settings.MODEL_REGISTRY_DIR) / "champion" / "current.json"
        if not current_path.exists():
            return model_path()
        current = json.loads(current_path.read_text())
        manifest_path = Path(current["manifest_path"])
        manifest = json.loads(manifest_path.read_text())
        members = manifest.get("members", [])
        if len(members) != 1:
            raise SubmissionGuardError(
                "The current champion must expose one frozen callable ensemble artifact."
            )
        return Path(members[0]["artifact"]["path"])

    @staticmethod
    def _submission_window(napi: Any, round_number: int) -> dict[str, Any]:
        """Use closeTime, not NumerAPI.check_round_open's staking deadline."""
        if hasattr(napi, "raw_query"):
            query = """
                query($tournament: Int!, $round: Int!) {
                  rounds(tournament: $tournament, number: $round) {
                    number
                    openTime
                    closeTime
                    closeStakingTime
                  }
                }
            """
            tournament = int(getattr(napi, "tournament_id", 8))
            response = napi.raw_query(
                query,
                {"tournament": tournament, "round": int(round_number)},
            )
            rows = response.get("data", {}).get("rounds", [])
            if rows:
                row = rows[0]
                now = datetime.now(timezone.utc)
                open_at = datetime.fromisoformat(row["openTime"].replace("Z", "+00:00"))
                close_at = datetime.fromisoformat(row["closeTime"].replace("Z", "+00:00"))
                return {
                    **row,
                    "accepting_submissions": open_at < now < close_at,
                }
        is_open = bool(napi.check_round_open()) if hasattr(napi, "check_round_open") else True
        return {"number": int(round_number), "accepting_submissions": is_open}

    def _audit(self, agent: str, event: str, payload: dict[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        canonical = json.dumps(payload, sort_keys=True, default=str)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "event": event,
            "payload_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "payload": payload,
        }
        with self.audit_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def prepare(
        self,
        *,
        target_model: str | None = None,
        artifact_path: str | Path | None = None,
    ) -> dict[str, Any]:
        napi = self._api()
        round_number = int(napi.get_current_round())
        round_window = self._submission_window(napi, round_number)
        if not round_window["accepting_submissions"]:
            raise SubmissionGuardError(f"Numerai round {round_number} is not open.")
        models = napi.get_models()
        model_name, model_id = resolve_single_model(
            models,
            target_model or settings.SUBMISSION_TARGET_MODEL,
        )
        self._audit(
            "platform_scout",
            "round_and_target_resolved",
            {
                "round_number": round_number,
                "target_model": model_name,
                "target_model_id": model_id,
                "round_window": round_window,
            },
        )

        data_report = sync_datasets(napi)
        self._audit("data_steward", "data_synchronized", data_report)

        selected_artifact = (
            Path(artifact_path)
            if artifact_path
            else self._default_artifact()
        )
        if not selected_artifact.exists():
            raise SubmissionGuardError(f"Approved model artifact is missing: {selected_artifact}")
        frame = build_submission_dataframe(path=str(selected_artifact))
        artifact_sha256 = hashlib.sha256(selected_artifact.read_bytes()).hexdigest()
        self._audit(
            "prediction_specialist",
            "predictions_generated",
            {"artifact": str(selected_artifact), "artifact_sha256": artifact_sha256, "rows": len(frame)},
        )

        validation = validate_submission_dataframe(
            frame,
            min_unique_predictions=settings.SUBMISSION_MIN_UNIQUE_PREDICTIONS,
        )
        validation["model_artifact"] = str(selected_artifact.resolve())
        validation["model_artifact_sha256"] = artifact_sha256
        robustness = evaluate_artifact_robustness(
            selected_artifact,
            model_name=selected_artifact.stem,
        )
        recommendation = promotion_recommendation(robustness, champion=None)
        validation["robustness"] = robustness
        validation["promotion_recommendation"] = recommendation
        if recommendation["decision"] != "PROMOTE":
            raise SubmissionGuardError(
                "Model robustness failed readiness policy: "
                + ", ".join(recommendation["failures"])
            )
        self._audit("risk_judge", "submission_validated", validation)

        submission_dir = self.state_dir / "rounds" / str(round_number) / model_name
        submission_dir.mkdir(parents=True, exist_ok=True)
        submission_path = submission_dir / "submission.csv"
        frame.to_csv(submission_path, index=False)
        packet, packet_path = create_readiness_packet(
            self.state_dir,
            round_number=round_number,
            target_model=model_name,
            target_model_id=model_id,
            submission_path=submission_path,
            validation=validation,
            approval_ttl_minutes=settings.APPROVAL_TTL_MINUTES,
        )
        self._audit(
            "governance_guard",
            "readiness_packet_created",
            {
                "run_id": packet.run_id,
                "packet_path": str(packet_path),
                "approval_expires_at": packet.approval_expires_at,
            },
        )
        return {
            "ok": True,
            "status": "AWAITING_HUMAN_APPROVAL",
            "run_id": packet.run_id,
            "round_number": round_number,
            "target_model": model_name,
            "target_model_id": model_id,
            "packet_path": str(packet_path),
            "approval_challenge": packet.approval_challenge,
            "approval_expires_at": packet.approval_expires_at,
            "validation": validation,
            "data_report": data_report,
            "round_window": round_window,
            "delegated_agents": list(self.AGENTS[:5]),
        }

    def approve(self, *, run_id: str, challenge: str, actor: str) -> dict[str, Any]:
        packet_path = self.state_dir / "runs" / run_id / "readiness.json"
        if not packet_path.exists():
            raise SubmissionGuardError(f"Unknown readiness run: {run_id}")
        approval_path = approve_readiness_packet(
            packet_path,
            challenge=challenge,
            actor=actor,
        )
        self._audit(
            "governance_guard",
            "human_approval_recorded",
            {"run_id": run_id, "actor": actor, "approval_path": str(approval_path)},
        )
        return {
            "ok": True,
            "status": "APPROVED",
            "run_id": run_id,
            "approval_path": str(approval_path),
        }

    def submit(self, *, run_id: str) -> dict[str, Any]:
        packet_path = self.state_dir / "runs" / run_id / "readiness.json"
        packet = validate_approval(packet_path)
        napi = self._api()
        current_round = int(napi.get_current_round())
        if current_round != int(packet["round_number"]):
            raise SubmissionGuardError(
                f"Approval is for round {packet['round_number']}, current round is {current_round}."
            )
        round_window = self._submission_window(napi, current_round)
        if not round_window["accepting_submissions"]:
            raise SubmissionGuardError(f"Numerai round {current_round} is closed.")
        if self.ledger.contains(current_round, packet["target_model_id"]):
            raise SubmissionGuardError("This round/model already has a local submission record.")

        models = napi.get_models()
        resolved_name, resolved_id = resolve_single_model(models, packet["target_model"])
        if resolved_id != packet["target_model_id"]:
            raise SubmissionGuardError("Numerai target model mapping changed after approval.")

        frame = pd.read_csv(packet["submission_path"])
        validate_submission_dataframe(
            frame,
            min_unique_predictions=settings.SUBMISSION_MIN_UNIQUE_PREDICTIONS,
        )
        self._audit(
            "submission_executor",
            "upload_started",
            {"run_id": run_id, "round_number": current_round, "target_model": resolved_name},
        )
        submission_id = str(
            napi.upload_predictions(df=frame, model_id=resolved_id, timeout=600)
        )
        history = napi.submission_ids(resolved_id)
        verified = any(str(item.get("id")) == submission_id for item in history)
        self.ledger.record(
            round_number=current_round,
            model_id=resolved_id,
            submission_id=submission_id,
            run_id=run_id,
            verified=verified,
        )
        status = "SUBMITTED_VERIFIED" if verified else "SUBMITTED_UNVERIFIED"
        self._audit(
            "verification_listener",
            status.lower(),
            {"run_id": run_id, "submission_id": submission_id, "verified": verified},
        )
        return {
            "ok": verified,
            "status": status,
            "run_id": run_id,
            "round_number": current_round,
            "target_model": resolved_name,
            "submission_id": submission_id,
            "verified": verified,
            "delegated_agents": list(self.AGENTS[5:]),
        }
