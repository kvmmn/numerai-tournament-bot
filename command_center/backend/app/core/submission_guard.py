from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


class SubmissionGuardError(RuntimeError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def resolve_single_model(
    models: Mapping[str, str],
    configured_name: str | None,
) -> tuple[str, str]:
    if not models:
        raise SubmissionGuardError("No Numerai models are available on the account.")
    if not configured_name:
        if len(models) == 1:
            return next(iter(models.items()))
        raise SubmissionGuardError(
            "Multiple Numerai models exist. Set SUBMISSION_TARGET_MODEL explicitly; "
            "identical predictions must not be broadcast to every model."
        )

    matches = [(name, model_id) for name, model_id in models.items() if name.casefold() == configured_name.casefold()]
    if len(matches) != 1:
        available = ", ".join(sorted(models))
        raise SubmissionGuardError(
            f"Configured target model {configured_name!r} does not uniquely match an "
            f"account model. Available: {available}"
        )
    return matches[0]


def validate_submission_dataframe(
    frame: pd.DataFrame,
    *,
    expected_ids: pd.Index | None = None,
    min_unique_predictions: int = 100,
) -> dict[str, Any]:
    if list(frame.columns) != ["id", "prediction"]:
        raise SubmissionGuardError("Submission columns must be exactly: id, prediction.")
    if frame.empty:
        raise SubmissionGuardError("Submission is empty.")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise SubmissionGuardError("Submission ids must be present and unique.")
    predictions = pd.to_numeric(frame["prediction"], errors="coerce")
    if predictions.isna().any() or not predictions.map(math.isfinite).all():
        raise SubmissionGuardError("Predictions must be finite numeric values.")
    if ((predictions < 0.0) | (predictions > 1.0)).any():
        raise SubmissionGuardError("Predictions must be within [0, 1].")
    unique_count = int(predictions.nunique())
    required_unique = min(min_unique_predictions, len(frame))
    if unique_count < required_unique:
        raise SubmissionGuardError(
            f"Predictions have only {unique_count} unique values; at least "
            f"{required_unique} are required."
        )
    if expected_ids is not None:
        actual = set(frame["id"].astype(str))
        expected = set(pd.Index(expected_ids).astype(str))
        if actual != expected:
            raise SubmissionGuardError(
                f"Submission/live id mismatch: missing={len(expected - actual)}, "
                f"unexpected={len(actual - expected)}."
            )
    return {
        "row_count": int(len(frame)),
        "unique_ids": int(frame["id"].nunique()),
        "unique_predictions": unique_count,
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "prediction_mean": float(predictions.mean()),
    }


def validate_raw_prediction_series(
    predictions: pd.Series,
    *,
    min_unique_predictions: int = 100,
) -> dict[str, Any]:
    numeric = pd.to_numeric(predictions, errors="coerce")
    unique_count = int(numeric.nunique())
    required_unique = min(min_unique_predictions, len(numeric))
    if numeric.isna().any() or not numeric.map(math.isfinite).all():
        raise SubmissionGuardError("Raw predictions must be finite numeric values.")
    if unique_count < required_unique:
        raise SubmissionGuardError(
            "Raw model predictions failed diversity validation before ranking: "
            f"unique={unique_count}, required={required_unique}."
        )
    return {
        "row_count": int(len(numeric)),
        "unique_predictions": unique_count,
        "prediction_std": float(numeric.std()),
    }


@dataclass(frozen=True)
class ReadinessPacket:
    run_id: str
    round_number: int
    target_model: str
    target_model_id: str
    submission_path: str
    submission_sha256: str
    created_at: str
    approval_expires_at: str
    validation: dict[str, Any]
    validation_sha256: str
    approval_challenge: str
    status: str = "AWAITING_APPROVAL"


def create_readiness_packet(
    state_dir: str | Path,
    *,
    round_number: int,
    target_model: str,
    target_model_id: str,
    submission_path: str | Path,
    validation: dict[str, Any],
    approval_ttl_minutes: int,
    now: datetime | None = None,
) -> tuple[ReadinessPacket, Path]:
    now = now or utc_now()
    submission_path = Path(submission_path).resolve()
    fingerprint = sha256_file(submission_path)
    validation_sha256 = hashlib.sha256(
        json.dumps(validation, sort_keys=True, default=str).encode()
    ).hexdigest()
    identity = f"{round_number}:{target_model_id}:{fingerprint}:{validation_sha256}"
    run_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
    challenge = hashlib.sha256(f"APPROVE:{run_id}:{fingerprint}".encode()).hexdigest()[:12].upper()
    packet = ReadinessPacket(
        run_id=run_id,
        round_number=int(round_number),
        target_model=target_model,
        target_model_id=target_model_id,
        submission_path=str(submission_path),
        submission_sha256=fingerprint,
        created_at=now.isoformat(),
        approval_expires_at=(now + timedelta(minutes=approval_ttl_minutes)).isoformat(),
        validation=validation,
        validation_sha256=validation_sha256,
        approval_challenge=challenge,
    )
    path = Path(state_dir) / "runs" / run_id / "readiness.json"
    _atomic_json(path, asdict(packet))
    return packet, path


def approve_readiness_packet(
    packet_path: str | Path,
    *,
    challenge: str,
    actor: str,
    now: datetime | None = None,
) -> Path:
    now = now or utc_now()
    packet_path = Path(packet_path)
    packet = json.loads(packet_path.read_text())
    if packet_path.with_name("revocation.json").exists():
        raise SubmissionGuardError("Readiness packet has been revoked.")
    if challenge.strip().upper() != packet["approval_challenge"]:
        raise SubmissionGuardError("Approval challenge does not match the readiness packet.")
    if now > datetime.fromisoformat(packet["approval_expires_at"]):
        raise SubmissionGuardError("Readiness packet approval window has expired.")
    grant = {
        "run_id": packet["run_id"],
        "round_number": packet["round_number"],
        "target_model_id": packet["target_model_id"],
        "submission_sha256": packet["submission_sha256"],
        "validation_sha256": packet["validation_sha256"],
        "actor": actor,
        "approved_at": now.isoformat(),
        "expires_at": packet["approval_expires_at"],
    }
    approval_path = packet_path.with_name("approval.json")
    _atomic_json(approval_path, grant)
    return approval_path


def validate_approval(
    packet_path: str | Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or utc_now()
    packet_path = Path(packet_path)
    packet = json.loads(packet_path.read_text())
    if packet_path.with_name("revocation.json").exists():
        raise SubmissionGuardError("Readiness packet has been revoked.")
    approval_path = packet_path.with_name("approval.json")
    if not approval_path.exists():
        raise SubmissionGuardError("No explicit human approval exists for this run.")
    approval = json.loads(approval_path.read_text())
    for key in (
        "run_id",
        "round_number",
        "target_model_id",
        "submission_sha256",
        "validation_sha256",
    ):
        if approval.get(key) != packet.get(key):
            raise SubmissionGuardError(f"Approval does not match readiness field: {key}.")
    if now > datetime.fromisoformat(approval["expires_at"]):
        raise SubmissionGuardError("Approval has expired.")
    submission_path = Path(packet["submission_path"])
    if not submission_path.exists() or sha256_file(submission_path) != packet["submission_sha256"]:
        raise SubmissionGuardError("Submission artifact changed after readiness approval.")
    return packet


class SubmissionLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"submissions": {}}
        return json.loads(self.path.read_text())

    @staticmethod
    def key(round_number: int, model_id: str) -> str:
        return f"{int(round_number)}:{model_id}"

    def contains(self, round_number: int, model_id: str) -> bool:
        return self.key(round_number, model_id) in self._read().get("submissions", {})

    def record(
        self,
        *,
        round_number: int,
        model_id: str,
        submission_id: str,
        run_id: str,
        verified: bool,
    ) -> None:
        payload = self._read()
        key = self.key(round_number, model_id)
        if key in payload.setdefault("submissions", {}):
            raise SubmissionGuardError(
                f"Round {round_number} already has a recorded submission for model {model_id}."
            )
        payload["submissions"][key] = {
            "round_number": int(round_number),
            "model_id": model_id,
            "submission_id": submission_id,
            "run_id": run_id,
            "verified": bool(verified),
            "recorded_at": utc_now().isoformat(),
        }
        _atomic_json(self.path, payload)
