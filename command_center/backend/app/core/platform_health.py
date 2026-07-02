from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Callable, Mapping

from .agentic_control_plane import AgenticControlPlane
from .config import settings
from .numerai_ops import build_napi


REQUIRED_DATASETS = (
    "features.json",
    "train.parquet",
    "validation.parquet",
    "live.parquet",
)
REQUIRED_API_METHODS = (
    "get_account",
    "get_models",
    "get_current_round",
    "list_datasets",
    "stake_get",
)
VERSION_PATTERN = re.compile(r"^v(\d+(?:\.\d+)*)/")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def _version_key(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.removeprefix("v").split("."))
    except ValueError:
        return ()


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


class PlatformHealthMonitor:
    """Read-only contract monitor for Numerai APIs, data, and account mapping."""

    def __init__(
        self,
        state_dir: str | Path | None = None,
        registry_dir: str | Path | None = None,
        data_root: str | Path | None = None,
        *,
        data_version: str | None = None,
        feature_set: str | None = None,
        napi: Any | None = None,
        now: datetime | None = None,
        round_window_resolver: Callable[[Any, int], Mapping[str, Any]] | None = None,
    ):
        self.state_dir = Path(state_dir or settings.CONTROL_PLANE_DIR)
        self.registry_dir = Path(registry_dir or settings.MODEL_REGISTRY_DIR)
        self.data_root = Path(data_root or settings.DATA_DIR).expanduser().resolve()
        self.data_version = data_version or settings.DATA_VERSION
        self.feature_set = feature_set or settings.FEATURE_SET
        self.napi = napi
        self.now = now or datetime.now(timezone.utc)
        self.round_window_resolver = (
            round_window_resolver or AgenticControlPlane._submission_window
        )

    @staticmethod
    def _check(
        code: str,
        status: str,
        summary: str,
        **evidence: Any,
    ) -> dict[str, Any]:
        return {
            "code": code,
            "status": status,
            "summary": summary,
            "evidence": evidence,
        }

    @staticmethod
    def _error(exc: Exception) -> str:
        return f"{type(exc).__name__}: {str(exc)[:400]}"

    def _api(self) -> Any:
        if self.napi is None:
            self.napi = build_napi()
        return self.napi

    def _portfolio_names(self) -> set[str]:
        path = self.registry_dir / "portfolio" / "current.json"
        if not path.exists():
            return set()
        payload = json.loads(path.read_text())
        return {
            str(row["model_name"]).casefold()
            for row in payload.get("assignments", [])
        }

    def _local_data_check(self) -> dict[str, Any]:
        version_dir = self.data_root / self.data_version
        missing = [
            name for name in REQUIRED_DATASETS if not (version_dir / name).is_file()
        ]
        if missing:
            return self._check(
                "LOCAL_DATA_CONTRACT",
                "FAIL",
                "Required local datasets are missing.",
                version=self.data_version,
                missing=missing,
                path=str(version_dir),
            )

        features_payload = json.loads((version_dir / "features.json").read_text())
        feature_sets = features_payload.get("feature_sets", {})
        selected = feature_sets.get(self.feature_set)
        if not isinstance(selected, list) or not selected:
            return self._check(
                "LOCAL_DATA_CONTRACT",
                "FAIL",
                "Configured feature set is unavailable or empty.",
                version=self.data_version,
                feature_set=self.feature_set,
                available_feature_sets=sorted(feature_sets),
            )

        import pyarrow.parquet as pq

        metadata: dict[str, Any] = {
            "features.json": {
                "feature_set": self.feature_set,
                "feature_count": len(selected),
            }
        }
        for name in REQUIRED_DATASETS[1:]:
            parquet = pq.ParquetFile(version_dir / name)
            required_columns = {"era"}
            if name != "live.parquet":
                required_columns.add("target")
            columns = set(parquet.schema.names)
            missing_columns = sorted(required_columns - columns)
            rows = int(parquet.metadata.num_rows)
            if missing_columns or rows <= 0:
                return self._check(
                    "LOCAL_DATA_CONTRACT",
                    "FAIL",
                    "A local parquet dataset violates the required schema.",
                    file=name,
                    missing_columns=missing_columns,
                    rows=rows,
                )
            metadata[name] = {
                "rows": rows,
                "row_groups": int(parquet.num_row_groups),
            }
        return self._check(
            "LOCAL_DATA_CONTRACT",
            "PASS",
            "Configured local data is present and schema-compatible.",
            version=self.data_version,
            path=str(version_dir),
            datasets=metadata,
        )

    def snapshot(self) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        generated_at = self.now.astimezone(timezone.utc)

        try:
            napi = self._api()
        except Exception as exc:
            checks.append(
                self._check(
                    "API_AUTHENTICATION",
                    "FAIL",
                    "Numerai API client could not be initialized.",
                    error=self._error(exc),
                )
            )
            return self._finalize(generated_at, checks)

        missing_methods = [
            name for name in REQUIRED_API_METHODS if not callable(getattr(napi, name, None))
        ]
        checks.append(
            self._check(
                "CLIENT_API_SURFACE",
                "FAIL" if missing_methods else "PASS",
                (
                    "Required NumerAPI methods are missing."
                    if missing_methods
                    else "Required read-only NumerAPI methods are available."
                ),
                missing_methods=missing_methods,
                numerapi_version=self._numerapi_version(),
            )
        )

        try:
            account = napi.get_account()
            required_account_fields = {"username", "availableNmr"}
            missing_account_fields = sorted(required_account_fields - set(account))
            username = str(account.get("username") or "")
            available_nmr = _as_float(account.get("availableNmr"))
            if not username or missing_account_fields:
                raise ValueError(
                    f"missing account fields: {missing_account_fields or ['username']}"
                )
            checks.append(
                self._check(
                    "ACCOUNT_SCHEMA",
                    "PASS",
                    "Authenticated account schema is compatible.",
                    username=username,
                    available_nmr=available_nmr,
                )
            )
        except Exception as exc:
            checks.append(
                self._check(
                    "ACCOUNT_SCHEMA",
                    "FAIL",
                    "Authenticated account response is incompatible.",
                    error=self._error(exc),
                )
            )

        models: dict[str, str] = {}
        try:
            raw_models = napi.get_models()
            if not isinstance(raw_models, Mapping) or not raw_models:
                raise ValueError("get_models returned no model mapping")
            models = {
                str(name): str(model_id)
                for name, model_id in raw_models.items()
                if name and model_id
            }
            if len(models) != len(raw_models):
                raise ValueError("model mapping contains empty names or ids")
            portfolio_names = self._portfolio_names()
            account_names = {name.casefold() for name in models}
            missing_assignments = sorted(account_names - portfolio_names)
            unknown_assignments = sorted(portfolio_names - account_names)
            mapping_status = (
                "PASS"
                if portfolio_names
                and not missing_assignments
                and not unknown_assignments
                else "FAIL"
            )
            checks.append(
                self._check(
                    "MODEL_PORTFOLIO_MAPPING",
                    mapping_status,
                    (
                        "Account models and active portfolio match."
                        if mapping_status == "PASS"
                        else "Account models and active portfolio do not match."
                    ),
                    model_count=len(models),
                    account_models=sorted(models),
                    missing_assignments=missing_assignments,
                    unknown_assignments=unknown_assignments,
                    portfolio_present=bool(portfolio_names),
                )
            )
        except Exception as exc:
            checks.append(
                self._check(
                    "MODEL_PORTFOLIO_MAPPING",
                    "FAIL",
                    "Model mapping could not be verified.",
                    error=self._error(exc),
                )
            )

        try:
            current_round = int(napi.get_current_round())
            window = dict(self.round_window_resolver(napi, current_round))
            required_window_fields = {"number", "openTime", "closeTime"}
            missing_window_fields = sorted(required_window_fields - set(window))
            if int(window.get("number", -1)) != current_round or missing_window_fields:
                raise ValueError(
                    f"round window mismatch or missing fields: {missing_window_fields}"
                )
            checks.append(
                self._check(
                    "ROUND_WINDOW_SCHEMA",
                    "PASS",
                    "Current round and submission-window schema are compatible.",
                    current_round=current_round,
                    open_time=window.get("openTime"),
                    close_time=window.get("closeTime"),
                    close_staking_time=window.get("closeStakingTime"),
                    accepting_submissions=bool(window.get("accepting_submissions")),
                )
            )
        except Exception as exc:
            checks.append(
                self._check(
                    "ROUND_WINDOW_SCHEMA",
                    "FAIL",
                    "Current round window could not be verified.",
                    error=self._error(exc),
                )
            )

        try:
            remote_datasets = {str(name) for name in napi.list_datasets()}
            versions = sorted(
                {
                    match.group(0).rstrip("/")
                    for name in remote_datasets
                    if (match := VERSION_PATTERN.match(name))
                },
                key=_version_key,
            )
            required_remote = {
                f"{self.data_version}/{name}" for name in REQUIRED_DATASETS
            }
            missing_remote = sorted(required_remote - remote_datasets)
            if missing_remote:
                checks.append(
                    self._check(
                        "REMOTE_DATA_VERSION",
                        "FAIL",
                        "Configured Numerai data version is missing required datasets.",
                        configured_version=self.data_version,
                        available_versions=versions,
                        missing=missing_remote,
                    )
                )
            else:
                newest = versions[-1] if versions else None
                newer_available = bool(
                    newest
                    and _version_key(newest) > _version_key(self.data_version)
                )
                checks.append(
                    self._check(
                        "REMOTE_DATA_VERSION",
                        "WARN" if newer_available else "PASS",
                        (
                            "A newer Numerai data version requires compatibility review."
                            if newer_available
                            else "Configured Numerai data version is current and complete."
                        ),
                        configured_version=self.data_version,
                        newest_version=newest,
                        available_versions=versions,
                        required_datasets=sorted(required_remote),
                    )
                )
        except Exception as exc:
            checks.append(
                self._check(
                    "REMOTE_DATA_VERSION",
                    "FAIL",
                    "Remote Numerai dataset catalog could not be verified.",
                    error=self._error(exc),
                )
            )

        try:
            checks.append(self._local_data_check())
        except Exception as exc:
            checks.append(
                self._check(
                    "LOCAL_DATA_CONTRACT",
                    "FAIL",
                    "Local data contract could not be verified.",
                    error=self._error(exc),
                )
            )

        try:
            if not models:
                raise ValueError("no verified account model mapping")
            stakes = {
                name: _as_float(napi.stake_get(name))
                for name in sorted(models)
            }
            checks.append(
                self._check(
                    "STAKE_READ_API",
                    "PASS",
                    "Read-only stake API is compatible for every account model.",
                    stakes_nmr=stakes,
                )
            )
        except Exception as exc:
            checks.append(
                self._check(
                    "STAKE_READ_API",
                    "FAIL",
                    "Read-only stake API is incompatible.",
                    error=self._error(exc),
                )
            )

        return self._finalize(generated_at, checks)

    @staticmethod
    def _numerapi_version() -> str | None:
        try:
            return package_version("numerapi")
        except PackageNotFoundError:
            return None

    def _finalize(
        self,
        generated_at: datetime,
        checks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        failures = [
            check["code"] for check in checks if check["status"] == "FAIL"
        ]
        warnings = [
            check["code"] for check in checks if check["status"] == "WARN"
        ]
        status = (
            "PLATFORM_CONTRACT_BROKEN"
            if failures
            else "PLATFORM_MIGRATION_REVIEW"
            if warnings
            else "PLATFORM_COMPATIBLE"
        )
        report: dict[str, Any] = {
            "ok": not failures,
            "status": status,
            "read_only": True,
            "generated_at": generated_at.isoformat(),
            "configured_data_version": self.data_version,
            "feature_set": self.feature_set,
            "checks": checks,
            "failures": failures,
            "warnings": warnings,
            "summary": (
                f"{len(checks)} contracts checked; "
                f"{len(failures)} failed; {len(warnings)} require review."
            ),
        }
        reports_dir = self.state_dir / "platform"
        history_path = (
            reports_dir / f"{generated_at.strftime('%Y%m%d_%H%M%S')}.json"
        )
        latest_path = reports_dir / "latest.json"
        _atomic_json(history_path, report)
        _atomic_json(latest_path, report)
        report["report_path"] = str(history_path)
        report["latest_path"] = str(latest_path)
        return report
