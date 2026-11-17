"""Canonical, provenance-rich project and tournament metrics snapshots.

Counts with different meanings stay separate: raw uploads, model-round API
records, resolved scores, and full-portfolio rounds are never conflated.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping

from .config import BACKEND_ROOT, settings
from .numerai_ops import build_napi
from .staking import _score_value


SCHEMA_VERSION = 1
REQUIRED_SOURCES = ("numerai", "dataset", "operations", "quality")


def _utc(value: datetime | None = None) -> datetime:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _round_number(row: Mapping[str, Any]) -> int | None:
    for key in ("roundNumber", "round_number", "round", "tournamentRound"):
        value = _as_int(row.get(key))
        if value is not None:
            return value
    return None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return fmean(clean) if clean else None


def _metric(
    value: Any,
    *,
    unit: str,
    source: str,
    measured_at: str,
    scope: str,
    definition: str,
    window: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "value": value,
        "unit": unit,
        "source": source,
        "measured_at": measured_at,
        "scope": scope,
        "definition": definition,
    }
    if window:
        result["window"] = dict(window)
    return result


def _default_parquet_reader(path: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    return {
        "rows": int(metadata.num_rows),
        "columns": int(metadata.num_columns),
        "row_groups": int(metadata.num_row_groups),
    }


class MetricsLedger:
    """Collect and persist a non-destructive metrics snapshot."""

    def __init__(
        self,
        *,
        state_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        backend_root: str | Path | None = None,
        napi: Any | None = None,
        now: datetime | None = None,
        start_round: int | None = None,
        project_start_utc: str | datetime | None = None,
        recent_rounds: int | None = None,
        parquet_reader: Callable[[Path], Mapping[str, Any]] | None = None,
    ):
        self.state_dir = Path(state_dir or settings.CONTROL_PLANE_DIR)
        self.data_dir = Path(data_dir or settings.DATA_DIR)
        self.backend_root = Path(backend_root or BACKEND_ROOT)
        self.napi = napi
        self.now = _utc(now)
        self.start_round = int(start_round or settings.METRICS_START_ROUND)
        self.project_start = _as_datetime(
            project_start_utc or settings.METRICS_PROJECT_START_UTC
        )
        if self.project_start is None:
            raise ValueError("METRICS_PROJECT_START_UTC must be an ISO-8601 timestamp")
        self.recent_rounds = int(recent_rounds or settings.METRICS_RECENT_ROUNDS)
        self.parquet_reader = parquet_reader or _default_parquet_reader
        self.metrics_dir = self.state_dir / "metrics"

    @property
    def measured_at(self) -> str:
        return self.now.isoformat()

    def _api(self) -> Any:
        if self.napi is None:
            self.napi = build_napi()
        return self.napi

    def _collect_numerai(self) -> tuple[dict[str, Any], dict[str, Any]]:
        napi = self._api()
        models = dict(napi.get_models())
        current_round = int(napi.get_current_round())
        all_rows: list[dict[str, Any]] = []
        uploads: list[dict[str, Any]] = []
        upload_errors: dict[str, str] = {}
        model_details: dict[str, Any] = {}
        observations_by_model: dict[str, dict[int, dict[str, Any]]] = {}

        for model_name, model_id in models.items():
            returned = list(napi.round_model_performances_v2(model_id))
            if any(not isinstance(row, Mapping) or _round_number(row) is None for row in returned):
                raise ValueError(
                    f"Numerai performance schema changed for {model_name}: "
                    "one or more records have no round number"
                )
            project_rows = [
                dict(row)
                for row in returned
                if (_round_number(row) or -1) >= self.start_round
            ]
            all_rows.extend(project_rows)
            observations: dict[int, dict[str, Any]] = {}
            for row in project_rows:
                round_number = _round_number(row)
                if round_number is None or not row.get("roundResolved"):
                    continue
                corr = _score_value(row, ("canon_corr", "corr", "v2_corr20"))
                if corr is None:
                    continue
                observations[round_number] = {
                    "round_number": round_number,
                    "corr": corr,
                    "mmc": _score_value(row, ("canon_mmc", "mmc")),
                    "at_risk_nmr": float(row.get("atRisk") or 0.0),
                }
            observations_by_model[model_name] = observations
            ordered = [observations[key] for key in sorted(observations)]
            recent = ordered[-self.recent_rounds :]
            model_details[model_name] = {
                "model_id": model_id,
                "api_performance_records_since_start": len(project_rows),
                "resolved_scored_rounds_since_start": len(ordered),
                "latest_resolved_round": recent[-1]["round_number"] if recent else None,
                "recent": {
                    "requested_rounds": self.recent_rounds,
                    "observed_rounds": len(recent),
                    "mean_corr": _mean(row["corr"] for row in recent),
                    "mean_mmc": _mean(row["mmc"] for row in recent),
                    "positive_corr_rate": _mean(
                        1.0 if row["corr"] > 0 else 0.0 for row in recent
                    ),
                    "round_numbers": [row["round_number"] for row in recent],
                },
            }
            try:
                returned_uploads = list(napi.submission_ids(model_id))
                project_uploads = []
                unclassified_uploads = 0
                for row in returned_uploads:
                    if not isinstance(row, Mapping):
                        unclassified_uploads += 1
                        continue
                    round_number = _round_number(row)
                    inserted_at = _as_datetime(row.get("insertedAt"))
                    if round_number is None and inserted_at is None:
                        unclassified_uploads += 1
                        continue
                    if (
                        (round_number is not None and round_number >= self.start_round)
                        or (
                            round_number is None
                            and inserted_at is not None
                            and inserted_at >= self.project_start
                        )
                    ):
                        project_uploads.append(dict(row))
                if unclassified_uploads:
                    upload_errors[model_name] = (
                        f"{unclassified_uploads} upload record(s) lacked both "
                        "round number and a valid insertedAt timestamp"
                    )
                    model_details[model_name]["raw_upload_records_since_start"] = None
                else:
                    uploads.extend(project_uploads)
                    model_details[model_name]["raw_upload_records_since_start"] = len(project_uploads)
            except Exception as exc:
                upload_errors[model_name] = f"{type(exc).__name__}: {exc}"
                model_details[model_name]["raw_upload_records_since_start"] = None

        distinct_rounds = sorted(
            round_number
            for round_number in {
                _round_number(row) for row in all_rows
            }
            if round_number is not None
        )
        common_rounds = sorted(
            set.intersection(*(set(rows) for rows in observations_by_model.values()))
            if observations_by_model
            else set()
        )
        recent_common = common_rounds[-self.recent_rounds :]
        portfolio_observations = []
        for round_number in recent_common:
            rows = [rows[round_number] for rows in observations_by_model.values()]
            portfolio_observations.append(
                {
                    "round_number": round_number,
                    "mean_corr": _mean(row["corr"] for row in rows),
                    "mean_mmc": _mean(row["mmc"] for row in rows),
                    "model_count": len(rows),
                }
            )
        portfolio = {
            "method": "equal-weight mean across every account model",
            "coverage_rule": "round included only when every model has a resolved CORR score",
            "requested_rounds": self.recent_rounds,
            "observed_full_coverage_rounds": len(portfolio_observations),
            "round_numbers": recent_common,
            "mean_corr": _mean(row["mean_corr"] for row in portfolio_observations),
            "mean_mmc": _mean(row["mean_mmc"] for row in portfolio_observations),
            "positive_corr_rate": _mean(
                1.0 if (row["mean_corr"] or 0.0) > 0 else 0.0
                for row in portfolio_observations
            ),
            "observations": portfolio_observations,
        }
        window = {"start_round": self.start_round, "end_round": current_round}
        metrics = {
            "numerai.current_round": _metric(
                current_round, unit="round", source="Numerai API:get_current_round",
                measured_at=self.measured_at, scope="account",
                definition="Current tournament round reported by Numerai.",
            ),
            "numerai.models": _metric(
                len(models), unit="models", source="Numerai API:get_models",
                measured_at=self.measured_at, scope="account",
                definition="Model slots currently owned by the authenticated account.",
            ),
            "numerai.performance_records": _metric(
                len(all_rows), unit="model-round records",
                source="Numerai API:round_model_performances_v2",
                measured_at=self.measured_at, scope="all account models",
                definition="API performance rows; one model in one round is one record.",
                window=window,
            ),
            "numerai.distinct_rounds": _metric(
                len(distinct_rounds), unit="rounds",
                source="Numerai API:round_model_performances_v2",
                measured_at=self.measured_at, scope="union across all account models",
                definition="Distinct round numbers represented by performance records.",
                window=window,
            ),
            "numerai.resolved_scored_model_rounds": _metric(
                sum(len(rows) for rows in observations_by_model.values()),
                unit="model-round records", source="Numerai API:round_model_performances_v2",
                measured_at=self.measured_at, scope="all account models",
                definition="Resolved performance records with an available CORR score.",
                window=window,
            ),
        }
        for model_name, model in model_details.items():
            slug = model_name.casefold().replace(" ", "_")
            recent = model["recent"]
            for key, unit, definition in (
                ("mean_corr", "correlation", "Arithmetic mean CORR over the model's latest resolved scored rounds."),
                ("mean_mmc", "MMC", "Arithmetic mean MMC over the model's latest resolved scored rounds with MMC available."),
                ("positive_corr_rate", "fraction", "Fraction of the model's latest resolved scored rounds with CORR above zero."),
            ):
                metrics[f"numerai.model.{slug}.recent_{key}"] = _metric(
                    recent[key], unit=unit,
                    source="derived from Numerai authenticated API",
                    measured_at=self.measured_at, scope=model_name,
                    definition=definition,
                    window={"latest_resolved_rounds": self.recent_rounds},
                )
        for key, unit, definition in (
            ("mean_corr", "correlation", portfolio["coverage_rule"]),
            ("mean_mmc", "MMC", portfolio["coverage_rule"]),
            ("positive_corr_rate", "fraction", "Fraction of included rounds with portfolio mean CORR above zero."),
        ):
            metrics[f"numerai.portfolio_recent_{key}"] = _metric(
                portfolio[key], unit=unit, source="derived from Numerai authenticated API",
                measured_at=self.measured_at, scope="equal-weight full account portfolio",
                definition=definition,
                window={"latest_full_coverage_rounds": self.recent_rounds},
            )
        if not upload_errors:
            metrics["numerai.raw_upload_records"] = _metric(
                len(uploads), unit="upload records", source="Numerai API:submission_ids",
                measured_at=self.measured_at, scope="all account models",
                definition=(
                    "Raw upload records; retries and replacements remain separate records. "
                    "The API omits round numbers, so production records are filtered by insertedAt."
                ),
                window={
                    "start_timestamp": self.project_start.isoformat(),
                    "end_timestamp": self.measured_at,
                },
            )
        return {
            "current_round": current_round,
            "start_round": self.start_round,
            "models": model_details,
            "distinct_round_numbers": distinct_rounds,
            "portfolio_recent": portfolio,
            "raw_upload_errors": upload_errors,
            "complete": not upload_errors,
        }, metrics

    def _collect_dataset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        version_dir = self.data_dir / settings.DATA_VERSION
        datasets: dict[str, Any] = {}
        total_rows = 0
        total_bytes = 0
        for name in ("train.parquet", "validation.parquet", "live.parquet"):
            path = version_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Required dataset is missing: {path}")
            metadata = dict(self.parquet_reader(path))
            size = path.stat().st_size
            row_count = int(metadata["rows"])
            total_rows += row_count
            total_bytes += size
            datasets[name] = {
                "path": str(path),
                "rows": row_count,
                "bytes": size,
                "columns": _as_int(metadata.get("columns")),
                "row_groups": _as_int(metadata.get("row_groups")),
                "modified_at": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
            }
        features_path = version_dir / "features.json"
        features = _load_object(features_path)
        feature_sets = features.get("feature_sets", {})
        if not isinstance(feature_sets, dict):
            feature_sets = {}
        details = {
            "data_version": settings.DATA_VERSION,
            "version_dir": str(version_dir),
            "files": datasets,
            "totals": {"rows": total_rows, "bytes": total_bytes},
            "feature_sets": {
                str(name): len(values) if isinstance(values, list) else None
                for name, values in feature_sets.items()
            },
        }
        source = "local Parquet footer metadata and filesystem stat"
        metrics = {
            "dataset.total_rows": _metric(
                total_rows, unit="rows", source=source,
                measured_at=self.measured_at, scope=settings.DATA_VERSION,
                definition="Sum of physical rows in train, validation, and live files; splits are not deduplicated.",
            ),
            "dataset.total_bytes": _metric(
                total_bytes, unit="bytes", source=source,
                measured_at=self.measured_at, scope=settings.DATA_VERSION,
                definition="Sum of filesystem byte sizes for train, validation, and live files.",
            ),
        }
        for filename, row in datasets.items():
            key = filename.removesuffix(".parquet")
            metrics[f"dataset.{key}_rows"] = _metric(
                row["rows"], unit="rows", source="local Parquet footer metadata",
                measured_at=self.measured_at, scope=f"{settings.DATA_VERSION}/{filename}",
                definition=f"Physical row count in {filename}.",
            )
            metrics[f"dataset.{key}_bytes"] = _metric(
                row["bytes"], unit="bytes", source="local filesystem stat",
                measured_at=self.measured_at, scope=f"{settings.DATA_VERSION}/{filename}",
                definition=f"Filesystem size of {filename}.",
            )
        return details, metrics

    @staticmethod
    def _report_mode(path: Path) -> str:
        parts = path.stem.split("_", 2)
        return parts[2] if len(parts) == 3 else path.stem

    def _collect_operations(self) -> tuple[dict[str, Any], dict[str, Any]]:
        automation = self.backend_root / "automation"
        reports_dir = automation / "reports"
        logs_dir = automation / "logs"
        json_reports = sorted(reports_dir.glob("*.json"))
        markdown_reports = sorted(reports_dir.glob("*.md"))
        json_stems = {path.stem for path in json_reports}
        markdown_stems = {path.stem for path in markdown_reports}
        modes = Counter(self._report_mode(path) for path in json_reports)
        log_files = [path for path in logs_dir.glob("*") if path.is_file()]
        plist_files = sorted(automation.glob("com.numerai.*.plist"))
        ledger_path = self.state_dir / "submission_ledger.json"
        submission_ledger = _load_object(ledger_path)
        verified_submissions = sum(
            1
            for value in submission_ledger.values()
            if isinstance(value, dict) and value.get("verified")
        )
        runtime_root = self.backend_root.parents[1]
        backup_dir = runtime_root / "backups" / "state"
        backups = sorted(backup_dir.glob("numerai-state-*.tar.gz"))
        state_snapshots = {}
        state_payloads = {}
        for name in ("platform", "competition", "system_health"):
            path = self.state_dir / name / "latest.json"
            payload = _load_object(path)
            state_payloads[name] = payload
            state_snapshots[name] = {
                "path": str(path),
                "exists": path.is_file(),
                "ok": payload.get("ok"),
                "status": payload.get("status"),
                "generated_at": payload.get("generated_at"),
            }
        details = {
            "reports": {
                "directory": str(reports_dir),
                "json_files": len(json_reports),
                "markdown_files": len(markdown_reports),
                "paired_runs": len(json_stems & markdown_stems),
                "unpaired_json": sorted(json_stems - markdown_stems),
                "unpaired_markdown": sorted(markdown_stems - json_stems),
                "json_by_mode": dict(sorted(modes.items())),
            },
            "logs": {"directory": str(logs_dir), "files": len(log_files)},
            "schedules": {
                "files": len(plist_files),
                "labels": [path.stem for path in plist_files],
            },
            "submissions": {
                "ledger_path": str(ledger_path),
                "ledger_records": len(submission_ledger),
                "verified_records": verified_submissions,
            },
            "backups": {"directory": str(backup_dir), "files": len(backups)},
            "latest_state": state_snapshots,
        }
        source = "local durable automation artifacts"
        metrics = {
            "operations.report_artifacts": _metric(
                len(json_reports) + len(markdown_reports), unit="files", source=source,
                measured_at=self.measured_at, scope=str(reports_dir),
                definition="JSON plus Markdown files in the reports directory.",
            ),
            "operations.paired_report_runs": _metric(
                len(json_stems & markdown_stems), unit="runs", source=source,
                measured_at=self.measured_at, scope=str(reports_dir),
                definition="Unique report stems having both JSON and Markdown artifacts.",
            ),
            "operations.log_files": _metric(
                len(log_files), unit="files", source=source,
                measured_at=self.measured_at, scope=str(logs_dir),
                definition="Regular files currently present in the automation log directory.",
            ),
            "operations.scheduled_jobs_defined": _metric(
                len(plist_files), unit="job definitions",
                source="repository launchd property lists",
                measured_at=self.measured_at, scope=str(automation),
                definition="Version-controlled job definitions; not proof that services are loaded.",
            ),
            "operations.verified_submission_records": _metric(
                verified_submissions, unit="ledger records",
                source="local submission_ledger.json",
                measured_at=self.measured_at, scope="installed control plane",
                definition="Ledger entries explicitly marked verified; not raw API uploads.",
            ),
        }
        health = state_payloads["system_health"]
        for key, unit, definition in (
            ("healthy_job_count", "jobs", "Native workers whose service and durable evidence checks are healthy."),
            ("expected_job_count", "jobs", "Native workers expected by the system-health specification."),
        ):
            if health.get(key) is not None:
                metrics[f"operations.{key}"] = _metric(
                    health[key], unit=unit,
                    source=str(self.state_dir / "system_health" / "latest.json"),
                    measured_at=self.measured_at, scope="native automation",
                    definition=definition,
                )
        competition = state_payloads["competition"]
        coverage = competition.get("current_round_coverage") or {}
        qualification = competition.get("season_qualification") or {}
        for name, value, unit, definition in (
            ("current_round_verified_models", coverage.get("submitted_verified"), "models", "Account models with a verified local submission record for the current round."),
            ("current_round_account_models", coverage.get("account_models"), "models", "Account model slots expected in the current competition snapshot."),
            ("season_qualified_rounds", qualification.get("qualified_round_count"), "rounds", "Season rounds meeting the configured total-at-risk qualification rule."),
        ):
            if value is not None:
                metrics[f"operations.{name}"] = _metric(
                    value, unit=unit,
                    source=str(self.state_dir / "competition" / "latest.json"),
                    measured_at=self.measured_at, scope="account competition state",
                    definition=definition,
                )
        return details, metrics

    def _collect_quality(self) -> tuple[dict[str, Any], dict[str, Any]]:
        roots = {
            "application": self.backend_root / "app",
            "automation": self.backend_root / "automation",
            "tests": self.backend_root / "tests",
        }
        counts: dict[str, Any] = {}
        parse_errors: list[str] = []
        test_functions = 0
        for label, root in roots.items():
            files = sorted(root.rglob("*.py")) if root.exists() else []
            lines = 0
            nonblank = 0
            for path in files:
                try:
                    text = path.read_text()
                    lines += len(text.splitlines())
                    nonblank += sum(bool(line.strip()) for line in text.splitlines())
                    if label == "tests":
                        tree = ast.parse(text, filename=str(path))
                        test_functions += sum(
                            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name.startswith("test_")
                            for node in ast.walk(tree)
                        )
                except (OSError, SyntaxError, UnicodeDecodeError) as exc:
                    parse_errors.append(f"{path}: {type(exc).__name__}: {exc}")
            counts[label] = {
                "python_files": len(files),
                "physical_lines": lines,
                "nonblank_lines": nonblank,
            }
        commit = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=self.backend_root,
                capture_output=True, text=True, check=False,
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
        except OSError:
            pass
        definition = (
            "AST-counted Python functions/methods named test_*; "
            "this is not a passing-test count."
        )
        details = {
            "source_counts": counts,
            "test_functions_defined": test_functions,
            "definition": definition,
            "git_commit": commit,
            "parse_errors": parse_errors,
        }
        metrics = {
            "quality.test_functions_defined": _metric(
                test_functions, unit="test functions", source="Python AST scan",
                measured_at=self.measured_at, scope=str(roots["tests"]),
                definition=definition,
            ),
            "quality.production_python_files": _metric(
                counts["application"]["python_files"] + counts["automation"]["python_files"],
                unit="files", source="repository filesystem scan",
                measured_at=self.measured_at, scope="app and automation",
                definition="Python source files under app/ and automation/.",
            ),
            "quality.production_nonblank_lines": _metric(
                counts["application"]["nonblank_lines"] + counts["automation"]["nonblank_lines"],
                unit="nonblank physical lines", source="repository filesystem scan",
                measured_at=self.measured_at, scope="app and automation",
                definition="Nonblank physical Python lines; not logical or authored-only LOC.",
            ),
        }
        return details, metrics

    def _collect_research(self) -> tuple[dict[str, Any], dict[str, Any]]:
        reports_dir = self.backend_root / "automation" / "reports"
        candidates = list(reports_dir.glob("*_research-evaluate.json"))
        candidates += list((self.state_dir / "research").glob("*_evaluation_packet.json"))
        if not candidates:
            return {"available": False}, {}
        latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        payload = _load_object(latest)
        evaluation = payload.get("evaluation", payload)
        if not isinstance(evaluation, dict):
            return {"available": False, "path": str(latest)}, {}
        overall = evaluation.get("overall", {})
        bootstrap = evaluation.get("bootstrap_mean_correlation", {})
        details = {
            "available": True,
            "path": str(latest),
            "model_name": evaluation.get("model_name"),
            "generated_at": evaluation.get("generated_at"),
            "overall": overall,
            "recent": evaluation.get("recent"),
            "feature_exposure": evaluation.get("feature_exposure"),
            "bootstrap_mean_correlation": bootstrap,
            "recommendation": payload.get("recommendation"),
        }
        metrics: dict[str, Any] = {}
        definitions = {
            "era_count": ("eras", "Validation eras represented in the robustness evaluation."),
            "mean_correlation": ("correlation", "Arithmetic mean of per-era validation correlations."),
            "sharpe": ("ratio", "Mean per-era correlation divided by sample standard deviation."),
            "max_drawdown": ("correlation units", "Maximum drawdown of cumulative per-era correlation."),
            "hit_rate": ("fraction", "Fraction of validation eras with correlation above zero."),
        }
        for key, (unit, definition) in definitions.items():
            if isinstance(overall, dict) and overall.get(key) is not None:
                metrics[f"research.latest_{key}"] = _metric(
                    overall[key], unit=unit, source=str(latest),
                    measured_at=self.measured_at,
                    scope=str(evaluation.get("model_name") or "latest candidate"),
                    definition=definition,
                )
        if evaluation.get("feature_exposure") is not None:
            metrics["research.latest_feature_exposure"] = _metric(
                evaluation["feature_exposure"], unit="absolute correlation",
                source=str(latest), measured_at=self.measured_at,
                scope=str(evaluation.get("model_name") or "latest candidate"),
                definition="Maximum absolute feature-to-prediction correlation in validation.",
            )
        for key in ("lower", "upper"):
            if isinstance(bootstrap, dict) and bootstrap.get(key) is not None:
                metrics[f"research.latest_bootstrap_{key}"] = _metric(
                    bootstrap[key], unit="correlation", source=str(latest),
                    measured_at=self.measured_at,
                    scope=str(evaluation.get("model_name") or "latest candidate"),
                    definition=f"Bootstrap interval {key} bound for mean validation correlation.",
                )
        return details, metrics

    @staticmethod
    def render_markdown(snapshot: Mapping[str, Any]) -> str:
        lines = [
            "# Canonical Metrics Snapshot", "",
            f"- Generated: `{snapshot.get('generated_at')}`",
            f"- Status: `{snapshot.get('status')}`",
            f"- Snapshot: `{snapshot.get('snapshot_id')}`", "",
            "## Source status", "",
            "| Source | OK | Error |", "|---|---:|---|",
        ]
        for name, row in snapshot.get("sources", {}).items():
            error = str(row.get("error") or "").replace("|", "\\|")
            lines.append(f"| {name} | {row.get('ok')} | {error} |")
        lines += [
            "", "## Metrics", "",
            "| Metric | Value | Unit | Scope | Definition |",
            "|---|---:|---|---|---|",
        ]
        for name, row in sorted(snapshot.get("metrics", {}).items()):
            definition = str(row.get("definition", "")).replace("|", "\\|")
            scope = str(row.get("scope", "")).replace("|", "\\|")
            lines.append(
                f"| `{name}` | {row.get('value')} | {row.get('unit')} | {scope} | {definition} |"
            )
        lines += [
            "",
            "Every value includes source, measurement time, scope, definition, and window (when applicable) in `latest.json`.",
            "",
        ]
        return "\n".join(lines)

    def _persist(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        stamp = self.now.strftime("%Y%m%dT%H%M%S%fZ")
        history_path = self.metrics_dir / "snapshots" / f"{stamp}.json"
        latest_path = self.metrics_dir / "latest.json"
        markdown_path = self.metrics_dir / "latest.md"
        successful_path = self.metrics_dir / "latest_successful.json"
        index_path = self.metrics_dir / "index.json"
        snapshot["artifacts"] = {
            "snapshot": str(history_path),
            "latest": str(latest_path),
            "latest_markdown": str(markdown_path),
            "latest_successful": str(successful_path),
            "index": str(index_path),
        }
        _atomic_json(history_path, snapshot)
        _atomic_json(latest_path, snapshot)
        _atomic_text(markdown_path, self.render_markdown(snapshot))
        if snapshot["ok"]:
            _atomic_json(successful_path, snapshot)
        index = _load_object(index_path)
        entries = index.get("snapshots", [])
        if not isinstance(entries, list):
            entries = []
        entries.append(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "generated_at": snapshot["generated_at"],
                "ok": snapshot["ok"],
                "status": snapshot["status"],
                "path": str(history_path),
            }
        )
        _atomic_json(index_path, {"schema_version": 1, "snapshots": entries[-400:]})
        return snapshot

    def snapshot(self) -> dict[str, Any]:
        collectors = {
            "numerai": self._collect_numerai,
            "dataset": self._collect_dataset,
            "operations": self._collect_operations,
            "quality": self._collect_quality,
            "research": self._collect_research,
        }
        metrics: dict[str, Any] = {}
        details: dict[str, Any] = {}
        sources: dict[str, Any] = {}
        for name, collect in collectors.items():
            try:
                source_details, source_metrics = collect()
                details[name] = source_details
                metrics.update(source_metrics)
                source_ok = bool(source_details.get("complete", True))
                sources[name] = {
                    "ok": source_ok,
                    "metric_count": len(source_metrics),
                    "error": (
                        None
                        if source_ok
                        else "Source returned incomplete data; see source details."
                    ),
                }
            except Exception as exc:
                sources[name] = {
                    "ok": False,
                    "metric_count": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        required_ok = all(sources[name]["ok"] for name in REQUIRED_SOURCES)
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": self.now.strftime("metrics-%Y%m%dT%H%M%S%fZ"),
            "generated_at": self.measured_at,
            "ok": required_ok,
            "status": "METRICS_SNAPSHOT_OK" if required_ok else "METRICS_SNAPSHOT_PARTIAL",
            "summary": (
                f"Collected {len(metrics)} canonical metrics from "
                f"{sum(row['ok'] for row in sources.values())}/{len(sources)} sources."
            ),
            "configuration": {
                "data_version": settings.DATA_VERSION,
                "start_round": self.start_round,
                "project_start_utc": self.project_start.isoformat(),
                "recent_rounds": self.recent_rounds,
                "required_sources": list(REQUIRED_SOURCES),
            },
            "sources": sources,
            "metrics": metrics,
            "details": details,
        }
        return self._persist(snapshot)


def read_metrics_history(
    state_dir: str | Path | None = None,
    *,
    limit: int = 30,
) -> dict[str, Any]:
    metrics_dir = Path(state_dir or settings.CONTROL_PLANE_DIR) / "metrics"
    index = _load_object(metrics_dir / "index.json")
    entries = index.get("snapshots", [])
    if not isinstance(entries, list):
        entries = []
    safe_limit = max(1, min(int(limit), 400))
    return {
        "ok": True,
        "total_indexed": len(entries),
        "limit": safe_limit,
        "snapshots": list(reversed(entries[-safe_limit:])),
    }
