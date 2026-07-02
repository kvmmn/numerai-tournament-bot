#!/usr/bin/env python3
"""Governed Numerai operations runner.

Submission, portfolio promotion, and staking are separate workflows. Every
mutation requires a current challenge-bound approval; legacy direct-submit
modes remain disabled.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.numerai_mcp_client import NumeraiMCPError  # noqa: E402
from app.core.numerai_mcp_workflow import NumeraiMCPWorkflowRunner  # noqa: E402
from app.core.numerai_ops import (  # noqa: E402
    NumeraiOpsError,
    build_napi,
    build_submission_dataframe,
    build_ensemble_submission,
    compare_with_champion,
    evaluate_model,
    evaluate_model_suite,
    has_last_known_good,
    last_known_good_path,
    model_path,
    resolve_target_models,
    save_champion,
    save_last_known_good,
    submit_to_models,
    sync_datasets,
    train_model,
    train_model_suite,
)
from app.core.config import settings  # noqa: E402
from app.core.agentic_control_plane import AgenticControlPlane  # noqa: E402
from app.core.submission_guard import SubmissionGuardError  # noqa: E402
from app.core.performance_listener import PerformanceListener  # noqa: E402
from app.core.portfolio import (  # noqa: E402
    PortfolioControlPlane,
    activate_portfolio,
    approve_portfolio_proposal,
    create_portfolio_proposal,
)
from app.core.research import (  # noqa: E402
    evaluate_artifact_robustness,
    promotion_recommendation,
    write_evaluation_artifacts,
)
from app.core.model_registry import (  # noqa: E402
    approve_candidate_bundle,
    promote_candidate_bundle,
)
from app.core.staking import StakeControlPlane  # noqa: E402
from app.core.competition import CompetitionTracker  # noqa: E402
from app.core.alerts import AlertDispatcher  # noqa: E402
from app.core.platform_health import PlatformHealthMonitor  # noqa: E402
from app.core.system_health import SystemHealthMonitor  # noqa: E402


REPORTS_DIR = BACKEND_ROOT / "automation" / "reports"
LOGS_DIR = BACKEND_ROOT / "automation" / "logs"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def _write_reports(mode: str, payload: Dict[str, Any]) -> Tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _now().strftime("%Y%m%d_%H%M%S")
    base = REPORTS_DIR / f"{stamp}_{mode}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")

    with json_path.open("w") as f:
        json.dump(_serialize(payload), f, indent=2)

    status = payload.get("status", "unknown")
    ok = payload.get("ok", False)
    summary = payload.get("summary", "")
    with md_path.open("w") as f:
        f.write(f"# Daily Numerai Run\n\n")
        f.write(f"- `mode`: `{mode}`\n")
        f.write(f"- `ok`: `{ok}`\n")
        f.write(f"- `status`: `{status}`\n")
        if summary:
            f.write(f"- `summary`: {summary}\n")
        f.write(f"- `generated_at_utc`: `{_now().isoformat()}`\n\n")

        # Model metrics table (if available)
        suite = payload.get("suite_results", {})
        if suite and suite.get("models"):
            f.write("## Model Metrics\n\n")
            f.write("| Model | Type | Corr | Sharpe | Drawdown | Exposure |\n")
            f.write("|-------|------|------|--------|----------|----------|\n")
            for m in suite["models"]:
                metrics = m.get("metrics", {})
                if metrics:
                    f.write(
                        f"| {m.get('name', '?')} | {m.get('model_type', '?')} "
                        f"| {metrics.get('validation_correlation', 0):.5f} "
                        f"| {metrics.get('sharpe', 0):.4f} "
                        f"| {metrics.get('max_drawdown', 0):.4f} "
                        f"| {metrics.get('feature_exposure', 0):.4f} |\n"
                    )
            f.write("\n")

        ens = payload.get("ensemble_metrics")
        if ens:
            f.write("## Ensemble Metrics\n\n")
            f.write(f"- Correlation: {ens.get('validation_correlation', 0):.5f}\n")
            f.write(f"- Sharpe: {ens.get('sharpe', 0):.4f}\n")
            f.write(f"- Max Drawdown: {ens.get('max_drawdown', 0):.4f}\n\n")

        comp = payload.get("champion_comparison")
        if comp:
            f.write("## Champion Comparison\n\n")
            f.write(f"- Decision: **{comp.get('decision', '?')}**\n")
            f.write(f"- Reason: {comp.get('reason', '?')}\n\n")

        f.write("## Full Result\n\n")
        f.write("```json\n")
        f.write(json.dumps(_serialize(payload), indent=2))
        f.write("\n```\n")

    return json_path, md_path


# ---------------------------------------------------------------------------
# MCP modes (unchanged)
# ---------------------------------------------------------------------------

def run_mcp_preflight() -> Dict[str, Any]:
    runner = NumeraiMCPWorkflowRunner()
    result = asyncio.run(runner.preflight())
    result["ok"] = True
    result["status"] = "mcp_preflight_ok"
    result["summary"] = f"Discovered {result.get('tool_count', 0)} MCP tools."
    return result


def run_mcp_cycle(approve_submission: bool) -> Dict[str, Any]:
    if approve_submission:
        raise SubmissionGuardError(
            "MCP submission is disabled. Use agent-prepare, agent-approve, "
            "then agent-submit."
        )
    runner = NumeraiMCPWorkflowRunner()
    result = asyncio.run(
        runner.run_cycle(approve_submission=approve_submission)
    )
    result["ok"] = True
    result["status"] = "mcp_dry_run_ok"
    result["summary"] = "MCP workflow executed."
    return result


def run_numerapi_preflight() -> Dict[str, Any]:
    napi = build_napi()
    model_map = resolve_target_models(napi)
    result = {
        "ok": True,
        "status": "numerapi_preflight_ok",
        "summary": f"Resolved {len(model_map)} model(s).",
        "current_round": napi.get_current_round(),
        "models": model_map,
    }
    return result


def run_numerapi_submit() -> Dict[str, Any]:
    """Legacy single-model submission."""
    napi = build_napi()
    data_report = sync_datasets(napi)
    artifact = train_model()
    metrics = evaluate_model(path=artifact["path"])
    submission_df = build_submission_dataframe(path=artifact["path"])
    model_map = resolve_target_models(napi)
    submission_results = submit_to_models(napi, submission_df, model_map)
    success_count = sum(
        1 for v in submission_results.values() if v.get("status") == "submitted"
    )
    return {
        "ok": success_count > 0,
        "status": "numerapi_submit_ok" if success_count > 0 else "numerapi_submit_failed",
        "summary": f"Submitted to {success_count}/{len(model_map)} model(s).",
        "data_report": data_report,
        "artifact": artifact,
        "metrics": metrics,
        "submission_results": submission_results,
    }


# ---------------------------------------------------------------------------
# Full-auto mode (NEW)
# ---------------------------------------------------------------------------

def run_full_auto() -> Dict[str, Any]:
    """
    Complete autonomous pipeline:
    1. Sync data
    2. Train model suite (6 models)
    3. Evaluate all + build ensemble
    4. Compare with champion
    5. Auto-approve if thresholds pass (or use fallback)
    6. Submit
    7. Update champion if promoted
    """
    print("[full-auto] Starting autonomous pipeline...")
    result: Dict[str, Any] = {"mode": "full-auto", "steps": []}

    # Step 1: Sync data
    print("[full-auto] Step 1: Syncing datasets...")
    napi = build_napi()
    data_report = sync_datasets(napi)
    result["data_report"] = data_report
    result["steps"].append("data_sync_ok")

    # Step 2: Train model suite
    print("[full-auto] Step 2: Training model suite...")
    artifacts = train_model_suite()
    success_count = sum(1 for a in artifacts if not a.get("error"))
    result["training"] = {
        "total": len(artifacts),
        "success": success_count,
        "artifacts": artifacts,
    }
    result["steps"].append(f"training_ok ({success_count}/{len(artifacts)})")

    if success_count == 0:
        result["ok"] = False
        result["status"] = "full_auto_failed"
        result["summary"] = "All models failed to train."
        return result

    # Step 3: Evaluate all models + build ensemble
    print("[full-auto] Step 3: Evaluating models and building ensemble...")
    suite_result = evaluate_model_suite(artifacts)
    ensemble_metrics = suite_result.get("ensemble_metrics")
    result["suite_results"] = suite_result
    result["ensemble_metrics"] = ensemble_metrics
    result["steps"].append("evaluation_ok")

    # Determine which metrics to use for approval
    if ensemble_metrics:
        candidate_metrics = ensemble_metrics
        candidate_type = "ensemble"
    else:
        # Pick best individual model
        best_individual = None
        best_corr = -999
        for m in suite_result.get("models", []):
            met = m.get("metrics")
            if met and met.get("validation_correlation", 0) > best_corr:
                best_corr = met["validation_correlation"]
                best_individual = m
        if best_individual:
            candidate_metrics = best_individual["metrics"]
            candidate_type = best_individual.get("name", "individual")
        else:
            result["ok"] = False
            result["status"] = "full_auto_failed"
            result["summary"] = "No models could be evaluated."
            return result

    result["candidate_type"] = candidate_type
    result["candidate_metrics"] = candidate_metrics

    # Step 4: Compare with champion
    print("[full-auto] Step 4: Comparing with champion...")
    comparison = compare_with_champion(candidate_metrics)
    result["champion_comparison"] = comparison
    result["steps"].append(f"champion_comparison: {comparison['decision']}")

    # Step 5: Auto-approve with thresholds
    print("[full-auto] Step 5: Checking approval thresholds...")
    corr = candidate_metrics.get("validation_correlation", 0)
    sharpe = candidate_metrics.get("sharpe", 0)
    exposure = candidate_metrics.get("feature_exposure", 0)

    threshold_pass = (
        corr >= settings.MIN_VALIDATION_CORR
        and sharpe >= settings.MIN_SHARPE_RATIO
        and (exposure <= settings.MAX_FEATURE_EXPOSURE or exposure == 0)
    )

    use_fallback = False
    if not threshold_pass:
        print(f"[full-auto] Thresholds FAILED (corr={corr:.5f}, sharpe={sharpe:.4f}, exp={exposure:.4f})")
        if has_last_known_good():
            print("[full-auto] Using last-known-good fallback.")
            use_fallback = True
            result["steps"].append("thresholds_failed_using_fallback")
        else:
            result["ok"] = False
            result["status"] = "full_auto_rejected"
            result["summary"] = (
                f"Thresholds not met (corr={corr:.5f}, sharpe={sharpe:.4f}) "
                f"and no fallback available."
            )
            result["steps"].append("thresholds_failed_no_fallback")
            return result
    else:
        result["steps"].append("thresholds_passed")

    # Step 6: Submit
    print("[full-auto] Step 6: Submitting predictions...")
    if use_fallback:
        submission_df = build_submission_dataframe(path=str(last_known_good_path()))
    elif candidate_type == "ensemble":
        submission_df = build_ensemble_submission()
    else:
        # Find the best individual model's path
        best_path = None
        for a in artifacts:
            if a.get("name") == candidate_type and not a.get("error"):
                best_path = a["path"]
                break
        if not best_path:
            best_path = artifacts[0]["path"]
        submission_df = build_submission_dataframe(path=best_path)

    model_map = resolve_target_models(napi)
    submission_results = submit_to_models(napi, submission_df, model_map)
    ok_submissions = sum(
        1 for v in submission_results.values() if v.get("status") == "submitted"
    )
    result["submission_results"] = submission_results
    result["steps"].append(f"submitted ({ok_submissions}/{len(model_map)})")

    # Step 7: Post-submission housekeeping
    if ok_submissions > 0 and not use_fallback:
        # Save last-known-good
        if candidate_type == "ensemble":
            # Save first model as representative for fallback
            for a in artifacts:
                if not a.get("error") and a.get("path"):
                    save_last_known_good(a["path"])
                    break
        else:
            for a in artifacts:
                if a.get("name") == candidate_type and a.get("path"):
                    save_last_known_good(a["path"])
                    break

        # Update champion if promoted
        if comparison.get("decision") == "PROMOTE":
            save_champion(candidate_metrics)
            result["steps"].append("champion_promoted")

    result["ok"] = ok_submissions > 0
    result["status"] = "full_auto_ok" if ok_submissions > 0 else "full_auto_submit_failed"
    result["summary"] = (
        f"{'Ensemble' if candidate_type == 'ensemble' else candidate_type} "
        f"{'(fallback)' if use_fallback else ''} "
        f"submitted to {ok_submissions}/{len(model_map)} model(s). "
        f"corr={corr:.5f}, sharpe={sharpe:.4f}."
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_mode(mode: str, args: argparse.Namespace | None = None) -> Dict[str, Any]:
    args = args or argparse.Namespace()
    if mode == "agent-prepare":
        return AgenticControlPlane().prepare(
            target_model=getattr(args, "target_model", None),
            artifact_path=getattr(args, "artifact_path", None),
        )
    if mode == "agent-approve":
        return AgenticControlPlane().approve(
            run_id=getattr(args, "run_id", None),
            challenge=getattr(args, "challenge", None),
            actor=getattr(args, "actor", None) or "human-operator",
        )
    if mode == "agent-submit":
        return AgenticControlPlane().submit(run_id=getattr(args, "run_id", None))
    if mode == "portfolio-prepare":
        return PortfolioControlPlane().prepare_all()
    if mode == "portfolio-status":
        return PortfolioControlPlane().inspect()
    if mode == "portfolio-propose":
        assignments = json.loads(Path(getattr(args, "assignments_path", "")).read_text())
        proposal, proposal_path = create_portfolio_proposal(
            settings.MODEL_REGISTRY_DIR,
            assignments,
        )
        return {
            "ok": True,
            "status": "AWAITING_HUMAN_PORTFOLIO_APPROVAL",
            "proposal_id": proposal["proposal_id"],
            "proposal_path": str(proposal_path),
            "approval_challenge": proposal["approval_challenge"],
            "approval_expires_at": proposal["approval_expires_at"],
            "assignments": proposal["assignments"],
        }
    if mode == "portfolio-approve":
        approval_path = approve_portfolio_proposal(
            getattr(args, "portfolio_proposal_path", None),
            challenge=getattr(args, "challenge", None),
            actor=getattr(args, "actor", None) or "human-operator",
        )
        return {
            "ok": True,
            "status": "PORTFOLIO_APPROVED",
            "approval_path": str(approval_path),
        }
    if mode == "portfolio-activate":
        current_path = activate_portfolio(
            settings.MODEL_REGISTRY_DIR,
            getattr(args, "portfolio_proposal_path", None),
        )
        return {
            "ok": True,
            "status": "PORTFOLIO_ACTIVATED",
            "current_path": str(current_path),
            "summary": "Portfolio assignments activated. No submission or stake change occurred.",
        }
    if mode == "stake-status":
        return StakeControlPlane().inspect()
    if mode == "stake-propose":
        return StakeControlPlane().propose(
            target_model=getattr(args, "target_model", None),
            action=getattr(args, "stake_action", None),
            amount_nmr=getattr(args, "amount_nmr", None),
            rationale=(
                getattr(args, "rationale", None)
                or "Operator-requested governed stake adjustment."
            ),
            deployment_round=getattr(args, "deployment_round", None),
        )
    if mode == "stake-approve":
        return StakeControlPlane().approve(
            proposal_path=getattr(args, "stake_proposal_path", None),
            challenge=getattr(args, "challenge", None),
            actor=getattr(args, "actor", None) or "human-operator",
        )
    if mode == "stake-execute":
        return StakeControlPlane().execute(
            proposal_path=getattr(args, "stake_proposal_path", None),
            confirmation=getattr(args, "confirmation", None),
        )
    if mode == "competition-status":
        return CompetitionTracker().snapshot()
    if mode == "platform-status":
        return PlatformHealthMonitor().snapshot()
    if mode == "system-health":
        return SystemHealthMonitor().snapshot()
    if mode == "alert-dispatch":
        return AlertDispatcher(
            Path(settings.CONTROL_PLANE_DIR),
            REPORTS_DIR,
        ).dispatch()
    if mode == "score-listen":
        return PerformanceListener(
            Path(settings.CONTROL_PLANE_DIR),
        ).poll()
    if mode == "research-evaluate":
        artifact_path = (
            getattr(args, "artifact_path", None)
            or str(AgenticControlPlane._default_artifact())
        )
        model_name = getattr(args, "model_name", None) or Path(artifact_path).stem
        packet = evaluate_artifact_robustness(
            artifact_path,
            model_name=model_name,
        )
        recommendation = promotion_recommendation(packet, champion=None)
        artifacts = write_evaluation_artifacts(
            Path(settings.CONTROL_PLANE_DIR) / "research",
            packet=packet,
            recommendation=recommendation,
        )
        return {
            "ok": True,
            "status": f"RESEARCH_{recommendation['decision']}",
            "summary": (
                f"{model_name}: {recommendation['decision']} "
                f"({', '.join(recommendation['failures']) or 'all checks passed'})."
            ),
            "evaluation": packet,
            "recommendation": recommendation,
            "artifacts": artifacts,
        }
    if mode == "model-approve":
        approval_path = approve_candidate_bundle(
            getattr(args, "manifest_path", None),
            challenge=getattr(args, "challenge", None),
            actor=getattr(args, "actor", None) or "human-operator",
        )
        return {
            "ok": True,
            "status": "MODEL_PROMOTION_APPROVED",
            "approval_path": str(approval_path),
        }
    if mode == "model-promote":
        current_path = promote_candidate_bundle(
            settings.MODEL_REGISTRY_DIR,
            getattr(args, "manifest_path", None),
        )
        return {
            "ok": True,
            "status": "MODEL_PROMOTED",
            "current_manifest": str(current_path),
            "summary": "Champion pointer updated. No Numerai submission or stake change occurred.",
        }
    if mode == "mcp-preflight":
        return run_mcp_preflight()
    if mode == "mcp-dry-run":
        return run_mcp_cycle(approve_submission=False)
    if mode == "mcp-submit":
        return {
            "ok": False,
            "status": "unsafe_legacy_mode_disabled",
            "summary": "Direct MCP submission is disabled. Use agent-prepare, agent-approve, then agent-submit.",
        }
    if mode == "mcp-auto":
        result = run_mcp_cycle(approve_submission=False)
        result["auto_submit_enabled"] = False
        result["summary"] = (
            "MCP dry-run executed. File-flag submission is permanently disabled; "
            "use the durable human approval workflow."
        )
        result["status"] = "mcp_auto_dry_run_ok"
        return result
    if mode == "numerapi-preflight":
        return run_numerapi_preflight()
    if mode == "numerapi-submit":
        return {
            "ok": False,
            "status": "unsafe_legacy_mode_disabled",
            "summary": "Direct NumerAPI submission is disabled. Use agent-prepare, agent-approve, then agent-submit.",
        }
    if mode == "full-auto":
        return {
            "ok": False,
            "status": "unsafe_legacy_mode_disabled",
            "summary": "Ungoverned full-auto submission is disabled. Use agent-prepare, agent-approve, then agent-submit.",
        }
    raise ValueError(f"Unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily Numerai automation runner."
    )
    parser.add_argument(
        "--mode",
        default="agent-prepare",
        choices=[
            "agent-prepare",
            "agent-approve",
            "agent-submit",
            "portfolio-prepare",
            "portfolio-status",
            "portfolio-propose",
            "portfolio-approve",
            "portfolio-activate",
            "stake-status",
            "stake-propose",
            "stake-approve",
            "stake-execute",
            "competition-status",
            "platform-status",
            "system-health",
            "alert-dispatch",
            "score-listen",
            "research-evaluate",
            "model-approve",
            "model-promote",
            "mcp-preflight",
            "mcp-dry-run",
            "mcp-submit",
            "mcp-auto",
            "numerapi-preflight",
            "numerapi-submit",
            "full-auto",
        ],
        help="Execution mode (default: agent-prepare).",
    )
    parser.add_argument("--target-model", help="Single Numerai model name for agent-prepare.")
    parser.add_argument("--artifact-path", help="Approved pickle artifact for agent-prepare.")
    parser.add_argument("--model-name", help="Model label for research artifacts.")
    parser.add_argument("--manifest-path", help="Frozen candidate manifest for promotion.")
    parser.add_argument("--assignments-path", help="JSON list of proposed portfolio assignments.")
    parser.add_argument(
        "--portfolio-proposal-path",
        help="Frozen portfolio proposal for approval or activation.",
    )
    parser.add_argument("--run-id", help="Readiness run id for approval or submission.")
    parser.add_argument("--challenge", help="Human approval challenge from readiness packet.")
    parser.add_argument("--actor", help="Human operator identity recorded in the audit log.")
    parser.add_argument(
        "--stake-proposal-path",
        help="Frozen stake proposal for stake-approve or stake-execute.",
    )
    parser.add_argument(
        "--stake-action",
        choices=["increase", "decrease"],
        help="Stake proposal action.",
    )
    parser.add_argument("--amount-nmr", type=float, help="Stake change amount in NMR.")
    parser.add_argument(
        "--deployment-round",
        type=int,
        help="Verified deployment round for a proposed stake increase.",
    )
    parser.add_argument("--rationale", help="Reason recorded in a stake proposal.")
    parser.add_argument(
        "--confirmation",
        help="Exact stake execution confirmation emitted by stake-propose.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if run result is not ok.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload: Dict[str, Any]
    try:
        payload = run_mode(args.mode, args)
    except (NumeraiMCPError, NumeraiOpsError, SubmissionGuardError, Exception) as exc:
        payload = {
            "ok": False,
            "status": "failed",
            "summary": str(exc),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }

    json_path, md_path = _write_reports(args.mode, payload)
    print(
        json.dumps(
            {
                "ok": payload.get("ok", False),
                "status": payload.get("status"),
                "summary": payload.get("summary", ""),
                "report_json": str(json_path),
                "report_md": str(md_path),
            },
            indent=2,
        )
    )

    if args.strict and not payload.get("ok", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
