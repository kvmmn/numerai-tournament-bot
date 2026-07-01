from ..state import AgentState, WorkflowLog
from datetime import datetime
from ...core.config import settings
from ...core.numerai_ops import (
    NumeraiOpsError,
    build_napi,
    build_submission_dataframe,
    build_ensemble_submission,
    resolve_target_models,
    submit_to_models,
    save_last_known_good,
    save_champion,
)


def _candidate_path(state: AgentState):
    candidate = state.get("best_candidate")
    if not candidate:
        return None
    if isinstance(candidate, dict):
        return candidate.get("path")
    return getattr(candidate, "path", None)


def _candidate_id(state: AgentState) -> str:
    candidate = state.get("best_candidate")
    if not candidate:
        return "unknown"
    if isinstance(candidate, dict):
        return candidate.get("model_id", "unknown")
    return getattr(candidate, "model_id", "unknown")


def _candidate_name(state: AgentState) -> str:
    candidate = state.get("best_candidate")
    if not candidate:
        return ""
    if isinstance(candidate, dict):
        return candidate.get("name", "")
    return getattr(candidate, "name", "")


def submission_agent(state: AgentState) -> dict:
    """
    Submits the approved predictions to Numerai.

    Supports:
    - Ensemble submission (if candidate is ensemble)
    - Single model submission
    - Saves last-known-good and champion after successful submission
    """
    logs = []
    approval_status = state.get("human_approval_status")

    if approval_status != "APPROVED":
        return {"error": "Submission attempted without approval", "status": "FAILED"}

    try:
        candidate_id = _candidate_id(state)
        candidate_name = _candidate_name(state)
        is_ensemble = candidate_name == "ensemble" or candidate_id == "ensemble"

        logs.append(
            WorkflowLog(
                step="Submission Agent",
                message=f"Preparing {'ensemble' if is_ensemble else 'single model'} predictions.",
                timestamp=datetime.now().isoformat(),
            )
        )

        # Build submission dataframe
        if is_ensemble and settings.USE_ENSEMBLE:
            submission_df = build_ensemble_submission()
        else:
            submission_df = build_submission_dataframe(path=_candidate_path(state))

        napi = build_napi()
        model_map = resolve_target_models(napi)

        logs.append(
            WorkflowLog(
                step="Submission Agent",
                message=f"Submitting to {len(model_map)} model(s): {', '.join(model_map.keys())}",
                timestamp=datetime.now().isoformat(),
            )
        )

        submission_results = submit_to_models(napi, submission_df, model_map)
        ok_count = sum(
            1 for result in submission_results.values() if result.get("status") == "submitted"
        )
        status = "SUBMITTED" if ok_count > 0 else "FAILED"

        # Post-submission: save last-known-good and update champion
        if status == "SUBMITTED":
            path = _candidate_path(state)
            if path:
                try:
                    save_last_known_good(path)
                    logs.append(
                        WorkflowLog(
                            step="Submission Agent",
                            message="Saved as last-known-good model.",
                            timestamp=datetime.now().isoformat(),
                        )
                    )
                except Exception as exc:
                    logs.append(
                        WorkflowLog(
                            step="Submission Agent",
                            message=f"Warning: failed to save last-known-good: {exc}",
                            timestamp=datetime.now().isoformat(),
                        )
                    )

            # Update champion if comparison said PROMOTE
            champion_comparison = state.get("champion_comparison", {})
            if champion_comparison and champion_comparison.get("decision") == "PROMOTE":
                challenger_metrics = champion_comparison.get("challenger_metrics", {})
                try:
                    save_champion(challenger_metrics, path)
                    logs.append(
                        WorkflowLog(
                            step="Submission Agent",
                            message="Champion updated with new model.",
                            timestamp=datetime.now().isoformat(),
                        )
                    )
                except Exception as exc:
                    logs.append(
                        WorkflowLog(
                            step="Submission Agent",
                            message=f"Warning: failed to update champion: {exc}",
                            timestamp=datetime.now().isoformat(),
                        )
                    )

        logs.append(
            WorkflowLog(
                step="Submission Agent",
                message=f"Submission finished. success={ok_count}, total={len(submission_results)}",
                timestamp=datetime.now().isoformat(),
            )
        )

        return {
            "status": status,
            "submission_results": submission_results,
            "audit_log": logs,
        }

    except NumeraiOpsError as exc:
        logs.append(
            WorkflowLog(
                step="Submission Agent",
                message=f"Submission failed: {exc}",
                timestamp=datetime.now().isoformat(),
            )
        )
        return {"status": "FAILED", "error": str(exc), "audit_log": logs}
