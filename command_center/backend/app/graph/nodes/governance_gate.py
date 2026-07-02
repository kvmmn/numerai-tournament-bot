from ..state import AgentState, WorkflowLog, DecisionContext, DecisionOption
from datetime import datetime
from ...core.config import settings


def _read(candidate, key: str, default=0.0):
    if candidate is None:
        return default
    if isinstance(candidate, dict):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _get_metrics(candidate):
    """Extract metrics dict from candidate (handles both dict and object)."""
    if candidate is None:
        return {}
    metrics = _read(candidate, "metrics", {})
    if metrics is None:
        return {}
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()
    if hasattr(metrics, "dict"):
        return metrics.dict()
    if isinstance(metrics, dict):
        return metrics
    return {}


def _passes_thresholds(metrics: dict) -> tuple[bool, str]:
    """Check if metrics pass auto-approval thresholds."""
    corr = metrics.get("validation_correlation", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    exposure = metrics.get("feature_exposure", 0.0)

    reasons = []
    if corr < settings.MIN_VALIDATION_CORR:
        reasons.append(f"corr {corr:.5f} < {settings.MIN_VALIDATION_CORR}")
    if sharpe < settings.MIN_SHARPE_RATIO:
        reasons.append(f"sharpe {sharpe:.4f} < {settings.MIN_SHARPE_RATIO}")
    if exposure > settings.MAX_FEATURE_EXPOSURE and exposure > 0:
        reasons.append(f"exposure {exposure:.4f} > {settings.MAX_FEATURE_EXPOSURE}")

    if reasons:
        return False, "Thresholds not met: " + "; ".join(reasons)
    return True, "All thresholds passed."


def governance_gate(state: AgentState) -> dict:
    """
    Decides whether to approve submission.

    Three modes:
    1. AUTO_APPROVE_SUBMISSION=True: Auto-approve if thresholds pass, fallback if not
    2. Manual: Halt and wait for human approval with decision context
    """
    logs = []

    best_candidate = state.get("best_candidate")
    candidate_id = _read(best_candidate, "model_id", "Unknown")
    metrics = _get_metrics(best_candidate)
    champion_comparison = state.get("champion_comparison", {})

    # AUTO_APPROVE_SUBMISSION is retained only for backwards-compatible
    # configuration parsing. Governance is fail-closed: metrics can recommend,
    # but never create the human approval required by the submission executor.
    if settings.AUTO_APPROVE_SUBMISSION:
        logs.append(
            WorkflowLog(
                step="Governance Gate",
                message="AUTO_APPROVE_SUBMISSION was ignored; explicit human approval is mandatory.",
                timestamp=datetime.now().isoformat(),
            )
        )

    # Mode 2: Manual approval - halt and present decision context
    logs.append(
        WorkflowLog(
            step="Governance Gate",
            message=f"Halting for approval of candidate {candidate_id}.",
            timestamp=datetime.now().isoformat(),
        )
    )

    corr = metrics.get("validation_correlation", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    exposure = metrics.get("feature_exposure", 0.0)
    drawdown = metrics.get("max_drawdown", 0.0)

    passes, threshold_msg = _passes_thresholds(metrics)
    comparison_msg = ""
    if champion_comparison:
        comparison_msg = f" Champion: {champion_comparison.get('decision', '?')} - {champion_comparison.get('reason', '')}"

    context = DecisionContext(
        title="Promote Candidate Model?",
        description=(
            f"Model: {candidate_id} | "
            f"Corr={corr:.5f}, Sharpe={sharpe:.4f}, "
            f"Exposure={exposure:.4f}, Drawdown={drawdown:.4f}. "
            f"Thresholds: {'PASS' if passes else 'FAIL'} ({threshold_msg})."
            f"{comparison_msg}"
        ),
        timestamp=datetime.now().isoformat(),
        options=[
            DecisionOption(label="Approve & Submit", value="APPROVED", style="success", description="Proceed to submission."),
            DecisionOption(label="Reject", value="REJECTED", style="danger", description="Discard this candidate."),
            DecisionOption(label="Retry Training", value="RETRY_TRAINING", style="secondary", description="Discard and retrain."),
        ],
    )

    return {
        "status": "AWAITING_APPROVAL",
        "decision_context": context,
        "audit_log": logs,
    }
