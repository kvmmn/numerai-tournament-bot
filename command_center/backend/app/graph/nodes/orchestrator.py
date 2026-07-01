from ..state import AgentState, WorkflowLog
from datetime import datetime

def orchestrator(state: AgentState) -> dict:
    """
    The Central Brain.
    Decides the next step based on the current context.
    """
    current_status = state.get("status", "IDLE")
    logs = []
    updates = {}

    # Logic: State Transition Rules
    if current_status == "IDLE":
        # Start of cycle -> Check Data
        updates["status"] = "CHECKING_DATA"
        logs.append(WorkflowLog(step="Orchestrator", message="New cycle started. directing to Data Steward.", timestamp=datetime.now().isoformat()))

    elif current_status == "DATA_VALIDATED":
        # Data is good -> Start Training/Experimentation
        updates["status"] = "TRAINING"
        logs.append(WorkflowLog(step="Orchestrator", message="Data valid. Directing to Model Factory.", timestamp=datetime.now().isoformat()))

    elif current_status == "MODELS_TRAINED":
        # Models ready -> Evaluate them
        updates["status"] = "EVALUATING"
        logs.append(WorkflowLog(step="Orchestrator", message="Models trained. Directing to Evaluation Judge.", timestamp=datetime.now().isoformat()))

    elif current_status == "CANDIDATE_SELECTED":
        # We have a winner -> Ask for Human Approval
        updates["status"] = "AWAITING_APPROVAL"
        logs.append(WorkflowLog(step="Orchestrator", message="Candidate selected. Requesting Governance Approval.", timestamp=datetime.now().isoformat()))

    elif current_status == "APPROVED":
        # Human said YES -> Submit
        updates["status"] = "SUBMITTING"
        logs.append(WorkflowLog(step="Orchestrator", message="Approved. Directing to Submission Agent.", timestamp=datetime.now().isoformat()))

    elif current_status == "SUBMITTED":
        # Done -> Go back to Monitoring/Idle
        updates["status"] = "IDLE"
        logs.append(WorkflowLog(step="Orchestrator", message="Cycle complete. System IDLE.", timestamp=datetime.now().isoformat()))

    elif current_status == "RETRY_TRAINING":
        updates["status"] = "TRAINING"
        logs.append(
            WorkflowLog(
                step="Orchestrator",
                message="Retry requested. Directing to Model Factory.",
                timestamp=datetime.now().isoformat(),
            )
        )

    elif current_status == "REJECTED":
        updates["status"] = "IDLE"
        logs.append(
            WorkflowLog(
                step="Orchestrator",
                message="Candidate rejected. Returning to IDLE.",
                timestamp=datetime.now().isoformat(),
            )
        )

    elif current_status == "FAILED":
        updates["status"] = "IDLE"
        logs.append(
            WorkflowLog(
                step="Orchestrator",
                message="Failure detected. Returning to IDLE for inspection.",
                timestamp=datetime.now().isoformat(),
            )
        )

    return {"status": updates.get("status", current_status), "audit_log": logs}
