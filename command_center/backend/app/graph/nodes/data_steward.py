from datetime import datetime
from ..state import AgentState, WorkflowLog
from ...core.numerai_ops import NumeraiOpsError, build_napi, sync_datasets

def data_steward(state: AgentState) -> dict:
    """
    Ensures required datasets exist and refreshes live data.
    """
    logs = []

    try:
        napi = build_napi()
        report = sync_datasets(napi)
        msg = (
            "Data synchronized. "
            f"found={len(report['found'])}, downloaded={len(report['downloaded'])}, refreshed={len(report['refreshed'])}"
        )
        logs.append(
            WorkflowLog(
                step="Data Steward",
                message=msg,
                timestamp=datetime.now().isoformat(),
            )
        )
        return {
            "status": "DATA_VALIDATED",
            "audit_log": logs,
            "data_health_report": report,
        }
    except NumeraiOpsError as exc:
        logs.append(
            WorkflowLog(
                step="Data Steward",
                message=f"Data synchronization failed: {exc}",
                timestamp=datetime.now().isoformat(),
            )
        )
        return {"status": "FAILED", "error": str(exc), "audit_log": logs}
