import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from ..graph.workflow import app as graph_app
from ..graph.state import AgentState
from ..core.config import settings, BACKEND_ROOT
from ..core.numerai_ops import (
    NumeraiOpsError,
    build_napi,
    resolve_target_models,
    load_champion_metrics,
    champion_metrics_path,
)
from ..core.numerai_mcp_client import (
    NumeraiMCPClient,
    NumeraiMCPError,
    infer_tool_suggestions,
    tool_info_to_dict,
)
from ..core.numerai_mcp_workflow import NumeraiMCPWorkflowRunner

router = APIRouter()

REPORTS_DIR = BACKEND_ROOT / "automation" / "reports"

# --- Request/Response Models ---


class GraphInput(BaseModel):
    command: str = "START"  # START, APPROVE, REJECT


class ApprovalRequest(BaseModel):
    decision: str  # APPROVED, REJECTED


class MCPCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = {}


class MCPRunRequest(BaseModel):
    approve_submission: bool = False
    args: Dict[str, Dict[str, Any]] = {}


class ConfigUpdate(BaseModel):
    key: str
    value: str


# --- MCP Routes ---


@router.get("/mcp/tools")
async def mcp_list_tools():
    try:
        client = NumeraiMCPClient()
        tools = await client.list_tools()
        return {
            "ok": True,
            "tool_count": len(tools),
            "tools": [tool_info_to_dict(t) for t in tools],
            "suggested_tools": infer_tool_suggestions(tools),
        }
    except NumeraiMCPError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mcp/preflight")
async def mcp_preflight():
    try:
        runner = NumeraiMCPWorkflowRunner()
        return await runner.preflight()
    except NumeraiMCPError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mcp/call")
async def mcp_call_tool(req: MCPCallRequest):
    privileged_terms = (
        "submit",
        "upload",
        "stake",
        "create",
        "delete",
        "assign",
        "trigger",
        "mutation",
    )
    if any(term in req.tool_name.casefold() for term in privileged_terms):
        raise HTTPException(
            status_code=403,
            detail="Privileged MCP mutations are disabled on the generic API route.",
        )
    try:
        client = NumeraiMCPClient()
        result = await client.call_tool(req.tool_name, req.arguments)
        return {"ok": True, "tool_name": req.tool_name, "result": result}
    except NumeraiMCPError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mcp/run")
async def mcp_run_cycle(req: MCPRunRequest):
    if req.approve_submission:
        raise HTTPException(
            status_code=403,
            detail="MCP submission is disabled on the generic API route.",
        )
    try:
        runner = NumeraiMCPWorkflowRunner()
        return await runner.run_cycle(
            approve_submission=req.approve_submission,
            args=req.args,
        )
    except NumeraiMCPError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Numerai Routes ---


@router.get("/numerai/preflight")
def numerai_preflight():
    try:
        napi = build_napi()
        models = resolve_target_models(napi)
        return {
            "ok": True,
            "target_models": models,
            "target_model_count": len(models),
            "current_round": napi.get_current_round(),
        }
    except NumeraiOpsError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Graph OS Routes ---


@router.get("/os/state")
def get_os_state():
    try:
        config = {"configurable": {"thread_id": "main_thread"}}
        current_state = graph_app.get_state(config)

        if not current_state.values:
            return {"status": "OFFLINE", "visual_graph": None}

        state_values = current_state.values
        next_step = current_state.next

        return {
            "values": state_values,
            "next": next_step,
            "visual_graph": None,
        }
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}


@router.post("/os/start")
def start_os_cycle(background_tasks: BackgroundTasks):
    def run_graph():
        config = {"configurable": {"thread_id": "main_thread"}}
        initial_input = {"status": "IDLE", "human_approval_status": "PENDING"}
        for output in graph_app.stream(initial_input, config=config):
            print(f"[Graph] Output: {output}")

    background_tasks.add_task(run_graph)
    return {"message": "OS Cycle Started"}


@router.post("/os/approve")
def approve_gate(req: ApprovalRequest, background_tasks: BackgroundTasks):
    raise HTTPException(
        status_code=410,
        detail=(
            "Legacy graph approval is disabled. Use the challenge-bound "
            "agent-approve workflow."
        ),
    )


@router.post("/os/reset")
def reset_os_state():
    config = {"configurable": {"thread_id": "main_thread"}}
    graph_app.update_state(
        config,
        {
            "status": "IDLE",
            "audit_log": [],
            "experiment_queue": [],
            "models": [],
            "human_approval_status": "PENDING",
        },
    )
    return {"message": "State reset to IDLE."}


# --- Dashboard Routes (NEW) ---


@router.get("/models/comparison")
def get_model_comparison():
    """Return metrics for all models from the latest suite evaluation."""
    try:
        config = {"configurable": {"thread_id": "main_thread"}}
        current_state = graph_app.get_state(config)

        if not current_state.values:
            return {"ok": True, "models": [], "ensemble_metrics": None, "champion_metrics": None}

        state_values = current_state.values
        suite_results = state_values.get("suite_results", {})
        ensemble_metrics = state_values.get("ensemble_metrics")
        champion_metrics = load_champion_metrics()

        models = []
        if suite_results and suite_results.get("models"):
            for m in suite_results["models"]:
                models.append({
                    "name": m.get("name", m.get("model_id", "?")),
                    "model_type": m.get("model_type", "?"),
                    "metrics": m.get("metrics"),
                    "error": m.get("error"),
                })

        return {
            "ok": True,
            "models": models,
            "ensemble_metrics": ensemble_metrics,
            "champion_metrics": champion_metrics,
            "model_names_in_ensemble": suite_results.get("model_names_in_ensemble", []),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/history")
def get_submission_history():
    """Return historical submission reports."""
    try:
        reports = []
        if REPORTS_DIR.exists():
            json_files = sorted(REPORTS_DIR.glob("*.json"), reverse=True)[:20]
            for jf in json_files:
                try:
                    data = json.loads(jf.read_text())
                    reports.append({
                        "filename": jf.name,
                        "timestamp": jf.stem.split("_")[0] + "_" + jf.stem.split("_")[1]
                        if "_" in jf.stem
                        else jf.stem,
                        "ok": data.get("ok", False),
                        "status": data.get("status", "unknown"),
                        "summary": data.get("summary", ""),
                        "ensemble_metrics": data.get("ensemble_metrics"),
                        "candidate_metrics": data.get("candidate_metrics"),
                        "champion_comparison": data.get("champion_comparison"),
                    })
                except Exception:
                    pass

        return {"ok": True, "reports": reports, "total": len(reports)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/config")
def get_config():
    """Return current configuration (non-sensitive values)."""
    return {
        "ok": True,
        "config": {
            "DATA_VERSION": settings.DATA_VERSION,
            "FEATURE_SET": settings.FEATURE_SET,
            "TRAIN_ERA_STRIDE": settings.TRAIN_ERA_STRIDE,
            "AUTO_APPROVE_SUBMISSION": settings.AUTO_APPROVE_SUBMISSION,
            "USE_ENSEMBLE": settings.USE_ENSEMBLE,
            "MIN_VALIDATION_CORR": settings.MIN_VALIDATION_CORR,
            "MIN_SHARPE_RATIO": settings.MIN_SHARPE_RATIO,
            "MAX_FEATURE_EXPOSURE": settings.MAX_FEATURE_EXPOSURE,
            "MODEL_NAME": settings.MODEL_NAME,
            "NUMERAI_MODEL_NAMES": settings.NUMERAI_MODEL_NAMES,
            "has_credentials": bool(settings.NUMERAI_PUBLIC_ID and settings.NUMERAI_SECRET_KEY),
        },
    }


@router.get("/champion")
def get_champion():
    """Return current champion model metrics."""
    try:
        metrics = load_champion_metrics()
        return {
            "ok": True,
            "has_champion": metrics is not None,
            "metrics": metrics,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
