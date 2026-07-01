from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .config import settings
from .numerai_mcp_client import (
    MCPToolInfo,
    NumeraiMCPClient,
    NumeraiMCPError,
    infer_tool_suggestions,
    tool_info_to_dict,
)


@dataclass
class MCPWorkflowToolMap:
    current_round: Optional[str]
    sync_data: Optional[str]
    train_model: Optional[str]
    evaluate_model: Optional[str]
    submit_predictions: Optional[str]

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "current_round": self.current_round,
            "sync_data": self.sync_data,
            "train_model": self.train_model,
            "evaluate_model": self.evaluate_model,
            "submit_predictions": self.submit_predictions,
        }


class NumeraiMCPWorkflowRunner:
    def __init__(self, client: Optional[NumeraiMCPClient] = None):
        self.client = client or NumeraiMCPClient()

    def _resolve_tool_map(self, suggestions: Dict[str, Optional[str]]) -> MCPWorkflowToolMap:
        return MCPWorkflowToolMap(
            current_round=settings.NUMERAI_MCP_TOOL_CURRENT_ROUND or suggestions.get("current_round"),
            sync_data=settings.NUMERAI_MCP_TOOL_SYNC_DATA or suggestions.get("sync_data"),
            train_model=settings.NUMERAI_MCP_TOOL_TRAIN_MODEL or suggestions.get("train_model"),
            evaluate_model=settings.NUMERAI_MCP_TOOL_EVALUATE_MODEL or suggestions.get("evaluate_model"),
            submit_predictions=settings.NUMERAI_MCP_TOOL_SUBMIT_PREDICTIONS
            or suggestions.get("submit_predictions"),
        )

    async def preflight(self) -> Dict[str, Any]:
        tools = await self.client.list_tools()
        suggestions = infer_tool_suggestions(tools)
        selected = self._resolve_tool_map(suggestions)

        selected_dict = selected.as_dict()
        required = ("current_round", "sync_data", "train_model", "evaluate_model")
        missing = [stage for stage in required if not selected_dict.get(stage)]
        return {
            "ok": not missing,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "mcp_url": self.client.url,
            "tool_count": len(tools),
            "tools": [tool_info_to_dict(t) for t in tools],
            "suggested_tools": suggestions,
            "selected_tools": selected_dict,
            "missing_required_stages": missing,
        }

    async def run_cycle(
        self,
        approve_submission: bool,
        args: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        args = args or {}
        preflight = await self.preflight()
        if not preflight["ok"]:
            missing = ", ".join(preflight["missing_required_stages"])
            raise NumeraiMCPError(f"MCP workflow is incomplete; missing required stages: {missing}")
        selected = preflight["selected_tools"]
        tool_names = {tool["name"] for tool in preflight["tools"]}

        events = []
        outputs: Dict[str, Any] = {}

        async def run_stage(stage: str, tool_name: Optional[str]):
            if not tool_name:
                events.append(f"skip:{stage}:tool-not-configured")
                return
            if tool_name not in tool_names:
                raise NumeraiMCPError(
                    f"Configured tool for stage '{stage}' not found: {tool_name}"
                )
            stage_args = args.get(stage) or {}
            events.append(f"start:{stage}:{tool_name}")
            outputs[stage] = await self.client.call_tool(tool_name, stage_args)
            events.append(f"done:{stage}:{tool_name}")

        await run_stage("current_round", selected.get("current_round"))
        await run_stage("sync_data", selected.get("sync_data"))
        await run_stage("train_model", selected.get("train_model"))
        await run_stage("evaluate_model", selected.get("evaluate_model"))

        if approve_submission:
            await run_stage("submit_predictions", selected.get("submit_predictions"))
        else:
            events.append("skip:submit_predictions:approval-required")

        return {
            "ok": True,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "approved": approve_submission,
            "events": events,
            "outputs": outputs,
            "selected_tools": selected,
        }
