from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from .config import settings

MCP_IMPORT_ERROR: Optional[str] = None

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
except Exception as exc:  # pragma: no cover - import guard
    ClientSession = None
    streamable_http_client = None
    MCP_IMPORT_ERROR = str(exc)

try:
    from mcp.client.sse import sse_client
except Exception:
    sse_client = None


class NumeraiMCPError(RuntimeError):
    pass


@dataclass
class MCPToolInfo:
    name: str
    description: str
    input_schema: Dict[str, Any]


def _tool_to_dict(tool: Any) -> MCPToolInfo:
    if isinstance(tool, dict):
        return MCPToolInfo(
            name=tool.get("name", ""),
            description=tool.get("description", ""),
            input_schema=tool.get("inputSchema") or tool.get("input_schema") or {},
        )
    return MCPToolInfo(
        name=getattr(tool, "name", ""),
        description=getattr(tool, "description", ""),
        input_schema=getattr(tool, "inputSchema", {}) or getattr(tool, "input_schema", {}) or {},
    )


def _result_to_jsonable(result: Any) -> Dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()

    payload: Dict[str, Any] = {}

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        payload["structured_content"] = structured

    text_chunks: List[str] = []
    contents = getattr(result, "content", None)
    if isinstance(contents, list):
        for part in contents:
            if isinstance(part, dict):
                txt = part.get("text")
            else:
                txt = getattr(part, "text", None)
            if txt:
                text_chunks.append(txt)
    if text_chunks:
        payload["text"] = "\n".join(text_chunks)

    raw = repr(result)
    if not payload:
        payload["raw"] = raw
    else:
        payload["raw"] = raw
    return payload


class NumeraiMCPClient:
    def __init__(
        self,
        url: Optional[str] = None,
        auth_header: Optional[str] = None,
        use_sse: Optional[bool] = None,
    ):
        self.url = url or settings.NUMERAI_MCP_URL
        self.auth_header = auth_header or settings.NUMERAI_MCP_AUTH
        self.use_sse = settings.NUMERAI_MCP_USE_SSE if use_sse is None else use_sse

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.auth_header:
            headers["Authorization"] = self.auth_header
        return headers

    async def _with_session(self, handler):
        if MCP_IMPORT_ERROR or ClientSession is None or streamable_http_client is None:
            raise NumeraiMCPError(
                "MCP Python SDK is not installed. Install dependency `mcp` first."
            )

        headers = self._headers()
        timeout = httpx.Timeout(60.0, connect=20.0)

        # Prefer SSE when URL points to /sse and SDK transport is available.
        if self.use_sse and sse_client is not None and self.url.endswith("/sse"):
            async with sse_client(self.url, headers=headers) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await handler(session)

        async with httpx.AsyncClient(headers=headers, timeout=timeout) as http_client:
            async with streamable_http_client(
                self.url,
                http_client=http_client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await handler(session)

    async def list_tools(self) -> List[MCPToolInfo]:
        async def _handler(session):
            response = await session.list_tools()
            tools = getattr(response, "tools", response)
            return [_tool_to_dict(t) for t in tools]

        return await self._with_session(_handler)

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        args = arguments or {}

        async def _handler(session):
            result = await session.call_tool(name, args)
            return _result_to_jsonable(result)

        return await self._with_session(_handler)

    def list_tools_sync(self) -> List[MCPToolInfo]:
        return asyncio.run(self.list_tools())

    def call_tool_sync(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return asyncio.run(self.call_tool(name, arguments))


def infer_tool_suggestions(tools: List[MCPToolInfo]) -> Dict[str, Optional[str]]:
    def pick(*keywords: str) -> Optional[str]:
        for tool in tools:
            n = tool.name.lower()
            if all(k in n for k in keywords):
                return tool.name
        for tool in tools:
            n = tool.name.lower()
            if any(k in n for k in keywords):
                return tool.name
        return None

    return {
        "current_round": pick("round"),
        "sync_data": pick("data", "sync") or pick("download"),
        "train_model": pick("train") or pick("model"),
        "evaluate_model": pick("eval") or pick("score"),
        "submit_predictions": pick("submit") or pick("upload"),
    }


def tool_info_to_dict(tool: MCPToolInfo) -> Dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def parse_json_or_none(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    return json.loads(raw)
