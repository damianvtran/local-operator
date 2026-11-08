"""Responses-API shim that lets Codex's client-side `tool_search` survive a
Responses->Messages translator.

WHY: Codex (rust-v0.147.0) defers MCP tools behind its own `tool_search` tool
when the model's catalog entry says `supports_search_tool: true`. That tool is
a Responses-API construct: the request carries `{"type": "tool_search",
"execution": "client", ...}`, the model answers with a `tool_search_call`
output item, Codex runs BM25 locally and sends back a `tool_search_output`
input item whose `tools` array holds the matching tool specs; the OpenAI
Responses server then makes those tools callable. LiteLLM 1.106.6 (the rig's
Responses->Messages translator) DROPS tool type `tool_search` ("it has no
Chat Completions equivalent", litellm/responses/litellm_completion_transformation/
transformation.py) and has no handling for `tool_search_call` /
`tool_search_output` items. So through LiteLLM, deferral silently removes the
MCP tools from the model's view instead of deferring them. Measured at $0 with
codex_dev_capture.sh: 11 tools reach Anthropic, no search tool, no MCP tools.

WHAT this shim does (it sits between Codex and LiteLLM, both speaking Responses):
  request  * `tool_search` tool  -> ordinary function tool `tool_search`
             (same description and JSON schema Codex generated);
           * input `tool_search_call`   -> `function_call` named tool_search
             (arguments re-serialised as a JSON string);
           * input `tool_search_output` -> `function_call_output` (a short
             text listing of the loaded tools), AND each loaded tool spec is
             appended to `tools`, the same thing the Responses server does;
             appended in first-loaded order and deduplicated, so the tools
             block only grows and the prompt-cache prefix stays as stable as
             it can be;
  response * a streamed `function_call` item named `tool_search` is re-emitted
             as a `tool_search_call` item (`execution: "client"`, arguments
             parsed to an object) on `response.output_item.done`; its
             `.added` and argument-delta events are withheld because Codex
             parses items on `.done`, and in `response.completed` the item is
             rewritten the same way.
  refuse   * any server-side web search tool (`type` starting `web_search`,
             e.g. `web_search`, `web_search_preview`): 400, nothing forwarded
             (S9e). codex-best runs with `web_search = "disabled"`; LiteLLM
             would turn the tool into Anthropic's unpriced server tool
             `web_search_20250305`; no arm gets a server tool (R14 M2,
             adapters.SERVER_TOOL_AUDIT). Refused, never
             stripped, so a config regression is visible at $0 instead of
             silently changing what Codex asked for. A *function* tool that
             merely is named `web_search` is client-side and passes.
Everything else passes through byte-for-byte. Headers are forwarded
untouched except hop-by-hop ones; nothing is logged except one JSON line per
request with counts (no bodies, no headers, no credentials).

Constraints: stateless (all state lives in Codex's own request history);
127.0.0.1 only; one upstream (BRIDGE_UPSTREAM, default LiteLLM on :4000).
This is NOT Codex as shipped: present runs through it as a separately
labelled "best-config (translator bridge)" arm.

Usage: BRIDGE_UPSTREAM=http://127.0.0.1:4000 python codex_bridge.py <port>
       then point Codex's provider base_url at http://127.0.0.1:<port>/v1
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import TypedDict, TypeGuard

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

if __package__:
    from ..sse_stream import iter_sse
else:
    # The coding driver launches this file directly, outside the parent path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sse_stream import iter_sse

UPSTREAM = os.environ.get("BRIDGE_UPSTREAM", "http://127.0.0.1:4000").rstrip("/")
LOG = Path(os.environ.get("BRIDGE_LOG", "codex_bridge.jsonl"))
NAME = "tool_search"
HOP = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}
client = httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30))


class _ToolSpec(TypedDict, total=False):
    type: str
    name: str
    tools: list[_ToolSpec]
    description: str
    parameters: dict[str, object]
    strict: bool


class _InputItem(TypedDict, total=False):
    type: str
    call_id: str | None
    name: str
    arguments: object
    tools: list[_ToolSpec]
    output: str


class _RequestBody(TypedDict, total=False):
    tools: list[_ToolSpec]
    input: list[_InputItem | str]


class _RewriteStats(TypedDict):
    tool_search_tool: bool
    search_calls: int
    loaded_specs: int


class _SearchFunctionCall(TypedDict, total=False):
    type: str
    name: str
    id: str
    call_id: str
    arguments: str


def _tool_names(spec: _ToolSpec) -> list[str]:
    if spec.get("type") == "namespace":
        return [f'{spec.get("name")}.{t.get("name")}' for t in spec.get("tools") or []]
    return [str(spec.get("name"))]


def rewrite_request(body: _RequestBody) -> tuple[_RequestBody, _RewriteStats]:
    """Map Codex's tool_search request shapes onto plain function tools/items."""
    stats: _RewriteStats = {"tool_search_tool": False, "search_calls": 0, "loaded_specs": 0}
    tools: list[_ToolSpec] = []
    for t in body.get("tools") or []:
        if t.get("type") == NAME:
            stats["tool_search_tool"] = True
            tools.append(
                {
                    "type": "function",
                    "name": NAME,
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                    "strict": False,
                }
            )
        else:
            tools.append(t)
    present = {n for t in tools for n in _tool_names(t)}
    new_input: list[_InputItem | str] = []
    for item in body.get("input") or []:
        typ = item.get("type") if isinstance(item, dict) else None
        if typ == "tool_search_call" and isinstance(item, dict):
            stats["search_calls"] += 1
            args = item.get("arguments")
            new_input.append(
                {
                    "type": "function_call",
                    "call_id": item.get("call_id"),
                    "name": NAME,
                    "arguments": args if isinstance(args, str) else json.dumps(args or {}),
                }
            )
        elif typ == "tool_search_output" and isinstance(item, dict):
            loaded = item.get("tools") or []
            listing = []
            for spec in loaded:
                names = _tool_names(spec)
                listing += names
                if set(names) <= present:
                    continue
                if spec.get("type") == "namespace":
                    # A later search can return more members of a namespace
                    # already loaded: append only the new members, so no
                    # flattened tool name is ever declared twice.
                    fresh = [
                        t
                        for t in spec.get("tools") or []
                        if f'{spec.get("name")}.{t.get("name")}' not in present
                    ]
                    fresh_spec: _ToolSpec = {**spec, "tools": fresh}
                    spec = fresh_spec
                    names = _tool_names(spec)
                tools.append(spec)
                present.update(names)
                stats["loaded_specs"] += 1
            text = (
                ("Loaded tools (now callable): " + ", ".join(listing))
                if listing
                else "No matching tools."
            )
            new_input.append(
                {"type": "function_call_output", "call_id": item.get("call_id"), "output": text}
            )
        else:
            new_input.append(item)
    body["tools"] = tools
    body["input"] = new_input
    return body, stats


def _convert_item(item: _SearchFunctionCall) -> dict[str, object]:
    try:
        args = json.loads(item.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {"query": item.get("arguments") or ""}
    return {
        "type": "tool_search_call",
        "id": item.get("id"),
        "call_id": item.get("call_id"),
        "status": "completed",
        "execution": "client",
        "arguments": args,
    }


def _is_search_fc(item: object) -> TypeGuard[_SearchFunctionCall]:
    return (
        isinstance(item, dict) and item.get("type") == "function_call" and item.get("name") == NAME
    )


def _server_web_search(body: _RequestBody) -> bool:
    return any(
        isinstance(t, dict) and str(t.get("type", "")).startswith("web_search")
        for t in body.get("tools") or []
    )


async def responses(request: Request) -> Response:
    raw = await request.body()
    body = json.loads(raw)
    if _server_web_search(body):
        reason = "server-side web search tool"
        with LOG.open("a") as fh:
            fh.write(json.dumps({"ts": time.time(), "status": 400, "refused": reason}) + "\n")
        return Response(json.dumps({"error": "refused by the bench bridge", "reason": reason}),
                        status_code=400, media_type="application/json")
    body, stats = rewrite_request(body)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    # Only the host transport receives auth; never load a private YAML/auth DB.
    key = os.environ.get("CODING_TRANSLATOR_API_KEY")
    if key:
        headers["authorization"] = "Bearer " + key
    req = client.build_request(
        "POST", f"{UPSTREAM}/v1/responses", headers=headers, content=json.dumps(body).encode()
    )
    resp = await client.send(req, stream=True)
    out_headers = {
        k: v
        for k, v in resp.headers.items()
        if k.lower()
        not in {"content-length", "content-encoding", "transfer-encoding", "connection"}
    }
    row = {
        "ts": time.time(),
        "status": resp.status_code,
        "n_tools": len(body.get("tools") or []),
        **stats,
        "search_calls_out": 0,
    }

    if "text/event-stream" not in resp.headers.get("content-type", ""):
        content = await resp.aread()
        await resp.aclose()
        with LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        return Response(content, status_code=resp.status_code, headers=out_headers)

    async def gen():
        hidden: set[str | None] = set()  # item ids of tool_search function calls
        try:
            async for event, data in iter_sse(resp.aiter_raw()):
                if data is None:
                    yield event
                    continue
                try:
                    d = json.loads(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    yield event
                    continue
                if not isinstance(d, dict):
                    yield event
                    continue
                t = d.get("type", "")
                item = d.get("item")
                if t == "response.output_item.added" and _is_search_fc(item):
                    hidden.add(item.get("id"))
                    continue
                if t.startswith("response.function_call_arguments.") and d.get("item_id") in hidden:
                    continue
                if t == "response.output_item.done" and _is_search_fc(item):
                    d["item"] = _convert_item(item)
                    row["search_calls_out"] += 1
                elif t in ("response.completed", "response.incomplete") and isinstance(
                    d.get("response"), dict
                ):
                    d["response"]["output"] = [
                        _convert_item(i) if _is_search_fc(i) else i
                        for i in d["response"].get("output") or []
                    ]
                else:
                    yield event
                    continue
                yield f"event: {t}\ndata: {json.dumps(d)}\n\n".encode()
        finally:
            await resp.aclose()
            with LOG.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

    return StreamingResponse(gen(), status_code=resp.status_code, headers=out_headers)


app = Starlette(routes=[Route("/v1/responses", responses, methods=["POST"])])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
