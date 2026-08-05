"""Bridge between MCP server tools and harness ``AgentTool``s.

Ports the omp tool bridge (``src/mcp/tool-bridge.ts``): tool-name mangling,
schema normalization, outbound argument hygiene, result flattening, and the
retriable-connection-error classification that drives the manager's
reconnect-once + retry-once policy. Pure functions only — no MCP SDK imports,
so this module stays importable anywhere.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from local_operator.harness.types import AgentTool, TextContent, ToolResult

# The harness injects an ``i`` (intent) field into every tool's wire schema;
# strict-schema MCP servers reject calls that carry undeclared fields.
INTENT_FIELD = "i"

# Network-level and stale-session errors that warrant a reconnect + single
# retry. Conservative: only errors where the server is likely alive but the
# connection object is stale (dead SSE, expired session, refused after
# restart). Mirrors omp's RETRIABLE_PATTERNS.
_RETRIABLE_PATTERNS = (
    "econnrefused",
    "econnreset",
    "epipe",
    "enetunreach",
    "ehostunreach",
    "connection refused",
    "connection reset",
    "broken pipe",
    "network is unreachable",
    "no route to host",
    "fetch failed",
    "transport not connected",
    "transport closed",
    "connection closed",
    "network error",
    "http 404",
    "http 502",
    "http 503",
    "404 not found",
    "502 bad gateway",
    "503 service unavailable",
)

_SANITIZE_RE = re.compile(r"[^a-z_]+")
_RUN_RE = re.compile(r"_+")


def _sanitize_part(value: str, fallback: str) -> str:
    """Lowercase, map non ``[a-z_]`` runs to ``_``, collapse runs, trim edges.

    An empty result falls back to ``fallback`` so the minted tool name always
    carries both parts (omp behavior).
    """
    sanitized = _RUN_RE.sub("_", _SANITIZE_RE.sub("_", value.lower())).strip("_")
    return sanitized or fallback


def create_mcp_tool_name(server_name: str, tool_name: str) -> str:
    """Mint the harness-visible name for an MCP tool.

    Shape: ``mcp__<server>_<tool>`` with both parts sanitized to lowercase
    ``[a-z_]``. A redundant server prefix on the tool name is stripped: server
    ``puppeteer`` + tool ``puppeteer_screenshot`` becomes
    ``mcp__puppeteer_screenshot``, not ``mcp__puppeteer_puppeteer_screenshot``.
    """
    server = _sanitize_part(server_name, "server")
    tool = _sanitize_part(tool_name, "tool")

    prefix = f"{server}_"
    tool = tool.removeprefix(prefix)
    return f"mcp__{server}_{tool}"


def normalize_input_schema(input_schema: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize an MCP ``inputSchema`` for provider tool declarations.

    Providers expect a JSON Schema object: ensure ``type: "object"`` and fill
    ``properties``/``required`` when missing. Unknown keys pass through
    (servers occasionally add extensions).
    """
    schema = dict(input_schema) if isinstance(input_schema, dict) else {}
    if schema.get("type") != "object":
        schema["type"] = "object"
    if not isinstance(schema.get("properties"), dict):
        schema["properties"] = {}
    if not isinstance(schema.get("required"), list):
        schema["required"] = []
    return schema


def _is_unused_optional_placeholder(value: Any) -> bool:
    """Empty string / empty dict / None — placeholders models invent."""
    return value is None or value == "" or (isinstance(value, dict) and not value)


def prepare_outbound_args(
    args: dict[str, Any] | None,
    declared_properties: dict[str, Any] | None,
    required: list[str] | None = None,
    additional_properties: Any = None,
) -> dict[str, Any]:
    """Clean harness-side arguments before forwarding to ``tools/call``.

    1. Drop the harness-injected ``i`` intent field when the schema does not
       declare it — unconditionally: strict-schema servers (Linear, anything
       with ``additionalProperties: false`` or Zod strict) reject every call
       carrying it, and the MCP boundary is the authoritative guard so no
       caller has to pre-strip.
    2. Drop other undeclared keys ONLY when ``additionalProperties`` is
       explicitly ``False``. An absent ``additionalProperties`` or an empty
       sub-schema (``{}``) is open per JSON Schema — the schema may be
       composed via ``$ref``/``allOf``/``oneOf`` or accept anything, so we
       forward the caller's keys untouched rather than silently running the
       call with ``{}``.
    3. Drop optional (non-required) keys whose value is an empty placeholder
       (``None``, ``""``, ``{}``). Falsy values that are not placeholders
       (``0``, ``False``, ``[]``) are always kept.

    Returns a new dict; the input is never mutated.
    """
    if not isinstance(args, dict):
        return {}
    properties = declared_properties if isinstance(declared_properties, dict) else {}
    required_set = set(required or [])
    strict = additional_properties is False

    cleaned: dict[str, Any] = {}
    for key, value in args.items():
        declared = key in properties
        if key == INTENT_FIELD and not declared:
            continue  # harness intent never reaches a server that didn't ask for it
        if strict and not declared:
            continue  # schema explicitly forbids unknown fields
        if key not in required_set and _is_unused_optional_placeholder(value):
            continue
        cleaned[key] = value
    return cleaned


def format_mcp_result(
    result: Any,
    tool_call_id: str = "",
    tool_name: str = "",
) -> ToolResult:
    """Flatten an MCP ``tools/call`` result into a harness ``ToolResult``.

    Content blocks are joined text with separators: text blocks pass through,
    image blocks become ``[Image: <mime>]`` placeholders, embedded resources
    become ``[Resource: <uri>]`` plus their text when present. ``isError``
    maps to ``is_error`` (with an ``Error:`` prefix, matching omp).
    """
    content_blocks = getattr(result, "content", None)
    if content_blocks is None and isinstance(result, dict):
        content_blocks = result.get("content")
    content_blocks = content_blocks or []

    if isinstance(result, dict):
        is_error = bool(result.get("is_error", result.get("isError")))
    else:
        is_error = bool(getattr(result, "is_error", False))

    parts: list[str] = []
    for item in content_blocks:
        item_type = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")
        if item_type == "text":
            text = getattr(item, "text", "") if not isinstance(item, dict) else item.get("text", "")
            parts.append(text or "")
        elif item_type == "image":
            mime = (
                getattr(item, "mime_type", "image")
                if not isinstance(item, dict)
                else item.get("mimeType") or item.get("mime_type") or "image"
            )
            parts.append(f"[Image: {mime}]")
        elif item_type == "resource":
            resource = (
                getattr(item, "resource", None)
                if not isinstance(item, dict)
                else item.get("resource")
            )
            uri = (
                getattr(resource, "uri", "")
                if resource is not None and not isinstance(resource, dict)
                else (resource or {}).get("uri", "")
            )
            text = (
                getattr(resource, "text", None)
                if resource is not None and not isinstance(resource, dict)
                else (resource or {}).get("text")
            )
            if text:
                parts.append(f"[Resource: {uri}]\n{text}")
            else:
                parts.append(f"[Resource: {uri}]")

    text = "\n\n".join(parts)
    if is_error:
        text = f"Error: {text}"
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=[TextContent(text=text)],
        is_error=is_error,
        details={
            "server_result": (
                result
                if isinstance(result, dict)
                else getattr(result, "model_dump", lambda: None)()
            ),
        },
    )


def is_retriable_connection_error(error: BaseException) -> bool:
    """Classify an exception as a stale-connection error worth one reconnect.

    Matches econnrefused/reset-class network failures, "transport closed"-style
    disconnects, and HTTP 404/502/503 (stale session after a server restart).
    """
    message = str(error).lower()
    return any(pattern in message for pattern in _RETRIABLE_PATTERNS)


def build_agent_tool(
    server_name: str,
    mcp_tool: Any,
    call_fn: Callable[..., Any],
) -> AgentTool:
    """Wrap one MCP tool as a harness ``AgentTool``.

    ``mcp_tool`` may be an SDK ``Tool`` model or a cached plain dict with
    ``name`` / ``description`` / ``inputSchema``. ``call_fn`` is the manager's
    execute coroutine: ``call_fn(tool_call_id, args, signal, on_update,
    context) -> ToolResult`` — deferred vs live behavior lives there, so this
    wrapper is identical for both.
    """
    if isinstance(mcp_tool, dict):
        raw_name = mcp_tool.get("name", "")
        description = mcp_tool.get("description", "") or ""
        input_schema = mcp_tool.get("inputSchema") or mcp_tool.get("input_schema") or {}
    else:
        raw_name = getattr(mcp_tool, "name", "") or ""
        description = getattr(mcp_tool, "description", "") or ""
        input_schema = (
            getattr(mcp_tool, "input_schema", None) or getattr(mcp_tool, "inputSchema", None) or {}
        )

    async def _execute(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any,
        on_update: Any,
        context: Any,
    ) -> ToolResult:
        return await call_fn(tool_call_id, args, signal, on_update, context)

    return AgentTool(
        name=create_mcp_tool_name(server_name, raw_name),
        label=f"{server_name}/{raw_name}",
        description=description,
        parameters=normalize_input_schema(input_schema),
        approval_tier="exec",  # unknown external side effects default to exec
        interruptible=True,
        execute=_execute,
    )
