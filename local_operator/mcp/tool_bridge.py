"""Bridge between MCP server tools and harness ``AgentTool``s.

Implements the MCP tool bridge: tool-name mangling,
schema normalization, outbound argument hygiene, result flattening, and the
retriable-connection-error classification that drives the manager's
reconnect-once + retry-once policy. Pure functions only, and no MCP SDK
import at module scope, so this module stays importable (and cheap) when the
SDK extra is absent; the one SDK reference sits inside the branch that only
runs once an SDK-validated content block has actually arrived.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from local_operator.harness.intent import INTENT_FIELD, apply_intent_schema
from local_operator.harness.types import (
    AgentTool,
    TextContent,
    ToolExecuteFn,
    ToolResult,
)

if TYPE_CHECKING:
    # Annotation-only: the SDK is an optional extra, and this module must
    # stay importable (and cheap) for config-only callers.
    from mcp.types import CallToolResult, ContentBlock, Tool

# Network-level and stale-session errors that warrant a reconnect + single
# retry. Conservative: only errors where the server is likely alive but the
# connection object is stale (dead SSE, expired session, refused after
# restart). Mirrors the established retriable error patterns.
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
    carries both parts (established behavior).
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


def _is_unused_optional_placeholder(value: object) -> bool:
    """Empty string / empty dict / None — placeholders models invent."""
    return value is None or value == "" or (isinstance(value, dict) and not value)


def prepare_outbound_args(
    args: dict[str, Any] | None,
    declared_properties: dict[str, Any] | None,
    required: list[str] | None = None,
    additional_properties: bool | dict[str, Any] | None = None,
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


def _dict_block_text(block: dict[str, Any]) -> str | None:
    """Render one raw-JSON content block, or ``None`` for kinds we drop.

    Servers that answer ``tools/call`` with unvalidated JSON (and the tool
    cache) hand us plain dicts rather than SDK models, so both wire shapes of
    the image mime key are accepted.
    """
    block_type = block.get("type")
    if block_type == "text":
        return block.get("text") or ""
    if block_type == "image":
        return f"[Image: {block.get('mimeType') or block.get('mime_type') or 'image'}]"
    if block_type == "resource":
        raw_resource = block.get("resource")
        resource: dict[str, Any] = raw_resource if isinstance(raw_resource, dict) else {}
        uri = resource.get("uri", "")
        text = resource.get("text")
        return f"[Resource: {uri}]\n{text}" if text else f"[Resource: {uri}]"
    return None


def _model_block_text(block: ContentBlock) -> str | None:
    """Render one SDK content block, or ``None`` for kinds we drop.

    Audio and resource links carry nothing a text transcript can show, so
    they contribute no part at all rather than an empty one.
    """
    if block.type == "text":
        return block.text or ""
    if block.type == "image":
        return f"[Image: {block.mime_type}]"
    if block.type == "resource":
        # Only the text variant of an embedded resource has inline content;
        # blob resources are base64 payloads we summarize by URI alone.
        from mcp.types import TextResourceContents

        resource = block.resource
        text = resource.text if isinstance(resource, TextResourceContents) else None
        return f"[Resource: {resource.uri}]\n{text}" if text else f"[Resource: {resource.uri}]"
    return None


def format_mcp_result(
    result: CallToolResult | dict[str, Any],
    tool_call_id: str = "",
    tool_name: str = "",
) -> ToolResult:
    """Flatten an MCP ``tools/call`` result into a harness ``ToolResult``.

    Content blocks are joined text with separators: text blocks pass through,
    image blocks become ``[Image: <mime>]`` placeholders, embedded resources
    become ``[Resource: <uri>]`` plus their text when present. ``isError``
    maps to ``is_error`` (with an ``Error:`` prefix, matching the established behavior).

    ``result`` is the SDK model on every live call path; the raw-dict branch
    covers servers whose payload never went through SDK validation.
    """
    blocks: Sequence[ContentBlock] | Sequence[dict[str, Any]]
    server_result: dict[str, Any] | None
    if isinstance(result, dict):
        raw_blocks = result.get("content")
        blocks = raw_blocks if isinstance(raw_blocks, list) else []
        is_error = bool(result.get("is_error", result.get("isError")))
        server_result = result
    else:
        blocks = result.content
        is_error = bool(result.is_error)
        server_result = result.model_dump()

    parts: list[str] = []
    for block in blocks:
        rendered = _dict_block_text(block) if isinstance(block, dict) else _model_block_text(block)
        if rendered is not None:
            parts.append(rendered)

    text = "\n\n".join(parts)
    if is_error:
        text = f"Error: {text}"
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=[TextContent(text=text)],
        is_error=is_error,
        details={"server_result": server_result},
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
    mcp_tool: Tool | dict[str, Any],
    call_fn: ToolExecuteFn,
) -> AgentTool:
    """Wrap one MCP tool as a harness ``AgentTool``.

    ``mcp_tool`` may be an SDK ``Tool`` model or a cached plain dict with
    ``name`` / ``description`` / ``inputSchema``. ``call_fn`` is the manager's
    execute coroutine — deferred vs live behavior lives there, so this
    wrapper is identical for both.
    """
    raw_name: str
    description: str
    input_schema: dict[str, Any] | None
    if isinstance(mcp_tool, dict):
        raw_name = mcp_tool.get("name", "")
        description = mcp_tool.get("description", "") or ""
        input_schema = mcp_tool.get("inputSchema") or mcp_tool.get("input_schema") or {}
    else:
        raw_name = mcp_tool.name
        description = mcp_tool.description or ""
        input_schema = mcp_tool.input_schema

    return AgentTool(
        name=create_mcp_tool_name(server_name, raw_name),
        label=f"{server_name}/{raw_name}",
        description=description,
        # MCP tools get `i` too. Their schemas are server-owned, so this is
        # the one place where injection could reach a foreign process — but
        # the loop lifts the key before `execute`, `prepare_outbound_args`
        # drops it again against the SERVER's schema (never this one), and a
        # server that declares its own `i` is left alone by
        # `apply_intent_schema`. Skipping MCP instead would put a hole in the
        # narration exactly where the user has least idea what is happening:
        # a remote call whose name is `mcp__linear_save_issue`.
        parameters=apply_intent_schema(normalize_input_schema(input_schema)),
        approval_tier="exec",  # unknown external side effects default to exec
        interruptible=True,
        execute=call_fn,
    )
