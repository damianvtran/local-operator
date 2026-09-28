"""The episode action server: one episode's action surface, exported over MCP.

WHAT THIS IS. An evaluation episode's computer-use actions reach the model as
a real MCP tool when the episode runs as a session (the SDK engagement in
``local_operator/evaluation/session_arm.py``). This module is the server side
of that window: a stdio MCP server, declared in the episode's own ``mcp.json``,
whose one tool is the same projected ``ActionBatch`` contract the loop-driven
action tool uses (``runner/action_tool.py`` -- the projection is IMPORTED, not
re-stated, so the offer cannot drift from the contract).

WHY A SEPARATE PROCESS, AND WHY A BRIDGE. The action has to execute against
the episode's environment, whose connection, verification and evidence live in
the episode driver -- not in a child the session spawns. So the server owns no
environment state: each ``tools/call`` is forwarded, one newline-delimited
JSON frame each way, to a UNIX socket the driver listens on inside the episode
scratch. That keeps the MCP server a pure shape adapter (nothing
OSWorld-specific and nothing environment-specific crosses into it), and it
keeps the driver the only writer of episode state.

THE WIRE (``WIRE_PROTOCOL``). One connection per call; the request is
``{"protocol": 1, "tool": "apply_actions", "arguments": {...}}`` and the reply
is ``{"protocol": 1, "content": [text|image blocks], "is_error": bool}``. The
frame helpers below are the ONE statement of that wire -- the bridge imports
them from here rather than re-deriving the shape.

FAILURE CONTRACT. A call that cannot reach the driver is answered as an MCP
tool ERROR (``is_error=true``) with a sentence naming the cause, never raised
into the session: the model sees a tool result it can react to, and a driver
that has decided the episode is over answers refusals through this same
channel. The process's stdout is the protocol stream -- every log line goes to
stderr.

ISOLATION. This module imports no provider, config, credentials, session or
TUI; the MCP SDK itself is imported lazily inside the serving functions (the
same discipline ``local_operator/mcp/`` uses), so importing this module
without the ``mcp`` extra installed stays cheap and safe.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import asdict, fields
from typing import Any, Mapping, Sequence

from local_operator.evaluation.action_surface import ActionSurface
from local_operator.evaluation.protocol import MAX_ENVELOPE_BYTES
from local_operator.evaluation.runner.action_tool import (
    ACTION_TOOL_DESCRIPTION,
    ACTION_TOOL_NAME,
    action_tool_parameters,
)

#: The MCP server name the episode driver declares. The session-visible tool
#: name is minted by the harness from (server name, tool name) --
#: ``mcp__episode_actions_apply_actions`` -- so both ends of the declaration
#: derive the callable name from these two constants rather than pasting it.
SERVER_NAME = "episode-actions"

#: The frame protocol both ends of the UNIX socket speak. Versioned because
#: the two processes are started separately and a skew must fail loudly on its
#: first frame rather than behave plausibly; the checks live in
#: :func:`decode_call` / :func:`decode_response` and are shared by the bridge.
WIRE_PROTOCOL = 1

#: Read limit for one wire frame, on BOTH ends of the socket. A reply carries
#: the observation's images base64-encoded, so one reply routinely exceeds
#: asyncio's 64 KiB default line limit: measured on the paid session arm
#: 2026-09-28 (task_017), a 476 KiB frame produced a reply whose read raised
#: "Separator is not found, and chunk exceed the limit", and every EXECUTED
#: batch was answered to the model as unreachable -- while the desktop had
#: already acted. A policy ceiling far above the largest frame measured
#: (~26x that ~640 KiB reply), not a derived invariant: the cap it is sized
#: off (the protocol envelope) bounds frame REFERENCES, while this wire
#: inlines the images themselves. Still bounded, so a rogue reply cannot
#: read unbounded; the client opens its connection with it and the bridge
#: binds its server with it.
WIRE_READ_LIMIT_BYTES = MAX_ENVELOPE_BYTES + 4096

#: How long one forwarded call may wait for its reply. Deliberately generous:
#: an executed batch is funded by the adapter's own per-call deadlines (a batch
#: may legally wait 64x60 s), and cutting it here would report a live
#: environment as dead. ``None`` (default) leaves the bound to the MCP
#: client's own timeout, which the episode driver already sizes.
DEFAULT_CALL_TIMEOUT_S: float | None = None

_TEXT = "text"
_IMAGE = "image"


class ActionBridgeUnreachable(RuntimeError):
    """The driver's bridge socket could not be reached for one call."""


def surface_to_json(surface: ActionSurface) -> str:
    """Serialize the negotiated surface for the server's argv.

    Field-by-field from the frozen dataclass: a field added to ``ActionSurface``
    joins the wire without a second list to update, which is the same reason
    the tool schema is derived rather than written by hand.
    """

    return json.dumps(asdict(surface), sort_keys=True)


def surface_from_json(payload: str) -> ActionSurface:
    """Rebuild the surface the driver negotiated; refuse anything malformed.

    The reconstruction is the VALIDATION: ``ActionSurface`` is frozen, so a
    payload naming a field it does not have (a skew between the two ends) is
    refused here, at process start, rather than degrading into a server whose
    offer disagrees with the adapter's admission.
    """

    values: Any = json.loads(payload)
    if not isinstance(values, dict):
        raise ValueError("surface payload must be a JSON object")
    # Strict in BOTH directions: a field the model does not know (a newer
    # driver) and a field this build requires but the payload omits (an older
    # driver) are the same defect -- two ends that would otherwise negotiate
    # DIFFERENT offers while both look healthy.
    expected = {field.name for field in fields(ActionSurface)}
    if set(values) != expected:
        raise ValueError(
            f"surface payload must name exactly {sorted(expected)}; " f"got {sorted(values)}"
        )
    return ActionSurface(**values)


# ---------------------------------------------------------------------------
# The wire -- one implementation, imported by both ends
# ---------------------------------------------------------------------------

_CONTENT_TYPES = (_TEXT, _IMAGE)


def encode_call(arguments: Mapping[str, Any], *, call_id: str | None = None) -> bytes:
    """One request frame: the tool, its arguments and a per-call id for logs."""

    payload = {
        "protocol": WIRE_PROTOCOL,
        "tool": ACTION_TOOL_NAME,
        "call_id": call_id or f"call-{uuid.uuid4().hex[:12]}",
        "arguments": dict(arguments),
    }
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def decode_call(line: bytes) -> dict[str, Any]:
    """Parse one request frame; refuse a frame this build cannot serve."""

    payload = _decode_object(line, "call frame")
    if payload.get("protocol") != WIRE_PROTOCOL:
        raise ValueError(f"unsupported call protocol {payload.get('protocol')!r}")
    if payload.get("tool") != ACTION_TOOL_NAME:
        raise ValueError(f"unknown tool {payload.get('tool')!r}")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("call arguments must be a JSON object")
    return payload


def encode_response(
    content: Sequence[Any],
    *,
    is_error: bool = False,
    details: Mapping[str, Any] | None = None,
) -> bytes:
    """One reply frame; content blocks are flattened to the wire vocabulary.

    Accepts harness ``Content`` blocks (the driver's renderer output) and wire
    dicts alike, so the refusal path and the observation path converge here --
    the one place the model-facing result shape is stated.
    """

    payload: dict[str, Any] = {
        "protocol": WIRE_PROTOCOL,
        "content": [_block_to_wire(block) for block in content],
        "is_error": bool(is_error),
    }
    if details is not None:
        payload["details"] = dict(details)
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def decode_response(line: bytes) -> dict[str, Any]:
    """Parse one reply frame; validate every block before it reaches the model."""

    payload = _decode_object(line, "response frame")
    if payload.get("protocol") != WIRE_PROTOCOL:
        raise ValueError(f"unsupported response protocol {payload.get('protocol')!r}")
    content = payload.get("content")
    if not isinstance(content, list):
        raise ValueError("response content must be a JSON array")
    for block in content:
        _validate_wire_block(block)
    return payload


def _decode_object(line: bytes, what: str) -> dict[str, Any]:
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed {what}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{what} must be a JSON object")
    return payload


def _block_to_wire(block: Any) -> dict[str, Any]:
    """One content block on the wire. Harness blocks are read by attribute."""

    block_type = block.get("type") if isinstance(block, Mapping) else getattr(block, "type", None)
    if block_type == _TEXT:
        text = block.get("text") if isinstance(block, Mapping) else block.text
        return {"type": _TEXT, "text": text or ""}
    if block_type == _IMAGE:
        if isinstance(block, Mapping):
            data = block.get("data")
            mime_type = block.get("mime_type") or block.get("mimeType")
        else:
            data = block.data
            mime_type = block.mime_type
        return {"type": _IMAGE, "data": data or "", "mime_type": mime_type or "image/png"}
    raise ValueError(f"unsupported content block type {block_type!r}")


def _validate_wire_block(block: Any) -> None:
    if not isinstance(block, dict) or block.get("type") not in _CONTENT_TYPES:
        raise ValueError(f"unsupported content block {block!r}")
    if block["type"] == _TEXT and not isinstance(block.get("text"), str):
        raise ValueError("text block must carry a string 'text'")
    if block["type"] == _IMAGE:
        if not isinstance(block.get("data"), str) or not isinstance(block.get("mime_type"), str):
            raise ValueError("image block must carry base64 'data' and 'mime_type'")


async def forward_call(
    endpoint: str,
    arguments: Mapping[str, Any],
    *,
    timeout: float | None = DEFAULT_CALL_TIMEOUT_S,
) -> dict[str, Any]:
    """One call, one connection, one reply: the whole client half of the wire."""

    try:
        reader, writer = await asyncio.open_unix_connection(endpoint, limit=WIRE_READ_LIMIT_BYTES)
    except OSError as error:
        raise ActionBridgeUnreachable(
            f"the episode's action bridge is not reachable at {endpoint!r} ({error})"
        ) from error
    line = b""
    try:
        writer.write(encode_call(arguments))
        await writer.drain()
        if timeout is None:
            line = await reader.readline()
        else:
            line = await asyncio.wait_for(reader.readline(), timeout)
    except asyncio.TimeoutError as error:
        raise ActionBridgeUnreachable(
            f"the episode's action bridge did not answer within {timeout}s"
        ) from error
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
    if not line:
        raise ActionBridgeUnreachable("the episode's action bridge closed without a reply")
    try:
        return decode_response(line)
    except ValueError as error:
        raise ActionBridgeUnreachable(
            f"the episode's action bridge sent a bad reply: {error}"
        ) from error


def format_forward_error(error: BaseException) -> tuple[list[dict[str, Any]], bool]:
    """The model-facing tool result for a failed call to the action bridge.

    A transport failure is reported, never raised into the session: a raised
    call would read to the loop as a broken tool, while a reported one lets the
    episode driver's own state (which the model cannot see) decide whether the
    run continues.

    The sentence deliberately does NOT claim the call never reached the driver:
    it is used for every reply-side fault too -- a read-limit overrun, a bad
    reply, a timeout after the driver may already have executed -- and the paid
    session-arm episode that motivated ``WIRE_READ_LIMIT_BYTES`` was taught
    "the channel is dead" by exactly that false claim while its batches had
    already acted. Non-reachability is stated by the cause itself ("not
    reachable at ..."), never asserted here.
    """

    text = f"Action call failed while talking to the episode's action bridge: {error}"
    return [{"type": _TEXT, "text": text}], True


# ---------------------------------------------------------------------------
# The MCP server
# ---------------------------------------------------------------------------


def build_mcp_server(endpoint: str, surface: ActionSurface, *, timeout: float | None):
    """The low-level MCP server, wired by hand for one reason: the tool's
    ``inputSchema`` is the FLATTENED action projection, which a decorated
    Python function cannot express -- signature introspection would emit a
    ``$ref``/``anyOf``-laden schema some providers reject, and the projection
    exists precisely to avoid that. Everything else is the SDK's own server.
    """

    import mcp.types as types
    from mcp.server.lowlevel import Server

    server = Server(SERVER_NAME)

    async def on_list_tools(
        ctx: Any, params: types.PaginatedRequestParams
    ) -> types.ListToolsResult:
        del ctx, params
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=ACTION_TOOL_NAME,
                    description=ACTION_TOOL_DESCRIPTION,
                    input_schema=action_tool_parameters(surface),
                )
            ]
        )

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        del ctx
        if params.name != ACTION_TOOL_NAME:
            return types.CallToolResult(
                content=[types.TextContent(type=_TEXT, text=f"Unknown tool {params.name!r}")],
                is_error=True,
            )
        arguments = params.arguments or {}
        try:
            reply = await forward_call(endpoint, arguments, timeout=timeout)
        except (ActionBridgeUnreachable, ValueError) as error:
            blocks, is_error = format_forward_error(error)
            return _to_mcp_result(blocks, is_error=is_error)
        return _to_mcp_result(reply["content"], is_error=bool(reply.get("is_error")))

    server.add_request_handler("tools/list", types.PaginatedRequestParams, on_list_tools)
    server.add_request_handler("tools/call", types.CallToolRequestParams, on_call_tool)
    return server


def _to_mcp_result(blocks: Sequence[Any], *, is_error: bool):
    import mcp.types as types

    content: list[Any] = []
    for block in blocks:
        if block["type"] == _TEXT:
            content.append(types.TextContent(type=_TEXT, text=block["text"]))
        else:
            content.append(
                types.ImageContent(type=_IMAGE, data=block["data"], mime_type=block["mime_type"])
            )
    return types.CallToolResult(content=content, is_error=is_error)


async def serve(endpoint: str, surface: ActionSurface, *, timeout: float | None) -> None:
    """Serve one episode's action surface over stdio until stdin closes."""

    from mcp.server.stdio import stdio_server

    server = build_mcp_server(endpoint, surface, timeout=timeout)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve one episode's action surface over MCP (stdio)."
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="the episode driver's action-bridge UNIX socket",
    )
    parser.add_argument(
        "--surface",
        required=True,
        help="the negotiated ActionSurface as JSON (see surface_to_json)",
    )
    parser.add_argument(
        "--call-timeout",
        type=float,
        default=DEFAULT_CALL_TIMEOUT_S,
        help="seconds to wait for a forwarded call's reply (default: no bound here)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        surface = surface_from_json(args.surface)
    except (ValueError, TypeError) as error:
        print(f"action_server: bad --surface: {error}", file=sys.stderr)
        return 2
    try:
        asyncio.run(serve(args.endpoint, surface, timeout=args.call_timeout))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
