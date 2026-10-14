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

import base64
import copy
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from local_operator.harness.intent import INTENT_FIELD, apply_intent_schema
from local_operator.harness.types import (
    AgentTool,
    CapturedFrame,
    Content,
    ImageContent,
    TextContent,
    ToolContext,
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
    # A tools/call can surface an OAuth challenge AFTER initialize. Reconnect
    # recreates the HTTP transport through OAuthClientProvider (refreshing the
    # access token) and the call site retries ONCE — no replay loop.
    "http 401",
    "401 unauthorized",
    "www-authenticate",
    "mcp/www_authenticate",
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
    """Render one raw-JSON TEXT-bearing content block, or ``None`` when dropped.

    Servers that answer ``tools/call`` with unvalidated JSON (and the tool
    cache) hand us plain dicts rather than SDK models. Image blocks never
    reach here -- :func:`_block_image` routes them before this -- so the kinds
    this handles are text and embedded resources.
    """
    block_type = block.get("type")
    if block_type == "text":
        return block.get("text") or ""
    if block_type == "resource":
        raw_resource = block.get("resource")
        resource: dict[str, Any] = raw_resource if isinstance(raw_resource, dict) else {}
        uri = resource.get("uri", "")
        text = resource.get("text")
        return f"[Resource: {uri}]\n{text}" if text else f"[Resource: {uri}]"
    return None


def _model_block_text(block: ContentBlock) -> str | None:
    """Render one SDK TEXT-bearing content block, or ``None`` for kinds we drop.

    Audio and resource links carry nothing a text transcript can show, so
    they contribute no part at all rather than an empty one. Images are
    routed through :func:`_block_image` before this is consulted.
    """
    if block.type == "text":
        return block.text or ""
    if block.type == "resource":
        # Only the text variant of an embedded resource has inline content;
        # blob resources are base64 payloads we summarize by URI alone.
        from mcp.types import TextResourceContents

        resource = block.resource
        text = resource.text if isinstance(resource, TextResourceContents) else None
        return f"[Resource: {resource.uri}]\n{text}" if text else f"[Resource: {resource.uri}]"
    return None


def _image_content(data: str, mime_type: str) -> ImageContent | None:
    """One image block as model content, or ``None`` when it may not be shown.

    Two checks stand between a server's payload and the conversation, and a
    payload that fails either falls back to the text placeholder:

    * the bytes must DECODE. Forwarding a corrupt payload as an image block
      earns a provider-side ``Could not process image`` 400 with the block
      already in history -- ``imaging`` refuses the same thing at its own
      boundary for the same reason -- and the decode is cheap (bounded by the
      cap below).
    * the decoded size must fit :data:`~local_operator.imaging.IMAGE_MAX_BYTES`,
      the bound a local image read honours, imported inside the branch so
      text-only traffic never pays for the imaging module. Without a bound,
      one content block from an arbitrary configured server could claim an
      unbounded slice of a session's context.

    Bytes that fit are forwarded VERBATIM. Re-encoding someone else's media
    here would make the harness a second definition of what a server
    published -- and for evaluation frames it would silently break the
    adapter's own pixel geometry, which clicks are calibrated against.
    """
    if not data or not mime_type:
        return None
    from local_operator.imaging import IMAGE_MAX_BYTES

    try:
        decoded = base64.b64decode(data, validate=True)
    except ValueError:
        return None
    if len(decoded) > IMAGE_MAX_BYTES:
        return None
    return ImageContent(data=data, mime_type=mime_type)


def _block_image(block: Any) -> tuple[ImageContent | None, str | None]:
    """Classify one content block, both wire shapes, as image or not.

    Returns ``(image, placeholder)``: ``(content, None)`` for an image that
    may be inlined, ``(None, text)`` for an image that must fall back to a
    placeholder (missing or oversized payload), and ``(None, None)`` for a
    block of any other kind -- the caller renders those as text.
    """
    if isinstance(block, dict):
        if block.get("type") != "image":
            return None, None
        mime = str(block.get("mimeType") or block.get("mime_type") or "image")
        data = block.get("data")
    else:
        if block.type != "image":
            return None, None
        mime = block.mime_type or "image"
        data = block.data
    placeholder = f"[Image: {mime}]"
    if isinstance(data, str) and data:
        return _image_content(data, mime), placeholder
    return None, placeholder


#: MCP ``_meta`` key under which a producer may attach its own bounded capture
#: claim (``{"stream_id": ..., "revision_id": ...}``) to an image block. The
#: claim carries ONLY the producer's own stream/revision strings; the
#: authority-bearing producer scope is stamped by the harness from the actual
#: call site (see :func:`capture_producer_scope`), so a block claiming an
#: identity of its own changes nothing. An old reader can ignore the key and
#: still receives the image; readers that never see the key get images exactly
#: as before it existed.
CAPTURE_CLAIM_META_KEY = "local_operator/capture"

#: Upper bound for the harness-composed producer scope. A composition past it
#: refuses the stamp outright rather than getting truncated into a colliding
#: identity: an unstamped image is PROTECTED, the safe direction.
_CAPTURE_SCOPE_MAX = 256


def capture_producer_scope(server_name: str, mcp_tool_name: str, session_id: str) -> str | None:
    """Compose the harness-stamped producer scope for one MCP server+tool.

    Identity comes from the ORIGIN pair -- the actual configured server and
    its tool name -- never the collision-suffixed agent-visible label
    (:func:`create_mcp_tool_name` output can carry a ``_2`` suffix while the
    origin pair cannot) and never anything a producer supplied. The owning
    session is part of the scope so a fork or a copied transcript can never
    let one conversation's producer supersede another's captures.

    Returns ``None`` when a component is missing or the result exceeds the
    bound: the caller then stamps nothing, which keeps images PROTECTED.
    """
    if not session_id or not server_name or not mcp_tool_name:
        return None
    scope = f"mcp:{session_id}:{server_name}:{mcp_tool_name}"
    if len(scope) > _CAPTURE_SCOPE_MAX:
        return None
    return scope


def _capture_claim_of(block: Any) -> tuple[str, str] | None:
    """One image block's producer claim, or ``None`` when it cannot be used.

    Both wire shapes are read: dict rows (``_meta``, with ``meta`` as the
    by-name dump spelling) and SDK models (``.meta``). Anything missing,
    mistyped or outside the id contract returns ``None`` and the caller
    stamps nothing -- malformed metadata means PROTECTED, not a guess. The
    ids are validated by constructing the value object itself so there is
    exactly ONE definition of the bounds.
    """
    if isinstance(block, dict):
        meta = block.get("_meta", block.get("meta"))
    else:
        meta = getattr(block, "meta", None)
    if not isinstance(meta, dict):
        return None
    claim = meta.get(CAPTURE_CLAIM_META_KEY)
    if not isinstance(claim, dict):
        return None
    stream_id = claim.get("stream_id")
    revision_id = claim.get("revision_id")
    if not isinstance(stream_id, str) or not isinstance(revision_id, str):
        return None
    try:
        # A probe scope: only the id bounds are under test here; identity is
        # the harness's to stamp, never the claim's to state.
        CapturedFrame(stream_id=stream_id, revision_id=revision_id, producer_scope="probe")
    except ValidationError:
        return None
    return stream_id, revision_id


def _with_capture_claim(
    image: ImageContent, block: Any, producer_scope: str | None
) -> ImageContent:
    """Stamp harness-scoped capture provenance when the producer claimed one.

    No scope (the caller had no real producer identity) or no valid claim
    means the image keeps ``captured_frame=None`` -- exactly what compaction
    reads as PROTECTED, so every failure mode here degrades to retention.
    """
    if producer_scope is None:
        return image
    claim = _capture_claim_of(block)
    if claim is None:
        return image
    return image.model_copy(
        update={
            "captured_frame": CapturedFrame(
                stream_id=claim[0], revision_id=claim[1], producer_scope=producer_scope
            )
        }
    )


def _without_redundant_content_text(server_result: dict[str, Any]) -> dict[str, Any]:
    """Retain protocol metadata while dropping text already held by the spill.

    MCP content may mix text with images and embedded resources. Removing the
    whole ``content`` field would lose media and URI metadata, while retaining
    it unchanged would persist the oversized text a second time. Only an actual
    spill calls this helper, so ordinary payloads remain byte-for-byte stable.
    """
    sanitized = copy.deepcopy(server_result)
    raw_content = sanitized.get("content")
    if not isinstance(raw_content, list):
        return sanitized

    content: list[Any] = []
    for block in raw_content:
        if not isinstance(block, dict):
            content.append(block)
            continue
        if block.get("type") == "text":
            continue
        if block.get("type") == "resource" and isinstance(block.get("resource"), dict):
            block["resource"].pop("text", None)
        content.append(block)
    sanitized["content"] = content
    return sanitized


def format_mcp_result(
    result: CallToolResult | dict[str, Any],
    tool_call_id: str = "",
    tool_name: str = "",
    context: ToolContext | None = None,
    *,
    producer_scope: str | None = None,
) -> ToolResult:
    """Flatten an MCP ``tools/call`` result into a harness ``ToolResult``.

    Content blocks are joined text with separators: text blocks pass through,
    image blocks become ``[Image: <mime>]`` placeholders ONLY when their payload
    is absent or over the inline cap; a payload that fits is forwarded as a
    real :class:`ImageContent` block the model can see. Embedded resources
    become ``[Resource: <uri>]`` plus their text when present. ``isError``
    maps to ``is_error`` (with an ``Error:`` prefix, matching the established behavior).

    ``result`` is the SDK model on every live call path; the raw-dict branch
    covers servers whose payload never went through SDK validation.

    ``producer_scope`` is the harness-stamped identity of the ACTUAL producer
    (server+tool origin plus owning session -- see
    :func:`capture_producer_scope`). When given, an image block carrying a
    valid bounded capture claim in its ``_meta`` is stamped with
    :class:`~local_operator.harness.types.CapturedFrame` provenance so
    compaction may later reclaim SUPERSEDED captures; without it, or without
    a claim, the image stays unannotated and therefore PROTECTED. The
    bookkeeping never reaches a provider: encoders build image parts from
    data/mime explicitly.
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
    images: list[ImageContent] = []
    for block in blocks:
        image, placeholder = _block_image(block)
        if placeholder is not None:
            if image is not None:
                images.append(_with_capture_claim(image, block, producer_scope))
            else:
                parts.append(placeholder)
            continue
        rendered = _dict_block_text(block) if isinstance(block, dict) else _model_block_text(block)
        if rendered is not None:
            parts.append(rendered)

    text = "\n\n".join(parts)
    if is_error:
        text = f"Error: {text}" if text else "Error"

    details: dict[str, Any] = {"server_result": server_result}
    if context is not None:
        # Keep the optional SDK isolated at this module boundary: importing the
        # established built-in spill path locally preserves config-only callers'
        # cheap ``tool_bridge`` import when MCP support is absent.
        from local_operator.tools.builtin import spill_truncate

        text, spill = spill_truncate(text, tool_name, context)
        if spill is not None:
            details.update(spill)
            if server_result is not None:
                details["server_result"] = _without_redundant_content_text(server_result)

    content: list[Content] = []
    if text:
        content.append(TextContent(text=text))
    # Images ride AFTER the text: the text is their caption, the same order a
    # local image read uses. An image-only result carries no text block at all
    # -- every provider renderer here already anticipates that shape (the
    # OpenAI tool renderer exists in part to keep image-only results from
    # flattening to an empty string).
    content.extend(images)
    if not content:
        content.append(TextContent(text=text))
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=content,
        is_error=is_error,
        details=details,
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
    annotations: dict[str, Any] | None
    if isinstance(mcp_tool, dict):
        raw_name = mcp_tool.get("name", "")
        description = mcp_tool.get("description", "") or ""
        input_schema = mcp_tool.get("inputSchema") or mcp_tool.get("input_schema") or {}
        raw_annotations = mcp_tool.get("annotations")
    else:
        raw_name = mcp_tool.name
        description = mcp_tool.description or ""
        input_schema = mcp_tool.input_schema
        raw_annotations = getattr(mcp_tool, "annotations", None)
    # The annotations ride onto the AgentTool for ONE consumer: the monitor
    # read-only evaluator, whose rule is ``annotations.readOnlyHint is True``
    # (contract §6.5). The SDK model dumps by ALIAS — the wire spelling is
    # ``readOnlyHint``; its snake-case field name would read as "no hint" and
    # refuse every monitorable tool. Absent/false stays absent/false: nothing
    # else in the harness reads the hint, and the bridge's own exec tier for
    # MCP tools is unchanged.
    if raw_annotations is None:
        annotations = None
    elif isinstance(raw_annotations, dict):
        annotations = dict(raw_annotations)
    else:
        try:
            annotations = dict(raw_annotations.model_dump(by_alias=True, exclude_none=True))
        except Exception:  # noqa: BLE001 — an unreadable hint is "no hint"
            annotations = None

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
        mcp_annotations=annotations,
        execute=call_fn,
    )
