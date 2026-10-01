"""On-demand tool documentation for ``tool://`` reads.

Every core tool ships its full JSON Schema on **every** API request, in the
same prompt-cache prefix as the system prompt, whether or not the tool is ever
called (``AGENTS.md``'s tool-surface footprint ladder). That is the right
default for DISPATCH — the model needs schemas to call tools at all — but it
is a poor home for REFERENCE detail: per-op acceptance rules, field semantics
and failure modes are read once and then re-sent on every message.

This module makes that detail readable on demand instead. ``read
tool://<name>`` renders one tool's purpose, its authored per-op notes, and a
recursive walk of its parameters — names, types, enum literals and defaults
verbatim; ``read tool://`` lists the tools this session actually holds. The
bytes cost nothing until a reader asks for them, and the resolver is a pure
read: it never routes through the tool's ``execute`` (approval tiers, side
effects) and never changes dispatch.

Contract, mirroring ``skills/api.py`` so the chain behaves identically:

* :func:`make_tool_doc_resolver` returns ``None`` for every non-``tool://``
  URL and never raises; an unknown name is served AS CONTENT naming the
  available set, which is the model's one-round self-correction path.
* :func:`chain_tool_docs` returns a :class:`ToolDocsLink` — an ID-BEARING
  wrapper, not a bare closure — because ``Session.__init__`` installs it in
  place of the host's resolver, and the host-field parity guard must still be
  able to recognise the wrapper and recover the host value from it
  (:func:`is_tool_docs_link` / :func:`unwrap_tool_docs`).
* Property names, types and enum literals are NEVER truncated — they are the
  point of the mechanism. Prose is elided only at authoring time.
* Rendering is deterministic: a pure function of (name, label, description,
  parameters, notes) — no timestamps or ids, sorted listings, insertion-order
  properties.
* The soft 8 KiB/doc cap is enforced by TEST, never at render time; the
  runtime hard bound stays ``read``'s 16 KiB internal-document shaping.

Authored attachments live in module-level mappings rather than a new
``AgentTool`` field — a field would ride the provider tools array this
mechanism exists to shrink. :data:`TOOL_NOTES` carries per-tool notes and ops;
:data:`SPECIAL_RENDERERS` (see :func:`register_tool_doc_renderer`) lets a
surface that must be byte-identical to another view of the same information —
the sessions pilot's ``op='help'`` — register its own pure renderer here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard
from urllib.parse import unquote, urlsplit

from local_operator.harness.types import AgentTool

#: The URL scheme this module serves. The resolver is chained AHEAD of the
#: guide/skill/mcp walker in ``Session.__init__`` (and, structurally, in every
#: subagent — each constructs its own Session and wraps it over its own live
#: inventory), so a scheme that is not ``tool://`` costs the chain one
#: ``startswith`` and reaches the base walker exactly as before.
TOOL_DOC_SCHEME = "tool://"

#: Nested property levels rendered BELOW the root properties. 2 covers every
#: schema on the default surface today (deepest: ``ask`` — questions[] ->
#: options[] fields — plus one nested model on edit/todo/task/project). The
#: drift test fails if the live surface needs a third level, so a future nested
#: model forces an explicit cap decision instead of silently dropping its
#: field names from every reference.
_PARAM_DEPTH_CAP = 2

#: Section headings, as constants so the drift tests can assert against the
#: same strings the renderer writes rather than re-spelling them.
_PURPOSE_HEADING = "## Purpose"
_OPS_HEADING = "## Ops"
_PARAMS_HEADING = "## Parameters"
_NOTES_HEADING = "## Notes"


@dataclass(frozen=True)
class ToolDocOp:
    """One op in an authored per-op table.

    The sessions pilot's ``op='help'`` renders exactly this table, which is the
    only place the per-op accepted-field contract can live for op-based tools
    whose fields all sit in one flat schema union; ``blurb`` and ``fields`` are
    both optional so a pilot that only names its ops is served too.
    """

    op: str
    blurb: str = ""
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolDocNotes:
    """Authored attachments for one tool's reference.

    ``ops`` renders as the per-op table (op, blurb, accepted fields); ``notes``
    renders as a trailing prose section. Both optional: a tool with no entry in
    :data:`TOOL_NOTES` gets the generic render.
    """

    ops: tuple[ToolDocOp, ...] = ()
    notes: str = ""


#: Authored notes, keyed by tool name. Deliberately a MODULE-LEVEL mapping and
#: not a new ``AgentTool`` field: the field would serialize into every
#: provider request's tools array — the cost this mechanism exists to reduce —
#: while this mapping costs nothing until a ``tool://`` read looks it up. A
#: drift test flags a key that is not a real tool name. Promote to a field
#: only if it ever grows beyond reference prose.
TOOL_NOTES: dict[str, ToolDocNotes] = {}

#: Tool-specific renderers for surfaces that must be BYTE-IDENTICAL to another
#: view of the same information (the sessions pilot's ``op='help'``). Consulted
#: at RESOLVE time, so registration order does not matter. A renderer's
#: contract: deterministic, no raises — a raising renderer degrades to the
#: generic render below, never to an error result.
SPECIAL_RENDERERS: dict[str, Callable[[AgentTool], str]] = {}


def register_tool_doc_renderer(name: str, fn: Callable[[AgentTool], str]) -> None:
    """Register ``fn`` as the renderer for ``tool://<name>`` (last call wins)."""
    SPECIAL_RENDERERS[name] = fn


def render_tool_doc(tool: AgentTool, *, notes: ToolDocNotes | None = None) -> str:
    """The full reference document for one tool.

    Pure and never raises; on an unreadable schema it degrades to
    :func:`_fallback` — name, description and a plain "unavailable" note —
    because a reader has no retry path for "renderer crashed". ``notes=None``
    looks the tool's entry up in :data:`TOOL_NOTES`; pass an explicit value to
    render candidate notes that are not registered yet (tests).
    """
    if notes is None:
        notes = TOOL_NOTES.get(tool.name)
    try:
        return _render(tool, notes)
    except Exception:  # noqa: BLE001 — the public contract is "never raises"
        return _fallback(tool)


def _render(tool: AgentTool, notes: ToolDocNotes | None) -> str:
    """Section order per the design note §2: title, purpose, ops, parameters,
    authored notes, re-read footer."""
    lines = [_title(tool)]
    description = (tool.description or "").strip()
    if description:
        lines += ["", _PURPOSE_HEADING, "", description]
    if notes is not None and notes.ops:
        lines += ["", _OPS_HEADING, ""]
        lines += [_op_line(op) for op in notes.ops]
    parameters = tool.parameters if isinstance(tool.parameters, dict) else {}
    param_lines: list[str] = []
    _render_properties(parameters, root=parameters, depth=0, out=param_lines)
    if param_lines:
        lines += ["", _PARAMS_HEADING, ""]
        lines += param_lines
    extras = (notes.notes if notes is not None else "").strip()
    if extras:
        lines += ["", _NOTES_HEADING, "", extras]
    lines += ["", f"Read again: `{TOOL_DOC_SCHEME}{tool.name}`"]
    return "\n".join(lines)


def _fallback(tool: AgentTool) -> str:
    head = f"# Tool: `{tool.name}`"
    description = (tool.description or "").strip()
    parts = [head]
    if description:
        parts += ["", description]
    parts += ["", "(tool reference unavailable)"]
    return "\n".join(parts)


def _title(tool: AgentTool) -> str:
    title = f"# Tool: `{tool.name}`"
    label = (tool.label or "").strip()
    # A label that repeats the name in different case ("Read" for ``read``)
    # says nothing the title does not; only a genuinely different label
    # ("Shell", "Agent roles", "Mesh network") earns its bytes.
    if label and label.lower() != tool.name.lower():
        title += f" — {label}"
    return title


def _op_line(op: ToolDocOp) -> str:
    line = f"- {op.op}"
    blurb = " ".join(op.blurb.split())
    if blurb:
        line += f": {blurb}"
    if op.fields:
        line += f" — fields: {', '.join(op.fields)}"
    return line


def _render_properties(
    schema: dict[str, Any],
    *,
    root: dict[str, Any],
    depth: int,
    out: list[str],
    indent: str = "",
) -> None:
    """Walk one ``properties`` container, one line per property.

    Property order is the schema's insertion order (pydantic emits declaration
    order, which the tools array already depends on for prompt-cache
    stability), so the doc reads in the same order as the schema the model
    dispatches against.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    required = schema.get("required")
    required_names = set(required) if isinstance(required, list) else set()
    for name, subschema in properties.items():
        if not isinstance(name, str):
            continue
        out.append(
            _property_line(
                name,
                subschema,
                root=root,
                required=name in required_names,
                indent=indent,
            )
        )
        if depth >= _PARAM_DEPTH_CAP:
            continue
        for child in _child_schemas(subschema, root):
            _render_properties(child, root=root, depth=depth + 1, out=out, indent=indent + "  ")


def _property_line(
    name: str,
    subschema: Any,
    *,
    root: dict[str, Any],
    required: bool,
    indent: str,
) -> str:
    annotations = [_type_of(subschema, root)]
    if required:
        annotations.append("required")
    literals = _enum_literals(subschema, root)
    if literals:
        annotations.append("enum: " + " | ".join(str(value) for value in literals))
    resolved = _resolved(subschema, root)
    if isinstance(resolved, dict) and "default" in resolved:
        annotations.append("default: " + _format_default(resolved["default"]))
    line = f"{indent}- `{name}` ({', '.join(annotations)})"
    description = ""
    if isinstance(resolved, dict):
        raw = resolved.get("description")
        if isinstance(raw, str):
            description = " ".join(raw.split())
    if description:
        line += f": {description}"
    return line


def _child_schemas(subschema: Any, root: dict[str, Any]) -> list[dict[str, Any]]:
    """Object schemas whose properties belong one level deeper in the doc.

    A property's fields can arrive through three shapes in these schemas:
    directly (an inline object), through ``items`` (an array of a ``$defs``
    model — ``edits``/``questions``/``milestones``), or through an ``anyOf``
    branch (a union whose object arm carries fields). All three are walked so
    a field name can never hide behind the shape of its container.
    """
    children: list[dict[str, Any]] = []
    for candidate in _object_candidates(subschema, root):
        if isinstance(candidate.get("properties"), dict) and candidate["properties"]:
            children.append(candidate)
    return children


def _object_candidates(subschema: Any, root: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(subschema, dict):
        return []
    resolved = _resolved(subschema, root)
    if not isinstance(resolved, dict):
        return []
    candidates = [resolved]
    for branch in resolved.get("anyOf") or ():
        branch_schema = _resolved(branch, root)
        if isinstance(branch_schema, dict):
            candidates.append(branch_schema)
    items = resolved.get("items")
    if isinstance(items, dict):
        item_schema = _resolved(items, root)
        if isinstance(item_schema, dict):
            candidates.append(item_schema)
    return candidates


def _resolved(subschema: Any, root: dict[str, Any]) -> Any:
    """Dereference a local ``$ref`` against the root ``$defs``; non-refs pass
    through unchanged. A dangling ref resolves to itself, which the walkers
    treat as a schema with nothing to add rather than an error."""
    if not isinstance(subschema, dict):
        return subschema
    ref = subschema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        defs = root.get("$defs")
        target = defs.get(ref[len("#/$defs/") :]) if isinstance(defs, dict) else None
        if isinstance(target, dict):
            return target
    return subschema


def _type_of(subschema: Any, root: dict[str, Any]) -> str:
    if not isinstance(subschema, dict):
        return "any"
    ref = subschema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        # Name the model, not "object": "array of EditHunk" is the useful
        # answer to "what goes in this list".
        return ref[len("#/$defs/") :]
    any_of = subschema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        parts: list[str] = []
        for branch in any_of:
            part = _type_of(branch, root)
            if part not in parts:
                parts.append(part)
        return " | ".join(parts)
    declared = subschema.get("type")
    if isinstance(declared, str):
        if declared == "array":
            items = subschema.get("items")
            if isinstance(items, dict):
                return f"array of {_type_of(items, root)}"
            return "array"
        return declared
    return "any"


def _enum_literals(subschema: Any, root: dict[str, Any]) -> list[Any]:
    """Enum literals attached to a property, through unions and item schemas.

    Collected recursively because the same JSON Schema expresses "one of these
    values" three ways in this codebase: a direct ``enum``, an ``enum`` on one
    ``anyOf`` branch (nullable enums), and an ``enum`` on ``items``. All three
    reach the doc — a dropped literal is a wrong answer to the exact question
    the reader came with.
    """
    if not isinstance(subschema, dict):
        return []
    resolved = _resolved(subschema, root)
    if not isinstance(resolved, dict):
        return []
    literals: list[Any] = []
    values = resolved.get("enum")
    if isinstance(values, list):
        literals.extend(values)
    for branch in resolved.get("anyOf") or ():
        literals.extend(_enum_literals(branch, root))
    items = resolved.get("items")
    if isinstance(items, dict):
        literals.extend(_enum_literals(items, root))
    return literals


def _format_default(value: Any) -> str:
    """JSON-literal rendering, so ``false``/``null``/``""`` read as the schema
    spells them rather than Python's spellings."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _url_name(url: str) -> str:
    """The decoded name a ``tool://`` URL addresses, or ``""``.

    Netloc is where the name lives (mirrors ``skills/api.py::_url_name``); a
    malformed URL is simply not a name rather than an error. The name is never
    case-folded: tool names are lowercase, and a case-mismatch answer that
    names the real tool is a better self-correction than a silent hit.
    """
    try:
        return unquote(urlsplit(url).netloc)
    except Exception:  # noqa: BLE001 — a malformed URL is simply not a name
        return ""


def make_tool_doc_resolver(
    inventory: Callable[[], Sequence[AgentTool]],
) -> Callable[[str], str | None]:
    """Build the ``tool://`` adapter for the ``read`` tool.

    Contract (mirrors ``skills/api.py``): returns content for ``tool://`` URLs,
    ``None`` for every other scheme (the caller chains other resolvers), and
    never raises — so a reviewer can reason about this link exactly as they do
    about the skill and guide links it is chained with.

    ``inventory`` is a CALLABLE, not a snapshot: a session's tool list is
    rebound mid-session (an MCP enable, a declared inventory narrowing, the
    prune in ``harness.subagent``), and the reader must see the list that is
    live at READ time, not the one that existed when the resolver was built.
    A hidden tool is omitted from LISTINGS but still served on a direct hit —
    hidden tools remain callable (``prompts_api``), and a reader who knows the
    name is asking precisely because it is not listed.
    """

    def resolver(url: str) -> str | None:
        if not url.startswith(TOOL_DOC_SCHEME):
            return None
        try:
            return _resolve_tool_url(url, inventory)
        except Exception as exc:  # noqa: BLE001 — the resolver contract is "never raises"
            return f"Tool reference unavailable: {exc}"

    return resolver


def _resolve_tool_url(url: str, inventory: Callable[[], Sequence[AgentTool]]) -> str:
    tools = list(inventory())
    name = _url_name(url)
    available = ", ".join(sorted(tool.name for tool in tools if not tool.hidden)) or "(none)"
    if not name:
        # Bare ``tool://`` is the discovery door, mirroring ``read skill://``:
        # the listing is what tells the reader which names exist at all.
        return f"Tool URL missing a name: expected tool://<name>\nAvailable tools: {available}"
    tool = next((candidate for candidate in tools if candidate.name == name), None)
    if tool is None:
        # Served AS CONTENT (like ``Unknown skill``): the available set is the
        # model's self-correction path, in one tool round.
        return f"Unknown tool: {name}\nAvailable: {available}"
    renderer = SPECIAL_RENDERERS.get(name)
    if renderer is not None:
        try:
            return renderer(tool)
        except Exception:  # noqa: BLE001 — a broken special renderer must not
            # cost the reader the generic reference entirely; fall through to
            # it. The special renderer's own drift test pins the byte parity.
            pass
    return render_tool_doc(tool, notes=TOOL_NOTES.get(name))


class ToolDocsLink:
    """The resolver link :func:`chain_tool_docs` installs — id-bearing on purpose.

    A plain closure would work identically at CALL time, and this class exists
    for the one consumer that must not treat it as identical: the host-field
    parity guard (``tests/unit/session/test_tool_context_parity.py``) asserts
    by IDENTITY that every value the host hands ``Session.__init__`` still
    reaches the executor, and the session legitimately replaces the host's
    resolver with this link (``Session.__init__`` chains ``tool://`` ahead of
    it). That guard may keep its drop-detection only if the wrapper is
    RECOGNISABLE and the host value is RECOVERABLE from it —
    :func:`is_tool_docs_link` / :func:`unwrap_tool_docs` are the seam, and the
    guard accepts the wrapper in place of identity ONLY when unwrapping
    recovers the host's own object. Do not collapse this back into a closure
    without carrying that permission somewhere else.
    """

    __slots__ = ("_base", "_tool_resolver")

    def __init__(
        self,
        base: Callable[[str], str | None] | None,
        tool_resolver: Callable[[str], str | None],
    ) -> None:
        self._base = base
        self._tool_resolver = tool_resolver

    def __call__(self, url: str) -> str | None:
        if url.startswith(TOOL_DOC_SCHEME):
            handled = self._tool_resolver(url)
            if handled is not None:
                return handled
        base = self._base
        return base(url) if base is not None else None


def is_tool_docs_link(value: object) -> TypeGuard[ToolDocsLink]:
    """True for a resolver link :func:`chain_tool_docs` built.

    Exported for consumers that need to tell a tool-docs wrapper apart from
    any other callable (the parity guard's acceptance path); everything else
    should just call the resolver. A ``TypeGuard``, so ``assert
    is_tool_docs_link(x)`` narrows ``x`` for the type checker as well as the
    runtime — the guard's call sites depend on that.
    """
    return isinstance(value, ToolDocsLink)


def unwrap_tool_docs(value: object) -> Callable[[str], str | None] | None:
    """The host resolver a :class:`ToolDocsLink` wraps, or ``None``.

    ``None`` for anything that is not a link (including a dropped field's
    ``None``) AND for a link wrapping no base; a caller that must tell those
    two apart checks :func:`is_tool_docs_link` first.
    """
    return value._base if isinstance(value, ToolDocsLink) else None


def chain_tool_docs(
    base: Callable[[str], str | None] | None,
    inventory: Callable[[], Sequence[AgentTool]],
) -> Callable[[str], str | None]:
    """Put the ``tool://`` resolver AHEAD of ``base``; every other scheme
    reaches ``base`` exactly as before, and with no base configured the
    ``tool://`` link still answers (a session with no knowledge resolver has
    nowhere else to route its own tool docs to).

    ``base`` is the guide->skill->mcp walker the session factory composes;
    this wrapper is installed in ``Session.__init__`` so every session — root
    or subagent, each with its own live inventory — answers ``tool://`` without
    touching the factory chain or the subagent wiring. Returns a
    :class:`ToolDocsLink` rather than a bare closure so the identity-sensitive
    parity guard can recognise it (see the class docstring).
    """
    return ToolDocsLink(base, make_tool_doc_resolver(inventory))
