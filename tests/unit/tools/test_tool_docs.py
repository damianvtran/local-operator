"""The ``tool://`` on-demand tool reference: renderer, resolver and drift tests.

``tool://`` is the reference half of the tool surface: schemas stay lean and
ride every request, while the detail a reader needs ONCE — per-op accepted
fields, parameter semantics — renders only when ``read tool://<name>`` asks
for it. That trade holds only if the on-demand copy cannot rot, so these tests
pin the properties that make it trustworthy:

* COMPLETENESS — every property name and enum literal on the default surface
  reaches its doc. Names, types and enum literals are NEVER truncated; that is
  the point of the mechanism, and a drop here is a wrong answer to the exact
  question the reader came with.
* DETERMINISM — a render is a pure function of the tool: two renders are
  byte-equal, including across a JSON round-trip of the schema (the shape a
  persisted/reloaded schema has).
* RESOLVER CONTRACT — a hit serves the render bytes, bare ``tool://`` lists,
  an unknown name names the set (hidden tools excluded from listings but still
  served on a direct hit), and every non-``tool://`` URL returns ``None`` so
  the chained guide/skill/mcp walker is preserved.
* NEVER RAISES — malformed schemas and a broken inventory still produce text,
  because a reader has no retry path for "renderer crashed".
* BUDGET — each doc stays under the 8 KiB soft cap (the runtime hard bound is
  ``read``'s 16 KiB internal-document shaping) and the :data:`MEASURED_TOKENS`
  ledger is current: a diff here is a deliberate, visible cost change to the
  reference surface, not an accident to wave through.

Plus one integration test against a real ``Session``, because the wiring — not
the renderer — is what a subagent inherits: ``Session.__init__`` chains the
``tool://`` resolver ahead of the knowledge resolver over the session's LIVE
inventory, and ``read`` reaches it through the turn's own ``ToolContext``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.compaction.tokens import count_text_tokens
from local_operator.harness.types import (
    AgentTool,
    StreamEndEvent,
    TextContent,
    ToolResult,
)
from local_operator.tools import builtin
from local_operator.tools.registry import DEFAULT_TOOL_NAMES, create_tools
from local_operator.tools.tool_docs import (
    SPECIAL_RENDERERS,
    TOOL_NOTES,
    ToolDocNotes,
    ToolDocOp,
    chain_tool_docs,
    is_tool_docs_link,
    make_tool_doc_resolver,
    render_tool_doc,
    unwrap_tool_docs,
)
from scripts.real_tool_surface import build_real_tools
from tests.unit.session.test_session import ScriptedStream, make_session
from tests.unit.tools.test_effort_tier_schema import THREE, write_tiers

#: Soft cap for ONE rendered doc, in characters. Enforced by TEST, never at
#: render time — names/types/enums must not be truncated to fit, so an
#: oversized doc is fixed by authoring less prose, not by clipping.
DOC_SOFT_CAP_CHARS = 8 * 1024

#: The nesting level of ``properties`` containers the renderer walks below the
#: root (design note §2). Spelled here rather than imported from the module so
#: this walker is a check ON the renderer, not a mirror of its internals —
#: :func:`test_default_surface_fits_the_documented_depth_cap` fails if the live
#: surface ever outgrows it.
_WALK_DEPTH_CAP = 2

#: Token ledger of the rendered docs, ``count_text_tokens`` under cl100k_base
#: (tiktoken is a BASE dependency, so the ruler is identical on every install).
#: When this drifts the test prints the new table; paste it back only after
#: deciding the change is worth its cost — the whole mechanism exists to make
#: reference detail opt-in, and the ledger is what keeps it honest.
#:
#: CANONICAL ARM: every doc here is rendered against an ISOLATED, empty config
#: (see :func:`hermetic_config`) — the arm CI runs. ``agent`` and ``task`` are
#: the two docs a tiers-configured ``values.subagents`` changes (they advertise
#: ``subagents.model_choice`` / ``subagents.models``); without the isolation
#: this table could only ever be green on one of the two machines. The OTHER
#: arm is pinned by :func:`test_the_tier_configured_arm_is_pinned_too`.
#:
#: RE-MEASURED 2026-10-01 by ``fix/sessions-resume-0930-9d2e`` (PR #1863,
#: folded after #1862): ``sessions`` 718 -> 738 and ``eval`` 333 -> 348 are
#: the ONLY entries that moved — the sessions per-op advertisement (plus the
#: ``help`` op's enum value) and the eval failure notice — and they were
#: re-measured through this file's own renderer, not pasted from another
#: machine. Every other entry is byte-identical to #1862's table.
MEASURED_TOKENS: dict[str, int] = {
    "agent": 687,
    "ask": 837,
    "bash": 271,
    "browser": 1050,
    "console": 1006,
    "edit": 372,
    "eval": 348,
    "glob": 94,
    "grep": 237,
    "hub": 638,
    "jobs": 339,
    "list_variables": 61,
    "lsp": 303,
    "monitor": 369,
    "network": 697,
    "patience": 361,
    "project": 967,
    "project_delete": 98,
    "read": 312,
    "read_variable": 74,
    "secret": 209,
    "send": 614,
    "sessions": 738,
    "task": 422,
    "team": 412,
    "team_delete": 93,
    "todo": 463,
    "wait": 275,
    "wake": 357,
    "web_fetch": 367,
    "web_read": 234,
    "web_search": 202,
    "write": 100,
}

#: The SAME ledger's other arm: ``agent``/``task`` rendered against a
#: tiers-configured ``values.subagents`` (the synthetic selectors the effort
#: tests already use, so the config shape lives in one place). Recorded so a
#: change to the config-sensitive branch is a visible edit here rather than a
#: machine-dependent surprise.
TIER_ARM_TOKENS: dict[str, int] = {"agent": 716, "task": 585}


async def _noop_execute(*_args: Any, **_kwargs: Any) -> ToolResult:
    raise AssertionError("tool bodies are never executed by these tests")


def _tool(
    name: str,
    *,
    label: str = "",
    description: str = "",
    parameters: Any = None,
    hidden: bool = False,
) -> AgentTool:
    return AgentTool(
        name=name,
        label=label,
        description=description or f"{name} tool",
        parameters=parameters if parameters is not None else {"type": "object", "properties": {}},
        hidden=hidden,
        execute=_noop_execute,
    )


def _text_of(result: ToolResult) -> str:
    return "".join(block.text for block in result.content if isinstance(block, TextContent))


@pytest.fixture()
def hermetic_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An ISOLATED, EMPTY config directory — the arm CI renders.

    ``agent`` and ``task`` build their docs from ``values.subagents`` config
    (``model_choice`` / ``models``, read through ``local_operator.config``), so
    a run without this renders whatever the RECORDER's machine is configured
    with: measured, the two docs are 687/422 tokens clean and 714/581 with
    tiers configured — exactly how this suite once passed on CI and failed on
    a tiers box (B2/Q-2). Empty here = the canonical arm.
    """
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    return config


@pytest.fixture()
def default_surface(hermetic_config: Path) -> list[AgentTool]:
    """The full default tool surface, exactly as the budget bench measures it.

    Reuses ``scripts/real_tool_surface.build_real_tools`` so these drift tests
    and the ``context-budget`` CI guard cannot disagree about what "the default
    surface" is: it forces the two machine-probing createIf gates (browser and
    console) deterministically ON, which is what makes a CI runner and a
    developer box measure the same 33 tools. Function-scoped (not module-scoped)
    because the builders read the config at BUILD time, so each test's
    :func:`hermetic_config` must apply to its own render; the fixture is taken
    for its env side effect alone.
    """
    tools = build_real_tools(".")
    assert {tool.name for tool in tools} == set(DEFAULT_TOOL_NAMES), (
        "the default surface changed shape (a builder is gated off, or a tool "
        "was added/removed); update this suite's expectations with it"
    )
    return tools


# ---------------------------------------------------------------------------
# An independent schema walker (the drift oracle for completeness above)
# ---------------------------------------------------------------------------


def _deref(schema: Any, root: dict[str, Any]) -> Any:
    """Resolve a local ``$ref`` against the root ``$defs``; other schemas pass."""
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            defs = root.get("$defs")
            if isinstance(defs, dict):
                return defs.get(ref.split("/")[-1], schema)
    return schema


def _nested_containers(subschema: Any, root: dict[str, Any]) -> list[dict[str, Any]]:
    """Object schemas whose properties belong one level deeper in the doc.

    A field name can hide behind three container shapes — an inline object, an
    array's ``items``, or an ``anyOf`` branch with an object arm — and all
    three are how this codebase's schemas nest their models.
    """
    resolved = _deref(subschema, root)
    if not isinstance(resolved, dict):
        return []
    candidates = [resolved]
    for branch in resolved.get("anyOf") or ():
        candidates.append(_deref(branch, root))
    if isinstance(resolved.get("items"), dict):
        candidates.append(_deref(resolved["items"], root))
    return [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and isinstance(candidate.get("properties"), dict)
        and candidate["properties"]
    ]


def _collect_enums(subschema: Any, root: dict[str, Any], enums: set[str]) -> None:
    resolved = _deref(subschema, root)
    if not isinstance(resolved, dict):
        return
    for literal in resolved.get("enum") or []:
        enums.add(str(literal))
    for branch in resolved.get("anyOf") or ():
        _collect_enums(branch, root, enums)
    if isinstance(resolved.get("items"), dict):
        _collect_enums(resolved["items"], root, enums)


def _collect_entries(
    schema: Any,
    root: dict[str, Any],
    depth: int,
    entries: dict[tuple[int, str], set[str]],
) -> None:
    """``(depth, name) -> enum literals`` the doc must carry at that position.

    ``depth`` is the CONTAINER level: root properties live at 0, and the
    renderer indents two spaces per level, so the pair pins both the name AND
    the line it must appear on — not merely that the bytes occur somewhere.
    """
    resolved = _deref(schema, root)
    if not isinstance(resolved, dict):
        return
    properties = resolved.get("properties")
    if not isinstance(properties, dict) or not properties:
        return
    for name, subschema in properties.items():
        literals = entries.setdefault((depth, str(name)), set())
        _collect_enums(subschema, root, literals)
        if depth >= _WALK_DEPTH_CAP:
            continue
        for child in _nested_containers(subschema, root):
            _collect_entries(child, root, depth + 1, entries)


def _structured_lines(doc: str, depth: int, name: str) -> list[str]:
    """The rendered parameter line(s) for ``name`` at container ``depth``.

    The renderer writes one line per property, ``{indent}- `name` (annotations)``
    with two spaces of indent per nesting level. Matching THAT form — rather
    than "the name occurs somewhere in the doc" — is what lets the oracle see
    a dropped line: a name surviving only in prose (a description quoting it,
    the Purpose paragraph) does not count (M1).
    """
    prefix = f"{'  ' * depth}- `{name}` ("
    return [line for line in doc.splitlines() if line.startswith(prefix)]


def _max_properties_depth(schema: Any, root: dict[str, Any], depth: int = 0) -> int:
    """Deepest ``properties`` container in the schema, unbounded."""
    resolved = _deref(schema, root)
    if not isinstance(resolved, dict):
        return -1
    best = -1
    properties = resolved.get("properties")
    if isinstance(properties, dict) and properties:
        best = depth
        for subschema in properties.values():
            for child in _nested_containers(subschema, root):
                best = max(best, _max_properties_depth(child, root, depth + 1))
    for branch in resolved.get("anyOf") or ():
        best = max(best, _max_properties_depth(branch, root, depth))
    if isinstance(resolved.get("items"), dict):
        best = max(best, _max_properties_depth(resolved["items"], root, depth + 1))
    return best


# ---------------------------------------------------------------------------
# Completeness (§3.1) and the depth cap it depends on
# ---------------------------------------------------------------------------


def test_every_default_surface_property_name_and_enum_literal_reaches_its_doc(
    default_surface: list[AgentTool],
) -> None:
    """No name, type or enum literal may be dropped from a rendered doc.

    This is the reason the mechanism can replace reference prose: the reader
    gets the FULL accepted surface, not a summary of it. A missing enum
    literal is a wrong answer to the exact question ``read tool://<tool>`` is
    asked; a missing property name reads as "this field does not exist".

    The name check is STRUCTURAL (M1): each collected ``(depth, name)`` pair
    must appear as the rendered parameter line at its own indentation, and
    each enum literal must appear INSIDE that line. A substring check would
    pass whenever a dropped line's name (or literal) still occurs in the
    surrounding prose — e.g. ``timeout`` inside another field's description —
    which is exactly the drift this oracle exists to catch.
    """
    for tool in default_surface:
        doc = render_tool_doc(tool)
        entries: dict[tuple[int, str], set[str]] = {}
        _collect_entries(tool.parameters or {}, tool.parameters or {}, 0, entries)
        missing_lines: list[str] = []
        missing_literals: list[str] = []
        for (depth, name), literals in sorted(entries.items()):
            lines = _structured_lines(doc, depth, name)
            rendered = f"{'  ' * depth}- `{name}` ("
            if not lines:
                missing_lines.append(rendered)
                continue
            for literal in sorted(literals):
                if not any(literal in line for line in lines):
                    missing_literals.append(f"{rendered} lacks enum literal {literal!r}")
        assert (
            not missing_lines
        ), f"{tool.name}: parameter line(s) missing from its tool:// doc: {missing_lines}"
        assert not missing_literals, (
            f"{tool.name}: enum literals missing from their parameter line(s): "
            f"{missing_literals}"
        )


def test_default_surface_fits_the_documented_depth_cap(
    default_surface: list[AgentTool],
) -> None:
    """The depth cap is a standing decision, not a silent truncation.

    If a future schema nests its models one level deeper than the renderer
    walks, the doc would silently drop those field names (and the completeness
    test above, written to the same cap, would not notice). This fails FIRST,
    by name, so the choice is explicit: trim the nesting, or raise the cap and
    the walkers together.
    """
    too_deep = {
        tool.name: _max_properties_depth(tool.parameters or {}, tool.parameters or {})
        for tool in default_surface
    }
    offenders = {name: depth for name, depth in too_deep.items() if depth > _WALK_DEPTH_CAP}
    assert not offenders, (
        f"schemas nested deeper than the {_WALK_DEPTH_CAP}-level cap: {offenders}. "
        "The doc drops those field names; raise the cap deliberately or flatten "
        "the schema."
    )


# ---------------------------------------------------------------------------
# Determinism (§3.2)
# ---------------------------------------------------------------------------


def test_render_is_deterministic_including_a_schema_json_round_trip(
    default_surface: list[AgentTool],
) -> None:
    """Two renders are byte-equal; a JSON round-tripped schema renders equal too.

    The round trip is the shape a schema has after persistence, and the one
    the drift tests compare against (``json.dumps/loads``) — an implementation
    leaning on dict identity, insertion quirks or formatting locale would pass
    the first assertion and fail this one. It also pins that the renderer does
    not mutate the tool it reads.
    """
    for tool in default_surface:
        before = json.dumps(tool.parameters, sort_keys=True)
        first = render_tool_doc(tool)
        assert render_tool_doc(tool) == first, f"{tool.name}: two renders differ"
        reloaded = tool.model_copy(deep=True)
        reloaded.parameters = json.loads(json.dumps(tool.parameters))
        assert (
            render_tool_doc(reloaded) == first
        ), f"{tool.name}: render differs after a JSON schema round-trip"
        assert (
            json.dumps(tool.parameters, sort_keys=True) == before
        ), f"{tool.name}: the renderer mutated its input schema"


# ---------------------------------------------------------------------------
# Token ledger + soft cap (§3.3)
# ---------------------------------------------------------------------------


def test_docs_stay_within_the_soft_cap_and_the_token_ledger_is_current(
    default_surface: list[AgentTool],
) -> None:
    """The 8 KiB soft cap holds, and the ledger changes are DELIBERATE.

    The cap keeps ``read``'s runtime shaping (16 KiB) dormant; the ledger
    makes a doc getting more expensive a visible, reviewed edit in the same
    diff rather than creep nobody notices. The failure prints the whole new
    table so the update is a paste, not an archaeology dig.
    """
    current: dict[str, int] = {}
    oversized: list[str] = []
    for tool in default_surface:
        doc = render_tool_doc(tool)
        if len(doc) > DOC_SOFT_CAP_CHARS:
            oversized.append(f"{tool.name}: {len(doc)} chars")
        current[tool.name] = count_text_tokens(doc)
    assert not oversized, (
        f"tool:// docs exceed the {DOC_SOFT_CAP_CHARS}-char soft cap: {oversized}. "
        "Names/types/enums are never truncated — trim authored prose instead."
    )
    if current != MEASURED_TOKENS:
        raise AssertionError(
            "The tool:// token ledger moved. Re-measure deliberately before "
            "updating MEASURED_TOKENS in this file:\n\n"
            f"MEASURED_TOKENS = {json.dumps(current, indent=4, sort_keys=True)}"
        )


def test_the_tier_configured_arm_is_pinned_too(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The config-sensitive branch is PINNED, not ignored (B2/Q-2).

    ``agent``/``task`` advertise ``values.subagents.models`` and
    ``subagents.model_choice``, so their docs move with the operator's config —
    the non-hermeticity that once made the canonical ledger green on one
    machine and red on the other. The canonical table above renders the CLEAN
    arm under :func:`hermetic_config`; this test renders the tiers arm against
    a synthetic config and pins its sizes too, so a change to EITHER branch is
    a visible edit rather than a machine-dependent surprise.
    """
    config = tmp_path / "tier-config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_tiers(config, THREE)
    tools = {tool.name: tool for tool in build_real_tools(".")}
    measured = {
        "agent": count_text_tokens(render_tool_doc(tools["agent"])),
        "task": count_text_tokens(render_tool_doc(tools["task"])),
    }
    assert measured == TIER_ARM_TOKENS, (
        f"The tier-configured rendering moved: {measured}. Re-measure "
        "deliberately and update TIER_ARM_TOKENS."
    )
    # The branch is real: both docs differ from the clean arm's bytes.
    assert measured["agent"] != MEASURED_TOKENS["agent"]
    assert measured["task"] != MEASURED_TOKENS["task"]


def test_authored_notes_and_renderers_name_real_tools(
    default_surface: list[AgentTool],
) -> None:
    """``TOOL_NOTES``/``SPECIAL_RENDERERS`` keys must be real tool names.

    Both are module-level mappings consulted at resolve time, so a typo in a
    key would not crash anything — the entry would simply never be used, and
    the doc would silently miss its authored ops. This makes the typo loud.
    """
    names = {tool.name for tool in default_surface}
    assert set(TOOL_NOTES) <= names, f"TOOL_NOTES keys not real: {sorted(set(TOOL_NOTES) - names)}"
    assert (
        set(SPECIAL_RENDERERS) <= names
    ), f"SPECIAL_RENDERERS keys not real: {sorted(set(SPECIAL_RENDERERS) - names)}"


# ---------------------------------------------------------------------------
# Authored attachments: ops table + notes hook
# ---------------------------------------------------------------------------


def test_authored_ops_and_notes_render_in_the_documented_shape() -> None:
    """The pilot seam: ``ToolDocNotes`` renders as the per-op table (§2.3)."""
    notes = ToolDocNotes(
        ops=(
            ToolDocOp(op="list", blurb="Show things.", fields=("id", "owner")),
            ToolDocOp(op="delete", fields=("id",)),
        ),
        notes="Only the owner may delete.",
    )
    doc = render_tool_doc(_tool("widget"), notes=notes)
    assert "## Ops" in doc
    assert "- list: Show things. — fields: id, owner" in doc
    assert "- delete — fields: id" in doc
    assert "## Notes" in doc
    assert "Only the owner may delete." in doc


def test_special_renderer_wins_and_a_broken_one_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registered renderer serves ``tool://<name>``; a raising one degrades
    to the generic doc instead of an error result — the seam's whole contract
    is that the sessions pilot can pin byte parity without the reader ever
    losing the fallback."""
    tool = _tool("widget")
    monkeypatch.setitem(SPECIAL_RENDERERS, "widget", lambda _tool: "SPECIAL BYTES")
    resolver = make_tool_doc_resolver(lambda: [tool])
    assert resolver("tool://widget") == "SPECIAL BYTES"

    def _boom(_tool: AgentTool) -> str:
        raise RuntimeError("renderer exploded")

    monkeypatch.setitem(SPECIAL_RENDERERS, "widget", _boom)
    fallback = resolver("tool://widget")
    assert fallback == render_tool_doc(tool)


# ---------------------------------------------------------------------------
# Resolver contract (§3.4, §3.6)
# ---------------------------------------------------------------------------


def test_resolver_serves_a_hit_with_the_render_bytes() -> None:
    tool = _tool("bash", label="Shell", description="Run a shell command.")
    resolver = make_tool_doc_resolver(lambda: [tool])
    doc = resolver("tool://bash")
    assert doc == render_tool_doc(tool)
    # The name is URL-decoded, mirroring ``skills/api.py::_url_name``: a tool
    # whose name needs escaping must stay addressable.
    assert resolver("tool://ba%73h") == doc


def test_bare_tool_url_lists_visible_tools_in_sorted_order() -> None:
    resolver = make_tool_doc_resolver(
        lambda: [_tool("zeta"), _tool("alpha"), _tool("hidden-one", hidden=True)]
    )
    text = resolver("tool://")
    assert text == ("Tool URL missing a name: expected tool://<name>\nAvailable tools: alpha, zeta")
    assert "hidden-one" not in text


def test_unknown_name_names_the_available_set() -> None:
    resolver = make_tool_doc_resolver(lambda: [_tool("zeta"), _tool("alpha", hidden=True)])
    text = resolver("tool://bogus")
    assert text == "Unknown tool: bogus\nAvailable: zeta"
    assert "alpha" not in text


def test_a_direct_hit_on_a_hidden_tool_still_serves() -> None:
    """Hidden tools remain callable; a reader who knows the name is asking
    precisely because it is not listed."""
    hidden = _tool("hidden-one", hidden=True)
    resolver = make_tool_doc_resolver(lambda: [hidden])
    assert resolver("tool://hidden-one") == render_tool_doc(hidden)


def test_non_tool_urls_return_none_so_the_chain_is_preserved() -> None:
    resolver = make_tool_doc_resolver(lambda: [_tool("bash")])
    for url in ("skill://alpha", "guide://beta", "mcp://server/tool", "https://example.com"):
        assert resolver(url) is None, url


def test_resolver_never_raises_when_the_inventory_itself_fails() -> None:
    def broken() -> list[AgentTool]:
        raise RuntimeError("inventory down")

    resolver = make_tool_doc_resolver(broken)
    text = resolver("tool://bash")
    assert isinstance(text, str)
    assert "Tool reference unavailable" in text
    # The scheme check runs FIRST: a foreign URL never touches the inventory.
    assert resolver("skill://alpha") is None


@pytest.mark.parametrize(
    "parameters",
    [
        None,
        {},
        {"properties": None},
        {"properties": {}},
        {"properties": {"x": "not-a-schema"}},
        {"properties": {"x": {"type": "array"}}},
        {"properties": {"x": {"$ref": "#/$defs/Missing"}}},
        {"properties": {"x": {"anyOf": []}}},
        # A self-referential $defs graph must terminate (the depth cap bounds
        # the walk) rather than recurse forever.
        {
            "$defs": {"A": {"properties": {"y": {"$ref": "#/$defs/A"}}}},
            "properties": {"x": {"anyOf": [{"$ref": "#/$defs/A"}]}},
        },
    ],
)
def test_render_never_raises_on_malformed_schemas(parameters: Any) -> None:
    doc = render_tool_doc(_tool("widget", parameters=parameters))
    assert isinstance(doc, str)
    assert "# Tool: `widget`" in doc


# ---------------------------------------------------------------------------
# The chain: ``tool://`` in front, everything else untouched
# ---------------------------------------------------------------------------


def test_chain_serves_tool_urls_ahead_of_the_base_walker() -> None:
    calls: list[str] = []

    def base(url: str) -> str | None:
        calls.append(url)
        return {"skill://s": "SKILL", "guide://g": "GUIDE"}.get(url)

    tools = [_tool("bash")]
    chained = chain_tool_docs(base, lambda: tools)
    assert chained("tool://bash") == render_tool_doc(tools[0])
    assert calls == [], "tool:// must never reach the base walker"
    assert chained("skill://s") == "SKILL"
    assert chained("mcp://m/t") is None
    assert calls == ["skill://s", "mcp://m/t"]
    # The link is ID-BEARING (B1/Q-1): the host-field parity guard recognises
    # the wrapper and recovers the host resolver through it instead of failing
    # an identity check that a plain closure could not satisfy.
    assert is_tool_docs_link(chained)
    assert unwrap_tool_docs(chained) is base
    assert is_tool_docs_link(base) is False
    assert unwrap_tool_docs(base) is None


def test_chain_still_answers_tool_urls_with_no_base_configured() -> None:
    """A session constructed with no knowledge resolver still serves its own
    tool docs; every other scheme degrades to None exactly as before."""
    solo = chain_tool_docs(None, lambda: [_tool("bash")])
    assert solo("tool://bash") == render_tool_doc(_tool("bash"))
    assert solo("skill://s") is None
    # The seam reports the (absent) base honestly: a link with no host
    # resolver unwraps to None, so the parity guard still REJECTS it for a
    # host value — that is the drop class the guard must keep seeing.
    assert is_tool_docs_link(solo)
    assert unwrap_tool_docs(solo) is None


# ---------------------------------------------------------------------------
# Integration: the Session wiring a subagent inherits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_wrapper_serves_read_tool_urls_over_live_inventory(tmp_path) -> None:
    """Exercise the real path: Session -> ToolContext -> read tool.

    The wrapper is installed in ``Session.__init__`` over a live-inventory
    lambda, and the turn's own ``ToolContext`` is what hands it to ``read`` —
    so this drives the same objects a real session does, not a test stub
    standing in for them. The refresh half pins the lambda (a snapshot would
    keep serving the construction-time list forever).
    """
    bash_tool = _tool("bash", label="Shell", description="Run a shell command.")
    session = make_session(
        tmp_path,
        ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[bash_tool],
    )
    try:
        resolver = session._skill_resolver
        assert resolver is not None
        context = session._build_tool_context()
        assert context.resolve_internal_url is resolver
        read = {tool.name: tool for tool in create_tools(context)}["read"]

        result = await read.execute("call-1", {"path": "tool://bash"}, None, None, context)
        assert result.is_error is False
        assert _text_of(result) == render_tool_doc(bash_tool)
        assert result.details is not None
        assert result.details["url"] == "tool://bash"

        # ``refresh_tools`` rebinds ``self._tools``; the resolver must follow.
        session.refresh_tools([_tool("zsh", description="Other.")])
        zsh_doc = resolver("tool://zsh")
        assert zsh_doc is not None and "# Tool: `zsh`" in zsh_doc
        bash_doc = resolver("tool://bash")
        assert bash_doc is not None and bash_doc.startswith("Unknown tool: bash")

        # This session configured NO knowledge resolver: every other scheme
        # returns None, exactly as before the wrapper existed.
        assert resolver("skill://anything") is None
    finally:
        await session.dispose()


def test_read_description_advertises_the_tool_scheme() -> None:
    """The cue lives in system.md; the read description carries only the bare
    scheme list — and ``tool://`` must be in it, or the model never learns the
    address from the tool it would read the doc with."""
    description = builtin.build_read_tool().description
    assert "tool://" in description
