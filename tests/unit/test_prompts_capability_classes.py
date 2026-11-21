"""Capability-class prose gating (``CAPABILITY_CLASS_FLAGS``).

A session ships prose about files, shell, readable resources and web tools only
when it holds a tool of that class. The two guarantees pinned here are the ones
the change rests on: a session holding the interactive tool set renders byte-
identical to the template's unconditional (pre-gate) output, and every
combination of the flags that share a line joins its sentences cleanly.

Kept in its own file so the all-combination test stays separate from
``test_prompts_api.py``'s per-tool combination test.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

import pytest

from local_operator.harness.intent import apply_intent_schema
from local_operator.harness.types import AgentTool
from local_operator.prompts_api import (
    CAPABILITY_CLASS_FLAGS,
    CHANNEL_ASK,
    CHANNEL_HUB,
    TOOL_GATED_PROSE_FLAGS,
    build_system_blocks,
    render_template,
    split_tail_sections,
)
from local_operator.session_factory import _env_details

_HOST = {"has_browser": False, "no_browser": False, "has_console": False, "no_console": False}

#: Every tool any gated flag names, i.e. a session for which nothing is gated.
_EVERY_GATED_TOOL = sorted(
    {name for names in TOOL_GATED_PROSE_FLAGS.values() for name in names}
    | {name for names in CAPABILITY_CLASS_FLAGS.values() for name in names}
)

#: The benchmark configuration (``lop exec --tools mcp__bd_search,mcp__bd_fetch``).
_MCP_ONLY = ["mcp__bd_fetch", "mcp__bd_search"]

_KNOWLEDGE = (
    "<guides>\n- delivery: Allocate bounded engineering delivery.\n</guides>\n\n"
    "<mcps>Find MCP tools: `mcp://?search=terms`; list: `mcp://`.</mcps>"
)

#: Prose that needs a class the MCP-only session lacks, one marker per sentence.
_CLASS_GATED_MARKERS = (
    "Hand-writing an app's file format?",
    "Deliverables surface themselves.",
    "Edit, don't rewrite.",
    "Treat unknown files as the user's work",
    "(`guide://credentials`)",
    "Never delete or move a path you did not create",
    "Guardrail rules: `<config_dir>/rules/*.md`",
    "Prefer the most specific tool",
    "`read tool://` lists tools",
    "`spill://` handle",
    "`scratchpad://`",
    "### Resources",
    "`<guides>` lists Local Operator procedures",
    "`web_search`/`web_fetch`",
    "delegate self-contained slices",
    # Wave 5: workspace-, shell- and subagent-specific prose.
    "This harness is Local Operator",
    "Trust code and guides, not memory.",
    "Act directly with tools: shell commands, file reads/edits",
    "Run commands, read files, search the workspace",
    "Exercise behaviour changes through the real path",
    "**Reuse existing patterns**",
    "(bang-mode)",
    "Respect denials of a prompted write or command",
    "`preview_tool_call`",
    "`reviewer`, `architect`, or `designer` subagent",
)

#: The operator's always-on list: judgement and safety, never capability prose.
_ALWAYS_ON_MARKERS = (
    "You are Local Operator, a personal assistant",
    "**Use tools before answering.**",
    "**Verify results.**",
    "**Be concise.**",
    "**Do real work fully.**",
    "**Recover, don't stop.**",
    "**Look up off-machine answers.**",
    "## Narration",
    "Destructive or irreversible operations",
    "If an approval request is declined, stop that action and say so.",
    "Keep secrets secret: never print credentials, tokens or keys",
    # Both arms (the full paragraph and the one-line form) open with this.
    "`BLOCKED by guardrail` is final for that effect",
    "Deciding is your job",
    "Authorization never extends to a destructive or irreversible step",
    "## Session state updates",
    "[session-state] records are host context, not a task or permission to act.",
)


def _tool(name: str, *, intent: bool = True) -> AgentTool:
    """A tool in its PRODUCTION shape: every builtin and MCP tool passes
    ``apply_intent_schema``, which ``has_intent`` is derived from."""

    async def _noop(*_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("not called")

    return AgentTool(
        name=name,
        description=f"{name} tool",
        parameters=apply_intent_schema({}) if intent else {},
        execute=_noop,
    )


def _blocks(names: list[str], skills_block: str = _KNOWLEDGE, **kwargs: Any) -> list[str]:
    kwargs.setdefault("host_has_browser", True)
    kwargs.setdefault("host_has_console", True)
    return build_system_blocks(
        [_tool(name) for name in names],
        skills_block=skills_block,
        env_details="env",
        date_str="2026-01-01",
        **kwargs,
    )


def test_a_session_holding_every_class_renders_the_ungated_bytes() -> None:
    """The acceptance gate: block 0 is the template with no flag stated (absent
    means True, the historical unconditional render), block 1 keeps its absence
    notes, and the tail is the knowledge block unchanged."""
    head, inventory, _env, tail = _blocks(_EVERY_GATED_TOOL, interactive=False)
    plain = render_template("system.md", dict(_HOST))
    stated = render_template(
        "system.md", {**_HOST, **{flag: True for flag in CAPABILITY_CLASS_FLAGS}}
    )
    assert plain == stated
    # Wave 5 (g) is the ONLY full-tool-set delta: the template render is
    # unchanged and the session-state section follows it after one blank line.
    assert head.startswith(plain.rstrip("\n") + "\n\n## Session state updates\n\n")
    for marker in _CLASS_GATED_MARKERS:
        assert marker in head, marker
    assert "your web search tools" not in head
    assert tail.startswith(_KNOWLEDGE + "\n\n<interactivity>")
    # Role-restricted notes stay (the tool set has bash but not every host tool
    # is absent: build a set without browser/console to see them).
    restricted = _blocks([n for n in _EVERY_GATED_TOOL if n not in {"browser", "console"}])[1]
    assert "This session was not given the browser tool." in restricted
    assert "This session was not given the `console` tool." in restricted
    assert inventory.startswith("## Available tools")


def test_mcp_only_session_drops_class_prose_and_keeps_the_always_on_rules() -> None:
    head, inventory, _env, tail = _blocks(_MCP_ONLY, interactive=False)
    for marker in _CLASS_GATED_MARKERS:
        assert marker not in head, marker
    for marker in _ALWAYS_ON_MARKERS:
        assert marker in head, marker
    assert "use\n  your web search tools for third-party errors" in head
    assert "- **Parallelize independent work:** batch calls.\n" in head
    assert "never print credentials, tokens or keys.\n" in head
    # Wave 5's short arms, each in place of the workspace prose it replaces.
    assert "\nAct directly with your tools; use them before answering from memory.\n\n" in head
    assert "- **Use tools before answering.** Start shallow, deepen on signals" in head
    assert "- **Verify results.** Read actual responses; errors aren't success." in head
    assert "Claim success only when observable" in head
    assert "- `BLOCKED by guardrail` is final for that effect: report it; never reroute.\n" in head
    assert "resolve the question yourself via\ntools or documents, then act." in head
    # The detached no-channel ``<interactivity>`` body says "proceed and state
    # what you would have asked"; the head's no-ask line is not repeated.
    assert "No `ask` tool in this session" not in head
    assert "Prefer to PROCEED with what you have" in tail
    # MCP tools carry the injected ``i`` property, so its rule stays.
    assert "Most tools take `i`" in head
    assert "Never restate the\ncall's `i` intent" in head
    # No shell: nothing to install a browser engine or terminal emulator with.
    assert inventory == "## Available tools\n\n- mcp__bd_fetch\n- mcp__bd_search"
    # No read: the catalogues point at URLs this session cannot open.
    for marker in ("<guides>", "<skills", "<mcps>", "mcp://", "guide://"):
        assert marker not in tail, marker
    assert tail.startswith("<interactivity>")
    assert split_tail_sections(tail) == {"interactivity": tail}


def test_an_otherwise_empty_tail_keeps_the_fixed_arity_placeholder() -> None:
    """Providers place cache breakpoints per block index (bd19aafbcbd3), so the
    tail is never an empty string even when every section is withheld."""
    blocks = _blocks(_MCP_ONLY)
    assert len(blocks) == 4
    assert blocks[3] == "<skills/>"


def test_a_shell_only_session_keeps_the_filesystem_safety_rules() -> None:
    """``rm`` needs no file tool: the delete/overwrite rules ride
    ``has_workspace`` (files OR shell), never ``has_files`` alone."""
    head, inventory, _env, tail = _blocks(["bash"], host_has_console=False)
    for kept in (
        "Never delete or move a path you did not create",
        "Treat unknown files as the user's work",
        "(bang-mode)",
        "Deliverables surface themselves.",
        "When the `console` tool is NOT in your tool list",
    ):
        assert kept in head, kept
    assert "Guardrail rules: `<config_dir>/rules/*.md`; Rules only add restriction." in head
    for dropped in ("Edit, don't rewrite.", "### Resources", "guide://", "`read scratchpad://`"):
        assert dropped not in head, dropped
    assert "This session was not given the browser tool." in inventory
    assert "<guides>" not in tail


@pytest.mark.parametrize("names", [["write"], ["grep"], _MCP_ONLY])
def test_no_shell_omits_browser_setup_and_bang_rule_follows_shell_or_ask(names: list[str]) -> None:
    """Bang-mode provenance rides ``has_shell`` OR ``has_ask``: #4332 R1-m1's
    case is a TUI session without a shell tool whose operator still types
    ``! cmd``, and that host holds ``ask``. A headless run without either can
    neither receive a bang command nor re-run one."""
    head, _inventory, _env, _tail = _blocks(names, host_has_browser=False)
    assert "(bang-mode)" not in head
    for pointer in ("pages with curl", '"Headless nodes"', "follow `guide://browser`"):
        assert pointer not in head
    with_ask = _blocks([*names, "ask"], host_has_browser=False)[0]
    assert "(bang-mode)" in with_ask


@pytest.mark.parametrize("names", [["write"], ["grep"], ["bash"]])
def test_workspace_without_read_keeps_rules_but_not_resource_pointers(names: list[str]) -> None:
    head, _inventory, _env, _tail = _blocks(names)
    assert "Guardrail rules: `<config_dir>/rules/*.md`; Rules only add restriction." in head
    assert "`read scratchpad://`" not in head
    assert "guide://guardrails" not in head


def test_read_resource_sentence_does_not_require_file_class() -> None:
    text = render_template("system.md", {**_HOST, "has_files": False, "has_resources": True})
    assert "`read scratchpad://` lists it; details: `guide://scratchpad`." in text
    assert "Text scratch of your own" not in text


@pytest.mark.parametrize("files,shell", list(itertools.product([False, True], repeat=2)))
def test_direct_render_derives_workspace_from_files_or_shell(files: bool, shell: bool) -> None:
    text = render_template("system.md", {**_HOST, "has_files": files, "has_shell": shell})
    assert ("Guardrail rules:" in text) == (files or shell)
    assert ("Treat unknown files as the user's work" in text) == (files or shell)


def test_a_deferred_class_tool_counts_as_held() -> None:
    """Deferred = callable via ``read tool://<name>``; its class prose stays."""
    head, _inv, _env, tail = _blocks([*_MCP_ONLY, "read"], deferred_tools=("read",))
    assert "### Resources" in head
    assert "(`guide://credentials`)" in head
    assert tail.startswith(_KNOWLEDGE)


def test_has_web_and_no_web_cannot_both_be_true() -> None:
    with pytest.raises(ValueError, match="has_web and no_web"):
        render_template("system.md", {**_HOST, "has_web": True, "no_web": True})


#: Flags whose prose shares a line or a paragraph break with a capability class.
_SEAM_FLAGS = (
    *sorted(CAPABILITY_CLASS_FLAGS),
    "has_intent",
    "has_wake_monitor",
    "has_no_reply",
    "has_notify",
    "has_ask",
    "has_eval",
    "has_subagents",
)


def _seam_faults(text: str) -> list[str]:
    faults: list[str] = []
    body = text.rstrip("\n")
    if "\n\n\n" in body:
        at = body.index("\n\n\n")
        faults.append(f"blank-line run near {body[max(0, at - 60) : at]!r}")
    for line in body.splitlines():
        # Bullet continuation lines are indented two spaces on purpose.
        content = line[2:] if line.startswith("  ") and not line.startswith("   ") else line
        if content.startswith(" "):
            faults.append(f"leading space: {line!r}")
        if content.endswith(" "):
            faults.append(f"trailing space: {line!r}")
        if "  " in content:
            faults.append(f"double space: {line!r}")
        # Two sentences glued without their separator ("tools.`wake`").
        if re.search(r"\.(`|[A-Z])", content):
            faults.append(f"glued join: {line!r}")
    return faults


def test_every_capability_flag_combination_joins_its_sentences_cleanly() -> None:
    """#4318 review R3 / #4326's rule, over the new flags: every combination of
    the classes and the per-tool flags sharing their lines renders with no
    leading, trailing or double space, no glued sentences and no blank-line
    run. Asserted to hold on the ungated render first, so a fault is the gate's."""
    assert _seam_faults(render_template("system.md", dict(_HOST))) == []
    for combo in itertools.product([False, True], repeat=len(_SEAM_FLAGS)):
        flags = dict(zip(_SEAM_FLAGS, combo, strict=True))
        text = render_template("system.md", {**_HOST, **flags})
        assert _seam_faults(text) == [], flags


def test_the_tools_heading_never_renders_empty() -> None:
    """With every lead sentence gated off the section still carries the ``i``
    guidance directly under its heading; with that gated off too (no tool has
    the injected schema property) the ``### Deciding and asking`` subsection
    follows it, so the heading still has a body."""
    off = {flag: False for flag in (*TOOL_GATED_PROSE_FLAGS, *_SEAM_FLAGS, "has_bridge")}
    text = render_template("system.md", {**_HOST, **off, "has_intent": True})
    assert "## Tools\n\nMost tools take `i`" in text
    bare = render_template("system.md", {**_HOST, **off})
    assert "## Tools\n\n### Deciding and asking\n\n" in bare


@pytest.mark.parametrize(
    ("interactive", "channel", "no_ask_line"),
    [
        # No ``<interactivity>`` block (no runtime probe): the head line is the
        # only statement, so it stays.
        (None, None, True),
        # The PROCEED bodies say it already; said once, in the tail.
        (False, None, False),
        (True, None, False),
        (False, CHANNEL_HUB, False),
        # An attached child is routed through ``hub``, not told to proceed, so
        # the head keeps its decide-yourself line.
        (True, CHANNEL_HUB, True),
    ],
)
def test_the_no_ask_line_is_said_once_between_head_and_interactivity(
    interactive: bool | None, channel: str | None, no_ask_line: bool
) -> None:
    head, _inv, _env, tail = _blocks(_MCP_ONLY, interactive=interactive, channel=channel)
    assert ("No `ask` tool in this session" in head) is no_ask_line
    proceeds = "PROCEED" in tail or "proceed on the best reading" in tail
    assert proceeds is not no_ask_line or (interactive, channel) == (True, CHANNEL_HUB)


def test_a_session_holding_ask_keeps_its_ask_prose_whatever_the_tail_says() -> None:
    head, _inv, _env, tail = _blocks([*_MCP_ONLY, "ask"], interactive=False, channel=CHANNEL_ASK)
    assert "Reach for `ask` only when" in head
    assert "No `ask` tool in this session" not in head


def test_the_intent_rule_follows_the_schemas_not_the_names() -> None:
    """``has_intent`` reads the declared schemas: the same names without the
    injected ``i`` property drop the rule and its Narration clause."""
    plain = build_system_blocks(
        [_tool(name, intent=False) for name in _MCP_ONLY],
        skills_block="",
        env_details="",
        date_str="2026-01-01",
        host_has_browser=True,
        host_has_console=True,
    )[0]
    assert "Most tools take `i`" not in plain
    assert "`i` intent" not in plain
    assert "Never open with filler" in plain
    assert "## Tools\n\n### Deciding and asking" in plain
    assert _seam_faults(plain.split("## Session state updates")[0]) == []


@pytest.mark.parametrize(
    ("names", "workspace", "session_id"),
    [
        (_MCP_ONLY, False, False),
        (["web_search", "web_fetch"], False, False),
        (["bash"], True, False),
        (["read"], True, False),
        ([*_MCP_ONLY, "sessions"], False, True),
        (["send"], False, True),
        (["tags"], False, True),
        (["bash", "read", "sessions", "send", "tags"], True, True),
    ],
)
def test_env_details_ship_only_lines_a_held_tool_can_use(
    names: list[str], workspace: bool, session_id: bool
) -> None:
    text = _env_details("/work", "sid123", names)
    assert ("Working directory: /work" in text) is workspace
    assert ("Platform: " in text) is workspace
    assert ("Python: " in text) is workspace
    assert ("Session ID: sid123" in text) is session_id


def test_env_details_without_a_tool_list_keep_every_line() -> None:
    """``None`` is the caller that states no membership: historical shape."""
    text = _env_details("/work", "sid123")
    assert text.splitlines()[2:] == ["Working directory: /work", "Session ID: sid123"]
    assert text.startswith("Platform: ")


def test_an_mcp_only_env_block_is_the_date_and_model() -> None:
    blocks = build_system_blocks(
        [_tool(name) for name in _MCP_ONLY],
        skills_block="",
        env_details=_env_details("/work", "sid123", _MCP_ONLY),
        date_str="2026-01-01",
        model_label="bedrock/ca.anthropic.claude-haiku-5-5",
        host_has_browser=True,
        host_has_console=True,
    )
    assert blocks[2] == "Today is 2026-01-01.\n\nModel: bedrock/ca.anthropic.claude-haiku-5-5"


def test_has_workspace_and_no_workspace_cannot_both_be_true() -> None:
    with pytest.raises(ValueError, match="has_workspace and no_workspace"):
        render_template("system.md", {**_HOST, "has_workspace": True, "no_workspace": True})
