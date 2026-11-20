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

from local_operator.harness.types import AgentTool
from local_operator.prompts_api import (
    CAPABILITY_CLASS_FLAGS,
    TOOL_GATED_PROSE_FLAGS,
    build_system_blocks,
    render_template,
    split_tail_sections,
)

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
    "(bang-mode)",
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
    "Keep secrets secret: never print credentials, tokens or keys",
    "Respect denials of a prompted write or command",
    "Deciding is your job",
    "Authorization never extends to a destructive or irreversible step",
)


def _tool(name: str) -> AgentTool:
    async def _noop(*_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("not called")

    return AgentTool(name=name, description=f"{name} tool", execute=_noop)


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
    assert head.startswith(plain)
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
    for dropped in ("Edit, don't rewrite.", "`scratchpad://`", "### Resources", "guide://"):
        assert dropped not in head, dropped
    assert "This session was not given the browser tool." in inventory
    assert "<guides>" not in tail


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
    """With every lead sentence gated off the section still carries the
    always-on ``i`` guidance directly under its heading."""
    text = render_template(
        "system.md",
        {
            **_HOST,
            **{flag: False for flag in (*TOOL_GATED_PROSE_FLAGS, *_SEAM_FLAGS, "has_bridge")},
        },
    )
    assert "## Tools\n\nMost tools take `i`" in text
