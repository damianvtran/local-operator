"""The console's prompt note, its flag pair, and the three-state diagnosis.

The browser's note is a PAIR of strings because one string would have to assert
something false about the host, and the console inherits that whole problem — the
`{{#if}}` engine has no `else` and no negation, so a half-supplied pair silently
drops a section, and a session's ROLE can lack the tool on a host that has it
perfectly well. That third state is the one an earlier revision of the browser
note got wrong (`prompts_api`'s own record), and it is the one a `reviewer` child
on a console-capable host would hit: told "there is no console on this host" when
the truth is "this role was not given it" — a false, actionable diagnosis.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import AgentTool
from local_operator.prompts_api import (
    build_system_blocks,
    render_template,
    render_tool_inventory_block,
)
from local_operator.tools import builtin
from local_operator.tools.registry import DEFAULT_TOOL_NAMES, create_tools

USAGE = "goes through the `console` tool"
ABSENCE = "When the `console` tool is NOT in your tool list"


def _tool(name: str, description: str = "x") -> AgentTool:
    async def execute(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - never called
        raise AssertionError

    return AgentTool(name=name, description=description, execute=execute)


@pytest.fixture
def host_without_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host whose app cannot serve a console (the default in CI)."""
    monkeypatch.setattr(builtin, "ui_console_advertisable", lambda: False)
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)


def test_the_two_console_sections_are_mutually_exclusive_for_every_input() -> None:
    values: list[dict[str, Any]] = [{}]
    for key in ("has_console", "no_console"):
        values += [{key: True}, {key: False}]
    for has in (True, False):
        for absent in (True, False):
            if not (has and absent):
                values.append({"has_console": has, "no_console": absent})

    for data in values:
        text = render_template("system.md", data)
        assert not (USAGE in text and ABSENCE in text), data


def test_a_bare_render_derives_the_consoleless_arm() -> None:
    """`{{#if}}` on a missing key drops its body with no marker, so the pair is
    completed rather than left to the caller — and the conservative arm is the
    one that claims nothing about a capability nobody stated."""
    bare = render_template("system.md", {})
    assert ABSENCE in bare and USAGE not in bare


def test_half_supplying_the_pair_never_ships_both_sections() -> None:
    only_has = render_template("system.md", {"has_console": True})
    assert USAGE in only_has and ABSENCE not in only_has
    only_absent = render_template("system.md", {"no_console": True})
    assert ABSENCE in only_absent and USAGE not in only_absent


def test_both_true_is_refused_rather_than_rendered() -> None:
    """A session cannot hold the tool on a host with no console, so the pair with
    no meaning raises — these callers are all in this repo, and an impossible
    prompt is a build bug rather than input to tolerate."""
    with pytest.raises(ValueError, match="cannot both be true"):
        render_template("system.md", {"has_console": True, "no_console": True})


def test_the_console_note_never_asks_the_model_to_set_one_up(host_without_console: None) -> None:
    """A three-line prohibition, not a setup playbook.

    The browser's absence arm IS a playbook — a user can install that host in a
    minute — and the console's is not: it needs the desktop app, so telling the
    model to arrange one would invite the dead end the playwright incident was
    (design §14.5).
    """
    inventory = render_tool_inventory_block([_tool("bash")], host_has_console=False)
    assert "This session has no `console` tool" in inventory
    assert "never install or script a terminal emulator" in inventory
    # No setup instruction of any kind, and no claim that the tool is broken.
    lowered = inventory.lower()
    assert "install the app" not in lowered
    assert "sign in" not in lowered


def test_a_role_without_the_tool_on_a_capable_host_is_told_the_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """State 3: the host HAS a console; this role simply was not granted it.

    The defect this pins: `reviewer`, `scout`, `manager` and `architect` seeds all
    omit `console` from their tool allowlists (exactly as they omit `browser`), so
    every such child on a console-capable host takes this branch. A child told
    "no console on this host" would answer a user's question about a console
    surface with a wrong diagnosis and no route to the right one.
    """
    monkeypatch.setattr(builtin, "ui_console_advertisable", lambda: True)
    inventory = render_tool_inventory_block([_tool("bash")], host_has_console=True)
    assert "was not given the `console` tool" in inventory
    assert "IS available on this host" in inventory
    assert "This session has no `console` tool" not in inventory


def test_the_probe_is_the_one_the_builder_gates_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The note's host diagnosis must track the gate, or the two disagree.

    Measured failures of exactly this kind exist for the browser (a predicate
    naming fewer hosts than the gate it mirrors told a restricted child the wrong
    thing), so the console's probe reads the same predicate the `createIf` entry
    reads rather than inferring it from the tool list.
    """
    seen: list[int] = []

    def probe() -> bool:
        seen.append(1)
        return True

    monkeypatch.setattr(builtin, "ui_console_advertisable", probe)
    block = render_tool_inventory_block([_tool("bash")])
    assert "IS available on this host" in block
    assert seen, "the note did not consult the console gate's own predicate"


def test_the_console_tool_present_suppresses_the_absence_note() -> None:
    block = render_tool_inventory_block([_tool("bash"), _tool("console")], host_has_console=False)
    assert "This session has no `console` tool" not in block
    assert "was not given the `console` tool" not in block


def test_a_hidden_console_tool_still_suppresses_the_note() -> None:
    """Membership, not visibility: a hidden tool is still callable, and telling
    the model a capability does not exist while one answers is worse than
    silence. (The console is not hidden — this pins the RULE, not a state.)"""
    hidden = _tool("console")
    hidden.hidden = True
    block = render_tool_inventory_block([_tool("bash"), hidden], host_has_console=False)
    assert "This session has no `console` tool" not in block


def test_build_system_blocks_wires_the_console_flags_it_renders_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the WIRING, not the template's branching.

    A template-only test hand-writes the flags and therefore cannot see a caller
    that passes the wrong ones — that is how the browser pair shipped both
    sections at once, and the same mutation (hardcoding ``has_console=True``,
    inverting the pair, or passing ``{}``) must not survive here either.
    """
    monkeypatch.setattr(builtin, "ui_console_advertisable", lambda: True)
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)

    with_console = build_system_blocks([_tool("bash"), _tool("console")], "", "env", "2026-01-01")
    assert USAGE in with_console[0]
    assert ABSENCE not in with_console[0]

    monkeypatch.setattr(builtin, "ui_console_advertisable", lambda: False)
    without = build_system_blocks([_tool("bash")], "", "env", "2026-01-01")
    assert USAGE not in without[0]
    assert ABSENCE in without[0]
    # The note rides the inventory block, which is the only block the session
    # re-renders, so the diagnosis follows a tool that appears mid-session too.
    assert "This session has no `console` tool" in without[1]


def test_a_console_capable_host_offers_the_tool_through_the_real_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prompt note and the gate agree on one machine state, end to end.

    A fully-capable context plus the console gate forced on must produce the tool
    in the registry AND the usage prose in the prompt: two renderers of one
    decision (the gate and the note) disagreeing is precisely the class of bug
    the browser's history records.
    """
    monkeypatch.setattr(builtin, "ui_console_advertisable", lambda: True)
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tools = create_tools(_full_context())
    assert "console" in [tool.name for tool in tools]
    blocks = build_system_blocks(tools, "", "env", "2026-01-01")
    assert USAGE in blocks[0]


def _full_context() -> Any:
    """The same all-capabilities context `scripts/real_tool_surface` builds."""
    import sys

    sys.path.insert(0, ".")
    from scripts.real_tool_surface import build_real_tool_context

    return build_real_tool_context(".")


def test_the_read_only_role_seeds_still_omit_the_console() -> None:
    """§14.6's enforcement lever, asserted rather than assumed.

    The role seeds' `tools` allowlists are what actually keeps `console_*` out of
    an architect/reviewer/scout/manager child — prose is not enforcement — so the
    omission is pinned here. `coder` and the UX roles carry `tools: null` (the
    full surface) and are deliberately not in this set.
    """
    import json

    from local_operator.agent_profiles import SEEDS_DIR

    manifest = json.loads((SEEDS_DIR / "manifest.json").read_text(encoding="utf-8"))
    constrained = {
        seed["name"]: seed["tools"] for seed in manifest["seeds"] if seed["tools"] is not None
    }
    assert {"architect", "manager", "reviewer", "scout"} <= set(constrained)
    for name, tools in constrained.items():
        assert "console" not in tools, name


def test_the_console_tool_is_in_the_default_surface() -> None:
    """A `createIf`-gated tool is still a DEFAULT tool: the gate decides whether it
    builds, not whether the session asked for it. A tool reachable only by a host
    config edit would be one nobody has."""
    assert "console" in DEFAULT_TOOL_NAMES
    assert callable(builtin.build_console_tool)
