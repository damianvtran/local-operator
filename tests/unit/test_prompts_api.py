"""Tests for the template renderer and system-prompt block builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from local_operator.harness.types import AgentTool, ToolContext
from local_operator.prompts_api import (
    build_system_blocks,
    render_string,
    render_template,
)
from local_operator.tools import builtin


@pytest.fixture(autouse=True)
def _force_browser_available(monkeypatch):
    """The inventory assertion spans the full default surface including
    ``browser``, whose builder is gated on a reachable CMUX browser that CI
    lacks; force the predicate so the ordering test is deterministic."""
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)


if TYPE_CHECKING:
    from local_operator.harness.wake import WakeSchedule


def _tool(name: str, description: str, hidden: bool = False) -> AgentTool:
    async def _noop(*_args, **_kwargs):  # pragma: no cover — never executed here
        raise AssertionError("not called")

    return AgentTool(name=name, description=description, hidden=hidden, execute=_noop)


class _FakeSchedulerForBlocks:
    """Minimal scheduler so create_tools includes wake for ordering checks."""

    @property
    def schedules(self) -> list["WakeSchedule"]:
        return []

    async def update(self, schedules: list["WakeSchedule"]) -> None:
        pass


class _FakeJobsForBlocks:
    """Minimal job manager so create_tools includes task/wait/jobs."""

    def get(self, job_id: str, *, owner_id: str | None = None) -> Any:
        return None

    def list(self, *, owner_id: str | None = None) -> list[Any]:
        return []

    async def cancel(self, job_id: str, *, owner_id: str | None = None) -> bool:
        return False


# ---------------------------------------------------------------------------
# render_string / render_template engine
# ---------------------------------------------------------------------------


def test_plain_variable() -> None:
    assert render_string("Hello {{name}}!", {"name": "World"}) == "Hello World!"


def test_missing_variable_renders_empty() -> None:
    assert render_string("a{{missing}}b", {}) == "ab"


def test_if_true_and_false() -> None:
    assert render_string("{{#if flag}}on{{/if}}", {"flag": True}) == "on"
    assert render_string("{{#if flag}}on{{/if}}", {"flag": False}) == ""
    assert render_string("{{#if flag}}on{{/if}}", {}) == ""
    # truthy non-bool values activate the branch
    assert render_string("{{#if x}}y{{/if}}", {"x": "text"}) == "y"


def test_nested_if() -> None:
    template = "{{#if a}}A{{#if b}}B{{/if}}{{/if}}"
    assert render_string(template, {"a": True, "b": True}) == "AB"
    assert render_string(template, {"a": True, "b": False}) == "A"
    assert render_string(template, {}) == ""


def test_each_over_strings() -> None:
    template = "{{#each items}}- {{this}}\n{{/each}}"
    assert render_string(template, {"items": ["x", "y"]}) == "- x\n- y\n"


def test_each_over_dicts_exposes_keys() -> None:
    template = "{{#each tools}}{{name}}={{value}} {{/each}}"
    out = render_string(template, {"tools": [{"name": "a", "value": 1}, {"name": "b", "value": 2}]})
    assert out == "a=1 b=2 "


def test_each_missing_or_empty_renders_nothing() -> None:
    assert render_string("[{{#each items}}x{{/each}}]", {}) == "[]"
    assert render_string("[{{#each items}}x{{/each}}]", {"items": []}) == "[]"


def test_unclosed_block_raises() -> None:
    with pytest.raises(ValueError):
        render_string("{{#if x}}never closed", {"x": True})


def test_stray_closing_tag_raises() -> None:
    # RT-20/RT-33: a closer with no opener is a build bug, not silent text.
    for template in ("{{/if}}", "text {{/each}}", "{{#if x}}{{/each}}{{/if}}"):
        with pytest.raises(ValueError):
            render_string(template, {"x": True})


def test_nested_each_inside_if() -> None:
    # RT-33: block nesting composes; the each body re-renders per item.
    template = "{{#if show}}[{{#each items}}{{this}},{{/each}}]{{/if}}"
    assert render_string(template, {"show": True, "items": [1, 2]}) == "[1,2,]"
    assert render_string(template, {"show": False, "items": [1, 2]}) == ""


def test_dotted_path_lookup() -> None:
    # RT-33: dotted paths resolve through nested dicts; a missing hop -> empty.
    assert render_string("{{a.b.c}}", {"a": {"b": {"c": "deep"}}}) == "deep"
    assert render_string("{{a.x.c}}", {"a": {"b": {"c": "deep"}}}) == ""


def test_system_md_loads_and_renders() -> None:
    text = render_template("system.md", {})
    assert "Local Operator" in text


def test_compaction_summary_renders_optional_sections() -> None:
    full = render_template(
        "compaction_summary.md",
        {"transcript": "TRANS", "previous_summary": "PREVSUMMARY", "files": "a.py"},
    )
    assert "## Goal" in full
    assert "TRANS" in full
    assert "PREVSUMMARY" in full
    assert "<files>" in full and "a.py" in full

    bare = render_template("compaction_summary.md", {"transcript": "TRANS"})
    assert "Previous summary" not in bare
    assert "<files>" not in bare
    assert "TRANS" in bare


# ---------------------------------------------------------------------------
# build_system_blocks
# ---------------------------------------------------------------------------

TOOLS = [
    _tool("bash", "Run a shell command."),
    _tool("read", "Read a file."),
    _tool("secret", "Hidden thing.", hidden=True),
]
SKILLS = "## Skills\n\n- demo: A demo skill."
ENV = "cwd: /tmp/project\nOS: Darwin"
DATE = "2026-08-04"


def test_blocks_isolate_volatile_content() -> None:
    # Fixed arity, cache-layout order: [instructions, inventory, env, skills].
    # The per-turn volatile skills block rides LAST so a selection change can
    # never invalidate the conversation prefix after it.
    blocks = build_system_blocks(TOOLS, SKILLS, ENV, DATE)
    assert len(blocks) == 4
    instructions, inventory, env_block, skills = blocks

    # block 0: stable instructions only
    assert "Local Operator" in instructions
    assert "demo" not in instructions
    assert DATE not in instructions
    assert "Darwin" not in instructions

    # block 1: tool inventory ONLY — skills never leak in here
    assert "- bash: Run a shell command." in inventory
    assert "- read: Read a file." in inventory
    assert "secret" not in inventory
    assert "demo" not in inventory
    assert DATE not in inventory and "Darwin" not in inventory

    # block 2: date + env only
    assert DATE in env_block
    assert "Darwin" in env_block
    assert "demo" not in env_block

    # last block: the skills listing verbatim
    assert skills == SKILLS


def test_no_skills_keeps_fixed_arity_with_placeholder() -> None:
    # The block list is fixed-arity: an empty selection emits the constant
    # placeholder, never drops the block (breakpoint derivation counts
    # blocks).
    blocks = build_system_blocks(TOOLS, "", ENV, DATE)
    assert len(blocks) == 4
    assert "- bash: Run a shell command." in blocks[1]
    assert blocks[2].startswith(f"Today is {DATE}.")
    assert blocks[3] == "<skills/>"


def test_block_zero_and_one_are_byte_stable_across_turns() -> None:
    """Different per-turn inputs must not perturb the stable prefix — that is
    the whole point of the split (prompt-cache stability, >=90% cache rate)."""
    b0, b1 = build_system_blocks(TOOLS, SKILLS, ENV, DATE)[:2]
    b0_again, b1_again = build_system_blocks(TOOLS, "different skills", "other env", "2027-01-01")[
        :2
    ]
    assert b0 == b0_again
    assert b1 == b1_again


def test_skills_only_ever_appear_in_their_own_block() -> None:
    blocks = build_system_blocks(TOOLS, SKILLS, ENV, DATE)
    for index, block in enumerate(blocks):
        if index == 3:
            continue  # the skills block itself (last, volatile)
        assert "demo: A demo skill." not in block


def test_env_and_date_never_in_the_stable_head() -> None:
    # The stable head (instructions, inventory) must stay byte-stable; env
    # and date ride in their own block (index 2), skills last.
    blocks = build_system_blocks(TOOLS, SKILLS, ENV, DATE)
    for block in blocks[:2]:
        assert DATE not in block
        assert "Darwin" not in block


def test_inventory_block_matches_default_tool_order() -> None:
    # RT-33: block-1 ordering follows DEFAULT_TOOL_NAMES, which keeps the
    # inventory byte-stable against provider tool-array ordering.
    from local_operator.tools.registry import DEFAULT_TOOL_NAMES, create_tools

    tools = create_tools(
        ToolContext(
            cwd=".",
            wake_scheduler=_FakeSchedulerForBlocks(),
            subagent_launcher=lambda label, prompt: "job-x",
            jobs=_FakeJobsForBlocks(),
        )
    )
    blocks = build_system_blocks(tools, "", ENV, DATE)
    lines = [line for line in blocks[1].splitlines() if line.startswith("- ")]
    expected = list(DEFAULT_TOOL_NAMES)  # scheduler attached -> wake included
    assert [line.split(":")[0][2:] for line in lines] == expected


def test_env_block_handles_empty_env_details() -> None:
    blocks = build_system_blocks(TOOLS, "", "", DATE)
    assert blocks[2] == f"Today is {DATE}."
