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


class _FakeCommsForBlocks:
    """Minimal comms so create_tools includes hub for ordering checks."""

    def is_child(self, job_id: str | None) -> bool:
        return False


async def _fake_ask_for_blocks(questions: list[Any]) -> dict[str, list[str]] | None:
    """Minimal ask hook so create_tools includes ask for ordering checks."""
    return None


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


def test_system_md_states_harness_identity() -> None:
    """The prompt must tell the agent it IS the local-operator harness, run as
    ``lop``, so a question about the harness/itself is answered rather than
    treated as an unknown third-party tool. Asserts the behaviour contract, not
    exact wording — reword freely, but keep the identity and the command name.
    """
    text = render_template("system.md", {})
    assert "harness" in text
    # The standard command name must be named so the agent can tell a user how
    # to run/update it.
    assert "`lop`" in text


def test_system_md_requires_owned_browser_cleanup_before_final_response() -> None:
    """Long-lived interactive sessions do not dispose between turns, so the
    packaged invariant must make normal cleanup the agent's responsibility."""
    text = render_template("system.md", {})
    assert "action=close" in text
    assert "BEFORE the final response" in text
    assert "Never close another" in text
    assert "session teardown is a fallback" in text


def test_system_md_teaches_eval_digest_pipeline() -> None:
    """The prompt must steer multi-step work into one ``eval`` that prints a
    compact digest, with full output kept fetchable via ``spill://`` — the
    token-efficiency guidance. Contract, not wording.
    """
    text = render_template("system.md", {})
    assert "eval" in text
    assert "spill://" in text


def test_system_md_frames_the_web_as_a_verification_surface() -> None:
    """Verification must not be defined as a closed set of LOCAL actions.

    The first working principle used to enumerate verification as "run the
    command, read the file, search the workspace" — three members, all local
    and complete-sounding. An agent following it faithfully never reaches the
    web, because by the definition it was handed it has already verified. That
    is why lookups only happened when a user asked for them explicitly.
    Contract, not wording: the enumeration must keep a web member.
    """
    # Normalized: the source is hard-wrapped at 79 columns, so any asserted
    # phrase may straddle a line break.
    text = " ".join(render_template("system.md", {}).split())
    # The persona paragraph enumerates the toolset and states when to use it.
    # Both halves carried the closed-local framing, and this one is read FIRST
    # — a later bullet asking for web research argues uphill against a persona
    # that already scoped tool use to "the machine's state".
    assert "searching the workspace and the web" in text
    assert "or on something you would otherwise recall" in text
    # The working principle's own enumeration of what verifying means.
    assert "search the workspace, look it up on the web" in text


def test_system_md_teaches_when_to_research_without_making_it_reflexive() -> None:
    """The research bullet has to carry four clauses that pull against each
    other, so all four are pinned as behaviour.

    Naming the tools alone would produce an agent that searches on every task —
    a latency and token cost paid on work that never needed it. The trigger is
    deliberately self-observational ("you notice you are guessing") rather than
    a topic list, because a topic list invites keyword matching and generalises
    to nothing outside it. The last clause exists so a result informs the work
    without becoming it.
    """
    text = " ".join(render_template("system.md", {}).split())
    # 1. The tools are named as the answer to stale knowledge.
    assert "`web_search`/`web_fetch`" in text
    # 2. Staleness is the stated reason, not a preference.
    assert "cutoff" in text
    # 3. Not reflexive: a trigger the agent evaluates against itself, plus an
    #    explicit brake. Stated trigger-then-brake so the agent learns what
    #    fires the search before it learns what suppresses it.
    assert "you notice you are guessing" in text
    assert "not on every task" in text
    # 4. Results inform, never dictate, are never copied wholesale, and — the
    #    clause that is easy to miss — never bound the options considered. An
    #    agent can honour "never the answer" and still weigh only the three
    #    approaches a search surfaced.
    assert "input, never the answer" in text
    assert "never the edge of your options" in text


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


def test_the_system_prompt_names_user_run_bang_receipts() -> None:
    """The model must be able to tell a command the USER ran (bang-mode) from
    one it issued itself: a `! <command>` user message + bash call + result is
    user-produced context, never the model's own earlier action."""
    blocks = build_system_blocks(TOOLS, SKILLS, ENV, DATE)
    head = blocks[0]
    assert "bang-mode" in head
    assert "the USER ran directly" in head
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


def test_model_label_rides_the_env_block_not_the_stable_head() -> None:
    # The running-model line must be visible to the model, and it belongs in the
    # env block (index 2) rather than the byte-stable head, because it can change
    # mid-session (a /model switch, a failover fallback) and must not invalidate
    # the cached conversation prefix.
    blocks = build_system_blocks(TOOLS, SKILLS, ENV, DATE, model_label="anthropic/claude-opus-4-8")
    instructions, inventory, env_block, _skills = blocks
    assert "Model: anthropic/claude-opus-4-8" in env_block
    # Not in the stable head/inventory — those must stay model-agnostic.
    assert "claude-opus-4-8" not in instructions
    assert "claude-opus-4-8" not in inventory


def test_model_label_absent_by_default_and_when_blank() -> None:
    # Sessions/hosts that pass no label (or a blank one) must not grow a dangling
    # "Model:" line — the feature is additive.
    for label in ("", "   "):
        blocks = build_system_blocks(TOOLS, SKILLS, ENV, DATE, model_label=label)
        assert "Model:" not in blocks[2]
    assert "Model:" not in build_system_blocks(TOOLS, SKILLS, ENV, DATE)[2]


def test_model_label_does_not_perturb_the_stable_prefix() -> None:
    # Two different model labels must leave blocks 0 and 1 byte-identical, so a
    # switch never busts the prompt-cache prefix.
    a = build_system_blocks(TOOLS, SKILLS, ENV, DATE, model_label="anthropic/x")
    b = build_system_blocks(TOOLS, SKILLS, ENV, DATE, model_label="zai/y")
    assert a[:2] == b[:2]
    # Arity is still fixed with the label present.
    assert len(a) == 4


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
            subagent_launcher=lambda label, prompt, *, agent="task", effort=None: "job-x",
            jobs=_FakeJobsForBlocks(),
            subagent_comms=_FakeCommsForBlocks(),
            # `ask` is createIf-gated on a UI that can draw its picker AND on the
            # hook that answers it, so the fully-capable context this test needs
            # has to carry both or the tool drops out of the inventory.
            has_ui=True,
            ask_user=_fake_ask_for_blocks,
            # `agent` is createIf-gated on a registry to persist roles into;
            # only its PRESENCE is read here, since this test is about the
            # ORDER of a fully-capable inventory.
            agent_registry=object(),
            team_registry=object(),
        )
    )
    blocks = build_system_blocks(tools, "", ENV, DATE)
    lines = [line for line in blocks[1].splitlines() if line.startswith("- ")]
    expected = list(DEFAULT_TOOL_NAMES)  # scheduler attached -> wake included
    assert [line.split(":")[0][2:] for line in lines] == expected


def test_inventory_explains_a_missing_browser_instead_of_leaving_a_hole() -> None:
    """The browser builder is createIf-gated on a reachable cmux, so on a host
    without one the model can only see an ABSENCE — and an absence reads as
    "arrange your own". A real session answered a screenshot request by writing
    a playwright script and downloading a Chromium that could not carry the
    user's logins. The inventory says the capability is missing and why."""
    blocks = build_system_blocks(TOOLS, SKILLS, ENV, DATE)

    inventory = blocks[1]
    assert "NO browser tool" in inventory
    assert "playwright" in inventory
    assert "never install or script a browser engine" in inventory.lower()
    # The note is a capability statement, not a fake inventory entry: the
    # ordering test reads every "- " line as a tool name.
    assert not any(line.startswith("- ") and "browser" in line for line in inventory.splitlines())


def test_inventory_note_is_absent_when_a_browser_tool_exists() -> None:
    """Never tell the model a browser is unavailable while one would answer."""
    tools = [*TOOLS, _tool("browser", "Drive the user's real browser.")]

    inventory = build_system_blocks(tools, SKILLS, ENV, DATE)[1]
    assert "NO browser tool" not in inventory
    assert "playwright" not in inventory


def test_inventory_note_follows_membership_not_visibility() -> None:
    """A hidden tool is still callable; claiming it does not exist is worse
    than saying nothing."""
    tools = [*TOOLS, _tool("browser", "Drive the user's real browser.", hidden=True)]

    assert "NO browser tool" not in build_system_blocks(tools, SKILLS, ENV, DATE)[1]


def test_env_block_handles_empty_env_details() -> None:
    blocks = build_system_blocks(TOOLS, "", "", DATE)
    assert blocks[2] == f"Today is {DATE}."


# ---------------------------------------------------------------------------
# user custom instructions
# ---------------------------------------------------------------------------


def test_user_instructions_ride_the_stable_head_block() -> None:
    """Custom instructions are as stable as the persona, so they belong in
    block 0 — not the volatile tail, where a long file would be re-sent
    ahead of every skills/goal change."""
    blocks = build_system_blocks(
        TOOLS, SKILLS, ENV, DATE, user_instructions="- Use conventional commits."
    )

    assert len(blocks) == 4
    assert "- Use conventional commits." in blocks[0]
    assert "<user_instructions>" in blocks[0]
    for block in blocks[1:]:
        assert "conventional commits" not in block


def test_user_instructions_are_delimited_from_the_packaged_persona() -> None:
    # The model must be able to tell operator preference apart from the
    # packaged rules; without the tag a long file reads as a continuation.
    blocks = build_system_blocks(TOOLS, "", ENV, DATE, user_instructions="- Prefer tabs.")

    head = blocks[0]
    assert "## User's custom instructions" in head
    assert head.index("You are Local Operator") < head.index("<user_instructions>")
    assert "</user_instructions>" in head


def test_absent_user_instructions_leave_the_head_byte_identical() -> None:
    # Nobody who has not set the file may pay for the feature.
    baseline = build_system_blocks(TOOLS, SKILLS, ENV, DATE)[0]

    assert build_system_blocks(TOOLS, SKILLS, ENV, DATE, user_instructions="")[0] == baseline
    assert build_system_blocks(TOOLS, SKILLS, ENV, DATE, user_instructions="   \n ")[0] == baseline


def test_user_instructions_keep_the_head_stable_across_turns() -> None:
    instructions = "- Always ask before force-pushing."
    first = build_system_blocks(TOOLS, SKILLS, ENV, DATE, user_instructions=instructions)[0]
    second = build_system_blocks(
        TOOLS, "other skills", "other env", "2027-01-01", user_instructions=instructions
    )[0]

    assert first == second


def test_credentials_ride_the_volatile_tail_as_names_only() -> None:
    """A newly stored key must invalidate only the tail, and the value must
    never appear in any block."""
    blocks = build_system_blocks(
        TOOLS, SKILLS, ENV, DATE, credentials=["GITHUB_TOKEN", "DEPLOY_KEY"]
    )
    assert len(blocks) == 4
    tail = blocks[3]
    assert "<session-credentials>" in tail
    assert "`GITHUB_TOKEN`" in tail
    assert "`DEPLOY_KEY`" in tail
    # The discoverability sentence: the live failure was a model that never
    # connected "I just added the API key" with this block, because nothing
    # told it the names below are where to look.
    assert "added a key or credential" in tail
    assert "list_variables" in tail
    for block in blocks[:3]:
        assert "GITHUB_TOKEN" not in block
    # Idle sessions pay nothing: no names means no block.
    idle = build_system_blocks(TOOLS, SKILLS, ENV, DATE)
    assert "<session-credentials>" not in idle[3]


def test_agent_brief_rides_the_volatile_tail() -> None:
    """A /agent attach must not invalidate the cached persona prefix, and it
    must land AFTER the team brief — the later, more specific instruction."""
    blocks = build_system_blocks(
        TOOLS,
        SKILLS,
        ENV,
        DATE,
        team_brief="[team: release]\nYou coordinate.",
        agent_brief="[role: reviewer]\nYou review.",
    )
    assert "<agent>" in blocks[3]
    assert "[role: reviewer]" in blocks[3]
    assert blocks[3].index("<team>") < blocks[3].index("<agent>")
    for block in blocks[:3]:
        assert "role: reviewer" not in block
    idle = build_system_blocks(TOOLS, SKILLS, ENV, DATE)
    assert "<agent>" not in idle[3]


def test_team_brief_rides_the_volatile_tail() -> None:
    """A /team attach must not invalidate the cached persona prefix."""
    blocks = build_system_blocks(
        TOOLS, SKILLS, ENV, DATE, team_brief="[team: release]\nYou coordinate."
    )
    assert "<team>" in blocks[3]
    assert "[team: release]" in blocks[3]
    for block in blocks[:3]:
        assert "team: release" not in block
    idle = build_system_blocks(TOOLS, SKILLS, ENV, DATE)
    assert "<team>" not in idle[3]
    # Same names keep the head byte-stable.
    again = build_system_blocks(
        TOOLS, "other skills", "other env", "2027-01-01", credentials=["GITHUB_TOKEN", "DEPLOY_KEY"]
    )
    assert blocks[0] == again[0]
    assert blocks[1] == again[1]
