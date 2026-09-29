"""Child knowledge tails: names survive, repeated furniture does not.

A child's block 3 is re-sent on every call of the delegated run, so these are
structural guards on what rides it, not machine-speed bets. The contract:

* every guide/skill NAME in the inherited listing survives — that is the
  discoverability invariant, and the full text plus reference files stay one
  ``guide://``/``skill://`` read away; and
* the parent's ``<resource_recommendations>``, a DUPLICATED ``<mcps>``
  catalogue and over-cap listing descriptions do not.

The reductions are composed by ``subagent._slim_child_knowledge`` and gated by
``subagents.slim_child_knowledge``; the integration cases build a real child
through ``_build_child_session`` exactly as a reviewer/QA delegation does.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.harness import subagent as subagent_mod
from local_operator.skills.discovery import Skill
from local_operator.skills.index import render_block as render_skills_block
from tests.unit.session.test_session import ScriptedStream, make_session

# The real producers' furniture, so a rename on either side of the seam fails
# here rather than silently turning a strip into a no-op.
_GENERIC_CATALOGUE = "<mcps>Find MCP tools: `mcp://?search=terms`; list: `mcp://`.</mcps>"

# A description comfortably past the child cap. The exact content is
# irrelevant; the length is the point.
LONG_DESCRIPTION = (
    "Full testing procedures with reproduction steps, environment notes, "
    "expected outputs and several more clauses that exist only to push this "
    "description well past the per-line cap a child tail carries."
)


def _parent_block() -> str:
    """The shape ``session_factory._select_knowledge_block`` composes.

    Skills render (guides section, then skills section, ``"\\n\\n"``-joined),
    then the ``<mcps>`` catalogue, then the classification layer's
    ``<resource_recommendations>`` block — the same order the producer appends
    them.
    """
    guides = render_skills_block(
        [
            Skill(
                name="repository-testing",
                description=LONG_DESCRIPTION,
                file_path=Path("guide-a/SKILL.md"),
                base_dir=Path("guide-a"),
                source="test",
                resource_type="guide",
            ),
        ]
    )
    skills = render_skills_block(
        [
            Skill(
                name="minerva-router",
                description="Routes Minerva work to the right skill.",
                file_path=Path("skill-a/SKILL.md"),
                base_dir=Path("skill-a"),
                source="test",
            ),
        ]
    )
    recommendations = (
        "<resource_recommendations>\n"
        "These may help with this request — read the ones that actually fit, ignore the rest:\n"
        "- `skill://minerva-router` (0.91)\n"
        "</resource_recommendations>"
    )
    return "\n\n".join([guides, skills, _GENERIC_CATALOGUE, recommendations])


class _OwnedStream(ScriptedStream):
    """A parent stream whose children take their own handle (see the runner)."""

    def __init__(self) -> None:
        super().__init__([])
        self.children: list[tuple[str, "_OwnedStream"]] = []
        self.closed = False

    def fork(self, session_id: str) -> "_OwnedStream":
        child = _OwnedStream()
        self.children.append((session_id, child))
        return child

    async def close(self) -> None:
        self.closed = True


def _stub_child_mcp(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Wire a fake ``_ChildMcp`` whose catalogue is identifiable.

    The real wiring needs a live ``McpManager``; what the tail composition
    cares about is only that the child HAS a catalogue of its own, composed
    against the child's task prompt. Returns the list of queries the child's
    catalogue was rendered with.
    """
    calls: list[str] = []

    def catalogue(query: str) -> str:
        calls.append(query)
        return "<mcps>\n- own-server: The child's own catalogue. Read `mcp://own-server`.\n</mcps>"

    stub = subagent_mod._ChildMcp(
        tools=[],
        catalogue=catalogue,
        resolve=lambda url: None,
        attach=lambda session: None,
    )
    monkeypatch.setattr(subagent_mod, "_child_mcp_wiring", lambda *args, **kwargs: stub)
    return calls


def _parent_session(tmp_path, block: str):
    def provider(model_label: str = "") -> list[str]:
        return ["standing instructions", "tools", "date today", block]

    setattr(provider, "append_only_state", True)
    setattr(provider, "repo_guidance", "")
    setattr(provider, "knowledge_hooks", SimpleNamespace(frozen_block=block))
    stream = _OwnedStream()
    return make_session(tmp_path, stream, system_blocks_provider=provider), stream


async def _child_tail(child) -> str:
    blocks = child._system_blocks_provider()
    if inspect.isawaitable(blocks):
        blocks = await blocks
    return blocks[3]


# ---------------------------------------------------------------------------
# The pure composition
# ---------------------------------------------------------------------------


def test_slimmer_markers_track_their_producers() -> None:
    """The strip markers must be the producers' actual furniture.

    A renamed marker on either side would turn the strip into a silent no-op
    (the child keeps paying for the block) with every test still green — this
    is the one place the seam is pinned in both directions.
    """
    from local_operator.mcp.resources import render_mcp_suggestions
    from local_operator.session_factory import (
        _RECOMMENDATION_BLOCK_CLOSE,
        _RECOMMENDATION_BLOCK_OPEN,
    )

    assert subagent_mod._RECOMMENDATIONS_OPEN == _RECOMMENDATION_BLOCK_OPEN
    assert subagent_mod._RECOMMENDATIONS_CLOSE == _RECOMMENDATION_BLOCK_CLOSE
    rendered = render_mcp_suggestions([], "")
    assert rendered.startswith(subagent_mod._MCP_CATALOGUE_OPEN)
    assert rendered.endswith(subagent_mod._MCP_CATALOGUE_CLOSE)


def test_bool_normalisation_matches_the_settings_accessor() -> None:
    """The local spelling table cannot drift from the page's reading.

    ``harness/subagent.py`` may not import ``settings_io`` (the boundary pin in
    ``tests/unit/test_approval_source_boundary.py`` forbids the reach — a
    function-local import is still a reach), so the normaliser is spelled there
    as well as in ``settings_io.strict_bool``. This walks the two against each
    other across the spellings a YAML file and the settings page can produce:
    a change to either table fails here instead of silently splitting the two
    readers of one switch (the ``resume.py`` title-type shape).
    """
    from local_operator.settings_io import strict_bool

    spellings: list[object] = [
        True,
        False,
        0,
        1,
        2,
        -1,
        3.5,
        "true",
        "True",
        " yes ",
        "on",
        "1",
        "false",
        "FALSE",
        "no",
        "off",
        "0",
        "maybe",
        "",
        None,
        [],
        {},
    ]
    for spelling in spellings:
        for default in (True, False):
            assert subagent_mod._strict_bool(spelling, default) == strict_bool(spelling, default), (
                spelling,
                default,
            )


def test_strip_collapses_the_seam_and_keeps_neighbours() -> None:
    text = "head\n\n<mcps>dup\n</mcps>\n\ntail"
    assert subagent_mod._strip_marked_sections(text, "<mcps>", "</mcps>") == "head\n\ntail"


def test_strip_handles_first_and_last_sections() -> None:
    strip = subagent_mod._strip_marked_sections
    assert strip("<mcps>x</mcps>\n\nbody", "<mcps>", "</mcps>") == "body"
    assert strip("body\n\n<mcps>x</mcps>", "<mcps>", "</mcps>") == "body"


def test_strip_leaves_an_unclosed_marker_alone() -> None:
    """An unbounded marker must not eat the rest of the block."""
    text = "body\n\n<mcps>never closed"
    assert subagent_mod._strip_marked_sections(text, "<mcps>", "</mcps>") == text


def test_cap_clips_only_listing_bullets() -> None:
    limit = subagent_mod._CHILD_KNOWLEDGE_MAX_DESCRIPTION_CHARS
    text = "\n".join(
        [
            "Guides are procedures for this harness. An imperative that must survive.",
            "<guides>",
            f"- short: {'a' * limit}",  # at the cap: untouched
            f"- long: {'b' * (limit + 40)}",
            "</guides>",
            "<mcps>",
            f"- srv: {'c' * (limit + 40)}",  # outside a listing: untouched
            "</mcps>",
        ]
    )
    lines = subagent_mod._cap_listing_descriptions(text).splitlines()
    assert lines[0] == "Guides are procedures for this harness. An imperative that must survive."
    assert lines[2] == f"- short: {'a' * limit}"
    assert lines[3] == f"- long: {'b' * (limit - 1)}…"
    assert len(lines[3].split(": ", 1)[1]) <= limit
    assert lines[6] == f"- srv: {'c' * (limit + 40)}"


def test_slim_keeps_names_and_imperatives_and_drops_furniture() -> None:
    slim = subagent_mod._slim_child_knowledge(_parent_block(), drop_catalogue=True)
    # Names and the protocol imperatives survive byte-identically...
    assert "- repository-testing: " in slim
    assert "- minerva-router: " in slim
    assert "MUST read `guide://<name>` before acting" in slim
    assert "skills are virtual resources" in slim
    # ...and the repeated furniture is gone.
    assert "<mcps>" not in slim
    assert "<resource_recommendations>" not in slim
    assert LONG_DESCRIPTION not in slim  # clipped, not retained whole


def test_slim_keeps_the_inherited_catalogue_when_it_has_nothing_to_replace_it() -> None:
    slim = subagent_mod._slim_child_knowledge(_parent_block(), drop_catalogue=False)
    assert _GENERIC_CATALOGUE in slim
    assert "<resource_recommendations>" not in slim


# ---------------------------------------------------------------------------
# The composed child tail (integration, through the real build path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_child_tail_is_slimmed_by_default_with_exactly_one_catalogue(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stub_calls = _stub_child_mcp(monkeypatch)
    parent, _stream = _parent_session(tmp_path, _parent_block())
    try:
        child = await subagent_mod._build_child_session(
            label="review",
            prompt="review the change",
            parent_session=parent,
            model_spec=None,
            job_id="child-tail-job",
        )
        try:
            tail = await _child_tail(child)
            assert tail.count("<mcps>") == 1
            assert tail.count("</mcps>") == 1
            # The survivor is the CHILD's own catalogue, rendered against its
            # task prompt — not the parent's inherited copy.
            assert "own-server" in tail
            assert "Find MCP tools" not in tail
            # Every catalogue the child rendered came from its own task prompt
            # (construction may render the provider more than once).
            assert stub_calls
            assert set(stub_calls) == {"review the change"}
            assert "<resource_recommendations>" not in tail
            assert "- repository-testing: " in tail
            assert "- minerva-router: " in tail
            description = next(
                line for line in tail.splitlines() if line.startswith("- repository-testing:")
            ).split(": ", 1)[1]
            assert len(description) <= subagent_mod._CHILD_KNOWLEDGE_MAX_DESCRIPTION_CHARS
            assert description.endswith("…")
        finally:
            await child.dispose()
    finally:
        await parent.dispose()


@pytest.mark.asyncio
async def test_child_tail_is_the_parent_block_verbatim_when_the_gate_is_off(
    tmp_path, monkeypatch
) -> None:
    """``subagents.slim_child_knowledge: false`` restores the pre-slimmer shape.

    Including the duplicate catalogue: the switch disables the composition
    wholesale, which is what "inherit the parent block" has to mean.
    """
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.yml").write_text(
        "values:\n  subagents:\n    slim_child_knowledge: false\n", encoding="utf-8"
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    _stub_child_mcp(monkeypatch)
    parent, _stream = _parent_session(tmp_path, _parent_block())
    try:
        child = await subagent_mod._build_child_session(
            label="review",
            prompt="review the change",
            parent_session=parent,
            model_spec=None,
            job_id="child-tail-job",
        )
        try:
            tail = await _child_tail(child)
            assert tail.count("<mcps>") == 2
            assert _GENERIC_CATALOGUE in tail
            assert "<resource_recommendations>" in tail
            assert LONG_DESCRIPTION in tail
        finally:
            await child.dispose()
    finally:
        await parent.dispose()


@pytest.mark.asyncio
async def test_child_without_mcp_wiring_keeps_the_inherited_catalogue(
    tmp_path, monkeypatch
) -> None:
    """Nothing to replace it with: the inherited copy is the child's only hint."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(subagent_mod, "_child_mcp_wiring", lambda *args, **kwargs: None)
    parent, _stream = _parent_session(tmp_path, _parent_block())
    try:
        child = await subagent_mod._build_child_session(
            label="review",
            prompt="review the change",
            parent_session=parent,
            model_spec=None,
            job_id="child-tail-job",
        )
        try:
            tail = await _child_tail(child)
            assert _GENERIC_CATALOGUE in tail
            assert tail.count("<mcps>") == 1
            assert "<resource_recommendations>" not in tail
            assert "- repository-testing: " in tail
        finally:
            await child.dispose()
    finally:
        await parent.dispose()


@pytest.mark.asyncio
async def test_review_child_still_resolves_its_skills(tmp_path, monkeypatch) -> None:
    """The accuracy bar: a slimmed child that sees a name can still read it.

    Resolution is not a property of the tail text, so this drives the child's
    real resolver chain: ``skill://``/``guide://`` URLs fall through to the
    PARENT's resolver (which is what reaches the skill store), while
    ``mcp://`` stays on the child's own wiring.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    _stub_child_mcp(monkeypatch)
    parent, _stream = _parent_session(tmp_path, _parent_block())
    resolved: list[str] = []

    def parent_resolver(url: str) -> str:
        resolved.append(url)
        return f"body of {url}"

    parent._skill_resolver = parent_resolver
    try:
        child = await subagent_mod._build_child_session(
            label="review",
            prompt="verify the slimming",
            parent_session=parent,
            model_spec=None,
            job_id="child-tail-job",
        )
        try:
            tail = await _child_tail(child)
            assert "repository-testing" in tail
            assert "minerva-router" in tail
            # ``_skill_resolver`` is Optional on the session; the child build
            # always installs the closure, and narrowing it here is what keeps
            # the assertions callable under pyright's strict optional mode.
            resolver = child._skill_resolver
            assert resolver is not None
            skill_url = "skill://minerva-router"
            assert resolver(skill_url) == f"body of {skill_url}"
            assert resolver("guide://repository-testing") == "body of guide://repository-testing"
            assert resolved == ["skill://minerva-router", "guide://repository-testing"]
            # The child's own MCP resolver answered first and declined, and an
            # mcp:// URL is never routed to the parent (it would activate on
            # the wrong session).
            assert resolver("mcp://somewhere/tool") is None
            assert resolved == ["skill://minerva-router", "guide://repository-testing"]
        finally:
            await child.dispose()
    finally:
        await parent.dispose()
