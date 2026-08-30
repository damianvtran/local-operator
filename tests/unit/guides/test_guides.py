"""Packaged guide discovery, routing, and progressive-disclosure contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.guides import discover_guides, make_guide_resolver
from local_operator.prompts_api import render_template
from local_operator.session_factory import (
    _KnowledgeHooks,
    _registered_agent_hints,
    _select_knowledge_block,
)
from local_operator.skills.discovery import Skill
from local_operator.skills.embeddings import LocalEmbedder
from local_operator.skills.index import SkillIndex, render_block


def test_packaged_catalog_is_small_and_descriptions_are_prompt_sized() -> None:
    guides = discover_guides()

    assert [guide.name for guide in guides] == [
        "agents",
        "browser",
        "configuration",
        "extensions",
        "failover",
        "mcp",
        "mobile",
        "peer-messaging",
        "teams",
    ]
    assert all(guide.resource_type == "guide" for guide in guides)
    assert all(40 <= len(guide.description) <= 180 for guide in guides)


def test_guide_protocol_reads_body_only_on_demand() -> None:
    guides = {guide.name: guide for guide in discover_guides()}
    resolver = make_guide_resolver(guides)

    assert resolver("skill://configuration") is None
    body = resolver("guide://configuration")
    assert body is not None
    assert "# Local Operator configuration" in body
    assert "LOCAL_OPERATOR_CONFIG_DIR" in body
    error = resolver("guide://configuration/../../credentials.env")
    assert error is not None
    assert "not allowed" in error


def test_guide_listing_never_contains_guide_body() -> None:
    configuration = next(guide for guide in discover_guides() if guide.name == "configuration")

    block = render_block([configuration])

    assert configuration.description in block
    assert "LOCAL_OPERATOR_CONFIG_DIR" not in block

    assert "guide://<name>" in render_template("system.md", {})
    assert "<skills>" not in block


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("install and configure an MCP server in Local Operator", "mcp"),
        ("set the default provider and model configuration", "configuration"),
        ("create a Local Operator skill or executable plugin", "extensions"),
        ("list available agents or spawn a subagent", "agents"),
        ("set up phone access so I can drive lop from my mobile", "mobile"),
    ],
)
async def test_each_guide_routes_from_representative_task(
    tmp_path: Path, query: str, expected: str
) -> None:
    guides = discover_guides()
    # A Guide satisfies the Skill protocol the index consumes; the annotation
    # names the concrete class, so the cast is what admits the sibling type.
    index = SkillIndex(cast(Any, guides), LocalEmbedder(), cache_dir=tmp_path / "cache")
    await index.build()

    selected = await index.select(query)

    assert expected in {guide.name for guide in selected}


@pytest.mark.asyncio
async def test_registered_agent_metadata_surfaces_only_generic_guide(
    tmp_path: Path,
) -> None:
    specialist = SimpleNamespace(
        id="db-1",
        name="Database specialist",
        description="PostgreSQL query tuning database indexes and execution plans",
        tags=["postgresql", "database"],
        categories=["performance"],
    )
    registry = SimpleNamespace(config_dir=tmp_path, list_agents=lambda: [specialist])
    hints = _registered_agent_hints(cast(Any, registry))
    agents_guide = next(guide for guide in discover_guides() if guide.name == "agents")

    main_index = SkillIndex([agents_guide], LocalEmbedder(), cache_dir=tmp_path / "main-cache")
    hint_index = SkillIndex(hints, LocalEmbedder(), cache_dir=tmp_path / "hint-cache")
    await main_index.build()
    await hint_index.build()
    hooks = _KnowledgeHooks(
        index=main_index,
        agent_hint_index=hint_index,
        guides_by_name={"agents": agents_guide},
    )

    block = await _select_knowledge_block(
        hooks, "Tune this PostgreSQL query using its indexes and execution plan"
    )

    assert "- agents:" in block
    assert "Database specialist" not in block
    assert specialist.description not in block
    assert "<skills>" not in block


def test_agent_hint_rows_are_never_rendered(tmp_path: Path) -> None:
    specialist = SimpleNamespace(
        id="review-1",
        name="Reviewer",
        description="Review Python code",
        tags=[],
        categories=[],
    )
    registry = SimpleNamespace(config_dir=tmp_path, list_agents=lambda: [specialist])

    assert render_block(_registered_agent_hints(cast(Any, registry))) == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["config", "edit", "hosting", "openrouter"],
        ["--hosting", "openrouter", "--model", "openai/gpt-4.1", "exec", "summarize"],
        ["mcp", "add", "demo", "--command", "demo-mcp", "--arg", "serve"],
        ["agents", "list", "--page", "2", "--perpage", "10"],
        ["exec", "review this", "--agent", "research"],
    ],
)
def test_documented_cli_shapes_parse(argv: list[str]) -> None:
    from local_operator.cli import build_cli_parser

    build_cli_parser().parse_args(argv)


@pytest.mark.asyncio
async def test_custom_instructions_task_routes_to_the_configuration_guide(
    tmp_path: Path,
) -> None:
    """The phrasings a user actually reaches for when moving standing rules
    into Local Operator must select the guide that names the real file."""
    guides = discover_guides()
    index = SkillIndex(cast(Any, guides), LocalEmbedder(), cache_dir=tmp_path / "cache")
    await index.build()

    for query in (
        "update the system prompt / custom instructions",
        "copy my AGENTS.md standing rules into local-operator",
    ):
        selected = {guide.name for guide in await index.select(query)}
        assert "configuration" in selected, query


def test_mobile_guide_requires_a_password_delivery_ask() -> None:
    """An agent that 'just prints the password' is the failure this guide exists
    to prevent. The four channels and the ask-first rule have to be in the
    body, not implied."""
    body = make_guide_resolver({guide.name: guide for guide in discover_guides()})("guide://mobile")

    assert body is not None
    assert "ask" in body.lower()
    assert "Keychain" in body
    assert "pbcopy" in body
    assert "0600" in body
    assert "Never invent a fourth channel" in body
    assert "lop mobile install" in body
    assert "Show it once" not in body
    assert "context window" in body


def test_browser_and_agent_guides_require_terminal_surface_cleanup() -> None:
    guides = {guide.name: guide for guide in discover_guides()}
    resolver = make_guide_resolver(guides)

    browser = resolver("guide://browser")
    agents = resolver("guide://agents")
    assert browser is not None and agents is not None
    assert "Close before your final answer" in browser
    assert "Long-lived TUI/cmux processes stay alive between turns" in browser
    assert "close failed and the handle was dropped" in browser
    assert "Before a subagent's terminal handoff" in agents
    assert "put child disposal in `finally`" in agents


def test_configuration_guide_names_the_real_instructions_file() -> None:
    # The guide exists so an agent does not have to infer this from source and
    # end up editing a file nothing reads.
    body = make_guide_resolver({guide.name: guide for guide in discover_guides()})(
        "guide://configuration"
    )

    assert body is not None
    assert "system_prompt.md" in body
    # The two mechanisms that look authoritative but are not.
    assert "no `custom_instructions` key" in body
    assert "next session, not the running one" in body


def test_system_prompt_demands_a_guide_read_before_acting() -> None:
    # A soft "may appear" was not enough to make the protocol fire on a real
    # configuration question; the rule has to be imperative.
    text = render_template("system.md", {})

    assert "guide://<name>" in text
    assert "MUST" in text


def test_the_guide_resolver_never_raises_on_an_unreadable_body(tmp_path: Path) -> None:
    """The adapter's contract is "never raises", on both resource surfaces.

    ``guide://`` and ``skill://`` share ``resolve_resource_url``, so every
    filesystem failure reachable through one is reachable through the other.
    This adapter caught only ``ValueError``, so a GUIDE.md that exists but
    denies permission (deleted or chmod-000'd between discovery and the read)
    escaped as a ``PermissionError`` out of a resolver the read tool trusts not
    to raise — while the identical skill URL returned the message as content
    (review F7). The two must stay in step.
    """
    base = tmp_path / "packaged"
    base.mkdir()
    body = base / "GUIDE.md"
    body.write_text("# body\n", encoding="utf-8")
    body.chmod(0o000)
    guide = Skill(
        name="demo",
        description="d" * 60,
        file_path=body,
        base_dir=base,
        source=str(tmp_path),
        resource_type="guide",
    )

    resolver = make_guide_resolver({"demo": guide})
    try:
        result = resolver("guide://demo")
    finally:
        body.chmod(0o600)  # let tmp_path cleanup remove it

    # Returned AS CONTENT, not raised: the model sees why and can self-correct.
    assert result is not None
    assert "Permission denied" in result


def test_a_looping_guide_base_dir_does_not_escape_the_resolver(tmp_path: Path) -> None:
    """A looping ``base_dir`` is a bad resource, not a crash (review F6).

    ``Path.resolve()`` reports ELOOP as ``RuntimeError`` on 3.12/3.13 and as a
    non-strict success on 3.14, so the guards around it must catch both or the
    behaviour forks by interpreter. Driven through the guide adapter because
    that is where the "never raises" contract lives.
    """
    loop = tmp_path / "loopbase"
    loop.symlink_to(tmp_path / "loopbase", target_is_directory=True)
    guide = Skill(
        name="demo",
        description="d" * 60,
        file_path=loop / "GUIDE.md",
        base_dir=loop,
        source=str(tmp_path),
        resource_type="guide",
    )

    resolver = make_guide_resolver({"demo": guide})

    # Neither the child listing nor the bare read may raise; both report.
    child = resolver("guide://demo/references")
    assert child is not None and "not found" in child
    assert resolver("guide://demo") is not None
