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
from local_operator.skills.embeddings import LocalEmbedder
from local_operator.skills.index import SkillIndex, render_block


def test_packaged_catalog_is_small_and_descriptions_are_prompt_sized() -> None:
    guides = discover_guides()

    assert [guide.name for guide in guides] == [
        "agents",
        "browser",
        "configuration",
        "extensions",
        "mcp",
        "mobile",
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
