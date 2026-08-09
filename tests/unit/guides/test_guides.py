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
        "configuration",
        "extensions",
        "mcp",
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
