"""Packaged guide discovery, routing, and progressive-disclosure contracts."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.guides import discover_guides, make_guide_resolver
from local_operator.prompts_api import render_template
from local_operator.scratchpad import SCRATCHPAD_PATH_ENV
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
        "classification",
        "configuration",
        "console",
        "credentials",
        "extensions",
        "failover",
        "mcp",
        "mobile",
        "peer-messaging",
        "qwencloud",
        "scratchpad",
        "teams",
        "tunnel",
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
        ("create a Radient personal tunnel with OpenCode routes and billing", "tunnel"),
        ("why is my usage blank", "qwencloud"),
        (
            "turn on the smart agent hints and check which vendor served the call",
            "classification",
        ),
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


def test_scratchpad_guide_states_the_rules_no_tool_schema_can() -> None:
    """The properties that belong to the guide: the lifetime (claimed once),
    the text-only rule, that a one-off SCRIPT and a data file are as much at
    home here as a note, the naming rule the desktop canvas depends on, the
    fallback when the host has no session folder, and the fact that every
    result carries the absolute path a shell needs.
    """
    resolver = make_guide_resolver({guide.name: guide for guide in discover_guides()})
    body = resolver("guide://scratchpad")

    assert body is not None
    assert body.count("deleted with the session") == 1
    assert "one-off script" in body
    assert "Data you are still shaping" in body
    assert "real extension" in body
    assert "absolute path" in body
    assert "mktemp" in body
    assert "do not put scratch in the user's" in body
    # The guide is the authority on the five calls, and must list five.
    assert "Five calls" in body
    assert body.count("| `") == 5


def test_scratchpad_guide_prints_no_absolute_path_shaped_example() -> None:
    """A guide is prompt text that lands in a transcript, and the desktop Files
    panel infers its tiles from absolute paths found in transcript text. An
    ELIDED or templated path in prose therefore produced phantom tiles on the
    real app — two of them reproduced from a guide's own example lines
    (measured against the desktop build). So examples are URL-shaped, and a
    resolved path is described in words rather than templated.
    """
    resolver = make_guide_resolver({guide.name: guide for guide in discover_guides()})
    body = resolver("guide://scratchpad")
    assert body is not None

    assert "/…" not in body
    assert "<id>" not in body
    # No bare absolute path of any kind: every example must carry the scheme.
    assert (
        re.search(r"(?<![\w./:-])/(Users|home|tmp|var|private|sessions|scratchpad)/", body) is None
    )


def test_scratchpad_guide_says_where_binary_scratch_goes() -> None:
    """The guide separates the two cases an earlier revision conflated, and the
    separation is a MEASURED correction rather than a preference: a PNG written
    into the pad by ``bash`` reads back through the scheme as a viewable image,
    while a non-image binary has no text to return and is refused. Only the
    second needs a real temp dir — the home this bullet names (the per-user temp
    directory, NOT the one macOS reaps) plus where to record the path it made —
    so the omission ``system.md`` leaves is answered where the reader lands
    instead of being a gap to fall into.
    """
    resolver = make_guide_resolver({guide.name: guide for guide in discover_guides()})
    body = resolver("guide://scratchpad")
    assert body is not None
    section = body[body.index("## Use something else for") : body.index("## The protocol")]
    assert "**A NON-image binary**" in section
    assert "mktemp" in section
    assert "$TMPDIR" in section
    # The verified mechanism, named rather than gestured at, and the window it
    # prunes on — the reason the guide gives for avoiding that one directory.
    assert "com.apple.tmp_cleaner" in section
    assert "three days" in section
    # Where the made temp dir's path is recorded, so a later turn finds the files.
    assert "scratchpad://" in section


def test_scratchpad_guide_makes_a_rendered_frame_first_class_content() -> None:
    """The correction this pin exists for. The guide used to say a binary put in
    the pad "cannot be read back — the reader refuses it", and it named an image
    FIRST, so the single most common scratch artifact this fleet makes (681 PNGs
    sat in ``/tmp`` at the time of the audit) was sent away from the pad by the
    document that is supposed to hold it. Measured 2026-09-21: a PNG written into
    the pad by ``bash`` and read back through the scheme renders as an image, and
    only a NON-image binary is refused — for having no text to return, which is a
    different fact from being unreadable.

    The stale claim and its replacement are pinned together, because the failure
    this guards is a future revision restoring the blanket refusal: that reads as
    a harmless simplification and would silently re-create the funnel.
    """
    resolver = make_guide_resolver({guide.name: guide for guide in discover_guides()})
    body = resolver("guide://scratchpad")
    assert body is not None

    assert "cannot be read back" not in body
    assert "A rendered frame or a still" in body
    # Whitespace-collapsed: the guide is PROSE and re-wraps as it is edited, so
    # an assertion on the raw bytes pins the line width rather than the claim —
    # it broke on a rewording that changed nothing else.
    permitted = " ".join(
        body[body.index("## Use it for") : body.index("## Use something else for")].split()
    )
    assert "VIEWABLE image" in permitted
    # The distinction is stated where the reader decides, not only where the
    # reader is told to go elsewhere.
    assert "NON-image binary" in body


def test_scratchpad_guide_names_the_exported_path_and_the_pad_local_mktemp() -> None:
    """The path is what a shell actually needs, and the guide is where an agent
    looks after being nudged. Both the variable and the idiom have to be here:
    `mktemp -d` with no template is the sanctioned escape hatch, so an agent that
    wants a private rig directory has to be shown the TEMPLATED form that keeps
    the directory inside the pad — otherwise the nudge and the guide disagree and
    the escape hatch wins.
    """
    resolver = make_guide_resolver({guide.name: guide for guide in discover_guides()})
    body = resolver("guide://scratchpad")
    assert body is not None

    assert f"${SCRATCHPAD_PATH_ENV}" in body
    assert f'mktemp -d "${SCRATCHPAD_PATH_ENV}/rig.XXXXXX"' in body
    # The unset case is stated, because the operator's own terminal does not
    # have it and a recipe that assumes it is a recipe that fails there.
    assert "unset in the user's terminal" in body


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


def test_classification_guide_names_the_switch_the_logins_and_the_log_line() -> None:
    """The facts a reader cannot infer from the code they are standing in.

    The switch is off by default and the section is new-session-scoped, so an
    agent asked "why are there no hints" needs the exact key; the cascade is
    three legs with three separate logins, and the guide is the only place that
    says which command buys which one. The provider ids are asserted against the
    live registry rather than merely spelled, because a renamed leg would
    otherwise leave the guide advertising a login that no longer exists — the
    one failure mode this guide can have that costs a user a round trip.
    """
    from local_operator.providers.registry import known_provider_ids

    resolver = make_guide_resolver({guide.name: guide for guide in discover_guides()})
    body = resolver("guide://classification")

    assert body is not None
    assert "lop config edit classification.auto true" in body
    assert "values.classification" in body
    assert "classification: vendor=" in body
    for leg in ("radient", "typesafe", "openrouter"):
        assert leg in known_provider_ids()
        assert f"lop login {leg}" in body


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
