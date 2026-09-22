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
        "system-tools",
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
        # The lexical router is deliberately crude (a hashed n-gram embedder,
        # not a model), so this row is the SHAPE it can actually see: the
        # missing-command signal and the package-manager vocabulary. A task that
        # merely needs a conversion ("convert this video to mp4") is NOT matched
        # by this router and is not asserted here — the classification layer's
        # LLM roster is the path that carries those, and overstating the router
        # in a test would pin a claim it cannot keep.
        ("ffmpeg: command not found, I need to install it", "system-tools"),
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


def test_every_guide_cross_reference_resolves() -> None:
    """A `guide://<name>` that names no discovered guide is a silent dead end.

    The corpus tells a model to go read another guide at the exact moment it is
    stuck (a missing tool, a refused call), and nothing verified the name it is
    sent to. Nothing in the read path can catch it either: ``read`` on an
    unknown guide returns the available names as CONTENT, which is a recovery
    for the model and invisible to a reviewer, so a one-word typo in a
    cross-reference ships as a working-looking sentence.

    Scoped to the packaged corpus, which is the only place a guide may be
    referenced from — a guide in a user's own skill tree is not this catalog's
    business, and the catalogue is what ships.
    """
    guides = discover_guides()
    names = {guide.name for guide in guides}
    assert names, "the packaged catalog discovered nothing to check"

    unreachable: list[str] = []
    self_refs: list[str] = []
    for guide in guides:
        body = guide.file_path.read_text(encoding="utf-8", errors="replace")
        for target in set(re.findall(r"guide://([a-zA-Z0-9_-]+)", body)):
            if target not in names:
                unreachable.append(f"{guide.name} -> guide://{target}")
            if target == guide.name:
                self_refs.append(f"{guide.name} -> guide://{target}")

    assert not unreachable, f"cross-reference names no discovered guide: {unreachable}"
    assert not self_refs, f"a guide must not tell the model to read itself: {self_refs}"


def test_every_guide_reference_in_the_code_resolves() -> None:
    """The other half of the dead end, one directory over (review round 1, R1-8).

    The corpus walk above cannot see the references the HARNESS itself prints:
    the missing-tool advisory names `guide://system-tools` from `builtin.py`, and
    a rename of that guide would leave the harness pointing at a name the
    resolver reports as unknown — with nothing but the model's own recovery to
    notice. The test that guards guide-to-guide links should guard
    code-to-guide links with it, because the same rename breaks both and only
    one of them was checked.

    Scoped to the packaged `local_operator/` tree: that is the code that ships
    beside the guides and therefore the only code whose references this catalog
    can promise. A reference in a user's own script is theirs to get right.

    BOTH `.py` and `.md`, the latter because the highest-traffic reference site in
    the harness is the packaged system prompt: `prompts_md/system.md` carries four
    `guide://` pointers and rides every session on every turn, so a rename there
    is a dead end in front of every model — and a `.py`-only walk could not see it
    (QA round 1, Q4, which demonstrated exactly that by breaking the prompt and
    watching the test stay green).
    """
    root = Path(discover_guides()[0].file_path).resolve().parents[2]
    assert (root / "guides").is_dir(), f"unexpected package layout at {root}"
    names = {guide.name for guide in discover_guides()}

    # `guide://<name>` placeholders are excluded by the pattern itself (the
    # character class stops at `<`), which is why the protocol's own prose in
    # `skills/index.py` and the prompts does not trip this.
    dangling: list[str] = []
    walked = 0
    for pattern in ("*.py", "*.md"):
        for path in sorted(root.rglob(pattern)):
            walked += 1
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                for target in set(re.findall(r"guide://([a-zA-Z0-9_-]+)", line)):
                    if target not in names:
                        dangling.append(f"{path.relative_to(root)}:{lineno} -> guide://{target}")

    assert walked > 100, f"the walk found only {walked} files, which is not this tree"
    assert not dangling, f"code references a guide that does not exist: {dangling}"


def test_system_tools_guide_agrees_with_the_console_guide_on_approval() -> None:
    """The one rule two guides state, so the two texts cannot drift apart.

    The console guide owns the rule that the harness gate authorises the CALL
    while ``ask`` authorises the CHANGE to the user's machine. The install guide
    is the situation where a model is most likely to compress the two into one,
    so it has to send the model to that rule rather than paraphrase it into a
    subtly weaker one of its own.
    """
    body = make_guide_resolver({guide.name: guide for guide in discover_guides()})(
        "guide://system-tools"
    )

    assert body is not None
    assert "console guide already carries that rule" in body
    assert "Never install anything silently" in body
    # The Windows elevation limit is the finding this guide exists to state
    # rather than paper over: the surface cannot answer a UAC dialog.
    assert "surface cannot" in body and "answer it" in body
    assert "UAC" in body


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
    # The lifetime, stated once and stated so it cannot be read as ephemeral. The
    # earlier wording ("deleted with the session") was read by a measured session
    # as meaning "like a temp directory", and it kept a duplicate copy of its state
    # outside the pad for an hour rather than test that reading: the pad is
    # session-DIR backed and rides out runtime restarts and rollovers.
    assert body.count("survives runtime restarts") == 1
    assert "deleted with the session" not in body
    assert "dies with the session" not in body
    # The shell channel's one limit, in the copy an agent reads BEFORE choosing
    # where to write (round 1, R4): a relative redirect is not resolved.
    assert "needs the path named absolutely" in body
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


def test_scratchpad_guide_states_the_boundary_of_the_content_policy() -> None:
    """The content policy is enforced at the TOOLS, and this guide is the
    document read right next to the ``$LOCAL_OPERATOR_SCRATCHPAD`` recipe that
    hands a shell the pad path.

    Review round 1 (F1): the bullet said build output was "refused by name if you
    try" beside that recipe, which reads as a property of the pad — while the
    channel that produced the measured 34.8 GB (a compiler and a package manager
    in a shell) is not policed at all, and the check has one call site either
    way. The claim and its boundary are pinned TOGETHER, because the failure this
    guards is a later revision keeping the rule and dropping the honest half —
    which is exactly how the over-claim got written.
    """
    resolver = make_guide_resolver({guide.name: guide for guide in discover_guides()})
    body = resolver("guide://scratchpad")
    assert body is not None

    section = body[body.index("## Use something else for") : body.index("## The protocol")]
    # Whitespace-collapsed: the guide is PROSE and re-wraps as it is edited, so an
    # assertion on the raw bytes would pin the line width rather than the claim.
    collapsed = " ".join(section.split())

    assert "Build output" in collapsed
    assert "git worktree add" in collapsed
    # The boundary: the tools are checked, a shell is not.
    assert "enforced at the TOOLS and not in a shell" in collapsed
    assert "NOT policed" in collapsed


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
