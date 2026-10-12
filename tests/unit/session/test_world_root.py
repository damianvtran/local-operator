"""``SessionSpec.world_root`` — the model-world boundary, over the REAL SDK.

WHY THIS FILE EXISTS. An episode session runs a real model on the operator's
host. Before the boundary, an episode's tools were confined to the whole
scratch tree — which carries the seeded controller store (the secret store,
``auth.db``, provider config) — and project discovery walked cwd ancestors to
``/``, so even an isolated ``HOME`` discovered the real operator's skills and
guidance (measured in the architect's canary). This slice adds
``SessionSpec.world_root``: validated once before construction, installed as
the Session's confinement AT construction, and clipped into the three
discovery walks — nothing above the world is silently inherited, packaged
knowledge and explicitly scoped controller sources stay, and a confined
session's shell runs under the strict environment allowlist.

These tests drive the REAL SDK session and AgentLoop over a scripted provider
transport (``clients.client_for_spec`` patched for the ``test`` hosting,
the same seam the session-arm rig uses) — never a fake Session — so every
assertion is about the machinery an episode actually runs.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator import sdk
from local_operator.evaluation.session_arm import (
    EPISODE_TOOL_SURFACE,
    SessionArmError,
    validate_episode_session_spec,
)
from local_operator.session.spec import (
    ApprovalPolicy,
    SessionRoots,
    SessionSpec,
    SessionSpecError,
)

CONTROLLER_CANARY = "CONTROLLER_CANARY_NOT_A_SECRET"
WORKSPACE_CANARY = "WORKSPACE_CANARY"
OUTER_GUIDANCE = "OUTER-GUIDANCE-SENTINEL"
INNER_GUIDANCE = "IN-WORLD-GUIDANCE-SENTINEL"
OUTER_SKILL = "outer-sentinel-skill"
INNER_SKILL = "inner-sentinel-skill"
CHILD_PROBE = "CHILD-WORLD-PROBE"
MCP_WORLD_FIXTURE = "world-echo-fixture"
MCP_OUTER_CURSOR = "outer-cursor-sentinel"
MCP_OUTER_PROJECT = "outer-project-sentinel"
MCP_OUTER_MARKER = "OUTER-CURSOR-BODY"

#: The task tool's result names its background job as ``job <hex id>`` (the
#: same shape the session-arm rig parses).
_JOB_RE = re.compile(r"job ([0-9a-f]{6,})")

_SKILL_TEMPLATE = "---\nname: {name}\ndescription: {description}\n---\n\nbody\n"


@pytest.fixture
def episode(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A controller/world split in the shape the episode caller builds.

    ``home/.local-operator`` is the controller store (and HOME), ``world`` is
    the model's world, and the tree ABOVE the world carries the sentinels a
    boundary must hide: an outer AGENTS.md, an outer skill, and the dummy
    controller file. Everything the model may legitimately see is planted
    inside ``world``.

    NOT ``tmp_path``: the seatbelt profile deliberately permits READS of
    ``/private/var/folders`` (the per-user temp — dyld caches and mktemp
    machinery live there), so a world sited there cannot demonstrate the
    kernel denial at all — and since QA round 1 the SDK preflight REFUSES
    layouts whose stores sit under such a tree outright (see
    ``test_controller_state_under_kernel_exposed_trees_is_refused``): the
    profile cannot deny a child path back out of a broad machinery allow.
    The tree lives under the REAL home (the same idiom
    ``test_build_and_scripts.durable_path`` uses) and is removed afterwards.
    """
    import pwd
    import shutil
    import uuid

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    base = real_home / ".cache" / "lop-world-root-tests" / uuid.uuid4().hex[:12]
    base.mkdir(parents=True, exist_ok=True)
    request.addfinalizer(lambda: shutil.rmtree(base, ignore_errors=True))
    home = base / "home"
    config = home / ".local-operator"
    agent_home = home / "agent-home"
    world = base / "world"
    for directory in (config, agent_home, world):
        directory.mkdir(parents=True, exist_ok=True)

    (world / "allowed.txt").write_text(WORKSPACE_CANARY + "\n")
    (world / "AGENTS.md").write_text("# World guidance\n" + INNER_GUIDANCE + "\n")
    inner_skill = world / ".local-operator" / "skills" / INNER_SKILL
    inner_skill.mkdir(parents=True, exist_ok=True)
    (inner_skill / "SKILL.md").write_text(
        _SKILL_TEMPLATE.format(name=INNER_SKILL, description=INNER_GUIDANCE)
    )

    (config / "dummy-controller.txt").write_text(CONTROLLER_CANARY + "\n")
    (base / "AGENTS.md").write_text("# Outer guidance\n" + OUTER_GUIDANCE + "\n")
    outer_skill = base / ".local-operator" / "skills" / OUTER_SKILL
    outer_skill.mkdir(parents=True, exist_ok=True)
    (outer_skill / "SKILL.md").write_text(
        _SKILL_TEMPLATE.format(name=OUTER_SKILL, description=OUTER_GUIDANCE)
    )

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(agent_home))
    # A live harness exports its own shell markers; the ordinary caller path
    # is what this file exercises, so they are stripped here.
    from local_operator import agent_shell

    monkeypatch.delenv(agent_shell.AGENT_SHELL_ENV, raising=False)
    monkeypatch.delenv(agent_shell.MAY_DELEGATE_ENV, raising=False)
    monkeypatch.delenv(agent_shell.ALLOW_NESTED_SESSION_ENV, raising=False)

    roots = SessionRoots(config_dir=config, agent_home=agent_home, cwd=world, allow_volatile=True)
    return SimpleNamespace(
        base=base, home=home, config=config, agent_home=agent_home, world=world, roots=roots
    )


class ScriptedWorldClient:
    """Deterministic transport: the parent probes, the child reports.

    Stateless per request (the decision reads only the request), so parent and
    child streams may interleave on one loop without corrupting each other.
    Every request's user-visible text is recorded for the discovery
    assertions; every tool-role text is recorded for the outcome assertions.
    """

    def __init__(self, episode: SimpleNamespace) -> None:
        self.episode = episode
        self.requests: list[str] = []
        self.system_texts: list[str] = []
        self.tool_texts: list[str] = []
        #: ``"probe"`` runs the world script; ``"echo"`` answers every request
        #: with a text stop — the shape a spec with no scripted tools needs.
        self.mode = "probe"

    async def stream(
        self, request: Any, api_key: str | None = None, oauth_access: str | None = None
    ) -> AsyncIterator[Any]:
        messages = list(getattr(request, "messages", None) or [])
        joined = "\n".join(getattr(m, "text", "") or "" for m in messages)
        rows = [
            getattr(m, "text", "") or "" for m in messages if getattr(m, "role", None) == "tool"
        ]
        self.requests.append(joined)
        self.system_texts.append(
            "\n".join(str(block) for block in (getattr(request, "system_blocks", None) or []))
        )
        self.tool_texts.extend(rows)
        if self.mode == "echo":
            events = self._text_stop("echo", len(rows))
        else:
            is_child = any(
                CHILD_PROBE in (getattr(m, "text", "") or "")
                for m in messages
                if getattr(m, "role", None) == "user"
            )
            events = self._child_events(rows) if is_child else self._parent_events(rows)
        for event in events:
            yield event

    # -- event shapes (copied from the session-arm rig's wire client) ---------

    def _tool_call(self, index: int, name: str, arguments: dict[str, Any]) -> list[Any]:
        from local_operator.harness.types import (
            StreamEndEvent,
            StreamStartEvent,
            StreamToolCallDelta,
            StreamUsageEvent,
            Usage,
        )

        return [
            StreamStartEvent(response_id=f"world-probe-{index}"),
            StreamToolCallDelta(
                index=0,
                id=f"world-probe-call-{index}",
                name=name,
                argument_delta=json.dumps(arguments),
            ),
            StreamUsageEvent(usage=Usage(input_tokens=10, output_tokens=5)),
            StreamEndEvent(stop_reason="toolUse", usage=Usage(input_tokens=10, output_tokens=5)),
        ]

    def _text_stop(self, text: str, index: int) -> list[Any]:
        from local_operator.harness.types import (
            StreamEndEvent,
            StreamStartEvent,
            StreamTextDelta,
            StreamUsageEvent,
            Usage,
        )

        return [
            StreamStartEvent(response_id=f"world-probe-text-{index}"),
            StreamTextDelta(delta=text),
            StreamUsageEvent(usage=Usage(input_tokens=5, output_tokens=2)),
            StreamEndEvent(stop_reason="stop", usage=Usage(input_tokens=5, output_tokens=2)),
        ]

    # -- the parent's script --------------------------------------------------

    def _parent_events(self, rows: list[str]) -> list[Any]:
        stage = len(rows)
        episode = self.episode
        controller = episode.config / "dummy-controller.txt"
        world = episode.world
        if stage == 0:
            return self._tool_call(0, "read", {"path": str(world / "allowed.txt")})
        if stage == 1:
            return self._tool_call(1, "read", {"path": str(controller)})
        if stage == 2:
            return self._tool_call(2, "bash", {"command": f"cat {controller}"})
        if stage == 3:
            # The strict shell allowlist: a credential-shaped ambient variable
            # the deployment exported must not be visible to the model's own
            # command; PATH/HOME (the base allowlist) must be.
            return self._tool_call(3, "bash", {"command": "env && pwd"})
        if stage == 4:
            return self._tool_call(
                4,
                "bash",
                {"command": f"ln -sf {controller} {world}/escape.txt; cat {world}/escape.txt"},
            )
        if stage == 5:
            return self._tool_call(5, "read", {"path": str(world / "escape.txt")})
        if stage == 6:
            # A controller capability the pinned surface excludes: it must not
            # resolve at all, declared inventory or not.
            return self._tool_call(6, "secret", {"op": "list"})
        if stage == 7:
            return self._tool_call(
                7,
                "task",
                {
                    "label": "world-probe-child",
                    "prompt": (
                        CHILD_PROBE
                        + ": read the controller canary file and the world's allowed.txt, "
                        "then report."
                    ),
                    "agent": "scout",
                },
            )
        if stage in (8, 9):
            for row in reversed(rows):
                match = _JOB_RE.search(row)
                if match:
                    return self._tool_call(
                        stage, "wait", {"job_id": match.group(1), "wait_ms": 120000}
                    )
            # No job id yet (or the task ran inline): ask the job table, then
            # wind down regardless.
            return self._tool_call(stage, "jobs", {"op": "list"})
        if not any("child-report" in row for row in rows) and stage < 15:
            # The child has not reported yet: keep witnessing on its job.
            for row in reversed(rows):
                match = _JOB_RE.search(row)
                if match:
                    return self._tool_call(
                        stage, "wait", {"job_id": match.group(1), "wait_ms": 120000}
                    )
            return self._tool_call(stage, "jobs", {"op": "list"})
        if not any(f"name: {INNER_SKILL}" in row for row in rows):
            # THE KNOWLEDGE BOUNDARY, through the real read tool: an in-world
            # skill resolves and its content comes back.
            return self._tool_call(stage, "read", {"path": f"skill://{INNER_SKILL}"})
        if not any("Unknown skill" in row for row in rows):
            # ... and the outer ancestor's twin does NOT resolve — the
            # discovery clip itself, observed from the session.
            return self._tool_call(stage, "read", {"path": f"skill://{OUTER_SKILL}"})
        return self._text_stop("world-probe complete", stage)

    # -- the child's script ---------------------------------------------------

    def _child_events(self, rows: list[str]) -> list[Any]:
        stage = len(rows)
        episode = self.episode
        if stage == 0:
            return self._tool_call(
                0, "read", {"path": str(episode.config / "dummy-controller.txt")}
            )
        if stage == 1:
            return self._tool_call(1, "read", {"path": str(episode.world / "allowed.txt")})
        if stage == 2:
            # Packaged knowledge is trusted by provenance and must reach the
            # confined CHILD too (QA round 2, Q5 — it disappeared here).
            return self._tool_call(2, "read", {"path": "skill://design-qa"})
        joined = "\n".join(rows)
        denied = ("outside it" in joined) or ("Operation not permitted" in joined)
        allowed = WORKSPACE_CANARY in joined
        return self._text_stop(f"child-report denied={denied} allowed={allowed}", stage)


@pytest.fixture
def scripted_(episode: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> ScriptedWorldClient:
    """The scripted transport, wired the way the session-arm rig wires one."""
    import local_operator.providers.clients as clients_mod

    client = ScriptedWorldClient(episode)
    original = clients_mod.client_for_spec

    def patched(spec: Any, **kwargs: Any) -> Any:
        if getattr(spec, "provider", None) == "test":
            return client
        return original(spec, **kwargs)

    monkeypatch.setattr(clients_mod, "client_for_spec", patched)
    return client


def _episode_spec(world: Path) -> SessionSpec:
    """The pinned episode posture: world boundary + declared surface."""
    return SessionSpec(
        hosting="test",
        model="mock",
        world_root=world,
        tools=EPISODE_TOOL_SURFACE,
        approvals=ApprovalPolicy.declared(EPISODE_TOOL_SURFACE),
    )


@pytest.mark.asyncio
async def test_the_world_boundary_holds_end_to_end_over_the_real_sdk(
    episode: SimpleNamespace,
    scripted_: ScriptedWorldClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LO_TEST_API_KEY", "should-never-reach-the-model-shell")
    async with sdk.open_session(_episode_spec(episode.world), roots=episode.roots) as session:
        # The SDK hands back ``SessionProtocol``; the concrete-only members
        # read here are widened the way ``test_sdk.py`` widens them.
        concrete = cast(Any, session)
        await session.prompt("world boundary probe")
        # 1. Construction-time confinement: the world root itself.
        assert Path(concrete._confinement_root).resolve() == episode.world.resolve()
        # 2. The declared surface, exactly: no controller capability is live.
        inventory = set(concrete.tool_inventory)
        expected = {name for name in EPISODE_TOOL_SURFACE if not name.startswith("mcp__")}
        assert inventory == expected, inventory ^ expected

    results = "\n".join(scripted_.tool_texts)
    # 3. The filesystem boundary, per the canary's own probes: no route —
    #    allowed read, refused read, shell, or shell-through-symlink — may
    #    return the controller file's CONTENT to the model.
    assert (
        CONTROLLER_CANARY not in results
    ), "the controller canary's CONTENT must never come back through any tool"
    assert "outside it" in results, "the read of a controller sibling must be refused"
    # The shell is stopped by the KERNEL on hosts that carry the sandbox
    # (darwin: seatbelt's "Operation not permitted"); a host WITHOUT one
    # refuses to run the command at all (``spawn_refusal``) rather than run
    # it unconfined. Either is the boundary holding — the canary-CONTENT
    # assertion above is the one that must pass on every host — and asserting
    # the macOS sentence everywhere made this test fail on Linux while the
    # sandbox was working as designed (QA round 1, Q4).
    assert any(
        marker in results
        for marker in (
            "Operation not permitted",
            "cannot enforce that boundary for shell children",
        )
    ), "the shell must be stopped by the kernel sandbox or refused outright"
    # 4. The excluded capability does not resolve.
    assert "Tool not found: secret" in results, results[-600:]
    # 5. The shell runs under the strict allowlist.
    assert "LO_TEST_API_KEY" not in results, "an ambient credential leaked into the model shell"
    # 6. The child inherited the same boundary and still reached the world.
    assert "child-report denied=True allowed=True" in results, results[-600:]
    # 6b. Packaged knowledge (trusted by provenance) must reach the CHILD
    #     too (QA round 2, Q5 — it disappeared from confined children).
    assert "name: design-qa" in results, "the packaged catalog must resolve in the child"
    # 7. The knowledge boundary, observed from the session: the in-world skill
    #    resolved through the real read tool, and the outer ancestor's twin
    #    produced the unknown-skill answer instead. (The skills LISTING in the
    #    system prompt is query-selected, so it is the loader-level test below
    #    that pins the full root walk; these two probes pin the same boundary
    #    at the session's own knowledge namespace.)
    assert (
        f"name: {INNER_SKILL}" in results
    ), "the in-world skill must resolve through the session's own read tool"
    assert (
        "Unknown skill" in results and OUTER_SKILL in results
    ), "an ancestor skill's twin resolved: discovery was not clipped"
    # 8. Discovery: in-world guidance is present; outer ancestors are not.
    systems = "\n".join(scripted_.system_texts)
    assert INNER_GUIDANCE in systems, "the world's own AGENTS.md must reach the session"
    assert OUTER_GUIDANCE not in systems, "an ancestor AGENTS.md leaked past the boundary"
    assert OUTER_SKILL not in systems, "an ancestor skill leaked past the boundary"


@pytest.mark.asyncio
async def test_world_root_none_keeps_the_ordinary_unclipped_session(
    episode: SimpleNamespace,
    scripted_: ScriptedWorldClient,
) -> None:
    """The default path: no boundary, no confinement, discovery unchanged.

    The one behavioural discriminator that proves the clip is opt-in: with
    ``world_root=None`` the same cwd discovers the ancestor guidance — the
    exact leak the boundary exists to stop. The transport echoes text only:
    nothing in this test may execute a command.
    """
    scripted_.mode = "echo"
    spec = SessionSpec(hosting="test", model="mock")
    async with sdk.open_session(spec, roots=episode.roots) as session:
        concrete = cast(Any, session)
        assert concrete._confinement_root is None
        # The historical surface is untouched: a controller capability the
        # pinned episode surface excludes is live here.
        assert "secret" in set(concrete.tool_inventory)
        await session.prompt("plain session probe")
    systems = "\n".join(scripted_.system_texts)
    assert OUTER_GUIDANCE in systems, "an ordinary session keeps the historical ancestor walk"
    assert INNER_GUIDANCE in systems
    assert not scripted_.tool_texts, "no tool call may have run in the unconfined session"


def test_world_root_validation_refuses_incoherent_layouts(tmp_path: Path) -> None:
    """The SDK preflight: normalized once, refused before any construction."""
    from local_operator.sdk import _resolve_world_root

    world = tmp_path / "world"
    world.mkdir()
    controller = tmp_path / "controller-home"
    config = controller / ".local-operator"
    agent_home = controller / "agent-home"
    for directory in (config, agent_home):
        directory.mkdir(parents=True)

    def _roots(cwd: Path, config_dir: Path = config, home: Path = agent_home) -> SessionRoots:
        return SessionRoots(config_dir=config_dir, agent_home=home, cwd=cwd, allow_volatile=True)

    # Happy path: resolved once, created if missing.
    import pwd
    import shutil
    import uuid

    durable = (
        Path(pwd.getpwuid(os.getuid()).pw_dir)
        / ".cache"
        / "lop-world-root-tests"
        / uuid.uuid4().hex[:12]
    )
    durable.mkdir(parents=True, exist_ok=True)
    try:
        d_world = durable / "world"
        d_world.mkdir()
        d_controller = durable / "controller-home"
        d_config = d_controller / ".local-operator"
        d_agent_home = d_controller / "agent-home"
        for directory in (d_config, d_agent_home):
            directory.mkdir(parents=True)

        def _durable_roots(cwd: Path) -> SessionRoots:
            return SessionRoots(
                config_dir=d_config, agent_home=d_agent_home, cwd=cwd, allow_volatile=True
            )

        assert (
            _resolve_world_root(SessionSpec(world_root=d_world), _durable_roots(d_world / "sub"))
            == d_world
        )
        fresh = durable / "fresh-world"
        assert _resolve_world_root(SessionSpec(world_root=fresh), _durable_roots(fresh)) == fresh
    finally:
        shutil.rmtree(durable, ignore_errors=True)

    # cwd outside the world.
    with pytest.raises(SessionSpecError, match="does not contain the session's cwd"):
        _resolve_world_root(SessionSpec(world_root=world / "other"), _roots(world))
    # store roots inside the world.
    with pytest.raises(SessionSpecError, match="contains the config_dir"):
        _resolve_world_root(SessionSpec(world_root=tmp_path), _roots(tmp_path))
    # the world nested inside controller state.
    with pytest.raises(SessionSpecError, match="sits inside the agent_home"):
        _resolve_world_root(SessionSpec(world_root=agent_home / "w"), _roots(agent_home / "w"))
    # a filesystem root is never a world.
    with pytest.raises(SessionSpecError, match="filesystem root"):
        _resolve_world_root(SessionSpec(world_root="/"), _roots(world))
    # None stays side-effect-free.
    assert _resolve_world_root(SessionSpec(), _roots(world)) is None


def test_world_root_is_refused_where_the_planes_do_not_carry_it(tmp_path: Path) -> None:
    """Attach viewers and spawned runtimes refuse the field rather than drop it."""
    from local_operator.sdk import _refuse_attach_extras, _refuse_spawn_extras

    spec = SessionSpec(world_root=tmp_path / "w")
    with pytest.raises(SessionSpecError):
        _refuse_attach_extras(spec)
    with pytest.raises(SessionSpecError):
        _refuse_spawn_extras(spec)


def test_validate_episode_session_spec_pins_world_and_surface(tmp_path: Path) -> None:
    """The arm's single seam: an episode without boundary or surface is refused."""
    good = _episode_spec(tmp_path / "world")
    assert validate_episode_session_spec(spec=good) is good

    with pytest.raises(SessionArmError, match="world_root"):
        validate_episode_session_spec(spec=SessionSpec(hosting="test", model="mock"))
    with pytest.raises(SessionArmError, match="DECLARE"):
        validate_episode_session_spec(
            spec=SessionSpec(hosting="test", model="mock", world_root=tmp_path / "world")
        )
    with pytest.raises(SessionArmError, match="pinned"):
        validate_episode_session_spec(
            spec=SessionSpec(
                hosting="test",
                model="mock",
                world_root=tmp_path / "world",
                tools=("read", "bash"),
                approvals=ApprovalPolicy.declared(("read", "bash")),
            )
        )
    with pytest.raises(SessionArmError, match="declared"):
        validate_episode_session_spec(
            spec=SessionSpec(
                hosting="test",
                model="mock",
                world_root=tmp_path / "world",
                tools=EPISODE_TOOL_SURFACE,
                approvals=ApprovalPolicy.auto(),
            )
        )


def test_discovery_roots_clip_at_the_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three discovery loaders, clipped and unclipped, side by side.

    Loader-level by design: the session-level proof (the read tool resolving
    an in-world skill and refusing an outer one through the real prompt) lives
    in the boundary test above; this test pins the semantics of the clip
    itself — which roots are walked, which sources are read — including the
    negative for the unclipped walk, so a future edit that silently reopens
    the ancestor walk fails here with a small, direct failure.
    """
    from local_operator.context_files import load_repo_guidance
    from local_operator.mcp.config import load_all_mcp_configs
    from local_operator.skills.api import default_skill_roots
    from local_operator.skills.discovery import discover_skills

    base = tmp_path / "episode"
    world = base / "world"
    home = base / "home"
    for directory in (world, home, base / "config"):
        directory.mkdir(parents=True, exist_ok=True)
    # The config dir must be the test's own: the loaders read the user-scope
    # ``mcp.json`` from ``config_dir()``, which otherwise resolves from the
    # ambient environment.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(base / "config"))
    monkeypatch.setenv("HOME", str(home))
    (world / "AGENTS.md").write_text("# World\n" + INNER_GUIDANCE + "\n")
    (base / "AGENTS.md").write_text("# Outer\n" + OUTER_GUIDANCE + "\n")
    for name, directory in (
        (INNER_SKILL, world / ".local-operator" / "skills" / INNER_SKILL),
        (OUTER_SKILL, base / ".local-operator" / "skills" / OUTER_SKILL),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SKILL.md").write_text(_SKILL_TEMPLATE.format(name=name, description=name))
    (world / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"inner-mcp-sentinel": {"command": "/bin/echo"}}})
    )
    (home / ".cursor").mkdir()
    (home / ".cursor" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"outer-mcp-sentinel": {"command": "/bin/echo"}}})
    )

    clipped_names = {
        skill.name
        for skill in discover_skills(default_skill_roots(world, boundary=world), boundary=world)[0]
    }
    assert INNER_SKILL in clipped_names
    assert OUTER_SKILL not in clipped_names, "the clipped walk reached an ancestor"
    assert "design-qa" in clipped_names, "packaged skills must survive the clip (Q5)"

    # The unclipped walk IS what reaches the ancestor (the clip is the fix,
    # not the layout): assert at the root level so this test never scans the
    # operator's real home.
    unclipped_roots = default_skill_roots(world)
    assert base / ".local-operator" / "skills" in unclipped_roots

    guidance = load_repo_guidance(world, boundary=world)
    assert INNER_GUIDANCE in guidance
    assert OUTER_GUIDANCE not in guidance

    clipped_configs, _ = load_all_mcp_configs(world, boundary=world)
    assert "inner-mcp-sentinel" in clipped_configs
    assert "outer-mcp-sentinel" not in clipped_configs
    unclipped_configs, _ = load_all_mcp_configs(world)
    assert "outer-mcp-sentinel" in unclipped_configs

    # A cwd outside the boundary is refused, never silently unclipped.
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(ValueError):
        default_skill_roots(outside, boundary=world)
    with pytest.raises(ValueError):
        load_repo_guidance(outside, boundary=world)
    with pytest.raises(ValueError):
        load_all_mcp_configs(outside, boundary=world)

    # ---------------------------------------------------------------------
    # Symlinked SOURCES are not world sources (QA round 1, Q2): an in-world
    # skill dir symlinked to an ancestor's skill, and a world project
    # ``.local-operator/mcp.json`` symlinked to an out-of-world file, both
    # resolve outside the boundary and must be dropped by the same
    # canonical-containment pass the other sources go through. The unclipped
    # reads stay permissive, so the clip (not the spelling) is the fix.
    # ---------------------------------------------------------------------
    leak = world / ".local-operator" / "skills" / "leaky-name"
    os.symlink(base / ".local-operator" / "skills" / OUTER_SKILL, leak)
    assert OUTER_SKILL not in {
        skill.name
        for skill in discover_skills(default_skill_roots(world, boundary=world), boundary=world)[0]
    }, "an in-world symlink re-admitted an ancestor skill"

    outer_mcp_file = base / "outer-mcp.json"
    outer_mcp_file.write_text(
        json.dumps({"mcpServers": {"symlink-mcp-sentinel": {"command": "/bin/echo"}}})
    )
    os.symlink(outer_mcp_file, world / ".local-operator" / "mcp.json")
    sym_configs, sym_sources = load_all_mcp_configs(world, boundary=world)
    assert "symlink-mcp-sentinel" not in sym_configs
    assert all(
        Path(source).resolve().is_relative_to(world.resolve()) for source in sym_sources.values()
    ), sym_sources

    # ---------------------------------------------------------------------
    # READ time (QA round 1, Q2 second half): a skill directory REPLACED by
    # an out-of-world symlink after discovery cannot be served through the
    # stale mapping entry — the resolver refuses before anything is read,
    # with the same unknown shape a name miss uses.
    # ---------------------------------------------------------------------
    import shutil as _shutil

    from local_operator.skills.api import make_skill_resolver

    inner_dir = world / ".local-operator" / "skills" / INNER_SKILL
    skills_map = {
        skill.name: skill
        for skill in discover_skills(default_skill_roots(world, boundary=world), boundary=world)[0]
    }
    guard = make_skill_resolver(skills_map, None, boundary=world)
    assert "body" in (guard(f"skill://{INNER_SKILL}") or "")
    assert "Unknown skill" in (guard(f"skill://{OUTER_SKILL}") or "")
    # A swapped BODY FILE (directory intact) must refuse too — the residual
    # the directory-level check cannot see (QA round 2, Q6).
    _shutil.rmtree(inner_dir)
    inner_dir.mkdir()
    (inner_dir / "SKILL.md").write_text(
        _SKILL_TEMPLATE.format(name=INNER_SKILL, description=INNER_GUIDANCE)
    )
    assert "body" in (guard(f"skill://{INNER_SKILL}") or "")
    (inner_dir / "SKILL.md").unlink()
    os.symlink(
        base / ".local-operator" / "skills" / OUTER_SKILL / "SKILL.md",
        inner_dir / "SKILL.md",
    )
    body_swapped = guard(f"skill://{INNER_SKILL}") or ""
    assert "Unknown skill" in body_swapped, "a swapped body file served out-of-world content"
    assert OUTER_GUIDANCE not in body_swapped
    # A swapped REFERENCE FILE refuses through the child resolver's own
    # containment relative to the (in-world) skill directory.
    refs = inner_dir / "references"
    refs.mkdir()
    (refs / "note.md").write_text("INNER-REFERENCE\n")
    os.unlink(inner_dir / "SKILL.md")
    (inner_dir / "SKILL.md").write_text(
        _SKILL_TEMPLATE.format(name=INNER_SKILL, description=INNER_GUIDANCE)
    )
    assert "INNER-REFERENCE" in (guard(f"skill://{INNER_SKILL}/references/note.md") or "")
    (refs / "note.md").unlink()
    os.symlink(
        base / ".local-operator" / "skills" / OUTER_SKILL / "SKILL.md",
        refs / "note.md",
    )
    ref_swapped = guard(f"skill://{INNER_SKILL}/references/note.md") or ""
    assert "escapes the skill directory" in ref_swapped, ref_swapped
    assert OUTER_GUIDANCE not in ref_swapped
    _shutil.rmtree(inner_dir)
    os.symlink(base / ".local-operator" / "skills" / OUTER_SKILL, inner_dir)
    swapped = guard(f"skill://{INNER_SKILL}") or ""
    assert "Unknown skill" in swapped, "a post-discovery dir swap served out-of-world content"
    # And the same resolver WITHOUT a boundary is the unchanged ordinary
    # path: the guard is opt-in, not a global behaviour change.
    ordinary = make_skill_resolver(skills_map, None)
    assert ordinary(f"skill://{INNER_SKILL}") is not None


def test_controller_state_under_kernel_exposed_trees_is_refused() -> None:
    """QA round 1, Q3 + round 2, Q7: the exposed-tree layout never constructs.

    The seatbelt profile must expose ``/private/var/folders`` (dyld caches,
    mktemp machinery) read-only to shell children. Measured: a broad allow
    defeats a specific later DENY in seatbelt in both rule orders, so the
    profile CANNOT carve the stores back out; accepted, the confined shell
    read the dummy controller directly AND through a world symlink. The SDK
    preflight therefore refuses the layout before anything is allocated.

    The exposed tree is DERIVED from the backend (``getconf
    DARWIN_USER_TEMP_DIR``) rather than assumed from ambient ``mkdtemp``
    placement: under ``env -i`` the temp fallback resolves to ``/private/tmp``
    — an intentionally supported unexposed control — and the round-1 version
    of this test built the wrong layout there (QA round 2, Q7). A world under
    the exposed tree with its stores elsewhere, a ``/private/tmp`` layout
    (stores included) and the durable-home session tests above all keep their
    supported behaviour; only the store roots' placement refuses.
    """
    import shutil
    import tempfile

    from local_operator.sdk import _resolve_world_root

    if sys.platform != "darwin":
        # No kernel sandbox on this host: confined shells are refused
        # outright (``spawn_refusal``), so there is no kernel path to guard
        # and the layout is fine.
        pytest.skip("the kernel-exposure refusal is only meaningful on hosts with the sandbox")

    # Derive the exposed tree from the backend itself, NOT from ambient
    # ``tempfile`` placement: with ``env -i`` (the canonical isolated run)
    # Python's temp fallback resolves to ``/private/tmp``, an intentionally
    # SUPPORTED unexposed control, so a bare ``mkdtemp()`` built the wrong
    # layout and the assertion failed for fixture reasons only (QA round 2,
    # Q7). ``getconf DARWIN_USER_TEMP_DIR`` is the machinery tree the seatbelt
    # profile actually exposes; ``/usr/bin/getconf`` by absolute path so the
    # derivation is PATH-independent too.
    import subprocess

    conf = subprocess.run(
        ["/usr/bin/getconf", "DARWIN_USER_TEMP_DIR"],
        capture_output=True,
        text=True,
        check=False,
    )
    exposed_root = Path(conf.stdout.strip()) if conf.returncode == 0 else None
    if exposed_root is None or not exposed_root.is_dir():
        pytest.skip("getconf DARWIN_USER_TEMP_DIR is unavailable on this host")

    base = Path(tempfile.mkdtemp(dir=exposed_root))
    try:
        world = base / "world"
        world.mkdir()
        config = base / "home" / ".local-operator"
        agent_home = base / "home" / "agent-home"
        for directory in (config, agent_home):
            directory.mkdir(parents=True)
        roots = SessionRoots(
            config_dir=config, agent_home=agent_home, cwd=world, allow_volatile=True
        )
        with pytest.raises(SessionSpecError, match="must expose read-only to shell children"):
            _resolve_world_root(SessionSpec(world_root=world), roots)

        # Stores outside the exposed trees, world inside: accepted.
        d_config = Path.home() / ".cache" / "lop-world-root-tests"
        d_config = d_config / os.urandom(4).hex() / ".local-operator"
        d_agent = d_config.parent / "agent-home"
        for directory in (d_config, d_agent):
            directory.mkdir(parents=True)
        roots2 = SessionRoots(
            config_dir=d_config, agent_home=d_agent, cwd=world, allow_volatile=True
        )
        assert _resolve_world_root(SessionSpec(world_root=world), roots2) == world.resolve()
        shutil.rmtree(d_config.parent, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    # The unexposed controls keep their behaviour: ``/private/tmp`` is NOT one
    # of the machinery trees, so a layout there — stores included — is
    # accepted by the preflight (the kernel denies per-world by other means;
    # QA rows 65–69).
    tmp_base = Path(tempfile.mkdtemp(dir="/private/tmp"))
    try:
        t_world = tmp_base / "world"
        t_world.mkdir()
        t_config = tmp_base / "home" / ".local-operator"
        t_agent = tmp_base / "home" / "agent-home"
        for directory in (t_config, t_agent):
            directory.mkdir(parents=True)
        t_roots = SessionRoots(
            config_dir=t_config, agent_home=t_agent, cwd=t_world, allow_volatile=True
        )
        assert _resolve_world_root(SessionSpec(world_root=t_world), t_roots) == t_world.resolve()
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)


class McpSkillScriptedClient(ScriptedWorldClient):
    """The Q1/Q2 probe script: skills before/after a swap, MCP names live.

    Parenting and stages work exactly as the base rig's do (``stage =
    len(rows)``); the only difference is WHAT the parent asks for. The swap
    itself happens in the test process BETWEEN prompts, so it works on every
    host — the scripted shell could not do it on Linux, where confined bash
    is refused by design.
    """

    #: Armed by the test between prompts: prompt two reads after the skill
    #: DIR was swapped; prompt three after only the BODY FILE was (Q6).
    swapped = False
    body_swapped = False

    def _parent_events(self, rows: list[str]) -> list[Any]:
        stage = len(rows)
        if stage == 0:
            return self._tool_call(0, "read", {"path": f"skill://{INNER_SKILL}"})
        if stage == 1:
            return self._tool_call(1, "read", {"path": f"skill://{OUTER_SKILL}"})
        if stage == 2:
            # The packaged catalog: trusted by PROVENANCE, so it must resolve
            # in a bounded session although it lives outside the world
            # (QA round 2, Q5).
            return self._tool_call(2, "read", {"path": "skill://design-qa"})
        if stage == 3:
            return self._tool_call(3, "read", {"path": f"mcp://{MCP_OUTER_CURSOR}"})
        if stage == 4:
            return self._tool_call(4, "read", {"path": f"mcp://{MCP_WORLD_FIXTURE}"})
        if stage == 5:
            # Prompt one ends HERE; the test swaps the in-world skill dir and
            # arms ``swapped`` before prompt two, which is where the read of
            # the swapped name happens — the swap must never be observable as
            # a pre-swap read.
            if not self.swapped:
                return self._text_stop("part one complete", stage)
            return self._tool_call(5, "read", {"path": f"skill://{INNER_SKILL}"})
        if stage == 6:
            # Prompt two ends HERE; the test swaps ONLY the body file and
            # arms ``body_swapped`` before prompt three (QA round 2, Q6).
            if not self.body_swapped:
                return self._text_stop("part two complete", stage)
            return self._tool_call(6, "read", {"path": f"skill://{INNER_SKILL}"})
        return self._text_stop("mcp-skill-probe complete", stage)


@pytest.fixture
def scripted_mcp(
    episode: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> McpSkillScriptedClient:
    """The MCP/skill scripted transport, wired like ``scripted_``."""
    import local_operator.providers.clients as clients_mod

    client = McpSkillScriptedClient(episode)
    original = clients_mod.client_for_spec

    def patched(spec: Any, **kwargs: Any) -> Any:
        if getattr(spec, "provider", None) == "test":
            return client
        return original(spec, **kwargs)

    monkeypatch.setattr(clients_mod, "client_for_spec", patched)
    return client


@pytest.mark.asyncio
async def test_late_mcp_wiring_and_symlinked_sources_stay_inside_the_world(
    episode: SimpleNamespace,
    scripted_mcp: McpSkillScriptedClient,
) -> None:
    """QA round 1, Q1+Q2 over the real SDK session and the LIVE manager.

    Planted before the open: the outer home's ``.cursor`` server, a world
    project file that is a SYMLINK to an out-of-world file, one harmless
    in-world fixture server (``true``; it exits immediately — nothing is
    spoken to), and an in-world symlink to the ancestor skill dir. Then:
    the live manager must know ONLY the in-world server — at startup, at
    connect-by-name, and after a reload (Q1's exact leak shape was the
    reload) — and the skill surface must refuse the ancestor both by
    discovery and after a mid-session dir swap (Q2).
    """
    from local_operator.mcp.manager import McpConnectionError

    cursor = episode.home / ".cursor"
    cursor.mkdir(parents=True, exist_ok=True)
    (cursor / "mcp.json").write_text(
        json.dumps({"mcpServers": {MCP_OUTER_CURSOR: {"type": "stdio", "command": "true"}}})
    )
    outer_file = episode.base / "outer-mcp-sentinel.json"
    outer_file.write_text(
        json.dumps({"mcpServers": {MCP_OUTER_PROJECT: {"type": "stdio", "command": "true"}}})
    )
    os.symlink(outer_file, episode.world / ".local-operator" / "mcp.json")
    (episode.world / ".mcp.json").write_text(
        json.dumps({"mcpServers": {MCP_WORLD_FIXTURE: {"type": "stdio", "command": "true"}}})
    )
    os.symlink(
        episode.base / ".local-operator" / "skills" / OUTER_SKILL,
        episode.world / ".local-operator" / "skills" / "sneaky-link",
    )

    async with sdk.open_session(_episode_spec(episode.world), roots=episode.roots) as session:
        concrete = cast(Any, session)
        await session.prompt("mcp and skill probe, part one")

        manager = concrete.mcp_manager
        assert manager is not None, "the world session must carry a live MCP manager"
        # The in-world server is the only CONFIGURED one at the live manager.
        assert MCP_WORLD_FIXTURE in manager._configs, manager._configs
        for absent in (MCP_OUTER_CURSOR, MCP_OUTER_PROJECT):
            assert absent not in manager._configs, manager._configs
            with pytest.raises(McpConnectionError):
                await manager.connect_configured_server(absent, interactive=False)
        # Every admitted source file lives inside the world.
        for source in manager._sources.values():
            assert Path(source).resolve().is_relative_to(episode.world.resolve()), source
        # The RELOAD path answers with the same clipped set (Q1's leak was
        # exactly this call re-reading the unclipped sources).
        await manager.discover_and_connect()
        assert set(manager._configs) == {MCP_WORLD_FIXTURE}, manager._configs

        # Prompt two: swap the in-world skill dir for an out-of-world symlink
        # (test-side, so the scenario itself is host-independent), then read
        # it through the session's own read tool.
        inner_dir = episode.world / ".local-operator" / "skills" / INNER_SKILL
        import shutil

        shutil.rmtree(inner_dir)
        os.symlink(episode.base / ".local-operator" / "skills" / OUTER_SKILL, inner_dir)
        scripted_mcp.swapped = True
        await session.prompt("mcp and skill probe, part two")

        # Prompt three (QA round 2, Q6): restore a REAL in-world directory,
        # then swap ONLY its SKILL.md for a symlink to the outer file — the
        # case the directory-level guard cannot see.
        os.unlink(inner_dir)
        inner_dir.mkdir()
        (inner_dir / "SKILL.md").write_text(
            _SKILL_TEMPLATE.format(name=INNER_SKILL, description=INNER_GUIDANCE)
        )
        os.unlink(inner_dir / "SKILL.md")
        os.symlink(
            episode.base / ".local-operator" / "skills" / OUTER_SKILL / "SKILL.md",
            inner_dir / "SKILL.md",
        )
        scripted_mcp.body_swapped = True
        await session.prompt("mcp and skill probe, part three")

    texts = scripted_mcp.tool_texts
    assert "name: design-qa" in "\n".join(
        texts
    ), "packaged knowledge (trusted by provenance) must resolve in a bounded session"
    first_inner = next((i for i, t in enumerate(texts) if f"name: {INNER_SKILL}" in t), None)
    first_unknown = next((i for i, t in enumerate(texts) if "Unknown skill" in t), None)
    assert first_inner is not None, "the in-world skill must resolve before any swap"
    assert first_unknown is not None, "the ancestor skill must stay unresolvable"
    assert first_inner < first_unknown, "resolution ordering changed"
    assert "Unknown skill" in texts[-1], "the swapped body file served out-of-world content"
    joined = "\n".join(texts)
    assert OUTER_GUIDANCE not in joined, "an ancestor sentinel leaked through a tool result"
    systems = "\n".join(scripted_mcp.system_texts)
    for name in (MCP_OUTER_CURSOR, MCP_OUTER_PROJECT, "sneaky-link"):
        assert name not in systems, f"the seed leaked {name} into the system prompt"
