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
    kernel denial at all. The tree lives under the REAL home (the same idiom
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
    assert "Operation not permitted" in results, "the shell must be stopped by the kernel sandbox"
    # 4. The excluded capability does not resolve.
    assert "Tool not found: secret" in results, results[-600:]
    # 5. The shell runs under the strict allowlist.
    assert "LO_TEST_API_KEY" not in results, "an ambient credential leaked into the model shell"
    # 6. The child inherited the same boundary and still reached the world.
    assert "child-report denied=True allowed=True" in results, results[-600:]
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
    spec = SessionSpec(world_root=world)
    assert _resolve_world_root(spec, _roots(world / "sub")) == world
    fresh = tmp_path / "fresh-world"
    assert _resolve_world_root(SessionSpec(world_root=fresh), _roots(fresh)) == fresh

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
        skill.name for skill in discover_skills(default_skill_roots(world, boundary=world))[0]
    }
    assert INNER_SKILL in clipped_names
    assert OUTER_SKILL not in clipped_names, "the clipped walk reached an ancestor"
    assert "design-qa" in clipped_names, "packaged skills must survive the clip"

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
