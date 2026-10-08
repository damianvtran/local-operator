"""``local_operator.sdk`` — same machinery, explicit roots, additive surface.

What these tests defend, in the order the task names it:

* **It is the same machinery, not a parallel one.** ``open_session`` builds
  through ``session_factory.create_session``; the tool surface it reaches is
  compared against ``tools.registry.create_tools`` for the same allow-list;
  the event stream is driven by a real turn through the mock provider
  (``hosting="test"``), not by a stub loop. The import guard keeps the engine
  off ``local_operator.sdk``'s own import graph (function-local discipline).
* **Isolation is explicit and default-safe.** The scoping test proves the
  ambient environment is overridden and restored; the tripwire tests prove the
  uid-default roots and an un-redirected cache are refused by default; the
  spawn test proves the child environment is built from ``SessionRoots`` (HOME,
  the two overrides, ``CMUX_*``/``LOP_*`` stripped) rather than inherited.
* **PR 1 is additive.** The pinned-surface test fixes ``__all__`` and the four
  public signatures; nothing here touches exec/TUI/mobile/server code paths —
  the surface is exercised only through the SDK's own entry points.

Fixtures use ``tmp_path`` roots with ``allow_volatile=True``: pytest's tmp_path
lives under ``$TMPDIR``, which the durable rule refuses by design, and the
refusal itself is covered in ``tests/unit/session/test_spec.py``. Every test
that expects construction gives the session a scratch ``HOME`` the way a
launchd episode does, so the cache check passes the honest way.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator import sdk
from local_operator.session.spec import (
    ApprovalPolicy,
    SessionIsolationError,
    SessionRoots,
    SessionSpec,
    SessionSpecError,
)

REPO = Path(__file__).resolve().parents[2]

_PROBE = """
import json, importlib, sys
importlib.import_module(sys.argv[1])
print(json.dumps(sorted(sys.modules)))
"""


def _imported_modules(target: str) -> set[str]:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, target],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, f"importing {target} failed:\n{proc.stderr[-3000:]}"
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SessionRoots:
    """A scratch home in the shape the isolation contract expects.

    ``HOME`` is redirected (the reliable method — a cache built from the real
    home is refused by the tripwire otherwise), the config dir lives under the
    scratch home, and a DECOY ambient root is exported so the scoping test can
    prove the SDK overrides it rather than agreeing with it.

    The agent-shell markers are stripped here because this shell carries them
    (the harness sets them on every agent-run command) and the ordinary caller
    path is what most tests exercise; the guard tests set them back, which is
    where the refusal belongs.
    """
    from local_operator import agent_shell

    home = tmp_path / "home"
    root = home / ".local-operator"
    agent_home = home / "local-operator-home"
    work = home / "work"
    for directory in (root, agent_home, work):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "decoy-config"))
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path / "decoy-home"))
    monkeypatch.delenv(agent_shell.AGENT_SHELL_ENV, raising=False)
    monkeypatch.delenv(agent_shell.MAY_DELEGATE_ENV, raising=False)
    monkeypatch.delenv(agent_shell.ALLOW_NESTED_SESSION_ENV, raising=False)
    return SessionRoots(config_dir=root, agent_home=agent_home, cwd=work, allow_volatile=True)


def _mock_spec(**overrides: Any) -> SessionSpec:
    values: dict[str, Any] = {"hosting": "test", "model": "mock"}
    values.update(overrides)
    return SessionSpec(**values)


async def _never_gate(tool_name: str, description: str) -> bool:  # pragma: no cover
    """An attach-refused callback gate: reaching it at all is the failure."""
    raise AssertionError("a refused attach spec's gate must never be consulted")


# --- same machinery --------------------------------------------------------------


def test_importing_the_sdk_leaves_the_engine_off_the_graph() -> None:
    """The import contract: heavy imports are function-local, re-exports lazy.

    A fresh subprocess (pytest has everything imported in-process). The banned
    set is exactly what a naive module-scope re-export would drag in — the
    composition root, the runtime package, the event types, the provider stack.
    """
    modules = _imported_modules("local_operator.sdk")
    for banned in (
        "local_operator.session_factory",
        "local_operator.session.session",
        "local_operator.session.runtime.launch",
        "local_operator.session.runtime.serving",
        "local_operator.harness.types",
        "local_operator.harness.approval",
        "local_operator.providers",
        "local_operator.mcp",
    ):
        offenders = sorted(m for m in modules if m == banned or m.startswith(banned + "."))
        assert not offenders, f"{banned} is back on local_operator.sdk's import path"
    assert "local_operator.session.spec" in modules
    assert "local_operator.paths" in modules


@pytest.mark.asyncio
async def test_open_session_builds_through_the_factory_and_scopes_the_roots(
    scratch: SessionRoots,
) -> None:
    """Own mode: a real in-process session against exactly the declared roots.

    Asserts on the resolved store path (the apparatus's own proof method — a
    read of config would only say what it should be, not what it resolved to)
    and on the ambient decoy being back afterwards.
    """
    from local_operator import paths

    seen: dict[str, Any] = {}
    async with sdk.open_session(_mock_spec(), roots=scratch) as session:
        # The SDK hands back ``SessionProtocol``; the concrete-only members this
        # test reads (transcript_path, tool_inventory, the gate accessor) are
        # widened here rather than declared on the protocol, which deliberately
        # does not carry them.
        session = cast(Any, session)
        seen["config_dir"] = paths.config_dir().resolve()
        seen["agent_home"] = paths.agent_home_dir().resolve()
        seen["transcript"] = Path(session.transcript_path)
        assert session.owns_runtime is True, "own mode must run the loop in this process"
        assert session.outcome_is_synchronous is True

    assert seen["config_dir"] == scratch.config_path
    assert seen["agent_home"] == scratch.agent_home_path
    # config_dir/sessions/<id>/transcript.jsonl — the session DIRECTORY is
    # created at construction; the transcript file itself is lazy (the first
    # write), which `test_a_real_turn_streams_on_the_same_engine` asserts.
    assert seen["transcript"].parent.parent.parent == scratch.config_path
    assert seen["transcript"].parent.exists()

    # The scoped environment is restored: the decoy is what the environment
    # says again, and the session's root is NOT baked into the process.
    assert paths.config_dir().resolve() == Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"]).resolve()
    assert "decoy-config" in str(paths.config_dir())


def _seed_lopdev(roots: SessionRoots) -> None:
    """A team this root holds, so team resolution has something to resolve."""
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    TeamRegistry(roots.config_path).create_team(
        TeamEditFields(
            name="lopdev",
            manager="manager",
            members=[TeamMember(role="coder")],
            instructions="ship it",
            project="local-operator",
        )
    )


@pytest.mark.asyncio
async def test_open_session_refuses_the_team_profile_pair_with_the_shared_sentence(
    scratch: SessionRoots,
) -> None:
    """Q-MAJOR-3 (#2050's re-review): the pair is refused, not silently split.

    Once ``to_runner_args`` forwards the team, ``resolve_startup`` sees it and
    the #2014 rule fires for the SDK exactly as it does for ``lop exec`` — the
    same shared sentence, re-raised as the spec surface's own error type.
    """
    _seed_lopdev(scratch)
    with pytest.raises(SessionSpecError, match="cannot be combined with --team"):
        async with sdk.open_session(_mock_spec(team="lopdev", profile="reviewer"), roots=scratch):
            pass


@pytest.mark.asyncio
async def test_open_session_attaches_a_named_team(scratch: SessionRoots) -> None:
    """Team-only used to be dropped silently; now it is the session's team.

    The attach rides the same post-open ``session.attach_team`` exec uses, so
    ``active_team`` is the registry row the name resolved to — not a string the
    SDK kept to itself.
    """
    _seed_lopdev(scratch)
    async with sdk.open_session(_mock_spec(team="lopdev"), roots=scratch) as session:
        session = cast(Any, session)
        assert session.active_team is not None
        assert session.active_team.name == "lopdev"


@pytest.mark.asyncio
async def test_open_session_refuses_a_team_this_root_does_not_hold(scratch: SessionRoots) -> None:
    """A missing team is refused by name — the other silent-drop half of Q-MAJOR-3."""
    with pytest.raises(SessionSpecError, match="No team named"):
        async with sdk.open_session(_mock_spec(team="nobody-here"), roots=scratch):
            pass


@pytest.mark.asyncio
async def test_tool_surface_matches_create_tools_for_the_same_allow_list(
    scratch: SessionRoots,
) -> None:
    """The construction-half of the design's §7 parity: same reach, same names.

    ``create_tools(context, enabled=…)`` is the function the session's own
    inventory is built by; comparing the live session against it for the same
    allow-list is what "same tool surface" means at this layer. Names compare
    as a SET: the session builds in its canonical order (which the prompt cache
    depends on), create_tools in the caller's order — the surface is the set,
    and both equal the declaration.
    """
    from local_operator.tools.registry import create_tools

    allow = ("read", "grep", "bash")
    async with sdk.open_session(_mock_spec(tools=allow), roots=scratch) as session:
        session = cast(Any, session)
        expected = {
            tool.name for tool in create_tools(session._build_tool_context(), enabled=list(allow))
        }
        assert set(session.tool_inventory) == expected == set(allow)
        assert session.unresolved_declared_tools() == (), "every declared name must resolve"


@pytest.mark.asyncio
async def test_the_default_surface_is_the_same_set_as_create_tools(scratch: SessionRoots) -> None:
    """An unrestricted SDK session reaches the same surface a session built
    anywhere else with the same context reaches — the full default registry
    plus the capability tools the context gates in."""
    from local_operator.tools.registry import create_tools

    async with sdk.open_session(_mock_spec(), roots=scratch) as session:
        session = cast(Any, session)
        expected = {tool.name for tool in create_tools(session._build_tool_context())}
        assert set(session.tool_inventory) == expected


@pytest.mark.asyncio
async def test_a_real_turn_streams_on_the_same_engine(scratch: SessionRoots) -> None:
    """The event stream, driven by a real turn through the mock provider.

    ``subscribe first, prompt once`` — the exec order. The events asserted are
    the ones the whole apparatus keys on: ``agent_start`` opens a turn and
    ``agent_end`` is its terminal outcome; ``provider_turn_start`` is the
    acceptance boundary supervisors (like the benchmark) rely on.
    """
    async with sdk.open_session(_mock_spec(), roots=scratch) as session:
        session = cast(Any, session)
        stream = sdk.events(session)
        await session.prompt("hello from the sdk test")
        types = []
        while not stream._queue.empty():
            types.append(stream._queue.get_nowait().type)
        await stream.aclose()
    assert "agent_start" in types and "agent_end" in types
    assert "provider_turn_start" in types
    assert types.index("agent_start") < types.index("agent_end")
    assert Path(session.transcript_path).exists(), "a completed turn has written the transcript"


# --- approval policy -------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_policies_map_to_the_pinned_semantics(scratch: SessionRoots) -> None:
    """The mapping table from ``ApprovalPolicy``'s docstring, enforced.

    The gate object is read the way the loop reads it for a turn
    (``Session._tool_approval_gate``), so this pins the wiring a call would
    meet, not just the value object: refuse raises the TYPED refusal (never a
    bare ``False`` rendered as "user denied"); auto approves; a declared
    inventory answers its own members and refuses the rest the honest way; a
    callback is consulted verbatim.
    """
    from local_operator.harness.approval import ApprovalUnavailableError

    async with sdk.open_session(_mock_spec(approvals=ApprovalPolicy.refuse()), roots=scratch) as s:
        s = cast(Any, s)
        with pytest.raises(ApprovalUnavailableError):
            await s._tool_approval_gate()("bash", "write something")

    async with sdk.open_session(_mock_spec(approvals=ApprovalPolicy.auto()), roots=scratch) as s:
        s = cast(Any, s)
        assert await s._tool_approval_gate()("bash", "write something") is True

    policy = ApprovalPolicy.declared(["read", "grep"])
    async with sdk.open_session(_mock_spec(approvals=policy), roots=scratch) as s:
        s = cast(Any, s)
        assert set(s.tool_inventory) == {"read", "grep"}
        gate = s._tool_approval_gate()
        assert await gate("read", "open a file") is True, "the declaration stands as approval"
        with pytest.raises(ApprovalUnavailableError):
            await gate("write", "write something")


@pytest.mark.asyncio
async def test_callback_policy_is_consulted_verbatim(
    scratch: SessionRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    async def callback(tool_name: str, description: str) -> bool:
        asked.append(tool_name)
        return tool_name == "read"

    async with sdk.open_session(
        _mock_spec(approvals=ApprovalPolicy.callback(callback)), roots=scratch
    ) as session:
        session = cast(Any, session)
        gate = session._tool_approval_gate()
        assert await gate("read", "x") is True
        assert await gate("bash", "x") is False
    assert asked == ["read", "bash"]


@pytest.mark.asyncio
async def test_declared_policy_and_spec_tools_must_agree(scratch: SessionRoots) -> None:
    """One bound, stated once: agreement passes, disagreement refuses.

    The policy's own list is the declaration when ``spec.tools`` is unset (the
    design's example spelling); when both name a list and they differ, that is
    two sources of truth for a security control, refused before construction.
    """
    policy = ApprovalPolicy.declared(["read", "grep"])
    async with sdk.open_session(
        _mock_spec(tools=("read", "grep"), approvals=policy), roots=scratch
    ) as session:
        session = cast(Any, session)
        assert set(session.tool_inventory) == {"read", "grep"}

    with pytest.raises(SessionSpecError, match="different inventories"):
        async with sdk.open_session(
            _mock_spec(tools=("read",), approvals=ApprovalPolicy.declared(["read", "grep"])),
            roots=scratch,
        ):
            pass


# --- isolation tripwires ----------------------------------------------------------


@pytest.mark.asyncio
async def test_default_and_cache_tripwires_refuse(
    scratch: SessionRoots, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The uid-default roots, and an un-redirected cache, refuse by default.

    These are the two halves of the incident class the design's §4 exists for:
    a session that "just happens" to target the operator's real store, and one
    that looks isolated while its cache still reads the real home. Both fail
    BEFORE construction — asserted here by the store not existing.
    """
    from local_operator.session import spec as spec_mod

    fake_uid = tmp_path / "fake-uid-home"
    monkeypatch.setattr(spec_mod, "uid_home_dir", lambda: fake_uid)

    with pytest.raises(SessionIsolationError, match="DEFAULT config dir"):
        async with sdk.open_session(
            _mock_spec(),
            roots=SessionRoots(
                config_dir=fake_uid / ".local-operator",
                agent_home=tmp_path / "ah",
                cwd=tmp_path,
                allow_volatile=True,
            ),
        ):
            pass
    assert not fake_uid.exists(), "the refusal must come before anything is written"

    with pytest.raises(SessionIsolationError, match="DEFAULT"):
        async with sdk.open_session(
            _mock_spec(),
            roots=SessionRoots(
                config_dir=tmp_path / "cfg",
                agent_home=fake_uid / "local-operator-home",
                cwd=tmp_path,
                allow_volatile=True,
            ),
        ):
            pass

    # The cache: HOME is this uid's real home, and the cache root lands under
    # it while the session roots point elsewhere.
    monkeypatch.setenv("HOME", str(fake_uid))
    with pytest.raises(SessionIsolationError, match="cache resolves"):
        async with sdk.open_session(
            _mock_spec(),
            roots=SessionRoots(
                config_dir=tmp_path / "cfg2",
                agent_home=tmp_path / "ah2",
                cwd=tmp_path,
                allow_volatile=True,
            ),
        ):
            pass

    # ... and the opt-in makes it representable: allow_ambient is the deliberate
    # spelling of "yes, the operator's own store".
    async with sdk.open_session(
        _mock_spec(),
        roots=SessionRoots(
            config_dir=fake_uid / ".local-operator",
            agent_home=fake_uid / "local-operator-home",
            cwd=tmp_path,
            allow_volatile=True,
            allow_ambient=True,
        ),
    ) as session:
        assert session.session_id


# --- the agent-shell guard (the CLI's policy, applied to SDK creation) ------------


@pytest.mark.asyncio
async def test_the_agent_shell_guard_refuses_creation_and_the_escape_stamps(
    scratch: SessionRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same policy the CLI applies: no silent top-level sessions, ever.

    A command an agent's ``bash`` tool started may not open sessions unless its
    session holds ``task`` (the allowance) or the documented QA escape waived
    the refusal — and a session opened under either is STAMPED, so it stays out
    of the operator's picker/sidebar/phone list. The stamp is written by
    ``session_factory._prepare`` (the one place every session gets its
    directory), which is exactly why the SDK cannot drift from exec on it.
    """
    from local_operator import agent_shell
    from local_operator.resume import ORIGIN_AGENT_SHELL, session_origin

    monkeypatch.setenv(agent_shell.AGENT_SHELL_ENV, "1")

    with pytest.raises(sdk.SessionOpenRefused):
        async with sdk.open_session(_mock_spec(), roots=scratch):
            pass
    with pytest.raises(sdk.SessionOpenRefused):
        await sdk.deliver("any-id", roots=scratch, errand=sdk.PromptErrand(text="hi"))
    with pytest.raises(sdk.SessionOpenRefused):
        await sdk.spawn_session(_mock_spec(), roots=scratch, errand=sdk.PromptErrand(text="hi"))

    # The QA escape waives the refusal — and the session it opens is stamped.
    monkeypatch.setenv(agent_shell.ALLOW_NESTED_SESSION_ENV, "1")
    async with sdk.open_session(_mock_spec(), roots=scratch) as session:
        session = cast(Any, session)
        origin = session_origin(Path(session.transcript_path).parent)
    assert origin == ORIGIN_AGENT_SHELL

    # So does the delegation allowance.
    monkeypatch.delenv(agent_shell.ALLOW_NESTED_SESSION_ENV)
    monkeypatch.setenv(agent_shell.MAY_DELEGATE_ENV, "1")
    async with sdk.open_session(_mock_spec(), roots=scratch) as session:
        session = cast(Any, session)
        assert session_origin(Path(session.transcript_path).parent) == ORIGIN_AGENT_SHELL


# --- spawn / deliver (runtime-hosted) ---------------------------------------------


@pytest.mark.asyncio
async def test_spawn_session_mints_warms_delivers_and_scopes_the_child_env(
    scratch: SessionRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NEW session: viewer-style id, a warm engage carrying the birth sample,
    then the errand — over the one engagement router, with the child's
    environment built from the roots rather than inherited.

    The environment snapshot is taken INSIDE the scoped call, because that is
    the environment ``launch._spawn_runtime`` copies into the child; the decoys
    prove the strip and the restore.
    """
    import local_operator.session.runtime.launch as launch_mod

    monkeypatch.setenv("LOP_SHOULD_BE_STRIPPED", "1")
    monkeypatch.setenv("CMUX_SHOULD_BE_STRIPPED", "1")
    # Deterministic for the notification arm: earlier tests in this process may
    # have suppressed notifications stickily (the mock provider's rule), and
    # this test's claim is about what the SDK does from a clean posture.
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)

    calls: list[tuple[Any, ...]] = []
    snapshots: list[dict[str, str]] = []

    async def fake_engage(session_id, cwd, work, *, config_dir, **kwargs):
        calls.append((session_id, cwd, work, config_dir, kwargs))
        snapshots.append(dict(os.environ))
        return launch_mod.EngageOutcome(session_id=session_id, detail="accepted")

    monkeypatch.setattr(launch_mod, "engage_runtime", fake_engage)

    spec = SessionSpec(
        hosting="openrouter", model="deepseek/deepseek-v4.1-flash", birth_effort="high"
    )
    outcome = await sdk.spawn_session(
        spec, roots=scratch, errand=launch_mod.PromptErrand(text="go")
    )

    assert len(calls) == 2, "a new session warms first (birth sample), then delivers"
    session_id = calls[0][0]
    assert outcome.session_id == session_id == calls[1][0]
    assert len(session_id) == 12 and all(c in "0123456789abcdef" for c in session_id)

    warm = calls[0][2]
    assert isinstance(warm, launch_mod.WarmErrand)
    assert warm.initial_model.provider == "openrouter"
    assert warm.initial_model.model_id == "deepseek/deepseek-v4.1-flash"
    assert warm.initial_model.reasoning_effort == "high"
    assert warm.model_selection_override is True
    assert calls[1][2].text == "go"
    assert calls[0][3] == scratch.config_path

    snapshot = snapshots[0]
    assert snapshot["LOCAL_OPERATOR_CONFIG_DIR"] == str(scratch.config_path)
    assert snapshot["LOCAL_OPERATOR_HOME"] == str(scratch.agent_home_path)
    assert snapshot["HOME"] == str(scratch.agent_home_path)
    assert snapshot["LOCAL_OPERATOR_NO_NOTIFICATIONS"] == "1"
    assert not [k for k in snapshot if k.startswith(("CMUX_", "LOP_"))]

    # Restored after the spawn: the caller's environment is its own again.
    assert os.environ.get("LOP_SHOULD_BE_STRIPPED") == "1"
    assert os.environ.get("CMUX_SHOULD_BE_STRIPPED") == "1"
    assert "decoy-config" in (os.environ.get("LOCAL_OPERATOR_CONFIG_DIR") or "")


@pytest.mark.asyncio
async def test_spawn_notifications_true_lets_the_child_inherit(
    scratch: SessionRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    import local_operator.session.runtime.launch as launch_mod

    snapshots: list[dict[str, str]] = []

    async def fake_engage(session_id, cwd, work, *, config_dir, **kwargs):
        snapshots.append(dict(os.environ))
        return launch_mod.EngageOutcome(session_id=session_id, detail="accepted")

    monkeypatch.setattr(launch_mod, "engage_runtime", fake_engage)
    # The mock provider suppresses notifications for this PROCESS, stickily, the
    # moment a mock session is built — and earlier tests in the same pytest
    # process built one. Clearing the switch first is what makes this test
    # answer the SPEC's decision rather than the fixture history's; with
    # notifications=True the SDK leaves the launcher's posture alone.
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    await sdk.spawn_session(
        SessionSpec(hosting="test", model="mock", notifications=True),
        roots=scratch,
        errand=launch_mod.PromptErrand(text="go"),
    )
    assert "LOCAL_OPERATOR_NO_NOTIFICATIONS" not in snapshots[-1]


@pytest.mark.asyncio
async def test_spawn_resume_skips_the_warm_step(
    scratch: SessionRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    import local_operator.resume as resume_mod
    import local_operator.session.runtime.launch as launch_mod

    monkeypatch.setattr(
        resume_mod, "resolve_resume_id", lambda config_dir, requested: "abc123def456"
    )
    calls: list[tuple[Any, ...]] = []

    async def fake_engage(session_id, cwd, work, *, config_dir, **kwargs):
        calls.append((session_id, cwd, work, config_dir, kwargs))
        return launch_mod.EngageOutcome(session_id=session_id, detail="accepted")

    monkeypatch.setattr(launch_mod, "engage_runtime", fake_engage)
    spec = SessionSpec().with_resume("some-id")
    outcome = await sdk.spawn_session(
        spec, roots=scratch, errand=launch_mod.PromptErrand(text="go")
    )

    assert len(calls) == 1, "a resumed conversation keeps its own saved selection"
    assert calls[0][0] == outcome.session_id == "abc123def456"
    assert isinstance(calls[0][2], launch_mod.PromptErrand)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"agent_name": "reviewer-bot"},
        {"train": True},
        {"workstream": True},
        {"team": "release"},
        {"profile": "reviewer"},
        {"tools": ("read",)},
        {"name": "nightly"},
        {"goal": "finish"},
        {"approvals": ApprovalPolicy.auto()},
        {"yolo": True, "approvals": ApprovalPolicy.auto()},
    ],
)
async def test_spawn_refuses_state_the_spawn_contract_cannot_deliver(
    scratch: SessionRoots, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any]
) -> None:
    """Loud refusal, never silent dropping: the runtime child is composed from
    its environment, and post-open attachment state has no sanctioned channel
    to a NEW child. The message names the composition that does work."""
    import local_operator.session.runtime.launch as launch_mod

    engaged: list[Any] = []

    async def fake_engage(*args, **kwargs):  # pragma: no cover — must not run
        engaged.append(args)
        return launch_mod.EngageOutcome(session_id="x", detail="?")

    monkeypatch.setattr(launch_mod, "engage_runtime", fake_engage)
    with pytest.raises(SessionSpecError, match="post-open state"):
        await sdk.spawn_session(
            _mock_spec(**overrides), roots=scratch, errand=launch_mod.PromptErrand(text="go")
        )
    assert not engaged


@pytest.mark.asyncio
async def test_spawn_refuses_a_half_model_pair_or_an_orphan_birth_effort(
    scratch: SessionRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The birth sample is all-or-nothing (a half pair would crash the spawn or
    be dropped silently — both refuted in ``_model_sample``'s docstring)."""
    import local_operator.session.runtime.launch as launch_mod

    async def fake_engage(*args, **kwargs):  # pragma: no cover — must not run
        raise AssertionError("engage must not be reached")

    monkeypatch.setattr(launch_mod, "engage_runtime", fake_engage)
    with pytest.raises(SessionSpecError, match="state both, or neither"):
        await sdk.spawn_session(
            _mock_spec(hosting="test", model=None),
            roots=scratch,
            errand=launch_mod.PromptErrand(text="go"),
        )
    with pytest.raises(SessionSpecError, match="birth_effort"):
        await sdk.spawn_session(
            _mock_spec(hosting=None, model=None, birth_effort="high"),
            roots=scratch,
            errand=launch_mod.PromptErrand(text="go"),
        )


@pytest.mark.asyncio
async def test_deliver_resolves_the_id_against_the_scoped_root(
    scratch: SessionRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deliver`` = one engage against an existing id, resolved against THIS
    root (so a scratch store can never resolve the operator's sessions)."""
    import local_operator.resume as resume_mod
    import local_operator.session.runtime.launch as launch_mod

    resolved: list[tuple[Path, str]] = []

    def fake_resolve(config_dir: Path, requested: str) -> str:
        resolved.append((Path(config_dir), requested))
        return "abc123def456"

    monkeypatch.setattr(resume_mod, "resolve_resume_id", fake_resolve)
    calls: list[tuple[Any, ...]] = []

    async def fake_engage(session_id, cwd, work, *, config_dir, **kwargs):
        calls.append((session_id, cwd, work, config_dir, kwargs))
        return launch_mod.EngageOutcome(session_id=session_id, detail="accepted")

    monkeypatch.setattr(launch_mod, "engage_runtime", fake_engage)
    errand = launch_mod.SteerErrand(text="course correct")
    outcome = await sdk.deliver("@latest", roots=scratch, errand=errand)

    assert resolved == [(scratch.config_path, "@latest")]
    assert len(calls) == 1
    assert calls[0][0] == outcome.session_id == "abc123def456"
    assert calls[0][1] == str(scratch.cwd_path)
    assert calls[0][2] is errand
    assert calls[0][3] == scratch.config_path


# --- single-root invariant --------------------------------------------------------


@pytest.mark.asyncio
async def test_one_root_per_process_unless_opted_out(scratch: SessionRoots, tmp_path: Path) -> None:
    """Two live roots in one process is the failure the contract refuses.

    Refused for a second ``open_session`` and for a spawn/deliver targeting the
    other root while one is live; ``allow_multi_root=True`` is the deliberate
    escape; once the first root's session is closed, the process may move on.
    """
    import local_operator.session.runtime.launch as launch_mod

    other = SessionRoots(
        config_dir=tmp_path / "other" / ".local-operator",
        agent_home=tmp_path / "other",
        cwd=tmp_path / "other",
        allow_volatile=True,
    )
    for directory in (other.config_path, other.agent_home_path, other.cwd_path):
        directory.mkdir(parents=True, exist_ok=True)

    async with sdk.open_session(_mock_spec(), roots=scratch):
        with pytest.raises(SessionIsolationError, match="one SessionRoots per process"):
            async with sdk.open_session(_mock_spec(), roots=other):
                pass
        with pytest.raises(SessionIsolationError):
            await sdk.deliver("x", roots=other, errand=launch_mod.PromptErrand(text="hi"))
        with pytest.raises(SessionIsolationError):
            await sdk.spawn_session(
                SessionSpec(resume="abc123def456"),
                roots=other,
                errand=launch_mod.SteerErrand(text="hi"),
            )
        async with sdk.open_session(_mock_spec(), roots=other, allow_multi_root=True):
            pass

    async with sdk.open_session(_mock_spec(), roots=other):
        pass


@pytest.mark.asyncio
async def test_a_delivery_in_flight_holds_its_root_against_another(
    scratch: SessionRoots, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2, cross-root: a delivery is a live root while it runs.

    The reproduced interleave: ``gather(deliver(A), open_session(B))`` used to
    run BOTH — B was not refused (a delivery never registered its root) and
    A's engage then read B's environment mid-flight. Registration is one
    synchronous step before the first await, so a second, different-root
    operation must now be refused while the delivery is in flight — and the
    delivery's engage must always read its OWN root.
    """
    import local_operator.resume as resume_mod
    import local_operator.session.runtime.launch as launch_mod

    other = SessionRoots(
        config_dir=tmp_path / "other" / ".local-operator",
        agent_home=tmp_path / "other",
        cwd=tmp_path / "other",
        allow_volatile=True,
    )
    for directory in (other.config_path, other.agent_home_path, other.cwd_path):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        resume_mod, "resolve_resume_id", lambda config_dir, requested: "held12345678"
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    seen: list[str | None] = []

    async def held_engage(session_id, cwd, work, *, config_dir, **kwargs):
        seen.append(os.environ.get("LOCAL_OPERATOR_CONFIG_DIR"))
        entered.set()
        await release.wait()
        return launch_mod.EngageOutcome(session_id=session_id, detail="accepted")

    monkeypatch.setattr(launch_mod, "engage_runtime", held_engage)
    delivery = asyncio.create_task(
        sdk.deliver("held12345678", roots=scratch, errand=launch_mod.SteerErrand(text="hold"))
    )
    await entered.wait()
    assert seen == [str(scratch.config_path)], "the engage must read its own root"

    with pytest.raises(SessionIsolationError, match="one SessionRoots per process"):
        async with sdk.open_session(_mock_spec(), roots=other):
            pass

    release.set()
    outcome = await delivery
    assert outcome.session_id == "held12345678"


@pytest.mark.asyncio
async def test_overlapping_same_root_scopes_restore_the_environment_exactly(
    scratch: SessionRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2, same-root half: interleaved scopes compose; nothing leaks.

    Two deliveries on the SAME root may overlap (one root, two scopes) and may
    exit out of order: A holds its scope, B enters, then A exits while B is
    still in flight. Under a save/restore pair the earlier exit restores
    values captured before the sibling entered — the sibling then reads
    pre-scope values mid-flight and the later exit leaks scope values past
    both. Asserted: mid-flight the live scope's values still hold; after both
    close, EVERY tracked key is exactly its pre-scope value.
    """
    import local_operator.resume as resume_mod
    import local_operator.session.runtime.launch as launch_mod

    monkeypatch.setenv("CMUX_WORKSPACE_ID", "ws-original")
    monkeypatch.setenv("LOP_MOBILE_CHILD_SESSION", "child-original")
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)
    tracked = (
        "HOME",
        "LOCAL_OPERATOR_CONFIG_DIR",
        "LOCAL_OPERATOR_HOME",
        "LOCAL_OPERATOR_NO_NOTIFICATIONS",
        "CMUX_WORKSPACE_ID",
        "LOP_MOBILE_CHILD_SESSION",
    )
    before = {key: os.environ.get(key) for key in tracked}

    monkeypatch.setattr(
        resume_mod, "resolve_resume_id", lambda config_dir, requested: "inter1234567"
    )
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    calls = 0

    async def held_engage(session_id, cwd, work, *, config_dir, **kwargs):
        nonlocal calls
        index = calls
        calls += 1
        entered[index].set()
        await release[index].wait()
        return launch_mod.EngageOutcome(session_id=session_id, detail="accepted")

    monkeypatch.setattr(launch_mod, "engage_runtime", held_engage)
    errand = launch_mod.SteerErrand(text="hold")

    task_a = asyncio.create_task(sdk.deliver("a", roots=scratch, errand=errand))
    await entered[0].wait()
    task_b = asyncio.create_task(sdk.deliver("b", roots=scratch, errand=errand))
    await entered[1].wait()

    # Non-LIFO, the order a save/restore pair gets wrong: the FIRST scope
    # exits while the second is still in flight.
    release[0].set()
    await task_a
    mid = {key: os.environ.get(key) for key in tracked}
    assert mid["LOCAL_OPERATOR_CONFIG_DIR"] == str(scratch.config_path)
    assert mid["LOCAL_OPERATOR_HOME"] == str(scratch.agent_home_path)
    assert mid["HOME"] == str(scratch.agent_home_path)
    assert mid["CMUX_WORKSPACE_ID"] is None, "a live scope's strip must survive its sibling"
    assert mid["LOP_MOBILE_CHILD_SESSION"] is None
    assert mid["LOCAL_OPERATOR_NO_NOTIFICATIONS"] == "1"

    release[1].set()
    await task_b
    after = {key: os.environ.get(key) for key in tracked}
    assert after == before, "both scopes closed: every key returns to its pre-scope value"


# --- attach mode ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attach_returns_the_viewer_and_cold_ids_get_a_remedy(
    scratch: SessionRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator import session_factory

    calls: list[dict[str, Any]] = []

    class FakeViewer:
        owns_runtime = False

        def __init__(self) -> None:
            self.disposed = 0

        async def dispose(self) -> None:
            self.disposed += 1

    viewer = FakeViewer()

    async def fake_create(session_args, *managers, **kwargs):
        calls.append(dict(kwargs))
        return viewer

    monkeypatch.setattr(session_factory, "create_session", fake_create)
    spec = SessionSpec().with_resume("abc123def456")
    async with sdk.open_session(spec, roots=scratch, mode="attach") as attached:
        assert attached is viewer
    assert viewer.disposed == 1, "the viewer's side is released on exit"
    assert calls[0]["has_ui"] is True

    class ColdOwner:
        owns_runtime = True

        def __init__(self) -> None:
            self.disposed = 0

        async def dispose(self) -> None:
            self.disposed += 1

    cold = ColdOwner()

    async def fake_cold(session_args, *managers, **kwargs):
        return cold

    monkeypatch.setattr(session_factory, "create_session", fake_cold)
    with pytest.raises(SessionSpecError, match="no live runtime"):
        async with sdk.open_session(spec, roots=scratch, mode="attach"):
            pass
    assert cold.disposed == 1, "a cold build must not be left owning the id"

    with pytest.raises(SessionSpecError, match="needs spec.resume"):
        async with sdk.open_session(_mock_spec(), roots=scratch, mode="attach"):
            pass

    with pytest.raises(SessionSpecError, match="does not attach teams"):
        async with sdk.open_session(
            SessionSpec(team="release").with_resume("abc"), roots=scratch, mode="attach"
        ):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "approvals",
    [
        ApprovalPolicy.refuse(),
        ApprovalPolicy.auto(),
        ApprovalPolicy.declared(["read"]),
        ApprovalPolicy.callback(_never_gate),
    ],
    ids=["refuse", "auto", "declared", "callback"],
)
async def test_attach_refuses_every_non_default_approval_policy(
    scratch: SessionRoots,
    monkeypatch: pytest.MonkeyPatch,
    approvals: ApprovalPolicy,
) -> None:
    """A policy attach cannot install must be refused, never silently inert.

    ``mode="attach"`` returns before ``_install_approval_policy``, so the only
    policy it may accept is the default ``refuse()`` (the preset of a spec
    that never mentioned approvals). Every other enum value is refused BEFORE
    any construction — one parametrized case per preset, because a check that
    only works for one of them is exactly the bug this test pins — and the
    private preflight is exercised directly, which is where the fix lives.
    """
    from local_operator import session_factory

    created: list[dict[str, Any]] = []

    class FakeViewer:
        owns_runtime = False

        async def dispose(self) -> None:
            pass

    async def fake_create(session_args, *managers, **kwargs):
        created.append(dict(kwargs))
        return FakeViewer()

    monkeypatch.setattr(session_factory, "create_session", fake_create)
    spec = SessionSpec(resume="abc123def456", approvals=approvals)

    if approvals == ApprovalPolicy.refuse():
        async with sdk.open_session(spec, roots=scratch, mode="attach") as attached:
            assert isinstance(attached, FakeViewer)
        assert created, "the default policy must leave attach usable"
        return

    with pytest.raises(SessionSpecError, match="Offending fields: approvals"):
        sdk._refuse_attach_extras(spec)
    with pytest.raises(SessionSpecError, match="approval policies"):
        async with sdk.open_session(spec, roots=scratch, mode="attach"):
            pass
    assert not created, "the refusal must land before any construction"


@pytest.mark.asyncio
async def test_attach_in_a_root_without_hosting_names_both_blockers(
    scratch: SessionRoots,
) -> None:
    """Q-2 (QA round 1): the attach remedy must survive a hosting-less root.

    ``create_session``'s cold-attach fall-through hits the hosting preflight
    BEFORE the viewer check can produce "no live runtime is serving …", so in
    a root with nothing to start a runtime with, the refusal must still say
    BOTH facts — the id has no live runtime, and this root cannot start one —
    not just the lower-level "Hosting platform is not configured."
    """
    with pytest.raises(SessionSpecError, match="no live runtime is serving") as raised:
        async with sdk.open_session(
            SessionSpec().with_resume("abc123def456"), roots=scratch, mode="attach"
        ):
            pass
    message = str(raised.value)
    assert "cannot start" in message
    assert "Hosting platform is not configured" in message, "the cause stays in the message"


# --- events adapter ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_stream_delivers_unsubscribes_and_survives_handler_errors() -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.handlers: list[Any] = []
            self.unsubscribed = 0

        def subscribe(self, handler: Any) -> Any:
            self.handlers.append(handler)

            def unsubscribe() -> None:
                self.unsubscribed += 1

            return unsubscribe

    session = FakeSession()
    stream = sdk.events(session)
    handler = session.handlers[0]
    handler("one")
    handler("two")
    assert await stream.__anext__() == "one"
    assert await stream.__anext__() == "two"

    await stream.aclose()
    assert session.unsubscribed == 1
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()
    await stream.aclose()  # idempotent
    assert session.unsubscribed == 1, "close must not unsubscribe twice"

    # The async-with spelling.
    session2 = FakeSession()
    async with sdk.events(session2) as stream2:
        session2.handlers[0]("x")
        assert await stream2.__anext__() == "x"
    assert session2.unsubscribed == 1


# --- pinned surface ---------------------------------------------------------------


def test_public_surface_is_pinned() -> None:
    """Additive-stable means the names and shapes are pinned by a test.

    ``__all__`` exactly; every name resolves (the PEP 562 table stays
    complete); the re-export identity holds; and the four entry points keep
    their parameter names and kinds, which callers write against.
    """
    assert sdk.__all__ == [
        "AgentEvent",
        "ApprovalPolicy",
        "EngageOutcome",
        "Errand",
        "EventHandler",
        "MarkdownSchema",
        "PeerMessageErrand",
        "PromptErrand",
        "SessionEventStream",
        "SessionIsolationError",
        "SessionOpenRefused",
        "SessionRoots",
        "SessionSpec",
        "SessionSpecError",
        "SteerErrand",
        "ViewerSessionProtocol",
        "WakeErrand",
        "WarmErrand",
        "decode_output",
        "deliver",
        "events",
        "open_session",
        "spawn_session",
        "SessionProtocol",
    ]
    for name in sdk.__all__:
        assert getattr(sdk, name) is not None, name
    with pytest.raises(AttributeError):
        sdk.no_such_export  # noqa: B018

    from local_operator.session.runtime.launch import PromptErrand as Direct

    assert sdk.PromptErrand is Direct

    def params(func: Any) -> list[tuple[str, str]]:
        return [(p.name, p.kind.name) for p in inspect.signature(func).parameters.values()]

    P = "POSITIONAL_OR_KEYWORD"
    K = "KEYWORD_ONLY"
    assert params(sdk.open_session) == [
        ("spec", P),
        ("roots", K),
        ("mode", K),
        ("allow_multi_root", K),
    ]
    assert params(sdk.spawn_session) == [
        ("spec", P),
        ("roots", K),
        ("errand", K),
        ("deadline_s", K),
        ("allow_multi_root", K),
    ]
    assert params(sdk.deliver) == [
        ("session_id", P),
        ("roots", K),
        ("errand", K),
        ("deadline_s", K),
        ("allow_multi_root", K),
    ]
    assert params(sdk.events) == [("session", P)]


# --- output contract enforcement ---------------------------------------------------


@pytest.mark.asyncio
async def test_open_session_applies_the_output_contract_through_the_session_method(
    scratch: SessionRoots,
) -> None:
    """A spec's ``output_*`` fields reach the SAME post-open method exec
    installs them with, before the first turn — and the contract is announced
    in the session's system blocks (otherwise every first attempt is a coin
    flip)."""
    async with sdk.open_session(
        _mock_spec(
            output_format="json",
            output_schema={"type": "object", "required": ["name"]},
            output_retries=1,
        ),
        roots=scratch,
    ) as session:
        concrete = cast(Any, session)
        contract = concrete._output_contract
        assert contract is not None
        assert contract.format == "json"
        assert contract.max_attempts == 2
        blocks = await concrete._prepare_system_blocks(commit_state=False)
        assert any(block.startswith("Output contract:") for block in blocks)
        assert any('"required"' in block for block in blocks)


@pytest.mark.asyncio
async def test_open_session_refuses_a_contract_it_cannot_build(scratch: SessionRoots) -> None:
    """The common ``OutputContract`` validation, surfaced as ``SessionSpecError``
    (the spec surface's own error class) rather than a bare ValueError."""
    from local_operator.output_contract import MarkdownSchema

    with pytest.raises(SessionSpecError, match="MarkdownSchema applies to format"):
        async with sdk.open_session(
            _mock_spec(output_format="json", output_schema=MarkdownSchema()), roots=scratch
        ):
            pass


@pytest.mark.asyncio
async def test_spawn_refuses_output_enforcement_with_a_named_remedy(
    scratch: SessionRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    import local_operator.session.runtime.launch as launch_mod

    async def fake_engage(*args, **kwargs):  # pragma: no cover — must not run
        raise AssertionError("engage must not be reached")

    monkeypatch.setattr(launch_mod, "engage_runtime", fake_engage)
    with pytest.raises(SessionSpecError, match="cannot carry output enforcement"):
        await sdk.spawn_session(
            _mock_spec(output_format="json"),
            roots=scratch,
            errand=launch_mod.PromptErrand(text="go"),
        )


def test_attach_refuses_output_enforcement() -> None:
    """An attach spec that carried enforcement would be inert by construction:
    the viewer never installs owner-side state. Refused, never silently set."""
    with pytest.raises(SessionSpecError, match="the owning runtime decides"):
        sdk._refuse_attach_extras(SessionSpec(resume="abc123def456", output_format="json"))


def test_the_output_contract_helpers_are_lazy_reexports() -> None:
    """``decode_output``/``MarkdownSchema`` resolve through the PEP 562 table
    to the contract module's own objects (identity, not a copy)."""
    from local_operator import output_contract

    assert sdk.decode_output is output_contract.decode_output
    assert sdk.MarkdownSchema is output_contract.MarkdownSchema
