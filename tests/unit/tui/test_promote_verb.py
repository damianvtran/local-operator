"""``/agent promote`` and ``/team promote``: the grammar and the rehearsal (S4).

Two layers, because they answer different questions:

* :func:`parse_promote` is pure — a mistyped flag must be a usage refusal, never a
  copy under a name the user did not mean;
* the shipped ``OperatorApp`` is the surface — a bare call must write NOTHING and
  name exactly what a confirmed call would copy (and, under ``replace``, exactly
  what it would overwrite), and only ``yes`` may commit.

The account store is the assertion target throughout: a rehearsal that quietly
wrote would still print the right sentence, so "nothing changed" is checked on
disk, not in the notice.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.profile_scope import scope_session_registries
from local_operator.promote_slash import PromoteRequest, parse_promote
from local_operator.teams import TeamEditFields, TeamRegistry, parse_members
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_projects_view import _boot, _notices

pytestmark = pytest.mark.asyncio


def _agent_fields(**overrides: Any) -> AgentEditFields:
    """``AgentEditFields`` with every field spelled out (it is strict-validated)."""
    base: dict[str, Any] = dict(
        name=None,
        label=None,
        description=None,
        tags=None,
        categories=None,
        security_prompt=None,
        hosting=None,
        model=None,
        last_message=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        current_working_directory=None,
    )
    base.update(overrides)
    return AgentEditFields(**base)


# --- the grammar ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("rest", "expected"),
    [
        ("one", PromoteRequest("one")),
        ("one yes", PromoteRequest("one", confirmed=True)),
        ("one replace", PromoteRequest("one", replace=True)),
        ("one replace yes", PromoteRequest("one", replace=True, confirmed=True)),
        ("one as two", PromoteRequest("one", as_name="two")),
        ("one as two yes", PromoteRequest("one", as_name="two", confirmed=True)),
        ("=promote", PromoteRequest("promote")),
        ("one AS two", PromoteRequest("one", as_name="two")),
    ],
)
def test_the_grammar_parses_every_documented_spelling(rest, expected) -> None:
    assert parse_promote(rest) == expected


@pytest.mark.parametrize(
    "rest",
    ["", "   ", "one as", "one wat", "one replace as", "=", "one yes extra"],
)
def test_a_mistyped_command_is_a_usage_refusal(rest) -> None:
    assert parse_promote(rest) is None


# --- the shipped app -----------------------------------------------------------


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, OperatorApp, Any, Any]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    global_agents = AgentRegistry(tmp_path)
    global_teams = TeamRegistry(tmp_path)
    agents, teams = scope_session_registries(
        global_agents, global_teams, tmp_path / "sessions" / "one"
    )
    session = FakeSession()
    session.agent_registry = agents
    session.team_registry = teams
    return session, OperatorApp(lambda: _factory(session)), global_agents, global_teams


def _session_role(agents: Any, name: str, prompt: str = "Do it.\n") -> str:
    row = agents.create_agent(_agent_fields(name=name, tags=["role"]), scope="session")
    agents.set_agent_system_prompt(row.id, prompt)
    return row.id


async def _run(pilot: Any, app: OperatorApp, command: str) -> None:
    app._run_slash_command(command)
    for _ in range(4):
        await pilot.pause()


async def test_a_bare_promote_rehearses_and_writes_nothing(tmp_path, monkeypatch) -> None:
    session, app, global_agents, _global_teams = setup(tmp_path, monkeypatch)
    _session_role(session.agent_registry, "release-check")
    async with app.run_test(size=(80, 30)) as pilot:
        await _boot(pilot, app)
        await _run(pilot, app, "/agent promote release-check")

        assert global_agents.get_agent_by_name("release-check") is None
        assert "Add `yes` to copy release-check into your account" in _notices(app)[-1]
        assert "nothing has changed yet" in _notices(app)[-1]


async def test_yes_commits_and_the_receipt_names_the_copy(tmp_path, monkeypatch) -> None:
    session, app, global_agents, _global_teams = setup(tmp_path, monkeypatch)
    _session_role(session.agent_registry, "release-check")
    async with app.run_test(size=(80, 30)) as pilot:
        await _boot(pilot, app)
        await _run(pilot, app, "/agent promote release-check yes")

        account = global_agents.get_agent_by_name("release-check")
        assert account is not None
        assert global_agents.get_agent_system_prompt(account.id) == "Do it.\n"
        assert "Copied release-check to your account." in _notices(app)[-1]
        assert "This conversation keeps its own copy." in _notices(app)[-1]
        # A copy-only receipt must not claim a replacement (design round 1, D2).
        assert "Replaced" not in _notices(app)[-1]
        # COPY, not move: this conversation still launches its own row.
        assert session.agent_registry.get_agent_by_name("release-check") is not None


async def test_a_team_rehearsal_names_the_members_it_would_copy(tmp_path, monkeypatch) -> None:
    session, app, _global_agents, global_teams = setup(tmp_path, monkeypatch)
    _session_role(session.agent_registry, "helper")
    session.team_registry.create_team(
        TeamEditFields(
            name="pod",
            manager="manager",
            members=parse_members(["helper"]),
            instructions="Work together.",
        ),
        scope="session",
    )
    async with app.run_test(size=(80, 30)) as pilot:
        await _boot(pilot, app)
        await _run(pilot, app, "/team promote pod")

        assert global_teams.get_team_by_name("pod") is None
        assert "helper, pod" in _notices(app)[-1]


async def test_replace_rehearses_the_overwrite_separately(tmp_path, monkeypatch) -> None:
    session, app, global_agents, _global_teams = setup(tmp_path, monkeypatch)
    _session_role(session.agent_registry, "release-check")
    global_agents.create_agent(_agent_fields(name="release-check", tags=["role"]))
    async with app.run_test(size=(80, 30)) as pilot:
        await _boot(pilot, app)
        # The rehearsal is a separate step, and it names the OVERWRITE rather
        # than a copy: replace is the one destructive spelling of this verb.
        await _run(pilot, app, "/agent promote release-check replace")
        rehearsal = _notices(app)[-1]
        assert "Replace existing account definitions: release-check" in rehearsal
        assert "with this conversation's copy" in rehearsal
        # D1: the destructive rehearsal is a WARNING notice, not the quiet tier
        # an ordinary copy uses — the frame's own ink has to say what the
        # sentence says.
        assert _last_notice_kind(app) == "warning"

        await _run(pilot, app, "/agent promote release-check replace yes")
        receipt = _notices(app)[-1]
        assert "Replaced in your account: release-check." in receipt
        assert "This conversation keeps its own copy." in receipt
        # The replacement is named in its OWN sentence, never folded into the
        # copy list (design round 1, D2 / UX round 1, U1).
        assert "Copied release-check" not in receipt
        assert global_agents.get_agent_by_name("release-check") is not None


async def test_a_refusal_is_reported_and_nothing_is_written(tmp_path, monkeypatch) -> None:
    _session, app, global_agents, _global_teams = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(80, 30)) as pilot:
        await _boot(pilot, app)
        await _run(pilot, app, "/agent promote ghost")

        assert global_agents.get_agent_by_name("ghost") is None
        assert "could not promote" in _notices(app)[-1]
        assert "no role or specialist named 'ghost'" in _notices(app)[-1]


async def test_a_mistyped_verb_shows_the_usage(tmp_path, monkeypatch) -> None:
    _session, app, _global_agents, _global_teams = setup(tmp_path, monkeypatch)
    async with app.run_test(size=(80, 30)) as pilot:
        await _boot(pilot, app)
        await _run(pilot, app, "/agent promote")

        assert "Use /agent promote <name> [replace] [yes] [as <new name>]." in _notices(app)[-1]


async def test_a_team_named_promote_is_still_reachable_through_the_escape(
    tmp_path, monkeypatch
) -> None:
    """The reserved verb must not make a team unaddressable (the `chart` rule)."""

    session, app, _global_agents, _global_teams = setup(tmp_path, monkeypatch)
    session.team_registry.create_team(
        TeamEditFields(name="promote", manager="manager", members=[], instructions="x"),
        scope="session",
    )
    async with app.run_test(size=(80, 30)) as pilot:
        await _boot(pilot, app)
        await _run(pilot, app, "/team =promote go")

        assert session.prompts[-1] == "go"
        assert session.active_team_name == "promote"


def _last_notice_kind(app: OperatorApp) -> str:
    """The tier of the newest notice — the ink, not just the sentence (D1/D3)."""
    from local_operator.tui.widgets.transcript import NoticeBlock as _Block
    from local_operator.tui.widgets.transcript import TranscriptView as _View

    blocks = [b for b in app.query_one(_View).blocks() if isinstance(b, _Block)]
    return blocks[-1].notice_kind


async def test_a_mixed_plan_leads_with_the_overwrite(tmp_path, monkeypatch) -> None:
    """Overwrites come FIRST and raise the notice to warning (design round 1, D1).

    The team's member collides in the account while the team itself is new, so
    one plan both replaces and copies — the shape the designer measured with the
    safe sentence on top.
    """

    session, app, global_agents, _global_teams = setup(tmp_path, monkeypatch)
    _session_role(session.agent_registry, "helper")
    session.team_registry.create_team(
        TeamEditFields(
            name="pod",
            manager="manager",
            members=parse_members(["helper"]),
            instructions="Work together.",
        ),
        scope="session",
    )
    global_agents.create_agent(_agent_fields(name="helper", tags=["role"]))
    async with app.run_test(size=(80, 30)) as pilot:
        await _boot(pilot, app)
        await _run(pilot, app, "/team promote pod replace")

        notice = _notices(app)[-1]
        assert notice.index("Replace existing account definitions: helper") < notice.index(
            "also copies"
        )
        assert _last_notice_kind(app) == "warning"


async def test_the_receipt_names_created_and_replaced_separately(tmp_path, monkeypatch) -> None:
    """A mixed completion reads `Copied …` AND `Replaced …`, in that order."""

    session, app, global_agents, _global_teams = setup(tmp_path, monkeypatch)
    _session_role(session.agent_registry, "helper")
    session.team_registry.create_team(
        TeamEditFields(
            name="pod",
            manager="manager",
            members=parse_members(["helper"]),
            instructions="Work together.",
        ),
        scope="session",
    )
    global_agents.create_agent(_agent_fields(name="helper", tags=["role"]))
    async with app.run_test(size=(80, 30)) as pilot:
        await _boot(pilot, app)
        await _run(pilot, app, "/team promote pod replace yes")

        receipt = _notices(app)[-1]
        assert "Copied pod to your account." in receipt
        assert "Replaced in your account: helper." in receipt
        assert receipt.index("Copied pod") < receipt.index("Replaced in your account: helper")
        assert _last_notice_kind(app) == "success"


# --- the OWNER dispatchers (QA round 1, Q1) ------------------------------------
#
# `/agent` and `/team` are authoritative-session commands, so in a live session
# the line reaches `ServingSessionHandle._agent_slash` / `_team_slash` and the
# app-hosted handler never runs. QA measured the regression on that seam: the
# owner read `promote` as a NAME and answered "no agent named 'promote'" to the
# verb the TUI offered. A FakeSession-based test cannot catch that (it never
# routes), so these call the real dispatcher methods against real composites on
# disk. The handle is a stub because the promote branch touches no handle state.


def _dispatcher_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    global_agents = AgentRegistry(tmp_path)
    global_teams = TeamRegistry(tmp_path)
    agents, teams = scope_session_registries(
        global_agents, global_teams, tmp_path / "sessions" / "one"
    )
    session = SimpleNamespace(agent_registry=agents, team_registry=teams)
    return session, global_agents, global_teams


def _owner_slash(kind: str, session: Any, args: str) -> Any:
    """Run one line through the RUNTIME OWNER's dispatcher, not the app's."""
    import asyncio
    from typing import cast

    from local_operator.session.frontend_state import SlashResult
    from local_operator.session.runtime.serving import ServingSessionHandle

    # A stub handle: the promote branches touch no handle state, and casting
    # keeps pyright honest about which object the REAL dispatcher is called on.
    handle = cast("ServingSessionHandle", SimpleNamespace())
    if kind == "agent":
        return asyncio.run(ServingSessionHandle._agent_slash(handle, session, args, SlashResult))
    return ServingSessionHandle._team_slash(handle, session, args, SlashResult)


def test_the_owner_dispatcher_reserves_the_agent_promote_verb(tmp_path, monkeypatch) -> None:
    session, global_agents, _teams = _dispatcher_session(tmp_path, monkeypatch)
    _session_role(session.agent_registry, "release-check")

    result = _owner_slash("agent", session, "promote release-check")

    assert result.kind == "notice"
    assert "Add `yes` to copy release-check into your account" in result.text
    assert "no agent named" not in result.text
    assert global_agents.get_agent_by_name("release-check") is None


def test_the_owner_dispatcher_commits_an_agent_promote(tmp_path, monkeypatch) -> None:
    session, global_agents, _teams = _dispatcher_session(tmp_path, monkeypatch)
    _session_role(session.agent_registry, "release-check")

    result = _owner_slash("agent", session, "promote release-check yes")

    assert "Copied release-check to your account." in result.text
    assert global_agents.get_agent_by_name("release-check") is not None


def test_the_owner_dispatcher_reserves_the_team_promote_verb(tmp_path, monkeypatch) -> None:
    session, _agents, global_teams = _dispatcher_session(tmp_path, monkeypatch)
    _session_role(session.agent_registry, "helper")
    session.team_registry.create_team(
        TeamEditFields(
            name="pod",
            manager="manager",
            members=parse_members(["helper"]),
            instructions="Work together.",
        ),
        scope="session",
    )

    rehearsal = _owner_slash("team", session, "promote pod")
    assert "Add `yes` to copy helper, pod into your account" in rehearsal.text
    assert "no team named" not in rehearsal.text
    assert global_teams.get_team_by_name("pod") is None

    committed = _owner_slash("team", session, "promote pod yes")
    assert "Copied helper, pod to your account." in committed.text
    assert global_teams.get_team_by_name("pod") is not None


def test_the_owner_dispatcher_names_the_overwrite_first(tmp_path, monkeypatch) -> None:
    session, global_agents, _teams = _dispatcher_session(tmp_path, monkeypatch)
    _session_role(session.agent_registry, "release-check")
    global_agents.create_agent(_agent_fields(name="release-check", tags=["role"]))

    result = _owner_slash("agent", session, "promote release-check replace")

    assert result.style == "warning"
    assert result.text.startswith("Replace existing account definitions: release-check")
