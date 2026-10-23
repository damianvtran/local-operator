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
from typing import Any

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.profile_scope import scope_session_registries
from local_operator.teams import TeamEditFields, TeamRegistry, parse_members
from local_operator.tui.app import OperatorApp
from local_operator.tui.promote_verb import PromoteRequest, parse_promote
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
        assert "copied release-check to your account" in _notices(app)[-1]
        assert "this conversation keeps its own copy" in _notices(app)[-1]
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
        assert "to replace release-check in your account" in _notices(app)[-1]

        await _run(pilot, app, "/agent promote release-check replace yes")
        assert "copied release-check to your account" in _notices(app)[-1]
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

        assert "Use /agent promote <name> [replace] [yes]." in _notices(app)[-1]


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
