"""The status band after a resume has to tell the truth about what is attached.

``Session`` restores a stored ``/team`` and ``/agent`` at construction (see
``tests/unit/session/test_attachment_persistence.py``); these tests cover the
front end's half, driving the REAL ``OperatorApp`` against a real resumed
session rather than a fake, because the contract under test is precisely that
the band is driven FROM the session:

* a restored attachment must reach the band on adopt, or the user sees a blank
  segment beside a manager that is in force;
* a stale one must leave the segment blank AND say why, because the alternative
  readings — a segment naming a team whose briefs are not in the prompt, or a
  silent downgrade to the base voice — are the two ways this can lie.

Both cold ``--resume <id>`` and the in-TUI ``/resume`` picker land in
``_adopt_session``, so exercising that sink covers both routes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.teams import Team, TeamRegistry
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.unit.session.test_session import MODEL, ScriptedStream


def _role_fields(**overrides: Any) -> AgentEditFields:
    base: dict[str, Any] = dict(
        name=None,
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


def _registries(root: Path) -> tuple[AgentRegistry, TeamRegistry]:
    agents = AgentRegistry(root)
    agents.create_agent(_role_fields(name="auditor", description="Audits", tags=["role"]))
    teams = TeamRegistry(root)
    teams.save_team(
        Team(
            id="t-lopdev",
            name="lopdev",
            created_date=datetime.now(timezone.utc),
            manager="manager",
            instructions="Ship reviewed work.",
            project="local-operator",
        )
    )
    return agents, teams


def _session(root: Path, agents: Any, teams: Any) -> Session:
    return Session(
        model=MODEL,
        stream_fn=ScriptedStream([[]]),
        tools=[],
        transcript=Transcript(root / "sess"),
        system_blocks_provider=lambda: [],
        agent_registry=agents,
        team_registry=teams,
    )


async def _adopted(app: OperatorApp, pilot: Any) -> None:
    """Let the boot worker adopt the session and the band settle."""
    for _ in range(10):
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_band_names_the_team_and_agent_a_resume_restored(tmp_path) -> None:
    """Before this, both segments were blank on every resume — honestly so, the
    persona really was gone. Now the state comes back and the band shows it."""
    agents, teams = _registries(tmp_path)
    first = _session(tmp_path, agents, teams)
    first.attach_team(teams.get_team_by_name("lopdev"))
    first.attach_agent_profile("auditor")

    resumed = _session(tmp_path, agents, teams)

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot)
        assert app._status is not None
        assert app._status._team == "lopdev"
        assert app._status._agent_profile == "auditor"


@pytest.mark.asyncio
async def test_a_stale_team_leaves_the_segment_blank_and_says_why(tmp_path) -> None:
    """The band must never paint a name that is not stamped into the prompt, and
    the downgrade must not be silent."""
    agents, teams = _registries(tmp_path)
    first = _session(tmp_path, agents, teams)
    first.attach_team(teams.get_team_by_name("lopdev"))

    # The team is gone by the time the session is reopened (renamed or deleted).
    resumed = _session(tmp_path, agents, TeamRegistry(tmp_path / "elsewhere"))

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot)
        assert app._status is not None
        assert app._status._team == ""
        notices = [(n.text() or "") for n in app.query(NoticeBlock)]
        assert any("lopdev" in text and "without it" in text for text in notices), notices


@pytest.mark.asyncio
async def test_a_clean_resume_raises_no_notice(tmp_path) -> None:
    """A working restore is not news; only a miss is."""
    agents, teams = _registries(tmp_path)
    first = _session(tmp_path, agents, teams)
    first.attach_team(teams.get_team_by_name("lopdev"))

    resumed = _session(tmp_path, agents, teams)

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot)
        notices = [(n.text() or "") for n in app.query(NoticeBlock)]
        assert not any("could not restore" in text for text in notices), notices


@pytest.mark.asyncio
async def test_the_stale_notice_does_not_end_the_empty_state(tmp_path) -> None:
    """It is infrastructure news the user did not ask for, so it lands under the
    splash the way the MCP startup record does rather than collapsing the boot
    composition — the failure that once made the centred prompt unreachable."""
    agents, teams = _registries(tmp_path)
    first = _session(tmp_path, agents, teams)
    first.attach_team(teams.get_team_by_name("lopdev"))

    resumed = _session(tmp_path, agents, TeamRegistry(tmp_path / "elsewhere"))

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot)
        # The notice IS on screen; what must not happen is the splash retiring
        # for it. ``_welcome_visible`` is the authoritative "the conversation
        # has started" edge both boot layouts hang off.
        assert app._welcome_visible is True
        notices = [(n.text() or "") for n in app.query(NoticeBlock)]
        assert any("could not restore" in text for text in notices), notices
