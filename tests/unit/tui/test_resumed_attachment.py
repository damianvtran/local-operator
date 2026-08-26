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


async def _adopted(app: OperatorApp, pilot: Any, session: Any) -> None:
    """Wait until ``session`` has actually been adopted, then let layout settle.

    Polls for the EVENT rather than spending a fixed pause budget, which is the
    file's house idiom (``test_app_pilot.py`` waits ``if app._session is
    session: break`` with a bound of 50). The session is built by an async boot
    worker, so how many pauses the adopt takes is a function of machine load,
    not of the code under test: measured latency for these tests ranged from 1
    to 23 pauses across runs. A fixed budget of 10 therefore sat mid
    distribution and failed roughly half of all runs — and the assertions it
    failed were this file's own guard for the feature, including the D1 notice
    check. Raising the budget would only lengthen the fuse; waiting on the
    condition removes it.

    Identity, not ``is not None``: these tests hand the factory one specific
    session and every assertion below is about THAT session's restored state,
    so waking on any adopt would reintroduce a race with a different shape.

    The trailing pauses are for LAYOUT, not for the adopt: ``_adopt_session`` is
    synchronous end to end, so once it has run the blocks are mounted, but
    ``test_the_stale_notice_is_on_screen_on_a_session_with_history`` asserts on
    ``region``, which is only meaningful after a layout pass has placed them.
    Bounded and load-independent, unlike waiting for the adopt itself.
    """
    for _ in range(50):
        await pilot.pause()
        if app._session is session:
            break
    assert app._session is session, "the boot worker never adopted the session"
    for _ in range(2):
        await pilot.pause()


async def _with_history(session: Session, turns: int) -> None:
    """Give a session enough replayed turns to fill more than one screen.

    A stale attachment is only reachable on a session that ALREADY RAN, so a
    test asserting on that notice against an EMPTY transcript is testing the
    one shape the feature never meets in the field (D1).
    """
    from local_operator.harness.types import Message, TextContent

    for i in range(turns):
        await session._transcript.append_message(
            Message(role="user", content=[TextContent(text=f"question {i}")])
        )
        await session._transcript.append_message(
            Message(role="assistant", content=[TextContent(text=f"answer {i}: disk full.")])
        )


def _restore_notice(app: OperatorApp) -> Any:
    """The stale-attachment NoticeBlock, or ``None``."""
    for block in app.query(NoticeBlock):
        if "could not restore" in (block.text() or ""):
            return block
    return None


def _is_on_screen(app: OperatorApp, block: Any) -> bool:
    """Whether ``block`` actually falls inside the transcript's painted area.

    ``Widget.region`` is SCREEN-relative and already carries the scroll, so
    comparing it against the transcript's own ``region`` (also screen-relative)
    answers "is this on screen" — where ``window_region`` is in virtual space
    and would compare two different coordinate systems. Verified to
    discriminate: it is ``False`` against the pre-fix ordering, where the notice
    sat at ``y=-63`` under a viewport starting at ``y=65``, and ``True`` after.
    """
    from local_operator.tui.widgets.transcript import TranscriptView

    return block.region.overlaps(app.query_one(TranscriptView).region)


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
        await _adopted(app, pilot, resumed)
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
        await _adopted(app, pilot, resumed)
        assert app._status is not None
        assert app._status._team == ""
        notices = [(n.text() or "") for n in app.query(NoticeBlock)]
        assert any("lopdev" in text and "re-attach" in text for text in notices), notices


@pytest.mark.asyncio
async def test_the_stale_notice_is_on_screen_on_a_session_with_history(tmp_path) -> None:
    """D1: the notice has to be VISIBLE, not merely mounted.

    A stale attachment can only happen on a session that already ran, so this is
    the only shape that matters. Raised before the history replay the notice
    became block 0, the replay was appended after it, the transcript scrolled to
    the bottom, and it landed at ``y=-63`` under a viewport starting at ``y=65``
    — the silent downgrade the notice exists to prevent, reproduced by the
    notice itself. Membership in ``app.query(NoticeBlock)`` cannot see that;
    only viewport containment can.
    """
    agents, teams = _registries(tmp_path)
    first = _session(tmp_path, agents, teams)
    await _with_history(first, 20)
    first.attach_team(teams.get_team_by_name("lopdev"))

    resumed = _session(tmp_path, agents, TeamRegistry(tmp_path / "elsewhere"))

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(100, 26)) as pilot:
        await _adopted(app, pilot, resumed)
        notice = _restore_notice(app)
        assert notice is not None, "the stale-attachment notice was never mounted"
        assert _is_on_screen(
            app, notice
        ), f"notice mounted but off screen: region={tuple(notice.region)}"
        # And it is the LAST block, i.e. directly above the composer where the
        # user's eye already is, rather than merely somewhere on screen.
        from local_operator.tui.widgets.transcript import TranscriptView

        blocks = list(app.query_one(TranscriptView).children)
        assert blocks[-1] is notice, "the notice is not the most recent block"


@pytest.mark.asyncio
async def test_a_restored_goal_is_reported(tmp_path) -> None:
    """D4: a standing goal steers ``/loop``, so a silently restored one is
    invisible state driving the conversation. The band is the wrong home for a
    sentence; a one-line receipt is the right weight."""
    agents, teams = _registries(tmp_path)
    first = _session(tmp_path, agents, teams)
    first.set_goal("ship the resume fix and cut the release")

    resumed = _session(tmp_path, agents, teams)

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot, resumed)
        notices = [(n.text() or "") for n in app.query(NoticeBlock)]
        assert any(
            "goal restored" in text and "cut the release" in text for text in notices
        ), notices


@pytest.mark.asyncio
async def test_the_takeover_adopt_paints_the_restored_attachment(tmp_path) -> None:
    """R4: owner death swaps a RemoteSession facade for a REAL Session, which
    restores its attachment at construction. ``StatusLine.update`` treats
    ``None`` as leave-alone, so omitting the two segments left whatever the
    remote facade had painted — the one adopt sink that did not show them."""
    agents, teams = _registries(tmp_path)
    first = _session(tmp_path, agents, teams)
    first.attach_team(teams.get_team_by_name("lopdev"))
    first.attach_agent_profile("auditor")

    # Stands in for the remote facade: a different session, nothing attached.
    bare = Session(
        model=MODEL,
        stream_fn=ScriptedStream([[]]),
        tools=[],
        transcript=Transcript(tmp_path / "other"),
        system_blocks_provider=lambda: [],
    )

    async def factory() -> Session:
        return bare

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot, bare)
        assert app._status is not None
        assert app._status._team == ""

        await app._adopt_takeover_session(_session(tmp_path, agents, teams))
        for _ in range(6):
            await pilot.pause()
        assert app._status._team == "lopdev"
        assert app._status._agent_profile == "auditor"


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
        await _adopted(app, pilot, resumed)
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
        await _adopted(app, pilot, resumed)
        # The notice IS on screen; what must not happen is the splash retiring
        # for it. ``_welcome_visible`` is the authoritative "the conversation
        # has started" edge both boot layouts hang off.
        assert app._welcome_visible is True
        notices = [(n.text() or "") for n in app.query(NoticeBlock)]
        assert any("could not restore" in text for text in notices), notices
