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
from local_operator.resume import write_session_attachment
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


# Known limit (Q-NIT-2): a ``session_cls`` subclass that strips ``attach_team``
# cannot restore a STORED team — the constructor attaches the resolved team
# through that seam and raises TypeError; build stripped shapes on fresh transcripts.
def _session(root: Path, agents: Any, teams: Any, session_cls: type[Session] = Session) -> Session:
    return session_cls(
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


#: What a bound team whose definition is gone at reopen reports. The refusal
#: path cannot tell "deleted" from "not synced to this device yet" (both are
#: ``_DefinitionNotLoaded``), so it no longer claims "could not restore … —
#: renamed or deleted"; ``test_attachment_persistence.py`` pins the same string
#: at the session layer. Pinned here too so a TUI that drops or rewrites the
#: session's notice on adopt fails loudly instead of matching nothing.
STALE_TEAM_NOTICE = "lopdev: team definition is unavailable; re-attach via /team"


def _restore_notice(app: OperatorApp, session: Session) -> Any:
    """The NoticeBlock carrying ``session``'s restore notice, or ``None``.

    Keyed on the notice the SESSION recorded rather than on a copy literal: the
    old ``"could not restore"`` key went silently stale when the refusal copy
    changed, which made the on-screen check report "never mounted" for a notice
    that was in fact mounted — and made the clean-resume negative vacuous.
    """
    expected = session.attachment_restore_notice
    assert expected, "the session recorded no restore notice to look for"
    for block in app.query(NoticeBlock):
        if (block.text() or "") == expected:
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
async def test_the_band_names_the_team_and_its_manager_a_resume_restored(tmp_path) -> None:
    """Before this, both segments were blank on every resume — honestly so, the
    persona really was gone. Now the state comes back and the band shows it.

    Issue #2014: a team OWNS the agent slot, so the two segments a resume has to
    bring back are the team and the MANAGER it claimed. The sidecar here is the
    pre-#2014 shape — a team AND an unrelated profile — written directly, since
    the two attaches that would produce it are now refused; what must paint is
    the team and its manager, NOT the dropped profile.

    The team is BOUND first through the real ``attach_team`` path: a sidecar
    name alone is no longer authority (see the name-only sibling below), so the
    legacy pair only restores on a session whose binding record exists.
    """
    agents, teams = _registries(tmp_path)
    first = _session(tmp_path, agents, teams)  # creates tmp_path/sess
    first.attach_team(teams.get_team_by_name("lopdev"))
    write_session_attachment(tmp_path / "sess", team="lopdev", agent="auditor", goal="")

    resumed = _session(tmp_path, agents, teams)

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot, resumed)
        assert app._status is not None
        # The band paints the shared display form: a derived default that
        # adds nothing over the key paints the RAW NAME (D2), while a custom
        # label would paint itself.
        assert app._status._team == "lopdev"
        # The manager the TEAM claimed — never the unrelated profile the legacy
        # sidecar also named. Lowercase is what this path paints: the band's
        # segment goes through the shared display rule, and with no registry row
        # and no seed label to add information it falls back to the raw key.
        assert app._status._agent_profile == "manager"
        assert resumed.attachment_restore_notice == ""


@pytest.mark.asyncio
async def test_a_name_only_team_is_not_painted_and_says_why(tmp_path) -> None:
    """The complement of the bound case above: the same legacy sidecar with NO
    binding record names a team, but a name is not authority. The band must stay
    blank (painting it would claim a manager the prompt does not carry) and the
    refusal must reach the screen rather than downgrade silently.

    The mesh stamp without team identity marks the session as having ARRIVED
    name-only: an unmarked local journal is grandfathered instead (the legacy
    test below), so the arrival marker is what makes this the refused shape."""
    from local_operator.session.placement import MeshStamp, write_stamp_into

    agents, teams = _registries(tmp_path)
    _session(tmp_path, agents, teams)  # creates tmp_path/sess, nothing bound
    write_session_attachment(tmp_path / "sess", team="lopdev", agent="auditor", goal="")
    write_stamp_into(tmp_path / "sess", MeshStamp(session_id="sess", origin={}))

    resumed = _session(tmp_path, agents, teams)

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot, resumed)
        assert app._status is not None
        assert app._status._team == ""
        assert app._status._agent_profile == ""
        assert resumed.attachment_restore_notice == (
            "lopdev: recorded authority is unavailable; re-attach via /team"
        )
        assert _restore_notice(app, resumed) is not None


@pytest.mark.asyncio
async def test_a_legacy_local_team_is_painted(tmp_path) -> None:
    """The grandfathered sibling of the name-only refusal: the same legacy
    sidecar with no binding AND no arrival marker is a pre-binding session
    created on this machine, so it restores, is stamped, and paints the team's
    manager with no downgrade notice."""
    from local_operator.network.move_team import LOCAL_BINDING_NAME

    agents, teams = _registries(tmp_path)
    _session(tmp_path, agents, teams)  # creates tmp_path/sess, nothing bound
    write_session_attachment(tmp_path / "sess", team="lopdev", agent="auditor", goal="")

    resumed = _session(tmp_path, agents, teams)

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot, resumed)
        assert app._status is not None
        assert app._status._team == "lopdev"
        assert app._status._agent_profile == "manager"
        assert resumed.attachment_restore_notice == ""
    assert (tmp_path / "sess" / LOCAL_BINDING_NAME).exists()


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
        assert resumed.attachment_restore_notice == STALE_TEAM_NOTICE
        notice = _restore_notice(app, resumed)
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
    """R4: owner death swaps a AttachedSession facade for a REAL Session, which
    restores its attachment at construction. ``StatusLine.update`` treats
    ``None`` as leave-alone, so omitting the two segments left whatever the
    remote facade had painted — the one adopt sink that did not show them."""
    agents, teams = _registries(tmp_path)
    first = _session(tmp_path, agents, teams)
    first.attach_team(teams.get_team_by_name("lopdev"))

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
        assert app._status._team == "lopdev"  # the shared form, as on resume
        assert app._status._agent_profile == "manager"


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
        # Both halves: the session recorded nothing, AND nothing re-attach-shaped
        # reached the screen by another route. The old literal key could not
        # see the refusal copy at all, so it passed whatever was mounted.
        assert resumed.attachment_restore_notice == ""
        notices = [(n.text() or "") for n in app.query(NoticeBlock)]
        assert not any("re-attach" in text for text in notices), notices


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
        assert STALE_TEAM_NOTICE in notices, notices


@pytest.mark.asyncio
async def test_the_local_detach_verb_paints_both_segments(tmp_path) -> None:
    """Agent review round 1, MINOR-2: the LOCAL ``/team clear`` branch had no
    test at all, which is how MINOR-1 (the follower seam skipping the agent-band
    sync) shipped unnoticed.

    Two facts in one walk. When a team was in force, BOTH segments move — the
    roster and the manager name the team claimed for the agent slot — so the band
    cannot disagree with what was released. (The NO-team half — the verb as a true
    no-op that keeps an attached profile, design round 1 D1 — is asserted on this
    process's other two seams: ``test_slash_team_clear_with_no_team_keeps_the_
    profile`` for the routed path and ``test_clearing_a_team_that_is_not_
    attached_keeps_the_profile`` for the mutator itself; this cell is the
    coverage the local seam needed at all, agent review round 1 MINOR-2.)
    """
    agents, teams = _registries(tmp_path)
    first = _session(tmp_path, agents, teams)
    first.attach_team(teams.get_team_by_name("lopdev"))

    resumed = _session(tmp_path, agents, teams)

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot, resumed)
        assert app._status is not None
        assert app._status._team == "lopdev"
        assert app._status._agent_profile == "manager"

        app._cmd_team("clear", app._notice)
        for _ in range(6):
            await pilot.pause()

        assert resumed.active_team_name == ""
        assert app._status._team == ""
        # The manager the team claimed leaves with it.
        assert app._status._agent_profile == ""


@pytest.mark.asyncio
async def test_the_follower_detach_verb_repaints_in_the_same_turn(tmp_path) -> None:
    """MINOR-1's seam, asserted the way it can actually fail (agent review round
    2, MINOR-2).

    The first version of this cell paused the pilot six times before asserting,
    which is exactly enough for the frontend-state delta to repaint the agent
    segment on its own — so it passed against the PREVIOUS head too and guarded
    nothing. The band syncs synchronously inside the seam, so the assertion is
    made in the SAME turn as the call: no `await` between them. The receipt and
    the released state are checked afterwards.
    """
    from local_operator.session.frontend_state import SlashResult

    agents, teams = _registries(tmp_path)
    first = _session(tmp_path, agents, teams)
    first.attach_team(teams.get_team_by_name("lopdev"))

    resumed = _session(tmp_path, agents, teams)

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot, resumed)
        assert app._status is not None
        assert app._status._agent_profile == "manager"

        result = app._team_attach_slash_result("clear", resumed.team_registry, SlashResult)

        # SAME TURN: only the seam's own sync can have produced this.
        assert app._status._team == ""
        assert app._status._agent_profile == ""

        for _ in range(6):
            await pilot.pause()

        assert result.kind == "notice"
        assert result.text == "no team active; this session uses its base instructions."
        assert result.data["type"] == "team_attached"
        assert resumed.active_team_name == ""


@pytest.mark.asyncio
async def test_the_follower_no_op_receipt_is_the_bare_sentence_with_its_keys(tmp_path) -> None:
    """The follower host's half of the bare no-op — sentence AND its 4-key data.

    The routed twin is pinned in ``test_team_agent_precedence.py``; with nothing
    in force the verb moved nothing, and this half must word it identically and
    carry the same ``team_attached`` payload — including ``request`` — or a
    follower's band keys drift from the owner's. Two producers, one sentence:
    pinned on both, so a drift in either reddens (agent review round 1, R1).
    """
    from local_operator.session.frontend_state import SlashResult

    agents, teams = _registries(tmp_path)
    resumed = _session(tmp_path, agents, teams)

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot, resumed)

        result = app._team_attach_slash_result("clear", resumed.team_registry, SlashResult)

        assert result.kind == "notice"
        assert result.style == "info"
        assert result.text == "no team is attached, so nothing was detached."
        assert result.data == {
            "type": "team_attached",
            "team": "",
            "manager": "",
            "request": "",
        }


@pytest.mark.asyncio
async def test_the_follower_detach_refusal_names_what_the_session_can_do(tmp_path) -> None:
    """The contract's no-team refusal, byte-pinned on the follower seam.

    A session that cannot run a team at all answers the detach verb with a
    WARNING, and a refusal must not be published as a detach: no ``data`` rides
    it, or a client painting from the frame would clear a segment that never
    moved. Reachable on a viewer (``AttachedSession``, no local detach seam);
    the words come from the same ``_team_detach_receipt`` the local handler
    pushes as a plain notice, so the two surfaces cannot drift.
    """
    from local_operator.session.frontend_state import SlashResult

    agents, teams = _registries(tmp_path)

    # The viewer's shape: the LISTING resolves, the DETACH does not. A subclass
    # rather than attribute surgery on ``Session`` itself (``test_slash_echo``'s
    # NoAttachSession idiom), so no later cell loses the method.
    class NoRunTeamSession(Session):
        attach_team = None  # type: ignore[assignment]

    resumed = _session(tmp_path, agents, teams, session_cls=NoRunTeamSession)

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot, resumed)

        result = app._team_attach_slash_result("clear", resumed.team_registry, SlashResult)

        assert result.kind == "notice"
        assert result.style == "warning"
        assert result.text == (
            "this session can list and chart teams, but not run one, "
            "so there is no team to detach"
        )
        assert result.data == {}


@pytest.mark.asyncio
async def test_the_follower_attach_refusal_names_what_the_session_can_do(tmp_path) -> None:
    """The attach half of the no-team refusal, byte-pinned on the follower seam.

    M12 follow-up (#2072/#2078): the follower seam said "this session cannot
    run a team. /team chart …" where the TUI-local handler says what the
    session CAN do — the same drift shape as the detach refusal above. The
    routed twin is pinned in ``tests/unit/session/test_team_agent_precedence``
    ``::TestTheRoutedCommands``, so the surfaces cannot disagree silently.
    """
    from local_operator.session.frontend_state import SlashResult

    agents, teams = _registries(tmp_path)

    class NoRunTeamSession(Session):
        attach_team = None  # type: ignore[assignment]

    resumed = _session(tmp_path, agents, teams, session_cls=NoRunTeamSession)

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot, resumed)

        result = app._team_attach_slash_result("lopdev", resumed.team_registry, SlashResult)

        assert result.kind == "notice"
        assert result.style == "warning"
        assert result.text == (
            "this session can list and chart teams, but not run one. "
            "/team chart <name> shows a roster"
        )
        assert result.data == {}


@pytest.mark.asyncio
async def test_the_team_listing_footer_advertises_the_detach_verb(tmp_path) -> None:
    """Design round 1, D3: the verb that LEAVES the state is advertised where the
    listing that invites the state is painted — mirroring ``Detach: /agent
    clear`` — and disappears when no team is in force, so the hint cannot outlive
    the attachment it describes."""
    agents, teams = _registries(tmp_path)
    first = _session(tmp_path, agents, teams)
    first.attach_team(teams.get_team_by_name("lopdev"))

    resumed = _session(tmp_path, agents, teams)

    async def factory() -> Session:
        return resumed

    app = OperatorApp(factory)
    async with app.run_test(size=(120, 24)) as pilot:
        await _adopted(app, pilot, resumed)

        authoring = " · /team chart <name> · /team new · /team edit <name>"
        assert app._team_listing_footer() == (
            "Send: /team <name> <message> · Detach: /team clear" + authoring
        )

        app._cmd_team("clear", app._notice)
        for _ in range(6):
            await pilot.pause()

        assert app._team_listing_footer() == "Send: /team <name> <message>" + authoring
