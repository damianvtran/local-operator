"""The sidebar's switch path is a conversation SWAP, and must reset like one.

Two regressions the session sidebar introduced, both reported against `/new`
and both only reachable THROUGH a sidebar switch — plain `/new` was never
broken, which is why they survived the original round:

* the splash did not come back on `/new`, because the prepared view for a
  target WITH history carries no ``WelcomeView`` and `_welcome` became None;
* the band kept the previous conversation's cost and context, because the
  commit path reset neither and the incoming snapshot is leave-alone on the
  absent values a never-used session legitimately reports.

The over-fix guard is asserted in the same file and matters as much as the
fixes: a switch BACK onto a conversation with real spend has to land on that
spend, not on the zero the reset writes.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from local_operator.session.frontend_state import (
    FrontendModelSpec,
    FrontendSessionState,
    FrontendStateStore,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from local_operator.tui.widgets.status_line import FORK_PENDING_TEXT
from local_operator.tui.widgets.toast import Toast
from local_operator.tui.widgets.welcome import WelcomeView
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def isolated_swap(tmp_path, monkeypatch):
    # Headless apps must never rename the caller's real multiplexer workspace.
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_terminal_title", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_multiplexer_broadcast", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_herdr_reporter", lambda _self: None)


class SidebarRemote(FakeSession):
    """A ``FakeSession`` wearing the owner-backed surface the sidebar requires.

    A ``MagicMock(spec=RemoteSession)`` cannot stand in here: the commit path
    renders the band and the splash from this object, and a mock's auto-created
    attributes reach Rich as non-string values. Extending the suite's existing
    fake keeps every protocol method real and adds only the navigation surface.

    The frontend store is a REAL one so ``_adopt_session`` runs the production
    repaint — which is precisely the leave-alone-on-None behaviour the band
    regression turns on.
    """

    is_remote = True

    def __init__(
        self,
        session_id: str,
        *,
        history=(),
        cost=None,
        context=None,
        pending_fork: bool = False,
    ) -> None:
        super().__init__()
        self._id = session_id
        #: Opt-in, because `has_pending_fork` is the fact the band's `forking`
        #: segment reads and only ONE test wants it true. A `/fork` issued on a
        #: streaming session defers to a turn boundary (`owned.py:2208`), so the
        #: request stays live — and cancellable — across a park.
        self._pending_fork = pending_fork
        self._history = list(history)
        self._store = FrontendStateStore(
            FrontendSessionState(
                session_id=session_id,
                epoch=f"epoch-{session_id}",
                # `model_label` is DERIVED from this spec, not a field: the band
                # and the splash both read the label, so a snapshot without a
                # spec would repaint them empty and mask what is being asserted.
                selected_model=FrontendModelSpec(provider="test", model_id="model"),
                cumulative_parent_cost=cost,
                context_tokens=context,
                context_window=200_000 if context is not None else None,
            )
        )
        self.is_cold = False
        self.has_pending_gate_reply = False
        self.display_history_current = True
        self.display_history_revision = 1

    @property
    def session_id(self) -> str:
        return self._id

    def has_pending_fork(self) -> bool:
        return self._pending_fork

    @property
    def frontend_state(self):
        return self._store.state

    def subscribe_frontend(self, handler):
        return self._store.subscribe(handler)

    def display_history_window(self):
        return list(self._history)

    def history(self):
        return list(self._history)

    @property
    def history_message_count(self) -> int:
        return len(self._history)

    async def ensure_display_current(self) -> None:
        return None

    async def ensure_display_anchor(self, _anchor) -> bool:
        return True

    def resume_viewer_gates(self) -> None:
        return None

    def suspend_viewer_gates(self, *, auto_approve=False, keep_answer=False) -> None:
        return None

    def set_takeover_callback(self, _callback) -> None:
        return None

    def set_stopped_callback(self, _callback) -> None:
        return None

    def set_owner_gone_callback(self, _callback) -> None:
        return None


def _message(role: str, text: str):
    return SimpleNamespace(role=role, text=text, tool_calls=None, content=text)


async def _switch(app: OperatorApp, pilot, remote: SidebarRemote) -> None:
    """Drive the REAL prepare/commit pair, the way a sidebar click does.

    The lease is stubbed because leasing reaches for an owner record on disk;
    everything after it — the prepared replay, the parked outgoing view, the
    commit, the adopt — is production code, which is where both bugs lived.
    """
    source = app._sidebar_sources.get(remote.session_id)
    if source is None:
        source = SessionInteraction(remote)
        app._sidebar_sources[remote.session_id] = source

    async def lease(_session_id, *, speculative=False):
        source.preparations += 1
        return source

    app._lease_sidebar_source = lease  # type: ignore[method-assign]
    prepare = asyncio.ensure_future(app._prepare_sidebar_session(remote.session_id))
    # Preparation waits on a laid-out frame, so the pilot has to keep pumping.
    for _ in range(400):
        if prepare.done():
            break
        await pilot.pause()
    prepared = prepare.result()
    future = app._commit_sidebar_session(remote.session_id, prepared, 0)
    for _ in range(20):
        await pilot.pause()
    if future is not None and not future.done():
        # The ready-frame future settles on a real paint; the pilot's frames are
        # enough for the assertions here and the timer must not outlive the app.
        future.cancel()
    for _ in range(10):
        await pilot.pause()


@pytest.mark.asyncio
async def test_new_after_a_sidebar_switch_still_shows_the_splash():
    """`/new` must land on the splash even when the view came from the sidebar.

    The splash is composed once, into the boot transcript. Sidebar navigation
    mounts a NEW ``TranscriptView`` per target and gives it a ``WelcomeView``
    only when the target has no history, so after a switch onto a conversation
    with messages the app held no usable splash at all: `/new` applied the boot
    layout over an empty transcript with nothing centred in it.
    """
    fresh = SidebarRemote("fresh-session")
    home = SidebarRemote("home-session")
    busy = SidebarRemote(
        "busy-session",
        history=[_message("user", "a question"), _message("assistant", "an answer")],
    )

    async def resume_factory(_resume_id):
        return fresh

    app = OperatorApp(lambda: _factory(home), resume_factory=resume_factory)
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()

            await _switch(app, pilot, busy)
            # The precondition, asserted so the test still describes the bug if
            # the sidebar ever starts mounting an empty state unconditionally.
            assert app._transcript_view().blocks()

            app._run_slash_command("/new")
            for _ in range(80):
                await pilot.pause()
            await asyncio.sleep(0.4)
            for _ in range(40):
                await pilot.pause()

            view = app._transcript_view()
            welcome = app._welcome
            assert welcome is not None, "the app holds no splash after a sidebar switch"
            assert welcome in list(view.children), "the splash is not in the visible transcript"
            assert welcome.display, "the splash is mounted but hidden"
            assert app._welcome_visible is True
            assert app.screen.has_class("boot")
            # One empty state per view, however many switches preceded it.
            assert len([c for c in view.children if isinstance(c, WelcomeView)]) == 1


@pytest.mark.asyncio
async def test_sidebar_commit_clears_the_previous_conversations_cost_and_context():
    """Switching back to an untouched `/new` must not show the other's spend.

    ``_commit_sidebar_session`` used to reset none of the band and rely on the
    incoming snapshot to repaint it. That snapshot is leave-alone on absent
    values, and a conversation that has never had a turn reports exactly that
    (`cumulative_cost` None, `context_tokens` None) — so the busy session's
    figures stayed on screen over the fresh conversation.

    The second half is the over-fix guard: the reset runs BEFORE the adopt, so a
    conversation with real spend is repainted from its own snapshot rather than
    left on the zero.
    """
    fresh = SidebarRemote("fresh-session")
    home = SidebarRemote("home-session")
    busy = SidebarRemote(
        "busy-session",
        history=[_message("user", "a question"), _message("assistant", "an answer")],
        cost=12.3456,
        context=98_765,
    )

    async def resume_factory(_resume_id):
        return fresh

    app = OperatorApp(lambda: _factory(home), resume_factory=resume_factory)
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()
            assert app._status is not None

            await _switch(app, pilot, busy)
            assert app._status._cost == "$12.35"
            assert app._status._context_tokens == 98_765

            await _switch(app, pilot, fresh)
            assert app._status._cost == "", "the previous conversation's cost is still painted"
            assert not app._status._context_tokens, "the previous context reading is still painted"

            # Back onto the conversation that genuinely spent it: the snapshot
            # puts the real figure back, so the reset must not have cost it.
            await _switch(app, pilot, busy)
            assert app._status._cost == "$12.35", "restored spend was zeroed by the reset"
            assert app._status._context_tokens == 98_765


@pytest.mark.asyncio
async def test_a_parked_conversation_keeps_its_splash_notice():
    """A sidebar round trip must not discard the setup warning on the splash.

    `_splash_notice` describes ONE conversation's empty state — the `/login`,
    unknown-provider and no-model-set warnings. `/new`, `/resume` and `/reload`
    RETIRE the conversation they leave, so clearing it there is right; the
    sidebar PARKS one and comes back to it, and nothing on that path puts the
    notice back.

    The loss was delayed rather than immediate, which is what made it worth a
    guard: `WelcomeView` reads the notice through a closure, so the row survives
    the switch frame and vanishes at the next `refresh_info()` — the 0.25 s poll
    or any model change. Asserting the ATTRIBUTE rather than the painted row is
    therefore deliberate: it is the state the next repaint reads, and it fails
    at the moment of the loss instead of a quarter-second later.
    """
    home = SidebarRemote("home-session")
    busy = SidebarRemote(
        "busy-session",
        history=[_message("user", "a question"), _message("assistant", "an answer")],
    )

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()

            warning = "/login openai to get started - no provider configured."
            app._announce_on_splash(warning, "warning")
            assert app._splash_notice == warning

            await _switch(app, pilot, busy)
            await _switch(app, pilot, home)

            assert app._splash_notice == warning, (
                "the parked conversation's setup warning was discarded by a " "sidebar round trip"
            )
            # And it must survive the repaint that exposes the loss. The splash
            # reads its facts through a closure, so a cleared notice leaves the
            # already-drawn row standing and takes it away at the next poll —
            # this is that poll, driven explicitly rather than waited for.
            assert app._welcome is not None
            app._welcome.refresh_info()
            for _ in range(5):
                await pilot.pause()
            assert (
                app._welcome._info.notice == warning
            ), "the warning survived on the app but the splash repainted without it"


@pytest.mark.asyncio
async def test_a_parked_conversation_keeps_its_dock_density():
    """A `ctrl+g` set on the conversation the user returns to must survive.

    `reset_density()` exists for a swap that retires: "a `ctrl+g` pressed
    against the old children does not pin the new ones" (#525 design §2). On
    the sidebar path the old children ARE the ones being returned to, and the
    re-seed happens at the panel's next non-empty sync — so the user's explicit
    choice was discarded by the act of glancing elsewhere.
    """
    home = SidebarRemote("home-session")
    busy = SidebarRemote(
        "busy-session",
        history=[_message("user", "a question"), _message("assistant", "an answer")],
    )

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()

            panel = app._subagent_panel
            assert panel is not None
            # The state a `ctrl+g` leaves behind: seeded, and pinned by the user.
            panel._density_seeded = True
            panel._user_density = True

            await _switch(app, pilot, busy)
            await _switch(app, pilot, home)

            assert panel._user_density, "the user's ctrl+g was discarded by a sidebar switch"
            assert panel._density_seeded, "the dock density was re-seeded on a park"


@pytest.mark.asyncio
async def test_returning_to_an_empty_conversation_shows_the_splash_under_a_notice():
    """The operator's reported flow, end to end, with an infrastructure notice.

    `/new` → click a session with history → click back. The returned-to
    conversation is the same untouched `/new` it was, so it must land on the
    splash — but `_adopt_session` re-emits infrastructure notices (build skew, a
    failed MCP server) on EVERY swap, and the commit path asked
    `bool(view.blocks())`, which counts them. That is the wrong question: those
    blocks are appended with `ends_empty_state=False` precisely so they land
    UNDER the splash rather than retiring it, and a user on a skewed build or
    with one broken MCP server therefore never got an empty state back.

    The notice is raised through the production `_system_notice`, the same call
    `_check_build_skew` makes, so this pins the predicate and not a fixture.
    """
    fresh = SidebarRemote("fresh-session")
    home = SidebarRemote("home-session")
    busy = SidebarRemote(
        "busy-session",
        history=[_message("user", "a question"), _message("assistant", "an answer")],
        cost=12.3456,
        context=98_765,
    )

    async def resume_factory(_resume_id):
        return fresh

    app = OperatorApp(lambda: _factory(home), resume_factory=resume_factory)
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()

            app._run_slash_command("/new")
            for _ in range(80):
                await pilot.pause()
            await asyncio.sleep(0.4)
            for _ in range(40):
                await pilot.pause()

            # The infrastructure notice the swap re-emits, posted the way the
            # product posts it. The splash stays up under it — that is the
            # documented contract of `ends_empty_state=False`, asserted here so
            # step 4 is compared against a step 2 that is known to be right.
            app._system_notice("this session is running an older version than this window", "note")
            for _ in range(10):
                await pilot.pause()
            assert app._transcript_view().blocks(), "the notice did not reach the transcript"
            assert app._welcome_visible is True, "the notice retired the splash on `/new` itself"

            await _switch(app, pilot, busy)
            assert app._welcome_visible is not True

            await _switch(app, pilot, fresh)
            for _ in range(20):
                await pilot.pause()

            view = app._transcript_view()
            # The precondition that made the old predicate wrong: the returned-to
            # transcript is NOT empty, it just has not started a conversation.
            assert view.blocks(), "no notice on the return leg — the flow is not reproduced"
            assert not view.conversation_started()
            welcome = app._welcome
            assert welcome is not None, "no splash after returning to the `/new` conversation"
            assert welcome in list(view.children), "the splash is not in the visible transcript"
            assert welcome.display, "the splash is mounted but hidden"
            assert app._welcome_visible is True
            assert app.screen.has_class("boot")


@pytest.mark.asyncio
async def test_a_parked_conversation_keeps_its_pending_fork_indicator():
    """A `/fork` still awaiting its boundary must survive a park-and-return.

    The third RETIRE-only clear, found by the same audit that produced the other
    two (review round 2, MAJOR-3). `fork_pending` is the one field the ungated
    `status.update` writes that NEITHER restore path repaints: `_adopt_session`
    does not touch it, `FrontendSessionState` carries no fork-pending field so a
    snapshot cannot either, and the only writer of the truth
    (`_sync_fork_pending`) is called from neither adoption path. So a clear on
    the park leg is permanent for the lifetime of the request.

    Why that matters more than a missing glyph: a deferred fork stays LIVE and
    Ctrl+C-cancellable until the turn ends, so a band that stops saying `forking`
    is the inverse of the lie `_schedule_fork_report` forbids — the user gets no
    cue that a fork is coming and it lands minutes later out of nowhere.

    Asserted on the RENDERED row, not on the flag: round 1's bug was state that
    was set while the user saw nothing, so `_status._fork_pending` alone cannot
    close this. `is_showing` confirms the drop ladder kept the segment at this
    width, which is what makes reading the row a fair test rather than a
    width-sensitive one.
    """
    home = SidebarRemote("home-session", pending_fork=True)
    busy = SidebarRemote(
        "busy-session",
        history=[_message("user", "a question"), _message("assistant", "an answer")],
    )

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()

            # The request, as `/fork` on a streaming session makes it: the
            # session reports the pending fork and the band is synced from it.
            app._sync_fork_pending()
            for _ in range(5):
                await pilot.pause()
            assert app._status is not None
            assert app._status.is_showing("fork"), "the band never showed `forking` to begin with"
            assert FORK_PENDING_TEXT in app._status.render_text(100).plain

            await _switch(app, pilot, busy)
            await _switch(app, pilot, home)
            for _ in range(10):
                await pilot.pause()

            # The truth is unchanged — the fork is still waiting for a boundary.
            assert home.has_pending_fork() is True
            assert app._status.is_showing("fork")
            assert FORK_PENDING_TEXT in app._status.render_text(100).plain, (
                "the band stopped saying `forking` after a sidebar round trip, "
                "while the request is still live and still cancellable"
            )


@pytest.mark.asyncio
async def test_the_fork_indicator_describes_the_session_the_user_is_looking_at():
    """A parked conversation's `forking` must not be painted over another one.

    The other half of the park gate above, and a defect that predates this PR:
    on base, nothing on the sidebar path wrote `fork_pending` at all, so the
    outgoing conversation's segment simply stayed lit over the session switched
    TO. `retire=False` kept that behaviour rather than causing it — it fixed the
    return leg — so both legs are only correct once `_commit_sidebar_session`
    re-derives the flag from the incoming session after its adopt.

    Why this is worse than a stale readout (design round 3, D3): the segment is
    an AFFORDANCE. It advertises `esc`, and `action_stop` probes `self._session`
    — now a different conversation — so the offered cancel cannot reach the fork
    it names, and nothing ever clears it. That is precisely the lie
    `_schedule_fork_report` brings the segment down to avoid, one session over.

    Asserted on the RENDERED band, like its sibling, because `_fork_pending`
    alone cannot see this: a code round can verify the `None` leave-alone is
    correct and still miss that the resulting FRAME describes the wrong
    conversation. Both legs live in one test because they are the pair a
    one-sided fix breaks: clearing unconditionally would take the live
    indicator back off the return leg (review round 2, MAJOR-3).
    """
    home = SidebarRemote("home-session", pending_fork=True)
    busy = SidebarRemote(
        "busy-session",
        history=[_message("user", "a question"), _message("assistant", "an answer")],
        cost=12.35,
        context=98_765,
    )

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()

            app._sync_fork_pending()
            for _ in range(5):
                await pilot.pause()
            assert app._status is not None
            assert app._status.is_showing("fork"), "the band never showed `forking` to begin with"
            assert FORK_PENDING_TEXT in app._status.render_text(100).plain

            await _switch(app, pilot, busy)
            for _ in range(10):
                await pilot.pause()

            # The park leg. `busy` has no fork of any kind, so the segment it
            # would be offering `esc` for does not exist on this conversation.
            assert busy.has_pending_fork() is False
            assert FORK_PENDING_TEXT not in app._status.render_text(100).plain, (
                "the parked conversation's `forking` is painted over the session "
                "switched TO, advertising an `esc` that cannot reach that fork"
            )

            await _switch(app, pilot, home)
            for _ in range(10):
                await pilot.pause()

            # The return leg, unchanged: the fork is still waiting for its
            # boundary, so the segment it can still be cancelled from is back.
            assert home.has_pending_fork() is True
            assert app._status.is_showing("fork")
            assert FORK_PENDING_TEXT in app._status.render_text(100).plain, (
                "resolving the indicator for the incoming session cost the "
                "live fork its indicator on the conversation that owns it"
            )


@pytest.mark.asyncio
async def test_a_park_withdraws_the_splash_toast_but_keeps_the_notice():
    """The park drops the outgoing toast and keeps the outgoing splash row.

    The notice and the toast are raised together by `_announce_on_splash` and
    have opposite lifetimes, which is the distinction this asserts (design round
    2, D2). The NOTICE is the parked conversation's own empty-state content and
    must survive the round trip — that is what this PR exists to deliver. The
    TOAST is a transient overlay ABOUT the conversation being left, so gating it
    with the notice made a park carry it onto the session switched TO: "No
    provider configured" sitting over a working session with real spend.

    Both halves are asserted in ONE test deliberately, because they are a pair
    that can drift apart: withdrawing the toast must not cost the notice, and
    keeping the notice must not keep the toast. Split across two tests, a fix
    for either could silently break the other and still show green.
    """
    home = SidebarRemote("home-session")
    busy = SidebarRemote(
        "busy-session",
        history=[_message("user", "a question"), _message("assistant", "an answer")],
        cost=12.35,
        context=98_765,
    )

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()

            warning = "/login openai to get started - no provider configured."
            app._announce_on_splash(warning, "warning")
            for _ in range(5):
                await pilot.pause()

            live = [toast for toast in app.query(Toast) if toast.display]
            assert live, "no toast was raised — the park has nothing to withdraw"

            await _switch(app, pilot, busy)
            for _ in range(10):
                await pilot.pause()

            # ON the session switched TO: nothing of the parked conversation's
            # interruption is left standing over it. Counted from `display`
            # rather than from the owner, because what the user sees is a card
            # on screen; a hidden toast still holding a tag is invisible.
            assert (
                len([toast for toast in app.query(Toast) if toast.display]) == 0
            ), "the parked conversation's toast followed the user onto another session"

            await _switch(app, pilot, home)
            for _ in range(10):
                await pilot.pause()

            # And the row the toast was raised for is still there, past the
            # repaint that exposes a late loss.
            assert app._splash_notice == warning
            assert app._welcome is not None
            app._welcome.refresh_info()
            for _ in range(5):
                await pilot.pause()
            assert (
                app._welcome._info.notice == warning
            ), "withdrawing the toast cost the splash notice it was raised for"
            rendered = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
            assert warning in rendered, "the notice is on the app but not on the painted splash"
