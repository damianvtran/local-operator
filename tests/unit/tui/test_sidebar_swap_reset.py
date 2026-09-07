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

    def __init__(self, session_id: str, *, history=(), cost=None, context=None) -> None:
        super().__init__()
        self._id = session_id
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
