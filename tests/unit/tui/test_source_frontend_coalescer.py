"""A burst of owner deltas must cost ONE source callback, not one per delta.

``OperatorApp._watch_source_frontend`` installs a per-source subscription whose
callback scheduled ``self.call_later(self._source_frontend_changed, source)`` on
EVERY canonical delta — there was no coalescer on this path, unlike the current
session's own subscription (``_on_frontend_update``), which has had a
scheduled-bit guard since it was measured repeating whole-roster work per update
in a coalesced burst. A burst here cost N timers and N retention predicates, each
of them a full release decision about the source, for one answer.

These assertions are STRUCTURAL — how many callbacks ran, and which source they
belonged to — per AGENTS.md "Timing, flakes". The app is driven through the real
pilot, so ``call_later`` is the real Textual scheduler rather than a stand-in.
"""

from __future__ import annotations

import os

import pytest

from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    # An inherited CMUX_* variable lets a headless app rename the operator's
    # real workspaces; HOME is redirected too because the cache root is derived
    # from it independently of LOCAL_OPERATOR_CONFIG_DIR.
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(OperatorApp, "_start_update_check", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_terminal_title", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_multiplexer_broadcast", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_herdr_reporter", lambda _self: None)


def _recording(sink: list[object]):
    """Stand in for ``run_worker``: record the coroutine and close it unwritten."""

    def run(coro, **_kwargs):  # type: ignore[no-untyped-def]
        sink.append(coro)
        coro.close()
        return None

    return run


class _ViewerSource:
    """A leased source: a real ``AttachedSession`` facade with a real store.

    A real subclass rather than a mock because ``_sidebar_source_releasable``
    starts with an ``isinstance(..., AttachedSession)`` test (through
    ``_is_viewer``), so a duck-typed stand-in would make the release half of
    these tests vacuous.
    """

    def __init__(self, session_id: str) -> None:
        from local_operator.session.attached import AttachedSession

        class _Facade(AttachedSession):
            def __init__(self) -> None:  # deliberately not AttachedSession.__init__
                self._session_id = session_id
                self._frontend_store = FrontendStateStore(
                    FrontendSessionState(session_id=session_id, epoch="e1")
                )
                self.handlers: list[object] = []

            @property
            def session_id(self) -> str:
                return self._session_id

            @property
            def has_pending_gate_reply(self) -> bool:
                return False

            def subscribe_frontend(self, handler):  # type: ignore[no-untyped-def]
                self.handlers.append(handler)

                class _Subscription:
                    unwatch = False
                    unsubscribe = staticmethod(lambda: None)

                return _Subscription()

        self.session = _Facade()
        self.source = SessionInteraction(self.session)

    def deliver(self, count: int) -> None:
        """Fire the subscription the way the store does: once per delta."""
        for index in range(count):
            self.session.handlers[0](object())  # type: ignore[operator]


@pytest.mark.asyncio
async def test_a_burst_of_deltas_schedules_one_callback() -> None:
    """N deltas in one loop turn: one timer, one release decision."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    wired = _ViewerSource("child-0")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._watch_source_frontend(wired.source)
        calls: list[SessionInteraction] = []
        app._source_frontend_changed = calls.append  # type: ignore[method-assign]

        wired.deliver(5)
        assert len(calls) == 0, "the callback ran inline rather than off the pump"
        await pilot.pause()

        assert calls == [wired.source], "a burst of five deltas cost more than one callback"


@pytest.mark.asyncio
async def test_a_later_delta_after_the_callback_schedules_again() -> None:
    """The bit must be cleared by the callback itself, or the source goes deaf."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    wired = _ViewerSource("child-0")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._watch_source_frontend(wired.source)
        calls: list[SessionInteraction] = []
        app._source_frontend_changed = calls.append  # type: ignore[method-assign]

        wired.deliver(2)
        await pilot.pause()
        assert len(calls) == 1
        assert wired.source.frontend_change_scheduled is False, "the scheduled bit stuck"

        wired.deliver(2)
        await pilot.pause()
        assert len(calls) == 2, "the source stopped reacting after one coalesced burst"


@pytest.mark.asyncio
async def test_a_retired_source_still_clears_its_bit_and_acts_on_nothing() -> None:
    """Retirement is decided by the real callback, and the bit is not its casualty."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    wired = _ViewerSource("child-0")
    releases: list[object] = []
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._watch_source_frontend(wired.source)
        app.run_worker = _recording(releases)  # type: ignore[method-assign]
        wired.source.retired = True
        app._sidebar_sources["child-0"] = wired.source

        wired.deliver(3)
        await pilot.pause()

        assert not releases, "a retired source was handed to the release worker"
        assert wired.source.frontend_change_scheduled is False

        wired.deliver(1)
        await pilot.pause()
        assert not releases


@pytest.mark.asyncio
async def test_a_superseded_source_does_not_act() -> None:
    """Another source is bound to the session id: this callback is not wanted."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    stale = _ViewerSource("child-0")
    live = _ViewerSource("child-0")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._watch_source_frontend(stale.source)
        calls: list[SessionInteraction] = []
        app._source_frontend_changed = calls.append  # type: ignore[method-assign]
        app._sidebar_sources["child-0"] = live.source

        stale.deliver(1)
        await pilot.pause()

        assert calls == [], "a superseded interaction acted on its owner's delta"
        assert stale.source.frontend_change_scheduled is False


@pytest.mark.asyncio
async def test_a_source_with_no_session_id_is_still_delivered() -> None:
    """No row to compare against is not evidence of supersession.

    Dropping on an absent row would strand such a source's gate draft and its
    release decision for the life of the app.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    wired = _ViewerSource("")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._watch_source_frontend(wired.source)
        calls: list[SessionInteraction] = []
        app._source_frontend_changed = calls.append  # type: ignore[method-assign]

        wired.deliver(1)
        await pilot.pause()

        assert calls == [wired.source]


@pytest.mark.asyncio
async def test_the_direct_call_sites_are_not_coalesced() -> None:
    """Subagent events and the close drain keep their own cadence.

    Those sites call ``_source_frontend_changed`` directly, so the coalescer must
    live on the SUBSCRIPTION rather than inside the callback: a source with a
    queued callback would otherwise swallow an event-driven change. What this
    pins is that the callback itself carries no gate.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    wired = _ViewerSource("child-0")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._watch_source_frontend(wired.source)
        calls: list[SessionInteraction] = []
        app._source_frontend_changed = calls.append  # type: ignore[method-assign]
        app._sidebar_sources["child-0"] = wired.source

        wired.deliver(1)  # a queued coalesced callback is now pending
        app._source_frontend_changed(wired.source)  # what a direct site invokes
        assert len(calls) == 1, "the callback itself was gated"
        await pilot.pause()

        assert len(calls) == 2, "a direct call site was gated by the coalescer"
