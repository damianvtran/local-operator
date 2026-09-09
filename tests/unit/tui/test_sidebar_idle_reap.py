"""A parked viewer releases its runtime on a clock, but only when truly idle.

The defect these tests pin: ``_sidebar_source_releasable`` refused to release
any source the presentation LRU was holding, so a widget cache sized for
rendering (``RETAINED_PRESENTATIONS = 12``) became a *process* working set. One
reporting host held 15 attach sockets and 15 runtime children (~1.9 GB RSS)
after merely clicking through conversations, every genuine retention term
false.

The fix is a time bound rather than deleting that clause, because deleting it
reintroduces the cold-switch cost ``RETAINED_PRESENTATIONS`` was raised 4->12 to
remove. So there are two properties here and they pull in opposite directions:

* a source parked LONGER than the window is released (the leak);
* a source parked SHORTER than the window keeps its socket AND its presentation
  (the performance property that made the timed design worth building).

The second is a regression guard in the strict sense: an implementation that
simply drops the LRU clause passes every leak test in this file and fails that
one.

Assertions are STRUCTURAL — releasable booleans, map membership, dispose call
counts — and the clock is injected, never slept on, per AGENTS.md
Section "Timing, flakes".
"""

from __future__ import annotations

import os
from unittest.mock import Mock

import pytest

from local_operator.tui import app as app_module
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


class _Clock:
    """A monotonic clock the test advances explicitly.

    The sweep's whole contract is "how long has this been parked", so a real
    sleep would be both slow and the only flaky thing in the file.
    """

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def app_with_clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(app_module.time, "monotonic", clock)
    current = FakeSession()
    app = OperatorApp(lambda: _factory(current))
    app._session = current
    app._interaction = SessionInteraction(current)
    app._interactions[id(current)] = app._interaction
    return app, clock


def _make_remote(app, session_id, clock, parked_for, *, history=4):
    """Build a parked source whose session passes the RemoteSession check.

    A real subclass rather than a mock: the predicate's first clause is an
    ``isinstance(..., RemoteSession)`` test, so a duck-typed stand-in would make
    every assertion here vacuously true.
    """
    from local_operator.session.remote import RemoteSession

    class _ParkedRemote(RemoteSession):
        def __init__(self) -> None:  # deliberately not RemoteSession.__init__
            self._session_id = session_id
            self.disposed = 0

        @property
        def session_id(self) -> str:
            return self._session_id

        @property
        def has_pending_gate_reply(self) -> bool:
            return False

        @property
        def history_message_count(self) -> int:
            return history

        async def dispose(self) -> None:
            self.disposed += 1

    session = _ParkedRemote()
    source = SessionInteraction(session)
    source.parked_at = clock.now - parked_for
    app._sidebar_sources[session_id] = source
    app._interactions[id(session)] = source
    return source


def test_source_parked_past_the_window_is_releasable(app_with_clock):
    """The leak itself, in the units the predicate speaks."""
    app, clock = app_with_clock
    source = _make_remote(app, "aged", clock, app_module.SIDEBAR_IDLE_RELEASE_S + 1)
    # PRECONDITION: it is the LRU clause alone that would refuse, so the test
    # cannot pass vacuously against an unrelated retention.
    assert not source.retained_for_local_work
    assert not source.retained_for_auto_work
    assert not source.preparations
    app._sidebar_presentations["aged"] = object()

    assert not app._sidebar_source_releasable(source, reason="idle")
    assert app._sidebar_source_releasable(source, reason="expired")


def test_a_recently_parked_source_keeps_its_socket_and_presentation(app_with_clock):
    """REGRESSION GUARD: the performance property the timed design protects.

    An implementation that just deletes the LRU clause passes every other test
    in this file and fails this one, which is the entire point of it.
    """
    app, clock = app_with_clock
    presentation = object()
    source = _make_remote(app, "fresh", clock, 1.0)
    app._sidebar_presentations["fresh"] = presentation
    released = []
    app.run_worker = lambda coro, **kw: released.append(coro) or coro.close()

    clock.advance(app_module.SIDEBAR_IDLE_RELEASE_S / 2)
    app._sweep_idle_sidebar_sources()

    assert not released, "a source parked under the window must not be released"
    assert app._sidebar_sources.get("fresh") is source, "its viewer must survive"
    assert app._sidebar_presentations.get("fresh") is presentation, (
        "its presentation must survive, or the switch-back is a cold rebuild "
        "and RETAINED_PRESENTATIONS bought nothing"
    )


def test_the_sweep_releases_an_expired_source_and_evicts_its_presentation(app_with_clock):
    """A released source mints a new token, so its presentation can never hit."""
    app, clock = app_with_clock
    _make_remote(app, "aged", clock, 0.0)
    app._sidebar_presentations["aged"] = object()
    released = []
    app.run_worker = lambda coro, **kw: released.append(coro) or coro.close()

    clock.advance(app_module.SIDEBAR_IDLE_RELEASE_S + 1)
    app._sweep_idle_sidebar_sources()

    assert released, "an expired source must be handed to a release worker"
    assert "aged" not in app._sidebar_presentations, (
        "the presentation must be evicted with the source: re-leasing mints a "
        "fresh token, so a retained presentation is dead weight"
    )


def test_the_current_session_is_never_swept(app_with_clock):
    """No deadline exists for the displayed conversation, by construction."""
    app, clock = app_with_clock
    released = []
    app.run_worker = lambda coro, **kw: released.append(coro) or coro.close()
    # The current source carries no park stamp at all; that absence, not a
    # clause someone must remember, is what makes this structural.
    assert app._interaction.parked_at is None

    clock.advance(app_module.SIDEBAR_IDLE_RELEASE_S * 10)
    app._sweep_idle_sidebar_sources()

    assert not released


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("preparations", 1),
        ("retired", True),
    ],
)
def test_the_sweep_fails_closed_on_viewer_retention(app_with_clock, attribute, value):
    """Doubt keeps the source: a wrong keep costs one window, a wrong release
    costs the user a cold rebuild — and `preparations` is what stops a source
    being disposed while a prepare is in flight for it."""
    app, clock = app_with_clock
    source = _make_remote(app, "busy", clock, 0.0)
    setattr(source, attribute, value)
    released = []
    app.run_worker = lambda coro, **kw: released.append(coro) or coro.close()

    clock.advance(app_module.SIDEBAR_IDLE_RELEASE_S + 1)
    app._sweep_idle_sidebar_sources()

    assert not released


def test_the_sweep_suppresses_prewarm_for_what_it_just_released(app_with_clock):
    """Otherwise the 2 s prewarm poll re-leases it: a spawn treadmill.

    Prewarm's candidate filter selects entries with no retained presentation —
    exactly what the sweep produces — so without this damper a 5-minute reap
    becomes a 2-second respawn.
    """
    app, clock = app_with_clock
    _make_remote(app, "aged", clock, 0.0)
    app._sidebar_presentations["aged"] = object()
    app.run_worker = lambda coro, **kw: coro.close()

    clock.advance(app_module.SIDEBAR_IDLE_RELEASE_S + 1)
    app._sweep_idle_sidebar_sources()

    assert app._sidebar_prewarm_refused(
        "aged"
    ), "a swept source must not be immediately re-prepared by prewarm"


def test_an_empty_parked_source_still_waits_out_the_clock(app_with_clock):
    """A parked `/new` is REACHABLE, so emptiness is not a licence to dispose.

    An earlier revision of this fix released an apparently-empty source at park
    time, reasoning that there is no transcript to rebuild. That broke a shipped
    flow — `/new`, click a session with history, click back — because the user
    can still return to the empty conversation, and disposing it at park killed
    the source behind it (caught by
    ``test_sidebar_swap_reset.py::test_returning_to_an_empty_conversation_...``).
    Emptiness changes what the RUNTIME does when the clock finally expires (it
    accepts the retire offer instead of waiting for the drain), never when the
    viewer lets go.
    """
    app, clock = app_with_clock
    source = _make_remote(app, "untitled", clock, 0.0, history=0)
    # PRECONDITION: genuinely empty, so the claim is about emptiness rather
    # than about some other term happening to keep it.
    assert getattr(source.session, "history_message_count", None) == 0
    released = []
    app.run_worker = lambda coro, **kw: released.append(coro) or coro.close()

    app._sweep_idle_sidebar_sources()

    assert not released, "an empty conversation is still clickable while parked"
    assert app._sidebar_sources.get("untitled") is source


def test_the_sweep_never_raises_into_the_timer(app_with_clock, monkeypatch):
    """A Textual interval that throws stops repeating, disabling reaping."""
    app, clock = app_with_clock
    _make_remote(app, "aged", clock, 0.0)
    monkeypatch.setattr(
        app, "_sidebar_source_releasable", Mock(side_effect=RuntimeError("probe exploded"))
    )

    clock.advance(app_module.SIDEBAR_IDLE_RELEASE_S + 1)
    app._sweep_idle_sidebar_sources()  # must not raise
