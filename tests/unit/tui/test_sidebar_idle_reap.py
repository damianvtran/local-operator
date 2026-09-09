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

WHICH TEST CATCHES THE "just delete the LRU clause" MUTATION:
``test_source_parked_past_the_window_is_releasable``, and only it. Running that
mutation (replacing the clause at ``_sidebar_source_releasable`` with ``True``)
fails exactly that test and leaves the other eight green — measured, review
round 1 MINOR-3, not assumed. It is the one that asserts the ``"idle"`` refusal
AND the ``"expired"`` release on the same source, so deleting the clause breaks
the refusal half.

``test_a_recently_parked_source_keeps_its_socket_and_presentation`` pins the
TIME BOUND instead, and cannot catch that mutation: it parks 150 s into a 300 s
window, so the sweep hits ``now - parked_at < deadline`` and skips the source
before the predicate is consulted at all. Do not delete the first test on the
belief that the second covers it — that trade loses the only guard there is.

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


def _make_remote(app, session_id, clock, parked_for, *, history=4, poison=False):
    """Build a parked source whose session passes the RemoteSession check.

    A real subclass rather than a mock: the predicate's first clause is an
    ``isinstance(..., RemoteSession)`` test, so a duck-typed stand-in would make
    every assertion here vacuously true.

    ``poison`` makes ``frontend_state`` raise the way the real property does
    before its store syncs (``remote.py``), which is the reachable way a probe
    can throw — the predicate reads it through ``retained_for_auto_work``.
    Declared on the class rather than assigned afterwards so the raise is part
    of the type, as it is in production.
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

        @property
        def frontend_state(self):
            if poison:
                raise RuntimeError("frontend state has not synchronized")
            return super().frontend_state

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

    This pins THE TIME BOUND ITSELF — that a sub-window source is skipped on
    the clock, keeping both its socket and its presentation, which is the whole
    reason the fix is a deadline rather than a deletion of the LRU clause.

    It does NOT catch the "just delete the LRU clause" mutation, and an earlier
    version of this docstring wrongly claimed it did (review round 1 MINOR-3).
    It cannot: parking at half the window means the sweep returns on
    ``now - parked_at < deadline`` before the predicate is ever consulted.
    ``test_source_parked_past_the_window_is_releasable`` is the test that fails
    under that mutation.
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


def test_one_raising_source_does_not_starve_the_healthy_ones_behind_it(app_with_clock):
    """PER-SOURCE fail-closed, not per-tick — the difference is permanent leak.

    `_sidebar_sources` iterates in insertion order, so a `try` at loop scope
    abandons the same suffix of the map on EVERY tick, not just this one. The
    sweep is the only reaper, so a single bad probe ordered first would restore
    the original leak for every source behind it, forever (review round 1
    MINOR-1: 21 consecutive ticks released 0 of 3 healthy sources).

    The raise is planted on `frontend_state`, which is the reachable shape:
    `retained_for_auto_work` reads it via `getattr` BEFORE the predicate's
    `isinstance(..., RemoteSession)` guard, and `getattr` does not swallow an
    exception raised by a property — only a missing attribute.
    """
    app, clock = app_with_clock
    bad = _make_remote(app, "poisoned", clock, 0.0, poison=True)
    # PRECONDITION: the probe really does raise, so the test cannot pass
    # because nothing ever threw.
    with pytest.raises(RuntimeError):
        _ = bad.retained_for_auto_work
    healthy = [_make_remote(app, f"healthy{index}", clock, 0.0) for index in range(3)]
    # PRECONDITION: the bad source really is first, so a pass cannot come from
    # the raiser happening to sort last.
    assert list(app._sidebar_sources) == ["poisoned", "healthy0", "healthy1", "healthy2"]
    released = []
    app.run_worker = lambda coro, **kw: released.append(coro) or coro.close()

    clock.advance(app_module.SIDEBAR_IDLE_RELEASE_S + 1)
    app._sweep_idle_sidebar_sources()  # must not raise

    assert len(released) == len(healthy), (
        "a source whose probe raises must cost only itself; every healthy "
        "source ordered after it still has to be released"
    )
    assert app._sidebar_sources.get("poisoned") is bad, "a failed probe is a KEEP"


def test_the_runtime_retire_offer_is_bounded(monkeypatch):
    """A wedged runtime must not hold the closing socket for `ACK_TIMEOUT_S`.

    The offer crosses the socket and `dispose()` closes that socket, so the
    caller cannot proceed until it answers. Unbounded, a sidebar close would
    wait 15 s per source against a runtime that never replies. A timeout is
    just another "kept": the runtime's residency drain collects it.

    Deliberately NOT on the `app_with_clock` fixture: that fixture freezes
    `time.monotonic`, which is the clock the asyncio event loop schedules
    timeouts on, so a real `wait_for` inside it can never fire. This one needs
    a real loop, so it takes a bare instance instead of the injected clock.
    """
    import asyncio

    # PRECONDITION: 3 s is the number, matching `mobile/daemon.py`'s bound on
    # the identical call. Compressed below only so the test costs milliseconds
    # instead of the production bound.
    assert app_module.RETIRE_OFFER_TIMEOUT_S == 3.0
    monkeypatch.setattr(app_module, "RETIRE_OFFER_TIMEOUT_S", 0.05)

    # No Textual app is booted: the method touches nothing but its argument,
    # and booting one would drag the whole harness into a timing assertion.
    app = OperatorApp.__new__(OperatorApp)
    started: list[bool] = []

    async def _wedged() -> str:
        started.append(True)
        await asyncio.sleep(300)  # a runtime that never answers
        raise AssertionError("unreachable: the bound must fire first")

    session = Mock()
    session.retire_if_unused = _wedged

    async def _drive() -> float:
        loop = asyncio.get_running_loop()
        begin = loop.time()
        # The OUTER bound is the test's own failure mode, not the assertion:
        # without the bound under test this await never returns, and a hang is
        # a far worse test failure than a raise. `CancelledError` is a
        # `BaseException`, so the method's `except Exception` cannot eat it.
        await asyncio.wait_for(OperatorApp._retire_unused_runtime(app, session), timeout=2.0)
        return loop.time() - begin

    # Must RETURN (a timeout is a "kept"), never raise into teardown.
    elapsed = asyncio.run(_drive())

    assert started, "the offer must actually have been made"
    assert elapsed < 1.0, (
        "an unanswered offer must be abandoned on the bound, not held for the "
        f"AttachClient default of 15 s (took {elapsed:.2f}s)"
    )


def test_the_sweep_re_reads_the_window_constant_every_tick(app_with_clock, monkeypatch):
    """The only way to reach this path in a test is patching the constant.

    `SIDEBAR_IDLE_RELEASE_S` is deliberately not an env knob (see its
    docstring), so every test here — and the two e2e repros — compress the
    window by setting the module attribute and relying on the sweep reading it
    per tick. Hoisting it into a local or binding it at `on_mount` would leave
    all of them green against a five-minute wait they never reach, and the
    coverage would evaporate silently. QA round 1, Q1: this pins it.
    """
    app, clock = app_with_clock
    _make_remote(app, "aged", clock, 0.0)
    released = []
    app.run_worker = lambda coro, **kw: released.append(coro) or coro.close()

    clock.advance(10.0)
    monkeypatch.setattr(app_module, "SIDEBAR_IDLE_RELEASE_S", 1_000_000.0)
    app._sweep_idle_sidebar_sources()
    assert not released, "a window widened after startup must be honoured"

    monkeypatch.setattr(app_module, "SIDEBAR_IDLE_RELEASE_S", 1.0)
    app._sweep_idle_sidebar_sources()
    assert released, (
        "a window narrowed after startup must be honoured too: the sweep reads "
        "the module attribute per tick, which is what makes the e2e repros real"
    )
