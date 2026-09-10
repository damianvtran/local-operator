"""A sidebar connect never publishes a session it did not actually bind.

THE DEFECT THESE PIN. ``RemoteSession._ensure_bound`` has three outcomes and
only two of them are distinguishable at the call site: it binds, it raises, or
— while the facade is ``_recovering`` after an owner loss — it returns
SILENTLY, without binding, leaving the session cold. ``_connect_sidebar_source``
read that third outcome as success. What followed depended only on whether the
display window happened to be invalidated:

* invalidated: the next call raised ``ConnectionError`` and the bare
  ``except Exception`` latched a terminal failure, so a condition that clears
  itself in under ``COLD_FALLBACK_S`` seconds became a dead session only a
  manual reselect could revive;
* not invalidated: nothing raised, the commit went through, and the user
  watched a live-looking transcript of a COLD session for 15 s — 1,799 refused
  frames, each requesting a full relayout — until ``_await_sidebar_frame``'s
  timer gave up and reverted to the failure view.

Every existing test of this path drives a facade that either binds or raises.
None covered the silent third outcome, which is exactly why it shipped. The
load-bearing assertion in this file is therefore the postcondition:
**after a connect commits, the session is not cold.**

WHY THE DOUBLE IS A REAL SUBCLASS. ``_connect_sidebar_source`` opens with an
``isinstance(session, RemoteSession)`` check and returns quietly for anything
else, so a duck-typed stand-in would make every assertion here vacuously true
by never reaching the code under test.

NO WALL-CLOCK ASSERTIONS. Per AGENTS.md "Timing, flakes": the retry budget is
asserted as a RELATIONSHIP between two constants and as a COUNT of attempts,
never as an elapsed duration, and the gate-spin guard asserts the structural
fact (the readiness gate is never armed for a bind that failed) rather than a
rate of refusals.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import Mock

import pytest

from local_operator.session.remote import COLD_FALLBACK_S, RemoteSession
from local_operator.tui import app as app_module
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    # An inherited CMUX_* variable lets a headless app rename the operator's
    # real multiplexer workspaces; HOME is redirected too because the cache
    # root is derived from it independently of LOCAL_OPERATOR_CONFIG_DIR.
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_update_check", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_terminal_title", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_multiplexer_broadcast", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_herdr_reporter", lambda _self: None)


class RecoveringRemote(RemoteSession):
    """A viewer that reproduces the ``_recovering`` silent return.

    ``_ensure_bound`` returns without raising and without clearing ``is_cold``,
    which is what the real method does at ``remote.py`` when the facade is
    mid-recovery. ``heals_after`` binds on the Nth call so the retry budget has
    something to succeed against, mirroring the real timeline where the facade
    stops being ``_recovering`` once ``COLD_FALLBACK_S`` elapses.

    Deliberately does NOT call ``RemoteSession.__init__``: constructing a real
    one dials a runtime. Only the surface the connect path reads is provided.
    """

    def __init__(self, session_id: str, *, heals_after: int | None = None) -> None:
        self._session_id = session_id
        self._cold = True
        self._heals_after = heals_after
        self.bind_calls = 0
        self.disposed = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_cold(self) -> bool:
        return self._cold

    # Both are read-only properties on the real class, so they are overridden
    # rather than assigned. "Current" is the state that makes the SILENT bind
    # failure reachable: an invalidated window would make the next call raise
    # instead, which is the symptom this defect's other face already produced.
    @property
    def display_history_current(self) -> bool:
        return True

    @property
    def display_history_revision(self) -> int:
        return 1

    async def _ensure_bound(self, *, foreground: bool = True) -> None:
        self.bind_calls += 1
        if self._heals_after is not None and self.bind_calls >= self._heals_after:
            self._cold = False

    async def ensure_display_current(self) -> None:
        return None

    async def dispose(self) -> None:
        # App teardown disposes every source it is holding. The real method
        # walks socket and waiter state this double never built, so it is
        # replaced rather than left to fail the unmount of every test here.
        self.disposed = True


class UnreachableRemote(RecoveringRemote):
    """A viewer that BINDS but whose owner is momentarily unreachable.

    The other face of the same disruption: ``_ensure_bound`` succeeds, and the
    display refresh is what fails (``ConnectionError("history owner is
    unavailable")``, raised when the client is gone and the window is stale).
    This is the shape that used to latch terminally with no auto-retry.
    """

    def __init__(self, session_id: str, *, heals_after: int | None = None) -> None:
        super().__init__(session_id, heals_after=None)
        self._cold = False
        self._display_heals_after = heals_after
        self.display_calls = 0

    async def _ensure_bound(self, *, foreground: bool = True) -> None:
        self.bind_calls += 1

    async def ensure_display_current(self) -> None:
        self.display_calls += 1
        heals = self._display_heals_after
        if heals is not None and self.display_calls >= heals:
            return None
        raise ConnectionError("history owner is unavailable")


async def _current_source(
    app: OperatorApp, pilot: Any, session: RemoteSession
) -> SessionInteraction:
    """Make ``session`` the app's current sidebar source, as a click would.

    ``display_only`` starts True because that is the state the connect task is
    always entered from: the saved excerpt is on screen and the socket work has
    not happened yet.

    The pilot is pumped to completion FIRST because boot installs its own
    interaction asynchronously: a source assigned before that lands is replaced
    by the boot session's, `_is_current` goes False, and the retry branch —
    which is conditioned on the source still being on screen — silently stops
    being exercised. That produced a test that passed while asserting nothing.
    """
    for _ in range(60):
        await pilot.pause()
    source = SessionInteraction(session)
    source.display_only = True
    app._interaction = source
    app._interactions[id(session)] = source
    app._sidebar_sources[session.session_id] = source
    return source


async def _drain_retries(app: OperatorApp, source: SessionInteraction, *, rounds: int = 40) -> None:
    """Await the whole retry chain, round by round, as the app drives it.

    Each round is a NEW task: the `finally` block re-arms through `call_soon`,
    so following the chain means awaiting the current task, yielding twice to
    let the callback run, and picking up whatever task it installed. Awaiting
    the FIRST task alone would only ever see attempt one.

    Awaited rather than pumped with `pilot.pause()`, which costs ~1 s a turn
    here and made an earlier version of this file take eight minutes. `rounds`
    is a runaway backstop, not a budget: a chain that has settled breaks out
    immediately, and one that has not is the bug.
    """
    for _ in range(rounds):
        task = source.connection_task
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            return
        # Two turns: one for the `call_soon` re-arm to run, one for the task it
        # creates to be scheduled.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        if source.connection_task is task or source.connection_task is None:
            return
    raise AssertionError("the retry chain never settled")


def _instant_backoff(monkeypatch, *, attempts: int | None = None) -> None:
    """Collapse the backoff schedule so the retry policy is testable at speed.

    Only the WAIT is shortened. The attempt COUNT and the exhaustion behaviour
    are the properties under test, and the derivation that sizes the real count
    is asserted separately in
    ``test_the_retry_budget_outlasts_the_recovery_give_up_bound``.
    """
    monkeypatch.setattr(app_module, "SIDEBAR_CONNECT_BACKOFF_S", 0.0)
    monkeypatch.setattr(app_module, "SIDEBAR_CONNECT_BACKOFF_CEILING_S", 0.0)
    if attempts is not None:
        monkeypatch.setattr(app_module, "SIDEBAR_CONNECT_ATTEMPTS", attempts)


@pytest.mark.asyncio
async def test_a_bind_that_did_not_bind_is_not_committed(monkeypatch):
    """THE ROOT CAUSE, in one assertion: a cold session is never published.

    Fails on the pre-fix tree, where ``_ensure_bound``'s silent return was read
    as success and the commit ran against ``is_cold=True``.
    """
    session = RecoveringRemote("recovering")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        # RECORDED, never raised. An earlier version of this test stubbed both
        # with `side_effect=AssertionError(...)` and passed against a tree with
        # the fix reverted: the retry's `except Exception` catches an
        # AssertionError like any other, so the test asserted nothing. Mutation
        # testing found it; the collaborators now record and the assertions are
        # made afterwards, out of reach of the handler.
        reached: list[bool] = []
        prepared: Any = (source, object())

        async def prepare(*_args: Any, **_kwargs: Any) -> Any:
            reached.append(session.is_cold)
            return prepared

        commit = Mock(return_value=None)
        monkeypatch.setattr(app, "_prepare_sidebar_session", prepare)
        monkeypatch.setattr(app, "_commit_sidebar_session", commit)
        _instant_backoff(monkeypatch, attempts=2)
        # Stubbed so this test observes exactly ONE attempt: the re-arm is a
        # separate property, asserted in the retry tests below, and letting the
        # chain run here would obscure which attempt refused to commit.
        rearmed = Mock()
        monkeypatch.setattr(app, "_start_sidebar_connection", rearmed)

        await app._connect_sidebar_source(source)
        await asyncio.sleep(0)  # the re-arm is a `call_soon` callback

        assert session.bind_calls == 1
        # THE POSTCONDITION. Nothing downstream of the bind ran at all, so no
        # cold session could reach a commit; `reached` would carry a True on
        # the pre-fix tree, where preparation proceeded against `is_cold`.
        assert reached == []
        commit.assert_not_called()
        assert source.display_only is True
        # The failure is not the end of the story — it re-arms rather than
        # latching — but nothing was published in the meantime.
        rearmed.assert_called_once_with(source)


@pytest.mark.asyncio
async def test_no_frame_ever_shows_a_cold_session_as_connected(monkeypatch):
    """The invariant the false-connected window violated, sampled per loop turn.

    Every state the user can see is published between awaits, so sampling on
    each turn of the loop for the life of the connect task covers the whole
    window without measuring how long it was. States the pre-fix code passed
    through — ``display_only=False`` while ``is_cold=True`` — are what the
    assertion excludes.
    """
    session = RecoveringRemote("recovering", heals_after=3)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        prepared: Any = (source, object())
        monkeypatch.setattr(app, "_prepare_sidebar_session", _always(prepared))
        monkeypatch.setattr(app, "_commit_sidebar_session", Mock(return_value=None))
        _instant_backoff(monkeypatch)

        observed: list[tuple[bool, bool]] = []
        task = asyncio.ensure_future(app._connect_sidebar_source(source))
        while not task.done():
            observed.append((source.display_only, session.is_cold))
            await asyncio.sleep(0)
        observed.append((source.display_only, session.is_cold))
        await task
        await pilot.pause()

        assert observed, "the watcher never sampled the connect task"
        assert not [
            state for state in observed if state == (False, True)
        ], "a cold session was published as connected"
        # The connect still SUCCEEDS once the facade heals: the invariant must
        # not be satisfied by simply never connecting.
        assert source.display_only is False
        assert session.is_cold is False


@pytest.mark.asyncio
async def test_a_transient_failure_retries_instead_of_latching(monkeypatch):
    """A momentarily unreachable owner heals with no user action at all.

    The asymmetry this corrects: ``PreparationInvalidated`` — a purely LOCAL
    staleness race — already retried, while a transient owner unreachability
    did not. Both are transient; only exhaustion is terminal.
    """
    session = UnreachableRemote("blipping", heals_after=3)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        prepared: Any = (source, object())
        monkeypatch.setattr(app, "_prepare_sidebar_session", _always(prepared))
        monkeypatch.setattr(app, "_commit_sidebar_session", Mock(return_value=None))
        _instant_backoff(monkeypatch)

        # The production re-arm goes through `call_soon`; drive it the same way
        # the app does rather than calling the coroutine in a loop, so the
        # `finally` block's re-arm is what is under test.
        app._start_sidebar_connection(source)
        await _drain_retries(app, source)

        assert session.display_calls == 3, "the connect did not retry the transient failure"
        assert source.display_only is False
        assert source.connection_error == ""
        # A successful commit refills the budget for the NEXT disruption.
        assert source.connect_attempts == 0


@pytest.mark.asyncio
async def test_the_status_stays_on_connecting_while_the_retry_is_live(monkeypatch):
    """Mid-retry the app has not given up, so it must not tell the user it has.

    ``connection_error`` is what flips the status from ``Saved · Connecting…``
    to ``Saved · Connection unavailable · Reselect to retry``. Asking the user
    to act while the app is still working asks them to fix something that is
    fixing itself.
    """
    session = RecoveringRemote("recovering")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        _instant_backoff(monkeypatch, attempts=4)
        # One attempt only: the status under test is the one shown BETWEEN
        # attempts, which a chain run to exhaustion would immediately overwrite
        # with the terminal copy.
        monkeypatch.setattr(app, "_start_sidebar_connection", Mock())

        await app._connect_sidebar_source(source)

        assert source.connect_attempts == 1
        assert source.connection_error == ""
        assert app._status is not None  # the band exists from on_mount
        status = app._status.render_text(120).plain
        assert "Connecting" in status
        assert "Reselect" not in status


@pytest.mark.asyncio
async def test_only_exhaustion_latches_and_it_says_so_honestly(monkeypatch):
    """A genuinely unreachable owner still reaches the user — after the budget.

    The terminal state is what the retry is FOR, not an alternative to it, so
    the exhausted case must still land on the existing failure copy.
    """
    session = RecoveringRemote("gone")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        _instant_backoff(monkeypatch, attempts=3)

        app._start_sidebar_connection(source)
        await _drain_retries(app, source)

        assert session.bind_calls == app_module.SIDEBAR_CONNECT_ATTEMPTS + 1
        assert source.display_only is True
        assert source.connection_error == "the runtime is not responding"
        assert app._status is not None
        status = app._status.render_text(120).plain
        assert "Connection unavailable" in status
        assert "Reselect" in status
        # Surrendering hands the user a FULL budget, not one last attempt.
        assert source.connect_attempts == 0


def test_the_retry_budget_outlasts_the_recovery_give_up_bound():
    """THE RELATIONSHIP, not the number — the way this fix regresses quietly.

    A viewer that lost its owner refuses to bind for up to ``COLD_FALLBACK_S``
    and says so by returning silently, so every attempt inside that window
    fails identically and healing is only possible on an attempt that lands
    AFTER it. Measured during the investigation: a ~4 s total span did not heal
    an evicted viewer; a span past ``COLD_FALLBACK_S`` healed it at t=9.9 s.

    Pinned as an inequality between the two constants because that is the
    property. Asserting either number instead would go green on a tree where
    ``COLD_FALLBACK_S`` had grown past the budget — which is the original bug,
    silently restored.
    """
    schedule = [
        app_module.sidebar_connect_backoff_s(attempt)
        for attempt in range(1, app_module.SIDEBAR_CONNECT_ATTEMPTS + 1)
    ]
    assert sum(schedule) > COLD_FALLBACK_S
    # And it is not merely wider by a rounding error: the last attempt has to
    # land clearly past the recovery bound on a loaded machine, not tie with it.
    assert sum(schedule) > COLD_FALLBACK_S * 1.25
    # One attempt fewer must NOT clear the bound, which is what makes the
    # derived count minimal rather than an arbitrary large number.
    assert sum(schedule[:-1]) <= COLD_FALLBACK_S * 1.5


@pytest.mark.asyncio
async def test_a_failed_connect_never_arms_the_readiness_gate(monkeypatch):
    """NO GATE SPIN, in #856's own counters.

    ``_sidebar_gate_surface_ready``'s first check is ``is_cold``, so a cold
    session committed as connected made the gate STRUCTURALLY unsatisfiable:
    ``post_display_hook``'s recovery branch bought a full-screen relayout for
    it on every frame until the 15 s timer fired. Measured on this tree —
    post-#856, both halves re-run here — **1,268 refusals, every one of them a
    recovery**, against 0 after the fix.

    #856 is what makes this assertion sharp rather than merely small. Its
    arming refresh dirties the whole screen region, so the first post-commit
    paint is a full ``LayoutUpdate`` the gate passes on, and
    ``_sidebar_gate_recoveries`` is expected to be ZERO on a healthy switch —
    its own test asserts the branch stays dead, and its comment says a firing
    recovery is now a signal. A failed connect must not resurrect it.

    ``_sidebar_gate_reached`` is asserted alongside for the reason #856's own
    comment gives: a zero meaning "the gate was never armed" reads identically
    to one meaning "the gate never refused", and only the second is worth
    pinning. Refusing to commit means it is never armed, so both are zero here
    — and the direct spy proves the stronger fact that the predicate is not
    consulted for this source even once.

    Facts, not rates, per AGENTS.md: counter deltas and an unarmed ready-frame.
    """
    session = RecoveringRemote("recovering")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        _instant_backoff(monkeypatch, attempts=2)

        refusals: list[bool] = []
        original = OperatorApp._sidebar_gate_surface_ready

        def counting(self, candidate):
            result = original(self, candidate)
            if candidate is source:
                refusals.append(result)
            return result

        monkeypatch.setattr(OperatorApp, "_sidebar_gate_surface_ready", counting)
        reached_before = app._sidebar_gate_reached
        recoveries_before = app._sidebar_gate_recoveries

        await app._connect_sidebar_source(source)
        for _ in range(20):
            await pilot.pause()

        assert refusals == []
        assert app._sidebar_gate_recoveries - recoveries_before == 0
        assert app._sidebar_gate_reached - reached_before == 0
        assert app._sidebar_ready_frame is None


@pytest.mark.asyncio
async def test_navigating_away_mid_retry_stops_the_retry(monkeypatch):
    """The backoff keeps the task alive ~10 s; a hidden source must not hold it.

    A live connection task counts as ``retained_for_local_work``, so retrying a
    source the user has navigated away from would strand an evicted hidden
    viewer for the whole budget instead of releasing it. The retry is therefore
    conditioned on the source still being current.
    """
    session = RecoveringRemote("recovering")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        _instant_backoff(monkeypatch, attempts=6)
        # The user switched away before the connect task ran.
        app._interaction = SessionInteraction(FakeSession())

        await app._connect_sidebar_source(source)
        await pilot.pause()

        assert session.bind_calls == 1, "a hidden source kept retrying"
        assert source.connection_error == "the runtime is not responding"


@pytest.mark.asyncio
async def test_cancelling_during_the_backoff_is_a_cancellation_not_a_failure(monkeypatch):
    """``CancelledError`` must survive the grown ``except`` body.

    The retry adds an ``await`` inside the exception handler, which is a new
    place a cancellation can land. It has to propagate — and suppress the
    status write and the re-arm — exactly as a cancellation anywhere else in
    the body does, rather than being absorbed as one more failed attempt.
    """
    session = RecoveringRemote("recovering")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        # A real (non-zero) backoff, so the cancellation lands inside the sleep.
        monkeypatch.setattr(app_module, "SIDEBAR_CONNECT_BACKOFF_S", 30.0)
        monkeypatch.setattr(app_module, "SIDEBAR_CONNECT_BACKOFF_CEILING_S", 30.0)
        rearmed = Mock()
        monkeypatch.setattr(app, "_start_sidebar_connection", rearmed)

        task = asyncio.ensure_future(app._connect_sidebar_source(source))
        for _ in range(50):
            await pilot.pause()
            await asyncio.sleep(0)
            if session.bind_calls:
                break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await pilot.pause()

        assert task.cancelled()
        rearmed.assert_not_called()


@pytest.mark.asyncio
async def test_the_retry_releases_its_preparation_before_waiting(monkeypatch):
    """A ~10 s task must not hold prepared widgets across a wait nothing watches.

    The failure can arrive AFTER ``_prepare_sidebar_session`` has handed over a
    preparation — a commit that raises. Leaving it held for the backoff is a
    leak in all but name, and the ``finally`` block only runs once, at the end
    of the whole retry sequence.
    """
    session = UnreachableRemote("blipping")
    session._display_heals_after = None
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        prepared: Any = (source, object())
        monkeypatch.setattr(app, "_prepare_sidebar_session", _always(prepared))
        monkeypatch.setattr(
            app,
            "_commit_sidebar_session",
            Mock(side_effect=ConnectionError("history owner is unavailable")),
        )
        released: list[Any] = []

        async def release(item):
            released.append(item)

        monkeypatch.setattr(app, "_release_sidebar_preparation", release)
        # Bind and display succeed; the COMMIT is what fails, so a preparation
        # is in hand when the retry branch is reached.
        session.ensure_display_current = _noop  # type: ignore[method-assign]
        _instant_backoff(monkeypatch, attempts=3)

        app._start_sidebar_connection(source)
        await _drain_retries(app, source)

        assert source.connection_error
        # Once per attempt, not once at the end of the whole sequence.
        assert len(released) == app_module.SIDEBAR_CONNECT_ATTEMPTS + 1
        assert source.command_frame_pending is False


async def _noop() -> None:
    return None


def _always(value: Any) -> Any:
    """A coroutine-function stub returning ``value`` on EVERY call.

    A ``Mock(return_value=<coroutine>)`` cannot stand in: a coroutine object is
    single-use, so the second retry attempt would await an already-consumed one
    and fail with a ``RuntimeError`` that has nothing to do with the code under
    test. The retry is the whole point here, so the stub has to be re-callable.
    """

    async def prepare(*_args: Any, **_kwargs: Any) -> Any:
        return value

    return prepare
