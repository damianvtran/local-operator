"""A sidebar connect never publishes a session it did not actually bind.

THE DEFECT THESE PIN. ``AttachedSession._ensure_bound`` has three outcomes and
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
``isinstance(session, AttachedSession)`` check and returns quietly for anything
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

from local_operator.session.attached import COLD_FALLBACK_S, AttachedSession
from local_operator.tui import app as app_module
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from local_operator.tui.session_navigation import OwnerWentCold, SurfaceNotReady
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: The shipped schedule, captured at import BEFORE any test patches it for
#: speed. `_shipped_span_from` asserts against these rather than against the
#: live module attributes, because the property under test is what a user's
#: next connect actually gets.
_SHIPPED_BACKOFF_S = app_module.SIDEBAR_CONNECT_BACKOFF_S
_SHIPPED_CEILING_S = app_module.SIDEBAR_CONNECT_BACKOFF_CEILING_S
_SHIPPED_ATTEMPTS = app_module.SIDEBAR_CONNECT_ATTEMPTS


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


class RecoveringRemote(AttachedSession):
    """A viewer that reproduces the ``_recovering`` silent return.

    ``_ensure_bound`` returns without raising and without clearing ``is_cold``,
    which is what the real method does at ``attached.py`` when the facade is
    mid-recovery. ``heals_after`` binds on the Nth call so the retry budget has
    something to succeed against, mirroring the real timeline where the facade
    stops being ``_recovering`` once ``COLD_FALLBACK_S`` elapses.

    Deliberately does NOT call ``AttachedSession.__init__``: constructing a real
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


def _shipped_span_from(spent: int) -> float:
    """Total backoff still available after `spent` attempts, on the SHIPPED schedule.

    Deliberately reads the module's real constants rather than whatever a test
    has patched for speed: the property under test is what a user's next connect
    actually gets, which is a fact about the shipped numbers.
    """
    return sum(
        min(_SHIPPED_BACKOFF_S * (2 ** (attempt - 1)), _SHIPPED_CEILING_S)
        for attempt in range(spent + 1, _SHIPPED_ATTEMPTS + 1)
    )


async def _current_source(
    app: OperatorApp, pilot: Any, session: AttachedSession
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
        # latching — but nothing was published in the meantime. `continues_retry`
        # is what stops that re-arm refilling the budget it is spending; see
        # `test_the_retry_loops_own_rearm_still_spends_the_budget`.
        rearmed.assert_called_once_with(source, continues_retry=True)


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
        # The terminal copy now says the app already TRIED (design round D3):
        # "Connection unavailable" described a state, so the advice that
        # followed read as "try the thing I was already doing".
        assert "Reconnect failed" in status
        assert "Select again to retry" in status
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


class BindsThenLosesItsOwner(RecoveringRemote):
    """A viewer that BINDS, then loses its owner before its frame paints.

    The window the defect lives in, in one object: the bind postcondition
    passes and the owner disappears in the gap between that check and the first
    painted frame — which is where a real ``ATTACH_MAX_CLIENTS`` LRU eviction
    landed for the user (the architect's rig reproduced it at 15.1 s with 1,820
    refusals).

    ``heals_after`` is deliberately unused: this double never comes back, so
    the arms under test are the refusal and the BOUND, not the heal. The heal
    is asserted against real runtimes in
    ``tests/e2e/test_sidebar_reconnect_e2e.py``.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id, heals_after=None)
        self._cold = False

    async def _ensure_bound(self, *, foreground: bool = True) -> None:
        self.bind_calls += 1

    def go_cold(self) -> None:
        self._cold = True


async def _drain_with_paints(app: OperatorApp, pilot: Any, source: SessionInteraction) -> None:
    """``_drain_retries``, but pumping the pilot so the app actually PAINTS.

    A cold pending frame is decided by ``post_display_hook``, and that only runs
    when Textual renders. A drain that merely awaits the task leaves the
    compositor idle, so the readiness gate's own 15 s timer becomes the only
    thing that can settle the frame — which IS the pre-fix behaviour, and the
    reason every assertion below is on the exception TYPE and the counters
    rather than on how long anything took.

    ``wait_for`` is a runaway backstop, never an assertion: a chain that has
    settled breaks out immediately.
    """
    for _ in range(40):
        task = source.connection_task
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), 30)
        except asyncio.CancelledError:
            return
        await pilot.pause()
        if source.connection_task is task or source.connection_task is None:
            return
    raise AssertionError("the retry chain never settled")


def _cold_frame_rig(
    app: OperatorApp,
    source: SessionInteraction,
    session: Any,
    monkeypatch: Any,
    *,
    deferred: bool = True,
):
    """Arm a REAL frame, then lose the owner before it can paint.

    Returned as the recorder for what the frame finally settled with.

    ``deferred`` selects WHICH HALF of the window is exercised, and the two are
    not interchangeable — they are the two places this PR closes it:

    * ``True`` (the hook's half): the loss is scheduled with ``call_soon``, so
      it lands while the connect body is suspended on ``await ready`` and the
      next paint is what notices. Inline would be caught by the belt instead,
      because the belt is SYNCHRONOUS between ``_commit_sidebar_session``
      returning and its ``is_cold`` read — the test would then prove nothing
      about the hook.
    * ``False`` (the belt's half): the loss is applied inline, i.e. before the
      commit returns, which is what an owner lost during preparation looks
      like. The belt refuses it before the body ever waits.
    """
    outcomes: list[BaseException | None] = []
    loop = asyncio.get_running_loop()

    def commit(*_args: Any, **_kwargs: Any) -> Any:
        future = app._await_sidebar_frame(source, app._sidebar_navigation.generation)
        future.add_done_callback(lambda settled: outcomes.append(settled.exception()))
        if deferred:
            loop.call_soon(session.go_cold)
        else:
            session.go_cold()
        return future

    monkeypatch.setattr(app, "_commit_sidebar_session", commit)
    prepared: Any = (source, object())
    monkeypatch.setattr(app, "_prepare_sidebar_session", _always(prepared))
    return outcomes


@pytest.mark.asyncio
async def test_a_cold_frame_does_not_wait_out_the_readiness_gate(monkeypatch):
    """THE CORE FIX: a cold committed frame fails at once, and spends the budget.

    ``_sidebar_gate_surface_ready``'s first check is ``is_cold``, so for as long
    as the owner is gone the gate's verdict is already known. Waiting it out is
    not a longer check — it is 15 s of refusals, each buying a forced
    full-screen relayout (measured pre-fix: 1,820 of them, ~127/s), ending in
    ``SurfaceNotReady``, which #883 made terminal-on-first for the good reason
    that a PAINT failure should not be retried. That latched a TRANSIENT owner
    loss and told the user to reselect the session the reselect healed in 0.17 s.

    Asserted as the exception the frame settled with plus a COUNT of attempts —
    the defect is "the wrong arm decided this: the timer instead of the cold
    check, and no retry", which is a fact about types and counts, not durations.
    """
    monkeypatch_setup = monkeypatch
    session = BindsThenLosesItsOwner("cold-at-paint")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        outcomes = _cold_frame_rig(app, source, session, monkeypatch_setup)
        # A SHORT budget: the property is that this arm SPENDS it and latches
        # only on exhaustion, not the shipped length of it.
        _instant_backoff(monkeypatch, attempts=3)

        app._start_sidebar_connection(source)
        await _drain_with_paints(app, pilot, source)

        assert outcomes, "no frame was ever armed"
        assert isinstance(outcomes[0], OwnerWentCold), (
            f"the cold frame settled as {outcomes[0]!r}: a SurfaceNotReady means the "
            "gate's 15 s timer decided it, not the cold branch"
        )
        # THE BOUND HOLDS: attempts are spent per round, and only exhaustion
        # latches — one bind that succeeded, then one per refusing round.
        assert session.bind_calls == app_module.SIDEBAR_CONNECT_ATTEMPTS + 1
        assert source.display_only is True
        assert source.connection_error == "the runtime is not responding"
        # Surrendering hands the user a full budget for their reselect.
        assert source.connect_attempts == 0


@pytest.mark.asyncio
async def test_a_loss_during_preparation_is_refused_before_the_wait(monkeypatch):
    """THE BELT: an owner lost between the bind check and the commit.

    The commit->paint half is the hook's; this is the other half, and it is
    closed by re-reading ``is_cold`` once between the commit and the ``await``.
    Without it the frame is armed against a session that is already cold, and
    the only thing that can settle it is the gate's 15 s timer.

    Its signature is DISTINCT from the hook's, which is why both are pinned:
    the gate is consulted ZERO times here, because nothing was ever painted
    before the refusal, where the hook's half consults it exactly once.
    """
    session = BindsThenLosesItsOwner("cold-at-commit")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        outcomes = _cold_frame_rig(app, source, session, monkeypatch, deferred=False)
        _instant_backoff(monkeypatch, attempts=3)
        reached_before = app._sidebar_gate_reached

        app._start_sidebar_connection(source)
        await _drain_with_paints(app, pilot, source)

        assert outcomes, "no frame was ever armed"
        assert isinstance(outcomes[0], OwnerWentCold), (
            f"the cold commit settled as {outcomes[0]!r}: the frame was waited on "
            "instead of refused"
        )
        assert (
            app._sidebar_gate_reached - reached_before == 0
        ), "the gate was consulted for a commit the belt can already refuse"
        # Spends the budget like every other transient arm, and latches only on
        # exhaustion — the whole reason it is not routed at the terminal one.
        assert session.bind_calls == app_module.SIDEBAR_CONNECT_ATTEMPTS + 1
        assert source.display_only is True
        assert source.connection_error == "the runtime is not responding"


@pytest.mark.asyncio
async def test_a_cold_frame_buys_no_relayout(monkeypatch):
    """NO GATE SPIN, for the commit->paint window this time.

    #856's counters are the sharpest instrument here: ``_sidebar_gate_reached``
    "reached the gate" and ``_sidebar_gate_recoveries`` "bought a full-screen
    relayout chasing it". A cold frame that waits can only ever produce the
    second — the gate refuses on ``is_cold`` before it can pass — so the fix's
    signature is one consultation and ZERO recoveries, against 1,820 before it.

    Counts, not rates: a busy machine changes how long the frames take, not how
    many the gate refuses.
    """
    session = BindsThenLosesItsOwner("cold-at-paint")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        _cold_frame_rig(app, source, session, monkeypatch)
        _instant_backoff(monkeypatch, attempts=3)
        reached_before = app._sidebar_gate_reached
        recoveries_before = app._sidebar_gate_recoveries

        app._start_sidebar_connection(source)
        await _drain_with_paints(app, pilot, source)

        assert (
            app._sidebar_gate_reached - reached_before == 1
        ), "the gate was consulted for a frame that was already refused"
        assert (
            app._sidebar_gate_recoveries - recoveries_before == 0
        ), "a relayout was bought for a verdict the gate had already reached"


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


@pytest.mark.asyncio
async def test_a_reselect_after_navigating_away_gets_a_full_budget(monkeypatch):
    """F2/Q1: the budget available ON A RESELECT, not merely at birth.

    THE DEFECT THIS PINS. A source navigated away from mid-retry keeps its spent
    `connect_attempts`: the chain stops because the `finally` declines to re-arm
    a source that is no longer current, and nothing on that path resets the
    counter. The user's next reselect then started partway through the budget —
    reproduced independently by review (carried 4, resuming at 5 of 7) and by QA
    against a real attach-cap eviction (carried 6 → 3.0 s remaining → failed,
    where the identical eviction from a fresh budget healed).

    WHY THE EXISTING RELATIONSHIP TEST DOES NOT COVER THIS.
    `test_the_retry_budget_outlasts_the_recovery_give_up_bound` asserts the
    span of the FULL schedule and passes while this defect is live, because the
    arithmetic never changed — what changed was how much of it a reselect gets
    to use. This test therefore asserts the property in the units the defect
    speaks: the remaining span at the moment a user-initiated connect begins.
    """
    session = RecoveringRemote("recovering")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        # A SHORT REAL BACKOFF, not a zeroed one. The residue is left by a
        # chain stopped BETWEEN rounds — the user switches away while a backoff
        # is sleeping — so the window has to exist for the test to enter it.
        monkeypatch.setattr(app_module, "SIDEBAR_CONNECT_BACKOFF_S", 0.05)
        monkeypatch.setattr(app_module, "SIDEBAR_CONNECT_BACKOFF_CEILING_S", 0.05)
        monkeypatch.setattr(app_module, "SIDEBAR_CONNECT_ATTEMPTS", 7)

        app._start_sidebar_connection(source)
        # Part-way through, NOT to exhaustion: an exhausted budget resets itself
        # on surrender, and the residue this pins is the one a chain leaves when
        # it is stopped early.
        for _ in range(3):
            await asyncio.sleep(0.06)
        assert source.connect_attempts > 0, "the precondition never spent any budget"
        assert source.connect_attempts < app_module.SIDEBAR_CONNECT_ATTEMPTS

        # The user navigates away mid-backoff; the chain stops itself.
        app._interaction = SessionInteraction(FakeSession())
        task = source.connection_task
        if task is not None:
            await task
        await asyncio.sleep(0)
        carried = source.connect_attempts
        assert carried > 0, "the precondition left no residue to clear"
        assert not source.connection_error, "the budget was spent, not carried"

        # The user comes back and reselects. Sampled SYNCHRONOUSLY, before the
        # new task gets a turn: the budget a connect begins with is the property
        # under test, and letting the task run first hides it — a successful
        # connect zeroes the counter on its own, so a later sample reads 0 even
        # on a tree where the reselect inherited a spent budget. That made an
        # earlier version of this test survive its own mutation.
        app._interaction = source
        app._start_sidebar_connection(source)

        assert source.connect_attempts == 0, "a reselect inherited a spent budget"

        # AND IN THE UNITS THE DEFECT SPEAKS: the span still available at the
        # moment a user-initiated connect begins must outlast the bound the
        # whole constant is derived from. Computed from the SHIPPED schedule
        # (this test patches the backoff to keep itself fast, and a span
        # measured off the patched values would assert nothing about what
        # users get).
        shipped = _shipped_span_from(source.connect_attempts)
        assert shipped > COLD_FALLBACK_S
        # NOT VACUOUS, in two steps. Carrying a residue strictly shortens what
        # the next connect gets...
        assert _shipped_span_from(carried) < shipped
        # ...and carrying enough of one takes it below the bound entirely, which
        # is the failure QA reproduced against a real eviction (carried 6 → 3.0 s
        # → failed, where the same eviction healed from a fresh budget). Pinned
        # as the existence of such a value rather than as one number, so it
        # keeps meaning the same thing if the schedule is retuned.
        assert any(
            _shipped_span_from(spent) < COLD_FALLBACK_S for spent in range(1, _SHIPPED_ATTEMPTS + 1)
        ), "no residue could ever break the bound; this test would be vacuous"


@pytest.mark.asyncio
async def test_the_retry_loops_own_rearm_still_spends_the_budget(monkeypatch):
    """The other half of F2/Q1: the reset must not make the budget infinite.

    The reviewers' proposed one-liner — clear the counter beside
    `connection_error` in `_start_sidebar_connection` — is the right place and
    the right intent, but the retry loop re-arms through that same method. Taken
    unconditionally it zeroes the counter the previous round just incremented,
    and a permanently unreachable owner is re-dialled forever instead of
    surrendering: measured at `attempts=1` on every round for 16 rounds with no
    terminal state, which is a worse bug than the one being fixed.

    So this asserts the pair: a user start refills, the loop's own re-arm does
    not, and exhaustion is still reachable.
    """
    session = RecoveringRemote("gone")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        _instant_backoff(monkeypatch, attempts=3)

        app._start_sidebar_connection(source)
        await _drain_retries(app, source)

        assert source.connection_error == "the runtime is not responding"
        assert session.bind_calls == app_module.SIDEBAR_CONNECT_ATTEMPTS + 1


@pytest.mark.asyncio
async def test_a_readiness_gate_timeout_is_not_retried(monkeypatch):
    """F1: a PAINT failure is terminal on the first occurrence.

    `_await_sidebar_frame` runs its own 15 s timer to expiry before raising, so
    folding its failure into the reconnect budget does not re-dial anything — it
    re-commits an already-bound session and waits the timer out again, once per
    attempt. Reviewer measured 8 commits and ~120 s of "Connecting…" where the
    honest behaviour is a single 15 s failure, plus 8 forced full-screen arming
    relayouts through a recovery branch whose author documents any firing as a
    signal.

    Asserted as a COUNT of commits, not a duration: the defect is "the budget is
    spent on this", which is a fact about how many times the body ran.
    """
    session = RecoveringRemote("painting", heals_after=1)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        prepared: Any = (source, object())
        monkeypatch.setattr(app, "_prepare_sidebar_session", _always(prepared))
        commits = {"n": 0}

        def commit(*_args: Any, **_kwargs: Any) -> Any:
            commits["n"] += 1
            future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            # Exactly what `expired()` does when the 15 s timer fires.
            future.set_exception(
                SurfaceNotReady(
                    "The conversation connected but its input surface did not become ready"
                )
            )
            return future

        monkeypatch.setattr(app, "_commit_sidebar_session", commit)
        _instant_backoff(monkeypatch, attempts=7)

        app._start_sidebar_connection(source)
        await _drain_retries(app, source)

        assert commits["n"] == 1, "a readiness-gate timeout was retried"
        assert source.display_only is True
        assert source.connection_error == (
            "The conversation connected but its input surface did not become ready"
        )
        # Surrendering on a paint failure still hands the user a full budget for
        # the connection attempt their reselect will make.
        assert source.connect_attempts == 0


@pytest.mark.asyncio
async def test_the_retry_window_animates_and_the_failed_state_does_not(monkeypatch):
    """D1/D4: the band moves while it is working, and stops when it is not.

    The retry window was a byte-identical frozen frame for its whole ~12.75 s —
    a design round measured six frames from t=0.5s to t=12.0s hashing to one
    file, because `_sync_spinner_timer` animates only on `_streaming or
    _starting` and the reconnect path sets neither. Seven attempts behind a
    surface that acknowledged none of them reads as a hang.

    Asserted on the band's own STATE rather than on rendered pixels, so it
    cannot flake on paint timing; the rendered frames are attached to the PR.
    """
    session = RecoveringRemote("recovering")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        _instant_backoff(monkeypatch, attempts=4)
        assert app._status is not None

        # Mid-retry: working, so it animates. Published through the same method
        # the connect body calls, which is what owns this state.
        source.display_only = True
        source.connect_attempts = 2
        source.connection_error = ""
        app._show_sidebar_connection(source)
        await pilot.pause()
        assert app._status._connecting is True
        assert app._status._spinner_timer is not None, "the retry window has no animation"
        frame = app._status._spinner_index
        app._status._advance_spinner()
        assert app._status._spinner_index != frame

        # Exhausted: not working, so the motion stops — which is the state
        # change the eye catches without reading the words.
        source.connection_error = "the runtime is not responding"
        app._show_sidebar_connection(source)
        await pilot.pause()
        assert app._status._connecting is False
        assert app._status._spinner_timer is None


@pytest.mark.asyncio
async def test_guidance_answers_wait_or_act_in_every_disconnected_state(monkeypatch):
    """D2: the state with the most uncertainty must not carry the least advice.

    `connection_error` is cleared between attempts so the band can say
    "Connecting…", and it was also the only thing gating the refusal hint — so
    pressing Enter mid-retry produced a bare "unavailable" with no answer, while
    pressing it after the app gave up produced the fuller sentence. Inverted
    against need. Suppressing the reselect advice mid-retry is right; nothing
    had replaced it.
    """
    session = RecoveringRemote("recovering")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        source.display_only = True
        notices: list[str] = []
        monkeypatch.setattr(app, "_notice", lambda text, kind="info": notices.append(text))

        # Mid-retry: the app is working and the user need not act.
        source.connect_attempts = 2
        source.connection_error = ""
        app.composer_submission_refused()
        assert "Reconnecting" in notices[-1]
        assert "Select this session again" not in notices[-1]
        # The SLASH-COMMAND refusal carries the same three-state guidance. It is
        # a second caller of `_unavailable_hint` with eight call sites of its
        # own, and it reads the hint through a different guard — so the composer
        # assertions above do not cover it, and the inverted-guidance bug was
        # fixed here separately.
        assert app._allow_source_command() is False, "a cold source must refuse the command"
        assert "Reconnecting" in notices[-1]
        assert "Select this session again" not in notices[-1]

        # Exhausted: the app has stopped, and reselecting is the right advice.
        source.connection_error = "the runtime is not responding"
        app.composer_submission_refused()
        assert "Select this session again to retry." in notices[-1]
        assert app._allow_source_command() is False, "a cold source must refuse the command"
        assert "Select this session again to retry." in notices[-1]

        # Never-attempted: nothing to say beyond the refusal itself.
        source.connect_attempts = 0
        source.connection_error = ""
        app.composer_submission_refused()
        assert notices[-1] == "Send unavailable until connected."
        assert app._allow_source_command() is False, "a cold source must refuse the command"
        assert notices[-1] == "Commands unavailable until connected."


def test_the_derivation_refuses_a_backoff_it_cannot_solve(monkeypatch):
    """F4: a mistuned constant fails loudly at the edit, not at import.

    `_attempts_outlasting` runs at module scope, so its unbounded loop put a
    hang — or, with a zeroed backoff, an `OverflowError` from the doubling — at
    IMPORT time of `app.py`. Unreachable from the shipped values, but the
    trigger is a one-character edit to constants a future agent has explicit
    reason to tune, which is the worst place for a silent failure.
    """
    monkeypatch.setattr(app_module, "SIDEBAR_CONNECT_BACKOFF_S", 0.0)
    monkeypatch.setattr(app_module, "SIDEBAR_CONNECT_BACKOFF_CEILING_S", 0.0)
    with pytest.raises(ValueError, match="cannot outlast"):
        app_module._attempts_outlasting(COLD_FALLBACK_S)


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
