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

THE OTHER HALF OF THE SAME POSTCONDITION: A COLD FACADE MUST BE ABLE TO BIND.
A connect that commits against a facade which cannot bind is one face of this
defect; the other is a facade that can NEVER bind, where every round of the same
budget is a silent no-op by construction. The sidebar leases sources through two
branches and both must build the SAME contract, because the source cache hands
either branch's facade to the user's later click: `saved_preview` is a viewer,
and the speculative (prewarm) branch now asks `connect` for that contract with
`viewer=True`. The tests under "the viewer contract" below pin the flag at the
call site, its effect through the real `connect`, and what it is for — the cold
prewarm row that heals on the click instead of latching.

The two levers are complementary and neither replaces the other: the LEASE now
builds a facade that can bind, and a facade that cannot bind AT ALL (a
deliberate stop on the legacy attach contract, which no lease in this file can
reach) gets one honest report instead of a budget spent on rounds that dial
nothing. That second half is what the tests appended to "the viewer contract"
below pin, including the arm that must still be retried — a facade
mid-recovery.

WHAT THESE DOUBLES CANNOT SHOW (UX round 2's gap note, carried here so nobody
reads them as covering it). Every double in this file models a facade whose bind
CANNOT complete: an isolated env has no launcher, so a round fails without ever
starting a successor runtime. On a host where the engage's spawn path can heal —
a production owner record whose successor gets published — the same click may
HEAL instead of reporting, and the verdict arm is then NARROWER in production
than these tests suggest. The claims here hold wherever the verdict is reached,
not that it is reached everywhere a stopped owner is involved.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from rich.text import Text

from local_operator.mobile.attach_client import AttachClient
from local_operator.session.attached import (
    COLD_FALLBACK_S,
    FRONTEND_ATTACH_MIN_PROTOCOL,
    FRONTEND_CAPABILITY,
    AttachedSession,
)
from local_operator.tui import app as app_module
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from local_operator.tui.session_navigation import OwnerWentCold, SurfaceNotReady
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
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

    THE FLAGS ARE PART OF THAT SURFACE, and they are the real ones the inherited
    ``can_ever_bind`` reads: this double is a VIEWER that HEALS, so it carries
    the capability (``_can_go_cold`` True), no loop of its own (``_recovering``
    False, because the SILENT RETURN is what this double models and the budget is
    what the tests below spend against it), and no socket at all (``_client``
    None — cold means nothing is connected here). The mid-recovery answer is
    pinned separately, and so is the arm that has a socket while it is cold — see
    ``MidRecoveryRemote`` and ``RefreshNotSyncedRemote``.
    """

    _can_go_cold = True
    _recovering = False
    _disposed = False
    _client = None
    #: Read FIRST by the real ``session_was_stopped``, which the connect now
    #: consults on a failed round: this double never issued a stop, so the real
    #: method falls through to the durable marker and answers False — the "no
    #: stop proven" shape. The path is deliberately one that cannot exist, so no
    #: real marker can leak into these arms; the marker arm sets its own config
    #: directory per instance (``StoppedOwnerRemote``).
    _deliberate_stop = False
    _config_dir = Path("/nonexistent-local-operator-config")

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


def _notice_rows(app: OperatorApp) -> list[str]:
    """What the notices on screen actually PAINT, one entry per row.

    `NoticeBlock.text()` is the authored string; `_build()` is what the surface
    shows. The claim these tests make is about the surface — a row that still
    promises a dial is a rendered sentence, and a restate that changed nothing
    on screen would pass a check on the authored list and fail the user.
    """
    rows: list[str] = []
    for view in app.query(TranscriptView):
        for block in view.blocks():
            if not isinstance(block, NoticeBlock):
                continue
            # `_build` is typed as any renderable; these blocks build `Text`,
            # and `plain` is the string the surface shows.
            built = block._build()
            rows.extend((built if isinstance(built, Text) else Text(str(built))).plain.split("\n"))
    return rows


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
    to ``Saved · Reconnect failed · Select again to retry``. Asking the user
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
    signature is ZERO recoveries, against 1,820 before it.

    THE CONSULTATION COUNT DROPPED FROM ONE TO ZERO (the geometry-settled
    arming below), and that is a deliberate, smaller-cost arm rather than a
    weakened assertion: the readiness frame is armed only once the incoming
    geometry has settled
    (`_arm_sidebar_frame_when_settled`), so a frame painted while the dock is
    still reflowing is not asked about at all — and a cold owner refuses it
    through the hook's cold branch, which never needed the gate's verdict
    (``_sidebar_gate_surface_ready``'s own first check is ``is_cold``). What
    this test is FOR is the second counter, which is unchanged at zero: no
    full-screen relayout is bought for a verdict the app already holds. Pinning
    a consultation that is now avoided would pin the cost, not the property.

    Counts, not rates: a busy machine changes how long the frames take, not how
    many the gate refuses.
    """
    session = BindsThenLosesItsOwner("cold-at-paint")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        outcomes = _cold_frame_rig(app, source, session, monkeypatch)
        _instant_backoff(monkeypatch, attempts=3)
        reached_before = app._sidebar_gate_reached
        recoveries_before = app._sidebar_gate_recoveries

        app._start_sidebar_connection(source)
        await _drain_with_paints(app, pilot, source)

        # The frame really was refused, and by the cold branch rather than by
        # the timer: without this the two zero deltas below would also hold for
        # a frame that was never armed at all.
        assert outcomes, "no frame was ever armed"
        assert isinstance(outcomes[0], OwnerWentCold), (
            f"the cold frame settled as {outcomes[0]!r}: the wait, not the cold check, "
            "decided it"
        )
        assert (
            app._sidebar_gate_reached - reached_before == 0
        ), "a frame that was already refused was put to the gate anyway"
        assert (
            app._sidebar_gate_recoveries - recoveries_before == 0
        ), "a relayout was bought for a verdict the gate had already reached"


@pytest.mark.asyncio
async def test_a_frame_whose_geometry_never_settles_is_armed_anyway(monkeypatch):
    """THE SETTLE WINDOW IS BOUNDED: a geometry that never settles cannot stall.

    `_arm_sidebar_frame_when_settled` holds the arm back while the incoming
    geometry is still moving, and that is only safe because the wait is a COUNT
    of refresh hops rather than a condition on the geometry. The pathological
    case is driven here directly, by a geometry probe that returns a different
    sample on every call -- so "has it stopped moving?" can never be answered
    yes -- and the two facts that make the bound real are asserted: the frame is
    ARMED anyway, and the readiness future still settles.

    Why this is the blocker-shaped question rather than a nicety: the switch
    holds its input boundary until that future settles, so an arm that waited on
    a condition the geometry never satisfies would hang every switch behind the
    gate's 15 s timer. A deferred arm that can stall is worse than the refusals
    it removes.
    """
    session = RecoveringRemote("never-settles", heals_after=1)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        samples: list[tuple[float, ...]] = []

        def never_settles(_self: OperatorApp) -> tuple[float, ...]:
            sample = (float(len(samples)), float(len(samples)))
            samples.append(sample)
            return sample

        monkeypatch.setattr(OperatorApp, "_sidebar_frame_geometry", never_settles)
        armed: list[int] = []
        real_arm = OperatorApp._arm_sidebar_frame

        def arm(_self: OperatorApp, src: SessionInteraction, fut: Any, gen: int) -> None:
            armed.append(len(samples))
            real_arm(_self, src, fut, gen)

        monkeypatch.setattr(OperatorApp, "_arm_sidebar_frame", arm)
        _instant_backoff(monkeypatch, attempts=1)

        def commit(*_args: Any, **_kwargs: Any) -> Any:
            return app._await_sidebar_frame(source, app._sidebar_navigation.generation)

        monkeypatch.setattr(app, "_commit_sidebar_session", commit)
        monkeypatch.setattr(app, "_prepare_sidebar_session", _always((source, object())))

        app._start_sidebar_connection(source)
        await _drain_with_paints(app, pilot, source)

        assert armed, (
            "the frame was never armed: a geometry that never settles stalled the switch "
            f"behind {len(samples)} samples"
        )
        assert len(samples) <= app_module.SIDEBAR_ARM_SETTLE_HOPS + 2, (
            f"the arm waited on {len(samples)} samples, past the "
            f"{app_module.SIDEBAR_ARM_SETTLE_HOPS}-hop "
            "bound; the wait is no longer bounded"
        )
        assert app._sidebar_gate_reached >= 1, "the armed frame was never put to the gate"
        assert source.can_never_bind is False, "an armed frame was refused on a healthy source"


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

        def composer_notice() -> str:
            """The text of the row a refused Enter is speaking through.

            The composer's refusal OWNS its row now — one block per state,
            restated in place and retired when the state ends (UX U1) — so this
            reads the block it holds rather than the last block in the view:
            the slash-command refusal below appends its own row AFTER it, and a
            restated row is not the newest one on screen. `latest_notice` is the
            probe for that second path.
            """
            notice = app._composer_refusal_notice
            assert notice is not None and notice.is_attached, "the refused Enter said nothing"
            return notice.text()

        def latest_notice() -> str:
            """The text of the LAST notice in the transcript."""
            views = list(app.query(TranscriptView))
            blocks = [b for b in views[0].blocks() if isinstance(b, NoticeBlock)]
            assert blocks, "nothing was said at all"
            return blocks[-1].text()

        # Mid-retry: the app is working and the user need not act.
        source.connect_attempts = 2
        source.connection_error = ""
        app.composer_submission_refused()
        assert "Reconnecting" in composer_notice()
        assert "Select this session again" not in composer_notice()
        # The SLASH-COMMAND refusal carries the same three-state guidance. It is
        # a second caller of `_unavailable_hint` with eight call sites of its
        # own, and it reads the hint through a different guard — so the composer
        # assertions above do not cover it, and the inverted-guidance bug was
        # fixed here separately.
        assert app._allow_source_command() is False, "a cold source must refuse the command"
        assert "Reconnecting" in latest_notice()
        assert "Select this session again" not in latest_notice()

        # Exhausted: the app has stopped, and reselecting is the right advice.
        source.connection_error = "the runtime is not responding"
        app.composer_submission_refused()
        assert "Select this session again to retry." in composer_notice()
        assert app._allow_source_command() is False, "a cold source must refuse the command"
        assert "Select this session again to retry." in latest_notice()

        # NEVER-ATTEMPTED: the composer's row is RESTATED, not stacked, so the
        # second refusal leaves one row where it had one before (UX U1).
        source.connect_attempts = 0
        source.connection_error = ""
        app.composer_submission_refused()
        assert composer_notice() == "Send unavailable until connected."
        assert app._allow_source_command() is False, "a cold source must refuse the command"
        assert latest_notice() == "Commands unavailable until connected."


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


class _LostDuringPreparation(RecoveringRemote):
    """Binds cleanly, then loses its owner inside the prepare→commit window.

    The state the belt's unconditional form exists for: the bind postcondition
    sees a reachable owner, the owner is gone by the time the commit returns, and
    the commit returned ``None`` (no frame to fail and no ``post_display_hook`` to
    notice) — see `test_a_cold_session_whose_commit_returns_no_frame_is_still_refused`.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self._cold = False

    async def _ensure_bound(self, *, foreground: bool = True) -> None:
        self.bind_calls += 1

    async def ensure_display_current(self) -> None:
        # The loss lands HERE, between the bind postcondition and the commit.
        self._cold = True


@pytest.mark.asyncio
async def test_a_cold_session_whose_commit_returns_no_frame_is_still_refused(monkeypatch):
    """F/m2: the postcondition is not conditional on there being a frame.

    ``ready is None`` is ``_commit_sidebar_session``'s early return for a
    prepared replay that already IS the current transcript view. With the belt
    gated on ``ready is not None`` the body fell straight through it having
    already set ``display_only = False``, so an owner lost in this window was
    published live with no frame to refuse and no hook left to notice — the
    ``connect_attempts`` reset below then credited the round with a connect it
    never made.
    """
    session = _LostDuringPreparation("lost")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        prepared: Any = (source, object())
        monkeypatch.setattr(app, "_prepare_sidebar_session", _always(prepared))
        monkeypatch.setattr(app, "_commit_sidebar_session", Mock(return_value=None))
        # Observed rather than allowed to run, so this test sees ONE refusal
        # rather than the whole chain (the chain is pinned elsewhere).
        rearmed = Mock()
        monkeypatch.setattr(app, "_start_sidebar_connection", rearmed)
        _instant_backoff(monkeypatch, attempts=2)

        await app._connect_sidebar_source(source)
        await asyncio.sleep(0)  # the re-arm is a `call_soon` callback

        assert session.is_cold is True
        assert source.display_only is True, "a cold session was published with no frame to refuse"
        assert source.connect_attempts == 1, "the cold commit was counted as a completed connect"
        rearmed.assert_called_once_with(source, continues_retry=True)


@pytest.mark.asyncio
async def test_the_connect_prepares_with_refresh_so_no_frame_return_is_unreachable(monkeypatch):
    """F/m2: WHY that return cannot be reached from the sidebar connect path.

    Two things together, rather than a prose claim. (1) The trigger that return
    compares against is a presentation whose replay view already IS the current
    transcript — demonstrated below on the real helper, so the premise is
    executed rather than asserted. (2) The ONLY way to be handed that
    presentation is `_prepare_sidebar_session`'s two shortcuts, and both are
    guarded on ``not refresh``; the connect body passes ``refresh=True``, which
    is what this pins. So the belt's extension above is defensive on this path —
    it is kept because one property read is cheaper than a silent hole in the
    postcondition this whole file exists for.
    """
    session = RecoveringRemote("fresh", heals_after=1)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        prepared: Any = (source, object())
        seen: list[dict[str, Any]] = []

        async def prepare(*_args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return prepared

        monkeypatch.setattr(app, "_prepare_sidebar_session", prepare)
        monkeypatch.setattr(app, "_commit_sidebar_session", Mock(return_value=None))

        await app._connect_sidebar_source(source)
        await asyncio.sleep(0)

        # (1) the trigger, on the real helper.
        assert app._capture_sidebar_presentation().replay.view is app._transcript_view()
        # (2) the connect never asks for it.
        assert seen == [{"refresh": True}], (
            "the connect prepared without `refresh`, which re-opens the "
            "current-view shortcuts `_commit_sidebar_session`'s early return needs"
        )
        # And a bound, cold-free connect with no frame still settles as connected.
        assert source.display_only is False
        assert session.is_cold is False


@pytest.mark.asyncio
async def test_a_refused_enter_stops_speaking_once_the_connect_lands(monkeypatch):
    """UX U1's other half: the refusal row ends when the state it describes does.

    `Send unavailable until connected.` is present-progressive about a state the
    app owns, so it may not survive the connect that ended it — a durable row
    would have the transcript say "until connected" directly above a session that
    IS connected, and it accumulated one row per refused Enter while the wait
    lasted. Retired on the COMPLETED connect only (same condition as the budget
    refill): an attempt that returned early proved nothing about the owner.
    """
    session = UnreachableRemote("blipping", heals_after=3)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        prepared: Any = (source, object())
        monkeypatch.setattr(app, "_prepare_sidebar_session", _always(prepared))
        monkeypatch.setattr(app, "_commit_sidebar_session", Mock(return_value=None))
        _instant_backoff(monkeypatch)

        app._start_sidebar_connection(source)
        # Mid-connect, the state the composer refuses to send in.
        assert source.display_only is True
        app.composer_submission_refused()
        app.composer_submission_refused()
        assert app._composer_refusal_notice is not None, "the refusal said nothing"
        assert app._composer_refusal_notice.is_attached

        await _drain_retries(app, source)

        assert source.connect_attempts == 0, "the connect did not land"
        assert app._composer_refusal_notice is None, "the refusal outlived the connect"
        views = list(app.query(TranscriptView))
        rows = [b for b in views[0].blocks() if isinstance(b, NoticeBlock)]
        assert [b.text() for b in rows if b.text().startswith("Send unavailable")] == []


@pytest.mark.asyncio
async def test_a_refused_enter_does_not_survive_the_latch(monkeypatch):
    """UX U1's third exit: the LATCH is a state end too (round 3).

    The row ends on the redial's exits and on a COMPLETED connect, but a
    sidebar connect that LATCHES left it speaking in the present tense under the
    band's own verdict — measured on the round-3 head at 14.23s:

        LATCHED: status='Saved · Reconnect failed · Select again to retry'
        notices at the latch: ['Send unavailable until connected. Reconnecting —
                              it will keep trying for a few more seconds.']

    The band had retracted the retry while the transcript still promised one,
    in exactly the state the original bug report was about. Restated into the
    register the user's next Enter already produced, so the two surfaces agree
    whether or not they press it — and asserted on the PAINTED rows, because
    the defect is what the user reads.
    """
    session = RecoveringRemote("gone")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        # NOT the collapsed backoff here: the Enter has to land in the mid-retry
        # register ("Reconnecting — it will keep trying …"), which only exists
        # while a round is PARKED. With a zeroed backoff the whole chain latches
        # inside one `pilot.pause()` and the row is written in the
        # pre-first-dial register instead — which would make this test pass on
        # the very defect it exists to catch.
        monkeypatch.setattr(app_module, "SIDEBAR_CONNECT_ATTEMPTS", 3)
        monkeypatch.setattr(app_module, "sidebar_connect_backoff_s", lambda _attempt: 0.2)

        app._start_sidebar_connection(source)
        assert source.display_only is True
        # Pump until the first attempt has failed, so the Enter lands in the
        # register the user actually sees mid-retry (the `connect_attempts`
        # hint) rather than in the pre-first-dial state.
        for _ in range(400):
            await pilot.pause()
            if source.connect_attempts >= 1:
                break
        assert source.connect_attempts >= 1, "the connect never attempted a dial"
        app.composer_submission_refused()
        assert app._composer_refusal_notice is not None, "the refusal said nothing"
        # The state the row describes, while it is still true.
        assert any("will keep trying" in row for row in _notice_rows(app))

        await _drain_retries(app, source)

        assert source.connect_attempts == 0, "the connect did not latch"
        assert app._status is not None
        status = app._status.render_text(120).plain
        assert "Reconnect failed" in status
        assert "Select again to retry" in status

        rows = _notice_rows(app)
        assert "will keep trying" not in " ".join(
            rows
        ), "the refusal row outlived the latch and kept promising a dial"
        assert any("Select this session again to retry." in row for row in rows)
        # One row, not two: the restate replaces rather than stacks.
        assert sum(1 for row in rows if "Send unavailable" in row) == 1


# --------------------------------------------------------------------------
# The viewer contract: a cold prewarm source must be able to bind again
# --------------------------------------------------------------------------


def _viewer_record() -> Any:
    """A discovery record ``connect`` will dial.

    Only two fields are read before the dial — ``frontend_attach_refusal``
    decides on ``protocol`` and ``capabilities`` — and the dial is what these
    tests replace, so standing up a real runtime would buy nothing. That is
    deliberately different from the e2e stage, which dials real sockets.
    """
    return SimpleNamespace(
        protocol=FRONTEND_ATTACH_MIN_PROTOCOL,
        capabilities=(FRONTEND_CAPABILITY,),
    )


async def _refuse_to_take_over() -> None:
    raise AssertionError("a sidebar viewer must never take over a session")


class _StopAtTheDial(Exception):
    """Sentinel from the stub ``_dial``: the contract is set before the dial."""


class PrewarmRemote(RecoveringRemote):
    """A prewarm-created facade that has LOST its owner, guard and all.

    ``_ensure_bound``'s first guard is ``if not self._can_go_cold or
    self._disposed: return``, so a cold facade returns without dialling while
    that capability is unset — and no retry count can change it. Mirroring the
    guard here is what makes the two cases below differ by the capability alone,
    rather than by which double happened to be constructed.
    """

    def __init__(self, session_id: str, *, can_go_cold: bool) -> None:
        super().__init__(session_id)
        self._can_go_cold = can_go_cold
        # The real guard and the latch diagnostics both read `_disposed`;
        # `RecoveringRemote` carries a public `disposed`, so both are kept here
        # rather than letting the latch line print None for the field it exists
        # to print.
        self._disposed = False
        self.dials = 0

    async def _ensure_bound(self, *, foreground: bool = True) -> None:
        self.bind_calls += 1
        if not self._can_go_cold or self._disposed:
            return
        self.dials += 1
        self._cold = False


@pytest.mark.asyncio
async def test_connect_keeps_the_legacy_contract_unless_it_is_asked_for_a_viewer(
    tmp_path, monkeypatch
) -> None:
    """``viewer=True`` is the only way in, and False is every other caller today.

    The two contracts are what a facade does when it loses its owner: take the
    conversation over, or go cold and let the next action rebind. ``/resume``,
    the startup attach and ``lop --resume`` are the first kind — the user asked
    to be put in front of that conversation — so this keyword must not change
    them by accident, which is only testable where the flag is set.
    """
    built: list[AttachedSession] = []

    async def stop_at_the_dial(self: AttachedSession, record: Any) -> Any:
        built.append(self)
        raise _StopAtTheDial

    monkeypatch.setattr(AttachedSession, "_dial", stop_at_the_dial)
    for viewer in (False, True):
        with pytest.raises(_StopAtTheDial):
            await AttachedSession.connect(
                _viewer_record(),
                "prewarm",
                config_dir=tmp_path,
                takeover_factory=_refuse_to_take_over,
                viewer=viewer,
            )
        assert built[-1]._can_go_cold is viewer


@pytest.mark.asyncio
async def test_the_speculative_sidebar_lease_asks_for_the_viewer_contract(monkeypatch) -> None:
    """THE WIRING, at the one call site that decides it.

    The prewarm branch used to call ``connect`` with no contract asked for,
    which left ``_can_go_cold`` False on a facade the source cache then handed
    to the user's click — so a cold state on it was permanent. The flag is read
    off a facade built by the REAL ``connect`` (the dial is the only stub), so
    this fails if either the call site drops the keyword or the keyword stops
    setting the capability.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(60):
            await pilot.pause()
        with monkeypatch.context() as patched:
            built: list[AttachedSession] = []

            async def stop_at_the_dial(self: AttachedSession, record: Any) -> Any:
                built.append(self)
                raise _StopAtTheDial

            patched.setattr(AttachedSession, "_dial", stop_at_the_dial)
            patched.setattr(
                "local_operator.mobile.attach_client.find_runtime_record",
                lambda *_args, **_probe: (_viewer_record(), 4242),
            )
            with pytest.raises(_StopAtTheDial):
                await app._lease_sidebar_source("prewarm", speculative=True)
            assert (
                built[-1]._can_go_cold is True
            ), "the speculative lease built a facade that cannot rebind once cold"


@pytest.mark.asyncio
async def test_a_cold_prewarm_facade_heals_instead_of_latching(monkeypatch) -> None:
    """THE POLICY: the click's facade can bind, so a cold prewarm row heals.

    The facade is cold WITH the capability — the state the speculative lease now
    always builds — and the connect commits on its first dial, so the user sees
    the live transcript rather than being told to select again.
    """
    session = PrewarmRemote("prewarm", can_go_cold=True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        prepared: Any = (source, object())
        monkeypatch.setattr(app, "_prepare_sidebar_session", _always(prepared))
        monkeypatch.setattr(app, "_commit_sidebar_session", Mock(return_value=None))
        _instant_backoff(monkeypatch)

        app._start_sidebar_connection(source)
        await _drain_retries(app, source)

        assert session.dials == 1, "the cold prewarm facade never dialled"
        assert session.is_cold is False
        assert source.display_only is False, "the connect did not commit"
        assert source.connection_error == ""
        assert app._status is not None
        status = app._status.render_text(160).plain
        assert "Reconnect failed" not in status
        assert "Select again to retry" not in status


@pytest.mark.asyncio
async def test_a_facade_that_can_never_bind_reports_once_and_spends_no_budget(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """THE POLICY: a viewer closed to dialling gets one report, not a budget.

    Cold is NOT one state. A facade mid-recovery is cold and WILL bind (its loop
    releases it rebindable, which is what the budget exists to wait out); a
    facade that cannot bind at all is cold and will NOT, so every further round
    is pure backoff over a round that dials nothing — the operator's latch, whose
    signature was ``elapsed=0.00s`` over eight attempts.

    The reachable shape is a DELIBERATE STOP on the legacy attach contract: the
    session `lop --resume` boots is built by ``connect`` without ``viewer=True``,
    ``_adopt_session`` registers it as a sidebar source, and ``/stop`` ends it -
    and ``_on_disconnected`` returns BEFORE any recovery loop starts, so nothing
    ever sets the capability ``_ensure_bound``'s first guard reads. The session
    is stopped, so the honest sentence is the app's own for that state and the
    reselect promise is withdrawn: a reselection reuses this very facade, and a
    message is REFUSED rather than served (``_no_session_notice`` says so).

    Retrying stays right one arm over; ``MidRecoveryRemote`` pins that.
    """
    session = PrewarmRemote("stopped", can_go_cold=False)
    app = OperatorApp(lambda: _factory(FakeSession()))
    caplog.set_level(logging.WARNING, logger="local_operator.tui.app")
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        # A budget of four, so "one round" cannot be confused with "the budget".
        _instant_backoff(monkeypatch, attempts=4)

        app._start_sidebar_connection(source)
        await _drain_retries(app, source)

        # THE STRUCTURAL CLAIM: one bind attempt, out of a budget of four, and
        # not one dial — asserted as counts, never as a span.
        assert session.bind_calls == 1, "the un-bindable arm spent the budget anyway"
        assert session.dials == 0, "a facade without the capability dialled"
        assert source.display_only is True
        assert source.connect_attempts == 0
        assert app._status is not None
        status = app._status.render_text(160).plain
        assert _stopped_sentence(session) in status
        assert "Select again to retry" not in status
        # The composer's own row answers with the same verdict rather than the
        # reselect it cannot honour, and with no "until connected" prefix: ONE
        # string, so the two surfaces cannot drift (UX U3/U4, round 1).
        assert app._unavailable_notice("Send") == _stopped_sentence(session)

        # AND THE LATCH LINE NAMES THE ARM, so the arm is diagnosable from the
        # log the way the budget arm already is. `attempts=1` is the count of
        # rounds spent, and it is the field that separates this arm from the
        # eight-round no-op sequence that used to print the same shape.
        latch_lines = [
            record.getMessage()
            for record in caplog.records
            if "sidebar connect latched" in record.getMessage()
        ]
        assert latch_lines, "the latch was not logged at all"
        assert "can never bind this session" in latch_lines[0]
        assert "can_go_cold=False" in latch_lines[0]
        assert "attempts=1" in latch_lines[0]


class MidRecoveryRemote(PrewarmRemote):
    """Cold, without the viewer capability, and with a RECOVERY LOOP RUNNING.

    The one shape that looks like the arm above and must NOT be treated as it:
    ``_recovering`` means a loop is chasing the owner, and every exit of that
    loop either attaches this facade or releases it rebindable, so the state is
    transient by construction. Deliberately keeps ``_can_go_cold`` False, so the
    budget below is spent on the ``_recovering`` term of ``can_ever_bind``
    ALONE — without it the one-shot arm above would fire and this test fails.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id, can_go_cold=False)
        self._recovering = True


@pytest.mark.asyncio
async def test_the_recovering_arm_still_spends_the_whole_budget(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """MID-RECOVERY IS TRANSIENT, so the budget is still spent waiting it out.

    The same cold state as the arm above, differing only in ``_recovering`` -
    which is the whole point of the distinction: an owner loss the recovery loop
    is chasing heals on its own within ``COLD_FALLBACK_S``, and a caller that
    treated it as final would tell the user a session on its way back is gone.
    So this arm must still reach the latch the old way: the whole budget, and
    the exhausted-retry copy rather than the stopped-session one.
    """
    session = MidRecoveryRemote("mid-recovery")
    app = OperatorApp(lambda: _factory(FakeSession()))
    caplog.set_level(logging.WARNING, logger="local_operator.tui.app")
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        _instant_backoff(monkeypatch, attempts=3)

        app._start_sidebar_connection(source)
        await _drain_retries(app, source)

        assert (
            session.bind_calls == app_module.SIDEBAR_CONNECT_ATTEMPTS + 1
        ), "a recovering facade was cut short instead of waited out"
        assert session.dials == 0
        assert source.display_only is True
        assert source.connect_attempts == 0
        assert app._status is not None
        status = app._status.render_text(160).plain
        assert "Reconnect failed" in status
        assert "Select again to retry" in status
        assert _stopped_sentence(session) not in status

        latch_lines = [
            record.getMessage()
            for record in caplog.records
            if "sidebar connect latched" in record.getMessage()
        ]
        assert latch_lines, "the latch was not logged at all"
        assert "retry budget exhausted" in latch_lines[0]
        assert "can never bind this session" not in latch_lines[0]


def _live_client() -> Any:
    """A REAL ``AttachClient`` whose socket is up, built without dialling.

    ``AttachClient.connected`` reads ``self._connected``, so a bare instance with
    that flag set IS the real object in the real state these arms read: no
    double, and pyright sees the attribute's own type. Nothing else on the client
    is reached by ``is_cold``, ``can_ever_bind`` or ``session_was_stopped``.
    """
    client = AttachClient.__new__(AttachClient)
    client._connected = True
    return client


class RefreshNotSyncedRemote(RecoveringRemote):
    """A LIVE session mid display-refresh, on the LEGACY attach contract.

    The third cause of coldness, and the one the verdict this file pins must NOT
    read as a permanent state: ``is_cold``'s last disjunct is
    ``not _ready_for_events``, and ``_refresh_display_history`` clears that flag
    while the client STAYS CONNECTED and the runtime keeps serving
    (``attached.py`` says so at its own ``/move`` seam). The socket is up, the
    session is alive, and the window ends on its own — so this must retry and
    then commit, exactly as the reviewer's A/B showed the old budget doing.

    ``is_cold`` is the REAL expression rather than a hand-flipped flag, which is
    what makes the state visible to the suite at all: a double that hard-codes
    coldness cannot represent it, and this arm's defect (a live session told it
    was stopped) passed every test in this file. ``_can_go_cold`` stays False —
    the legacy contract — because that is what made the wrong verdict decisive:
    with the capability set, the predicate answers True for another reason and
    the bug hides.
    """

    def __init__(self, session_id: str, *, syncs_on_bind: int = 2) -> None:
        super().__init__(session_id)
        self._client = _live_client()
        self._ready_for_events = False
        self._can_go_cold = False
        self._syncs_on_bind = syncs_on_bind
        self.dials = 0

    @property
    def is_cold(self) -> bool:
        # The real predicate, term for term (``attached.py``): a double that
        # restated it as a flag is how MAJOR-1 stayed invisible.
        return self._client is None or not self._client.connected or not self._ready_for_events

    async def _ensure_bound(self, *, foreground: bool = True) -> None:
        self.bind_calls += 1
        # The refresh COMPLETING, not a dial, and on the round after the one that
        # finds the window open — which is the shape under test: the first round
        # sees cold, the retry waits, the second round finds the sync landed. The
        # real class dials nothing here either (a legacy facade still returns
        # from `_ensure_bound`'s first guard), so what heals is the SYNC.
        if self.bind_calls >= self._syncs_on_bind:
            self._ready_for_events = True


@pytest.mark.asyncio
async def test_a_live_session_mid_refresh_is_retried_not_reported_as_stopped(
    monkeypatch,
) -> None:
    """THE STATE MAJOR-1 WAS ABOUT: connected, not yet synced, and alive.

    A legacy-contract facade with its socket UP while ``_refresh_display_history``
    rebuilds the window is cold by ``is_cold``'s third disjunct and un-bindable by
    ``_can_go_cold`` — two facts that together look exactly like the deliberate
    stop the one-shot arm reports. They are not: the refresh finishes, the facade
    stops being cold, and the NEXT round commits. So a verdict that reads only
    the guard's two flags tells the user a live session was stopped and drops the
    retry that would have healed it.

    The window is one round wide, so this fails on the head that had the one-shot
    arm alone: one attempt, the stopped sentence, no commit.
    """
    session = RefreshNotSyncedRemote("mid-refresh")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        prepared: Any = (source, object())
        monkeypatch.setattr(app, "_prepare_sidebar_session", _always(prepared))
        monkeypatch.setattr(app, "_commit_sidebar_session", Mock(return_value=None))
        _instant_backoff(monkeypatch)

        app._start_sidebar_connection(source)
        await _drain_retries(app, source)

        # THE RETRY HAPPENED AND IT HEALED: a second round, and a commit.
        assert session.bind_calls == 2, "the mid-refresh window was not waited out"
        assert session.is_cold is False
        assert source.display_only is False, "the connect never committed"
        assert source.connection_error == ""
        assert source.can_never_bind is False
        assert app._status is not None
        status = app._status.render_text(160).plain
        assert _stopped_sentence(session) not in status
        assert app._unavailable_notice("Send") != _stopped_sentence(session)


def _stopped_sentence(session: Any) -> str:
    """The app's stopped-session sentence for ``session``, id spliced in.

    Built from the module constant rather than retyped, so a copy change is not
    a silent test rewrite: what these tests assert is that the SURFACES carry
    this sentence, in this shape (UX U3, round 1 — the id is the point of it).
    """
    return app_module.STOPPED_SESSION_NOTICE.format(session_id=session.session_id)


class StoppedOwnerRemote(RecoveringRemote):
    """A VIEWER whose owner was stopped before its row was clicked.

    The arm UX round 1 measured on the surface users reach most naturally (F9,
    then a click on a stopped session's row): the row leases a viewer facade, the
    DIAL fails on every round (``Connect call failed``), and the app has no stop
    notice of its own to go on — ``_on_disconnected`` never ran for this viewer,
    because it never had a socket to lose. The durable marker is the only signal,
    so ``session_was_stopped`` is the REAL implementation here, reading a real
    wake-index entry the test writes through the store; nothing about the marker
    read is stubbed.

    ``_can_go_cold`` is True — a viewer, exactly as the sidebar's lease builds —
    which is the point: ``can_ever_bind`` cannot catch this arm, and before the
    marker was consulted it spent the whole budget promising a reselect.
    """

    def __init__(self, session_id: str, *, config_dir: Any) -> None:
        super().__init__(session_id)
        self._config_dir = config_dir
        self._can_go_cold = True
        self.dials = 0

    async def _ensure_bound(self, *, foreground: bool = True) -> None:
        self.bind_calls += 1
        # The measured shape: a record that exists with nothing listening, so
        # every round fails the dial rather than returning from a guard.
        raise ConnectionError("Connect call failed ('127.0.0.1', 54321)")


def _write_stop_marker(config: Path, session_id: str, *, stopped: bool = True) -> None:
    """The wake-index entry a real stop leaves behind, or the same entry reopened.

    ``stopped=True`` is the durable marker ``control._mark_wakes_dormant``
    stamps through the real stop ladder: one schedule, so an entry exists at all
    — a wake-less session writes NO entry and therefore no marker, which is the
    LIMIT (UX U7) this rig also pins by leaving the marker absent.

    ``stopped=False`` is what reopening the session leaves: the same schedules
    with ``stopped_at`` dropped, exactly as the open-time rewrite does it
    (``write_entry``'s ``clear``). The two are the shapes the repeat-click arm
    has to tell apart, so they are built by one function and differ in one key.
    """
    from local_operator.harness.wake import WakeSchedule
    from local_operator.wakes import store as wake_store

    wake_store.write_entry(
        config,
        session_id,
        cwd=str(config),
        schedules=[
            WakeSchedule(id="w1", message="check in", next_due_at=1_700_000_060_000, created_at=1)
        ],
        preserve={"stopped_at": 1_700_000_000_000} if stopped else None,
    )


@pytest.mark.asyncio
async def test_a_row_whose_owner_was_stopped_reports_it_once(
    tmp_path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """UX U2: the stopped ROW reaches the same verdict as the stopped facade.

    Eight rounds and 16 s of ``Connecting…`` used to end on "Select again to
    retry", which the reviewer then followed for another 15 s to the same
    verdict. The durable stop marker says why: this owner is not coming back.

    Asserted structurally: NO round at all — the marker is read before the
    dial, so the round that used to spend the engage envelope on an owner that
    cannot answer never starts (PR #1049 round 2, U6.1 measured that envelope at
    12.8-13.9 s of `Saved · Connecting…` before this sentence arrived) — the
    honest sentence on the band and in the refusal row, and the latch naming the
    arm. The marker itself is a real file written through the wake store — which
    is also what pins its LIMIT, since a session with no schedules writes no
    entry at all.
    """
    session = StoppedOwnerRemote("stoppedbypeer", config_dir=tmp_path)
    _write_stop_marker(tmp_path, session.session_id)
    app = OperatorApp(lambda: _factory(FakeSession()))
    caplog.set_level(logging.WARNING, logger="local_operator.tui.app")
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        _instant_backoff(monkeypatch, attempts=4)

        app._start_sidebar_connection(source)
        await _drain_retries(app, source)

        # ZERO, not one: the pre-flight reads the durable marker before
        # `bind_runtime`, so this row never opens a round. Asserting the
        # POSTCONDITION the fix exists for rather than the round count keeps the
        # test honest in both directions -- a future change that reintroduced a
        # dial here would spend the envelope again, and this is what would say so.
        assert session.bind_calls == 0, "the stopped row opened a round anyway"
        assert source.display_only is True
        assert source.connect_attempts == 0
        assert app._status is not None
        status = app._status.render_text(160).plain
        assert _stopped_sentence(session) in status
        assert "Select again to retry" not in status
        assert app._unavailable_notice("Send") == _stopped_sentence(session)

        latch_lines = [
            record.getMessage()
            for record in caplog.records
            if "sidebar connect latched" in record.getMessage()
        ]
        assert latch_lines, "the latch was not logged at all"
        assert "the session was stopped" in latch_lines[0]
        assert "attempts=1" in latch_lines[0]


@pytest.mark.asyncio
async def test_the_stopped_row_verdict_needs_the_marker_not_just_a_cold_source(
    tmp_path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The same rows and rounds WITHOUT the marker still spend the budget.

    The negative half of the arm above, and the reason it cannot pass vacuously:
    a cold viewer whose dial keeps failing and whose session carries no stop
    marker is a DEAD OWNER as far as the app can tell, and a dead owner is what
    the budget exists for (a successor can republish its record). Without the
    marker the round count and the exhausted-retry copy are unchanged.
    """
    session = StoppedOwnerRemote("neverstarted", config_dir=tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()))
    caplog.set_level(logging.WARNING, logger="local_operator.tui.app")
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        _instant_backoff(monkeypatch, attempts=3)

        app._start_sidebar_connection(source)
        await _drain_retries(app, source)

        assert session.bind_calls == app_module.SIDEBAR_CONNECT_ATTEMPTS + 1
        assert app._status is not None
        status = app._status.render_text(160).plain
        assert "Reconnect failed" in status
        assert _stopped_sentence(session) not in status
        latch_lines = [
            record.getMessage()
            for record in caplog.records
            if "sidebar connect latched" in record.getMessage()
        ]
        assert latch_lines and "the session was stopped" not in latch_lines[0]


def _stopped_session_row_click(app: OperatorApp, session_id: str) -> None:
    """Click a row the way the sidebar does, through the real message."""
    from local_operator.tui.widgets.session_sidebar import SessionSidebar

    app.post_message(SessionSidebar.Selected(session_id))


async def _reach_the_stopped_verdict(app: OperatorApp, source: SessionInteraction) -> str:
    """Drive one real connect to its held verdict; return the error text it holds.

    Shared by the two tests below because the arm they differ about is what a
    SECOND click does — if they each built the verdict their own way, a
    difference between them could come from the setup rather than from the
    click.
    """
    app._start_sidebar_connection(source)
    await _drain_retries(app, source)
    assert source.can_never_bind is True, "the connect never published a verdict"
    assert source.connection_error, "the verdict carried no error text"
    return source.connection_error


@pytest.mark.asyncio
async def test_reclicking_a_stopped_row_answers_without_starting_a_connect(
    tmp_path, monkeypatch
) -> None:
    """UX U6.2: no second click may spend a wait the app has the answer to.

    The two facts this pins are the ones the reviewer measured on the round-2
    head: the click painted `Saved · Connecting…` OVER the verdict the app had
    published seconds earlier, then spent the full connect envelope (12.8 s in
    their run) to re-derive the identical sentence. The verdict here is reached
    through the real connect, so `can_never_bind` and `connection_error` are the
    app's own — not values this test wrote onto the source.

    Asserted as a structural ABSENCE rather than as band text: the band comes
    back to the same sentence in the old code too, so a text-only assertion
    would pass against the defect. What must not exist is a second connect task,
    and what must not be painted is the `Connecting…` frame that the old path
    put over the answer — that frame is the lie, and it is only visible in the
    task's existence and in the fields it clears on entry.
    """
    session = StoppedOwnerRemote("reclickedstopped", config_dir=tmp_path)
    _write_stop_marker(tmp_path, session.session_id)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        app._session = session
        _instant_backoff(monkeypatch, attempts=4)
        held_error = await _reach_the_stopped_verdict(app, source)
        dials_at_verdict = session.bind_calls
        # The verdict's own task, still assigned to the source: a finished
        # connect is not cleared from `connection_task` (the retry re-arm
        # replaces it, and nothing else writes it), so the identity of THIS
        # object is what a second click must not change.
        task_at_verdict = source.connection_task

        _stopped_session_row_click(app, session.session_id)
        for _ in range(8):
            await pilot.pause()

        assert source.connection_task is task_at_verdict, (
            "the repeat click started another connect; the verdict it already held "
            "was cleared and re-derived"
        )
        assert session.bind_calls == dials_at_verdict, "the repeat click opened a round"
        assert source.can_never_bind is True
        assert source.connection_error == held_error
        assert app._status is not None
        status = app._status.render_text(160).plain
        assert _stopped_sentence(session) in status
        assert "Connecting" not in status, "the band went back to promising a dial"


@pytest.mark.asyncio
async def test_reclicking_a_reopened_row_dials_again(tmp_path, monkeypatch) -> None:
    """The negative control for the arm above: a reopen must still be dialled.

    A held verdict is a fact about the past. Reopening the session — here, the
    same wake-index entry with `stopped_at` dropped, which is what the
    open-time rewrite leaves — means an owner may be back, and the app must not
    answer that click from a verdict it can no longer justify. Without this
    test the arm above would also pass if the re-click branch gave up whenever
    `can_never_bind` was set, which is the shape that would strand a session
    that came back.
    """
    session = StoppedOwnerRemote("reopened", config_dir=tmp_path)
    _write_stop_marker(tmp_path, session.session_id)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        app._session = session
        _instant_backoff(monkeypatch, attempts=2)
        await _reach_the_stopped_verdict(app, source)

        # The session is reopened from elsewhere: the marker the verdict rested
        # on is gone, and nothing about this source has been touched.
        _write_stop_marker(tmp_path, session.session_id, stopped=False)
        _stopped_session_row_click(app, session.session_id)
        # The click's own handler is a worker, so the loop has to turn before an
        # ordinary connect exists to drain: draining first would await the
        # FINISHED verdict task and return on the very next line.
        for _ in range(6):
            await pilot.pause()
        await _drain_retries(app, source)

        assert session.bind_calls >= 1, "a reopened session was never dialled again"
        assert app._status is not None
        status = app._status.render_text(160).plain
        assert (
            _stopped_sentence(session) not in status
        ), "the app answered a reopened session with the stopped verdict"


@pytest.mark.asyncio
async def test_clicking_the_current_sessions_stopped_row_answers_with_the_verdict(
    monkeypatch,
) -> None:
    """UX U5: the click on the CURRENT stopped session is no longer inert.

    The session was live, then its owner was stopped while it was on screen, so
    ``display_only`` is False and no connect runs on the click — the row did
    nothing at all. The app already knows the stop here (``_stopped_session_id``
    is set by the very handler that painted the notice), so the click publishes
    the same verdict a stopped row reached from elsewhere does.

    ``display_only`` is deliberately NOT set: this session is not a saved view,
    its composer is open, and `/resume` is runnable from it — which is what the
    verdict says to do.
    """
    session = RecoveringRemote("currentstopped")
    session._recovering = False
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        # The click handler's current-session branch is keyed on the app's own
        # session, so the row's session has to BE it — the state under test is
        # "the session on screen lost its owner ages ago".
        app._session = session
        # Cold, as a stopped session's facade is, with the app's own record of
        # the stop it announced. `display_only` is FALSE — that is the whole
        # point of this arm: the session was live and lost its owner, so no
        # connect runs and the row click has nothing to repaint.
        session._cold = True
        source.display_only = False
        app._stopped_session_id = session.session_id

        _stopped_session_row_click(app, session.session_id)
        for _ in range(6):
            await pilot.pause()

        assert source.can_never_bind is True
        assert source.display_only is False
        assert app._status is not None
        status = app._status.render_text(160).plain
        assert _stopped_sentence(session) in status
        assert "Select again to retry" not in status


@pytest.mark.asyncio
async def test_resume_is_runnable_in_the_state_whose_verdict_names_it(monkeypatch) -> None:
    """UX U1: the command the verdict names must not be the one gate refuses.

    Drives the real gate and the real dispatch: in the stopped/un-bindable state
    the composer accepts `/resume <id>` and the dispatcher REACHES the handler.
    Two states that legitimately refuse are asserted in the same test so the
    widening cannot be read as "everything is allowed": a session transition
    already in flight (``_session_transition_pending``, which answers before the
    allowlist is consulted) and a command that is not a saved-view command.
    """
    session = PrewarmRemote("stoppedargv", can_go_cold=False)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        _instant_backoff(monkeypatch, attempts=2)
        app._start_sidebar_connection(source)
        await _drain_retries(app, source)
        assert source.can_never_bind is True, "the state under test did not arrive"

        called: list[tuple[str, str]] = []
        monkeypatch.setattr(app, "_cmd_resume", lambda arg, notice: called.append(("resume", arg)))

        # THE GATE, then the DISPATCH: both consult `_SAVED_LOCAL_COMMANDS`, and
        # both used to refuse this command in this state.
        assert app.composer_submission_blocked(f"/resume {session.session_id}") is False
        dispatched = app._run_slash_command(f"/resume {session.session_id}")
        if dispatched is not None:
            await dispatched
        assert called == [("resume", session.session_id)]

        # STILL REFUSED WHERE A REFUSAL IS THE TRUTH. A transition in flight owns
        # the composer; a command that needs an owner is not a saved-view one.
        app._session_transition_pending = True
        try:
            assert app.composer_submission_blocked(f"/resume {session.session_id}") is True
            assert app.composer_submission_blocked("/model gpt-4o") is True
        finally:
            app._session_transition_pending = False
        assert app.composer_submission_blocked("/model gpt-4o") is True
        assert called == [("resume", session.session_id)]

        # And the refusal ROW for a command that needs an owner is the verdict
        # with no "until connected" prefix (UX U4, round 1).
        assert app._unavailable_notice("Commands") == _stopped_sentence(session)
        assert "until connected" not in app._unavailable_notice("Commands")


@pytest.mark.asyncio
async def test_the_hint_helper_does_not_advise_the_reselect_the_verdict_withholds(
    monkeypatch,
) -> None:
    """QA Q5 (round 2): the row and a direct read of its helper must agree.

    `_unavailable_hint`'s reselect value is the answer for every arm that latched
    WITHOUT a verdict, and its sole caller answers a verdict-carrying source one
    line earlier — so that value reached no surface while still being what the
    helper returned for it. Asserted here rather than left implicit, because the
    way a later reader (or agent) takes a helper's answer for the row's text IS a
    direct read. The control keeps the reselect branch honest: the same latched
    error with no verdict still gets the advice it exists for.
    """
    session = PrewarmRemote("stoppedhint", can_go_cold=False)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        source = await _current_source(app, pilot, session)
        _instant_backoff(monkeypatch, attempts=2)
        app._start_sidebar_connection(source)
        await _drain_retries(app, source)

        assert source.can_never_bind is True and source.connection_error
        assert app._unavailable_hint() == "", "the helper must not offer a reselect"
        # The row itself is unchanged: the verdict, and no "until connected".
        assert app._unavailable_notice("Send") == _stopped_sentence(session)

        # CONTROL: no verdict behind the same latched error -> the branch is live.
        source.can_never_bind = False
        assert app._unavailable_hint() == " Select this session again to retry."


@pytest.mark.parametrize(
    "can_go_cold,recovering,disposed,connected,expected",
    [
        # The ordinary viewer, cold right now: it binds as soon as it is dialled.
        (True, False, False, False, True),
        # A viewer mid-recovery, and a LEGACY facade mid-recovery: both on their
        # way back, because the loop's exits leave the facade bindable (`_go_cold`
        # for a viewer, `_give_up_recovery` setting the flag for the legacy one).
        (True, True, False, False, True),
        (False, True, False, False, True),
        # A LIVE session the display is mid-refresh on: the socket is UP and the
        # runtime is serving, so `is_cold` is true through its third disjunct
        # alone. Retrying is what waits the window out — MAJOR-1, round 1.
        (False, False, False, True, True),
        (True, False, False, True, True),
        # The deliberate stop on the legacy contract: no loop ran, no socket is
        # up, so nothing ever sets the flag this guard reads. The reachable False.
        (False, False, False, False, False),
        # Disposed: refuses every path. Latent, and still False.
        (True, False, True, False, False),
        (False, False, True, False, False),
    ],
)
def test_can_ever_bind_is_the_truth_table_the_sidebar_decides_on(
    can_go_cold: bool, recovering: bool, disposed: bool, connected: bool, expected: bool
) -> None:
    """The real property, on the real class, at every combination of its inputs.

    Read through the facade rather than restated by a double: the app's verdict
    (and therefore the sentence the user reads) is this property's answer, and a
    double that reimplemented it would let the two drift silently.

    ``connected`` is a separate axis rather than a footnote because it is a
    separate CAUSE of coldness: `_ready_for_events` is cleared while the socket
    stays up, so a facade can be cold, un-diallable and perfectly alive at once.
    """
    session = AttachedSession.__new__(AttachedSession)
    session._can_go_cold = can_go_cold
    session._recovering = recovering
    session._disposed = disposed
    session._client = _live_client() if connected else None

    assert session.can_ever_bind is expected
