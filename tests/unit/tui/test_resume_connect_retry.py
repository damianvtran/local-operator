"""`/resume`'s dial is bounded-retried, not one-shot.

THE DEFECT THESE PIN. `_attach_or_refuse` called `AttachedSession.connect` once
and handed any exception straight to the user, which is the pre-#883 shape at
the other seam: #883 gave the SIDEBAR's connect a budget so a transient owner
loss heals by itself, and left `/resume` — the path a user reaches for when the
sidebar has already failed them — reporting a permanent verdict for what is
usually a runtime that is mid-restart (a `kill -9` republishes its record within
a second or two). The visible cost was the operator re-typing `/resume`.

What is asserted here is the POLICY — retried, bounded, record re-read per
attempt, static refusals not retried — driven through the real `_attach_or_refuse`
with only its seams (`find_runtime_record`, `AttachedSession.connect`, the redial
clock) stubbed, exactly as `test_stop_command.py` drives it.

THE BOUND IS WALL CLOCK, SO THE CLOCK IS A SEAM. `_attach_or_refuse` reads its
time through `_resume_redial_clock` and waits through `_resume_redial_pause`;
those two exist so a test can drive a VIRTUAL clock past the deadline instead of
measuring a duration, which AGENTS.md's "Timing, flakes" section forbids and
which would be a bet on machine load in any case. The backoff is collapsed to
zero (or replaced outright) for the same reason, and the derivation that sizes
the real budget is asserted as a RELATIONSHIP in
`test_the_budget_outlasts_the_transient_window_and_stays_in_tens_of_seconds`.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.attached import (
    COLD_FALLBACK_S,
    FRONTEND_ATTACH_MIN_PROTOCOL,
    FRONTEND_SYNC_FOREGROUND_S,
    frontend_attach_refusal,
)
from local_operator.session.frontend_state import FRONTEND_CAPABILITY
from local_operator.session.runtime.types import SessionRecord
from local_operator.tui import app as app_module
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: The shipped schedule, captured at import BEFORE any test patches it for
#: speed: the property under test is what a user's next `/resume` actually gets.
_SHIPPED_BACKOFF_S = app_module.SIDEBAR_CONNECT_BACKOFF_S
_SHIPPED_CEILING_S = app_module.SIDEBAR_CONNECT_BACKOFF_CEILING_S
_SHIPPED_ATTEMPTS = app_module.RESUME_CONNECT_ATTEMPTS
_SHIPPED_WALL_S = app_module.RESUME_CONNECT_WALL_S
_SHIPPED_BOUND_S = app_module.RESUME_CONNECT_BOUND_S


#: The notice measure the retry row has to fit: `resume-after.geometry.json`
#: reports the `NoticeBlock` at `[75, 2]` at 100 columns, and it stays 75 at
#: every wider terminal — so a row longer than this always wraps and splits its
#: own parenthetical (design D3, where the 77-78 cell row did exactly that).
_NOTICE_MEASURE_CELLS = 75


class _Stranded(Exception):
    """Raised in the swap window to stop a test at the point it asserts."""


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


def _notices(app: OperatorApp) -> list[str]:
    """Notices on screen, tolerating a transcript that has not been built yet.

    The modal boot composition mounts no `TranscriptView` until the app is
    booted, and a notice written before that lands on the screen rather than in
    the view — which is a state these tests should not depend on either way.
    """
    views = list(app.query(TranscriptView))
    if not views:
        return []
    return [block._text for block in views[0].blocks() if isinstance(block, NoticeBlock)]


def _gave_up(detail: str, *, session_id: str = "remote-1") -> str:
    """The terminal sentence, spelled the way the app spells it.

    ONE concept, two spellings would drift: the app's copy names the session and
    the next step (UX U4) around whatever the last failure's own text was, so
    the test builds it from the same two pieces rather than pasting the whole
    string twice.
    """
    return f"could not resume {session_id} — {detail}. Run /resume {session_id} again to retry."


class _RedialClock:
    """A VIRTUAL clock for the redial's two seams, measured in loop turns.

    `RESUME_CONNECT_WALL_S` is a wall-clock bound, so the property under test is
    "this loop stops when its budget is spent" — and AGENTS.md's "Timing,
    flakes" section forbids asserting that with a duration: a test that sleeps
    then asserts is a bet on machine load. This drives the deadline past itself
    with no sleeping at all, the same shape `session_picker`'s cadence test uses
    when it patches that widget's own module clock.

    `dial_s` is what ONE attempt costs, so the interesting case — an owner that
    is alive and silent, spending a full `FRONTEND_SYNC_FOREGROUND_S` envelope
    per dial — is expressible as a number the test chooses rather than as a
    machine speed it hopes for.
    """

    def __init__(self, *, dial_s: float = 0.0, start: float = 1_000.0) -> None:
        self.started = start
        self.now = start
        self.dial_s = dial_s
        #: What each backoff was asked to wait for, in order. The clamp is
        #: asserted against this rather than against elapsed time.
        self.pauses: list[float] = []

    def clock(self) -> float:
        return self.now

    async def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.now += seconds

    def spend_dial(self) -> None:
        self.now += self.dial_s

    def install(self, monkeypatch, *, wall_s: float | None = None) -> None:
        monkeypatch.setattr(app_module, "_resume_redial_clock", self.clock)
        monkeypatch.setattr(app_module, "_resume_redial_pause", self.pause)
        if wall_s is not None:
            monkeypatch.setattr(app_module, "RESUME_CONNECT_WALL_S", wall_s)


def _record(
    pid: int, name: str, *, protocol: int = 5, capabilities: list[str] | None = None
) -> SessionRecord:
    return SessionRecord(
        pid=pid,
        kind="tui",
        session_id=f"sid-{pid}",
        conversation_name=name,
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="k",
        protocol=protocol,
        capabilities=[FRONTEND_CAPABILITY] if capabilities is None else capabilities,
    )


def _app(monkeypatch, tmp_path: Path) -> OperatorApp:
    """A booted-independent app whose `/resume` seams the caller stubs."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    app._resume_factory = lambda _sid: _factory(FakeSession())  # type: ignore[assignment]
    monkeypatch.setattr(app_module, "sidebar_connect_backoff_s", lambda _attempt: 0.0)
    return app


@asynccontextmanager
async def _running(app: OperatorApp):
    """An app with a built transcript, yielded so the dialog can be driven.

    Booted before yielding because the notices these tests read are rendered
    into the transcript view, which the modal boot composition has not mounted
    yet — and kept OPEN across the assertions, because the view goes with the
    app the moment `run_test` exits.
    """
    async with app.run_test(size=(100, 30)) as pilot:  # type: ignore[attr-defined]
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        yield pilot


@pytest.mark.asyncio
async def test_a_transient_connect_failure_is_retried_rather_than_reported(monkeypatch, tmp_path):
    """The pre-#883 shape: one dial, then the exception as a verdict.

    The retry is what makes a runtime that is mid-restart (a republished record
    a second later) invisible to the user, so the assertion is that the SECOND
    dial happened — and that the user was shown a live retry rather than an
    error in the meantime.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")
    dials: list[Any] = []
    # RECORDED, never raised: anything raised inside the stub is caught by the
    # connect body's own handler and read as one more transient failure, so an
    # assertion here would turn into a retry instead of a failure (the sidebar
    # file learned this the hard way).
    seen_between_attempts: list[list[str]] = []

    async def connect(record_arg, *_args, **_kwargs):
        dials.append(record_arg)
        if len(dials) == 1:
            raise ConnectionError("attach refused")
        seen_between_attempts.append(_notices(app))
        return FakeSession()

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)
    # The swap window's first step, so the test stops where it was going to
    # assert rather than stepping into a real adoption.
    monkeypatch.setattr(
        OperatorApp, "_reset_ledger_for_swap", lambda _self: (_ for _ in ()).throw(_Stranded())
    )

    async with _running(app):
        with pytest.raises(_Stranded):
            await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        assert len(dials) == 2, "a transient failure was not retried"
        notices = _notices(app)
        assert not [
            n for n in notices if n == "attach refused"
        ], "the user was shown a verdict for a dial that was already being retried"
        # BETWEEN the attempts the user is told a redial is running, which is
        # the whole point of narrating it: the alternative they used to get was
        # an error for a dial that was about to succeed.
        assert seen_between_attempts, "the second dial never ran"
        assert [
            n for n in seen_between_attempts[0] if n.startswith("reconnecting to session")
        ], seen_between_attempts[0]
        # ...and once the dial SUCCEEDS the row is retired, rather than left
        # claiming a reconnect that has already happened.
        assert not [n for n in notices if n.startswith("reconnecting to session")]


@pytest.mark.asyncio
async def test_only_exhaustion_latches_and_it_says_so_honestly(monkeypatch, tmp_path):
    """A genuinely unreachable owner still reaches the user — after the budget.

    The terminal state is what the budget is FOR, not an alternative to it. The
    count is asserted against the derived constant rather than a literal, so a
    retuned schedule cannot turn this into a permanently-failing loop or into a
    one-shot by accident.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")
    dials: list[Any] = []

    async def connect(record_arg, *_args, **_kwargs):
        dials.append(record_arg)
        raise ConnectionError("attach refused")

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app):
        await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        assert len(dials) == _SHIPPED_ATTEMPTS, "the redial did not spend its budget"
        # THE ROW IS RESTATED INTO THE VERDICT, so the transcript holds exactly
        # ONE statement about the redial and it is in the past tense (design D1,
        # UX U3). Asserted as the whole list because the old shape left the live
        # row ABOVE the verdict — "reconnecting to session … (retry 48 of 48)"
        # over "the runtime is not responding" — and a second run stacked a
        # second pair on top of it.
        assert _notices(app) == [_gave_up("attach refused")]


@pytest.mark.asyncio
async def test_a_second_resume_does_not_strand_the_first_ones_row(monkeypatch, tmp_path):
    """Two `/resume`s leave two verdicts, not four rows (design D1, UX U3).

    The give-up arm used to post the verdict and leave its live row standing, so
    every later run inherited the previous one's — measured on the PR's own
    frame: `notice rows on screen: 4` after a second run.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")

    async def connect(*_args, **_kwargs):
        raise ConnectionError("attach refused")

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app):
        await app._attach_or_refuse(tmp_path, "remote-1", 90909)
        first = _notices(app)
        await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        assert first == [_gave_up("attach refused")]
        assert _notices(app) == [_gave_up("attach refused")] * 2
        assert not [
            n for n in _notices(app) if n.startswith("reconnecting to session")
        ], "a retry row survived the run it belonged to"


@pytest.mark.asyncio
async def test_the_runtime_record_is_re_read_before_every_attempt(monkeypatch, tmp_path):
    """A retired runtime republishes under a NEW pid.

    Reusing the first record would spend the whole budget redialling a socket
    that cannot answer, so the lookup is per attempt — the same reason
    `AttachedSession._recover_runtime` re-reads. The second dial must therefore
    be handed the SECOND record, not the first.
    """
    app = _app(monkeypatch, tmp_path)
    first = _record(90909, "the remote")
    republished = _record(91555, "the remote")
    lookups: list[str] = []
    dialled: list[Any] = []

    def lookup(_root, concrete):
        lookups.append(concrete)
        return (first, 90909) if len(lookups) == 1 else (republished, 91555)

    async def connect(record_arg, *_args, **_kwargs):
        dialled.append(record_arg)
        if len(dialled) == 1:
            raise ConnectionError("attach refused")
        return FakeSession()

    monkeypatch.setattr("local_operator.mobile.attach_client.find_runtime_record", lookup)
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)
    monkeypatch.setattr(
        OperatorApp, "_reset_ledger_for_swap", lambda _self: (_ for _ in ()).throw(_Stranded())
    )

    async with _running(app):
        with pytest.raises(_Stranded):
            await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        assert len(lookups) == 2, "the record was captured once and reused"
        assert [r.pid for r in dialled] == [
            90909,
            91555,
        ], "the redial was pointed at the retired pid instead of the republished one"


@pytest.mark.asyncio
async def test_a_static_refusal_is_not_retried(monkeypatch, tmp_path):
    """A capability gap is a property of the owner, not of the moment.

    Every redial would raise the identical refusal, so a budget spent on it is
    just a longer way to the same sentence — and since `frontend_attach_refusal`
    already answers the question `connect` refuses on (its docstring's "ONE
    RULE, TWO CALLERS"), the refusal is taken BEFORE spending even one dial.
    The expected sentence is built by that function rather than pasted, so the
    two cannot drift apart.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote", capabilities=[])
    dials: list[Any] = []

    async def connect(*_args, **_kwargs):
        dials.append(1)
        # The SAME sentence `connect` refuses with, taken from the canonical
        # function rather than spelled out, so the two cannot drift apart.
        raise ConnectionError(frontend_attach_refusal(record))

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app):
        await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        assert len(dials) == 1, "a static refusal was retried"
        assert _notices(app) == [frontend_attach_refusal(record)]


@pytest.mark.asyncio
async def test_an_empty_record_on_the_first_attempt_is_paced_not_refused(monkeypatch, tmp_path):
    """THE REPUBLISH GAP: `(None, None)` is what the redial exists for.

    A runtime that has just been restarted publishes no record between the old
    pid retiring and the new one arriving, and a `/resume` typed inside that
    window used to be told the session was "open in an older Local Operator
    process" — a statement about a process that did not exist — and returned
    without trying again (review m1). It is paced now, and the FIRST attempt
    doing so is the assertion: the lookup returns nothing, and the second one
    gets a record and is dialled.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")
    lookups: list[str] = []
    dials: list[Any] = []
    seen: list[list[str]] = []

    def lookup(_root, concrete):
        lookups.append(concrete)
        return (None, None) if len(lookups) == 1 else (record, 90909)

    async def connect(record_arg, *_args, **_kwargs):
        dials.append(record_arg)
        # Sampled on the dial that FOLLOWS the paced attempt, which is the only
        # place the row exists on this path: the success path removes it (by
        # design — see the give-up tests for the arm that restates it instead).
        seen.append(_notices(app))
        return FakeSession()

    monkeypatch.setattr("local_operator.mobile.attach_client.find_runtime_record", lookup)
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)
    monkeypatch.setattr(
        OperatorApp, "_reset_ledger_for_swap", lambda _self: (_ for _ in ()).throw(_Stranded())
    )

    async with _running(app):
        with pytest.raises(_Stranded):
            await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        assert len(dials) == 1, "the empty first attempt did not reach the republished record"
        assert lookups == ["remote-1", "remote-1"]
        assert not [
            n for n in _notices(app) if "older Local Operator process" in n
        ], "an absence was reported as an older process running the session"
        # The wait is narrated while it happens rather than staying silent until
        # it succeeds, and the row is gone once it HAS succeeded.
        assert [n for n in seen[0] if n.startswith("reconnecting to session")]
        assert not [n for n in _notices(app) if n.startswith("reconnecting to session")]


@pytest.mark.asyncio
async def test_an_owner_that_publishes_no_record_is_refused_by_its_own_pid(monkeypatch, tmp_path):
    """`(None, pid)`: an owner is there, and it is one no dial can reach.

    The marker names a live process and the scan finds no dialable record for
    it — an older binary, or a registrant that failed to start. That IS the case
    the upgrade advice belongs to, and the pid it names must be the live owner's
    rather than the (possibly stale) one the command was invoked with.
    """
    app = _app(monkeypatch, tmp_path)
    dials: list[Any] = []

    async def connect(*_args, **_kwargs):
        dials.append(1)
        raise AssertionError("an owner with no dialable record was dialled")

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (None, 91234),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app):
        await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        assert dials == []
        assert _notices(app)[-1] == (
            "session remote-1 is open in an older Local Operator process (pid 91234) — "
            "update or close that process, then resume again"
        )


@pytest.mark.asyncio
async def test_a_record_below_the_attach_protocol_is_refused_with_the_older_process_advice(
    monkeypatch, tmp_path
):
    """One version concept, one threshold: `FRONTEND_ATTACH_MIN_PROTOCOL`.

    The guard this replaces compared against a literal `4` while the canonical
    refusal used `5`, so a protocol-4 record was dialled and then refused by the
    very rule the guard was duplicating (review n3). A record below the current
    build has no full-TUI attach, so the user is told to update the process —
    the advice that belongs to this case and only this one (design D5).
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote", protocol=FRONTEND_ATTACH_MIN_PROTOCOL - 1)
    dials: list[Any] = []

    async def connect(*_args, **_kwargs):
        dials.append(1)
        raise AssertionError("an owner below the attach protocol was dialled")

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app):
        await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        assert dials == []
        assert _notices(app)[-1] == (
            "session remote-1 is open in an older Local Operator process (pid 90909) — "
            "update or close that process, then resume again"
        )


@pytest.mark.asyncio
async def test_an_owner_pid_that_moved_between_the_reads_is_paced_not_refused(
    monkeypatch, tmp_path
):
    """A moved marker is a republish, which is the case the retry is FOR.

    `find_runtime_record` derives its owner from the same `.session.pid` marker
    the caller read, so a mismatch can only mean the marker moved between the
    two reads. Refusing there — as the suggested guard did — would reproduce
    m1's own defect (a sentence about a pid, and no retry) for a different
    input, so the second attempt's record is dialled instead.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(91555, "the remote")
    dials: list[Any] = []

    async def connect(record_arg, *_args, **_kwargs):
        dials.append(record_arg)
        return FakeSession()

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 91555),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)
    monkeypatch.setattr(
        OperatorApp, "_reset_ledger_for_swap", lambda _self: (_ for _ in ()).throw(_Stranded())
    )

    async with _running(app):
        with pytest.raises(_Stranded):
            await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        assert len(dials) == 1, "the redial refused a record whose owner pid had moved"
        assert not [
            n for n in _notices(app) if "older Local Operator process" in n
        ], "the moved pid was reported as an older process"


@pytest.mark.asyncio
async def test_the_in_flight_row_states_the_bound_and_fits_the_notice_measure(
    monkeypatch, tmp_path
):
    """The row's copy, pinned: bounded, no denominator, no internal noun.

    Three findings land on this one string. It used to assert a diagnosis the
    app had not established ("the owner is not answering" — wrong whenever there
    is no record at all), use a noun this TUI uses nowhere else ("owner"), and
    restate an implementation counter the user cannot act on (`retry 48 of 48`)
    while the wait itself had no stated end (design D2/D3/D4, UX U2). It is now
    62 cells for a typical id — inside the 75-cell notice measure, so it fits on
    one line instead of splitting its own parenthetical.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")
    seen: list[list[str]] = []

    async def connect(*_args, **_kwargs):
        # Sampled on the SECOND dial: the row is narrated after the first one
        # fails, which is why `seen[1]` is the first frame that can hold it.
        seen.append(_notices(app))
        raise ConnectionError("attach refused")

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app):
        await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        rows = [n for n in seen[1] if n.startswith("reconnecting to session")]
        assert rows == [
            f"reconnecting to session remote-1 — still trying for up to " f"{_SHIPPED_BOUND_S:g} s"
        ]
        assert len(rows[0]) <= _NOTICE_MEASURE_CELLS, "the row splits in the notice measure"
        assert "owner" not in rows[0]
        assert "retry" not in rows[0]


def test_the_budget_outlasts_the_transient_window_and_stays_in_tens_of_seconds():
    """THE RELATIONSHIP, not the number — the way this fix regresses quietly.

    What the budget has to outlast is the TRANSIENT WINDOW this loop exists for,
    not the facade's `RECOVERY_GIVE_UP_S`: a record moving between
    `live_runtime_pid` and `find_runtime_record`, an attach refusal, and a facade
    mid-recovery for up to `COLD_FALLBACK_S` — the same window the sidebar's
    budget is sized against, and the derivation that grew to ~14.5 minutes was
    sizing for 90 s of a bound this command never has to survive (review M1).

    Pinned as inequalities between the constants rather than as literals, and
    deliberately with NO duration in it: a test that sleeps and asserts is a bet
    on machine load (AGENTS.md, "Timing, flakes"). The wall-clock behaviour
    itself is driven on a virtual clock in the tests below.
    """
    # 50% margin over the recovery bound, the sidebar's own rule: the last paced
    # attempt must land clearly after `_go_cold`, not tie with it on a loaded
    # machine.
    assert _SHIPPED_WALL_S > COLD_FALLBACK_S * 1.5
    # ...and the fence on the OTHER side, which is the actual regression: the
    # budget is a small number of full-envelope dials, not 49 of them.
    envelopes = _SHIPPED_WALL_S / FRONTEND_SYNC_FOREGROUND_S
    assert envelopes <= 2, "the wall-clock cap allows more than a couple of full-envelope dials"
    assert _SHIPPED_BOUND_S == _SHIPPED_WALL_S + FRONTEND_SYNC_FOREGROUND_S
    assert _SHIPPED_BOUND_S < 60, "a user-initiated command should not wait a minute"
    # The secondary cap is the SAME span, so the two caps cannot disagree: a dial
    # that fails instantly cannot buy more dials than the paced schedule itself
    # would have spent the budget on.
    assert _SHIPPED_ATTEMPTS == app_module._attempts_outlasting(_SHIPPED_WALL_S)


@pytest.mark.asyncio
async def test_the_redial_stops_when_the_wall_clock_is_spent_not_when_attempts_run_out(
    monkeypatch, tmp_path
):
    """M1, stated as behaviour: attempt time comes off the budget.

    The measured regression this closes: each dial of a live-but-silent owner
    costs a full `FRONTEND_SYNC_FOREGROUND_S` envelope before it raises, and the
    budget that counted only the backoff was therefore 49 dials and ~14.5
    minutes for one `/resume`. On a virtual clock where every dial spends that
    envelope the loop must end after TWO of them — well before
    `RESUME_CONNECT_ATTEMPTS` — so what stopped it was the clock, not the count.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")
    clock = _RedialClock(dial_s=FRONTEND_SYNC_FOREGROUND_S)
    clock.install(monkeypatch)
    dials: list[Any] = []

    async def connect(record_arg, *_args, **_kwargs):
        dials.append(record_arg)
        clock.spend_dial()
        raise ConnectionError("the runtime is not responding")

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app):
        await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        assert len(dials) == 2, "a silent owner spent more than two full envelopes"
        assert len(dials) < _SHIPPED_ATTEMPTS, "the ATTEMPT cap ended the loop, not the budget"
        # The promise the row makes to the user holds on this path: the span is
        # the deadline plus at most the one envelope already in flight.
        assert clock.now - clock.started <= _SHIPPED_BOUND_S
        assert _notices(app) == [_gave_up("the runtime is not responding")]


@pytest.mark.asyncio
async def test_the_backoff_is_clamped_to_what_is_left_of_the_budget(monkeypatch, tmp_path):
    """The paced wait is measured against the REMAINING budget, not against itself.

    A schedule that slept its own backoff regardless would push the loop past
    its own deadline by up to one full backoff, which is how a "bounded" loop
    ends up unbounded in practice. Read off the pause the loop actually asked
    for, so no duration is measured.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")
    wall_s = 12.0
    clock = _RedialClock()
    clock.install(monkeypatch, wall_s=wall_s)
    # Instant dials and a backoff LONGER than the whole budget, so every pause
    # has to be the clamp rather than the schedule.
    monkeypatch.setattr(app_module, "sidebar_connect_backoff_s", lambda _attempt: 5.0)

    async def connect(*_args, **_kwargs):
        raise ConnectionError("the runtime is not responding")

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app):
        await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        # 5, then 5, then the 2 s that were left — never the 5 the schedule asked
        # for, and never a wait that outlives the deadline.
        assert clock.pauses == [5.0, 5.0, 2.0]
        assert clock.now - clock.started == wall_s
        assert max(clock.pauses) <= wall_s


@pytest.mark.asyncio
async def test_the_composer_says_what_is_happening_during_a_resume_redial(monkeypatch, tmp_path):
    """U1: a refused Enter speaks, and names the session it is reconnecting to.

    Enter is refused for the whole redial — `composer_submission_blocked` decides
    that and is unchanged — but the refusal used to say NOTHING, because it only
    spoke when the source was `display_only` or frame-pending and a `/resume`
    redial runs over a still-live session. Measured for 136 s: draft kept, no
    notice, no band change, no toast, and a retyped `/resume target` left sitting
    in the buffer. Read from INSIDE the real loop, so this pins live state rather
    than a state the test assembled by hand.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")
    seen: dict[str, Any] = {}

    async def connect(*_args, **_kwargs):
        seen["target"] = app._resume_retry_target
        seen["blocked"] = app.composer_submission_blocked("/resume other")
        app.composer_submission_refused()
        raise ConnectionError("the runtime is not responding")

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app):
        app._session_transition_pending = True
        try:
            await app._attach_or_refuse(tmp_path, "remote-1", 90909)
        finally:
            app._session_transition_pending = False

        assert seen["target"] == "remote-1", "the redial never published its target"
        assert seen["blocked"] is True, "a submit the redial swallows was not refused"
        assert (
            "Send unavailable until connected. Reconnecting to remote-1 — it will keep "
            f"trying for up to {_SHIPPED_BOUND_S:g} s; switching session stops the wait."
        ) in _notices(app), "the refused submit was silent"
        # Cleared when the loop ends, so a later transition cannot inherit it.
        assert app._resume_retry_target == ""
