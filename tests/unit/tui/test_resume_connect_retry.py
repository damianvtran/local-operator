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

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from rich.text import Text

from local_operator.session.attached import (
    COLD_FALLBACK_S,
    FRONTEND_ATTACH_MIN_PROTOCOL,
    FRONTEND_SYNC_FOREGROUND_S,
    frontend_attach_refusal,
)
from local_operator.session.frontend_state import FRONTEND_CAPABILITY
from local_operator.session.runtime import registry
from local_operator.session.runtime.types import HEARTBEAT_TIMEOUT_S, SessionRecord
from local_operator.tui import app as app_module
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_navigation import UNREACHABLE_OWNER_MESSAGE
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


def _rendered_text(app: OperatorApp) -> str:
    """Every notice row the view paints, as one whitespace-normalised string."""
    return " ".join(" ".join(_rendered_notice_rows(app)).split())


def _notice_blocks(app: OperatorApp) -> list[NoticeBlock]:
    """The notice BLOCKS on screen, by identity, in visual order.

    `_notices` compares authored strings, which cannot tell one identical block
    from two. A claim that a run settled INTO an existing row rather than beside
    it is a claim about identity, so it has to read the widgets.
    """
    blocks: list[NoticeBlock] = []
    for view in app.query(TranscriptView):
        blocks.extend(block for block in view.blocks() if isinstance(block, NoticeBlock))
    return blocks


def _rendered_notice_rows(app: OperatorApp) -> list[str]:
    """What the notices actually PAINT, one entry per rendered row.

    Distinct from `_notices`, which reads the authored string: these rows are
    the block's own `_build` output, so a claim that a row is gone is a claim
    about the surface rather than about a list the app happens to hold. The two
    disagree exactly when a block is still ATTACHED — which is the defect class
    here: QA's repro read a correct-looking `pending=False` beside a row that was
    still on screen when the user returned to the session (QA Q2).
    """
    views = list(app.query(TranscriptView))
    if not views:
        return []
    rows: list[str] = []
    for block in views[0].blocks():
        if not isinstance(block, NoticeBlock):
            continue
        # `_build` IS the renderable the block paints; asking it here rather than
        # reading `_text` is what makes these rows the surface's own, and the
        # isinstance is what tells pyright which renderable it is.
        renderable = block._build()
        if isinstance(renderable, Text):
            rows.extend(renderable.plain.split("\n"))
    return rows


def _gave_up(detail: str, *, session_id: str = "remote-1") -> str:
    """The terminal sentence, spelled the way the app spells it.

    ONE concept, two spellings would drift: the app's copy names the session and
    the next step (UX U4) around whatever the last failure's own text was, so
    the test builds it from the same two pieces rather than pasting the whole
    string twice. The next step is its OWN authored row (design D2) — as one
    flowing paragraph the 80-column wrap split `/resume` from the id beside it.
    """
    return (
        f"could not resume {session_id} — {detail}.\n" f"Run /resume {session_id} again to retry."
    )


def _stopped(*, session_id: str = "remote-1") -> str:
    """The sentence a redial that never reached a verdict leaves behind.

    The cancellation exit. It names NO cause, on purpose: the sidebar's switch
    and Textual's own shutdown (`workers.cancel_all()` when the message loop
    ends) take the same arm, so a sentence claiming a switch can be false for a
    quit (review round 3, MINOR-1) — and the row is durable, so a false cause
    can outlive the process that wrote it.
    """
    return (
        f"resume of {session_id} stopped — the wait ended without connecting.\n"
        f"Run /resume {session_id} again to retry."
    )


def _static_refused(*, session_id: str = "remote-1") -> str:
    """The terminal sentence for a capability gap no redial can change (UX U2)."""
    return (
        f"could not resume {session_id} — the process hosting it does not serve the "
        "full TUI session state. Update or restart that process, then resume again."
    )


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
            await app._attach_or_refuse(tmp_path, "remote-1")

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
        # NOTHING AT ALL, which is the sharper form of the same claim: a row
        # restated into "the wait ended without connecting" also stops
        # starting with "reconnecting", and it would be FALSE here — the attach
        # succeeded. That is what the settled flag at the break is for.
        assert notices == [], notices


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
        await app._attach_or_refuse(tmp_path, "remote-1")

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
        await app._attach_or_refuse(tmp_path, "remote-1")
        first = _notices(app)
        await app._attach_or_refuse(tmp_path, "remote-1")

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

    def lookup(_root, concrete, **_probe):
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
            await app._attach_or_refuse(tmp_path, "remote-1")

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
    RULE, TWO CALLERS"), the arm reads the answer off the same record rather
    than inferring it from the exception.

    ONE DIAL IS SPENT FIRST, and that is the behaviour (review n5, QA Q3):
    `frontend_attach_refusal` is consulted only AFTER `connect` raises —
    `connect_calls=1 socket_dials=0`, on this head and on the pre-delta base —
    so the assertion is `len(dials) == 1`, not zero. Zero is the record
    BELOW the attach protocol, which the first-attempt guard refuses before any
    dial (see the neighbouring test). This docstring used to claim the refusal
    was taken "BEFORE spending even one dial" three lines above that assertion.

    The sentence is the app's own (UX U2), not the owner's internal one: the
    token it used to print (`owner lacks tui_state_v1; … protocol >= 5`) is not
    something the user can act on, and the arm had no session and no next step.
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
        await app._attach_or_refuse(tmp_path, "remote-1")

        assert len(dials) == 1, "a static refusal was retried"
        assert _notices(app) == [_static_refused()]
        assert not [
            n for n in _notices(app) if "tui_state_v1" in n
        ], "the owner's internal capability token reached the transcript"


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

    def lookup(_root, concrete, **_probe):
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
            await app._attach_or_refuse(tmp_path, "remote-1")

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

    THE ABSENCE IS ESTABLISHED HERE, NOT ASSUMED FROM THE TUPLE (review m3):
    this test's registry really is empty, so `dialable_record_exists` answers
    `False` and the refusal is the established one. Its sibling
    `test_an_owner_with_a_live_record_under_another_session_id_is_paced_not_refused`
    is the same `(None, pid)` input with a live record behind it, and it must NOT
    reach this sentence.
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
        await app._attach_or_refuse(tmp_path, "remote-1")

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
        await app._attach_or_refuse(tmp_path, "remote-1")

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
            await app._attach_or_refuse(tmp_path, "remote-1")

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
    while the wait itself had no stated end (design D2/D3/D4, UX U2).

    THE BOUND IS ANCHORED AS THE WHOLE WAIT'S, not a countdown (UX U4): said
    verbatim, "still trying for up to 42 s" at t=13.3 s of a 24.86 s wait reads
    as "42 s from now". "about … s in total" cannot be read that way, and the
    hedge is also what keeps the number true — one loop-top record read per pass
    is charged only after the dial that follows it, so the real worst case is
    the stated bound plus that read (review n4). The non-breaking space keeps
    `42 s` from splitting across a wrap in the narrower block a sidebar-open
    layout leaves (design D2); it is asserted so a copy edit cannot quietly
    drop it.
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
        await app._attach_or_refuse(tmp_path, "remote-1")

        rows = [n for n in seen[1] if n.startswith("reconnecting to session")]
        assert rows == [
            f"reconnecting to session remote-1 — still trying, about "
            f"{_SHIPPED_BOUND_S:g}\u00a0s in total"
        ]
        assert len(rows[0]) <= _NOTICE_MEASURE_CELLS, "the row splits in the notice measure"
        assert "owner" not in rows[0]
        assert "retry" not in rows[0]
        # The whole-operation anchor, and the unit that must not break from its
        # own number.
        assert "in total" in rows[0]
        assert f"{_SHIPPED_BOUND_S:g}\u00a0s" in rows[0]


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
        await app._attach_or_refuse(tmp_path, "remote-1")

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
        await app._attach_or_refuse(tmp_path, "remote-1")

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
        seen["during"] = list(_notices(app))
        raise ConnectionError("the runtime is not responding")

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app):
        app._session_transition_pending = True
        try:
            await app._attach_or_refuse(tmp_path, "remote-1")
        finally:
            app._session_transition_pending = False

        assert seen["target"] == "remote-1", "the redial never published its target"
        assert seen["blocked"] is True, "a submit the redial swallows was not refused"
        assert (
            "Send unavailable until connected. Reconnecting to session remote-1 — "
            "switching session stops the wait."
        ) in seen["during"], "the refused submit was silent"
        # The TARGET is named one way and the bound is NOT restated: the retry row
        # directly above is already carrying the number, and naming the session
        # two ways (`session <id>` / bare `<id>`) is what design D3 found.
        assert not [n for n in seen["during"] if "42" in n and n.startswith("Send")]
        # Cleared when the loop ends, so a later transition cannot inherit it.
        assert app._resume_retry_target == ""
        # AND THE REFUSAL ROW DID NOT OUTLIVE IT. The refusal speaks for a state
        # the loop owns, so the transcript after the verdict holds the verdict —
        # nothing underneath still claiming the dial is running (UX U1).
        assert _notices(app) == [_gave_up("the runtime is not responding")]
        assert app._composer_refusal_notice is None


@pytest.mark.asyncio
async def test_a_redial_cancelled_by_a_sidebar_switch_settles_its_row(monkeypatch, tmp_path):
    """The THIRD exit: a switch away leaves no live promise behind (D1/m4/Q2).

    Three independent round-2 streams found this one: `verdict()` restated the
    row on the give-up and static arms only, while the sidebar's switch cancels
    this worker group (`_select_sidebar_session`) — so the loop died on
    `CancelledError`, `finally` cleared `_resume_retry_target`, and the row went
    on saying "still trying" in the session the user returned to. One more stuck
    row per abandoned run, on the exit the new composer copy TEACHES.

    Driven the way the key drives it — `cancel_group(app, "session")`, the same
    edge the sidebar uses — with the loop parked inside a real dial, which is
    where a silent owner holds it for a full 15 s envelope.

    Asserted on the RENDERED notice, not only on the block list: QA's round-2
    repro showed a correct-looking `pending=False`/`retry_target=''` beside a row
    that was still on screen when the user came back (`repro_r3_stranded.py`,
    `stranded-back-in-origin.png`).
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")
    parked = asyncio.Event()
    dials: list[Any] = []

    async def connect(*_args, **_kwargs):
        dials.append(1)
        if len(dials) == 1:
            raise ConnectionError("attach refused")
        # The envelope of a live-but-silent owner: the loop is parked HERE when
        # the user switches away.
        parked.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app) as pilot:
        app._run_session_transition(app._attach_or_refuse(tmp_path, "remote-1"))
        for _ in range(200):
            await pilot.pause()
            if parked.is_set():
                break
        assert parked.is_set(), "the redial never reached its second dial"
        assert _notices(app) == [
            f"reconnecting to session remote-1 — still trying, about "
            f"{_SHIPPED_BOUND_S:g}\u00a0s in total"
        ]
        # And a refused Enter mid-wait, which writes the composer's own row: the
        # same abandon falsifies its forward-looking clause too, so BOTH rows
        # have to end with the wait (design D1 — "the composer's refusal row
        # needs the same treatment on that exit").
        app.composer_submission_refused()
        assert len(_notices(app)) == 2, "the refusal did not speak"

        # The sidebar's own abort. `cancel_group` is what a switch calls before
        # the navigation starts, so this is the real edge rather than a
        # hand-raised CancelledError.
        app.workers.cancel_group(app, "session")
        for _ in range(200):
            await pilot.pause()
            if not app._session_transition_pending:
                break

        assert len(dials) == 2, "the cancelled redial kept dialling"
        assert app._resume_retry_target == ""
        assert app._session_transition_pending is False
        # ONE settled statement, in the past tense: the row the loop narrated,
        # restated — not a second row under it and not the live one.
        assert _notices(app) == [_stopped()]
        rendered = _rendered_text(app)
        assert "still trying" not in rendered, "a dead redial is still promising a dial"
        # The AUTHORED statement, whitespace-collapsed: what the view paints is
        # the same sentence wrapped to the block's own measure, so the claim is
        # about the words on screen rather than about where the fold landed.
        assert " ".join(_stopped().split()) in " ".join(rendered.split())
        # Nothing here is a live promise: the composer answers as it does with no
        # redial running, because none is.
        assert app._unavailable_hint() == ""


@pytest.mark.asyncio
async def test_a_redial_cancelled_by_a_shutdown_names_no_cause(monkeypatch, tmp_path):
    """MINOR-1 (round 3): Textual's shutdown takes the same arm as a switch.

    `workers.cancel_all()` — which `_process_messages_loop`'s `finally` calls
    when the message loop ends — cancels this worker on a quit exactly as the
    sidebar's switch does, so the sentence that used to read "the wait ended when
    you switched session" was false for every shutdown. The row is durable, so a
    false cause could be read on the next launch beside a session the user never
    left. What is true of both is that the resume never completed, and the row
    now says that instead.

    Driven on the real shutdown edge (`cancel_all`, the call Textual itself
    makes) rather than by hand-raising a `CancelledError`, so the equivalence
    the copy now rests on is executed. The how-to-retry half is asserted too: a
    neutral cause must not cost the user their next step.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")
    parked = asyncio.Event()
    dials: list[Any] = []

    async def connect(*_args, **_kwargs):
        dials.append(1)
        if len(dials) == 1:
            raise ConnectionError("attach refused")
        parked.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app) as pilot:
        app._run_session_transition(app._attach_or_refuse(tmp_path, "remote-1"))
        for _ in range(200):
            await pilot.pause()
            if parked.is_set():
                break
        assert parked.is_set(), "the redial never reached its second dial"

        app.workers.cancel_all()
        for _ in range(200):
            await pilot.pause()
            if not app._session_transition_pending:
                break

        assert _notices(app) == [_stopped()]
        # The cause is not asserted, because the code cannot establish one: no
        # `switch`/`switched` anywhere on the rendered surface.
        assert "switch" not in _rendered_text(app).lower()
        assert "Run /resume remote-1 again to retry." in _rendered_text(app)


@pytest.mark.asyncio
async def test_a_second_abandonment_does_not_repeat_the_first_ones_sentence(monkeypatch, tmp_path):
    """D1 (round 3): one settled sentence per conversation, because the words
    carry no per-run information.

    The cancellation sentence names only the session, so a SECOND abandoned
    redial settled its fresh row into a byte-identical copy of the one already
    in the view — measured on the round-3 head as `notice rows on screen: 2`,
    regions `(12,14,75,3)` and `(12,18,75,3)`. Two adjacent identical blocks
    say nothing the first one did not, so the run contributes no second copy
    and the block that goes is its own live row.

    NOT the verdict case: `test_a_second_resume_does_not_strand_the_first_ones_row`
    keeps two identical VERDICTS (approved in rounds 1-2), and this narrowing
    matches on the settle text only.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")
    parked: list[asyncio.Event] = []
    dials: list[Any] = []

    async def connect(*_args, **_kwargs):
        dials.append(1)
        # Odd dials fail, so every run narrates a row before its wait; even ones
        # park where a switch finds the loop — the shape the `cancel2` rig drove.
        if len(dials) % 2:
            raise ConnectionError("attach refused")
        event = asyncio.Event()
        parked.append(event)
        event.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app) as pilot:
        previous: NoticeBlock | None = None
        for run in (1, 2):
            app._run_session_transition(app._attach_or_refuse(tmp_path, "remote-1"))
            for _ in range(200):
                await pilot.pause()
                if len(parked) == run:
                    break
            assert len(parked) == run, f"run {run} never reached its parked dial"
            assert _notices(app)[-1].startswith("reconnecting to session remote-1")
            app.workers.cancel_group(app, "session")
            for _ in range(200):
                await pilot.pause()
                if not app._session_transition_pending:
                    break
            if run == 1:
                # The row the first abandonment leaves behind. The second run
                # must settle INTO this block, not append a copy beside it
                # (design D1) — hence the identity check below.
                previous = _notice_blocks(app)[0]

        assert len(parked) == 2, "the second run never started, so D1 is untested"
        assert previous is not None, "the first run's settled row was never captured"
        rows = _rendered_notice_rows(app)
        # ONE settled sentence for the conversation, and it is the FIRST run's
        # own row, kept and restated rather than a second copy appended.
        assert _notice_blocks(app) == [previous]
        assert sum(1 for row in rows if "resume of remote-1 stopped" in row) == 1
        assert _notices(app) == [_stopped()]
        assert "still trying" not in _rendered_text(app)


@pytest.mark.asyncio
async def test_a_redial_cancelled_during_its_first_dial_invents_no_row(monkeypatch, tmp_path):
    """The settle restates the row the loop NARRATED; it never invents one.

    A `/resume` abandoned inside its very first silent envelope has nothing on
    screen yet, so there is nothing to correct — and the copy that would
    otherwise be written ("the wait ended without connecting") describes a wait
    the user never saw. The invariant is about promises made, not about exits
    taken.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")
    parked = asyncio.Event()
    dials: list[Any] = []

    async def connect(*_args, **_kwargs):
        dials.append(1)
        parked.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app) as pilot:
        app._run_session_transition(app._attach_or_refuse(tmp_path, "remote-1"))
        for _ in range(200):
            await pilot.pause()
            if parked.is_set():
                break
        assert parked.is_set()
        assert _notices(app) == [], "the first envelope narrated a row already"

        app.workers.cancel_group(app, "session")
        for _ in range(200):
            await pilot.pause()
            if not app._session_transition_pending:
                break

        assert len(dials) == 1
        assert _rendered_notice_rows(app) == []


@pytest.mark.asyncio
async def test_a_refused_enter_restates_one_row_and_never_outlives_the_verdict(
    monkeypatch, tmp_path
):
    """U1: the composer's row is ONE row, and it ends with the operation.

    Enter is the user's response to silence, so a durable notice per refusal made
    the count unbounded by construction: three Enters during one redial put
    three two-line blocks of present-tense text on screen, each retracting the
    sentence above it — and because `verdict()` restates the retry row IN PLACE,
    a refusal in the redial's last second was left BELOW the verdict, making the
    stale claim the last thing read. Seven refusals here, not three, so the check
    is about the mechanism rather than about a count that happened to fit.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")
    during: list[list[str]] = []
    refusals: list[int] = []

    async def connect(*_args, **_kwargs):
        if len(refusals) < 7:
            refusals.append(1)
            app.composer_submission_refused()
            during.append(_notices(app))
        raise ConnectionError("the runtime is not responding")

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app):
        app._session_transition_pending = True
        try:
            await app._attach_or_refuse(tmp_path, "remote-1")
        finally:
            app._session_transition_pending = False

        assert len(refusals) == 7, "the refusals did not all land in the redial"
        assert (
            len([n for n in during[-1] if n.startswith("Send unavailable")]) == 1
        ), f"one Enter, one row — got {during[-1]}"
        # THE VERDICT IS THE LAST THING READ, and the only notice left at all.
        assert _notices(app) == [_gave_up("the runtime is not responding")]
        assert "Send unavailable" not in "\n".join(_rendered_notice_rows(app))
        assert app._composer_refusal_notice is None


@pytest.mark.asyncio
async def test_an_owner_with_a_live_record_under_another_session_id_is_paced_not_refused(
    monkeypatch, tmp_path
):
    """review m3: `(None, pid)` is not only the publishes-no-record case.

    `find_runtime_record` also returns it for the rebind race — a record for that
    pid that is live, protocol-5 and perfectly dialable, still stamped with the
    PREVIOUS `session_id` (its own docstring calls the welcome projection the
    arbiter of that race). The first-attempt guard used to read the tuple as
    "this owner is an old binary", which claimed an age the code never checked
    and invited the user to close a healthy current-version runtime.

    Paced, not dialled: the loop re-reads the record each pass and the timeout's
    own sentence is the outcome, which is what a refusal that cannot name a
    cause should degrade to. The pause list is asserted, so this pins the PACING
    rather than merely the absence of the sentence.
    """
    app = _app(monkeypatch, tmp_path)
    clock = _RedialClock()
    clock.install(monkeypatch, wall_s=12.0)
    monkeypatch.setattr(app_module, "sidebar_connect_backoff_s", lambda _attempt: 5.0)
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (None, 91234),
    )
    # A live, dialable record for that pid exists under ANOTHER session_id.
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.dialable_record_exists",
        lambda _root, _pid: True,
    )

    async def explode(*_args, **_kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a paced absence dialled a record it never got")

    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", explode)

    async with _running(app):
        await app._attach_or_refuse(tmp_path, "remote-1")

        assert clock.pauses == [5.0, 5.0, 2.0], "the refusal's pacing was skipped"
        assert not [
            n for n in _notices(app) if "older Local Operator process" in n
        ], "the older-process sentence was printed for an owner the code never aged"
        assert _notices(app) == [_gave_up(UNREACHABLE_OWNER_MESSAGE)]


@pytest.mark.asyncio
async def test_an_unreadable_registry_is_paced_rather_than_refused(monkeypatch, tmp_path):
    """A failed read is not evidence of absence — so it cannot be a refusal.

    `dialable_record_exists` answers `None` when the registry cannot be read at
    all, which is why it is not a plain bool: the refusal's sentence names a
    cause ("open in an older Local Operator process"), and a scan that raised has
    established nothing about the process. The redial ends on the timeout's own
    honest sentence instead.
    """
    app = _app(monkeypatch, tmp_path)
    clock = _RedialClock()
    clock.install(monkeypatch, wall_s=12.0)
    monkeypatch.setattr(app_module, "sidebar_connect_backoff_s", lambda _attempt: 5.0)
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (None, 91234),
    )
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.dialable_record_exists",
        lambda _root, _pid: None,
    )

    async with _running(app):
        await app._attach_or_refuse(tmp_path, "remote-1")

        assert clock.pauses == [5.0, 5.0, 2.0]
        assert not [n for n in _notices(app) if "older Local Operator process" in n]
        assert _notices(app) == [_gave_up(UNREACHABLE_OWNER_MESSAGE)]


@pytest.mark.asyncio
async def test_a_wedged_owner_is_paced_rather_than_told_it_is_an_older_process(
    monkeypatch, tmp_path
):
    """review round 3 MINOR-2: `wedged` is not `absent`.

    The registry has a third state — the pid is alive and the heartbeat is older
    than `HEARTBEAT_TIMEOUT_S`, i.e. the owner is stuck — and it keeps that
    record for exactly the reason the redial exists: a stuck owner may recover
    on its own, which is the transient the budget outlasts. `dialable_record_exists`
    asked only for `live`, so a wedged owner answered `False` and earned the
    older-process sentence with the pacing skipped: a cause the code had not
    established, printed for a current-version process that a redial could have
    healed.

    Driven through the REAL registry and the REAL `.session.pid` marker rather
    than a stubbed classification of them, so the state under test is the one
    `scan` actually produces — and the instrument is asserted first, because a
    test that quietly drove the `live` branch would say nothing about this
    finding.
    """
    app = _app(monkeypatch, tmp_path)
    clock = _RedialClock()
    clock.install(monkeypatch, wall_s=12.0)
    monkeypatch.setattr(app_module, "sidebar_connect_backoff_s", lambda _attempt: 5.0)

    # A record for a REAL, live pid — this test process — aged past the
    # heartbeat timeout, plus the claim marker naming it. Both are the files
    # `find_runtime_record` reads in production.
    record = replace(_record(os.getpid(), "the remote"), session_id="remote-1")
    registry.publish(record, root=tmp_path)
    published = registry.record_path(record.pid, root=tmp_path)
    payload = json.loads(published.read_text())
    payload["heartbeat_at"] = time.time() - (HEARTBEAT_TIMEOUT_S + 5.0)
    published.write_text(json.dumps(payload))
    marker = tmp_path / "sessions" / "remote-1"
    marker.mkdir(parents=True, exist_ok=True)
    (marker / ".session.pid").write_text(str(record.pid))

    states = [state for found, state in registry.scan(tmp_path) if found.pid == record.pid]
    assert states == ["wedged"], "the record is not wedged, so nothing below is about MINOR-2"

    async def explode(*_args, **_kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a wedged owner's record is never handed to a dial")

    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", explode)

    async with _running(app):
        await app._attach_or_refuse(tmp_path, "remote-1")

        assert clock.pauses == [5.0, 5.0, 2.0], "the wedged owner's pacing was skipped"
        assert not [
            n for n in _notices(app) if "older Local Operator process" in n
        ], "the older-process sentence was printed for a process the code never aged"
        assert _notices(app) == [_gave_up(UNREACHABLE_OWNER_MESSAGE)]


def test_the_redial_outranks_a_latched_source_in_the_hint(monkeypatch, tmp_path):
    """UX U3: the composer answers for the state the app is actually in.

    `_unavailable_hint` is an ordered chain, and `self._interaction` is — for the
    current session — the very object `_connect_sidebar_session` writes, so a
    latched `connection_error` or a spent `connect_attempts` can sit on the
    object a live redial is asking about. Behind those two fields, a refusal
    mid-redial would answer "Select this session again to retry." while the retry
    row directly above says the app is already retrying and that reselection is
    not what is owed.

    UX could not DRIVE that combination (both attempts healed), so the failure
    itself stays unreproduced; what is pinned here is that the precedence no
    longer depends on the order two fields happen to be tested in. The retry is
    the more specific state — the app sets it for exactly the life of its loop.
    """
    app = _app(monkeypatch, tmp_path)
    app._session_transition_pending = True
    app._resume_retry_target = "remote-1"
    app._interaction.connection_error = "the runtime is not responding"
    app._interaction.connect_attempts = 3

    assert app._unavailable_hint() == (
        " Reconnecting to session remote-1 — switching session stops the wait."
    )

    # With no redial live, the latched state answers exactly as it did before.
    app._resume_retry_target = ""
    assert app._unavailable_hint() == " Select this session again to retry."

    app._interaction.connection_error = ""
    assert app._unavailable_hint() == " Reconnecting — it will keep trying for a few more seconds."


@pytest.mark.asyncio
async def test_the_verdict_keeps_its_next_step_on_one_row_at_eighty_columns(monkeypatch, tmp_path):
    """design D2: the actionable half must not be split from its argument.

    As one flowing paragraph the wrap fell wherever the cells landed, and at 80
    columns — a width the original rig never filmed — it broke between `/resume`
    and the session id the user has to type (measured 71.4 cells then 23.1),
    i.e. the half they must act on was the half that got split. A newline before
    the final sentence costs nothing at the widths where the row already wraps,
    and this pins the rendered result rather than the authored string: the whole
    sentence has to survive as ONE row of the block.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")

    async def connect(*_args, **_kwargs):
        raise ConnectionError("the runtime is not responding")

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with app.run_test(size=(80, 24)) as pilot:  # type: ignore[attr-defined]
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        await app._attach_or_refuse(tmp_path, "remote-1")

        rows = [row.strip() for row in _rendered_notice_rows(app)]
        assert (
            "Run /resume remote-1 again to retry." in rows
        ), f"the next step was split across rows: {rows}"
        # And it is still TWO authored statements, so the wider widths cannot
        # silently rejoin it into the paragraph this replaced.
        assert _gave_up("the runtime is not responding").count("\n") == 1


@pytest.mark.asyncio
async def test_a_resume_onto_a_live_owner_paints_first_and_binds_behind(monkeypatch, tmp_path):
    """PAINT FIRST, ATTACH BEHIND: a conversation on disk opens without a dial.

    The blocking `connect` held `/resume` on the owner's canonical sync before
    anything was drawn — 15 s and a refusal against a busy owner, and the whole
    redial budget (about 42 s) against a silent one. With a transcript on disk
    the conversation is adopted as a cold viewer instead and the ordinary eager
    engage binds it behind the paint, so this asserts the three halves: NO dial
    on the request path, the adopted session is that cold viewer, and the engage
    was started for it. The redial tests above keep the dial path (their root
    holds no transcript), which is the fallback this branch degrades to.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote")
    directory = tmp_path / "sessions" / "remote-1"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    dials: list[Any] = []
    colds: list[str] = []
    engaged: list[str] = []
    cold_viewer = FakeSession()

    async def connect(*_args, **_kwargs):
        dials.append("connect")
        return FakeSession()

    async def cold(session_id, **_kwargs):
        colds.append(session_id)
        return cold_viewer

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.cold", cold)
    monkeypatch.setattr(
        OperatorApp, "_engage_runtime_eagerly", lambda self: engaged.append("engage")
    )

    async with _running(app):
        engaged.clear()
        await app._attach_or_refuse(tmp_path, "remote-1")

        assert dials == [], "a conversation on disk must not wait on the owner's sync to paint"
        assert colds == ["remote-1"]
        assert app._session is cold_viewer
        assert engaged == ["engage"], "the bind behind the paint was never started"
        assert _notices(app) == [], "a paint-first open narrates no redial"


# --- The paint-first attach's own narration and bound (UX round 1, U1-U5) ---
#
# Paint-first opens a REAL cold viewer and the ordinary background engage binds
# it behind the paint. Against a frozen owner that engage says nothing until it
# fails (~45 s), so these drive the real `/resume` arm with a real
# `AttachedSession.cold` and a stand-in `engage_runtime` that parks until the
# test releases it — the frozen owner, without a process — and shortened
# narrate/bound timers.


class _FrozenOwner:
    """`engage_runtime` for an owner that does not answer until ``thaw()``."""

    def __init__(self) -> None:
        self.calls = 0
        self._gate = asyncio.Event()
        self.fail: BaseException | None = None

    def thaw(self, *, fail: BaseException | None = None) -> None:
        self.fail = fail
        self._gate.set()

    async def __call__(self, *_args, **_kwargs) -> None:
        self.calls += 1
        await self._gate.wait()
        if self.fail is not None:
            raise self.fail


async def _paint_first(monkeypatch, tmp_path, frozen: _FrozenOwner) -> OperatorApp:
    from local_operator.config import ConfigManager

    ConfigManager(config_dir=tmp_path).update_config({"hosting": "test", "model_name": "mock"})
    directory = tmp_path / "sessions" / "frozen-1"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (_record(90909, "frozen"), 90909),
    )
    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", frozen)
    monkeypatch.setattr(app_module, "ATTACH_BEHIND_NARRATE_S", 0.2)
    # Room between the two marks, because the tests read "still trying" and
    # the verdict as separate states: at 0.6 s, a host at load ~200 fired both
    # timers inside one `pilot.pause()`, and the narration was never observable
    # (the round-2 suite run, 2 of these tests, pre-existing on `63bc96f4`).
    monkeypatch.setattr(app_module, "ATTACH_BEHIND_BOUND_S", 3.0)
    return _app(monkeypatch, tmp_path)


async def _pump(pilot, predicate, *, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause()
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


@pytest.mark.asyncio
async def test_a_frozen_owner_is_narrated_then_judged_not_left_starting(monkeypatch, tmp_path):
    """U1: the wait is told something true, on a schedule, and ends in a verdict.

    Before: the band read `starting…` alone for 42 s and the first sentence came
    at ~45 s (measured in a real pty). The redial's own "still trying, about N s"
    row now arrives at the narrate mark, and the bound restates it as the owner
    not answering and ends the pending state rather than staying optimistic.
    """
    frozen = _FrozenOwner()
    app = await _paint_first(monkeypatch, tmp_path, frozen)
    async with _running(app) as pilot:
        await app._attach_or_refuse(tmp_path, "frozen-1")
        # The engage is a worker: it reaches `engage_runtime` a loop turn later.
        assert await _pump(pilot, lambda: frozen.calls == 1), "the attach never started"
        assert app._starting_shown, "the band did not say the attach is pending"
        assert await _pump(
            pilot, lambda: any("still trying" in n for n in _notices(app))
        ), _notices(app)
        assert await _pump(
            pilot, lambda: any("is not answering" in n for n in _notices(app))
        ), _notices(app)
        rows = [n for n in _notices(app) if "frozen-1" in n]
        assert len(rows) == 1, ("the verdict must restate the narration, not add a row", rows)
        assert "no runtime yet" not in rows[0]
        assert not app._starting_shown, "`starting…` outlived the verdict"
        frozen.thaw(fail=ConnectionError("gone"))
        await _pump(pilot, lambda: not app._warm_engage_started)


@pytest.mark.asyncio
async def test_a_bind_landing_inside_the_bound_retires_the_narration(monkeypatch, tmp_path):
    """U3: the "still trying" row does not outlive the attach it described."""
    frozen = _FrozenOwner()
    app = await _paint_first(monkeypatch, tmp_path, frozen)
    monkeypatch.setattr(app_module, "ATTACH_BEHIND_BOUND_S", 30.0)
    async with _running(app) as pilot:
        await app._attach_or_refuse(tmp_path, "frozen-1")
        assert await _pump(pilot, lambda: any("still trying" in n for n in _notices(app)))
        # The owner answers: the facade leaves cold the way a real bind does.
        monkeypatch.setattr(type(app._session), "is_cold", property(lambda _self: False))
        frozen.thaw()
        assert await _pump(pilot, lambda: not app._starting_shown)
        assert not [n for n in _notices(app) if "frozen-1" in n], _notices(app)


@pytest.mark.asyncio
async def test_a_bind_landing_after_the_verdict_restates_it(monkeypatch, tmp_path):
    """U3: a late answer restates the verdict rather than leaving it standing."""
    frozen = _FrozenOwner()
    app = await _paint_first(monkeypatch, tmp_path, frozen)
    async with _running(app) as pilot:
        await app._attach_or_refuse(tmp_path, "frozen-1")
        assert await _pump(pilot, lambda: any("is not answering" in n for n in _notices(app)))
        monkeypatch.setattr(type(app._session), "is_cold", property(lambda _self: False))
        frozen.thaw()
        assert await _pump(
            pilot, lambda: any("is answering again" in n for n in _notices(app))
        ), _notices(app)
        assert not any("is not answering" in n for n in _notices(app))


@pytest.mark.asyncio
async def test_a_prompt_sent_before_the_attach_keeps_the_pending_cue(monkeypatch, tmp_path):
    """U4: `starting…` survives the prompt being accepted, until its send returns.

    The background engage yields to the prompt's foreground bind and clears its
    own flag; the band used to flip to `working` in that frame while nothing had
    reached the owner.
    """
    frozen = _FrozenOwner()
    app = await _paint_first(monkeypatch, tmp_path, frozen)
    monkeypatch.setattr(app_module, "ATTACH_BEHIND_BOUND_S", 30.0)
    release = asyncio.Event()

    async def parked_prompt(self, *_args, **_kwargs):  # noqa: ANN001
        await release.wait()
        return ""

    async with _running(app) as pilot:
        await app._attach_or_refuse(tmp_path, "frozen-1")
        monkeypatch.setattr(type(app._session), "prompt", parked_prompt)
        # The engage surrenders, which is what cleared the cue before.
        app._set_starting(False)
        app._start_turn_for(app._interaction, "sent while attaching")
        await pilot.pause()
        assert app._starting_shown, "the pending cue vanished when the prompt was accepted"
        band = app._status
        assert band is not None and band._starting and band._streaming
        assert "starting…" in band.render_text(100).plain
        release.set()
        assert await _pump(pilot, lambda: not app._starting_shown)
        frozen.thaw(fail=ConnectionError("gone"))


# --- A message sent during the paint-first wait (UX round 2, U6/U8) ---
#
# The message's own FOREGROUND bind preempts the background engage, which then
# fails by design. Before: the watch settled on that failure (no row, no bound,
# no verdict for as long as the bind took), and when the bind itself failed —
# "owner did not send its state", the welcome read against a frozen owner — the
# generic branch printed that transport sentence, left the echo standing as if
# sent and did not give the text back: the message was gone, 4/4 in UX round 2.


def _user_rows(app: OperatorApp) -> list[str]:
    from local_operator.tui.widgets.transcript import UserBlock

    views = list(app.query(TranscriptView))
    if not views:
        return []
    return [block.text() for block in views[0].blocks() if isinstance(block, UserBlock)]


async def _compose_and_send(pilot, app: OperatorApp, text: str) -> None:  # noqa: ANN001
    """Type into the REAL composer and press enter: the echo row and the accepted
    draft are painted by the submit path, which is the half U6 lost."""
    from local_operator.tui.widgets.editor import Editor

    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    editor.text = text
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


async def _send_during_attach(  # noqa: ANN001
    monkeypatch, tmp_path, frozen, prompt, *, bound_s: float = 2.0
):
    """`/resume` a frozen owner, then SUBMIT a message while it is still pending.

    ``prompt`` stands in for the facade's send, so the test controls what the
    message's own bind does. The engage is failed the way a preempted
    background engage fails — the moment the prompt is in flight.
    """
    app = await _paint_first(monkeypatch, tmp_path, frozen)
    # AFTER `_paint_first`, which sets its own; and before the attach, because
    # the watch arms its timers from the module values when it is created. The
    # narration and the verdict are read as separate states, so the bound sits
    # well past the narrate mark on a loaded host.
    monkeypatch.setattr(app_module, "ATTACH_BEHIND_BOUND_S", bound_s)
    ctx = _running(app)
    pilot = await ctx.__aenter__()
    await app._attach_or_refuse(tmp_path, "frozen-1")
    assert await _pump(pilot, lambda: frozen.calls == 1)
    monkeypatch.setattr(type(app._session), "prompt", prompt)
    await _compose_and_send(pilot, app, "sent while attaching")
    # The background engage yields to the foreground bind and fails.
    frozen.thaw(fail=ConnectionError("preempted"))
    await _pump(pilot, lambda: not app._warm_engage_started)
    return app, pilot, ctx


@pytest.mark.asyncio
async def test_a_message_sent_early_in_the_wait_keeps_the_narration(monkeypatch, tmp_path):
    """U6: a preempted engage must not silence the wait while the attach is pending."""
    frozen = _FrozenOwner()
    release = asyncio.Event()

    async def parked(self, *_a, **_k):  # noqa: ANN001
        await release.wait()
        raise ConnectionError("owner did not send its state")

    # EVENT-BASED, not a wall-clock window (QA round 3, Q-1): at 0.2 s / 2.0 s a
    # loaded host's single pump could straddle both marks, so the narration was
    # never observed as its own state. The bound is parked far away, the
    # narration is awaited by predicate, and the bound is then FIRED — the
    # attempt's own timer callback — so the verdict is a second observed state.
    app, pilot, ctx = await _send_during_attach(
        monkeypatch, tmp_path, frozen, parked, bound_s=600.0
    )
    try:
        # The engage has FAILED; the narration and the bound must still come.
        assert not app._warm_engage_started, "the fixture must fail the engage first"
        assert await _pump(pilot, lambda: any("still trying" in n for n in _notices(app))), (
            "the preempted engage silenced the narration",
            _notices(app),
        )
        (attempt,) = app._attach_behind_attempts.values()
        assert attempt.phase == "narrated" and attempt._timers, "the bound is not armed"
        attempt._judge()
        assert await _pump(
            pilot, lambda: any("is not answering" in n for n in _notices(app))
        ), _notices(app)
        assert not app._starting_shown, "`starting…` outlived the verdict"
        release.set()
        assert await _pump(pilot, lambda: any("was not sent" in n for n in _notices(app)))
        # Each fact is stated ONCE: the session's verdict (the narration row
        # restated, never a second copy) and the message's failure notice (its
        # row stays; the notice carries `send again` / `edit`). The narration
        # must not linger beside them.
        texts = _notices(app)
        assert not [n for n in texts if "still trying" in n], texts
        assert len([n for n in texts if "is not answering" in n]) == 1, texts
        assert len([n for n in texts if "did not answer" in n]) == 1, (
            "one account per failed message",
            texts,
        )
    finally:
        await ctx.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_a_message_whose_bind_fails_during_the_wait_keeps_its_row(monkeypatch, tmp_path):
    """U6/U8 under the boundary rule: never delivered ⇒ the row stays, the fate is told.

    SUPERSEDES "echo down, draft back" (the reviewed preference this change
    reverses): the message the user can see is not erased, and the payload is
    reached through the notice's own verbs rather than by refilling the
    composer.
    """
    frozen = _FrozenOwner()

    async def refused(self, *_a, **_k):  # noqa: ANN001
        raise ConnectionError("owner did not send its state")

    app, pilot, ctx = await _send_during_attach(monkeypatch, tmp_path, frozen, refused)
    try:
        assert await _pump(pilot, lambda: any("was not sent" in n for n in _notices(app)))
        from local_operator.tui.widgets.editor import Editor

        assert app.query_one(Editor).text.strip() == "", "the composer was refilled"
        assert _user_rows(app) == ["sent while attaching"], "the row was withdrawn"
        texts = _notices(app)
        assert not any("owner did not send its state" in n for n in texts), texts
        assert not app._starting_shown
        # A second message against the same silent owner is a NEW send with its
        # own record: its row and notice join the first, never restate it.
        await _compose_and_send(pilot, app, "again")
        assert await _pump(pilot, lambda: not app._interaction.active_workers)
        assert _user_rows(app) == ["sent while attaching", "again"], _user_rows(app)
        assert len([n for n in _notices(app) if "was not sent" in n]) == 2, _notices(app)
    finally:
        await ctx.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_a_message_whose_bind_lands_settles_the_narration(monkeypatch, tmp_path):
    """U6: the preempting message's successful bind settles the wait it took over."""
    frozen = _FrozenOwner()
    release = asyncio.Event()

    async def delivered(self, *_a, **_k):  # noqa: ANN001
        await release.wait()
        monkeypatch.setattr(type(self), "is_cold", property(lambda _self: False))
        return ""

    app, pilot, ctx = await _send_during_attach(
        monkeypatch, tmp_path, frozen, delivered, bound_s=30.0
    )
    try:
        assert await _pump(pilot, lambda: any("still trying" in n for n in _notices(app)))
        release.set()
        assert await _pump(pilot, lambda: not app._attach_behind_attempts)
        assert await _pump(pilot, lambda: not [n for n in _notices(app) if "frozen-1" in n])
        assert _user_rows(app) == ["sent while attaching"], "a delivered message lost its row"
    finally:
        await ctx.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_the_paint_first_verdict_keeps_its_next_step_whole_at_eighty_columns(
    monkeypatch, tmp_path
):
    """U9: at an 80-column pane "send a message to retry" split across two rows.

    Measured in UX round 2 on the one-sentence verdict. The next step is now its
    own authored row, the redial give-up's pattern, and this pins the RENDERED
    rows at 80 columns rather than the authored string.
    """
    frozen = _FrozenOwner()
    app = await _paint_first(monkeypatch, tmp_path, frozen)
    async with app.run_test(size=(80, 24)) as pilot:  # type: ignore[attr-defined]
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        await app._attach_or_refuse(tmp_path, "frozen-1")
        assert await _pump(pilot, lambda: any("is not answering" in n for n in _notices(app)))
        rows = [row.strip() for row in _rendered_notice_rows(app)]
        assert "Send a message to retry." in rows, rows
        frozen.thaw(fail=ConnectionError("gone"))
        await _pump(pilot, lambda: not app._warm_engage_started)


@pytest.mark.asyncio
async def test_the_paint_first_narration_quotes_its_own_bound(monkeypatch, tmp_path):
    """U7/D5: the row said "about 42 s in total" and was judged at 27 s."""
    frozen = _FrozenOwner()
    app = await _paint_first(monkeypatch, tmp_path, frozen)
    monkeypatch.setattr(app_module, "ATTACH_BEHIND_BOUND_S", 27.0)
    async with _running(app) as pilot:
        await app._attach_or_refuse(tmp_path, "frozen-1")
        assert await _pump(pilot, lambda: any("still trying" in n for n in _notices(app)))
        row = next(n for n in _notices(app) if "still trying" in n)
        assert "about 27\u00a0s in total" in row, row
        assert f"{app_module.RESUME_CONNECT_BOUND_S:g}\u00a0s" not in row, row
        frozen.thaw(fail=ConnectionError("gone"))
        await _pump(pilot, lambda: not app._warm_engage_started)


# --- The same wait across conversation SWITCHES (UX round 3, U10-U12) ---
#
# Every step of the return path used to decide WHICH conversation and WHICH echo
# row it was repairing from the screen: the withdrawal took the newest submit's
# rows out of the view in front, the "back in the composer" row was painted into
# whatever conversation was showing, and the narration watch bailed the moment
# the binding epoch moved. These drive the REAL sidebar prepare/commit between a
# paint-first conversation (a real `AttachedSession.cold` behind `/resume`) and
# ordinary sidebar conversations, and assert per conversation.


def _view_rows(view: TranscriptView) -> tuple[list[str], list[str]]:
    """(user row texts, notice texts) of ONE transcript, whichever is on screen."""
    from local_operator.tui.widgets.transcript import UserBlock

    users = [b.text() for b in view.blocks() if isinstance(b, UserBlock)]
    notices = [b._text for b in view.blocks() if isinstance(b, NoticeBlock)]
    return users, notices


def _returned_rows(notices: list[str]) -> list[str]:
    # Matched on the stable FACT of the sentence (the owner did not answer the
    # message), not its wording, so the arms discriminate on behaviour: the
    # pre-fix head's copy ("…is back in the composer") matches too.
    return [n for n in notices if "did not answer" in n]


class _HeldSend:
    """`prompt` for the paint-first facade: each send parks until the test settles it."""

    def __init__(self) -> None:
        self.gates: list[asyncio.Event] = []
        self.outcomes: list[BaseException | None] = []
        self.delivered: list[str] = []

    def install(self, monkeypatch, session) -> None:  # noqa: ANN001
        held = self

        async def prompt(self_, text, *_a, **_k):  # noqa: ANN001
            gate = asyncio.Event()
            held.gates.append(gate)
            held.outcomes.append(None)
            index = len(held.gates) - 1
            await gate.wait()
            outcome = held.outcomes[index]
            if outcome is not None:
                raise outcome
            held.delivered.append(text)
            return ""

        monkeypatch.setattr(type(session), "prompt", prompt)

    def refuse_all(self) -> None:
        for index, gate in enumerate(self.gates):
            if not gate.is_set():
                self.outcomes[index] = ConnectionError("owner did not send its state")
                gate.set()


async def _to_sidebar(app: OperatorApp, pilot, remote) -> None:  # noqa: ANN001
    from tests.unit.tui.test_sidebar_swap_reset import _switch

    await _switch(app, pilot, remote)


def _sidebar_remote(name: str):  # noqa: ANN202
    from tests.unit.tui.test_sidebar_swap_reset import SidebarRemote, _message

    return SidebarRemote(
        name,
        history=[_message("user", f"{name} question"), _message("assistant", f"{name} answer")],
    )


@asynccontextmanager
async def _paint_first_with_sends(monkeypatch, tmp_path, *, bound_s: float = 30.0):
    """A paint-first `/resume` of a frozen owner, with sends the test settles."""
    frozen = _FrozenOwner()
    app = await _paint_first(monkeypatch, tmp_path, frozen)
    monkeypatch.setattr(app_module, "ATTACH_BEHIND_BOUND_S", bound_s)
    sends = _HeldSend()
    async with _running(app) as pilot:
        await app._attach_or_refuse(tmp_path, "frozen-1")
        assert await _pump(pilot, lambda: frozen.calls >= 1)
        owner = app._session
        sends.install(monkeypatch, owner)
        try:
            yield app, pilot, owner, sends, frozen
        finally:
            sends.refuse_all()
            frozen.thaw(fail=ConnectionError("gone"))
            for _ in range(10):
                await pilot.pause()


async def _back_to(app: OperatorApp, pilot, owner) -> None:  # noqa: ANN001
    """Return to the paint-first conversation through the same sidebar commit."""
    await _to_sidebar(app, pilot, owner)
    assert app._session is owner, "the switch back did not land on the sending conversation"


@pytest.mark.asyncio
async def test_a_send_then_a_switch_away_and_back_keeps_one_row_and_its_verdict(
    monkeypatch, tmp_path
):
    """Arm 1: send, switch away, come back; the return lands where the message was sent.

    Before: the round trip moved the binding epoch, the watch bailed on every
    step, and the conversation was left with a sent row, a "still trying" that
    never judged, and the text in the composer with no account (U10/U12).
    """
    async with _paint_first_with_sends(monkeypatch, tmp_path, bound_s=3.0) as ctx:
        app, pilot, owner, sends, _frozen = ctx
        await _compose_and_send(pilot, app, "GONE-A")
        assert await _pump(pilot, lambda: len(sends.gates) == 1)
        await _to_sidebar(app, pilot, _sidebar_remote("side-b"))
        await _back_to(app, pilot, owner)
        # The bound is the attempt's: it still lands, in THIS conversation.
        assert await _pump(
            pilot,
            lambda: any("is not answering" in n for n in _view_rows(app._transcript_view())[1]),
        ), _view_rows(app._transcript_view())
        sends.refuse_all()
        from local_operator.tui.widgets.editor import Editor

        assert await _pump(
            pilot, lambda: bool(_returned_rows(_view_rows(app._transcript_view())[1]))
        )
        users, notices = _view_rows(app._transcript_view())
        assert users.count("GONE-A") == 1, (
            "the row of a failed send left its own transcript (superseded preference)",
            users,
        )
        # The session's verdict and the message's notice, each once.
        assert len([n for n in notices if "is not answering" in n]) == 1, notices
        assert len(_returned_rows(notices)) == 1, notices
        assert app.query_one(Editor).text.strip() == "", "the payload returned by itself"
        # The row's own verb against an owner that is STILL silent: `send again`
        # retires the failed row+notice and resubmits in ONE handler (J1), so
        # the transcript still shows exactly one row for the message. The second
        # bind hangs on the still-frozen owner (its gate was created AFTER
        # `refuse_all`, so it is open) — the pending state, with no failure
        # notice yet and the payload back nowhere.
        editor = app.query_one(Editor)
        from local_operator.tui.session_presentation import SendFailureNotice

        notice_row = next(
            b
            for b in app._transcript_view().blocks()
            if isinstance(b, SendFailureNotice) and "did not answer" in b._text
        )
        notice_row.focus()
        await pilot.pause()
        await pilot.press("enter")
        assert await _pump(
            pilot, lambda: len(sends.gates) == 2
        ), "the resend never reached the owner"
        users, notices = _view_rows(app._transcript_view())
        assert users.count("GONE-A") == 1, ("the resend stacked a row", users)
        assert not _returned_rows(notices), ("a retired offer was left standing", notices)
        assert editor.text.strip() == "", editor.text


@pytest.mark.asyncio
async def test_two_sends_then_a_switch_keep_both_rows_in_their_own_view(monkeypatch, tmp_path):
    """Arm 2: two sends inside the window, a switch, both keep their rows (U10 repro B)."""
    async with _paint_first_with_sends(monkeypatch, tmp_path) as ctx:
        app, pilot, owner, sends, _frozen = ctx
        view_a = app._transcript_view()
        await _compose_and_send(pilot, app, "TWO-1")
        assert await _pump(pilot, lambda: len(sends.gates) == 1)
        await _compose_and_send(pilot, app, "TWO-2")
        assert await _pump(pilot, lambda: _view_rows(view_a)[0].count("TWO-2") == 1)
        side = _sidebar_remote("side-b")
        await _to_sidebar(app, pilot, side)
        view_b = app._transcript_view()
        assert view_b is not view_a
        sends.refuse_all()
        # The second send's bind runs once the first releases the provider lock.
        assert await _pump(pilot, lambda: len(sends.gates) == 2)
        sends.refuse_all()
        source_a = app._sidebar_sources["frozen-1"]
        assert await _pump(pilot, lambda: not source_a.active_workers)
        users_a, notices_a = _view_rows(view_a)
        assert users_a.count("TWO-1") == 1 and users_a.count("TWO-2") == 1, (
            "a kept row of a failed send was withdrawn from its own view",
            users_a,
        )
        users_b, notices_b = _view_rows(view_b)
        assert not _returned_rows(notices_b), ("the other conversation was told", notices_b)
        await _back_to(app, pilot, owner)
        users, notices = _view_rows(app._transcript_view())
        assert users.count("TWO-1") == 1 and users.count("TWO-2") == 1, users
        assert len(_returned_rows(notices)) == 2, ("one notice per failed message", notices)
        from local_operator.tui.widgets.editor import Editor

        assert app.query_one(Editor).text.strip() == "", "a payload returned by itself"
        assert source_a.unsent == [], "a failure must not park as a returned draft"


@pytest.mark.asyncio
async def test_two_failed_sends_then_a_delivered_resend_leave_one_row(monkeypatch, tmp_path):
    """U10 repro C / QA Q-2 restated: a delivered resend replaces its failed row.

    Two sends fail under one owner; each keeps its row and its notice. The
    resend goes out ONCE — from a notice's `send again`, which retires that
    failed row in the same handler that resubmits — so the delivered message
    has one row and the other failure is untouched.
    """
    async with _paint_first_with_sends(monkeypatch, tmp_path) as ctx:
        app, pilot, _owner, sends, _frozen = ctx
        from local_operator.tui.session_presentation import SendFailureNotice

        await _compose_and_send(pilot, app, "TWO-1")
        assert await _pump(pilot, lambda: len(sends.gates) == 1)
        await _compose_and_send(pilot, app, "TWO-2")
        sends.refuse_all()
        assert await _pump(pilot, lambda: len(sends.gates) == 2)
        sends.refuse_all()
        assert await _pump(pilot, lambda: not app._interaction.active_workers)
        users, notices = _view_rows(app._transcript_view())
        assert users.count("TWO-1") == 1 and users.count("TWO-2") == 1, users
        assert len(_returned_rows(notices)) == 2, notices

        # The first failure's notice: `send again` retires its row and resubmits.
        first = next(b for b in app._transcript_view().blocks() if isinstance(b, SendFailureNotice))
        sends.gates.clear()
        sends.outcomes.clear()
        first.focus()
        await pilot.pause()
        await pilot.press("enter")
        assert await _pump(pilot, lambda: len(sends.gates) == 1)
        sends.gates[0].set()
        assert await _pump(pilot, lambda: [d.strip() for d in sends.delivered] == ["TWO-1"])
        users, notices = _view_rows(app._transcript_view())
        assert users.count("TWO-1") == 1 and users.count("TWO-2") == 1, (
            "one delivered message, one row; the other failure stays",
            users,
        )
        assert len(_returned_rows(notices)) == 1, notices


@pytest.mark.asyncio
async def test_a_return_while_another_conversation_is_shown_lands_in_its_own(monkeypatch, tmp_path):
    """Arm 3: the refusal fires while a DIFFERENT conversation is on screen (U11)."""
    async with _paint_first_with_sends(monkeypatch, tmp_path) as ctx:
        app, pilot, owner, sends, _frozen = ctx
        view_a = app._transcript_view()
        await _compose_and_send(pilot, app, "GONE-H")
        assert await _pump(pilot, lambda: len(sends.gates) == 1)
        await _to_sidebar(app, pilot, _sidebar_remote("side-b"))
        from local_operator.tui.widgets.editor import Editor

        sends.refuse_all()
        source_a = app._sidebar_sources["frozen-1"]
        assert await _pump(pilot, lambda: not source_a.active_workers)
        users_b, notices_b = _view_rows(app._transcript_view())
        assert not _returned_rows(notices_b), (
            "painted into the conversation switched TO",
            notices_b,
        )
        assert app.query_one(Editor).text.strip() == "", "the other composer was filled"
        assert "GONE-H" in _view_rows(view_a)[0], "the row left its parked view"
        assert source_a.draft.text.strip() == "", "a failure must not park a draft"
        await _back_to(app, pilot, owner)
        users, notices = _view_rows(app._transcript_view())
        assert users.count("GONE-H") == 1, users
        assert len(_returned_rows(notices)) == 1, notices
        assert app.query_one(Editor).text.strip() == "", "the payload returned by itself"


@pytest.mark.asyncio
async def test_a_second_and_third_conversation_never_see_the_return(monkeypatch, tmp_path):
    """Arm 4: switch to a second, then a third conversation; the return stays home."""
    async with _paint_first_with_sends(monkeypatch, tmp_path) as ctx:
        app, pilot, owner, sends, _frozen = ctx
        await _compose_and_send(pilot, app, "GONE-3")
        assert await _pump(pilot, lambda: len(sends.gates) == 1)
        await _to_sidebar(app, pilot, _sidebar_remote("side-b"))
        view_b = app._transcript_view()
        await _to_sidebar(app, pilot, _sidebar_remote("side-c"))
        view_c = app._transcript_view()
        sends.refuse_all()
        source_a = app._sidebar_sources["frozen-1"]
        assert await _pump(pilot, lambda: not source_a.active_workers)
        for view in (view_b, view_c):
            assert not _returned_rows(_view_rows(view)[1]), _view_rows(view)
        await _back_to(app, pilot, owner)
        users, notices = _view_rows(app._transcript_view())
        assert users.count("GONE-3") == 1 and len(_returned_rows(notices)) == 1, (users, notices)


@pytest.mark.asyncio
async def test_a_row_narrated_while_away_reaches_its_verdict_after_the_switch_back(
    monkeypatch, tmp_path
):
    """Arm 5 (U12): away across the narrate mark, back before the bound.

    Before: the switch back re-bound the facade, the watch's `_current()` was
    false from then on, and "still trying, about 27 s in total" stood 19 s past
    its own bound with nothing in its place.
    """
    async with _paint_first_with_sends(monkeypatch, tmp_path, bound_s=3.0) as ctx:
        app, pilot, owner, _sends, _frozen = ctx
        started = time.monotonic()
        await _to_sidebar(app, pilot, _sidebar_remote("side-b"))
        # Across the narrate mark while away: nothing about it in B.
        assert await _pump(pilot, lambda: time.monotonic() - started > 0.5)
        assert not [n for n in _view_rows(app._transcript_view())[1] if "frozen-1" in n]
        await _back_to(app, pilot, owner)
        assert await _pump(
            pilot,
            lambda: any("is not answering" in n for n in _view_rows(app._transcript_view())[1]),
        ), ("the re-narrated row never reached its verdict", _view_rows(app._transcript_view()))
        rows = [n for n in _view_rows(app._transcript_view())[1] if "frozen-1" in n]
        assert len(rows) == 1, ("one account of the wait", rows)
        assert not app._starting_shown, "`starting…` beside the verdict"


@pytest.mark.asyncio
async def test_the_failure_notices_verbs_survive_an_eighty_column_wrap(monkeypatch, tmp_path):
    """U13 applied to the new notice: the verbs the user acts on stay readable."""
    frozen = _FrozenOwner()

    async def refused(self, *_a, **_k):  # noqa: ANN001
        raise ConnectionError("owner did not send its state")

    app = await _paint_first(monkeypatch, tmp_path, frozen)
    async with app.run_test(size=(80, 24)) as pilot:  # type: ignore[attr-defined]
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        await app._attach_or_refuse(tmp_path, "frozen-1")
        assert await _pump(pilot, lambda: frozen.calls >= 1)
        monkeypatch.setattr(type(app._session), "prompt", refused)
        await _compose_and_send(pilot, app, "NARROW-E")
        assert await _pump(pilot, lambda: any("was not sent" in n for n in _notices(app)))
        rows = [row.strip() for row in _rendered_notice_rows(app)]
        joined = " ".join(rows)
        # The sentence may wrap; the state and the verbs must survive it as
        # words. (OQ3: the balance of the two halves is the design round's call;
        # what is pinned here is that neither half disappears behind a wrap.)
        assert "your message was not sent" in joined, rows
        assert "send again \u23ce · edit e" in joined, rows
        frozen.thaw(fail=ConnectionError("gone"))
        await _pump(pilot, lambda: not app._warm_engage_started)


# --- The account OUTLIVES the attempt (agent review round 4, F-1; UX U14; QA Q-1) ---
#
# Every switch arm above asserts BEFORE any bind lands, which is why they all
# passed while the account could still be lost: the row's only holder was the
# attempt, which ends when a carrier binds, and the view it was painted into is
# replaced when the switch back's connect re-commits the conversation. These
# arms land a bind at each of the three points that matter and assert that the
# account is either on screen or genuinely no longer owed — never silently gone.


class _Coldness:
    """`is_cold` for the paint-first facade, moved by the test the way a bind moves it.

    Only the paint-first facade's class is patched; the sidebar doubles are
    `SidebarRemote`s and keep their own answer.
    """

    def __init__(self, monkeypatch, session) -> None:  # noqa: ANN001
        self.cold = True
        state = self
        monkeypatch.setattr(type(session), "is_cold", property(lambda _self: state.cold))


@pytest.mark.asyncio
async def test_a_landing_between_the_capture_and_the_return_still_accounts_for_it(
    monkeypatch, tmp_path
):
    """F-1a: the engage binds, the facade resyncs, THEN the message's bind fails.

    Before (``c09c0608``): ``landed`` closed the attempt, so the return found it
    "not alive" and said nothing — the text went back to the composer with no
    row and no notice at all. ``3424719f`` painted the row; this pins that.
    """
    async with _paint_first_with_sends(monkeypatch, tmp_path) as ctx:
        app, pilot, owner, sends, frozen = ctx
        from local_operator.tui.widgets.editor import Editor

        coldness = _Coldness(monkeypatch, owner)
        await _compose_and_send(pilot, app, "SILENT")
        assert await _pump(pilot, lambda: len(sends.gates) == 1)
        # The engage binds the attach: the attempt is over.
        coldness.cold = False
        frozen.thaw()
        assert await _pump(pilot, lambda: not app._attach_behind_attempts)
        # ...and the facade is cold again (a resync: `not _ready_for_events`)
        # when the message's own bind is refused.
        coldness.cold = True
        sends.refuse_all()
        source = app._interaction
        assert await _pump(pilot, lambda: not source.active_workers)
        users, notices = _view_rows(app._transcript_view())
        assert "SILENT" in users, ("the row of a failed send was withdrawn", users)
        assert app.query_one(Editor).text.strip() == "", "the composer was filled"
        assert len(_returned_rows(notices)) == 1, (
            "the send failed with no account of how it ended",
            notices,
        )


@pytest.mark.asyncio
async def test_a_landing_after_the_row_is_painted_keeps_it_through_the_recommit(
    monkeypatch, tmp_path
):
    """F-1b / QA Q-1: the switch back's connect re-commits a fresh view on the bind.

    Before: the row lived only in the widget it was painted into, and the
    connect worker's re-commit swapped that view out as soon as the owner
    answered — taking the "is answering again" restatement with it (U3 says a
    row that stated an outcome is restated, never silently removed).
    """
    async with _paint_first_with_sends(monkeypatch, tmp_path) as ctx:
        app, pilot, owner, sends, frozen = ctx
        from local_operator.tui.widgets.editor import Editor

        coldness = _Coldness(monkeypatch, owner)
        await _compose_and_send(pilot, app, "GONE-R")
        assert await _pump(pilot, lambda: len(sends.gates) == 1)
        await _to_sidebar(app, pilot, _sidebar_remote("side-b"))
        sends.refuse_all()
        source_a = app._sidebar_sources["frozen-1"]
        assert await _pump(pilot, lambda: not source_a.active_workers)
        await _back_to(app, pilot, owner)
        before = app._transcript_view()
        assert len(_returned_rows(_view_rows(before)[1])) == 1, _view_rows(before)
        # The owner answers: the engage binds and the switch back's connect
        # worker re-commits the conversation over a fresh replay. The failure
        # notice must survive the swap — projected into whatever view the
        # conversation has in front (`_sync_send_failure_notices` at commit) —
        # never left behind with the replaced view.
        coldness.cold = False
        frozen.thaw()
        assert await _pump(pilot, lambda: not app._attach_behind_attempts)
        assert await _pump(
            pilot, lambda: len(_returned_rows(_view_rows(app._transcript_view())[1])) == 1
        ), (
            "the failure notice left with the replaced view",
            _view_rows(app._transcript_view()),
        )
        assert app.query_one(Editor).text.strip() == "", "the payload returned by itself"


@pytest.mark.asyncio
async def test_the_owner_answering_while_the_user_is_away_is_shown_on_return(monkeypatch, tmp_path):
    """U14: the account of a returned message, when the owner answers while away.

    Before: the bind restated the row in the parked view (or the attempt closed
    first), and the conversation came back without it — the text sat in the
    composer with nothing saying how it got there, or that the owner is back.
    """
    async with _paint_first_with_sends(monkeypatch, tmp_path) as ctx:
        app, pilot, owner, sends, frozen = ctx
        from local_operator.tui.widgets.editor import Editor

        coldness = _Coldness(monkeypatch, owner)
        await _compose_and_send(pilot, app, "GONE-W")
        assert await _pump(pilot, lambda: len(sends.gates) == 1)
        sends.refuse_all()
        source_a = app._interaction
        assert await _pump(pilot, lambda: not source_a.active_workers)
        assert len(_returned_rows(_view_rows(app._transcript_view())[1])) == 1
        await _to_sidebar(app, pilot, _sidebar_remote("side-b"))
        view_b = app._transcript_view()
        # The owner answers while the user is away.
        coldness.cold = False
        frozen.thaw()
        assert await _pump(pilot, lambda: not app._attach_behind_attempts)
        assert not [n for n in _view_rows(view_b)[1] if "frozen-1" in n], _view_rows(view_b)
        # The return is served by a REBUILT presentation, not the parked one:
        # the cache misses whenever the conversation changed while away (a
        # bind that landed is such a change), which is the shape UX walked in
        # a real pty. A row that lived only in the parked widget is gone then.
        parked = app._sidebar_presentations.pop("frozen-1", None)
        assert parked is not None, "precondition: the conversation was parked"
        await _back_to(app, pilot, owner)
        assert app._transcript_view() is not parked.replay.view
        for _ in range(20):
            await pilot.pause()
        # The rebuilt view gets the notice projected into it (the record rides
        # the interaction; `_sync_send_failure_notices` runs at this commit).
        # The row's own re-seed is the following slice.
        assert len(_returned_rows(_view_rows(app._transcript_view())[1])) == 1, (
            "the conversation came back without its failure notice",
            _view_rows(app._transcript_view()),
        )
        assert app.query_one(Editor).text.strip() == "", "the payload returned by itself"


@pytest.mark.asyncio
async def test_a_retired_conversation_leaves_nothing_in_the_attempt_table(monkeypatch, tmp_path):
    """m-1: a judged attempt whose conversation is then retired does not linger."""
    async with _paint_first_with_sends(monkeypatch, tmp_path, bound_s=0.6) as ctx:
        app, pilot, _owner, _sends, _frozen = ctx
        assert await _pump(pilot, lambda: any("is not answering" in n for n in _notices(app)))
        source = app._interaction
        assert source.token in app._attach_behind_attempts
        app._retire_attach_behind(source)
        assert source.token not in app._attach_behind_attempts
        assert source.attach_behind_account is None


# --- A bind that is NOT one of the attempt's carriers settles the account too ---
# (UX round 5, U15 and its item 4; QA round 5, Q-1/Q-2.) The arms above release
# the bind through the attempt's own warm engage, which is a carrier, so they
# restate the row on every head. In the real product the carriers have long
# failed by the time the owner answers after a switch back, and the bind is
# delivered by the SIDEBAR'S OWN CONNECT (`_start_sidebar_connection` →
# `bind_runtime` → re-commit), which reported to nothing: the row stayed
# "did not answer — send it again" for the rest of the session, above the reply
# the resend produced.


def _refreeze(frozen: _FrozenOwner) -> None:
    """Every engage from now on waits again, like an owner that is still silent."""
    frozen._gate = asyncio.Event()
    frozen.fail = None


async def _returned_then_back_with_carriers_spent(
    app, pilot, owner, sends, frozen, text
):  # noqa: ANN001, ANN202
    """Send, switch away, the message comes back, switch back; no carrier is left."""
    await _compose_and_send(pilot, app, text)
    assert await _pump(pilot, lambda: len(sends.gates) == 1)
    await _to_sidebar(app, pilot, _sidebar_remote("side-b"))
    sends.refuse_all()
    source = app._sidebar_sources["frozen-1"]
    assert await _pump(pilot, lambda: not source.active_workers)
    await _back_to(app, pilot, owner)
    # The switch back's engage fails like the real one does against a silent
    # owner (its own envelope), so NO carrier of the attempt is left alive.
    frozen.thaw(fail=ConnectionError("gone"))
    assert await _pump(pilot, lambda: not app._warm_engage_started)
    _refreeze(frozen)
    for _ in range(5):
        await pilot.pause()
    assert len(_returned_rows(_view_rows(app._transcript_view())[1])) == 1
    return source


@pytest.mark.asyncio
async def test_a_bind_by_the_sidebar_connect_after_a_switch_back_settles_the_account(
    monkeypatch, tmp_path
):
    """U15 / Q-1: the switch back's own connect binds; the row says the owner is back.

    Before (``91b361dc``): only a CARRIER's bind reached the account, so the
    connect bound, re-committed, and the row kept "did not answer — send it
    again" with the owner answering. The settled row must then survive further
    view swaps, exactly once each time.
    """
    async with _paint_first_with_sends(monkeypatch, tmp_path) as ctx:
        app, pilot, owner, sends, frozen = ctx
        coldness = _Coldness(monkeypatch, owner)
        source = await _returned_then_back_with_carriers_spent(
            app, pilot, owner, sends, frozen, "GONE-C"
        )
        # The owner answers, and the bind is the SIDEBAR CONNECT's, not a carrier's.
        app._start_sidebar_connection(source)
        coldness.cold = False
        frozen.thaw()

        def rows() -> list[str]:
            return [n for n in _view_rows(app._transcript_view())[1] if "frozen-1" in n]

        assert await _pump(pilot, lambda: any("is answering again" in n for n in rows())), (
            "a bind that was not a carrier left the outcome row unsettled",
            rows(),
        )
        # Two statements, one each: the account's session-level outcome and the
        # message's failure notice. Neither may duplicate, and neither may be
        # erased by the bind's re-commit.
        assert len([n for n in rows() if "is answering again" in n]) == 1, rows()
        assert len(_returned_rows(rows())) == 1, rows()
        # Survives the next swaps, one copy each time.
        for _ in range(2):
            await _to_sidebar(app, pilot, _sidebar_remote("side-b"))
            await _back_to(app, pilot, owner)
            assert len([n for n in rows() if "is answering again" in n]) == 1, rows()
            assert len(_returned_rows(rows())) == 1, rows()


@pytest.mark.asyncio
async def test_a_delivered_resend_leaves_no_failure_row_standing(monkeypatch, tmp_path):
    """UX round 5, item 4: following "send it again" after the owner came back.

    The facade is bound by a path that is not a carrier and does not re-commit
    the conversation, so the resend goes out over an already-bound facade and is
    never an ``_AttachBehindSend``. Before (``91b361dc``): delivered exactly
    once, and the row above it still said the message was not sent and to send
    it again, over an empty composer.
    """
    async with _paint_first_with_sends(monkeypatch, tmp_path) as ctx:
        app, pilot, owner, sends, frozen = ctx
        from local_operator.tui.widgets.editor import Editor

        coldness = _Coldness(monkeypatch, owner)
        source = await _returned_then_back_with_carriers_spent(
            app, pilot, owner, sends, frozen, "GONE-D"
        )
        coldness.cold = False
        source.display_only = False
        sends.gates.clear()
        sends.outcomes.clear()
        from local_operator.tui.session_presentation import SendFailureNotice

        # The resend is the notice's own `send again` (the payload is not in the
        # composer under the boundary rule).
        notice = next(
            b for b in app._transcript_view().blocks() if isinstance(b, SendFailureNotice)
        )
        notice.focus()
        await pilot.pause()
        await pilot.press("enter")
        assert await _pump(pilot, lambda: len(sends.gates) == 1)
        sends.gates[0].set()
        assert await _pump(pilot, lambda: [d.strip() for d in sends.delivered] == ["GONE-D"])
        for _ in range(5):
            await pilot.pause()
        users, notices = _view_rows(app._transcript_view())
        assert users.count("GONE-D") == 1, users
        assert not _returned_rows(notices), (
            "a failure notice stands above the delivered message",
            notices,
        )
        assert app.query_one(Editor).text.strip() == ""


@pytest.mark.asyncio
async def test_retiring_a_conversation_takes_its_row_off_the_view(monkeypatch, tmp_path):
    """QA round 5, Q-2: the row goes with the account, even detached from its ancestry."""
    async with _paint_first_with_sends(monkeypatch, tmp_path, bound_s=600.0) as ctx:
        app, pilot, _owner, _sends, _frozen = ctx
        assert await _pump(pilot, lambda: any("still trying" in n for n in _notices(app)))
        source = app._interaction
        view = app._transcript_view()
        # QA's `test_c3` end state: the table and the account were cleared,
        # and the row stood on the view with nothing left to take it down.
        assert [n for n in _view_rows(view)[1] if "frozen-1" in n]
        app._retire_attach_behind(source)
        for _ in range(3):
            await pilot.pause()
        assert not [n for n in _view_rows(view)[1] if "frozen-1" in n], _view_rows(view)
        assert source.attach_behind_account is None
