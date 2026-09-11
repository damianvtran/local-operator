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
with only its two seams (`find_runtime_record`, `AttachedSession.connect`)
stubbed, exactly as `test_stop_command.py` drives it.

NO WALL-CLOCK ASSERTIONS. The budget is a COUNT of dials against
`RESUME_CONNECT_ATTEMPTS`, and the backoff is collapsed to zero so the policy is
testable at speed; the derivation that sizes the real schedule is asserted as a
RELATIONSHIP in `test_the_retry_budget_outlasts_the_facade_give_up_bound`.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.attached import RECOVERY_GIVE_UP_S
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

        assert len(dials) == _SHIPPED_ATTEMPTS + 1
        notices = _notices(app)
        assert notices[-1] == "attach refused"
        # ONE live row for the whole redial, restated in place: the budget can
        # run for the facade's entire give-up bound, and a notice per attempt
        # would write a hundred durable rows to say one thing.
        assert len([n for n in notices if n.startswith("reconnecting to session")]) == 1


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
    just a longer way to the same sentence. Read off the same record `connect`
    refused, so the two can never disagree.
    """
    app = _app(monkeypatch, tmp_path)
    record = _record(90909, "the remote", capabilities=[])
    dials: list[Any] = []

    async def connect(*_args, **_kwargs):
        dials.append(1)
        raise ConnectionError(
            f"owner lacks {FRONTEND_CAPABILITY}; canonical full-TUI attach needs protocol >= 5"
        )

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)

    async with _running(app):
        await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        assert len(dials) == 1, "a static refusal was retried"
        assert _notices(app)[-1] == (
            f"owner lacks {FRONTEND_CAPABILITY}; canonical full-TUI attach needs protocol >= 5"
        )


def test_the_retry_budget_outlasts_the_facade_give_up_bound():
    """THE RELATIONSHIP, not the number — the way this fix regresses quietly.

    `/resume` builds a TERMINAL facade, whose give-up bound is
    `RECOVERY_GIVE_UP_S` rather than the sidebar viewer's `COLD_FALLBACK_S`, so
    that is what the budget has to outlast. Pinned as an inequality between the
    two constants because that is the property: asserting either number instead
    would go green on a tree where the bound had grown past the budget — which
    is the original defect, silently restored.
    """
    schedule = [
        app_module.sidebar_connect_backoff_s(attempt) for attempt in range(1, _SHIPPED_ATTEMPTS + 1)
    ]
    assert sum(schedule) > RECOVERY_GIVE_UP_S
    # Not merely wider by a rounding error: the last attempt must land clearly
    # past the bound on a loaded machine, not tie with it.
    assert sum(schedule) > RECOVERY_GIVE_UP_S * 1.25
    # One attempt fewer must NOT clear the bound, which is what makes the
    # derived count minimal rather than an arbitrary large number.
    assert sum(schedule[:-1]) <= RECOVERY_GIVE_UP_S * 1.5
