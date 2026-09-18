"""The viewer-watch signal is chrome and must never take the app down.

`_note_viewer_watching` tells a session's owner whether this terminal is still
displaying it, so the owner's notification routing can stop counting a
switched-away viewer as somebody watching. It runs on both switch edges, which
makes its failure mode the important part: `run_worker` defaults to
`exit_on_error=True`, so a raise from the AWAITED coroutine arrives as
`WorkerFailed` at the app's exception handler and ends the TUI. A `try` around
the call cannot see that, because the call only schedules.

Both triggers are ordinary rather than exotic -- an owner too old to know the
op answers with an error frame, and a connection dropped mid-switch raises on
the write -- so a staged rollout of the op is itself the mixed-version case
that would crash a switch (review round 1, F1).

These tests drive `_note_viewer_watching` ITSELF rather than a hand-rolled
analog of its call shape. An earlier version demonstrated Textual's
`exit_on_error` mechanism on a throwaway app and passed with the production
guard reverted, which made it evidence about Textual rather than about this
method (review round 2, F2).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import Mock

import pytest

from local_operator.tui.app import OperatorApp


def _app_with_capture() -> tuple[OperatorApp, list[Any], list[Any]]:
    """An unbooted app that records what it schedules and what killed it.

    No Textual app is booted for the same reason
    ``test_sidebar_idle_reap`` does not: the method under test touches its
    argument and ``run_worker``, and booting the harness would trade a precise
    assertion for a timing one. ``run_worker`` is replaced with the real
    signature so the test can assert on ``exit_on_error`` AND still await the
    coroutine the method built -- which is where the guard being tested lives.
    """
    app = OperatorApp.__new__(OperatorApp)
    scheduled: list[Any] = []
    died: list[Any] = []

    def _run_worker(work: Any, *, exclusive: bool = False, exit_on_error: bool = True, **kw: Any):
        scheduled.append({"work": work, "exit_on_error": exit_on_error})
        return None

    # `_handle_exception` is where a WorkerFailed lands, i.e. the thing that
    # ends the app. Recording it is how these tests tell "swallowed" from
    # "would have killed the TUI" without booting a real worker manager.
    object.__setattr__(app, "run_worker", _run_worker)
    object.__setattr__(app, "_handle_exception", lambda error: died.append(error))
    return app, scheduled, died


def _source(*, raising: bool) -> Mock:
    """A `SessionInteraction` stub whose owner client answers the op or fails.

    Shaped as the method reads it (`source.session._client.viewer_watch`) so a
    rename anywhere on that path fails this test rather than silently skipping
    the signal -- the `getattr` chain returns None on a miss, which is a valid
    no-op the method must keep for older owners.
    """

    async def _ok(*, displaying: bool) -> str:
        return "viewer watching" if displaying else "viewer away"

    async def _boom(*, displaying: bool) -> str:
        raise RuntimeError("unknown op: viewer_watch")

    source = Mock()
    source.session._client.viewer_watch = _boom if raising else _ok
    return source


@pytest.mark.asyncio
async def test_a_failing_viewer_watch_never_reaches_the_app() -> None:
    """The real method's coroutine swallows its own failure.

    Awaits the coroutine `_note_viewer_watching` actually built, so the inner
    guard is what is under test. Without it this await raises and the raise is
    what Textual would turn into `WorkerFailed`.
    """
    app, scheduled, died = _app_with_capture()

    OperatorApp._note_viewer_watching(app, _source(raising=True), displaying=False)

    assert len(scheduled) == 1, "the switch edge must still signal the owner"
    await scheduled[0]["work"]  # must not raise
    assert died == [], "a failed viewer-watch signal must never reach the app"


@pytest.mark.asyncio
async def test_the_worker_is_spawned_with_errors_disarmed() -> None:
    """Belt to the inner guard's braces, and the half a refactor would drop.

    Either mechanism alone suppresses the crash, so asserting only the
    behaviour above would let a future edit delete this flag unnoticed.
    """
    app, scheduled, _ = _app_with_capture()

    OperatorApp._note_viewer_watching(app, _source(raising=True), displaying=False)

    assert scheduled[0]["exit_on_error"] is False


@pytest.mark.asyncio
async def test_the_signal_carries_the_edge_it_was_given() -> None:
    """Both switch edges reach the owner with the right claim."""
    seen: list[bool] = []

    async def _record(*, displaying: bool) -> str:
        seen.append(displaying)
        return "ok"

    for edge in (False, True):
        app, scheduled, died = _app_with_capture()
        source = Mock()
        source.session._client.viewer_watch = _record
        OperatorApp._note_viewer_watching(app, source, displaying=edge)
        await scheduled[0]["work"]
        assert died == []

    assert seen == [False, True]


@pytest.mark.asyncio
async def test_an_owner_without_the_op_is_a_silent_no_op() -> None:
    """An older owner exposes no `viewer_watch`, which must schedule nothing.

    The `getattr` chain returning None is the compatibility path: that build
    counted every attach anyway, so there is nothing to tell it.
    """
    app, scheduled, died = _app_with_capture()
    source = Mock()
    source.session._client = object()  # no viewer_watch attribute

    OperatorApp._note_viewer_watching(app, source, displaying=False)

    assert scheduled == []
    assert died == []


@pytest.mark.asyncio
async def test_a_scheduling_failure_is_also_swallowed() -> None:
    """`run_worker` itself can raise (a torn-down app), and that is chrome too."""
    app = OperatorApp.__new__(OperatorApp)
    died: list[Any] = []

    def _explode(*_a: Any, **_k: Any):
        raise RuntimeError("app is shutting down")

    object.__setattr__(app, "run_worker", _explode)
    object.__setattr__(app, "_handle_exception", lambda error: died.append(error))

    OperatorApp._note_viewer_watching(app, _source(raising=False), displaying=True)

    assert died == []
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_the_claim_is_remembered_on_the_facade_for_a_redial() -> None:
    """The away claim outlives the connection that carried it.

    The claim is per CONNECTION: a fresh socket defaults to displaying, so a
    parked source that redialed would resume suppressing its owner's
    parked-gate notification. `AttachedSession._attach` re-asserts from this
    field, which only helps if the field is written here (independent review
    round 4, F1b).
    """
    app, scheduled, _ = _app_with_capture()
    source = _source(raising=False)

    OperatorApp._note_viewer_watching(app, source, displaying=False)
    assert source.session._viewer_displaying is False
    await scheduled[0]["work"]

    OperatorApp._note_viewer_watching(app, source, displaying=True)
    assert source.session._viewer_displaying is True


@pytest.mark.asyncio
async def test_the_claim_is_recorded_even_with_no_client_to_send_it() -> None:
    """An away claim raised while the socket is down must still be remembered.

    This is precisely the window the redial re-assert covers, so recording the
    claim only when a client exists would lose the one case that needs it.
    """
    app, scheduled, died = _app_with_capture()
    source = Mock()
    source.session._client = None

    OperatorApp._note_viewer_watching(app, source, displaying=False)

    assert source.session._viewer_displaying is False
    assert scheduled == [], "there is no connection to send it on"
    assert died == []
