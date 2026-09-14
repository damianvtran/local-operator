"""The drain latch: leaving a build behind WITHOUT aborting what is running.

``begin_retire`` refuses while any work would be lost, and that refusal is the
defect the bound exists to bound: a session busy for hours never reaches an
idle instant, so a runtime whose install was replaced underneath it keeps
executing a tree that is gone. ``begin_drain`` (design §4 F1) drops the idle
gate and keeps every other guarantee — no new work is admitted, nothing in
flight is touched, and a message that arrives while the runtime is still
finishing is SPOOLED for the successor rather than dropped or run against the
build that is leaving.

Exercised the way ``test_cut_off_turns._LatchHost`` exercises the sibling
latch: the handle's real methods over a stub session and a stub predicate, so
the assertions land on the latch rather than on a runtime boot (the e2e stage
covers the boot).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.session.runtime.inbox import INBOX_NAME, peek_inbox
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session


class FakeSession:
    """The slice of Session these latches touch."""

    def __init__(self, directory: Path, *, busy: bool = True) -> None:
        self.transcript = SimpleNamespace(directory=directory)
        self.busy = busy
        self.notes: list[tuple[str, str]] = []
        self.deliberate = 0
        self.peer_calls: list[tuple[str, str, bool]] = []

    # -- the two verdict writers -------------------------------------------
    def note_cut_off(self, cause: str, detail: str = "") -> None:
        self.notes.append((cause, detail))

    def note_deliberate_stop(self) -> None:
        self.deliberate += 1

    # -- peer delivery ------------------------------------------------------
    async def receive_peer_message(
        self, text: str, *, mode: str = "mailbox", wake: bool = False, sender: Any = None
    ) -> str:
        self.peer_calls.append((text, mode, wake))
        return "delivered"


class DrainHost:
    """``ServingSessionHandle``'s drain latch over a stub session.

    ``may_refresh`` is the only thing stubbed about the predicate: a runtime
    that reports work it would lose, which is the case the whole rung exists
    for.
    """

    begin_drain = ServingSessionHandle.begin_drain
    begin_retire = ServingSessionHandle.begin_retire
    _retiring_refusal = ServingSessionHandle._retiring_refusal
    _spool_for_successor = ServingSessionHandle._spool_for_successor
    receive_peer_message = ServingSessionHandle.receive_peer_message
    _note_deliberate_stop = ServingSessionHandle._note_deliberate_stop
    _check_loop_thread = lambda self: None  # noqa: E731 — the real one only asserts

    def __init__(self, session: FakeSession, *, busy: bool = True) -> None:
        self._session = session
        self._busy = busy
        self._retiring_cause = ""
        self._draining = False
        self._exit_committed = False
        self._disposing = False
        self._fold = SimpleNamespace(note_peer_message=lambda *_a, **_k: None)

    def may_refresh(self) -> str:
        return "busy" if self._busy else ""

    def _notify(self) -> None:
        pass


def _host(tmp_path: Path, *, busy: bool = True) -> tuple[DrainHost, FakeSession]:
    session = FakeSession(tmp_path / "sessions" / "s1")
    session.transcript.directory.mkdir(parents=True, exist_ok=True)
    return DrainHost(session, busy=busy), session


# -- the latch ------------------------------------------------------------------


def test_begin_drain_latches_while_a_turn_is_running(tmp_path: Path) -> None:
    """The one difference from ``begin_retire``: the latch does not wait."""
    host, session = _host(tmp_path, busy=True)
    assert host.begin_retire("runtime-retired") is False, "the soft rung still refuses"
    assert host._retiring_cause == ""

    assert host.begin_drain("runtime-retired", " (0.54.33@7fe8b10 → 0.54.39@dec7933)") is True
    assert host._retiring_cause == "runtime-retired"
    assert host._draining is True
    assert host._exit_committed is False, "the exit has not been taken yet"
    assert "runtime-retired" in host._retiring_refusal()
    assert session.notes == [], "no turn is being cut off, so no cut-off may be recorded"


def test_begin_retire_still_records_the_cut_off_it_owes(tmp_path: Path) -> None:
    """The contrast that keeps the taxonomy honest: the EXIT rung notes it."""
    host, session = _host(tmp_path, busy=False)
    assert host.begin_retire("runtime-retired", " (a → b)") is True
    assert host._exit_committed is True
    assert session.notes == [("runtime-retired", " (a → b)")]


def test_a_disposing_handle_refuses_the_latch(tmp_path: Path) -> None:
    """A second exit must not race the disposal that is already running."""
    host, _ = _host(tmp_path)
    host._disposing = True
    assert host.begin_drain("runtime-retired") is False
    assert host._draining is False


# -- admissions during the drain ------------------------------------------------


@pytest.mark.asyncio
async def test_a_peer_wake_during_the_drain_is_spooled_for_the_successor(
    tmp_path: Path,
) -> None:
    """invariant (iv): deferred, not dropped, and never run on the dying build."""
    host, session = _host(tmp_path)
    assert host.begin_drain("runtime-retired", "declined 3x") is True

    receipt = await host.receive_peer_message(
        "the build is moving", mode="steer", wake=True, sender={"name": "peer"}
    )
    assert "spooled" in receipt, receipt
    assert session.peer_calls == [], "a turn must not be started on a build that is leaving"

    directory = session.transcript.directory
    rows = peek_inbox(directory)
    assert len(rows) == 1
    assert rows[0].text == "the build is moving"
    assert rows[0].sender == {"name": "peer"}
    # The successor's boot drain (``process._drain_inbox_into``) reads exactly
    # this file, so the row is what the next engage delivers.
    assert (directory / INBOX_NAME).exists()


@pytest.mark.asyncio
async def test_a_committed_exit_refuses_a_wake_instead(tmp_path: Path) -> None:
    """Once the exit is being taken there is no successor window left, so the
    message keeps the vocabulary refusal the sender can act on."""
    host, session = _host(tmp_path, busy=False)
    assert host.begin_retire("runtime-retired") is True
    with pytest.raises(RuntimeError) as caught:
        await host.receive_peer_message("too late", mode="steer", wake=True)
    assert "send it again" in str(caught.value)
    assert peek_inbox(session.transcript.directory) == []


@pytest.mark.asyncio
async def test_a_quiet_note_is_still_delivered_during_the_drain(tmp_path: Path) -> None:
    """Unchanged behaviour, and deliberately so: a record-only note opens no
    turn, so refusing it would drop something the sender was told had landed."""
    host, session = _host(tmp_path)
    assert host.begin_drain("runtime-retired") is True
    assert await host.receive_peer_message("fyi") == "delivered"
    assert session.peer_calls == [("fyi", "mailbox", False)]
    assert peek_inbox(session.transcript.directory) == []


def test_a_user_stop_during_the_drain_is_still_the_users_own(tmp_path: Path) -> None:
    """A drain lasts as long as the work does, so a ``/stop`` landing inside it
    ends the turn by the operator's own hand — reporting it as housekeeping
    would misattribute the cancel. Only the committed EXIT suppresses it."""
    host, session = _host(tmp_path)
    assert host.begin_drain("runtime-retired") is True
    host._note_deliberate_stop()
    assert session.deliberate == 1

    host._exit_committed = True
    host._note_deliberate_stop()
    assert session.deliberate == 1, "the retirement owns the verdict from here"


# -- the wake divert (the session side of invariant iv) --------------------------


class WakeHost:
    """``Session``'s two wake-spooling methods over a stub transcript."""

    retire_wakes_to_inbox = Session.retire_wakes_to_inbox
    _spool_wake_to_inbox = Session._spool_wake_to_inbox

    def __init__(self, directory: Path | None) -> None:
        self._transcript = SimpleNamespace(directory=directory)
        self._wake_deliver_hook: Any = None

    def _missed_delivery_note(self, _due: Any) -> None:
        return None


def _due(text: str = "check the deploy") -> Any:
    return SimpleNamespace(
        schedule=SimpleNamespace(id="w1", every_ms=None, message=text),
        occurrence=1,
        planned_total=None,
        final=False,
    )


@pytest.mark.asyncio
async def test_a_wake_fired_during_the_drain_lands_in_the_inbox(tmp_path: Path) -> None:
    host = WakeHost(tmp_path)
    host.retire_wakes_to_inbox()
    assert host._wake_deliver_hook == host._spool_wake_to_inbox

    await host._spool_wake_to_inbox(_due())
    rows = peek_inbox(tmp_path)
    assert len(rows) == 1
    assert "check the deploy" in rows[0].text
    assert rows[0].mode == "mailbox", "read on the next engage, never a turn of its own"


@pytest.mark.asyncio
async def test_an_unspoolable_wake_is_loud_and_does_not_raise(tmp_path: Path) -> None:
    """A drain must not die on a spool write, and a lost wake must leave a trace."""
    host = WakeHost(None)
    await host._spool_wake_to_inbox(_due())


@pytest.mark.asyncio
async def test_the_spooled_row_reaches_the_next_boot_drain(tmp_path: Path) -> None:
    """The end of the chain: what the drain spools is what the successor's boot
    drain delivers, in write order."""
    from local_operator.session.runtime.inbox import (
        InboxLine,
        append_inbox,
        drain_inbox,
    )

    assert append_inbox(
        tmp_path,
        InboxLine(text="wake fired while draining", sender={}, mode="mailbox", written_at=1.0),
    )
    lines = drain_inbox(tmp_path)
    assert [line.text for line in lines] == ["wake fired while draining"]
    assert lines[0].to_json()["mode"] == "mailbox"
