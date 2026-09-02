"""The kill switch (``session/runtime/control.py``) — one implementation of stop.

The contract these tests pin, from design §12:

- the escalation ladder: graceful ``stop`` op → identity-confirmed SIGTERM →
  SIGKILL, with a REFUSAL when identity cannot be confirmed ahead of a signal;
- one target vocabulary shared with `send` (delegated to
  ``resolve_peer_target`` — tested in ``tests/unit/mobile/test_peer_send.py``);
- wakes go DORMANT (``stopped_at`` stamped into the index entry), never
  deleted, and reopening clears the stamp — the semantics
  ``tests/unit/wakes/test_session_index.py`` established at the store level;
- the refusal exits cleanly with ``already exited`` when the pid died under us.

These run against REAL sockets where it matters (identity confirmation, the
graceful op) — a mocked stream cannot exhibit the welcome-first sequencing the
identity check depends on. Signals are exercised against real child processes
only in the live evidence for the PR; here the ladder's signal rungs are
exercised through the monkeypatched seams the runtime cannot avoid exposing
(``os.kill``, pid liveness).
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from local_operator.session.runtime import control, registry
from local_operator.session.runtime.types import SessionRecord
from tests.unit.session.runtime.test_server import FakeHandle, _wait_record


def _record_for(server_record: SessionRecord, **overrides: Any) -> SessionRecord:
    """A resolvable record for a live runtime: pid/port/key/session from it."""
    fields = {
        "pid": server_record.pid,
        "kind": "tui",
        "session_id": server_record.session_id,
        "conversation_name": server_record.conversation_name,
        "cwd": server_record.cwd,
        "model_label": server_record.model_label,
        "control_port": server_record.control_port,
        "control_key": server_record.control_key,
    }
    fields.update(overrides)
    return SessionRecord(**fields)


async def _serve(handle: FakeHandle | None = None):
    """Start an in-process runtime the way test_server's suites do."""
    from local_operator.session.runtime.server import RuntimeServer

    server = RuntimeServer(handle or FakeHandle(), kind="tui")
    await server.start_in_process()
    record = await _wait_record()
    return server, record


class _NeverStopsHandle(FakeHandle):
    """A handle that ignores the stop hook — the 'runtime will not exit' case.

    Models an old runtime (the op is unknown to it) or a wedged one: the
    graceful rung gets no ack it can use, so the ladder must escalate. The
    ``stop`` dispatch raises by leaving ``request_stop`` undefined.
    """


class _StoppingHandle(FakeHandle):
    """A handle whose ``request_stop`` "ends the process" — observably.

    The runtime under test is IN-PROCESS, so its record's pid is the test
    runner's own pid. A hook that really exited would kill pytest, and a
    hook that did nothing would let the ladder escalate to a real SIGTERM
    against the runner (which is exactly how the first draft of this file
    died with exit 143). So the hook flips a flag, and the ``no_signals``
    fixture makes ``pid_alive`` read that flag instead of the process table.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stops: list[bool] = []
        self.exited = False

    def request_stop(self) -> None:
        self.stops.append(True)
        self.exited = True


@pytest.fixture
def no_signals(monkeypatch: pytest.MonkeyPatch):
    """Never let the ladder signal the test process; make liveness follow
    the handle's ``exited`` flag. Returns the list of real signals sent
    (always empty on a correct ladder) and a hook to bind the handle."""
    sent: list[tuple[int, int]] = []
    state: dict[str, Any] = {"handle": None}
    real_kill = control.os.kill

    def spy_kill(pid: int, sig: int) -> None:
        # Signal 0 is the liveness PROBE, not a signal.
        if sig == 0:
            real_kill(pid, sig)
            return
        sent.append((pid, sig))

    def alive(pid: int) -> bool:
        handle = state["handle"]
        if handle is not None and getattr(handle, "exited", False):
            return False
        return pid > 0 and pid != -1

    monkeypatch.setattr(control.os, "kill", spy_kill)
    monkeypatch.setattr(control.registry, "pid_alive", alive)
    return sent, state


@pytest.mark.asyncio
async def test_graceful_socket_rung_stops_the_runtime(no_signals) -> None:
    """The ``stop`` op acks and the runtime exits; the outcome names socket,
    and no signal was ever sent."""
    sent, state = no_signals
    handle = _StoppingHandle()
    state["handle"] = handle
    server, record = await _serve(handle)
    try:
        outcome = await control.stop_session(
            _record_for(record), timeout_s=3.0, _root=registry.run_dir()
        )
        assert outcome.method == "socket"
        assert record.conversation_name in outcome.line
        assert "stopped" in outcome.line
        # The handle's hook ran exactly once.
        assert handle.stops == [True]
        assert sent == []
    finally:
        server.close()


@pytest.mark.asyncio
async def test_identity_mismatch_refuses_and_signals_nobody(no_signals) -> None:
    """A pid serving a DIFFERENT session id is never signalled.

    The pid-reuse rule (§12): an unconfirmed identity is a refusal, because a
    recycled pid means the process under the record may be an unrelated
    stranger. ``os.kill`` is monkeypatched to FAIL the test if reached —
    the assertion is that the ladder stops before its signal rungs.
    """
    server, record = await _serve()
    try:
        impostor = _record_for(record, session_id="someone-elses-session")
        signalled, _ = no_signals
        outcome = await control.stop_session(impostor, timeout_s=2.0, _root=registry.run_dir())
        assert outcome.method == "refused"
        assert signalled == []
        assert "refused" in outcome.line
    finally:
        server.close()


@pytest.mark.asyncio
async def test_dead_pid_reports_already_exited(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pid that died under the stop resolves as already-exited, code-clean."""
    server, record = await _serve()
    server.close()
    ghost = _record_for(record)
    # Re-publish the record so the registry still lists it after the close,
    # and make pid liveness read dead (the record's pid is OURS in-process,
    # so the real probe would say live).
    registry.publish(ghost)
    monkeypatch.setattr(registry, "pid_alive", lambda pid: False)
    try:
        outcome = await control.stop_session(ghost, timeout_s=1.0, _root=registry.run_dir())
        assert outcome.method == "gone"
        assert "already exited" in outcome.line
    finally:
        registry.unpublish(ghost.pid)


@pytest.mark.asyncio
async def test_wakes_go_dormant_not_deleted(tmp_path: Path, no_signals) -> None:
    """A stopped session's index entry survives with ``stopped_at`` stamped.

    Schedules are never deleted by a stop — the transcript is the authority —
    so dormancy is a marker on the derived index entry, and the count rides
    the receipt (``2 wakes dormant``).
    """
    from local_operator.wakes import store as wake_store

    session_id = "waketest"
    wake_store.write_entry(
        tmp_path,
        session_id,
        cwd="/tmp/waketest",
        schedules=[
            {"id": "w1", "message": "one", "every_ms": 60000, "next_due_at": 1},
            {"id": "w2", "message": "two", "every_ms": 60000, "next_due_at": 2},
        ],
    )
    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    try:
        target = _record_for(record, session_id=session_id)
        outcome = await control.stop_session(target, timeout_s=3.0, _root=tmp_path)
        assert outcome.wakes_dormant == 2
        assert "2 wakes dormant" in outcome.line
        entry = wake_store.read_entry(tmp_path, session_id)
        assert entry is not None, "a stop must never delete the index entry"
        assert isinstance(entry.get("stopped_at"), int)
        assert len(entry["schedules"]) == 2
    finally:
        server.close()


def test_mark_wakes_dormant_no_entry_is_zero(tmp_path: Path) -> None:
    """A session with no wake entry parks nothing and the receipt says none."""
    record = SessionRecord(
        pid=1,
        kind="tui",
        session_id="no-wakes",
        conversation_name="x",
        cwd="/tmp",
        model_label="m",
        control_port=0,
        control_key="k",
    )
    assert control._mark_wakes_dormant(record, tmp_path) == 0


@pytest.mark.asyncio
async def test_stop_all_never_targets_the_callers_own_pid(
    monkeypatch: pytest.MonkeyPatch, no_signals
) -> None:
    """``stop_all`` stops every OTHER live target and skips ``own_pid``
    entirely — never a socket op to itself, never a signal to itself.

    The first draft appended the own record LAST and walked it down the
    ladder; the TUI's own handle had no ``request_stop``, so identity was
    confirmed over its own socket and the terminal SIGTERMed itself (R1-1).
    The caller ends its own session in-process; this module must not.
    """
    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    try:
        # The in-process runtime's record carries the runner's own pid, so
        # the CALLER here is modelled as a distinct pid: what is pinned is
        # that the pid handed in as ``own_pid`` never reaches the ladder,
        # whichever pid that is.
        own = _record_for(record, pid=424242, session_id="own-session", control_port=1)
        other = record
        monkeypatch.setattr(
            control.registry,
            "scan",
            lambda root=None: [(own, "live"), (other, "live")],
        )
        monkeypatch.setattr(control, "_same_uid", lambda rec: True)
        outcomes = await control.stop_all(own_pid=424242, _root=registry.run_dir())
        assert [o.session_id for o in outcomes] == [other.session_id]
        assert [o.method for o in outcomes] == ["socket"]
        assert no_signals[0] == []
        assert handle.stops == [True]
    finally:
        server.close()


@pytest.mark.asyncio
async def test_stop_all_only_pids_restricts_to_the_listed_set(
    monkeypatch: pytest.MonkeyPatch, no_signals
) -> None:
    """A target that was not on the caller's listing is skipped, not stopped:
    the listing is the confirmation (R1-6)."""
    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    try:
        newcomer = _record_for(record, pid=99999, session_id="newcomer", control_port=1)
        monkeypatch.setattr(
            control.registry, "scan", lambda root=None: [(record, "live"), (newcomer, "live")]
        )
        monkeypatch.setattr(control, "_same_uid", lambda rec: True)
        outcomes = await control.stop_all(
            own_pid=None, only_pids={record.pid}, _root=registry.run_dir()
        )
        assert [o.session_id for o in outcomes] == [record.session_id]
        assert no_signals[0] == []
    finally:
        server.close()


def test_summarize_reconciles_with_the_listing() -> None:
    """The summary leads with the total and folds the caller's own session
    in, so the count matches what the listing promised (D2)."""
    outcomes = [
        control.StopOutcome(1, "a", "a", "socket", "x"),
        control.StopOutcome(2, "b", "b", "sigkill", "y"),
        control.StopOutcome(3, "c", "c", "refused", "z"),
        control.StopOutcome(4, "d", "d", "gone", "w"),
    ]
    own = control.StopOutcome(5, "e", "e", "socket", "")
    assert (
        control.summarize(outcomes, own=own)
        == "5 sessions: 2 stopped, 1 killed, 1 already exited, 1 refused"
    )
    assert control.summarize([]) == "no sessions to stop"
    assert control.summarize([outcomes[0]]) == "1 session: 1 stopped"


@pytest.mark.asyncio
async def test_refusal_line_names_the_session_then_the_pid_once(no_signals) -> None:
    """The refusal a user must act on is name-first with the pid once, so
    it matches the listing's row shape (D2/D8)."""
    server, record = await _serve()
    try:
        impostor = _record_for(record, session_id="someone-elses-session", conversation_name="x")
        outcome = await control.stop_session(impostor, timeout_s=2.0, _root=registry.run_dir())
        assert outcome.method == "refused"
        assert outcome.line.startswith(f'refused "x" (pid {record.pid}) — it serves session "')
        assert no_signals[0] == []
    finally:
        server.close()


@pytest.mark.asyncio
async def test_stop_op_is_dispatchable_over_the_wire() -> None:
    """A daemon-class dial can send ``stop`` and read the ack — the shape
    ``_exchange`` produces for a real runtime, independent of the ladder."""
    from local_operator.session.runtime.server import RuntimeServer

    handle = _StoppingHandle()
    server = RuntimeServer(handle, kind="tui")
    await server.start_in_process()
    record = await _wait_record()
    try:
        reply = await control._exchange(record, {"op": "stop"}, reply_timeout_s=3.0)
        assert reply is not None and reply.get("op") == "ack"
        assert handle.stops == [True]
    finally:
        server.close()


@pytest.mark.asyncio
async def test_old_runtime_unknown_op_is_a_miss_not_a_failure() -> None:
    """An error reply (old runtime, unknown op) leaves the graceful rung
    WITHOUT raising — the ladder's scheduled-miss path for mixed versions."""
    server, record = await _serve(FakeHandle())  # no request_stop capability
    try:
        reply = await control._exchange(record, {"op": "stop"}, reply_timeout_s=3.0)
        assert reply is not None and reply.get("op") == "error"
    finally:
        server.close()


@pytest.mark.asyncio
async def test_force_escalates_past_a_fresh_heartbeat_on_record_identity(no_signals) -> None:
    """A heartbeating runtime whose socket never answers (a TUI burning
    100% CPU, its socket loop queued behind the runaway) is refused without
    --force and signalled with it.

    The runtime here stays UP — its port stays bound and its record fresh —
    but the ladder's dial is starved, which is the shape the CLI evidence
    reproduced with SIGSTOP. --force reads identity from the record's own
    fields, which is sound precisely because the pid still holds the
    recorded control port.
    """
    server, record = await _serve()
    try:
        target = _record_for(record, pid=os.getpid())
        target.heartbeat_at = time.time()
        registry.publish(target)

        # Starve the dial without freeing the port: the socket is up, but no
        # identity answer comes back inside the ladder's window.
        async def _never(*args, **kwargs):
            # The starved shape: the dial never yields an identity answer,
            # exactly as _confirmed_session_id reports a silent socket.
            return False, control._SOCKET_SILENT

        with mock.patch.object(control, "_confirmed_session_id", _never):
            refused = await control.stop_session(target, timeout_s=0.5, _root=registry.run_dir())
            assert refused.method == "refused"
            assert "must lapse" in refused.line  # the named wait (U2-3)
            assert no_signals[0] == []
            stopped = await control.stop_session(
                target, timeout_s=0.5, force=True, _root=registry.run_dir()
            )
        assert stopped.method in ("sigterm", "sigkill")
        assert no_signals[0] != []  # the force gate opened
    finally:
        server.close()


@pytest.mark.asyncio
async def test_force_still_refuses_a_stale_record_over_a_recycled_pid(no_signals) -> None:
    """--force widens WHICH identity proof is admissible, never whether one
    is required.

    Caught in real testing: an earlier --force accepted the record-file check
    alone, which a stale record trivially satisfies, and SIGTERMed an
    unrelated process holding the recycled pid. Both halves are now
    mandatory — a fresh heartbeat AND the pid still holding the recorded
    control port — so a record that outlived its process refuses even under
    --force, and nothing is signalled.
    """
    server, record = await _serve()
    try:
        server.close()
        await asyncio.sleep(0.3)
        stale = _record_for(record, pid=os.getpid())
        stale.heartbeat_at = time.time() - 3600  # long past the window
        registry.publish(stale)
        outcome = await control.stop_session(
            stale, timeout_s=1.0, force=True, _root=registry.run_dir()
        )
        assert outcome.method == "refused"
        assert no_signals[0] == []  # nothing was signalled
    finally:
        pass
