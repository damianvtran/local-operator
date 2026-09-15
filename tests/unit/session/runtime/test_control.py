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

from local_operator.paths import config_dir
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

    def alive(pid: int, *, check_zombie: bool = False) -> bool:
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
        outcome = await control.stop_session(_record_for(record), timeout_s=3.0, _root=config_dir())
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
        outcome = await control.stop_session(impostor, timeout_s=2.0, _root=config_dir())
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
    monkeypatch.setattr(registry, "pid_alive", lambda pid, **_: False)
    try:
        outcome = await control.stop_session(ghost, timeout_s=1.0, _root=config_dir())
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
        outcomes = await control.stop_all(own_pid=424242, _root=config_dir())
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
        outcomes = await control.stop_all(own_pid=None, only_pids={record.pid}, _root=config_dir())
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
        outcome = await control.stop_session(impostor, timeout_s=2.0, _root=config_dir())
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
            refused = await control.stop_session(target, timeout_s=0.5, _root=config_dir())
            assert refused.method == "refused"
            # The remedy is named as the WHOLE command (the TUI's /stop paints
            # this string and has no spelling for a flag — U2-3/U3-2), and it is
            # the FORCED rung, which is the one that reaches a heartbeating
            # owner. The sentence used to promise that the heartbeat "must
            # lapse (~45s)" and to advise retrying afterwards: that is true only
            # while the owner keeps failing to report — a single turn in the
            # window resets it — and the ladder meanwhile has a rung that works
            # NOW, so the promise was both unfalsifiable and unnecessary. See
            # ``_identity_by_start_time``.
            assert f"lop stop --pid {target.pid} --force" in refused.line
            assert "must lapse" not in refused.line
            assert no_signals[0] == []
            stopped = await control.stop_session(
                target, timeout_s=0.5, force=True, _root=config_dir()
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
        outcome = await control.stop_session(stale, timeout_s=1.0, force=True, _root=config_dir())
        assert outcome.method == "refused"
        assert no_signals[0] == []  # nothing was signalled
    finally:
        pass


def test_withdrawal_takes_back_only_this_ladders_own_rung_one_marker(tmp_path: Path) -> None:
    """The refusal's withdrawal is targeted, not a blanket unlink.

    A marker attests to an ACT, and the run it names may have several possible
    authors: a concurrent ``lop stop`` from another terminal, a retry from the
    same front end, the ladder's own later rung. Only THIS ladder's own rung-1
    statement may be taken back on a refusal — a marker another killer staged
    for the same run is evidence for a stop that really is happening, and a
    later rung's marker is not this rung's to withdraw.
    """
    from local_operator.session.runtime.types import session_dir

    record = SessionRecord(
        pid=2**22 + 71,
        kind="tui",
        session_id="withdraw",
        conversation_name="withdraw",
        cwd="/tmp",
        model_label="m",
        control_port=1,
        control_key="k",
    )
    conversation = session_dir(tmp_path, record.session_id)
    conversation.mkdir(parents=True)

    def staged(**overrides: Any) -> dict[str, Any]:
        payload = control._stop_marker_payload(record, "socket", command="/stop")
        payload.update(overrides)
        return payload

    # The markers that must SURVIVE the withdrawal, each for its own reason.
    for label, payload in (
        ("another killer's marker", staged(killer={"pid": os.getpid() + 1, "command": "/stop"})),
        ("a later rung's marker", staged(rung="sigkill")),
        ("another run's marker", staged(started_at=record.started_at + 5.0)),
    ):
        registry.write_stop_marker(conversation, payload)
        control._withdraw_staged_stop_marker(record, tmp_path)
        assert registry.read_stop_marker(conversation) is not None, label

    # ...and the one it must withdraw: our own rung-1 statement for this run.
    registry.write_stop_marker(conversation, staged())
    control._withdraw_staged_stop_marker(record, tmp_path)
    assert registry.read_stop_marker(conversation) is None


def test_record_retired_reads_the_root_the_caller_injected(tmp_path: Path) -> None:
    """A stop's landing is decided against the root the caller TARGETED.

    ``_record_retired`` used to read the ambient ``registry.run_dir()`` while the
    ladder's other root reads (``_park_wakes``, ``_recover_record``) used the
    injected one, so an injected-root caller's missing file read as "retired"
    and the ladder returned a confident ``socket`` receipt — ``0.01 s`` — for a
    stop that had landed nowhere, with the target still alive and serving
    (design §1e, found by running the ladder against an isolated root). The
    three assertions below are the whole defect: what is in the injected root
    decides, what is only in the ambient root must not.
    """
    injected = tmp_path / "injected"
    record = SessionRecord(
        pid=2**22 - 5,
        kind="tui",
        session_id="s-root",
        conversation_name="root",
        cwd="/tmp",
        model_label="m",
        control_port=1,
        control_key="k",
    )
    registry.publish(record, root=injected)

    assert control._record_retired(record, injected) is False
    # The ambient root (the suite's isolated config dir) holds nothing for this
    # pid: before the fix this answer was the one the ladder used for BOTH.
    assert control._record_retired(record, config_dir()) is True

    # A record that now names another session is retired too — the pid was
    # recycled by a new runtime, which is what the read is guarding against.
    other = SessionRecord(**{**record.to_json(), "session_id": "s-someone-else"})
    registry.publish(other, root=injected)
    assert control._record_retired(record, injected) is True


def test_the_marker_payload_names_the_run_the_rung_the_killer_and_the_build() -> None:
    """The durable marker's fields are the attribution, so they are pinned.

    Each one answers a question the incident could not answer afterwards: which
    RUN (the ``(session_id, pid, started_at)`` key a stale marker is refused
    by), which RUNG (how hard the stop pushed), whether it was ASKED FOR (the
    flag that must never be set by anything but this ladder), WHO (the fact the
    investigation could not recover at all) and which BUILD the target was
    running.
    """
    record = SessionRecord(
        pid=4242,
        kind="exec",
        session_id="s-payload",
        conversation_name="payload",
        cwd="/tmp",
        model_label="m",
        control_port=1,
        control_key="k",
    )
    record.started_at = 1_760_000_000.0
    record.version = "0.54.39"
    record.source_ref = "dec7933"

    payload = control._stop_marker_payload(record, "sigkill", command="lop stop --all")

    assert payload["session_id"] == "s-payload"
    assert payload["pid"] == 4242
    assert payload["started_at"] == 1_760_000_000.0
    assert payload["rung"] == "sigkill"
    assert payload["deliberate"] is True
    assert payload["killer"] == {
        "pid": os.getpid(),
        "argv0": payload["killer"]["argv0"],
        "command": "lop stop --all",
    }
    assert payload["killer"]["argv0"]
    assert payload["build"] == "0.54.39@dec7933"


class _AckingHandle(FakeHandle):
    """A handle whose ``request_stop`` ACKS and then does not exit.

    The shape rung 1 reaches and the ladder then refuses: the op is answered
    (a live, cooperative socket), the session keeps running, and the IDENTITY
    gate is what stops the ladder — the ``/resume`` record-lag window the gate
    exists for, where the record still names the previous session while the
    process serves the new one. Unlike ``_StoppingHandle`` (which reports
    itself exited, so the ladder calls the rung landed), this one stays alive,
    which is the whole point: a refusal leaves a live target behind.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stops = 0

    def request_stop(self) -> None:
        self.stops += 1


@pytest.mark.asyncio
async def test_the_rung_that_fires_stages_the_marker_naming_that_rung(no_signals) -> None:
    """The rung that ACTS leaves exactly one marker, naming itself.

    The other half of the invariant below: a rung that escalates must attest,
    because at SIGKILL the target cannot. Driven through ``--force``, the route
    that reaches the signal rungs against an in-process double whose socket
    cannot be dialled into an identity answer.
    """
    from local_operator.session.runtime.types import session_dir

    server, record = await _serve(handle=_NeverStopsHandle())
    try:
        target = _record_for(record, pid=os.getpid())
        target.heartbeat_at = time.time()
        registry.publish(target, root=config_dir())
        conversation = session_dir(config_dir(), target.session_id)
        # The runtime that owns a session has a conversation directory (its
        # transcript lives there) — the in-process double does not, and the
        # marker writer deliberately refuses to invent one.
        conversation.mkdir(parents=True, exist_ok=True)

        # A starved-but-fresh runtime: the dial says nothing, the heartbeat says
        # it was alive moments ago, so no signal is safe without --force.
        async def _never(*args: Any, **kwargs: Any) -> Any:
            return False, control._SOCKET_SILENT

        with mock.patch.object(control, "_confirmed_session_id", _never):
            stopped = await control.stop_session(
                target, timeout_s=0.5, force=True, _root=config_dir()
            )
        assert stopped.method in ("sigterm", "sigkill")
        assert no_signals[0] != []  # the force gate opened
        marker = registry.read_stop_marker(conversation)
        assert marker is not None, "the rung that acted must have staged its evidence"
        assert marker["rung"] == stopped.method
        assert marker["deliberate"] is True
        assert marker["killer"]["pid"] == os.getpid()
        assert marker["killer"]["command"] == "control.stop_session"
        assert marker["pid"] == target.pid
        assert marker["started_at"] == target.started_at
    finally:
        server.close()


@pytest.mark.asyncio
async def test_a_refused_stop_leaves_no_marker_and_the_death_stays_unattributed(
    no_signals,
) -> None:
    """Rung 1 can ACK and the ladder still refuse — and then nothing is signed.

    THE DEFECT THIS PINS: rung 1 stages its marker on the ack (the ack is what
    sets the target's exit in motion), and the ladder may then refuse before
    any signal — a socket that answers naming ANOTHER session id is a live
    stranger (the ``/resume`` record-lag window), and a start-time proof can
    fail. The target is left ALIVE with a durable marker keyed to its own run,
    so its next, quite involuntary death would classify as the user's own stop:
    the wrong-verdict class this evidence exists to remove, in the direction
    that HIDES a crash.

    The old version of this test could not see it: ``_serve()``'s default
    ``FakeHandle`` has no ``request_stop``, so the stop op answered ``error``
    and ``_graceful_stop`` returned BEFORE the staging line. Here the handle
    acks (``stops == 1`` proves the staging path ran) and the refusal comes from
    the real identity gate, then the run is classified.
    """
    import uuid

    from local_operator.session.attention import AttentionStore, bootstrap_transcript
    from local_operator.session.runtime.types import session_dir
    from local_operator.session.transcript import Transcript

    handle = _AckingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    session_id = "someone-elses-session"
    dead_pid = 2**22 + 61  # no such process; the ladder's liveness is patched
    try:
        # The record claims a session the socket does not serve: the dial
        # answers with the runtime's REAL id, which is exactly what the
        # identity gate refuses on.
        target = _record_for(record, pid=dead_pid, session_id=session_id)
        registry.publish(target, root=config_dir())
        conversation = session_dir(config_dir(), session_id)
        transcript = Transcript(conversation)
        await transcript.append_custom(
            "attention_started",
            {
                "conversation_id": f"session/{session_id}",
                # A real UUID: the store validates it, so a placeholder here
                # would fail the classification rather than the assertion.
                "token": str(uuid.uuid4()),
            },
        )

        outcome = await control.stop_session(target, timeout_s=0.5, _root=config_dir())

        assert handle.stops == 1, "the rung-1 staging path must have run"
        assert outcome.method == "refused"
        assert "it serves session" in outcome.line
        assert no_signals[0] == []  # nothing was signalled
        assert registry.read_stop_marker(conversation) is None, (
            "a refusal must withdraw the marker its own rung staged: the target is "
            "still alive and its later death must not read as the user's own stop"
        )

        # ...and the death that comes later is still unattributed, which is what
        # the withdrawal buys: the record names the run, the marker does not.
        with mock.patch.object(registry, "pid_alive", lambda *a, **k: False):
            result = bootstrap_transcript(transcript, AttentionStore(config_dir() / "a.db"))
        assert result is not None
        assert (result[0], result[1]) == ("error", "runtime-killed"), result
        assert f"pid {dead_pid}" in result[2], result[2]
    finally:
        server.close()
