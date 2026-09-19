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
from local_operator.session.runtime.types import (
    LEAVING_FOR_BUILD,
    LEAVING_ON_SIGNAL,
    SessionRecord,
)
from tests.unit.session.runtime.test_server import FakeHandle, _wait_record


def _bare_record(**overrides: Any) -> SessionRecord:
    """A resolvable record with no live runtime behind it.

    For the cells that read a record rather than dial it: the ``/info``,
    ``lop sessions`` and receipt builders take a record and nothing else, and a
    plain one keeps a wording cell from needing a server to state itself.
    """
    fields: dict[str, Any] = {
        "pid": 4242,
        "kind": "tui",
        "session_id": "stub01234567",
        "conversation_name": "stub",
        "cwd": "/tmp",
        "model_label": "test/mock",
        "control_port": 1,
        "control_key": "k",
        "version": "0.55.4",
        "source_ref": "f4a70b9" + "0" * 33,
    }
    fields.update(overrides)
    return SessionRecord(**fields)


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
async def test_force_escalates_past_a_fresh_heartbeat_on_record_identity(
    no_signals, monkeypatch: pytest.MonkeyPatch
) -> None:
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
            # A short rung-2 wait, INJECTED rather than shortened in the
            # product: the process behind this record is the test runner
            # itself, which never exits, so the production grace (minutes by
            # construction — see ``SIGTERM_GRACE_S``) would be spent in full.
            # What this cell pins is WHICH rung fires. The CONSTANT is patched
            # rather than a parameter passed (NIT, PR #1141): a knob on the one
            # value whose purpose is that it cannot be shortened is one
            # production caller away from landing SIGKILL inside the receiver's
            # drain.
            monkeypatch.setattr(control, "SIGTERM_GRACE_S", 0.5)
            stopped = await control.stop_session(
                target,
                timeout_s=0.5,
                force=True,
                _root=config_dir(),
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


def test_an_involuntary_marker_keeps_the_run_key_and_never_claims_a_stop(
    tmp_path: Path,
) -> None:
    """I3: the same schema, written by a party that is not the stop ladder.

    Every path that can take a runtime away — the prune that removes the generation
    it is importing from, an in-place install rewriting it — stages this BEFORE it
    acts, because afterwards the runtime cannot record anything and on 2026-09-18
    twenty-five of them left no artifact naming an actor. Two properties make the
    reader work unchanged: the RUN KEY is spelled from the same three fields (so
    ``attention._stop_marker_covers_run`` refuses a stale marker exactly as before),
    and ``deliberate`` is false (so nothing can read the death as the user's own
    stop).

    The deliberate payload is asserted to have NO ``actor``/``mechanism`` keys in the
    same breath: absent rather than empty, so a marker of that kind stays
    byte-identical to what every reader has already been taught.
    """
    from local_operator.session.runtime.types import session_dir

    record = SessionRecord(
        pid=4243,
        kind="daemon",
        session_id="s-involuntary",
        conversation_name="involuntary",
        cwd="/tmp",
        model_label="m",
        control_port=1,
        control_key="k",
    )
    record.started_at = 1_760_000_000.0
    record.version = "0.59.2"
    record.source_ref = "abc1234"
    conversation = session_dir(tmp_path, record.session_id)
    conversation.mkdir(parents=True)

    assert control.note_involuntary_stop(
        record,
        tmp_path,
        mechanism="generation-prune",
        actor="lop install prune",
    )

    marker = registry.read_stop_marker(conversation)
    assert marker is not None
    assert marker["session_id"] == "s-involuntary"
    assert marker["pid"] == 4243
    assert marker["started_at"] == 1_760_000_000.0
    assert marker["deliberate"] is False
    assert marker["mechanism"] == "generation-prune"
    assert marker["actor"] == "lop install prune"
    assert marker["killer"] == {
        "pid": os.getpid(),
        "argv0": marker["killer"]["argv0"],
        "command": "lop install prune",
    }
    assert marker["build"] == "0.59.2@abc1234"

    deliberate = control._stop_marker_payload(record, "socket", command="/stop")
    assert "actor" not in deliberate and "mechanism" not in deliberate
    assert deliberate["deliberate"] is True


def test_an_involuntary_marker_reads_a_boot_record_as_the_same_run(tmp_path: Path) -> None:
    """A runtime in its FIRST SECOND is attestable, and its build is spelled its way.

    The window a prune can catch a runtime in is the ~1.2 s between the boot record
    and the first heartbeat, so the writer must accept a ``BootRecord`` — which
    spells the build ``build_version``/``build_ref`` where a ``SessionRecord`` says
    ``version``/``source_ref``. ONE payload shape for both, rather than a second
    builder free to disagree about the run key.
    """
    from local_operator.session.runtime.journal import BootRecord
    from local_operator.session.runtime.types import session_dir

    record = BootRecord(
        pid=4244, session_id="s-booting", build_version="0.59.2", build_ref="def5678"
    )
    conversation = session_dir(tmp_path, record.session_id)
    conversation.mkdir(parents=True)

    assert control.note_involuntary_stop(
        record, tmp_path, mechanism="generation-prune", actor="lop update"
    )

    marker = registry.read_stop_marker(conversation)
    assert marker is not None
    assert (marker["session_id"], marker["pid"]) == ("s-booting", 4244)
    assert marker["build"] == "0.59.2@def5678"
    assert marker["deliberate"] is False


def test_withdrawing_an_involuntary_marker_takes_back_only_our_own(tmp_path: Path) -> None:
    """MINOR 2: the writer's counterpart, and everything it must refuse to touch.

    The prune and the in-place install stage a marker BEFORE the irreversible step;
    when the step is then not taken — ``_remove_tree`` reports the tree still there,
    the installer exits non-zero — the marker is the only artifact left saying
    otherwise, and it is keyed to the live RUN, so it would narrate any later death
    of that runtime as this act's. The withdrawal decides what is ours by READING
    the file, exactly as the ladder's own withdrawal does, rather than by trusting
    that we wrote one: another front end's marker for the same run and a rung's
    marker both have to survive it.
    """
    from local_operator.session.runtime.types import session_dir

    record = SessionRecord(
        pid=4245,
        kind="daemon",
        session_id="s-withdraw",
        conversation_name="withdraw",
        cwd="/tmp",
        model_label="m",
        control_port=1,
        control_key="k",
    )
    record.started_at = 1_760_000_000.0
    conversation = session_dir(tmp_path, record.session_id)
    conversation.mkdir(parents=True)

    # ANOTHER ACT'S MARKER FOR THE SAME RUN is not this withdrawal's to take: the
    # act it names is still the truth about why this runtime is gone.
    assert control.note_involuntary_stop(
        record, tmp_path, mechanism="in-place-install", actor="lop update"
    )
    assert (
        control.withdraw_involuntary_stop(record, tmp_path, mechanism="generation-prune") is False
    )
    assert registry.read_stop_marker(conversation) is not None

    # ...and neither is a marker whose killer is a DIFFERENT process.
    staged = registry.read_stop_marker(conversation)
    assert staged is not None
    staged["killer"] = dict(staged["killer"], pid=os.getpid() + 1)
    registry.write_stop_marker(conversation, staged)
    assert (
        control.withdraw_involuntary_stop(record, tmp_path, mechanism="in-place-install") is False
    )

    # A DIFFERENT RUN of the same session is not this run: the key is the pid too,
    # which is what keeps an older run's attestation from narrating a newer death.
    older = SessionRecord(
        pid=9999,
        kind="daemon",
        session_id="s-withdraw",
        conversation_name="withdraw",
        cwd="/tmp",
        model_label="m",
        control_port=1,
        control_key="k",
    )
    older.started_at = 1_700_000_000.0
    assert control.withdraw_involuntary_stop(older, tmp_path, mechanism="in-place-install") is False

    # OURS, for the mechanism that staged it: taken back, and the file is gone.
    staged["killer"] = dict(staged["killer"], pid=os.getpid())
    registry.write_stop_marker(conversation, staged)
    assert control.withdraw_involuntary_stop(record, tmp_path, mechanism="in-place-install") is True
    assert registry.read_stop_marker(conversation) is None

    # THE LADDER'S OWN RUNG MARKER IS NEVER WITHDRAWN HERE even when it covers this
    # run: it says a person asked for the stop, which is a statement about an act
    # that did happen.
    control._write_stop_marker(record, tmp_path, "socket", command="/stop")
    assert (
        control.withdraw_involuntary_stop(record, tmp_path, mechanism="generation-prune") is False
    )
    assert registry.read_stop_marker(conversation) is not None


def test_an_involuntary_marker_without_a_conversation_is_refused_not_invented(
    tmp_path: Path,
) -> None:
    """The writer reports the gap instead of creating a directory to hide it in.

    ``registry.write_stop_marker`` deliberately does not create the conversation
    directory, and an involuntary act must not either: a ``lop serve`` daemon has no
    conversation at all, and a session whose directory was deleted has no reader for
    a marker that would only be found by recreating it. Both return False, which the
    caller (``update.note_doomed_runtimes``) logs — an unattestable runtime is a real
    gap in the artifact, and silence there would read afterwards as "nobody was
    affected".
    """
    from local_operator.session.runtime.types import session_dir

    record = SessionRecord(
        pid=4245,
        kind="daemon",
        session_id="s-gone",
        conversation_name="gone",
        cwd="/tmp",
        model_label="m",
        control_port=1,
        control_key="k",
    )
    assert not control.note_involuntary_stop(record, tmp_path, mechanism="generation-prune")
    assert not session_dir(tmp_path, record.session_id).exists()

    session_less = mock.MagicMock()
    session_less.session_id = ""
    assert not control.note_involuntary_stop(session_less, tmp_path, mechanism="generation-prune")


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
async def test_the_rung_that_fires_stages_the_marker_naming_that_rung(
    no_signals, monkeypatch: pytest.MonkeyPatch
) -> None:
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
            # A short rung-2 wait, INJECTED rather than shortened in the
            # product: the process behind this record is the test runner
            # itself, which never exits, so the production grace (minutes by
            # construction — see ``SIGTERM_GRACE_S``) would be spent in full.
            # What this cell pins is WHICH rung fires. The CONSTANT is patched
            # rather than a parameter passed (NIT, PR #1141): a knob on the one
            # value whose purpose is that it cannot be shortened is one
            # production caller away from landing SIGKILL inside the receiver's
            # drain.
            monkeypatch.setattr(control, "SIGTERM_GRACE_S", 0.5)
            stopped = await control.stop_session(
                target,
                timeout_s=0.5,
                force=True,
                _root=config_dir(),
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


# ---------------------------------------------------------------------------
# Rotation (`lop refresh`) and the ladder's duty not to cut work in flight
# ---------------------------------------------------------------------------


class _RefreshableHandle(_StoppingHandle):
    """A handle that can judge itself idle and answer a retirement request."""

    def __init__(self, *, reason: str = "") -> None:
        super().__init__()
        self.reason = reason
        self.probes = 0

    def may_refresh(self) -> str:
        self.probes += 1
        return self.reason


def test_the_sigkill_rung_outlasts_the_receivers_drain_bound() -> None:
    """THE INVARIANT between the two halves of the work-aware signal fix.

    A runtime that receives SIGTERM with a turn in flight defers its own
    disposal by up to ``SIGNAL_DRAIN_S`` (``process._drain_for_signal``). This
    ladder escalates to SIGKILL after ``SIGTERM_GRACE_S``. If the second number
    were ever the smaller, the escalation would kill a runtime that was
    deliberately and correctly finishing a turn — with the one signal nothing
    can catch — and the receiver-side fix would be worse than useless.
    """

    from local_operator.session.runtime.types import SIGNAL_DRAIN_S

    assert (
        control.SIGTERM_GRACE_S > SIGNAL_DRAIN_S
    ), "the ladder's SIGTERM→SIGKILL grace must outlast the runtime's drain bound"
    assert (
        control.SIGTERM_GRACE_S - SIGNAL_DRAIN_S >= 5.0
    ), "the margin must leave the receiver real time to dispose after its drain"


@pytest.mark.asyncio
async def test_a_busy_target_is_skipped_and_nobody_signals_it(no_signals) -> None:
    """Our own kill switch must not cut a turn in flight.

    Rung 1 (the socket op) is tried first and would stop a cooperative runtime
    promptly even mid-turn — a stop the user asked for is one they want. What
    this pins is the case the sweep needs: the socket did NOT answer, and the
    record says a turn is in flight, so the ladder declines to signal and says
    so, naming both ways forward.
    """
    from local_operator.paths import config_dir as ambient_config_dir

    server, record = await _serve()
    server.close()  # nothing listening: rung 1 is a scheduled miss
    target = _record_for(record, busy=True)
    try:
        outcome = await control.stop_session(target, timeout_s=0.5, _root=config_dir())
        assert outcome.method == "busy"
        assert "turn is in flight" in outcome.line
        assert "--force" in outcome.line
        assert no_signals[0] == [], "a busy target is never signalled"
    finally:
        registry.unpublish(target.pid)
    assert ambient_config_dir  # the ambient root is never used by these tests


@pytest.mark.asyncio
async def test_force_signals_the_busy_target_the_plain_stop_left_alone(
    no_signals, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--force`` is the escape hatch, and it escalates the whole ladder.

    The flag already means "signal this runtime I cannot reach"; someone who
    types it has accepted that the turn goes with it.
    """
    import signal as signal_mod

    server, record = await _serve()
    server.close()
    target = _record_for(record, busy=True)
    monkeypatch.setattr(control, "_identity_by_record", lambda _r: (True, ""))
    # The target dies the moment it is signalled. This test is about WHICH rung
    # runs with --force (the skip is bypassed), not about the escalation order
    # or the grace budgets, which the constant tests own; without this the
    # ladder would spend its real SIGTERM grace waiting for a process that the
    # fixture keeps alive on purpose.
    monkeypatch.setattr(control.registry, "pid_alive", lambda _pid, **_: not no_signals[0])
    try:
        outcome = await control.stop_session(target, timeout_s=0.5, force=True, _root=config_dir())
        assert [sig for _pid, sig in no_signals[0]] == [signal_mod.SIGTERM]
        assert outcome.method == "sigterm"
    finally:
        registry.unpublish(target.pid)


@pytest.mark.asyncio
async def test_a_healthy_busy_session_is_still_stopped_promptly_by_the_socket_rung(
    no_signals,
) -> None:
    """The skip never applies to a target that ANSWERS: a deliberate stop wins.

    This is the behaviour the whole ladder exists for — ``lop stop <busy
    session>`` must end it, mid-turn, without waiting for the receiver's drain
    bound. The skip is scoped to targets whose own socket is silent, which is
    why it sits after rung 1.
    """
    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    try:
        outcome = await control.stop_session(
            _record_for(record, busy=True), timeout_s=3.0, _root=config_dir()
        )
        assert outcome.method == "socket"
        assert handle.stops == [True]
        assert no_signals[0] == []
    finally:
        server.close()


@pytest.mark.asyncio
async def test_a_runtime_that_is_already_leaving_is_asked_before_it_is_stopped(
    no_signals,
) -> None:
    """U1: the drain is real work, so the ladder must not quietly cut it.

    A signalled runtime keeps working for up to ``SIGNAL_DRAIN_S`` and publishes
    ``leaving`` while it does (``SessionRecord.leaving``). Every OTHER rung
    reaches it — it is cooperative, its socket answers, and rung 1 ends it in
    milliseconds — which is the harm rather than the safeguard: the operator's
    own stop cuts the very turn the signal asked the runtime to finish, and
    nothing they could read said so (`lop sessions` reported ``live``, and no
    surface mentioned the signal at all). So this refusal runs BEFORE rung 1.

    Two properties are pinned together, because the fix is only correct with
    both: the plain stop declines and says what insisting would cost, and
    ``--force`` still ends it PROMPTLY through the socket — a confirmation, never
    a postponement. A deliberate stop that got slower or was deferred would be a
    regression on the property this ladder was built to protect.
    """
    from local_operator.session.runtime.types import LEAVING_ON_SIGNAL

    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    target = _record_for(record, busy=True, leaving=LEAVING_ON_SIGNAL)
    try:
        outcome = await control.stop_session(target, timeout_s=3.0, _root=config_dir())
        assert outcome.method == "draining", outcome.line
        assert LEAVING_ON_SIGNAL in outcome.line, outcome.line
        assert "cuts the turn" in outcome.line, outcome.line
        assert "--force" in outcome.line, outcome.line
        # The refusal is a PARTIAL result (the target is still running) and is
        # in the shared set the front ends read, not a method nobody classified.
        assert outcome.method not in control.ENDED_METHODS
        assert outcome.method in control.LEFT_ALONE_METHODS
        # Rung 1 never ran, and nothing was signalled.
        assert handle.stops == [], "the socket rung must not run for a draining target"
        assert no_signals[0] == []

        forced = await control.stop_session(target, timeout_s=3.0, force=True, _root=config_dir())
        assert forced.method == "socket", forced.line
        assert handle.stops == [True], "--force stops it now, by the socket rung"
        assert no_signals[0] == [], "and still without a signal"
    finally:
        server.close()


@pytest.mark.asyncio
async def test_the_refusal_quotes_the_drains_own_reason(no_signals) -> None:
    """The refusal is trigger-agnostic, because the drain now is (PR #1108).

    Two things commit a runtime to leaving — a termination signal and a build
    replaced on disk while a turn was in flight — and ``process._commit_to_leaving``
    publishes both to the same record field through the same call. The ladder
    therefore refuses to cut either one, and the sentence it paints has to come
    from the record: a hard-coded "it was signalled" would state the wrong
    reason for every build-driven drain, on the one line the operator reads to
    decide whether to insist.
    """
    from local_operator.session.runtime.types import LEAVING_FOR_BUILD

    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    target = _record_for(record, busy=True, leaving=LEAVING_FOR_BUILD)
    try:
        outcome = await control.stop_session(target, timeout_s=3.0, _root=config_dir())
        assert outcome.method == "draining", outcome.line
        assert LEAVING_FOR_BUILD in outcome.line, outcome.line
        assert "was signalled" not in outcome.line, outcome.line
        assert handle.stops == [], "the socket rung must not run for a draining target"
        assert no_signals[0] == []
    finally:
        server.close()


@pytest.mark.asyncio
async def test_the_refusal_offers_only_a_remedy_its_reader_can_act_on(
    no_signals,
) -> None:
    """UX round 2, U7: the TUI paints this line and parses no flags at all.

    The refusal is composed in the shared module and painted verbatim by both
    front ends, so it used to name ``--force`` — a flag of ``lop stop`` — on a
    surface whose ``/stop`` takes only a target: ``/stop --force <id>`` answers
    "no live session matches '--force <id>'" about a session that is live and
    listed. A surface must never offer an action it cannot accept, so the remedy
    is named in the reader's own vocabulary: the flag for the CLI, the shell for
    the TUI.

    Both directions are asserted. A fix that dropped the flag everywhere would
    remove the escape from the one front end that HAS it, which is the half this
    test exists to keep.
    """
    from local_operator.session.runtime.types import LEAVING_ON_SIGNAL

    handle = _StoppingHandle()
    no_signals[1]["handle"] = handle
    server, record = await _serve(handle)
    target = _record_for(record, busy=True, leaving=LEAVING_ON_SIGNAL)
    try:
        cli = await control.stop_session(
            target, timeout_s=3.0, _root=config_dir(), _command="lop stop"
        )
        assert cli.method == "draining", cli.line
        assert "--force to stop it anyway" in cli.line, cli.line

        tui = await control.stop_session(
            target, timeout_s=3.0, _root=config_dir(), _command="/stop"
        )
        assert tui.method == "draining", tui.line
        assert "--force to stop it anyway" not in tui.line, tui.line
        assert f"lop stop --force {target.pid}" in tui.line, tui.line

        # AND THE FIRST REMEDY IS NOTHING. Exit 2 plus a lone "--force to stop
        # it anyway" reads as "this did not work, retry if you meant it", when
        # the exit is already scheduled and the answer for almost everyone is to
        # let the turn finish (UX round 2, NIT-2).
        for outcome in (cli, tui):
            assert "it leaves by itself, nothing to do" in outcome.line, outcome.line

        # Neither line stopped or signalled anything, on either surface.
        assert handle.stops == []
        assert no_signals[0] == []
    finally:
        server.close()


def test_the_wait_line_names_its_bound_once_and_in_one_unit() -> None:
    """D4/U5: two bounds, one unit — and neither vocabulary in the wrong mouth.

    The drain bound and the ladder's grace are 120 s and 150 s. Printed as ``(up
    to 2 min)`` and ``waiting up to 150s`` they read as different KINDS of
    number, inviting the reader to wonder whether they are the same wait (design
    round 2, D4). ``bound_text`` is the one formatter both go through; this
    pins its output and the two sentences that carry it.

    The TUI's form is asserted to stay out of the kill vocabulary for the reason
    its reader is different: ``SIGKILL`` and "drain" are the CLI's words, and the
    app paints this line verbatim into a notice that had always promised
    "waiting for it to answer" (design round 2, D2).
    """
    assert control.bound_text(120.0) == "2 min"
    assert control.bound_text(150.0) == "2.5 min"
    assert control.bound_text(3.0) == "3s"
    assert control.bound_text(1800.0) == "30 min"

    cli = control._wait_line("beta", 1676, from_a_shell=True)
    assert cli == 'waiting up to 2.5 min for "beta" (pid 1676) to drain before SIGKILL'
    tui = control._wait_line("beta", 1676, from_a_shell=False)
    assert "SIGKILL" not in tui and "drain" not in tui, tui
    # The BOUND survives into the TUI's own words — the whole point of (U5): the
    # pause is minutes long and an unannounced one reads as a hang.
    assert "2.5 min" in tui, tui

    # Both bound-bearing sentences now state their bound through the ONE
    # formatter, so neither can drift into a second unit on its own.
    assert control.bound_text(control.SIGTERM_GRACE_S) in cli
    stub = SessionRecord(
        pid=4242,
        kind="tui",
        session_id="stub01234567",
        conversation_name="stub",
        cwd="/tmp",
        model_label="test/mock",
        control_port=1,
        control_key="k",
    )
    receipt = control._refresh_line(stub, "0.55.0@46a4e9b", "draining", "")
    assert f"(up to {control.bound_text(control.SIGNAL_DRAIN_S)})" in receipt, receipt
    # ...and it is the receiver's bound in that sentence, not the sender's: the
    # receipt describes how long the RUNTIME will finish its turn for.
    assert control.SIGNAL_DRAIN_S != control.SIGTERM_GRACE_S


@pytest.mark.asyncio
async def test_the_settle_question_keeps_no_phrase_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D8 is RECORDED, not closed — this cell is the record.

    The design round's D8 measured a draining runtime from the round-1 lineage
    being answered "ask again", and the temptation is to make ``_settle_question``
    answer "draining" whenever the record carries a phrase. That branch would be
    DEAD: ``Server._refresh_if_idle`` returns ``kept: already leaving`` whenever
    ``_leaving`` is set, and ``announce_retiring`` sets it in the same call that
    writes the phrase — so a phrase-carrying runtime never routes here at all,
    while the buildings that DO route here (drain on a signal, publish no
    phrase) carry nothing the CLI can distinguish "leaving" from "not settled"
    with. Pinned as a negative so the shortcut is not re-added as a fix: the
    CLI's answer for that population is the marker's, and the population is
    named in the function's docstring.
    """
    record = _bare_record()
    monkeypatch.setattr(control, "moved_and_unsettled", lambda *_a, **_k: True)

    assert control._settle_question(record) == ("unsettled", "")
    assert control._settle_question(_record_for(record, leaving=LEAVING_ON_SIGNAL)) == (
        "unsettled",
        "",
    )


@pytest.mark.asyncio
async def test_the_drain_reads_as_prose_in_every_receipt() -> None:
    """D7: one vocabulary, and it carries a verb where a person reads it.

    The record's phrase is a CELL VALUE (lowercase, subject-less) because
    ``lop sessions`` and ``/info`` print it in a column, so concatenating it
    into a sentence produced '"name" (pid 12, running …) signalled; leaving when
    its turn ends' — a fragment with no verb. Both prose slots hang their own
    subject on one copula supplied by ``_drain_phrase``, so the same words are a
    cell in one place and a clause in the other.
    """
    record = _bare_record()
    for phrase in (LEAVING_ON_SIGNAL, LEAVING_FOR_BUILD):
        receipt = control._refresh_line(
            _record_for(record, leaving=phrase), "0.55.4@f4a70b9", "draining", ""
        )
        assert receipt.endswith(f"is {phrase}"), receipt

    handle = _StoppingHandle()
    server, served = await _serve(handle)
    target = _record_for(served, busy=True, leaving=LEAVING_ON_SIGNAL)
    try:
        outcome = await control.stop_session(target, timeout_s=0.2, _root=config_dir())
        assert outcome.method == "draining", outcome.line
        assert f"— it is {LEAVING_ON_SIGNAL};" in outcome.line, outcome.line
    finally:
        server.close()
        registry.unpublish(target.pid)


@pytest.mark.asyncio
async def test_a_dead_pid_behind_a_leaving_record_is_reported_as_gone(
    no_signals, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal is liveness-gated: a stale record cannot refuse a stop.

    A drain always ends in an exit, and a record outlives its process by a
    moment. The honest report for a target that has already gone is the ladder's
    own "already exited" — a CLEAN resolution, exit 0 — rather than a refusal
    about a drain that is over. So the check needs an alive pid, and this cell is
    the other half of that condition.
    """
    from local_operator.session.runtime.types import LEAVING_ON_SIGNAL

    server, record = await _serve()
    server.close()
    target = _record_for(record, leaving=LEAVING_ON_SIGNAL)
    monkeypatch.setattr(control.registry, "pid_alive", lambda *_a, **_k: False)
    try:
        outcome = await control.stop_session(target, timeout_s=0.2, _root=config_dir())
        assert outcome.method == "gone", outcome.line
        assert outcome.method in control.ENDED_METHODS
    finally:
        registry.unpublish(target.pid)


@pytest.mark.asyncio
async def test_refresh_moves_a_stale_idle_runtime_over_the_real_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``lop refresh`` against a REAL runtime: the op moves it, and it says so.

    The whole acceptance argument for the rotation path is that it ends no
    session and cuts no work, so this drives the real dial and the real op
    rather than a stubbed reply: the runtime answers ``retiring`` and its own
    ``request_stop`` runs, which is exactly what "moved" means.
    """
    handle = _RefreshableHandle(reason="")
    server, record = await _serve(handle)
    _make_stale(monkeypatch, server)
    try:
        outcome = await control.refresh_session(_record_for(record), timeout_s=3.0)
        assert outcome.method == "moved", outcome
        assert "retiring now" in outcome.line
        # And it NAMES the build it is leaving for (NIT, PR #1141): "which of
        # these is still on the old build" is the question this command exists
        # to answer, and a version-only label cannot answer it on a host whose
        # common handover is a same-version rebuild. The label comes from the
        # runtime's own committed decision, not a second disk read.
        assert "for the build on disk (0.49.9)" in outcome.line, outcome.line
        assert handle.stops == [True], "the runtime really was asked to leave"
    finally:
        server.close()


@pytest.mark.asyncio
async def test_an_unsettled_install_is_reported_as_unsettled_not_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D1/M2 at the front end: the runtime's hedge survives into the METHOD.

    The runtime answers two different sentences now — "build on disk matches"
    and "the install on disk has not settled yet" — and this is the half that
    matters to a script: only the first is a settled outcome (exit 0). Collapsing
    the second into ``current`` is exactly the defect, because `lop refresh`'s
    own documentation says its first run is `lop-update`, so it lands inside
    ``BUILD_SETTLE_S`` for a whole fleet.
    """
    server, record = await _serve()
    try:

        async def _answer(*_args: Any, **_kwargs: Any) -> dict[str, str]:
            return {"op": "ack", "detail": "kept: the install on disk has not settled yet"}

        monkeypatch.setattr(control, "_exchange", _answer)
        outcome = await control.refresh_session(_record_for(record), timeout_s=1.0)
        assert outcome.method == "unsettled", outcome
        assert (
            outcome.method not in control.REFRESH_SETTLED_METHODS
        ), "an unsettled install is NOT a completed rotation"
        assert "ask again" in outcome.line, outcome.line

        # The sibling answer keeps its own, settled meaning: the two are one
        # line of code apart and must not be merged by a later reader. The stamp
        # and the marker are set so that "current" is PROVEN here rather than
        # assumed — the record says it runs the build on disk, the disk says the
        # same thing, and the marker has aged past the settle, which is the only
        # combination that earns a zero exit.
        from local_operator import update as update_mod
        from local_operator.update import BuildStamp

        monkeypatch.setattr(
            update_mod,
            "installed_build",
            lambda *_a, **_k: BuildStamp(version="0.49.9", source_ref="f4a70b9cdef"),
        )
        monkeypatch.setattr(
            update_mod,
            "disk_build",
            lambda *_a, **_k: BuildStamp(version="0.49.9", source_ref="f4a70b9cdef"),
        )
        monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 999.0)

        async def _matches(*_args: Any, **_kwargs: Any) -> dict[str, str]:
            return {"op": "ack", "detail": "kept: build on disk matches"}

        monkeypatch.setattr(control, "_exchange", _matches)
        matching = _record_for(record, version="0.49.9", source_ref="f4a70b9cdef")
        matched = await control.refresh_session(matching, timeout_s=1.0)
        assert matched.method == "current", matched
        assert matched.method in control.REFRESH_SETTLED_METHODS
    finally:
        server.close()


@pytest.mark.asyncio
async def test_the_retired_hedge_is_settled_here_instead_of_read_as_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D1/U6/O1: the fleet alive when the install moves answers the OLD sentence.

    ``lop refresh``'s own docstring says its first run is ``lop-update``, so the
    command is invoked INSIDE the settle window — and inside that window every
    live runtime is still executing the PREVIOUS build's code, which knows ONE
    sentence for both "matches" and "has not settled". Folding that sentence
    into ``current`` (exit 0) tells a rotating script the fleet is done while
    every member of it is about to retire: the harm the runtime-side fix exists
    to remove, on a runtime that cannot carry the fix. The design, UX and QA
    rounds reproduced it independently on this head (design D1, UX U6, QA O1,
    all three quoting the machine having the real tool install as the witness).

    So the CLI settles the question itself, from the stamp the record PUBLISHED
    and the marker on disk now. Both directions are asserted, because a check
    that answered ``unsettled`` for everything would be as wrong as the one it
    replaces: the same record, past the settle, is honestly ``current``.
    """
    from local_operator import buildwatch
    from local_operator import update as update_mod
    from local_operator.update import BuildStamp

    server, record = await _serve()
    try:

        async def _hedge(*_args: Any, **_kwargs: Any) -> dict[str, str]:
            # Verbatim the pre-#1141 runtime's answer, quoted from the constant
            # rather than retyped: if that sentence ever moves, this cell fails
            # loudly instead of quietly testing a string nobody sends.
            return {"op": "ack", "detail": buildwatch.KEPT_MATCHES_OR_UNSETTLED}

        monkeypatch.setattr(control, "_exchange", _hedge)
        target = _record_for(record, version="0.49.8", source_ref="46a4e9b1234567")
        monkeypatch.setattr(
            update_mod,
            "installed_build",
            lambda *_a, **_k: BuildStamp(version="0.49.9", source_ref="f4a70b9cdef"),
        )
        monkeypatch.setattr(
            update_mod,
            "disk_build",
            lambda *_a, **_k: BuildStamp(version="0.49.9", source_ref="f4a70b9cdef"),
        )
        # The install moved, and NOBODY has judged it yet — which is what the
        # hedge admits with its second clause and what the old caller threw away.
        monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 0.5)
        outcome = await control.refresh_session(target, timeout_s=1.0)
        assert outcome.method == "unsettled", outcome
        assert outcome.method not in control.REFRESH_SETTLED_METHODS
        assert "ask again" in outcome.line, outcome.line

        # Past the settle the same hedge is the true answer: the disk has not
        # moved beyond what the record says, so there is nothing to ask again
        # for. The ladder still does not call it ``moved`` — the runtime never
        # committed to retiring — but it is settled.
        monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 999.0)
        settled = await control.refresh_session(target, timeout_s=1.0)
        assert settled.method == "current", settled
        assert settled.method in control.REFRESH_SETTLED_METHODS

        # A record that published NO stamp cannot be second-guessed: with
        # nothing to compare, the runtime's own answer stands.
        monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 0.5)
        anonymous = await control.refresh_session(_record_for(record), timeout_s=1.0)
        assert anonymous.method == "current", anonymous
    finally:
        server.close()


def test_the_retired_hedge_is_a_prefix_of_the_live_answer() -> None:
    """The cross-version contract itself, pinned in one assertion.

    A runtime started before this change answers ``kept: build on disk matches
    (or has not settled)`` and will keep answering it until build skew retires
    it, so the matcher has to keep accepting that sentence for ever. It does so
    by matching the shorter, live answer as a PREFIX — the two constants are one
    edit apart and the day somebody "tidies" the retired string out of the
    module is the day the fleet that exists at update time stops being routed at
    all: the answer falls to the generic ``kept`` branch and a diagnosis becomes
    "was not moved: …".
    """
    from local_operator import buildwatch

    assert buildwatch.KEPT_MATCHES_OR_UNSETTLED.startswith(buildwatch.KEPT_MATCHES)
    assert buildwatch.KEPT_MATCHES != buildwatch.KEPT_MATCHES_OR_UNSETTLED


def _make_stale(monkeypatch: pytest.MonkeyPatch, server: Any) -> None:
    """The disk now carries a NEWER build than the one ``server`` booted on.

    The same three inputs ``test_server_refresh`` uses, and they are the whole
    gate: the op compares the boot stamp with the install on disk before it ever
    asks whether the runtime is idle, so a runtime on the CURRENT build answers
    "already current" whatever it is busy with. That ordering is what makes the
    queued move below meaningful — the build really has moved, and the only
    thing left is whether this runtime may leave yet.
    """
    from local_operator import update as update_mod
    from local_operator.update import BuildStamp

    server._boot_build = BuildStamp(version="0.49.8", source_ref="46a4e9b1234567")
    monkeypatch.setattr(
        update_mod, "installed_build", lambda *_a, **_k: BuildStamp(version="0.49.9")
    )
    monkeypatch.setattr(update_mod, "disk_build", lambda *_a, **_k: BuildStamp(version="0.49.9"))
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 999.0)
    monkeypatch.delenv("LOP_BUILD_PREFIX", raising=False)


@pytest.mark.asyncio
async def test_refresh_reports_a_busy_runtime_as_a_queued_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A busy runtime is NOT a failure: it retires by itself when its turn ends.

    The move is queued rather than refused, and that is the difference from the
    kill switch: nothing is ending this session, so its own reaper gets to
    decide when it can leave. It also needs no bound of its own — a turn that
    runs for an hour is simply waited out by the process that owns it.
    """
    handle = _RefreshableHandle(reason="busy")
    server, record = await _serve(handle)
    _make_stale(monkeypatch, server)
    try:
        outcome = await control.refresh_session(_record_for(record), timeout_s=3.0)
        assert outcome.method == "busy", outcome
        assert "a turn in flight" in outcome.line
        assert handle.stops == [], "a busy runtime is never retired by the ask"
    finally:
        server.close()


@pytest.mark.asyncio
async def test_refresh_reports_a_current_runtime_as_nothing_to_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative arm of the rotation: a matching build is left alone.

    The disk still carries the build this runtime booted from, which is the
    ordinary state of a host that has not run ``lop-update`` since — and the
    state in which ``lop refresh`` must say "nothing to do" rather than retire
    a working runtime for a build it is already running.
    """
    from local_operator import update as update_mod
    from local_operator.update import BuildStamp

    handle = _RefreshableHandle(reason="")
    server, record = await _serve(handle)
    same = BuildStamp(version="0.49.8", source_ref="46a4e9b1234567")
    server._boot_build = same
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: same)
    monkeypatch.setattr(update_mod, "disk_build", lambda *_a, **_k: same)
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 999.0)
    monkeypatch.delenv("LOP_BUILD_PREFIX", raising=False)
    try:
        outcome = await control.refresh_session(_record_for(record), timeout_s=3.0)
        assert outcome.method == "current", outcome
        assert "already runs the build on disk" in outcome.line
        assert handle.stops == []
    finally:
        server.close()


@pytest.mark.asyncio
async def test_refresh_reports_an_unreachable_runtime_without_failing(
    tmp_path: Path,
) -> None:
    """A silent socket is reported, not fatal — and it is the partial case."""
    record = SessionRecord(
        pid=2**22 + 71,
        kind="daemon",
        session_id="silentsession",
        conversation_name="the quiet one",
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="k",
        version="0.49.8",
    )
    outcome = await control.refresh_session(record, timeout_s=0.2)
    assert outcome.method == "unreachable"
    assert "did not answer its control socket" in outcome.line
    assert outcome.method not in control.REFRESH_SETTLED_METHODS
    assert tmp_path.exists()  # no ambient config root is touched by the dial


@pytest.mark.asyncio
async def test_refresh_all_asks_only_live_sessions_and_reports_each(
    no_signals, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--all`` walks the record set and reports per session, never signalling."""
    handle = _RefreshableHandle(reason="busy")
    server, record = await _serve(handle)
    _make_stale(monkeypatch, server)
    try:
        live = _record_for(record, conversation_name="working")
        ghost = _record_for(
            record,
            pid=2**22 + 73,
            session_id="gonequiet",
            conversation_name="quiet",
        )
        monkeypatch.setattr(control, "_stop_targets", lambda root, own_pid=None: [live, ghost])
        outcomes = await control.refresh_all(timeout_s=0.2, _root=tmp_path)
        by_method = {o.method: o for o in outcomes}
        assert by_method["busy"].session_id == live.session_id
        assert by_method["unreachable"].session_id == ghost.session_id
        assert no_signals[0] == [], "the rotation path signals nobody, ever"
        summary = control.summarize_refresh(outcomes)
        assert "1 will move when its turn ends" in summary
        assert "1 unreachable" in summary
    finally:
        server.close()


def test_summarize_refresh_on_an_empty_machine_says_so() -> None:
    assert control.summarize_refresh([]) == "no live sessions to refresh"
