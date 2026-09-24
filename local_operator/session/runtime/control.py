"""The kill switch: stop one session, or every session on this machine.

``stop_session`` / ``stop_all`` are the ONE implementation of "end that
agent". The TUI's ``/stop``, the CLI's ``lop stop`` and (later) the phone all
call them; a second implementation of "stop" is the kind of drift that gets a
process killed by the wrong branch of an escalation ladder. The design calls
this control-plane rule "one implementation, three front ends"
(design §12, the detached-architecture series).

Import-light by contract, matching :mod:`.registry` and :mod:`.types`: `lop
stop` runs on the CLI startup path, so nothing heavy may load at module
import. asyncio is imported inside the functions that dial a socket, exactly
as :func:`local_operator.session.runtime.serving.spawn_owned_session` defers
its heavy imports. ``tests/unit/test_import_graph.py`` is the guard.

**The escalation ladder** (see :data:`SIGTERM_GRACE_S` and
:data:`SIGKILL_CONFIRM_S` for the two rungs' budgets, which differ on purpose):
a graceful ``stop`` control op the runtime serves itself → a SKIP for
a target whose record reports a turn in flight → SIGTERM (the runtime's signal
handler defers its own disposal to the end of that turn, bounded by
``SIGNAL_DRAIN_S``; ``SIGTERM_GRACE_S`` outlasts the bound so this rung cannot
escalate inside it) → SIGKILL (state orphaned; the existing stale-record/lease
machinery recovers, and the outcome is reported as ``killed``). The skip and
the drain are the two halves of one rule — this ladder is one of the senders
that must not cut work in flight, and the receiver is the one that can tell
whether it is mid-turn.

**Pid-reuse safety.** Before ANY signal is sent, the target's identity is
confirmed. A pid is not proof of identity — a SIGKILLed runtime leaves its
record behind for up to one scan, and the OS may have recycled the pid into
an unrelated process — and killing the wrong process is the one
unrecoverable mistake a kill switch can make. Two proofs, tried in order:

1. **The socket.** The session id in the record must match the session id
   the process at that pid serves, read off the welcome frame every runtime
   pushes on an authenticated dial. A process that answers with a DIFFERENT
   id is a live stranger and is refused outright.
2. **The start time**, only when the socket is silent (the wedged runtime —
   alive, not answering — is the one the kill switch exists for). The
   process must have started before the record's last heartbeat; a recycled
   pid cannot pass, because the stranger began after the recorded process
   died, which was after it last heartbeat.

When neither proof can be made the stop REFUSES (``refused``) and says why;
refusing is always acceptable for a kill switch, killing the wrong process
never is. Stale-record reaping then cleans the file up on the next scan,
which is the correct recovery for a dead owner.

**Never a process-group signal.** Backgrounded ``bash`` jobs deliberately
outlive their turn, and :mod:`local_operator.tools.group_reaper` owns their
lifecycle. Escalating to ``killpg`` would tear through a job the user asked
to keep running — the exact thing ``background=true`` exists for — so every
signal here targets the single recorded pid.

**Every rung leaves durable evidence BEFORE it acts.** Each rung stages a stop
marker (``registry.STOP_MARKER_NAME``) into the target's conversation
directory immediately before the irreversible thing it does — the target's own
clean exit once the ``stop`` op has been acked, or the signal it sends itself.
The kill switch is the only party that CAN attest to a forced stop: at the
SIGKILL rung the target is not executing, and
the 2026-09-13 kill wave is what the absence of that attestation cost — the
same event reached the operator as ``runtime-killed``, as no cause at all, and
as ``owner-lost``, because the classifier could only reconstruct a deliberate
act out of a missing record and a silent socket. The marker is what turns "the
runtime disappeared without exiting cleanly" into "the user's stop reached the
sigkill rung, run by pid N". It is staged per rung and OVERWRITTEN as the
ladder escalates, so the file always names the rung that actually acted — and
withdrawn when the ladder refuses, because a refusal signs nothing (see
:func:`_withdraw_staged_stop_marker`). "Durable" here is process-durability:
the file is visible to every reader before the signal, which is the semantics
this needs (the target dies; the host does not) — see :func:`_write_stop_marker`
for what that does and does not cover.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from local_operator import procstate
from local_operator.buildwatch import KEPT_MATCHES, KEPT_UNSETTLED, moved_and_unsettled
from local_operator.paths import config_dir
from local_operator.session.runtime import registry
from local_operator.session.runtime.types import (
    HEARTBEAT_TIMEOUT_S,
    RUN_DIRNAME,
    SIGNAL_DRAIN_S,
    SessionRecord,
    bound_text,
    session_dir,
)

if TYPE_CHECKING:
    from pathlib import Path

#: What ``/resume`` does to the session you are LEAVING when the setting is
#: absent. True keeps its runtime working in the background — the whole point
#: of a session that outlives the terminal, and the behaviour a user switching
#: between two conversations expects. False is for anyone who would rather a
#: session they walked away from stop spending tokens; they can still leave it
#: running deliberately with ``/stop`` as the explicit end.
DEFAULT_BACKGROUND_ON_RESUME = True


#: Which stop method produced an outcome, in escalation order. The runtime's
#: own control socket (``socket``) is the graceful rung; the two signals are
#: the escalation; the remaining three name the non-signalled resolutions.
#: ``gone`` is a process that left the table before the ladder reached it
#: (already exited — nothing to do, not a failure); ``refused`` is the
#: ladder declining to signal because identity could not be confirmed;
#: ``busy`` is the ladder declining to signal because the target reports a
#: turn in flight (see :func:`stop_session`). Kept distinct so a front end can
#: decide "partial" from the method alone rather than by parsing the receipt
#: line.
Method = str  # "socket" | "sigterm" | "sigkill" | "gone" | "refused" | "busy"


def _from_a_shell(command: str) -> bool:
    """Did this stop come from the CLI rather than the TUI?

    Read off ``_command``, the front end's own name for the request, which both
    callers already set verbatim for the stop marker (``lop stop`` / ``lop stop
    --all`` from the CLI, ``/stop`` / ``/stop --all`` from the TUI). Deriving it
    from the value they already pass keeps ONE convention alive rather than
    adding a second parameter that would have to be kept in step with it.
    """
    return not command.startswith("/")


def _force_remedy(pid: int, from_a_shell: bool) -> str:
    """The one next step a refusal may name, in words the CALLER can act on.

    A SURFACE MUST NEVER OFFER AN ACTION IT CANNOT ACCEPT (UX round 2, U7).
    The refusal below is composed here, in the module both front ends share,
    and it used to name ``--force`` unconditionally — which the TUI paints
    verbatim and cannot parse: ``/stop --force <id>`` answers "no live session
    matches '--force <id>'" about a session that is live and listed one
    keystroke away. The flag is real and worth naming, so the fix is to name it
    in the vocabulary of the surface that will read the line: ``lop stop``
    takes it directly, and the TUI's route to it is a shell (the pattern the
    app already uses elsewhere for a remedy it does not own).
    """
    if from_a_shell:
        return "--force to stop it anyway"
    return f"to force it, run lop stop --force {pid} in a shell"


def _wait_line(name: str, pid: int, *, from_a_shell: bool) -> str:
    """The bound-bearing line a rung announces before it spends it.

    Both forms name the bound for the reason ``on_wait`` exists (U5): the pause
    is minutes long and a front end that says nothing for its whole length
    reads as a hang. Only the vocabulary differs, and ``SIGKILL``/"drain" are
    the CLI's: the TUI paints this line verbatim and promised "waiting for it
    to answer" in its own copy, so entering the kill vocabulary there for the
    first time is a sentence the app has never used (design round 2, D2).
    """
    if from_a_shell:
        return (
            f'waiting up to {bound_text(SIGTERM_GRACE_S)} for "{name}" (pid {pid}) '
            "to drain before SIGKILL"
        )
    return (
        f'waiting up to {bound_text(SIGTERM_GRACE_S)} for "{name}" (pid {pid}) '
        "to answer; it is killed at the bound if it never does"
    )


#: How long to wait, after the graceful ``stop`` op is acked, for the process
#: to actually exit before escalating to SIGTERM. The op acks before its clean
#: exit finishes (gates deny, dispose, unpublish), and 10 s comfortably covers
#: a session draining an interruptible tool on a loaded machine — the default
#: every front end can afford — while ``lop stop --timeout`` lets a script
#: trade patience for promptness.
DEFAULT_TIMEOUT_S = 10.0

#: The SIGTERM rung's budget: how long the ladder waits for a signalled
#: runtime to exit before escalating to SIGKILL.
#:
#: THE INVARIANT: this MUST be longer than ``types.SIGNAL_DRAIN_S``, the bound
#: on how long a runtime with a turn in flight defers its own disposal after a
#: signal (``process._drain_for_signal``). A ladder that escalated inside that
#: window would SIGKILL a runtime that was deliberately, correctly finishing a
#: turn — the escalation would destroy the very work the receiver's drain exists
#: to save, and do it with the one signal nothing can catch. Derived rather than
#: written as a second number so the two cannot drift apart, plus a margin: the
#: receiver still has to deny parked gates, dispose, flush the transcript and
#: unpublish its record after the drain closes before its pid goes away.
#:
#: The cost is honest and bounded: a target that is silent on its socket AND
#: reports work in flight takes this long to resolve, because the ladder cannot
#: tell "draining politely" from "wedged" while the socket is silent. That is
#: the correct trade — see the drain's own docstring — and ``stop_session``
#: skips a target whose record still says it is busy, so the wait is only paid
#: for a runtime whose published state has gone stale.
SIGTERM_GRACE_S = SIGNAL_DRAIN_S + 30.0

#: How long to wait for a SIGKILLed process to actually disappear.
#:
#: NOT the same budget as ``SIGTERM_GRACE_S``, deliberately, and the difference
#: is what the two rungs mean. SIGTERM asks, so its grace has to outlast the
#: receiver's own drain (see above); SIGKILL cannot be refused at all, so there
#: is nothing to wait FOR — this covers only a process wedged in an
#: uninterruptible syscall, and it is the 3 s the mobile child already used as
#: its process drain budget. Spending the SIGTERM grace here too would add two
#: and a half minutes to every escalated kill and buy no information.
SIGKILL_CONFIRM_S = 3.0

#: Budget for one identity-confirming socket round trip. Identity
#: confirmation is a ping-class exchange, not a turn: if the runtime cannot
#: answer inside this window it is wedged, and the ladder's job is to report
#: that rather than to hold the caller hostage. Wedged is not refused — the
#: identity check never ran, so the refuse rule does not apply — and SIGTERM
#: (which a wedged runtime's handler still runs, via the signal thread) is
#: the correct next rung.
_IDENTITY_TIMEOUT_S = 3.0

#: The one ``why_not`` the ladder treats as "try the other proof": the socket
#: did not answer at all (as opposed to answering with someone else's id).
_SOCKET_SILENT = "did not answer the control socket"

#: Bytes buffered for a SINGLE frame before it is judged oversized and
#: discarded (see ``_read_frames_until``). Generous for any ack or identity
#: welcome, bounded so a pathological runtime cannot make this module buffer
#: without limit. Same values ``peer_client`` chose, for the same reason.
_MAX_FRAME_BYTES = 1 << 23
#: One socket read: the granularity at which bytes are pulled from the kernel
#: while scanning for a newline. Independent of the frame cap above.
_READ_CHUNK = 1 << 16


@dataclass
class StopOutcome:
    """What happened when one target was stopped.

    ``method`` names the rung that worked; ``line`` is the human receipt the
    front ends paint verbatim, so the TUI, the CLI and the phone report the
    same event in the same words. ``wakes_dormant`` carries the count of
    schedules parked dormant by the stop (see :func:`_mark_wakes_dormant`) for
    front ends that compose their own receipt.
    """

    pid: int
    session_id: str
    name: str
    method: Method
    line: str
    wakes_dormant: int = 0


@dataclass
class _Dial:
    """One daemon-class control-socket conversation with a runtime."""

    reader: Any
    writer: Any


async def _dial(record: SessionRecord, timeout_s: float) -> _Dial | None:
    """Connect and authenticate against ``record``'s control socket.

    Returns ``None`` when the connection is refused — for the escalation
    ladder that is not an error but the cheapest possible answer: the process
    is already gone, so the rung above the socket is the stale-record reap.
    ``lop send``'s sender client (``mobile/peer_client.py``) dials the same
    daemon-class way for the same reason: a stop must not perturb the
    runtime's attach accounting (a dial as ``client: "attach"`` would count
    against ``ATTACH_MAX_CLIENTS`` and hold a runtime warm, and this whole
    module exists to do the opposite).
    """
    import asyncio
    import json

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", record.control_port),
            timeout=timeout_s,
        )
    except (OSError, TimeoutError):
        return None
    try:
        writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
        await writer.drain()
    except (OSError, ConnectionError):
        # The port accepted the connect but died before auth completed. Same
        # answer as a refused connect for the same reason: already gone.
        writer.close()
        return None
    return _Dial(reader=reader, writer=writer)


async def _read_frames_until(dial: _Dial, predicate: Any, timeout_s: float) -> Any:
    """Read newline-framed control frames until ``predicate(frame)`` holds.

    A daemon-class dial receives the runtime's unsolicited ``welcome``
    projection FIRST — the same sequencing ``peer_client.send_peer_message``
    handles with its own no-line-limit reader. Projections are the identity
    answer (see :func:`_confirmed_session_id`) and never the ack this
    function is asked to wait for, so they are skipped here.

    Read in bounded chunks and framed HERE rather than by ``readline``:
    a projection is unbounded in principle (a large transcript tail in one
    line), and ``StreamReader.readline`` raises ``LimitOverrunError`` instead of
    RETURNING such a line — the frame can never be read, so one big welcome
    leaves the ack this function waits for unreachable (the defect U1 fixed for
    `lop send`). ``readline`` does drain the offending bytes (CPython deletes
    through the separator when it found one), so a later read would not
    re-raise; that is why the cost of a skip is "a frame is lost", not "the
    reader wedges" — and a lost welcome is exactly the loss this reader exists
    to avoid. Discarding an oversized line (over ``_MAX_FRAME_BYTES`` with no
    newline yet) keeps memory bounded while the frames AFTER it — including
    the ack this function may be waiting for — survive in the same buffer.
    """
    import asyncio
    import json

    buf = bytearray()
    skipping = False

    async def _next_line() -> bytes:
        # `buf` is mutated in place, never rebound, so it needs no nonlocal;
        # `skipping` is a reassigned flag and does.
        nonlocal skipping
        while True:
            nl = buf.find(b"\n")
            if nl != -1:
                line = bytes(buf[: nl + 1])
                del buf[: nl + 1]
                if skipping:
                    # This newline ends the oversized line being discarded;
                    # anything after it is a real frame again.
                    skipping = False
                    continue
                return line
            if len(buf) > _MAX_FRAME_BYTES:
                buf.clear()
                skipping = True
                continue
            chunk = await dial.reader.read(_READ_CHUNK)
            if not chunk:
                raise ConnectionError("runtime closed the connection")
            buf.extend(chunk)

    async def _read() -> Any:
        while True:
            line = await _next_line()
            try:
                frame = json.loads(line.decode("utf-8", "replace"))
            except ValueError:
                continue  # noise on an authenticated loopback socket
            if not isinstance(frame, dict):
                continue
            if predicate(frame):
                return frame

    return await asyncio.wait_for(_read(), timeout=timeout_s)


async def _close_dial(dial: _Dial) -> None:
    try:
        dial.writer.close()
        await dial.writer.wait_closed()
    except (OSError, ConnectionError):
        pass


async def _exchange(
    record: SessionRecord, op: dict[str, Any], *, reply_timeout_s: float
) -> dict[str, Any] | None:
    """Send one op and return its matching reply, or ``None`` if unreachable.

    ``None`` means only "no conversation was possible" (refused connect,
    half-open socket, timeout). A runtime that ANSWERS with an error frame is
    reachable — the caller decides what the error means for its rung.
    """
    import json

    dial = await _dial(record, _IDENTITY_TIMEOUT_S)
    if dial is None:
        return None
    try:
        req = 1
        op_frame = {"op": op["op"], "req": req, **{k: v for k, v in op.items() if k != "op"}}
        dial.writer.write(json.dumps(op_frame).encode() + b"\n")
        await dial.writer.drain()
        return await _read_frames_until(
            dial,
            lambda frame: frame.get("req") == req and frame.get("op") in ("ack", "error"),
            timeout_s=reply_timeout_s,
        )
    except (OSError, ConnectionError, TimeoutError):
        return None
    finally:
        await _close_dial(dial)


async def _confirmed_session_id(record: SessionRecord) -> tuple[bool, str]:
    """Confirm the record's identity against the process serving its port.

    Returns ``(confirmed, why_not)``. The identity answer is the WELCOME
    projection every runtime pushes unsolicited on an authenticated dial:
    its ``session_id`` is read live off the handle the process at that port
    is serving RIGHT NOW. That is the whole trick — the record says "pid N
    serves session S"; the socket answer says "the process listening on
    this port is serving session S"; when they agree, pid N really is the
    owner of session S and a signal aimed at N cannot hit a pid-recycled
    stranger.

    Deliberately not the ``stop`` op, and not ``snapshot`` either: the
    welcome has been the first frame since the first version that ever
    published a record, so confirmation works against every runtime in
    existence, including one too old to know any op this module sends.
    """
    dial = await _dial(record, _IDENTITY_TIMEOUT_S)
    if dial is None:
        # Unreachable: nothing confirmed. The caller decides — a wedged or
        # dead runtime escalates; a live-but-unknown one refuses.
        return False, _SOCKET_SILENT

    try:
        frame = await _read_frames_until(
            dial,
            lambda frame: frame.get("op") == "projection" and isinstance(frame.get("data"), dict),
            timeout_s=_IDENTITY_TIMEOUT_S,
        )
    except (OSError, ConnectionError, TimeoutError, ValueError):
        return False, _SOCKET_SILENT
    finally:
        await _close_dial(dial)

    data = frame.get("data") or {}
    session_id = str(data.get("session_id") or "")
    if session_id and session_id == record.session_id:
        return True, ""
    return (
        False,
        f'it serves session "{session_id}", not "{record.session_id}"',
    )


def _process_started_at(pid: int) -> float | None:
    """Epoch seconds the process at ``pid`` started, or ``None`` if unknown.

    ``ps -o lstart=`` is the one portable, stdlib-reachable source (macOS
    has no ``/proc``; ``psutil`` is deliberately not a dependency). One-second
    resolution, which is enough for the comparison it feeds.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    for fmt in ("%a %b %d %H:%M:%S %Y", "%a %d %b %H:%M:%S %Y"):
        try:
            return time.mktime(time.strptime(out, fmt))
        except ValueError:
            continue
    return None


def _identity_by_record(record: SessionRecord) -> tuple[bool, str]:
    """Identity from the record plus what the OS says about that pid.

    Admissible ONLY under ``--force`` AND only when the socket was SILENT
    (the caller gates both): the socket answer is the load-bearing proof
    everywhere else, and this fallback exists for the one shape the socket
    cannot reach — a heartbeating-but-starved process (a TUI burning 100%
    CPU in a tight tool-error loop, its socket loop queued behind the
    runaway) that the refusal rule would otherwise hold forever.

    What is actually checked — stated precisely, because the next person to
    widen this will trust this paragraph. NOTHING here asks the process what
    session it serves; that question is the socket's, and a socket that
    answers a DIFFERENT session is a live stranger that stays refused rather
    than reaching this function at all. The control key is likewise never
    verified against the process. Three facts must all hold:

    1. ``_same_uid`` — the record on disk is ours and still names this pid
       and session. Self-consistent by construction: it compares the record
       against the same file, so it adds no evidence about the PROCESS. It
       is an authorization and freshness check, not an identity proof.
    2. A heartbeat fresher than the timeout, and the pid still HOLDING the
       recorded control port. Both are WINDOW checks: a lapsed beat says the
       recorded owner stopped reporting, and a port that has moved on says the
       name belongs to a different process now. Neither asks the process
       anything, which is why this clause narrows the accident rather than
       closing it — the beat is written by the runtime's OWN event loop (see
       ``registry.classify``), so it is not evidence about the socket loop and
       cannot separate "alive but not answering" from "record outlived its
       process" on its own.
    3. The process did not start AFTER the record's last heartbeat. This is
       the clause that carries the weight: (1) and (2) together are still
       satisfiable by a stranger that inherited a dead lop's pid AND its
       ephemeral port inside the heartbeat window, which is a real accident
       rather than a hypothetical — it was reproduced deterministically
       against unrelated processes (round-3 Q3-1).
    """
    if not _same_uid(record):
        return False, "the record on disk no longer names this pid and session"
    # A lapsed heartbeat means the recorded process is gone, whatever now
    # holds its pid. Refusing here is the conservative answer a kill switch
    # owes: --force widens WHICH proof is admissible, never whether one is.
    age = time.time() - record.heartbeat_at
    if age > HEARTBEAT_TIMEOUT_S:
        return False, (
            f"its record stopped heartbeating {int(age)}s ago, so the pid may "
            f"belong to another process now; identity cannot be confirmed"
        )
    if not _pid_holds_port(record):
        return False, "the pid no longer holds the control port its record claims"
    # Third clause, and the one that closes the actual accident: a stranger
    # can inherit BOTH a dead lop's pid and its ephemeral port inside the
    # heartbeat window — ports are recycled alongside pids, so the port check
    # bounds that window rather than closing it, and QA killed unrelated
    # processes four times over deterministically on the two clauses alone
    # (round-3 Q3-1). A process that started AFTER the record's last
    # heartbeat cannot be the process that wrote it, whatever pid and port it
    # now holds. Measured not to cost the case --force exists for: on a
    # genuinely starved runtime the start time precedes the heartbeat, so the
    # legitimate target is still admitted.
    started = _process_started_at(record.pid)
    if started is None:
        return False, "could not read the process start time; identity cannot be confirmed"
    if started > record.heartbeat_at + 1.0:
        return (
            False,
            "the process started after the record's last heartbeat "
            "(the pid was reused); identity cannot be confirmed",
        )
    return True, ""


def _pid_holds_port(record: SessionRecord) -> bool:
    """True when ``record.pid`` still owns the record's control port.

    The kernel's own answer to "is this pid the process that published this
    record": the port was bound at spawn, so a pid that holds it IS the
    runtime rather than a stranger who inherited the number. ``lsof`` is the
    portable reach for this (macOS has no /proc and psutil is deliberately
    not a dependency); an unreadable answer means unproven, which refuses.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["lsof", "-nP", "-a", "-p", str(record.pid), "-iTCP", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    # Anchored: lsof prints "TCP 127.0.0.1:53213 (LISTEN)", so an unanchored
    # ``":5321" in out`` matches every PREFIX of a real listening port. Parse
    # the address column and compare the port as an integer instead — this
    # function is the load-bearing half of a fix for a bug that killed an
    # unrelated process, so a loose match here is exactly the wrong economy.
    for line in out.splitlines():
        for field in line.split():
            _, sep, port = field.rpartition(":")
            if sep and port.isdigit() and int(port) == record.control_port:
                return True
    return False


async def _identity_by_start_time(record: SessionRecord) -> tuple[bool, str]:
    """Identity for a process that will not answer its socket.

    A runtime whose owner has stopped reporting — alive, beat stale, socket
    silent — is the one the kill switch exists for, and the socket check cannot
    confirm it. So a second source that needs NO cooperation from the process is
    admitted, ONLY when the heartbeat is stale (see the inline comment for why a
    fresh heartbeat forbids it): the process at this pid must have STARTED
    before the record's last heartbeat. A
    recycled pid cannot pass that — the stranger now holding the pid began
    after the recorded process died, and it died after its last heartbeat
    (the heartbeat is the process proving it was alive). Same-host clock on
    both sides, so the comparison is sound; one second of slack covers
    ``ps``'s resolution.

    Also requires the record on disk to still name this pid and session
    (``_same_uid``): a record rewritten by a NEW lop that took the same pid
    would otherwise describe a different session.
    """
    if not _same_uid(record):
        return False, "the record on disk no longer names this pid and session"
    # Only a record whose beat has LAPSED earns this proof. A heartbeat inside
    # the timeout says the recorded process reported moments ago, so a silent
    # socket may be a loop that is simply behind (the beat is authored by the
    # runtime's own event loop — ``registry.classify``) or a stale file whose
    # pid a stranger now holds (seen live: a fresh fake record over a ``sleep``
    # passed the start-time check and was signalled). Neither can be proved
    # from the record's timing, so this rung refuses and names the opt-in.
    age = time.time() - record.heartbeat_at
    if age <= HEARTBEAT_TIMEOUT_S:
        # Name the remedy as the WHOLE COMMAND, not a bare `--force`: this
        # string is painted by the TUI's /stop as well as by the CLI, and the
        # TUI has no spelling for a flag (round-3 U3-2).
        #
        # The sentence promises NOTHING about waiting. An earlier version told
        # the user the heartbeat "must lapse (~45s) ... retry then", which is
        # advice the operator cannot act on with confidence: it is true only
        # while the owner keeps failing to report, and a loop that turns once
        # in that window resets it. The forced rung is reachable NOW for the
        # case this refusal exists for, and it says what it does.
        return (
            False,
            f"it is heartbeating but not answering its socket "
            f"(its last heartbeat was {int(age)}s ago), so a signal cannot be "
            f"proven to be aimed at it. Run `lop stop --pid {record.pid} --force` "
            f"from a shell to force-stop the process deliberately",
        )
    # Off the loop: this is a fork/exec of ``ps`` with a 5 s ceiling, and the
    # TUI runs the ladder on its event loop (``run_worker(thread=False)``).
    # A loaded host — exactly when someone reaches for a kill switch — is
    # where a synchronous call here would freeze the frame (the #401 class).
    started = await asyncio.to_thread(_process_started_at, record.pid)
    if started is None:
        return False, "could not read the process start time"
    if started > record.heartbeat_at + 1.0:
        return (
            False,
            "the process started after the recorded session's last heartbeat "
            "(the pid was reused)",
        )
    return True, ""


def _same_uid(record: SessionRecord) -> bool:
    """Defensive same-account check on a record before acting on it.

    The 0600 records under the 0700 run directory already make cross-account
    access impossible — anything that can READ the record (control key
    included) is already the owning account, which is the whole authorization
    model of the mobile stack. This check exists for the one gap the file
    mode cannot close: a record left world-readable by a downgraded directory
    mode on a machine nobody audited would otherwise let one user's ``--all``
    stop another user's agents. Refusing on a uid mismatch costs nothing and
    closes that gap; it is belt-and-braces, not the primary boundary.

    A record with no readable owner reads as not-ours, which refuses — the
    conservative answer for a kill switch.
    """
    import json

    path = registry.run_dir() / f"{record.pid}.json"
    try:
        stat = path.stat()
        if stat.st_uid != os.getuid():
            return False
        # The record on disk must still be THIS record: a recycled filename
        # (pid reused by a new lop) would otherwise pass the uid check while
        # describing a different session.
        data = json.loads(path.read_text())
        return data.get("session_id") == record.session_id
    except (OSError, ValueError):
        return False


def _mark_wakes_dormant(record: SessionRecord, root: Path) -> int:
    """Park a stopped session's wakes dormant by stamping ``stopped_at``.

    Schedules are NEVER deleted by a stop: the transcript is the authority
    (``Session._persist_wake_schedules``) and a stop that edited the model's
    schedule state would silently change what the agent was asked to do.
    Instead the wake INDEX entry — the derived file a cold process reads —
    gains ``stopped_at``, the key ``wakes/store.write_entry`` already
    preserves across rewrites and clears on the session's next open. The
    future wake supervisor skips entries carrying it (design §4.4/§12), so
    dormant means exactly "will not fire until someone reopens the session",
    and reopening is the un-stop.

    Returns the number of schedules parked (the receipt's "N wakes dormant"),
    or 0 when there is no entry — a session with no schedules has nothing to
    park, and absent-file-is-no-wakes is the store's own contract.
    """
    from local_operator.wakes import store as wake_store

    entry = wake_store.read_entry(root, record.session_id)
    if entry is None:
        return 0
    schedules = entry.get("schedules") or []
    if not schedules:
        return 0
    wake_store.write_entry(
        root,
        record.session_id,
        cwd=entry.get("cwd") or record.cwd,
        schedules=schedules,
        preserve=dict(entry, stopped_at=int(time.time() * 1000)),
    )
    return len(schedules)


async def _park_wakes(record: SessionRecord, root: Path) -> int:
    """``_mark_wakes_dormant`` off the loop: two small file operations, but
    on the same TUI loop as the ladder, and a cold disk under load is enough
    to show as a dropped frame. Best-effort — the index is derived."""
    try:
        return await asyncio.to_thread(_mark_wakes_dormant, record, root)
    except Exception:  # noqa: BLE001 — the index is derived; the stop is not
        return 0


def _record_retired(record: SessionRecord, root: Path) -> bool:
    """True once the record on disk no longer describes this session.

    The graceful op's observable end is NOT always a process exit: a
    TUI-owned session ends beneath a terminal that stays up, and its only
    trace of the stop is the unpublished (or rewritten) record. Reading
    the file rather than ``registry.scan`` keeps this a single stat+read on
    the 100 ms poll, and a file that now names a different session (the
    process reopened something else) counts as retired too.

    ``root`` is the root the CALLER targeted, never the ambient one. This
    read is the ladder's only proof that a graceful stop landed, and reading
    ``registry.run_dir()`` here made an injected-root caller answer against a
    DIFFERENT root's run directory: a missing file read as "retired", so the
    ladder returned a confident ``socket`` receipt — ``(socket, 0.01s)`` — for
    a stop that had been acked by nobody, while the target was still alive and
    still serving. Production callers pass ``root=config_dir()``, where the two
    agree; the defect was only reachable from another root (the phone's, a
    supervisor's, a test's), and that is precisely where the receipt was then
    taken at face value (design §1e).
    """
    import json

    path = root / RUN_DIRNAME / f"{record.pid}.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return True
    return data.get("session_id") != record.session_id


async def _await_stopped(record: SessionRecord, timeout_s: float, root: Path) -> bool:
    """Wait for the stop to land, bounded by ``timeout_s``.

    Landed means the pid left the process table (a runtime process) OR the
    record was unpublished (a TUI owner whose process survives with the
    session ended beneath it). Polls rather than waiting on a child: these
    processes are not this process's children, so there is no waitpid
    right to hang on. 100 ms is fine-grained enough that a clean exit
    (sub-second in the common case) is observed almost immediately, and
    cheap enough to hold for 10 s without cost. Returns False on timeout —
    the caller's next escalation rung.

    ``root`` rides through to :func:`_record_retired` so the check runs
    against the same root the caller targeted (see the note there).
    """
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        if not registry.pid_alive(record.pid) or _record_retired(record, root):
            return True
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.1)


async def _await_pid_exit(pid: int, timeout_s: float) -> bool:
    """The signal rungs' wait: only a process exit counts. A signalled
    process that stays up did not honour the signal, whatever its record
    says, and the next rung is the answer."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if not registry.pid_alive(pid):
            return True
        await asyncio.sleep(0.1)
    return not registry.pid_alive(pid)


def _stopped_line(
    record: SessionRecord, method: Method, wakes: int, *, forced: bool = False
) -> str:
    """The one human receipt line every front end paints for one stop.

    ``socket`` reads as a plain "stopped"; the signal rungs SAY so —
    ``stopped … (sigterm)`` / ``killed …`` — because a kill switch that
    reports an escalated stop as a graceful one hides the one fact the
    user would act on next time (that runtime was not answering its
    socket). ``sigkill`` gets its own verb: state was orphaned.

    ``forced`` names the proof that authorised the signal. A forced stop is
    the one outcome whose receipt has to say the socket proof was bypassed
    and identity came from the record instead, so the transcript records
    which evidence the kill rested on (round-3 D3-4).
    """
    name = record.conversation_name or record.session_id
    verb = "killed" if method == "sigkill" else "stopped"
    rung = " (sigterm)" if method == "sigterm" else ""
    proof = " (--force: identity confirmed from its record)" if forced else ""
    wakes_part = (
        f" — {wakes} wake{'s' if wakes != 1 else ''} dormant until you reopen it" if wakes else ""
    )
    return f'{verb} "{name}"{rung}{proof}{wakes_part}'


def _stop_marker_payload(
    record: Any,
    rung: Method,
    *,
    command: str,
    deliberate: bool = True,
    actor: str = "",
    mechanism: str = "",
) -> dict[str, Any]:
    """The durable evidence ONE act on a runtime leaves behind.

    Written by the PARTY THAT ACTS (this process), never by the target: at the
    SIGKILL rung the target is not executing and cannot record anything, which is
    why the 2026-09-13 wave could only be described as a runtime that
    "disappeared without exiting cleanly". The fields are exactly the facts a
    reader needs to attribute the death in one look:

    * ``session_id`` / ``pid`` / ``started_at`` — the RUN KEY. The classifier
      refuses a marker that does not match the run it is classifying, so a
      marker left by an EARLIER run of the same session cannot narrate a
      later, involuntary death as the user's own act.
    * ``rung`` — which rung actually acted, and therefore how hard the target
      resisted: ``socket`` is the runtime's own clean exit, ``sigterm`` its
      handler, ``sigkill`` nothing at all (state orphaned). Empty on an
      involuntary act, which has no rung — nothing asked the runtime to stop.
    * ``deliberate`` — what the act WAS, and the one field the classifier reads
      to decide which sentence this death gets. True from the ladder and only
      from the ladder (a stop a person asked for); false from every other party
      that can take a runtime away (see :func:`note_involuntary_stop`). A kill
      by a supervisor or by a stray shell must not set it — the flag is what
      keeps an involuntary death reading as one.
    * ``killer`` — pid, argv0 and the front end's command name, so "who did
      this" is answered by the artifact rather than by the operator's shell
      history — being unrecoverable from the artifacts on this host is
      exactly what made one incident read as three different stories.
    * ``actor`` / ``mechanism`` — present ONLY on an involuntary act (see
      :func:`note_involuntary_stop`): the acting component's own name and the
      machine token for what it did. Absent rather than empty on a deliberate
      stop, so a marker of that kind stays byte-identical to what every reader
      has already been taught.
    * ``build`` — the TARGET's ``version@source_ref``, because the question
      this answers is which runtime died.

    ``Any`` rather than ``SessionRecord`` because two record kinds carry the same
    run: a ``SessionRecord`` for a runtime that is up, and the ``BootRecord`` of
    one that is still booting (which is exactly the window the harness can take a
    tree away in). Both spell the build differently — ``version``/``source_ref``
    against ``build_version``/``build_ref`` — and duck-typing the four reads here
    keeps ONE payload shape for both instead of a second builder that could
    disagree.

    NO FREE-TEXT ``reason`` FIELD, and its absence is deliberate (design round
    1, D7): a constant sentence ("a deliberate stop was requested through the
    control plane") that no reader consumed sat in the artifact looking
    load-bearing, and the next agent to touch this schema would have assumed
    something rendered it. Every reader that needs a sentence renders one from
    the fields above on the spot — ``incidents.render_stop_attribution`` takes
    exactly the rung and the killer, ``incidents.render_involuntary_attribution``
    exactly the mechanism, actor and pid — so a stored copy could only drift from
    them. ``deliberate`` is what says an act was asked for, and it does so
    without prose.
    """
    version = str(getattr(record, "version", "") or getattr(record, "build_version", "") or "")
    source_ref = str(getattr(record, "source_ref", "") or getattr(record, "build_ref", "") or "")
    if version and source_ref:
        build = f"{version}@{source_ref}"
    else:
        build = version or source_ref or ""
    argv0 = os.path.basename(sys.argv[0] or "") or sys.executable
    payload: dict[str, Any] = {
        "session_id": record.session_id,
        "pid": record.pid,
        "started_at": record.started_at,
        "at": time.time(),
        "rung": rung,
        "deliberate": bool(deliberate),
        "killer": {"pid": os.getpid(), "argv0": argv0, "command": command},
        "build": build,
    }
    if not deliberate:
        payload["actor"] = actor
        payload["mechanism"] = mechanism
    return payload


def note_involuntary_stop(
    record: Any,
    root: Path,
    *,
    mechanism: str,
    actor: str = "",
    command: str = "",
) -> bool:
    """Stage the durable stop marker for a runtime an INVOLUNTARY act is about to take away.

    WHY EVERY HARNESS-CAUSED DEATH NEEDS ONE. The marker schema above already
    exists, is already read by the classifier, and is already written by the
    ladder that stops a session a person asked to stop — so an involuntary death,
    which is the kind that comes in waves, was the only kind that stayed
    anonymous. That is the gap the operator's requirement names: "nothing should
    kill runtimes en masse, ever; and if it does happen, it must be attributable."
    On 2026-09-18 twenty-five runtimes on one machine vanished inside thirteen
    seconds, each one leaving a record whose pid was gone and no statement about
    who took its tree — so an investigation could prove the wave happened and
    could not name a single actor.

    Called BEFORE the act, by the party that is about to perform it — the same
    ordering invariant :func:`_write_stop_marker` documents, and for the same
    reason: at the moment this matters the runtime cannot record anything, so the
    acting process is the only party that can. Two fields make the act nameable
    (``incidents.INVOLUNTARY_MECHANISM_LABELS`` renders the token): ``mechanism``
    is the machine token for WHAT was done, and ``actor`` the acting component's
    own name.

    ``deliberate: False`` IS WHAT KEEPS THIS OUT OF THE USER-STOP CLASS, and the
    run key is spelled from the same three fields a deliberate marker uses, so
    ``attention._stop_marker_covers_run`` needs no change: a marker that does not
    describe the run being classified is refused exactly as before, and one that
    does can never be read as the user's own stop, because the classifier asks
    ``deliberate`` before it renders anything.

    ``command`` defaults to ``actor`` because for this writer the two ARE the same
    fact — the front end's own name for what it is doing — and a reader that found
    ``killer.command`` empty beside a populated ``actor`` would be looking at two
    fields meaning one thing.

    BEST-EFFORT, like every other evidence write: a stop the harness is taking
    must not be abandoned because a sidecar could not be written. Returns whether
    a marker is on disk, for the caller that wants to say so — and the caller
    should say so, because a runtime that could NOT be attested is a gap in the
    artifact rather than an absence of victims.

    LAST WRITER WINS, as it already does between the ladder's own rungs: one marker
    file serves the whole conversation, and this one describes the most recent act
    on it. The run key is what keeps the two losses asymmetric in the right
    direction — an older run's marker can never narrate a newer death (it is refused
    for not covering the run), whereas a newer act's marker displaces an older run's.
    That direction is the one to prefer: the displaced verdict was published into
    ``attention.db`` when its successor re-engaged, while an unattested NEWER death
    is the whole failure this function exists to end.

    ``record`` is anything carrying the run key — an ``update``-side caller has a
    ``SessionRecord`` (live or reaped) or a ``BootRecord`` (a runtime still in its
    first second, which is the window a prune used to leave unprotected).
    """
    session_id = str(getattr(record, "session_id", "") or "")
    if not session_id:
        # No conversation to attest into. Not an error: a ``lop serve`` record is
        # a daemon, not a session, and it has no transcript of its own.
        return False
    payload = _stop_marker_payload(
        record,
        "",
        command=command or actor,
        deliberate=False,
        actor=actor,
        mechanism=mechanism,
    )
    try:
        registry.write_stop_marker(session_dir(root, session_id), payload)
    except OSError:
        return False
    return True


def withdraw_involuntary_stop(record: Any, root: Path, *, mechanism: str) -> bool:
    """Take back OUR involuntary marker when the act it attests does not complete.

    The counterpart of :func:`note_involuntary_stop`, and the same shape
    :func:`_withdraw_staged_stop_marker` already takes for the ladder. A marker is
    written BEFORE an irreversible step so it survives the process the step is done
    to; when the step is then not taken — a prune whose ``_remove_tree`` reports the
    tree still there, an in-place install whose installer exited non-zero — the
    marker is the only artifact left saying otherwise. It is keyed to the live RUN,
    so it covers every later death of that same process, and an unrelated crash an
    hour later would be narrated as this act (review round 1, MINOR 2). The
    command's own report says the act failed; the evidence has to agree with it.

    WHAT IS REMOVED IS OURS — decided by READING the file rather than by
    remembering that we wrote one, exactly as the ladder's withdrawal does: the run
    key must match the record, ``rung`` must be the empty one an involuntary act
    carries (a deliberate rung's marker is never this call's to take), ``deliberate``
    must be ``False``, ``killer.pid`` must be ours, and ``mechanism`` must be the act
    this withdrawal speaks for. So a marker another front end staged for the same
    run survives, and so does a marker for a different mechanism.

    ``root`` is the CONFIG ROOT the record was read under, because that is where the
    conversation directory — and therefore the marker — lives; the run key names the
    session, not the root.

    Best-effort and never raises, like every other evidence write here: a failure to
    clean up must not fail the act's own error report. Returns whether a marker was
    removed, for a caller that wants to say so.
    """
    session_id = str(getattr(record, "session_id", "") or "")
    if not session_id:
        return False
    conversation = session_dir(root, session_id)
    staged = registry.read_stop_marker(conversation)
    if not staged or staged.get("rung") != "" or staged.get("deliberate") is not False:
        return False
    if staged.get("mechanism") != mechanism:
        return False
    # The whole run key, for the reason ``_withdraw_staged_stop_marker`` states: the
    # marker's own three fields are what the classifier compares against a dead
    # record, so they are what decides whether this file is a statement about the
    # run whose act just failed.
    if staged.get("session_id") != session_id or staged.get("pid") != record.pid:
        return False
    if staged.get("started_at") != record.started_at:
        return False
    killer = staged.get("killer")
    if not isinstance(killer, dict) or killer.get("pid") != os.getpid():
        return False
    # ``remove_stop_marker`` is best-effort by its own contract (a withdrawal must
    # not fail over evidence cleanup), so there is no OSError to catch here.
    registry.remove_stop_marker(conversation)
    return True


def _write_stop_marker(record: SessionRecord, root: Path, rung: Method, *, command: str) -> None:
    """Stage the durable stop marker BEFORE the step it attests to.

    RETURNS NOTHING, because a return value here would be a second, weaker
    answer to a question the FILE already answers (design round 2, NIT-1). What
    the withdrawal needs to know is not "did this call write something" but
    "is the file on disk MY OWN rung-1 statement for THIS run", and only a
    re-read can answer that: a marker another ladder staged for the same run
    must survive this ladder's refusal (see
    :func:`_withdraw_staged_stop_marker`). So the failure it used to report —
    ``None`` when the write raised — was never read by either caller.

    Best-effort, and the swallow is the decision rather than an oversight: a
    stop the user asked for must still happen when a sidecar cannot be
    written, so failing to attest never aborts the ladder. The cost is bounded
    and stated — the receipt is unaffected, and that one death falls back to
    the dead-record rung, which is where it stood before this existed.

    THE GUARANTEE IS PROCESS-DURABILITY, NOT HOST-DURABILITY, and the ordering
    claim is worth stating at that strength: the file is in the page cache and
    visible to every reader before the signal, which is exactly the semantics
    this needs (the TARGET process dies; the HOST does not), and it is the same
    shape ``registry.publish`` already uses for the records themselves. A power
    loss between the write and the rename is not covered — added fsync would
    buy that for every heartbeat of every live session, which is a cost this
    evidence does not justify. A killer killed mid-write leaves a
    ``.runtime-stop.json.*.tmp`` behind: bounded, tiny, and never read as a
    marker (the reader names the file, not the pattern).

    Sits next to the rung it describes rather than in a wrapper, because the
    ordering IS the invariant: the file must be visible before the signal, and
    a caller that ever moves one of these calls below its rung has broken the
    thing the file is for.
    """
    payload = _stop_marker_payload(record, rung, command=command)
    try:
        registry.write_stop_marker(session_dir(root, record.session_id), payload)
    except OSError:
        # The one failure this can have, swallowed on purpose: see the
        # docstring — a missing attestation must never abort a stop the user
        # asked for.
        pass


def _withdraw_staged_stop_marker(record: SessionRecord, root: Path) -> None:
    """Take back OUR rung-1 marker when the ladder refuses to go further.

    Rung 1 stages its marker on the target's ACK — the ack is what sets the
    target's exit in motion — and the ladder can still REFUSE before any signal:
    a socket that answers naming a different session id is a live stranger, and
    a start-time proof can fail. The target is then ALIVE, with a durable marker
    on disk that is keyed to its very run, so its later and quite involuntary
    death (a crash, OOM, another kill wave) would classify as the user's own
    stop. That is the wrong-verdict class this evidence exists to remove, in the
    worst direction: it HIDES a crash rather than inventing a stop.

    WHAT IS REMOVED IS OURS, decided by READING the file rather than by
    remembering that we wrote one: same run key, ``rung == "socket"``, and
    ``killer.pid`` == ours. A marker another ladder staged for the same run (a
    concurrent ``lop stop`` from another front end) therefore survives, and so
    does a later rung's marker — the refusal branch cannot see either for this
    call, but the check is what makes that true rather than the call order.
    Best-effort like every other evidence write: a refusal must not fail over
    cleanup.
    """
    conversation = session_dir(root, record.session_id)
    staged = registry.read_stop_marker(conversation)
    if not staged or staged.get("rung") != "socket":
        return
    if staged.get("session_id") != record.session_id or staged.get("pid") != record.pid:
        return
    # The run KEY, all of it: the marker's own three fields are what the
    # classifier compares against a dead record, so they are what decides
    # whether this file is a statement about the run we are refusing.
    if staged.get("started_at") != record.started_at:
        return
    killer = staged.get("killer")
    if not isinstance(killer, dict) or killer.get("pid") != os.getpid():
        return
    registry.remove_stop_marker(conversation)


async def _graceful_stop(
    record: SessionRecord, timeout_s: float, root: Path, *, command: str
) -> bool:
    """Rung 1: ask the runtime to stop itself, wait out the clean exit.

    The op the runtime serves (``RuntimeServer._dispatch``'s ``stop`` case)
    runs deny-pending-gates → dispose → unpublish → exit; the ack comes back
    when the decision is made, the exit lands moments later. A TUI owner
    runs the same op but its PROCESS stays (the session ends beneath the
    terminal), so the wait accepts an unpublished record as the landing. An ``error``
    reply (an old runtime that predates the op) is a scheduled miss, not a
    failure — the ladder proceeds to identity confirmation and SIGTERM, which
    every runtime already handles, so mixed-version machines never wedge.

    THE MARKER GOES AFTER THE ACK AND BEFORE THE WAIT. This rung's
    irreversible step is the target's clean EXIT, which the ack sets in
    motion — not the request, which the target is free to refuse and a silent
    socket never even received. Staging it before the exchange would leave a
    "deliberate stop" marker behind for a rung that never acted, which is the
    one misreading the marker must not create: an unacked socket request to a
    process that had already crashed would then read as the user's own stop.

    AND A RUNG THAT STAGES BUT DOES NOT LAND DOES NOT KEEP ITS MARKER WHEN THE
    LADDER REFUSES: this rung can ack against a process the identity gate then
    refuses to signal (a socket answering another session id, a failed
    start-time proof), which leaves the target alive. `stop_session` withdraws
    our own marker on that path — see :func:`_withdraw_staged_stop_marker`.
    """
    reply = await _exchange(record, {"op": "stop"}, reply_timeout_s=timeout_s)
    if reply is None or reply.get("op") != "ack":
        return False
    _write_stop_marker(record, root, "socket", command=command)
    return await _await_stopped(record, timeout_s, root)


async def _signal_and_confirm(record: SessionRecord, sig: int | None, grace_s: float) -> bool:
    """Rungs 2–3: signal the confirmed pid and wait out its grace window.

    Called only AFTER identity confirmation — this is the rung that can hit
    a process, which is exactly why nothing reaches it unconfirmed. SIGTERM
    rides the runtime's existing handler, which is NO LONGER the same shape as
    the socket op: the socket op is a deliberate stop and cuts a live turn,
    while the handler defers its own disposal to the end of any turn in flight
    and is bounded by ``types.SIGNAL_DRAIN_S`` (see ``process._drain_for_signal``).
    That is why ``SIGTERM_GRACE_S`` is derived from that bound rather than
    chosen: a grace shorter than the drain would escalate to the next line
    mid-drain. SIGKILL has no handler by definition — state is orphaned and
    recovered by the stale-record reap plus the lease's dead-owner recovery,
    which is exactly what those mechanisms exist for.
    """
    if sig is None:
        # NO SIGNAL TO SEND. ``signal.SIGKILL`` is documented "Availability:
        # Unix", so naming it on the rung reached exactly when a runtime is
        # wedged raised ``AttributeError`` before anything was killed — the
        # ladder's last resort failing on the platform that needs one most.
        # Windows offers no signal here at all (a runtime is spawned detached,
        # so it has no console to receive ``CTRL_BREAK_EVENT``):
        # ``TerminateProcess`` is the stop the kernel has, and
        # :func:`procstate.terminate_process_tree` walks the tree with
        # ``taskkill /T /F``. The rung is unchanged by intent — the marker this
        # rung writes still names it — only the mechanism differs per platform,
        # which is what :func:`procstate.hard_kill_signal` reports.
        delivered = await asyncio.to_thread(
            procstate.terminate_process_tree, record.pid, force=True
        )
        if not delivered:
            return not registry.pid_alive(record.pid)
        return await _await_pid_exit(record.pid, grace_s)
    try:
        os.kill(record.pid, sig)
    except OSError:
        # Covers ProcessLookupError, PermissionError, AND Windows' OSError for
        # a pid that is already gone (``os.kill`` there reports a WinError, not
        # ``ESRCH``); all three mean "ask the record instead of assuming".
        return not registry.pid_alive(record.pid)
    return await _await_pid_exit(record.pid, grace_s)


def _recover_record(record: SessionRecord, root: Path) -> None:
    """Best-effort stale-record cleanup after a confirmed exit.

    A clean stop unpublishes its own record; the SIGKILL rung cannot (the
    process is gone). ``registry.scan`` reaps dead-pid records on any
    reader's next pass, so this is not load-bearing — it is the polite
    version that makes `lop sessions` correct immediately instead of at the
    next scan, and it must never raise over a file that is already gone.

    ``root`` for the same reason :func:`_record_retired` takes it: the
    record to clean up is the one in the CALLER's root. Unpublishing the
    ambient root's ``<pid>.json`` while the caller targeted another one left
    the real record behind (seen as a stale row until a scan reaped it) and,
    worse, could delete an unrelated record that happened to share the pid.
    """
    if registry.pid_alive(record.pid):
        return
    registry.unpublish(record.pid, root)


async def stop_session(
    record: SessionRecord,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    force: bool = False,
    _root: Path | None = None,
    _command: str = "control.stop_session",
    on_wait: Callable[[str], None] | None = None,
) -> StopOutcome:
    """Stop one live session by its discovery record. Never raises.

    The escalation ladder in order: a skip when the record says the runtime is
    ALREADY LEAVING a signal it received → graceful socket op → a skip when the
    record says a turn is in flight → identity-confirmed SIGTERM →
    identity-confirmed SIGKILL, with a refuse when identity cannot be
    confirmed ahead of a signal. See the module docstring for the rules;
    this function is where they are enforced in order.

    The first skip is the one exception to the order below, and it has to be
    first: a draining runtime answers its socket, so every later rung would
    reach it and stop it promptly — which is the harm. The bottom skip sits
    BETWEEN the socket rung and the identity gate on purpose, and the
    distinction it rests on still holds for a target that is merely busy: a
    cooperative runtime is stopped deliberately and promptly by the socket op
    even mid-turn — a stop the user asked for IS a stop they want — and that
    skip only applies once that rung has failed, i.e. to a target whose own
    socket will not answer.

    ``force`` admits the record-field identity proof when the socket cannot
    answer — the explicit opt-in for a heartbeating-but-starved process the
    refusal rule would otherwise hold forever (see ``_identity_by_record``) —
    and, for the same reason, it is what signals a target that reports a turn
    in flight, or one that is already leaving, instead of skipping it.

    ``_root`` is the config root (tests inject one); production callers use
    the ambient ``config_dir()``.

    ``on_wait`` is told, once, before rung 2's wait — the only silence in this
    ladder long enough to be mistaken for a hang (``SIGTERM_GRACE_S``, ~150 s
    for a wedged target). A callback rather than a print because the module is
    shared by the CLI and the TUI, which paint in different places, and
    optional because a caller that already shows progress (the TUI's own
    "waiting for it to answer" block) may not want a second line. Nothing about
    the wait itself depends on it.

    THERE IS NO GRACE KNOB HERE, deliberately (NIT, PR #1141). One used to
    exist — ``_sigterm_grace_s`` — for the unit cells that keep a target alive
    to force the escalation, because the production grace is minutes long by
    construction and turned two of them into 153-second tests. It was removed
    for the reason its own docstring gave: it guarded the one constant whose
    whole purpose is that it cannot be shortened, so a production caller passing
    it could have landed SIGKILL inside the receiver's drain. The cells now
    monkeypatch ``SIGTERM_GRACE_S`` itself, as the drain cells already do for
    ``SIGNAL_DRAIN_S``.

    ``_command`` is the front end the stop came from, carried verbatim into
    every rung's stop marker so the artifact can name its author — the tokens
    are the user's own entries: ``lop stop`` (the CLI's single stop),
    ``lop stop --all`` (its sweep), ``/stop`` and ``/stop --all`` (the TUI's).
    It is the caller's to set because only the caller knows which request the
    user actually made — and the marker's whole value is that the answer
    survives the process that knew it. The default names this function only so
    an in-process caller (a test, a future supervisor) is honest about being
    one rather than borrowing a front end's name.
    """
    root = _root if _root is not None else config_dir()
    name = record.conversation_name or record.session_id

    # FIRST, AND AHEAD OF RUNG 1 DELIBERATELY: a runtime that has ALREADY been
    # signalled and is finishing the turn in flight is the one target whose own
    # graceful stop is destructive.
    #
    # WHY THIS ISN'T SATISFIED BY THE BUSY SKIP BELOW. Everything else in this
    # ladder exists to protect work from signals; the socket rung protects it by
    # being the user's own deliberate stop, which is prompt BY CONSTRUCTION and
    # stays that way (rung 1 is unchanged, and the skip below still sits after
    # it). But a target inside the drain is finishing work a signal asked it to
    # finish: there, "deliberate and prompt" is exactly what cuts the turn the
    # drain was saving, and the old ordering could not say so — the busy skip is
    # unreachable for a cooperative runtime by design. So the one state that has
    # to be published is the one state that has to skip EARLY (U1/U2, PR #1141).
    #
    # WHAT THIS IS NOT: a deferral. Nothing here waits, retries or re-orders
    # anything — the refusal is immediate and synchronous, the exit is already
    # scheduled by the runtime itself, and ``--force`` still stops the target
    # promptly and deliberately, classified `user-stop` as always. The operator
    # pays one keystroke to convert a lost turn into a finished one, and nothing
    # is made slower (a MAJOR if that were to change).
    #
    # LIVENESS-GATED, because the honest report for a target that has already
    # gone is the ladder's own "already exited" — a clean resolution, exit 0 —
    # and not a refusal about a drain that is over. A record can outlive its
    # process by a moment; that case falls through and rung 1 reports it.
    # Read defensively, like every probe on this path: the ladder's contract is
    # that it never raises, and a record-shaped double handed in by a caller (or
    # one written by a runtime that predates the field) may simply lack it.
    leaving = getattr(record, "leaving", "") or ""
    # ...AND GATED ON PROGRESS, because the skip's whole premise is that the turn
    # boundary the latch waits on will ARRIVE. For a runtime that has stopped
    # reporting it will not: on 2026-09-24 two runtimes carried "leaving for the build
    # on disk when its turn ends" with heartbeats 5.6 h and 5.9 h stale, `lop sessions`
    # listed both ``wedged``, and a plain `lop stop` answered "it leaves by itself,
    # nothing to do" — a promise no evidence supported. ``stalled`` is the reason the
    # drain can no longer be trusted to finish, and a stalled drain falls through to
    # the ordinary ladder instead of being skipped.
    stalled = _drain_stalled(record) if leaving and not force else ""
    if stalled and on_wait is not None:
        # Said BEFORE the ladder runs, because the ladder that follows can take the
        # whole signal grace, and the reader has just been told by every other
        # surface that this runtime is leaving by itself: the line is why this
        # stop did not believe that.
        #
        # IT NAMES THE DECISION, NEVER THE RESULT (QA round 1, Q-2). The rungs below can
        # still REFUSE — a recycled pid, a socket that is silent while the beat is fresh
        # — and a line promising "stopping it" followed by "refused …" read as a
        # contradiction. It is said here rather than after the identity gate because
        # rung 1 can end the runtime before that gate runs, and that stop needs its
        # reason too.
        on_wait(
            f'"{name}" (pid {record.pid}) {_drain_phrase(record)}, but {stalled}, so it is '
            "not left to drain; trying the ordinary stop"
        )
    if leaving and not force and not stalled and registry.pid_alive(record.pid):
        # THE REMEDY IS TWO-SIDED, and the line is read by whichever front end
        # asked: ``--force`` is a flag of ``lop stop`` and the TUI's ``/stop``
        # takes no flags at all (U7). Named in the reader's own vocabulary —
        # see ``_force_remedy`` — because a surface that offers an action it
        # cannot accept is the defect, not the wording.
        #
        # AND THE FIRST REMEDY IS NOTHING. The exit is already scheduled by the
        # runtime itself, so "skipped … (--force …)" alone reads as "this did
        # not work — retry if you meant it", when the answer for almost every
        # operator is to let it finish (UX round 2, NIT-2).
        return StopOutcome(
            pid=record.pid,
            session_id=record.session_id,
            name=name,
            method="draining",
            line=(
                f'skipped "{name}" (pid {record.pid}) — it {_drain_phrase(record)}; stopping '
                "it now cuts the turn it is finishing — it leaves by itself, nothing "
                f"to do ({_force_remedy(record.pid, _from_a_shell(_command))})"
            ),
        )

    # Rung 1 — the graceful op. Both its failure shapes are scheduled misses:
    # an unreachable socket means already-gone-or-crashed, an error reply
    # means an older runtime. Either way the ladder continues.
    if await _graceful_stop(record, timeout_s, root, command=_command):
        wakes = await _park_wakes(record, root)
        _recover_record(record, root)
        method: Method = "socket"
        return StopOutcome(
            pid=record.pid,
            session_id=record.session_id,
            name=name,
            method=method,
            line=_stopped_line(record, method, wakes),
            wakes_dormant=wakes,
        )

    # The pid is gone but the graceful op never acked: it died under us
    # (crash, or an old runtime that exited on its own). Nothing to signal;
    # reap the record, park the wakes, report it as already gone — a clean
    # resolution, not a refusal, so `--all` over a dead record exits 0.
    #
    # No stop marker here, deliberately: this branch is the one shape where
    # the ladder proved NO rung acted (no ack, and nothing left to signal), so
    # attesting a deliberate stop would be a fiction. Note this is also why
    # the marker is staged AFTER the graceful ack and not before the request:
    # a request the target never received is not a stop.
    if not registry.pid_alive(record.pid):
        wakes = await _park_wakes(record, root)
        _recover_record(record, root)
        method = "gone"
        return StopOutcome(
            pid=record.pid,
            session_id=record.session_id,
            name=name,
            method=method,
            line=f'"{name}" already exited',
            wakes_dormant=wakes,
        )

    # A SIGNAL MUST NOT CUT WORK IN FLIGHT — and this ladder is one of the
    # senders that owes that. With the receiver's drain in place a SIGTERM is no
    # longer fatal to a mid-turn runtime, so the real cost of signalling here is
    # not data loss but a wait: the receiver drains to the end of its turn, while
    # this caller sits in ``_await_pid_exit`` for up to ``SIGTERM_GRACE_S``. The
    # record already publishes the one fact needed to avoid both, so the ladder
    # asks it FIRST and skips a target that reports a turn in flight.
    #
    # WHY THE RECORD'S BIT AND NOT THE SOCKET'S: a cooperative runtime never
    # reaches this branch — rung 1 above stopped it, deliberately and promptly —
    # so by construction this is the case where the socket did NOT answer the
    # request, and for a silent target the record is the only evidence available.
    #
    # IT IS ALSO A NARROWER PREDICATE THAN THE RECEIVER'S, and that is a chosen
    # trade rather than an oversight (M1, PR #1141). The record publishes
    # ``is_conversationally_active()`` — the spinner bit, whose own docstring
    # names it the authority for the picker and explicitly NOT ``is_busy()`` —
    # while the receiver drains on ``is_busy()``, which additionally counts live
    # subagents, background jobs, MCP grant/reload tasks and retained background
    # tasks. So a socket-silent runtime holding only, say, a detached ``bash``
    # job publishes ``busy=False``, is signalled here, and the ladder then waits
    # out the receiver's drain before rung 2 resolves: bounded extra LATENCY on
    # the socket-silent tail, never a lost turn, because the receiver's drain is
    # exactly the mechanism that makes a stale-false safe in either direction.
    #
    # Why not widen this to match: the field's primary reader is the picker and
    # the sessions listing, where the narrow meaning is the load-bearing one
    # ("this conversation is mid-turn") — widening ``SessionRecord.busy`` would
    # put a spinner on every session holding a background job. A second,
    # ladder-only bit on the record is the alternative, and it buys a shorter
    # wait on a shape whose stop already works; it is not worth a field whose
    # staleness could then refuse a stop the operator asked for. Stated here so
    # the difference reads as intended.
    #
    # It is derived state, stale by up to one heartbeat (15 s), and both
    # directions of that staleness are already covered: stale-true skips a
    # target that has since gone idle (reported, and one command away),
    # stale-false signals one that is in fact busy and is then drained by the
    # receiver under the invariant on ``SIGTERM_GRACE_S``. Neither can cut a
    # turn.
    #
    # ``--force`` ESCALATES PAST IT, and that is deliberate: the flag already
    # means "use the weaker identity proof and signal this process I cannot reach"
    # — the operator explicitly asking for signals against a runtime that will
    # not answer. Someone who types it has accepted that the turn goes too. A
    # plain stop refuses rather than surprising them, and the refusal names both
    # ways forward.
    # A STALLED DRAIN IS NOT SKIPPED HERE EITHER. The busy bit on such a record is the
    # turn the drain is waiting on, published by a runtime that has since stopped
    # reporting, so it is exactly as stale as the drain above — and the target has
    # already committed to leaving, so a signal asks it to do what it decided to do.
    if record.busy and not force and not stalled:
        method = "busy"
        return StopOutcome(
            pid=record.pid,
            session_id=record.session_id,
            name=name,
            method=method,
            line=(
                f'skipped "{name}" (pid {record.pid}) — a turn is in flight; '
                "stop it again once the turn ends, or --force to signal it now"
            ),
        )

    # Identity gate before ANY signal — the pid-reuse rule. Confirmed here,
    # after the graceful rung (which signals nobody), so a cooperative
    # runtime is never asked to prove itself and a wedged one cannot
    # fast-path to a signal it did not earn.
    confirmed, why_not = await _confirmed_session_id(record)
    # Whether the SOCKET was silent, remembered before the rungs below
    # overwrite ``why_not`` with their own reasons. Both fallback proofs are
    # gated on this and not on "unconfirmed": a socket that ANSWERED naming a
    # different session is a live stranger, and no weaker evidence may
    # override it (round-3 BLOCKER-2).
    socket_silent = not confirmed and why_not == _SOCKET_SILENT
    if socket_silent:
        # Alive but silent: the wedged case. The socket cannot vouch for
        # it, so fall back to the start-time proof — still a proof, still
        # refusing when it cannot be made. A socket that ANSWERED with a
        # different session id never reaches here: that is a live stranger
        # and stays refused.
        confirmed, why_not = await _identity_by_start_time(record)
    forced = False
    if not confirmed and force and socket_silent:
        # --force reads identity from the record's own fields for the one
        # case the socket cannot answer: a process so starved it never
        # services its loop.
        #
        # Gated on SILENCE, exactly like the start-time rung above, and for
        # the same reason. A socket that ANSWERED "I serve session X, not Y"
        # is a live stranger telling you to your face that it is not your
        # target — the strongest negative the ladder can obtain — and record
        # fields are strictly weaker evidence that the very record in
        # question trivially satisfies. This needs no attacker to go wrong:
        # ``session_id`` reaches the record only on the 15 s heartbeat, so
        # after /resume there is a window where the record still names the
        # PREVIOUS session while the process serves the new one, and a user
        # who re-ran with --force as the refusal invites would kill the
        # conversation that replaced it (round-3 BLOCKER-2).
        #
        # Off the loop: this forks ``lsof`` and ``ps``, and the TUI runs the
        # ladder on its event loop — the constraint the probes above carry.
        confirmed, why_not = await asyncio.to_thread(_identity_by_record, record)
        if confirmed:
            why_not = ""
            forced = True
    if not confirmed:
        # A refusal signs nothing — and must leave nothing signed either. Rung 1
        # may already have staged its socket marker before this gate refused
        # (an acked stop against a process that then answered a different
        # session id, or a failed start-time proof), so the target is ALIVE and
        # holding evidence keyed to its own run; left there it would publish
        # that target's next, involuntary death as the user's own stop.
        _withdraw_staged_stop_marker(record, root)
        method = "refused"
        return StopOutcome(
            pid=record.pid,
            session_id=record.session_id,
            name=name,
            method=method,
            line=f'refused "{name}" (pid {record.pid}) — {why_not}',
        )

    # Rung 2 — SIGTERM: the runtime's existing handler runs the same clean
    # exit the socket op would have. Marker first, then the signal: this rung
    # cannot be attested by its target any more than rung 3 can — a SIGTERM
    # that the process never gets to handle (frozen, starved) is
    # indistinguishable afterwards from a crash, UNLESS the sender said so
    # before sending.
    _write_stop_marker(record, root, "sigterm", command=_command)
    # Rung 2 is a REQUEST, so its wait must outlast the receiver's own drain
    # (``SIGTERM_GRACE_S``). Rung 3's budget is deliberately not this one: see
    # ``SIGKILL_CONFIRM_S``.
    #
    # THE WAIT IS ANNOUNCED FIRST. This is the longest silence in the ladder by
    # two orders of magnitude — a wedged, mid-turn target pays the whole derived
    # grace (~150 s) before SIGKILL — and a front end that paints its receipts
    # only at the end leaves the operator unable to tell a working command from a
    # hung one for that entire time; the natural response to that is Ctrl-C,
    # which leaves the outcome genuinely ambiguous (U5, PR #1141). The bound is
    # named because it is the thing being waited on: without it the pause reads
    # as a stall at whatever it happened to be printing. ``on_wait`` is a
    # callback rather than a print because this module is shared by the CLI and
    # the TUI, which paint in different places.
    if on_wait is not None:
        on_wait(_wait_line(name, record.pid, from_a_shell=_from_a_shell(_command)))
    if await _signal_and_confirm(
        record,
        signal.SIGTERM,
        SIGTERM_GRACE_S,
    ):
        wakes = await _park_wakes(record, root)
        _recover_record(record, root)
        method = "sigterm"
        return StopOutcome(
            pid=record.pid,
            session_id=record.session_id,
            name=name,
            method=method,
            line=_stopped_line(record, method, wakes, forced=forced),
            wakes_dormant=wakes,
        )

    # Rung 3 — SIGKILL. State is orphaned by design; stale-record reaping
    # and the lease's dead-owner recovery pick it up. Report the rung used.
    #
    # THE MARKER IS THE POINT OF THIS RUNG. Everything the runtime could have
    # said about its own death is gone the moment this signal lands — the
    # frozen runtime in the 2026-09-13 repro logged NOTHING at all, and the
    # operator was left reading "disappeared without exiting cleanly" for a
    # stop the user had asked for. The write is immediately before the signal
    # and names sigkill, so the next reader learns which rung killed it, that
    # it was deliberate, and who did it.
    _write_stop_marker(record, root, "sigkill", command=_command)
    # ``hard_kill_signal()`` is SIGKILL on POSIX — the identical value this rung
    # has always sent there — and None on Windows, where it selects the
    # terminate-the-tree path above instead of raising AttributeError.
    await _signal_and_confirm(record, procstate.hard_kill_signal(), SIGKILL_CONFIRM_S)
    wakes = await _park_wakes(record, root)
    _recover_record(record, root)
    method = "sigkill"
    return StopOutcome(
        pid=record.pid,
        session_id=record.session_id,
        name=name,
        method=method,
        line=_stopped_line(record, method, wakes, forced=forced),
        wakes_dormant=wakes,
    )


def _drain_stalled(record: SessionRecord) -> str:
    """Why a LEAVING runtime cannot be trusted to reach its turn boundary, or ``""``.

    TWO FACTS, EACH OWNED ELSEWHERE, and no threshold invented here:

    * ``registry.classify`` says ``wedged`` AND the beat is older than the stall
      watchdog's own steady bound (``stall_watchdog.bound_seconds()``, 300 s unless
      configured). ``wedged`` alone is the listing's 45 s "not answering", and that is
      NOT enough to cut a drain (agent review round 1, M2): ``classify``'s own docstring
      records beats 105.8 s and 205.8 s late on runtimes whose CPU was advancing, and
      the stall bound is the number this codebase sized ABOVE those measurements to
      mean "stopped, not slow". A drain in the 45 s-to-bound band is still skipped,
      exactly as before; the incident runtimes were 5.6 h and 5.9 h stale.
    * ``stall_watchdog.held_now`` — its own bound fired with work in flight and it has
      not re-armed since. The same predicate behind the listing's ``bound held`` cell,
      and the arm that covers a parked workload loop behind a fresh serving beat.

    THE BOUND IS READ IN THIS PROCESS, which is a stated limit rather than a proof: the
    knob is an environment variable and the runtime may have been started with another
    value. A switched-off or unreadable knob here falls back to ``DEFAULT_STALL_S``
    rather than to the 45 s arm, so the uncertain direction is the one that skips.

    DEGRADED EVIDENCE, NOT A DIAGNOSIS (``classify``'s own caveat). That is why this
    does not SIGNAL anything by itself — it only withdraws the promise that the runtime
    leaves on its own, and hands the target to the ordinary ladder, whose socket rung
    stops a runtime that does answer and whose signal rungs still require confirmed
    identity.

    NEVER RAISES, like every probe on this path: a record-shaped double without the
    fields, or an unreadable dump, answers ``""`` — the pre-existing skip.
    """
    try:
        from local_operator.session.runtime import stall_watchdog

        verdict = registry.classify(record)
        bound = stall_watchdog.bound_seconds() or stall_watchdog.DEFAULT_STALL_S
        if verdict.state == "wedged" and verdict.heartbeat_age_s >= bound:
            # One unit ladder with `lop sessions`' HB_AGE column (45s/12m/3h/2d), so the
            # number in this line is the number beside the row the operator just read.
            from local_operator.wakes.display import format_age

            return f"it has not reported for {format_age(verdict.heartbeat_age_s)}"
        if stall_watchdog.held_now(record.pid, record.started_at):
            return "its stall bound fired with work in flight and it has not re-armed since"
    except Exception:  # noqa: BLE001 — the ladder never raises over a probe
        return ""
    return ""


def _stop_targets(root: Path, own_pid: int | None = None) -> list[SessionRecord]:
    """Every OTHER agent on THIS machine, in scan order.

    ``live`` AND ``wedged`` are both targets — a wedged owner is exactly the
    agent the user most needs to be able to stop, and its socket will not
    answer, which is what the signal rungs are for. NOT merely detached
    sessions: a TUI-owned session in another terminal gets the same graceful
    op, and its process survives with the session ended beneath it (the
    TUI's ``request_stop`` hook owns that in-process case).

    The caller's OWN record is never a target here. A process cannot walk
    itself down the ladder: its graceful op is a socket call to itself, and
    the signal rungs would terminate the very front end painting the report
    (seen live: the TUI SIGTERMed itself). An in-process caller ends its own
    session through its own in-process path, after this list — the TUI's
    ``_stop_all_worker`` does exactly that, last, so the receipt is the last
    line the user reads. A CLI caller has no own session (``own_pid=None``).

    Other users' sessions are excluded twice over: the OS already makes their
    records unreadable (0600 under 0700), and the same-uid check in
    :func:`stop_all` refuses to act on any record that slips through a
    downgraded mode.
    """
    return [
        rec
        for rec, state in registry.scan(root)
        if state in ("live", "wedged") and (own_pid is None or rec.pid != own_pid)
    ]


async def stop_all(
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    own_pid: int | None = None,
    only_pids: "frozenset[int] | set[int] | None" = None,
    force: bool = False,
    _root: Path | None = None,
    _command: str = "control.stop_all",
    on_wait: Callable[[str], None] | None = None,
) -> list[StopOutcome]:
    """Stop every OTHER agent on this machine. Never raises.

    ``own_pid`` is excluded outright (see :func:`_stop_targets`); the caller
    ends its own session in-process afterwards. ``only_pids`` restricts the
    run to a set the user was SHOWN — the TUI's arm listing is the
    confirmation, so a session that appeared between arm and repeat must
    not be stopped on the strength of a listing it was never on.

    ``_command`` is the front end the sweep came from, forwarded to every
    target's marker (see :func:`stop_session`): ``lop stop --all`` from the CLI,
    ``/stop --all`` from the TUI's kill switch. A sweep's marker has to name
    the sweep, not a single stop, or the artifact cannot tell the operator
    whether they pressed a key on one session or the whole machine.

    THE DEFAULT IS THE IN-PROCESS NAME, matching :func:`stop_session`'s, and the
    match is the point (review round 2, NIT-3). Both parameters exist for a
    caller with no user behind it, and a default that named a front end made
    this one's silent fallback a CLAIM about a keystroke nobody pressed —
    ``lop stop --all`` — while its twin's said ``control.stop_session``. Two
    defaults in two registers for one concept is a drift waiting to be read as
    evidence, so both now say which function ran; every front end passes its
    own token explicitly, and ``tests/unit/test_cli_stop.py`` pins that the CLI
    does.

    Sequential, not concurrent: the graceful rung waits up to ``timeout_s``
    per uncooperative session, and a fan-out would hold every target's wait
    open at once — the opposite of what a user pressing a kill switch wants
    when one wedged runtime sits in front of twelve healthy ones. One at a
    time, healthiest first (scan order), so the common case is fast and the
    wedged tail is paid only by whoever actually needs the signals.

    ``on_wait`` is forwarded to every target unchanged (see
    :func:`stop_session`), because the sweep has the same silence problem the
    single stop does — worse, since one wedged runtime sits in front of the
    rest.
    """
    root = _root if _root is not None else config_dir()
    outcomes: list[StopOutcome] = []
    for record in _stop_targets(root, own_pid=own_pid):
        if only_pids is not None and record.pid not in only_pids:
            continue
        if not _same_uid(record):
            outcomes.append(
                StopOutcome(
                    pid=record.pid,
                    session_id=record.session_id,
                    name=record.conversation_name or record.session_id,
                    method="refused",
                    line=(
                        f'refused "{record.conversation_name or record.session_id}" '
                        f"(pid {record.pid}) — not owned by this account"
                    ),
                )
            )
            continue
        outcomes.append(
            await stop_session(
                record,
                timeout_s=timeout_s,
                force=force,
                _root=root,
                _command=_command,
                on_wait=on_wait,
            )
        )
    return outcomes


#: Outcomes that count as "the session is no longer running", i.e. the stop
#: did its job. Everything else (``refused``, ``busy``, ``draining``) is the
#: partial case — and all three are partial for the same reason: the target is
#: still running. Read by the front ends' exit codes and by :func:`summarize`,
#: so `lop stop --all` reports the same partial-vs-clean verdict whichever rung
#: declined.
ENDED_METHODS = frozenset({"socket", "sigterm", "sigkill", "gone"})

#: Outcomes where the target is alive and was deliberately left alone, so a
#: front end has to NAME it: the grouped count can say how many, never which
#: one, and "which agent did not stop" is the only thing the user can act on.
#: Two members because the reasons differ and the copy does — a turn in flight
#: ends and the runtime stays; a drain is already ending it — and both mean the
#: same thing to the caller's arithmetic (not stopped, not broken). The CLI
#: needed no set for this: it paints every outcome's own line. The TUI does,
#: which is what this exists for (M3, PR #1141).
LEFT_ALONE_METHODS = frozenset({"busy", "draining"})


def summarize(outcomes: list[StopOutcome], *, own: StopOutcome | None = None) -> str:
    """The grouped report every front end paints after ``stop all``.

    Reconciles with the promise the listing made: leads with the total the
    user was told would be stopped, then the rung grouping — the honest
    summary of an escalation is how many stopped cleanly, how many needed a
    signal, how many were already gone, how many were left alone because a
    turn was in flight (or because they were already leaving), and how many
    were refused, not twelve identical lines.
    ``own`` is the caller's in-process outcome (the TUI's own session), folded
    into the total and the ``stopped`` count so the numbers add up on one line
    instead of across three.

    EVERY METHOD THIS LADDER CAN RETURN IS LISTED, and that is a property worth
    keeping: a group missing from ``order`` is a stopped target the total does
    not account for (a ``3 sessions:`` line whose parts sum to 2).
    """
    everything = list(outcomes) + ([own] if own is not None else [])
    if not everything:
        return "no sessions to stop"
    order: list[tuple[str, str]] = [
        ("socket", "stopped"),
        ("sigterm", "stopped via sigterm"),
        ("sigkill", "killed"),
        ("gone", "already exited"),
        ("busy", "left alone (a turn is in flight)"),
        ("draining", "left alone (it is already leaving)"),
        ("refused", "refused"),
    ]
    parts: list[str] = []
    for method, label in order:
        count = sum(1 for o in everything if o.method == method)
        if count:
            parts.append(f"{count} {label}")
    total = len(everything)
    return f"{total} session{'s' if total != 1 else ''}: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Rotation: ask a runtime to move to the build on disk, instead of killing it
# ---------------------------------------------------------------------------

#: The op a runtime serves for "retire iff you are idle and the build on disk has
#: moved" (``RuntimeServer._refresh_if_idle``). Deliberately the SAME op a viewer
#: already sends (``attach_client.refresh_if_idle``, the belt for the seconds
#: after ``lop-update``): one implementation of "move to the new build" means the
#: rotation command and a viewer's own recovery cannot disagree about what is
#: allowed to be retired.
REFRESH_OP = "refresh_if_idle"

#: How long to wait for that op's ack. It is a DECISION, not a turn: the runtime
#: answers with what it decided before it disposes itself, so the budget is a
#: round trip on a loopback socket rather than anything a session could be busy
#: doing. ``lop refresh --timeout`` trades patience for promptness per call.
DEFAULT_REFRESH_TIMEOUT_S = 10.0


@dataclass
class RefreshOutcome:
    """What one runtime answered when it was asked to move to the build on disk.

    ``method`` is the resolution, and the front ends read it rather than the
    prose: ``moved`` (retiring now), ``busy`` (a turn is in flight; it moves
    when that turn ends), ``draining`` (a signal already has it leaving at its
    next boundary), ``current`` (already on the build on disk), ``unsettled``
    (the install on disk changed too recently for any runtime to have judged
    it), ``kept`` (the runtime's own refusal, quoted in the line),
    ``unsupported`` (it predates the op), ``unreachable`` (its control socket
    did not answer).
    """

    pid: int
    session_id: str
    name: str
    method: str
    line: str


#: The resolutions that mean "this session is on the build on disk now, or will
#: be as soon as its own work lets it". This is the ROTATION counterpart of
#: ``ENDED_METHODS``: everything here needs nothing more from the caller, and
#: the one shape that does (``unreachable``) is the partial case a front end
#: reports as non-zero.
#:
#: ``busy``, ``draining`` and ``kept`` are settled rather than partial on
#: purpose, and it is the whole point of the command: a busy runtime retires BY
#: ITSELF when its turn ends (the reaper's ``_should_refresh`` branch asks
#: ``may_refresh`` every ``BUILD_CHECK_S``), so "still busy" is a queued move,
#: not a failure. This is why no drain bound is needed here — nothing is being
#: killed, so a long turn can simply be waited out by the process that owns it.
#: ``draining`` is the same fact with the exit already scheduled (a signal got
#: there first), which is why it does not need a second ask either.
#:
#: ``unsettled`` IS DELIBERATELY ABSENT, and that is the fix rather than an
#: oversight (D1/M2, PR #1141). It means "the install on disk moved less than
#: ``BUILD_SETTLE_S`` ago and no runtime has judged it yet" — an honest answer,
#: but not one a script may read as "the fleet is on the new build". Including
#: it here would restore exactly the bug: ``lop refresh``'s first run is
#: ``lop-update``, so it lands INSIDE the settle window for every session on the
#: machine, and a zero exit would report a rotation that has not started.
REFRESH_SETTLED_METHODS = frozenset({"moved", "busy", "draining", "current", "kept", "unsupported"})


def _build_label(record: SessionRecord) -> str:
    """``version@ref[:7]`` — the record's own build, as the TUI already names it.

    A VERSION ALONE CANNOT ANSWER THE QUESTION THESE RECEIPTS ARE READ FOR, and
    the sibling module says why in one line: ``lop-update`` builds from ``main``
    while ``pyproject.toml`` still names the last release, so two genuinely
    different builds share one version string — the same-version rebuild is the
    dominant handover on this host. Labelled by version only, ``running 0.55.0``
    printed on the row of the runtime LEAVING the old build and on the row of the
    one already on the new one, which is precisely the distinction the command
    exists to draw (D2, PR #1141).

    Mirrors ``update.BuildStamp.label()`` — the TUI's build-skew notice prints
    the same form — but is built from a discovery record, a different object:
    the record is what the runtime published about ITSELF, and where it carries
    no ref (a runtime too old to publish one, a PyPI install) the version alone,
    or the ref alone, is the honest answer rather than an invented pair.
    """
    version = record.version or ""
    ref = record.source_ref or ""
    if version and ref:
        return f"{version}@{ref[:7]}"
    return version or ref[:7] or "an unrecorded build"


def _settle_question(record: SessionRecord) -> tuple[str, str]:
    """``("unsettled", "")`` or ``("current", "")`` for a "matches" answer.

    THE DECIDING HALF OF THE ANSWER LIVES IN THE OTHER PROCESS, AND THAT
    PROCESS IS THE OLD CODE. ``Server._refresh_if_idle`` learned to separate
    "the install on disk has moved and has not settled" from "matches" — but a
    runtime keeps the code it booted with until build skew retires it, so the
    fleet alive when ``lop-update`` replaces the install answers the ONE
    hedged sentence its own build knows: ``kept: build on disk matches (or has
    not settled)``. Folding that into ``current`` (exit 0) tells a rotating
    script the fleet is done in exactly the window ``lop refresh``'s own
    docstring says it is for — its first run IS ``lop-update`` — while every
    member of that fleet is about to retire. The design, UX and QA rounds
    reproduced that independently on this head (design D1, UX U6, QA O1).

    SO THE QUESTION IS ASKED HERE, CLI-SIDE, and it can be: the record carries
    the runtime's own boot stamp (``version``/``source_ref``, published since
    long before this change) and the marker read is stdlib-only
    (``buildwatch.moved_and_unsettled``). Two facts meet — the stamp the record
    says it is running, and the marker on disk now — and only their combination
    can distinguish "it is on the build on disk" from "the disk has moved past
    it and nobody has judged it yet". ``build_changed`` deliberately refuses to
    make that distinction (it answers "may I act"), which is why this asks
    ``pending_build``'s question instead of reusing its result.

    Read against the marker THIS process can see, which in production is the
    same install the runtime published — the record's stamp came from that
    install, and both ends resolve the same prefix (``LOP_BUILD_PREFIX`` or
    ``sys.prefix``). A caller run from a DIFFERENT install than the fleet (a
    worktree CLI against the tool-install runtimes, a test harness) is the one
    shape where the two reads can disagree, and there the doubt resolves to
    ``unsettled`` — "ask again", which is the honest instruction when the two
    builds in play are not the same one.

    WHAT IS LEFT AS ``current``: a runtime that answers "matches" while the
    marker has moved AND settled. Its own read of the same file says otherwise,
    so it is either a race of microseconds against a settling marker or a
    disagreement about which install is on disk; either way the runtime answers
    ``retiring`` on the next ask, seconds later, and there is no ladder method
    for "its stamp is not the disk's but it has not committed". Reported as
    ``current`` rather than dressed up as an ``unsettled`` whose sentence ("the
    install changed a moment ago") would be false.

    Never raises: ``moved_and_unsettled`` folds every probe failure into
    ``False``, and this is read while composing a receipt for a person.

    WHAT THIS CANNOT DO, measured rather than inferred (design round 3, D8). A
    runtime whose build DRAINS ON A SIGNAL but PREDATES
    ``SessionRecord.leaving`` — every build before ``4faad653b``, whose
    pre-rebase twin ``efee31e42`` is the head the design round drove its
    ``crossver_drain.py`` cell with — publishes no phrase and answers the
    retained hedge while it is on its way out, so this classifies it from the
    marker and the receipt says "ask again in a few seconds" about a session that
    will be gone by then. It is RECORDED rather than closed because the CLI
    cannot close it: neither the record nor the wire distinguishes "leaving"
    from "has not settled" on those builds — the install is what the settle
    question is about and a signal drain never touches it, while the runtime's
    own work is a fact an ordinary busy session shares. The cost is one stale
    receipt on a transient state (the next ask finds the session gone), and the
    alternative — guessing "leaving" from "busy" — would relabel ordinary busy
    runtimes for every reader.

    That population is exactly the builds that publish no phrase, which is also
    why no phrase-carrying runtime reaches here: ``_refresh_if_idle`` answers
    ``kept: already leaving`` whenever ``_leaving`` is set, and
    ``announce_retiring`` sets it in the same call that writes the record field,
    so the phrase and the answer arrive together (both from ``4faad653b`` on).
    """
    if moved_and_unsettled(record.version or "", record.source_ref or ""):
        return "unsettled", ""
    return "current", ""


def _drain_phrase(record: SessionRecord) -> str:
    """What to say about a drain whose TRIGGER this front end cannot see, as a
    CLAUSE that a prose slot can hang a subject on.

    Two things now commit a runtime to leaving, and only the runtime knows which
    it was: a termination signal, and a build replaced on disk while a turn was
    in flight. A hard-coded "it was signalled" is therefore false for every
    build-driven drain — and the refusal below fires for those too, because the
    record carries the commit whichever trigger made it. So the phrase is
    quoted from the record, where the trigger wrote it, and this one vocabulary
    then serves ``lop sessions``, ``/info``, this ladder and the rotation
    receipt (UX round 2, U8/U9; the reconciliation of PR #1108).

    IT IS A CLAUSE, NOT THE CELL VALUE VERBATIM, and that is the difference
    design round 3 (D7) filed: the record's phrase is written as a CELL
    (lowercase, subject-less) because ``lop sessions`` and ``/info`` print it in
    a column, so '"name" (pid 12, running …) signalled; leaving when its turn
    ends' reads as a fragment — the receipt had no verb. One copula restores the
    grammar in every prose slot without a second vocabulary: the caller supplies
    the subject (``{where}``, ``it``) and this supplies ``is signalled; …``.

    The fallback is for a peer running a build that predates the field: it
    answers ``kept: already leaving`` without publishing a phrase, and the
    clause it gets is the one that was true before the field existed.
    """
    if record.leaving:
        return f"is {record.leaving}"
    return "was signalled and is leaving at its next boundary"


def _refresh_line(record: SessionRecord, running: str, method: str, detail: str) -> str:
    """The one human receipt line for one rotation, per resolution.

    Every branch names the session, its pid and the build it is running
    (``_build_label``), because the question the caller actually has is "which
    of these is still on the old build, and what is it doing instead".
    ``running`` is the record's own build stamp — the runtime's reported build,
    not an assumption about the disk.

    TWO BRANCHES EXIST ONLY TO STOP THIS COMMAND LYING, and both are answers a
    runtime gives that the old code round-tripped into a stronger claim:
    ``draining`` (a signal already has it leaving — not "busy, will move when
    the turn ends") and ``unsettled`` (nobody has judged the install yet — not
    "already current"). Both are named because an operator's next decision
    differs: one waits or forces a stop, the other simply asks again.
    """
    name = record.conversation_name or record.session_id
    where = f'"{name}" (pid {record.pid}, running {running})'
    if method == "moved":
        # The runtime names the build it is leaving FOR (``retiring to <label>``,
        # from its own committed decision — no second read of a marker that may
        # have moved again since). An older runtime answers a bare ``retiring``,
        # and the receipt then says exactly what it said before.
        target = f" ({detail})" if detail else ""
        return f"{where} is retiring now for the build on disk{target}"
    if method == "current":
        return f"{where} already runs the build on disk"
    if method == "busy":
        return f"{where} has a turn in flight — it moves when that turn ends"
    if method == "draining":
        # THE RUNTIME'S OWN WORDS, quoted rather than paraphrased. Two triggers
        # commit a runtime to a drain — a termination signal, and a build
        # replaced on disk while a turn was in flight — and this receipt cannot
        # tell them apart, so a sentence that hard-codes "was signalled" states
        # the wrong one for half its readers. The phrase comes from the trigger,
        # carries its own bound where it has one, and is the same string
        # ``lop sessions`` prints and the TUI's ``/info`` row shows: one
        # vocabulary for one state, written by the one call that also sends the
        # drain to the app (UX round 2, U8/U9).
        #
        # A runtime that committed to a drain without publishing a phrase (a
        # peer running an older build of this same branch) still gets an
        # honest receipt: the answer it gave IS its own sentence.
        if record.leaving:
            return f"{where} {_drain_phrase(record)}"
        return (
            f"{where} is already leaving — the exit is scheduled, not queued "
            f"(up to {bound_text(SIGNAL_DRAIN_S)})"
        )
    if method == "unsettled":
        return (
            f"{where} has not judged the build on disk yet — the install changed a "
            "moment ago and may still be settling; ask again in a few seconds"
        )
    if method == "unsupported":
        return (
            f"{where} cannot be asked to move{detail} "
            "— it retires on its own when it next goes idle"
        )
    if method == "unreachable":
        return f"{where} did not answer its control socket — ask it again once it is responsive"
    return f"{where} was not moved: {detail}"


async def refresh_session(
    record: SessionRecord, *, timeout_s: float = DEFAULT_REFRESH_TIMEOUT_S
) -> RefreshOutcome:
    """Ask one live runtime to move to the build on disk. Never raises.

    The non-destructive counterpart of :func:`stop_session`, and deliberately
    NOT a signal: SIGTERM now means "leave at your next boundary, bounded",
    which is a safe but impatient request, while this asks the runtime to judge
    its own readiness and leaves the timing to it. That difference is the whole
    reason this function exists — the 2026-09-14 sweep needed a build to take
    effect NOW and reached for the only tool it had, and killing in-flight work
    must never be the shortest path to a new build.

    Identity is NOT proven first. The risks are asymmetric in exactly the
    opposite direction from the kill ladder: a stale record's port may now be a
    recycled stranger, but the most a stranger can be asked to do here is retire
    ITSELF, and any runtime that answers this op does so only on its own
    ``may_refresh`` verdict. Spending the ladder's pid-reuse proofs on a request
    that cannot hurt anyone would make the common case (ask twelve healthy
    runtimes to move) pay for the rare one.
    """
    name = record.conversation_name or record.session_id
    running = _build_label(record)
    # ``say`` is the build label the receipt quotes back: the record's stamp is
    # what the runtime published about ITSELF, which is the fact under question.
    reply = await _exchange(record, {"op": REFRESH_OP}, reply_timeout_s=timeout_s)
    if reply is None:
        method = "unreachable"
        detail = ""
    elif reply.get("op") != "ack":
        # An ``error`` reply is how a runtime that predates the op answers (and
        # how any other refusal arrives). Reachable, but not rotatable on
        # request — quote whatever it said and stay honest about the rest.
        method = "unsupported"
        message = str(reply.get("message") or "").strip()
        detail = f" ({message[:120]})" if message else ""
    else:
        answer = str(reply.get("detail") or "")
        # The runtime's own vocabulary, parsed rather than re-derived: ``retiring``
        # is the retirement it just committed to (optionally followed by the build
        # it is leaving FOR), and every other answer is one of its
        # ``kept: <reason>`` refusals (see ``Server._retire_for``).
        #
        # THE NEW ``kept:`` ANSWERS ARE COMPARED AS WHOLE SENTENCES, and both sit
        # ahead of the generic ``kept`` fallback and of the ``matches`` prefix
        # below, because each is a state the caller must not read as its
        # neighbour does: ``already leaving`` is not busy-with-a-queued-move (the
        # exit is already scheduled), and "the install on disk has not settled
        # yet" is not "matches" — collapsing those two into "already current"
        # with a zero exit status IS the D1/M2 defect. Both sentences are
        # module constants so a reword cannot silently stop matching.
        if answer.startswith("retiring"):
            method, detail = "moved", answer.removeprefix("retiring").removeprefix(" to ").strip()
        elif answer == "kept: already leaving":
            method, detail = "draining", ""
        elif answer == KEPT_UNSETTLED:
            method, detail = "unsettled", ""
        elif answer.startswith(KEPT_MATCHES):
            # A PREFIX MATCH, deliberately, and it covers TWO sentences: this
            # head's ``KEPT_MATCHES`` and the RETIRED hedge
            # ``KEPT_MATCHES_OR_UNSETTLED`` that every runtime started before
            # this change still answers. That is a cross-version contract — see
            # the constant — so the retired string must keep routing here
            # rather than being tidied out of the matcher.
            #
            # AND IT DOES NOT MEAN "current" ON ITS OWN. See ``_settle_question``.
            method, detail = _settle_question(record)
        elif answer == "kept: busy":
            method, detail = "busy", ""
        else:
            method, detail = "kept", answer.removeprefix("kept: ") or "no reason given"
    return RefreshOutcome(
        pid=record.pid,
        session_id=record.session_id,
        name=name,
        method=method,
        line=_refresh_line(record, running, method, detail),
    )


async def refresh_all(
    *,
    timeout_s: float = DEFAULT_REFRESH_TIMEOUT_S,
    own_pid: int | None = None,
    only_pids: "frozenset[int] | set[int] | None" = None,
    _root: Path | None = None,
) -> list[RefreshOutcome]:
    """Ask every OTHER live runtime on this machine to move to the build on disk.

    Sequential like :func:`stop_all`, and for the same reason: each dial can
    wait out ``timeout_s`` against a runtime that has stopped answering, and a
    fan-out would hold every one of those waits open at once. The waits here are
    the caller's only cost — nothing is signalled, so a slow target holding up
    the report is a delay, never a loss.
    """
    root = _root if _root is not None else config_dir()
    outcomes: list[RefreshOutcome] = []
    for record in _rotation_targets(root, own_pid=own_pid):
        if only_pids is not None and record.pid not in only_pids:
            continue
        if not _same_uid(record):
            name = record.conversation_name or record.session_id
            outcomes.append(
                RefreshOutcome(
                    pid=record.pid,
                    session_id=record.session_id,
                    name=name,
                    method="unreachable",
                    line=f'"{name}" (pid {record.pid}) is not owned by this account',
                )
            )
            continue
        outcomes.append(await refresh_session(record, timeout_s=timeout_s))
    return outcomes


def _rotation_targets(root: Path, own_pid: int | None = None) -> list[SessionRecord]:
    """Every live runtime this account may ask to move, in scan order.

    The same record set the stop ladder targets (``_stop_targets``): ``live``
    AND ``wedged``, because a wedged runtime is one the caller needs a truthful
    answer about — it will simply report unreachable, which is the answer that
    tells a user something is actually wrong. The caller's own record is
    excluded for the same reason it is there: a process asking itself to retire
    would be answering with the front end's own runtime.
    """
    return _stop_targets(root, own_pid=own_pid)


def summarize_refresh(outcomes: list[RefreshOutcome]) -> str:
    """The grouped one-line report ``lop refresh`` paints after the per-target lines.

    Leads with what moved, because that is what the caller asked for, then what
    will move by itself, then what needs nothing. A caller who sees
    ``1 unreachable`` knows exactly which session to look at; a caller who sees
    only counts of the settled ones knows the rotation is done.

    EVERY LABEL CARRIES ITS OWN SINGULAR AND PLURAL FORM, because a count and
    its noun have to agree — ``1 will move when their turn ends`` was the shipped
    output, a singular count with a plural possessive, and the per-target line
    above it ("it moves when that turn ends") was already singular and correct
    (D3, PR #1141). Only the labels that actually inflect carry two: the rest
    read the same at any count, and pairing them with themselves keeps ONE list
    of every method this function knows about.
    """
    if not outcomes:
        return "no live sessions to refresh"
    # ``(method, singular, plural)``. Every method ``refresh_session`` can return
    # appears here — a method missing from this list is a target the total counts
    # and the parts do not (D1's original question: "is the rotation complete"
    # has to be answerable from the summary alone).
    order: list[tuple[str, str, str]] = [
        ("moved", "retiring now", "retiring now"),
        ("busy", "will move when its turn ends", "will move when their turn ends"),
        ("draining", "already leaving", "already leaving"),
        ("current", "already current", "already current"),
        ("unsettled", "not settled yet — ask again", "not settled yet — ask again"),
        ("kept", "not moved (see above)", "not moved (see above)"),
        ("unsupported", "too old to ask", "too old to ask"),
        ("unreachable", "unreachable", "unreachable"),
    ]
    parts: list[str] = []
    for method, one, many in order:
        count = sum(1 for o in outcomes if o.method == method)
        if count:
            parts.append(f"{count} {one if count == 1 else many}")
    total = len(outcomes)
    return f"{total} session{'s' if total != 1 else ''}: " + ", ".join(parts)
