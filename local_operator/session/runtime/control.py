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
as :func:`local_operator.session.runtime.owned.spawn_owned_session` defers
its heavy imports. ``tests/unit/test_import_graph.py`` is the guard.

**The escalation ladder** (see :data:`SIGTERM_GRACE_S` for each rung's
budget): a graceful ``stop`` control op the runtime serves itself → SIGTERM
(the runtime's existing signal handler runs the same clean exit) → SIGKILL
(state orphaned; the existing stale-record/lease machinery recovers, and the
outcome is reported as ``killed``).

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
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from local_operator.paths import config_dir
from local_operator.session.runtime import registry
from local_operator.session.runtime.types import HEARTBEAT_TIMEOUT_S, SessionRecord

if TYPE_CHECKING:
    from pathlib import Path

#: Which stop method produced an outcome, in escalation order. The runtime's
#: own control socket (``socket``) is the graceful rung; the two signals are
#: the escalation; the remaining two name the non-signalled resolutions.
#: ``gone`` is a process that left the table before the ladder reached it
#: (already exited — nothing to do, not a failure); ``refused`` is the
#: ladder declining to signal because identity could not be confirmed. Kept
#: distinct so a front end can decide "partial" from the method alone
#: rather than by parsing the receipt line.
Method = str  # "socket" | "sigterm" | "sigkill" | "gone" | "refused"

#: How long to wait, after the graceful ``stop`` op is acked, for the process
#: to actually exit before escalating to SIGTERM. The op acks before its clean
#: exit finishes (gates deny, dispose, unpublish), and 10 s comfortably covers
#: a session draining an interruptible tool on a loaded machine — the default
#: every front end can afford — while ``lop stop --timeout`` lets a script
#: trade patience for promptness.
DEFAULT_TIMEOUT_S = 10.0

#: The SIGTERM rung's budget: the runtime's existing handler runs the same
#: deny → dispose → unpublish ordering the socket op does, and 3 s is the
#: process drain budget the mobile child already uses elsewhere
#: (``process.DEFAULT_GRACE_S``). After this, the runtime is not listening to
#: anyone and SIGKILL is the only remaining answer.
SIGTERM_GRACE_S = 3.0

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
    line), and ``StreamReader.readline`` raises ``LimitOverrunError`` WITHOUT
    consuming the buffer — every later read re-raises on the same bytes, so
    one big welcome wedges the reader, the exact defect U1 fixed for `lop
    send`. Discarding an oversized line (over ``_MAX_FRAME_BYTES`` with no
    newline yet) keeps memory bounded while the frames AFTER it — including
    the ack this function may be waiting for — survive in the same buffer.
    """
    import asyncio
    import json

    buf = bytearray()
    skipping = False

    async def _next_line() -> bytes:
        nonlocal buf, skipping
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
    """Identity from the RECORD FIELDS alone — no socket, no clock.

    The record the runtime wrote says: this pid, this session, this kind,
    this cwd, this port/key. A process at that pid that serves the SAME
    session on the SAME port with the SAME key is the owner beyond pid-reuse
    doubt, because the port+key are chosen at spawn and recycled-pid
    coincidence across three independent fields is not a real shape.

    Admissible ONLY under ``--force`` (the caller's explicit opt-in): the
    socket identity is the load-bearing proof everywhere else, and this
    fallback exists for the one shape the socket cannot reach — a
    heartbeating-but-starved process (a TUI burning 100% CPU in a tight
    tool-error loop, its socket loop queued behind the loop) that the
    refusal rule would otherwise hold forever. ``kind == "tui"`` is the
    load-bearing case; a ``daemon``-kind runtime starves the same way but
    has no terminal to lose, so the SIGTERM is safe there too once identity
    is proven from the record.

    Two facts must BOTH hold, and neither is optional:

    1. The record on disk still names this pid and session (``_same_uid``).
    2. The pid still HOLDS the port the record claims, and its heartbeat is
       fresh. A starved runtime keeps heartbeating (the heartbeat thread is
       SIGSTOP-immune and independent of the socket loop), so a fresh
       heartbeat is precisely what separates "alive but not answering" from
       "record outlived its process". Without this, --force degenerates into
       "signal whatever holds this pid": a stale record whose pid a stranger
       inherited passed check 1 alone and SIGTERMed an unrelated process in
       testing — the exact pid-reuse accident the ladder exists to prevent.
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
            f"belong to another process now — identity cannot be confirmed"
        )
    if not _pid_holds_port(record):
        return False, "the pid no longer holds the control port its record claims"
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
    return f":{record.control_port}" in out


async def _identity_by_start_time(record: SessionRecord) -> tuple[bool, str]:
    """Identity for a process that will not answer its socket.

    A wedged runtime — alive, heartbeat stale, socket silent — is the one
    the kill switch exists for, and the socket check cannot confirm it. So
    a second source that needs NO cooperation from the process, admitted
    ONLY when the heartbeat is stale (see the inline comment for why a fresh
    heartbeat forbids it): the process at this pid must have STARTED before
    the record's last heartbeat. A
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
    # Only a WEDGED record earns this proof. A heartbeat inside the timeout
    # says the recorded process was alive and serving moments ago, so a
    # silent socket means the socket is not that process's — the record is
    # a stale file whose pid a stranger now holds (seen live: a fresh fake
    # record over a ``sleep`` passed the start-time check and was
    # signalled). Stale heartbeat + old start time is the one shape that
    # is both wedged and provably the original process.
    age = time.time() - record.heartbeat_at
    if age <= HEARTBEAT_TIMEOUT_S:
        # Name the remedy: the refusal is only true while the heartbeat is
        # fresh, and the wait is bounded (one heartbeat window). --force is
        # the opt-in past it for a starved process the socket cannot reach.
        wait_s = int(HEARTBEAT_TIMEOUT_S - age) + 1
        return (
            False,
            f"it is heartbeating but not answering its socket "
            f"(its last heartbeat was {int(age)}s ago; it must lapse "
            f"(~{wait_s}s) before a forced stop is safe — retry then, "
            f"or pass --force)",
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


def _record_retired(record: SessionRecord) -> bool:
    """True once the record on disk no longer describes this session.

    The graceful op's observable end is NOT always a process exit: a
    TUI-owned session ends beneath a terminal that stays up, and its only
    trace of the stop is the unpublished (or rewritten) record. Reading
    the file rather than ``registry.scan`` keeps this a single stat+read on
    the 100 ms poll, and a file that now names a different session (the
    process reopened something else) counts as retired too.
    """
    import json

    path = registry.run_dir() / f"{record.pid}.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return True
    return data.get("session_id") != record.session_id


async def _await_stopped(record: SessionRecord, timeout_s: float) -> bool:
    """Wait for the stop to land, bounded by ``timeout_s``.

    Landed means the pid left the process table (a runtime process) OR the
    record was unpublished (a TUI owner whose process survives with the
    session ended beneath it). Polls rather than waiting on a child: these
    processes are not this process's children, so there is no waitpid
    right to hang on. 100 ms is fine-grained enough that a clean exit
    (sub-second in the common case) is observed almost immediately, and
    cheap enough to hold for 10 s without cost. Returns False on timeout —
    the caller's next escalation rung.
    """
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        if not registry.pid_alive(record.pid) or _record_retired(record):
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


def _stopped_line(record: SessionRecord, method: Method, wakes: int) -> str:
    """The one human receipt line every front end paints for one stop.

    ``socket`` reads as a plain "stopped"; the signal rungs SAY so —
    ``stopped … (sigterm)`` / ``killed …`` — because a kill switch that
    reports an escalated stop as a graceful one hides the one fact the
    user would act on next time (that runtime was not answering its
    socket). ``sigkill`` gets its own verb: state was orphaned.
    """
    name = record.conversation_name or record.session_id
    verb = "killed" if method == "sigkill" else "stopped"
    rung = " (sigterm)" if method == "sigterm" else ""
    wakes_part = (
        f" — {wakes} wake{'s' if wakes != 1 else ''} dormant until you reopen it" if wakes else ""
    )
    return f'{verb} "{name}"{rung}{wakes_part}'


async def _graceful_stop(record: SessionRecord, timeout_s: float) -> bool:
    """Rung 1: ask the runtime to stop itself, wait out the clean exit.

    The op the runtime serves (``RuntimeServer._dispatch``'s ``stop`` case)
    runs deny-pending-gates → dispose → unpublish → exit; the ack comes back
    when the decision is made, the exit lands moments later. A TUI owner
    runs the same op but its PROCESS stays (the session ends beneath the
    terminal), so the wait accepts an unpublished record as the landing. An ``error``
    reply (an old runtime that predates the op) is a scheduled miss, not a
    failure — the ladder proceeds to identity confirmation and SIGTERM, which
    every runtime already handles, so mixed-version machines never wedge.
    """
    reply = await _exchange(record, {"op": "stop"}, reply_timeout_s=timeout_s)
    if reply is None or reply.get("op") != "ack":
        return False
    return await _await_stopped(record, timeout_s)


async def _signal_and_confirm(record: SessionRecord, sig: "signal.Signals", grace_s: float) -> bool:
    """Rungs 2–3: signal the confirmed pid and wait out its grace window.

    Called only AFTER identity confirmation — this is the rung that can hit
    a process, which is exactly why nothing reaches it unconfirmed. SIGTERM
    rides the runtime's existing handler (the same clean exit as the socket
    op); SIGKILL has no handler by definition — state is orphaned and
    recovered by the stale-record reap plus the lease's dead-owner recovery,
    which is exactly what those mechanisms exist for.
    """
    try:
        os.kill(record.pid, sig)
    except (ProcessLookupError, PermissionError):
        return not registry.pid_alive(record.pid)
    return await _await_pid_exit(record.pid, grace_s)


def _recover_record(record: SessionRecord) -> None:
    """Best-effort stale-record cleanup after a confirmed exit.

    A clean stop unpublishes its own record; the SIGKILL rung cannot (the
    process is gone). ``registry.scan`` reaps dead-pid records on any
    reader's next pass, so this is not load-bearing — it is the polite
    version that makes `lop sessions` correct immediately instead of at the
    next scan, and it must never raise over a file that is already gone.
    """
    if registry.pid_alive(record.pid):
        return
    registry.unpublish(record.pid)


async def stop_session(
    record: SessionRecord,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    force: bool = False,
    _root: Path | None = None,
) -> StopOutcome:
    """Stop one live session by its discovery record. Never raises.

    The escalation ladder in order: graceful socket op → identity-confirmed
    SIGTERM → identity-confirmed SIGKILL, with a refuse when identity cannot
    be confirmed ahead of a signal. See the module docstring for the rules;
    this function is where they are enforced in order.

    ``force`` admits the record-field identity proof when the socket cannot
    answer — the explicit opt-in for a heartbeating-but-starved process the
    refusal rule would otherwise hold forever (see ``_identity_by_record``).

    ``_root`` is the config root (tests inject one); production callers use
    the ambient ``config_dir()``.
    """
    root = _root if _root is not None else config_dir()
    name = record.conversation_name or record.session_id

    # Rung 1 — the graceful op. Both its failure shapes are scheduled misses:
    # an unreachable socket means already-gone-or-crashed, an error reply
    # means an older runtime. Either way the ladder continues.
    if await _graceful_stop(record, timeout_s):
        wakes = await _park_wakes(record, root)
        _recover_record(record)
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
    if not registry.pid_alive(record.pid):
        wakes = await _park_wakes(record, root)
        _recover_record(record)
        method = "gone"
        return StopOutcome(
            pid=record.pid,
            session_id=record.session_id,
            name=name,
            method=method,
            line=f'"{name}" already exited',
            wakes_dormant=wakes,
        )

    # Identity gate before ANY signal — the pid-reuse rule. Confirmed here,
    # after the graceful rung (which signals nobody), so a cooperative
    # runtime is never asked to prove itself and a wedged one cannot
    # fast-path to a signal it did not earn.
    confirmed, why_not = await _confirmed_session_id(record)
    if not confirmed and why_not == _SOCKET_SILENT:
        # Alive but silent: the wedged case. The socket cannot vouch for
        # it, so fall back to the start-time proof — still a proof, still
        # refusing when it cannot be made. A socket that ANSWERED with a
        # different session id never reaches here: that is a live stranger
        # and stays refused.
        confirmed, why_not = await _identity_by_start_time(record)
    if not confirmed and force:
        # --force: the one identity the socket cannot give (a starved
        # process that never services its loop) is read from the record's
        # own fields — port, key, session, kind — which a recycled pid
        # cannot collide on. Gated on the caller's explicit opt-in and
        # named in the receipt, because the refusal rule this bypasses is
        # the one that keeps a wrong pid alive.
        # Off the loop: this forks ``lsof`` and the TUI runs the ladder on
        # its event loop, the same constraint the ``ps`` probe above carries.
        confirmed, why_not = await asyncio.to_thread(_identity_by_record, record)
        if confirmed:
            why_not = ""
    if not confirmed:
        method = "refused"
        return StopOutcome(
            pid=record.pid,
            session_id=record.session_id,
            name=name,
            method=method,
            line=f'refused "{name}" (pid {record.pid}) — {why_not}',
        )

    # Rung 2 — SIGTERM: the runtime's existing handler runs the same clean
    # exit the socket op would have.
    if await _signal_and_confirm(record, signal.SIGTERM, SIGTERM_GRACE_S):
        wakes = await _park_wakes(record, root)
        _recover_record(record)
        method = "sigterm"
        return StopOutcome(
            pid=record.pid,
            session_id=record.session_id,
            name=name,
            method=method,
            line=_stopped_line(record, method, wakes),
            wakes_dormant=wakes,
        )

    # Rung 3 — SIGKILL. State is orphaned by design; stale-record reaping
    # and the lease's dead-owner recovery pick it up. Report the rung used.
    await _signal_and_confirm(record, signal.SIGKILL, SIGTERM_GRACE_S)
    wakes = await _park_wakes(record, root)
    _recover_record(record)
    method = "sigkill"
    return StopOutcome(
        pid=record.pid,
        session_id=record.session_id,
        name=name,
        method=method,
        line=_stopped_line(record, method, wakes),
        wakes_dormant=wakes,
    )


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
) -> list[StopOutcome]:
    """Stop every OTHER agent on this machine. Never raises.

    ``own_pid`` is excluded outright (see :func:`_stop_targets`); the caller
    ends its own session in-process afterwards. ``only_pids`` restricts the
    run to a set the user was SHOWN — the TUI's arm listing is the
    confirmation, so a session that appeared between arm and repeat must
    not be stopped on the strength of a listing it was never on.

    Sequential, not concurrent: the graceful rung waits up to ``timeout_s``
    per uncooperative session, and a fan-out would hold every target's wait
    open at once — the opposite of what a user pressing a kill switch wants
    when one wedged runtime sits in front of twelve healthy ones. One at a
    time, healthiest first (scan order), so the common case is fast and the
    wedged tail is paid only by whoever actually needs the signals.
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
        outcomes.append(await stop_session(record, timeout_s=timeout_s, force=force, _root=root))
    return outcomes


#: Outcomes that count as "the session is no longer running", i.e. the stop
#: did its job. Everything else (``refused``) is the partial case.
ENDED_METHODS = frozenset({"socket", "sigterm", "sigkill", "gone"})


def summarize(outcomes: list[StopOutcome], *, own: StopOutcome | None = None) -> str:
    """The grouped report every front end paints after ``stop all``.

    Reconciles with the promise the listing made: leads with the total the
    user was told would be stopped, then the rung grouping — the honest
    summary of an escalation is how many stopped cleanly, how many needed a
    signal, how many were already gone and how many were refused, not twelve
    identical lines. ``own`` is the caller's in-process outcome (the TUI's
    own session), folded into the total and the ``stopped`` count so the
    numbers add up on one line instead of across three.
    """
    everything = list(outcomes) + ([own] if own is not None else [])
    if not everything:
        return "no sessions to stop"
    order: list[tuple[str, str]] = [
        ("socket", "stopped"),
        ("sigterm", "stopped via sigterm"),
        ("sigkill", "killed"),
        ("gone", "already exited"),
        ("refused", "refused"),
    ]
    parts: list[str] = []
    for method, label in order:
        count = sum(1 for o in everything if o.method == method)
        if count:
            parts.append(f"{count} {label}")
    total = len(everything)
    return f"{total} session{'s' if total != 1 else ''}: " + ", ".join(parts)
