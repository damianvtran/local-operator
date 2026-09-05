"""``engage_runtime``: the one way work reaches a session, running or not.

Every path that has something for a session to do — a viewer's first message, a
peer note, a scheduled wake, a phone continuation — calls this. It answers one
question ("is there a runtime for this session, and if not, whose job is it to
start one?") in one place, because the answer involves a race that is easy to
get wrong in each caller separately and impossible to get wrong once here.

**The invariant: at most one runtime per session, ever.** Two runtimes on one
transcript is a forked trajectory — both append, neither sees the other's rows,
and the conversation silently splits. The arbiter is the transcript LEASE, not
a check before spawning: a check-then-spawn has a window between the check and
the spawn, and that window is exactly as wide as session construction (~1.2 s),
which is long enough for ten contenders to walk through it together.

So every contender is allowed to spawn a candidate, and the lease decides.
Losers exit 0 — they lost a race that was designed to be lost, not encountered
an error (``process.py`` logs the loss and returns 0 for that reason).

**The loop** (design §11.3), in order, until the deadline:

1. **A live record?** Deliver over its socket and return. The common case.
2. **A lease naming a live pid?** Someone is CONSTRUCTING a runtime right now
   — it has claimed the transcript but has not published a record yet, a
   window about as long as session construction. Wait and re-loop rather than
   spawning a second candidate that is doomed to lose. This is what keeps the
   spawn count at one for N simultaneous engagements.
3. **Neither?** Spawn once, then re-loop. Only once per call: a second spawn
   from the same caller cannot help — if the first is still constructing, (2)
   now covers it — and would just be another loser to reap.

The backoff (``0.05 → ×1.7 → cap 1.0``) is shaped for step 2's wait: fast
enough that a warm start feels immediate, and backing off to seconds so a
30-second deadline does not become a spin.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)

#: How long an engagement will keep trying before giving up. Sized for the
#: worst realistic cold start (session construction plus MCP settling) with
#: room to spare, because the alternative to waiting is telling a user their
#: message went nowhere.
DEFAULT_DEADLINE_S = 30.0

#: Poll shape while waiting for a contender's runtime to publish its record.
_POLL_INITIAL_S = 0.05
_POLL_FACTOR = 1.7
_POLL_CAP_S = 1.0
#: How many runtimes one engage may spawn before it stops trying. Only the
#: FIRST spawn is ordinary; the rest are respawns after a candidate proved to
#: have died during construction. Three is enough to ride out a transient
#: (a momentarily unreadable credential file, a port in TIME_WAIT) while
#: keeping a genuinely unconstructable session from respawning for the whole
#: deadline and burying its real error under a crash loop.
_MAX_SPAWNS = 3


@dataclass(frozen=True, slots=True)
class PromptErrand:
    """A user turn. The reason a session usually starts."""

    text: str
    images: list[dict[str, str]] = field(default_factory=list)
    command_id: str = ""


@dataclass(frozen=True, slots=True)
class SteerErrand:
    """A mid-turn injection into a session that is already working."""

    text: str
    images: list[dict[str, str]] = field(default_factory=list)
    command_id: str = ""


@dataclass(frozen=True, slots=True)
class PeerMessageErrand:
    """A message from another local lop session (``lop send``)."""

    text: str
    mode: str = "mailbox"
    wake: bool = False
    sender: dict[str, Any] = field(default_factory=dict)
    command_id: str = ""


@dataclass(frozen=True, slots=True)
class WakeErrand:
    """Start a cold session because one of its wakes is due.

    **It delivers nothing, and that is the whole design.** The obvious shape —
    a ``wake_fire`` op telling the runtime which occurrence to deliver — fires
    every wake TWICE, because a session already delivers its own overdue wakes
    on load: ``WakeScheduler.load`` re-arms anything whose ``next_due_at`` has
    passed to ``now + LOAD_GRACE_MS`` and records it for the resume catch-up
    (``harness/wake.py``), so the mere existence of the runtime is what fires
    the wake. An op on top of that would append the occurrence a second time.

    So the supervisor's job is strictly to make a runtime EXIST for a session
    whose wake is due; the session then does what it would have done had a
    terminal been open. ``schedule_id`` and ``occurrence_ms`` are carried for
    the log line and for the derived ``command_id``, which is what keeps a
    supervisor retry from starting two runtimes for one occurrence.
    """

    schedule_id: str
    occurrence_ms: int
    command_id: str = ""


@dataclass(frozen=True, slots=True)
class WarmErrand:
    """Start the runtime, deliver nothing.

    The speculative engage: the viewer fires this on the first keystroke so
    that by the time a message is actually submitted the runtime is already
    constructed, turning a ~1.2 s wait into no wait at all. It returns as soon
    as the record is live — there is nothing to deliver — and the runtime it
    starts defers materialising the session directory until real work arrives,
    so an abandoned draft leaves nothing behind.
    """

    command_id: str = ""


Errand = Union[PromptErrand, SteerErrand, PeerMessageErrand, WakeErrand, WarmErrand]


@dataclass(frozen=True, slots=True)
class EngageOutcome:
    """What the engagement did, for the caller's receipt and for metrics."""

    session_id: str
    #: The ack line the runtime returned, or a short local description for a
    #: warm engage that delivered nothing.
    detail: str
    #: True when this call started the runtime rather than finding one.
    spawned: bool = False
    #: True when the runtime recognised the ``command_id`` as already admitted
    #: and did nothing — a retry that correctly declined to double-deliver.
    duplicate: bool = False


def new_command_id() -> str:
    return str(uuid.uuid4())


def _session_dir(config_dir: Path, session_id: str) -> Path:
    return config_dir / "sessions" / session_id


def _lease_holder(config_dir: Path, session_id: str) -> int | None:
    """Pid currently holding the transcript lease, if it is alive.

    Read directly rather than through ``acquire_session_lease``: this is a
    PROBE, and acquiring in order to find out would take the very lease the
    runtime needs. Uses the lease's own claim reader so both agree on the
    format.
    """
    from local_operator.session_lease import LEASE_NAME, _pid_state, _read_claim

    path = _session_dir(config_dir, session_id) / LEASE_NAME
    if not path.exists():
        return None
    _generation, pid = _read_claim(path)
    if pid is None:
        return None
    return pid if _pid_state(pid) == "live" else None


def _spawn_runtime(
    session_id: str, cwd: str, *, defer_materialise: bool
) -> "subprocess.Popen[bytes]":
    """Start one detached runtime candidate for ``session_id``.

    Returns the ``Popen`` so the engage loop can tell a candidate that is
    still CONSTRUCTING from one that died: the lease is acquired a few
    hundred milliseconds after the exec, and in that window (no record) and
    (no lease) is true of a perfectly healthy candidate — so liveness of the
    process we spawned is the only sound death signal (round 2, Q8).

    Its stdio is CAPTURED to a file, recorded on the returned object as
    ``lop_capture_path``. Both streams used to go
    to ``DEVNULL``, which made a candidate that died before
    ``logging.basicConfig`` ran completely silent — the failure had no
    traceback, no message and no exit reason anywhere on the system, so a
    session that could never start looked identical to a slow one. That
    silence is what turned a clear "hosting is not configured" into a
    30-second wait and an unexplained 503 (QA Q1).

    The path rides on the Popen rather than widening the return type, so every
    existing caller and test double keeps working with a plain process object.
    The file is small, per-candidate, created 0600, and unlinked as soon as it
    is read.

    Only routing data enters the environment — prompt text, images and command
    identity travel over the authenticated loopback socket, never through
    ``ps``-readable state. These are the two variables ``process.py`` already
    reads, plus the deferred-materialisation flag; the spawn stays bare by
    design (design §11.3 C1).
    """
    env = dict(os.environ)
    env["LOP_MOBILE_CHILD_CWD"] = cwd
    env["LOP_MOBILE_CHILD_RESUME"] = session_id
    if defer_materialise:
        env["LOP_RUNTIME_DEFER_MATERIALISE"] = "1"
    else:
        # A parent that set this for an earlier speculative engage must not
        # leak it into a runtime that has real work to do.
        env.pop("LOP_RUNTIME_DEFER_MATERIALISE", None)
    # 0600 at CREATION, via mkstemp. `Path.open("wb")` takes the process umask
    # (measured 0o644 here), leaving the child's entire stdout+stderr --
    # tracebacks, provider error bodies, config echoes -- world-readable in a
    # shared /tmp. mkstemp also generates the random suffix itself, so the
    # session id no longer has to carry the uniqueness and a directory listing
    # stops disclosing live session ids to other local users. The prefix keeps
    # these recognisable as this project's spawn captures.
    handle_fd, capture_path = tempfile.mkstemp(prefix="lop-runtime-", suffix=".log")
    capture = Path(capture_path)
    handle = os.fdopen(handle_fd, "wb")
    try:
        process = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            [sys.executable, "-m", "local_operator.session.runtime.process"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        # The child holds its own duplicated descriptor; this one is ours to
        # drop so the file is not kept open for the life of the server.
        handle.close()
    setattr(process, "lop_capture_path", capture)
    return process


#: Upper bound on captured child output quoted back to a caller. Enough for a
#: traceback's final frames, small enough that a runaway child cannot turn an
#: error message into a memory problem.
_CAPTURE_TAIL_BYTES = 4096

#: Startup conditions whose cause is CONFIGURATION the user can act on, mapped
#: to the sentence to show them. A curated map rather than the child's raw text
#: because a construction failure can quote a provider endpoint or a filesystem
#: path, and an error surface is not the place to discover that. Anything not
#: listed stays generic; the full traceback is in the runtime log either way.
_ACTIONABLE_STARTUP_REASONS = {
    "HostingNotConfiguredError": (
        "No model provider is configured yet. Connect one in Settings > Providers, "
        "then send the message again."
    ),
    "HostingUnknownError": (
        "This session's model provider is not recognised. Choose a provider in "
        "Settings > Providers, then send the message again."
    ),
    "ModelNotConfiguredError": (
        "No model is selected for this session. Pick one with /model, then send "
        "the message again."
    ),
}


class RuntimeStartupError(RuntimeError):
    """No runtime could be started, with a reason worth showing a person.

    Distinct from ``TimeoutError`` (nothing answered in time) because the two
    call for opposite responses: a timeout invites a retry, whereas this says
    retrying changes nothing until something is configured.
    """

    def __init__(self, message: str, *, actionable: str = "") -> None:
        super().__init__(message)
        #: A vetted, user-facing sentence, or "" when the cause was not one of
        #: the known configuration conditions.
        self.actionable = actionable


class ActionableConnectionError(ConnectionError):
    """A ``ConnectionError`` whose MESSAGE is a vetted, user-facing sentence.

    The type is the permission slip. Callers that relay owner failures to a user
    surface (the desktop HTTP routes) may echo ``str(error)`` for this class and
    must fall back to a generic sentence for every other ``ConnectionError``.

    That distinction cannot be recovered from the message text, and the previous
    round proved it: the relay echoed EVERY ``ConnectionError`` verbatim on the
    strength of a docstring claiming they were limited to the vetted set, and
    shipped ``owner socket unreachable: [Errno 61] Connect call failed
    ('127.0.0.1', 54321)`` (an internal control port) and ``owner moved to
    another conversation (abc123secretsession)`` (another session's id) into the
    renderer. ``attach_client`` raises bare ``ConnectionError`` from a dozen
    places carrying socket errors, peer ids and provider text; only the
    configuration reasons curated in :data:`_ACTIONABLE_STARTUP_REASONS` are
    fit to show, so only they get this type.
    """

    #: Marks this error's message as vetted for display. An attribute rather
    #: than a bare `isinstance` so a caller reads as "is this actionable?"
    #: rather than having to know the class hierarchy.
    actionable = True


def _spawn_failure_reason(capture: Path) -> tuple[str, str]:
    """The child's last words: ``(log_detail, user_facing)``.

    ``log_detail`` is the traceback's terminal line, for the server log. The
    second element is filled only when that line names one of the known
    configuration conditions, so nothing unvetted reaches a user surface.
    """
    try:
        raw = capture.read_bytes()[-_CAPTURE_TAIL_BYTES:].decode("utf-8", "replace")
    except OSError:
        return "", ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        # A traceback's terminal line is "module.QualifiedError: message".
        if line.startswith(("Traceback", "  ", "File ")) or ": " not in line:
            continue
        qualified, _, _message = line.partition(": ")
        name = qualified.rsplit(".", 1)[-1].strip()
        return line, _ACTIONABLE_STARTUP_REASONS.get(name, "")
    return (lines[-1] if lines else ""), ""


async def _deliver(record: Any, session_id: str, work: Errand) -> tuple[str, bool]:
    """Hand one errand to a live runtime. Returns ``(detail, duplicate)``."""
    from local_operator.mobile.peer_client import send_peer_message

    if isinstance(work, (WarmErrand, WakeErrand)):
        # Neither delivers anything: a warm engage exists to pay the start-up
        # cost early, and a wake is delivered by the session's own scheduler
        # the moment it loads (see WakeErrand). Reaching a live runtime IS the
        # completed errand for both.
        return "runtime ready", False
    if isinstance(work, PeerMessageErrand):
        detail = await send_peer_message(
            record,
            text=work.text,
            mode=work.mode,
            wake=work.wake,
            sender=work.sender,
        )
        return detail, False

    from local_operator.mobile.attach_client import AttachClient

    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        await client.connect(record, session_id)
        op = "prompt" if isinstance(work, PromptErrand) else "steer"
        return await client.request_ack_with_duplicate(
            op,
            text=work.text,
            images=work.images,
            command_id=work.command_id,
        )
    finally:
        client.close()


async def engage_runtime(
    session_id: str,
    cwd: str,
    work: Errand,
    *,
    config_dir: Path,
    deadline_s: float = DEFAULT_DEADLINE_S,
) -> EngageOutcome:
    """Ensure a runtime exists for ``session_id`` and give it ``work``.

    The single arbitration point; see the module docstring for the loop and
    why the lease rather than a pre-spawn check decides who runs.

    Raises ``TimeoutError`` if no runtime could be reached within the
    deadline, and ``RuntimeError`` — carrying the child's own reason — as soon
    as every candidate it is allowed to start has died. Every other failure (a
    refused op, a dead socket) surfaces as the underlying error from the
    delivery attempt, since those are the caller's to report.
    """
    from local_operator.mobile.attach_client import find_owner_record

    if not getattr(work, "command_id", ""):
        # Identity is what makes a retry safe. A caller that did not supply one
        # gets one here rather than being silently non-idempotent.
        work = type(work)(**{**_fields(work), "command_id": new_command_id()})

    deadline = time.monotonic() + deadline_s
    delay = _POLL_INITIAL_S
    spawned = False
    # The Popen of the most recent candidate, so the respawn branch can tell
    # a live constructor from a dead one without reading the lease.
    candidate: "subprocess.Popen[bytes] | None" = None
    # Where the current candidate's stdio is being captured, so a death can be
    # reported with the child's own reason instead of a generic timeout.
    capture: Path | None = None
    # The child's terminal traceback line (for the log) and, when the cause was
    # a known configuration condition, the sentence to show the user.
    spawn_reason = ""
    spawn_actionable = ""
    # Counted separately from ``spawned`` because a respawn after a candidate
    # died mid-construction is a different event from the first spawn, and
    # only the retries need a cap. See the respawn branch below.
    spawns = 0
    last_error: Exception | None = None
    # Deferred materialisation is exactly the speculative case: a warm engage
    # must not create a session directory for a draft the user may abandon.
    # A wake engage is NOT speculative — the session already exists on disk.
    defer = isinstance(work, WarmErrand)

    while time.monotonic() < deadline:
        record, _owner = await asyncio.to_thread(find_owner_record, config_dir, session_id)
        if record is not None:
            try:
                detail, duplicate = await _deliver(record, session_id, work)
                if capture is not None:
                    # The candidate became the owner (or someone else's did);
                    # its captured stdio has served its purpose.
                    capture.unlink(missing_ok=True)
                    capture = None
                return EngageOutcome(
                    session_id=session_id,
                    detail=detail,
                    spawned=spawned,
                    duplicate=duplicate,
                )
            except (ConnectionError, TimeoutError) as exc:
                # The runtime died between the scan and the dial. Re-loop: the
                # record will be gone next pass and we spawn a fresh one.
                last_error = exc
                logger.debug("engage: dial failed for %s; retrying", session_id, exc_info=True)

        if not spawned or spawns < _MAX_SPAWNS:
            holder = await asyncio.to_thread(_lease_holder, config_dir, session_id)
            if holder is not None:
                # STARTING: a contender holds the transcript but has not
                # published yet. Spawning here would create a doomed candidate,
                # so wait for its record instead. This is the whole reason the
                # loop looks at the lease at all.
                logger.debug("engage: %s is starting under pid %s; waiting", session_id, holder)
            elif not spawned:
                logger.debug("engage: spawning a runtime for %s", session_id)
                candidate = await asyncio.to_thread(
                    _spawn_runtime, session_id, cwd, defer_materialise=defer
                )
                capture = getattr(candidate, "lop_capture_path", None)
                spawned = True
                spawns += 1
            elif candidate is not None and candidate.poll() is not None:
                # THE CANDIDATE WE SPAWNED IS GONE. It exited while no record
                # exists and nobody holds the lease — a winner dying DURING
                # construction (`process.py`'s own `return 2`, an OOM kill, a
                # bad credential, an MCP hang taking the process down). This
                # is not the designed-loser path: a loser exits 0 without ever
                # holding the lease, and the winner it lost to still holds it,
                # so this branch cannot fire for one.
                #
                # ``poll() is not None`` is the death signal, not the absence
                # of a lease: the lease is acquired a few hundred milliseconds
                # after the exec, and in that window (no record) and (no
                # lease) is true of a perfectly healthy candidate. Round 1's
                # R1 fix read that window as death and respawned on EVERY
                # engage — three processes per first message, two of them
                # doomed (round 2, Q8). A LIVE candidate is by definition
                # still constructing, so the loop waits for its record like
                # any other contender.
                #
                # Bounded because a session that cannot construct at all
                # (missing credential, unreadable transcript) would otherwise
                # respawn until the deadline, turning one clear failure into a
                # crash loop. Past the cap the loop STOPS rather than waiting
                # out the deadline: see the fast-fail below.
                if capture is not None:
                    reason, actionable = await asyncio.to_thread(_spawn_failure_reason, capture)
                    spawn_reason = reason or spawn_reason
                    spawn_actionable = actionable or spawn_actionable
                    capture.unlink(missing_ok=True)
                logger.info(
                    "engage: the candidate for %s died during construction (rc=%s): %s",
                    session_id,
                    candidate.returncode,
                    spawn_reason or "no output captured",
                )
                candidate = await asyncio.to_thread(
                    _spawn_runtime, session_id, cwd, defer_materialise=defer
                )
                capture = getattr(candidate, "lop_capture_path", None)
                spawns += 1

        if (
            spawned
            and spawns >= _MAX_SPAWNS
            and candidate is not None
            and candidate.poll() is not None
        ):
            # FAIL FAST. Every candidate we are allowed to start has died, and
            # nobody else holds the lease, so no amount of further waiting can
            # produce a runtime. Blocking out the rest of the deadline turned a
            # diagnosable startup failure into ~30 s of apparent hang followed
            # by a generic 503 (QA Q1); the caller can now say WHAT failed,
            # immediately, using the child's own message.
            if capture is not None:
                reason, actionable = await asyncio.to_thread(_spawn_failure_reason, capture)
                spawn_reason = reason or spawn_reason
                spawn_actionable = actionable or spawn_actionable
                capture.unlink(missing_ok=True)
                capture = None
            raise RuntimeStartupError(
                f"could not start a runtime for session {session_id}"
                + (f": {spawn_reason}" if spawn_reason else ""),
                actionable=spawn_actionable,
            ) from last_error

        await asyncio.sleep(min(delay, max(0.0, deadline - time.monotonic())))
        delay = min(delay * _POLL_FACTOR, _POLL_CAP_S)

    if capture is not None:
        capture.unlink(missing_ok=True)
    raise TimeoutError(
        f"could not reach a runtime for session {session_id} within {deadline_s:.0f}s"
    ) from last_error


def _fields(work: Errand) -> dict[str, Any]:
    """Field values of one errand, for rebuilding it with an id attached."""
    return {name: getattr(work, name) for name in work.__slots__}
