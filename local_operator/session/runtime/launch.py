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

**What the parent's dead time is NOT.** Two other suspects were measured and
acquitted, so nobody re-derives them. (1) The first ``find_owner_record``
scan is a guaranteed miss for a freshly minted ``/new`` session, which looks
like serialized latency ahead of the spawn — but in the warm parent that
actually runs ``/new`` (a long-lived TUI) engage-entry to fork measures a
median of 2.1 ms. The ~24 ms a cold process shows is function-local import
cost the TUI has already paid, so skipping the scan would buy single-digit
milliseconds while weakening the lease arbitration this module exists to
protect. (2) The child's import graph is dominated by
``local_operator.harness.jobs`` at 99 ms cumulative, but that is a SHARED
subtree: given ``session_factory`` (the composition root the child imports
regardless), its marginal cost is 2.2 ms. Deferring it would move ~2 ms.

**The poll shape has two regimes, because the loop has two waits.** When a
construction is KNOWN to be in flight — no record exists yet, AND either a
candidate we spawned is still alive or a contender holds the lease — we have
a strong prior on when a record will appear (~0.4 s for a deferred warm
start, ~1.2 s for a full cold session; both measured on an M-series dev box
via ``scripts/bench_runtime_attach.py``), so the grid is DENSE and flat.
When nothing is known to be constructing the wait is open-ended and the
exponential backoff takes over — and that includes retrying a record which
already exists but will not answer the dial, a slow path that must stay one.
See :func:`_poll_delay` for the measurements behind the constants.
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

#: Poll shape for an OPEN-ENDED wait — nothing is known to be constructing, so
#: there is no prior on when a record might appear and a spin would be pure
#: waste against a 30-second deadline.
_POLL_INITIAL_S = 0.05
_POLL_FACTOR = 1.7
_POLL_CAP_S = 1.0

#: Poll interval while a construction is KNOWN to be in flight. Flat, not
#: exponential: see :func:`_poll_delay` for why the two waits get two shapes.
_CONSTRUCTING_POLL_S = 0.01

#: How long the dense regime may last before the open-ended backoff takes over.
#: A construction still unfinished after this is not "about to publish" — it is
#: wedged, slow, or a contender doing something we cannot see — and continuing
#: to poll it 100 times a second for the rest of a 30-second deadline would be
#: a spin. Sized at ~2.5x the ~1.2 s a full cold session construction takes
#: **as measured on an M-series dev box**.
#:
#: The one-sidedness of that calibration was MEASURED, not argued: modelling a
#: slower host by lengthening the construction, dead time is 11.1/11.7/7.4 ms
#: at 0.4/1.2/2.5 s (window covers it), 67.0 ms at 3.5 s (window lapses
#: mid-construction), and 716.3 vs 726.2 ms at 5.0 s and 726.1 vs 778.6 ms at
#: 8.0 s (fully degraded). Past the window the new shape CONVERGES to the old
#: one and never exceeds it, because the fallback restarts the exponential
#: from ``_POLL_INITIAL_S`` rather than from a value that decayed while we
#: were watching (QA round 1, Q2). So a host 3x slower than this one gets
#: today's behaviour, not a regression.
#:
#: Deliberately NOT structural. Deriving it from an observed construction time
#: needs a measurement the loop does not have on a process's first ``/new``,
#: and buys nothing over a heuristic whose worst case is the status quo.
#: Nothing asserts on this value — it is a poll-frequency heuristic, not a
#: correctness bound.
_CONSTRUCTING_WINDOW_S = 3.0
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
    # The open-ended wait's exponential state. ``delay`` is what we actually
    # sleep on a given pass, which the dense regime overrides without
    # disturbing ``backoff``.
    backoff = _POLL_INITIAL_S
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
    # When the current construction episode was first OBSERVED, which is what
    # selects the dense poll regime (see ``_poll_delay``). Reset to None the
    # moment nothing is known to be constructing, so a respawned candidate
    # gets its own fresh dense window rather than inheriting a spent one.
    constructing_since: float | None = None
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

        holder: int | None = None
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

        # Is a construction KNOWN to be in flight? It requires BOTH that no
        # record exists yet AND that someone is working on one — a contender
        # holding the lease, or the candidate we spawned still being alive.
        #
        # ``record is None`` is load-bearing, not belt-and-braces. The lease is
        # NOT a construction signal on its own: ``session_factory`` registers
        # its release as a dispose hook, so a fully constructed, happily
        # serving runtime holds its lease for its entire life. Without this
        # term, a runtime that has published a record but refuses the dial
        # (wedged control socket, port exhaustion) takes the ``ConnectionError``
        # retry path above and is read as "constructing" on every pass — which
        # turned a slow retry into a hot spin: measured at 301 dial attempts in
        # a 3 s window against 8 under the open-ended grid (review round 1,
        # MAJOR-1). A record that already exists means construction is over,
        # whatever the lease says, so the wait for that runtime to become
        # dialable is open-ended and belongs on the backoff.
        #
        # It also bounds the scan cost the dense interval assumes. A quiet
        # record makes `scan()` fork `ps`, but `find_owner_record` only reaches
        # `scan()` once an owner marker exists; requiring `record is None`
        # keeps the dense regime off the marker-plus-record case entirely. See
        # `_poll_delay` for the measured table and the one overlap that
        # remains.
        constructing = record is None and (
            holder is not None or (candidate is not None and candidate.poll() is None)
        )
        now = time.monotonic()
        if constructing:
            if constructing_since is None:
                constructing_since = now
        else:
            constructing_since = None

        delay, backoff = _poll_delay(
            backoff, None if constructing_since is None else now - constructing_since
        )
        await asyncio.sleep(min(delay, max(0.0, deadline - time.monotonic())))

    if capture is not None:
        capture.unlink(missing_ok=True)
    raise TimeoutError(
        f"could not reach a runtime for session {session_id} within {deadline_s:.0f}s"
    ) from last_error


def _poll_delay(backoff: float, constructing_for_s: float | None) -> tuple[float, float]:
    """Pick the next poll interval, and the backoff to carry forward.

    ``constructing_for_s`` is how long a construction has been KNOWN to be in
    flight — no record has been published yet, and either a candidate we
    spawned is alive or a contender holds the lease — or ``None`` when nothing
    is known to be constructing. Returns ``(sleep_for, next_backoff)``.

    The "no record yet" half is not redundant: a held lease alone does not mean
    construction, because it is released only on session disposal. See the
    caller's comment on ``constructing`` for the retry storm that omitting it
    caused.

    WHY TWO REGIMES
    ===============
    The single exponential grid this replaces served both of the loop's waits,
    but they are not the same wait and the shape that suits one is wrong for
    the other:

    * **Known construction.** No record exists yet, and we have a live
      ``Popen`` (or a lease naming a live pid) plus a good prior on when the
      record lands — measured at ~0.4 s for a deferred warm start and ~1.2 s
      for a full cold session, on an M-series dev box. The record's arrival is
      an EVENT we are already close to; the only question is how soon after it
      we look. A flat, dense grid answers within one interval.
    * **Open-ended wait.** Nothing is known to be constructing. There is no
      prior at all, the wait may run the whole 30-second deadline, and
      polling it densely is a spin that buys nothing. That is the wait the
      exponential backoff was written for, and it keeps it unchanged. A
      published record that refuses the dial lives HERE, not above.

    THE MEASUREMENT THAT FORCED THIS
    ================================
    Under one grid (``0.05 → ×1.7 → cap 1.0``) the polls fire at 50, 135,
    280, 525, 943 ms. A warm ``/new`` child publishes its record at ~400 ms,
    which falls between the 280 ms and 525 ms wakes — so the parent slept
    ~145 ms past a runtime that was already serving, and a child landing just
    after 525 ms waited until 943 ms. Measured over 7 isolated runs
    (``scripts/bench_runtime_attach.py``): child ready at a median of 401 ms,
    parent noticed at 538 ms, **144 ms median dead time and 318 ms at worst**.
    The totals clustered bimodally at ~540 ms and ~990 ms precisely because
    they were quantized to that grid.

    **What this change controls is the dead time, and only that.** An
    independent QA round reproduced the collapse in every one of three
    interleaved A/B runs (~94-136 ms down to ~10-16 ms) but measured the
    TOTAL attach improving by 2.1%, 15.0% and 16.6% \u2014 not the 34.9% an idle
    box shows. The child's own construction dominates the total and varies far
    more than the dead time removed here, so the percentage is a property of
    how loaded the machine is, not of this code. State the dead time when
    quoting this change; the total is a consequence, and a variable one.

    WHY DENSE POLLING IS AFFORDABLE HERE
    ====================================
    The backoff was protecting against a cost that does not exist at these
    timescales. One full poll iteration — ``find_owner_record`` (a miss scan),
    ``_lease_holder``, and ``Popen.poll`` — measures 23-30 µs against a run
    directory of 200 real records, and is FLAT from 0 to 200 because
    ``find_owner_record`` returns before ``scan()`` when there is no owner
    marker (QA round 1, Q3; an earlier author estimate of 339 µs on an
    11-record dir was pessimistic). At a 10 ms interval that is a 0.2-0.3%
    duty cycle on one thread, for at most ``_CONSTRUCTING_WINDOW_S``, and only
    while a session is genuinely starting.

    The nominal 100 Hz is also not what the loop achieves: real work per
    iteration plus scheduling put the measured rate at ~42 Hz (126 dense wakes
    in 3033 ms), so "10 ms" overstates how often the parent actually looks.

    The dense window is bounded rather than open-ended for exactly the reason
    the backoff exists: a construction still unfinished after 3 s is not about
    to publish, so the loop stops guessing and falls back to the open-ended
    shape.

    **That 339 µs assumes a tidy run directory, and record COUNT is not what
    threatens it.** ``scan()`` forks ``ps`` for any record whose heartbeat is
    older than ``HEARTBEAT_INTERVAL_S * 1.5``, and such a record is not reaped,
    so it pays that fork on every scan (review round 1, MINOR-1).

    Where that lands is narrower than it first appears, because
    ``find_owner_record`` returns BEFORE ``scan()`` when the session has no
    ``.session.pid`` owner marker. Measured on this host, 8 fresh records plus
    N quiet ones:

    ====================  ===========  ===========  ===========
    owner marker          0 quiet      1 quiet      3 quiet
    ====================  ===========  ===========  ===========
    absent (no owner)     0.01 ms      0.01 ms      0.01 ms
    present               0.48 ms      14.02 ms     32.13 ms
    ====================  ===========  ===========  ===========

    So the expensive column needs an owner marker, and the cheap row is the
    ordinary ``/new`` dense window — a freshly minted session has no owner, so
    it never reaches ``scan()`` at all and the 339 µs figure holds.

    The requirement that no record exists retires the WORST case (a marker
    plus a dialable record is now always coarse), but one reachable overlap
    remains and is stated here rather than glossed: a marker present with no
    usable record yet — the genuine pre-publish window — can pay ~14 ms per
    poll against a 10 ms interval. That is accepted rather than floored,
    because it is SELF-LIMITING: the scans run sequentially on a ``to_thread``
    worker, so a scan slower than its interval simply yields fewer scans
    instead of a growing backlog, and the loop degrades toward the coarse
    grid's own frequency. A minimum-interval floor would not help — the cost
    is in the scan, not in the sleep — and the window is bounded by
    ``_CONSTRUCTING_WINDOW_S`` regardless.
    """
    if constructing_for_s is not None and constructing_for_s < _CONSTRUCTING_WINDOW_S:
        # Hold the backoff where it is: if the dense window expires, the
        # open-ended wait starts from the top rather than from a value that
        # decayed while we were watching a healthy construction.
        return _CONSTRUCTING_POLL_S, backoff
    return backoff, min(backoff * _POLL_FACTOR, _POLL_CAP_S)


def _fields(work: Errand) -> dict[str, Any]:
    """Field values of one errand, for rebuilding it with an id attached."""
    return {name: getattr(work, name) for name in work.__slots__}
