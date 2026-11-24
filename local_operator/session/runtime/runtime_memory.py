"""The per-runtime memory guard: the one guard that may end a session runtime itself.

**WHY THIS EXISTS, in one incident.** On 2026-11-24 (00:00Z and again 00:38Z) the
runtime of session ``96cf57e7acb7`` cold-booted to 2.0-2.4 GB within seconds while
idle — ``Transcript.__init__`` loading a 974 MB journal (fixed at the load by PR
#4513) — and drove the Mac into swap death twice. Nothing in the product could end
it: ``memory_guard.Guard`` bounds one command's process group, and the machine pass
(:mod:`.machine_memory`) deliberately ends only NON-runtime fragments ("a
conversation is not a runaway, the command under it is"). That rule is right for
the fleet's SUM and leaves one class with no bound at all: a runtime whose OWN
process is the runaway. This module is that bound.

**WHAT IT MEASURES.** Each live runtime of ONE config root (the registry the
machine pass reads), by its OWN pid only, through the fork-free footprint reader
(``mobile.resources.direct_footprint_bytes``: ``ri_phys_footprint`` on macOS, Pss
on Linux — compressed and swapped pages counted at their original size, never a
``ps`` RSS). The runtime's process GROUP is deliberately NOT summed: its children
are commands, already bounded per command by ``memory_guard.Guard`` and in sum by
the machine pass, and a group sum here would end a conversation for its command's
memory — the exact thing the machine pass refuses to do.

**WHEN IT ACTS.** Only when a runtime stays over :data:`RUNTIME_MEMORY_LIMIT_MB_DEFAULT`
(``session.runtime_memory.limit_mb``) for :data:`RUNTIME_MEMORY_SAMPLES` CONSECUTIVE
passes AND for at least :data:`RUNTIME_MEMORY_SUSTAINED_S` of wall time. Both,
because the pass cadence is not a promise (a supervisor restart runs a pass at
once), and a sample count alone could be met in seconds.

**INVARIANT — A NORMAL BIG SESSION IS NEVER ENDED MID-BOOT** (``BOOT_TRANSIENT``):
the default ceiling sits above the largest live runtime measured on the operator's
machine, and a first-open peak lasts seconds, so it cannot satisfy the sustained
window. ``tests/unit/session/runtime/test_runtime_memory.py`` pins both halves.

**WHAT IT DOES.** In order, each step a precondition of the next:

1. proves the pid is still the run the record describes (:func:`verify_run`:
   birth by syscall against the record's ``started_at``, and the process's own
   config root) — a recycled pid or a foreign store's runtime is never touched;
2. stages the involuntary stop marker (``control.note_involuntary_stop``,
   mechanism :data:`MECHANISM`) — the durable sanction the runtime's signal gate
   (``process._route_signal``) requires before it will honour a SIGTERM, and the
   incident the session renders on its next open;
3. stamps the session's wakes and monitors DORMANT (``control.STOP_MARKER``,
   ``stopped_at``) — kept, never deleted, but not re-armed (see
   :func:`end_runaway_runtime` for why not ``parked_at``);
4. SIGTERMs the exact pid, waits :data:`RUNTIME_GUARD_TERM_GRACE_S`, re-proves the
   birth, and SIGKILLs it if it is still there;
5. writes one ``runtime memory guard:`` WARNING line to the supervisor log and
   spools a notice into the session's inbox (the machine guard's owner-notice
   path), which the session reads when it next opens a runtime.

Unknown never acts — the per-command sampler's rule, kept here: a footprint the
host cannot read resets the run of samples, and a birth or root that cannot be
proved refuses the stop.

**WHAT IT NEVER ENDS: THE OPERATOR'S TUI** (``kind == "tui"``, agent review round 1
R5). The TUI publishes a registry record like any runtime, but that process is the
operator's foreground window: ending it from a background pass would close the
screen they are looking at, mid-keystroke, with no surface left to say why. An
over-ceiling TUI gets ONE warning line per run in the supervisor log instead; the
operator, who is by definition at it, decides. ``daemon`` and ``exec`` runtimes are
background processes and are guarded.

**IT KEEPS ITS HOST ALIVE** (QA round 1 Q1). The pass runs in the wake supervisor,
which retires once nothing is fireable — including right after this guard made a
session's wakes dormant, which left a second runaway unguarded for 12 minutes in
QA. :func:`holds_supervisor` is the supervisor's reason to stay: while the guard
is enabled and any guardable runtime of the store is live, it does not retire.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: ``session.runtime_memory.enabled`` — the off switch. On by default: the guard is
#: the only bound on a runtime's own memory, and its default ceiling is sized so
#: that no measured normal session reaches it.
RUNTIME_MEMORY_ENABLED_PATH: tuple[str, ...] = ("session", "runtime_memory", "enabled")
RUNTIME_MEMORY_ENABLED_DEFAULT = True

#: ``session.runtime_memory.limit_mb`` — the per-runtime footprint ceiling, MiB.
#:
#: MEASURED, NOT GUESSED (2026-11-23, the operator's 36 GiB Mac, read-only): 17 live
#: runtimes sampled six times 10 s apart by ``direct_footprint_bytes`` — per-runtime
#: peak median 991 MiB, p90 1,945 MiB, max 5,028 MiB (an idle session with a
#: 1,211 MiB journal on a build without #4513's load fix). Known loads besides:
#: a 1.27 GB journal peaks at 1,193 MiB on first open WITH #4513 and 3+ GiB without
#: it. 8 GiB is 1.6x the observed max — headroom for a boot transient on a build
#: that predates #4513 — and ~22 % of that machine's RAM, low enough that one
#: runaway is ended long before it alone can push a 36 GiB host into swap death.
RUNTIME_MEMORY_LIMIT_MB_PATH: tuple[str, ...] = ("session", "runtime_memory", "limit_mb")
RUNTIME_MEMORY_LIMIT_MB_DEFAULT = 8192

#: The lowest ceiling the guard will accept. A configured value below it is INVALID
#: and the DEFAULT applies (warned once per value), never the floor itself: the
#: measured median runtime is ~1 GiB (p90 1,945 MiB), so clamping a units typo such
#: as ``limit_mb: 8`` (meant as GiB) up to 1024 would still end about half the fleet
#: every few minutes (agent review round 1 R3) — a typo must not become a fleet kill.
RUNTIME_MEMORY_LIMIT_MB_FLOOR = 1024

#: The below-floor values already warned about in this process, so a supervisor
#: that re-reads the setting every pass logs a bad value ONCE rather than once a
#: minute forever (QA round 1 Q2). Keyed by value: a different bad value is a new
#: edit and earns its own line.
_WARNED_LIMITS: set[int] = set()

#: Consecutive over-ceiling passes before the guard acts. With the supervisor's
#: 60 s cadence (``wakes.supervisor.MACHINE_MEMORY_INTERVAL_S``) three samples span
#: two full intervals; see :data:`RUNTIME_MEMORY_SUSTAINED_S` for the wall-time half.
RUNTIME_MEMORY_SAMPLES = 3

#: The least wall time between the first over-ceiling sample and the act. The
#: cadence alone does not bound this: a supervisor restart runs a pass at once, so
#: three samples could otherwise land seconds apart and catch a boot transient.
#: 120 s against a first-open peak that lasts seconds (measured on the #4513 load).
RUNTIME_MEMORY_SUSTAINED_S = 120.0

#: How long the guard waits after its SIGTERM before SIGKILL. Deliberately SHORTER
#: than the receiver's busy drain (``types.SIGNAL_DRAIN_S``, 120 s): the ladder a
#: PERSON runs must never land SIGKILL inside a drain, but this guard acts only on a
#: runtime that has held more memory than any measured session for minutes, and
#: every further second of drain is a second of the host in swap. An IDLE runtime
#: honours the sanctioned SIGTERM promptly (the real-path proof in the PR measured
#: it); a busy one loses its turn, and the involuntary marker attributes that loss.
RUNTIME_GUARD_TERM_GRACE_S = 30.0

#: How long to confirm a SIGKILL landed — the ladder's own bound
#: (``control.SIGKILL_CONFIRM_S``), restated so this module imports no ladder at
#: import time.
RUNTIME_GUARD_KILL_CONFIRM_S = 3.0

#: One second of slack on the birth comparison: macOS ``lstart`` resolution.
_BIRTH_SLACK_S = 1.0

#: The involuntary-stop mechanism token, rendered by
#: ``incidents.INVOLUNTARY_MECHANISM_LABELS``.
MECHANISM = "runtime-memory-guard"

#: The acting component named on the marker and the notice.
ACTOR = "runtime memory guard"


@dataclass(frozen=True)
class RuntimeGuardSettings:
    """The three facts one pass decides with."""

    enabled: bool = RUNTIME_MEMORY_ENABLED_DEFAULT
    limit_mb: int = RUNTIME_MEMORY_LIMIT_MB_DEFAULT
    samples: int = RUNTIME_MEMORY_SAMPLES
    sustained_s: float = RUNTIME_MEMORY_SUSTAINED_S


def read_settings(config_dir: Path) -> RuntimeGuardSettings:
    """The guard's settings, read LIVE from ``config_dir`` (a fresh ``ConfigManager``).

    Config trouble falls back to the defaults rather than disabling the guard: an
    unreadable ``config.yml`` is not the operator saying "off". A ceiling below
    :data:`RUNTIME_MEMORY_LIMIT_MB_FLOOR` is invalid and the default applies, and
    ``0`` or a negative ceiling means the default too (the off switch is
    ``enabled``, never a magic number — the ``bash.memory.limit_mb`` rule).
    """
    enabled: Any = RUNTIME_MEMORY_ENABLED_DEFAULT
    limit: Any = RUNTIME_MEMORY_LIMIT_MB_DEFAULT
    try:
        from local_operator.config import ConfigManager

        config = ConfigManager(config_dir)
        enabled = config.get_nested_value(RUNTIME_MEMORY_ENABLED_PATH, enabled)
        limit = config.get_nested_value(RUNTIME_MEMORY_LIMIT_MB_PATH, limit)
    except Exception:  # noqa: BLE001 — config trouble keeps the guard on its defaults
        logger.debug("runtime memory guard: config unreadable; using defaults", exc_info=True)
    try:
        limit_mb = int(limit)
    except (TypeError, ValueError):
        limit_mb = RUNTIME_MEMORY_LIMIT_MB_DEFAULT
    if limit_mb <= 0:
        limit_mb = RUNTIME_MEMORY_LIMIT_MB_DEFAULT
    if limit_mb < RUNTIME_MEMORY_LIMIT_MB_FLOOR:
        if limit_mb not in _WARNED_LIMITS:
            _WARNED_LIMITS.add(limit_mb)
            logger.warning(
                "runtime memory guard: session.runtime_memory.limit_mb=%s MiB is below the "
                "%s MiB floor, so it is ignored; using the default %s MiB",
                limit_mb,
                RUNTIME_MEMORY_LIMIT_MB_FLOOR,
                RUNTIME_MEMORY_LIMIT_MB_DEFAULT,
            )
        limit_mb = RUNTIME_MEMORY_LIMIT_MB_DEFAULT
    return RuntimeGuardSettings(enabled=enabled is not False, limit_mb=limit_mb)


@dataclass
class _Run:
    """The over-ceiling streak of ONE run (pid + its record's ``started_at``)."""

    count: int = 0
    first_over_at: float = 0.0
    #: The config-root proof, cached for the run: a process's launch environment
    #: cannot change after birth, so one read (a ``ps`` fork on macOS) answers for
    #: the run's life — and is taken while the host is still answering, before the
    #: streak matures into the moment a fork is least likely to return.
    root_ok: bool | None = None


@dataclass
class RuntimeGuardTracker:
    """What the guard remembers between passes: each run's streak, nothing else.

    Lost on a supervisor restart, which costs at most a delayed act (the streak
    restarts) and never a wrong one.
    """

    runs: dict[tuple[int, float], _Run] = field(default_factory=dict)
    #: Runs of a TUI already warned about as over the ceiling (see the module
    #: docstring): one line per run, never one per pass.
    tui_warned: set[tuple[int, float]] = field(default_factory=set)


@dataclass(frozen=True)
class GuardAct:
    """One delivered (or refused) end of a runaway runtime."""

    session_id: str
    pid: int
    footprint_mb: int
    limit_mb: int
    method: str  # "sigterm" | "sigkill" | "gone" | "withheld" | "refused"
    detail: str = ""
    wakes_dormant: int = 0
    monitors_dormant: int = 0
    notice: str = ""  # "spooled" | "not_told"


@dataclass(frozen=True)
class RuntimeGuardReport:
    """One pass: what was read and what (if anything) was ended."""

    state: str  # "off" | "ok" | "watching" | "acted"
    runtimes: int = 0
    unmeasured: int = 0
    over: tuple[tuple[int, int, int], ...] = ()  # (pid, mb, streak)
    acts: tuple[GuardAct, ...] = ()

    def summary(self) -> str:
        """One line for the supervisor log."""
        if self.state == "off":
            return "runtime memory guard: off (session.runtime_memory.enabled=false)"
        parts = [f"runtime memory guard: {self.runtimes} runtime(s), {self.unmeasured} unmeasured"]
        for pid, mb, streak in self.over:
            parts.append(
                f"pid {pid} over the ceiling at {mb} MiB (sample {streak})"
            )  # i18n: ignore supervisor developer log line
        for act in self.acts:
            parts.append(
                f"pid {act.pid} {act.method}{f' ({act.detail})' if act.detail else ''}"
            )  # i18n: ignore supervisor developer log line
        return "; ".join(parts)


def _live_records(config_dir: Path) -> list[Any]:
    """This config root's live runtime records (never stale ones; never reaping)."""
    from local_operator.session.runtime import registry

    return [rec for rec, state in registry.scan(config_dir, reap=False) if state != "stale"]


#: The record kind the guard measures but never ends (module docstring, R5).
OPERATOR_WINDOW_KIND = "tui"


def _guardable(record: Any) -> bool:
    """A record this guard could ever END: a real pid, not this process, not a TUI."""
    pid = getattr(record, "pid", 0)
    if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        return False
    return getattr(record, "kind", "") != OPERATOR_WINDOW_KIND


def holds_supervisor(
    config_dir: Path,
    *,
    records: Callable[[Path], list[Any]] | None = None,
    settings: RuntimeGuardSettings | None = None,
) -> bool:
    """Whether the wake supervisor must stay up to keep this guard running (Q1).

    True while the guard is enabled AND at least one guardable runtime of this
    config root is live. The registry is read first because it is the cheap half
    (a directory scan, no fork) and the common answer on an idle store is "none";
    the settings are read only when there is something to guard. Never raises: a
    failed read answers False, which leaves the supervisor's own retirement rule
    exactly as it was before this guard existed.
    """
    try:
        if not any(_guardable(rec) for rec in (records or _live_records)(config_dir)):
            return False
        return (settings or read_settings(config_dir)).enabled
    except Exception:  # noqa: BLE001 — a hold question must never add a crash path
        logger.debug("runtime memory guard: hold check failed", exc_info=True)
        return False


def _default_footprint_mb(pid: int) -> int | None:
    from local_operator.mobile.resources import direct_footprint_bytes

    value = direct_footprint_bytes(pid)
    if value is None or value <= 0:
        return None
    return int(value // (1024 * 1024))


def birth_matches(record: Any) -> bool | None:
    """Whether the process at ``record.pid`` is the run the record was written for.

    Fork-free on macOS and Linux (``procstate.start_epochs``: ``sysctl`` /
    ``/proc``). The runtime writes ``started_at`` after its process is born, so the
    process now holding the pid must have started no later than that (plus the
    clock's one-second resolution); a pid handed to a stranger after the runtime
    died started AFTER it. ``None`` = cannot tell, and every caller refuses on it.
    """
    from local_operator import procstate

    started_at = getattr(record, "started_at", None)
    if not isinstance(started_at, (int, float)) or started_at <= 0:
        return None
    try:
        epoch = procstate.start_epochs([record.pid]).get(record.pid)
    except Exception:  # noqa: BLE001 — a probe that raises is "cannot tell"
        return None
    if epoch is None:
        return None
    return epoch <= float(started_at) + _BIRTH_SLACK_S


def root_matches(record: Any, config_dir: Path) -> bool | None:
    """Whether the process's own launch environment names ``config_dir``.

    The reclaim sweep's proof (``reclaim.config_root_matches``), with its rule for a
    SIGNALLING caller: only ``True`` licenses an act.
    """
    from local_operator.session.runtime import reclaim

    try:
        env = reclaim.pid_environment(record.pid)
    except Exception:  # noqa: BLE001 — unreadable is "cannot tell"
        return None
    if not env:
        return None
    return reclaim.config_root_matches(env, config_dir)


def notice_text(act: GuardAct) -> str:
    """The row the session shows when it next opens: operator first, model second.

    WRITTEN FOR THE OPERATOR (UX round 1 U3): what happened, what was kept, how
    to resume and which setting governs it, by its /settings label — the sender
    is already named on the row, so the text does not shout it again. The units
    are the setting's own (MiB, U5/N1), so the number told and the number
    editable compare without a conversion. The last sentence is the model's: it
    reads the same row, and without it would repeat the work that grew.

    English by design: this is transcript CONTENT, persisted once and replayed
    to every viewer and to the model, like every other peer row.
    """
    return (
        f"The memory guard ended this session's runtime: it held {act.footprint_mb:,} MiB, "
        f"over the {act.limit_mb:,} MiB ceiling (Settings → Session memory limit → "
        f"Session ceiling (MiB)), for several consecutive checks. Its wakes and monitors "
        f"were made dormant, not deleted, and reopening the session (resume it or open it "
        f"from the session list) re-arms them. If the same load grows again, the guard "
        f"ends it again. For the model: before continuing, find what grew (usually a very "
        f"large transcript or tool result held in memory) instead of repeating the work "
        f"that preceded the stop."
    )


def _spool_notice(config_dir: Path, act: GuardAct) -> str:
    """Queue the notice in the session's inbox — read the next time it opens.

    Never dialled: the runtime has just been ended, so the machine guard's first
    step (``notify_owner_of_kill``'s dial) has nobody to answer it.
    """
    try:
        from local_operator.session.runtime.inbox import InboxLine, append_inbox

        written = append_inbox(
            config_dir / "sessions" / act.session_id,
            InboxLine(
                text=notice_text(act),
                sender={"conversation_name": "runtime memory guard"},
                mode="mailbox",
                written_at=time.time(),
            ),
        )
        return "spooled" if written else "not_told"
    except Exception:  # noqa: BLE001 — a notice never undoes a stop
        logger.warning("runtime memory guard: could not spool the notice", exc_info=True)
        return "not_told"


def _alive_not_zombie(pid: int) -> bool:
    """Alive AND not a zombie: an exited runtime whose parent (a TUI, a test, a
    launcher) has not reaped it yet still answers ``kill(pid, 0)``, and waiting out
    the grace on a corpse would end in a pointless SIGKILL and a wrong receipt."""
    from local_operator import procstate

    return procstate.pid_alive(pid) and not procstate.is_zombie(pid)


def _await_exit(pid: int, timeout_s: float, alive: Callable[[int], bool]) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if not alive(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def end_runaway_runtime(
    config_dir: Path,
    record: Any,
    *,
    footprint_mb: int,
    limit_mb: int,
    verify: Callable[[Any], bool | None] = birth_matches,
    send: Callable[[int, int], None] = os.kill,
    alive: Callable[[int], bool] | None = None,
    term_grace_s: float | None = None,
    kill_confirm_s: float | None = None,
) -> GuardAct:
    """End ONE runaway runtime: marker, dormant schedules, SIGTERM, then SIGKILL.

    ``verify`` is asked before EVERY signal, not once: the SIGTERM grace is long
    enough for the runtime to exit and its pid to be handed on, and SIGKILL to a
    stranger is the one mistake no attribution repairs.

    Synchronous; the supervisor runs it on its pass's worker thread.
    """
    from local_operator import procstate
    from local_operator.session.runtime import control

    is_alive = alive or _alive_not_zombie
    grace = RUNTIME_GUARD_TERM_GRACE_S if term_grace_s is None else term_grace_s
    confirm = RUNTIME_GUARD_KILL_CONFIRM_S if kill_confirm_s is None else kill_confirm_s
    base = {
        "session_id": str(record.session_id),
        "pid": int(record.pid),
        "footprint_mb": footprint_mb,
        "limit_mb": limit_mb,
    }

    if verify(record) is not True:
        return GuardAct(**base, method="refused", detail="birth not proven")

    # The sanction FIRST: the runtime's signal gate refuses a SIGTERM no marker
    # covers, and the marker must be on disk before the signal can arrive.
    control.note_involuntary_stop(record, config_dir, mechanism=MECHANISM, actor=ACTOR)
    # DORMANT (``control.STOP_MARKER``, ``stopped_at``: kept), NOT ``PARK_MARKER``
    # (armed). A park re-arms the schedules so the wake supervisor cold-engages the
    # session on its next due wake — and the incident this guard exists for WAS a
    # wake cold-booting a runaway. A runtime that has held more than any measured
    # session for minutes will do it again on the next engage, so re-arming would
    # turn one stop into a kill loop on the wake's cadence. Dormant keeps every
    # schedule (reopening the session restores them) and leaves the re-arm to a
    # person who has read the notice. The ladder's own constant, never a second
    # spelling: the copy-set guard classifies path-shaped literals per module.
    marker = control.STOP_MARKER
    wakes = monitors = 0
    try:
        wakes = control._mark_wakes_dormant(record, config_dir, marker)
    except Exception:  # noqa: BLE001 — the index is derived; the stop is not
        logger.warning("runtime memory guard: could not stamp wakes dormant", exc_info=True)
    try:
        monitors = control._mark_monitors_dormant(record, config_dir, marker)
    except Exception:  # noqa: BLE001 — same
        logger.warning("runtime memory guard: could not stamp monitors dormant", exc_info=True)
    base.update(wakes_dormant=wakes, monitors_dormant=monitors)

    method = "sigterm"
    try:
        send(record.pid, signal.SIGTERM)
    except ProcessLookupError:
        method = "gone"
    except OSError as exc:
        # Nothing was delivered, so the marker must not stay to narrate this run's
        # next, unrelated death as the guard's act.
        control.withdraw_involuntary_stop(record, config_dir, mechanism=MECHANISM)
        return GuardAct(**base, method="refused", detail=f"SIGTERM failed: {exc}")
    detail = ""
    if method == "sigterm" and not _await_exit(record.pid, grace, is_alive):
        if verify(record) is not True:
            # Re-used, or no longer provable: never SIGKILL on doubt. The SIGTERM
            # did go out, so the run's marker stays.
            method = "withheld"
            detail = "still present after the SIGTERM grace, but its birth could not be re-proved"
        else:
            kill_sig = procstate.hard_kill_signal()
            try:
                if kill_sig is None:
                    procstate.terminate_process_tree(record.pid, force=True)
                else:
                    send(record.pid, kill_sig)
                method = "sigkill"
            except ProcessLookupError:
                method = "gone"
            if not _await_exit(record.pid, confirm, is_alive):
                detail = f"still present {confirm:g} s after SIGKILL"
    try:
        control._recover_record(record, config_dir)
    except Exception:  # noqa: BLE001 — record cleanup is polite, never load-bearing
        pass
    return GuardAct(**base, method=method, detail=detail)


def runtime_memory_pass(
    config_dir: Path,
    tracker: RuntimeGuardTracker,
    *,
    apply: bool = True,
    settings: RuntimeGuardSettings | None = None,
    records: Callable[[Path], list[Any]] | None = None,
    footprint_mb: Callable[[int], int | None] | None = None,
    verify: Callable[[Any], bool | None] | None = None,
    root_proof: Callable[[Any, Path], bool | None] | None = None,
    end: Callable[..., GuardAct] | None = None,
    notify: Callable[[Path, GuardAct], str] | None = None,
    now: float | None = None,
) -> RuntimeGuardReport:
    """Sample every live runtime once; end any whose streak has matured.

    Blocking (a footprint syscall per runtime, and on an act a ``ps`` for the root
    proof plus the signal waits); the supervisor hands it to a worker thread.
    ``apply=False`` samples and advances streaks but never ends anything.
    """
    conf = settings or read_settings(config_dir)
    if not conf.enabled:
        tracker.runs.clear()
        return RuntimeGuardReport(state="off")
    moment = time.time() if now is None else now
    read = footprint_mb or _default_footprint_mb
    live = (records or _live_records)(config_dir)
    seen: set[tuple[int, float]] = set()
    unmeasured = 0
    over: list[tuple[int, int, int]] = []
    acts: list[GuardAct] = []
    for record in live:
        pid = getattr(record, "pid", 0)
        if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
            continue
        key = (pid, float(getattr(record, "started_at", 0.0) or 0.0))
        seen.add(key)
        try:
            mb = read(pid)
        except Exception:  # noqa: BLE001 — a failed read is "unknown"
            mb = None
        if not _guardable(record):
            # The operator's own window: measured, warned once per run, never
            # ended and never streaked (module docstring, R5).
            if mb is not None and mb > conf.limit_mb and key not in tracker.tui_warned:
                tracker.tui_warned.add(key)
                logger.warning(
                    "runtime memory guard: the TUI (pid %s, session %s) holds %s MiB, over "
                    "the %s MiB ceiling; a TUI is the operator's own window and is never "
                    "ended by this guard",
                    pid,
                    getattr(record, "session_id", ""),
                    mb,
                    conf.limit_mb,
                )
            continue
        run = tracker.runs.setdefault(key, _Run())
        if mb is None:
            # Unknown never acts — and never counts: a streak is CONSECUTIVE readings.
            unmeasured += 1
            tracker.runs[key] = _Run(root_ok=run.root_ok)
            continue
        if mb <= conf.limit_mb:
            tracker.runs[key] = _Run(root_ok=run.root_ok)
            continue
        if run.count == 0:
            run.first_over_at = moment
        run.count += 1
        if run.root_ok is None:
            try:
                run.root_ok = (root_proof or root_matches)(record, config_dir)
            except Exception:  # noqa: BLE001
                run.root_ok = None
        over.append((pid, mb, run.count))
        matured = run.count >= conf.samples and moment - run.first_over_at >= conf.sustained_s
        if not matured or not apply:
            continue
        if run.root_ok is not True:
            logger.warning(
                "runtime memory guard: pid %s (session %s) is over the ceiling at %s MiB but "
                "its config root could not be proved to be %s; not acting",
                pid,
                getattr(record, "session_id", ""),
                mb,
                config_dir,
            )
            continue
        act = (end or end_runaway_runtime)(
            config_dir,
            record,
            footprint_mb=mb,
            limit_mb=conf.limit_mb,
            **({"verify": verify} if verify is not None else {}),
        )
        if act.method != "refused":
            act = dataclasses.replace(act, notice=(notify or _spool_notice)(config_dir, act))
        _log_act(act)
        acts.append(act)
        tracker.runs.pop(key, None)
    for stale in set(tracker.runs) - seen:
        tracker.runs.pop(stale, None)
    tracker.tui_warned &= seen
    state = "acted" if acts else ("watching" if over else "ok")
    return RuntimeGuardReport(
        state=state,
        runtimes=len(seen),
        unmeasured=unmeasured,
        over=tuple(over),
        acts=tuple(acts),
    )


def _log_act(act: GuardAct) -> None:
    """ONE grep-able line per act: who, what, how big, against which ceiling."""
    logger.warning(
        "runtime memory guard: session=%s pid=%s method=%s footprint_mib=%s limit_mib=%s "
        "wakes_dormant=%s monitors_dormant=%s notice=%s detail=%r",
        act.session_id,
        act.pid,
        act.method,
        act.footprint_mb,
        act.limit_mb,
        act.wakes_dormant,
        act.monitors_dormant,
        act.notice,
        act.detail,
    )
