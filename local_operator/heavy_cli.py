"""``lop services heavy {wait,run,status}``: the heavy-job slot from a shell.

WHY THIS EXISTS. The machine-wide heavy-job lease (:mod:`local_operator.heavy_lease`,
PR #4104) queues heavy builds and test runs that go through the agent's ``bash``
tool. Everything else had no first-class way to take part: sessions and agents on
mesh nodes coordinated the slot by hand with peer messages ("slot granted" /
"slot released"), or by polling ``pgrep -f`` loops, which match the waiter's own
shell and parents and so never see the slot free (the delivery guide forbids
them). These three verbs give a shell, a script or an operator the same queue:

* ``wait``   blocks until a slot is FREE and returns WITHOUT taking it, for a
  caller that only needs to know the machine is quiet;
* ``run``    takes a slot, runs one command as a child holding it, and frees it
  when the command exits (normally, on SIGINT/SIGTERM/SIGHUP, or killed);
* ``status`` prints the ``lop services status`` row plus one line per holder.

CONSTRAINTS. Locking semantics belong to ``heavy_lease`` and are reused, never
re-implemented: ``run`` acquires through :func:`heavy_lease.acquire` exactly as
the ``bash`` tool does, so a CLI holder and a tool holder queue behind each other
in one FIFO-less slot pool. ``wait`` never probes a flock — a probe would briefly
take the slot and could make a real acquirer's non-blocking attempt fail — it
reads the holder records, the same source ``lop services status`` trusts.

Exit codes: ``wait``/``run`` return :data:`EXIT_TIMED_OUT` (75, ``EX_TEMPFAIL``)
when the queue bound runs out, so a script can tell "never ran" from a command
that ran and failed; ``run`` otherwise returns the child's code (``128 + N`` for a
child ended by signal N, the shell convention). Off (``bash.heavy_jobs.enabled=
false``) or an unsupported platform (no POSIX flock) is not an error: ``wait`` and
``status`` say so and return 0, ``run`` says so and runs the command unqueued.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Sequence, TextIO

from local_operator import heavy_audit, heavy_lease

#: ``EX_TEMPFAIL`` from sysexits.h: "try again later" is exactly what a full queue
#: means, and it is distinct from the 1/2 a failing command usually returns.
EXIT_TIMED_OUT = 75

#: How often a waiter prints its "queued behind …" line. The bash tool streams
#: one a second into a live UI; a terminal or a log file only needs to show it is
#: alive and who it waits for, and one line a second floods a build log.
NOTE_EVERY_SECONDS = 5.0

#: Audit events ``status`` shows: enough to see the last two jobs' take/release
#: pairs, short enough to stay a glance. The whole trail is the jsonl file.
STATUS_EVENTS = 5

#: How often ``wait`` re-reads the holder records. Matches the lease's own poll
#: so a freed slot is noticed as fast as a real acquirer would notice it.
_POLL_SECONDS = 0.25

#: The signals ``run`` forwards to its child's group and treats as "stop now".
#:
#: Built defensively because this module is imported on every platform (the
#: CLI parser reaches it) and Windows has neither ``signal.SIGHUP`` nor
#: ``os.killpg``: a literal ``signal.SIGHUP`` here made the whole module raise
#: AttributeError at import there. Without ``killpg`` there is no group to
#: forward to, so the tuple is empty and ``run`` installs no handlers at all --
#: the unqueued fallback then never reaches ``killpg``, and a console Ctrl-C
#: still reaches the child because it shares the console. On POSIX the tuple
#: is exactly ``(SIGINT, SIGTERM, SIGHUP)``, as before.
_FORWARDED: tuple[int, ...] = (
    tuple(
        sig
        for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None))
        if sig is not None
    )
    if hasattr(os, "killpg")
    else ()
)


def _queue_off_line() -> str | None:
    """The off/unsupported line ``lop services status`` prints, or None when live.

    Reused from :func:`local_operator.services._heavy_job_lines` so the two
    commands can never word the same state differently.
    """
    if heavy_lease.supported() and heavy_lease.configured_settings().enabled:
        return None
    from local_operator.services import _heavy_job_lines

    lines = _heavy_job_lines()
    return lines[0] if lines else None


def _free_slots(directory: Any, slots: int) -> tuple[bool, list[heavy_lease.Holder]]:
    """Whether some slot below ``slots`` has no live holder, plus the holders.

    Only slots below the CURRENT count matter: :func:`heavy_lease.try_acquire`
    only ever tries those, so a holder left on a higher index after auto sizing
    shrank the count does not free anything an acquirer could take.
    """
    holders = heavy_lease.read_holders(directory, slots)
    return len({holder.slot for holder in holders}) < max(1, slots), holders


def wait_command(
    timeout: float | None,
    *,
    out: TextIO | None = None,
    sleep: Callable[[float], Any] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Block until a heavy-job slot is free; never take it.

    ``timeout`` defaults to the configured queue bound
    (``LOCAL_OPERATOR_HEAVY_JOB_WAIT_S``), so ``wait`` gives up exactly when a
    heavy ``bash`` command started now would have been refused.
    """
    out = out if out is not None else sys.stderr
    off = _queue_off_line()
    if off is not None:
        print(off, file=out)
        return 0
    settings = heavy_lease.configured_settings()
    directory = heavy_lease.lease_dir()
    bound = timeout if timeout is not None else settings.wait_seconds
    started = clock()
    last_note: float | None = None
    while True:
        free, holders = _free_slots(directory, settings.slots)
        if free:
            return 0
        waited = clock() - started
        if waited >= bound:
            message = heavy_lease.refusal_message(holders, waited, "lop services heavy wait", bound)
            print(message, file=out)
            return EXIT_TIMED_OUT
        if last_note is None or waited - last_note >= NOTE_EVERY_SECONDS:
            last_note = waited
            print(heavy_lease.queued_line(holders, waited), file=out, flush=True)
        sleep(min(_POLL_SECONDS, max(0.01, bound - waited)))


def _exit_code(returncode: int) -> int:
    """Popen's ``-N`` for "killed by signal N" as the shell's ``128 + N``."""
    return 128 - returncode if returncode < 0 else returncode


def run_command(
    argv: Sequence[str],
    timeout: float | None,
    *,
    out: TextIO | None = None,
) -> int:
    """Run ``argv`` holding a heavy-job slot; release it when the child exits.

    The child leads its OWN session and process group (``start_new_session``,
    the same spawn shape the ``bash`` tool uses) and the lease record names
    that group (``note_pgid``), which is what the lease was designed around: if
    this ``lop`` process is SIGKILLed the child keeps the slot through its
    inherited lock descriptor and ``read_holders`` still reports it by its live
    group, and a forwarded signal reaches the whole build tree (``make -j``
    workers too), not just its first process.

    Its own group also means a terminal's Ctrl-C reaches only this process,
    which forwards it exactly once (a shared group would deliver it twice). The
    cost is that the child has no controlling terminal, so a tty stdin is
    replaced with ``/dev/null`` — heavy jobs are builds and tests, not
    interactive programs, and an inherited tty fd would be one it cannot own.
    A piped stdin is passed through untouched.
    """
    out = out if out is not None else sys.stderr
    if not argv:
        # Refused before queueing: a slot taken for nothing would make every
        # other heavy job on the machine wait for it.
        from local_operator.i18n.keys import cli_services_heavy as copy
        from local_operator.i18n.text import tr

        print(tr("cli.services_heavy")(copy.no_command()), file=out)
        return 2
    received: list[int] = []
    child: list[subprocess.Popen[bytes]] = []

    #: Signals that arrived while no child existed yet, so nothing was forwarded.
    unforwarded: list[int] = []

    def _on_signal(signum: int, _frame: Any) -> None:
        received.append(signum)
        if not child:
            unforwarded.append(signum)
            return
        try:
            os.killpg(child[0].pid, signum)
        except OSError:
            pass  # already gone; the wait below returns on its own

    previous = {sig: signal.signal(sig, _on_signal) for sig in _FORWARDED}
    lease: heavy_lease.Lease | None = None
    exit_code: list[int] = []
    try:
        off = _queue_off_line()
        if off is not None:
            print(off, file=out, flush=True)
        else:
            settings = heavy_lease.configured_settings()
            bound = timeout if timeout is not None else settings.wait_seconds
            last_note: list[float] = []

            def _on_wait(line: str) -> None:
                # acquire() offers a line about once a second; print one per
                # NOTE_EVERY_SECONDS (the first one at once).
                now = time.monotonic()
                if not last_note or now - last_note[0] >= NOTE_EVERY_SECONDS:
                    last_note[:] = [now]
                    print(line, file=out, flush=True)

            result = asyncio.run(
                heavy_lease.acquire(
                    heavy_lease.lease_dir(),
                    settings.slots,
                    wait_seconds=bound,
                    reason="lop services heavy run",
                    session_id=os.environ.get("LOCAL_OPERATOR_SESSION_ID", ""),
                    # The operator typed this line, so there is no session
                    # credential store to redact against; it is shown verbatim
                    # by `lop services status`, like a `ps` listing would.
                    command=shlex.join(argv),
                    on_wait=_on_wait,
                    aborted=lambda: bool(received),
                )
            )
            if result is None:
                return 128 + received[0]
            if isinstance(result, heavy_lease.Refusal):
                print(result.message, file=out)
                return EXIT_TIMED_OUT
            lease = result
        if received:
            return 128 + received[0]
        fd = lease.fileno() if lease is not None else None
        stdin = subprocess.DEVNULL if _stdin_is_tty() else None
        inherited: tuple[int, ...] = (fd,) if fd is not None else ()
        try:
            proc = subprocess.Popen(
                list(argv),
                stdin=stdin,
                pass_fds=inherited,
                start_new_session=True,
            )
        except OSError as exc:
            # The shell's own codes: 127 "not found", 126 "found, not runnable".
            print(f"{argv[0]}: {exc.strerror}", file=out)  # i18n: ignore OS error text
            return 127 if isinstance(exc, FileNotFoundError) else 126
        child.append(proc)
        # A signal handled after Popen returned but before the append above saw
        # no child and was only recorded; forward it now so the child is never
        # left running with the slot held. Python runs handlers between
        # bytecodes on this thread, so each signal lands in exactly one of the
        # two paths. (Blocking signals with pthread_sigmask instead would not
        # work: the child inherits the mask and would ignore what we forward.)
        for signum in unforwarded:
            try:
                os.killpg(proc.pid, signum)
            except OSError:
                pass
        if lease is not None:
            lease.note_pgid(proc.pid)
        # PEP 475: the handler runs inside wait() and wait() resumes, so the
        # child's real exit status is what comes back after a forwarded signal.
        exit_code.append(_exit_code(proc.wait()))
        return exit_code[0]
    finally:
        if lease is not None:
            # The child's code goes into the audit trail's release line; None
            # when the child never ran (spawn failure, signal before Popen).
            lease.release(exit_code=exit_code[0] if exit_code else None)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _stdin_is_tty() -> bool:
    try:
        return os.isatty(sys.stdin.fileno())
    except (AttributeError, OSError, ValueError):
        return False


def _age(seconds: float) -> str:
    """``8s``, ``3m05s``, ``1h02m``: compact enough for a one-line holder row."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def status_command(*, out: TextIO | None = None, now: float | None = None) -> int:
    """The ``lop services status`` heavy-jobs row, then one line per holder."""
    from local_operator.i18n.keys import cli_services_heavy as copy
    from local_operator.i18n.text import tr

    out = out if out is not None else sys.stdout
    off = _queue_off_line()
    if off is not None:
        print(off, file=out)
        return 0
    settings = heavy_lease.configured_settings()
    directory = heavy_lease.lease_dir()
    print(heavy_lease.status_line(directory, settings.slots, now=now), file=out)
    now = time.time() if now is None else now
    text = tr("cli.services_heavy")
    for holder in heavy_lease.read_holders(directory):
        line = copy.holder(
            slot=str(holder.slot),
            holder=holder.who(),
            age=_age(now - holder.started_at),
            command=holder.command or "?",
        )
        print(text(line), file=out)
    # The audit trail's tail (local_operator/heavy_audit.py), so "did those two
    # jobs serialize?" is answerable here without opening the file. The holder
    # line above already gives pid/session/age; the matching ``take`` line
    # below adds its label and absolute start time.
    #
    # No waiter count, deliberately: waiters poll a non-blocking flock and
    # leave no trace until they win or time out, so counting them would need a
    # new per-waiter registry (writes on every queue entry, plus stale cleanup
    # for killed waiters) — not cheap, and not needed to prove serialization.
    path = heavy_audit.audit_path(directory)
    events = heavy_audit.tail(directory, STATUS_EVENTS)
    if not events:
        print(text(copy.no_events(path=str(path))), file=out)
        return 0
    print(text(copy.events_header(path=str(path))), file=out)
    for entry in events:
        line = copy.event_line(
            time=heavy_audit.iso_time(entry.get("ts")), record=heavy_audit.describe(entry)
        )
        print(text(line), file=out)
    return 0
