"""Every non-runtime service on this machine, moved onto the build ``current`` names.

WHAT THIS IS FOR
----------------
``lop update`` has always replaced the install and left the processes alone. The
supervised ones (mobile, browser bridge, tunnel, wakes) were repaired afterwards
by :func:`local_operator.update.refresh_daemons_after_upgrade`, but a ``lop
serve`` daemon — the process a desktop app actually talks to — kept serving the
build it had loaded, indefinitely and by design: :mod:`local_operator.server
.retire` refuses to exit on a build change because a marker proves neither a
ready successor nor that scheduler-owned work can stop.

The result was a machine whose INSTALL was current and whose BACKEND was not, and
an app whose update button reported exactly that honestly and then had nothing to
offer: it had no way to move a server it did not start, and no way to make the
install it *did* move become the one being served. This module is the missing
step, and :mod:`local_operator.server.reload` is the mechanism it drives — the
daemon replaces its own image while keeping its pid and its listening socket, so
nothing has to own, kill or respawn it.

WHAT IS AND IS NOT A SERVICE
----------------------------
A SERVICE is a long-lived non-conversational process: a ``serve`` daemon, the
mobile daemon, the browser bridge, the tunnel, the wakes supervisor. A RUNTIME
is a conversation (``session/runtime/process.py``), and this module never touches
one: runtimes are detached, own their transcript leases, and converge on the new
build by themselves when they next go idle (``buildwatch`` + ``launch
._spawn_interpreter``). Restarting a runtime would drop the turn it is running;
restarting the daemons around it does not.

ORDER IS LOAD-BEARING, AND IT IS THE SAME ONE ``lop update`` USES
-----------------------------------------------------------------
The supervised daemons are repaired FIRST and by a CHILD running the NEW build,
because the repair RENDERS plists: run in-process from a superseded interpreter
it would write the previous build's plist shape and report success (see
``update._post_upgrade_invocation``). The ``serve`` reloads follow, and they are
requests rather than actions — a daemon that is mid-spawn refuses and says so.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from local_operator.server import registry as serve_registry
from local_operator.server import reload as serve_reload

logger = logging.getLogger("local_operator.services")

#: How long one daemon has to come back on the new build before it is reported.
#:
#: Sized against the phases it covers, MEASURED rather than assumed (serve-reload
#: review round 2, R2-2: the first version was 30 s while the daemon's own budgets
#: summed to 40 s, so the caller could give up while the daemon was still
#: working and report a failure for a reload that then succeeded):
#:
#:   drain   up to ``reload.DRAIN_BUDGET_S``  = 10 s   (asked TWICE — see below)
#:   smoke   up to ``reload.SMOKE_TIMEOUT_S`` = 10 s
#:   start   interpreter + record publish     = ~1.5-2 s on the reporting host
#:
#: THE DRAIN IS PAID TWICE (serve-reload review round 4, R4-2: this and the reload's own
#: comment both described one drain, which invites a future smoke bump that
#: silently overruns the caller — a reload is drained, smoke-checked, and drained
#: AGAIN, because the smoke is off the loop and a spawn can arrive during it).
#: Worst honest sum ≈ 32 s against 45 s. A change to either budget has to re-do
#: that arithmetic rather than assume this number still covers it.
#:
#: 45 s is a margin over that sum rather than a restatement of it, and it is the
#: CALLER's patience, not a phase's deadline: an expiry is a warning on a
#: successful update, never a kill.
RELOAD_WAIT_S = 45.0

#: The poll period while waiting for a daemon's replacement to publish.
RELOAD_POLL_S = 0.2

#: How long the identity probe before a signal is given.
#:
#: One loopback round trip to a daemon that is answering right now; the timeout is
#: the same order as ``update._mobile_healthz_answers``'s, for the same reason —
#: the answer is either immediate or it is not an answer.
HEALTH_TIMEOUT_S = 1.0


def _answers_as_record(record: Any) -> str | None:
    """Why this record must NOT be trusted, or ``None`` when it is proven.

    THE RECORD IS NOT PROOF OF WHO IS LISTENING (serve-reload review round 1, R1-4). It is a
    file whose name is a pid, and a pid is recycled: review constructed a live
    ``sleep`` named by a hand-written live, ``reloadable: true`` record, and
    ``reload_serve_daemons`` signalled it — rc ``-30``, a process killed by a
    command whose entire purpose is to NOT kill anything. The ``/health``
    ``instance_id`` is the one value that ties the address to the process, which
    is the check daemon discovery already performs for the same reason
    (``server/registry``'s module docstring: "a 200 alone is not
    identification").

    FAIL-CLOSED, and asymmetrically so against the rule that governs the
    DAEMON's own probes: there, an unreadable probe means "stay" because leaving
    is the irreversible act. Here the irreversible act is SIGNALLING somebody
    else's process, so an unreadable answer means "do not signal" — the daemon
    keeps serving and the operator is told the address did not identify itself.
    """
    import urllib.error
    import urllib.request

    # AN IPv6 LITERAL MUST BE BRACKETED, and the first version of this line was
    # not: `f"http://{[record.host]}:…"` renders the list `['::1']`, so every
    # probe of a v6 daemon asked a URL that cannot parse and the daemon was
    # reported as "did not identify itself" while answering perfectly. Found by
    # the test the review asked for (serve-reload review round 2, R2-6), which is the
    # whole reason it was asked for.
    authority = f"[{record.host}]" if ":" in record.host else record.host
    url = f"http://{authority}:{record.port}/health"
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT_S) as response:
            import json

            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return f"{record.host}:{record.port} did not identify itself ({exc})"
    if not isinstance(payload, dict):  # pragma: no cover - a non-object body
        return f"{record.host}:{record.port} answered with something that is not a health report"
    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    served = result.get("instance_id") if isinstance(result, dict) else None
    if not served or served != record.instance_id:
        return (
            f"{record.host}:{record.port} is answering as {served or 'no instance'}, "
            f"not the {record.instance_id[:8]}… this record names"
        )
    return None


@dataclass(frozen=True)
class ServiceRefresh:
    """One service's outcome, in the shape the CLI already prints.

    Deliberately the same three buckets ``update.DaemonRefresh`` uses — ``lines``
    for what happened, ``warnings`` for what did not and what to do about it —
    because the two lists are printed by one printer and a second vocabulary
    would make an operator read two formats for one command's output.
    """

    name: str
    lines: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ServeOutcome:
    """Internal: what happened to one daemon, before it is worded for a reader."""

    record: Any
    moved: bool = False
    skipped: bool = False
    warning: str = ""
    state: dict[str, Any] = field(default_factory=dict)


def live_serve_daemons() -> list[Any]:
    """Every LIVE ``serve`` record on this machine, in pid order.

    ``live`` only — the shared registry's classification, which is the same one
    every other reader acts on: a record whose pid is gone has already been reaped
    by ``scan``, and a ``wedged`` daemon (pid alive, heartbeat stopped) is not one
    to signal, because a stuck process is not a process that will handle SIGUSR1.

    Sorted by pid so the output is stable run to run: an operator comparing two
    invocations should not have to diff a shuffled list.
    """
    found = []
    for record, state in serve_registry.scan():
        if state == "live":
            found.append(record)
    return sorted(found, key=lambda record: record.pid)


def _current_stamp() -> Any:
    """The build the pointer names, or ``None`` when it cannot be read.

    The SUBJECT of the whole module, and read from the pointer rather than from
    this process (``update.disk_build()``'s argument) on purpose: the caller may
    itself be a superseded build — the desktop app runs ``lop update`` through the
    stable shim, and the shim's interpreter is whatever ``current`` named when the
    process started. Asking ``sys.prefix`` would answer "every daemon is already
    current" for the one process that just made them all stale.
    """
    from local_operator import update

    try:
        return update.disk_build()
    except Exception:  # noqa: BLE001 — an unreadable install is "no answer"
        logger.debug("install stamp unreadable", exc_info=True)
        return None


def _serves_current_build(record: Any, stamp: Any) -> bool:
    """Is this daemon already running the build the pointer names?

    Both halves, because this machine's common drift is a SAME-VERSION rebuild
    (``lop-update`` rebuilds the checkout, so ``pyproject.toml`` names the last
    release and only ``source_ref`` moves) — a version-only test would call a
    stale daemon current on exactly the host this is for.
    """
    if stamp is None:
        return False
    return (record.version, record.source_ref) == (stamp.version, stamp.source_ref)


def reload_serve_daemons(
    *,
    wait_s: float = RELOAD_WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
    scan: Callable[[], list[Any]] = live_serve_daemons,
    kill: Callable[[int, int], None] = os.kill,
    probe: Callable[[Any], str | None] = _answers_as_record,
    monotonic: Callable[[], float] = time.monotonic,
) -> list[ServiceRefresh]:
    """Ask every stale ``serve`` daemon to move itself onto the current build.

    ONE REQUEST PER DAEMON, THEN ONE WAIT — not request/wait/request/wait. The
    daemons move in parallel by construction (each is an independent process), so
    serialising the waits would add their drains together: three busy daemons
    would cost three times the budget to discover something that was true two
    waits ago. The whole fleet gets one deadline instead.

    NEVER RAISES. A service that could not be moved is a warning on a successful
    update, exactly like ``update.refresh_daemons_after_upgrade``'s contract, and
    for the same reason: the install has already been replaced, and a failed
    nudge must not be reported as a failed update.
    """
    # SCAN FIRST, so "nothing to say" stays silent. A machine with no serve
    # daemons running is not a machine with a broken install, and a warning on
    # every `lop update` on such a host would be noise that trains its reader to
    # ignore the line that matters.
    daemons = scan()
    if not daemons:
        return []
    # NO EARLY RETURN FOR A PLATFORM WITHOUT SIGUSR1 (serve-reload review round 8,
    # R8-1). The first version refused the whole fleet up front, which was redundant
    # for signalling — every daemon reports `reloadable: false` there, so the loop
    # below takes its "cannot move itself" branch for each one anyway — and LOSSY for
    # the report, because it replaced those per-daemon lines with a single sentence.
    # The operator on such a platform now sees exactly which daemons are running and
    # what each needs, which is the same information the POSIX path gives.
    stamp = _current_stamp()
    if stamp is None:
        # NO STAMP MEANS NO PREDICATE (serve-reload review round 2, R2-1). ``_serves_current_build``
        # answers False when the stamp is unreadable, so without this guard EVERY
        # daemon looks stale and the whole fleet gets signalled by a caller that
        # has no build to move anything onto. That is not hypothetical: it was
        # reachable the moment the services stage started running on ``lop
        # update``'s "nothing to install" path, and a source checkout is exactly
        # such a caller — its ``disk_build()`` is None, so a developer running
        # ``lop update`` in their worktree would have signalled this machine's serve
        # daemon. Demonstrated end to end in review with a fabricated-root daemon:
        # stale → signalled → really reloaded. (serve-reload review round 4, R4-4: the first two
        # versions of this sentence listed the mobile daemon, the browser bridge and
        # the tunnel as well, and both overclaimed — see below.)
        #
        # THE BLAST RADIUS IS THE SERVES, not the whole fleet (serve-reload review
        # round 3, R3-4: an earlier version of this comment listed the mobile daemon, the
        # browser bridge and the tunnel, and overclaimed — `_repair_refusal`
        # already refuses an editable caller inside the plist refresh child).
        #
        # AND NOT THE MOBILE DAEMON EITHER, except on one path (serve-reload review
        # round 4, R4-4): a checkout's `lop update --no-services` does still reach the mobile
        # bounce, which has no guard of this kind and never did. That is
        # pre-existing behaviour rather than anything this change added —
        # `--no-services` is deliberately the pre-change path — so it is named here
        # and left alone rather than quietly changed under a flag whose whole job is
        # to reproduce the old behaviour.
        #
        # The invariant this restores: "stale" is a comparison, and a comparison
        # against an absent right-hand side is not a verdict. Fail-closed, and
        # said out loud — an operator whose pointer is unreadable AND who has
        # daemons running has a real problem and should hear about it.
        return [
            ServiceRefresh(
                "serve daemons",
                warnings=(
                    "warning: this install has no build the pointer can name, so no "
                    "service can be moved onto it and NOTHING was signalled. Run "
                    "`lop services status` to see what is running, and `lop update` "
                    "from the install that owns them.",
                ),
            )
        ]
    refreshes: list[ServiceRefresh] = []
    pending: list[Any] = []

    # The SCAN IS DONE ONCE and reused: scanning twice would let a daemon arrive
    # or depart between the guard and the loop, which is exactly the kind of gap
    # the guard exists to close.
    for record in daemons:
        name = _serve_name(record)
        if _serves_current_build(record, stamp):
            refreshes.append(ServiceRefresh(name, lines=(f"{name} is already on {_label(stamp)}",)))
            continue
        if not getattr(record, "reloadable", False):
            # The capability is absent for every daemon built before this
            # existed, and for a --reload child whose port belongs to uvicorn's
            # supervisor. Sending the signal anyway is NOT harmless: SIGUSR1's
            # default disposition is to terminate, so signalling a daemon that
            # never installed the handler kills the backend this command exists
            # to bring along. Report it by name instead.
            refreshes.append(
                ServiceRefresh(
                    name,
                    warnings=(
                        f"warning: {name} is on {record.version or 'an older build'} and "
                        "cannot move itself ("
                        + (
                            "this platform has no SIGUSR1"
                            if serve_reload.RELOAD_SIGNAL is None
                            else "it predates in-place reload"
                        )
                        + "); restart that "
                        "server by hand, or stop and start whatever supervises it",
                    ),
                )
            )
            continue
        try:
            # PROVE THE PROCESS BEFORE TOUCHING IT (serve-reload review round 1, R1-4). A pid
            # is not an identity, and the request is a signal whose default
            # disposition is death.
            mismatch = probe(record)
            if mismatch is not None:
                refreshes.append(
                    ServiceRefresh(
                        name,
                        warnings=(
                            f"warning: {name} was not asked to reload — {mismatch}. "
                            "Nothing was signalled.",
                        ),
                    )
                )
                continue
            # The family: a mismatch here is silent, which is why it has its own test.
            kill(record.pid, serve_reload.RELOAD_SIGNAL)  # type: ignore[arg-type]  # guarded above
        except OSError as exc:
            refreshes.append(
                ServiceRefresh(
                    name,
                    warnings=(f"warning: {name} could not be asked to reload: {exc}",),
                )
            )
            continue
        pending.append(record)
        refreshes.append(
            ServiceRefresh(
                name,
                lines=(
                    f"{name} was asked to move onto {_label(stamp)}; "
                    "its conversations keep running",
                ),
            )
        )

    if pending:
        refreshes.extend(
            _await_relocations(pending, wait_s=wait_s, sleep=sleep, scan=scan, monotonic=monotonic)
        )
    return refreshes


def _await_relocations(
    pending: Sequence[Any],
    *,
    wait_s: float,
    sleep: Callable[[float], None],
    scan: Callable[[], list[Any]],
    monotonic: Callable[[], float],
) -> list[ServiceRefresh]:
    """Wait for the asked daemons to republish under a NEW ``instance_id``.

    ``instance_id`` is the proof and the version is not: it is minted once per
    process (``server/app.py``), so a changed one can only mean the process that
    published before has been replaced — whereas the version is equal across a
    same-version rebuild, which is this host's ordinary case. Comparing versions
    here would report "did not move" for a reload that worked.

    A daemon that vanishes while we wait is NOT reported as a failure of the
    request: a pid that is gone may have exited for its own reasons, and the
    honest report is that it stopped answering — the reader can see the record is
    gone too.
    """
    before = {record.pid: record.instance_id for record in pending}
    deadline = monotonic() + wait_s
    settled: dict[int, Any] = {}
    while True:
        for record in scan():
            if record.pid in before and record.pid not in settled:
                if record.instance_id != before[record.pid]:
                    settled[record.pid] = record
        if len(settled) == len(before):
            break
        if monotonic() >= deadline:
            break
        sleep(RELOAD_POLL_S)

    out: list[ServiceRefresh] = []
    for record in pending:
        name = _serve_name(record)
        moved = settled.get(record.pid)
        if moved is not None:
            out.append(
                ServiceRefresh(
                    name,
                    lines=(f"{name} is now serving {moved.version or 'the current build'}",),
                )
            )
        else:
            out.append(
                ServiceRefresh(
                    name,
                    warnings=(
                        f"warning: {name} did not come back on the current build within "
                        f"{wait_s:.0f}s; it is still serving the build it loaded and the "
                        "next `lop services restart` will try again",
                    ),
                )
            )
    return out


def restart_services(*, wait_s: float = RELOAD_WAIT_S) -> list[ServiceRefresh]:
    """Bring EVERY non-runtime service onto the current build.

    THE ONE OWNER of that sentence, so ``lop update`` and ``lop services
    restart`` cannot drift into two different ideas of what "everything" means.

    The supervised daemons go first and are repaired by a child of the NEW build
    (``update.refresh_daemons_after_upgrade``, which owns that rule and its
    reasons); the ``serve`` reloads follow. A run that moved nothing is a
    success with nothing to say — the list comes back with skips in it rather
    than with an error.
    """
    from local_operator import update

    refreshes: list[ServiceRefresh] = []
    try:
        for refresh in update.refresh_daemons_after_upgrade():
            refreshes.append(
                ServiceRefresh(
                    refresh.name, lines=tuple(refresh.lines), warnings=tuple(refresh.warnings)
                )
            )
    except Exception:  # noqa: BLE001 — a failed repair must not fail the command
        logger.warning("supervised daemon refresh failed", exc_info=True)
        refreshes.append(
            ServiceRefresh(
                "service daemons",
                warnings=(
                    "warning: the installed daemons could not be refreshed; run "
                    "`lop browser restart`, `lop mobile restart` and `lop tunnel restart`",
                ),
            )
        )
    refreshes.extend(reload_serve_daemons(wait_s=wait_s))
    return refreshes


def status_lines() -> list[str]:
    """One line per non-runtime service, with the build it is serving.

    The reader this exists for is an operator who has just been told "the server
    is on an older build than the install" by a desktop app and wants to see
    WHICH process said so. Nothing here acts, so it is safe to run at any time.
    """
    stamp = _current_stamp()
    lines: list[str] = []
    daemons = live_serve_daemons()
    if not daemons:
        lines.append("serve daemons: none running")
    for record in daemons:
        current = _serves_current_build(record, stamp)
        verdict = "current" if current else "STALE"
        caps = "reloadable" if getattr(record, "reloadable", False) else "not reloadable"
        lines.append(
            f"serve daemon pid {record.pid} on {record.host}:{record.port} — "
            f"{verdict} ({record.version or 'unknown build'}, {caps})"
        )
    for path in _supervised_daemon_plists():
        lines.append(f"supervised daemon: {path.stem}")
    return lines


def _supervised_daemon_plists() -> list[Any]:
    """The installed LaunchAgents this product owns, via ``update``'s own probe.

    Asked of ``update`` rather than re-derived so the two cannot disagree about
    which labels are ours: ``_installed_daemon_plists`` is also what decides
    whether the repair is worth a child process at all.
    """
    from local_operator import update

    return list(update._installed_daemon_plists())


def _serve_name(record: Any) -> str:
    """``serve pid <pid>``, the name every line about one daemon is prefixed with."""
    return f"serve pid {record.pid}"


def _label(stamp: Any) -> str:
    """``BuildStamp.label()`` when there is a stamp, else a phrase that says so."""
    return stamp.label() if stamp is not None else "the current build"


def print_refreshes(refreshes: Sequence[ServiceRefresh]) -> None:
    """Print a refresh list the way ``lop update`` already prints its daemons.

    ``lines`` first and in the order the callers built them, then ``warnings`` —
    the same discipline ``update._print_daemon_refreshes`` states, and for the
    same reason: a warning is what the reader has to act on, so it goes where the
    eye lands last rather than being interleaved with success.
    """
    for refresh in refreshes:
        for line in refresh.lines:
            print(line)
    for refresh in refreshes:
        for warning in refresh.warnings:
            print(warning, file=sys.stderr)
