"""A detached session process: ``python -m local_operator.session.runtime.process``.

The daemon spawns one of these per phone-started session instead of hosting
the session in-process, for one reason: **lifetime**. The daemon is
supervised state — launchd restarts it on crash and on ``lop mobile
restart`` — and a session living inside it would die with every restart,
taking an in-flight turn with it. A child with its own pid has terminal
session lifetime: the daemon going away costs the phone its view, never the
session its work.

The child builds a session with the CLI's composition root, wraps it in the
owned-session handle (approval/ask gates resolved from the phone), registers
it through the normal record + control socket path, and idles until a signal
arrives or the residency predicate (:func:`_should_exit`) holds for one
sustained drain. Environment variables are the
spawn contract (``LOP_MOBILE_CHILD_CWD``, ``_PROVIDER``, ``_MODEL``) — argv
would be ps-readable.

**Residency (design §6.1).** The runtime is a unit of WORK, not of state; it
runs its trajectory to completion and exits when idle, so a closed terminal
costs nothing and a wake fires in a fresh process later. It stays resident
while any of three things holds — see :func:`_should_exit` for each term and
the reasoning behind it.

**Self-refresh (design-runtime-autorefresh §3.2).** Independently of the
quiet exit, an idle runtime whose install on disk has moved under it
(``lop-update`` ran) ANNOUNCES ``retiring`` to its viewers and exits, so the
next engage runs the new build. An attached viewer does not hold this — it
re-engages a successor itself — which is what keeps a five-hour-stale
runtime from staying resident because someone was looking at it. See
:func:`_should_refresh` and :func:`_refresh_for`.

This was ``mobile/child.py``. Only the phone spawns one today, but nothing in
it is phone-specific: it is the generic "a session running with no interface
owner" process, which is what later work needs for wakes and background
automations. The ``LOP_MOBILE_CHILD_*`` environment names keep their spelling
for the same reason ``RUN_DIRNAME`` does — they are a cross-process contract,
and during an upgrade a daemon of one version spawns a child of another.
``local_operator.mobile.child`` still resolves and still runs this ``main``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from local_operator import buildwatch as _buildwatch

if TYPE_CHECKING:
    from local_operator.update import BuildStamp

logger = logging.getLogger(__name__)

#: Fine polling makes the 3-second drain predictable while remaining cheap for
#: one event loop. Viewer TRAFFIC is not an input to this loop (a chatty
#: viewer does not reset the drain); viewer PRESENCE is, through term 3 of
#: the predicate.
REAP_CHECK_S = 0.25

#: Idle runtimes are disposable once their durable session state is
#: quiescent. This is a drain for newly arriving work, not a reconnect grace.
DEFAULT_GRACE_S = 3.0

# The build-watch timings, the env readers that shorten them for the e2e stage,
# and the changed-and-settled comparison `_build_changed` live in ONE module that
# the ``serve`` daemon (`server/retire.py`) imports too — a second copy of a
# settle rule would be free to disagree with this one about the same torn
# install (see that module's docstring).
#
# Bound here under the private names this module has always published
# (`_build_changed`, `_build_settle_seconds`, …) because `session.runtime.server`
# and this module's own tests reach for them HERE; a definition moving does not
# move its name. Assignments rather than `import ... as`, because they are
# re-exports of a module whose own names are public: the alias is this module's
# historical spelling, not a second definition.
BUILD_CHECK_S = _buildwatch.BUILD_CHECK_S
BUILD_SETTLE_S = _buildwatch.BUILD_SETTLE_S
BUILD_STAGGER_S = _buildwatch.BUILD_STAGGER_S
# The warm-window term of the residency predicate moved there too, and for a
# sharper reason than tidiness: this module is RUN as ``__main__``, so an import
# of it is answered from DISK rather than from ``sys.modules`` — which is the
# one moment the file may already be gone. ``may_refresh`` reached the helper
# that way and raised ``ImportError`` exactly when the drain needed it, so the
# exit was unreachable (QA round 1, Q-1). See that module's docstring.
WARM_WINDOW_S = _buildwatch.WARM_WINDOW_S
_wake_within_window = _buildwatch.wake_within_window
_build_changed = _buildwatch.build_changed
_build_pair = _buildwatch.build_pair
_build_prefix = _buildwatch.build_prefix
_build_settle_seconds = _buildwatch.build_settle_seconds
_build_stagger_seconds = _buildwatch.build_stagger_seconds

#: A runtime must not refuse the same newer build FOREVER. ``_should_refresh``
#: only acts on an instant where nothing would be lost, so a session busy for
#: hours never reaches one — and while it is busy ``lop-update`` replaces the
#: install tree WHOLESALE (``uv tool install --force`` writes all 929 files
#: with the new install time; nothing older survives). Measured on this host:
#: eight runtimes still executing 0.54.33 and two on 0.54.35 while the install
#: had moved to 0.54.39 across six generations, with no retire line for the
#: last replacement because those runtimes were never idle.
#:
#: So a declined newer stamp is COUNTED, and the runtime that keeps declining
#: it becomes hard-stale: it stops admitting new work and leaves at the first
#: instant its own work is done (:class:`_BuildWatch` / :func:`_drain_for`).
#: Nothing in flight is aborted, and the bound is keyed on the BUILD, never on
#: how long a turn has run: a turn or a wake that lands mid-count is
#: unaffected.
#:
#: Both bounds are deliberate and both are needed. The count bounds the common
#: shape (one install lands under a runtime that stays busy); measured against
#: the observation cadence rather than in the abstract, three checks is ~15 s
#: of ``BUILD_CHECK_S``, so a runtime that is STILL busy three checks after an
#: install has landed commits to leaving then — eager on purpose, because the
#: alternative is paying for turns executed against a build whose files are
#: gone. The clock is the belt for the shape the count cannot see: a stamp that
#: keeps MOVING under the counter (six generations in four hours), where every
#: change would otherwise reset the count back to one.
#:
#: The bound is the one piece of the build watch the ``serve`` daemon does NOT
#: share, and the asymmetry is argued rather than incidental. The daemon
#: (``server/retire.py``) refuses new work and leaves when its own in-flight
#: terms clear, unbounded — each of its terms names live work, and it has no
#: viewer to re-engage a successor. A session runtime can be busy for hours
#: with nothing but a turn, and the process that would run its next engage is
#: waiting on THIS one leaving; that is what needs a bound.
BUILD_MAX_STALE_GENERATIONS = 3
BUILD_MAX_STALENESS_S = 30 * 60.0


def _grace_seconds() -> float:
    raw = os.environ.get("LOP_SESSION_GRACE_S", "")
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_GRACE_S
    return value if value > 0 else DEFAULT_GRACE_S


def _idle_for_refresh(handle: object) -> bool:
    """Is this runtime free to act on a newer build RIGHT NOW?

    The idle GATE alone, split out of :func:`_should_refresh` because the
    staleness bound has to tell "nothing newer is on disk" apart from
    "something newer is, and this runtime declined it" — and a busy runtime and
    a broken probe answer the same way to the gate while meaning very
    different things. ``ServingSessionHandle.may_refresh`` is the one predicate
    for it, shared with the viewer-driven ``refresh_if_idle`` op so both sides
    agree on what idle means. A handle without the probe (an older host, a
    reduced test handle) is never idle for this purpose: unknown state is not
    an invitation to leave. Probe failures read the same way — which is also
    why the drain path re-checks through :func:`_idle_for_refresh` rather than
    trusting an earlier sample.
    """
    may_refresh = getattr(handle, "may_refresh", None)
    if not callable(may_refresh):
        return False
    try:
        return not may_refresh()
    except Exception:  # noqa: BLE001 — uncertainty keeps the runtime
        logger.debug("refresh predicate failed; keeping runtime", exc_info=True)
        return False


def _should_refresh(handle: object, boot: "BuildStamp | None") -> "BuildStamp | None":
    """Retire so the next engage spawns from the build now on disk?

    Returns the NEW stamp when yes, ``None`` when no. Not a fourth term of
    :func:`_should_exit`: that predicate answers "may I exit *quietly*", and a
    refresh must ANNOUNCE (the ``retiring`` frame) so a viewer re-engages
    rather than reading the exit as owner death. Only when the runtime is
    doing NOTHING it would lose — ``ServingSessionHandle.may_refresh`` is the
    one predicate for that, shared with the viewer-driven ``refresh_if_idle``
    op so both sides agree what idle means. An attached viewer does NOT hold
    (the operator's rule; the viewer re-engages on its own), and neither does
    pristineness — a pristine stale runtime is the cheapest refresh there is.
    A handle without the probe (an older host, a reduced test handle) never
    refreshes: unknown state is not an invitation to exit.

    This is the SOFT rung, and it keeps its refusal: a busy runtime is left
    exactly as it is, because a refresh can wait. What no longer waits forever
    is the whole question — see :data:`BUILD_MAX_STALE_GENERATIONS` and
    :class:`_BuildWatch` for the bound on how long the refusal may repeat.
    """
    if not _idle_for_refresh(handle):
        return None
    return _build_changed(boot)


@dataclass(frozen=True, slots=True)
class _BuildPoll:
    """One build check's verdict, so the reaper acts on ONE read of the disk.

    ``newer`` is the settled build on disk when it differs from what this
    process loaded, else ``None``. ``idle`` is whether the runtime is free to
    act on it right now. ``hard_stale`` is the bound having tripped — a newer
    build this runtime has repeatedly declined. ``files_gone`` is the tree the
    process loaded having disappeared from disk, sustained past the install
    settle (which is a different fact from "a newer build is available", and a
    stronger one: a runtime whose tree is gone cannot import anything at all).
    """

    newer: "BuildStamp | None" = None
    idle: bool = False
    declines: int = 0
    hard_stale: bool = False
    files_gone: bool = False

    def refreshable(self) -> bool:
        """Does the ordinary idle refresh own this poll?"""
        return self.idle and self.newer is not None

    def draining(self) -> bool:
        """Must the runtime commit to leaving even though it is not idle?"""
        return self.hard_stale or self.files_gone


class _BuildWatch:
    """How long this runtime has been serving a build that is no longer the
    one on disk, and whether the tree it loaded still exists.

    Two independent answers, because they cover different shapes:

    * **The bound.** ``poll`` counts each observation of a newer settled stamp
      that the idle gate declined (see :data:`BUILD_MAX_STALE_GENERATIONS`),
      and trips :attr:`_BuildPoll.hard_stale` once that count or the age of the
      decline crosses the limit. The soft rung above still decides first: an
      idle runtime retires on its very first observation, which is why the
      count only ever grows for a runtime with work in flight.
    * **The files-gone probe.** Three stats of the paths this process actually
      loaded. A tree that is GONE answers no build stamp at all, so no amount
      of stamp comparison can see it: ``installed_build`` degrades to an empty
      version and ``build_marker_age_s`` to ``None``, and ``_build_changed``
      then answers "nothing to do" forever while the process serves a tree that
      is no longer there. Never mtime — see :meth:`_files_gone`.

    Both read the disk through :mod:`local_operator.buildwatch`, the one
    definition of the build-watch timings and the changed-and-settled rule
    shared with the ``serve`` daemon (``server/retire.py``): a second copy of
    the settle window would be free to disagree with this one about the same
    torn install.

    State is per-reaper-run (one instance, one process) and deliberately thin:
    the counters exist to bound a refusal, not to record history.
    """

    def __init__(
        self,
        boot: "BuildStamp | None",
        *,
        paths: "tuple[Path, ...] | None" = None,
        armed: bool | None = None,
    ) -> None:
        self.boot = boot
        #: The probe's two inputs are injectable so a test can STATE the tree
        #: it means (a real install is not arrangeable in a unit test) rather
        #: than arrange one. ``None`` reads them from the running process, which
        #: is what every production caller does.
        self._paths = _loaded_tree_paths() if paths is None else tuple(paths)
        #: The arming verdict is NOT read here: ``_tree_is_replaceable`` pulls
        #: in the update module, and the reaper's start path is timed by the
        #: idle-exit tests (a one-off import there showed up as an extra ~80 ms
        #: on the first run). It is read on the first probe, which is either the
        #: first build check or never — a runtime that exits quietly never pays
        #: it at all.
        self._armed = armed
        self._arm_read = armed is not None
        self._declined: "BuildStamp | None" = None
        self._declines = 0
        self._stale_since: float | None = None
        self._missing_since: float | None = None

    def poll(self, handle: object, *, now: float | None = None) -> _BuildPoll:
        """One check of the disk against what this process loaded.

        NOTHING HERE EVER CLEARS THE BOUND, and that is deliberate (review
        round 1, MINOR 1). An observation that cannot establish the build's
        identity — the ~2 checks per install where ``build_changed`` answers
        ``None`` because ``.lop-source`` is younger than the settle — used to
        reset both counters. That made the belt the age of the last
        uninterrupted run of declines rather than the age of the first one, and
        because an install is exactly what produces those ``None``
        observations, the event the belt bounds was also the event that reset
        it: the "stamp that keeps moving" shape was left to the per-stamp count
        alone. Reproduced on the previous head — 40 checks, a fresh stamp
        every other check, one decline each, ``hard_stale`` False forever.

        There is nothing to reset TO. The count is per stamp (a different newer
        build restarts it in :meth:`_count_decline`), and the belt is monotone
        for the life of the process, which is what "must not refuse the same
        newer build forever" means. Neither is consulted unless a settled newer
        stamp is on disk AND this runtime is not free to act on it, so a
        runtime that is on the current build never trips either — and the one
        way out of a trip is the process leaving.
        """
        at = time.monotonic() if now is None else now
        idle = _idle_for_refresh(handle)
        newer = _build_changed(self.boot)
        files_gone = self._files_gone(at)
        if newer is None or idle:
            # Nothing to be hard-stale about: no settled newer build, or one the
            # soft rung will act on at its next check. The probe still stands on
            # its own — a tree that is gone is exactly the case that answers no
            # stamp at all.
            return _BuildPoll(newer=newer, idle=idle, files_gone=files_gone)
        return _BuildPoll(
            newer=newer,
            idle=False,
            declines=self._count_decline(newer, at),
            hard_stale=self._hard_stale(at),
            files_gone=files_gone,
        )

    def _count_decline(self, newer: "BuildStamp", at: float) -> int:
        """Record one observation of a newer stamp this runtime did not act on.

        Per stamp: a DIFFERENT newer build is a different fact (a fresh
        ``lop-update``) and restarts the count, while ``_stale_since`` — the
        belt — keeps the age of the FIRST decline this process ever recorded
        and is never cleared (see :meth:`poll`).

        TWO SHAPES THIS MUST NOT BE READ AS COVERING, both measured by QA
        round 1 rather than argued:

        * a stamp stream with no settle gap at all. ``build_changed`` answers
          ``None`` while ``.lop-source`` is younger than ``BUILD_SETTLE_S``, so
          installs closer together than that leave ``newer is None`` on every
          check, and neither bound can arm (Q-2: 40 polls, ``declines=0``,
          ``_stale_since=None``, ``hard_stale=False``). It is left that way on
          purpose: the settle is what makes an unreadable stamp mean "do not
          act", and acting on a persistently torn tree is the torn-tree race
          the settle exists to prevent — a successor booted out of a
          half-written site-packages. The precondition is installs faster than
          ``BUILD_SETTLE_S`` apart indefinitely, which ``lop-update`` does not
          do; the PR body states the bound rather than overstating it.
        * a marker that is GONE or blank (Q-3). ``installed_build`` then
          answers a stamp whose ``source_ref`` is empty, which differs from the
          boot stamp, so the count moves and a busy runtime retires ~15 s later
          announcing an empty ref. Self-healing — the successor boots the same
          build — and intended: an install whose identity cannot be read is not
          a build this runtime gets to keep declining to notice.
        """
        if self._declined != newer:
            self._declined = newer
            self._declines = 0
        self._declines += 1
        if self._stale_since is None:
            self._stale_since = at
        return self._declines

    def _hard_stale(self, at: float) -> bool:
        if self._declines >= BUILD_MAX_STALE_GENERATIONS:
            return True
        return self._stale_since is not None and at - self._stale_since >= BUILD_MAX_STALENESS_S

    @property
    def _probe_armed(self) -> bool:
        """Is the files-gone probe armed for this runtime? Read once, lazily."""
        if not self._arm_read:
            self._armed = _tree_is_replaceable()
            self._arm_read = True
        return bool(self._armed)

    def _files_gone(self, at: float) -> bool:
        """Has the module tree this process loaded disappeared from disk, and
        STAYED gone past the install settle?

        Armed only for a tree an updater can replace wholesale
        (:func:`_tree_is_replaceable`) — an editable worktree is the negative
        control and must never trip this, however its files are touched.

        Existence, never mtime. An editor's atomic save gives a source file a
        new mtime and a new inode while the build is exactly the same one, and
        a developer's checkout legitimately looks stale by either clock; a file
        that is no longer there is not a judgement call.

        SUSTAINED, not sampled, and the settle is the same constant the stamp
        path uses. An installer rewrites site-packages over several seconds, so
        a single missing-path observation is the expected shape of a NORMAL
        in-place upgrade rather than evidence of a dead tree — and a runtime
        that retired inside that window would have its viewer spawn a successor
        against a half-written install, which is the torn-tree race
        ``BUILD_SETTLE_S`` exists to prevent. Waiting it out costs one more
        check; a tree that is genuinely gone stays gone.
        """
        if not self._probe_armed or not self._paths:
            return False
        try:
            present = all(path.exists() for path in self._paths)
        except OSError:  # noqa: BLE001 — an unstattable path is not a missing tree
            return False
        if present:
            self._missing_since = None
            return False
        if self._missing_since is None:
            self._missing_since = at
            logger.warning(
                "session runtime: the module tree this process loaded (%d paths probed) is "
                "missing from disk; waiting out the install settle before retiring",
                len(self._paths),
            )
            return False
        return at - self._missing_since >= _build_settle_seconds()


def _loaded_tree_paths() -> "tuple[Path, ...]":
    """The module paths whose disappearance means "my files are gone".

    A SAMPLE, captured once at boot from the running tree: this module's own
    file, the package ``__init__`` when the package can name it, and the PARENT
    of the first of those that resolved — the package root in the ordinary
    case, and merely this module's own directory when
    ``local_operator.__file__`` is unavailable. That difference is harmless on
    purpose: any one of the sampled paths missing means the tree is gone, so
    the sample is a probe and never a claim about the tree's shape. Three stats
    on the existing ``BUILD_CHECK_S`` cadence spanning the top-level package
    and a subpackage. Deliberately not a walk of the ~930 installed files: this
    runs on the runtime's own loop, and a dirty-flag storm on a busy filesystem
    is a failure mode of its own.
    """
    import local_operator

    paths: list[Path] = []
    for candidate in (getattr(local_operator, "__file__", None), __file__):
        if not candidate:
            continue
        try:
            resolved = Path(candidate).resolve()
        except OSError:  # noqa: BLE001 — an unresolvable path cannot be probed
            continue
        paths.append(resolved)
    if paths:
        paths.append(paths[0].parent)
    return tuple(dict.fromkeys(paths))


def _tree_is_replaceable() -> bool:
    """Is the tree this process loaded one an updater REPLACES underneath it?

    THE ARMING RULE of the files-gone probe, and it is the negative control
    rather than a precaution. An editable checkout — every development
    worktree, including the one this suite runs in — has no install of its own:
    its "build" is the working tree, where files legitimately appear, vanish
    and change under a long-lived runtime (a branch switch, an editor's atomic
    rename, a ``git checkout``). Retiring on that would kill sessions for doing
    nothing wrong.

    Asked of the installer's own metadata (``update.install_kind``) rather than
    by looking for a marker file, because a PyPI wheel installed by ``lop
    update`` is replaced wholesale just the same while carrying no git ref. An
    ``unknown`` layout — a vendored tree, a distro package, the fake prefix the
    e2e stage points at — keeps today's behaviour: no probe. Any failure to
    read the kind disarms it, because the cost of a wrong retire is a cold
    start the user did not ask for.
    """
    try:
        from local_operator import update as update_mod

        kind = update_mod.install_kind(prefix=_build_prefix())
    except Exception:  # noqa: BLE001 — an unreadable install kind disarms the probe
        logger.debug("install kind unreadable; files-gone probe disarmed", exc_info=True)
        return False
    return kind not in (update_mod.InstallKind.EDITABLE, update_mod.InstallKind.UNKNOWN)


def _drain_detail(poll: _BuildPoll, boot: "BuildStamp | None") -> str:
    """The parenthetical riding with the retirement cause.

    Names WHICH build the runtime left for and WHY NOW, because "the runtime
    retired" alone is not actionable to whoever reads the log later, and the
    why-now is the part an investigation cannot reconstruct after the fact: a
    build pair says what changed, the trigger says whether this runtime was
    still working or had lost its tree.
    """
    reasons: list[str] = []
    if poll.files_gone:
        reasons.append("the loaded module tree is gone")
    if poll.declines:
        reasons.append(f"declined {poll.declines}x")
    if not reasons:
        reasons.append("hard-stale")
    pair = _build_pair(boot, poll.newer) if poll.newer is not None else ""
    return f"{', '.join(reasons)}{pair}"


def _viewer_attached(runtime: object) -> bool:
    """Term 3 of the predicate: is an INTERACTIVE viewer connected?

    Only ``ClientKind == "attach"`` counts — a TUI following this session, or
    the phone's interactive attach while the user has the session open.
    ``"daemon"`` clients (the mobile daemon's adoption dial, ``lop send``,
    ``lop stop``, the future supervisor) deliberately do not: the daemon
    adopts EVERY session on the machine, so if its connection held runtimes
    warm nothing would ever exit. ``RuntimeServer.attach_clients()`` already
    computes exactly this count for the attach cap; it is probed rather than
    required so the reduced handles in tests keep working.
    """
    count = getattr(runtime, "attach_clients", None)
    if not callable(count):
        return False
    try:
        live = count()
    except Exception:  # noqa: BLE001 — uncertainty here must not pin the runtime
        logger.debug("attach_clients failed; treating as no viewer", exc_info=True)
        return False
    return isinstance(live, int) and live > 0


def _should_exit(handle: object, runtime: object) -> bool:
    """The residency predicate (design §6.1): exit when ALL three hold.

    1. ``handle.is_busy()`` is False — no turn, compaction, subagents, jobs,
       queued prompts, or gate parked on a user's answer. Work is
       authoritative: nothing below can end a turn early.
    2. No wake is due within :data:`WARM_WINDOW_S` — a runtime about to fire
       its own wake is cheaper kept than re-spawned (see the constant).
    3. No interactive viewer is attached — a user looking at the session is
       about to type, and holding the process warm turns "every message after
       a 3 s pause costs a cold start" into "the first message of a
       conversation costs one".

    Reconciling term 3 with the older rule "watchers and replicas observe
    work; they do not own it": both are still true, and they are about
    different things. OWNERSHIP of the work is the turn's — a viewer leaving
    does not abort a turn (term 1 is checked first and alone decides that),
    and a daemon-class client never holds anything. Term 3 is about
    READINESS: an attached interactive viewer is the one signal that the
    next message is imminent, so residency follows it. The phone's SSE
    watcher count (``phone_watchers``) stays out of the predicate — the
    daemon's connection is not the user's attention, and the phone's
    interactive attach dials as ``"attach"`` when it wants warmth.
    """
    is_busy = getattr(handle, "is_busy", None)
    if is_busy is not None and is_busy():
        return False
    if _wake_within_window(handle):
        return False
    if _viewer_attached(runtime):
        return False
    return True


async def _clean_exit(handle: object, runtime: object, *, reason: str = "idle-exit") -> None:
    """Dispose the quiescent session, then unpublish its owner record.

    The reaper reaches this only after ordinary gate timeouts and all resumed
    work have drained, so injecting a shutdown denial here would violate the
    same no-interruption invariant that selected this state.

    ``reason`` names WHY this runtime is leaving, and it is logged here rather
    than by the caller because this is the one place every planned exit
    converges. That line is not decoration: the reference investigation could
    not tell a refresh retirement from a SIGTERM from a torn install, because an
    exiting runtime logged nothing about itself and no exit record survived
    (design §1.6/§5.3).
    """
    boot = getattr(runtime, "_boot_build", None)
    logger.info(
        "session runtime: exiting (%s, pid %d, %s)",
        reason,
        os.getpid(),
        boot.label() if boot is not None else "<unknown>",
    )
    try:
        await handle.dispose()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — dispose is best-effort at exit
        logger.warning("child session dispose failed", exc_info=True)
    try:
        await runtime.aclose()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        logger.debug("child runtime aclose failed", exc_info=True)


async def _reaper(handle: object, runtime: object, stop: asyncio.Event) -> None:
    """Exit the disposable session runtime after one uninterrupted idle drain.

    The drain is re-checked every ``REAP_CHECK_S`` against the full predicate,
    so any term flipping back — work arriving, a viewer attaching, a wake
    entering the warm window — cancels it and the clock restarts from the
    next fully-idle tick. A wake that fires during the drain starts a turn,
    which flips ``is_busy()``: that is how "due within the drain" fires once,
    in-process, with no supervisor involvement.

    TWO WAYS TO LEAVE, and the difference between them is the whole point of
    :class:`_BuildWatch`. The QUIET exit above is the residency policy: it waits
    for a fully idle runtime and is cancelled by anything that comes back. The
    BUILD exit is an obligation: once a runtime has been observed declining a
    replaced build past the bound — or has lost the tree it loaded — it stops
    admitting work and leaves at the first instant its OWN work is done, viewer
    or no viewer. Nothing in flight is ever aborted by either path.
    """
    grace_s = _grace_seconds()
    boot: BuildStamp | None = getattr(runtime, "_boot_build", None)
    watch = _BuildWatch(boot)
    next_build_check = time.monotonic() + BUILD_CHECK_S
    #: Set once this runtime has committed to leaving for the build on disk.
    #: Non-local to the tick: the exit is re-attempted on both loop levels.
    drain: _Drain | None = None

    async def refresh_check() -> bool:
        """The build branch on its own slower cadence. True once the
        runtime has retired (the caller returns). Runs on BOTH loop levels
        below: outside the drain — where an attached viewer, which holds the
        quiet exit, must not hold this — and inside it, because a grace of
        minutes (``LOP_SESSION_GRACE_S``) would otherwise starve the check
        for an unwatched runtime that is exactly the one nobody else will
        ever refresh."""
        nonlocal next_build_check, drain
        if time.monotonic() < next_build_check:
            return False
        next_build_check = time.monotonic() + BUILD_CHECK_S
        poll = watch.poll(handle)
        if drain is not None:
            # A LATCHED DRAIN WINS, and it is tested FIRST (review round 1,
            # MINOR 2). The drain latches while the runtime is busy by
            # construction, so the first build check after the work ends finds
            # it idle-and-newer; taking the soft rung there would draw a SECOND
            # ``BUILD_STAGGER_S`` slice at the exit — the exact delay
            # ``_Drain.stagger_until`` is drawn at drain start to avoid — and
            # announce ``retiring`` a second time for one departure.
            return await _drain_for(drain, handle, runtime, stop)
        if poll.refreshable():
            return await _refresh_for(cast("BuildStamp", poll.newer), handle, runtime, stop)
        if not poll.draining():
            return False
        drain = await _begin_drain(poll, handle, runtime, stop)
        if drain is None:
            return False
        return await _drain_for(drain, handle, runtime, stop)

    while not stop.is_set():
        await asyncio.sleep(REAP_CHECK_S)
        if stop.is_set():
            continue
        if await refresh_check():
            return
        if stop.is_set():
            continue
        if drain is not None:
            # Committed to leaving. Neither the quiet-exit grace nor term 3 of
            # ``_should_exit`` applies here: a viewer that is looking at this
            # session is re-engaging it (the ``retiring`` frame went out at
            # drain start at the latest, and often minutes ago), and a grace
            # window is for a runtime that might still be wanted — this one has
            # already stopped taking work.
            if await _drain_for(drain, handle, runtime, stop):
                return
            continue
        if not _should_exit(handle, runtime):
            continue
        deadline = time.monotonic() + grace_s
        drained = False
        while time.monotonic() < deadline:
            await asyncio.sleep(REAP_CHECK_S)
            if await refresh_check():
                return
            if stop.is_set() or not _should_exit(handle, runtime):
                break  # a predicate term flipped back (or shutdown began)
        else:
            drained = True
        if not drained:
            continue
        # The LAST instant before the exit, and the reason it is a latch rather
        # than another ``_should_exit`` sample: the grace loop's condition can
        # go false on the same tick that a ``prompt`` or ``peer_message`` opens
        # a turn, and this branch then disposes without re-reading the
        # predicate — aborting work it had just refused to wait for. The latch
        # commits the runtime to leaving in the same synchronous step that
        # checks it, so from here the admissions REFUSE and the claim is true by
        # construction (design §5.1).
        begin_retire = getattr(handle, "begin_retire", None)
        if callable(begin_retire) and not begin_retire("idle-exit"):
            logger.info("session runtime: work arrived as the idle drain closed; keeping")
            continue
        logger.info(
            "session runtime: idle for %.1fs (no work, no viewer, no wake within %.0fs); "
            "exiting cleanly",
            grace_s,
            WARM_WINDOW_S,
        )
        await _clean_exit(handle, runtime, reason="idle-exit")
        stop.set()  # amain's wait() returns; exit code stays 0
        return


async def _refresh_for(
    newer: "BuildStamp", handle: object, runtime: object, stop: asyncio.Event
) -> bool:
    """Announce and retire so the next engage runs ``newer``. True if exited.

    Three checks of the predicate, and each is load-bearing:

    1. Before the stagger (the caller's) — the cheap gate.
    2. After the stagger — work may have arrived while sixteen siblings
       spread their exits over ``BUILD_STAGGER_S``; a runtime that picked up
       a turn meanwhile keeps it and tries again next check.
    3. After the announce — ``announce_retiring`` is an await (it drains each
       viewer's writer), and a ``peer_message`` or ``prompt`` can open a turn
       in that gap. Same shape as ``RuntimeServer._retire_if_pristine``'s
       re-check after its ``stopping`` broadcast. A turn that starts between
       THIS re-check and ``_clean_exit`` is aborted by the dispose exactly as
       a ``stop`` op racing a turn is; the message is persisted and the
       sender's next engage runs the new build.

    ``retiring`` is announced AFTER the stagger, immediately before exit, so
    a viewer never waits on a runtime that is merely "about to" leave.
    """
    boot: BuildStamp | None = getattr(runtime, "_boot_build", None)
    delay = random.uniform(0, _build_stagger_seconds())  # noqa: S311 — jitter, not security
    logger.info(
        "session runtime: build on disk is %s but this process loaded %s; idle, retiring in "
        "%.1fs so the next engage runs the new build",
        newer.label(),
        boot.label() if boot is not None else "<unknown>",
        delay,
    )
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
        return False  # a stop landed during the stagger; its path owns the exit
    except asyncio.TimeoutError:
        pass
    if _should_refresh(handle, boot) is None:
        logger.info("session runtime: work arrived during the refresh stagger; keeping")
        return False
    announce = getattr(runtime, "announce_retiring", None)
    if callable(announce):
        try:
            await cast(Callable[..., Awaitable[None]], announce)("stale-build", to=newer.label())
        except Exception:  # noqa: BLE001 — a viewer that misses this goes cold the slow way
            logger.debug("retiring announcement failed", exc_info=True)
    if stop.is_set():
        return False
    # The final check is the LATCH, not another sample: a retirement that acted
    # on a sampled "idle" and then met a turn during the announce would abort
    # work it had just decided not to disturb. ``begin_retire`` commits the
    # runtime to leaving in one synchronous step, and from that instant the
    # admission paths refuse, so no turn can open between here and the dispose.
    begin_retire = getattr(handle, "begin_retire", None)
    if callable(begin_retire):
        if not begin_retire("runtime-retired", _build_pair(boot, newer)):
            logger.info("session runtime: work arrived while retiring was announced; keeping")
            return False
    elif _should_refresh(handle, boot) is None:
        # A handle without the latch (an older or reduced host, e.g. the tests'
        # stub handles): keep today's re-check rather than retiring unguarded.
        # Refusing AFTER announcing is safe for the same reason it is for
        # ``stopping``: ``retiring`` only latches the disconnect REASON in an
        # attach client, and does nothing unless the socket then closes.
        logger.info("session runtime: work arrived while retiring was announced; keeping")
        return False
    logger.info("session runtime: retiring for %s", newer.label())
    await _clean_exit(handle, runtime, reason="retiring for " + newer.label())
    stop.set()
    return True


@dataclass
class _Drain:
    """A departure already committed to, waiting only on this runtime's work.

    ``stagger_until`` is drawn ONCE, when the drain begins, and is the same
    ``BUILD_STAGGER_S`` slice :func:`_refresh_for` sleeps: sixteen runtimes that
    all become idle on one tick must not spawn sixteen successors together.
    Drawn at the START rather than at the exit, because the drain's exit is
    reached from a busy state by construction — the runtime was declining a
    build precisely because it had work — so the wait it implies has usually
    elapsed by the time the work is done, and a runtime that becomes idle
    immediately waits it out. Either way the exits are spread.
    """

    detail: str
    to: str
    reason: str
    stagger_until: float


async def _begin_drain(
    poll: _BuildPoll, handle: object, runtime: object, stop: asyncio.Event
) -> "_Drain | None":
    """Announce the departure, then stop admitting work. ``None``: not ours.

    ANNOUNCE FIRST, LATCH SECOND, and the order is invariant (iii): a viewer
    must learn the runtime is leaving BEFORE it starts refusing, or the first
    refused message reads as an error rather than as a handover. The latch
    (``ServingSessionHandle.begin_drain``) is the commit — from that instant no
    new work is admitted while the live turn, its subagents and its jobs run to
    completion, and the exit waits for exactly that.

    A handle without the latch (an older host, a reduced test double) does NOT
    drain. The bound's whole guarantee is that admissions stop; a runtime that
    kept accepting work it had already decided to walk away from would be
    serving the replaced build for longer, not less. It keeps the old behaviour
    — keep serving, ask again on the next check — which is the status quo rather
    than a regression.

    THE ANNOUNCEMENT PRECEDES THE LATCH. The daemon reaches the same order by a
    different route, and the difference is worth reading off rather than
    paraphrased: ``server/retire.py`` publishes ``retiring_from``/``retiring_to``
    into the RECORD the moment a settled change is detected, keeps serving
    while anything is attached, and latches its typed refusal only once the
    drain has emptied — with a jittered ``BUILD_STAGGER_S`` slice between the
    latch and the exit. Both serve the same goal, a reader must learn the
    process is leaving before it is refused, and each mechanism decides how much
    time that reader gets: a daemon's readers POLL its record, so announcing
    early costs it nothing and it can keep working until they let go, while this
    runtime's announcement is a FRAME on the very connection a prompt arrives
    on — and waiting for that viewer is precisely the failure being fixed here,
    because viewer presence is the term that kept five-hour-stale runtimes
    resident. So the runtime announces and latches in the SAME synchronous step
    and leaves when its own work is done, rather than waiting to be let go. See
    ``ServingSessionHandle.begin_drain`` for the runtime's half of the shape.
    """
    begin_drain = getattr(handle, "begin_drain", None)
    if not callable(begin_drain):
        return None
    if getattr(handle, "_disposing", False):
        # The disposal owns the exit already. Announcing a departure here would
        # repeat on every check (nothing latches, so ``drain`` stays unset) and
        # would invite a viewer to re-engage a session that is on its way out
        # for a reason the disposal has stated itself.
        return None
    boot: BuildStamp | None = getattr(runtime, "_boot_build", None)
    to = poll.newer.label() if poll.newer is not None else ""
    detail = _drain_detail(poll, boot)
    if poll.files_gone:
        reason = "retiring: the build this process loaded is gone from disk"
    elif to:
        reason = "retiring for " + to
    else:
        reason = "retiring for a build replaced on disk"
    delay = random.uniform(0, _build_stagger_seconds())  # noqa: S311 — jitter, not security
    logger.info(
        "session runtime: %s (loaded %s; %s); no new work will be admitted, in-flight work "
        "finishes first (pid %d)",
        reason,
        boot.label() if boot is not None else "<unknown>",
        detail,
        os.getpid(),
    )
    announce = getattr(runtime, "announce_retiring", None)
    if callable(announce):
        try:
            await cast(Callable[..., Awaitable[None]], announce)("stale-build", to=to)
        except Exception:  # noqa: BLE001 — a viewer that misses this goes cold the slow way
            logger.debug("retiring announcement failed", exc_info=True)
    if stop.is_set():
        return None  # a stop arrived during the announcement; its path owns the exit
    try:
        latched = begin_drain("runtime-retired", detail)
    except Exception:  # noqa: BLE001 — uncertainty keeps the runtime
        logger.warning("could not latch the drain; keeping runtime", exc_info=True)
        return None
    if not latched:
        return None
    return _Drain(
        detail=detail,
        to=to,
        reason=reason,
        stagger_until=time.monotonic() + delay,
    )


async def _drain_for(drain: _Drain, handle: object, runtime: object, stop: asyncio.Event) -> bool:
    """Leave at the first instant this runtime's own work is done. True if exited.

    THE DIFFERENCE FROM :func:`_refresh_for` IS THE FIX. That path samples the
    idle predicate and acts only on an instant at which nothing would be lost —
    right for a refresh that can wait, and unusable for a runtime whose build
    has been replaced under it: a session busy for hours never reaches such an
    instant, which is how eight runtimes kept executing a tree that was gone.
    Here the runtime has ALREADY stopped admitting work
    (``ServingSessionHandle.begin_drain``), so the same predicate converges by
    itself — the running turn finishes, no successor turn can open, and the
    idle instant arrives. Nothing in flight is aborted: the wait is bounded by
    the work, never by a clock.

    The viewer term of :func:`_should_exit` is deliberately absent, exactly as
    it is absent from ``may_refresh``. The ``retiring`` frame went out at drain
    start, so a viewer re-engages onto the new build instead of holding this
    one — and holding for it is what kept five-hour-stale runtimes resident.

    The exit commits through ``begin_retire``, so the last instant still says
    "idle" by construction and the cut-off note a retirement owes is written by
    the rung that owns it. Retried every ``REAP_CHECK_S`` until it lands.
    """
    if time.monotonic() < drain.stagger_until:
        return False
    if not _idle_for_refresh(handle):
        return False
    begin_retire = getattr(handle, "begin_retire", None)
    if callable(begin_retire) and not begin_retire("runtime-retired", drain.detail):
        logger.info("session runtime: work arrived as the drain closed; keeping")
        return False
    logger.info("session runtime: %s; exiting cleanly", drain.reason)
    await _hand_wakes_to_successor(handle)
    await _clean_exit(handle, runtime, reason=drain.reason)
    stop.set()  # amain's wait() returns; exit code stays 0
    return True


async def _hand_wakes_to_successor(handle: object) -> int:
    """Write the wakes this drain swallowed, so a successor is raised for them.

    Called at the EXIT rather than from the drain's own deliver hook, because
    the hook runs inside ``WakeScheduler.pump``'s write lock: the write this
    needs would deadlock against it, and the pump persists its post-retire list
    moments later, which would overwrite a write made from there. By the exit
    that persist has landed — see ``Session.hand_wakes_to_successor``.

    The wakes that need it are the ones whose fire RETIRED their schedule: the
    index row the supervisor raises a wake errand from is gone with the
    schedule, so an unwatched session would keep the reminder and never run the
    work until a human opened the conversation (review round 1, MINOR 3).

    Never raises: a runtime that has already stopped admitting work must not be
    held by a failed handover, and the session logs the loss itself.
    """
    session = getattr(handle, "_session", None)
    probed = getattr(session, "hand_wakes_to_successor", None)
    if not callable(probed):
        return 0
    hand_over = cast("Callable[[], Awaitable[int]]", probed)
    try:
        handed = await hand_over()
    except Exception:  # noqa: BLE001 — a failed handover must not block the exit
        logger.warning("could not hand the drain's wakes to a successor", exc_info=True)
        return 0
    return int(handed)


async def _drain_inbox_into(handle: object) -> int:
    """Deliver every message spooled while this session was cold. Count sent.

    Also the handover path: the same file is where a DRAINING runtime spools what
    arrives while it finishes (``serving._spool_for_successor``), so this drains
    both "nothing was running" and "what was running had already committed to a
    replaced build".

    Called from :func:`amain` after the session exists and before the control
    socket listens — see the call site for why that ordering is the delivery
    guarantee rather than an implementation detail.

    Delivery honours the row's ``wake`` and NEVER its ``mode`` (``mailbox``
    either way — see ``serving._spool_for_successor`` for why a boot cannot
    honour a steer, and why the row keeps it anyway): a row
    spooled by a runtime that was leaving a replaced build carries what its
    sender asked for, and a wake — a fired alarm, or a peer ``send --wake`` —
    asked for a TURN. Delivering it as a quiet note would keep the message and
    never do the work, which for a recurring automation is the "scheduled work
    silently not running" shape this drain exists to avoid (review round 1,
    MINOR 3). Rows written before the field existed read as notes, unchanged.

    Best-effort per message: one malformed or rejected row must not stop the
    rest, and none of it may prevent the runtime from starting.
    """
    from local_operator.session.runtime.inbox import drain_inbox

    session = getattr(handle, "_session", None)
    directory = getattr(getattr(session, "transcript", None), "directory", None)
    if directory is None:
        return 0
    try:
        lines = await asyncio.to_thread(drain_inbox, directory)
    except Exception:  # noqa: BLE001 — a bad spool must not block the runtime
        logger.warning("inbox drain failed", exc_info=True)
        return 0
    probed = getattr(handle, "receive_peer_message", None)
    if not lines or not callable(probed):
        return 0
    receive = cast(Callable[..., Awaitable[str]], probed)
    delivered = 0
    for line in lines:
        try:
            await receive(
                line.text,
                mode="mailbox",
                wake=bool(getattr(line, "wake", False)),
                sender=line.sender,
            )
            delivered += 1
        except Exception:  # noqa: BLE001 — one bad row is not the others' problem
            logger.warning("spooled message could not be delivered", exc_info=True)
    if delivered:
        logger.info("delivered %d spooled message(s) at open", delivered)
    return delivered


def _install_sighup_ignore(loop: asyncio.AbstractEventLoop) -> None:
    """Make ``SIGHUP`` a no-op for this runtime, logged once per process.

    WHY AN IGNORE AND NOT A CLEAN EXIT. Only two things may end a session's
    work: the runtime's own residency predicate (``_should_exit``) and the
    deliberate kill switch (``control.stop_session`` → socket stop → SIGTERM →
    SIGKILL). SIGHUP is neither — no front end and no user asked for anything —
    and it is the classic "the terminal that started me is gone" signal, the one
    ``nohup`` exists to ignore, while this process is spawned detached
    (``start_new_session=True``) and writes its log to a file. Left at its
    DEFAULT disposition a HUP kills the interpreter outright: no caused turn
    outcome, no lease release, no record unpublish, not even the
    ``session runtime: exiting`` line — the session is left reading as an
    anonymous "cause could not be determined" cut-off.

    CALLED FROM THE FIRST STATEMENT OF ``amain``, so the guarantee covers the
    whole substantive boot (lease arbitration, session construction, MCP
    bring-up, ``start_in_process``). The residual window is ``main()``'s logging
    setup and the ``asyncio.run`` bootstrap — milliseconds, and nothing is
    spawned in it. A process-wide ``signal.signal(SIG_IGN)`` installed in
    ``main()`` instead would close even that, and is deliberately NOT done:
    ``main()`` is callable in-process (the suite does exactly that for the CLI —
    see ``procname.is_own_launch``, which exists for the same hazard), and a
    disposition set there would leak into the embedder's process for the rest of
    its life, with nothing to restore it.

    The latch, not a per-signal log line: a terminal that re-delivers on its way
    down must not be able to fill the runtime log.
    """
    hup_logged = False

    def _on_sighup() -> None:
        nonlocal hup_logged
        if hup_logged:
            return
        hup_logged = True
        logger.info(
            "session runtime: ignoring SIGHUP (pid %d); this runtime is detached from interfaces",
            os.getpid(),
        )

    # ``SIGHUP`` is POSIX-only, and a platform without it must not fail to boot a
    # runtime over a signal it could not have received.
    sighup = getattr(signal, "SIGHUP", None)
    if sighup is None:
        return
    try:
        loop.add_signal_handler(sighup, _on_sighup)
        return
    except (NotImplementedError, RuntimeError, ValueError):
        pass
    # THE FALLBACK CANNOT BE ALLOWED TO RAISE, and it is the branch that plants
    # an INHERITABLE ignore: ``signal.signal`` works only on the main thread
    # (``ValueError`` otherwise, which is exactly how a loop that refused for
    # that reason would then kill the boot this function protects), and a SIG_IGN
    # survives ``exec`` — CPython's ``restore_signals`` resets only
    # SIGPIPE/SIGXFZ/SIGXFSZ — so anything spawned after it would inherit an
    # ignored HUP. Nothing reaches here on the shipped path (the loop takes the
    # callback), so this is the belt for a platform whose loop will not.
    try:
        signal.signal(sighup, signal.SIG_IGN)
    except (ValueError, OSError, RuntimeError):
        logger.warning("could not install the SIGHUP ignore", exc_info=True)


async def amain() -> int:
    # SIGHUP FIRST, before the deferred imports below, the lease arbitration,
    # session construction, MCP bring-up and ``start_in_process``: the guarantee
    # is "no interface can end this session's work", and a HUP during boot has
    # exactly the unattributed shape it exists to remove (review round 1,
    # MINOR-3). See ``_install_sighup_ignore`` for the scope this does and does
    # not cover.
    _install_sighup_ignore(asyncio.get_running_loop())

    # Deferred for startup cost, not to break a cycle: importing the owned
    # handle pulls the composition root, and `python -m` on this module must
    # not pay for it before the log file is configured in main().
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import (
        ServingSessionHandle,
        spawn_owned_session,
    )
    from local_operator.session_lease import SessionLeaseHeldError

    cwd = os.environ.get("LOP_MOBILE_CHILD_CWD") or os.path.expanduser("~")
    provider = os.environ.get("LOP_MOBILE_CHILD_PROVIDER") or None
    model_id = os.environ.get("LOP_MOBILE_CHILD_MODEL") or None
    # The reasoning level this conversation is BORN with, when its viewer chose
    # one. It is a birth sample like the pair above: ``session_factory`` uses it
    # in place of the configured default so the spec this process constructs
    # carries the level, which is what makes the first frontend snapshot, the
    # first provider call and the selection row written at admission agree. A
    # value this model cannot express is CLAMPED there rather than refused — a
    # stored choice may outlive the ladder that offered it.
    birth_effort = os.environ.get("LOP_MOBILE_CHILD_EFFORT") or None
    resume = os.environ.get("LOP_MOBILE_CHILD_RESUME") or None
    if resume:
        # A runtime ADOPTS the id it was given rather than requiring a
        # directory to already exist. The viewer mints the session id before
        # anything is on disk (it is a name, not a directory, until there is
        # work), so the first engage of a brand-new session arrives here with
        # nothing to resume — and the strict `--resume` path would refuse it.
        # See ``session_factory._transcript_dir_and_agent_id``.
        os.environ["LOP_RUNTIME_ADOPT_SESSION"] = "1"

    loop = asyncio.get_running_loop()
    try:
        handle: ServingSessionHandle = await spawn_owned_session(
            loop,
            cwd=cwd,
            provider=provider,
            model_id=model_id,
            birth_effort=birth_effort,
            resume=resume,
            model_selection_override=os.environ.get("LOP_MODEL_SELECTION_OVERRIDE") == "1",
        )
    except SessionLeaseHeldError as exc:
        # LOSING THE LEASE IS NOT AN ERROR. Under ``engage_runtime`` every
        # contender is allowed to spawn a candidate and the lease decides which
        # one lives (session/runtime/launch.py) — so a loser is a race working
        # exactly as designed, and it exits 0. Returning non-zero here made an
        # ordinary ten-way engage look like nine crashes in the logs, and would
        # make a supervisor's KeepAlive treat normal arbitration as a failure
        # loop.
        logger.info(
            "runtime lost the lease for %s to pid %s; exiting",
            resume or "<new>",
            exc.pid,
        )
        return 0
    except Exception as error:
        logger.exception("session runtime child: session construction failed")
        # ALSO to stderr, which the spawning parent captures. `main()` points
        # logging at the daemon's own file, so the traceback above is written
        # where only a person reading logs later can find it -- the parent saw
        # nothing at all and could only report a generic timeout after burning
        # its whole deadline (QA Q1). One line on stderr is what lets an engage
        # fail fast and name the actual cause.
        print(
            f"{type(error).__module__}.{type(error).__qualname__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    # THE ORDERING IS THE GUARANTEE (design §11.4). Messages spooled while the
    # session was cold are delivered here, BEFORE the control socket begins
    # listening, so they cannot be interleaved with an errand a client sends
    # over that socket — there is no socket yet. Draining after
    # ``start_in_process`` would race the engaging caller's own prompt and
    # deliver a note written minutes ago after one written just now.
    await _drain_inbox_into(handle)

    # The wake scheduler is armed HERE, after the inbox drain and before the
    # socket listens. A runtime the supervisor starts for an overdue wake has
    # no errand that delivers a prompt — the WakeErrand carries nothing by
    # design — so the wake's turn comes from the session's own catch-up path,
    # which only runs once the scheduler is pumped. Round 2 (U4/Q9) found the
    # cold runtime never called ``async_init``: the runtime started, idled,
    # and exited, and a one-shot wake was consumed without ever running.
    #
    # After the drain so spooled quiet notes land before the wake's turn
    # starts; before the socket so a client's prompt cannot race the catch-up.
    # ``async_init`` is idempotent and degrades to ``ensure_future`` for
    # background work, so a host that also calls it later changes nothing.
    session = getattr(handle, "_session", None)
    init = getattr(session, "async_init", None)
    if callable(init):
        try:
            result = init()
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 — an unarmable scheduler is not a dead runtime
            logger.warning("wake scheduler did not arm at boot", exc_info=True)

    runtime = RuntimeServer(handle, kind="daemon")
    await runtime.start_in_process()

    stop = asyncio.Event()
    # What ASKED this runtime to leave. Named because the exit itself is the one
    # event the reference investigation could not attribute: a refresh, a
    # SIGTERM and a torn install all ended the process with nothing written
    # about which it was (design §1.6/§5.3). The reaper and the retire path log
    # their own reason from ``_clean_exit``; this covers the two triggers that
    # dispose directly.
    trigger: dict[str, str] = {}

    def _on_signal(sig: signal.Signals) -> None:
        trigger.setdefault("why", sig.name)
        stop.set()

    def _on_socket_stop() -> None:
        """The graceful ``stop`` op (``ServingSessionHandle.request_stop``)."""
        trigger.setdefault("why", "socket-stop")
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _on_signal, sig)
    if os.environ.get("LOP_RUNTIME_DEBUG_STACKS") == "1":
        # SIGUSR1 prints every asyncio task's stack to the child log. The
        # child has no terminal and no attached debugger, and a wedged turn
        # (round 2, U6) is exactly the state whose cause is "which await is
        # the turn parked in" — invisible to py-spy without root and to the
        # main-thread faulthandler dump, which shows the loop idling under a
        # parked task. Opt-in so a normal runtime pays nothing.
        def _dump_task_stacks() -> None:
            session = getattr(handle, "_session", None)
            try:
                subagents = session.running_subagents() if session is not None else 0
            except Exception as exc:  # noqa: BLE001 — the dump must not die
                subagents = f"RAISES {type(exc).__name__}: {exc}"
            logger.info(
                "state: streaming=%s compacting=%s lock=%s queue=%s drain_done=%s "
                "subagents=%s is_busy=%s",
                getattr(session, "_is_streaming", "?"),
                getattr(session, "_compacting", "?"),
                getattr(getattr(session, "_turn_lock", None), "locked", lambda: "?")(),
                len(getattr(handle, "_prompt_queue", [])),
                getattr(getattr(handle, "_prompt_drain_task", None), "done", lambda: "?")(),
                subagents,
                handle.is_busy(),
            )
            sig = getattr(session, "_signal", None)
            logger.info(
                "signal: present=%s aborted=%s abort_requested=%s",
                sig is not None,
                getattr(sig, "aborted", None),
                getattr(session, "_abort_requested", "?"),
            )
            for task in asyncio.all_tasks(loop):
                if task.done():
                    continue
                # The parked await is at the BOTTOM of the coroutine chain:
                # each awaited coroutine's frame hangs off the outer one's
                # cr_await, not its f_back, so format_stack alone prints only
                # the outermost frame. Walk the chain to see where the turn
                # is actually parked.
                lines: list[str] = []
                obj: Any = task.get_coro()
                while obj is not None:
                    frame = getattr(obj, "cr_frame", None) or getattr(obj, "gi_frame", None)
                    if frame is None:
                        break
                    code = frame.f_code
                    lines.append(f"  {code.co_filename}:{frame.f_lineno} in {code.co_name}")
                    obj = getattr(obj, "cr_await", None) or getattr(obj, "gi_yieldfrom", None)
                logger.info("task %r await-chain:\n%s", task.get_name(), "\n".join(lines))

        loop.add_signal_handler(signal.SIGUSR1, _dump_task_stacks)
    # The socket ``stop`` op (the kill switch's graceful rung) and SIGTERM
    # converge on the same event, so the deny → dispose → aclose ordering
    # below runs once, identically, for both triggers.
    handle.on_stop_requested = _on_socket_stop
    # The self-reaper: a phone session nobody watches and nothing runs is a
    # live process doing nothing, and before this it idled FOREVER. Runs
    # beside the signal wait; whichever fires first wins.
    reaper = asyncio.ensure_future(_reaper(handle, runtime, stop))
    reaper_ran_clean_exit = False
    await stop.wait()
    if not reaper.done():
        reaper.cancel()
    elif reaper.exception() is None:
        # The reaper completed (not was cancelled): it already ran the clean
        # ordering. A signal-initiated stop still owes it.
        reaper_ran_clean_exit = True
    if not reaper_ran_clean_exit:
        # The reaper logs its own line from ``_clean_exit``; these are the
        # direct-dispose triggers (a signal, or the graceful ``stop`` op).
        boot = getattr(runtime, "_boot_build", None)
        logger.info(
            "session runtime: exiting (%s, pid %d, %s)",
            trigger.get("why") or "unknown",
            os.getpid(),
            boot.label() if boot is not None else "<unknown>",
        )
        try:
            handle._deny_pending_gates()
        except Exception:  # noqa: BLE001 — shutdown must proceed
            logger.debug("child gate deny failed", exc_info=True)
        try:
            await handle.dispose()
        except Exception:  # noqa: BLE001
            logger.warning("child session dispose failed", exc_info=True)
    await runtime.aclose()

    # Under `LOP_RUNTIME_DEFER_MATERIALISE` the transcript and roster sidecar
    # never create the session directory, but the LEASE cannot be deferred —
    # it arbitrates "at most one runtime per session" and lives inside the
    # directory — so a speculatively warmed runtime that was never given real
    # work leaves a lease-only `sessions/<id>/` behind. This exit path USED
    # TO `rmdir` that directory (#622, `_remove_unwritten_session_dir`). It
    # no longer does, and must not: the operator's logs show it firing on a
    # real store, and the rule after that incident is that no exit hook, no
    # startup hook and no sweep removes a session directory on its own
    # judgement — a lease-only directory is exactly what the user-enabled
    # `session.cleanup.remove_empty` policy exists for, and it is off by
    # default. `tests/unit/session/test_no_session_deletion.py` forbids
    # reintroducing an `rmdir` here.
    return 0


def main() -> int:
    # A child has no terminal and no inherited log stream — without this its
    # warnings (a failed prompt, a dead provider) vanish, which is how a
    # silently-dropped turn went undiagnosed. It writes a file of its own; why
    # that is not the daemon's is the paragraph below.
    #
    # BOUNDED and quiet, unlike the `logging.basicConfig(level=INFO,
    # filename=...)` this replaces, which wrote an UNBOUNDED file and handed the
    # root level to INFO so every wire client logged one record per request.
    # `configure_file_logging` pins those clients to WARNING and bounds the file
    # at LOG_TOTAL_MAX_BYTES, so a chatty or wedged runtime costs a fixed
    # ceiling per writer instead of the disk.
    #
    # The file is the runtimes' OWN (`runtime.log`), not the daemon's
    # `mobile.log`: the daemon's log is a launchd StandardOutPath whose fd can
    # never be reopened, and bounding a file means RENAMING it, so a runtime
    # rotating the daemon's log would move the daemon's stream out of what
    # `lop mobile logs` reads. `lop mobile logs` reads both.
    from local_operator.logger import configure_file_logging
    from local_operator.paths import runtime_log_path

    target = runtime_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if configure_file_logging(path=target, level=logging.INFO) is None:
        logger.warning("session runtime could not open a log file; records stay on stderr")
    # One record per runtime, naming its own process: this file is shared by every
    # runtime child, so a reader has to be able to attribute a line to the
    # process that wrote it.
    logger.info("session runtime started: pid %d", os.getpid())
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
