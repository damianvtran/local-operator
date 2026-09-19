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

**Work-aware termination.** A termination signal is no longer an exemption
from the residency rule above. When SIGTERM or SIGINT arrives with work in
flight, the runtime COMMITS to leaving through the same seam a replaced build
uses (:func:`_commit_to_leaving`) — announced as it commits, so the operator is
told before anything is refused, and latched so no new work is admitted — then
leaves at the next boundary at which nothing would be lost, bounded by
``types.SIGNAL_DRAIN_S`` (see :func:`_drain_for_signal` for the
three properties and why each is load-bearing). A signal with nothing in
flight is byte-for-byte the old behaviour, and SIGKILL remains unrefusable.
The graceful paths were always idle-gated; the signal path was the gap, and it
is the one an unnamed outside sweep can actually reach.
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
from local_operator.session.runtime.types import (
    BUILD_DRAIN_OVERDUE_CAUSE,
    BUILD_DRAIN_PROGRESS_S,
    LEAVING_FOR_BUILD,
    LEAVING_FOR_BUILD_OVERDUE,
    LEAVING_ON_SIGNAL,
    SIGNAL_DRAIN_CAUSE,
    SIGNAL_DRAIN_S,
)

if TYPE_CHECKING:
    from local_operator.session.runtime import journal
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
#: Added with the exit-time build pair (2026-09-17): the same reading the drain
#: uses to detect a move, asked again at the EXIT so the reason names the build
#: the process is actually leaving for. Aliased here rather than reached through
#: ``_buildwatch`` for the reason the block above exists — this module is RUN as
#: ``__main__``, so its own names are the stable seam for tests and for any
#: reader, and the call site should not depend on the attribute being reachable.
_handover_build = _buildwatch.handover_build

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


def _drain_reasons(poll: _BuildPoll) -> tuple[str, ...]:
    """WHY NOW, as the phrases the retirement's why-now carries.

    Split out of :func:`_drain_detail` because the reason it names is a fact
    about the LATCH — this runtime was declining a settled newer build, or its
    tree was gone — and it stays true of the exit however long the work in
    between took. The build PAIR is the half that can go stale, so the two are
    composed separately (see :func:`_drain_detail_at_exit`).

    EACH PHRASE NAMES ITS SUBJECT. ``declined 3x`` was read against the build
    it followed ("a newer build ... declined", i.e. refused) when the runtime
    was the party declining to hand over — and it was read by the operator, so
    the ambiguity cost something (design round 1, D2).
    """
    reasons: list[str] = []
    if poll.files_gone:
        reasons.append("the loaded module tree is gone")
    if poll.declines:
        reasons.append(f"the runtime declined to hand over {poll.declines}x")
    if not reasons:
        reasons.append("hard-stale")
    return tuple(reasons)


def _drain_detail(
    reasons: "tuple[str, ...]", boot: "BuildStamp | None", newer: "BuildStamp | None"
) -> str:
    """The why-now riding with the retirement, for the runtime LOG.

    Names WHICH build the runtime left for and WHY NOW, because "the runtime
    retired" alone is not actionable to whoever reads the log later, and the
    why-now is the part an investigation cannot reconstruct after the fact: a
    build pair says what changed, the trigger says whether this runtime was
    still working or had lost its tree.

    IT IS NOT A TURN'S REASON, and that is the round-1 correction: a retirement
    that latched through :meth:`ServingSessionHandle.begin_retire` proved nothing
    was in flight, so there is no cut for it to label — the string is logged at
    the exit (``serving.ServingSessionHandle.begin_retire``) and nowhere else.
    See ``process._drain_detail_at_exit`` for the pair's own re-read.

    ``newer`` is the build on disk at the moment the pair is being read, so a
    caller that has one in hand from a poll and a caller re-reading the disk at
    its exit both come through here — one spelling of the pair, and therefore
    one sentence shape, whichever moment it describes.
    """
    pair = _build_pair(boot, newer) if newer is not None else ""
    return f"{', '.join(reasons)}{pair}"


def _drain_detail_at_exit(drain: "_Drain") -> str:
    """The why-now the EXIT logs, with its build pair RE-READ.

    WHY NOT ``drain.detail``. That string was composed at the latch, and the
    gap between the latch and the exit is exactly the wait this runtime's work
    buys — hours, on a busy session. Two ``lop-update`` runs fit in that gap,
    and replaying the latch's pair then asserts a transition the process has
    already left: measured on the reporting host, five latches at 01:58 named
    ``(0.56.2 → 0.56.6)`` and the record that replayed one of them at 09:56
    named that same pair while 0.56.9 was the install on disk (2026-09-17). A
    log line naming a build that has not been on disk for hours is its own
    false report, and the operator's request was to stop backend updates from
    producing false traces.

    The honest pair at the exit is boot → whatever the install names NOW, which
    is what :func:`local_operator.buildwatch.handover_build` answers. ``None``
    from it asserts no transition at all — the install is back to the boot
    stamp, the stamp cannot be resolved into a build, or there is no install to
    compare against (:func:`handover_build`'s own three shapes, one of which is
    a build strictly OLDER than the boot stamp) — and that is the same rule the
    drain already follows before it acts on a move. ANSWERING THE ROLLBACK EDGE
    (QA round 1, Q3): with the pair dropped, the exit keeps only the reasons,
    which is a statement about what this runtime declined and not a promise
    that a newer build is on disk; the exit's own log line carries it, and the
    durable row this used to feed no longer exists (a retirement that proved
    nothing was in flight brands no turn — see
    ``ServingSessionHandle._note_retirement_cut_off``).

    The REASONS are deliberately still the latch's: this runtime was declining
    three settled builds, or its tree was gone, and neither fact expires. See
    :func:`_drain_reasons`.
    """
    return _drain_detail(drain.reasons, drain.boot, _handover_build(drain.boot))


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


def _work_in_flight(handle: object) -> bool:
    """Is there work in flight that disposing NOW would destroy?

    THE one work predicate of this module, and the reason it is a function
    rather than two inline ``handle.is_busy()`` calls: the reaper's residency
    predicate (:func:`_should_exit`, term 1) and the signal drain
    (:func:`_drain_for_signal`) must agree on what "would lose nothing" means.
    They did not — the reaper sampled the handle, the signal handler never
    asked and disposed on the spot — and that asymmetry is what cut 32 turns
    off in the 2026-09-14 sweep: the same event the reaper would have deferred
    to a safe boundary was fatal when it arrived as a signal instead of as an
    idle tick.

    ``is_busy`` is the authority (never the record's derived ``busy`` bit): it
    covers a live turn, a parked approval, a running goal loop, live subagents,
    background jobs and a queued prompt.

    Absent probe -> ``False``, matching the long-standing treatment of reduced
    test handles and older implementations that never grew the accessor. A
    RAISING probe -> ``True``, and the direction is deliberate: ``is_busy``
    documents itself as failing closed, and a predicate that cannot be
    evaluated must never be the thing that ends a turn. (Letting it propagate,
    which the inline form did, also took the reaper down with it.)
    """
    probe = getattr(handle, "is_busy", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:  # noqa: BLE001 — uncertainty must keep the runtime working
        logger.debug("busy probe failed; treating work as in flight", exc_info=True)
        return True


def _should_exit(handle: object, runtime: object) -> bool:
    """The residency predicate (design §6.1): exit when ALL three hold.

    1. :func:`_work_in_flight` is False (``handle.is_busy()``) — no turn,
       compaction, subagents, jobs, queued prompts, or gate parked on a user's
       answer. Work is authoritative: nothing below can end a turn early. That
       predicate is shared with the signal drain, so "would lose nothing" has
       exactly one definition on this process.
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
    if _work_in_flight(handle):
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

    The turn journal is told the same fact before the boot record is withdrawn,
    and in that order: a row left OPEN by a runtime that leaves anyway is the
    evidence a successor reads, and withdrawing the boot record first would
    strip the half that says which build the row's pid was running.
    """
    boot = getattr(runtime, "_boot_build", None)
    logger.info(
        "session runtime: exiting (%s, pid %d, %s)",
        reason,
        os.getpid(),
        boot.label() if boot is not None else "<unknown>",
    )
    _note_journal_exit(handle, reason)
    try:
        await handle.dispose()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — dispose is best-effort at exit
        logger.warning("child session dispose failed", exc_info=True)
    try:
        # ``aclose`` RAISES off the runtime's owning loop by design, and this
        # exit path runs on the SESSION's loop — after the serving plane moved
        # to its own thread, that is every daemon and exec teardown. The raise
        # here is swallowed by the ``except`` below, so the failure was quiet:
        # teardown still STARTED (``aclose`` requests the close before raising),
        # but nothing waited for it and the process could exit mid-teardown.
        # ``aclose_remote`` is the same teardown awaited across the thread hop;
        # a reduced test double that has only the owner-loop form falls back to
        # it, which is the behaviour that double was written against.
        remote = getattr(runtime, "aclose_remote", None)
        if callable(remote):
            await cast(Callable[[], Awaitable[None]], remote)()
        else:
            await runtime.aclose()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        logger.debug("child runtime aclose failed", exc_info=True)
    _clear_boot_record()


#: The pid whose boot record THIS process published, or ``None``.
#:
#: Process-level rather than per-handle because the artifact is: one boot record
#: per process, written once at boot. It exists so the exit path can answer "do I
#: have a record to withdraw?" with a comparison instead of a filesystem call
#: (see :func:`_clear_boot_record` for why that matters).
_boot_record_pid: int | None = None


def _bind_boot_instrumentation(
    handle: object, *, session_id: str = "", cwd: str = ""
) -> "journal.TurnJournal | None":
    """Write this runtime's boot record and attach its turn journal.

    Both halves are INSTRUMENTS (design-session-survival §5) and both are
    therefore best-effort: a runtime that cannot write its own account must
    still run its turns, so every failure degrades to "no evidence" and is
    logged. The failure that would be worse than the incident they exist to
    diagnose is a session that cannot boot because an observability write
    failed.

    The build is read with ``LOP_BUILD_PREFIX`` exactly as ``server.py`` reads
    it for the session record — the same seam, read the same way, so the two
    artifacts cannot disagree about the build a comparison between them rests
    on.
    """
    global _boot_record_pid
    try:
        from local_operator import update as update_mod
        from local_operator.session.runtime import journal

        session = getattr(handle, "_session", None)
        identity = str(getattr(session, "session_id", "") or session_id)
        directory = getattr(getattr(session, "_transcript", None), "directory", None)
        build = update_mod.installed_build(os.environ.get("LOP_BUILD_PREFIX") or None)
    except Exception:  # noqa: BLE001 — no identity, no artifacts; the turn still runs
        logger.warning("session runtime: boot instrumentation unavailable", exc_info=True)
        return None

    try:
        journal.write_boot_record(identity, build, cwd=cwd)
        _boot_record_pid = os.getpid()
        # THIS BOOT IS THE ONE MOMENT A NEW WRITER JOINS THE NAMESPACE, so it is
        # where the namespace is bounded: nothing else reaps ``run/host`` (see
        # ``journal.prune_boot_records``), and a directory that only ever grows
        # is what makes a recycled pid's stale record reachable.
        journal.prune_boot_records()
    except OSError:
        logger.warning("session runtime: could not write its boot record", exc_info=True)

    if directory is None:
        # A session with no transcript directory has nothing to attach a ROW to
        # (the record above still stands). Reachable only through a reduced
        # handle in a test, never through ``spawn_owned_session``.
        return None
    try:
        writer = journal.TurnJournal(directory, identity, build)
        attach = getattr(handle, "attach_turn_journal", None)
        if callable(attach):
            attach(writer)
        return writer
    except Exception:  # noqa: BLE001
        logger.warning("session runtime: could not attach its turn journal", exc_info=True)
        return None


def _note_journal_exit(handle: object, cause: str) -> None:
    """Tell the turn journal why this runtime is leaving. Best-effort.

    Only a turn still OPEN when this runs is affected — a quiescent exit says
    nothing, which is correct: there is no unfinished turn to attribute.
    """
    writer = getattr(handle, "_turn_journal", None)
    note = getattr(writer, "note_exit", None)
    if callable(note):
        note(cause)


def _clear_boot_record() -> None:
    """Withdraw this pid's boot record on a CLEAN exit. Best-effort.

    The asymmetry is the evidence: a record that survives its process says the
    process stopped without running its own exit ordering, which is the fact
    the 2026-09-15 fleet could not establish about itself. Withdrawing here is
    also what keeps ``run/host`` bounded — one record per unaccounted death,
    not one per runtime ever spawned.

    THE GUARD IS NOT AN OPTIMISATION. Nothing is attempted when this process
    never published a record, and that is load-bearing twice over. The exit path
    is TIMED by the reaper's own tests (``test_process_reaper`` measures the
    spread of ``_clean_exit`` elapsed times to 40 ms), so an instrument must not
    add work to it for a host that has no record to withdraw — an in-process
    session, a reduced handle, every test that never binds instrumentation.
    And the withdrawal is not free even when it does nothing: it resolves
    ``run_dir()``, which MKDIRS the namespace, so an unconditional call would
    make every clean exit create a directory it has nothing to put in.
    """
    if _boot_record_pid != os.getpid():
        return
    try:
        from local_operator.session.runtime import journal

        journal.clear_boot_record()
    except Exception:  # noqa: BLE001 — an exit path never fails over an instrument
        logger.debug("session runtime: could not withdraw its boot record", exc_info=True)


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
        # THIS RUNG PROVES NOTHING WAS IN FLIGHT, and the cut-off note now says
        # so by construction rather than by an argument here: the note is
        # written only for a turn the disposal is about to ABORT
        # (``ServingSessionHandle._note_retirement_cut_off``), and the grace
        # loop above only falls through with the whole residency predicate
        # holding — no turn, no job, no gate parked on the user — while the
        # latch refuses admissions from the same instant, so no turn can appear
        # between them either. Arming a cause at the LATCH (what this branch did
        # until 2026-09-17) claimed the opposite, and the claim was not free:
        # the note is consumed by whichever run end comes next, and the teardown
        # synthesises one for a run whose outcome was never published — so a
        # quiet update that caught nothing published a durable "error" row for a
        # run that had already ended, rendered as an unexplained cut-off because
        # ``idle-exit`` is a retirement label and not a cause in
        # ``incidents.CUT_OFF_CAUSES``. Six such rows on the reporting host
        # (2026-09-17); the one read in full is session ``1ee642a5a098``, whose
        # last turn row is 09:57:24 and whose ``error`` row was published at
        # 09:59:31 against the run that had already ended. The retirement is
        # still latched — the refusal and the log line need it.
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
    #: The reasons half of ``detail``, kept so the EXIT can render its own
    #: parenthetical (:func:`_drain_detail_at_exit`) without parsing the latch's
    #: string back apart: the phrases are the latch's facts and outlive it, the
    #: build pair in ``detail`` does not.
    reasons: "tuple[str, ...]" = ()
    #: The stamp this process booted from, so the exit can ask the disk what it
    #: has moved to SINCE — the pair the reason is allowed to assert. ``None``
    #: is a process that never had a comparable stamp, which asserts no pair.
    boot: "BuildStamp | None" = None
    #: The token this departure latches and names itself with at the exit. Two
    #: values, because the two triggers make different claims: ``runtime-retired``
    #: is the build vocabulary every viewer-driven retirement already carried
    #: ("this process left so the next engage runs the new build"), while a
    #: termination signal makes no claim about the build at all and says
    #: ``runtime-shutdown`` — the same token the dispose rung would have written
    #: had the signal been fatal on arrival, so a cut turn is classified
    #: identically whether the drain expired or never ran.
    cause: str = "runtime-retired"
    #: The progress clock (:class:`_DrainProgress`), created on the drain's FIRST
    #: tick rather than here, and the difference is not cosmetic: this dataclass is
    #: built by :func:`_commit_to_leaving`, whose callers have just announced a
    #: departure and know nothing about the work, while the tick is the one place
    #: that can observe it. A drain that never ticks never needs a clock, and the
    #: first tick is within ``REAP_CHECK_S`` of the latch.
    progress: "_DrainProgress | None" = None


#: The wire label the BACKSTOP announces when a build drain's work has stopped
#: moving, so a frame and the record it was written with both name the bound.
#:
#: IT IS A PREFIX OF ``stale-build`` FOR INSURANCE, NOT FOR A PATH THIS TREE TAKES.
#: ``types.leaving_phrase_for_frame`` matches these labels with ``startswith`` and
#: answers the build sentence for this one, but it is reached only for a frame that
#: carries NO ``leaving`` — and this rung always sends one, so within this tree the
#: phrase short-circuits ahead of the label (agent review round 1, R6). What the
#: prefix buys is the reader that has no phrase vocabulary yet: a RELEASED app, or
#: another build of this branch, reads ``reason``/``to`` and resolves a departure it
#: cannot place to the build sentence instead of to nothing. It stays a prefix
#: rather than a new vocabulary word for that population, not because anything here
#: reads it.
_BUILD_OVERDUE_REASON = "stale-build-overdue"

#: What :func:`_leave_overdue` logs this departure as, and what it hands
#: ``_clean_exit`` as the journal's exit cause — the TOKEN
#: (``types.BUILD_DRAIN_OVERDUE_CAUSE``), never the sentence and never
#: ``drain.reason``: the row's cause is what a successor renders through
#: ``incidents.CUT_OFF_CAUSES``, and free text there is unrenderable, which is how
#: the bound stayed invisible to every durable surface (QA round 1, Q-2). The
#: sentence a person reads is composed in :func:`_leave_overdue`'s own log line
#: and in the ``CUT_OFF_CAUSES`` entry, in one place each.
_BUILD_OVERDUE_EXIT_REASON = BUILD_DRAIN_OVERDUE_CAUSE


#: What :func:`_leave_overdue` warns with — the human line, and the only place
#: the elapsed figure is stated while the departure is happening. The token above
#: is what is durable; this is what is readable.
_BUILD_OVERDUE_LOG = (
    "session runtime: %s; no movement reported from the work in flight for %.0fs "
    "(bound %.0fs); leaving without waiting for it"
)


def _transcript_footprint(transcript: object) -> "tuple[Any, ...]":
    """The newest durable row of each transcript kind, or ``()`` unreadable.

    THE TURN'S DURABLE FOOTPRINT. The writer is the step's own pairing boundary:
    ``Transcript.append_messages`` commits the assistant message and every tool
    result of one step together, so a turn that is stepping moves this — and a
    step that has not finished yet does not, which is the whole of the bound's
    residual (see :func:`_work_motion`). Read through ``latest_entry``, which is
    O(1) per kind and says so ("without copying history") — this runs on every
    reaper tick, and ``entries()`` would copy the whole transcript four times a
    second. A compaction or prune row counts too: both REWRITE history, and both
    are work the runtime did.

    ``_note_turn_boundary`` is NOT this signal and used to be named here; it
    writes the TURN JOURNAL's ``last_boundary`` (``serving.py``), which is
    evidence for a successor about which step completed, de-duplicated by tool
    name — not a transcript row and not a movement marker (agent review round 1,
    R7).

    The kind constants are imported HERE rather than at module scope because this
    module is RUN as ``__main__`` and its import block is the child's boot path —
    the reason ``_drain_inbox_into`` imports its own the same way.
    """
    latest = getattr(transcript, "latest_entry", None)
    if not callable(latest):
        return ()
    from local_operator.session.transcript import (
        ENTRY_COMPACTION,
        ENTRY_CUSTOM,
        ENTRY_MESSAGE,
    )

    newest: list[Any] = []
    for kind in (ENTRY_MESSAGE, ENTRY_COMPACTION, ENTRY_CUSTOM):
        try:
            entry = latest(kind)
        except Exception:  # noqa: BLE001 — unreadable state is not movement
            entry = None
        newest.append((kind, getattr(entry, "id", ""), getattr(entry, "ts", 0.0)))
    return tuple(newest)


def _job_footprint(session: object) -> "tuple[Any, ...]":
    """Every job row as ``(id, status, output_seq, progress)``, or ``()``.

    FOUR FACTS PER ROW, because between them they are the only way a JOB that is
    genuinely working can be told from one that has stopped, and the difference is
    the whole finding (agent review round 1, R1):

    * ``id``/``status`` — a job settling, a queued job admitted, a subagent lane
      opening or closing. This is also where "the subagent count changed" is
      read, and it is read as ROWS rather than through ``running_subagents()``
      because that predicate is a count derived from these same rows: one lane
      finishing as another starts is invisible to a count and visible here. Sorted,
      so a reordered table is not read as movement.
    * ``output_seq`` — the LIVE OUTPUT OFFSET ``AsyncJobManager.append_output``
      keeps (``harness/jobs.py``: "counts every char ever appended and never
      rewinds"). This is the one field that separates a background job that is
      PRINTING (a build, a test run, a mirrored bash child) from one whose child is
      alive at 0.1% CPU and silent — the shape that was force-cut before this
      signal existed.
    * ``progress`` — the child relay's activity string for a lane
      (``report_progress`` -> ``latest_details["progress"]``): coarser than a step
      boundary, but written only when the lane's own event stream moves
      (``harness/subagent.py``), so it separates a lane that is THINKING or
      RESPONDING from one parked inside a tool.

    A DEAD CHILD CANNOT ADVANCE ANY OF THESE, which is what keeps them honest as
    motion rather than noise: ``append_output`` is called from a pipe reader that
    ends when its pipes close (and once, at backgrounding, to seed what the
    foreground phase already collected), ``latest_details`` is written by the
    child's own relay, and a settled row's status does not move again. A child that
    dies leaves all three frozen, so the clock keeps running toward the bound —
    which is the failure mode anyway, and the one that must not be silent.

    ``is_busy`` already builds this list on the same tick, so the cost is one list
    comprehension over a table that is small by construction (capacity is capped),
    and a manager that cannot list is not movement.
    """
    manager = getattr(session, "jobs", None)
    listing = getattr(manager, "list", None)
    if not callable(listing):
        return ()
    try:
        rows = cast("list[Any]", listing())
    except Exception:  # noqa: BLE001 — unreadable state is not movement
        return ()
    footprint: list[Any] = []
    for job in rows:
        details = getattr(job, "latest_details", None)
        progress = details.get("progress", "") if isinstance(details, dict) else ""
        footprint.append(
            (
                str(getattr(job, "id", "")),
                str(getattr(job, "status", "")),
                int(getattr(job, "output_seq", 0) or 0),
                str(progress),
            )
        )
    return tuple(sorted(footprint))


def _spool_footprint(transcript: object) -> int:
    """How much the successor's spool holds, in bytes; ``-1`` when there is none.

    WHAT IT PROVES, and what it does not: a spool row is written by the drain's
    OWN delivery path when a peer message or a fired wake arrives, so a change
    here means work reached this runtime and was preserved for its successor —
    not that the turn in flight advanced. That is why it is the one field an
    OUTSIDE actor can move, and it is stated rather than hidden: a peer that keeps
    sending extends the bound by another window each time. It belongs in the
    clock anyway, because the reading it replaces — "the work is stalled, so the
    messages queueing behind it are irrelevant" — would cut a session whose
    successor is being kept fed, and because the write is the drain's own hop
    rather than any timer's tick.

    Size rather than rows: it is one ``stat`` against a file the drain appends to
    (``inbox.append_inbox`` opens with ``O_APPEND``), and a reader that counted
    rows would have to parse the file on every tick.
    """
    directory = getattr(transcript, "directory", None)
    if directory is None:
        return -1
    from local_operator.session.runtime.inbox import inbox_path

    try:
        return inbox_path(Path(directory)).stat().st_size
    except OSError:
        return -1


def _work_motion(handle: object) -> "tuple[Any, ...]":
    """Every observable sign that the work a drain is holding for has MOVED.

    MOTION, NOT WORK — and the distinction is where this mechanism is honest and
    where it is blind (agent review round 1, R1). What this tuple can read is what
    REACHES this process: a step's committed rows, a lane's roster movement, a
    job's own live output and activity, a spool write. A step that is running but
    reports nothing — a foreground tool call, whose result (and therefore whose
    transcript row) lands only when it returns, and which mirrors nothing into a
    job row unless it was backgrounded — is invisible here for its whole duration.
    The runtime cannot tell that step from a hung one, so the clock it feeds says
    "no movement reported", and the phrase it publishes says exactly that rather
    than asserting a cause (``types.LEAVING_FOR_BUILD_OVERDUE``).

    NOT A LIVENESS PROBE EITHER. The record heartbeat, the reaper's own tick, a
    viewer's repaint and ``is_streaming`` all keep reporting for a session whose
    work has stopped — the incident's runtime answered ``busy`` and ``live`` for
    two hours while three subagent lanes sat behind a bash child that had not
    printed anything in 23 minutes. A field belongs in this tuple only if
    something OTHER than a clock changes it, only if a change means the work
    advanced, and only if a DEAD child cannot produce it (see
    :func:`_job_footprint` for the three job fields against that bar). Five are
    read:

    * the transcript's newest row per kind — the turn's durable footprint;
    * the subagent ROSTER GENERATION — the one LANE-level signal that reaches the
      parent: a lane's every completed assistant message, model change and
      lifecycle event bumps it (``Session._schedule_subagent_persist``, driven by
      the child relay in ``harness.subagent``), so a lane that is STEPPING keeps
      this moving while its parent's own transcript stays frozen for the whole
      lane. Read as a private attribute because there is no public seam for it,
      and because the durable rendering of the same fact (the roster sidecar) is
      a COALESCED, threaded write whose latency belongs to its own writer rather
      than to the work — the failure mode is the same one ``_idle_for_refresh``
      documents for a sampled predicate, on a signal that has a cheaper exact
      source in this very process;
    * the job rows, as ``(id, status, output_seq, progress)`` per row — a job that
      is PRINTING, or whose lane is stepping, is a job that is moving;
    * the spool, as the inbox file's own size (:func:`_spool_footprint`).

    UNREADABLE STATE IS NOT MOVEMENT: a probe that raises contributes a constant
    and the clock keeps running. The direction is deliberate, and it is the
    OPPOSITE of the fail-closed rule the residency predicates use — those answer
    "may I destroy this work?" and must say no when unsure, while this tuple only
    decides when to stop waiting for a session whose own work cannot tell anyone
    it is alive, which is the case the bound exists for.
    """
    session = getattr(handle, "_session", None)
    transcript = getattr(session, "transcript", None)
    return (
        _transcript_footprint(transcript),
        getattr(session, "_subagent_roster_generation", None),
        _job_footprint(session),
        _spool_footprint(transcript),
    )


@dataclass
class _DrainProgress:
    """Has the work a drain is holding for moved lately? ``_drain_for``'s clock.

    ``moved_at`` is the instant of the last observation that DIFFERED from the
    one before it, and it is what :data:`types.BUILD_DRAIN_PROGRESS_S` is
    measured against. Nothing here is advanced by the caller's own cadence, and
    that is what makes this a bound on the WORK rather than a second timeout: a
    runtime whose turn is stepping, whose lane is reporting, whose jobs are
    settling or whose spool is filling keeps pushing ``moved_at`` forward, so a
    hold that is moving is never cut however long it runs.

    ``overdue`` is the drain's own record that its hold was ended by the
    backstop, and it is also the guard that keeps the rung from running twice.
    """

    motion: "tuple[Any, ...]" = ()
    moved_at: float = 0.0
    overdue: bool = False

    @classmethod
    def started(cls, handle: object, at: float) -> "_DrainProgress":
        """Open the clock on the drain's first tick, already sampled."""
        return cls(motion=_work_motion(handle), moved_at=at)

    def sample(self, handle: object, at: float) -> bool:
        """One observation. True when the work moved since the last one."""
        motion = _work_motion(handle)
        if motion == self.motion:
            return False
        self.motion = motion
        self.moved_at = at
        return True

    def stalled_s(self, at: float) -> float:
        """How long this drain's work has shown no movement, in seconds."""
        return at - self.moved_at


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
    boot: BuildStamp | None = getattr(runtime, "_boot_build", None)
    to = poll.newer.label() if poll.newer is not None else ""
    reasons = _drain_reasons(poll)
    detail = _drain_detail(reasons, boot, poll.newer)
    if poll.files_gone:
        reason = "retiring: the build this process loaded is gone from disk"
    elif to:
        reason = "retiring for " + to
    else:
        reason = "retiring for a build replaced on disk"
    return await _commit_to_leaving(
        handle,
        runtime,
        stop,
        label="stale-build",
        reason=reason,
        detail=detail,
        to=to,
        loaded=boot.label() if boot is not None else "<unknown>",
        cause="runtime-retired",
        stagger_s=random.uniform(0, _build_stagger_seconds()),  # noqa: S311 — jitter, not security
        leaving=LEAVING_FOR_BUILD,
        reasons=reasons,
        boot=boot,
    )


def _drain_loaded_label(runtime: object) -> str:
    """What the runtime loaded, for a departure's log line.

    One helper rather than the same inline ternary at every call site: the
    "<unknown>" fallback is what makes the line readable for a runtime whose
    boot stamp is unreadable (which is exactly the runtime an investigation
    wants to see named).
    """
    boot: BuildStamp | None = getattr(runtime, "_boot_build", None)
    return boot.label() if boot is not None else "<unknown>"


async def _commit_to_leaving(
    handle: object,
    runtime: object,
    stop: asyncio.Event,
    *,
    label: str,
    reason: str,
    detail: str,
    loaded: str,
    to: str = "",
    cause: str = "runtime-retired",
    stagger_s: float = 0.0,
    leaving: str = "",
    reasons: "tuple[str, ...]" = (),
    boot: "BuildStamp | None" = None,
) -> "_Drain | None":
    """Announce a departure, then stop admitting work. ``None``: not ours.

    THE ONE PLACE A DEPARTURE IS COMMITTED TO, whichever trigger asked for it —
    the build replaced on disk (:func:`_begin_drain`) or a termination signal
    (:func:`_drain_for_signal`). Two triggers, ONE state, and that is the point
    rather than tidiness: the ``retiring`` frame's ``draining`` flag and the
    phrase the fleet surfaces read (``SessionRecord.leaving``) are two
    renderings of this single commit, written by the one call below
    (``RuntimeServer.announce_retiring``), so no surface can report a drain that
    another surface does not, and the bound a caller imposes is the only thing
    the two triggers do differently.

    ANNOUNCE FIRST, LATCH SECOND, and the order is invariant (iii): a viewer
    must learn the runtime is leaving BEFORE it starts refusing, or the first
    refused message reads as an error rather than as a handover. The latch
    (``ServingSessionHandle.begin_drain``) is the commit — from that instant no
    new work is admitted while the live turn, its subagents and its jobs run to
    completion, and the exit waits for exactly that. It deliberately does NOT
    write the cut-off cause (that is ``begin_retire``, reached at the boundary
    by :func:`_drain_for`), which is what makes it safe to commit while a turn
    the drain exists to save is still running.

    ``leaving`` IS THE RECORD'S HALF OF THE SAME COMMIT, and passing it here
    rather than writing the record from the trigger is the reconciliation PR
    #1108 forced: that PR landed its own drain state, whose ``draining`` flag on
    the ``retiring`` frame is what the APP paints its notice from at frame
    receipt, while this branch had added ``SessionRecord.leaving`` for the fleet
    surfaces (``lop sessions``, ``/info``, the catalogue, the stop ladder's
    refusal). Two renderings of one fact, so ONE writer: the frame is
    authoritative for the app and the phrase is authoritative for the fleet,
    and ``announce_retiring`` writes the phrase and sends the frame in the same
    call — a trigger cannot publish one without the other, which is what makes
    a disagreement impossible rather than merely unlikely. A trigger passes its
    OWN words, because the two reasons are not interchangeable and either
    phrase would be a lie about the other trigger; the frame's ``reason`` label
    does the same job on the wire.

    A handle without the latch (an older host, a reduced test double) does NOT
    drain. The bound's whole guarantee is that admissions stop; a runtime that
    kept accepting work it had already decided to walk away from would be
    serving the replaced build for longer, not less. It keeps the old behaviour
    — keep serving, ask again on the next check — which is the status quo rather
    than a regression.

    A LATCHED DRAIN IS NEVER TAKEN TWICE. The signal path and the reaper can
    both reach here for one departure (a sweep arriving while a hard-stale
    runtime is already draining, or the reverse), and ``_drain_for`` is not
    written to be run twice for one exit: the first commit owns the exit, and
    the second trigger waits for it or bounds it.

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
        # invite a viewer to re-engage a session that is on its way out for a
        # reason the disposal has stated itself.
        return None
    if getattr(handle, "_draining", False):
        # A departure is already committed to and announced. Its own call site
        # owns the exit; a second ``_Drain`` would race it into ``_clean_exit``.
        return None
    # THE SUCCESSOR-SPREAD DELAY IS THE CALLER'S, because the two triggers do
    # not share the reason for it: a BUILD change retires a whole fleet within
    # seconds of itself (every runtime sees the same stamp), so the build path
    # draws a jittered ``BUILD_STAGGER_S`` slice here and the successors spawn
    # spread out. A signal retires only what it hit, and those exits are spread
    # by the work each one is finishing — so the signal path draws nothing, and
    # a signalled runtime leaves the moment its own turn ends instead of holding
    # its process (and its record) for up to ``BUILD_STAGGER_S`` afterwards.
    delay = stagger_s
    if stop.is_set():
        # BEFORE the log and before the announce, and both orders are the point:
        # the log line below says no new work will be admitted and the frame
        # says the same to the operator, and a stop that has already landed
        # means nothing will refuse — the row stays in the transcript and the
        # claim is false (review round 4, MINOR 3, and NIT 3 one surface down).
        #
        # This closes the COMMONEST of the two ways the latch can be denied, not
        # the class: a ``_disposing`` flip inside ``begin_drain`` still leaves
        # the frame sent with no latch taken (review round 5, NIT 1), which the
        # announce-first ordering — invariant (iii) — makes structurally
        # unavoidable rather than fixable here.
        return None  # its path owns the exit
    logger.info(
        "session runtime: %s (loaded %s; %s); no new work will be admitted, in-flight work "
        "finishes first (pid %d)",
        reason,
        loaded,
        detail,
        os.getpid(),
    )
    announce = getattr(runtime, "announce_retiring", None)
    if callable(announce):
        try:
            # ``draining=True``: this frame is the operator's ONLY warning
            # before admissions start being refused, so it must carry the fact
            # that refusals are in force from the latch below until the drain
            # empties. A viewer that inferred that from its own state inferred
            # "cold", which is true of every handover (QA round 3, Q-1).
            await cast(Callable[..., Awaitable[None]], announce)(
                label, to=to, draining=True, leaving=leaving
            )
        except Exception:  # noqa: BLE001 — a viewer that misses this goes cold the slow way
            logger.debug("retiring announcement failed", exc_info=True)
    if stop.is_set():
        return None  # a stop arrived during the announcement; its path owns the exit
    try:
        latched = begin_drain(cause, detail)
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
        cause=cause,
        reasons=reasons,
        boot=boot,
    )


async def _drain_for(
    drain: _Drain,
    handle: object,
    runtime: object,
    stop: asyncio.Event,
    *,
    now: float | None = None,
) -> bool:
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
    the work.

    AND ONLY BY THE WORK THAT IS STILL MOVING — see :func:`_leave_overdue`, the
    one rung here that draws a bound at all. It reads no clock of the drain: the
    clock it reads is reset by every observable sign that the work advanced
    (:func:`_work_motion`), so it cannot fire on a turn that is merely long, and
    what it bounds is not the hold but STALENESS. The state it exists for has no
    other exit: ``is_busy()`` counts a gate parked on a user and a lane parked
    behind a child process, both of which can hold for hours, and a drain that
    holds forever takes its session with it (measured: 2 h and still refusing,
    ``state=wedged``, three lanes stalled behind a 23-minute bash child).

    The viewer term of :func:`_should_exit` is deliberately absent, exactly as
    it is absent from ``may_refresh``. The ``retiring`` frame went out at drain
    start, so a viewer re-engages onto the new build instead of holding this
    one — and holding for it is what kept five-hour-stale runtimes resident.
    PINNED BY TEST (``test_buildwatch_progress``): with an interactive attach
    client connected, the exit still happens at the first idle instant, and a
    viewer's presence resets no clock either.

    The exit commits through ``begin_retire``, so the last instant still says
    "idle" by construction and the cut-off note a retirement owes is written by
    the rung that owns it. Retried every ``REAP_CHECK_S`` until it lands.

    THE DETAIL IS RE-READ HERE rather than replayed from the latch
    (:func:`_drain_detail_at_exit`): this path's whole shape is that the exit
    waits for hours of work, and a reason that names the build pair of the
    latch asserts a transition the install left long ago.

    ``now`` is the caller's clock, injectable for the one test that has to cross
    a fifteen-minute bound without waiting it out — the same convention
    ``_BuildWatch.poll`` uses.
    """
    at = time.monotonic() if now is None else now
    if drain.progress is None:
        drain.progress = _DrainProgress.started(handle, at)
    else:
        drain.progress.sample(handle, at)
    # The stagger is respected before ANY exit, the forced one included: sixteen
    # runtimes that all went stale at once must not spawn sixteen successors
    # together, which is the only reason this line is above the backstop rather
    # than below it.
    if at < drain.stagger_until:
        return False
    if not _idle_for_refresh(handle):
        if drain.progress.stalled_s(at) >= BUILD_DRAIN_PROGRESS_S:
            return await _leave_overdue(
                drain, handle, runtime, stop, progress=drain.progress, at=at
            )
        return False
    begin_retire = getattr(handle, "begin_retire", None)
    if callable(begin_retire) and not begin_retire(drain.cause, _drain_detail_at_exit(drain)):
        logger.info("session runtime: work arrived as the drain closed; keeping")
        return False
    logger.info("session runtime: %s; exiting cleanly", drain.reason)
    await _hand_wakes_to_successor(handle)
    await _clean_exit(handle, runtime, reason=drain.reason)
    stop.set()  # amain's wait() returns; exit code stays 0
    return True


async def _leave_overdue(
    drain: _Drain,
    handle: object,
    runtime: object,
    stop: asyncio.Event,
    *,
    progress: _DrainProgress,
    at: float,
) -> bool:
    """The backstop: leave by FORCE, through the signal drain's own exit rung.

    Reached only when :data:`types.BUILD_DRAIN_PROGRESS_S` has passed with no
    movement in ANY of the signs :func:`_work_motion` reads — a hold whose work
    has stopped reporting anything at all, which is the state a build drain would
    otherwise sit in forever: ``is_busy()`` keeps answering True for a lane parked
    behind a bash child, the drain's promise ("in-flight work finishes first") is
    only as good as that work's willingness to finish, and nothing else in the
    drain draws any bound. The runtime is still refusing every admission while it
    holds, and a successor cannot be engaged while the predecessor holds the
    transcript lease, so not firing here does not cost a slow handover — it costs
    the session.

    THE EXIT IS THE SIGNAL DRAIN'S, in all three parts, and it is deliberately not
    a new exit path:

    * the record and the frame are RE-PUBLISHED through ``announce_retiring`` — the
      same one commit that writes ``SessionRecord.leaving`` and sends the frame —
      so ``lop sessions`` stops advertising a wait the runtime has given up on,
      and the label and phrase BOTH name the bound (see the constants above). The
      frame is the ordinary ``retiring`` one a drain already sent at its start, so
      a viewer that went cold on that first frame sees the same event again rather
      than a new one it has to learn;
    * the wakes this drain swallowed are handed to the successor FIRST, exactly as
      the clean rung hands them over (:func:`_hand_wakes_to_successor`). Not an
      extra: the wakes are the ones whose fire RETIRED their schedule, so a
      handover skipped here does not defer the reminder, it loses it, and the
      rung that cuts a turn is the last one that should also drop the user's
      scheduled work;
    * a gate still parked on a user's answer is DENIED rather than left holding a
      process that is leaving: the turn it belongs to is being cut, and amain's own
      direct-dispose block denies for exactly this reason before its dispose. It is
      NOT the memo's "do not deny from the drain" case — that refusal is about a
      drain that is still trying to preserve its turn, which is the case this rung
      has already given up on;
    * the exit runs ``_clean_exit``, the one convergence point every planned exit
      already goes through, with the TOKEN ``types.BUILD_DRAIN_OVERDUE_CAUSE`` as
      its reason — not a sentence. The row is the only account of this departure
      that outlives the process (``lop sessions --json`` returns an empty list
      ~97 ms after the escalation because the record goes with it), so the cause
      has to be a token the taxonomy can RENDER: ``death_verdict`` narrates a
      recorded non-signal cause ahead of its own inferences, and
      ``CUT_OFF_CAUSES`` turns that token into the sentence a successor repeats
      (agent review round 1, R3; QA round 1, Q-2). Written as free text before
      this, it reached the row and nothing read it;
    * the why-now the cut-off note brands the turn with is RE-READ here, and the
      CAUSE it brands it with becomes this departure's own token, so the turn the
      escalation cuts is narrated as a bounded handover on every surface that
      repeats a cut-off — the live error row, the attention record and the
      successor's incident (agent review round 1, R2/R3; QA round 1, Q-2). The
      latch's cause is still the truth for the WAIT; it is the wrong word for the
      CUT.

    WHY THE DRAIN'S CAUSE DOES NOT CHANGE, against the memo's "classified by
    ``SIGNAL_DRAIN_CAUSE``". ``begin_drain`` is the latch that token lives on, and
    calling it a second time is not a rename: it re-runs
    ``Session.retire_wakes_to_inbox``, which STARTS A FRESH ``_wake_rearms`` list —
    discarding the one-shot wakes this drain has already swallowed and would have
    handed to its successor at the exit, so a reminder that fired between the latch
    and this rung would be lost silently. Writing the handle's private
    ``_retiring_cause`` instead would be the same latch minus its bookkeeping, and
    it would ALSO make ``ServingSessionHandle._retiring_refusal`` name this
    departure SIGNALLED ("This session was signalled to stop"), which is the only
    token that accessor maps and maps for exactly this reason: a false sentence
    about which trigger took a session away is the class of falsehood agent review
    round 4 (MAJOR-2) filed in the other direction. The build drain's own
    ``runtime-retired`` is TRUE of this exit — the runtime is leaving so the next
    engage runs the build on disk — and that a BOUND ended the hold is carried by
    the phrase, which is the primary carrier of which trigger committed a drain.
    """
    if progress.overdue:
        # The drain is not exited twice: a second caller (the signal drain's own
        # loop, in principle) gets the same answer without a second announcement
        # or a second disposal.
        return True
    stalled = progress.stalled_s(at)
    progress.overdue = True
    logger.warning(_BUILD_OVERDUE_LOG, drain.reason, stalled, BUILD_DRAIN_PROGRESS_S)
    announce = getattr(runtime, "announce_retiring", None)
    if callable(announce):
        try:
            await cast("Callable[..., Awaitable[None]]", announce)(
                _BUILD_OVERDUE_REASON,
                to=drain.to,
                draining=True,
                leaving=LEAVING_FOR_BUILD_OVERDUE,
            )
        except Exception:  # noqa: BLE001 — a viewer that misses this goes cold the slow way
            logger.debug("overdue announcement failed", exc_info=True)
    # THE DEPARTURE'S ATTRIBUTION IS RE-STATED HERE, and both halves matter.
    #
    # The WHY-NOW is RE-READ at the exit exactly as the quiet rung does it
    # (:func:`_drain_detail_at_exit`), because this rung is only ever reached after
    # hours of a hold: the pair the latch composed can name builds the install left
    # long ago, and a why-now naming a build that has not been on disk for hours is
    # its own false report. A drained runtime has NO detail at all today —
    # ``begin_drain`` takes a ``detail`` and never stores it (only ``begin_retire``
    # assigns ``_retiring_detail``) — so the cut-off note this feeds was branded
    # with an empty parenthetical (agent review round 1, R2).
    #
    # The CAUSE becomes the token, which is what makes the CUT legible: the note is
    # consumed by the next ``AgentEndEvent`` as ``Session._cut_off_cause`` and
    # rendered through ``incidents.CUT_OFF_CAUSES`` on every surface that repeats a
    # cut-off — the live "Stopped with an error" row, the attention record, and the
    # successor's ``session_incident``. Left as the latch's ``runtime-retired``, a
    # turn cut BY A BOUND narrated the sentence every ordinary build handover
    # leaves, so the fact the operator needs was invisible on every durable surface
    # (QA round 1, Q-2: the record is gone ~97 ms after the escalation, so these are
    # the only places left to look).
    #
    # NOT A RE-LATCH, and NOT ``SIGNAL_DRAIN_CAUSE``. The round-1 objection stands:
    # ``begin_drain`` re-runs ``retire_wakes_to_inbox``, whose one-shot re-arms only
    # ``_hand_wakes_to_successor`` writes — a second call would drop a reminder this
    # drain had already swallowed; and classifying a build departure as
    # ``runtime-shutdown`` would make ``_retiring_refusal`` name a trigger that did
    # not happen. This is a token of its own, which that accessor maps to NO trigger
    # (its only named departure is the signal), so a refusal here is exactly as
    # unnamed as it already was for a build drain, and the phrase the frame carries
    # is what resolves the trigger on the far side. The two truthiness readers of
    # this field (the admission gate, the spool decision) see a non-empty string
    # either way and cannot tell the difference.
    detail = _drain_detail_at_exit(drain)
    try:
        setattr(handle, "_retiring_cause", _BUILD_OVERDUE_EXIT_REASON)
        setattr(handle, "_retiring_detail", detail)
    except Exception:  # noqa: BLE001 — the note is evidence, never a gate on the exit
        logger.debug("could not hand the exit attribution to the cut-off note", exc_info=True)
    deny = getattr(handle, "_deny_pending_gates", None)
    if callable(deny):
        try:
            deny()
        except Exception:  # noqa: BLE001 — the exit must not be held by a failed denial
            logger.debug("gate denial failed at the overdue exit", exc_info=True)
    await _hand_wakes_to_successor(handle)
    await _clean_exit(handle, runtime, reason=_BUILD_OVERDUE_EXIT_REASON)
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


#: The wire label a SIGNAL-driven retirement announces. Deliberately not one of
#: the build labels (``stale-build``, ``moved``): both of those mean "a NEWER
#: build is owed", and a viewer that read that off an ordinary SIGTERM would be
#: told the install had moved when it had not. The frame itself is the ordinary
#: ``retiring`` one — what a viewer does with it (engage a successor) is exactly
#: right here, because a runtime that has been signalled IS leaving.
_SIGNAL_DRAIN_REASON = "shutdown-drain"


async def _drain_for_signal(
    handle: object, runtime: object, stop: asyncio.Event, *, sig_name: str
) -> None:
    """Leave after a termination signal — at the next boundary, or at the bound.

    Called by ``amain``'s signal handler INSTEAD of ``stop.set()`` when
    :func:`_work_in_flight` is true. It is the fix for the asymmetry the
    2026-09-14 incident measured: the reaper refused to exit while a turn was in
    flight, and the signal handler disposed anyway, so a broadcast SIGTERM
    destroyed work the very same process had just decided not to disturb.
    SIGTERM is catchable and already handled on this loop, so the receiver can
    afford to be the polite one — which is the whole point: the sender of a
    sweep is unnamed and cannot be taught manners, while this process always
    knows whether it is mid-turn.

    IT COMMITS THROUGH :func:`_commit_to_leaving`, the same seam the
    build-replaced drain commits through, so a signalled runtime is in ONE state
    and every surface reports that state the same way: the ``retiring`` frame
    carries ``draining=True`` for the app, and the same call publishes the
    phrase the fleet surfaces read. What the two triggers do differently is the
    bound this function imposes — the build drain waits on its work alone.

    Three properties, each load-bearing:

    * THE WAIT IS BOUNDED by ``types.SIGNAL_DRAIN_S``. A wedged or runaway
      runtime must not become unkillable, and a signal must never turn into an
      unbounded wait. The deadline is absolute and the boundary is re-checked
      every ``REAP_CHECK_S``, so the wait ends at the FIRST tick after the work
      finishes.

    * THE COMMIT IS SAFE TO TAKE NOW, and PR #1108 is what made it so.
      ``begin_drain`` refuses new admissions and spools peer messages without
      touching the turn in flight, and it deliberately does NOT write the
      cut-off cause — that is ``begin_retire``, which :func:`_drain_for` reaches
      only at the boundary. A turn that finishes inside the window therefore
      cannot be relabelled an error. Round 1 had to announce at the boundary
      instead of committing at the signal, because the only latch that existed
      then did both jobs at once.

      WHICH latch that argument is about is worth spelling out, because only one
      of the two conceivable spellings could do that harm. Calling
      ``begin_retire`` BEFORE the wait would latch NOTHING: it returns ``False``
      the moment ``may_refresh()`` is non-empty, and ``may_refresh()`` reports
      ``"busy"`` whenever ``is_busy()`` is true (``serving.py``) — which is
      precisely the case this function exists for, since a signal only reaches
      here with work in flight. That spelling is harmless rather than correct,
      and saying so is not pedantry: it looks like the fix, and a reader who
      "simplified" the boundary latch into it would silently lose the cut-off
      cause for a turn the bound really does destroy. The hazard belongs to a
      latch that writes the cause directly (``note_cut_off`` plus
      ``_retiring_cause``), which is the one this path still does not take.

      What the early commit costs is stated rather than hidden: work that
      arrives mid-drain is refused (``prompt``) or spooled for the successor
      (``peer_message``) — the same behaviour the build drain has, from the same
      latch — where round 1 admitted it and cut it if the bound expired. And a
      refusal is only honest if the operator was told, which is why the commit
      announces the moment it is taken.

    * ON EXPIRY THE DISPOSAL IS THE ORDINARY ONE: no second announcement and no
      clean-exit convergence, so the dispose rung notes ``runtime-shutdown`` for
      the turn it aborts — the token this drain's own latch carries, so a turn
      is classified identically whether the drain expired or never ran. Say so
      HERE rather than at the exit line, because the fact that matters
      afterwards is that the BOUND cut this turn and not the signal.

    * THE PENDING EXIT IS PUBLISHED *BEFORE* THE WAIT, unlike the retirement
      latch — and the two are deliberately not the same moment. The latch is a
      statement about a turn's OUTCOME (it brands the next end), so it must wait
      for the boundary; ``leaving`` is a statement about the PROCESS (a signal
      arrived and is being honoured), which is already true the instant this
      function starts. Publishing it here is what makes the drain visible at
      all: without it a signalled-but-working runtime spends up to
      ``SIGNAL_DRAIN_S`` looking like an ordinary busy one on every surface an
      operator reads, and the natural remedy for "it is still working" — a
      plain ``lop stop`` — cuts the very turn the drain is finishing (U1/U2, PR
      #1141). See ``SessionRecord.leaving``; ``lop stop`` refuses on it too.

    ONE GAP IS DELIBERATE and is stated rather than closed: the predicate is
    read above and then ``announce_retiring`` is awaited, so a ``prompt``
    landing inside that window is admitted before ``begin_drain`` — the latch
    adjacent to the announce — starts refusing. What protects a turn taken
    through that gap is the BOUNDARY latch, not a second read here: ``_drain_for``
    re-reads the idle gate immediately before ``begin_retire`` (through
    ``_idle_for_refresh``, the same ``may_refresh`` gate the reaper samples) and
    skips the commit when work arrived, so such a turn is normally WAITED OUT and
    ``stop.set()`` cuts it only if ``SIGNAL_DRAIN_S`` expires first (pinned by
    ``test_process_refresh.py::test_work_arriving_after_the_announce_keeps_the_runtime``).
    Re-reading the gate at the announce would not close the window — the signal
    has already decided that this process leaves, so a re-read could only relabel
    a cut, never save the turn — and the re-read that DOES matter is the one at
    the boundary, where it can still refuse the exit. So the window stays,
    bounded by one socket write, and the label stays honest
    (``runtime-shutdown``).

    The signal name is passed for the log only; the disposal itself stays where
    it is and in the order it already had — ``amain`` owns deny -> dispose ->
    aclose, and this function only decides WHEN ``stop`` is set.
    """
    # ``time.monotonic`` rather than ``loop.time``: the deadline is compared
    # against a clock nobody can move, and the reaper's own waits use this one.
    deadline = time.monotonic() + SIGNAL_DRAIN_S
    logger.info(
        "session runtime: %s arrived with work in flight; leaving at the next boundary "
        "(bound %.0fs)",
        sig_name,
        SIGNAL_DRAIN_S,
    )
    if getattr(handle, "_draining", False) and not stop.is_set():
        # A departure is ALREADY committed to — the build on disk was replaced
        # first — so its own call site owns the exit and this signal adds only
        # the one thing that path does not have: a bound. Waiting on ``stop``
        # rather than on the work is what keeps the two from racing into
        # ``_clean_exit``; the drain the reaper is running sets it.
        while not stop.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(REAP_CHECK_S)
        if not stop.is_set():
            logger.warning(
                "session runtime: %s drain bound (%.0fs) expired with work still in flight; "
                "disposing now",
                sig_name,
                SIGNAL_DRAIN_S,
            )
            stop.set()
        return
    # ``leaving=`` is how the pending exit reaches the fleet surfaces, and the
    # seam is the ONLY writer of both halves of that fact: it publishes the
    # phrase on the record and sends the ``draining=True`` frame in one call, so
    # no surface can report a drain another surface does not (see
    # ``RuntimeServer.announce_retiring``). Best-effort inside that call: a
    # record that could not be rewritten must not stop a runtime from honouring
    # the signal it was given.
    drain = await _commit_to_leaving(
        handle,
        runtime,
        stop,
        label=_SIGNAL_DRAIN_REASON,
        reason=f"leaving after {sig_name}",
        detail=f"{sig_name}: drained to the end of the turn in flight",
        loaded=_drain_loaded_label(runtime),
        cause=SIGNAL_DRAIN_CAUSE,
        leaving=LEAVING_ON_SIGNAL,
        # The detail is the whole parenthetical on this trigger — a signal makes
        # no claim about any build — so it is carried as the one phrase the
        # exit re-renders, with no boot stamp (:func:`_drain_detail_at_exit`
        # then asserts no pair, which is what this path means).
        reasons=(f"{sig_name}: drained to the end of the turn in flight",),
    )
    if drain is None:
        if stop.is_set() or getattr(handle, "_disposing", False):
            # An exit is already under way and owns the ordering; a second
            # ``stop.set()`` here would only be noise.
            return
        # NO LATCH ON THIS HANDLE (a reduced host or a test double). The old
        # fallback, kept because the guarantee it buys is the point of this
        # function: wait the work out, bounded, announcing nothing — a runtime
        # that never latched has nothing to refuse, so there is no handover to
        # advertise — and let the disposal own the exit.
        while _work_in_flight(handle) and time.monotonic() < deadline:
            await asyncio.sleep(REAP_CHECK_S)
        if _work_in_flight(handle):
            logger.warning(
                "session runtime: %s drain bound (%.0fs) expired with work still in flight; "
                "disposing now",
                sig_name,
                SIGNAL_DRAIN_S,
            )
        stop.set()
        return
    while True:
        if await _drain_for(drain, handle, runtime, stop):
            return
        if stop.is_set():
            return
        if time.monotonic() >= deadline:
            break
        await asyncio.sleep(REAP_CHECK_S)
    logger.warning(
        "session runtime: %s drain bound (%.0fs) expired with work still in flight; "
        "disposing now",
        sig_name,
        SIGNAL_DRAIN_S,
    )
    stop.set()


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

    THE ROW'S ``source`` DECIDES WHO IS SPEAKING, and the two are delivered by
    different paths on purpose. A ``SOURCE_USER`` row is the OWNER's own prompt,
    which a draining runtime spooled instead of refusing: it is run through
    ``handle.prompt`` — the ordinary admission, on the build that is taking
    over — because the alternative (``receive_peer_message``) wraps the user's
    own words in a peer-session provenance envelope for the model and paints a
    ``peer`` card for a message the user typed in this session. Every other row
    is a peer's, exactly as before.

    Best-effort per message: one malformed or rejected row must not stop the
    rest, and none of it may prevent the runtime from starting.
    """
    from local_operator.session.runtime.inbox import SOURCE_USER, drain_inbox

    session = getattr(handle, "_session", None)
    directory = getattr(getattr(session, "transcript", None), "directory", None)
    if directory is None:
        return 0
    # AN UNENGAGED SESSION KEEPS ITS SPOOL. Rows can reach this inbox from a
    # sender on an older build (one whose record read is absent-as-``True``) or
    # from before any record existed, and draining them HERE would put a peer
    # row at the head of a conversation its owner has not started — the boot
    # drain runs before the socket listens, which is also before the owner's
    # first turn. Leaving the file alone is what makes the delivery happen
    # instead at that first turn (``Session._drain_spooled_peer_inbox``, once
    # the owner IS engaged), and THAT drain runs after the turn's own messages
    # are durable, so the deferred rows land behind the owner's opening prompt
    # rather than opening the history (review round 1, F-2: an earlier revision
    # of this comment claimed that ordering while the drain still ran at the top
    # of the turn pipeline). The engaged case — including the handover, where a
    # draining runtime spools for a successor — drains exactly as it always did.
    from local_operator.session.runtime.engagement import (
        TRANSCRIPT_FILENAME,
        durable_conversation_path,
    )

    if not durable_conversation_path(directory / TRANSCRIPT_FILENAME):
        logger.info(
            "inbox drain skipped for %s: no durable history yet, so the person "
            "has not engaged this session; rows stay for the first turn",
            directory,
        )
        return 0
    try:
        lines = await asyncio.to_thread(drain_inbox, directory)
    except Exception:  # noqa: BLE001 — a bad spool must not block the runtime
        logger.warning("inbox drain failed", exc_info=True)
        return 0
    probed = getattr(handle, "receive_peer_message", None)
    if not lines:
        return 0
    receive = cast(Callable[..., Awaitable[str]], probed)
    delivered = 0
    # Rows of ONE batch carrying the same owner ``command_id``, which the
    # ``drain_inbox`` contract makes reachable: the file is emptied by a read,
    # so a crash between the read and its receipt re-delivers the batch, and a
    # client retry can spool the same message twice. The durable index is the
    # authority for the turn arm (see ``_run_owner_prompt``); this covers the
    # rows whose delivery has not reached the transcript yet, i.e. one queued
    # behind another in THIS loop.
    seen_owner_ids: set[str] = set()
    for line in lines:
        owner_row = getattr(line, "source", "") == SOURCE_USER
        try:
            if owner_row:
                await _run_owner_prompt(handle, line, seen=seen_owner_ids)
            elif callable(probed):
                await receive(
                    line.text,
                    mode="mailbox",
                    wake=bool(getattr(line, "wake", False)),
                    sender=line.sender,
                )
            else:
                raise RuntimeError("this handle cannot receive a spooled peer message")
            delivered += 1
        except Exception:  # noqa: BLE001 — one bad row is not the others' problem
            if owner_row:
                # LOUDER, AND FOR A DIFFERENT REASON: the owner's message is
                # the one whose receipt already told the user it would run, and
                # it is in no transcript but this spool's — so a swallowed row
                # is a message destroyed while the runtime said it was kept
                # (QA round 1, Q-1). The row is named by its own id so an
                # operator can find it.
                logger.error(
                    "spooled OWNER message %s could not be delivered",
                    getattr(line, "command_id", "") or "<no id>",
                    exc_info=True,
                )
            else:
                logger.warning("spooled message could not be delivered", exc_info=True)
    if delivered:
        logger.info("delivered %d spooled message(s) at open", delivered)
    return delivered


async def _run_owner_prompt(handle: object, line: Any, *, seen: set[str]) -> None:
    """Run one spooled OWNER prompt on this runtime, at most once.

    The continuation of the drain's own promise: a runtime that latched a
    stale-build drain spools the owner's message rather than refusing it
    (``serving.ServingSessionHandle.prompt``), and this is where the successor
    makes good on that — the ordinary admission, through the same ``prompt``
    every front end uses, so the row it writes is the user row it would have
    been and carries the command id the viewer painted it under.

    IDEMPOTENT BY THE DURABLE INDEX, not by this file. ``inbox.jsonl`` is
    emptied by a read, but the SAME message can legitimately be spooled twice
    (a client retried the refused op, a crash landed between the append and its
    receipt) and the identity it carries is the append-only one — so
    ``has_admitted_command`` answers here exactly as it does for a retried wire
    prompt on ``server._already_admitted``. Without this the second row
    appended a second user turn. ``seen`` closes the window the index cannot:
    two rows of one batch carrying the same id, where the first is still
    in flight (its append is behind a turn that is already running) when the
    second is read.

    MID-TURN IS THE ORDINARY CASE HERE, not an edge. The rows are delivered in
    write order and a peer ``mailbox``+``wake`` row DRIVES A TURN, so an owner
    row spooled after one lands while the session is streaming — where
    ``Session.prompt`` rejects outright ("session is already streaming; use
    steer() to inject mid-turn"). That rejection used to be swallowed with the
    message inside it, while the receipt the user got said it would run and the
    boot still counted the row as delivered (QA round 1, Q-1). The owner's own
    words join the turn in flight instead, which is what the sibling first-turn
    drain already does (``Session._run_spooled_owner_prompt``) and the strongest
    thing this process can honestly do with them.

    Raises rather than swallowing: the caller's per-row handler logs it by name
    and moves on.
    """
    prompt = getattr(handle, "prompt", None)
    if not callable(prompt):
        raise RuntimeError("this handle cannot run a spooled prompt")
    run = cast(Callable[..., Awaitable[Any]], prompt)
    command_id = str(getattr(line, "command_id", "") or "")
    if command_id and command_id in seen:
        logger.info("spooled prompt %s is a repeat within this batch; skipping", command_id)
        return
    admitted = getattr(handle, "has_admitted_command", None)
    if command_id and callable(admitted) and admitted(command_id):
        logger.info("spooled prompt already in the transcript; not running it twice")
        return
    try:
        if command_id:
            await run(line.text, command_id=command_id)
            # RECORDED AFTER THE DELIVERY, not before it: the batch's own repeat
            # only needs suppressing when the first row LANDED. The file's
            # contract is at-least-once, and a second row carrying the same id is
            # exactly the retry that contract promises — skipping it because a
            # first attempt raised would turn at-least-once into at-most-once
            # (agent review round 2, MINOR-2).
            seen.add(command_id)
        else:
            # No identity to deduplicate on, which only a writer older than the
            # field can produce. It still runs: the message is the user's, and
            # dropping it is worse than a duplicate it cannot be compared
            # against.
            await run(line.text)
        return
    except RuntimeError as error:
        # STRUCTURALLY, with the old sentence as the cross-build fallback: the
        # typed class is this build's seam, and a runtime one version behind
        # raises a bare ``RuntimeError`` that only the text identifies. Matching
        # the text alone — the first shape of this fix — degraded the recovery
        # back to a logged drop the moment the wording changed (agent review
        # round 2, MINOR-1).
        from local_operator.session.errors import TurnInFlight

        if not isinstance(error, TurnInFlight) and "already streaming" not in str(error):
            raise
        steer = getattr(handle, "steer", None)
        if not callable(steer):
            raise
        logger.info(
            "spooled prompt %s arrived mid-turn; joining the turn in flight",
            command_id or "<no id>",
        )
        await cast(Callable[..., Awaitable[Any]], steer)(line.text, command_id=command_id or None)
        if command_id:
            seen.add(command_id)


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
    bring-up, ``RuntimeServer.start``). The residual window is ``main()``'s logging
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
    # session construction, MCP bring-up and ``RuntimeServer.start``: the guarantee
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

    # THE INSTRUMENTS, BEFORE ANYTHING HERE CAN START A TURN. Both the inbox
    # drain below and the wake scheduler after it can open a turn, so a journal
    # attached later would miss exactly the turns a boot-time kill lands in —
    # and a boot record published later would be absent for a spawn that dies
    # at load. The record also precedes the control socket by construction, so
    # "this pid listened" and "this pid only existed" are distinguishable
    # (design-session-survival §5, §8).
    _bind_boot_instrumentation(handle, session_id=resume or "", cwd=cwd)

    # THE ORDERING IS THE GUARANTEE (design §11.4). Messages spooled while the
    # session was cold are delivered here, BEFORE the control socket begins
    # listening, so they cannot be interleaved with an errand a client sends
    # over that socket — there is no socket yet. ``runtime.start()`` below is
    # what binds it, on the runtime's own thread, so the ordering this comment
    # describes is now a happens-before across two threads rather than two
    # statements in one coroutine: the drain completes before ``start()`` is
    # called, and the binding happens on the thread ``start()`` creates — so
    # listening strictly follows this point. Note that ``start()`` does NOT
    # WAIT for that bind: it returns while ``_serve`` is still binding (which is
    # why ``wait_until_published`` is awaited below, and why that wait and this
    # guarantee are independent — this one is about ORDER, that one about being
    # able to READ what the boot wrote). Draining after it would race the
    # engaging caller's own prompt and deliver a note written minutes ago after
    # one written just now.
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
    # THE SERVING PLANE GETS ITS OWN LOOP. ``start()`` rather than
    # ``start_in_process()``, and the whole of the operator-visible defect is
    # that one word: in process, the listener, the welcome, ``ping`` and the
    # heartbeat share an event loop with the turn, so ANY synchronous step of a
    # turn parks all four together. Measured on the audit's rig (50 s block, no
    # client attached): the record crossed into ``wedged`` at t=46.2 s, and a
    # fresh dial connected in 0.01 s and then received NO welcome within 15 s.
    # The same rig with ``start()``: welcome immediate, ``ping`` -> ``pong`` in
    # 0.00 s, and the heartbeat never past 14.2 s. A fresh beat now means the
    # SERVING PLANE ran, which is the claim the surfaces already make.
    runtime.start()
    # WAIT FOR PUBLICATION BEFORE ANYTHING READS THE RECORD. ``start()`` returns
    # while ``_serve`` is still binding on its thread, so ``control_port`` is the
    # constructor's 0 and the record file does not exist yet. The parent polls
    # for the record (``launch.py``'s wait-for-record loop) and so does the wake
    # supervisor, but the boot record's withdrawal and the exit ordering below
    # both read the runtime's own state, so the wait is taken here too rather
    # than left to a caller's convention. A bind that failed releases this latch
    # as well; the answer would be "no control surface", and this path's
    # behaviour on a dead socket is what it always was.
    await runtime.wait_until_published()

    stop = asyncio.Event()
    # What ASKED this runtime to leave. Named because the exit itself is the one
    # event the reference investigation could not attribute: a refresh, a
    # SIGTERM and a torn install all ended the process with nothing written
    # about which it was (design §1.6/§5.3). The reaper and the retire path log
    # their own reason from ``_clean_exit``; this covers the two triggers that
    # dispose directly.
    trigger: dict[str, str] = {}
    #: The in-flight drain, if a signal has asked for one. Held so that a
    #: REPEAT signal cannot start a second drain, and so nothing else needs to
    #: know whether one is running.
    draining: asyncio.Task[None] | None = None

    def _on_signal(sig: signal.Signals) -> None:
        """Leave — at the next boundary if a turn is in flight, right now if not.

        The asymmetry this closes: the reaper has always refused to exit under
        live work, while this handler disposed immediately, so a sweep that
        arrived as a signal destroyed exactly the turns residency protects (the
        2026-09-14 incident: 21 runtimes, 32 cut-off turns).

        A signal with NOTHING in flight behaves exactly as it always did — the
        event is set in this same synchronous step, no task, no added latency,
        no change to anything that watches ``stop``.

        A REPEAT signal does NOT shorten the bound. A second SIGTERM is either
        the same fire-and-forget sweep arriving twice or a person pressing a
        key twice, and neither may talk the runtime into discarding the turn it
        is finishing; SIGKILL remains the unrefusable way to end a process that
        truly must end now. The repeat IS logged, because "the signal arrived
        twice and was absorbed" is the sort of thing an incident review has to
        be able to see afterwards.
        """
        nonlocal draining
        trigger.setdefault("why", sig.name)
        if not _work_in_flight(handle):
            stop.set()
            return
        if draining is None:
            draining = asyncio.ensure_future(
                _drain_for_signal(handle, runtime, stop, sig_name=sig.name)
            )
            return
        logger.info(
            "session runtime: %s repeated while draining; the drain bound is unchanged",
            sig.name,
        )

    def _on_socket_stop() -> None:
        """The graceful ``stop`` op (``ServingSessionHandle.request_stop``).

        MEASURED ON THIS HEAD, this hook runs on the SESSION's loop: the
        registrant HOPS ``request_stop`` there
        (``RuntimeServer._handle_call_on_session_loop``), so the "arrives from
        another thread" premise this docstring used to state is not what makes
        the set below correct — and it is not stated as the repair any more.
        What makes it correct is ``call_soon_threadsafe``, which is right on
        BOTH loops: ``asyncio.Event.set()`` from a foreign thread sets the flag
        WITHOUT waking the waiter (its callback is scheduled with plain
        ``call_soon`` and no self-pipe write happens), so a loop parked in
        ``select()`` — which is what ``await stop.wait()`` below looks like at
        the syscall level — is not woken until some other timer fires; with
        nothing else armed, never. That failure is what
        ``publication.PublicationGate`` exists for, and why
        ``RuntimeServer._wake_close_wait`` goes through the threadsafe form too.
        Keeping the defensive shape costs one self-pipe write and removes the
        question of which thread a registrant's hop machinery delivers this on —
        a reduced handle, or a future caller that reaches the hook directly,
        included. ``trigger`` is written here and read by the loop below; the
        same hop orders the two, the same way it orders ``stop``.
        """
        trigger.setdefault("why", "socket-stop")
        loop.call_soon_threadsafe(stop.set)

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
    if draining is not None and not draining.done():
        # The stop came from somewhere else first (the reaper's idle drain, the
        # socket ``stop`` op, or the refresh branch) while a signal-driven drain
        # was still waiting. That drain's remaining job was to set ``stop``,
        # which has now happened, so it is cancelled rather than left to wake
        # against a session that is already disposing.
        draining.cancel()
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
        # BEFORE the dispose, and that is the whole ordering: a turn still open
        # here is about to be aborted, and the exit cause this runtime knows
        # about itself (a signal name, the socket stop, "unknown") is the one
        # fact a successor cannot reconstruct from a corpse.
        _note_journal_exit(handle, trigger.get("why") or "unknown")
        try:
            handle._deny_pending_gates()
        except Exception:  # noqa: BLE001 — shutdown must proceed
            logger.debug("child gate deny failed", exc_info=True)
        try:
            await handle.dispose()
        except Exception:  # noqa: BLE001
            logger.warning("child session dispose failed", exc_info=True)
    # ``aclose_remote``, not ``aclose``: this is the SESSION's loop and the
    # runtime now owns its own thread, so the owner-loop-only form would raise
    # here — on the exit path that withdraws the boot record two lines below,
    # which an unwrapped raise skips entirely, leaving a record the reaper reads
    # as a runtime that stopped without running its own exit ordering.
    remote = getattr(runtime, "aclose_remote", None)
    if callable(remote):
        await cast(Callable[[], Awaitable[None]], remote)()
    else:  # pragma: no cover - a reduced host that only answers the owner-loop form
        await runtime.aclose()
    # Clean exit, so the boot record goes with it (see ``_clear_boot_record``):
    # a record that outlives its process is the statement "this pid stopped
    # without running its own exit ordering", and this path ran it.
    _clear_boot_record()

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
    # THE TOKENIZER WARM STARTS FIRST, before the log file, the imports and the
    # lease: it is the only piece of boot work that nothing else on this path
    # touches until the user's first message has already been sent, so
    # overlapping it costs nothing and removes ~120 ms from this process's
    # time-to-first-token (see ``compaction.tokens.warm_tokenizer``). It is a
    # daemon thread whose failure mode is "the cost moves back to where it was"
    # — never a boot failure.
    #
    # WRAPPED, INCLUDING THE IMPORT. The callee cannot raise; the import can,
    # and this sits ABOVE ``configure_file_logging``, so an escaping exception
    # would kill the child before there is anywhere to write down why — a
    # failed attach with no traceback, which is the failure shape
    # ``_spawn_runtime``'s capture file exists to end.
    try:
        from local_operator.compaction.tokens import warm_tokenizer_in_background

        warm_tokenizer_in_background()
    except Exception:  # noqa: BLE001 — a warm-up must never be the failure
        logger.debug("tokenizer prewarm unavailable at boot", exc_info=True)

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
    # The Linux comm axis (see :func:`procname.brand_this_process`): macOS names
    # this child from the image its parent exec'd it through, Linux has no such
    # image, so the process has to name itself. Called in the ``__main__``
    # branch rather than inside ``main()`` because ``main()`` is callable
    # in-process, and a comm set on the CALLER's thread would outlive the call.
    from local_operator import procname

    procname.brand_this_process()
    sys.exit(main())
