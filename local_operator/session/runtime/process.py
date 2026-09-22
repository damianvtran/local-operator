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
sustained drain. Environment variables are the spawn contract
(``LOP_MOBILE_CHILD_CWD``, ``_PROVIDER``, ``_MODEL``) — argv would be
ps-readable. The ONE thing that rides in argv is ``--operator-fd <n>``, the
NUMBER of the descriptor the operator capability arrives on: a descriptor
number is not a secret and is useless without the 32 bytes written through it
(issue #1310, see ``harness/approval.py``).

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
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from local_operator import buildwatch as _buildwatch
from local_operator import procstate
from local_operator.procstate import install_loop_signal_handlers
from local_operator.session.runtime import stall_watchdog
from local_operator.session.runtime.types import (
    BUILD_DRAIN_OVERDUE_CAUSE,
    BUILD_DRAIN_PROGRESS_S,
    HEARTBEAT_INTERVAL_S,
    LEAVING_FOR_BUILD,
    LEAVING_FOR_BUILD_OVERDUE,
    LEAVING_ON_SIGNAL,
    SIGNAL_DRAIN_CAUSE,
    SIGNAL_DRAIN_S,
    UPDATE_UNNAMED_PAIR,
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

#: HOW LONG A BUSY PROBE THAT CANNOT BE EVALUATED MAY PIN A RUNTIME.
#:
#: Deliberately a linear COUNT and not a deadline: the reaper samples
#: ``_should_exit`` once per :data:`REAP_CHECK_S`, so counting the samples counts
#: the wall clock at the one cadence that exists, and a count cannot be warped
#: by a clock jump (the same reason ``_buildwatch`` counts checks). At 0.25 s
#: per sample this is ~60 s.
#:
#: THE STATE THIS BOUNDS is the one term that can pin a runtime BY
#: CONSTRUCTION: ``_work_in_flight`` answers "work is in flight, stay resident"
#: for every sample whose probe raises, and nothing else in the predicate can ever
#: contradict it — so a handle that has come apart keeps the process resident for
#: the life of the machine, invisible to every reader in the product, because they
#: all resolve a runtime through a record (or a boot record, or an environment)
#: that a broken handle stops maintaining. A probe that has answered nothing for a
#: minute is not evidence of work; it is a broken instrument, and a broken
#: instrument must not be able to keep a process alive for the life of the
#: machine.
#:
#: HOW MANY RUNTIMES ARE ACTUALLY IN THAT STATE IS NOT KNOWN, and the honest
#: figure matters more than the alarming one: the fleet on this machine on
#: 2026-09-17 was first counted as 34 of 57 runtimes with no record — but a second
#: census, asking each process for the config root it ACTUALLY uses, found every
#: one of them holding a fresh record in its OWN root (a sibling QA store), so that
#: population was a root-scoping artifact rather than a set of pinned corpses. What
#: this bound answers is the state that cannot be counted from outside at all: a
#: runtime whose own instrumentation has failed. The census, the refusals and the
#: sweep that can end such a process from outside are in ``reclaim``.
#:
#: WHY THE BOUND IS SO MUCH LONGER THAN THE DRAIN (3 s). The grace exists to
#: absorb work that ARRIVES; this bound exists to absorb a probe that is
#: TRANSIENTLY unevaluable, and it is chosen an order of magnitude beyond any
#: transient this module has a record of. Both errors are recoverable in the safe
#: direction: leaving early loses at most the turn the transcript resumes on the
#: next engage, while pinning forever costs a resident process that no surface in
#: the product can even see.
PROBE_DEFER_BOUND = 240

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
#: The update window's bounds are read through ``_buildwatch`` at the CALL SITE
#: rather than re-exported as names here, unlike the timings above: the tests and
#: the e2e stage shorten them with ``monkeypatch.setattr(buildwatch, ...)``, and a
#: module-level alias captured at import would keep applying the shipped bound
#: while the test believed it had moved it. The two constants are re-exported
#: because they are VALUES (nothing patches a value through this module).
UPDATE_LOCK_S = _buildwatch.UPDATE_LOCK_S
UPDATE_LOCK_HEARTBEAT_S = _buildwatch.UPDATE_LOCK_HEARTBEAT_S

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


class _ProbeDefers:
    """Consecutive samples whose busy probe RAISED, and the last failure's text.

    Process-scoped rather than per-handle for the same reason
    ``_boot_record_pid`` is: one runtime per process, so this is the runtime's
    own state and a handle that is replaced mid-life (``/new``) must not hand a
    fresh probe a fresh sixty seconds.

    A HEALTHY SAMPLE RESETS THE STREAK, whatever it answered. The count is of
    CONSECUTIVE failures, so a probe that answers ``True`` (real work) or
    ``False`` (idle) between two failures clears it: the runtime is not being
    pinned by an unusable probe, and a later streak starts from zero.
    """

    def __init__(self) -> None:
        self.streak = 0
        self.detail = ""

    def evaluated(self) -> None:
        """The probe answered (either way). Clears the streak."""
        self.streak = 0
        self.detail = ""

    def unevaluable(self, detail: str) -> int:
        """Record one failure and return the new streak length."""
        self.streak += 1
        self.detail = detail
        return self.streak

    def reset(self) -> None:
        """Forget the streak. For tests, which share this module's state."""
        self.evaluated()


#: The one instance, at the scope of the one runtime each process hosts.
_probe_defers = _ProbeDefers()


def _busy_verdict(handle: object) -> tuple[bool, str]:
    """``(busy, unevaluable_detail)`` — the probe's answer AND its usability.

    SEPARATES "THE PROBE SAID YES" FROM "THE PROBE DID NOT ANSWER", which the
    ``bool`` return cannot express and which the residency bound has to
    distinguish: the first is work and pins the runtime forever by design, the
    second is a broken instrument and may pin it only for
    :data:`PROBE_DEFER_BOUND` samples. ``detail`` is empty exactly when the probe
    answered, and it is what the log line and the exit reason quote — a bound
    that fires without naming what was wrong with the probe is an unattributable
    exit, the failure mode this module's whole instrumentation exists to remove.
    """
    probe = getattr(handle, "is_busy", None)
    if not callable(probe):
        return False, ""
    try:
        return bool(probe()), ""
    except Exception as exc:  # noqa: BLE001 — uncertainty must keep the runtime working
        logger.debug("busy probe failed; treating work as in flight", exc_info=True)
        return True, f"{type(exc).__name__}: {exc}"


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
    return _busy_verdict(handle)[0]


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

    AND THE FIRST TERM IS NOT ALLOWED TO PIN FOREVER BY FAILING. When the probe
    RAISES, the immediate answer stays fail-closed (``True`` — see
    :func:`_busy_verdict`: uncertainty must never be what ends a turn), but the
    failures are COUNTED, and a streak of :data:`PROBE_DEFER_BOUND` of them with
    nothing else holding the runtime is itself the verdict: a probe that has
    answered nothing for a minute is broken, not busy, and it may not keep a
    process alive for the life of the machine. Terms 2 and 3 must be consulted
    before that verdict is taken — a viewer or an imminent wake is a reason to
    stay that has nothing to do with the probe, and a streak measured through
    one would spend the bound on a runtime that was legitimately wanted — so
    this is the one path where the order is term 1, then 2 and 3, then the
    count. The exit is announced (WARNING at the first failure of a streak, and
    again with the reason at the bound) because an unattributable exit is the
    failure the whole of this module's instrumentation exists to remove.

    ORDER IS OTHERWISE UNCHANGED: for a probe that ANSWERS, this is the three
    terms in the order above and nothing else.

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
    busy, unevaluable = _busy_verdict(handle)
    if not unevaluable:
        # The ordinary path, byte for byte: the probe answered, so "would lose
        # nothing" has exactly one meaning here and in the signal drain.
        _probe_defers.evaluated()
        if busy:
            return False
        if _wake_within_window(handle):
            return False
        if _viewer_attached(runtime):
            return False
        return True
    # FAIL-CLOSED, COUNTED. A probe that raises pins for the immediate decision
    # (`_busy_verdict`), but only while nothing else is holding this runtime:
    # a viewer or an imminent wake is a reason to stay of its own, and counting
    # through one would spend the bound on a runtime whose residency was never
    # the probe's doing.
    if _wake_within_window(handle) or _viewer_attached(runtime):
        _probe_defers.reset()
        return False
    streak = _probe_defers.unevaluable(unevaluable)
    if streak == 1:
        logger.warning(
            "session runtime: the busy probe is unusable (%s); deferring, and leaving "
            "if it is still unusable after %d consecutive samples (~%.0fs)",
            unevaluable,
            PROBE_DEFER_BOUND,
            PROBE_DEFER_BOUND * REAP_CHECK_S,
        )
    if streak < PROBE_DEFER_BOUND:
        return False
    logger.warning(
        "session runtime: the busy probe has been unusable for %d consecutive samples "
        "(%s); work that cannot be read is not work still in flight, so leaving",
        streak,
        unevaluable,
    )
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


def _idle_exit_reason() -> str:
    """The cause string a quiet exit reports, naming the term that actually held.

    Two quiet exits reach ``_clean_exit`` from the same branch and they mean very
    different things, so the cause distinguishes them: an ordinary idle exit was
    PROVEN idle (no work, no viewer, no wake), while an exit on
    :data:`PROBE_DEFER_BOUND` was never proven anything — it was proven
    UNREADABLE. Folding the second into the first would put the most informative
    departure in this module's history behind the same word as a routine one,
    which is exactly the ambiguity ``_clean_exit``'s ``reason`` argument exists to
    remove (design §1.6/§5.3: an exit that says nothing about itself).

    A function rather than an expression at the call site because the streak, the
    bound and the last failure are three separate pieces of module state and the
    ONE place that reads them together is here.
    """
    if _probe_defers.streak < PROBE_DEFER_BOUND:
        return "idle-exit"
    return (
        f"idle-exit (busy probe unusable for {_probe_defers.streak} samples: "
        f"{_probe_defers.detail})"
    )


async def _reaper(handle: object, runtime: object, stop: asyncio.Event) -> bool:
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

    IT RETURNS WHETHER IT RAN THE CLEAN ORDERING, and that is not decoration:
    there are TWO normal returns and they mean opposite things to ``amain``.
    ``True`` is the exit leg — ``_clean_exit`` has run (deny → dispose →
    aclose) and the caller owes nothing. ``False`` is the empty-hands return,
    taken when ``stop`` was set by a task OTHER than this one: the signal
    drain's bound expiry or its no-latch fallback (``_drain_for_signal``),
    ``_on_signal``'s nothing-in-flight rung, the socket ``stop`` op
    (``_on_socket_stop``), or a drain another task ran to its own end. Nothing
    was disposed HERE, so the caller still owes the whole ordering. A caller
    that cannot tell the two apart skips an exit it owes — see
    :func:`_clean_ordering_already_ran` and issue #1250.
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
            return True
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
                return True
            continue
        if not _should_exit(handle, runtime):
            continue
        deadline = time.monotonic() + grace_s
        drained = False
        while time.monotonic() < deadline:
            await asyncio.sleep(REAP_CHECK_S)
            if await refresh_check():
                return True
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
        # THE REASON NAMES THE TERM THAT ACTUALLY HELD; see ``_idle_exit_reason``.
        reason = _idle_exit_reason()
        logger.info(
            "session runtime: idle for %.1fs (no work, no viewer, no wake within %.0fs); "
            "exiting cleanly (%s)",
            grace_s,
            WARM_WINDOW_S,
            reason,
        )
        await _clean_exit(handle, runtime, reason=reason)
        stop.set()  # amain's wait() returns; exit code stays 0
        return True

    # THE EMPTY-HANDS RETURN. ``stop`` was set by another task while this loop
    # was parked — the signal drain's bound expiry (``_drain_for_signal``), that
    # drain's no-latch fallback for a reduced handle, ``_on_signal``'s
    # nothing-in-flight rung, the socket ``stop`` op (``_on_socket_stop``), or a
    # drain another task ran to its own end — so an exit is under way and this
    # loop is not the one running it. ``False`` is what tells ``amain`` that it
    # still owes the deny → dispose → aclose ordering, and the ordering is
    # exactly what a bare ``exception() is None`` at that call site used to
    # lose: both returns look identical to it, so a 0.25 s tick landing in the
    # same loop iteration as a drain bound skipped the whole exit block (#1250).
    #
    # NB a build rung the REAPER runs is NOT among those producers, and the
    # distinction is worth keeping straight: ``_refresh_for``, ``_drain_for`` and
    # ``_leave_overdue`` each run ``_clean_exit`` in the task that calls them and
    # then return ``True``, so that rung is the exit leg, not this return.
    return False


def _clean_ordering_already_ran(reaper: "asyncio.Task[bool]") -> bool:
    """Did a FINISHED reaper run the clean exit ordering itself?

    The one question ``amain`` has to answer after ``await stop.wait()``:
    whether the exit block below it — the ``exiting`` line, the turn journal's
    exit note, ``_deny_pending_gates`` and ``dispose`` — is owed or already
    done. It reads the reaper's own return value, because that is the only
    thing that separates the two normal returns: a reaper that woke to an
    already-set ``stop`` also finishes with no exception, and reading that as a
    completed exit is how the block got skipped.

    What the skip cost, measured on ``macos-latest`` and reproduced on this
    host: no exit note reached the turn journal, so its row kept
    ``exit_cause=''`` and the cell asserting
    ``signal_exit_token(row.exit_cause) == "SIGTERM"`` failed with
    ``'' == 'SIGTERM'`` (issue #1250: reported 10 occurrences over four days, all
    on ``macos-latest``) — the row was then closed by the teardown's own dispose
    instead, recording an aborted
    turn as ``completed``. The gate deny and the dispose were skipped with it;
    only ``aclose_remote`` still ran, which is why the process exited 0 and the
    boot record was withdrawn while none of the ordering had happened.

    ``cancelled()`` is tested BEFORE ``exception()`` deliberately:
    ``exception()`` RAISES ``CancelledError`` on a cancelled task rather than
    answering ``None``, so a reaper that is FINISHED *and* cancelled would make
    this function raise instead of answering. Nothing in the tree cancels the
    reaper but ``amain``'s own ``reaper.cancel()`` branch, and that branch never
    reaches this read — so the ordering is defence against a future caller, not
    a closed live escape. It is what makes the answer TOTAL, and that is the
    reason it cannot be simplified away: the moment something else cancels a
    finished reaper, the three-way answer below is the only one that still
    holds.
    """
    if not reaper.done() or reaper.cancelled():
        return False
    if reaper.exception() is not None:
        return False
    return reaper.result() is True


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

    THE UPDATE WINDOW (2026-09-19). An idle runtime leaving is exactly the
    runtime a person is typing into — the TUI has just told them it will switch
    to the new version when next idle — so this rung opens a window BEFORE the
    announce: ``SessionRecord.updating`` is published, the admission lock is
    taken, and every admission that arrives from then on is SPOOLED for the
    successor rather than refused (``serving.ServingSessionHandle.prompt``;
    ``types.UPDATING`` carries the incident). The window closes one of two ways:

    * the handover completes — the process exits, the marker stays, and the
      successor's boot consumes it into ``record.updated`` (``_refresh_for`` does
      NOT close the window on this arm, deliberately: an ``end_update`` before
      ``_clean_exit`` would clear the marker the successor needs);
    * the bound expires — ``UPDATE_LOCK_S`` with no heartbeat. The rung then
      ABANDONS the handover rather than waiting on it: the lock is released, the
      build this runtime loaded is KEPT, the failure is published with
      ``types.UPDATE_FAILED_CAUSE``, and the messages this window queued are
      drained back in and run HERE (``_abandon_update_window``). That last part is
      what makes the queue safe: a receipt for a successor that never comes would
      be the same broken promise, one layer down.

    The bound is on the window's OWN work, enforced by ``_await_live_window`` against the
    LOCK's dead-or-alive deadline rather than by a total-duration timeout — so a stalled
    viewer writer is the failure it holds, and a handover that keeps beating is not. The
    two legs are bounded on different terms for the reason that function's own docstring
    gives: the handover's awaits can beat, while the exit leg cannot.
    """
    boot: BuildStamp | None = getattr(runtime, "_boot_build", None)
    pair = _buildwatch.update_pair_text(boot.label() if boot is not None else "", newer.label())
    # THE PAIR IS NEVER EMPTY ONCE A WINDOW OPENS. ``""`` is the record's "no window"
    # sentinel and the admission gate both, so publishing it would open an invisible
    # window that queues nothing while the sender still got a receipt (agent review
    # round 1, NIT 2). A runtime whose own stamp is unreadable still moves to the
    # build on disk, and the copy says exactly that.
    pair = pair or UPDATE_UNNAMED_PAIR
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

    # THE WINDOW OPENS HERE — after the last cheap gate and BEFORE the announce,
    # which is the order ``process._begin_drain`` documents: a reader must learn
    # the runtime is moving before anything is deferred to its successor, or the
    # first queued receipt arrives under a notice that says nothing is happening.
    begin_update = getattr(handle, "begin_update", None)
    opened = False
    if callable(begin_update):
        opened = bool(begin_update(pair, "stale-build"))
        if not opened:
            # Either another rung already holds the window (its move owns this
            # rung's too) or this pair has already failed a window and the
            # automatic rung has stopped retrying it. Both answers are "keep
            # serving", and both are cheap: the next check asks again.
            logger.info("session runtime: not opening an update window for %s; keeping", pair)
            return False

    async def _announce_and_latch() -> bool:
        """The handover's awaits, so the window can be bounded around them.

        Returns False when work turned up and the runtime is therefore keeping:
        the caller closes the window on that arm too, because a window left open
        over a runtime that did not leave queues for nobody.
        """
        announce = getattr(runtime, "announce_retiring", None)
        if callable(announce):
            try:
                # The window's build pair reaches the frame through the RECORD
                # (``RuntimeServer._announce_retiring_on_loop``), which the window
                # published before this call — deliberately not as a keyword here:
                # a peer runtime, or a reduced host, whose ``announce_retiring``
                # predates the parameter would take a TypeError inside the guard
                # meant for a viewer's writer, and lose the announcement entirely.
                await cast(Callable[..., Awaitable[None]], announce)(
                    "stale-build", to=newer.label()
                )
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
        return True

    bound = _buildwatch.update_lock_seconds()
    pump = asyncio.ensure_future(
        _pump_update_heartbeat(handle, interval=_buildwatch.update_lock_heartbeat_seconds())
    )
    handover = asyncio.ensure_future(_announce_and_latch())
    # THE PUMP IS REAPED ON EVERY PATH, including the early returns below: it is a task
    # on this runtime's loop, and one left running beats a released lock for the life
    # of the process (``UpdateLock.heartbeat`` is a no-op with nothing held, so the
    # leak is silent — a timer per second, forever).
    try:
        try:
            latched = await _await_live_window(handover, handle, bound)
        except asyncio.TimeoutError:
            handover.cancel()
            # AWAITED, not merely cancelled (agent review round 2, NIT A). The old
            # ``asyncio.wait_for`` both cancelled and consumed the task; ``cancel()``
            # alone leaves anything raised on its way out — a collaborator's error
            # rather than the ``CancelledError`` — as an unretrieved task exception.
            await asyncio.gather(handover, return_exceptions=True)
            await _abandon_update_window(handle, runtime, pair, bound)
            return False

        if not latched:
            if opened:
                # TWO DIFFERENT FACTS RETURN False FROM THE HANDOVER, AND THEY NEED
                # OPPOSITE ANSWERS (agent review round 1, MAJOR 1). ``stop`` set means
                # this process is EXITING: the spool must be LEFT ALONE, because the
                # next boot's drain is the only reader that will deliver it, and
                # draining it here moved the owner's message into the queue of a dying
                # process (measured: the row is consumed from ``inbox.jsonl``,
                # ``dispose`` rejects the prompt, and the text then exists NOWHERE
                # while the receipt told the user the successor would run it).
                # Anything else means work arrived and the runtime is KEEPING — the one
                # case where the spool belongs back inside the runtime that stayed,
                # because it is the only writer that owes those messages a turn now.
                await _close_update_window(handle, drain_back=not stop.is_set())
            return False

        logger.info("session runtime: retiring for %s", newer.label())
        # THE EXIT LEG IS BOUNDED TOO (agent review round 1, MINOR 1). ``_clean_exit``
        # awaits the turn abort and every viewer's writer — the stage that measured at
        # MINUTES in the incident — and the window used to stay open, beating for
        # nobody, across all of it: admissions kept getting receipts for a successor
        # that had not been spawned yet, with no bound at all. So the pump outlives the
        # latch and the exit is raced against the bound; the exit itself is NEVER
        # cancelled (a dispose cut in half is worse than a slow one), and the window's
        # promise is what gets withdrawn instead. Its own docstring says why this leg
        # takes the total-duration form rather than the heartbeat's.
        exit_task = asyncio.ensure_future(
            _clean_exit(handle, runtime, reason="retiring for " + newer.label())
        )
        try:
            await _await_live_window(exit_task, handle, bound, heartbeated=False)
        except asyncio.TimeoutError:
            await _retract_update_window(handle, pair, bound)
            await exit_task
    finally:
        pump.cancel()
    stop.set()
    return True


#: How often the handover is re-examined for liveness. Well under
#: ``UPDATE_LOCK_HEARTBEAT_S`` (1 s), so a window that has stopped beating is
#: noticed at the bound rather than a poll later; each pass is one ``asyncio.wait``
#: on the loop, so the cost of a 5 s window is on the order of a hundred turns.
_UPDATE_WINDOW_POLL_S = 0.05


async def _await_live_window(
    task: "asyncio.Future[Any]", handle: object, bound: float, *, heartbeated: bool = True
) -> bool:
    """Wait for a handover step while the window keeps proving it is ALIVE.

    THE BOUND IS THE HEARTBEAT'S, and that is the difference this function exists
    for (agent review round 1, MINOR 2). The previous shape was
    ``asyncio.wait_for(coro, timeout=UPDATE_LOCK_S)`` — a TOTAL-DURATION bound, so
    the lock's expiry, its beats and ``update_lock_remaining`` decided nothing: the
    reviewer removed the pump entirely and all 29 cells stayed green, because the
    timeout was not the mechanism being described. Here the deadline moves only when
    a beat arrives, so:

    * a handover that keeps beating may take as long as it needs (a slow-but-live
      announce is not a failure, and bounding it in total was the wrong promise);
    * a handover whose beats STOP — a blocked event loop, the one way a cooperating
      holder can silently die — expires at ``UPDATE_LOCK_S`` after the last beat;
    * deleting the pump now changes behaviour, which is what makes it load-bearing
      or wrong rather than decorative.

    ``heartbeated=False`` IS THE EXIT LEG, and the flag is not a convenience: the
    handover's awaits are the ones a beat can speak for, while ``_clean_exit`` is a
    DISPOSE, which has no progress to report and no collaborator that beats for it.
    So that leg gets the total-duration bound — 5 s from the commit to the process
    being gone — because "is it still making progress" has no honest answer there,
    and the failure it bounds is the real one: a successor that has not been spawned
    inside the bound is not one an admission may be told is coming (agent review
    round 1, MINOR 1). The pump still runs across it, so the record's ``updating``
    keeps being proven live for the surfaces reading it.

    RAISES ``asyncio.TimeoutError``. The caller owns the task; a hostile peer that
    never returns its writer is what the bound is for, so the caller cancels it.

    A HANDLE WITHOUT THE LOCK still gets a bound: the total-duration deadline is kept
    for the reduced hosts (the tests' stub handles, an older host wiring this rung to
    a handle that predates the window), because "bounded" must not depend on a
    method being there.
    """
    probed = getattr(handle, "update_lock_remaining", None)
    # AND THE LOCK HAS TO BE HELD, not merely readable (agent review round 2, R2-2).
    # The reader returns ``0.0`` when nothing is held, so gating on "the method is
    # there" handed a half-wired host — one that exposes the reader but never acquired
    # — a ZERO-second bound: immediate expiry, ``_abandon_update_window``, and an
    # ``update failed`` row for a window that never opened. The window string is the
    # second half of the question and the honest one: ``updating`` is non-empty exactly
    # while a window is open (``serving.ServingSessionHandle``), so a handle that says
    # nothing about a window gets the total-duration bound this function also
    # implements rather than a bound of zero.
    held = bool(getattr(handle, "updating", ""))
    remaining = (
        cast(Callable[[], float], probed) if heartbeated and held and callable(probed) else None
    )
    deadline = None if remaining is not None else time.monotonic() + bound
    # A shortened bound in a test must still be noticed promptly, so the poll
    # follows it down; the shipped pair (5 s / 1 s) leaves this at its constant.
    poll = max(0.001, min(_UPDATE_WINDOW_POLL_S, bound / 4))
    while True:
        done, _ = await asyncio.wait({task}, timeout=poll)
        if done:
            return bool(task.result())
        if remaining is not None:
            if remaining() <= 0:
                raise asyncio.TimeoutError
        elif deadline is not None and time.monotonic() >= deadline:
            raise asyncio.TimeoutError


async def _pump_update_heartbeat(handle: object, *, interval: float) -> None:
    """Beat an open update window while the handover runs. Never raises.

    A TASK RATHER THAN A BEAT BEFORE EACH AWAIT, and the reason is the awaits'
    shape rather than taste: the window's long await is inside
    ``RuntimeServer.announce_retiring``, which drains one writer per attached
    viewer — a count this rung cannot see and would have to guess at to place
    beats between. The pump is driven by the event loop's own turns, so if the
    loop is BLOCKED the beats stop, the deadline passes, and the bound fires.
    That is the intended failure mode: a loop that cannot turn is a runtime a
    front end must stop waiting on.
    """
    beat = getattr(handle, "heartbeat_update", None)
    if not callable(beat):
        return
    try:
        while True:
            await asyncio.sleep(interval)
            beat()
    except asyncio.CancelledError:
        # The window closed; a cancelled pump is the normal exit. Returning (as
        # opposed to re-raising) keeps the caller's ``cancel()`` from surfacing
        # as a task exception nobody awaits.
        return


async def _abandon_update_window(handle: object, runtime: object, pair: str, bound: float) -> None:
    """The bound expired: keep the build this runtime loaded, and say so.

    ORDER, and each step is one of the operator's requirements:

    1. the window CLOSES — the lock is released and the record's ``updating`` is
       cleared, so no admission is queued against a handover that is not coming
       and no surface keeps reading "updating";
    2. the messages the window QUEUED are drained back IN, so they run here, on
       the build the operator still has — the alternative is a receipt for a
       successor that never boots;
    3. the failure is PUBLISHED, on the record and as an incident row carrying
       ``types.UPDATE_FAILED_CAUSE``, which is what makes it reportable instead
       of silent (the operator's "so that it can be reported as an issue");
    4. the handle REMEMBERS the pair, so the automatic rung does not re-open the
       same window on its next check and burn the bound again forever.

    The runtime is never killed here — that is the whole contract. A failed
    update leaves a working session on the build it loaded.
    """
    logger.warning(
        "session runtime: the update window for %s held no heartbeat for %.1fs "
        "(bound %.1fs); abandoning the handover and keeping %s",
        pair or "the build on disk",
        bound,
        bound,
        _loaded_build_label(runtime),
    )
    end = getattr(handle, "end_update", None)
    if callable(end):
        end()
    remember = getattr(handle, "note_update_failed", None)
    if callable(remember):
        remember(pair, bound)
    await _drain_inbox_into(handle)
    note = getattr(runtime, "note_update_failed", None)
    if callable(note):
        try:
            await cast(Callable[..., Awaitable[None]], note)(pair, bound)
        except Exception:  # noqa: BLE001 — an unpublished failure is not a reason to die
            logger.warning("could not publish the failed update", exc_info=True)


async def _close_update_window(handle: object, *, drain_back: bool) -> None:
    """Close an open window whose runtime is KEEPING, and optionally re-admit its spool.

    The benign twin of :func:`_abandon_update_window`: work arrived, so the
    handover is not happening. Nothing is published — a window that closed
    because the runtime stayed is not a failure — and the marker goes, so a
    successor booting much later does not report an applied update that never
    happened (nor one that was abandoned when a stop landed).

    ``drain_back`` IS THE STOP ARM'S WHOLE POINT (agent review round 1, MAJOR 1).
    The spool belongs back inside the runtime that stayed, because it is the only
    writer that owes those messages a turn; on the arm where the process is
    EXITING, draining it here destroys the message — the row leaves the inbox for a
    queue that ``dispose`` then rejects — while the receipt the sender holds says
    the next runtime will run it. So the caller decides, from ``stop``, and the
    default is deliberately not a default: a caller that has not thought about
    which arm it is in cannot call this at all.
    """
    end = getattr(handle, "end_update", None)
    if callable(end) and not end():
        return
    if drain_back:
        await _drain_inbox_into(handle)


async def _retract_update_window(handle: object, pair: str, bound: float) -> None:
    """Stop ADVERTISING the window without calling the update failed.

    The exit leg's expiry (agent review round 1, MINOR 1): the handover is already
    applied and this process is going, so there is no failure to report and no
    spool to move — what changes is that new admissions stop being answered with a
    receipt. They are REFUSED instead, with the latch's own sentence and the draft
    back in the composer, which is the honest answer once no successor can be
    promised inside the bound.

    The spool is left exactly where it is: a ``SOURCE_USER`` row the window queued
    is delivered by the next boot's drain, which is the same guarantee the success
    arm gives.

    AND SO IS THE MARKER, WHICH IS THE ONE THING THIS ARM MUST NOT DO (agent review
    round 2, R2-1). ``begin_retire`` latched and ``_clean_exit`` completed, so the move
    genuinely happened: the successor boots the new build and the record owes the
    ``updated`` fact — the operator's own requirement, that the update be indicated as
    DONE. Clearing the marker here deleted that fact for exactly the case the incident
    measured at minutes (a dispose that outran its bound), and it is what separates this
    arm from the other two: on the STOP arm the move never happened (the next boot owes
    the operator their message, not an ``updated`` fact) and on the ABANDON arm the
    runtime kept the build it loaded. Both of those clear it, through ``end_update``'s
    default; this arm passes ``keep_marker=True``.
    """
    logger.warning(
        "session runtime: the update window for %s held no heartbeat for %.1fs (bound %.1fs) "
        "while this runtime was exiting; no longer advertising the move (new messages are "
        "refused until the successor is up)",
        pair,
        bound,
        bound,
    )
    end = getattr(handle, "end_update", None)
    if callable(end):
        end(keep_marker=True)


def _loaded_build_label(runtime: object) -> str:
    """The build label this runtime loaded, for the bound's log line. ``<unknown>``
    when the stamp is unreadable — the same fallback ``_refresh_for`` logs with, and
    the only honest answer for a process that cannot name its own install."""
    boot = getattr(runtime, "_boot_build", None)
    label = getattr(boot, "label", None)
    return str(label()) if callable(label) else "<unknown>"


def _session_dir_of(handle: object) -> "Path | None":
    """The session directory this handle serves, or ``None`` for a bare handle.

    The same two-hop read ``serving.ServingSessionHandle._session_directory``
    makes, duplicated here rather than reached through the handle because this
    runs at BOOT, before the handle is guaranteed to be the production one — and
    the marker's consumption must not depend on which handle a host built.
    """
    session = getattr(handle, "_session", None)
    transcript = getattr(session, "transcript", None) or getattr(session, "_transcript", None)
    directory = getattr(transcript, "directory", None)
    return directory if isinstance(directory, Path) else None


def _consume_update_marker(handle: object) -> str:
    """Read the handover marker and tell the handle an update APPLIED. ``""`` if none.

    The successor's half of the window, and the only place the "it worked" fact
    can be established: the predecessor is gone, and its record with it. Called
    from ``amain`` right after the inbox drain — the drain is what makes good on
    the queued messages, so the fact and the delivery are reported together, and
    a reader that saw "updated" before the messages ran would be reading a
    promise rather than a result.

    ONE-SHOT: the marker is cleared here, so a runtime that boots twice (a crash
    after the drain, before the socket) reports the update once. A marker that
    could not be cleared is corrected on the next boot rather than double-counted.
    """
    directory = _session_dir_of(handle)
    if directory is None:
        return ""
    from local_operator.session.runtime.inbox import (
        clear_update_window,
        read_update_window,
    )

    pair = read_update_window(directory)
    if not pair:
        return ""
    clear_update_window(directory)
    note = getattr(handle, "note_applied_update", None)
    if callable(note):
        note(pair)
    else:  # a reduced host: the fact lands on the attribute the record reads
        setattr(handle, "applied_update", pair)
    logger.info("session runtime: an update applied at boot (%s)", pair)
    return pair


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


#: The handle of the runtime this process is currently running, for the stall
#: bound's progress leg — and for nothing else.
#:
#: MODULE STATE BECAUSE THE TWO SIDES CANNOT SEE EACH OTHER: ``arm`` runs in the
#: ``__main__`` branch, before ``amain`` has built a session, while the probe it
#: needs a handle for only exists once ``spawn_owned_session`` has returned. A
#: closure over the handle is therefore impossible at the one moment arming is
#: allowed (see ``stall_watchdog`` on why that site is load-bearing), so the
#: handle is published here and the probe reads it lazily. Set by ``amain``,
#: cleared by ``main``'s ``finally`` beside ``disarm``.
_live_handle: object | None = None


def _tool_batch_in_flight(session: object) -> bool:
    """Is a tool batch EXECUTING in this session right now?

    Reads the live context through ``protocol.unanswered_tail_call_ids``, which IS
    this rule and the one scan behind both display questions that ask it
    (``pending_display_tool_ids`` / ``executing_display_tool_ids``). Agent review
    round 1 caught the first revision re-deriving it by hand: that copy agreed on
    today's shapes and would have drifted, because the published rule also scopes
    to the latest group after the latest user boundary (so an old interrupted turn
    cannot be revived by a later turn's liveness) and steps over the
    ``CustomMessage`` rows an incident can land on top of an open batch. The
    property itself is the message tail's, not a session's — ``AgentLoop`` appends
    the assistant message when a model turn ends and the tool results only once
    ``_execute_tool_calls`` returns, so for the whole duration of every batch the
    live list ends in unanswered calls.

    An unreadable context answers ``False`` — "no batch" — deliberately: this
    feeds a predicate that ends the process, so a state we cannot read must not be
    the thing that fires it, and a session with no context has nothing running.
    (The opposite direction is right for :func:`_work_motion`, whose docstring
    argues it: that tuple decides when to stop WAITING.)
    """
    context = getattr(session, "_context", None)
    messages = getattr(context, "messages", None)
    if not messages:
        return False
    # Function-local like every other import in this module: this file is the
    # child's boot path, and ``protocol`` pulls the session graph in.
    from local_operator.session.protocol import unanswered_tail_call_ids

    try:
        return bool(unanswered_tail_call_ids(messages))
    except Exception:  # noqa: BLE001 — an unreadable tail must not fire a bound
        logger.debug("stall watchdog: could not read the tool-batch tail", exc_info=True)
        return False


def _step_in_flight(handle: object) -> bool:
    """Is this process EXECUTING a step, i.e. is a spin not what we are seeing?

    The escape hatch that keeps the progress leg from cutting legitimate work,
    and it is deliberately two facts rather than ``handle.is_busy()``: that
    predicate is the RESIDENCY answer and is maximally inclusive by design — a
    running subagent lane or a detached background job makes it true — while the
    incident this leg exists for had four running lanes that had produced
    nothing for seven minutes. Reusing ``is_busy`` here would have spared the
    very state the leg is for, so what it asks is narrower and about THIS
    process: is a tool batch running, or a compaction rewriting history. Both
    burn CPU with no transcript movement while they last, which is exactly what
    the other two legs would otherwise read as a spin.

    A subagent lane's OWN in-process tool is not covered here, and the module
    docstring says so rather than leaving a reader to infer coverage. A lane that
    is stepping is covered from the other end: its step boundaries move
    ``_work_motion``'s roster generation, so the NO MOTION leg fails first.
    """
    session = getattr(handle, "_session", None)
    if session is None:
        return True
    if getattr(session, "_compacting", False):
        return True
    return _tool_batch_in_flight(session)


def _progress_probe() -> "tuple[object, bool]":
    """The stall bound's progress sample: ``(motion, in_flight)``.

    The seam between the two modules, and it is a plain callable rather than a
    handle so that ``stall_watchdog`` never imports this one — ``process`` is
    the module that imports IT, from the child's own entry point, and a cycle
    there would be paid by every runtime boot.

    NO HANDLE YET ANSWERS ``(no motion, in flight)``, i.e. "judge nothing". The
    arming happens before the session exists, so this is the ordinary state for
    the first seconds of every boot, and a probe that answered "not in flight"
    there would let a slow boot start a spin clock against a process that was
    still constructing itself. ``_work_motion`` is read through the same handle,
    so the two facts in the tuple are always about the same instant.
    """
    handle = _live_handle
    if handle is None:
        return (), True
    return _work_motion(handle), _step_in_flight(handle)


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
    session = getattr(handle, "_session", None)
    directory = getattr(getattr(session, "transcript", None), "directory", None)
    if directory is None:
        return 0
    # AN UNENGAGED SESSION KEEPS ITS PEER SPOOL — BUT NOT THE OWNER'S OWN WORDS.
    # Rows can reach this inbox from a sender on an older build (one whose record
    # read is absent-as-``True``) or from before any record existed, and draining
    # THEM HERE would put a peer row at the head of a conversation its owner has not
    # started — the boot drain runs before the socket listens, which is also before
    # the owner's first turn. Leaving those rows alone is what makes the delivery
    # happen instead at that first turn (``Session._drain_spooled_peer_inbox``, once
    # the owner IS engaged), and THAT drain runs after the turn's own messages are
    # durable, so the deferred rows land behind the owner's opening prompt rather
    # than opening the history (review round 1, F-2: an earlier revision of this
    # comment claimed that ordering while the drain still ran at the top of the turn
    # pipeline).
    #
    # A ``SOURCE_USER`` ROW IS NOT THAT CASE, AND THE GATE USED TO TREAT IT AS ONE
    # (agent review round 1, MAJOR 2). Those rows exist for exactly one reason: the
    # OWNER typed them into a session whose runtime was moving, and the receipt they
    # hold says the next runtime will run them. A session with no durable history is
    # not a session nobody engaged — it is the PRISTINE case this drain's own
    # callers name (``_should_refresh``: "a pristine stale runtime is the cheapest
    # refresh there is") — so gating them out left the message in the file with
    # nothing to run it: the runtime served on, idle, and the owner had to type a
    # second message before the first was delivered by the once-per-lifetime
    # first-turn drain. On the SUCCESSOR arm the same gate meant a record that said
    # ``updated`` while the message it was updated FOR was unrunnable.
    #
    # So the gate is scoped to the rows it was written for, and the peer rows are put
    # BACK rather than dropped: ``drain_inbox`` empties the file by contract, so a
    # reader that discards what it will not deliver has destroyed it.
    from local_operator.session.runtime.engagement import (
        TRANSCRIPT_FILENAME,
        durable_conversation_path,
    )
    from local_operator.session.runtime.inbox import (
        SOURCE_USER,
        append_inbox,
        drain_inbox,
        drop_owed_turn,
    )

    requires_engagement = not durable_conversation_path(directory / TRANSCRIPT_FILENAME)
    try:
        lines = await asyncio.to_thread(drain_inbox, directory)
    except Exception:  # noqa: BLE001 — a bad spool must not block the runtime
        logger.warning("inbox drain failed", exc_info=True)
        return 0
    if requires_engagement:
        keep = [line for line in lines if getattr(line, "source", "") != SOURCE_USER]
        lines = [line for line in lines if getattr(line, "source", "") == SOURCE_USER]
        for line in keep:
            # Best-effort and order-preserving among themselves; a row that cannot
            # be re-spooled is logged rather than lost silently.
            if not append_inbox(directory, line):
                logger.warning(
                    "could not re-spool a deferred inbox row for %s", directory.name, exc_info=True
                )
        if not lines:
            logger.info(
                "inbox drain deferred for %s: no durable history yet, so the person has "
                "not engaged this session; %d row(s) stay for the first turn",
                directory,
                len(keep),
            )
            # AND THE RAISE OBLIGATION GOES WITH THE DEFERRAL (review round 1,
            # R1-10). Every runtime raised for this record would boot, defer the
            # same rows and exit — real work, hourly, that delivers nothing,
            # because the one thing that would deliver them is the owner's first
            # turn. The rows stay in the spool (touch nothing else here) and that
            # turn still drains them, which is the deferral the sender's receipt
            # describes.
            drop_owed_turn(directory)
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
            # THE ROW GOES BACK. ``drain_inbox`` empties the file by contract, so a
            # delivery that raises has already consumed the message: without this the
            # operator's text is destroyed by a failure INSIDE this process, with the
            # receipt they hold still promising the successor would run it (agent review
            # round 1, MINOR 5 — measured: the row is consumed, ``_run_owner_prompt``
            # raises, and only an ERROR line in a log says so). Re-spooling is safe to
            # repeat because delivery is idempotent by the durable command index
            # (``_run_owner_prompt``) and ``inbox``'s contract is at-least-once.
            #
            # FOR A PEER ROW IT IS AT-LEAST-ONCE WITHOUT THAT SEAM (agent review round 2,
            # NIT B). ``inbox.InboxLine.command_id`` only ever rides a ``SOURCE_USER`` row,
            # and ``receive_peer_message`` has no equivalent dedupe: a peer row whose
            # delivery PERSISTED and then raised is appended a second time, and what the
            # peer sees is a duplicate ``peer_message`` card for one send. That is the
            # direction this file already chooses deliberately (a duplicated note is
            # visible and harmless, a dropped one is neither), and closing it properly
            # means an identity on the peer path — a change of its own, with its own
            # review, rather than a second half-seam here.
            if not append_inbox(directory, line):
                logger.error(
                    "could not re-spool an undelivered inbox row (command_id=%s); it is lost",
                    getattr(line, "command_id", "") or "<none>",
                    exc_info=True,
                )
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
    # THE DRAIN THAT EMPTIES THE SPOOL RETIRES THE OBLIGATION (``wakes.spooled``):
    # the rows are delivered, so nothing is owed any more, and the record would
    # otherwise leave the supervisor raising a runtime for a session with nothing
    # to run. Placed AFTER the loop rather than in a wrapper so the deferral
    # branch above keeps its own, opposite, decision.
    from local_operator.session.runtime.inbox import settle_owed_turn

    settle_owed_turn(directory, cwd=str(getattr(handle, "_desktop_cwd", "") or ""))
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


#: The switch that turns the SIGUSR1 task-stack dump off. ON BY DEFAULT since
#: 2026-09-20; ``0``/``off``/``no``/``false`` restores the previous disposition.
DEBUG_STACKS_ENV = "LOP_RUNTIME_DEBUG_STACKS"

#: The spellings that mean "off". A harness that needs SIGUSR1's default (fatal)
#: disposition back, or an operator debugging the dump itself, has one.
DEBUG_STACKS_OFF = ("0", "no", "false", "off")


def debug_stacks_enabled() -> bool:
    """Whether the SIGUSR1 task-stack dump is installed in this runtime.

    A function rather than an inline ``os.environ`` test so the default and its
    opt-out are testable without booting a runtime — the decision is the whole
    behaviour, and pinning it through a spawned child would make a one-line rule
    cost a process. The capability probe stays at the CALL SITE
    (``getattr(signal, "SIGUSR1", None)``), where the platform question belongs.
    """
    return os.environ.get(DEBUG_STACKS_ENV, "1").strip().lower() not in DEBUG_STACKS_OFF


#: How many deaths inside :data:`STALL_BEAT_WINDOW_S` are TOLERATED.
#:
#: TOLERATED, not "a storm": the panel below gives up on the death that comes
#: after this many, so the shipped constant of 3 tolerates three deaths and ends
#: the supervision on the FOURTH (agent review round 2, MINOR 1: an earlier
#: version of this comment said "the fifth", which is one more than the code
#: does — the number a reader takes from prose has to be the number the runtime
#: acts on).
#:
#: THE BUDGET IS A RATE, NOT A LIFETIME ALLOWANCE, and the shape of this constant
#: is the whole of that (agent review round 1, MAJOR 1). A counter that is only
#: ever incremented turns a storm guard into a slow disarmament: THREE unrelated
#: transient deaths spread over a long session were enough to spend it, and the
#: death AFTER them — hours later, no more related to those three than they were
#: to each other — ended the supervision for the rest of the session, so the
#: WORKLOAD stamp could then freeze for good and the bound could end a healthy
#: runtime, which is the incident this supervision exists to prevent. Counting
#: deaths inside a rolling window keeps what the cap is FOR (a hot failure cannot
#: spend unbounded resources re-creating a tick that will not run) without what it
#: accidentally had (one session's whole allowance, spent once and never
#: returned).
STALL_BEAT_RESTARTS = 3

#: The window :data:`STALL_BEAT_RESTARTS` deaths are counted over, in seconds.
#:
#: ONE BOUND'S WORTH (``types.DEFAULT_STALL_S`` is 300 s), and the scale is the
#: argument rather than a coincidence: "is this tick flapping?" is the same
#: question the bound asks of the plane it feeds, over the same period. Below this
#: rate a death is an incident the supervision absorbs; at this rate the plane's
#: reporter is not working and the bound is about to judge the plane anyway. A
#: window shorter than one heartbeat would make the budget unreachable, because
#: attempts cannot be closer together than the tick's own leading sleep.
STALL_BEAT_WINDOW_S = 300.0


async def _beat_stall_watchdog(stop: asyncio.Event) -> None:
    """Report the WORKLOAD loop's progress to the process's stall bound.

    The WORKLOAD plane's tick, and one of the two the bound tracks; the other is
    the serving plane's heartbeat (``RuntimeServer._heartbeat_loop``). They are
    tracked SEPARATELY (``stall_watchdog`` keeps a stamp per plane and arms for the
    earliest deadline), because the measured failure is exactly this plane going
    silent while the serving plane stays healthy — and a bound the healthy plane
    could keep re-arming would never fire on it.

    A plain sleep loop rather than a hook on the turn itself, and that is the
    property the bound needs: a turn that WAITS (on a model, a tool, a
    subprocess) yields and keeps this ticking, so what the bound measures is no
    sign of life anywhere — never a slow step or a long turn.

    Boot is not covered here: this starts with the reaper, after the runtime is
    published. The entry point's own arming covers that window, with the whole
    bound, which is longer than the engage deadline (180 s) the spawner sizes.
    """
    while not stop.is_set():
        # The SAME cadence the serving plane's heartbeat keeps, and the same
        # constant rather than a second copy of the number: the two ticks are
        # interchangeable as "this plane is alive" signals, so a drift between
        # them would be a drift in what the bound means.
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        stall_watchdog.beat(stall_watchdog.WORKLOAD)


def _record_tick_death(reason: str) -> str:
    """Best-effort write of one tick-death line, and the clause naming where it landed.

    MAY NEVER RAISE, AND THAT INCLUDES THE CLAUSE IT BUILDS. Two callers' CONTROL FLOW depends
    on this returning rather than on what it returns: the give-up has to reach its ``return``
    (agent review round 2, MINOR 2 — an un-totaled write there let a failing diagnostic defeat
    the terminal state, measured as 130 ticks, 130 guard tracebacks and no give-up line in 2 s)
    and a cancellation has to stay a cancellation (the same shape one branch over, where the
    write's exception replaced the ``CancelledError``).

    THE CLAUSE MOVED IN HERE FOR THE SAME REASON (agent review round 3, M2): the ``where`` string
    was built at the call site and interpolated ``stall_watchdog.dump_path()``, which reaches
    ``log_dir()`` — so even with the write itself total, a raise from the PATH LOOKUP hot-looped
    the supervisor (121 creations, no give-up) one line below the fix. Everything the give-up
    needs is therefore computed here, where nothing can raise, and the caller has no statement
    left between deciding to give up and returning.

    The two failure modes are reported apart, because they mean different things: a record that
    could not be written is a missing diagnostic, while a record that landed but could not be
    NAMED is present in the dump and only the log line is poorer for it.
    """
    try:
        recorded = stall_watchdog.note_tick_death(stall_watchdog.WORKLOAD, reason)
    except Exception:  # noqa: BLE001 — see the docstring: this must not move control flow
        _safe_warning(
            "session runtime: the stall bound's WORKLOAD tick-death record could not be "
            "written, so the log line for this event is the only trace of it",
            exc_info=True,
        )
        return "could NOT be recorded, so this log line is the only trace"
    if not recorded:
        return "could NOT be recorded, so this log line is the only trace"
    try:
        return f"is recorded in {stall_watchdog.dump_path()}"
    except Exception:  # noqa: BLE001 — naming the file must not decide the give-up either
        _safe_warning(
            "session runtime: the stall bound's WORKLOAD tick-death record landed, but its "
            "dump path could not be resolved for the log line",
            exc_info=True,
        )
        return "is recorded, in the dump beside this runtime's log"


def _safe_warning(message: str, *args: object, exc_info: bool = False) -> None:
    """``logger.warning`` FOR A PATH THAT MUST NOT BE BROKEN BY ITS OWN REPORT.

    EVERY log call in the supervision goes through here (agent review round 4, MINOR), because
    each one sits between a decision and the statement that carries it out — and this shape has
    now been found three times in two functions:

    * the give-up's own line: unprotected, a raise from it reached the cycle guard, which slept
      and went round again, so the give-up never happened (rigged: 321 creations in 6 s against
      4 for the control);
    * the two inside :func:`_record_tick_death`, i.e. in the function documented "MAY NEVER
      RAISE" (rigged together with the record: 282 creations);
    * the cancellation branch's, where a raise replaces the ``CancelledError`` a shutdown asked
      for, exactly as the un-totaled write did in round 2;
    * the cycle guard's own, which is the one that keeps the supervision alive at all — a raise
      there ends the supervisor, and a supervisor that has ended is the frozen stamp and the
      bound firing on a healthy runtime, which is the incident this whole change is about.

    REACHABLE BY REAL STATE, not only by a rig: this venv's ``StreamHandler.emit`` re-raises
    ``RecursionError`` instead of routing it to ``handleError``, and a runtime wedged enough to
    blow the recursion limit is exactly the runtime driving this loop.

    The cost is stated rather than hidden: if the log line cannot be written, the event is
    reported by the dump record that precedes it and by nothing else.
    """
    try:
        logger.warning(message, *args, exc_info=exc_info)
    except Exception:  # noqa: BLE001 — a report that cannot be made is not a second failure
        pass


async def _watch_stall_beats(stop: asyncio.Event) -> None:
    """Keep the WORKLOAD tick running for as long as the session is live.

    THE DEFECT THIS EXISTS FOR, measured on 0.62.0 (2026-09-21). The tick used to
    be started with a bare ``ensure_future`` and named exactly twice — there and
    at shutdown — so when it RAISED, nothing observed it: ``asyncio`` reports an
    unretrieved exception only at garbage collection, and a bound-firing
    ``_exit(1)`` never reaches GC at all. Nothing re-created the task either, so
    the WORKLOAD stamp froze FOREVER and the bound fired one deadline later on a
    runtime that was perfectly healthy, killing the turn in flight. Three
    readings were wrong at once: the runtime was reported as silent while it was
    working, the reporter's own death was reported nowhere, and the artifact
    could not tell the two apart — ``faulthandler`` dumps THREADS, and a dead
    task has neither a thread nor a frame, so the dump showed exactly what an
    idle healthy process shows.

    SO THE DEATH IS OBSERVED INSTEAD OF LEFT TO THE GARBAGE COLLECTOR, and the
    ``await`` below is what observes it: this coroutine drives the tick, so the
    raise lands in a live frame at the moment it happens. A done-callback would
    have to spawn the replacement from a synchronous callback (it cannot await
    the delay) and would still leave the exception for whoever remembered to
    call ``exception()``; awaiting it is the same visibility with the restart in
    the same frame, and it also means ``amain``'s shutdown cancels the LIVE tick
    through this await rather than skipping a dead one.

    LIVENESS SEMANTICS ARE UNCHANGED. The tick stays a plain sleep loop
    (:func:`_beat_stall_watchdog`, whose docstring says why a waiting turn must
    keep it ticking), a genuinely silent plane still trips the bound, and a tick
    that stays dead still trips it eventually: nothing here unbounds a plane
    whose reporter is gone, because the honest fail-safe is to leave on the
    deadline with the reason written into the dump rather than to run on with
    one leg silently switched off. What is new is only that a death is LOGGED,
    RECORDED, and — inside a rolling rate — UNDONE. The budget is
    ``STALL_BEAT_RESTARTS`` TOLERATED deaths inside ``STALL_BEAT_WINDOW_S``
    seconds, and the death after that budget (the FOURTH at the shipped constant)
    ends the supervision — not a lifetime allowance (agent review round 1,
    MAJOR 1: a counter that only ever increments is a storm guard that decays
    into a permanent disarmament, so a death hours after the ones that spent it
    re-ran the incident this function exists to prevent; agent review round 2,
    MINOR 1: that sentence said "the fifth" while the code gives up on the
    fourth). Past that rate the supervision GIVES UP, terminally and loudly, and
    hands the plane back to the bound -- see the give-up branch for what that
    costs and why continuing to re-create was rejected.

    THE SPACING THAT MAKES A HOT FAILURE BOUNDED IS THE TICK'S OWN LEADING
    SLEEP, so this loop adds no delay of its own on the normal path (the guard
    below is the one place that sleeps, for the case where even that cannot run).
    ``_beat_stall_watchdog`` is wait-then-beat, so the earliest a tick can die is
    one heartbeat after it was created, whatever the supervisor does.

    THE FIGURES ARE MEASURED, because a reasoned one was wrong twice (agent review
    round 1, MINOR 2; and the correction that replaced it was wrong too, caught by
    a real runtime before it reached the PR). A workload beat that NEVER works, at
    heartbeat 0.2 s with a 2 s bound: the tick raised at 0.21, 0.63, 1.10 and
    1.52 s — one heartbeat per attempt PLUS the guard's own heartbeat after each
    failed stamp, because a beat that is broken breaks the supervisor's stamp with
    it — and the bound ended the process 2.44 s after the arm. At the production
    cadence (15 s, bound 300 s) that shape gives a give-up ~105 s after the arm and
    a runtime that ends ~300 s after it.

    THE INVARIANT, which is the only figure worth quoting: the bound fires one
    deadline after the plane's LAST STAMP, and a stamp only ever comes from a beat
    that WORKED — so a persistently broken beat costs the runtime no extra life
    (its deadline is the arm's own), while a TRANSIENT death is exactly what the
    re-creation's stamp buys back, which is why this loop stamps at all.

    NOTHING IN THIS LOOP IS UNSUPERVISED, INCLUDING THE STAMP BELOW (agent
    review round 1, MINOR 1): the whole cycle runs inside a guard, so a fault in
    the recovery path is logged with its traceback rather than ending the
    supervision silently -- which is this function's own defect, one level up.
    """
    deaths: deque[float] = deque()
    while True:
        # THE WHOLE CYCLE IS GUARDED. Every statement here but ``await tick`` is
        # unsupervised state, and the failure it protects against is measured: a
        # raise from the stamp below ended the supervisor, froze the stamp, and
        # let the bound kill a healthy runtime. The guard is not a retry counter
        # and deliberately has no budget of its own, because it needs none: one
        # heartbeat per iteration bounds its cadence, and it does NOT stamp the
        # plane, so if it is the thing that keeps failing then the plane goes
        # genuinely unreported and the BOUND ends this runtime within one
        # deadline -- the fail-safe, already in place.
        try:
            tick = asyncio.ensure_future(_beat_stall_watchdog(stop))
            caught: Exception | None = None
            try:
                await tick
            except Exception as exc:  # noqa: BLE001 — a dying tick ends nothing in itself
                caught = exc
            if stop.is_set():
                # Ended because the session is ending, which is what it is for.
                return
            now = time.monotonic()
            deaths.append(now)
            while deaths and now - deaths[0] > STALL_BEAT_WINDOW_S:
                deaths.popleft()
            detail = (
                f"{type(caught).__name__}: {caught}"
                if caught is not None
                else "the tick returned early, with no stop and no exception"
            )
            # THE DECISION IS TAKEN BEFORE ANYTHING THAT CAN RAISE (agent review
            # round 2, MINOR 2). It used to be re-derived after the record write,
            # so a raise from that write skipped the `return` below: the guard
            # caught it, the loop went round again, and the FAILURE OF A REPORT
            # defeated the give-up — reproduced as 130 ticks, 130 guard
            # tracebacks and no give-up line in 2 s, with the plane never stamped.
            # A terminal state that a diagnostic can switch off is not terminal.
            giving_up = len(deaths) > STALL_BEAT_RESTARTS
            if giving_up:
                # THE GIVE-UP STATE, NAMED AND ARGUED (agent review round 1,
                # MAJOR 1 asked for the choice to be explicit). It is TERMINAL
                # for the session, and the reason is that the only thing that
                # could re-arm supervision is this supervisor, which is the
                # thing giving up. The rejected alternative -- keep re-creating
                # while a plane's reporter cannot run -- is a choice between
                # stamping on the plane's behalf (asserting a liveness nothing is
                # delivering: the instrument lying about the one thing it
                # measures) and not stamping (in which case the bound fires
                # anyway). So the bound firing IS the fail-safe, and this line is
                # what makes it attributable instead of mysterious.
                detail += (
                    f" -- and the supervision GIVES UP here: {len(deaths)} deaths inside "
                    f"{STALL_BEAT_WINDOW_S:g}s. Terminal for this session by design; what that "
                    f"costs is stated plainly: from here on nothing watches whether the "
                    f"workload plane reports, its stamp is left to freeze, and the bound ends "
                    f"this runtime one deadline after its last stamp"
                )
            # THE RECORD IS BEST-EFFORT AND THE GIVE-UP IS NOT, and since round 3 (M2)
            # that covers the CLAUSE as well as the write: everything the give-up needs is
            # computed inside ``_record_tick_death``, so there is no statement left here
            # that could keep this loop from reaching the ``return`` below — the path
            # lookup used to sit on this line, and rigging it to raise hot-looped the
            # supervisor with no give-up at all.
            #
            # THE RECORD ALSO COMES FIRST, so that a restart which is itself killed by the
            # bound (a tick that dies nearly a deadline late cannot be saved) still leaves
            # the reason in the file a reader will open.
            where = _record_tick_death(detail)
            if giving_up:
                # THE REPORT GOES THROUGH THE TOTAL CALL TOO (agent review round 4, MINOR):
                # this line sat between the decision and the ``return`` below, so a raise from
                # logging reached the guard, which slept and went round again — the give-up was
                # defeated by the announcement of the give-up (rigged: 321 creations, no
                # give-up line, control 4).
                _safe_warning(
                    "session runtime: the stall bound's WORKLOAD tick died (%s); the tick's "
                    "death %s",
                    detail,
                    where,
                )
                return
            _safe_warning(
                "session runtime: the stall bound's WORKLOAD tick died (%s); re-creating it "
                "(death %d inside %.0fs). Its plane's stamp is refreshed by the re-creation "
                "and then by the new tick, and the tick's death %s",
                detail,
                len(deaths),
                STALL_BEAT_WINDOW_S,
                where,
            )
            # THE RESTART IS ITSELF A TURN OF THE WORKLOAD LOOP, so stamping here
            # is a true reading rather than an optimistic one -- this coroutine
            # only runs when that loop runs. Without it the plane stays stamped as
            # silent for the new tick's leading sleep, and a death that happened
            # near the deadline would be completed by the bound instead of by the
            # recovery that just happened.
            stall_watchdog.beat(stall_watchdog.WORKLOAD)
        except asyncio.CancelledError:
            # A CANCELLATION WITH ``stop`` UNSET IS NOT A SHUTDOWN (agent review
            # round 1, NIT 1). ``amain`` sets ``stop`` and then cancels, so the two
            # cases are separable -- and they must be, because "the session is
            # ending" and "something else ended the ticker" are the pair this
            # module keeps insisting an instrument must not confuse. Nothing else
            # cancels this tick today; the guard is here so the day something
            # does, it is in the log and in the dump rather than silent.
            if not stop.is_set():
                _safe_warning(
                    "session runtime: the stall bound's WORKLOAD tick was cancelled while the "
                    "session was live, so the supervision ends with it: the plane has no "
                    "reporter and the bound will end this runtime one deadline after its last "
                    "stamp",
                    exc_info=True,
                )
                # THROUGH THE TOTAL WRITE, for the sibling reason: an un-totaled
                # ``note_tick_death`` raising HERE would replace this CancelledError
                # with its own exception, so the supervisor would end ``OSError``
                # where the shutdown asked for ``cancelled``.
                _record_tick_death(
                    "the tick was cancelled while the session was live -- not a shutdown, so "
                    "nothing re-created it"
                )
            raise
        except Exception:  # noqa: BLE001 — nothing here may end the supervision unobserved
            # THROUGH THE TOTAL CALL, and this one matters most: a raise from this log line
            # escapes the handler, so the ``while`` ends and the supervision dies — the tick is
            # never re-created, the stamp freezes and the bound ends a healthy runtime, which is
            # the incident this function exists to prevent (agent review round 4, MINOR).
            _safe_warning(
                "session runtime: the stall bound's WORKLOAD supervision raised in its own "
                "recovery path and is continuing; the plane is not stamped on this path, so "
                "the bound still bounds it",
                exc_info=True,
            )
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)


async def amain(operator_cap: bytes | None = None) -> int:
    """Run the owned session to completion.

    ``operator_cap`` is the capability handed over by the spawner on an
    inherited descriptor (see ``harness/approval.py``). It is threaded to the
    ``RuntimeServer`` rather than stashed on the handle, because the runtime is
    the seam that demands it and the handle is an injected collaborator that
    every test double also implements. ``None`` — a hand-run module, an older
    spawner — is the fail-closed state: ordinary operations keep working and
    nothing may loosen the gate.
    """
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
    # The progress leg's view of this process, published now that there is a
    # session to judge. Everything before this line is boot, and the probe
    # answers "judge nothing" there rather than starting a spin clock against a
    # process still constructing itself — see ``_live_handle`` and
    # ``_progress_probe``.
    global _live_handle
    _live_handle = handle
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

    # THE HANDOVER MARKER, consumed here for the same reason the drain is: this is
    # the successor's first act with a materialised session, and the fact it
    # carries ("this boot IS an update that applied") is only true once the
    # messages the window queued have actually been drained — which is the line
    # above. Read-and-clear is one-shot, so a runtime that boots twice reports it
    # once; the pair lands on the handle and ``RuntimeServer`` seeds the record
    # with it, because the record does not exist yet at this point.
    _consume_update_marker(handle)

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

    runtime = RuntimeServer(handle, kind="daemon", operator_cap=operator_cap)
    # EVERY WAY THIS RUNTIME CAN BE ASKED TO LEAVE IS ARMED HERE, BEFORE THE
    # SERVING PLANE CAN MAKE IT ADDRESSABLE. That order is the fix for a
    # measured race, not tidiness.
    #
    # ``runtime.start()`` hands ``_serve`` to a thread, and the first thing
    # ``_serve`` does once it holds a bound socket is publish the record
    # (``RecordPublisher.__init__``). The record is what every sender reads to
    # find this process: ``launch.py``'s wait-for-record loop, the wake
    # supervisor, and the stop ladder's rungs all address a target through it.
    # With the handlers installed after the wait for that publication — where
    # they used to sit — a runtime was ADDRESSABLE WHILE A SIGTERM STILL KILLED
    # IT with the default disposition. Measured on CI, twice, on two platforms
    # (``test_signal_drain_e2e``'s idle cell: exit ``-15`` with an empty
    # runtime-log tail, i.e. no ``exiting (SIGTERM`` line), which is the whole
    # symptom: the record is not unpublished, the lease is not released and the
    # documented drain never runs, because the process simply stops).
    #
    # WAITING ON PUBLICATION IS NOT A WAY AROUND THAT, which is what made the
    # window reachable at all: ``wait_until_published`` settles at the END of
    # ``_serve``'s boot prologue, so the record is already readable for as long
    # as those two boot registrations take to come back from the session's loop.
    #
    # AND NOT EARLIER THAN THIS EITHER, which is a bound rather than a
    # preference. The drain decision this handler takes announces through the
    # serving plane and waits on the handle (``_drain_for_signal``), so above
    # ``RuntimeServer`` there is nothing to decide WITH — and the prologue above
    # can already have a turn in flight (``_drain_inbox_into`` and the wake
    # scheduler's catch-up both open turns), so a handler armed there could only
    # either cut the turn this drain exists to protect or need a deferral
    # machine whose one possible trigger is a sender that signals a pid the
    # record does not yet name. Nothing in this process does that.
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

    # WINDOWS CANNOT INSTALL LOOP SIGNAL HANDLERS AT ALL (cross-platform work,
    # 2026-09-18): `add_signal_handler` is overridden only by the UNIX selector
    # loop, so the call raised NotImplementedError out of `amain` — the owner
    # process every attach dials never bound a socket on Windows. The shared
    # helper branches (see procstate.install_loop_signal_handlers). NOTHING IS
    # LOST where it degrades: the socket `stop` op below already converges on
    # the same event, and it is the rung the kill switch uses first.
    #
    # THE ORDER IS PINNED IN SOURCE: see
    # tests/unit/session/runtime/test_signal_drain.py, which fails if this call
    # ever moves back below ``start()``.
    install_loop_signal_handlers(
        loop,
        {
            signal.SIGTERM: lambda: _on_signal(signal.SIGTERM),
            signal.SIGINT: lambda: _on_signal(signal.SIGINT),
        },
    )
    # The socket ``stop`` op (the kill switch's graceful rung) and SIGTERM
    # converge on the same event, so the deny → dispose → aclose ordering
    # below runs once, identically, for both triggers. Armed HERE, with the
    # handlers, for their reason: an uninstalled hook is not an error but a
    # FALLBACK — ``request_stop`` disposes the session in place when no trigger
    # is set (``serving.py``), and that sets no stop event — so a ``stop`` op
    # landing in the window would have left a process that never exits, behind
    # a record that still reads as live.
    handle.on_stop_requested = _on_socket_stop
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

    # `SIGUSR1` is POSIX-only, so the debug dump is installed only where the
    # constant exists — an unguarded `signal.SIGUSR1` is an AttributeError that
    # would take the whole boot down over an opt-in diagnostic. `is_windows()`
    # is named alongside it because `loop.add_signal_handler` is UNIX-ONLY too
    # (the Windows default Proactor loop takes `BaseEventLoop`'s stub, which
    # raises `NotImplementedError`), so the two absences are one condition: this
    # block is safe to reach on any platform rather than merely skipped because
    # the constant happens to be missing.
    debug_stacks = getattr(signal, "SIGUSR1", None)
    # ON BY DEFAULT, and that is the change the 2026-09-20 freeze forced. This
    # was opt-in (``LOP_RUNTIME_DEBUG_STACKS=1``), and on every one of the five
    # runtime processes found wedged that day — 1.5 to 7.2 h each, all of them
    # parked in a C-level regex call with the transcript taking zero writes —
    # the variable was NOT set on the launcher, so the ONE instrument that could
    # have named the parked await was unavailable and the question "which line?"
    # stayed unanswerable for hours. A diagnostic that must be predicted before
    # the freeze it explains is a diagnostic nobody has when they need it.
    #
    # WHAT IT COSTS: one `add_signal_handler` under a capability probe, off the
    # hot path — nothing here is read per turn. WHAT IT CHANGES beyond the dump:
    # SIGUSR1's default disposition is fatal, so the signal used to kill a
    # runtime outright; it now prints this process's task stacks to the child log
    # instead, which is strictly more than the old behaviour offered and is the
    # convention a debugger and an operator both expect. The opt-out survives
    # (``=0``) for a harness that needs the default disposition back.
    #
    # It is NOT the whole answer to the freeze: this handler is an asyncio signal
    # handler, so like every other Python-level instrument it needs the loop, and
    # a loop parked in a C call never runs it. That class is what
    # ``stall_watchdog``'s C-thread dump covers; this one is for the state the
    # C-thread dump cannot show — a loop that is RUNNING but has its turn parked
    # on an await (round 2, U6).
    if not procstate.is_windows() and debug_stacks_enabled() and debug_stacks is not None:

        # SIGUSR1 prints every asyncio task's stack to the child log. The child
        # has no terminal and no attached debugger, and a wedged turn is exactly
        # the state whose cause is "which await is the turn parked in" —
        # invisible to py-spy without root and to the main-thread C dump, which
        # shows the loop idling under a parked task.
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

        loop.add_signal_handler(debug_stacks, _dump_task_stacks)
    # The self-reaper: a phone session nobody watches and nothing runs is a
    # live process doing nothing, and before this it idled FOREVER. Runs
    # beside the signal wait; whichever fires first wins.
    reaper = asyncio.ensure_future(_reaper(handle, runtime, stop))
    # The workload half of the stall bound, beside the reaper because it shares
    # its lifetime exactly: both run for the whole session and both stop with it.
    # SUPERVISED rather than a bare task, because an unobserved tick that dies
    # outlives this process's health by one deadline (see
    # :func:`_watch_stall_beats`, and the incident it documents). Nothing is
    # armed for an in-process host that never went through this module's entry
    # point, where ``beat`` is a no-op.
    stall_beats = asyncio.ensure_future(_watch_stall_beats(stop))
    reaper_ran_clean_exit = False
    await stop.wait()
    if draining is not None and not draining.done():
        # The stop came from somewhere else first (the reaper's idle drain, the
        # socket ``stop`` op, or the refresh branch) while a signal-driven drain
        # was still waiting. That drain's remaining job was to set ``stop``,
        # which has now happened, so it is cancelled rather than left to wake
        # against a session that is already disposing.
        draining.cancel()
    if not stall_beats.done():
        # The SUPERVISOR, so this reaches the live tick through its await rather
        # than skipping a dead one: a tick that died and was re-created is not
        # the task this handle names, which is exactly the state the old bare
        # handle could not cancel. A supervisor that has returned (its restart
        # budget exhausted) is ``done`` and owes nothing.
        stall_beats.cancel()
    if not reaper.done():
        reaper.cancel()
    elif _clean_ordering_already_ran(reaper):
        # The reaper completed AND ran the clean ordering, so this path owes it
        # nothing. A reaper that merely woke to find ``stop`` already set gets
        # no such credit — see :func:`_clean_ordering_already_ran`.
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
    # THE OPERATOR CAPABILITY IS READ HERE, and it is read from a DESCRIPTOR
    # rather than from argv/env/a file: a model-run ``bash`` tool can read any of
    # those (the record is 0600 under this same uid, which is the defect), and
    # the 32 bytes travel on an inherited fd the spawner closes as soon as it has
    # written them. Read above the lease and the session construction, and closed
    # inside the reader — by the time this process serves anything the descriptor
    # is gone from its table, so no tool subprocess can inherit it. ``None``
    # means no capability was handed over: ordinary operations keep working and
    # every authority-increasing request is refused. AFTER the logging setup, so
    # a malformed handoff's warning actually lands in ``runtime.log``.
    from local_operator.harness.approval import read_operator_cap_from_argv

    operator_cap = read_operator_cap_from_argv(sys.argv[1:])
    # The stall bound was armed in the entry point, before this file existed, so
    # its dump path is named here — next to the log a person reads after a
    # freeze, which is where they need it. A no-op when nothing is armed, i.e.
    # for every in-process caller of this function.
    stall_watchdog.announce()
    # One record per runtime, naming its own process: this file is shared by every
    # runtime child, so a reader has to be able to attribute a line to the
    # process that wrote it.
    logger.info("session runtime started: pid %d", os.getpid())
    try:
        return asyncio.run(amain(operator_cap=operator_cap))
    except KeyboardInterrupt:
        return 0
    finally:
        # A clean exit cancels the bound and writes its own outcome over the
        # header — the file STAYS, because the evidence is its content (the
        # fired marker), never its existence: a SIGKILL leaves the same file an
        # armed runtime has. See ``stall_watchdog``'s docstring for why nothing
        # here deletes anything. Reached on every graceful leave — the drain's,
        # the reaper's and a plain stop's — and never on the paths that exit
        # from the C thread (that is the point of the file).
        stall_watchdog.disarm()
        # Withdrawn with the bound, and for the same reason: a handle left here
        # would outlive its own runtime for any in-process caller of ``main``,
        # and the probe would then judge a disposed session.
        global _live_handle
        _live_handle = None


if __name__ == "__main__":
    # The Linux comm axis (see :func:`procname.brand_this_process`): macOS names
    # this child from the image its parent exec'd it through, Linux has no such
    # image, so the process has to name itself. Called in the ``__main__``
    # branch rather than inside ``main()`` because ``main()`` is callable
    # in-process, and a comm set on the CALLER's thread would outlive the call.
    from local_operator import procname

    procname.brand_this_process()
    # THE STALL BOUND IS ARMED HERE, AND ONLY HERE, for the same reason the comm
    # branding is: this branch is reachable by ``python -m`` alone, i.e. by a
    # real runtime child (``launch._spawn_runtime``, ``mobile/daemon.py``), while
    # ``main()`` and every constructor are reachable in-process. The C timer is
    # process-global and shared with the two pytest-side watchdogs
    # (``tests/e2e/watchdog.py``, ``tests/shard_stall_watchdog.py``), so arming
    # it from a library path would let an in-process boot silence a CI stage's
    # only bound; see ``stall_watchdog``'s docstring and
    # ``tests/unit/session/runtime/test_runtime_stall_watchdog.py``, which pins
    # both halves of that. Arming before ``main()`` also means a stall during
    # boot — the window nothing else can report — is bounded and named.
    stall_watchdog.arm(probe=_progress_probe)
    sys.exit(main())
