"""One definition of "the install on disk has been replaced under this process".

``lop-update`` (and :func:`local_operator.update.perform_upgrade`) replaces the
installed tree IN PLACE, so a long-lived process can find itself running a build
that no longer exists on disk. Two processes on this host must react the same
way to that: a session runtime (:mod:`local_operator.session.runtime.process`)
and the ``lop serve`` daemon (:mod:`local_operator.server.retire`). Three things
decide what such a process may do about it — how often to look
(:data:`BUILD_CHECK_S`), how long a fresh install must sit before it is trusted
(:data:`BUILD_SETTLE_S`), and how far to spread the exits when many processes
notice the same update (:data:`BUILD_STAGGER_S`).

They live in ONE module because the two callers must not disagree. A settle rule
that is 10 s in the runtime and 5 s in the daemon is a torn-tree race in whichever
copy a later edit forgets: the whole point of the settle is that a process about
to hand over to a successor does not spawn it out of a half-written
site-packages, and the guard only works where every participant shares it.

**Why this module is light.** It is stdlib only, and :mod:`local_operator.update`
— which reaches ``importlib.metadata``, ``urllib`` and ``subprocess`` — is
imported FUNCTION-LOCALLY inside :func:`build_changed`, the one function that
needs it. That matters for both importers: the runtime is spawned as a child on
every engage and the daemon's module is imported by ``generate_openapi`` and by
every ``lop serve``, so an import that dragged ``update`` in at module scope
would be paid by processes that may never watch a build at all. It is also why
this is NOT a section of :mod:`local_operator.update`: that module is the heavy
one, and neither of these callers may pay for it.

The constants are re-exported by ``session/runtime/process.py`` under the names
that module has always published (``BUILD_CHECK_S``, ``_build_changed``, …) so
nothing outside this module had to change when the definition moved here.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_operator.update import BuildStamp

logger = logging.getLogger(__name__)

#: How often an idle process re-reads the install on disk. Cheap (one
#: dist-info lookup + one 60-byte file), but there is no reason to do it on
#: every 250 ms reaper tick: a runtime that is already idle can afford to
#: notice an update within a few seconds, and a busy one never checks.
BUILD_CHECK_S = 5.0

#: A freshly written install is not a stable one: ``lop-update`` runs
#: ``uv tool install --force`` (which rewrites site-packages over several
#: seconds) and THEN writes ``.lop-source``. Retiring against a half-written
#: tree would spawn a successor that imports a mix of two builds. Require
#: the marker's mtime to be at least this old before acting on it.
#: Env-overridable ONLY so the e2e stage can flip a fake marker and observe
#: the retirement within its budget; production never sets the variable.
BUILD_SETTLE_S = 10.0

#: After the refresh predicate first holds, sleep a uniform random slice of
#: this before re-checking and retiring. This host runs ~16 resident
#: runtimes; ``lop-update`` would otherwise have all of them notice on the
#: same tick and their viewers spawn sixteen successors within a second.
#: Spread over 20 s the eager re-engages average ≤1 spawn/s. Same env
#: override rule as the settle: test-only.
#:
#: The daemon reuses it as its handover NOTICE as well as its spread: it
#: announces the retirement first, then waits this slice before leaving, which
#: is the window in which a record reader can actually see it (see
#: ``server/retire.py``).
BUILD_STAGGER_S = 20.0


def positive_seconds(raw: str, default: float) -> float:
    """``raw`` as a positive float, else ``default``.

    The build-watch timings are constants in production and only the e2e stage
    shortens them (``LOP_BUILD_SETTLE_S``, ``LOP_BUILD_STAGGER_S``), so a
    malformed or non-positive value falls back to the constant rather than
    disabling the protection it names. ``0`` is deliberately NOT accepted for
    the settle: a zero settle is the torn-tree race this constant exists to
    prevent, and a test that wants "fast" can say ``0.1``. The two readers
    below spell their variable names out as literals so the
    ambient-environment audit (``test_ambient_env_isolation``) can see them.
    """
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def build_settle_seconds() -> float:
    """``LOP_BUILD_SETTLE_S`` (test-only) or :data:`BUILD_SETTLE_S`."""
    return positive_seconds(os.environ.get("LOP_BUILD_SETTLE_S", ""), BUILD_SETTLE_S)


def build_stagger_seconds() -> float:
    """``LOP_BUILD_STAGGER_S`` (test-only) or :data:`BUILD_STAGGER_S`."""
    return positive_seconds(os.environ.get("LOP_BUILD_STAGGER_S", ""), BUILD_STAGGER_S)


def build_prefix() -> str | None:
    """Where to read the install stamp from: ``sys.prefix`` in production.

    ``LOP_BUILD_PREFIX`` exists ONLY so the e2e stage can point a real runtime
    (or a real ``lop serve``) at a temp directory carrying a fake
    ``.lop-source`` and flip it under the process. Nothing outside ``tests/e2e``
    sets it, and a production process that inherited it by accident would merely
    compare against a marker that never changes — it can never retire early.
    """
    return os.environ.get("LOP_BUILD_PREFIX") or None


def boot_build() -> "BuildStamp | None":
    """The stamp of the install on disk RIGHT NOW, or ``None`` if unreadable.

    The BOOT SAMPLE: a watcher takes this once at startup and compares every
    later read against it, which is the only way "the build moved under me" can
    be answered (see :func:`build_changed`). ``None`` — an editable checkout
    with no marker of its own, an unreadable dist-info — is not an error here:
    it means this process can never prove a move, so it never retires. That is
    the safe direction and it is deliberate (design-build-skew §6.5: a
    developer's worktree must not retire because they touched a file).
    """
    from local_operator import update as update_mod

    try:
        return update_mod.installed_build(build_prefix())
    except Exception:  # noqa: BLE001 — an unreadable boot stamp is "no baseline"
        logger.debug("boot build stamp unreadable; no build watch", exc_info=True)
        return None


def handover_build(boot: "BuildStamp | None") -> "BuildStamp | None":
    """The build the install on disk is PROVEN to have moved to, or ``None``.

    This is the reading both phases of the daemon's watch are made of: the
    detection that announces a handover, and the re-check every announced tick
    makes to ask whether the handover it announced is still the state of the
    install (``server/retire.py``). ``build_changed`` adds the settle window to
    it; the announced phase deliberately does not, because a move that has
    already been announced does not need to settle twice.

    ``None`` means the question is unproven, and it covers three shapes:

    * **the install is back to ``boot``** — a ``lop-update`` that failed and was
      rolled back, or one superseded by the running build. The process is the
      right one after all, so an announcement written earlier is withdrawn and
      the daemon goes on serving (review round 2, MINOR-2: the announcement used
      to be written once and never re-read, so the daemon still latched, exited
      and removed its record — leaving a reader a ``retiring_to`` that named a
      build no longer on disk);
    * **the stamp cannot be resolved into a build at all** — see
      :func:`proves_a_move`, the fail-closed rule QA round 2's OBS-1 asked for;
    * the exception/unreadable case below, the same direction for the same
      reason.

    A DIFFERENT build than the one announced is NOT ``None``: it is the newer
    move, and the caller re-announces onto it rather than leaving for a build
    that has already been replaced.
    """
    if boot is None:
        return None
    from local_operator import update as update_mod

    prefix = build_prefix()
    try:
        on_disk = update_mod.installed_build(prefix)
    except Exception:  # noqa: BLE001 — an unreadable stamp is "no change"
        logger.debug("build stamp unreadable; no refresh", exc_info=True)
        return None
    if on_disk == boot or not proves_a_move(boot, on_disk):
        return None
    return on_disk


def proves_a_move(boot: "BuildStamp", on_disk: "BuildStamp") -> bool:
    """Does ``on_disk`` read as a BUILD, rather than as a stamp nobody could read?

    WHY THIS GUARD IS HERE AND NOT IN ``update``. ``update.source_ref`` maps an
    absent, unreadable, empty or non-commit ``.lop-source`` to ``""`` (and
    ``write_source_marker`` documents why each of those must never raise), so
    ``installed_build`` answers a VERSION-ONLY stamp for all of them. That stamp
    is legitimate evidence of a move after a wheel upgrade — the marker reads
    ``pypi <version>``, the version itself moved, and the label
    ``0.54.39`` vs ``0.54.40`` names the handover honestly. It is *not* evidence
    of a move when the version is unchanged, because the ref is the PRIMARY key
    precisely for that case: ``lop-update`` builds from ``main`` while
    ``pyproject.toml`` still names the last release, so two genuinely different
    builds share one version string and only the recorded commit tells them
    apart. A version-only stamp that differs from a ``version@ref`` boot stamp
    therefore says "I could not read the ref", not "the build moved" — and
    leaving a process behind for a build it could not read is the failure QA
    round 2 measured on a real daemon (OBS-1: ``chmod 000`` on an aged marker,
    production settle, and the daemon announced, latched, exited and removed its
    record onto ``0.54.39``).

    It lives at this layer because the fix belongs to whoever ACTS on the answer
    rather than to the reader: ``update.py`` is deliberately total and silent
    (``installed_build`` is called from a runtime's construction path, where
    raising would stop every runtime on the host), and a caller that only
    COMPARES labels is right to treat an unreadable ref as a difference. Only a
    caller about to leave a process behind has to be strict, and both of this
    module's callers are.
    """
    if not on_disk.version and not on_disk.source_ref:
        # Nothing at all could be read: no dist-info, no marker. A stamp like
        # this labels as "unknown" and is not a build to leave for.
        return False
    if on_disk.source_ref:
        return True
    return on_disk.version != boot.version


def build_changed(boot: "BuildStamp | None") -> "BuildStamp | None":
    """The build now on disk, if it differs from ``boot`` AND has settled.

    ``None`` means "nothing to do": same stamp, an unreadable stamp, a boot
    stamp that was never captured (a reduced test server), a stamp that does not
    read as a build (see :func:`proves_a_move`), or a marker still inside the
    settle window (see ``BUILD_SETTLE_S``). Editable checkouts have no
    ``.lop-source`` and a constant version, so they never trip this — by design,
    matching ``design-build-skew.md`` §6.5: a developer's worktree runtime must
    not retire because they touched a file.
    """
    newer = handover_build(boot)
    if newer is None:
        return None
    from local_operator import update as update_mod

    prefix = build_prefix()
    try:
        age = update_mod.build_marker_age_s(prefix)
    except Exception:  # noqa: BLE001 — an unreadable marker is "not settled", not a dead watcher
        # The SETTLE read is guarded for the same reason the stamp read above it
        # is, and the reason is the consequence rather than the likelihood: this
        # function is called from a background watcher on BOTH sides (the daemon's
        # retirement poll and the runtime's refresh check), and an exception here
        # used to leave that task dead — for the daemon, a process that never
        # retires, silently (review round 1, MINOR-3).
        logger.debug(
            "build marker age unreadable; treating the install as unsettled", exc_info=True
        )
        return None
    if age is None or age < build_settle_seconds():
        # Younger than the settle, or unknowable: the install may still be
        # mid-write. Try again next check; the marker only gets older.
        return None
    return newer


def build_pair(boot: "BuildStamp | None", newer: "BuildStamp") -> str:
    """``" (old → new)"`` for the cut-off reason, or ``""`` without a boot stamp.

    The build pair is what makes a retirement self-explaining to whoever reads
    the reason later: "the process retired" is only actionable when it names
    which build it left for.
    """
    if boot is None:
        return ""
    return f" ({boot.label()} → {newer.label()})"
