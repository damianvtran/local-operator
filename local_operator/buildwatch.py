"""One definition of "the install on disk has been replaced under this process".

``lop-update`` (and :func:`local_operator.update.perform_upgrade`) used to
replace the installed tree IN PLACE, so a long-lived process could find itself
running a build that no longer exists on disk. Under the generation layout it no
longer does: an install lands in its OWN tree and the stable path is a pointer
(``local_operator.update`` documents the layout), so what moves is which build
the POINTER names, not the files any running process holds. The watch is
unchanged and still worth having, for a reason that has changed rather than
gone: it is now the CONVERGENCE path that retires a runtime on a superseded
generation once it is idle, where it used to be the only thing standing between
a running process and a tree being deleted under it. Two processes on this host
must react the same way to that: a session runtime
(:mod:`local_operator.session.runtime.process`)
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

It also carries the WARM-WINDOW term of the residency predicate — the "a wake is
about to fire" reason a runtime stays resident, :data:`WARM_WINDOW_S` and
:func:`wake_within_window` — and that one is here for a sharper version of the
same reason. The handle that asks it (``ServingSessionHandle.may_refresh``)
cannot import the runtime module to reach the helper: an import of a module with
no ``sys.modules`` entry is answered from DISK, and the state that matters is
exactly the one where the runtime's own files have been replaced — so the
function-local import raised ``ImportError``, the reaper read a failing
predicate as "not idle", and a draining runtime could never reach its exit
(QA round 1, Q-1). A stdlib-only module that BOTH sides already import is the
only home that is still importable at that moment.

The constants are re-exported by ``session/runtime/process.py`` under the names
that module has always published (``BUILD_CHECK_S``, ``_build_changed``, …) so
nothing outside this module had to change when the definition moved here.
"""

from __future__ import annotations

import logging
import os
import time
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
#: The daemon's internal injected-callback tests reuse this refusal window.
#: Production daemon polling announces only and never staggers toward an exit
#: (see ``server/retire.py``); runtime retirement above is unchanged.
BUILD_STAGGER_S = 20.0


#: THE ROTATION ANSWERS, one definition each.
#:
#: A runtime answers ``lop refresh`` with the sentence it decided from
#: (``Server._refresh_if_idle``) and the caller routes on that sentence rather
#: than re-deriving the decision (``session/runtime/control.refresh_session``),
#: so the two ends are a WIRE CONTRACT and a reword at either end fails
#: SILENTLY: the matcher misses and falls through to its generic ``kept``
#: branch, turning a diagnosis into "was not moved: …" (review round 2, NIT-1).
#: They live HERE, beside the settle question they describe, for the reason
#: ``types.LEAVING_ON_SIGNAL`` gives for living in ``types``: one module both
#: ends import, so a reword is a loud failure at one call site instead of a
#: silent drift across three literals.
KEPT_MATCHES = "kept: build on disk matches"

#: The install on disk has moved and has not settled yet. Distinct from
#: ``KEPT_MATCHES`` on purpose: a caller must never read one as the other, and
#: collapsing them into "already current" with a zero exit status IS the D1/M2
#: defect this answer exists to fix (PR #1141).
KEPT_UNSETTLED = "kept: the install on disk has not settled yet"

#: THE RETIRED HEDGE, honoured FOR EVER rather than only during a rollout.
#:
#: Before the two answers above existed, one runtime sentence covered both: a
#: runtime that had not judged a freshly moved marker said this, and so did one
#: whose install genuinely matched. It is still spoken by every runtime started
#: before this change — a running process keeps its own code until build skew
#: retires it, and that skew window is precisely the window ``lop refresh`` is
#: aimed at (its own docstring says its first run is ``lop-update``) — so a
#: caller that matched only ``KEPT_MATCHES`` would miss the whole fleet that
#: exists at update time and answer "already current" about an install that has
#: moved. ``KEPT_MATCHES`` is therefore matched as a PREFIX, which covers this
#: sentence too: honouring the retired string is a cross-version contract, not
#: an implementation detail to be tidied away later.
KEPT_MATCHES_OR_UNSETTLED = "kept: build on disk matches (or has not settled)"


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


#: A runtime whose own scheduler will fire a wake within this window stays
#: resident instead of exiting and paying a ~1.2 s cold start (plus the
#: supervisor's tick latency) to come back for it. Chosen to exceed
#: ``MIN_WAKE_INTERVAL_MS`` (60 s, harness.wake) by a margin: a session with
#: the tightest allowed recurrence then never thrashes exit → spawn → exit
#: once a minute, because the next fire is always inside the window. Anything
#: due further out is cheaper to leave to a cold spawn than to hold ~283 MB
#: for. Not env-tunable on purpose — it pairs with a constant in the wake
#: layer, and a knob would let the two drift apart.
WARM_WINDOW_S = 90.0


def wake_within_window(handle: object, *, now_ms: int | None = None) -> bool:
    """Term 2 of the residency predicate: does this runtime's OWN scheduler
    have a wake due within ``WARM_WINDOW_S``? Read through the handle (an
    optional capability, probed) so reduced test handles and older handle
    implementations that never grew the accessor behave as "no wakes" rather
    than crash the reaper.

    Failing open is the policy on purpose: a broken accessor must not pin the
    runtime it was asked about. See this module's docstring for why the
    function lives here rather than in the module that used to define it.
    """
    accessor = getattr(handle, "next_wake_due_at", None)
    if not callable(accessor):
        return False
    try:
        due_at = accessor()
    except Exception:  # noqa: BLE001 — a broken accessor must not pin the runtime
        logger.debug("next_wake_due_at failed; treating as no wake", exc_info=True)
        return False
    if not isinstance(due_at, int) or isinstance(due_at, bool):
        return False  # None, or a shape this reaper does not understand
    now = int(time.time() * 1000) if now_ms is None else now_ms
    return due_at - now <= WARM_WINDOW_S * 1000


def build_prefix() -> str | None:
    """Where to read THIS process's install stamp from: ``sys.prefix``.

    The BOOT sample's prefix, so the e2e stage's fake tree stands in for this
    process's own generation — which under the generation layout is a different
    question from :func:`disk_prefix`, and answering them with ONE prefix is how
    a stamp with this build's version and the pointer's ref gets built.

    ``LOP_BUILD_PREFIX`` exists ONLY so the e2e stage can point a real runtime
    (or a real ``lop serve``) at a temp directory carrying a fake
    ``.lop-source`` and flip it under the process. Nothing outside ``tests/e2e``
    sets it, and a production process that inherited it by accident would merely
    compare against a marker that never changes — it can never retire early.
    """
    return os.environ.get("LOP_BUILD_PREFIX") or None


def disk_marker_prefix() -> str | None:
    """Where the DISK install's marker — and its age — is read from.

    The POINTER's generation in production, which is the install whose freshness
    the settle window is about: under the generation layout this process's own
    tree is written once and never touched, so its marker's age says nothing
    about whether the build the pointer names has stopped being rewritten. Left
    as the boot prefix it would read the running tree's old marker and report
    "settled" about an install that had only just landed — disabling the settle
    exactly where it is still doing work.

    ``LOP_BUILD_PREFIX`` overrides it for the same reason it overrides the boot
    prefix: the e2e stage's temp tree stands in for a generation, and its fresh
    marker is what that stage flips.

    ``None`` means "this interpreter's own tree", which is what
    :func:`local_operator.update.build_marker_age_s` does with it and what the
    pre-generation behaviour was. Reachable only when the pointer cannot be
    resolved, in which case no move can be detected either and the settle is
    never consulted.
    """
    override = os.environ.get("LOP_BUILD_PREFIX")
    if override:
        return override
    from local_operator import update as update_mod

    try:
        root = update_mod.current_install_root()
    except Exception:  # noqa: BLE001 — an unreadable pointer is "no answer here"
        logger.debug("install pointer unreadable", exc_info=True)
        return None
    return str(root) if root is not None else None


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
    * **there is no install on disk to compare against** — an editable checkout,
      or a machine whose pointer cannot be resolved. See
      :func:`local_operator.update.disk_build`, which is the ONE place that
      decides; the exception/unreadable case below is the same direction for the
      same reason.

    A DIFFERENT build than the one announced is NOT ``None``: it is the newer
    move, and the caller re-announces onto it rather than leaving for a build
    that has already been replaced.
    """
    if boot is None:
        return None
    from local_operator import update as update_mod

    try:
        on_disk = update_mod.disk_build(build_prefix())
    except Exception:  # noqa: BLE001 — an unreadable stamp is "no change"
        logger.debug("build stamp unreadable; no refresh", exc_info=True)
        return None
    if on_disk is None or on_disk == boot or not proves_a_move(boot, on_disk):
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


def _settle_elapsed() -> bool:
    """Has the install marker on disk aged past the settle window?

    The settle term of :func:`build_changed`, extracted because
    :func:`pending_build` needs the SAME answer to the opposite question. Two
    readings of one marker that could disagree (one asking "may I act yet", the
    other "is the move merely young") is exactly the drift ``BUILD_SETTLE_S``
    exists to prevent, so both go through here.

    ``False`` for an unreadable or missing age — "younger than the settle, or
    unknowable: the install may still be mid-write" — which is "not settled" in
    the safe direction: it defers a retirement and describes a refresh as
    incomplete rather than claiming either is done. There is no retry logic
    here because none is needed: the marker only gets older, so the next check
    is the retry.
    """
    from local_operator import update as update_mod

    try:
        age = update_mod.build_marker_age_s(disk_marker_prefix())
    except Exception:  # noqa: BLE001 — an unreadable marker is "not settled", not a dead watcher
        # The SETTLE read is guarded for the same reason the stamp read in
        # ``handover_build`` is, and the reason is the consequence rather than
        # the likelihood: this function is called from a background watcher on
        # BOTH sides (the daemon's retirement poll and the runtime's refresh
        # check), and an exception here used to leave that task dead — for the
        # daemon, a process that never retires, silently (review round 1,
        # MINOR-3).
        logger.debug(
            "build marker age unreadable; treating the install as unsettled", exc_info=True
        )
        return False
    return age is not None and age >= build_settle_seconds()


def build_changed(boot: "BuildStamp | None") -> "BuildStamp | None":
    """The build now on disk, if it differs from ``boot`` AND has settled.

    ``None`` means "nothing to do": same stamp, an unreadable stamp, a boot
    stamp that was never captured (a reduced test server), a stamp that does not
    read as a build (see :func:`proves_a_move`), or a marker still inside the
    settle window (see ``BUILD_SETTLE_S``). Editable checkouts have no
    ``.lop-source`` and a constant version, so they never trip this — by design,
    matching ``design-build-skew.md`` §6.5: a developer's worktree runtime must
    not retire because they touched a file.

    THE LAST TWO SHAPES ARE DIFFERENT FACTS and :func:`pending_build` is how a
    caller tells them apart; this function answers only "may I act now", which
    is what both of its callers need.
    """
    newer = handover_build(boot)
    if newer is None:
        return None
    if not _settle_elapsed():
        return None
    return newer


def pending_build(boot: "BuildStamp | None") -> "BuildStamp | None":
    """The build on disk that has MOVED but has not settled yet, or ``None``.

    The complement of :func:`build_changed` on the same two reads, for the one
    caller that has to report the difference: ``lop refresh`` asks a runtime
    "are you on the build on disk", and the runtime used to answer
    ``kept: build on disk matches (or has not settled)`` for both shapes. Inside
    the settle window that sentence is FALSE about a runtime still on the old
    build, and the window is not an edge case for that command — its own
    docstring says its first run is ``lop-update``, i.e. the operator invokes it
    in exactly those seconds. Telling them "already current" (with a zero exit
    status) about a fleet that is about to rotate is the kind of wrong answer
    that stops an operator looking (D1/M2, PR #1141).

    ``None`` for a genuinely-matching install, and for every unreadable-stamp
    shape ``handover_build`` refuses — in those the settle window is not what is
    being described.
    """
    newer = handover_build(boot)
    if newer is None or _settle_elapsed():
        return None
    return newer


def moved_and_unsettled(version: str, source_ref: str) -> bool:
    """The settle question about a stamp somebody ELSE published.

    :func:`pending_build` asks it about the stamp THIS process booted from. A
    caller holding a DISCOVERY RECORD — the build a runtime published about
    itself — needs the same answer about a build it never loaded, and that
    caller is ``lop refresh``: the answer it reads comes from the runtime's own
    code, so inside the settle window after ``lop-update`` the whole live fleet
    is still running the PREVIOUS build and can only answer what that build
    knew. Reading the marker here is what makes the caller honest about those
    runtimes without asking anything of them (design round 2 D1 / UX round 2
    U6 / QA round 2 O1, PR #1141); ``control.refresh_session`` carries the full
    argument.

    ``False`` when the record published no stamp at all: with nothing to
    compare, the marker cannot be said to have moved PAST it, so the runtime's
    own "matches" stands rather than being second-guessed.

    An UNREADABLE OR ABSENT MARKER AGE counts as NOT SETTLED, which is
    ``_settle_elapsed``'s documented direction rather than a new rule — the
    install may still be mid-write, and the cost of the doubt is one more ask.
    A read that RAISES is the other fallback (``False``: no evidence of a
    move), because this is called while composing a receipt for a person and a
    failed probe there must not become a traceback.
    """
    if not version and not source_ref:
        return False
    try:
        from local_operator.update import BuildStamp

        # ``pending_build`` is the ONE definition of this question; asking it
        # again here in terms of a raw marker read is how the two ends would
        # come to disagree about the same file.
        return pending_build(BuildStamp(version=version, source_ref=source_ref)) is not None
    except Exception:  # noqa: BLE001 — a failed probe is "no evidence", not a crash
        logger.debug("settle question unanswerable for a published stamp", exc_info=True)
        return False


def build_pair(boot: "BuildStamp | None", newer: "BuildStamp") -> str:
    """``" (old → new)"`` for the cut-off reason, or ``""`` without a boot stamp.

    The build pair is what makes a retirement self-explaining to whoever reads
    the reason later: "the process retired" is only actionable when it names
    which build it left for.
    """
    if boot is None:
        return ""
    return f" ({boot.label()} → {newer.label()})"
