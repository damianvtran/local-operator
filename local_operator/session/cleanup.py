"""The ONE session-cleanup policy, and the only code allowed to remove a
session directory.

WHY THIS MODULE EXISTS, AND WHY IT IS OFF
=========================================

Every earlier automatic cleanup under ``sessions/`` is gone. The first
generation (age/count/byte ceilings) evicted a running session's transcript
out from under it. The second (an "empty directory" reaper) grew, in #576,
into a backfill that removed any directory whose transcript held no
``"role": "user"`` row — behind an opt-out toggle that ``/settings`` wrote
under a nested key and the reaper read under a flat one, so it never
worked. The runtime exit path carried a fourth (#622, an ``rmdir`` on
lease-only directories). On the night the incident was noticed, 225 of an
operator's 244 named sessions — 296,617 model calls of history — were gone.

The forensics could not pin every deletion on one of those reapers: none of
the logged reaps had a model call, and no reaper could remove a directory
holding a transcript. That is why this module is built around the
assumption that **something outside its own judgement may try to delete
the real store**, and refuses on the actor's behalf:

- **Nothing runs unless** ``session.cleanup.enabled`` **is true.** The
  default is false. Every limit below is inert while it is false, whatever
  value it holds — :func:`cleanup_from_config` returns before it lists the
  store.
- **The store must be marked.** :func:`remove_session_dir` refuses any
  target that is not directly under a ``sessions/`` directory carrying the
  :data:`STORE_MARKER_NAME` file, and refuses any target not under the
  config dir the process was given. A store under ``/tmp`` or a test's
  ``tmp_path`` has the marker only if its creator wrote it — the deliberate
  opt-in — and the operator's real store gets it once, from
  :func:`mark_store` on first use and from the config migration.
- **Every removal is logged at WARNING with the session id, the policy
  that chose it and the reason**, and appended to ``sessions/.cleanup-log.
  jsonl`` — an always-on, per-store record, because the earlier reaps were
  only discoverable by hunting through per-process log files.
- **Hard guards apply even when enabled.** A session with a live claim,
  lease or pid; one with an armed wake (DORMANT entries — the ``stopped_at``
  marker ``/stop`` writes, which the supervisor skips — do not guard, see
  :func:`_has_armed_wake`); one with unread spooled mail; the
  session being built right now; and the :data:`RECENT_KEEP` most recently
  active sessions are never candidates. The guards are checked per
  directory at decision time, not from a cached listing.
- **Fail closed.** Any guard that cannot be evaluated (unreadable file,
  stat error, import failure) keeps the directory.

The enforcement is ``tests/unit/session/test_no_session_deletion.py``: it
walks ``local_operator/`` and fails, naming file:line, on any call named
``rmtree``/``rmdir``/``removedirs``/``rename``/``renames``/``replace``/
``move``/``unlink``/``remove`` outside this module — through any import
alias and on any receiver — unless allow-listed with a reason; and it
asserts this module removes nothing when disabled, with every limit set to
1 over a 50-session store.

The policy is also runnable by hand: ``lop sessions cleanup --dry-run``
lists what the limits WOULD remove (it may do so with the switch off, and
says so), and ``lop sessions cleanup`` runs it — only with ``enabled:
true``, or with ``--force`` after listing and a typed confirmation. Both
honour every hard guard.

The store marker is a guard against foreign and unmarked targets and the
CLI on a store nothing has booted; it is NOT a second gate on the harness's
own startup pass, which marks its store in ``_prepare`` before maintenance
runs. That pass is gated by ``enabled`` alone (QA round 1, Q4).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from local_operator.session.retention import (
    _SIDECAR_NAMES,
    LIVE_MARKER_NAME,
    SESSIONS_DIRNAME,
    TRANSCRIPT_FILENAME,
    _is_claimed,
    _process_alive,
    session_activity,
)

logger = logging.getLogger(__name__)

#: The path of the cleanup block under ``values``, spelled ONCE. The
#: ``settings_io`` rows use ``CLEANUP_PATH + (leaf,)`` and the consumer reads
#: ``ConfigManager.get_nested_value(CLEANUP_PATH + (leaf,))``; both sides share
#: this tuple so the flat-vs-nested mismatch that silenced the #576 opt-out
#: cannot be reintroduced by a typo on one side.
CLEANUP_PATH: tuple[str, ...] = ("session", "cleanup")

#: Defaults, one per leaf. These are the consumer's defaults — the registry
#: (``settings_io``) is checked against them by
#: ``test_every_default_matches_its_consumer``. ``enabled`` is False and every
#: limit is "unlimited": a config that says nothing gets no cleanup.
DEFAULT_ENABLED = False
DEFAULT_MAX_SESSIONS = 0
DEFAULT_MAX_INACTIVE_DAYS = 0
DEFAULT_MAX_TOTAL_BYTES = 0
DEFAULT_REMOVE_EMPTY = False

#: Marker file that a ``sessions/`` directory must carry before
#: :func:`remove_session_dir` will remove anything inside it. Written by
#: :func:`mark_store` when the harness creates or migrates its own store.
#: A directory without it is either not a local-operator store, or a store
#: from before the marker existed — both mean "refuse". Dotted so it never
#: reads as a session id to a listing.
STORE_MARKER_NAME = ".local-operator-store"

#: Append-only record of every REAL removal, inside the store it describes.
#: One JSON object per line: ``{"at", "session", "title", "policy", "reason",
#: "actor", "pid"}``. Dry runs are NOT recorded — a rehearsal in the same file
#: as the losses made the log unreadable (13 of 22 rows in one QA store were
#: rehearsals). ``actor`` is ``"startup"`` (the maintenance pass, gated on
#: ``enabled``) or ``"cli"`` (``lop sessions cleanup``), which is the first
#: question after "why": who did it.
CLEANUP_LOG_NAME = ".cleanup-log.jsonl"

#: The LAST startup removal, as one JSON object beside the log: ``{"at",
#: "actor", "pid", "removed": N, "scanned": N, "policies": {name: count},
#: "record": path, "acknowledged": bool}``. Written only by a pass that
#: removed ≥1 session; overwritten by the next such pass. This is how a boot
#: that removed sessions gets to SAY SO on screen: the maintenance pass runs
#: in the runtime process, after the first frame, and the TUI may be a
#: viewer attached over a socket — so the fact is put on disk where every
#: viewer of that store can read it, and the first one to report it flips
#: ``acknowledged`` so the same removal is announced once, not on every
#: ``/resume``. Deliberately a file, not a notification bus: one fact, one
#: reader shape, durable across the process that produced it (UX round 1,
#: U1; the incident's launches painted an identical splash after 225
#: removals).
LAST_CLEANUP_NAME = "last-cleanup.json"

#: The N most recently active sessions are never candidates, whatever the
#: policy says. 10 matches the resume picker's first page: a session the user
#: can see at the top of ``/resume`` must not vanish between two launches.
RECENT_KEEP = 10


@dataclass(frozen=True)
class CleanupPolicy:
    """The five knobs, as read from config or the CLI. ``enabled`` gates
    everything; a limit of 0 means "no limit of that kind"."""

    enabled: bool = DEFAULT_ENABLED
    max_sessions: int = DEFAULT_MAX_SESSIONS
    max_inactive_days: int = DEFAULT_MAX_INACTIVE_DAYS
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    remove_empty: bool = DEFAULT_REMOVE_EMPTY

    @property
    def has_any_limit(self) -> bool:
        return bool(
            self.max_sessions or self.max_inactive_days or self.max_total_bytes or self.remove_empty
        )


@dataclass(frozen=True)
class Candidate:
    """One directory the policy chose, and why. Returned so the CLI's dry run
    and the tests can assert on decisions rather than scrape log lines.

    Carries what a user needs to JUDGE the decision, not only the id: a
    12-hex id says nothing, "Research thread 7 · 29d · 28 kB" does (UX round
    1, U4). ``title`` is resolved the way the resume picker resolves it.
    """

    session: str
    policy: str
    reason: str
    title: str = ""
    idle_days: float = 0.0
    size_bytes: int = 0
    #: ``"user"`` for the user's own conversation (no origin marker),
    #: otherwise the recorded origin (``"subagent"``, ``"fork"``). Shown on
    #: every dry-run row and counted in the launch notice because a store
    #: that is 85% subagent-origin reads as "159 removed" without it, and
    #: the user cannot tell whether those were theirs (UX round 3, U15).
    origin: str = "user"
    #: False for a directory that was never worked in (no transcript, no
    #: spool — an idle open-and-quit launch). The launch notice is silent
    #: when EVERY removal was one of these: a "removed 1 session" on every
    #: open-and-quit would train the user to ignore the one notice that
    #: matters (design round 3, N6). The record and the jsonl still say.
    active: bool = True


@dataclass
class CleanupResult:
    scanned: int = 0
    #: Directories the policy chose. In a dry run these were NOT removed.
    chosen: list[Candidate] = field(default_factory=list)
    removed: list[Candidate] = field(default_factory=list)
    #: ``(session, guard)`` for every directory a hard guard kept that a limit
    #: would otherwise have taken. Reported so a dry run shows what is
    #: protected, not only what is doomed.
    protected: list[tuple[str, str]] = field(default_factory=list)
    errors: int = 0
    dry_run: bool = False
    #: Why nothing ran, when nothing ran ("disabled", "no limits", "no store").
    skipped: str | None = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _coerce_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _coerce_bool(value: Any, default: bool) -> bool:
    """The settings registry's strict parser, so the page and the policy read
    the master switch identically (R1-6, R2-4). Imported lazily: this module
    must stay light for the runtime child, and ``settings_io`` is the page's
    module — but the PARSER is one function and there must be one of it."""
    from local_operator.settings_io import strict_bool

    return strict_bool(value, default)


def policy_from_config(config_manager: Any) -> CleanupPolicy:
    """Read the policy through :meth:`ConfigManager.get_nested_value` on
    :data:`CLEANUP_PATH` — the exact path ``settings_io`` writes.

    Duck-typed so a test can hand in any object with the method. A manager
    without ``get_nested_value`` (a stub predating it) yields the disabled
    default rather than guessing from ``get_config_value``: a reader that
    silently used the flat accessor is the bug this module replaces.
    """
    getter: Callable[..., Any] | None = getattr(config_manager, "get_nested_value", None)
    if getter is None:
        return CleanupPolicy()

    def leaf(name: str, default: Any) -> Any:
        try:
            return getter(CLEANUP_PATH + (name,), default)
        except Exception:  # noqa: BLE001 — a broken config is not a reason to run
            return default

    return CleanupPolicy(
        enabled=_coerce_bool(leaf("enabled", DEFAULT_ENABLED), DEFAULT_ENABLED),
        max_sessions=_coerce_int(leaf("max_sessions", DEFAULT_MAX_SESSIONS), DEFAULT_MAX_SESSIONS),
        max_inactive_days=_coerce_int(
            leaf("max_inactive_days", DEFAULT_MAX_INACTIVE_DAYS), DEFAULT_MAX_INACTIVE_DAYS
        ),
        max_total_bytes=_coerce_int(
            leaf("max_total_bytes", DEFAULT_MAX_TOTAL_BYTES), DEFAULT_MAX_TOTAL_BYTES
        ),
        remove_empty=_coerce_bool(leaf("remove_empty", DEFAULT_REMOVE_EMPTY), DEFAULT_REMOVE_EMPTY),
    )


# ---------------------------------------------------------------------------
# The store marker
# ---------------------------------------------------------------------------


def store_marker_path(sessions_dir: Path) -> Path:
    return sessions_dir / STORE_MARKER_NAME


def mark_store(sessions_dir: Path) -> None:
    """Stamp ``sessions_dir`` as a local-operator store.

    Idempotent and best-effort: a store that cannot be marked simply stays
    outside cleanup's reach, which is the safe side. Called when the harness
    creates its own store (session construction) and by the config migration,
    never by cleanup itself — the policy must not be able to authorise its
    own target.
    """
    try:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        marker = store_marker_path(sessions_dir)
        if not marker.exists():
            marker.write_text(
                "This directory is a local-operator session store. "
                "Its presence lets `lop sessions cleanup` remove sessions here "
                "when the user has enabled the policy.\n",
                encoding="utf-8",
            )
    except OSError as exc:
        logger.debug("session cleanup: cannot mark store %s: %s", sessions_dir, exc)


def _refusal(target: Path, config_dir: Path | None) -> str | None:
    """Why :func:`remove_session_dir` must not touch ``target``; ``None`` if it may.

    Three independent checks, each sufficient to refuse: the target must be
    directly under a ``sessions/`` directory; that directory must carry the
    store marker; and, when a ``config_dir`` is given, it must be THAT config
    dir's store. Paths are resolved so a symlink into the real store cannot
    launder itself through a marked scratch store.
    """
    try:
        resolved = target.resolve(strict=True)
    except OSError:
        return "target does not exist or cannot be resolved"
    parent = resolved.parent
    if parent.name != SESSIONS_DIRNAME:
        return f"not directly under a '{SESSIONS_DIRNAME}/' directory"
    if not store_marker_path(parent).is_file():
        return f"store carries no {STORE_MARKER_NAME} marker"
    if config_dir is not None:
        try:
            expected = (config_dir / SESSIONS_DIRNAME).resolve()
        except OSError:
            return "config dir cannot be resolved"
        if parent != expected:
            return f"store {parent} is not this process's store {expected}"
    if resolved.is_symlink() or not resolved.is_dir():
        return "not a directory"
    return None


def remove_session_dir(
    target: Path,
    *,
    config_dir: Path | None,
    policy: str,
    reason: str,
    actor: str,
    title: str = "",
    dry_run: bool = False,
) -> bool:
    """THE ONLY ``rmtree`` OF A SESSION DIRECTORY IN THIS CODEBASE.

    Refuses unless :func:`_refusal` clears the target; logs the refusal at
    WARNING so an attempt against an unmarked or foreign store is visible.
    Logs every real removal at WARNING (naming the record file) and appends
    it to the store's :data:`CLEANUP_LOG_NAME` BEFORE the ``rmtree``, so a
    crash mid-removal still leaves the record. A dry run refuses and decides
    exactly as a real run would but writes nothing and logs at DEBUG — the
    CLI prints the decisions itself, and a WARNING per rehearsal doubled
    every line in a terminal (UX round 1, U3). Returns whether the directory
    was (or, in a dry run, would have been) removed.
    """
    why_not = _refusal(target, config_dir)
    if why_not is not None:
        logger.warning(
            "session cleanup: REFUSED to remove %s (%s); policy=%s reason=%s",
            target,
            why_not,
            policy,
            reason,
        )
        return False
    if dry_run:
        logger.debug(
            "session cleanup (dry run): would remove %s; policy=%s reason=%s",
            target.name,
            policy,
            reason,
        )
        return True
    record = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "session": target.name,
        "title": title,
        "policy": policy,
        "reason": reason,
        "actor": actor,
        "pid": os.getpid(),
    }
    _append_cleanup_log(target.parent, record)
    logger.warning(
        "session cleanup: removing %s%s; policy=%s reason=%s actor=%s (record: %s)",
        target.name,
        f" ({title})" if title else "",
        policy,
        reason,
        actor,
        target.parent / CLEANUP_LOG_NAME,
    )
    shutil.rmtree(target)
    return True


#: The record ``policy`` string for a delete the USER asked for, as opposed to
#: one of the automatic limits. It is what tells the two apart in the cleanup
#: log, which is the only durable record either of them leaves.
EXPLICIT_DELETE_POLICY = "explicit-delete"

#: The user-facing sentence for each hard guard an explicit delete can be
#: refused by, keyed by the reason :func:`_guard` returns.
#:
#: ONE SENTENCE PER GUARD rather than one blanket refusal, because the remedies
#: differ: a running session is stopped, an armed wake is cancelled from the
#: conversation or its entry file, unread mail is read by reopening. Every
#: sentence names an action the user can actually take — review round 1 and UX
#: round 1 both found sentences naming doors that did not exist. The mapping is
#: closed on purpose — a reason it does not carry still refuses through the
#: fallback below, so a guard added to :func:`_guard` later cannot quietly become
#: a condition the user is allowed to delete through. That direction is the whole
#: safety property here: every path that does not
#: positively clear is a refusal.
_GUARD_REFUSALS: dict[str, str] = {
    "claimed by a live process": (
        "That conversation is open in a running session. Stop it before deleting it."
    ),
    "leased by a live process": (
        "That conversation is open in a running session. Stop it before deleting it."
    ),
    "has an armed wake": (
        # THE REMEDY HAS TO EXIST SOMEWHERE THE USER CAN REACH (UX round 1, U1).
        # "Cancel the wake before deleting it" named an action with no surface:
        # the composer offers no wake command, the wake band lists schedules with
        # no cancel action or binding, and `lop wake`'s own copy says there is no
        # cancel (`cli.py` names the entry FILE for the same reason, round 3
        # D20). The two doors that do exist are the conversation itself — the
        # model-facing `wake` tool cancels by schedule id, so asking is an action
        # a user can take — and the index entry, which is the same file remedy
        # the CLI's ghost row already names.
        #
        # ``{wake_file}`` IS RESOLVED AND STORE-RELATIVE (design round 2, D8;
        # desktop QA round 4, Q14). The sentence first printed a literal
        # ``<session-id>`` template on a screen where the app holds the id and
        # prints it elsewhere (the rehearsal one keystroke earlier says
        # ``(c3d4e5f6a1b2)``), so a user was told to delete a file whose name they
        # had to go and find. Resolving it to a machine-ABSOLUTE path fixed that
        # finding and broke a different rule: this sentence is read inside a
        # confirmation dialog for a conversation the user is looking at, and an
        # absolute path leaks the layout of the host and reads as an internal
        # detail. ``wakes/<id>.json`` is the form the CLI's own ghost row uses, so
        # the family reads in one register, and the id stays resolved so the door
        # is addressable without a template.
        "That conversation has a wake armed for it. Reopen it and ask it to cancel "
        "the wake, or delete its {wake_file} entry, before deleting it."
    ),
    "has unread spooled mail": (
        # "Read them" named no way to read them (UX round 1, U4). The spool
        # drains once, at open (`inbox.drain_inbox`), and no command reads
        # another conversation's inbox — so reopening that conversation IS the
        # action, and it is the one this names.
        "That conversation has unread messages waiting. Reopen it to read them "
        "before deleting it."
    ),
}

#: Used for a guard reason with no sentence of its own, INCLUDING a guard that
#: could not be evaluated. Deliberately says "could not be checked" rather than
#: naming a condition it did not observe: the user's next move differs, and
#: telling them to stop a session that is not running sends them looking for
#: something that is not there.
_GUARD_REFUSAL_FALLBACK = (
    "Whether that conversation is in use could not be checked, so nothing was deleted."
)

#: Used when the deletion itself declined for a reason that is not a guard —
#: an unmarked or foreign store, which :func:`_refusal` owns. Nothing was
#: removed and the caller must not report success, so it is a refusal rather
#: than an empty answer.
_DELETE_REFUSAL_FALLBACK = "That conversation could not be deleted, so nothing was removed."


@dataclass(frozen=True)
class DeleteOutcome:
    """What an explicit delete decided, in the shape both frontends answer from.

    ``found=False`` is the 404 case (unknown or malformed id) and carries no
    sentence: the caller cannot act on the difference between the two, and
    inventing a distinction with no remedy behind it is the rule the desktop
    pin route already applies to its own 404.

    ``refusal`` non-empty means NOTHING WAS DELETED and the sentence names why
    — the 409 case. It is a field rather than an exception because the desktop
    route has to map it to a status and the TUI paints it as a notice, and both
    want the same sentence.

    ``children`` is how many subagent runs this conversation launched, counted
    so a receipt can say they are KEPT. It is information about the blast
    radius, never an input to the decision: the deletion removes one directory.

    ``label`` is how a sentence names the conversation — title first, id in
    parentheses. The hosts all print it rather than re-deriving it, so a
    rehearsal cannot name the target in one frontend and not another (design
    round 1, D2). Empty when ``found`` is false: an id that resolves to nothing
    has no title to show and the 404 sentence names it already.
    """

    session_id: str
    found: bool
    deleted: bool
    refusal: str = ""
    children: int = 0
    label: str = ""

    def rehearsal(self) -> str:
        """The sentence a two-step delete shows BEFORE the confirmed one runs.

        ON THE OUTCOME rather than retyped in the three hosts that ask for it
        (review round 3, R3-2): ``label`` was shared but the sentence around it
        was copied verbatim into ``tui/app.py`` twice and
        ``session/runtime/serving.py`` once, so a wording edit in one host would
        silently give the terminal and the detached runtime different
        confirmations for the same irreversible act — and only the local host had
        a test. The children clause belongs here with it: it is the fact that says
        what the deletion does NOT touch, and it was duplicated on the same terms.

        PERMANENCE IS STATED ONCE. "for good — it cannot be undone" said the same
        thing twice in one breath and wrapped ``for good`` across lines at the
        standard width (design round 2, D10); the irreversibility is the part a
        user must take away, so it is the part that stays.
        """
        kept = f" {self.children} subagent run(s) it started are kept." if self.children else ""
        return (
            f"/delete removes {self.label} and its transcript — it cannot be "
            f"undone.{kept} Run /delete yes to confirm."
        )


def session_label(directory: Path) -> str:
    """How a destructive sentence names the conversation it will remove.

    TITLE FIRST because the id is not on the screen the rehearsal is typed into
    (design round 1, D2): the status band carries the model and the cwd, and the
    id appears only as a dim right-hand column inside a different screen
    (``/resume``). A rehearsal that named only the id asked a user to confirm the
    destruction of something they could not see — and in the PR's own frame the
    fixture's id was the string ``sess``, which reads as a truncated word.

    Falls back to ``this conversation (<id>)`` when the session has no stored
    title rather than inventing one: the lists have no label for it either, so a
    second spelling of "Untitled conversation" here would be a name the user
    never chose and cannot search for.
    """
    try:
        from local_operator.resume import stored_session_title

        title = stored_session_title(directory)
    except Exception:  # noqa: BLE001 — a label is never worth failing a delete over
        title = ""
    return f"“{title}” ({directory.name})" if title else f"this conversation ({directory.name})"


def _subagent_child_count(directory: Path) -> int:
    """How many subagent runs ``directory`` launched, best-effort, or 0.

    Read from the runtime's own roster sidecar through the same reader the
    desktop's child routes use (``session.session._read_roster_sidecar``) rather
    than re-parsing the file, so "how many children does this conversation
    have" has one answer per conversation.

    BEST-EFFORT IN THE SAFE DIRECTION: an unreadable roster reports 0, which
    understates what survives the deletion — and what survives is the point.
    The removal itself takes exactly one directory whether this answers or not,
    so a low count can never widen the blast radius; it can only make a receipt
    less informative.
    """
    try:
        from local_operator.session.session import (
            SUBAGENT_ROSTER_SIDECAR,
            _read_roster_sidecar,
        )

        payload = _read_roster_sidecar(directory / SUBAGENT_ROSTER_SIDECAR)
    except Exception:  # noqa: BLE001 — bookkeeping for a receipt, never a decision
        return 0
    if not payload:
        return 0
    records = payload.get("records")
    return len(records) if isinstance(records, list) else 0


def delete_session(
    config_dir: Path,
    session_id: str,
    *,
    actor: str,
    now: float | None = None,
    dry_run: bool = False,
) -> DeleteOutcome:
    """Delete ONE conversation the user explicitly asked to delete.

    THE ONLY ENTRY POINT BESIDES THE POLICY, and it goes through
    :func:`remove_session_dir` like everything else — this module stays the one
    place a session directory is removed, which is what
    ``tests/unit/session/test_no_session_deletion.py`` enforces by walking the
    package for a second ``rmtree``.

    WHAT AN EXPLICIT DELETE IS ALLOWED THAT THE POLICY IS NOT:

    * **``RECENT_KEEP`` does not apply.** That constant bounds the AUTOMATIC
      sweep — it exists so a routine run can never take the sessions someone is
      most likely to still want. A person who names a conversation and confirms
      a typed ``yes`` has answered the question ``RECENT_KEEP`` is a proxy for,
      so applying it would refuse the one deletion that is certainly intended.
    * **``CleanupPolicy.enabled`` does not apply.** The switch gates the
      automatic reapers (the module docstring's incident is why the default is
      off); an explicit delete is not one of them, and a user whose cleanup is
      disabled must still be able to delete a conversation they asked to
      delete.

    WHAT IT IS NOT ALLOWED, and every one of these is a refusal rather than a
    silent no-op:

    * **A running session.** Every hard guard in :func:`_guard` applies
      unchanged — a live claim, a live lease, an ARMED wake, unread spooled
      mail, and a guard that could not be evaluated. Deleting the directory of a
      session that is serving a turn leaves a process writing into a path that
      no longer exists, and the ledger the user reads would lose the record of
      the turn in flight. The refusal NAMES the guard it hit: the remedy is
      different for each, and a generic "in use" would send the user to stop a
      session that was never running.
    * **A session the user did not open.** A delegated subagent run resolves by
      id like any other session, but it is not a conversation anyone opened, so
      an id that is not ``is_user_session`` is answered as unknown rather than
      deleted. Deleting is irreversible and the user cannot see the row they are
      naming; the reversible verbs that keep hidden runs reachable (``/v1/desktop
      .../pin``, and ``/archive``) deliberately keep the looser id-shape-only
      admission, which is the asymmetry this sentence is here to explain.

    ``children`` on the outcome is the number of subagent runs the conversation
    launched, so a receipt can say those are KEPT: this removes exactly the
    addressed directory. Those runs live as siblings under ``sessions/`` and
    nothing here touches them — stated because the opposite is the natural
    assumption to make about a delete, and a user who assumed it would not look
    for their delegated work again.

    Deleting also leaves the PIN store and the ARCHIVE store consistent without
    touching either: both prune at read against the session store, so an id
    whose directory is gone reads back as neither pinned nor archived. That is
    asserted in this feature's tests rather than assumed here, and it is why
    this function has no import of either module. The wake index IS pruned
    explicitly, the same way the automatic path does it.
    """
    from local_operator.resume import is_user_session
    from local_operator.session.catalog import session_directory_name

    # The same id-shape guard the pin store applies to a stored entry, reused
    # rather than re-spelled: ``Path.__truediv__`` does not keep an id inside
    # ``sessions/`` (``sessions / "/tmp"`` IS ``/tmp``), so the shape test has
    # to come before the id is joined onto the store.
    if not session_directory_name(session_id) or session_id != Path(session_id).name:
        return DeleteOutcome(session_id=session_id, found=False, deleted=False)
    directory = config_dir / "sessions" / session_id
    if not directory.is_dir() or not is_user_session(directory):
        return DeleteOutcome(session_id=session_id, found=False, deleted=False)

    children = _subagent_child_count(directory)
    clock = time.time() if now is None else now
    reason = _guard(directory, config_dir, clock)
    if reason is not None:
        return DeleteOutcome(
            session_id=session_id,
            found=True,
            deleted=False,
            refusal=_guard_refusal(reason, session_id),
            children=children,
            label=session_label(directory),
        )
    removed = remove_session_dir(
        directory,
        config_dir=config_dir,
        policy=EXPLICIT_DELETE_POLICY,
        reason="the user deleted this conversation",
        actor=actor,
        title=_session_title(directory),
        dry_run=dry_run,
    )
    if removed and not dry_run:
        # Unreachable while the wake guard holds (an armed wake is a refusal
        # above), and still done: a wake entry whose session is gone is an index
        # pointing at nothing, and the guard is one config write away from being
        # bypassed by hand. Same call the automatic path makes, for the same
        # reason.
        _forget_wake_entry(config_dir, session_id)
    return DeleteOutcome(
        session_id=session_id,
        found=True,
        deleted=removed,
        refusal="" if removed else _DELETE_REFUSAL_FALLBACK,
        children=children,
        label=session_label(directory),
    )


def _append_cleanup_log(sessions_dir: Path, record: dict[str, Any]) -> None:
    try:
        with (sessions_dir / CLEANUP_LOG_NAME).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        # The removal still proceeds — the WARNING log line above carries the
        # same facts — but say so, because the jsonl is the record the user
        # is told to look at.
        logger.warning("session cleanup: cannot append to %s: %s", CLEANUP_LOG_NAME, exc)


# ---------------------------------------------------------------------------
# Hard guards
# ---------------------------------------------------------------------------


# EVERY GUARD FAILS CLOSED. A guard answers "may this directory be removed?"
# and the only safe answer to "I could not find out" is NO: the cost of a
# kept directory is bytes, the cost of a removed one is the incident. So
# each probe below treats any failure — an unreadable file, a corrupt
# payload, a missing module, a bug in the callee — as "keep", and
# :func:`_guard` wraps the lot so that an exception a probe did not
# anticipate is still "keep", named. Review round 3 (R3-1) found the
# recent-N guard returning an EMPTY set on failure and the run proceeding
# with zero recent protection; that shape is what this block forbids.


def _claimed(directory: Path, now: float) -> bool:
    """:func:`_is_claimed`, closed: an unreadable or unparseable marker keeps.

    ``_is_claimed`` itself reads a missing marker as "not claimed" — correct,
    absence is the common case — but an EACCES on a present marker also
    comes back as False there, because that function serves liveness
    questions where "assume dead" is the right default. Here it is not.
    """
    marker = directory / LIVE_MARKER_NAME
    try:
        if not marker.exists():
            return False
    except OSError:
        return True
    try:
        marker.read_text(encoding="utf-8")
    except OSError:
        return True  # present but unreadable: assume claimed
    return _is_claimed(directory, now)


def _lease_runtime_alive(directory: Path) -> bool | None:
    """Whether the ``.execution-lease`` names a live pid; ``None`` = no lease."""
    lease = directory / ".execution-lease"
    try:
        raw = lease.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return True  # unreadable: assume owned
    try:
        pid = json.loads(raw).get("pid")
    except (ValueError, AttributeError):
        return True  # corrupt lease: assume owned
    if not isinstance(pid, int):
        return True
    return _process_alive(pid)


def _has_armed_wake(config_dir: Path, session: str) -> bool:
    """Whether a wake for this session can still FIRE (U2).

    THE GUARD ASKS ABOUT FIRING, NOT ABOUT EXISTENCE. It used to be
    ``entry_path(...).exists()``, and `/stop` deliberately does not delete a
    schedule — it stamps ``stopped_at`` on the index entry, which the supervisor
    skips in every path it has (``wakes/supervisor.py``: the due scan, the
    delivery reconciliation and ``_next_wake_ms``), with its own comment saying
    the wakes "stay armed but do not fire until the user reopens it". So a
    conversation in which the user had ever set a reminder could NEVER be deleted
    from the moment they stopped it: `/delete` is refused with a sentence about a
    wake the product has itself put to sleep, and re-stopping or waiting does not
    change the answer. A marker that cannot fire is not pending live work — the
    invariant this guard protects is that nothing which can still happen is
    silently destroyed, and a dormant entry cannot happen.

    REOPENING REVIVES IT, and that is why dropping dormancy here is safe rather
    than merely convenient: the session's next open clears ``stopped_at``
    (``wakes/store.write_entry``'s ``clear`` argument), and a later delete of
    that conversation is refused again while the schedule is armed. Nothing the
    user can still receive is lost without a refusal; what stops being refused is
    the reminder attached to a conversation they have already ended and now ask
    to destroy.

    FAIL-CLOSED ON ANYTHING UNREADABLE, exactly as the existence check was: an
    entry that cannot be parsed may be armed, and the guard's contract is that
    every path which does not positively clear is a refusal. ``store.read_entry``
    is deliberately NOT used — it treats an unreadable file as absent for
    display's sake, which here would turn a corrupt entry into a delete.
    """
    try:
        from local_operator.wakes.store import entry_path

        path = entry_path(config_dir, session)
        if not path.exists():
            return False
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return True  # present but unreadable: assume it fires
        if not isinstance(entry, dict):
            return True
        return not entry.get("stopped_at")
    except Exception:  # noqa: BLE001 — unprovable is "yes"
        return True


def _has_spooled_mail(directory: Path) -> bool:
    try:
        from local_operator.session.runtime.inbox import INBOX_NAME

        spool = directory / INBOX_NAME
        return spool.exists() and spool.stat().st_size > 0
    except Exception:  # noqa: BLE001 — unprovable is "yes"
        return True


def _guard(directory: Path, config_dir: Path, now: float) -> str | None:
    """The hard guards, in one place. Returns the guard's name when the
    directory must be kept, ``None`` when the policy may consider it.

    Closed on every path: each probe already answers "keep" to its own
    failures, and the outer ``except`` makes an exception NONE of them
    anticipated (a bug in ``_process_alive``, a ``MemoryError`` mid-read)
    also "keep", with the probe named so the refusal is diagnosable.
    """
    try:
        if _claimed(directory, now):
            return "claimed by a live process"
        if _lease_runtime_alive(directory):
            return "leased by a live process"
        if _has_armed_wake(config_dir, directory.name):
            return "has an armed wake"
        if _has_spooled_mail(directory):
            return "has unread spooled mail"
    except Exception as exc:  # noqa: BLE001 — a guard that cannot answer keeps
        logger.warning("session cleanup: guard failed for %s (%r); keeping it", directory.name, exc)
        return f"guard failed: {type(exc).__name__}"
    return None


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def _guard_refusal(reason: str, session_id: str) -> str:
    """The refusal sentence for ``reason``, with its remedy RESOLVED.

    The armed-wake door is a file, and a file the user has to go and find is not a
    door (design round 2, D8): the sentence carries ``wakes/<id>.json`` — the id
    resolved, no template — RELATIVE to the config directory, which is the form
    the CLI's own ghost row uses. A machine-absolute path was tried and rejected
    (desktop QA round 4, Q14): this is a sentence in a confirmation dialog for a
    conversation the user is looking at, and the host's layout is not part of it.
    Substituted rather than ``.format`` because the sentences are prose that may
    legitimately contain braces.
    """
    template = _GUARD_REFUSALS.get(reason)
    if template is None:
        return _GUARD_REFUSAL_FALLBACK
    return template.replace("{wake_file}", f"wakes/{session_id}.json")


def _has_transcript(directory: Path) -> bool:
    try:
        return (directory / TRANSCRIPT_FILENAME).stat().st_size > 0
    except OSError:
        return False


def _dir_bytes(directory: Path) -> int:
    total = 0
    try:
        for entry in directory.rglob("*"):
            try:
                if entry.is_file() and entry.name not in _SIDECAR_NAMES:
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


@dataclass
class _Entry:
    path: Path
    activity: float | None
    has_transcript: bool
    size: int


class GuardUnavailable(Exception):
    """A guard could not be evaluated, so the run must not proceed.

    Raised, not swallowed, because the alternative — an empty protected set
    — is a run with the guard silently gone: review round 3 (R3-1) measured
    5 → 12 of 15 removable with no ``skipped`` and nothing logged when the
    picker raised. :func:`run_cleanup` turns this into ``skipped`` + an
    ``errors`` count and removes nothing.
    """


def _picker_rows(config_dir: Path) -> list[str]:
    """Every id ``/resume`` would list, in the picker's order.

        Owned by ``resume.recent_sessions`` — ONE ranking, consumed by the picker
        and by this policy — and imported lazily because ``resume`` is heavier
        than this module wants at import. The first :data:`RECENT_KEEP` are the
        recent guard; the whole list is the unit ``max_sessions`` counts in. A
        picker that cannot be listed is a guard that cannot be evaluated:
        :class:`GuardUnavailable`, never an empty list.

        ``revalidate=True`` IS THE GUARD, NOT AN OPTIMISATION KNOB. The scan
        normally serves an armed fast path that can report a session as hidden for
        up to ``REVALIDATE_EVERY`` polls after its marker was removed — cheap and
        correct for a sidebar that re-polls every 2 seconds and self-heals. But
        the recent-N guard IS this listing, so a session missing from it is not
        merely invisible here: it is UNPROTECTED, and this module deletes. Agent
        review round 1 (R2) reproduced it in production order — the TUI's first
        poll arms the path, the operator deletes ``origin.json`` by hand (the
        supported un-hide gesture), and startup maintenance 0.75 s later chose
        that session for removal while a fresh scan protected it as "one of the
        10 most recent". Deletion is irreversible; a stale display is not.

        The cost is deliberate and measured: this scan is 4,101 syscalls against
        an armed scan's 151 on a 4,000-directory store, paid ONCE per cleanup run
        — startup maintenance and periodic sweeps — rather than every 2 seconds,
        which is the poll this optimisation exists to make cheap. The sidebar's
        own steady-state poll is unchanged at 266 and still flat across a 40x
        store. It also makes the guard independent of whichever poll happened to
        precede it, which is the property that makes this decidable at all, and it
        does not starve the sidebar's repair: a forced revalidation rewrites the
        verdict cache, so a hand-un-hidden session becomes visible at that scan
        rather than later.

        ``include_archived=True`` IS LOAD-BEARING, and it is the one place in the
        codebase that must override the listing's default. An archived session is
        hidden from every list a user browses, but it is still the user's work and
        still something this policy protects: with the default filter it would
    drop out of the recent-N guard the moment it was archived — precisely the
        sessions a user archives are the OLDER ones, so it would drop straight
        into the ranked set every limit draws from — and a routine sweep would
        delete the archive. Archive hides a conversation; it does not declare it
        disposable, and nothing else in this module may read the listing with the
        archive filter on.
    """
    try:
        from local_operator.resume import recent_sessions

        return [
            name
            for name, _stamp in recent_sessions(
                config_dir, limit=None, revalidate=True, include_archived=True
            )
        ]
    except Exception as exc:  # noqa: BLE001 — re-raised as the typed refusal
        raise GuardUnavailable(f"recent-session picker: {type(exc).__name__}: {exc}") from exc


def _origin_label(directory: Path) -> str:
    """``"user"`` or the recorded origin, for a row a human has to judge."""
    try:
        from local_operator.resume import session_origin

        return session_origin(directory) or "user"
    except Exception:  # noqa: BLE001 — a label, never a reason to fail
        return "user"


def _session_title(directory: Path) -> str:
    """The name ``/resume`` would show for this directory, or ``""``.

    Resolved through ``resume.session_name`` so the CLI list and the picker
    agree on what a session is called. Import-lazy: ``resume`` is heavy and
    this runs only for the directories the policy CHOSE, never for the scan.
    """
    try:
        from local_operator.resume import session_name

        return session_name(directory, max_chars=48)
    except Exception:  # noqa: BLE001 — a name is a courtesy, never a reason to fail
        return ""


def run_cleanup(
    config_dir: Path,
    policy: CleanupPolicy,
    *,
    live_dir: Path | None = None,
    now: float | None = None,
    dry_run: bool = False,
    force: bool = False,
    actor: str = "startup",
) -> CleanupResult:
    """Apply ``policy`` to ``config_dir/sessions``.

    THE MASTER SWITCH GOVERNS EVERY CALLER. Round 1 let the CLI run with
    ``enabled: false`` on the theory that typing the command was consent;
    QA (Q1), UX (U2) and review (R1-5) each showed why that is the incident's
    shape: ``/settings`` leaves the limits in the file when the switch is
    turned OFF, so a user who read "off: nothing ever removes a session
    directory" and then ran the command "to see what it would do" lost 16 of
    34 sessions. Now ``enabled: false`` means nothing is removed by anyone
    unless ``force`` is set — and the CLI sets it only for ``--force``, after
    listing and confirming. A dry run with the switch off still LISTS what
    the limits would take (safe and useful) and reports ``skipped`` so the
    caller can say the switch is off.

    ``actor`` is recorded in the cleanup log: ``"startup"`` for the
    maintenance pass, ``"cli"`` for ``lop sessions cleanup``.

    Selection order, deliberately from least to most aggressive, with each
    limit re-checking the guards on its own candidates:

    1. ``remove_empty`` — directories with no non-empty transcript.
    2. ``max_inactive_days`` — last activity (:func:`session_activity`: the
       transcript's or the mail spool's mtime, never a sidecar's, never the
       directory's) older than the limit.
    3. ``max_sessions`` — beyond the N most recently active CONVERSATIONS
       SHOWN BY ``/resume``, oldest first.
    4. ``max_total_bytes`` — least recently active first until under budget.

    Only directories WITH activity are ranked; a never-active directory is
    outside the ranked set, never counts toward ``max_sessions`` or
    :data:`RECENT_KEEP`, and is only a ``remove_empty`` candidate (U11). The
    first :data:`RECENT_KEEP` rows of ``/resume`` — exactly what
    ``resume.recent_sessions`` lists — are excluded from every limit (Q8).

    ``max_sessions`` COUNTS IN THE PICKER'S UNIT. "Keep 50 sessions" to a
    user means the 50 conversations ``/resume`` shows; on a store that is
    85% subagent-origin, counting every transcript made ``max_sessions: 50``
    remove 21 of the user's 31 conversations while the picker read
    "31 sessions" (UX round 3, U15). So the cap ranks ONLY the rows
    ``recent_sessions`` lists (user-origin, with activity); a subagent-origin
    transcript is never a ``max_sessions`` candidate — a cap on those, if
    ever wanted, is a separately named setting. The age and byte limits are
    about staleness and disk, not about "how many conversations", and still
    consider every ranked directory.
    Ties on the activity clock break on the directory NAME so a dry run and
    the real run pick the same directories on any filesystem (R1-11).

    A real run is a plan followed by :func:`apply_cleanup` on that exact
    plan; a caller that previewed first should pass its preview to
    :func:`apply_cleanup` rather than calling this again (R2-2).
    """
    result = CleanupResult(dry_run=dry_run)
    if not policy.enabled and not force and not dry_run:
        result.skipped = "disabled"
        return result
    if not policy.has_any_limit:
        result.skipped = "no limits configured"
        return result
    sessions_dir = config_dir / SESSIONS_DIRNAME
    if not sessions_dir.is_dir():
        result.skipped = "no store"
        return result

    moment = now if now is not None else time.time()
    live_resolved: Path | None = None
    if live_dir is not None:
        try:
            live_resolved = live_dir.resolve()
        except OSError:
            live_resolved = None

    entries: list[_Entry] = []
    try:
        children = [child for child in sessions_dir.iterdir() if child.is_dir()]
    except OSError as exc:
        logger.warning("session cleanup: cannot scan %s: %s", sessions_dir, exc)
        result.errors += 1
        return result
    for child in children:
        result.scanned += 1
        try:
            if live_resolved is not None and child.resolve() == live_resolved:
                result.protected.append((child.name, "the current session"))
                continue
            entries.append(
                _Entry(child, session_activity(child), _has_transcript(child), _dir_bytes(child))
            )
        except OSError:
            continue

    # THE RANKED SET IS ONLY DIRECTORIES WITH ACTIVITY. A directory that has
    # never been worked in (no transcript, no spool — every idle open-and-quit
    # launch) has no place in "the N most recently active": it is not a
    # conversation, and ranking it by any fallback made each such launch
    # displace a real one (UX round 2, U11: eleven launches emptied an
    # 8-session store under `max_sessions: 5`). It is only ever a
    # `remove_empty` candidate. Newest first; equal stamps fall back to the
    # name so the order is a property of the store, not of `iterdir`.
    ranked = sorted(
        (entry for entry in entries if entry.activity is not None),
        key=lambda entry: (-(entry.activity or 0.0), entry.path.name),
    )
    never_active = [entry for entry in entries if entry.activity is None]
    # THE RECENT-N GUARD IS THE PICKER'S FIRST PAGE, LITERALLY. ``recent``
    # is what ``resume.recent_sessions(limit=RECENT_KEEP)`` returns — the
    # same function, the same rows, the same order the user sees on
    # ``/resume``. The policy used to rank every directory for this guard
    # while the picker lists only transcripted, user-origin sessions; on a
    # real store that is 179/210 subagent-origin the two sets did not
    # intersect at all, and ``max_sessions: 12`` would have removed 9 of the
    # 10 rows the user could see (QA round 2, Q8). Subagent-origin sessions
    # are real transcripts and rank under ``max_sessions`` by activity like
    # any other — they are never "empty" — but they are not on the picker,
    # so they are not what this guard protects.
    try:
        picker = _picker_rows(config_dir)
    except GuardUnavailable as exc:
        logger.warning("session cleanup: REFUSING to run, %s", exc)
        result.skipped = f"guard unavailable: {exc}"
        result.errors += 1
        return result
    recent = set(picker[:RECENT_KEEP])
    visible = set(picker)
    chosen: dict[str, Candidate] = {}

    # A directory several limits would take is reported once, under the first
    # guard that saved it; without this a dry run listed the same session
    # per limit.
    protected_seen: set[str] = set()

    def protect(name: str, guard: str) -> None:
        if name not in protected_seen:
            protected_seen.add(name)
            result.protected.append((name, guard))

    def consider(entry: _Entry, policy_name: str, reason: str) -> bool:
        name = entry.path.name
        if name in chosen:
            return True
        if name in recent:
            protect(name, f"one of the {RECENT_KEEP} most recent")
            return False
        guard = _guard(entry.path, config_dir, moment)
        if guard is not None:
            protect(name, guard)
            return False
        idle = 0.0 if entry.activity is None else max(0.0, (moment - entry.activity) / 86400.0)
        chosen[name] = Candidate(
            name,
            policy_name,
            reason,
            title=_session_title(entry.path) if entry.has_transcript else "",
            idle_days=idle,
            size_bytes=entry.size,
            origin=_origin_label(entry.path),
            active=entry.activity is not None,
        )
        return True

    if policy.remove_empty:
        for entry in [*never_active, *ranked]:
            if not entry.has_transcript:
                consider(entry, "remove_empty", "no transcript")

    if policy.max_inactive_days:
        cutoff = moment - policy.max_inactive_days * 86400.0
        for entry in ranked:
            activity = entry.activity or 0.0
            if activity < cutoff:
                # The row already carries the exact idle age; repeating it
                # rounded here read as `idle 10d > 10d` on a 10.0001-day
                # session. The reason states the LIMIT, the column the fact.
                consider(
                    entry,
                    "max_inactive_days",
                    f"idle over {policy.max_inactive_days}d",
                )

    if policy.max_sessions:
        # Walk oldest-first and stop once the KEPT count fits the cap. Not a
        # positional slice of the ranked list: a guarded row (a picker row,
        # a live claim) that sits inside the top-N slice would otherwise
        # shield everything newer than it from ever being considered, and
        # the cap would silently not apply (Q8's scenario: 10 older picker
        # rows guarded, 7 newer subagent runs, cap 12 -> the 5 oldest
        # subagent runs must go). Ranked over the PICKER'S rows only (U15):
        # ``kept`` is the number the user would read off ``/resume``.
        survivors = [
            entry
            for entry in ranked
            if entry.path.name in visible and entry.path.name not in chosen
        ]
        kept = len(survivors)
        for entry in reversed(survivors):  # oldest first
            if kept <= policy.max_sessions:
                break
            if consider(
                entry,
                "max_sessions",
                f"beyond newest {policy.max_sessions}",
            ):
                kept -= 1

    if policy.max_total_bytes:
        survivors = [entry for entry in ranked if entry.path.name not in chosen]
        total = sum(entry.size for entry in survivors)
        for entry in reversed(survivors):  # least recently active first
            if total <= policy.max_total_bytes:
                break
            if consider(
                entry,
                "max_total_bytes",
                f"store over {policy.max_total_bytes // 1024} kB",
            ):
                total -= entry.size

    result.chosen = list(chosen.values())
    if dry_run and not policy.enabled and not force:
        # Listed, not removed, and the caller is told why: the switch is off.
        result.skipped = "disabled"
    if dry_run:
        result.removed = list(result.chosen)
        return result
    return apply_cleanup(config_dir, result, actor=actor, now=moment)


def apply_cleanup(
    config_dir: Path,
    plan: CleanupResult,
    *,
    actor: str,
    now: float | None = None,
) -> CleanupResult:
    """Remove EXACTLY the set ``plan`` chose, re-checking only the hard guards.

    The CLI shows the user a plan (a dry run), asks, then removes. If the
    removal were a second scan, a session created between the prompt and the
    ``yes`` would shift the recent-N window and a row shown as KEPT could be
    removed (review round 2, R2-2 — reproduced). So the set is fixed here:
    nothing is re-ranked, nothing is added. What IS re-checked is whether a
    chosen directory has since acquired a live claim, an armed wake or
    unread mail, because those are facts about NOW and refusing is always the
    safe side. A directory that is no longer there is not an error.
    """
    moment = now if now is not None else time.time()
    sessions_dir = config_dir / SESSIONS_DIRNAME
    result = CleanupResult(scanned=plan.scanned, protected=list(plan.protected))
    result.chosen = list(plan.chosen)
    result.skipped = plan.skipped
    for candidate in plan.chosen:
        target = sessions_dir / candidate.session
        if not target.is_dir():
            continue
        guard = _guard(target, config_dir, moment)
        if guard is not None:
            result.protected.append((candidate.session, f"{guard} (since the preview)"))
            continue
        try:
            done = remove_session_dir(
                target,
                config_dir=config_dir,
                policy=candidate.policy,
                reason=candidate.reason,
                actor=actor,
                title=candidate.title,
            )
        except OSError as exc:
            logger.warning("session cleanup: cannot remove %s: %s", candidate.session, exc)
            result.errors += 1
            continue
        if done:
            result.removed.append(candidate)
            _forget_wake_entry(config_dir, candidate.session)
    return result


def _forget_wake_entry(config_dir: Path, session: str) -> None:
    """A removed session cannot have a wake fire for it; drop its index entry.

    REACHABLE since U2: a DORMANT entry no longer refuses the delete, so a
    stopped conversation whose reminder was put to sleep is removed together with
    that entry. An ARMED entry still refuses above, which is why this stays
    best-effort — it is cleanup for the case the guard now lets through, not a
    step the delete depends on."""
    try:
        from local_operator.wakes.store import remove_entry

        remove_entry(config_dir, session)
    except Exception:  # noqa: BLE001
        pass


def cleanup_from_config(
    config_manager: Any, config_dir: Path, *, live_dir: Path | None = None
) -> CleanupResult:
    """The startup entry point. Reads the policy and runs it — which, with the
    default config, means reading five keys and returning."""
    policy = policy_from_config(config_manager)
    if not policy.enabled:
        return CleanupResult(skipped="disabled")
    result = run_cleanup(config_dir, policy, live_dir=live_dir, actor="startup")
    if result.removed:
        logger.warning(
            "session cleanup: removed %d of %d sessions (see %s)",
            len(result.removed),
            result.scanned,
            config_dir / SESSIONS_DIRNAME / CLEANUP_LOG_NAME,
        )
        write_last_cleanup(config_dir / SESSIONS_DIRNAME, result, actor="startup")
    return result


def write_last_cleanup(sessions_dir: Path, result: CleanupResult, *, actor: str) -> None:
    """Record a removing pass in :data:`LAST_CLEANUP_NAME` for the TUI to announce."""
    policies: dict[str, int] = {}
    origins: dict[str, int] = {}
    for candidate in result.removed:
        policies[candidate.policy] = policies.get(candidate.policy, 0) + 1
        origins[candidate.origin] = origins.get(candidate.origin, 0) + 1
    # A pass that only swept never-active directories is recorded but
    # pre-acknowledged: nothing a user worked in went, so there is nothing
    # to announce (N6). ``quiet`` says why the flag is already set.
    quiet = bool(result.removed) and all(not candidate.active for candidate in result.removed)
    payload = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "actor": actor,
        "pid": os.getpid(),
        "removed": len(result.removed),
        "scanned": result.scanned,
        "policies": policies,
        "origins": origins,
        "record": str(sessions_dir / CLEANUP_LOG_NAME),
        "acknowledged": quiet,
        "quiet": "only never-active directories were removed" if quiet else None,
    }
    _write_record(sessions_dir / LAST_CLEANUP_NAME, payload)


def _write_record(path: Path, payload: dict[str, Any]) -> bool:
    """tmp + ``os.replace``: a viewer never reads a half-written record.

    Same shape as ``resume._save_origin_cache``. The ``os.replace`` here is
    on a FILE beside the session directories, never on one of them; the
    deletion guard allow-lists this call by name for that reason (R3-4).
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("session cleanup: cannot write %s: %s", path.name, exc)
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
    return True


def take_unannounced_cleanup(
    sessions_dir: Path, *, runtime_pid: int | None = None
) -> dict[str, Any] | None:
    """The last removing pass if no viewer has announced it yet, marking it
    announced; ``None`` otherwise.

    WHO ANNOUNCES: the viewer attached to the runtime that did the removing,
    when there is one. The record names the removing ``pid``; a caller whose
    ``runtime_pid`` is that pid takes the record outright. Any OTHER viewer
    defers while that pid is still alive — its own viewer is about to look
    — and takes it only once the writer is gone (a headless run, a runtime
    that exited before its viewer attached). Without this rule the notice
    went to whichever terminal read the file first: term2 announced what
    term1's runtime had removed while term1 stayed blank (UX round 3, U14).
    ``runtime_pid=None`` (a cold viewer, no runtime yet) defers the same way.

    Read-then-rewrite without a lock: two viewers of the SAME runtime
    adopting in the same instant could both announce, which is the harmless
    direction. A malformed or unreadable file is treated as "nothing to
    announce" — the jsonl and the WARNING still hold the facts.
    """
    path = sessions_dir / LAST_CLEANUP_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("acknowledged") or not payload.get("removed"):
        return None
    writer = payload.get("pid")
    if isinstance(writer, int) and not isinstance(writer, bool) and writer != runtime_pid:
        if _process_alive(writer) and writer != os.getpid():
            # The removing runtime is still up: its own viewer announces.
            return None
    payload["acknowledged"] = True
    if not _write_record(path, payload):
        # Announced anyway — the facts are true — but a record that cannot be
        # flipped announces on every boot, which is worth one diagnosable line.
        logger.debug("session cleanup: %s is not writable; the notice will repeat", path)
    return payload


def _counts(value: Any, joiner: str) -> str:
    """``"3 by remove_empty, 2 by max_sessions"`` (``joiner=" by "``) or
    ``"2 user, 3 subagent"`` (``joiner=" "``) from a ``{name: count}``
    mapping, tolerating any shape: a hand-edited or newer-schema record must
    still format (R3-2)."""
    if not isinstance(value, dict):
        return ""
    parts: list[str] = []
    for name, count in sorted(value.items(), key=lambda item: str(item[0])):
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            continue
        parts.append(f"{int(count)}{joiner}{name}")
    return ", ".join(parts)


def format_cleanup_notice(payload: Any) -> str:
    """The one-line transcript notice for a removing pass.

    Total, not partial: every field is read defensively because this runs at
    boot and a malformed record must never take the app down (review round
    3, R3-2: ``"removed": "many"`` crashed the first frame). ``payload`` is
    typed ``Any`` for the same reason — the caller has already checked it is
    a dict, but this function must not depend on that.
    """
    if not isinstance(payload, dict):
        payload = {}
    try:
        removed = int(payload.get("removed") or 0)
    except (TypeError, ValueError):
        removed = 0
    by = _counts(payload.get("policies"), " by ")
    origins = _counts(payload.get("origins"), " ")
    noun = "session" if removed == 1 else "sessions"
    record = payload.get("record")
    record = str(record) if isinstance(record, str) and record else CLEANUP_LOG_NAME
    home = os.path.expanduser("~")
    if record.startswith(home + os.sep):
        record = "~" + record[len(home) :]
    detail = "; ".join(part for part in (by, origins) if part)
    # Authored rows, one per clause: a NoticeBlock treats a newline as a row
    # boundary and wraps each row on its own, so the count, the record, the
    # preview command and the off-switch each keep their words together.
    # Left to the fold, 100 columns broke `lop / sessions cleanup` across
    # rows and stranded `/settings › Session cleanup` alone on the last one
    # (design round 3, N7). A clause that is still too wide for the column
    # wraps inside itself, which is the least bad break.
    return "\n".join(
        (
            f"session cleanup removed {removed} {noun} at launch"
            + (f" ({detail})" if detail else ""),
            f"record: {record}",
            "preview next time: lop sessions cleanup --dry-run",
            "turn off: /settings › Session cleanup",
        )
    )
