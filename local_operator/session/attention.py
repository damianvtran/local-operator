"""Durable, revision-bound completion receipts shared by local frontends.

A connection is not a read. Only an explicit acknowledgement of a published
completion token advances the watermark. Tokens are conversation-bound and do
not depend on a process epoch, a wall clock, or transcript modification time.
SQLite serializes the short transactions across TUI, relay and server processes;
callers on an event loop run these operations in a worker thread.

TWO WATERMARKS, DELIBERATELY SEPARATE. ``receipts.acknowledged`` says a human
demonstrably READ a result; ``deliveries.delivered`` says somebody already
NOTIFIED them about it. They ride the same monotonic ``completions.sequence``
but must never be conflated: notifying is cheap and reversible, marking-read is
destructive, and ``docs/SESSION_SIDEBAR.md`` pins the rule that routing a
notification never marks anything read. A session can be delivered-and-unread
forever, which is correct — the sidebar's checkmark stays until it is opened.

BULK ACKNOWLEDGEMENT IS TOKEN-BOUND, NEVER A SWEEP (:meth:`AttentionStore.
acknowledge_many`). A surface may acknowledge the completions it can ENUMERATE,
each by the token it actually rendered; a completion published after that render
is not in the batch and stays unread. The tempting shortcut — advance every
conversation's watermark to its own ``MAX(sequence)`` — clears a completion the
operator never saw, the exact hazard this mechanism exists to prevent, and it is
deliberately not offered (``docs/ATTENTION.md``, rule R10). No automatic path may
acknowledge anything either: a read receipt is a user gesture on a rendered
result, never a timer, a poll or a focus change.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any, TypeGuard, TypeVar

from local_operator.paths import config_dir

logger = logging.getLogger(__name__)

ATTENTION_CAPABILITY = "completion-ack-v1"
ATTENTION_CUSTOM_TYPE = "completion_attention"

#: How long SQLite itself waits for a competing writer before answering
#: ``SQLITE_BUSY``, in milliseconds, and the driver's matching window.
#:
#: THE HOUSE RANGE, AND THIS STORE WAS THE OUTLIER. Every sibling store sets a
#: busy timeout with a written rationale -- ``secrets/store.py`` 10000 ("covers
#: the brief writer-lock contention concurrent sessions do still produce"),
#: ``providers/usage_cache.py`` 5000, ``providers/auth_store.py`` 5000 -- and
#: ``attention.db`` shipped a bare ``connect(timeout=2.0)`` with no PRAGMA at
#: all, from when one process at a time wrote it. It is now the MOST contended
#: store on the machine: ~25 concurrent ``lop`` sessions plus the mobile daemon,
#: the tunnel connector and the browser bridge all publish completions into the
#: one file, and every publish is a ``BEGIN IMMEDIATE`` that also runs the
#: additive schema check. Two seconds expires first, the completion is lost, and
#: the caller is handed ``OperationalError`` -- which for a daemon request
#: handler is an outage rather than a late notification (2026-09-20,
#: ``~/Library/Logs/local-operator/mobile.log``).
_BUSY_TIMEOUT_MS = 5000

#: The driver's own busy handler, set alongside the PRAGMA for the reason
#: ``auth_store`` sets both: ``timeout`` covers the connection before the PRAGMA
#: has run, and the PRAGMA is what a reader of this file looks for.
_CONNECT_TIMEOUT_S = 5.0

#: How many times a contended write is re-attempted, and the wait before the
#: retry (seconds). The store owns the connection and the transaction, so it is
#: where riding out a lock belongs -- see :meth:`AttentionStore._retry_write`.
#:
#: ONE retry, and the bound is MEASURED rather than derived. SQLite's default
#: busy handler sleeps in increments and re-checks the elapsed time after each
#: one, so a configured window costs about 1.8x it in practice here (probed on
#: this host, sqlite 3.53.4: 50 ms -> 0.27 s, 2 s -> 3.8 s, 5 s -> 9.3 s). Two
#: attempts are therefore up to ~19 s measured (three attempts measured 29.75 s
#: in the PR's repro before the budget was tightened to two) -- and 19 s is
#: already the most a turn's ``finally`` should spend before answering, which is
#: what bounds this at two rather than three.
#: A second window absorbs a burst that outlasts the first; a lock that outlasts
#: both is what the journal beside the store is for (the outcome is durable in
#: the transcript before it is published, so the next boot re-imports it).
_WRITE_ATTEMPTS = 2
_WRITE_BACKOFF_S = (0.2,)

#: SQLite's two contended-verdict codes. ``SQLITE_BUSY`` is the busy-timeout
#: expiry ("database is locked"); ``SQLITE_LOCKED`` is the same verdict from the
#: other lock family ("database table is locked").
_CONTENTION_ERRONAMES = frozenset({"SQLITE_BUSY", "SQLITE_LOCKED"})

_T = TypeVar("_T")

#: The machine token a surface reads to tell "your token is stale, re-arm from
#: your own state" apart from a failure worth backing off on. Part of the wire
#: contract because the clients must act DIFFERENTLY on the two, and message
#: text is not a contract (the desktop error object carries it as `code`, the
#: mobile body as `code`, and `docs/DESKTOP_API.md` documents the row).
#:
#: THIS constant is the string's source of truth. A renderer cannot import
#: Python, so `SUPERSEDED_COMPLETION_TOKEN_CODE` in local-operator-ui's
#: `src/shared/desktop-session-contract.ts` is a copy of it, and neither repo's
#: tests can see the other's: each side therefore pins the literal it ships
#: (here `tests/unit/server/test_desktop_attention.py`; there
#: `scripts/completion-view-ack.test.mjs`). Changing the string is a cross-repo
#: change, not a local one.
SUPERSEDED_TOKEN_CODE = "superseded_completion_token"


class SupersededCompletionToken(ValueError):
    """A real token for this conversation that is no longer the current one.

    SUBCLASSES ``ValueError`` so every surface that already maps "unknown
    completion token" to 409 -- the desktop route's error ladder, the mobile
    `/seen` route, the runtime op -- keeps that mapping without a per-surface
    change, and a caller that catches ``ValueError`` cannot miss this one. The
    separate TYPE is what lets a surface that cares name the remedy instead of
    the generic sentence.

    RAISED INSTEAD OF ANSWERED, and that is the whole point (see
    :meth:`AttentionStore.acknowledge`). A superseded acknowledgement used to
    return a 200 whose body said `unseen: true` and whose receipt had not moved:
    a client that treats any resolved call as "read" (both shipped clients did)
    latched forever, and the operator's completion checkmark never cleared.
    """

    code = SUPERSEDED_TOKEN_CODE

    def __init__(self) -> None:
        super().__init__(
            "completion token superseded by a newer completion; "
            "acknowledge the conversation's current token"
        )


class AttentionWriteDeferred(sqlite3.OperationalError):
    """A write gave up: SQLite called the store busy through every attempt.

    SUBCLASSES ``sqlite3.OperationalError``, AND CARRIES ITS ``sqlite_errorname``,
    for the reason :class:`SupersededCompletionToken` subclasses ``ValueError``:
    every surface that already classifies a store failure keeps its mapping with
    no per-surface change. That is load-bearing here rather than tidy --
    ``session/store_failures.py`` chooses between 503-busy, 507-out-of-space and
    500-unavailable from the certified ``sqlite_errorname``, so a NEW type that
    dropped the code would downgrade a correctly-classified contention to
    "the store is broken" and tell the operator to check the machine.

    What the separate TYPE buys is the distinction the caller could not make
    before: *contended, come back later* versus *broken, stop retrying*. A
    caller that treats a store failure as fatal can now defer instead of
    failing -- see ``Session._publish_attention_outcome``, which is the caller
    whose raise took the mobile daemon's request handler down with it.

    RAISED ONLY AFTER THE BOUNDED RETRY in :meth:`AttentionStore._retry_write`
    is exhausted, so reaching it means contention outlasted ``_BUSY_TIMEOUT_MS``
    times ``_WRITE_ATTEMPTS`` -- never SQLite's first refusal.

    The name says DEFERRED rather than LOST because for the publish path it is:
    the outcome is journalled to the transcript *before* it is published, so the
    next boot's ``bootstrap_transcript`` re-imports it (see
    ``Session._publish_attention_outcome``). Callers that swallow this must say
    so out loud, because a completion missing from the store until the next boot
    is a real, user-visible delay.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _is_contention(error: sqlite3.OperationalError) -> bool:
    """Whether SQLite refused a statement for CONTENTION rather than for a cause.

    The discriminator is the certified code, exactly as ``store_failures``
    reasons: the message is localized prose that changes between releases. The
    text is consulted only when there is no code at all -- an error re-wrapped by
    an intermediate layer -- and even then only for SQLite's two lock verdicts,
    so a full disk or an unopenable store is never retried.
    """
    errorname = str(getattr(error, "sqlite_errorname", "") or "")
    if errorname:
        return errorname in _CONTENTION_ERRONAMES
    message = str(error).lower()
    return "locked" in message or "busy" in message


def _deferred(error: sqlite3.OperationalError) -> AttentionWriteDeferred:
    """Name a contended write's final refusal, keeping SQLite's own verdict.

    The code is copied rather than re-derived so the surface ladders keep
    classifying this as contention; when the original had none (the re-wrapped
    case ``_is_contention`` accepts on its text), the busy code is asserted
    explicitly, because this function is only ever called for a lock.
    """
    deferred = AttentionWriteDeferred(
        f"attention store stayed busy through {_WRITE_ATTEMPTS} attempts: {error}"
    )
    errorcode = getattr(error, "sqlite_errorcode", None)
    deferred.sqlite_errorcode = errorcode if errorcode is not None else sqlite3.SQLITE_BUSY
    errorname = str(getattr(error, "sqlite_errorname", "") or "")
    deferred.sqlite_errorname = errorname or "SQLITE_BUSY"
    return deferred


#: Named once because BOTH the minting side (`provisional_anchor`) and the
#: recognising side (`_supersedes_provisional`, on the stored AND the incoming
#: anchor) key on this exact shape; a literal in one of them drifting from the
#: other silently turns supersession off.
_PROVISIONAL_PREFIX = "completion-"

#: The delivery watermark. `delivered` is a highwater mark on the SAME
#: `completions.sequence` that `receipts.acknowledged` uses, which is what makes
#: the two comparable and the arbitration clock-free: sequence is assigned by
#: SQLite AUTOINCREMENT under BEGIN IMMEDIATE, so it is a total order across
#: every process on this machine. `delivered_at` and `backend` are DIAGNOSTICS
#: ONLY and no decision may read them \u2014 in particular, a wall clock must never
#: enter the claim, or two observers whose clocks disagree would both deliver.
_CREATE_DELIVERIES = (
    "CREATE TABLE deliveries ("
    "conversation TEXT PRIMARY KEY, delivered INTEGER NOT NULL, "
    "delivered_at REAL NOT NULL, backend TEXT NOT NULL)"
)

#: The no-flood rule, as one statement: everything already published counts as
#: already delivered. Runs once, inside the transaction that creates the table
#: (see `_connect`), so a machine upgrading into background-completion
#: notifications starts from "nothing outstanding" rather than announcing its
#: entire history.
_BASELINE_DELIVERIES = (
    "INSERT INTO deliveries(conversation,delivered,delivered_at,backend) "
    "SELECT conversation, MAX(sequence), ?, 'baseline' FROM completions "
    "GROUP BY conversation"
)

#: THE THIRD TERM OF `revision()`, and the reason it exists: a supersede is the
#: one durable change that moves NEITHER `completions.sequence` NOR
#: `receipts.acknowledged`, so without a counter it is invisible to every
#: consumer that gates on `revision()` (review round 1, major-1).
#:
#: The obvious alternative -- mint a new completions row so `sequence` moves --
#: is ruled out by the no-flood rule: `sequence` IS the receipt watermark, so a
#: new row resurrects an already-acknowledged turn as unread and re-fires a
#: notification for a result the human has read. Correcting a record is not a
#: new completion. The counter is therefore how a heal stays sequence-preserving
#: and still DETECTABLE.
#:
#: A single row by construction (`CHECK(id=1)`): this counts machine-global
#: mutations, exactly like the rest of `revision()`, not per-conversation ones.
_CREATE_MUTATIONS = (
    "CREATE TABLE mutations (id INTEGER PRIMARY KEY CHECK(id=1), supersedes INTEGER NOT NULL)"
)

#: Upsert rather than UPDATE because the table is created with NO row: both
#: `_CREATE_MUTATIONS` sites above create it empty and nothing seeds it, so the
#: first bump has nothing to update. A plain UPDATE would match zero rows and
#: silently succeed with rowcount 0, leaving every heal undetectable again --
#: the exact defect this table exists to fix. The `INSERT ... ON CONFLICT`
#: writes the seed and the increment in one statement, so the first caller and
#: every later one take the same path.
_BUMP_SUPERSEDES = (
    "INSERT INTO mutations(id,supersedes) VALUES(1,1) "
    "ON CONFLICT(id) DO UPDATE SET supersedes=mutations.supersedes+1"
)

#: WHICH conversation a heal touched — the missing half of the counter above
#: (review round 1, R4).
#:
#: `mutations.supersedes` is all `revision()` needs to be a correct change
#: detector: it proves a heal HAPPENED. It cannot say WHICH record moved, and the
#: machine-wide feed has to publish a corrected state for exactly that record. A
#: heal UPDATEs the row in place, so the healed conversation appears in neither
#: the feed's new-sequence delta nor its changed-acknowledgement delta — the feed
#: correctly accepted the new revision and then emitted nothing at all, leaving
#: every subscriber holding the provisional "Interrupted" outcome for a turn that
#: had actually completed.
#:
#: An append-only log rather than a column on `mutations`: two heals landing
#: between two ticks must BOTH be reported, and a single-row table silently merges
#: them into whichever was last. Pruned to the newest
#: `_SUPERSEDE_LOG_RETENTION` rows in the SAME transaction as the write, so it
#: stays bounded on a store that runs for years. The retention is orders of
#: magnitude deeper than any reader can fall — a consumer's cursor is never more
#: than one poll interval behind the write — and the consequence of over-running
#: it is a missed in-place correction rather than a missed completion, which is
#: why the bound is safe to hold this loosely.
_CREATE_SUPERSEDE_LOG = (
    "CREATE TABLE supersede_log ("
    "seq INTEGER PRIMARY KEY AUTOINCREMENT, conversation TEXT NOT NULL)"
)
_SUPERSEDE_LOG_RETENTION = 256
_APPEND_SUPERSEDE = "INSERT INTO supersede_log(conversation) VALUES(?)"
_PRUNE_SUPERSEDE_LOG = (
    "DELETE FROM supersede_log WHERE seq <= " "(SELECT COALESCE(MAX(seq),0) FROM supersede_log) - ?"
)


def conversation_identity(directory: Path) -> str:
    """Use the durable namespace, never the currently selected agent profile."""
    namespace = "agent" if directory.parent.name == "agents" else "session"
    return f"{namespace}/{directory.name}"


def provisional_anchor(token: str) -> str:
    """The anchor a run wears before it has a viewable result entry.

    A failed or interrupted run may have no assistant message to point at, so
    its outcome marker anchors to the run itself. The synthetic shape is what
    lets :meth:`AttentionStore.publish` recognise a record as provisional and
    therefore safe to replace once the real outcome lands.
    """
    return f"{_PROVISIONAL_PREFIX}{token}"


def _optional_column(row: sqlite3.Row, name: str) -> str:
    """One ADDITIVE column's value, ``""`` when a pre-taxonomy row lacks it.

    The read-only readers must not migrate: ``state_many`` opens the database
    ``mode=ro`` precisely so a frontend list can never write, while the schema
    migration lives on the WRITE path (``_connect``). An established database
    that no build of this version has written yet therefore genuinely has no
    ``reason``/``cause`` column, and a bare ``row[name]`` raises ``IndexError``
    on it — a sidebar crash on exactly the machine that has not restarted yet.
    Naming the absence here keeps the two readers in step without teaching
    either one to alter a schema it is only reading.
    """
    try:
        value = row[name]
    except IndexError:
        return ""
    return str(value or "")


#: How much of a reason the STORE keeps. This string rides every attach frame
#: (once per conversation in the canonical `attention` state, and once per child
#: for a job row), so an unbounded provider message would spend the frame ceiling
#: that `tests/unit/session/test_attach_frame_size.py` guards. The full text is
#: not lost: the live transcript notice and the journal row
#: (`completion_attention`, replayed by `_default_convert_to_llm`) both carry it
#: verbatim. 500 characters is past any harness-authored sentence and past the
#: opening of a provider error, which is all a one-line tooltip or a phone
#: banner can use.
REASON_WIRE_CHARS = 500


def _supersedes_provisional(
    existing: Any,
    conversation: str,
    token: str,
    anchor: str,
) -> bool:
    """May the incoming outcome overwrite the stored one for this token?

    Only for the SAME conversation, and only over a record still wearing the
    provisional shape. Cross-conversation reuse and overwrites of an already
    authoritative record both stay refusals — see :meth:`AttentionStore.publish`
    for why each half of that is load-bearing.

    THE INCOMING ANCHOR IS CHECKED TOO, not just the stored one (review round 1,
    minor-2). A publish for token T carrying `provisional_anchor(U)` — some
    OTHER token's synthetic anchor — otherwise superseded happily, and the row
    it left behind wore an anchor matching no token: unviewable, and permanently
    unhealable, because no later real outcome could ever supersede a record
    whose stored anchor is not `provisional_anchor(T)`. A one-way trip into a
    dead state. Reachability is low (`_publish_attention_outcome` always mints
    the anchor for the same token, and the journal path is gated on conversation
    identity), which is why the guard is cheap rather than elaborate: an
    incoming provisional anchor is legitimate only when it belongs to the token
    being published.

    THE STORED KIND IS NO LONGER PART OF THE SHAPE. It used to have to read
    `interrupted`, which was the only kind a provisional record could carry when
    "no outcome" was published as an interruption. The taxonomy change makes
    "no outcome" an `error`, and a provisional record published as `error` still
    has to be superseded by the same token's real `complete` — otherwise the
    change bricks exactly the session the docstring at :meth:`publish`
    describes. The ANCHOR, not the kind, is the "replaceable" signal: a record
    still wearing `completion-<token>` has no viewable result behind it and its
    only legitimate successor is the same token's real outcome.
    """
    stored_conversation, stored_anchor = tuple(existing)[:2]
    if anchor.startswith(_PROVISIONAL_PREFIX) and anchor != provisional_anchor(token):
        return False
    return stored_conversation == conversation and stored_anchor == provisional_anchor(token)


def bootstrap_transcript(
    transcript: Any,
    store: AttentionStore | None = None,
    *,
    witnessed_cut_off: tuple[str, str] | None = None,
    reaped_owner: Any | None = None,
) -> tuple[str, str, str, str] | None:
    """Import a conversation's durable outcome; NEVER fatal to the caller.

    Both call sites are boot paths that must survive a bad conversation:
    ``Session.__init__`` constructs the runtime process, and the mobile daemon
    sweeps up to 100 session directories in one loop. A raise here used to be
    unrecoverable rather than transient — the runtime failed to spawn on EVERY
    attach, so the session could never be opened again and the TUI showed only
    "Saved excerpt · Connection unavailable". An attention record is an
    observability nicety (who has an unread result); it may never outrank the
    ability to READ the conversation, and one poisoned session may never take
    the daemon's remaining 99 down with it.

    The failure is logged with the identity and the exception so a swallowed
    problem is still diagnosable from the log rather than silently invisible.

    Returns whatever :func:`_import_transcript_outcome` published for an
    orphaned run (or ``None``), so the caller that owns a ``Session`` can
    journal the cut-off once. A failure also returns ``None``: a swallowed
    problem journals nothing rather than half-narrating one.

    ``witnessed_cut_off`` is passed straight through; see
    :func:`_import_transcript_outcome` for the one caller that uses it and why
    the record it writes is provisional.

    ``reaped_owner`` is the record a caller's OWN reap has already deleted from
    the run directory; see :func:`_classify_orphaned_run` for why the evidence
    has to be handed in rather than re-read.
    """
    try:
        return _import_transcript_outcome(
            transcript,
            store,
            witnessed_cut_off=witnessed_cut_off,
            reaped_owner=reaped_owner,
        )
    except Exception as exc:  # noqa: BLE001 — attention must never block a boot
        # `getattr` so the handler cannot itself raise on a transcript that
        # never grew a `.directory` (a stub, a partially constructed instance)
        # and turn a swallowed failure back into a fatal one. It survives a
        # MISSING attribute only: `getattr` suppresses `AttributeError` and
        # nothing else, so a `.directory` raising `OSError` would still escape.
        # That case is unreachable — `Transcript.directory` is a plain instance
        # attribute, `transcript.py` defines no `@property`, `Path.name` does
        # not raise — so the read stays as-is rather than growing a third guard.
        # `session.py` mirrors this same defensive read.
        directory = getattr(transcript, "directory", None)
        logger.warning(
            "attention: skipping outcome import for %s (%r)",
            getattr(directory, "name", directory),
            exc,
            # An arbitrary unknown exception is being swallowed here, and `%r`
            # names its type but not the frame that raised it. The destination
            # is a real file (a spawned runtime's `log_dir()/runtime.log`), so
            # the traceback is what makes the next novel failure recoverable
            # from disk instead of only reproducible.
            exc_info=True,
        )


def _config_root_for(directory: Path) -> Path:
    """The config root that owns ``directory``.

    Derived from the transcript path rather than read from the ambient
    environment, because this classification must agree with the STORE it
    writes into: an isolated run redirects ``HOME`` (or
    ``LOCAL_OPERATOR_CONFIG_DIR``) and a store under one root must not be
    classified against another root's run records. ``sessions/<id>`` and
    ``agents/<id>`` are the only two layouts (``conversation_identity`` knows
    the same two), so anything else falls back to the ambient root, which is
    what a unit test's ``tmp_path`` transcript wants.
    """
    parent = directory.parent
    if parent.name in ("sessions", "agents"):
        return parent.parent
    return config_dir()


def _run_record_evidence(directory: Path) -> tuple[Any, Any]:
    """``(live_foreign_owner, dead_owner)`` for this conversation's records.

    Read RAW rather than through ``registry.scan``, and in ONE pass that returns
    both facts, because ``scan`` used to UNLINK every stale record it met: a scan
    performed first would destroy exactly the dead-owner evidence this
    classification depends on (the daemon's own sweep is a scan, so the ordering
    is not hypothetical). ``scan`` now MOVES a dead record to the run
    directory's ``reaped/`` sidecar instead, and this reader reads BOTH
    directories for that reason: the live directory is where an unreaped record
    still sits, and the sidecar is where a sweep has already put it, so the
    classification no longer depends on whether some other process happened to
    sweep first (INCIDENT 2026-09-13, session ``5e109d459222``, which read as
    "the cause could not be determined" for a death that had a recorded cause).

    ``live_foreign_owner`` is a live pid that is NOT this process. Excluding
    ourselves is load-bearing rather than tidiness: a successor runtime publishes
    its own record BEFORE it constructs its ``Session``, so counting our own pid
    as a live owner would make every orphaned run unclassifiable — precisely the
    bug this change fixes.
    """
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import RUN_DIRNAME, SessionRecord

    run = _config_root_for(directory) / RUN_DIRNAME
    live: Any = None
    dead: Any = None
    try:
        if not run.is_dir():
            return None, None
        # Both the live directory and the sidecar a ``scan`` moves dead records
        # into. A record is in exactly one of them (the move is a rename), so
        # a path appearing twice is impossible and no dedupe is needed.
        reaped = run / registry.REAPED_DIRNAME
        paths = sorted(run.glob("*.json")) + sorted(reaped.glob("*.json"))
    except OSError:
        return None, None
    for path in paths:
        try:
            record = SessionRecord.from_json(json.loads(path.read_text()))
        except (OSError, ValueError, TypeError):
            continue
        if record.session_id != directory.name:
            continue
        # The zombie probe costs a `ps` fork, and boot is not a hot path — and a
        # zombie is NOT a live owner, which is the whole point of asking.
        if not registry.pid_alive(record.pid, check_zombie=True):
            if dead is None or record.started_at >= dead.started_at:
                dead = record
            continue
        if record.pid != os.getpid() and (live is None or record.started_at >= live.started_at):
            live = record
    return live, dead


def _stopped_marker(directory: Path) -> bool:
    """Whether this conversation's wake index carries a recorded STOP.

    ``stopped_at`` is stamped by the stop path (``control._mark_wakes_dormant``)
    and cleared on the session's next open, so it is a durable, transcript-derived
    positive marker of a user's own cancel — the one cause a cut-off must not
    report as an error. A schedule-less session has no entry at all, which is why
    the deliberate stop is ALSO carried in the outcome marker itself; this is
    corroboration for the case where the runtime wrote no marker, not the only
    evidence.
    """
    from local_operator.wakes import store as wake_store

    try:
        entry = wake_store.read_entry(_config_root_for(directory), directory.name)
    except Exception:  # noqa: BLE001 — an unreadable index is not a stop
        return False
    return bool(isinstance(entry, dict) and entry.get("stopped_at"))


def _durable_stop_marker(directory: Path) -> dict[str, Any] | None:
    """The durable stop marker a KILLER staged before an irreversible step.

    Read from the CONVERSATION directory, not from a run directory derived from
    a config root: the writer (``registry.write_stop_marker``) is handed the
    same transcript directory this classifier starts from, so one spelling
    keeps the two sides in step — and the marker must sit somewhere that
    outlives the record it describes, because a clean stop unpublishes the
    record and a sweep moves a dead one aside while the transcript directory
    survives both. That durability is the whole reason this rung exists: at the
    SIGKILL rung the target records nothing, so the file below is the only
    artifact that can say the runtime was killed, by whom, and whether it was
    asked for.
    """
    from local_operator.session.runtime import registry

    return registry.read_stop_marker(directory)


def _stop_marker_covers_run(
    marker: dict[str, Any],
    directory: Path,
    dead: Any | None,
    *,
    run_started_at: float | None = None,
) -> bool:
    """Whether ``marker`` attests to the RUN this classification is about.

    A marker is keyed to a RUN — ``(session_id, pid, started_at)`` — not to a
    session, and this is where that is enforced. A session can be stopped
    deliberately (marker naming sigkill), reopened, run again, and then die
    involuntarily; without this check the SURVIVING marker would narrate the
    later, unexplained death as the user's own act, which is the one misreading
    a durable marker can introduce. So: the session id must be this
    conversation's, and the marker has to be shown to describe THIS run by
    whichever run key is available.

    THE RUN KEY HAS TWO SOURCES, and neither may be skipped.

    * A dead RECORD, when one survived: pid and start time must be the
      marker's, which is the strongest form of the check.
    * The run's own START, when there is no record at all — and there is no
      record in the NORMAL rung-3 shape, because the ladder that wrote the
      marker is the same ladder that unpublished the record
      (``control._recover_record``), and because a sweep can move it to the
      ``reaped/`` sidecar's retention bound. Skipping the check there (this
      function used to ``return True``) let a marker from an EARLIER deliberate
      stop of the same session narrate a LATER involuntary death as
      ``interrupted``/``user-stop`` — naming the earlier run's killer — once the
      sidecar's own bound evicted the record. That is a wrong-verdict hole in
      the direction that HIDES a crash, reported as QA round 1's Q-1.

    WHY ``at`` IS THE BOUND AND ``started_at`` IS NOT: ``started_at`` is the
    TARGET PROCESS's start, which for any runtime that was already resident
    when its turn began — the ordinary case — is EARLIER than the run, so
    bounding on it would refuse legitimate markers. ``at`` is the moment the
    killer staged the file, and a stop of THIS run necessarily happens after
    this run started. A marker with no usable ``at`` keeps the old permissive
    answer rather than inventing a refusal: no bound is no evidence, and
    refusing on no evidence would delete the attribution rung 3 exists for.
    """
    if str(marker.get("session_id") or "") != directory.name:
        return False
    if dead is None:
        return _marker_postdates_run(marker, run_started_at)
    if int(marker.get("pid") or -1) != int(getattr(dead, "pid", -2) or -2):
        return False
    started = marker.get("started_at")
    if _is_stamp(started):
        return abs(float(started) - float(getattr(dead, "started_at", 0.0) or 0.0)) < (
            _RUN_KEY_TOLERANCE_S
        )
    return True


#: How far two readings of the SAME run key may differ, in seconds. The key is
#: written by two processes (the killer stamps the marker, the target stamped
#: its record) from the same clock, so this only absorbs rounding.
_RUN_KEY_TOLERANCE_S = 1.0


def _is_stamp(value: object) -> TypeGuard[float | int]:
    """Whether ``value`` is a usable epoch stamp (``bool`` is an ``int``).

    A ``TypeGuard`` rather than a plain ``bool`` so a caller that has already
    asked can read the stamp without re-narrowing it: a ``dict.get`` returns
    ``Any | None``, and ``float()`` of that is a type error the guard makes go
    away instead of an inline ``isinstance`` chain repeated at each use.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0


def _marker_postdates_run(marker: dict[str, Any], run_started_at: float | None) -> bool:
    """Whether a marker with NO record to compare against still fits this run.

    See :func:`_stop_marker_covers_run` for why the bound is the marker's own
    stamp and not the target's start time, and why an absent or unusable stamp
    is permissive.
    """
    if run_started_at is None or not _is_stamp(marker.get("at")):
        return True
    # These fractional writer stamps order DIFFERENT events, unlike the two
    # rounded copies of one process key compared above. Any look-behind here
    # attributes a rapid re-engagement's death to the preceding turn's stop.
    return float(marker["at"]) >= float(run_started_at)


def _record_detail(record: Any, *, lead: str = "") -> str:
    """A parenthetical naming the record that outlived its process, or ``""``.

    Kept to the record's own facts (build, pid, started-at) rather than prose,
    because these are the fields a reader would otherwise have to reconstruct
    from the log to answer "which runtime was this". The started-at is ONE
    token rather than two — see the note on its format below (design round 3,
    D7).

    ``lead`` IS HOW THE VICTIM IS TOLD APART FROM AN ACTOR (QA round 1, Q4/Q5).
    Every other ``runtime-killed`` reason on this arm carries a marker's
    ATTRIBUTION in the same brackets (``(its install generation was pruned by lop
    install prune, killer pid N)``), so a bare ``(pid N, started …)`` — which is
    the DEAD runtime's own identity — reads one word away from naming the process
    that did it. This rung has no actor to name, and ``incidents.KILL_UNATTRIBUTED``
    is how a verdict says so: the same affirmative statement about the gap that
    ``journal.death_verdict``'s open-row rung makes, so a marked death and an
    unmarked one are not read as the same sentence. Comma-separated rather than a
    second set of brackets, exactly as ``journal.row_detail`` takes its own
    ``lead``.
    """
    build = str(getattr(record, "version", "") or "")
    ref = str(getattr(record, "source_ref", "") or "")
    stamp = f"{build}@{ref}" if build and ref else build or ref
    started = getattr(record, "started_at", None)
    when = ""
    if isinstance(started, (int, float)) and not isinstance(started, bool) and started:
        # THE DATE AND THE TIME ARE ONE TOKEN, joined by a NON-BREAKING space,
        # because the wrap is what made them read as two facts. Measured on the
        # real `NoticeBlock` at 60 columns (56 content cells): with a plain
        # space the boundary landed between them — `… started 2026-09-12` /
        # `05:15:53)`, a 9-cell orphan line under a timestamp the reader has to
        # reassemble. The non-breaking space carries the value whole on the last
        # line at 60, and moves nothing anywhere else: 4/3/2/2 lines at
        # 60/80/100/120 before and after, with the split removed (design round
        # 3, D7 — the one-character fix it named).
        when = time.strftime("%Y-%m-%d\u00a0%H:%M:%S", time.localtime(started))
    parts = [
        part for part in (stamp, f"pid {record.pid}", f"started {when}" if when else "") if part
    ]
    if lead:
        parts.insert(0, lead)
    return f" ({', '.join(parts)})" if parts else ""


def _classify_orphaned_run(
    directory: Path, *, reaped_owner: Any | None = None, run_started_at: float | None = None
) -> tuple[str, str, str]:
    """``(kind, cause, reason)`` for a started run whose owner is gone.

    The taxonomy's default flips here: "no evidence" used to mean
    ``interrupted`` and now means ``error``, because a cut-off we cannot explain
    is not a stop. Only POSITIVE evidence of a deliberate act records an
    interruption. Evidence, in order:

    * a DURABLE STOP MARKER staged by the process that took the irreversible
      step (``runtime-stop.json``) → ``interrupted`` / ``user-stop``, with the
      rung and the killer as the detail. It outranks everything below because
      it is the only evidence a forced stop can leave: the target at the
      SIGKILL rung is not executing;
    * a stop recorded in the wake index (the in-process stop path) →
      ``interrupted`` / ``user-stop``, no detail;
    * THE RUNTIME'S OWN OPEN TURN JOURNAL ROW (``runtime/journal.py``) → a
      NAMED involuntary cause: ``runtime-shutdown`` when the row itself recorded
      the signal the runtime was leaving on (a stop sweep that reached its
      target), ``install-mid-update`` when the install on disk moved away from
      the build the row booted from (the install-window tear), and
      ``runtime-killed`` otherwise. This rung is what turns the third case below
      from an inference into a fact: an open row IS the runtime's own statement
      that a turn was in flight and never ended, which is the evidence the two
      rungs below cannot have. It sits BELOW the marker rungs deliberately — a
      marker is a killer's attestation that someone asked for this, and nothing
      about an unfinished turn may outrank a record that the stop was ordered;
    * a record on disk whose pid is dead → ``error`` / ``runtime-killed``, with
      the record's build, pid and start time as the detail, prefixed by
      ``incidents.KILL_UNATTRIBUTED`` because the only party this rung can name is
      the victim — the legacy path, reached unchanged when no journal row survives;
    * nothing at all → ``error`` / no cause, saying plainly that the cause could
      not be determined.

    Why the marker is FIRST, and why the order is the fix. Before it existed,
    the same event (a stop that escalated to SIGKILL against a runtime that
    could not answer) reached the operator as three incompatible verdicts —
    ``runtime-killed`` here, no cause at all once a sweep had reaped the record,
    and ``owner-lost`` viewer-side — because every rung below reports what the
    process left behind and a SIGKILLed process leaves NOTHING. A marker is
    written by the killer before the signal, so the deliberate act survives the
    process it was done to, and one event yields exactly one verdict.

    ``reaped_owner`` IS THE DEAD RUNG'S EVIDENCE WHEN SOMEBODY ALREADY TOOK IT.
    ``registry.scan`` moves a stale record into a sidecar as it reports it (it
    used to DELETE it), and this reader reads that sidecar too — so the daemon's
    discovery loop can sweep before the classifier runs and the classification
    is unchanged. The caller's own record is still preferred when it has one,
    because it is strictly more evidence (the sweep may have run before the
    caller proved the pid dead). Only the DEAD rung uses it: the marker rungs
    answer for themselves, and the caller's live-owner gate
    (:func:`_run_record_evidence` inside :func:`_import_transcript_outcome`) has
    already run, so a successor that published while the record was being reaped
    still wins.

    ``run_started_at`` IS THE MARKER RUNG'S SECOND RUN KEY. The marker is keyed
    to a run, and when no record survives to compare pid and start time against,
    the run's own start — the in-flight ``attention_started`` entry's timestamp,
    passed by the caller that already read it — is what keeps a marker from an
    EARLIER stop of the same session from narrating a later involuntary death as
    the user's own act (QA round 1, Q-1; see :func:`_stop_marker_covers_run` for
    why the bound is the marker's own stamp and not the target's ``started_at``).
    ``None`` — no entry, or a caller that has no transcript — keeps the
    marker-only answer, because refusing on no evidence would delete the
    attribution the rung-3 shape exists for.

    THE NO-EVIDENCE ARM CARRIES NO CAUSE AND ITS OWN SENTENCE. It used to read
    the ``runtime-killed`` sentence with a ``(the cause could not be determined)``
    parenthetical bolted on, and every surface that prints the reason made that
    a contradiction: the sidebar's ``_error_label`` drops a parenthetical by
    design (it is built for the build pair), so the one surface that trims the
    detail stated a definite cause in the one case where nothing is known
    (design/UX review round 1, D3/U3). ``CUT_OFF_UNKNOWN`` already exists for
    this branch and needs no hedging, and leaving ``cause`` empty is what keeps
    ``cause_from_reason(reason)`` — the inverse every reader relies on —
    agreeing with the reason instead of naming a mechanism nobody observed.
    """
    from local_operator.incidents import (
        CUT_OFF_UNKNOWN,
        DELIBERATE_CUT_OFF_CAUSE,
        KILL_UNATTRIBUTED,
        involuntary_kill_detail,
        render_cut_off_reason,
        render_stop_attribution,
    )

    # The dead record is read unconditionally rather than on the third rung
    # only: the marker above is keyed to a RUN, so the rung that uses it has to
    # know whether a dead record contradicts it (see
    # :func:`_stop_marker_covers_run`). One directory read either way — the
    # helper reads both the live directory and the reaped sidecar in one pass.
    _, dead = _run_record_evidence(directory)
    if dead is None:
        dead = reaped_owner
    marker = _durable_stop_marker(directory)
    if marker is not None and _stop_marker_covers_run(
        marker, directory, dead, run_started_at=run_started_at
    ):
        raw_killer = marker.get("killer")
        killer: dict[str, Any] = raw_killer if isinstance(raw_killer, dict) else {}
        # ``deliberate is False`` EXACTLY, rather than a falsy test: the arm below
        # must not swallow a marker that predates the flag. A marker with no
        # ``deliberate`` field at all is a deliberate stop from a build older than
        # this one, and its refusal to claim otherwise is what keeps this change
        # from re-labelling history.
        if marker.get("deliberate") is False:
            # AN INVOLUNTARY ACT, ATTRIBUTED BY ITS OWN MARKER. The harness
            # recorded what it was about to do before it did it
            # (``control.note_involuntary_stop``), so this death is NAMED rather
            # than inferred — and it is emphatically not a ``user-stop``: the
            # marker says ``deliberate: false``, which is what every caller asking
            # "did someone ask for this" reads. A marker outranks the journal row
            # below because it is an act's own attestation where the row is only
            # the victim's last statement; the row is not consulted here, so the
            # reason is the shared sentence plus the attribution (or
            # ``(unattributed)`` when the marker named no actor).
            return (
                "error",
                "runtime-killed",
                render_cut_off_reason(
                    "runtime-killed",
                    detail=involuntary_kill_detail(
                        mechanism=str(marker.get("mechanism") or ""),
                        # The acting component's own name when the writer gave one, else
                        # the same command-or-argv0 the deliberate arm falls back to —
                        # because a marker ALWAYS knows which process staged it, and a
                        # pid alone is the weakest form of the attribution this arm
                        # exists to provide when its process is long gone.
                        actor=str(
                            marker.get("actor")
                            or killer.get("command")
                            or killer.get("argv0")
                            or ""
                        ),
                        killer_pid=killer.get("pid"),
                    ),
                ),
            )
        if marker.get("deliberate"):
            return (
                "interrupted",
                DELIBERATE_CUT_OFF_CAUSE,
                render_cut_off_reason(
                    DELIBERATE_CUT_OFF_CAUSE,
                    detail=render_stop_attribution(
                        rung=str(marker.get("rung") or ""),
                        command=str(killer.get("command") or killer.get("argv0") or ""),
                        killer_pid=killer.get("pid"),
                    ),
                ),
            )
    if _stopped_marker(directory):
        return (
            "interrupted",
            DELIBERATE_CUT_OFF_CAUSE,
            render_cut_off_reason(DELIBERATE_CUT_OFF_CAUSE),
        )
    # THE RUNTIME'S OWN STATEMENT, preferred over every inference below it.
    # Imported function-locally for the same reason the ``incidents`` import
    # above is: this runs at session boot, and an instrument that cannot be
    # read must degrade to the legacy rungs rather than stop a session opening.
    try:
        from local_operator.session.runtime import journal

        row = journal.open_row_after_death(directory)
        if row is not None:
            return journal.death_verdict(row)
    except Exception:  # noqa: BLE001 — unreadable evidence is not a dead session
        logger.debug("turn journal evidence unreadable for %s", directory.name, exc_info=True)
    if dead is not None:
        # ONLY THE VICTIM IS KNOWN HERE, so that is what the bracket says before it
        # names anything: this rung is reached when no marker, no row and no wake
        # index entry survives, and an aside that opens with a bare pid sits one
        # word away from the marker arms' actor (QA round 1, Q4/Q5).
        return (
            "error",
            "runtime-killed",
            render_cut_off_reason(
                "runtime-killed", detail=_record_detail(dead, lead=KILL_UNATTRIBUTED)
            ),
        )
    return "error", "", CUT_OFF_UNKNOWN


def _import_transcript_outcome(
    transcript: Any,
    store: AttentionStore | None = None,
    *,
    witnessed_cut_off: tuple[str, str] | None = None,
    reaped_owner: Any | None = None,
) -> tuple[str, str, str, str] | None:
    """Explicit one-time import; never called by GET, SSE or focus observation.

    Returns the ``(kind, cause, reason, token)`` this call PUBLISHED — for an
    orphaned in-flight run, or for the dying runtime's own saved marker when that
    marker reports a CUT-OFF — and ``None`` when it published nothing new. The
    caller that owns a ``Session`` (and can therefore journal) uses it to
    narrate the cut-off once per token; the daemon sweep, which has no session,
    ignores it.

    An in-flight ``attention_started`` with no matching outcome means a process
    died mid-turn — or is STILL RUNNING it. The two are told apart by the run
    registry, and only the second may publish nothing: a provisional marker
    written for a healthy run would be a wrong ``error`` row that every
    surface's ``busy`` suppression HIDES rather than corrects, so the guard
    removes the class instead of relying on every front end's suppression.
    ``witnessed_cut_off`` is the ONE caller that may override that guard, and
    only a caller holding POSITIVE evidence the owner is gone: a ``(cause,
    reason)`` pair from a viewer that has just DELIVERED a cut-off verdict after
    the whole cold window of failed dials and syncs. It publishes a PROVISIONAL
    record for the run's token, so the live owner's real outcome supersedes it
    when it lands (``_supersedes_provisional``) — which is what lets a
    live-but-silent stop reach the sidebar without inventing a second, permanent
    verdict (review round 1, MINOR-2).

    Old baselines were memory-only. Unknown historical work keeps that no-flood
    baseline, while a persisted seen stamp older than the actual final assistant
    entry preserves unread. Metadata file mtimes are deliberately irrelevant.
    """
    store = store or AttentionStore()
    identity = conversation_identity(transcript.directory)
    # Imported here rather than at module scope for the same reason
    # ``_classify_orphaned_run`` does it: a broken ``incidents`` import must not
    # stop a session from booting (see the guard around the bootstrap call).
    from local_operator.incidents import is_cut_off_cause, is_deliberate_cause

    saved = transcript.latest_custom(ATTENTION_CUSTOM_TYPE)
    # The ENTRY rather than only its details, because the run's own START is the
    # bound the stop marker is checked against when the record is gone
    # (``_stop_marker_covers_run``): the entry's timestamp is this turn's start,
    # and it is the only run key left once the ladder unpublishes the record or
    # the reaped sidecar's retention bound evicts it.
    started_entry = transcript.latest_custom_entry("attention_started")
    started = dict(started_entry.payload.get("details", {})) if started_entry is not None else None
    run_started_at = float(started_entry.ts) if started_entry is not None else None
    if (
        isinstance(started, dict)
        and started.get("conversation_id") == identity
        and (not isinstance(saved, dict) or saved.get("token") != started.get("token"))
    ):
        token = started["token"]
        live_owner, _ = _run_record_evidence(transcript.directory)
        if live_owner is not None:
            if witnessed_cut_off is None:
                # In flight. Publish NOTHING: the live runtime will publish the real
                # outcome when the turn ends, and a marker written here would be a
                # provisional row it then has to supersede — or, worse, a row no
                # later writer ever corrects if the turn completes with an anchor
                # this classifier did not predict.
                return None
            # A WITNESSED death the classifier cannot classify. The record on disk
            # points at a pid that is still alive (or unverifiable), so the branch
            # above has to assume a live run — and a viewer that already told the
            # user the turn was cut off is the one party that KNOWS otherwise. The
            # record is written PROVISIONALLY for exactly that reason: the live
            # owner's own outcome for the same token replaces it with the real
            # anchor and kind, so the worst case is a row that corrects itself
            # (review round 1, MINOR-2).
            cause, reason = witnessed_cut_off
            # KIND FROM THE CAUSE, not hardcoded, which is what makes the
            # witnessed writer agree with the two other writers that read this
            # vocabulary (`daemon._projection_frame`'s fill and the narration
            # guard below both route through `is_deliberate_cause`). The cause is
            # always our own — the viewer that watched the turn end supplies it —
            # so a deliberate one here must land `interrupted` and an involuntary
            # one `error`; writing `error` for whatever arrived would have been
            # the single place the taxonomy's deliberate half was not consulted
            # (review round 2, NIT-2).
            kind = "interrupted" if is_deliberate_cause(cause) else "error"
            store.publish(
                identity,
                token,
                provisional_anchor(token),
                kind,
                reason=reason,
                cause=cause,
            )
            return kind, cause, reason, str(token)
        kind, cause, reason = _classify_orphaned_run(
            transcript.directory, reaped_owner=reaped_owner, run_started_at=run_started_at
        )
        store.publish(identity, token, provisional_anchor(token), kind, reason=reason, cause=cause)
        return kind, cause, reason, str(token)
    if isinstance(saved, dict) and saved.get("conversation_id") == identity:
        if saved.get("eligible", True):
            # The dying runtime's own marker, replayed VERBATIM — including its
            # kind, cause and reason. This is the one path that can report a
            # deliberate stop as `interrupted` after the process is gone, so
            # nothing here may re-derive the kind from the absence of evidence.
            kind = str(saved.get("kind") or "")
            cause = str(saved.get("cause") or "")
            reason = str(saved.get("reason") or "")
            store.publish(
                identity,
                saved["token"],
                saved["anchor"],
                kind,
                reason=reason,
                cause=cause,
            )
            if kind == "error" and is_cut_off_cause(cause):
                # A CUT-OFF the dying runtime could not narrate itself. Its own
                # `_journal_cut_off_once` is refused by `journal_incident`'s
                # `_disposed` guard — the dispose rung sets that flag before
                # the turn's `finally` publishes — so the update/shutdown/retire
                # family reached every SURFACE and no MODEL (review round 1,
                # MINOR-2 + QA Q-2). Returning the tuple is what makes THIS
                # boot journal it: `Session._journal_restored_cut_off` narrates
                # it once per token, so the next turn's context opens with
                # `[session incident]` rather than a model re-guessing what its
                # last half-delivered request did.
                #
                # MEMBERSHIP, not the presence of a cause (review round 2, Q3),
                # AND not one token's identity (review round 1, NIT-1). The guard
                # used to be `cause` truthiness, which narrates a marker carrying
                # ANY string into the model's history — measured on a hand-written
                # marker with `cause='not-a-real-cause'`: one `session_incident`
                # card, and the malformed token imported as the durable outcome.
                # So `is_cut_off_cause` decides: the VOCABULARY says this build
                # can render the token, and `DELIBERATE_CUT_OFF_CAUSES` says
                # understanding a token is not enough to call the turn a cut-off.
                # Both halves are sets, so a future DELIBERATE cause joins a set
                # instead of becoming a second comparison nobody remembers to
                # write — which is how a user's own `/stop` would have been
                # narrated as a cut-off the moment its token was coined.
                # `kind` is still READ rather than re-derived: the dying
                # runtime's marker is replayed verbatim.
                #
                # The cost is stated rather than hidden: a NEWER runtime's cause
                # token is not narrated to the model by this build, because this
                # build cannot say what it means. The durable outcome is still
                # imported above — kind, cause and reason as recorded — so every
                # surface reads the truth; only the `[session incident]` card,
                # which is the model-facing form of a cause this build can
                # render, is withheld. Reachability today is a corrupted marker
                # (every in-tree `note_cut_off` caller passes a vocabulary token),
                # which is why this is a claim-vs-code correction and not a live
                # operator bug.
                return kind, cause, reason, str(saved["token"])
        return None
    if store.state(identity)["completion_token"]:
        return None
    history = transcript.build_llm_history()
    if not history:
        return None
    final = history[-1]
    if (
        getattr(final, "role", None) != "assistant"
        or not getattr(final, "text", "")
        or getattr(final, "tool_calls", None)
    ):
        return None
    entry = next((row for row in reversed(transcript.entries()) if row.id == final.id), None)
    if entry is None:
        return None
    seen = None
    try:
        raw = json.loads((store.path.parent / "mobile-seen.json").read_text())
        value = raw.get("sessions", {}).get(transcript.directory.name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            seen = value
    except (OSError, ValueError, AttributeError):
        pass
    token = str(uuid.uuid5(uuid.NAMESPACE_URL, f"local-operator:{identity}:{final.id}"))
    store.publish(
        identity, token, final.id, "complete", baseline_seen=seen is None or seen >= entry.ts
    )
    return None


class AttentionStore:
    """One database per config root; no in-memory authority to become stale."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else config_dir() / "attention.db"

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A new placeholder must be private before SQLite writes any contents.
        self.path.touch(mode=0o600, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=_CONNECT_TIMEOUT_S)
        conn.row_factory = sqlite3.Row
        # The busy handler spelled where the lock policy is read from, next to
        # the driver timeout for the reason the sibling stores set both. See
        # `_BUSY_TIMEOUT_MS`: this store's write path is the one ~25 concurrent
        # sessions, the mobile daemon, the tunnel connector and the browser
        # bridge all reach, and the 2 s it used to allow is what expired first.
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        try:
            # Publish the complete schema in one transaction. Concurrent readers
            # can see the positively identified empty database or both tables,
            # never an intermediate schema with a missing receipt table.
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                if self._uninitialized(conn):
                    conn.execute(
                        "CREATE TABLE completions ("
                        "sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "conversation TEXT NOT NULL, token TEXT NOT NULL UNIQUE, "
                        "anchor TEXT NOT NULL, kind TEXT NOT NULL, "
                        "reason TEXT NOT NULL DEFAULT '', cause TEXT NOT NULL DEFAULT '')"
                    )
                    conn.execute(
                        "CREATE INDEX completion_conversation "
                        "ON completions(conversation, sequence)"
                    )
                    conn.execute(
                        "CREATE TABLE receipts ("
                        "conversation TEXT PRIMARY KEY, acknowledged INTEGER NOT NULL)"
                    )
                    # A store being created here has no completions yet, so
                    # there is no backlog to baseline against — an empty
                    # delivery watermark IS the correct starting point.
                    conn.execute(_CREATE_DELIVERIES)
                    conn.execute(_CREATE_MUTATIONS)
                else:
                    # Missing tables/columns in an established database are
                    # corruption, not permission to rebuild an empty watermark.
                    #
                    # `deliveries` IS DELIBERATELY ABSENT FROM THIS PROBE, and
                    # adding it is the one "tidy-up" that would brick every
                    # existing machine: a database written by any release
                    # before the background-completion notifier legitimately
                    # lacks the table, so probing for it would read every one
                    # of them as corrupt. The probe stays byte-identical to
                    # what shipped, which is what keeps the meaning of an
                    # established database unchanged.
                    conn.execute(
                        "SELECT sequence,conversation,token,anchor,kind FROM completions LIMIT 0"
                    )
                    conn.execute("SELECT conversation,acknowledged FROM receipts LIMIT 0")
                    # Additive migration, and the baseline rides the SAME
                    # transaction as the CREATE so no concurrent reader can
                    # ever observe an unbaselined table. `unseen` is a LEVEL,
                    # not an edge: without this, the first observer to upgrade
                    # would claim every historical completion still unread and
                    # fire a banner for each one (measured on the maintainer's
                    # live store: 171 unseen conversations out of 332
                    # completions; the "8" an earlier draft of this comment
                    # cited was the test fixture's count, not a live
                    # measurement — review round 1, n1). Baselining at creation
                    # also means the
                    # eleventh observer to start INHERITS the baseline rather
                    # than re-deriving one of its own — the watermark is a
                    # property of the database, not of a process.
                    if not conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='deliveries'"
                    ).fetchone():
                        conn.execute(_CREATE_DELIVERIES)
                        conn.execute(_BASELINE_DELIVERIES, (time.time(),))
                    # `mutations` is additive for the SAME reason and stays out
                    # of the probe above for the same reason: every database
                    # written before this fix legitimately lacks it, and
                    # probing for it would read all of them as corrupt.
                    #
                    # NO BASELINE, unlike `deliveries`. That table baselines
                    # because `unseen` is a LEVEL and an unbaselined watermark
                    # would re-announce history. A change detector is an EDGE:
                    # it only has to move when something changes AFTER this
                    # point, so seeding at 0 is correct and a historical
                    # supersede count would be meaningless anyway.
                    if not conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='mutations'"
                    ).fetchone():
                        conn.execute(_CREATE_MUTATIONS)
                    # `supersede_log` is additive for the same reason and stays
                    # out of the probe above for the same reason: a database
                    # written before this fix legitimately lacks it. NO BASELINE,
                    # again because it is an EDGE not a LEVEL — readers start
                    # from "nothing was healed before I connected", and seeding a
                    # historical set would replay corrections nobody is stale for
                    # (a reconnect takes a fresh snapshot that already carries the
                    # healed state).
                    if not conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='supersede_log'"
                    ).fetchone():
                        conn.execute(_CREATE_SUPERSEDE_LOG)
                    # ``reason``/``cause`` are ADDITIVE and stay out of the probe
                    # above for the reason the two tables do: every database
                    # written before the cut-off taxonomy legitimately lacks
                    # them, and naming them in the probe would read all of them
                    # as corrupt. ``DEFAULT ''`` rather than a nullable column
                    # keeps the readers one shape: an old row's reason is the
                    # empty string, i.e. "no reason was recorded", which is
                    # exactly what it is — never a claim that there was none to
                    # record.
                    columns = {
                        str(row[1]) for row in conn.execute("PRAGMA table_info(completions)")
                    }
                    if "reason" not in columns:
                        conn.execute(
                            "ALTER TABLE completions ADD COLUMN reason TEXT NOT NULL DEFAULT ''"
                        )
                    if "cause" not in columns:
                        conn.execute(
                            "ALTER TABLE completions ADD COLUMN cause TEXT NOT NULL DEFAULT ''"
                        )
            return conn
        except BaseException:
            conn.close()
            raise

    def _connect_read_only(self) -> sqlite3.Connection:
        """A ``mode=ro`` connection that waits as long as the write path does.

        READS CONTEND FOR THE SAME LOCK, and this store's reads are on the hot
        paths: the mobile daemon's scan (``revision``), every frontend list
        (``state_many``), and the sidebar's deltas. This store keeps SQLite's
        default rollback journal, so a writer holding the lock blocks a reader
        at ``BEGIN`` exactly as it blocks another writer -- and the operator's
        log shows the daemon's own scan losing that race 36 times
        (`AttentionStore().revision` raising `database is locked` out of
        `_uninitialized`). A read that raises costs the caller its whole tick,
        so it gets the same 5 s window rather than 2.

        Deliberately NOT ``_connect``: the read paths must stay unable to create
        the file or migrate the schema (``mode=ro`` is the mechanism, and
        ``test_reads_do_not_create_or_mutate_storage`` pins it). Only the wait
        is shared.
        """
        conn = sqlite3.connect(
            f"{self.path.as_uri()}?mode=ro", uri=True, timeout=_CONNECT_TIMEOUT_S
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        return conn

    @staticmethod
    def _uninitialized(conn: sqlite3.Connection) -> bool:
        # A first publisher's private 0600 placeholder is a valid zero-schema
        # SQLite database. Check BOTH facts rather than swallowing OperationalError:
        # corrupt bytes, dropped tables and partial schemas must remain errors.
        return (
            conn.execute("PRAGMA schema_version").fetchone()[0] == 0
            and conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone() is None
        )

    @staticmethod
    def _state(conn: sqlite3.Connection, conversation: str) -> dict[str, Any]:
        row = conn.execute(
            "SELECT * FROM completions WHERE conversation=? ORDER BY sequence DESC LIMIT 1",
            (conversation,),
        ).fetchone()
        receipt = conn.execute(
            "SELECT acknowledged FROM receipts WHERE conversation=?", (conversation,)
        ).fetchone()
        acknowledged = receipt[0] if receipt else 0
        return {
            "conversation_id": conversation,
            "completion_token": row["token"] if row else None,
            "anchor_id": row["anchor"] if row else None,
            "kind": row["kind"] if row else None,
            # Why, in the operator's words, plus the machine token. Both empty
            # for a completion and for any record written before the cut-off
            # taxonomy; a surface that wants to append the cause must treat ""
            # as "nothing to say" rather than as an empty sentence.
            "reason": (row["reason"] or "") if row else "",
            "cause": (row["cause"] or "") if row else "",
            "unseen": bool(row and row["sequence"] > acknowledged),
            "revision": [row["sequence"] if row else 0, acknowledged],
        }

    def state_many(self, conversations: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Read a whole frontend list on one connection and one consistent snapshot.

        Call this in a worker, then merge/render the returned map on the UI loop.
        Chunking bounds SQL parameters, not connection count; empty/new stores
        remain genuinely read-only and return explicit no-completion states.
        """
        identities = list(dict.fromkeys(conversations))
        states = {
            identity: {
                "conversation_id": identity,
                "completion_token": None,
                "anchor_id": None,
                "kind": None,
                "reason": "",
                "cause": "",
                "unseen": False,
                "revision": [0, 0],
            }
            for identity in identities
        }
        if not identities or not self.path.exists():
            return states
        with closing(self._connect_read_only()) as conn:
            conn.execute("BEGIN")
            if self._uninitialized(conn):
                return states
            for offset in range(0, len(identities), 500):
                chunk = identities[offset : offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    "SELECT c.*, COALESCE(r.acknowledged,0) AS acknowledged FROM "
                    "(SELECT conversation, MAX(sequence) AS sequence FROM completions "
                    f"WHERE conversation IN ({placeholders}) GROUP BY conversation) latest "
                    "JOIN completions c ON c.sequence=latest.sequence "
                    "LEFT JOIN receipts r ON r.conversation=c.conversation",
                    chunk,
                )
                for row in rows:
                    states[row["conversation"]] = {
                        "conversation_id": row["conversation"],
                        "completion_token": row["token"],
                        "anchor_id": row["anchor"],
                        "kind": row["kind"],
                        "reason": _optional_column(row, "reason"),
                        "cause": _optional_column(row, "cause"),
                        "unseen": row["sequence"] > row["acknowledged"],
                        "revision": [row["sequence"], row["acknowledged"]],
                    }
        return states

    def state(self, conversation: str) -> dict[str, Any]:
        return self.state_many([conversation])[conversation]

    def published_since(self, sequence: int) -> list[dict[str, Any]]:
        """Publications NEWER than ``sequence``, oldest first, as deltas.

        THE MACHINE-WIDE FEED'S READ, and it exists for the same reason
        ``revision()`` does: a poller that must notice a completion cannot pay
        ``state_many`` over the whole store on every tick. ``sequence`` is the
        AUTOINCREMENT primary key, so this is an index scan over exactly what
        happened since the caller's cursor, not a per-conversation lookup.

        Returns ``(conversation, sequence, token, kind)`` per row because that
        is the whole of what "a completion was published" needs: the caller
        keys its own per-session baseline on ``token`` (the durable identity)
        and decides eligibility from ``kind`` (``BRIDGE_NOTIFIABLE_KINDS``).
        The caller has to read ``state_many`` for the affected sessions
        afterwards for the wire shape — this read answers "which sessions
        moved", never "what does the card say".

        Read-only and missing-store tolerant, exactly like its neighbours: a
        store that does not exist yet has published nothing, and a poller that
        raised here would lose cross-process completion sync for the life of
        its loop. A pre-taxonomy database reads without ``reason``/``cause``
        because neither is selected.
        """
        if not self.path.exists():
            return []
        with closing(self._connect_read_only()) as conn:
            conn.execute("BEGIN")
            if self._uninitialized(conn):
                return []
            return [
                {
                    "conversation": row["conversation"],
                    "sequence": int(row["sequence"]),
                    "token": row["token"],
                    "kind": row["kind"],
                }
                for row in conn.execute(
                    "SELECT conversation, sequence, token, kind FROM completions "
                    "WHERE sequence > ? ORDER BY sequence",
                    (int(sequence),),
                )
            ]

    def superseded_since(self, sequence: int) -> list[dict[str, Any]]:
        """``{conversation}`` entries healed AFTER ``sequence``, oldest first.

        THE FEED'S SECOND DELTA (review round 1, R4). ``revision()`` reports that
        a heal happened but not which record it moved, and a heal deliberately
        changes neither ``MAX(sequence)`` nor ``SUM(acknowledged)`` — so both
        :meth:`published_since` and :meth:`acknowledgement_map` come back empty
        for it. A consumer following the revision alone therefore advanced its
        change detector and then published nothing, leaving its subscribers on
        the stale outcome the heal had just corrected. This read is what turns
        "a heal happened" into "publish a corrected state for THIS session".

        Read-only and missing-store/missing-table tolerant, exactly like its
        neighbours: ``supersede_log`` is additive, so a database whose runtime
        has not reconnected yet legitimately lacks it and must read as "nothing
        was healed" rather than raising. A reader that raised here would lose
        in-place corrections for the life of its loop.
        """
        if not self.path.exists():
            return []
        with closing(self._connect_read_only()) as conn:
            conn.execute("BEGIN")
            if self._uninitialized(conn):
                return []
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='supersede_log'"
            ).fetchone():
                return []
            return [
                {"sequence": int(row["seq"]), "conversation": row["conversation"]}
                for row in conn.execute(
                    "SELECT seq, conversation FROM supersede_log WHERE seq > ? ORDER BY seq",
                    (int(sequence),),
                )
            ]

    def acknowledgement_map(self) -> dict[str, int]:
        """``{conversation: acknowledged}`` for every conversation with a receipt.

        The second half of the feed's delta: a read is a durable change to the
        same watermark the unseen mark is computed from (``sequence >
        acknowledged``), so a session that is READ must be able to publish an
        ``attention`` frame that clears its own mark without a full re-read of
        the store. The caller diffs this against the map it held last tick and
        re-reads state only for the sessions whose value moved.

        Deliberately its own small read rather than a term of ``revision()``:
        ``SUM(acknowledged)`` is enough to know *something* moved but not
        *which*, and guessing the conversation is what would make a late frame
        un-read a row.
        """
        if not self.path.exists():
            return {}
        with closing(self._connect_read_only()) as conn:
            conn.execute("BEGIN")
            if self._uninitialized(conn):
                return {}
            return {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT conversation, MAX(acknowledged) FROM receipts GROUP BY conversation"
                )
            }

    def _retry_write(self, operation: Callable[[], _T]) -> _T:
        """Run one write, riding out a contended lock before giving it a name.

        WHY THE RETRY IS IN THE STORE AND NOT AT THE CALL SITES. A lock is a
        statement about OTHER writers, not about the caller's request, so a
        transient one must not become the caller's failure. The store owns the
        connection and the transaction, so this is where the second attempt --
        and the eventual classification -- belongs; a caller cannot retry a
        transaction it cannot see.

        WHY IT EXISTS AT ALL. Publish sits on paths with no failure ladder to
        catch it: the turn's own ``finally`` (``Session._publish_attention_outcome``,
        which is what a runtime's ASGI request handler runs) and the mobile
        daemon's boot sweep. A raise there is not a late notification, it is a
        request that never completes -- the operator's 2026-09-20 outage, where
        one expiring lock answered the phone with an ASGI abort.

        BOUNDED, and stated: ``_WRITE_ATTEMPTS`` attempts, each with SQLite's own
        ``_BUSY_TIMEOUT_MS`` window, plus ``_WRITE_BACKOFF_S`` between them. A
        pathological wait is the price of not dropping a completion on the
        floor, and it is paid on a worker thread, never on the loop (about 19 s
        measured worst case -- see the constants).

        ONLY CONTENTION IS RETRIED (:func:`_is_contention`, on SQLite's certified
        code). A full disk, an unopenable store and a corrupt schema are not
        races; re-running those would only delay the report, and the surface
        ladders have different, better sentences for each.

        The failure that does escape is :class:`AttentionWriteDeferred`, which
        keeps SQLite's own code so the existing ladders classify it as
        contention rather than as a broken store.
        """
        for attempt in range(_WRITE_ATTEMPTS):
            try:
                return operation()
            except sqlite3.OperationalError as error:
                if not _is_contention(error):
                    raise
                if attempt + 1 >= _WRITE_ATTEMPTS:
                    raise _deferred(error) from error
                logger.debug(
                    "attention: write attempt %d/%d met a locked store; retrying in %s s",
                    attempt + 1,
                    _WRITE_ATTEMPTS,
                    _WRITE_BACKOFF_S[attempt],
                )
                time.sleep(_WRITE_BACKOFF_S[attempt])
        raise AssertionError("unreachable: _WRITE_ATTEMPTS >= 1")

    def publish(
        self,
        conversation: str,
        token: str,
        anchor: str,
        kind: str,
        *,
        baseline_seen: bool | None = None,
        reason: str = "",
        cause: str = "",
    ) -> dict[str, Any]:
        """Import a durable outcome idempotently, including after runtime restart.

        A TOKEN MAY LEGITIMATELY ADVANCE OUT OF ITS PROVISIONAL RECORD, and
        refusing that used to brick a session permanently. One turn writes
        ``attention_started`` with token T and, on completion, writes the SAME T
        with the real anchor and kind. Anything that bootstraps while that turn
        is in flight — a mobile daemon sweep, a resume attempt, a killed runtime —
        publishes the provisional ``(completion-T, interrupted)`` marker first.
        When the turn then finishes and the conversation is next opened, the
        journal's authoritative ``(<entry id>, complete)`` arrives for the same
        T: not a conflict, just the same turn described exactly. Rejecting it
        raised out of ``bootstrap_transcript`` on every single spawn, so the
        runtime could never start and the conversation was unopenable forever.

        Supersession is deliberately narrow: same conversation, and the stored
        record must still wear the provisional shape — anchor
        ``completion-<token>``. (The stored KIND is no longer part of that shape:
        see ``_supersedes_provisional`` for why requiring ``interrupted`` would
        now be the bug.) Two refusals it does NOT relax. A token appearing under
        a DIFFERENT conversation stays an error, which is the integrity property
        this check exists for: a forked transcript carrying its parent's journal
        must not capture the parent's receipt. And a record already anchored to
        a real entry is never replaced, so a late bootstrap racing a finished
        turn cannot drag a real outcome back to a synthetic one.

        A genuinely interrupted turn is stored in that same provisional shape,
        and that is intended rather than an ambiguity to resolve: the only
        writer that can present a different outcome for an existing token is
        this conversation's own journal replaying that same run, so the record
        it supersedes is by construction a description of the run that the
        journal now describes better. An unrelated turn always carries its own
        freshly minted token.

        The supersede is an UPDATE IN PLACE, which is what keeps it idempotent
        and flood-free: ``sequence`` is the receipt watermark, so minting a new
        row would resurrect an ALREADY-ACKNOWLEDGED turn as unread and re-fire
        a notification for a result the human has read. Correcting a record is
        not a new completion. Holding the sequence also means a corrected old
        turn stays where it belongs in the order instead of jumping ahead of
        newer ones.

        THAT CHOICE HAS A COST, and it is paid explicitly rather than left
        implicit (review round 1, major-1). Holding ``sequence`` means the heal
        moves neither term of the OLD ``revision()``, and two consumers gate on
        that value: the desktop bridge skips ``refresh_attention`` and keeps
        serving the stale ``interrupted`` record with its synthetic anchor to
        phone and desktop subscribers, and the TUI's background notifier skips
        its catalog scan. On a quiet machine a phone could show "Interrupted"
        for a turn that completed successfully, until some unrelated session
        happened to publish. So a supersede bumps ``mutations.supersedes``,
        which ``revision()`` folds into its tuple: the heal moves the change
        detector WITHOUT moving the watermark, and an acknowledged turn stays
        read. Detectable and flood-free are not in tension once they are
        separate counters.

        CONTENTION IS RIDDEN OUT HERE, NOT HANDED TO THE CALLER. The transaction
        runs under the bounded retry below (:meth:`_retry_write`), and only if
        every attempt meets SQLite's busy verdict does it raise
        :class:`AttentionWriteDeferred` -- never the bare ``OperationalError``
        this method used to give a daemon request handler, which answered the
        phone with an ASGI abort (2026-09-20, see ``_BUSY_TIMEOUT_MS``).
        """
        if kind not in {"complete", "error", "interrupted"} or not anchor:
            raise ValueError("invalid completion")
        reason = str(reason or "")[:REASON_WIRE_CHARS]
        if str(uuid.UUID(token)) != token:
            raise ValueError("invalid completion token")
        # Split from the transaction so the retry has something to call: the
        # statements below are unchanged, and re-running them is safe because a
        # replay is idempotent by token (``INSERT OR IGNORE``, plus a supersede an
        # identical replay cannot re-apply) and every attempt builds its own
        # connection, so no half-finished transaction is ever resumed.
        return self._retry_write(
            lambda: self._publish_outcome(
                conversation,
                token,
                anchor,
                kind,
                baseline_seen=baseline_seen,
                reason=reason,
                cause=cause,
            )
        )

    def _publish_outcome(
        self,
        conversation: str,
        token: str,
        anchor: str,
        kind: str,
        *,
        baseline_seen: bool | None,
        reason: str,
        cause: str,
    ) -> dict[str, Any]:
        """One attempt at :meth:`publish`'s transaction, on its own connection."""
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if (
                baseline_seen is not None
                and conn.execute(
                    "SELECT 1 FROM completions WHERE conversation=? LIMIT 1", (conversation,)
                ).fetchone()
            ):
                return self._state(conn, conversation)
            existing = conn.execute(
                "SELECT conversation, anchor, kind FROM completions WHERE token=?", (token,)
            ).fetchone()
            if existing and tuple(existing) != (conversation, anchor, kind):
                if not _supersedes_provisional(existing, conversation, token, anchor):
                    raise ValueError("completion token belongs to another outcome")
                conn.execute(
                    "UPDATE completions SET anchor=?, kind=?, reason=?, cause=? WHERE token=?",
                    (anchor, kind, reason, cause, token),
                )
                # Inside the SAME transaction as the UPDATE: a reader must
                # never observe a healed row whose change the detector has not
                # yet counted, or it would cache the new state under the old
                # revision and then ignore the next real change.
                conn.execute(_BUMP_SUPERSEDES)
                # ...and WHICH record moved, inside the same transaction and for
                # the same reason: a reader must never observe a healed row whose
                # identity has not been counted yet, or it would cache the healed
                # state under the old revision and then ignore the next change.
                conn.execute(_APPEND_SUPERSEDE, (conversation,))
                conn.execute(_PRUNE_SUPERSEDE_LOG, (_SUPERSEDE_LOG_RETENTION,))
            conn.execute(
                "INSERT OR IGNORE INTO completions(conversation,token,anchor,kind,reason,cause) "
                "VALUES(?,?,?,?,?,?)",
                (conversation, token, anchor, kind, reason, cause),
            )
            if baseline_seen:
                sequence = conn.execute(
                    "SELECT sequence FROM completions WHERE token=?", (token,)
                ).fetchone()[0]
                conn.execute(
                    "INSERT OR IGNORE INTO receipts(conversation,acknowledged) VALUES(?,?)",
                    (conversation, sequence),
                )
            return self._state(conn, conversation)

    def revision(self) -> tuple[int, int, int]:
        """Cheap process-independent change detector for existing polling loops.

        Three terms, because there are three kinds of durable change and the
        first two do not cover the third: ``MAX(sequence)`` moves on a publish,
        ``SUM(acknowledged)`` on a read, and ``supersedes`` on a heal that
        deliberately moves NEITHER of the others (see :meth:`publish`). Callers
        compare the tuple for equality and never interpret the terms, so widening
        it is safe for them; the per-conversation ``state()["revision"]`` pair is
        a separate, unchanged wire contract (``AttentionState.revision``, mirrored
        by the mobile client) and deliberately does not grow a third element.
        """
        if not self.path.exists():
            return (0, 0, 0)
        with closing(self._connect_read_only()) as conn:
            conn.execute("BEGIN")
            if self._uninitialized(conn):
                return (0, 0, 0)
            # This connection is READ-ONLY, so it cannot run the additive
            # migration itself: a database written before this fix, whose runtime
            # has not reconnected yet, legitimately has no `mutations` table and
            # must read as 0 rather than raising. A poller that raised here
            # would lose cross-process read sync for the life of the loop.
            row = conn.execute(
                "SELECT COALESCE(MAX(sequence),0), "
                "(SELECT COALESCE(SUM(acknowledged),0) FROM receipts), "
                "(SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='mutations') "
                "FROM completions"
            ).fetchone()
            supersedes = 0
            if row[2]:
                supersedes = conn.execute(
                    "SELECT COALESCE(MAX(supersedes),0) FROM mutations"
                ).fetchone()[0]
            return (row[0], row[1], supersedes)

    def acknowledge(self, conversation: str, token: str) -> dict[str, Any]:
        """Advance only through the observed token, never through server 'now'.

        THE TOKEN MUST BE THE CONVERSATION'S CURRENT COMPLETION, or the
        conversation must already be read. Anything else raises
        :class:`SupersededCompletionToken`, so a 200 from this method means one
        thing and only one: *this conversation is read now* (`unseen` false).

        WHY THE OLDER TOKEN IS REFUSED RATHER THAN RECORDED. The watermark is
        monotonic, so acknowledging an older token could only ever move the
        receipt to that token's sequence -- which, while a newer completion is
        still unseen, is a movement no surface can observe: `unseen` is computed
        against the NEWEST sequence, so the conversation stays unread either way.
        Returning 200 for it is what made the no-op indistinguishable from a read
        (the operator's defect: the desktop app sent a superseded token, got a
        200, latched, and its checkmark never cleared). The honest answer is that
        the caller is looking at a result the conversation has moved past: refuse
        it, and let the caller re-read the token that is current. THE REFUSAL
        CARRIES NO STATE, deliberately: the wire body is a machine ``code`` plus
        one operator-facing sentence (see :data:`SUPERSEDED_TOKEN_CODE`), because
        the state that settles this is the CALLER's own projection -- the thing it
        is already subscribed to and must refresh to learn which token is current
        now. A state computed here would be a second, already-stale opinion about
        a conversation the caller is watching, and a caller that trusted it
        instead of refreshing would be exactly as stuck as before. A DELAYED OR
        DUPLICATE RECEIPT STILL CONVERGES -- that is the
        case where the conversation is already read, and there this method answers
        with the read state exactly as before, which is what out-of-order
        delivery of a buffered receipt needs.

        The whole decision is made inside the write transaction: ``BEGIN
        IMMEDIATE`` serialises it against a concurrent :meth:`publish`, so the
        current sequence this compares against cannot be advanced under it, and
        the state returned was computed from the same snapshot.
        """
        if not isinstance(token, str) or len(token) != 36 or not self.path.exists():
            raise ValueError("unknown completion token")
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT sequence FROM completions WHERE conversation=? AND token=?",
                (conversation, token),
            ).fetchone()
            if row is None:
                raise ValueError("unknown completion token")
            current = conn.execute(
                "SELECT COALESCE(MAX(sequence),0), "
                "COALESCE((SELECT acknowledged FROM receipts WHERE conversation=?),0)"
                " FROM completions WHERE conversation=?",
                (conversation, conversation),
            ).fetchone()
            # `current[0] == 0` is a conversation with no completions at all,
            # which cannot happen for a token that was just found. Acknowledged
            # past the newest sequence is the already-read case: a reordered or
            # duplicate receipt lands here and must keep converging.
            if row[0] != current[0] and current[1] < current[0]:
                raise SupersededCompletionToken
            conn.execute(
                "INSERT INTO receipts(conversation,acknowledged) VALUES(?,?) "
                "ON CONFLICT(conversation) DO UPDATE SET acknowledged="
                "MAX(receipts.acknowledged,excluded.acknowledged)",
                (conversation, row[0]),
            )
            return self._state(conn, conversation)

    def acknowledge_many(self, items: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
        """Acknowledge several completions in ONE transaction, verdicts per item.

        The bulk half of :meth:`acknowledge`, and the SAME decision per item --
        it exists because a gesture that means "these" (the desktop sidebar's
        clear-all, the TUI's ``/notifications read``) cannot be honest as a
        watermark sweep. See the module docstring for the rule and
        ``docs/ATTENTION.md`` R10 for the one deliberate relaxation: a user may
        acknowledge the completions a surface ENUMERATES, token-bound.

        Returns one entry per input item, IN INPUT ORDER::

            {"conversation_id": str, "completion_token": str,
             "status": "read" | "superseded" | "unknown",
             "state": dict | None}   # post-write state iff status == "read"

        * ``read`` -- this token is the conversation's current completion, or
          the conversation is already read. The receipt moves to
          ``MAX(existing, sequence)`` and ``state`` is computed in the same
          snapshot. Already-read items report ``read`` and move nothing, so a
          repeated batch is idempotent and does not bump :meth:`revision`.
        * ``superseded`` -- a real completion of that conversation that is no
          longer current while the conversation is still unseen. Nothing is
          written and no state is returned, for the reason :meth:`acknowledge`
          documents at length: the settling state is the caller's, and a copy
          computed here would be stale before it rendered.
        * ``unknown`` -- no ``completions`` row for that pair, including a store
          file that does not exist. Nothing is written, and this call never
          MATERIALISES the database (the read-only tolerance :meth:`state_many`
          and :meth:`claim_delivery` share): a refused receipt must not create
          the store it could not find.

        ONE TRANSACTION, and each half of that is load-bearing. A per-item
        transaction would expose a half-applied batch to every poller and take N
        write-lock acquisitions; one ``BEGIN IMMEDIATE`` gives atomic visibility
        (an observer sees all of the batch or none of it), one snapshot for every
        verdict (so a concurrent :meth:`publish` cannot advance a sequence
        underneath a comparison), and a whole-batch rollback on failure -- a
        ``sqlite3.Error`` here leaves no receipt moved, which is what lets the
        route answer through the shared store-failure ladder while claiming
        nothing. Item verdicts are
        NOT errors, so ``superseded``/``unknown`` items never roll the batch back.

        Touches ``receipts`` ONLY. ``deliveries`` is a different fact (somebody
        was told, and notifying is not reading), ``supersede_log``/``mutations``
        record heals rather than reads, and no ``completions`` row is inserted or
        renumbered -- so nothing can be resurrected as unread by clearing a pile.

        DUPLICATES ARE LEGAL AND EVALUATED IN ORDER: ``[{a, T1}, {a, T2}]``
        yields exactly one ``read`` and one verdict for the other,
        deterministically. Deliberately no dedupe -- one policy site, and a
        caller that sends a pair twice gets the same answer the single-item walk
        would give it.
        """
        results = [
            {
                "conversation_id": conversation,
                "completion_token": token,
                "status": "unknown",
                "state": None,
            }
            for conversation, token in items
        ]
        if not results:
            return results
        # The missing-store case answers before `_connect()`, which creates the
        # file and its schema: an arbitration read that found nothing must not
        # leave a database behind.
        if not self.path.exists():
            return results
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            for result in results:
                token = result["completion_token"]
                if not isinstance(token, str) or len(token) != 36:
                    continue
                conversation = result["conversation_id"]
                row = conn.execute(
                    "SELECT sequence FROM completions WHERE conversation=? AND token=?",
                    (conversation, token),
                ).fetchone()
                if row is None:
                    continue
                current = conn.execute(
                    "SELECT COALESCE(MAX(sequence),0), "
                    "COALESCE((SELECT acknowledged FROM receipts WHERE conversation=?),0)"
                    " FROM completions WHERE conversation=?",
                    (conversation, conversation),
                ).fetchone()
                if row[0] != current[0] and current[1] < current[0]:
                    result["status"] = "superseded"
                    continue
                conn.execute(
                    "INSERT INTO receipts(conversation,acknowledged) VALUES(?,?) "
                    "ON CONFLICT(conversation) DO UPDATE SET acknowledged="
                    "MAX(receipts.acknowledged,excluded.acknowledged)",
                    (conversation, row[0]),
                )
                result["status"] = "read"
                result["state"] = self._state(conn, conversation)
        return results

    def claim_delivery(self, conversation: str, token: str, backend: str) -> bool:
        """True iff THIS caller may notify about ``token``. Exactly one ever wins.

        The arbitration point for N observer processes watching one shared
        store. Every frontend polls attention for every session, so without a
        claim, eleven running sessions would each announce the same completion
        eleven times. `BEGIN IMMEDIATE` is the serialization: both racers take
        the write lock, and the loser reads the winner's already-advanced
        watermark from inside its own transaction. This is the identical
        argument `receipts` already rests on, which is why the table lives here
        rather than beside the store in a lock file of its own.

        NEVER ACKNOWLEDGES. Delivering a toast about a session says nothing
        about whether the human read it, so this touches `deliveries` only and
        the sidebar's unseen mark survives untouched.

        CLAIM-THEN-DELIVER, deliberately. A process dying between the claim and
        the spawn loses that one toast; delivering first and claiming after
        would instead give a crash-looping process a repeating banner. The lost
        signal is only the transient nudge \u2014 `unseen` stays true, the checkmark
        stays in the sidebar, and `lop sessions` still reports it \u2014 whereas a
        duplicate is visible and repeats. Backends that can report their own
        failure hand the claim back through :meth:`release_delivery`.

        THE ACCEPTED RISK, stated plainly so the next reader does not have to
        rediscover it: a claimant killed between the claim and the spawn leaves
        that completion delivered-but-unannounced FOREVER. There is no lease,
        no holder pid and no expiry \u2014 a re-claim returns False for good, and
        nothing sweeps it. That is bounded in CONSEQUENCE (one transient toast,
        for one completion, on a process that crashed) and unbounded in TIME,
        and it is accepted here because the alternative trade is worse: a lease
        needs a clock, and a clock in this predicate is what lets two observers
        with disagreeing time both deliver. The durable signal survives
        regardless, which is what makes the transient one safe to lose. A lease
        carrying a holder pid is the natural fix if this ever stops being
        acceptable; the schema has no room for one today (review round 1 m1,
        QA round 1 Q3 \u2014 both judged the trade correct).

        A store that does not exist yet holds no completion to claim, so this
        never creates one: an arbitration read must not be the thing that
        materialises the database.
        """
        if not self.path.exists():
            return False
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT sequence FROM completions WHERE conversation=? AND token=?",
                (conversation, token),
            ).fetchone()
            if row is None:
                # An unknown token is never invented into the watermark: a
                # caller racing a store that was cleared behind it must not be
                # able to write a sequence no completion ever had.
                return False
            sequence = int(row[0])
            current = conn.execute(
                "SELECT delivered FROM deliveries WHERE conversation=?", (conversation,)
            ).fetchone()
            if current is not None and int(current[0]) >= sequence:
                return False
            conn.execute(
                "INSERT INTO deliveries(conversation,delivered,delivered_at,backend) "
                "VALUES(?,?,?,?) ON CONFLICT(conversation) DO UPDATE SET "
                "delivered=MAX(deliveries.delivered,excluded.delivered), "
                "delivered_at=excluded.delivered_at, backend=excluded.backend",
                (conversation, sequence, time.time(), backend),
            )
            # MAX on conflict mirrors `acknowledge` exactly, so reordered
            # writes converge upward instead of regressing the watermark.
            return True

    def release_delivery(self, conversation: str, token: str) -> bool:
        """Hand a claim back when the backend that took it delivered nothing.

        Without this, a backend failing AFTER the claim (notifications turned
        off between the two, `osascript` missing, a spawn refused) leaves the
        watermark asserting a toast that never reached anyone \u2014 a silent hole
        of exactly the kind this feature exists to close.

        Compare-and-swap, not `MAX`: rolling a watermark BACK is the one
        operation monotonic convergence cannot express. The update fires only
        while `delivered` is still the sequence this caller claimed, so an
        observer that has since claimed something newer is never clobbered.
        Rolling back to ``sequence - 1`` rather than deleting the row keeps
        every OLDER completion of this conversation delivered (their sequences
        are lower still), so a released claim re-opens exactly one event.
        """
        if not self.path.exists():
            return False
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT sequence FROM completions WHERE conversation=? AND token=?",
                (conversation, token),
            ).fetchone()
            if row is None:
                return False
            sequence = int(row[0])
            cursor = conn.execute(
                "UPDATE deliveries SET delivered=?, delivered_at=?, backend='released' "
                "WHERE conversation=? AND delivered=?",
                (sequence - 1, time.time(), conversation, sequence),
            )
            return cursor.rowcount > 0
