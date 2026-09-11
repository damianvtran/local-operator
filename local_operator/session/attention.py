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
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from local_operator.paths import config_dir

logger = logging.getLogger(__name__)

ATTENTION_CAPABILITY = "completion-ack-v1"
ATTENTION_CUSTOM_TYPE = "completion_attention"

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
    transcript: Any, store: AttentionStore | None = None
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
    """
    try:
        return _import_transcript_outcome(transcript, store)
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
            # is a real file (a spawned runtime's `log_dir()/mobile.log`), so
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
    both facts, because ``scan`` UNLINKS every stale record it meets: a scan
    performed first would destroy exactly the dead-owner evidence this
    classification depends on (the daemon's own sweep is a scan, so the ordering
    is not hypothetical).

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
        paths = sorted(run.glob("*.json"))
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


def _record_detail(record: Any) -> str:
    """A parenthetical naming the record that outlived its process, or ``""``.

    Kept to the record's own facts (build, pid, started-at) rather than prose,
    because these are the fields a reader would otherwise have to reconstruct
    from the log to answer "which runtime was this".
    """
    build = str(getattr(record, "version", "") or "")
    ref = str(getattr(record, "source_ref", "") or "")
    stamp = f"{build}@{ref}" if build and ref else build or ref
    started = getattr(record, "started_at", None)
    when = ""
    if isinstance(started, (int, float)) and not isinstance(started, bool) and started:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started))
    parts = [
        part for part in (stamp, f"pid {record.pid}", f"started {when}" if when else "") if part
    ]
    return f" ({', '.join(parts)})" if parts else ""


def _classify_orphaned_run(directory: Path) -> tuple[str, str, str]:
    """``(kind, cause, reason)`` for a started run whose owner is gone.

    The taxonomy's default flips here: "no evidence" used to mean
    ``interrupted`` and now means ``error``, because a cut-off we cannot explain
    is not a stop. Only POSITIVE evidence of a deliberate act records an
    interruption. Evidence, in order:

    * a recorded stop marker → ``interrupted`` / ``user-stop``;
    * a record on disk whose pid is dead → ``error`` / ``runtime-killed``, with
      the record's build, pid and start time as the detail — this is the case
      study's shape, and the one that used to read as a user cancel;
    * nothing at all → ``error`` / ``runtime-killed``, saying plainly that the
      cause could not be determined.
    """
    from local_operator.incidents import render_cut_off_reason

    if _stopped_marker(directory):
        return "interrupted", "user-stop", render_cut_off_reason("user-stop")
    _, dead = _run_record_evidence(directory)
    if dead is not None:
        return (
            "error",
            "runtime-killed",
            render_cut_off_reason("runtime-killed", detail=_record_detail(dead)),
        )
    return (
        "error",
        "runtime-killed",
        render_cut_off_reason("runtime-killed", detail=" (the cause could not be determined)"),
    )


def _import_transcript_outcome(
    transcript: Any, store: AttentionStore | None = None
) -> tuple[str, str, str, str] | None:
    """Explicit one-time import; never called by GET, SSE or focus observation.

    Returns the ``(kind, cause, reason, token)`` this call PUBLISHED for an
    orphaned in-flight run, or ``None`` when it published nothing new. The
    caller that owns a ``Session`` (and can therefore journal) uses it to
    narrate the cut-off once per token; the daemon sweep, which has no session,
    ignores it.

    An in-flight ``attention_started`` with no matching outcome means a process
    died mid-turn — or is STILL RUNNING it. The two are told apart by the run
    registry, and only the second may publish nothing: a provisional marker
    written for a healthy run would be a wrong ``error`` row that every
    surface's ``busy`` suppression HIDES rather than corrects, so the guard
    removes the class instead of relying on every front end's suppression.

    Old baselines were memory-only. Unknown historical work keeps that no-flood
    baseline, while a persisted seen stamp older than the actual final assistant
    entry preserves unread. Metadata file mtimes are deliberately irrelevant.
    """
    store = store or AttentionStore()
    identity = conversation_identity(transcript.directory)
    saved = transcript.latest_custom(ATTENTION_CUSTOM_TYPE)
    started = transcript.latest_custom("attention_started")
    if (
        isinstance(started, dict)
        and started.get("conversation_id") == identity
        and (not isinstance(saved, dict) or saved.get("token") != started.get("token"))
    ):
        token = started["token"]
        live_owner, _ = _run_record_evidence(transcript.directory)
        if live_owner is not None:
            # In flight. Publish NOTHING: the live runtime will publish the real
            # outcome when the turn ends, and a marker written here would be a
            # provisional row it then has to supersede — or, worse, a row no
            # later writer ever corrects if the turn completes with an anchor
            # this classifier did not predict.
            return None
        kind, cause, reason = _classify_orphaned_run(transcript.directory)
        store.publish(identity, token, provisional_anchor(token), kind, reason=reason, cause=cause)
        return kind, cause, reason, str(token)
    if isinstance(saved, dict) and saved.get("conversation_id") == identity:
        if saved.get("eligible", True):
            # The dying runtime's own marker, replayed VERBATIM — including its
            # kind, cause and reason. This is the one path that can report a
            # deliberate stop as `interrupted` after the process is gone, so
            # nothing here may re-derive the kind from the absence of evidence.
            store.publish(
                identity,
                saved["token"],
                saved["anchor"],
                saved["kind"],
                reason=str(saved.get("reason") or ""),
                cause=str(saved.get("cause") or ""),
            )
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
        conn = sqlite3.connect(self.path, timeout=2.0)
        conn.row_factory = sqlite3.Row
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
        with closing(
            sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=2.0)
        ) as conn:
            conn.row_factory = sqlite3.Row
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
        """
        if kind not in {"complete", "error", "interrupted"} or not anchor:
            raise ValueError("invalid completion")
        reason = str(reason or "")[:REASON_WIRE_CHARS]
        if str(uuid.UUID(token)) != token:
            raise ValueError("invalid completion token")
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
        with closing(
            sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=2.0)
        ) as conn:
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
        """Advance only through the observed token, never through server 'now'."""
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
            conn.execute(
                "INSERT INTO receipts(conversation,acknowledged) VALUES(?,?) "
                "ON CONFLICT(conversation) DO UPDATE SET acknowledged="
                "MAX(receipts.acknowledged,excluded.acknowledged)",
                (conversation, row[0]),
            )
            return self._state(conn, conversation)

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
