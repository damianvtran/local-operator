"""The durable ask log and its derived index.

WHY THIS MODULE EXISTS (design ``docs/design/ask-nonblocking.md`` §2.2). A
queued ask has to survive a runtime death, be answerable from a surface that has
no runtime, and be delivered exactly once by whichever runtime boots next. Three
properties follow, and each one dictates a shape:

* **Truth: ``<config_dir>/sessions/<sid>/asks.jsonl``, append-only.**
  The writers are several processes and not all of them hold the transcript
  lease (a desktop server, the phone daemon, ``lop`` in another terminal), so
  the log is append-only events under ``O_APPEND`` plus a bounded ``LOCK_NB``
  flock — the ``session/runtime/inbox.py`` discipline, for the same reason: on
  macOS/BSD a blocking flock makes ``close()`` block too, and a lock held by an
  unrelated process must never freeze a surface that is painting.
* **Status is a PURE FOLD of ``(events, now)``**, with no ``timed_out`` event.
  A written timeout would race the answer it is deciding against; a fold has
  exactly one answer for every input and no write to lose in a crash.
* **The index is derived and self-healing** (``<config_dir>/asks/<sid>.json``),
  the ``wakes/store.py`` contract exactly: corrupt or unknown-schema ⇒ absent,
  and any writer recomputes it from the log, so last-write-wins is safe.

**Reader tolerance is not optional.** The tick that reads this file runs every
60 s beside a writer, so a torn final line (a crash mid-``write``) must be
skipped rather than fatal — the same rule ``inbox._parse`` applies. And the
index lives OUTSIDE the session directory (a flat ``asks/`` directory, one file
per session with anything to show), so the cross-session view is
O(sessions-with-asks) rather than a scan of every session directory; the price
is that the session directory's own cleanup guards do not cover it, which is why
:func:`read_index` sweeps (see :func:`entry_is_stale`) and
``session/cleanup.py`` deletes the entry beside the directory it reaps.

**This module must stay stdlib-only.** Its readers include the aggregate list
view and the cleanup sweep, both of which run where the harness must not be
loaded. Import nothing from ``local_operator`` here —
``tests/unit/test_import_graph.py`` pins that in a fresh interpreter.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: The event log inside ``sessions/<id>/``.
ASKS_LOG_NAME = "asks.jsonl"

#: Subdirectory of the config dir holding one ``<session_id>.json`` per session
#: with anything to show. Flat, not nested under ``sessions/``: see the module
#: docstring for why the index deliberately does not live with the transcript.
ASKS_DIRNAME = "asks"

#: The session-directory root, needed by the staleness sweep (an entry whose
#: session directory is gone is an orphan) and by nothing else here.
SESSIONS_DIRNAME = "sessions"

#: Bumped only on an incompatible change to the entry shape. A reader skips an
#: entry whose schema it does not understand (the owning session rewrites it on
#: its next reconcile, so the bump heals like a deleted file).
INDEX_SCHEMA = 1

#: The event schema version, carried on every row.
EVENT_SCHEMA = 1

EVENT_QUEUED = "queued"
EVENT_ANSWERED = "answered"
EVENT_DECLINED = "declined"
EVENT_DISMISSED = "dismissed"
#: THE REVISION EVENT (design §10, #1936). It exists ONLY to SUPERSEDE an
#: already-recorded, not-yet-delivered answer: :func:`fold` keeps status,
#: ``answered_at`` and ``answered_by`` from the FIRST ``answered`` row and takes
#: the effective ``answers`` from the LATEST ``revised`` one. A revision that
#: arrives before any answer is recorded degrades to a plain ``answered`` and
#: never writes this kind (see ``AskQueue.revise``), so every ``revised`` row has
#: an ``answered`` row to supersede — the fold never honours a revision alone.
EVENT_REVISED = "revised"

#: THE WITHDRAWAL EVENT (design §12). Agent-authored: the asker retracting a
#: question it no longer needs answered. Terminal-on-write like ``dismissed``,
#: with ONE carve-out spelled in :func:`fold`: an ``answered`` row always wins
#: over a later ``withdrawn``, because the operator's real answer must never be
#: swallowed by the asker's retraction. Unknown to builds older than this one,
#: deliberately: an old fold skips a kind it does not branch on (the store's
#: "corrupt or unknown ⇒ tolerated" contract), so a log carrying these rows
#: stays readable everywhere.
EVENT_WITHDRAWN = "withdrawn"

#: Every status a folded ask can hold (design §2.2, §4).
STATUS_OPEN = "open"
STATUS_ANSWERED = "answered"
STATUS_DECLINED = "declined"
STATUS_TIMED_OUT = "timed_out"
STATUS_LATE = "late"
STATUS_DISMISSED = "dismissed"
STATUS_WITHDRAWN = "withdrawn"
STATUS_EXPIRED = "expired"

#: The statuses whose transcript row(s) are still injected; a `dismissed`,
#: `withdrawn` or `expired` ask injects NOTHING and must never attempt a
#: delivery (else ``reconcile`` loops forever on a row it can never write).
INJECTING_STATUSES = frozenset({STATUS_ANSWERED, STATUS_DECLINED, STATUS_TIMED_OUT, STATUS_LATE})

#: THE OUTSTANDING SET — the asks the user can still act on, and so the asks
#: every surface must agree are still live. ``open`` (nothing has happened to
#: it yet) and ``timed_out`` (its deadline fired, but a LATE answer is still
#: accepted and attributed — design §2.2, the spec's item 3: "a late answer is
#: still attributable and the agent still receives it"). It stops being
#: outstanding the moment it is SETTLED: ``answered``/``declined`` (the user
#: responded), ``dismissed`` (the user put a timed-out ask away), ``withdrawn``
#: (the asker retracted the question — design §12) and ``expired`` (an answer or
#: deadline past the 7-day window that injects nothing).
#:
#: ONE AUTHORITY, deliberately. Before this constant the same two-status rule
#: was spelled in four places — ``ask_wire``'s tally (open-only, which dropped a
#: timed-out-but-answerable ask from the count while it stayed on the bar),
#: ``AskQueue.projection``'s ordering, the TUI's ``_ANSWERABLE`` and the TUI
#: app's ``_open_ask_rows`` — and the one that was wrong was the count the
#: surfaces published. Anything that needs "is this ask still outstanding" reads
#: THIS set (or :func:`is_outstanding`); nothing re-lists the statuses.
OUTSTANDING_STATUSES = frozenset({STATUS_OPEN, STATUS_TIMED_OUT})

#: Row-id prefixes. The transcript row id IS the delivery marker (per
#: ``(ask_id, kind)``), so idempotence is structural rather than a boolean that
#: a crash can lose.
RESPONSE_ROW_PREFIX = "ask-response-"
TIMEOUT_ROW_PREFIX = "ask-timeout-"

#: How long past its deadline an ask can still be answered and ATTRIBUTED, and
#: how long a timeout notice or response is still worth injecting. Defined HERE
#: (the leaf module) because the FOLD needs it; ``asks/policy.py`` re-exports it
#: so callers have one import for the policy and this module keeps an empty
#: local-import closure (module docstring; pinned by test_import_graph).
LATE_WINDOW_S = 7 * 24 * 3600

_MILLIS = 1000


def now_ms() -> int:
    return int(time.time() * _MILLIS)


def new_ask_id(existing: Iterable[str] = ()) -> str:
    """A short, session-unique ask handle (``a-3f9c``).

    Session-unique rather than globally unique: the id names an ask within the
    session whose surfaces show it, and the transcript row that carries it lives
    in that session. Four hex characters over at most ``OPEN_ASK_CAP`` live asks
    makes a collision vanishingly unlikely, and the loop makes it impossible.
    """
    taken = {str(x) for x in existing}
    for _ in range(64):
        candidate = f"a-{secrets.token_hex(2)}"
        if candidate not in taken:
            return candidate
    raise RuntimeError("could not allocate a unique ask id")


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------


def session_dir(config_dir: Path | str, session_id: str) -> Path:
    return Path(config_dir) / SESSIONS_DIRNAME / session_id


def asks_log_path(session_directory: Path | str) -> Path:
    return Path(session_directory) / ASKS_LOG_NAME


_O_BINARY = getattr(os, "O_BINARY", 0)

#: Bounded retry when another writer holds the lock. Deliberately small: the
#: append below is ``O_APPEND`` on a line-sized write (atomic on every platform
#: we run on), so a contended writer proceeds unlocked rather than waiting.
_LOCK_ATTEMPTS = 20
_LOCK_RETRY_S = 0.005


class _NonBlockingLock:
    """``LOCK_EX | LOCK_NB`` with a bounded retry, or nothing at all.

    Mirrors ``session/runtime/inbox.py``'s helper deliberately rather than
    importing it: this module must stay stdlib-only (see the module docstring)
    and that one carries a ``local_operator.procstate`` import. The trade is
    the same one it documents — the lock protects the read-then-rewrite in a
    drain, and for a writer it is belt-and-braces over the atomic append.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self.acquired = False

    def __enter__(self) -> "_NonBlockingLock":
        if os.name == "nt":
            return self
        import fcntl

        for attempt in range(_LOCK_ATTEMPTS):
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.acquired = True
                return self
            except OSError:
                if attempt == _LOCK_ATTEMPTS - 1:
                    break
                time.sleep(_LOCK_RETRY_S)
        logger.debug("ask log lock contended; proceeding on the atomic append")
        return self

    def __exit__(self, *_exc: object) -> None:
        if not self.acquired or os.name == "nt":
            return
        import fcntl

        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass


def append_event(session_directory: Path | str, event: Mapping[str, Any]) -> bool:
    """Append one event row. True when it was written.

    ``O_APPEND`` with ONE ``write()`` per row, so concurrent writers interleave
    whole lines rather than overwriting each other at a shared offset.
    """
    directory = Path(session_directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("could not create %s for an ask event", directory, exc_info=True)
        return False
    payload = json.dumps(dict(event), separators=(",", ":"), sort_keys=True).encode() + b"\n"
    try:
        fd = os.open(
            asks_log_path(directory), os.O_CREAT | os.O_WRONLY | os.O_APPEND | _O_BINARY, 0o600
        )
    except OSError:
        logger.warning("could not open the ask log in %s", directory, exc_info=True)
        return False
    try:
        with _NonBlockingLock(fd):
            os.write(fd, payload)
        return True
    except OSError:
        logger.warning("ask-log append failed in %s", directory, exc_info=True)
        return False
    finally:
        os.close(fd)


def read_events(session_directory: Path | str) -> list[dict[str, Any]]:
    """Every readable event, in write order. A torn final line is skipped."""
    path = asks_log_path(session_directory)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return []
    except OSError:
        logger.warning("ask log unreadable at %s", path, exc_info=True)
        return []
    events: list[dict[str, Any]] = []
    for row in raw.splitlines():
        if not row.strip():
            continue
        try:
            payload = json.loads(row.decode("utf-8", "replace"))
        except ValueError:
            # A torn line (killed mid-write) is skipped, never fatal: this file
            # is read every minute beside a live writer.
            continue
        if isinstance(payload, dict) and payload.get("ask_id"):
            events.append(payload)
    return events


# ---------------------------------------------------------------------------
# The fold
# ---------------------------------------------------------------------------


def response_row_id(ask_id: str) -> str:
    return f"{RESPONSE_ROW_PREFIX}{ask_id}"


def timeout_row_id(ask_id: str) -> str:
    return f"{TIMEOUT_ROW_PREFIX}{ask_id}"


#: The separator in a MIRROR request id (design §4, "legacy mirror"): the
#: single-slot ``pending_gate``/``pending`` card carries
#: ``"<ask_id>.<qidx>"`` so an old client's per-question answer can be mapped
#: back onto the whole-ask queue. Kept here (the leaf module) because three
#: publishers build it and two answer routes parse it, and a second spelling of
#: the separator is how a mirrored answer would silently stop resolving.
MIRROR_REQUEST_SEPARATOR = "."


def mirror_request_id(ask_id: str, question_index: int = 0) -> str:
    """The legacy single-slot ``request_id`` for one question of a queued ask."""
    return f"{ask_id}{MIRROR_REQUEST_SEPARATOR}{int(question_index)}"


def parse_mirror_request_id(value: str) -> tuple[str, int] | None:
    """``(ask_id, question_index)`` for a mirrored request id, else ``None``.

    ``None`` is the ordinary answer: every request id that is NOT a mirrored
    question id is an approval or a live picker, and the callers fall through to
    the blocking path unchanged. The index is parsed permissively — a malformed
    tail is not a mirrored id rather than an error — because this runs on the
    answer path where the only two outcomes are "route to the queue" and
    "route to the gate".
    """
    raw = str(value or "")
    head, sep, tail = raw.rpartition(MIRROR_REQUEST_SEPARATOR)
    if not sep or not head or not tail:
        return None
    if not tail.isdigit():
        return None
    if not head.startswith("a-"):
        # Ask ids are ``a-<hex>`` (``new_ask_id``). Requiring the prefix keeps a
        # live picker's ``token_hex(8)`` request id — which is also hex digits —
        # from being misread as an ask.
        return None
    return head, int(tail)


def expected_row_ids(record: Mapping[str, Any]) -> list[str]:
    """The transcript row ids THIS status requires, in delivery order.

    ``timed_out`` needs the deadline row and nothing else: the notice is what the
    status means, and it is written in its own reconcile (there is no response to
    wait for). A ``late`` ask needs ONE row — the response — because a response row
    for the ask SUPERSEDES its deadline row for good (review round 1, MAJOR 2; the
    per-batch form of the suppression let the contradiction in one reconcile
    later, and §2.3's rule is that the timeout is suppressed for that ask, not
    postponed). That the two rows can both exist is a fact about the LOG, not about
    this status: an ask whose deadline fired while nobody was watching was written
    as ``timed_out`` first, and its answer then arrived as a second reconcile's
    response row — the lead on that response is what tells the model the window had
    closed.
    """
    ask_id = str(record.get("ask_id") or "")
    status = record.get("status")
    if not ask_id:
        return []
    if status in (STATUS_ANSWERED, STATUS_DECLINED, STATUS_LATE):
        return [response_row_id(ask_id)]
    if status == STATUS_TIMED_OUT:
        return [timeout_row_id(ask_id)]
    return []


def delivered_hint(ask_id: str, present_ids: Iterable[str], status: str) -> bool:
    """Whether the row(s) THIS status requires are DURABLE in the transcript.

    CONSUMPTION, not handoff (design §2.2/§4, amended 2026-10-04): a row is
    present here only once its durable append resolved, and the flag flips at
    that instant — handing a message to a delivery path, scheduling a turn or
    returning from the answering op closes nothing. What is required is
    per-status, straight off the §2.2 delivery table: ``answered``/``declined``/
    ``late`` need the RESPONSE row (for `late` the deadline notice delivers
    nothing — a timed-out ask answered late flips true→false until its row
    lands), ``timed_out`` needs the deadline row, ``dismissed``/``withdrawn``/
    ``expired`` are true when any row was written before, and ``open`` owes
    nothing.

    STICKY by construction otherwise: rows are never deleted from a transcript,
    so once the required row is durable this cannot flip back — the one
    sanctioned exception is the `timed_out`→`late` flip above.
    """
    present = set(present_ids)
    if status in (STATUS_ANSWERED, STATUS_DECLINED, STATUS_LATE):
        # The row that pins what the model was told. A LATE ask's deadline
        # notice is NOT the delivery of its answer; the response is.
        return response_row_id(ask_id) in present
    if status == STATUS_TIMED_OUT:
        return timeout_row_id(ask_id) in present
    if status == STATUS_OPEN:
        return False
    # dismissed / withdrawn / expired (and any status a future fold adds): any
    # row is the delivered fact — this is what keeps an answered ask from
    # flipping back when it folds to `expired` seven days later.
    return response_row_id(ask_id) in present or timeout_row_id(ask_id) in present


def _asks_in_order(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """Ask ids in the order their ``queued`` event was written (log seq)."""
    order: list[str] = []
    seen: set[str] = set()
    for event in events:
        ask_id = str(event.get("ask_id") or "")
        if not ask_id or ask_id in seen:
            continue
        seen.add(ask_id)
        order.append(ask_id)
    return order


def _first(events: Sequence[Mapping[str, Any]], ask_id: str, kind: str) -> dict[str, Any] | None:
    for event in events:
        if event.get("kind") == kind and str(event.get("ask_id")) == ask_id:
            return dict(event)
    return None


def _last(events: Sequence[Mapping[str, Any]], ask_id: str, kind: str) -> dict[str, Any] | None:
    """The LAST event of ``kind`` for ``ask_id`` (log order wins, as it does for
    every other rule here — see the ordering invariant stated at ``fold``).

    Successive revisions are all legal while the delivery window is open, so the
    effective answer is the newest ``revised`` row rather than any of the ones it
    supersedes; a later write cannot be beaten by an earlier reader.
    """
    found: dict[str, Any] | None = None
    for event in events:
        if event.get("kind") == kind and str(event.get("ask_id")) == ask_id:
            found = dict(event)
    return found


def fold(
    events: Sequence[Mapping[str, Any]],
    now: int,
    *,
    present_ids: Iterable[str] = (),
    session_id: str = "",
) -> list[dict[str, Any]]:
    """The ask records a log implies, ordered by log seq (design §2.2).

    Total by construction: the precedence table has one answer for every
    ``(events, now)``, so no two readers can disagree about an ask — and the
    answer-vs-deadline race is decided here rather than by which write landed
    first.

    ``withdrawn`` (design §12) is terminal-on-write from the moment it appears,
    with ONE carve-out: an ``answered`` row ALWAYS wins, in either write order.
    The asker retracts; the operator's real answer is never swallowed by that
    retraction — a racing withdrawal loses, and only a withdrawn row with NO
    ``answered`` sibling folds to ``withdrawn``. The check sits below every
    answered branch (and above the deadline branches, which must never judge a
    retracted ask).
    """
    records: list[dict[str, Any]] = []
    for ask_id in _asks_in_order(events):
        queued = _first(events, ask_id, EVENT_QUEUED)
        if queued is None:
            # An answer for an ask whose `queued` row is missing (a truncated
            # log) is not reconstructible: skip rather than invent a question.
            continue
        answered = _first(events, ask_id, EVENT_ANSWERED)
        declined = _first(events, ask_id, EVENT_DECLINED)
        dismissed = _first(events, ask_id, EVENT_DISMISSED)
        withdrawn = _first(events, ask_id, EVENT_WITHDRAWN)
        revised = _last(events, ask_id, EVENT_REVISED)
        expires_at = int(queued.get("expires_at") or 0)
        created_at = int(queued.get("at") or 0)
        answered_at = int(answered.get("at") or 0) if answered else 0
        window_end = expires_at + LATE_WINDOW_S * _MILLIS

        status: str
        if answered and answered_at <= expires_at:
            # Rule 1 sits ABOVE the view actions on purpose: a dismissal is an
            # action on an ask that already timed out, and it must never swallow
            # an answer that arrived inside the window (design §2.2).
            status = STATUS_ANSWERED
        elif declined is not None:
            status = STATUS_DECLINED
        elif dismissed is not None:
            status = STATUS_DISMISSED
        elif answered is not None:
            status = STATUS_LATE if now <= window_end else STATUS_EXPIRED
        elif withdrawn is not None:
            # THE ONE CARVE-OUT IS ABOVE, BY CONSTRUCTION (design §12): every
            # answered branch outranks this one, so a withdrawn row folds to
            # ``withdrawn`` only when no ``answered`` row exists — in either
            # write order. Below it, the deadline branches never run for a
            # retracted ask: ``withdrawn`` is terminal-on-write.
            status = STATUS_WITHDRAWN
        elif expires_at <= now <= window_end:
            status = STATUS_TIMED_OUT
        elif now > window_end:
            status = STATUS_EXPIRED
        else:
            status = STATUS_OPEN

        record: dict[str, Any] = {
            "ask_id": ask_id,
            "session_id": session_id,
            "created_at": created_at,
            "expires_at": expires_at,
            "timeout_s": int(queued.get("timeout_s") or 0),
            "urgent": bool(queued.get("urgent")),
            "tool_call_id": str(queued.get("tool_call_id") or ""),
            "questions": [
                dict(q) for q in (queued.get("questions") or ()) if isinstance(q, Mapping)
            ],
            "status": status,
            "delivered": delivered_hint(ask_id, present_ids, status),
        }
        if answered is not None:
            answers = answered.get("answers")
            record["answers"] = {
                str(k): [str(v) for v in (vals or ())]
                for k, vals in (answers.items() if isinstance(answers, Mapping) else ())
            }
            by = answered.get("by")
            if isinstance(by, Mapping):
                record["answered_by"] = dict(by)
            record["answered_at"] = answered_at
            if revised is not None:
                # The status, the stamp and the attribution stay the FIRST
                # answer's (design §10: a revision does not rewrite who answered
                # or when); only the ANSWERS move, and only to the newest
                # revision. ``revised_at`` is published beside them so the
                # revision path can name the write it supersedes without a
                # second read of the log.
                record["revised_at"] = int(revised.get("at") or 0)
                replacement = revised.get("answers")
                if isinstance(replacement, Mapping):
                    record["answers"] = {
                        str(k): [str(v) for v in (vals or ())] for k, vals in replacement.items()
                    }
        records.append(record)
    return records


def is_outstanding(status: Any) -> bool:
    """Whether a folded ask status still wants the user (see
    :data:`OUTSTANDING_STATUSES`).

    Takes the status VALUE rather than a record, because its callers read it off
    a wire row, whose shape is a dict on one surface and a model on another; the
    one thing they share is the status string. A missing/unknown status folds to
    settled — an ask nobody can name is not one a surface may ask the user to
    answer.
    """
    return str(status or "") in OUTSTANDING_STATUSES


def outstanding_asks(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Records the user can still act on: ``open`` or ``timed_out``.

    The row-list half of :func:`is_outstanding`; a surface that draws "asks
    waiting on you" (the TUI's bar/answerable set, a sidebar mark) uses this
    rather than re-listing the statuses.
    """
    return [dict(r) for r in records if is_outstanding(r.get("status"))]


def open_asks(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Records still expecting an answer (``open`` only — a timed-out ask is
    past its deadline and no longer counts against the open cap).

    NARROWER THAN :func:`outstanding_asks` ON PURPOSE, and the difference is the
    CAP's, not a surface's: ``OPEN_ASK_CAP`` bounds how many questions the agent
    may have in flight, and a timed-out ask is one the agent has already walked
    past — it must not keep a slot that a fresh question needs. A display that
    counted this instead of the outstanding set would drop a timed-out ask the
    user can still answer, which is the defect this module's constant exists to
    prevent.
    """
    return [dict(r) for r in records if r.get("status") == STATUS_OPEN]


def pending_asks(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Records whose timeout notice is still OWED.

    Delegates to :func:`outstanding_asks`, and the two coincide BY
    CONSTRUCTION rather than by accident: an ask owes its timeout notice exactly
    while it is outstanding — ``open`` (the notice is still ahead of it) or
    ``timed_out`` (the notice is owed now); a ``late`` ask's notice is
    suppressed (its response replaces it) and a settled one has none. If a
    future status ever makes the two diverge, SPLIT them here and say why rather
    than letting a second spelling of either rule appear.
    """
    return outstanding_asks(records)


def ask_ids(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """Ask ids in the order their ``queued`` event was written (log seq).

    Public, and used by the queue to look up exactly two row ids per ask with
    ``transcript.has_entry`` rather than scanning every transcript entry to
    build a membership set (see ``AskQueue.present_row_ids``).
    """
    return _asks_in_order(events)


def pending_row(record: Mapping[str, Any], draft: Iterable[str] | None = None) -> dict[str, Any]:
    """One folded record as the wire's ``PendingAsk`` (design §4, frozen).

    Secret answers hold the KEY ONLY (``[<key>]``) because they are stored that
    way in the log — this function copies, it never resolves. ``delivered`` is
    carried rather than derived so a surface reads one field instead of
    re-implementing the sticky rule.
    """
    row: dict[str, Any] = {
        "ask_id": str(record.get("ask_id") or ""),
        "created_at": int(record.get("created_at") or 0),
        "expires_at": int(record.get("expires_at") or 0),
        "timeout_s": int(record.get("timeout_s") or 0),
        "urgent": bool(record.get("urgent")),
        "status": str(record.get("status") or STATUS_OPEN),
        "delivered": bool(record.get("delivered")),
        "questions": [dict(q) for q in (record.get("questions") or ())],
    }
    answers = record.get("answers")
    if answers:
        row["answers"] = {str(k): list(v) for k, v in dict(answers).items()}
    by = record.get("answered_by")
    if by:
        row["answered_by"] = dict(by)
    answered_at = record.get("answered_at")
    if answered_at:
        row["answered_at"] = int(answered_at)
    drafted = sorted(str(qid) for qid in (draft or ()))
    if drafted:
        # THE LEGACY DRAFT (design §4, A2 addendum). The old single-slot client
        # answers a queued ask ONE QUESTION AT A TIME, so between its taps the
        # question already answered exists nowhere durable — the log is written
        # atomically, on the last question. It is published BESIDE ``answers``
        # rather than inside it on purpose: ``answers`` states what the log
        # holds, and a client that rendered a draft as a settled answer would be
        # showing an answer a runtime death would erase. Only the mirror needs
        # it, and only to know which question to put on the card next.
        row["draft_question_ids"] = drafted
    return row


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------


def asks_dir(config_dir: Path | str) -> Path:
    return Path(config_dir) / ASKS_DIRNAME


def entry_path(config_dir: Path | str, session_id: str) -> Path:
    return asks_dir(config_dir) / f"{session_id}.json"


def _entry_asks(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = entry.get("asks")
    return [dict(a) for a in raw if isinstance(a, Mapping)] if isinstance(raw, list) else []


def entry_is_stale(entry: Mapping[str, Any], now: int, *, session_exists: bool) -> bool:
    """Whether the index entry has nothing left to say and should be swept.

    Two ways to be stale, both about an ORPHAN the session-level guards cannot
    reach (the file lives outside the session directory — module docstring):

    * the session directory is gone, so nothing can ever answer the asks; or
    * no ask on the entry is still worth showing — every one is either settled
      or past ``expires_at + 7 d``, i.e. the fold would report it ``expired``.

    An ``open`` ask is kept however old it is: it has not reached its deadline,
    so the user can still answer it whenever they come back, and sweeping it
    would be the queue forgetting a question nobody has seen. A terminal ask is
    kept only while within the same 7-day window the delivery path uses, which
    is what makes the index's contents and the fold's statuses agree.
    """
    asks = _entry_asks(entry)
    if not asks:
        return True
    if not session_exists:
        # Nothing can answer them any more: the session directory is reaped and
        # cannot be reopened, and the log this entry projects lives with it. The
        # note's own unit test is exactly this (delete the session directory by
        # hand, reopen the index → the entry is gone).
        return True
    horizon = LATE_WINDOW_S * _MILLIS
    for ask in asks:
        if ask.get("status") == STATUS_OPEN:
            return False
        expires_at = int(ask.get("expires_at") or 0)
        if expires_at and now - expires_at <= horizon:
            return False
    return True


def read_entry(config_dir: Path | str, session_id: str) -> dict[str, Any] | None:
    """One session's entry, or ``None`` when absent or unreadable.

    Unreadable is treated exactly like absent — the log is the truth and the
    next reconcile rewrites the file — for the reason ``wakes/store.py`` gives:
    a scan over every session must not die on one bad entry.
    """
    path = entry_path(config_dir, session_id)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("ask index: unreadable entry %s; treating as absent", path, exc_info=True)
        return None
    if not isinstance(data, dict) or data.get("schema") != INDEX_SCHEMA:
        logger.debug("ask index: skipping entry %s with unknown schema", path)
        return None
    return data


def read_index(config_dir: Path | str, *, now: int | None = None) -> dict[str, dict[str, Any]]:
    """Every readable entry keyed by session id, with the TTL sweep applied.

    A missing directory is an empty index — the ordinary state of a machine
    that has no queued asks. The sweep (see :func:`entry_is_stale`) unlinks an
    orphaned entry best-effort; a failure to unlink is not reported, because
    the next read sweeps again and the file is derived in any case.
    """
    root = Path(config_dir)
    directory = asks_dir(root)
    stamp = now if now is not None else now_ms()
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return {}
    except OSError:
        logger.warning("ask index: cannot list %s", directory, exc_info=True)
        return {}
    index: dict[str, dict[str, Any]] = {}
    for name in names:
        if not name.endswith(".json") or name.startswith("."):
            continue  # staged temp files and stray dotfiles are never entries
        session_id = name[: -len(".json")]
        entry = read_entry(root, session_id)
        if entry is None:
            continue
        exists = session_dir(root, session_id).is_dir()
        if entry_is_stale(entry, stamp, session_exists=exists):
            logger.debug("ask index: sweeping stale entry %s", name)
            try:
                (directory / name).unlink()
            except OSError:
                pass
            continue
        # The filename is the lookup key; a body whose ``session_id`` disagrees
        # is a copied or hand-edited file, and the filename wins because that is
        # what the owning session will overwrite.
        entry["session_id"] = session_id
        index[session_id] = entry
    return index


def index_asks(config_dir: Path | str, *, now: int | None = None) -> list[dict[str, Any]]:
    """Every ask worth showing across sessions, newest-first within each status.

    The AGGREGATE view (design §4: ``GET /api/asks`` on the relay and
    ``GET /v1/desktop/asks`` on the desktop plane) — one read of the derived
    index, no runtime and no session directory walk, which is the whole reason
    the index lives outside the session dir. Each row is the frozen
    ``PendingAsk`` shape plus the two facts a row for ANOTHER session needs:
    ``session_id`` and ``cwd``.

    The same horizon rule as ``AskQueue.projection`` and the index's own
    staleness sweep is applied HERE too, so a reader of this function cannot
    disagree with the writer about whether a week-old settled ask is still worth
    showing: one horizon, three readers.
    """
    stamp = now if now is not None else now_ms()
    horizon_ms = LATE_WINDOW_S * _MILLIS
    #: NO CROSS-SESSION CAP, deliberately (review round 1, NIT 9). The population
    #: is (sessions with ask activity) × (rows the fold already caps at
    #: ``PROJECTION_CAP``), and the reader is an on-demand HTTP route rather than
    #: a per-frame path, so the honest bound today is "however many conversations
    #: have questions waiting" — a number the user can see and act on. It WOULD
    #: need one before it ever feeds a frame: if a surface starts polling this
    #: into a push, add the cap here (newest first, open-first) rather than in
    #: each caller.
    rows: list[dict[str, Any]] = []
    for session_id, entry in (read_index(config_dir, now=stamp) or {}).items():
        for ask in _entry_asks(entry):
            status = str(ask.get("status") or "")
            if status in (STATUS_EXPIRED, STATUS_DISMISSED):
                continue
            expires_at = int(ask.get("expires_at") or 0)
            if expires_at and stamp - expires_at > horizon_ms:
                continue
            row = dict(ask)
            row["session_id"] = session_id
            row["cwd"] = str(entry.get("cwd") or "")
            rows.append(row)
    rows.sort(key=lambda row: (row.get("status") != STATUS_OPEN, -int(row.get("created_at") or 0)))
    return rows


def write_entry(
    config_dir: Path | str,
    session_id: str,
    *,
    cwd: str,
    asks: Sequence[Mapping[str, Any]],
) -> Path | None:
    """Write (replace) one session's entry. ``None`` when there is nothing to
    show — so "no file" and "no asks" stay the same statement.

    Staged write + ``os.replace`` so a reader never sees a torn file: the
    aggregate view may scan while a session writes. The temp name starts with
    ``.`` so the directory scan above skips it.
    """
    path = entry_path(config_dir, session_id)
    if not asks:
        remove_entry(config_dir, session_id)
        return None
    entry = {
        "schema": INDEX_SCHEMA,
        "session_id": session_id,
        "cwd": cwd,
        "updated_at": now_ms(),
        "asks": [dict(a) for a in asks],
    }
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{session_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(entry, handle, separators=(",", ":"), sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def remove_entry(config_dir: Path | str, session_id: str) -> bool:
    """Delete one session's entry. Idempotent: absent is success."""
    path = entry_path(config_dir, session_id)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        logger.debug("ask index: could not remove %s", path, exc_info=True)
        return False
    return True


__all__ = [
    "ASKS_DIRNAME",
    "ASKS_LOG_NAME",
    "EVENT_ANSWERED",
    "EVENT_DECLINED",
    "EVENT_DISMISSED",
    "EVENT_QUEUED",
    "INJECTING_STATUSES",
    "LATE_WINDOW_S",
    "OUTSTANDING_STATUSES",
    "STATUS_ANSWERED",
    "STATUS_DECLINED",
    "STATUS_DISMISSED",
    "STATUS_EXPIRED",
    "STATUS_LATE",
    "STATUS_OPEN",
    "STATUS_TIMED_OUT",
    "append_event",
    "ask_ids",
    "asks_dir",
    "asks_log_path",
    "delivered_hint",
    "entry_is_stale",
    "entry_path",
    "expected_row_ids",
    "fold",
    "is_outstanding",
    "new_ask_id",
    "now_ms",
    "open_asks",
    "outstanding_asks",
    "pending_asks",
    "pending_row",
    "read_entry",
    "read_events",
    "read_index",
    "remove_entry",
    "response_row_id",
    "session_dir",
    "timeout_row_id",
    "write_entry",
]
