"""The cold-session inbox: messages that arrive when nothing is running.

A quiet peer note (``lop send --no-wake``, the ``send`` tool with
``wake=False``) to a session with no runtime has nowhere to go. Spawning a
whole runtime to receive it would contradict what "quiet" means — the point of
``wake=False`` is that the peer reads it on its NEXT turn, not that it starts
one — and dropping it would lose the message. So it is appended here, and the
runtime drains it the moment one exists.

**The ordering guarantee, and where it comes from.** ``process.py`` drains this
file after the session is constructed and BEFORE ``RuntimeServer`` begins
listening. That ordering *is* the guarantee: rows spooled while the session was
cold are delivered ahead of anything a socket client could send, because no
socket client can send anything yet. The alternative — draining after the
server starts — would race an errand that arrived in the same instant and
deliver messages out of the order they were written.

**Never a blocking ``flock``.** Every lock here is ``LOCK_NB`` with a bounded
retry. This is the #401 class: a blocking ``flock`` in the MCP OAuth refresh
lock deadlocked the Textual event loop, and on macOS/BSD a sibling ``flock``
even makes ``close()`` block. The appending side of this file runs from a
short-lived sender process, but ``peek_inbox`` is read by the VIEWER while it
paints, so a blocking lock here would freeze a terminal because an unrelated
process was mid-append. Contention is genuinely rare (two peers writing to the
same cold session in the same millisecond), and the correct response to losing
the race is to retry briefly and then give up — never to wait indefinitely.

**Not a sidecar.** ``inbox.jsonl`` is deliberately absent from
``retention._SIDECAR_NAMES``, so a directory holding one reads as having
CONTENT and the junk reap will not delete it. A spooled message the user has
not seen is exactly the thing that must survive a sweep.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The spool file inside ``sessions/<id>/``.
INBOX_NAME = "inbox.jsonl"

#: The two sender-facing receipts for a spooled message, ONE definition each for
#: the two writers (``serving._spool_for_successor`` for a draining runtime, and
#: ``peer_send._spool_quiet_note`` for a cold session). They led with the
#: mechanism verb ("spooled …") in both, which told the sender about our
#: plumbing before telling them what they had bought; the effect leads now and
#: the clause after the dash — a wake WILL be run by the next runtime, a quiet
#: note is only read — is the part they can act on (design round 1, D4).
#:
#: KEPT SHORT ON PURPOSE, and this is the constraint a future edit must respect:
#: design round 1 measured the receipts they replace at 54 and 50 characters and
#: verified each still fits ONE line at both 60 and 100 columns. The rewrite
#: above is 38 and 51 — inside that measured budget — because a longer receipt
#: buys precision the sender does not need at the cost of the one property that
#: was checked. A frame is not the place to discover a wrap.
#:
#: WHICH CONSUMER HONOURS A SPOOLED WAKE, since the string promises a turn
#: (QA round 3, Q9): the BOOT drain (``process.amain``, idle session, before the
#: socket listens) drives the turn this string promises, and that is the only
#: shape the send path can produce — ``deliver_peer_message`` spools only when
#: ``not wake and mode == "mailbox"``, so a wake row comes only from a successor
#: handover (``serving._spool_for_successor``, the one writer holding the
#: sender's ``wake``); a row written before the field existed parses as a
#: QUIET note (``InboxLine.from_json`` reads an absent ``wake`` as False), which
#: is the NOTE receipt, not this one (review round 4, NIT-2). The FIRST-TURN
#: drain reaches the receiver
#: mid-turn instead, where the row rides that turn's context (see the paragraph
#: at its call site in ``session._run_turn_pipeline``); the string is left as the
#: boot drain's promise rather than stretched to describe both, because the
#: over-claim needs a row no send can write.
SPOOL_RECEIPT_WAKE = "held for the next runtime — it runs it"
SPOOL_RECEIPT_NOTE = "held for the next runtime — read when it next opens"

#: The receipt a DRAINING runtime hands back for the OWNER's own prompt, which
#: is the third spooling writer and a different situation from both of the
#: above: nobody sent this to the session, its author typed it, and the session
#: they typed it into has already refused its turn.
#:
#: IT MUST NOT READ AS ADMITTED, and that is why it is not the wake receipt one
#: line up. ``prompt``'s normal ACK is the DURABLE TRANSCRIPT APPEND
#: (``serving.ServingSessionHandle.prompt``), so any other string on that op is
#: a weaker fact and has to be visibly weaker: the message has not been written
#: to this session's history, it has been written to the successor's spool. The
#: composer's own claim ("your message is back in the composer") is the
#: fallback's wording, not this one, because here the message is NOT coming
#: back: the successor runs it.
#:
#: Same short budget as its siblings for the same measured reason: a client
#: frame is not the place to discover a wrap.
SPOOL_RECEIPT_PROMPT = "queued for the next runtime — it will run it"

#: Which of the two PRODUCERS a row belongs to, carried on the row because the
#: successor delivers them differently and cannot guess which it holds.
#:
#: ``SOURCE_PEER`` — another local session's message (``send``, a scheduled
#: wake, a broadcast). It is delivered into the successor as a PEER message, with
#: the sender's provenance wrapped around it for the model, exactly as a live
#: dial would have delivered it.
#:
#: ``SOURCE_USER`` — the session OWNER's own prompt, spooled by a draining
#: runtime instead of refused. It must NOT be delivered as a peer message: the
#: model would read a foreign-session envelope over the user's own words and the
#: transcript would paint a ``peer`` card labelled 'another session' for a
#: message the user typed in this very session (``session.receive_peer_message``
#: builds that envelope from the sender dict). Its delivery is the ordinary
#: admission it would have had, run on the successor's build — see
#: ``process._drain_inbox_into`` and ``Session._drain_spooled_peer_inbox``.
#:
#: Absent in rows written before this field existed, which reads as
#: ``SOURCE_PEER``: every writer that predates it is the peer path.
SOURCE_PEER = "peer"
SOURCE_USER = "user"
#: NOT a message: the owner took a message back before it ran. It carries the
#: recalled row's ``command_id`` and nothing else, and both readers drop the row
#: it names (see :func:`withdraw_inbox`). A MARKER rather than a rewrite of the
#: spool, because this file is append-only by contract and every reader of it
#: must be able to see the recall without the file being rewritten under a
#: concurrent writer — see the measurement in :func:`withdraw_inbox`'s docstring.
SOURCE_RECALL = "recall"

#: Non-blocking lock retries, and the pause between them. Deliberately small:
#: the critical section is one ``write()`` of a few hundred bytes, so a
#: contender that cannot get in within ~50 ms is not merely slow, and waiting
#: longer trades a caller's responsiveness for a case that barely happens.
_LOCK_ATTEMPTS = 10
_LOCK_RETRY_S = 0.005

#: Refuse to spool past this many rows for one session. A cold session that
#: something is hammering must not grow an unbounded file that then has to be
#: replayed into a transcript in one go. Well above any legitimate use (a
#: handful of notes between sessions) and far below "this file is a problem".
MAX_INBOX_ROWS = 500


@dataclass(frozen=True, slots=True)
class InboxLine:
    """One spooled message, in the order it was written.

    ``wake`` is what the sender ASKED FOR, carried across the handover rather
    than decided by the reader. A row spooled because the receiving runtime was
    leaving a replaced build may have been a wake — a scheduled alarm the
    session owes a turn for, or a peer ``send --wake`` — and a successor that
    delivered it as a quiet note would keep the reminder and never do the work
    (review round 1, MINOR 3). Absent in rows written before this field
    existed, which reads as the old quiet-note behaviour: False.

    ``source`` says WHO is speaking in the row, which is what decides how the
    successor delivers it — see ``SOURCE_PEER``/``SOURCE_USER``. It is not a
    second ``mode``: ``mode`` is the peer sender's stated intent (and is
    deliberately not honoured on delivery), while ``source`` is the receiving
    side's own question about provenance.

    ``command_id`` is the producer identity the owner's prompt was admitted
    under, and it only ever rides a ``SOURCE_USER`` row. It exists so the
    successor's delivery is the SAME admission the predecessor refused rather
    than a second one: the transcript's append-only index
    (``transcript.has_admitted_command``) then answers a retried or
    twice-spooled row without appending it twice, and the viewer that painted a
    row for that id has its announcement matched rather than duplicated.
    """

    text: str
    sender: dict[str, Any]
    mode: str = "mailbox"
    written_at: float = 0.0
    wake: bool = False
    source: str = SOURCE_PEER
    command_id: str = ""

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "InboxLine":
        sender = payload.get("sender")
        return cls(
            text=str(payload.get("text", "") or ""),
            sender=dict(sender) if isinstance(sender, dict) else {},
            mode=str(payload.get("mode", "mailbox") or "mailbox"),
            written_at=float(payload.get("written_at", 0.0) or 0.0),
            wake=bool(payload.get("wake", False)),
            # Anything that is not one of the values THIS build knows is a
            # peer's: the peer path is the one that predates the field, so an
            # unknown value (a build this one has never heard of) must not
            # inherit the one delivery shape that skips the sender's provenance.
            # ``SOURCE_RECALL`` is one of ours and is preserved as itself —
            # collapsing it to a peer row would deliver the recall marker as an
            # empty peer message and, worse, hide the recall from both readers.
            #
            # (An OLDER runtime reading a spool that holds a marker does read it
            # as a peer row and delivers an empty note: a downgrade-only case,
            # not a shape this change creates, and the next drain consumes it.)
            source=_known_source(payload.get("source")),
            command_id=str(payload.get("command_id", "") or ""),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "sender": self.sender,
            "mode": self.mode,
            "written_at": self.written_at,
            "wake": self.wake,
            "source": self.source,
            "command_id": self.command_id,
        }


#: Every ``source`` value this build writes, so a reader can tell one of ours
#: from a value a NEWER build invented (which is read as a peer's row).
_KNOWN_SOURCES = frozenset({SOURCE_PEER, SOURCE_USER, SOURCE_RECALL})


def _known_source(raw: Any) -> str:
    """The row's own source when this build knows it, else the peer default."""
    return raw if isinstance(raw, str) and raw in _KNOWN_SOURCES else SOURCE_PEER


def inbox_path(session_dir: Path) -> Path:
    return session_dir / INBOX_NAME


class _NonBlockingLock:
    """``LOCK_EX | LOCK_NB`` with a bounded retry, or nothing at all.

    Failing to acquire is NOT an error: the append below is ``O_APPEND`` on a
    line-sized write, which the kernel already keeps atomic on every platform
    this runs on. The lock is what protects the read-then-rewrite in
    :func:`drain_inbox`, and for the writer it is belt-and-braces. So a
    contended writer proceeds unlocked rather than blocking a caller — the
    opposite trade from a correctness lock, and deliberate.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self.acquired = False

    def __enter__(self) -> "_NonBlockingLock":
        if os.name == "nt":
            # No fcntl on Windows, and msvcrt.locking's non-blocking mode
            # raises rather than waiting. The atomic O_APPEND write stands on
            # its own there.
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
        logger.debug("inbox lock contended; proceeding on the atomic append")
        return self

    def __exit__(self, *_exc: object) -> None:
        if not self.acquired or os.name == "nt":
            return
        import fcntl

        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass


def append_inbox(session_dir: Path, line: InboxLine) -> bool:
    """Spool one message for a session that is not running. True if written.

    ``O_APPEND`` so concurrent writers interleave whole lines rather than
    overwriting each other at a shared offset, and one ``write()`` per row so
    the atomicity that guarantees applies to the entire line.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    path = inbox_path(session_dir)
    payload = json.dumps(line.to_json(), separators=(",", ":")).encode() + b"\n"
    try:
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    except OSError:
        logger.warning("could not open inbox for %s", session_dir.name, exc_info=True)
        return False
    try:
        with _NonBlockingLock(fd):
            # Checked under the lock when we hold it: the bound exists to stop
            # a runaway producer, and an unlocked contender overshooting it by
            # a row or two is harmless.
            if _count_rows(path) >= MAX_INBOX_ROWS:
                logger.warning(
                    "inbox for %s is at its %d-row cap; message dropped",
                    session_dir.name,
                    MAX_INBOX_ROWS,
                )
                return False
            os.write(fd, payload)
        return True
    except OSError:
        logger.warning("inbox append failed for %s", session_dir.name, exc_info=True)
        return False
    finally:
        os.close(fd)


def _count_rows(path: Path) -> int:
    try:
        with open(path, "rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _parse(raw: bytes) -> list[InboxLine]:
    lines: list[InboxLine] = []
    for row in raw.splitlines():
        if not row.strip():
            continue
        try:
            payload = json.loads(row.decode("utf-8", "replace"))
        except ValueError:
            continue  # a torn line is skipped, never fatal
        if isinstance(payload, dict):
            lines.append(InboxLine.from_json(payload))
    return lines


def peek_inbox(session_dir: Path) -> list[InboxLine]:
    """Read the spool WITHOUT consuming it — the cold viewer's read.

    A viewer showing a session that is not running renders these as pending
    messages; the runtime is what actually delivers them. Deliberately
    lock-free: a reader that saw a half-written final line simply drops it
    (``_parse`` skips unparseable rows), which is cheaper and safer than
    taking a lock on a UI path.

    RECALLED ROWS ARE NOT IN THE ANSWER, and neither are the markers that recall
    them: the spool still holds both until the next drain consumes the file, so
    a reader that showed them would paint a message the user has taken back.
    """
    try:
        return _deliverable(_parse(inbox_path(session_dir).read_bytes()))
    except OSError:
        return []


def _deliverable(lines: list[InboxLine]) -> list[InboxLine]:
    """The rows of a spool batch that are FOR DELIVERY.

    One spelling for both readers, so a recall cannot be honoured by the runtime
    and ignored by the viewer (or the reverse). ``SOURCE_RECALL`` rows are the
    markers themselves — they say what to drop and are dropped with it.
    """
    recalled = {
        line.command_id for line in lines if line.source == SOURCE_RECALL and line.command_id
    }
    return [
        line
        for line in lines
        if line.source != SOURCE_RECALL
        and not (line.source == SOURCE_USER and line.command_id in recalled)
    ]


def _holds_owner_row(raw: bytes, command_id: str) -> bool:
    """Does this batch still hold the owner row carrying ``command_id``?

    The row a recall addresses is only ever removed by a drain, so "no longer
    here" means "the successor's batch has it" — the one fact the recall's answer
    turns on.
    """
    return any(line.source == SOURCE_USER and line.command_id == command_id for line in _parse(raw))


def withdraw_inbox(session_dir: Path, command_id: str) -> bool:
    """Record that the owner took a spooled message back. True if recorded.

    APPEND-ONLY, and that is a measured decision rather than a style one. The
    obvious implementation — rewrite the spool without that row — has to replace
    or truncate a file a concurrent ``append_inbox`` may be writing to, and this
    module's appenders never block for a lock (deliberately: a producer's message
    must not be dropped for one). Measured with three racing appenders and 48
    recalls: an in-place truncate destroyed acked rows in the reviewer's runs
    (1/241, 8/488; 0/300 and 0/180 in later ones, so that comparison is not a
    controlled A/B — what IS established is that the staged shape destroys acked
    rows: 10/27, 10/45, 14/42 in mine, because an appender's already-open
    ``O_APPEND`` descriptor points at the inode the replace orphans), while this
    marker lost NONE of 99, 181 and 354 acked rows. So the recall adds a MARKER
    row (``SOURCE_RECALL``) and rewrites nothing: no appender can lose a row, and
    both readers drop the marker and the row it names.

    ONE CRITICAL SECTION, WHICH IS WHAT MAKES THE ANSWER TRUE. The peek, the
    append and a verify all happen under the same non-blocking lock the drain
    takes for its read-and-consume, and the drain decides what to deliver from a
    read taken under that lock:

      * this call gets the lock first — the marker is in the file before the
        drain's decision, so the row is withheld and the receipt is true;
      * the drain gets it first — its batch is already gone by the time the
        append lands, and the VERIFY below sees that and answers ``False``, so
        the user is told the message will run rather than that it was recalled.

    The verify is what retires the lie QA reproduced at 4/120 trials: the marker
    alone cannot tell the difference, because a marker appended after the drain's
    read is a marker the drain never sees.

    THE RESIDUE IS MEASURED AND ITS SHAPE IS CONTENTION, NOT "A THIRD HOLDER".
    ``_NonBlockingLock`` gives up after roughly 50 ms (ten 5 ms attempts) and
    proceeds UNLOCKED — that is the file's deliberate discipline, not a corner —
    so a recall that loses the lock appends and verifies while a drain may hold
    the batch it read and not yet consumed; the verify then re-reads the same
    not-yet-taken bytes and answers True. Measured thresholds, by the round that
    filed it: 0 lies in 40 trials at ≤50 ms of contention, 1 in 12 at 60 ms, 12
    in 12 at 120 ms (QA); 4 in 1,600 stalled trials (UX); deterministic with a
    400 ms post-decision tail (reviewer). So the honest statement is: correct at
    the lock's own retry budget, degrading once a holder outlives it, and NOT
    the "a drain always gets there first" this docstring used to imply — the
    sentence mattered because it read like a closure. Closing it properly means
    either a blocking lock or a shared/exclusive discipline for appenders, which
    is a change to this file's write model rather than to the recall.

    ``False`` means the row was not in the spool when this was called, and the
    overwhelmingly likely reason is that the successor drained it — the message
    WILL run, and the caller must say that rather than let the user believe a
    recall worked.

    Keyed by the OWNER's own ``command_id``, which only a ``SOURCE_USER`` row
    carries (``serving._spool_for_successor`` writes it for that source alone),
    so a peer's message can never be recalled by this call even if ids were to
    collide.

    THE MARKER IS NOT CAPPED, unlike a message: ``append_inbox`` drops a row
    beyond ``MAX_INBOX_ROWS`` because the bound exists to stop a runaway
    PRODUCER, and a recall is a user action bounded by the UI — one row per
    press, gone with the next drain. Capping it instead would make the spool
    full answer ``False`` for a message that is still sitting in it, i.e. the
    recall would say "the next runtime already has that message" while nothing
    has it (agent review round 3, NIT-3).
    """
    if not command_id:
        return False
    path = inbox_path(session_dir)
    try:
        fd = os.open(path, os.O_RDWR | os.O_APPEND)
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning("could not open inbox for %s", session_dir.name, exc_info=True)
        return False
    try:
        with _NonBlockingLock(fd):
            raw = _read_all(fd)
            if not _holds_owner_row(raw, command_id):
                return False
            marker = InboxLine(text="", sender={}, source=SOURCE_RECALL, command_id=command_id)
            payload = json.dumps(marker.to_json(), separators=(",", ":")).encode() + b"\n"
            os.write(fd, payload)
            if not _holds_owner_row(_read_all(fd), command_id):
                logger.info(
                    "inbox recall for %s lost the race with a drain; the message will run",
                    command_id,
                )
                return False
            return True
    except OSError:
        logger.warning("inbox withdrawal failed for %s", session_dir.name, exc_info=True)
        return False
    finally:
        os.close(fd)


def _replace_remainder(path: Path, consumed: bytes) -> None:
    """Rewrite the spool with only the bytes written after ``consumed``.

    Staged through ``os.replace`` so a reader never sees a truncated file, and it
    is the ONLY place this module replaces the file: the recall does not rewrite
    the spool at all (see :func:`withdraw_inbox`).
    """
    staged = path.with_name(f"{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        current = path.read_bytes()
    except OSError:
        current = consumed
    remainder = current[len(consumed) :] if current.startswith(consumed) else b""
    try:
        staged.write_bytes(remainder)
        os.chmod(staged, 0o600)
        os.replace(staged, path)
    except OSError:
        logger.warning("inbox rewrite failed for %s", path.parent.name, exc_info=True)
        try:
            staged.unlink()
        except OSError:
            pass


def drain_inbox(session_dir: Path) -> list[InboxLine]:
    """Consume every spooled message, in write order. Called once, at open.

    Read and removal happen under one non-blocking lock so a concurrent
    appender cannot have its row consumed-but-not-delivered. Rows that arrive
    while the caller is delivering usually survive — the file is emptied here
    and a later append creates it again — but "NOT lost" is stronger than the
    mechanism holds, and QA measured the gap: with a FORCED window (a writer
    held inside the read-then-consume interval), 8 of 48 acked rows were lost on
    both this head and `4177803f0`, i.e. the claim was already false before this
    PR and is not repaired by it (PR #1319, QA round 4). It is recorded rather
    than fixed because closing it means changing the appender's deliberate
    proceed-unlocked discipline, which is a change of its own with its own
    review. The same shape is what `_replace_remainder`'s "the NEXT open drains"
    note below assumes; an unlocked appender's already-open descriptor points at
    the inode a replace orphans, so that row is gone rather than deferred.

    **The crash contract is at-least-once, not exactly-once.** The file is
    truncated after it is read, so a runtime killed between the read and its
    ``receive_peer_message`` calls loses the messages; a runtime killed between
    reading and truncating re-delivers them on the next open. The second is the
    safe direction and is the one this ordering chooses — a duplicated note is
    visible and harmless, a dropped one is neither.
    """
    path = inbox_path(session_dir)
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return []
    except OSError:
        logger.warning("could not open inbox for %s", session_dir.name, exc_info=True)
        return []
    try:
        with _NonBlockingLock(fd) as lock:
            raw = _read_all(fd)
            lines = _parse(raw)
            if not lines:
                return []
            # WHAT GETS DELIVERED IS DECIDED FROM THE LATEST BYTES, not from the
            # read above. A recall appends its marker without the lock (this
            # module's appenders never block for one) and the operator's receipt
            # says the message was taken back — so a marker that lands while this
            # call is reading must still withhold its row, or the drain delivers
            # a message the user was told would not run. Measured by QA at 4/120
            # trials before this re-read, 0/120 after (QA Q-1, round 3).
            latest = _read_all(fd)
            if latest != raw:
                lines = _parse(latest)
                if not lines:
                    # Another drain emptied the batch between the two reads.
                    # Nothing to deliver and nothing to consume.
                    return []
            # The markers are consumed with the rows they name: the whole file
            # goes, and a recall has done its job once the batch it applied to is
            # gone.
            deliverable = _deliverable(lines)
            if lock.acquired or os.name == "nt":
                os.ftruncate(fd, 0)
            else:
                # Unlocked, a truncate could discard a row an appender wrote
                # between the read and here. Stage the remainder instead: the
                # rename is atomic, and the worst case is that a racing
                # appender's row lands in a file we just replaced, which the
                # NEXT open drains.
                #
                # ``latest``, not ``raw``: rows that arrived between the two
                # reads are part of this batch, so they must not survive as a
                # remainder to be delivered twice.
                _replace_remainder(path, latest)
            return deliverable
    except OSError:
        logger.warning("inbox drain failed for %s", session_dir.name, exc_info=True)
        return []
    finally:
        os.close(fd)


def _read_all(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)
