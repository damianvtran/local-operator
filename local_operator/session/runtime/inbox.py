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
import re
import secrets
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from local_operator.procstate import O_BINARY

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
#:
#: AND THE BOOT DRAIN HAS TO EXIST FOR THE PROMISE TO HOLD, which is the half
#: that was missing until 0.61.19: a row spooled by a DRAINING runtime named no
#: one who would raise a successor, so a headless session could retire with the
#: message held and no runtime ever coming for it (measured 2026-09-21 — three
#: rows, no successor, no owner). ``serving._spool_for_successor`` therefore
#: records the turn as OWED (``local_operator.wakes.spooled``) and the wake
#: supervisor, whose whole job is to make a runtime exist for a session, raises
#: one for it. The promise is unchanged; what changed is that a process keeps it.
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

#: The ONE line a coalesced wake row carries, and the pattern that recovers its
#: count so a re-coalesce REPLACES the note instead of nesting a second one.
#: Derived from the string itself rather than re-spelled, because the two halves
#: drifting apart is a silent failure (the note stops being strippable, and every
#: handover adds another line) that only a test would catch.
#:
#: **IT SITS AFTER THE ENVELOPE, AND THAT IS THE POINT** (design round 1, D1). The
#: envelope — ``(alarm) Scheduled wake w1 (20, every 20m) — cancel with …`` — is
#: the identity of the row: it is what ``wake_receipt_headline`` folds into the
#: human headline, what ``is_harness_notice_text`` recognises as harness-minted,
#: and what the wake block keys on when it collapses. A note placed FIRST made
#: the row anonymous: the collapsed surface showed the coalesce bookkeeping where
#: the wake's name belongs, the receipt headline came out as this sentence, and
#: the row stopped looking like the harness's own notice. So the collapsed
#: surface carries identity, and the count shows on expand — the same rule the
#: envelope's own ``(20, every 20m)`` follows.
#:
#: TWO DIFFERENT FACTS, deliberately both spelled (review round 1, NIT 1): the
#: envelope's index is the OCCURRENCE the schedule advanced to (``20`` = the 20th
#: fire of that schedule), and this note's count is how many of those fires were
#: COALESCED into this row's delivery (``5`` = five spooled rows folded into one).
#: They differ whenever the runtime drained mid-series, and neither is derivable
#: from the other.
#:
#: One shape leads with something else: a LATE fire carries
#: ``Session._missed_delivery_note`` as its first ``\n\n`` segment, so there the
#: note follows that prefix — the same head a single un-coalesced fire would have,
#: which is why this composition never makes the coalesce bookkeeping the row's
#: identity. The prefix is rare by construction (only a one-shot overdue past
#: ``MAX_ARM_MS``, or a resumed series) and unchanged by this fix.
_COALESCED_NOTE = (
    "(This wake fired {count} times while the runtime was being replaced; " "the latest is below.)"
)
_COALESCED_NOTE_RE = re.compile(
    "^" + re.escape(_COALESCED_NOTE).replace(re.escape("{count}"), r"(\d+)") + r"\n\n"
)


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

    ``harness_injected`` is the STRUCTURAL provenance stamp, carried across the
    spool for the same reason ``wake`` is: it is what the producer knew and the
    successor cannot re-derive. The owner prompt that arrived during an update
    window may have been harness chrome — the goal judge's continuation is the
    producer — and the successor replays the row through the ordinary
    admission, so a lost stamp would mint an UNSTAMPED user row that the
    marker-only surfaces paint as the operator's own words (measured on a live
    session: 10 such rows). Absent in rows written before this field existed,
    which reads as False: the old behaviour, a plain message.
    """

    text: str
    sender: dict[str, Any]
    mode: str = "mailbox"
    written_at: float = 0.0
    wake: bool = False
    source: str = SOURCE_PEER
    command_id: str = ""
    harness_injected: bool = False
    #: The PEER sender's minted message identity (``peer-<32hex>``), carried so
    #: the successor's delivery is idempotent against a row that reached the
    #: spool twice (a crash between the write and the ack). Additive and
    #: defaulted: a row an older build wrote has no key, reads as ``""``, and
    #: the drain then delivers it exactly as it always did. It only ever rides a
    #: peer row — the owner-prompt paths own no message id.
    message_id: str = ""
    #: WHICH wake schedule a spooled fire belongs to (``WakeSchedule.id``). A
    #: draining runtime spools one row PER occurrence, and a schedule that
    #: repeats fast — ``every: 20m`` across a long handover — used to spool one
    #: row per fire, which the successor then delivered as one turn per row: the
    #: reported flood of dozens of identical alarms. This id is what lets
    #: :func:`coalesce_wake_rows` recognise the rows as the SAME wake without
    #: parsing their text, and what lets :func:`remove_wake_rows` purge the rows
    #: of a schedule the user has since cancelled. Additive and defaulted for the
    #: same reason ``message_id`` is: a row an older build wrote has no key,
    #: reads as ``""``, and delivers individually, exactly as it always did.
    wake_id: str = ""
    #: How many FIRINGS this one row stands for. Additive and defaulted to 1 (the
    #: ordinary, un-coalesced row). Carried STRUCTURALLY rather than re-derived
    #: from the note text so a coalesced row that the deferral path re-spools
    #: merges on the next drain without double-counting: the total is the sum of
    #: what each member already stood for. See :func:`coalesce_wake_rows`.
    wake_fires: int = 1

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
            harness_injected=bool(payload.get("harness_injected", False)),
            message_id=str(payload.get("message_id", "") or ""),
            wake_id=str(payload.get("wake_id", "") or ""),
            wake_fires=_fire_count(payload.get("wake_fires", 1)),
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
            "harness_injected": self.harness_injected,
            "message_id": self.message_id,
            "wake_id": self.wake_id,
            "wake_fires": self.wake_fires,
        }


#: Every ``source`` value this build writes, so a reader can tell one of ours
#: from a value a NEWER build invented (which is read as a peer's row).
_KNOWN_SOURCES = frozenset({SOURCE_PEER, SOURCE_USER, SOURCE_RECALL})


def _known_source(raw: Any) -> str:
    """The row's own source when this build knows it, else the peer default."""
    return raw if isinstance(raw, str) and raw in _KNOWN_SOURCES else SOURCE_PEER


def _fire_count(raw: Any) -> int:
    """``wake_fires`` off a persisted row, coerced rather than trusted.

    The spool is untrusted input — a truncated or hand-edited line reaches
    ``from_json`` — so this must not raise the way ``int(raw)`` can: a row that
    fails to parse takes the whole batch with it (``_parse`` catches only the
    JSON decode). Anything unusable reads as the default, one fire.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 1
    return value if value >= 1 else 1


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
            # raises rather than waiting. The lock is skipped rather than
            # emulated, and BOTH users are written to be correct without it:
            # the writer relies on the atomic O_APPEND write, and
            # `drain_inbox` falls through to the staged-remainder rewrite
            # whenever the lock was not acquired — on this platform as much as
            # on a contended POSIX one.
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
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND | O_BINARY, 0o600)
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


def settle_owed_turn(session_dir: Path, *, cwd: str = "") -> None:
    """Make the owed-turn record agree with this session's spool, after a drain.

    Called by both drains once they have finished with the spool, and it settles
    in BOTH directions because the spool is the authority and the record is a
    claim about it:

    * the spool still holds a row that asks for a turn → the record must EXIST
      (a raise is still owed). It usually does — the writer put it there — but a
      supervisor that read the file while ``drain_inbox`` had it emptied (the
      deferral path re-appends what it will not deliver) can have cleared it in
      that window, and re-noting here is what closes that hole (review round 1,
      R1-4).
    * the spool no longer holds one → drop the record, judged against the value
      this call read so a row that lands meanwhile re-arms it instead of being
      deleted under (``clear_spooled_turn``'s compare-and-delete guard).

    Best-effort: the drain's own delivery has already happened by the time this
    runs, and a store that cannot be written (or a config dir this process cannot
    see) must not turn a delivered message into a failed turn. The cost of
    leaving a record behind is one engage the supervisor should not have made; the
    cost of raising here is the turn.
    """
    from local_operator.paths import config_dir
    from local_operator.wakes.spooled import (
        clear_spooled_turn,
        note_spooled_turn,
        read_spooled_turn,
        spool_owes_turn,
    )

    try:
        root = config_dir()
        session_id = session_dir.name
        if spool_owes_turn(session_dir):
            if read_spooled_turn(root, session_id) is None:
                note_spooled_turn(root, session_id, cwd=cwd)
            return
        record = read_spooled_turn(root, session_id)
        if record is not None:
            clear_spooled_turn(root, session_id, expected_updated_at_ms=record.get("updated_at_ms"))
    except Exception:  # noqa: BLE001 — a bookkeeping failure is not a delivery failure
        logger.debug("could not settle the owed turn for %s", session_dir, exc_info=True)


def drop_owed_turn(session_dir: Path) -> None:
    """Drop this session's owed-turn record because NO raise can discharge it.

    The one case that calls it is the boot drain's deferral: a session with no
    durable history keeps its PEER rows until the owner's first turn
    (``process._drain_inbox_into``'s ``requires_engagement`` branch), so every
    runtime raised for that record would boot, defer the same rows and exit —
    real work, hourly, that delivers nothing (review round 1, R1-10). The spool
    row is untouched and the owner's first turn still drains it, which is the
    deferral the sender's receipt actually describes.

    TWO GUARDS, because this is the third unguarded unlink in the circuit and the
    first two both took a fix in review (QA round 2, R2-Q2):

    * IT REFUSES WHEN THE SPOOL HOLDS THE OWNER'S OWN WORDS. A ``SOURCE_USER`` row
      is dischargeable by exactly the raise this function is declining — the
      successor's boot drain RUNS it, durable history or not — so a record that
      arrived for one must survive; the deferral judgement is about the peer rows
      beside it. Without this guard the drop would delete a record the owner's own
      prompt had just written, which is defect (B) once more, this time caused by
      the fix for R1-10 rather than by the cap.
    * AND IT PASSES ``expected_updated_at_ms``, the same compare-and-delete the
      reconciler and :func:`settle_owed_turn` pass: the record is judged from the
      read a moment earlier, so a record that has changed since is left for the
      next pass rather than unlinked on a stale judgement. What the guard narrows
      is the window, from the whole deferral to the unlink itself.

    Best-effort, like every other write on this path.
    """
    from local_operator.paths import config_dir
    from local_operator.wakes.spooled import (
        clear_spooled_turn,
        read_spooled_turn,
        spool_has_owner_row,
    )

    try:
        if spool_has_owner_row(session_dir):
            return
        root = config_dir()
        record = read_spooled_turn(root, session_dir.name)
        if record is None:
            return
        clear_spooled_turn(
            root, session_dir.name, expected_updated_at_ms=record.get("updated_at_ms")
        )
    except Exception:  # noqa: BLE001 — see settle_owed_turn
        logger.debug("could not drop the owed turn for %s", session_dir, exc_info=True)


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


def coalesce_wake_rows(lines: list[InboxLine]) -> list[InboxLine]:
    """Fold repeated firings of ONE wake into a single row (the flood fix).

    **The defect this closes.** A draining runtime spools one row per FIRED
    occurrence (``Session._spool_wake_to_inbox``). A recurring wake whose runtime
    is draining for a long time — an ``every: 20m`` schedule across a multi-hour
    build handover — therefore spooled a row per occurrence: 52 identical rows
    over ~17 h, each of which the successor then delivered as its OWN wake and
    its own turn. The user comes back to dozens of alarms for one reminder.

    **What is merged.** Among the batch a drain just took, every group of rows
    that share a non-empty ``wake_id`` AND carry ``wake=True`` — i.e. repeated
    fires of the same schedule — becomes ONE row, placed where the group's LAST
    occurrence was, carrying the latest text and the summed ``wake_fires``. Rows
    without a ``wake_id`` (a peer note, an owner prompt, a row an older build
    wrote) are untouched and keep their order relative to everything else, so a
    batch that interleaves peer notes with wake fires keeps both its ordering and
    its meaning. A group of ONE is returned unchanged: a single fire is not a
    flood, and rewriting its text would be a change with no cause.

    **Why the count is structural AND in the text.** The predecessor is already
    in a batch that can be RE-SPOOLED: ``process._drain_inbox_into`` puts back the
    rows it will not deliver yet (no durable history), and those rows are merged
    again on the next drain. Carrying ``wake_fires`` on each row makes the second
    merge sum what each member already stood for — 5 then 1 is 6, never "two
    notes" — and :data:`_COALESCED_NOTE_RE` lets the earlier note be REPLACED
    instead of nested, so the text says what the count says. Both halves are
    tested (``test_a_re_coalesce_replaces_the_note_instead_of_nesting_it``).

    **The merged row keeps the envelope at its head** and puts the note between
    the envelope and the message: the envelope is the row's identity (what the
    receipt folds, what marks the text as harness-minted) and the note is
    bookkeeping about the delivery. ``_COALESCED_NOTE`` carries the reasoning.

    Pure and total: it never raises, never writes, and never reorders what it
    does not merge.
    """
    groups: dict[str, list[int]] = {}
    for index, line in enumerate(lines):
        if line.wake and line.wake_id:
            groups.setdefault(line.wake_id, []).append(index)
    merged: dict[int, InboxLine] = {}
    folded: set[int] = set()
    for indices in groups.values():
        if len(indices) < 2:
            continue
        last = indices[-1]
        latest = lines[last]
        fires = sum(lines[index].wake_fires for index in indices)
        # The note goes AFTER the text's leading segment, which is the envelope for
        # every fire that was not annotated as missed — see ``_COALESCED_NOTE``. A
        # note a previous merge left in the remainder is stripped from the REMAINDER
        # only, never from the leading segment, which is the identity this row's
        # collapse and receipt read.
        envelope, _, remainder = latest.text.partition("\n\n")
        remainder = _strip_coalesced_note(remainder)
        text = (
            f"{envelope}\n\n{_flagged_note(fires)}\n\n{remainder}"
            if remainder
            else f"{envelope}\n\n{_flagged_note(fires)}"
        )
        merged[last] = replace(latest, text=text, wake_fires=fires)
        folded.update(indices[:-1])
    if not merged:
        return list(lines)
    out: list[InboxLine] = []
    for index, line in enumerate(lines):
        if index in folded:
            continue
        out.append(merged.get(index, line))
    return out


def _flagged_note(fires: int) -> str:
    """The ONE line a coalesced row carries, after the leading envelope."""
    return _COALESCED_NOTE.format(count=fires)


def _strip_coalesced_note(remainder: str) -> str:
    """Drop a previous coalesce note so a re-coalesce can replace, not nest.

    Takes the text AFTER the envelope (see ``coalesce_wake_rows``) and is anchored
    at its start with the whole note — trailing blank line included — matched, so
    the sentence is only ever removed where a previous merge put it: the head of
    the remainder. The same words further down the wake's own message, or in a
    single-fire row that was never merged, are left alone.
    """
    return _COALESCED_NOTE_RE.sub("", remainder, count=1)


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
        fd = os.open(path, os.O_RDWR | os.O_APPEND | O_BINARY)
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


def remove_wake_rows(session_dir: Path, wake_id: str) -> int:
    """Drop every pending spooled fire of ONE wake. Returns how many went.

    **The defect half this closes.** Spooled rows outlive the schedule that wrote
    them: cancelling a wake stops the store entry and the supervisor's errand
    (``WakeScheduler.update``), but the rows a draining runtime already spooled
    stay in the file, so the successor still delivers the alarms the user just
    cancelled. This is what the cancel path calls to take them out.

    **Why it rewrites IN PLACE rather than through** ``os.replace``. The module's
    two other writers are asymmetric on purpose: :func:`_replace_remainder` uses
    a staged replace and can lose a row an appender wrote in the window
    (``withdraw_inbox``'s docstring records the measurement — 10/27, 10/45,
    14/42 acked rows lost, because an unlocked appender's already-open
    ``O_APPEND`` descriptor points at the inode the replace orphans). That loss is
    acceptable for a drain, which is delivering the rows it read; it is NOT
    acceptable here, where the rows we keep are exactly the ones nobody has seen.
    Truncating and rewriting through the SAME descriptor keeps that appender's
    descriptor pointing at the live file, so a row it writes after our rewrite
    lands at the end of our payload as a clean line instead of vanishing.

    **What it does NOT close, stated rather than left implicit.** An appender
    writes UNLOCKED (``_NonBlockingLock``'s contract is retry-then-proceed, because
    the appending side must never block the event loop), so a row written between
    our ``_read_all`` and our ``ftruncate`` is read by neither pass and is LOST —
    the same read→truncate window ``drain_inbox`` documents (PR #1319). Rewriting
    through the live descriptor narrows the loss to that window instead of the
    whole replace window (an orphaned ``O_APPEND`` fd loses everything after the
    replace), but it does not remove it: a purge is not atomic against an
    appender, and the honest claim is "keeps what it read, and what the appender
    wrote before the truncate".

    **Best-effort by contract, and it never corrupts to make a deadline.** If the
    non-blocking lock cannot be taken, another reader owns the spool and nothing is
    touched: it returns 0 with the file left byte-identical to what it found. The
    caller's fallback is the unchanged rows, which part 1 of this fix
    (:func:`coalesce_wake_rows`) still folds to ONE delivery rather than N. A
    failed open or write returns 0 the same way, leaving the rows to the next
    drain.
    """
    if not wake_id:
        return 0
    path = inbox_path(session_dir)
    try:
        fd = os.open(path, os.O_RDWR | O_BINARY)
    except FileNotFoundError:
        return 0
    except OSError:
        logger.warning("could not open inbox for %s", session_dir.name, exc_info=True)
        return 0
    try:
        with _NonBlockingLock(fd) as lock:
            if not lock.acquired:
                # Another reader owns the window (a drain consuming the file, or a
                # sibling cancel). Leaving its rows is the safe direction: they
                # are coalesced to one delivery, never lost.
                logger.info("inbox purge for wake %s skipped: a reader owns the spool", wake_id)
                return 0
            raw = _read_all(fd)
            kept, dropped = _without_wake_rows(raw, wake_id)
            if not dropped:
                return 0
            _rewrite_in_place(fd, b"".join(kept))
            return dropped
    except OSError:
        logger.warning("inbox purge failed for %s", session_dir.name, exc_info=True)
        return 0
    finally:
        os.close(fd)


def _without_wake_rows(raw: bytes, wake_id: str) -> tuple[list[bytes], int]:
    """``(kept line bytes, rows dropped)`` for the asked-for wake.

    Byte-faithful on purpose: the chunks that survive are the bytes that were
    there, never a re-serialisation of the parsed row. ``InboxLine.from_json``
    keeps only the fields this build knows, so a rewrite through it would strip
    whatever a newer build wrote beside them — the additivity this module relies
    on is exactly the set of keys an older reader must NOT launder. A chunk that
    does not parse (a torn final line, a hand-edited file) is kept untouched for
    the same reason.
    """
    kept: list[bytes] = []
    dropped = 0
    for chunk in raw.splitlines(keepends=True):
        if not chunk.strip():
            kept.append(chunk)
            continue
        try:
            payload = json.loads(chunk)
            line = InboxLine.from_json(payload) if isinstance(payload, dict) else None
        except (ValueError, TypeError, KeyError):
            line = None
        if line is not None and line.wake and line.wake_id == wake_id:
            dropped += 1
            continue
        kept.append(chunk)
    return kept, dropped


def _rewrite_in_place(fd: int, payload: bytes) -> None:
    """Truncate the open spool and write ``payload`` through the SAME handle.

    The ``O_APPEND`` appender discipline is a constraint, not a detail: its
    descriptor is the live file's, so it must stay the live file. Any short write
    is looped, and the caller holds the non-blocking lock, which is the same
    window :func:`drain_inbox` truncates in.
    """
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _replace_remainder(path: Path, consumed: bytes) -> None:
    """Rewrite the spool with only the bytes written after ``consumed``.

    Staged through ``os.replace`` so a reader never sees a truncated file, and it
    is the ONLY place this module replaces the file: the recall appends a marker
    rather than rewriting (see :func:`withdraw_inbox`), and
    :func:`remove_wake_rows` rewrites through the live descriptor without a
    replace — for the appender-orphan reason its docstring gives.
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
    closed = False
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
            if lock.acquired:
                os.ftruncate(fd, 0)
            else:
                # Unlocked, a truncate could discard a row an appender wrote
                # between the read and here. Stage the remainder instead: the
                # rename is atomic, and the worst case is that a racing
                # appender's row lands in a file we just replaced, which the
                # NEXT open drains.
                #
                # **Windows takes THIS branch too, and must close first.** The
                # ``or os.name == "nt"`` this replaces sent the platform with
                # no locking (see _NonBlockingLock) straight to the one
                # operation the sentence above identifies as able to discard a
                # row. Nothing about the remainder path is POSIX-specific, but
                # it DOES need the handle gone before it runs: on Windows
                # ``os.replace`` is ``MoveFileExW(MOVEFILE_REPLACE_EXISTING)``,
                # and an existing open of the DESTINATION path makes that call
                # fail — CPython's own issue 46003 records that delete sharing
                # does not save it, and our fd shares neither delete nor
                # anything else the CRT offers. Left open, every unlocked drain
                # would fail the rename, warn, and redeliver the whole spool.
                # Closing first is a no-op on POSIX, where a rename over an
                # open descriptor is ordinary.
                os.close(fd)
                closed = True
                # ``latest``, not ``raw``: rows that arrived between the two
                # reads are part of this batch, so they must not survive as a
                # remainder to be delivered twice.
                _replace_remainder(path, latest)
            return deliverable
    except OSError:
        logger.warning("inbox drain failed for %s", session_dir.name, exc_info=True)
        return []
    finally:
        if not closed:
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


# -- the update window's handover marker ---------------------------------------
#
# The inbox file above carries the MESSAGES a leaving runtime spooled. This
# carries the FACT that a handover was in flight, and the two are separate files
# because they have different lifetimes and different readers:
#
#   * the spool is read by the successor's own delivery path and is EMPTY until
#     somebody sends something;
#   * the marker is written by the window's OPEN and read (once) by the
#     successor's BOOT, whether or not a single message was queued.
#
# WHY A FILE AND NOT A FIELD. The pair has to cross a process boundary in the
# one direction the record cannot serve: the predecessor's record is gone by the
# time the successor imports, and the successor's own record is written before
# any of this could be consulted (``_bind_boot_instrumentation`` runs before the
# inbox drain). The session directory is the only durable handover surface both
# ends already share, and the predecessor ALREADY owns it.
#
# IT SURVIVES A FAILED WINDOW ON PURPOSE, in one direction only: the runtime that
# opens a window clears this marker when it aborts (``end_update``), so a marker
# that outlives its writer is evidence of a process that died mid-move. The
# successor still reports the update as applied, which is the honest reading —
# it IS running the newer build — and is exactly the case the operator could
# never see before this existed.
UPDATE_WINDOW_NAME = "update-window.json"


def update_window_path(session_dir: Path) -> Path:
    return session_dir / UPDATE_WINDOW_NAME


def write_update_window(session_dir: Path, pair: str) -> bool:
    """Record that an update window is moving to ``pair``. ``False`` on failure.

    Written through a temporary in the same directory and renamed into place, so
    a reader can never observe a half-written marker (the window opens
    synchronously inside the idle decision, where there is no second rung to
    retry from). Failure is NOT fatal to the window: the messages still spool and
    the successor still runs them — what is lost is only the "updated" fact, which
    is the cheaper half.

    THE TEMPORARY NAME IS UNIQUE PER WRITER, and that is a fix rather than
    tidiness (agent review round 1, MINOR 4). ``path.with_suffix('.tmp')`` — the
    obvious spelling, and the hazard ``model/catalogue.py`` documents for the same
    construct — is ONE name for every writer of one session directory, and two
    runtimes can serve one directory: that is exactly the shape a blocked loop
    provokes, where the supervisor spawns a replacement while the predecessor is
    still in its exit leg. Measured against that shape (2 writers x 3000 writes):
    **59** reads of an absent-or-corrupt marker, and **2374** writes reporting
    failure (``FileNotFoundError: update-window.tmp -> update-window.json`` — the
    other writer had already renamed it away). The two failure modes are both
    silent in the direction that matters: a writer whose ``os.replace`` installs
    the other's truncated file returns ``True``, and the successor then publishes
    no ``updated`` fact at all.

    ``mkstemp`` also gives O_EXCL, so two writers cannot open the same temporary
    in the first place. The temporary is removed on every failure path: a leaked
    ``.tmp`` inside a session directory is a file nothing else knows about.
    """
    path = update_window_path(session_dir)
    fd = -1
    tmp = ""
    try:
        # ``dir=`` puts the temporary on the same filesystem, which is what makes
        # the ``os.replace`` below atomic.
        fd, tmp = tempfile.mkstemp(dir=session_dir, prefix=UPDATE_WINDOW_NAME + ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            # ``os.fdopen`` takes ownership of ``fd``; ``stream`` marks that here so
            # the ``finally`` below does not close a descriptor the file object
            # already closed.
            fd = -1
            json.dump({"pair": pair, "pid": os.getpid()}, handle)
        os.replace(tmp, path)
        return True
    except OSError:
        logger.warning(
            "could not write the update-window marker for %s", session_dir.name, exc_info=True
        )
        return False
    finally:
        if fd >= 0:  # pragma: no cover - only the failed-fdopen path reaches this
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass  # the rename moved it, which is the success path
            except OSError:  # pragma: no cover - a leaked temp is not worth a raise
                logger.debug("could not remove the marker temporary %s", tmp, exc_info=True)


def read_update_window(session_dir: Path) -> str:
    """The pair a handover was moving to, or ``""`` when no marker is present.

    An unreadable or malformed marker reads as absent rather than raising: a boot
    must not fail because a sidecar from an older or killed build is malformed,
    and the cost of missing one is a fact that is merely nice to have.
    """
    try:
        raw = json.loads(update_window_path(session_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    pair = raw.get("pair") if isinstance(raw, dict) else ""
    return pair if isinstance(pair, str) else ""


def clear_update_window(session_dir: Path) -> bool:
    """Remove the marker, whether or not one is there. ``True`` if it was."""
    try:
        update_window_path(session_dir).unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning(
            "could not clear the update-window marker for %s", session_dir.name, exc_info=True
        )
        return False
