"""Arm, cancel and edit a session's wakes from OUTSIDE the session.

Three writers existed for schedule state and this module is the second one,
unified. ``Session._persist_wake_schedules`` is the one writer while a runtime
owns a session; this is the writer for a session with NO runtime, where there
is nobody to ask. ``lop wake create`` was already such a writer (its own
transcript append, its own index write, its own validation) and the desktop
routes added two more (arm and edit) — so the reason this module exists is not
tidiness. Each copy had drifted, and the drift was silent:

- the CLI allocated ``w{len(existing) + 1}``, so a cancel-then-arm REUSED a
  live id and a 17th wake was written as ``w17`` past the 16 cap;
- the CLI's index write passed no ``preserve``, so arming a wake on a STOPPED
  session un-parked it (dropping ``stopped_at``) and wiped the lateness
  telemetry the session records;
- and every copy re-derived the transcript-vs-index rule for itself.

**The invariant this module implements is three-part** (see
``wakes/store.py``'s docstring for the full statement): the transcript's
latest ``wake_schedules`` entry is the truth; the index is a derived
projection that a reader must never patch; and a write must leave the two
agreeing, or leave a marker saying it could not.

Consequences that shape every function here:

- **The base is the TRANSCRIPT, never the index.** An append REPLACES the
  session's schedule list, so reading a stale base — the index lags, and is
  absent for a session that has not been opened — would silently cancel live
  reminders. The CLI already did this and said why; it is the rule now.
- **Transcript first, index second, install hook third** — the same order as
  ``_persist_wake_schedules``, and the same asymmetry: only the transcript
  append may fail the request, because an index entry with no transcript is a
  reminder the supervisor faithfully fires into nothing.
- **Post-write verification.** Last-writer-wins on a full-list snapshot is
  safe *by construction* only while nothing else writes; the TUI and a
  resumed session can both append to the same transcript. So after the append
  this re-reads the latest entry and checks it is OURS — a mismatch means
  another writer landed on top of us, and the honest answer is to retry once
  from the new base and then refuse with a conflict rather than report a
  success that lost the user's wake.

**Not the place for either live-session policy, but it is the place for the
guard.** A live runtime's schedules are in memory and republished wholesale on
its next persist, so an append here would be overwritten from the transcript
AND from the index while the wake never fired (the supervisor skips a session
with a live record). The route resolves the owner first and asks it; when there
is nobody to ask this module writes — and immediately before every append it
re-asks whether an owner has appeared, refusing rather than writing behind one
(:func:`_refuse_if_owned`). That re-ask is the last line of defence, not the
first: the route already refused a live or wedged owner before calling here.

The other half of writing correctly from outside is serialisation. Two writers
whose reads interleave merge onto the same base, so one row is lost or written
twice; :mod:`local_operator.wakes.lock` is the per-session mutex that makes the
read-merge-append atomic between external writers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from local_operator.harness.wake import (
    WAKE_SCHEDULES_CUSTOM_TYPE,
    WakeSchedule,
    build_wake_edit,
    build_wake_schedule,
)
from local_operator.session.transcript import read_latest_custom_entry
from local_operator.wakes.lock import WakeLockBusy, WakeLockUnavailable, WakeWriteLock

logger = logging.getLogger(__name__)

#: What the caller of :func:`arm_wake` and friends is told to do about an
#: error. The status is carried HERE rather than mapped by each caller so the
#: CLI (which prints the sentence and exits 1) and the desktop route (which
#: raises that status) cannot disagree about what an error means.
STATUS_SESSION_NOT_FOUND = 404
STATUS_WAKE_NOT_FOUND = 404
STATUS_INVALID = 422
STATUS_CONFLICT = 409
#: 503 for both owner states and for a contended lock: every one of them means
#: "nothing was written, and retrying is what fixes it", which is what makes
#: them different from the 409s above (a conflict with the state itself).
STATUS_OWNER_BUSY = 503
STATUS_WRITE_BUSY = 503

#: The refusal for an owner that EXISTS but is not answering. One sentence, used
#: by the route's own pre-flight refusal and by the guard at the append below,
#: so a user cannot get two different descriptions of the same dead end.
WEDGED_MESSAGE = (
    "This conversation's runtime is not responding. Retry in a moment, or stop that conversation."
)

#: The clause every refusal MAY end on, kept as one constant because it is only true
#: on the paths where the write really did leave nothing behind — and the one path
#: that cannot promise it REPLACES the whole sentence rather than re-using this one
#: (review round 4, MINOR-3).
NOTHING_WRITTEN = "Nothing was written."

#: The refusal for an owner the server can SEE but cannot DIAL — a live pid in
#: ``.session.pid`` with no usable discovery record. A separate sentence because
#: the common one is FALSE in this state (review round 3, R6's minor): there may be
#: no session to change anything from, and "retry in a moment" promises a retry
#: that will not help while that pid lives. Both honest explanations are named,
#: because the code cannot tell them apart and the user can: an older build's
#: runtime (which will answer that session's own UI) or a marker left behind by a
#: pid the OS has since recycled onto an unrelated process.
OWNER_UNKNOWN_MESSAGE = (
    "This conversation's schedule file is claimed by process {pid}, which does not "
    "answer as a runtime — either a conversation open in an older build, or a stale "
    "owner marker left by a pid the system has since reused. " + NOTHING_WRITTEN + " "
    "If nothing has this conversation open, the marker at {marker} is stale and can "
    "be removed; otherwise change its schedules from that session."
)

#: The THIRD owner state (review round 4, MINOR-2): the pid is live and the
#: runtime registry could not be READ, so this server cannot say whether that owner
#: is dialable. ``dialable_record_exists`` returns None here, and its own docstring
#: is explicit that a failed read "is not evidence of absence". Reusing the
#: mirror-only sentence would assert a cause this code has not established, and
#: reusing the dialable one would assert the opposite; so the sentence names the
#: unread registry and asks for the retry that would resolve it. Refusing is the
#: conservative side of "pace rather than refuse": a write sent behind an owner we
#: could not identify is the lost reminder this guard exists to prevent, and the
#: retry IS the pacing.
OWNER_REGISTRY_UNREAD_MESSAGE = (
    "This conversation is held by another process (pid {pid}) and this server could "
    "not read the runtime registry to say whether it is answering, so the schedules "
    "were left alone. Retry in a moment."
)

#: Shown INSTEAD of a refusal's own sentence when the write could not be confirmed
#: undone (review round 4, MINOR-3): the older text ended "Nothing was written",
#: which is exactly what is unknown here. It keeps the refusal's status and code,
#: because the caller's branch is unchanged — only the promise is.
UNSETTLED_MESSAGE = (
    "This conversation is held by another process, and whether this request's write "
    "took effect could not be established. Reconcile the conversation's wake list "
    "before retrying."
)


class WakeWriteError(Exception):
    """A refusal the caller can render verbatim.

    ``message`` is always the user-facing sentence — for a validation failure
    it is the SAME text ``build_wake_schedule`` returns to the agent's tool, so
    one wording exists for "you cannot arm a 17th wake" across every surface.

    ``wrote`` says whether any of THIS request's bytes may still be in effect — the
    fact the receipt journal's release rule needs and cannot infer (review round 3,
    R8). The write sequence rolls its own appends back before refusing and reports
    the result, so it is ``True`` in exactly two places: the conflict refusal, whose
    verify loop cannot prove which attempt landed, and the undo-failure path, which
    hands the caller UNSETTLED_MESSAGE instead of the refusal's own sentence
    (review round 4, MINOR-3 and the NIT — this paragraph used to claim every raise
    carried ``False``, which the conflict raise contradicts).
    """

    def __init__(self, message: str, *, status: int, code: str, wrote: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.wrote = wrote


@dataclass(frozen=True)
class WakeWriteOutcome:
    """What a successful write did, for the caller's receipt.

    ``index_written`` is "the derived index now reflects this change, as far as
    this process can tell": true when the write did not raise, which includes
    a cancel that emptied the list and REMOVED the file (``index_path`` is then
    empty). It is False only when the index write itself failed — the
    transcript still won and the next open heals it, which is why that is a
    reported field rather than an error.
    """

    session_id: str
    wake_id: str
    next_due_at: int | None
    schedules: list[WakeSchedule]
    index_written: bool = False
    index_path: str = ""
    supervisor: str = ""


#: (wake_id, full new list, the row's due instant) — what a mutation produces
#: from the current base. Raising :class:`WakeWriteError` is how it refuses.
Mutation = Callable[[list[WakeSchedule], int], "tuple[str, list[WakeSchedule], int | None]"]


async def arm_wake(
    config_dir: Path,
    session_id: str,
    request: Mapping[str, Any],
    *,
    cwd: str | None = None,
    now_ms: int | None = None,
    request_id: str = "",
) -> WakeWriteOutcome:
    """Add one schedule to ``session_id``'s wakes. ``request`` is the same
    ``message``/``in``/``at``/``every``/``until``/``limit`` mapping the agent's
    ``wake`` tool takes, validated by the same function.

    ``request_id`` is the CALLER's request id when it has one (the desktop route
    always does). It is stamped on the row as its origin so that a later attempt
    of the same request can ask whether its write landed by IDENTITY rather than by
    comparing fields no further than the wake's own firing can change — the defect
    review round 4 (R9) drove. Writers with no request id (the agent's tool, the
    CLI) leave it unset and fall back to id-plus-message.
    """

    def mutate(
        existing: list[WakeSchedule], now: int
    ) -> tuple[str, list[WakeSchedule], int | None]:
        built = build_wake_schedule(dict(request), existing, now)
        if "error" in built:
            raise _refusal(built)
        schedule = built["schedule"]
        if request_id:
            schedule = schedule.model_copy(update={"request_id": request_id})
        return schedule.id, [*existing, schedule], schedule.next_due_at

    return await _apply(config_dir, session_id, mutate, cwd=cwd, now_ms=now_ms, intent=request_id)


async def cancel_wake(
    config_dir: Path,
    session_id: str,
    wake_id: str,
    *,
    cwd: str | None = None,
    now_ms: int | None = None,
) -> WakeWriteOutcome:
    """Drop one schedule. An empty remainder REMOVES the index entry rather
    than writing an empty one, which is also what releases the cleanup reap
    guard (``session/cleanup.py``) that a wake-carrying session holds."""

    def mutate(
        existing: list[WakeSchedule], now: int
    ) -> tuple[str, list[WakeSchedule], int | None]:
        if not any(schedule.id == wake_id for schedule in existing):
            raise _unknown_wake(wake_id, existing)
        remaining = [schedule for schedule in existing if schedule.id != wake_id]
        return wake_id, remaining, None

    return await _apply(config_dir, session_id, mutate, cwd=cwd, now_ms=now_ms)


async def edit_wake(
    config_dir: Path,
    session_id: str,
    wake_id: str,
    request: Mapping[str, Any],
    *,
    cwd: str | None = None,
    now_ms: int | None = None,
) -> WakeWriteOutcome:
    """Change an existing schedule's message and/or bounds.

    A key ABSENT from ``request`` keeps the row's current value; a key present
    with value ``None`` clears it. Callers that model the body as a pydantic
    request get that for free from ``exclude_unset=True`` — the distinction is
    the difference between "leave the limit" and "remove the limit", so a
    plain dump is not equivalent.
    """

    def mutate(
        existing: list[WakeSchedule], now: int
    ) -> tuple[str, list[WakeSchedule], int | None]:
        if not any(schedule.id == wake_id for schedule in existing):
            raise _unknown_wake(wake_id, existing)
        built = build_wake_edit(dict(request), existing, wake_id, now)
        if "error" in built:
            raise _refusal(built)
        row = built["schedule"]
        # IN PLACE: a row that moved to the end of the list on every edit would
        # reorder the page under the user for a message-only change.
        updated = [row if schedule.id == wake_id else schedule for schedule in existing]
        return row.id, updated, row.next_due_at

    return await _apply(config_dir, session_id, mutate, cwd=cwd, now_ms=now_ms)


# ---------------------------------------------------------------------------
# The shared write sequence
# ---------------------------------------------------------------------------


async def _apply(
    config_dir: Path,
    session_id: str,
    mutate: Mutation,
    *,
    cwd: str | None,
    now_ms: int | None,
    intent: str = "",
) -> WakeWriteOutcome:
    session_dir = Path(config_dir) / "sessions" / session_id
    if not await asyncio.to_thread(session_dir.is_dir):
        # Mirroring the CLI's refusal, for the same reason: a wake id in the
        # index with no transcript fires into nothing. The desktop route
        # creates the session before calling this, so a miss here is a real
        # miss rather than the cold-start case.
        raise WakeWriteError(
            f"no session {session_id!r}", status=STATUS_SESSION_NOT_FOUND, code="session_not_found"
        )
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)

    # ONE WRITER AT A TIME, and the lock spans the awaits below rather than just
    # the append: the base is read, merged and replaced, so two writers that
    # interleave their reads lose or duplicate a row (measured on the route: six
    # rows from five arms, all answered 200). It is taken on a worker thread and
    # released on one, so the event loop stays free, and a timed-out acquire is a
    # REFUSAL rather than "carry on unlocked": the failure that would follow is
    # the duplicate row, and a retryable 503 is the honest answer to contention.
    # See :mod:`local_operator.wakes.lock` for why this is not the session lease
    # and why the owning runtime is deliberately not a party to it.
    lock = WakeWriteLock(session_dir)
    try:
        await asyncio.to_thread(lock.acquire)
    except WakeLockBusy as busy:
        raise WakeWriteError(str(busy), status=STATUS_WRITE_BUSY, code="wake_write_busy") from None
    except WakeLockUnavailable as unusable:
        # The lock file itself could not be created — a read-only session
        # directory, or one that vanished between the existence check above and
        # the open. Nothing was written, and re-running cannot help until the
        # directory's mode changes, so this is a 409 that names the fix rather
        # than the 500 an untranslated OSError used to answer with (review
        # round 2, R7; QA Q5 — the base answered 200 here, so this sentence is
        # what a user gets instead of a silent success).
        raise WakeWriteError(
            f"Cannot schedule a wake for this conversation: {unusable}. "
            "Restore write permission on its directory, or reopen the conversation "
            "from a writable location, and retry.",
            status=STATUS_CONFLICT,
            code="wake_write_unavailable",
        ) from None
    try:
        rows, wake_id, due = await _mutate_locked(
            config_dir, session_dir, session_id, mutate, now, intent=intent
        )
        # The INDEX write is inside the lock too, and that is not incidental: it
        # is written from the list this call produced, so a writer that released
        # before writing it could overwrite a successor's newer entry with its
        # own older rows.
        previous = await asyncio.to_thread(_read_index_entry, config_dir, session_id)
        index_written = False
        index_path = ""
        try:
            written = await asyncio.to_thread(
                _write_index,
                config_dir,
                session_id,
                cwd or _resolved_cwd(session_dir, previous),
                rows,
                previous,
            )
            index_written = True
            index_path = str(written) if written is not None else ""
        except Exception:  # noqa: BLE001 — the transcript already has it (store.py's contract)
            logger.warning(
                "could not update the wake index entry for %s", session_id, exc_info=True
            )
    finally:
        await asyncio.to_thread(lock.release)

    supervisor = ""
    if rows:
        # Only when something is armed, and OUTSIDE the lock: installing the
        # supervisor shells out to launchd, and holding a per-session write lock
        # across a subprocess would queue every other writer behind it for no
        # reason — the hook touches no schedule state.
        #
        # A cancel that emptied the list leaves the supervisor to retire on its
        # own (it re-reads the index every slice and exits when nothing is
        # fireable), and installing one for an empty index would fight that
        # retirement.
        supervisor = await asyncio.to_thread(_ensure_supervisor, config_dir)

    return WakeWriteOutcome(
        session_id=session_id,
        wake_id=wake_id,
        next_due_at=due,
        schedules=rows,
        index_written=index_written,
        index_path=index_path,
        supervisor=supervisor,
    )


async def _mutate_locked(
    config_dir: Path,
    session_dir: Path,
    session_id: str,
    mutate: Mutation,
    now: int,
    *,
    intent: str = "",
) -> tuple[list[WakeSchedule], str, int | None]:
    """read -> mutate -> guard -> append -> verify -> guard, under the caller's lock.

    THE LOCK SPANS THE WHOLE SEQUENCE, and saying so is not incidental: the read
    that produces the base, the owner guard, the append, the re-read that verifies
    it, and the rollback below all happen while the per-session lock is held (the
    caller takes it in :func:`_apply` and releases it after the index write), so no
    second EXTERNAL writer can interleave with any of it. What the lock cannot
    exclude is a live session's own ``_persist_wake_schedules``, which is not a
    party to it — the runtime is not, and taking the session lease to change that
    would refuse every attach and paint this server's pid into ``.session.pid`` for
    the duration of a write. The owner guards below are therefore the answer to
    that writer, not a substitute for the lock.

    Returns the full new list, the row's id and its due instant.

    WHAT THE IDENTITY LEAVES BEHIND, stated rather than implied. A write that
    carries the caller's request id (every desktop arm does) is now settled by an
    exact lookup, so the duplicate-row class the last two rounds drove is closed
    for it in both directions: the retry of a write that may have landed finds its
    own row and reports APPLIED — before any mutation, so the 16-schedule cap
    cannot refuse a request that already succeeded and no second id slot is taken,
    and no second row means no second fire budget (``fired_count``/``limit``
    accounting stays on the ONE row, whose current ``next_due_at`` the reply now
    reports) — while a retry whose row is ABSENT re-arms it, which is correct:
    that wake really was lost. A write with NO identity (the agent's ``wake`` tool,
    the CLI) keeps the content-based settling of round 3, which cannot see through
    a re-timed copy of its own row; what it does have is the id-based rollback, so
    a refusal leaves its row behind and a caller that repeats the write gets one
    row per intent rather than a silent second one. There is no retry contract for
    those writers to break: neither has a request id, so neither can reach the
    released-then-re-armed path this whole mechanism exists for.
    """
    rows: list[WakeSchedule] = []
    wake_id = ""
    due: int | None = None
    before: list[WakeSchedule] = []
    #: The `(base, appended)` pair of the last snapshot this request actually
    #: wrote — the material a rollback needs. NOT the current iteration's pair:
    #: on attempt 1 they describe a write that has not happened yet.
    written: tuple[list[WakeSchedule], list[WakeSchedule]] | None = None
    for attempt in (0, 1):
        existing = await asyncio.to_thread(_read_rows, session_dir)
        landed = _settled(before, rows, existing, intent)
        if landed is not None:
            # THIS REQUEST'S CHANGE IS ALREADY IN THIS BASE (rounds 3 and 4,
            # R8/R9). The change is in effect, so the base IS the answer —
            # including the due instant the row now carries, which the session's
            # own firing may have advanced. No mutation runs, so a retry of an
            # applied arm cannot be refused by the 16-cap either, and cannot
            # consume a second id slot.
            logger.info("wake write for %s is already in the base it read", session_id)
            return existing, landed[0], landed[1]
        before = existing
        wake_id, rows, due = mutate(existing, now)
        try:
            await _refuse_if_owned(config_dir, session_id)
        except WakeWriteError as refusal:
            # A refusal on the SECOND attempt comes from a request that has
            # already appended — attempt 0's snapshot is on disk, whatever the
            # guard's reason is now. Roll our own write back before refusing, so
            # that every refusal this module raises really does leave nothing
            # durable behind (which is what lets the journal release its claim
            # and let a retry run — see ``WakeWriteError.wrote``).
            raise await _refusal_after_undoing(refusal, session_dir, written, intent)
        entry = await _append(session_dir, rows)
        written = (existing, rows)
        try:
            # THE POST-APPEND GUARD (review rounds 2-3, Q3). The pre-append one
            # is a moment before the write, and a runtime that claims the session
            # inside the append loads its schedule list before our row exists and
            # republishes that stale list on its next persist — deleting the row
            # from the transcript AND the index while the supervisor skips any
            # session with a live owner. Measured: ``200 index_written:true``, then
            # an empty snapshot and a removed index entry. Asking again HERE,
            # after the append and inside the same critical section, is what makes
            # that refusal instead of a silent loss: the window in which an owner
            # can appear and still be detected is now "before this check" rather
            # than "before the append".
            await _refuse_if_owned(config_dir, session_id)
        except WakeWriteError as refusal:
            # The append is undone, so the refusal leaves no lasting state for the
            # owner to delete — and the retry that follows routes through the
            # owner's own command instead of re-writing behind it.
            raise await _refusal_after_undoing(refusal, session_dir, written, intent)
        if await asyncio.to_thread(_latest_entry_id, session_dir) == entry:
            break
        # STILL REACHED, but no longer by a peer using this module: the lock
        # above serialises every external writer, so a mismatch now means a
        # writer that does not take it (a live session's own persist) landed on
        # top of us. Rebasing on ITS snapshot is what makes the retry safe — the
        # mutation is re-applied to the newer base, so the row is added once
        # against that base rather than appended again on top of our own write.
        if attempt:
            # ``wrote=True``: attempt 0's snapshot is on disk and this refusal does
            # NOT roll it back (the row may be in effect, in the base a peer
            # carried), so the journal must record this one rather than release it
            # — a retry would otherwise append the row a second time. The copy says
            # reconcile rather than retry blind, because that is what works here.
            raise WakeWriteError(
                "The conversation changed while your change was being applied. "
                "Reconcile its wake list before retrying.",
                status=STATUS_CONFLICT,
                code="wake_write_conflict",
                wrote=True,
            )
        logger.info("wake write for %s was overtaken by another writer; retrying", session_id)
    return rows, wake_id, due


def _is_ours(row: WakeSchedule, intent: str, our_version: WakeSchedule) -> bool:
    """Whether ``row`` is the row THIS write made — whatever its fields now say.

    By the request id when this write has one: that is what the field exists for,
    and it is the only test that survives the session's own firing advancing
    ``next_due_at``/``fired_count`` on the row it just absorbed (round 4, R9 —
    content comparison read that as "not mine", the loop re-mutated under a fresh
    id, and one request id left two rows for one intent).

    Without a request id (the agent's tool, the CLI) the fallback is the handle
    this write allocated plus the message: a handle alone can be reused after a
    cancel, and the message is what says the row is this intent's rather than a
    later row that happens to hold the same ``w`` number.
    """
    if intent:
        return row.request_id == intent
    return row.id == our_version.id and row.message == our_version.message


def _survives(
    row: WakeSchedule,
    ours: dict[str, WakeSchedule],
    before_by_id: dict[str, WakeSchedule],
    intent: str,
) -> bool:
    """Whether ``row`` is still something of THIS request, in the list just written.

    The question `_roll_back`'s return answers, asked of one row. A row THIS WRITE
    WROTE counts when :func:`_is_ours` says so — and "wrote" is `ours` minus the
    base, because ``ours`` holds the whole list we wrote, most of which we merely
    inherited. ANY row counts when it carries this request's identity, which is what
    catches a row of the same request under a different row id: the one shape the
    previous predicate could not see, and the reason it could never be False.
    """
    written = ours.get(row.id)
    if written is not None and written != before_by_id.get(row.id):
        return _is_ours(row, intent, written)
    return bool(intent) and row.request_id == intent


def _settled(
    before: list[WakeSchedule],
    after: list[WakeSchedule],
    latest: list[WakeSchedule],
    intent: str,
) -> tuple[str, int | None] | None:
    """This request's row and due instant, if ``latest`` already carries its change.

    THE IDEMPOTENCE THAT STOPS A RETRY FROM DUPLICATING A ROW, and it is asked on
    EVERY attempt — including the first — because with an identity it needs no
    history: a row carrying this request's id can only have come from this request,
    so "did my write land?" is one exact lookup with no reference to what the row
    looked like when we wrote it. That also settles the retry of an arm whose claim
    the journal RELEASED: the row is still standing, so the retry reports APPLIED
    with the same ``wake_id`` instead of arming a second one — and it does so
    before the mutation runs, so the 16-schedule cap cannot refuse a request that
    already succeeded, and no second id slot is consumed.

    A write with NO identity (the agent's tool, the CLI) keeps the earlier,
    content-based question, and only from the second attempt on — where this call
    has a base and an appended list of its own to compare: a row this write added
    and a value this write set are "landed" if the base still holds them; a row it
    cancelled is landed if the base no longer has it. Round 3's cancel-404 (the
    peer's snapshot had already absorbed the cancel, so re-applying raised
    not-found) is why the absence half exists at all.

    WHY "ALREADY LANDED" IS APPLIED RATHER THAN A FRESH RACE. The only writer that
    can carry our row into a NEW snapshot is one that read the transcript after our
    append, which means it holds our row in its own state — for a live session that
    is its in-memory list, so the wake is armed and will fire, and refusing it
    would be a false negative the owner's next persist would undo anyway.

    An edge worth naming: an identity match is the WHOLE answer, so a second POST
    that reuses a request id with a DIFFERENT body reports the first one's row
    rather than arming the new request. That is the same contract the receipt
    journal already gives that id (it replays the recorded outcome without looking
    at the body), so the two agree rather than diverge.
    """
    if intent:
        for row in latest:
            if row.request_id == intent:
                return row.id, row.next_due_at
        # ABSENT IS AN ANSWER TOO: the wake really was lost, so the mutation runs
        # afresh below and re-arms it. Deliberately not falling through to the
        # content question — an identity match is the only thing that can prove
        # this request's row is there, which is the whole point of the field.
        return None

    latest_by_id = {row.id: row for row in latest}
    before_by_id = {row.id: row for row in before}
    settled_id = ""
    settled_due: int | None = None
    for row in after:
        current = latest_by_id.get(row.id)
        if current is None:
            return None
        if row.id in before_by_id:
            # A row this write CHANGED: the value IS the intent, so the base has to
            # hold the value we set. A different one means a peer changed it again
            # and re-applying would overwrite that change.
            if current != row:
                return None
        elif not _is_ours(current, "", row):
            # A row this write CREATED and cannot identify by request id: the SAME
            # test the rollback uses (id plus message), deliberately not full
            # equality (review round 5, MINOR-5). Equality read a peer's persist
            # that merely RE-TIMED our row as "not landed", so the loop armed a
            # second row for one intent on its own verify retry — one module
            # answering one question two ways is what produced it.
            return None
        settled_id, settled_due = row.id, current.next_due_at
    for row in before:
        if row.id in {changed.id for changed in after}:
            continue
        if row.id in latest_by_id:
            return None
        settled_id, settled_due = row.id, None
    return (settled_id, settled_due) if settled_id else None


async def _refusal_after_undoing(
    refusal: WakeWriteError,
    session_dir: Path,
    written: tuple[list[WakeSchedule], list[WakeSchedule]] | None,
    intent: str,
) -> WakeWriteError:
    """The refusal to raise once this request's own write has been undone.

    ``written`` is None when nothing was appended (the attempt-0 case), and the
    refusal is returned untouched. Otherwise the rollback is what lets the refusal
    be RELEASED by the journal — a retry re-runs and can succeed, which is what
    these sentences promise — so when the rollback reports that the list it wrote
    still carries something of this request, the refusal has to say so instead:
    ``wrote=True`` makes the journal record it, and the sentence becomes one that
    never claims nothing was written (round 4, MINOR-3), because here that is
    exactly what cannot be guaranteed.
    """
    if written is None:
        return refusal
    try:
        complete = await _roll_back(session_dir, before=written[0], after=written[1], intent=intent)
    except Exception:  # noqa: BLE001 - any failure here leaves our bytes unproven
        logger.exception("could not undo the wake write for %s", session_dir.name)
        complete = False
    if not complete:
        return WakeWriteError(
            UNSETTLED_MESSAGE, status=refusal.status, code=refusal.code, wrote=True
        )
    return refusal


async def _roll_back(
    session_dir: Path,
    *,
    before: list[WakeSchedule],
    after: list[WakeSchedule],
    intent: str,
) -> bool:
    """Append a snapshot that undoes THIS write, preserving everyone else's.

    ``before`` is the base this write read and ``after`` the list it appended, so
    the difference between them is exactly this write's effect: rows it added or
    changed (undo them) and rows it removed (put them back). Everything the latest
    snapshot holds that is not this write's effect is kept as it stands — an owner
    that persisted a DIFFERENT change meanwhile must not have it reverted by a
    rollback that only ever meant to withdraw one arm.

    IDENTITY, NOT CONTENT, and that is the round-4 correction (R9): the previous
    rule was "keep any row that differs from what we wrote", so a re-timed copy of
    OUR OWN row was kept, the undo reported itself as done, and the released retry
    armed a second one. A row this write CREATED is now dropped whenever it is ours
    by :func:`_is_ours`, whatever the session's own firing has since done to its
    fields. A row this write EDITED is reverted only while the value standing is
    the one we wrote — a newer value means a peer changed it after us, and
    reverting that would be clobbering a change this request never made. A row this
    write CANCELLED comes back AT THE INDEX IT HELD (round 4, MINOR-1): a refused
    cancel must not reorder a list it was supposed to leave alone, which is the
    same principle ``edit_wake`` keeps when it edits in place.

    An APPEND rather than an edit of the file: the transcript is append-only, and
    rewriting it would delete a row a writer that does not take this lock appended
    between our two moments.

    RETURNS WHETHER THE LIST IT JUST WROTE STILL CARRIES ANYTHING OF THIS REQUEST, and
    that is all it returns — a check on the bytes this call wrote, never a proof
    about the durable state afterwards. The paragraph here used to claim the two
    differed "exactly when the rollback wrote a list that still carries our row",
    which is unsatisfiable (review round 5, MINOR-4): the drop rule is TOTAL over
    this request's rows — the ones this write wrote, when they are ours, and any row
    bearing this request's identity that the base did not hold — so today no path
    makes it False without raising, and a peer that restores a row after the return
    is beyond any check this module can make (the recorded Q15 case).

    It is kept, and named as what it is, because ``False`` IS reachable and the case it
    marks is the dangerous one: a row bearing this request's identity that the BASE
    already held is left standing deliberately (a live owner holding it in memory
    would restore it anyway — see the drop rule), which happens when a peer restores
    our row between the settle that found the base clear and this rollback. There the
    truthful answer is "something of this request is still in effect", so the refusal
    is RECORDED and a retry replays instead of arming a duplicate. It is also the
    tripwire for an edit that breaks the drop rule's totality: with that clause
    removed, this returns ``False`` on the shape whose row of ours sits under another
    row id, which is exactly how it was caught in review.
    """
    latest = await asyncio.to_thread(_read_rows, session_dir)
    ours = {row.id: row for row in after}
    before_by_id = {row.id: row for row in before}
    restored: list[WakeSchedule] = []
    for row in latest:
        our_version = ours.get(row.id)
        if our_version is None:
            # A row this write did not write. It is still THIS REQUEST'S if it
            # carries this request's identity and the base we read did not hold it
            # — the same request's row that an earlier attempt, or an owner
            # re-persisting a list that had absorbed it, put back. Dropped for the
            # reason the rows we wrote are dropped: this request is being refused,
            # so nothing of it may stand. A row the BASE held is left alone: this
            # write did not put it there, and a live owner holding it in memory
            # would only restore it on its next persist, so removing it would be a
            # write against a writer the lock does not serialise.
            if intent and row.request_id == intent and row.id not in before_by_id:
                continue
            restored.append(row)
            continue
        if row.id not in before_by_id and _is_ours(row, intent, our_version):
            continue  # our own creation: dropped, re-timed, edited or not
        previous = before_by_id.get(row.id)
        if previous is not None and previous != our_version and row == our_version:
            restored.append(previous)  # our edit, still standing: undone
            continue
        restored.append(row)  # a peer's own version of it: kept
    # Rows this write removed come back where they were, not at the end.
    present = {row.id for row in restored}
    for index, row in enumerate(before):
        if row.id in present or row.id in ours:
            continue
        restored.insert(min(index, len(restored)), row)
    complete = not any(_survives(row, ours, before_by_id, intent) for row in restored)
    if restored != latest:
        await _append(session_dir, restored)
    return complete


async def _refuse_if_owned(config_dir: Path, session_id: str) -> None:
    """Refuse to append to a transcript a runtime process owns.

    THE LOST-REMINDER GUARD, and it is here rather than only in the route
    because the route's decision is a moment old by the time this runs. An owner
    that appears in between — the desktop run pane's ``/watch`` heartbeat warming
    the visible session, a second app opening the conversation, the supervisor
    engaging one — loads its schedule list BEFORE this append and republishes
    that stale list on its next persist (a fire, a retirement, the agent's own
    ``wake`` tool), which deletes the row from both the transcript and the index
    while the supervisor skips any session with a live record. Measured: a 200
    with ``index_written: true``, then an empty transcript snapshot and a removed
    index entry once the owner persisted.

    THE WINDOW THIS LEAVES, stated rather than implied, and it is NOT the one
    round 2 described. The guard is called twice — before the append and again
    after it, both inside the per-session lock (see
    :func:`_mutate_locked`) — and an owner seen by the second call has the append
    rolled back before the refusal is raised. So the loss needs an owner that
    claims the session *after* the second call, and such an owner cannot have
    loaded a stale list: ``session_factory`` acquires the lease (which is what
    writes ``.session.pid``, the marker this predicate reads) "at the shared
    construction boundary, **before transcript creation**"
    (``session_factory.py:1951-1968``), and the session loads its schedules from
    that transcript afterwards (``session.py:2418``). Claiming after our check
    therefore means loading after our append, which means seeing the row we
    wrote. **The reasoning rests on that ordering and no other**: a future
    change that lets a runtime read its transcript before it claims would put a
    stale list on the far side of both checks and reopen this.

    WHY ``_persist_wake_schedules`` IS NOT ALSO PUT UNDER THE LOCK. It would not
    help: the owner's list is stale from the moment it loaded, so serialising the
    two writers cannot reveal that, and the loss is a CONTENT fact rather than a
    timing one (round 1's review says the same). What closes it is the pair of
    guard calls plus the rollback above, which is what makes the live session's
    own persist a writer whose effects we can always detect and undo.

    OWNED MEANS ANY LIVE PROCESS, NOT ONLY A DIALABLE ONE, and that distinction
    is the whole of review round 2's R6. Asking ``_has_live_runtime`` asks "does a
    *dialable discovery record* exist", which is the SUPERVISOR's question (it
    delivers into a record). An owner that exists as a live process holding the
    session but publishes no usable record — a mixed-version rollout, a
    registrant that failed to start, the rebind window where the record still
    names the previous session — is owned all the same, and the tree already says
    so twice: ``session_lease.acquire_session_lease`` refuses a new claim while a
    live legacy pid mirror exists ("It is still authoritative when live or
    uncertain"), and this route's own ``_rollback_created`` asks the wider
    question to decide a directory is not its own. Predicating on the narrower
    one appended the row behind such an owner and answered 200 with
    ``index_written: true`` — the lost-reminder leak this guard exists to
    prevent, one layer down. So the test is the owner PID
    (``find_runtime_record(...)[1]``), and the record is not part of it.

    ``[1]`` IS NOT "nothing else holds this pid", and round 3's minor is that
    claim: a marker left by a crash keeps naming a number the OS may since have
    recycled onto an unrelated process, and that stranger answers this guard
    forever. There is no bound to give it — a lease-style deadline exists to hand
    ownership to a COMPETITOR, and this writer competes with nobody (it has no
    claim to take) — so the honest fix is the copy, which is why the two states
    below get different sentences instead of one sentence that is false in one of
    them.

    BOTH OWNER STATES ARE REFUSED, and the order below is load-bearing: a WEDGED
    owner (alive, heartbeat stale, lease held so no engage can succeed) is asked
    FIRST because it is also a live pid — the owner question first would shadow
    the wedged answer with the generic one and lose the sentence that names the
    next step.
    """
    from local_operator.mobile.attach_client import (
        dialable_record_exists,
        find_runtime_record,
    )
    from local_operator.wakes.supervisor import wedged_runtime

    wedged = await asyncio.to_thread(wedged_runtime, Path(config_dir), session_id)
    if wedged is not None:
        raise WakeWriteError(WEDGED_MESSAGE, status=STATUS_OWNER_BUSY, code="wake_owner_wedged")
    # Off the loop: this reads ``.session.pid`` and, for the zombie proof, forks a
    # ``ps`` — the same probe the attach paths use, so this guard and the process
    # that would take the session agree about who is holding it.
    _record, owner = await asyncio.to_thread(find_runtime_record, Path(config_dir), session_id)
    if owner is None:
        return
    # TWO STATES, ONE PREDICATE, DIFFERENT SENTENCES (review round 3, R6 minor).
    # The pid mirror is the authority on "someone holds this transcript", and it
    # outlives the process that wrote it: a crash plus pid reuse leaves a live
    # number pointing at a stranger, which no amount of retrying will resolve and
    # which the "open in a running session" sentence describes falsely. A
    # dialable record is the difference the client can act on, and
    # ``dialable_record_exists`` is the tree's own answer for it.
    dialable = await asyncio.to_thread(dialable_record_exists, Path(config_dir), owner)
    if dialable is None:
        # A THIRD STATE, and the one the helper's own docstring warns about: the
        # registry could not be read, so "no usable record" and "a record I cannot
        # see" are indistinguishable from here. Both other sentences would assert a
        # cause this code has not established (review round 4, MINOR-2).
        raise WakeWriteError(
            OWNER_REGISTRY_UNREAD_MESSAGE.format(pid=owner),
            status=STATUS_OWNER_BUSY,
            code="wake_owner_present",
        )
    if dialable:
        raise WakeWriteError(
            "This conversation is open in a running session, which owns its schedules. "
            + NOTHING_WRITTEN
            + " Retry in a moment, or change them from that session.",
            status=STATUS_OWNER_BUSY,
            code="wake_owner_present",
        )
    raise WakeWriteError(
        OWNER_UNKNOWN_MESSAGE.format(
            pid=owner, marker=Path(config_dir) / "sessions" / session_id / ".session.pid"
        ),
        status=STATUS_OWNER_BUSY,
        code="wake_owner_present",
    )


def _refusal(outcome: Mapping[str, Any]) -> WakeWriteError:
    """A ``build_wake_schedule`` error as a refusal. Malformed (an unparseable
    duration, a bound on a one-shot) is the caller's input to fix and reads as
    a 422; everything else — the cap, a past time — is a conflict with the
    session's existing state and reads as a 409."""
    malformed = bool(outcome.get("malformed"))
    return WakeWriteError(
        str(outcome.get("error") or "the wake could not be scheduled"),
        status=STATUS_INVALID if malformed else STATUS_CONFLICT,
        code="wake_invalid" if malformed else "wake_refused",
    )


def _unknown_wake(wake_id: str, existing: list[WakeSchedule]) -> WakeWriteError:
    known = ", ".join(schedule.id for schedule in existing) or "none"
    return WakeWriteError(
        f"No wake schedule with id '{wake_id}' (known: {known}).",
        status=STATUS_WAKE_NOT_FOUND,
        code="wake_not_found",
    )


# ---------------------------------------------------------------------------
# Filesystem steps (each run off the event loop by its caller)
# ---------------------------------------------------------------------------


def _transcript(session_dir: Path):
    from local_operator.session.transcript import Transcript

    # ``defer_materialise``: a read must not create a session directory, and
    # the append path recreates a vanished one itself. This module never
    # invents a session — only the create+arm route does, and it does so
    # before calling here.
    return Transcript(session_dir, defer_materialise=True)


def _read_rows(session_dir: Path) -> list[WakeSchedule]:
    """The session's schedules, from the TRANSCRIPT's latest snapshot.

    The index is deliberately not consulted even as a fallback: it lags, it is
    absent for a session that has never been opened, and the append below
    REPLACES the list — so a stale base is how a live reminder gets cancelled
    by an unrelated edit.

    Read through :func:`read_latest_custom_entry`, NOT ``Transcript``. Both
    answer the same one-row question, but ``Transcript.__init__`` parses the
    WHOLE journal to do it: measured 8.84 s and 885 MB of traced peak on the
    operator's 262 MB transcript, and this function runs twice per arm attempt
    (plus once more on the rollback settle), i.e. 2-4 whole-journal parses to
    answer one question about one row. The backward reader is 17 ms and is the
    same reader the desktop's own checkpoint lookups use.
    """
    entry = read_latest_custom_entry(session_dir, WAKE_SCHEDULES_CUSTOM_TYPE)
    if entry is None:
        return []
    details = entry.payload.get("details")
    if not isinstance(details, Mapping):
        return []
    raw_rows = details.get("schedules")
    if raw_rows is None:
        return []
    if not isinstance(raw_rows, list):
        # An entry that EXISTS and is not a list is corrupt, not empty, and
        # guessing there would mean treating someone's armed reminders as
        # "none" and then writing that emptiness back.
        raise WakeWriteError(
            "The session's stored wake schedules are unreadable; reopen the session and retry.",
            status=STATUS_CONFLICT,
            code="wake_store_corrupt",
        )
    rows: list[WakeSchedule] = []
    for raw in raw_rows:
        try:
            rows.append(WakeSchedule.model_validate(raw))
        except Exception:  # noqa: BLE001 — one bad row costs one row, never the list
            logger.warning("dropping malformed persisted wake schedule: %r", raw)
    return rows


def _latest_entry_id(session_dir: Path) -> str | None:
    # The same bounded reader as ``_read_rows``, and for the same reason: this
    # asks one row's id, and the whole-journal parse it used to pay for is the
    # cost the arm path's rollback settle (``:720``) also pays.
    entry = read_latest_custom_entry(session_dir, WAKE_SCHEDULES_CUSTOM_TYPE)
    return entry.id if entry is not None else None


async def _append(session_dir: Path, rows: list[WakeSchedule]) -> str:
    """Append the new full-list snapshot and return its entry id.

    The ONLY step allowed to fail the write: everything after it is derived
    from what this put on disk.

    The transcript is CONSTRUCTED on the worker thread as well as appended from
    the loop. ``Transcript.__init__`` reads and parses the whole journal
    synchronously, which on a 103 MB transcript is seconds of event loop and
    the reason every other step here is already threaded (review round 1, R3).
    """
    transcript = await asyncio.to_thread(_transcript, session_dir)
    entry = await transcript.append_custom(
        WAKE_SCHEDULES_CUSTOM_TYPE,
        {"schedules": [schedule.model_dump() for schedule in rows]},
    )
    return entry.id


def _read_index_entry(config_dir: Path, session_id: str) -> dict[str, Any]:
    from local_operator.wakes.store import read_entry

    try:
        return read_entry(Path(config_dir), session_id) or {}
    except Exception:  # noqa: BLE001 — an unreadable entry preserves nothing
        logger.warning("could not read the wake index entry for %s", session_id, exc_info=True)
        return {}


def _write_index(
    config_dir: Path,
    session_id: str,
    cwd: str,
    rows: list[WakeSchedule],
    preserve: Mapping[str, Any],
):
    from local_operator.wakes.store import write_entry

    # ``preserve`` carries the keys this module does not own — ``stopped_at``
    # (a stopped session's wakes stay parked through an arm; dropping it would
    # un-park the whole session as a side effect of scheduling something) and
    # the session's own ``last_fired_at``/``last_attempt_at`` lateness stamps.
    return write_entry(
        Path(config_dir),
        session_id,
        cwd=cwd,
        schedules=rows,
        preserve=preserve,
    )


def _resolved_cwd(session_dir: Path, previous: Mapping[str, Any]) -> str:
    """Which directory the supervisor should start a runtime in.

    The index entry's own value first — it is what the supervisor reads today
    and must not be rewritten to something else by an unrelated arm — then the
    desktop marker a created session carries, then the session directory as a
    last resort (what the CLI did before this module existed)."""
    from local_operator.session.retention import DESKTOP_MARKER_NAME

    existing = previous.get("cwd")
    if isinstance(existing, str) and existing:
        return existing
    try:
        marker = json.loads((session_dir / DESKTOP_MARKER_NAME).read_text(encoding="utf-8"))
        cwd = marker.get("cwd")
        if isinstance(cwd, str) and cwd:
            return cwd
    except (OSError, ValueError):
        pass
    return str(session_dir)


def _ensure_supervisor(config_dir: Path) -> str:
    try:
        from local_operator.wakes.install import ensure_supervisor_installed

        return ensure_supervisor_installed(Path(config_dir)).reason
    except Exception:  # noqa: BLE001 — best-effort, exactly as the persist path treats it
        logger.debug("wake supervisor install failed", exc_info=True)
        return ""
