"""Pre-spindown drain for a pool member: the flush, the barrier and its bound.

WHY THIS MODULE EXISTS. A pool member is a device that is expected to disappear
(``mesh-compute-pool.md`` §3.5), and every turn it ran lives on it and nowhere
else until something copies it out. R22 is that something: before the pod powers
off, each session placed on it is FLUSHED to its home device — the copy is
brought up to a frontier the member chose, the destination keeps its own copy and
the session keeps its own id — and the pod does not power off until every flush
is acked. §6.2 states the order and calls it the barrier, and this module is that
barrier and nothing else.

THE PRIMITIVE IS MOBILITY'S, NOT THIS MODULE'S (§6.1). ``net_sync``'s ``plan`` /
``fetch`` / ``verify`` already copy a session to a chosen frontier, and
``sync.sync_from`` is their client half. A ``flush`` is that call with a
different purpose string and a different destination: home's REAL session
directory rather than a replica or a staging directory, because a flush keeps the
identity (§6.5 rule 1) and R22's whole promise is ``/resume <id>`` at home. The
one thing this module must never do is invent a cursor: the position is
``sync``'s ``(prefix_bytes, prefix_digest, frontier)`` over
``_safe_region``'s last complete line, because ``Transcript.compact_file``
rewrites the transcript with ``os.replace`` and a byte offset would then describe
somebody else's row boundary (mobility §7.3).

NO FENCE, ON PURPOSE (§6.1, OQ6). A flush copies from a source that is still
writing, which is why it cannot take ``exclusive-move-v1`` — and why the whole
guarantee rests on two other rules instead: the source serves only through its
last complete newline (so a torn tail is a divergence point, never corruption),
and **no second writer may exist at the destination** while the member is alive.
The second rule is a LAUNCH guard, not a sync one — it lives in
``session/placement.py`` (``pool_guard_refusal``) because it must refuse an
engage, not a copy.

THE BARRIER IS A DEADLINE, NEVER AN UNBOUNDED WAIT (§6.2). A home laptop that is
asleep must not keep a billable pod alive, and a pod that waits forever is a pod
billing for nothing. Both bounds are parameters here and are recorded in the
outcome, so the caller's report says which one it hit.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from local_operator.atomic_files import atomic_write_text
from local_operator.network.types import MemberLifecycle
from local_operator.session.placement import SessionPlacement, read_stamp, write_stamp
from local_operator.session.runtime.types import PRE_SPINDOWN_DRAIN_CAUSE

logger = logging.getLogger(__name__)

#: The pool's one cadence addition (§6.3): how often a member flushes sessions
#: whose bytes moved since their last flush. An un-drained crash (spot reclaim,
#: ``kill -9``) then costs at most one tick plus the in-flight turn, which is the
#: difference between "the tail is gone" and "the conversation is gone".
POOL_SYNC_TICK_S = 60.0

#: How long the drain waits for a turn in flight before cutting it at the turn
#: boundary (§3.6 step 2). Equal to ``types.SIGNAL_DRAIN_S`` on purpose: the
#: leaving phrase a viewer is shown for this trigger is the
#: ``LEAVING_ON_SIGNAL`` one, and its parenthetical names a 2-minute bound.
#: ``mesh-compute-pool.md`` §9.4 calibrates it from measurement; until that
#: measurement exists this is the design's recommended value.
DEFAULT_DRAIN_GRACE_S = 120.0

#: How long the member may wait for flush acks before it gives up and powers off
#: as ``lost`` (§6.2 step 4). The design's recommended value.
DEFAULT_SYNC_DRAIN_DEADLINE_S = 900.0

#: How often a bounded wait re-checks. Small enough that a turn settling is
#: noticed promptly, large enough that a 15-minute barrier is ~900 wakes.
DRAIN_POLL_S = 1.0


# ---------------------------------------------------------------------------
# The audit rows (§3.4, and mesh-incident-response.md §4.3's one log)
# ---------------------------------------------------------------------------


def _audit(event: str, root: Path | None, **fields: Any) -> None:
    """One row in the network audit log. Never raises: a record is not a gate.

    The literals at the call sites are what ``test_audit.py``'s drift guard walks,
    so each event name is spelled as a literal there rather than through a module
    constant — an unregistered name renders with NO ``detail`` at all, silently.
    """
    try:
        from local_operator.network.audit import write_event

        write_event(root, event=event, **fields)
    except Exception:  # noqa: BLE001 — an audit loss cannot undo a drain
        logger.debug("pool drain: could not record %s", event, exc_info=True)


# ---------------------------------------------------------------------------
# The flush (R22), at HOME
# ---------------------------------------------------------------------------


def flush_session_home(
    config_dir: Path,
    session_id: str,
    *,
    ask: Callable[[dict[str, Any]], dict[str, Any]],
    member_device: str,
    rewrite_placement: bool = False,
    sync_reason: str = "cadence",
) -> dict[str, Any]:
    """Bring home's copy of ``session_id`` to the member's frontier. Same id.

    RUNS ON HOME, not on the member: ``ask`` is a ``net_sync`` request toward the
    member (its relay's ``plan``/``fetch``/``verify`` handler, the shape
    ``sync._plan`` serves and ``test_sync_copy_set._ask`` reproduces in process),
    which is the direction the primitive is built in — a destination pulls a
    frontier, and the source cannot tell it to append. **The link is the relay's**
    (``sync.request_sync``'s rule: one process speaks the mesh), so the caller that
    owns the link supplies ``ask``; nothing here dials a peer itself.

    ``into`` is home's REAL session directory rather than a replica or a staging
    path, and that is the one difference from a move: §6.5 rule 1 keeps the id, so
    the ordinary resume path finds the conversation exactly where it looked before.
    ``attachments_root`` is the install's shared store — the same knob a move sets
    (``mobility``), so a flushed session does not grow a private copy of every blob
    it already has.

    Raises ``sync.SyncRefused`` (or an OSError) when the copy cannot be made; the
    caller decides whether that is a retry inside the barrier or a pending entry.

    ``rewrite_placement`` IS THE DRAIN'S, AND ONLY THE DRAIN'S (§6.5 rule 4). A
    cadence tick is a copy and nothing else (§6.3): rewriting home's stamp to
    ``local`` and clearing ``pool_member`` is the statement "the member has left",
    and a tick that said it while the member is still ``active`` would switch off
    :func:`pool_guard_refusal` — permanently — with the very copy it exists to make
    safe (review round 1, BLOCKER-1). ``sync_reason`` rides the audit row for the
    same reason: ``pre_spindown`` is a fact about a drain, ``cadence`` about a tick.
    """
    from local_operator.network import sync

    result = sync.sync_from(
        config_dir,
        session_id,
        ask=ask,
        owner_device=member_device,
        into=sync.session_dir(config_dir, session_id),
        attachments_root=Path(config_dir),
        purpose=sync.COPY_PURPOSE_FLUSH,
    )
    # PLACEMENT LAST, so a crash between the bytes and the stamp leaves a copy that
    # the launch guard still refuses (mode ``pool`` + a live member), never a
    # ``local`` stamp over a half-copied transcript.
    rewritten = _rewrite_placement_local(config_dir, session_id) if rewrite_placement else False
    _audit(
        "session_sync_completed",
        config_dir,
        subject=session_id,
        network_id=_network_of(config_dir, session_id),
        detail={
            "peer": member_device,
            "reason": sync_reason,
            "mode": "flush",
            "bytes": int(result.get("bytes") or 0),
            "placement": "local" if rewritten else "",
        },
    )
    return result


def _network_of(config_dir: Path, session_id: str) -> str:
    """The network the session's stamp names, or ``""``. For the audit row only."""
    stamp = read_stamp(config_dir, session_id)
    if stamp is None:
        return ""
    return stamp.placement.network_id or stamp.network_id


def _rewrite_placement_local(config_dir: Path, session_id: str) -> bool:
    """Rewrite home's stamp to ``local`` and audit it (§6.5 rule 4).

    ``stamp_revision`` is bumped because it is the idempotency key (§4 point 4): a
    relocation retried after a lost ack writes a NEWER revision rather than a
    duplicate, and a reader resolving "which stamp wins" has an answer. An empty
    or already-local stamp is left alone — a flush of a session whose stamp never
    named a pool is not a placement change, and rewriting it would be a lie in the
    one row an operator reads to find out where a conversation went.
    """
    stamp = read_stamp(config_dir, session_id)
    if stamp is None or stamp.placement.mode == "local":
        return False
    before = stamp.placement
    stamp.placement = SessionPlacement(
        mode="local",
        network_id=before.network_id or stamp.network_id,
        home_device=before.home_device or stamp.home_device,
        policy=before.policy,
        stamp_revision=before.stamp_revision + 1,
    )
    # THE MEMBER POINTER GOES WITH THE MODE. Leaving it set would keep
    # ``pool_guard_refusal`` looking at a member for a session that is now local —
    # and the guard's mode check is what makes a resumed home copy cheap.
    stamp.pool_member = ""
    write_stamp(config_dir, stamp)
    _audit(
        "session_placement_changed",
        config_dir,
        subject=session_id,
        network_id=stamp.network_id,
        detail={
            "placement_before": before.mode,
            "placement_after": "local",
            "policy": before.policy,
            "reason": "drain",
        },
    )
    return True


def make_flush(
    config_dir: Path,
    *,
    ask_for: Callable[[str], Callable[[dict[str, Any]], dict[str, Any]]],
    member_device: str,
    drain: bool = False,
) -> Callable[[str], str]:
    """A coordinator ``flush`` backed by real transfers: the acked frontier, or ``""``.

    ``ask_for`` supplies the member's ``net_sync`` frame handler for one session —
    the relay's, in production, which is why it is a callback and not a call this
    module makes. Every failure is an EMPTY answer rather than a raise: inside the
    barrier a refused copy is one more polling round, and an exception escaping
    here would abort a drain that should have kept trying until its deadline.

    THE RETURN VALUE IS THE FRONTIER THE COPY WAS VERIFIED TO, not a bare ``True``:
    the cadence records exactly that digest as "home holds this much", so bytes the
    member appends while the ack is in flight stay dirty for the next tick instead
    of being re-hashed away (review round 1, MINOR-2).

    ONE BOOLEAN, NOT TWO ARGUMENTS: §6.5 rule 4's placement rewrite and the
    ``pre_spindown`` audit reason travel together — only a drain may say the member
    left — so there is no spelling here that rewrites home's stamp without also
    recording why. ``drain=False``, the safe half, is what a cadence tick wants.
    """

    def flush(session_id: str) -> str:
        try:
            result = flush_session_home(
                config_dir,
                session_id,
                ask=ask_for(session_id),
                member_device=member_device,
                rewrite_placement=drain,
                sync_reason="pre_spindown" if drain else "cadence",
            )
        except Exception:  # noqa: BLE001 — see the docstring
            logger.debug("pool drain: flush of %s was not acked", session_id, exc_info=True)
            return ""
        cursor = result.get("cursor")
        return str((cursor if isinstance(cursor, dict) else {}).get("prefix_digest") or "")

    return flush


# ---------------------------------------------------------------------------
# The cadence (§6.3), and what "unsynced" means without a second cursor
# ---------------------------------------------------------------------------


def _region_digest(directory: Path) -> str:
    """``sha256`` of the transcript's SAFE REGION, or ``""`` when it has none.

    The region is ``sync._safe_region``'s — the bytes up to and including the last
    complete line — so this digest is comparable, field for field, with the
    ``prefix_digest`` a flush recorded. Reading the file whole is deliberate: the
    only alternative is a backward scan for a newline that has to grow its window
    until it finds one, and a single tool-result row can be megabytes long.
    """
    from local_operator.network import sync

    try:
        payload = (Path(directory) / sync.TRANSCRIPT_NAME).read_bytes()
    except OSError:
        return ""
    return sync.sha256_bytes(sync._safe_region(payload))


#: Where a member records "home holds at least this much of this session".
#:
#: OUTSIDE ``sessions/`` and outside ``network/replicas/``, for two different reasons:
#: nothing that lists, resumes or sweeps a session may see drain bookkeeping, and a
#: replica directory is the copy THIS device holds — the opposite direction from
#: "what the other device has pulled". One small file per session, like
#: ``network/pending-move.json``'s neighbour set, so one ``rm -rf`` of the mesh state
#: takes it with it.
FLUSHED_DIRNAME = "pool-flushed"
FLUSHED_NAME = "flushed.json"


def flushed_path(config_dir: Path, session_id: str) -> Path:
    """One session's marker path. Through ``store.safe_component``: the id came from
    a peer's stamp on the way in, and this is a path built from it."""
    from local_operator.network import store, sync

    return sync.network_dir(config_dir) / FLUSHED_DIRNAME / store.safe_component(session_id)


def record_flushed(config_dir: Path, session_id: str, digest: str) -> str:
    """Record the frontier digest a flush ACKED. Returns it, or ``""`` when empty.

    THE ARGUMENT IS THE INVARIANT: it is the safe region's sha256 the copy was
    verified against, so a later comparison answers "did I write anything since"
    without a second cursor and without trusting a byte offset — the same reason
    ``sync``'s own cursor is a digest (§7.3). This used to re-hash home's transcript
    after the ack returned, which recorded bytes that landed BETWEEN the verified
    frontier and that read as synced; the next tick then skipped exactly them
    (review round 1, MINOR-2). Nothing here re-reads the file.

    Written after the ack, never before: a marker on a failed flush would make the
    next tick skip the very bytes that were lost.
    """

    if not digest:
        return ""
    target = flushed_path(config_dir, session_id) / FLUSHED_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        target,
        json.dumps({"session_id": session_id, "prefix_digest": digest}, sort_keys=True),
        hidden=True,
    )
    return digest


def flushed_digest(config_dir: Path, session_id: str) -> str:
    """The digest the last acked flush recorded, or ``""`` when there is none.

    A row that cannot be read is ``""`` — which makes the session DIRTY, the
    direction that re-sends rather than skips.
    """
    try:
        raw = (flushed_path(config_dir, session_id) / FLUSHED_NAME).read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, ValueError):
        return ""
    return str(payload.get("prefix_digest") or "") if isinstance(payload, dict) else ""


def unsynced(config_dir: Path, session_id: str) -> bool:
    """Whether this session holds bytes the last acked flush did not carry.

    THE COMPARISON IS THE INVARIANT: a flush records the digest of the region it
    brought over, so "the region's digest now differs" means either bytes were
    appended or the transcript was rewritten (a compaction) — both cases a flush
    must repeat. A session with no transcript has nothing to carry and is not
    dirty; a session with no marker is dirty by construction (the marker is the
    only evidence it was ever copied).
    """
    from local_operator.network import sync

    digest = _region_digest(sync.session_dir(config_dir, session_id))
    if not digest:
        return False
    return digest != flushed_digest(config_dir, session_id)


def pool_sync_tick(
    config_dir: Path,
    session_ids: Iterable[str],
    *,
    flush: Callable[[str], str],
) -> tuple[str, ...]:
    """Flush every session whose bytes moved, and return the ones that acked.

    ONE PASS, called by whatever ticker owns this device's cadence every
    :data:`POOL_SYNC_TICK_S` — a loop here would be a second scheduler beside the
    relay's own, which is the shape "one owner per cadence" exists to prevent. An
    unchanged session costs one read and one string compare, and is SKIPPED rather
    than re-sent: a tick that re-flushed everything would spend the member's
    bandwidth on conversations nobody touched.

    ``flush`` MUST BE A CADENCE FLUSH (``make_flush``'s default; ``drain=False``).
    A tick is a copy and nothing else (§6.3): handing it a drain flush would rewrite
    home's stamp to ``local`` and lift the two-writer guard on a live member, which
    is the failure this split exists to make unspellable (review round 1,
    BLOCKER-1).
    """
    flushed: list[str] = []
    for session_id in session_ids:
        if not unsynced(config_dir, session_id):
            continue
        digest = flush(session_id)
        if not digest:
            continue
        record_flushed(config_dir, session_id, digest)
        flushed.append(session_id)
    return tuple(flushed)


# ---------------------------------------------------------------------------
# The bounded drain (§6.2, §3.6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrainOutcome:
    """What one drain did. Every field is evidence, none of it is a claim."""

    member_id: str
    #: Every session the member held when the drain started.
    sessions: tuple[str, ...]
    #: Sessions whose flush was acked.
    flushed: tuple[str, ...]
    #: Sessions still un-acked when the barrier deadline expired.
    pending: tuple[str, ...]
    #: Sessions whose in-flight turn the grace expired on, and which were cut.
    cut: tuple[str, ...]
    drained: bool
    barrier_expired: bool
    #: The member row could not be moved to ``expired``, so nothing here claims it was:
    #: the barrier completed but the durable record of it does not exist (§6.2 step 5).
    expire_failed: bool = False


@dataclass
class PoolDrainCoordinator:
    """One member's drain: announce, wait, flush, expire — each step bounded.

    THE COLLABORATORS ARE CALLBACKS BECAUSE THE DRAIN IS DEVICE-SIDE. This runs on
    the member and its four moving parts live elsewhere: which sessions are placed
    here (the catalogue), how to make a runtime commit to leaving (the relay's own
    signal path, with :data:`PRE_SPINDOWN_DRAIN_CAUSE`), whether a turn is still in
    flight (the runtime's record), and how to ask home to pull (``make_flush``).
    Naming them as parameters is what makes the barrier testable at all — a
    coordinator that reached into the relay could only be tested with one.

    ``sessions()`` is read ONCE, at the start: a session placed after the member
    entered ``draining`` cannot exist (the lifecycle is what refuses a placement),
    and re-reading would let a drain chase its own tail.
    """

    config_dir: Path
    member_id: str
    network_id: str
    sessions: Callable[[], Sequence[str]]
    #: Ask home to pull one session and answer the frontier it verified, or ``""``
    #: when the copy was not acked. ``make_flush(..., drain=True)`` in production:
    #: the placement rewrite (§6.5 rule 4) is part of the drain, so a cadence flush
    #: here would copy the bytes and leave home's stamp claiming a live member.
    flush: Callable[[str], str]
    #: Commit one placed session's runtime to leaving (§3.6 step 1). It takes the
    #: session and the CAUSE, because the cause is this module's to state and the
    #: runtime's leaving ladder keys on it (``types.PRE_SPINDOWN_DRAIN_CAUSE``):
    #: a caller that had to remember it would be a second place to get it wrong.
    signal: Callable[[str, str], None] = lambda _session_id, _cause: None
    #: Is a turn in flight for this session? ``False`` when unknown — an unknown
    #: turn is not a reason to hold a billable pod for the grace period.
    busy: Callable[[str], bool] = lambda _session_id: False
    #: Abandon the in-flight turn at its boundary. Called once per session the
    #: grace expired on, immediately before the journal row is closed.
    cut: Callable[[str], None] | None = None
    #: WHAT STARTED THIS DRAIN, from the design's closed set (§3.4): ``idle``,
    #: ``expiry``, ``user`` or ``error``. It rides the ``drain_started`` row because
    #: "why did the pod go away" is the first question an incident asks, and the
    #: coordinator cannot derive it (every one of the four reaches it as a call).
    trigger: str = "user"
    drain_grace_s: float = DEFAULT_DRAIN_GRACE_S
    sync_drain_deadline_s: float = DEFAULT_SYNC_DRAIN_DEADLINE_S
    poll_s: float = DRAIN_POLL_S
    #: Monotonic: every bound here is a duration, and a wall clock that steps must
    #: not be able to extend or truncate one.
    clock: Callable[[], float] = time.monotonic
    #: Wall clock, for the evidence timestamps only (the turn journal).
    wall_clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep

    def drain(self) -> DrainOutcome:
        placed = tuple(dict.fromkeys(str(item) for item in self.sessions() if str(item)))
        # 1. ANNOUNCE. The lifecycle IS the refusal for new placements and turns
        #    (§3.4: "no new turns are admitted"), so it is written before anything
        #    is signalled — a session told to drain while the member still reads
        #    ``active`` is a session that can be re-placed underneath the drain.
        #    A WRITE THAT FAILS DOES NOT STOP THE DRAIN: copying the bytes off a pod
        #    that is powering off is worth more than the row, and home's guard is
        #    what refuses an engage while the row still reads ``active``. The failure
        #    is recorded here rather than swallowed, because it is the durable
        #    "placements were still allowed during this drain" (review round 1,
        #    MAJOR-3 gap 1).
        announced = self._set_lifecycle("draining")
        _audit(
            "drain_started",
            self.config_dir,
            subject=self.member_id,
            network_id=self.network_id,
            detail={
                "member": self.member_id,
                "sessions": list(placed),
                "grace_s": float(self.drain_grace_s),
                "trigger": self.trigger,
                "lifecycle_written": announced,
            },
        )
        for session_id in placed:
            self._signal(session_id)

        # 2. THE IN-FLIGHT TURN, bounded by the grace (§3.6 step 2).
        cut = self._wait_for_turns(placed)

        # 3. THE BARRIER (§6.2 steps 2-4).
        flushed, pending = self._flush_until(placed)
        if pending:
            # 5. BOUNDED EXIT. The member stays ``draining`` — it is about to power
            #    off as ``lost`` (§6.6's row), and a row that said ``expired`` would
            #    claim a barrier that did not complete.
            _audit(
                "drain_barrier_timeout",
                self.config_dir,
                outcome="timeout",
                subject=self.member_id,
                network_id=self.network_id,
                detail={
                    "member": self.member_id,
                    "pending": list(pending),
                    "pending_count": len(pending),
                    "deadline_s": float(self.sync_drain_deadline_s),
                    "cut": list(cut),
                },
            )
            return DrainOutcome(
                member_id=self.member_id,
                sessions=placed,
                flushed=flushed,
                pending=pending,
                cut=cut,
                drained=False,
                barrier_expired=True,
            )

        # 4. ACKED → EXPIRED (§6.2 step 5), AND ONLY IF THE ROW SAYS SO. Shipping
        #    ``pool_member_expired`` (and ``drained=True``) beside a member row that
        #    still reads ``active``/``draining`` is two contradictory records of one
        #    fact — and the row is the one home's guard reads (review round 1,
        #    MAJOR-3). The copies are safe; what is missing is the claim.
        if not self._set_lifecycle("expired"):
            return DrainOutcome(
                member_id=self.member_id,
                sessions=placed,
                flushed=flushed,
                pending=(),
                cut=cut,
                drained=False,
                barrier_expired=False,
                expire_failed=True,
            )
        _audit(
            "pool_member_expired",
            self.config_dir,
            subject=self.member_id,
            network_id=self.network_id,
            detail={
                "member": self.member_id,
                "sessions": list(flushed),
                "cut": list(cut),
            },
        )
        return DrainOutcome(
            member_id=self.member_id,
            sessions=placed,
            flushed=flushed,
            pending=(),
            cut=cut,
            drained=True,
            barrier_expired=False,
        )

    # -- steps -------------------------------------------------------------

    def _wait_for_turns(self, placed: Sequence[str]) -> tuple[str, ...]:
        """Wait out the grace, then cut what is still running. Returns the cuts."""
        deadline = self.clock() + float(self.drain_grace_s)
        while True:
            busy = [session_id for session_id in placed if self._busy(session_id)]
            if not busy:
                return ()
            now = self.clock()
            if now >= deadline:
                for session_id in busy:
                    self._cut_turn(session_id)
                return tuple(busy)
            self.sleep(min(self.poll_s, deadline - now))

    def _flush_until(self, placed: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Flush until every session acks or the deadline expires.

        THE DEADLINE BOUNDS EACH ATTEMPT, not only the rounds: one home that is
        asleep costs a full ``net_sync`` round trip per session, so consulting the
        clock only between rounds let a wedged home overshoot a 900 s barrier by
        ``len(pending)`` attempts of it (review round 1, MAJOR-2 — measured at 2x the
        configured deadline). Each attempt therefore gets the budget that is LEFT,
        and a session that refuses is retried on the next round rather than dropped:
        the pod's next action on a failure is to power off, so a retry is free and a
        premature give-up is not.

        A SPENT BUDGET MEANS NO ATTEMPT. The old "every session is attempted at
        least once even when the deadline is already spent" promise is the one that
        made the bound unkeepable — an attempt with no budget left is the unbounded
        wait this deadline exists to prevent, so those sessions stay pending and the
        caller reports them (``drain_barrier_timeout``).
        """
        deadline = self.clock() + float(self.sync_drain_deadline_s)
        acked: list[str] = []
        pending = list(placed)
        while pending:
            still: list[str] = []
            for session_id in pending:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    still.append(session_id)
                    continue
                (acked if self._flush_one(session_id, remaining) else still).append(session_id)
            pending = still
            if not pending:
                break
            now = self.clock()
            if now >= deadline:
                break
            self.sleep(min(self.poll_s, deadline - now))
        return tuple(acked), tuple(pending)

    # -- collaborator calls, each one non-fatal ----------------------------

    def _signal(self, session_id: str) -> None:
        try:
            self.signal(session_id, PRE_SPINDOWN_DRAIN_CAUSE)
        except Exception:  # noqa: BLE001 — one session's runtime must not stop a drain
            logger.debug("pool drain: could not signal %s", session_id, exc_info=True)

    def _busy(self, session_id: str) -> bool:
        try:
            return bool(self.busy(session_id))
        except Exception:  # noqa: BLE001
            logger.debug("pool drain: could not ask about %s", session_id, exc_info=True)
            return False

    def _flush_one(self, session_id: str, budget_s: float) -> bool:
        """One flush attempt, bounded by what is left of the barrier budget.

        THE HANDLER IS THE CALLER'S, so it may block for as long as its own link
        allows (``sync.SYNC_OP_DEADLINE_S`` is 60 s per round trip). Running it
        inline would make "bounded by ``sync_drain_deadline_s``" a claim about the
        handler rather than about this loop, so the attempt runs in a DAEMON thread
        and is ABANDONED — not killed, nothing can kill it — when the budget runs
        out. Daemon because an abandoned attempt must not be able to hold the
        process open past the barrier: the member is powering off regardless.
        """
        answer: dict[str, Any] = {}

        def attempt() -> None:
            try:
                answer["acked"] = bool(self.flush(session_id))
            except Exception:  # noqa: BLE001
                logger.debug("pool drain: flush of %s failed", session_id, exc_info=True)
                answer["acked"] = False

        worker = threading.Thread(target=attempt, name=f"pool-flush-{session_id}", daemon=True)
        worker.start()
        worker.join(max(0.0, budget_s))
        if worker.is_alive():
            logger.warning("pool drain: the flush of %s outlived the barrier budget", session_id)
            return False
        return bool(answer.get("acked", False))

    def _cut_turn(self, session_id: str) -> None:
        if self.cut is not None:
            try:
                self.cut(session_id)
            except Exception:  # noqa: BLE001
                logger.debug("pool drain: could not cut the turn in %s", session_id, exc_info=True)
        _record_drain_cut(self.config_dir, session_id, now=self.wall_clock())

    def _set_lifecycle(self, lifecycle: MemberLifecycle) -> bool:
        """Move the member row to ``lifecycle``. True iff the durable row says so.

        THE ANSWER IS THE EFFECT, not "no exception was raised": the caller gates
        ``pool_member_expired`` and ``drained`` on it, and a row that still reads
        ``active`` beneath an audit log claiming the barrier completed is the
        contradiction review round 1 (MAJOR-3) found. A row that already reads
        ``lifecycle`` is the same durable fact, so an idempotent retry answers True.
        """
        from local_operator.network import store

        try:
            with store.mutate(self.network_id, self.config_dir) as record:
                row = record.member(self.member_id)
                if row is None:
                    logger.warning(
                        "pool drain: %s has no member row to mark %s", self.member_id, lifecycle
                    )
                    return False
                if row.lifecycle == lifecycle:
                    return True
                row.lifecycle = lifecycle
                store.save(record, self.config_dir)
        except (OSError, ValueError, FileNotFoundError):
            # The drain proceeds either way: a member row that cannot be written is
            # a reason for an operator to look, not a reason to leave sessions on a
            # pod that is being spun down.
            logger.warning(
                "pool drain: could not set %s to %s", self.member_id, lifecycle, exc_info=True
            )
            return False
        return True


def _record_drain_cut(config_dir: Path, session_id: str, *, now: float | None = None) -> bool:
    """Close the turn journal row the drain cut, in the runtime's own vocabulary.

    ONLY AN OPEN ROW IS TOUCHED, and only the drain writes here — the runtime's own
    close is a no-op once the row is closed, so this cannot paper over a turn that
    is still running. When the grace expires the drain IS the actor that ended the
    turn (it cut it at the boundary, §3.6 step 2), so the row it leaves is the truth
    a successor reads: turn N was in flight and the pre-spindown drain ended it.

    WHY THE CAUSE IS ``SIGNAL_DRAIN_CAUSE`` AND NOT ``PRE_SPINDOWN_DRAIN_CAUSE``.
    The journal's ``end_cause`` is read by every surface through
    ``incidents.CUT_OFF_CAUSES``, and a cause enters that table with an
    operator-facing sentence translated into every supported language — a reviewed
    i18n contribution of its own, not a token this module may mint. So the row
    carries the token the RUNTIME ITSELF writes when a signal cuts one of its turns
    (the pool drain IS a signalled drain: the relay signals it), and the pool's own
    trigger is stated where this slice owns the vocabulary — the ``drain_started`` /
    ``pool_member_expired`` / ``drain_barrier_timeout`` audit rows name the member and
    the sessions cut, and ``types.PRE_SPINDOWN_DRAIN_CAUSE`` is the reason on the wire.
    """
    from local_operator.network import sync
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import SIGNAL_DRAIN_CAUSE

    directory = sync.session_dir(config_dir, session_id)
    row = registry.read_turn_journal(directory)
    if not isinstance(row, dict) or not row.get("open"):
        return False
    stamped = float(now if now is not None else time.time())
    row.update(
        {
            "ended_at": stamped,
            "open": False,
            "end_cause": SIGNAL_DRAIN_CAUSE,
        }
    )
    try:
        registry.write_turn_journal(directory, row)
    except OSError:
        logger.debug("pool drain: could not record the cut in %s", session_id, exc_info=True)
        return False
    return True
