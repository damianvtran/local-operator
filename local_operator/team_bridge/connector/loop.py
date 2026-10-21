"""The events long-poll loop + cursor/resync walk (H4b).

TRANSPORT SHAPE, pinned by §6.7/§13.3. The connector's ONLY recurring bridge
traffic is ONE outstanding ``GET /v1/messaging/events?after=<cursor>&wait=25``:
``200`` → refs are recorded and drained, ``204`` → the read returns ``None``
and the loop simply reissues. There is deliberately no periodic status/inbox
call, no plain-poll fallback, and no scheduled model wake — a missed event
costs LATENCY, never state, because the inbox list is the source of truth and
the resync walk repairs gaps (empty ``events`` + fresh cursor, or a cursor the
server no longer honours).

CURSOR. The native ``sync`` row is the only store (`record_feed_batch` writes
it, `sync_state` reads it); ``feed_seq`` is the ``after=`` value, and a
``pending_resync`` flag routes the loop into the walk before the next poll.
Refs are durable in ``feed_refs`` and drained per cursor value — crash after
commit replays nothing, crash before replays the same batch.

RESYNC WALK. Drives ``pull.inbox`` pagination to a TERMINAL page
(``next_cursor is None``); H2's witness (`finish_resync`) refuses completion
unless every page was merged through the terminal one, so an interrupted walk
can never be recorded as complete. The baseline checkpoint captured BEFORE the
walk becomes the new resume point, so arrivals during the walk are not skipped.

``RADIENT_API_BASE_URL`` is read by the CALLER at process start and its value
rides the wire client's own resolution; the loop never re-reads the
environment mid-flight (restart-scoped, §13.3).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from local_operator.team_bridge.connector.conversation import (
    handle_conversation_available,
)
from local_operator.team_bridge.connector.wake import (
    handle_message_available,
    parse_event_ref,
)
from local_operator.team_bridge.events import plan_feed_refs
from local_operator.team_bridge.types import (
    CONVERSATION_FEED_KIND,
    ConversationFeedRef,
    parse_conversation_ref_value,
)

#: Bounded exponential backoff between loop iterations after a FAILURE (a
#: refused read, a transport error). A 204 timeout is NOT a failure — it
#: resets the backoff and reissues promptly.
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 60.0

#: Bounded walk: a page loop that never terminates is a server bug, not a
#: reason to spin — cap and surface so the next resync retries.
MAX_WALK_PAGES = 200

#: Reissue delay after a 204 timeout. Small deliberately: the server held the
#: request for the full wait window; a fresh poll is the contract's shape.
REISSUE_DELAY_S = 0.2

#: Held-item retry (§13.4 "retry with backoff"): base gap, cap, max attempts,
#: and the batch bound per turn. The per-message counter lives in THIS process;
#: a restart re-bases at RETRY_BASE_S — bounded and safe, because the gap only
#: schedules the next LOCAL re-screen and never grants delivery authority.
RETRY_BASE_S = 60.0
RETRY_MAX_S = 3600.0
RETRY_MAX_ATTEMPTS = 11
RETRY_BATCH_LIMIT = 8
_RETRY_ATTEMPTS: dict[str, int] = {}


async def resync_walk(owner: Any, *, config_dir: Path, access: Any) -> dict[str, Any]:
    """Drive the inbox pagination to a terminal page, then complete it.

    Completion is H2's witness, not this loop's confidence: `finish_resync`
    refuses unless the LAST merged page was terminal (`walk_terminal`).
    """

    from local_operator.team_bridge import pull as _pull
    from local_operator.team_bridge.store import Store

    store = Store(config_dir)
    baseline = store.sync_state(owner)["inbox_checkpoint"]
    cursor = baseline
    pages = 0
    while True:
        page = await _pull.inbox(owner, config_dir=config_dir, access=access, after=cursor)
        pages += 1
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
        if pages >= MAX_WALK_PAGES:
            raise RuntimeError("inbox resync walk exceeded the bounded page count")
    from local_operator.team_bridge.types import InboxCursor

    _pull.finish_resync(owner, InboxCursor(baseline or ""), config_dir=config_dir)
    return {"pages": pages}


def _conversation_hold_ref(value: str) -> ConversationFeedRef | None:
    """A hold row that names a conversation delivery, or ``None`` for a pair item."""

    parsed = parse_event_ref(value)
    if parsed is None:
        return None
    event_type, kind, raw = parsed
    if event_type != "message_available" or kind != CONVERSATION_FEED_KIND:
        return None
    return parse_conversation_ref_value(raw)


async def retry_due_holds(
    owner: Any,
    *,
    config_dir: Path,
    access: Any,
    auth_store: Any,
    settings: Any = None,
    limit: int = RETRY_BATCH_LIMIT,
) -> dict[str, Any]:
    """ONE bounded pass over DUE client-local holds (§13.4 retry-with-backoff).

    Mechanism: read the native ``screening_holds`` timing (local metadata
    only — never a server poll), then re-drive each DUE item through the SAME
    spine (``handle_message_available``), whose fetch performs the single
    documented message-bytes re-read per due item. Outcomes:

    * delivered / bounced — the receipt commit deletes the hold row; the
      in-process attempt counter resets;
    * held AGAIN — ``retry_at`` advances by a bounded doubling gap (60 s →
      3600 s cap; counter capped at RETRY_MAX_ATTEMPTS);
    * ``badge_only`` / ``claimed_elsewhere`` — the row is left for the next
      due window or the other claimant.

    Fail-closed: this path can never deliver or ack a held item itself; the
    spine's own laws decide (and a hold still never wakes). ``CancelledError``
    propagates — per-claim release lives in the spine's ``finally``.
    """

    import time as _time

    from local_operator.team_bridge.store import Store, StoreRefused

    store = Store(config_dir)
    due = store.due_screening_holds(owner, now=_time.time(), limit=limit)
    outcomes: list[dict[str, Any]] = []
    for row in due:
        message_id = row["message_id"]
        parsed = _conversation_hold_ref(message_id)
        if parsed is not None:
            # A conversation hold is keyed by its encoded feed ref, because a
            # bare message id cannot name the conversation again after a
            # restart. Never hand it to the pair spine.
            outcome = await handle_conversation_available(
                owner,
                parsed,
                config_dir=config_dir,
                auth_store=auth_store,
                access=access,
                settings=settings,
            )
        else:
            outcome = await handle_message_available(
                owner,
                message_id,
                config_dir=config_dir,
                auth_store=auth_store,
                access=access,
                settings=settings,
            )
        action = outcome.get("action")
        if action in ("delivered", "bounce", "delivered_unadmitted", "skipped"):
            _RETRY_ATTEMPTS.pop(message_id, None)
        elif action == "hold":
            attempts = _RETRY_ATTEMPTS.get(message_id, 0)
            gap = min(RETRY_BASE_S * (2**attempts), RETRY_MAX_S)
            try:
                store.advance_hold_retry(owner, message_id, retry_at=_time.time() + gap)
            except StoreRefused:
                pass  # fenced owner: the next live pass owns the row
            _RETRY_ATTEMPTS[message_id] = min(attempts + 1, RETRY_MAX_ATTEMPTS)
        outcomes.append({"message_id": message_id, "action": action})
    if len(_RETRY_ATTEMPTS) > 512:
        # Bound the counter map: it is a scheduling aid, not state worth growth.
        _RETRY_ATTEMPTS.clear()
    return {"due": len(due), "outcomes": outcomes}


def _persist_hold_backoff(store: Any, owner: Any, message_id: str, code: str) -> None:
    """Write/advance a durable ``screening_holds`` row for a held outcome.

    TB-04 + sec MINOR-1 (PR #2378): feed consumption is once-only (TB-01), so
    the re-drive path for a held item is the native holds schedule — the same
    bounded doubling gap ``retry_due_holds`` uses. A fenced owner is left to
    the next live pass; the counter map is the same scheduling aid.
    """

    import time as _time

    from local_operator.team_bridge.store import StoreRefused

    attempts = _RETRY_ATTEMPTS.get(message_id, 0)
    gap = min(RETRY_BASE_S * (2**attempts), RETRY_MAX_S)
    try:
        store.persist_hold(owner, message_id, str(code)[:64] or "hold", retry_at=_time.time() + gap)
    except StoreRefused:
        pass  # fenced owner: the next live pass owns the row
    _RETRY_ATTEMPTS[message_id] = min(attempts + 1, RETRY_MAX_ATTEMPTS)


async def run_iteration(
    owner: Any,
    *,
    config_dir: Path,
    access: Any,
    auth_store: Any,
    settings: Any = None,
) -> dict[str, Any]:
    """ONE loop turn: due-hold retries, one read, the drained refs, the walk.

    Returns metadata only (counts/actions) — no body, no excerpt.
    """

    from local_operator.team_bridge import pull as _pull
    from local_operator.team_bridge.store import Store

    # Held-item retries run FIRST and on every turn (their own `retry_at`
    # gates them): a quarantine must not wait for fresh traffic to retry, and
    # a 204 timeout is not a reason to skip it. Local metadata only.
    retries = await retry_due_holds(
        owner,
        config_dir=config_dir,
        access=access,
        auth_store=auth_store,
        settings=settings,
    )
    store = Store(config_dir)
    state = store.sync_state(owner)
    after = state["feed_seq"] or None
    batch = await _pull.events(owner, config_dir=config_dir, access=access, after=after)
    if batch is None:
        return {
            "events": 0,
            "reissued": True,
            "resync": False,
            "handled": [],
            "retries": retries,
        }
    refs = _pull.drain_feed_refs(owner, config_dir=config_dir, upto_seq=batch["cursor"])
    # H5e: the routing rule is stated ONCE (team_bridge.events) so the CoS
    # refresh consumer cannot drift from this loop: deliveries wake; the
    # state_changed/request_* refs are refresh hints and NEVER a wake kind.
    plan = plan_feed_refs(refs)
    handled: list[dict[str, Any]] = []
    for message_id in plan.deliveries:
        outcome = await handle_message_available(
            owner,
            message_id,
            config_dir=config_dir,
            auth_store=auth_store,
            access=access,
            settings=settings,
        )
        handled.append(outcome)
        if outcome.get("action") == "hold":
            # TB-04: a hold after consumption gets a DURABLE retry row — the
            # screening_holds schedule is the re-drive path now that refs are
            # consumed once-only, so a stuck delivery is never silently lost.
            _persist_hold_backoff(store, owner, message_id, str(outcome.get("code") or "hold"))
    for conversation_ref in plan.conversation_deliveries:
        # Conversation deliveries have their OWN card path; the pair inbox
        # reader above must never see one.
        outcome = await handle_conversation_available(
            owner,
            conversation_ref,
            config_dir=config_dir,
            auth_store=auth_store,
            access=access,
            settings=settings,
        )
        handled.append(outcome)
        if outcome.get("action") == "hold":
            _persist_hold_backoff(
                store, owner, conversation_ref.encode(), str(outcome.get("code") or "hold")
            )
    # TB-01: ONCE-ONLY consumption — the drained refs were just handled
    # (deliveries) or are state hints the drain itself consumes; deleting them
    # here is what stops the quadratic re-wake/ACK growth across polls and
    # restarts. A crash before this point replays the batch (at-least-once);
    # the receiver's own dedupe absorbs the replay.
    store.delete_feed_refs(owner, refs)
    resync = bool(batch["resync"]) or bool(store.sync_state(owner)["pending_resync"])
    walk: dict[str, Any] | None = None
    if resync:
        walk = await resync_walk(owner, config_dir=config_dir, access=access)
    return {
        "events": len(refs),
        "cursor": batch["cursor"],
        "reissued": False,
        "resync": resync,
        "walk": walk,
        "handled": handled,
        "retries": retries,
    }


async def run_forever(
    owner: Any,
    *,
    config_dir: Path,
    auth_store: Any,
    access_provider: Any,
    settings: Any = None,
    stop: asyncio.Event | None = None,
) -> None:
    """The connector's steady state: poll, handle, back off, repeat.

    ``access_provider`` is an async callable returning the pinned OAuth
    access for this owner (the H1 client's own ``current_access`` — the
    ordinary sign-in path, never a key cascade). A refused iteration backs
    off (`BACKOFF_BASE_S` doubling to `BACKOFF_MAX_S`); a 204 resets it.
    """

    backoff = BACKOFF_BASE_S
    while stop is None or not stop.is_set():
        try:
            access = await access_provider()
            summary = await run_iteration(
                owner,
                config_dir=config_dir,
                access=access,
                auth_store=auth_store,
                settings=settings,
            )
            if summary.get("reissued"):
                backoff = BACKOFF_BASE_S
                await asyncio.sleep(REISSUE_DELAY_S)
            else:
                backoff = BACKOFF_BASE_S
        except asyncio.CancelledError:
            raise
        except Exception:
            # No content in the backoff path (privacy): the failure is
            # operational, and the next iteration re-tries the ordinary way.
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, BACKOFF_MAX_S)
