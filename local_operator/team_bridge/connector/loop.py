"""The events long-poll loop + cursor/resync walk (H4b).

TRANSPORT SHAPE, pinned by §6.7/§13.3. The connector's ONLY recurring bridge
traffic is ONE outstanding ``GET /v1/messaging/events?after=<cursor>&wait=25``:
``200`` → refs are recorded and drained, ``204`` → the read returns ``None``
and any already-committed refs are drained before the loop reissues. There is
no periodic status/inbox call, no plain-poll fallback, and no scheduled model wake — a missed event
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
import logging
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
    feed_seq,
    is_conversation_message_id,
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

#: The server-side park of every connector read (§6.7: ``wait ≤ 30``). WHY:
#: without it the hub answers a caught-up cursor at once, and the live E2E on
#: local-operator-ui #1314 measured 124 reads in under 2 s, then 429s.
EVENTS_WAIT_S = 25

#: The floor between two iterations whose read came back AT ONCE (events,
#: a resync arm, a hub that ignores ``wait``). The long poll is the primary
#: pacing; this floor is what keeps a hub that never parks — a proxy that
#: strips the query, a persistent resync answer — from turning the loop back
#: into a storm. One second sits far inside the hub's 120/min poll bucket
#: and costs a busy feed at most a second of latency per batch.
MIN_ITERATION_S = 1.0

#: Cap on a server-named ``Retry-After``: honour the hub's own wait, but a
#: hostile or broken header must not park an always-on connector for hours.
RETRY_AFTER_MAX_S = 300.0

#: Held-item retry (§13.4 "retry with backoff"): base gap, cap, max attempts,
#: and the batch bound per turn. The per-message counter lives in THIS process;
#: a restart re-bases at RETRY_BASE_S — bounded and safe, because the gap only
#: schedules the next LOCAL re-screen and never grants delivery authority.
RETRY_BASE_S = 60.0
RETRY_MAX_S = 3600.0
RETRY_MAX_ATTEMPTS = 11
RETRY_BATCH_LIMIT = 8
_RETRY_ATTEMPTS: dict[str, int] = {}

logger = logging.getLogger(__name__)

# Operational codes only: exception messages, arbitrary .code values, feed
# refs and tracebacks may carry peer content or credentials. A pre-screen
# failure has no attempt/hold row yet, so record its stage and a closed code
# without fabricating a screening result or logging the exception itself.
_DIAGNOSTIC_FAILURE_CODES = frozenset(
    {
        "policy_stale",
        "bridge_off",
        "owner_changed",
        "credential_changed",
        "consent_revoked",
        "no_designation",
        "stale_generation",
        "schema_damaged",
        "owner_mismatch",
        "wrong_recipient",
        "wrong_tenant",
        "malformed_message",
        "mismatched_id",
        "rate_limited",
        "bridge_not_available",
        "bridge_disabled",
        "messaging_unavailable",
        "recipient_unavailable",
        "consent_version_unknown",
    }
)


def _log_delivery_failure(stage: str, failure: Exception) -> None:
    """Content-free diagnostic; the original exception still drives backoff."""
    code = getattr(failure, "code", None)
    safe_code = (
        code
        if isinstance(code, str) and code in _DIAGNOSTIC_FAILURE_CODES
        else "unexpected_failure"
    )
    logger.warning("bridge connector failure: stage=%s code=%s", stage, safe_code)


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


def _release_engaged_holds(store: Any, owner: Any, config_dir: Path, *, now: float) -> None:
    """Re-drive ``open_chief_of_staff`` holds the moment their reason is gone.

    WHY (review of local-operator-ui #1325): that hold is released by the owner
    SENDING their chief of staff a message — a user turn on disk, written by
    another process. No event reaches the connector when it happens, so the
    held message waited out the rest of its doubling backoff (60 s → 1 h), and
    the owner's "send it a message" notice stayed up that long after they did.

    So each pass checks the one local fact that releases it — cheap: a single
    indexed count first, and the transcript probe only when such holds exist.
    When ``engagement_hold`` now answers ``""`` (a durable turn exists, or the
    accepted consent allows engagement), those holds are made due NOW and the
    retry below re-drives them in this same pass. Every other hold code keeps
    its ordinary backoff. Best-effort: a failure leaves the backoff in charge.
    """

    from local_operator.team_bridge.connector import wake as _wake
    from local_operator.team_bridge.engagement import HOLD_OPEN_CHIEF_OF_STAFF
    from local_operator.team_bridge.store import StoreRefused

    try:
        if store.hold_count_readonly(owner, HOLD_OPEN_CHIEF_OF_STAFF) <= 0:
            return
        designation = store.designated_session_for(owner)
        # Through the module, so the spine and this check read ONE rule.
        if designation is None or _wake.engagement_hold(designation, config_dir):
            return
        store.make_holds_due(owner, HOLD_OPEN_CHIEF_OF_STAFF, now=now)
    except (StoreRefused, OSError):
        return


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
      due window or the other claimant;
    * ``retired`` — the row cannot be re-driven (a bare ``bcmsg_`` duplicate
      of a conversation hold, or a message the hub answers 404 for) and is
      deleted, so it is never re-asked on every iteration.

    Fail-closed: this path can never deliver or ack a held item itself; the
    spine's own laws decide (and a hold still never wakes). ``CancelledError``
    propagates — per-claim release lives in the spine's ``finally``.
    """

    import time as _time

    from local_operator.team_bridge.client import BridgeRefusal
    from local_operator.team_bridge.store import Store, StoreRefused

    store = Store(config_dir)
    _release_engaged_holds(store, owner, config_dir, now=_time.time())
    due = store.due_screening_holds(owner, now=_time.time(), limit=limit)
    outcomes: list[dict[str, Any]] = []
    for row in due:
        message_id = row["message_id"]
        if is_conversation_message_id(message_id):
            # A BARE conversation-message id (live E2E F5). The screener writes
            # its own hold (``provider_error``, ``policy_stale``, …) under the
            # message id it screened, and for a conversation that is the
            # ``bcmsg_`` id. That id cannot name the conversation again, and
            # the pair spine asked the hub for a PAIR message that does not
            # exist — raising on every iteration, before the poll, so the
            # whole connector stalled. The re-drive authority is the loop's
            # own row under the encoded feed ref (``_persist_hold_backoff``),
            # which every conversation hold also writes; the bare row is a
            # screening-side duplicate, so it is retired here, never re-driven.
            try:
                store.delete_hold(owner, message_id)
            except StoreRefused:
                pass  # fenced owner: the next live pass owns the row
            _RETRY_ATTEMPTS.pop(message_id, None)
            outcomes.append({"message_id": message_id, "action": "retired"})
            continue
        parsed = _conversation_hold_ref(message_id)
        try:
            if parsed is not None:
                # A conversation hold is keyed by its encoded feed ref, because
                # a bare message id cannot name the conversation again after a
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
        except asyncio.CancelledError:
            raise
        except BridgeRefusal as refusal:
            # BOUND (F5): one row's re-drive must never abort the pass — an
            # escape here skips the poll for every later iteration. A message
            # the hub says is gone (404; the hub's code is ``message_not_found``)
            # is terminal: retire the row instead of re-asking forever.
            # Anything else is held on the same backoff.
            if refusal.http_status == 404:
                try:
                    store.delete_hold(owner, message_id)
                except StoreRefused:
                    pass
                _RETRY_ATTEMPTS.pop(message_id, None)
                outcomes.append({"message_id": message_id, "action": "retired"})
                continue
            outcome = {"action": "hold", "message_id": message_id, "code": refusal.code}
        except Exception:  # noqa: BLE001 — one row never stops the connector
            outcome = {"action": "hold", "message_id": message_id, "code": "retry_failed"}
        action = outcome.get("action")
        if action in ("delivered", "bounce", "delivered_unadmitted", "skipped"):
            # A pre-upgrade conversation hold may still use the seven-part
            # label-bearing key. The spine clears the canonical locator; retire
            # the exact consumed row too, rather than replaying it forever.
            if parsed is not None and message_id != parsed.encode():
                try:
                    store.delete_hold(owner, message_id)
                except StoreRefused:
                    pass
            _RETRY_ATTEMPTS.pop(message_id, None)
        elif action == "hold":
            attempts = _RETRY_ATTEMPTS.get(message_id, 0)
            gap = min(RETRY_BASE_S * (2**attempts), RETRY_MAX_S)
            new_code = str(outcome.get("code") or "")[:64]
            try:
                advanced = store.advance_hold_retry(owner, message_id, retry_at=_time.time() + gap)
                if advanced and new_code and new_code != row["hold_code"]:
                    # The row's code is what the owner sees (F8): a message
                    # first held ``open_chief_of_staff`` and then held for a
                    # NEW reason once the owner messaged the session must not
                    # keep telling them to message it. Still conditional on the
                    # row existing (``advanced``), so nothing is resurrected.
                    store.persist_hold(owner, message_id, new_code, retry_at=_time.time() + gap)
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
    # F2 (live E2E): the stored cursor is a ``FeedSeq`` and 0 is a POSITION —
    # a fresh enrolment's. ``feed_seq or None`` turned it into "no cursor",
    # which the hub answers with a resync at the CURRENT head, so the first
    # message ever delivered to a fresh endpoint was skipped. Mint it through
    # the typed constructor and send it as-is, 0 included.
    after = feed_seq(state["feed_seq"])
    batch = await _pull.events(
        owner, config_dir=config_dir, access=access, after=after, wait=EVENTS_WAIT_S
    )
    # The feed cursor commits BEFORE screening. A failed handler leaves its
    # durable ref behind, but the hub can now legitimately answer 204 forever.
    # Drain that existing queue even on a timeout; never reset the cursor or
    # synthesize refs from historical inbox rows to recover a failed handoff.
    reissued = batch is None
    cursor = int(after) if batch is None else batch["cursor"]
    refs = _pull.drain_feed_refs(owner, config_dir=config_dir, upto_seq=cursor)
    if reissued and not refs:
        return {
            "events": 0,
            "reissued": True,
            "resync": False,
            "handled": [],
            "retries": retries,
        }
    # H5e: the routing rule is stated ONCE (team_bridge.events) so the CoS
    # refresh consumer cannot drift from this loop: deliveries wake; the
    # state_changed/request_* refs are refresh hints and NEVER a wake kind.
    plan = plan_feed_refs(refs)
    handled: list[dict[str, Any]] = []
    failures: list[tuple[str, Exception]] = []
    for message_id in plan.deliveries:
        try:
            outcome = await handle_message_available(
                owner,
                message_id,
                config_dir=config_dir,
                auth_store=auth_store,
                access=access,
                settings=settings,
            )
        except Exception as failure:  # one poison ref must not starve the rest
            _log_delivery_failure("pair_handling", failure)
            failures.append((f"message_available:message_id={message_id}", failure))
            continue
        handled.append(outcome)
        if outcome.get("action") == "hold":
            # TB-04: a hold after consumption gets a DURABLE retry row — the
            # screening_holds schedule is the re-drive path now that refs are
            # consumed once-only, so a stuck delivery is never silently lost.
            _persist_hold_backoff(store, owner, message_id, str(outcome.get("code") or "hold"))
    for conversation_ref in plan.conversation_deliveries:
        # Conversation deliveries have their OWN card path; the pair inbox
        # reader above must never see one.
        try:
            outcome = await handle_conversation_available(
                owner,
                conversation_ref,
                config_dir=config_dir,
                auth_store=auth_store,
                access=access,
                settings=settings,
            )
        except Exception as failure:  # preserve the conversation lane's own ref
            _log_delivery_failure("conversation_handling", failure)
            failures.append((conversation_ref.encode(), failure))
            continue
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
    # Retire only outcomes actually handled. Failed refs stay durable, while
    # successful neighbours do not repeat merely because a later ref failed.
    # One attempt per ref per iteration; re-raising preserves run_forever's
    # bounded failure backoff (including Retry-After), not a new hot retry loop.
    failed_refs = {ref for ref, _ in failures}

    # Compare canonical identities but delete ORIGINAL keys. An older row
    # can carry labels that the new projection intentionally discards; comparing
    # that raw key with a canonical failure would otherwise drop it on a 204
    # retry (#2835). This is ordinary consumption, never a history migration.
    def failed(raw: str) -> bool:
        conversation = _conversation_hold_ref(raw)
        return (conversation.encode() if conversation is not None else raw) in failed_refs

    store.delete_feed_refs(owner, [ref for ref in refs if not failed(ref)])
    if failures:
        raise failures[0][1]
    # A 204 is still NOT an inbox-resync signal. Keep the existing bootstrap /
    # re-enable replay fence and historical inbox materialization contract.
    resync = batch is not None and (
        bool(batch["resync"]) or bool(store.sync_state(owner)["pending_resync"])
    )
    walk: dict[str, Any] | None = None
    if resync:
        walk = await resync_walk(owner, config_dir=config_dir, access=access)
    return {
        "events": len(refs),
        "cursor": cursor,
        "reissued": reissued,
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

    import time as _time

    backoff = BACKOFF_BASE_S
    while stop is None or not stop.is_set():
        started = _time.monotonic()
        try:
            access = await access_provider()
            summary = await run_iteration(
                owner,
                config_dir=config_dir,
                access=access,
                auth_store=auth_store,
                settings=settings,
            )
            backoff = BACKOFF_BASE_S
            # F1 (live E2E): an answer used to re-poll IMMEDIATELY. Against a
            # real hub that answered at once (resync arms, a caught-up cursor
            # without ``wait``) it never yielded — 124 reads in under 2 s, then
            # 429s. The long poll paces the ordinary case; this floor paces a
            # hub that answers without parking, and it applies to a 204 as
            # well as a 200 (review R1-1): a proxy that strips ``wait`` turns
            # every caught-up poll into an INSTANT 204, which at the bare
            # reissue delay was ~300 polls/min against the hub's 120/min.
            floor = MIN_ITERATION_S - (_time.monotonic() - started)
            if summary.get("reissued"):
                floor = max(REISSUE_DELAY_S, floor)
            await _pause(stop, floor)
        except asyncio.CancelledError:
            raise
        except Exception as failure:
            # No content in the backoff path (privacy): the failure is
            # operational, and the next iteration re-tries the ordinary way.
            await _pause(stop, _failure_delay(failure, backoff))
            backoff = min(backoff * 2.0, BACKOFF_MAX_S)


def _failure_delay(failure: BaseException, backoff: float) -> float:
    """How long to wait after a failed iteration: the local backoff, or the
    hub's own ``Retry-After`` when it names a longer one (bounded).

    WHY: a 429 (the 120/min poll bucket, the ≤4-waiter census) carries the
    wait the hub will enforce; re-polling sooner only earns another refusal
    and keeps the bucket drained. Never SHORTER than the local backoff (the
    doubling still bounds a flapping hub), never longer than the cap.
    """

    retry_after = getattr(failure, "retry_after_seconds", None)
    if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool):
        return max(backoff, min(float(retry_after), RETRY_AFTER_MAX_S))
    return backoff


async def _pause(stop: asyncio.Event | None, seconds: float) -> None:
    """Sleep up to ``seconds``, waking early when ``stop`` is set.

    A plain sleep would make a long Retry-After hold a shutdown for minutes.
    """

    if seconds <= 0:
        # Still yield: a loop that never awaits starves every other task.
        await asyncio.sleep(0)
        return
    if stop is None:
        await asyncio.sleep(seconds)
        return
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
