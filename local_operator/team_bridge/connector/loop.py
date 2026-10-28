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
from collections.abc import Callable
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

#: STALE CLAIM RECOVERY (PR #3038 review R2-3). A connector killed mid-spine
#: (launchd bootout, reboot, OOM — anything that skips the spine's ``finally``)
#: leaves its claim ``held`` for up to ``wake.CLAIM_TTL_S``. The restarted
#: connector's spine is refused ``claimed_elsewhere``, and the loop used to
#: consume the feed ref anyway with no hold row: once the lease lapsed nothing
#: referenced the message and it was never delivered. The loop now parks such
#: a message under this hold code, due at the claim's expiry, and decides at
#: that point from the claim row alone (``Store.claim_status``):
#:
#: * still ``held`` and live — the holder renewed; wait for the new expiry;
#: * still ``held`` but EXPIRED — the holder died mid-handoff: re-drive;
#: * ``released`` by a connector — some spine reached its ``finally``, so ITS
#:   outcome stands (delivered, terminal, or its own hold row, which replaces
#:   this code under the same key): retire, never re-deliver;
#: * held/released by a SESSION (``ack working``) — a session already has
#:   the message: retire, as the ref consumption always did.
#:
#: Exactly-once in the live-holder case therefore rests on the claim row, not
#: on the receiver's dedupe: a successful holder releases, and a released row
#: is never re-driven. Only a holder that never released can be re-driven.
HOLD_STALE_CLAIM = "stale_claim"

#: Slack past the lease expiry before the parked message comes due, so the
#: re-drive's claim lands strictly after ``expires_at`` (``claim`` refuses a
#: row whose expiry is still ``> now``).
STALE_CLAIM_MARGIN_S = 1.0

#: OWNER WATCH (D2b, Team Bridge post-0.90.17). The connector used to resolve
#: its owner ONCE: after a re-enrolment (generation bump, new endpoint,
#: credential rebind) every iteration refused ``owner_mismatch`` and backed off
#: for ever, and an owner-less start exited 0 into launchd's KeepAlive, a whole
#: interpreter relaunch about every 10 s. Exiting on a refusal is NOT the fix
#: (review R4): the same codes answer an account switch against an UNCHANGED
#: owner, so a restart re-reads the same owner and refuses again, a respawn
#: storm. Instead the process stays resident and watches the store.
#:
#: ``OWNER_SIGNAL_POLL_S`` is how often an IDLE connector stats the bridge
#: database (``stat`` only, no SQLite open): a re-enrolment is picked up within
#: one poll. ``OWNER_RECHECK_S`` bounds everything else: an idle connector
#: re-reads the owner (and an identity-refused one retries the hub) at most
#: once per interval, so a missed signal costs latency, never correctness, and
#: an account that is never fixed costs one local read plus one hub attempt a
#: minute rather than a ramp of attempts or a relaunch loop.
OWNER_SIGNAL_POLL_S = 5.0
OWNER_RECHECK_S = BACKOFF_MAX_S

#: Refusals that say "this process's IDENTITY is wrong", not "the hub is
#: flaky". Retrying them on the 1-2-4-8 s ramp only repeats the refusal; what
#: fixes them is a store change (a re-enrolment, picked up by the owner watch)
#: or the operator signing back in (picked up at the bounded re-check). So they
#: idle the full ``OWNER_RECHECK_S`` at once and log on a state change only.
_IDENTITY_REFUSAL_CODES = frozenset(
    {
        "owner_mismatch",
        "not_enrolled",
        "owner_changed",
        "credential_changed",
        "stale_generation",
        "bridge_owner_mismatch",
        "bridge_owner_moved",
        "bridge_sign_in_required",
    }
)

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
        # Deterministic LOCAL identity refusals (no hub round trip): without
        # these in the closed set the 2026-10-26 outage logged as
        # ``unexpected_failure``, indistinguishable from a bug.
        "bridge_owner_mismatch",
        "bridge_sign_in_required",
        "bridge_owner_moved",
        # ``pull._guard_owner``'s refusal when the pinned credential row no
        # longer names a committed owner: an identity state the owner watch
        # idles on, so it must log as itself, not as ``PullRefused``.
        "not_enrolled",
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


def _iteration_failure_code(failure: BaseException) -> str:
    """The closed code, or the exception CLASS name, for a whole-iteration failure.

    The class name (never ``str(failure)``, never ``repr``) is what separates a
    local ``OperationalError`` (e.g. descriptors exhausted) from a hub refusal
    when nothing else about the failure may be written down: messages and
    tracebacks can carry bodies, names or bearer material.
    """
    code = getattr(failure, "code", None)
    if isinstance(code, str) and code in _DIAGNOSTIC_FAILURE_CODES:
        return code
    from local_operator.team_bridge.client import BridgeSignInRequired

    # The sign-in refusal is a ValueError subclass with no ``code`` attribute.
    if isinstance(failure, BridgeSignInRequired):
        return "bridge_sign_in_required"
    return type(failure).__name__


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
    * a ``stale_claim`` park — re-driven only when its connector claim is
      held past expiry (the holder died); ``wait`` re-arms it at the current
      expiry, ``retire`` deletes it (see ``HOLD_STALE_CLAIM``);
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
        # The spines claim by the bare message id, whichever key the hold uses.
        claim_id = parsed.message_id if parsed is not None else message_id
        if row["hold_code"] == HOLD_STALE_CLAIM:
            # A parked stale claim is re-driven ONLY once its holder provably
            # died (held past expiry); see HOLD_STALE_CLAIM for the cases.
            action, expires_at = _stale_claim_disposition(store, owner, claim_id, now=_time.time())
            try:
                if action == "retire":
                    store.delete_hold(owner, message_id)
                elif action == "wait" and expires_at is not None:
                    store.advance_hold_retry(
                        owner, message_id, retry_at=expires_at + STALE_CLAIM_MARGIN_S
                    )
            except StoreRefused:
                pass  # fenced owner: the next live pass owns the row
            if action != "redrive":
                # ``retired`` is the pass's existing word for a deleted row.
                reported = "retired" if action == "retire" else "waiting"
                outcomes.append({"message_id": message_id, "action": reported})
                continue
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
        if action in ("delivered", "bounce", "rejected", "skipped"):
            # A pre-upgrade conversation hold may still use the seven-part
            # label-bearing key. The spine clears the canonical locator; retire
            # the exact consumed row too, rather than replaying it forever.
            # A re-driven stale-claim park is retired here too: a cached or
            # replayed admission commits no new receipt, so nothing else
            # would delete it (the next pass would retire it anyway, on the
            # released claim — this just does not leave it visible).
            if (parsed is not None and message_id != parsed.encode()) or row[
                "hold_code"
            ] == HOLD_STALE_CLAIM:
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


def _stale_claim_disposition(
    store: Any, owner: Any, claim_id: str, *, now: float
) -> tuple[str, float | None]:
    """What a ``stale_claim``-parked message needs, from its claim row alone.

    ``("wait", expires_at)`` — a connector claim is still held and unexpired
    (a live holder, or a dead one whose lease has not lapsed): wait it out.
    ``("redrive", None)`` — a connector claim is held but EXPIRED (its holder
    died before ``finally``) or was fenced by a generation change: re-drive
    through the spine, whose own CAS claim replaces the dead row.
    ``("retire", None)`` — the claim was released (a spine finished, and its
    outcome — delivered, terminal or its own hold row — stands), a session
    holds it (the message already reached a session), or no row exists.
    Never ``redrive`` a released claim: that is what keeps a live second
    holder's delivery from being repeated.
    """

    status = store.claim_status(owner, claim_id)
    if status is None or not status["connector"]:
        return "retire", None
    if status["state"] == "held":
        if status["expires_at"] > now:
            return "wait", float(status["expires_at"])
        return "redrive", None
    if status["state"] == "fenced":
        return "redrive", None
    return "retire", None


def _park_stale_claim(
    store: Any, owner: Any, hold_key: str, claim_id: str, outcome: dict[str, Any]
) -> None:
    """Give a ``claimed_elsewhere`` outcome a durable retry instead of none.

    Only a refusal by a LIVE-LOOKING connector claim is parked (code
    ``claimed`` and a held ``connector:`` row): a session's own claim means
    the message already reached a session, and any other refusal code is an
    owner/identity fence the next live generation re-derives on its own. The
    row comes due at the lease expiry, when ``retry_due_holds`` decides from
    the claim row whether the holder finished or died. ``keep_existing``: a
    durable retry already there (the live holder's own hold) is never
    overwritten by this park.
    """

    import time as _time

    from local_operator.team_bridge.store import StoreRefused

    if outcome.get("code") != "claimed":
        return
    action, expires_at = _stale_claim_disposition(store, owner, claim_id, now=_time.time())
    if action == "retire":
        return  # released in between: the holder's own outcome stands
    retry_at = (expires_at or _time.time()) + STALE_CLAIM_MARGIN_S
    try:
        store.persist_hold(owner, hold_key, HOLD_STALE_CLAIM, retry_at=retry_at, keep_existing=True)
    except StoreRefused:
        pass  # fenced owner: the next live generation owns its own rows
    logger.info("bridge connector parked: stage=claim code=%s", HOLD_STALE_CLAIM)


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
        elif outcome.get("action") == "claimed_elsewhere":
            # R2-3: the ref is consumed below, so a claim left by a connector
            # that died mid-spine must leave a durable retry behind.
            _park_stale_claim(store, owner, message_id, message_id, outcome)
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
        elif outcome.get("action") == "claimed_elsewhere":
            # Same R2-3 park, keyed like every conversation hold (by ref).
            _park_stale_claim(
                store, owner, conversation_ref.encode(), conversation_ref.message_id, outcome
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
    owner_source: Callable[[], Any] | None = None,
) -> Any:
    """The connector's steady state: poll, handle, back off, repeat.

    ``access_provider`` is an async callable returning the pinned OAuth
    access for this owner (the H1 client's own ``current_access`` — the
    ordinary sign-in path, never a key cascade). A refused iteration backs
    off (`BACKOFF_BASE_S` doubling to `BACKOFF_MAX_S`); a 204 resets it.

    RE-ARM CONTRACT. Every iteration first re-reads the committed owner from
    ``owner_source`` (default: this config dir's ``Store.active_owner``, one
    local SQLite read, no network). A committed owner that DIFFERS from
    ``owner`` (endpoint, credential, tenant or generation) is RETURNED, and the
    caller re-arms on it in-process; ``None`` is returned only when ``stop`` is
    set. No owner at all, or an identity refusal against the unchanged owner,
    keeps the loop resident and idling (see ``OWNER_RECHECK_S``): it never
    returns for those, because the caller exiting is the respawn storm R4
    rejected.
    """

    import time as _time

    if owner_source is None:
        from local_operator.team_bridge.store import Store

        owner_source = Store(config_dir).active_owner
    backoff = BACKOFF_BASE_S
    # The idle state last LOGGED (``no_owner`` or a refusal code), so a state
    # that persists for hours writes one line, not one per re-check.
    idle_code: str | None = None
    while stop is None or not stop.is_set():
        current = owner_source()
        if current is not None and current != owner:
            logger.info("bridge connector rearm: stage=rearm code=owner_changed")
            return current
        if current is None:
            # Unenrolled under us (or the store is unreadable, which
            # ``active_owner`` reads as no authority): never touch the hub.
            if idle_code != "no_owner":
                logger.info("bridge connector idle: stage=idle code=no_owner")
                idle_code = "no_owner"
            await idle_until_owner_changes(
                None, config_dir=config_dir, owner_source=owner_source, stop=stop
            )
            continue
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
            idle_code = None
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
            # It used to be swallowed with no record at all, which is how a
            # connector that could no longer open a file looped silently for
            # hours. One structural line per failed iteration: stage + code.
            code = _iteration_failure_code(failure)
            # Every identity code is in the closed diagnostic set, so the
            # logged code is the classification: one source of truth.
            if code in _IDENTITY_REFUSAL_CODES:
                # Identity refusal: same line format, but once per state
                # change, then a watched idle instead of the ramp.
                if idle_code != code:
                    logger.warning("bridge connector failure: stage=iteration code=%s", code)
                    idle_code = code
                await idle_until_owner_changes(
                    owner,
                    config_dir=config_dir,
                    owner_source=owner_source,
                    stop=stop,
                    budget=_failure_delay(failure, OWNER_RECHECK_S),
                )
                continue
            logger.warning("bridge connector failure: stage=iteration code=%s", code)
            await _pause(stop, _failure_delay(failure, backoff))
            backoff = min(backoff * 2.0, BACKOFF_MAX_S)
    return None


def _owner_signal(config_dir: Path) -> tuple[Any, ...]:
    """A cheap change signal for the bridge store: ``stat`` of the database
    and its WAL, never an open.

    WHY a stat and not a read: an idle connector may sit for days, and a
    SQLite open per poll (``Store._connect`` also runs the layout check) is the
    tight polling the re-arm must avoid. Every enrolment write reaches the
    WAL (or, once checkpointed, the main file), so a commit, an invalidate or
    a first-ever store moves this tuple; plain reads do not (measured: three
    ``active_owner`` reads leave it identical). Other writers to the same
    store can move it too, which only costs one extra local owner read. Carries
    no names or content: inode, size and mtime only.
    """

    from local_operator.team_bridge.store import state_db_path

    database = state_db_path(config_dir)
    parts: list[Any] = []
    for path in (database, database.with_name(database.name + "-wal")):
        try:
            stat = path.stat()
        except OSError:
            parts.append(None)
            continue
        parts.append((stat.st_ino, stat.st_size, stat.st_mtime_ns))
    return tuple(parts)


async def idle_until_owner_changes(
    baseline: Any,
    *,
    config_dir: Path,
    owner_source: Callable[[], Any],
    stop: asyncio.Event | None,
    budget: float = OWNER_RECHECK_S,
) -> None:
    """Idle up to ``budget`` seconds, returning EARLY when the store's committed
    owner stops being ``baseline`` (or ``stop`` is set).

    The one owner-watch helper behind both idle states: the loop's (no owner,
    or an identity refusal) and ``__main__``'s owner-less start. Between polls
    it costs one ``stat`` per ``OWNER_SIGNAL_POLL_S``; the owner is READ only
    when that signal moves. The caller re-reads the owner after this returns,
    which is the bounded re-check that covers a change the signal missed.
    Takes the baseline signal BEFORE the first wait, so a commit landing at any
    point inside the window is seen.
    """

    signal = _owner_signal(config_dir)
    waited = 0.0
    while (stop is None or not stop.is_set()) and waited < budget:
        step = min(OWNER_SIGNAL_POLL_S, budget - waited)
        await _pause(stop, step)
        waited += step
        moved = _owner_signal(config_dir)
        if moved == signal:
            continue
        signal = moved
        if owner_source() != baseline:
            return


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
