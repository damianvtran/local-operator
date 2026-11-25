"""Coalesce conversation wakes: ONE card per (endpoint, conversation) window.

WHY THIS EXISTS (wake contract, 2026-10-30). An agent member is now woken for
EVERY message in a conversation it belongs to, not only mentions. A burst — a
human typing three short lines, two agents trading replies — would otherwise
cost the recipient one model turn per message. The connector therefore
buffers the CARD hand-off (never the screening): each message is still
fetched, hash-checked, screened through the unchanged ``screen_item`` path,
admitted and reserved one by one in ``handle_conversation_available``; only
the final ``deliver_wake`` is grouped, and the one card names every message it
covers (``message_ids:``, oldest → newest; ``message_id`` = the newest).

THE WINDOW IS A RATE, NOT A DELAY. A group flushes at the end of the connector
iteration that produced it when its conversation had no wake in the last
``window_s`` (so a single message on a quiet conversation is not slowed down),
and otherwise waits until ``window_s`` after the previous wake. That gives at
most one wake per window per (endpoint, conversation, thread, session) while a
burst that lands in one feed batch still produces exactly one card.

CRASH SAFETY IS WRITTEN BEFORE THE CARD CAN BE LOST, NEVER AFTER. The feed
ref is deleted by the loop as soon as its message is handled (once-only
consumption), so from ``enqueue`` onward the only records that a card is still
owed are the body reservation (still without its ``card_posted`` marker) and
the durable ``wake_coalescing`` hold this module writes AT ENQUEUE, keyed by
the message's own feed ref. A hard kill (SIGKILL, power loss) or a cancellation
between the ref delete and the flush therefore leaves a hold on the existing
retry schedule: the ordinary re-drive finds the owed card and posts it. That
ordering is the whole point — a hold written only at the END of an iteration
is a hold written after the window in which it was needed. On a successful
hand-off ``_settle`` marks the reservation posted and deletes the hold, so the
re-drive never fires in the normal path; and when the flush is itself
interrupted by a ``BaseException`` (a stop's cancellation, a SIGTERM, an
interrupt) it re-persists the group's holds and re-raises rather than
swallowing the stop.

The card-posted marker on the body reservation is still written only at
FLUSH, never earlier, so a reservation whose card is still buffered reads
``CARD_OWED`` — exactly the state a replay already knows how to finish.

Threads are their own groups: a reply card must name ONE thread root to route
an answer into it, so replies in two threads never share a card. Destination
sessions are their own groups too (B3, S-26): a card wakes ONE session, so a
reply routed to one session and a mention for another never share a card,
even in one thread inside one window (see ``GroupKey``).
"""

from __future__ import annotations

import contextvars
import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

#: Default coalescing window in seconds (wake contract: "~3 s, configurable").
#: Configurable per process through :class:`WakeCoalescer` — the connector
#: loop builds one with this default; tests and future settings pass another.
DEFAULT_COALESCE_WINDOW_S = 3.0

#: The hold code a still-buffered message carries across an iteration
#: boundary. Owner-visible like every hold code, and self-describing.
HOLD_WAKE_COALESCING = "wake_coalescing"

#: What a card hand-off returns when it BUFFERED the card instead of posting it.
CARD_COALESCED = "coalesced"


def card_is_buffered(detail: Any) -> bool:
    """True when a hand-off buffered its card, i.e. nothing is posted yet.

    THE ONE predicate for "is this delivery finished?". Both the first pass
    (``handle_conversation_available``'s terminal cleanup) and the hold re-drive
    (the connector loop's pass) decide it through this function, because the two
    drifting apart is precisely the lost wake of review round 2, M1′: the spine
    had been taught that a buffered hand-off is not terminal while the re-drive
    still deleted the row the enqueue had just written. A buffered card's only
    durable trace until the flush is that ``wake_coalescing`` row, which
    ``_settle`` retires once the card is actually posted.
    """

    return detail == CARD_COALESCED


#: Slack after a group's due time before its crash-recovery hold comes due, so
#: the in-memory flush always wins in a live process.
_HOLD_GRACE_S = 30.0

#: Upper bound on the messages one card names. A pathological burst past this
#: flushes early rather than growing an unbounded ``message_ids:`` line.
MAX_COALESCED_MESSAGES = 20

#: Bound on the "last wake" memory. A scheduling aid, not state worth growth.
_LAST_WAKE_CAP = 1024


@dataclass
class _Pending:
    """One buffered message: its card facts and its retry key."""

    metadata: Any  # ConversationCardMetadata
    ref_key: str
    mention: bool
    #: An urgent-failover standby (``conversation.classify_wake``): its row is still
    #: ``deferred`` and the flush must ask ``promote`` before it can wake anyone.
    staggered: bool = False


@dataclass
class _Group:
    owner: Any
    config_dir: Path
    session_id: str
    due_at: float
    items: list[_Pending] = field(default_factory=list)


#: (config dir, endpoint, conversation, thread root or "", destination session).
#: The config dir is part of the key because it IS the connector's identity: one
#: connector per config dir holds the singleton lock, and two config roots in one
#: process (tests, a future multi-owner host) must never share a rate window.
#:
#: The destination session is part of it because a card is delivered to ONE
#: session (``_Group.session_id``, fixed by the group's first enqueue). Once
#: replies are routed (B3, S-26), a reply routed to session S1 and a chief-of-
#: staff mention in the same thread can land inside one window; keyed without
#: the session they would merge into one card handed to whichever arrived first,
#: so one target never sees its message and the other is woken for a message
#: that was not routed to it. Two sessions are two wake targets, so ``_last_wake``
#: (which shares this key) rate-limits per destination too. With routing off
#: every card's session is the designated one, so the partition is unchanged.
GroupKey = tuple[str, str, str, str, str]

#: The scope the connector loop opens around one iteration. ``None`` (every
#: direct caller, including the existing unit tests) delivers immediately,
#: exactly as before this module existed.
_ACTIVE: contextvars.ContextVar["WakeCoalescer | None"] = contextvars.ContextVar(
    "team_bridge_wake_coalescer", default=None
)


def active_coalescer() -> "WakeCoalescer | None":
    return _ACTIVE.get()


def merge_metadata(items: list[Any]) -> Any:
    """The ONE card for several admitted messages (oldest → newest).

    The newest message supplies the locator (``message_id``, ``seq``) because
    reply routing binds to one id and the newest is what an answer follows.
    ``mention`` is the OR over the group: a mention anywhere in the burst makes
    the card directed. Thread direction needs no merge — a flush never mixes
    threads, one group per thread root. The sender is named only when every
    message shares one; otherwise the card says ``several members`` and the kind
    ``mixed``, so the agent-sender caution (which only applies when every
    message came from an agent) is never stamped on a human's line.
    """

    newest = items[-1]
    if len(items) == 1:
        return newest
    kinds = {item.sender_kind for item in items}
    labels = {item.sender_label for item in items}
    sender_kind = newest.sender_kind if len(kinds) == 1 else "mixed"
    displays = {item.session_agent_display for item in items}
    same_presentation = len(kinds) == len(labels) == len(displays) == 1
    # Names are presentation equality only, never endpoint/session routing keys.
    # Unknown session provenance cannot name an entire agent burst.
    if sender_kind == "agent" and newest.session_agent_display is None:
        same_presentation = False
    sender_label = newest.sender_label if same_presentation else "several members"
    session_display = newest.session_agent_display if same_presentation else None
    attachments = sum(int(item.attachments or 0) for item in items)
    note = next((item.attachments_note for item in reversed(items) if item.attachments_note), None)
    # Files coalesce the same way: the count is a sum, and one withheld line rides the
    # whole card because the agent reads every message of the group.
    files = sum(int(item.files or 0) for item in items)
    files_note = next((item.files_note for item in reversed(items) if item.files_note), None)
    return replace(
        newest,
        mention=any(item.mention for item in items),
        sender_kind=sender_kind,
        sender_label=sender_label,
        session_agent_display=session_display,
        attachments=attachments,
        attachments_note=note,
        files=files,
        files_note=files_note,
        coalesced_message_ids=tuple(item.message_id for item in items),
        # B4b: one released message in the burst puts the provenance line on the
        # whole card, because the agent reads every message of the group.
        owner_released=any(getattr(item, "owner_released", False) for item in items),
        # The gap BEFORE the burst: the oldest message's own count. Later
        # messages in the group follow it directly, so they add nothing.
        since_last_delivery=int(getattr(items[0], "since_last_delivery", 0) or 0),
        # Delayed delivery: the longest hold in the burst, so one late message
        # keeps its line on the shared card (the agent reads every message).
        held_for_s=max(int(getattr(item, "held_for_s", 0) or 0) for item in items),
    )


DeliverFn = Callable[..., Awaitable[str]]


class WakeCoalescer:
    """Per-process buffer of card hand-offs, keyed by endpoint/conversation/thread/session."""

    def __init__(self, *, window_s: float = DEFAULT_COALESCE_WINDOW_S) -> None:
        self.window_s = max(0.0, float(window_s))
        self._groups: dict[GroupKey, _Group] = {}
        self._last_wake: dict[GroupKey, float] = {}

    # -- scope ---------------------------------------------------------------

    def scope(self) -> "_Scope":
        return _Scope(self)

    # -- enqueue -------------------------------------------------------------

    def enqueue(
        self,
        owner: Any,
        metadata: Any,
        *,
        ref_key: str,
        session_id: str,
        config_dir: Path,
        now: float | None = None,
        min_interval_s: float | None = None,
        dwell_s: float | None = None,
        staggered_delay_s: float | None = None,
    ) -> None:
        """Buffer one admitted message's card. Idempotent per message id.

        ``dwell_s`` (channel-wake triage, operator 2026-11-21: batch messages that
        arrive close together) turns the rate into a short DELAY for a non-urgent
        wake: a new group is due ``dwell_s`` from now (still never before the rate
        window), so neighbours arriving meanwhile share its one card. ``0`` marks an
        IMMEDIATE wake (DM, mention, direct, owner): it pulls a dwelling group's due
        time back to the rate floor, so a mention is never held behind ambient
        chatter. ``None`` is the pre-dwell behaviour.

        ``staggered_delay_s`` buffers an urgent-failover standby in its OWN group
        (so an unrelated wake in the thread cannot flush it early), due after its
        slot's delay; the flush's ``promote`` hook decides whether it wakes at all.

        ``min_interval_s`` raises the gap after this destination's last wake
        above the ordinary window: the per-destination gate for ROUTED groups
        (``connector/routing.py``), whose target is not the session that always
        received everything. ``None`` keeps the ordinary window.

        The crash-recovery hold is written HERE, before the caller deletes the
        feed ref: everything after this point may not run (a stop, a kill, a
        cancel between the ref delete and the flush), and the hold is what a
        restart's ordinary re-drive looks for. A replay of a message already
        buffered (the hold re-driving while the group is still live) adds
        nothing and rewrites nothing.
        """

        now = time.time() if now is None else now
        staggered = staggered_delay_s is not None
        key = _group_key(config_dir, owner, metadata, session_id, staggered=staggered)
        group = self._groups.get(key)
        last = self._last_wake.get(key)
        # Rate, not delay: a quiet conversation is due NOW (flushed at the end of
        # this iteration); a recently woken one waits out the window.
        gap = self.window_s if min_interval_s is None else max(self.window_s, min_interval_s)
        floor = now if last is None else max(now, last + gap)
        if group is None:
            due = floor
            if staggered:
                due = max(floor, now + max(0.0, float(staggered_delay_s or 0.0)))
            elif dwell_s:
                due = max(floor, now + max(0.0, float(dwell_s)))
            group = _Group(owner=owner, config_dir=config_dir, session_id=session_id, due_at=due)
            self._groups[key] = group
        elif dwell_s is not None and dwell_s <= 0 and not staggered:
            group.due_at = min(group.due_at, floor)
        if any(item.metadata.message_id == metadata.message_id for item in group.items):
            return
        group.items.append(
            _Pending(
                metadata=metadata,
                ref_key=ref_key,
                mention=metadata.mention,
                staggered=staggered,
            )
        )
        group.items.sort(key=lambda item: item.metadata.conversation_seq)
        if len(group.items) >= MAX_COALESCED_MESSAGES:
            group.due_at = min(group.due_at, now)
        _persist_hold(owner, ref_key, config_dir, retry_at=group.due_at + _HOLD_GRACE_S)

    # -- timing --------------------------------------------------------------

    def seconds_until_due(self, *, now: float | None = None) -> float | None:
        """How long until the earliest buffered group is due (``None``: none)."""

        if not self._groups:
            return None
        now = time.time() if now is None else now
        return max(0.0, min(group.due_at for group in self._groups.values()) - now)

    def pending_keys(self) -> list[GroupKey]:
        return list(self._groups)

    # -- flush ---------------------------------------------------------------

    async def flush_due(
        self,
        *,
        deliver: DeliverFn,
        now: float | None = None,
        owner_filter: Any = None,
        config_dir: Path | None = None,
        promote: Callable[..., Awaitable[bool]] | None = None,
    ) -> list[dict[str, Any]]:
        """Deliver every due group: ONE ``deliver`` call per group.

        After a successful hand-off: mark every covered reservation's card
        posted, delete each message's crash-recovery hold, and record the wake
        time that rate-limits the next group. A failed hand-off keeps the group
        buffered for the next pass AND persists its holds, so neither an
        in-process retry nor a restart loses it.
        """

        now = time.time() if now is None else now
        outcomes: list[dict[str, Any]] = []
        for key in list(self._groups):
            group = self._groups[key]
            # Only THIS iteration's connector identity: an owner snapshot is
            # a value (equal across config roots), so the root is checked too.
            if owner_filter is not None and group.owner != owner_filter:
                continue
            if config_dir is not None and Path(group.config_dir) != Path(config_dir):
                continue
            if group.due_at > now or not group.items:
                continue
            # RE-CHECK AT FLUSH (the replay rule, applied to the buffer): a
            # group can outlive the state it was enqueued under — a hand-off
            # cancelled by a stop leaves it buffered while a replay or a body
            # turn settles the reservation. An item whose card is already
            # posted, or whose reservation is settled, must not wake anyone.
            #
            # An urgent-failover standby is still ``deferred`` here (settled), so
            # it is offered to ``promote`` FIRST: unanswered → its row becomes
            # ``pending`` and the owed check below keeps it; answered (or no hook)
            # → it stays deferred and the owed check drops it with no card.
            for item in group.items:
                if not item.staggered:
                    continue
                item.staggered = False
                if promote is None:
                    continue
                try:
                    await promote(group.owner, item.metadata, config_dir=group.config_dir)
                except Exception:  # noqa: BLE001 — closed code; the row stays deferred
                    logger.warning("bridge: staggered wake check failed code=promote_failed")
            group.items = _still_owed(group)
            if not group.items:
                # NOT a silent drop. Every item was already POSTED or SETTLED
                # (a replay settled it while the card sat buffered), so no wake
                # is owed — but the iteration's summary has to say that, because
                # "acked to the caller, woken nobody, no outcome recorded" is
                # indistinguishable from a lost card when a test or an owner
                # reads the summary afterwards (CI shard 1 remediation).
                #
                # NO delivery state is published here, and none is owed: the
                # contract (#3198) writes ``held`` from the hold paths and lets
                # ``delivered``/``refused`` ride the ADMIT write-once with the
                # verdict — an admit that already happened for every message in
                # this group, since only an admitted message is ever buffered.
                # A "delivered" write here would claim a wake this process did
                # not make (the very thing the verdict says it did not).
                del self._groups[key]
                logger.info(
                    "bridge: coalesced group settled before its flush code=card_already_posted"
                )
                outcomes.append(
                    {
                        "action": "already_posted",
                        "conversation_id": key[2],
                        "message_ids": [],
                    }
                )
                continue
            # ONE CARD PER PERSISTED HAND-OFF IDENTITY (PR #3478 QA r1). A replay
            # can re-group differently from the original flush (A+B went out,
            # B's marker failed, the retry buffers B with a new C). Re-merging
            # B+C under B's bound id would let the receiver swallow C as a
            # duplicate (a lost wake); a fresh id for B+C would post B twice.
            # So the group is split by the binding stored on each reservation:
            # members bound to one id re-send under it, unbound members get
            # their own new card. One partition in the ordinary case, so a
            # normal burst is still exactly one card.
            delivered_items: list[_Pending] = []
            failed_items: list[_Pending] = []
            for part in _handoff_partitions(group):
                metadata = merge_metadata([item.metadata for item in part])
                try:
                    detail = await deliver(
                        group.owner,
                        metadata,
                        session_id=group.session_id,
                        config_dir=group.config_dir,
                    )
                except Exception:  # noqa: BLE001 — closed code, never content
                    logger.warning("bridge: coalesced wake failed code=delivery_failed")
                    failed_items.extend(part)
                    outcomes.append({"action": "hold", "messages": len(part)})
                    continue
                except BaseException:
                    # A stop or an interrupt is NOT a failed delivery to retry in
                    # place: this process is going away and the buffer with it.
                    # The group's holds (written at enqueue) are refreshed past
                    # the grace period and the exception propagates, so the next
                    # start's ordinary hold retry re-drives every message in the
                    # group. Swallowing it here would strand the group in memory
                    # that is about to be discarded. Partitions already handed
                    # off are settled first so their markers are not lost too.
                    if delivered_items:
                        _settle(replace(group, items=delivered_items), now=now)
                        group.items = [i for i in group.items if i not in delivered_items]
                    _persist_holds(group, retry_at=time.time() + _HOLD_GRACE_S)
                    raise
                delivered_items.extend(part)
                outcomes.append(
                    {
                        "action": "delivered",
                        "conversation_id": metadata.conversation_id,
                        "message_ids": [item.metadata.message_id for item in part],
                        "detail": detail,
                    }
                )
            if delivered_items:
                _settle(replace(group, items=delivered_items), now=now)
                self._last_wake[key] = now
                if len(self._last_wake) > _LAST_WAKE_CAP:
                    self._last_wake.clear()
            if failed_items:
                group.items = failed_items
                group.due_at = now + max(self.window_s, 1.0)
                _persist_holds(group, retry_at=group.due_at + _HOLD_GRACE_S)
                continue
            del self._groups[key]
        return outcomes

    def persist_pending_holds(self, *, config_dir: Path | None = None) -> None:
        """Write a crash-recovery hold for every group still buffered."""

        for group in self._groups.values():
            if config_dir is not None and Path(group.config_dir) != Path(config_dir):
                continue
            _persist_holds(group, retry_at=group.due_at + _HOLD_GRACE_S)


class _Scope:
    """``with coalescer.scope():`` — card hand-offs inside are buffered."""

    def __init__(self, coalescer: WakeCoalescer) -> None:
        self._coalescer = coalescer
        self._token: contextvars.Token[Any] | None = None

    def __enter__(self) -> WakeCoalescer:
        self._token = _ACTIVE.set(self._coalescer)
        return self._coalescer

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _ACTIVE.reset(self._token)


def _group_key(
    config_dir: Path, owner: Any, metadata: Any, session_id: str, *, staggered: bool = False
) -> GroupKey:
    # session_id last: ``flush_due`` reads ``key[2]`` as the conversation id.
    # A staggered standby keys apart from the thread's ordinary group (a suffix
    # no thread id can carry) so it keeps its own due time and rate memory — and
    # PER MESSAGE (review r1 m2): ``due`` is set only when a group is created, so
    # a second urgent message in the same thread with another slot would inherit
    # the first one's due time and be checked early (a duplicate) or late.
    thread = metadata.thread_root_id or ""
    return (
        str(config_dir),
        str(getattr(owner, "endpoint_id", "") or ""),
        metadata.conversation_id,
        f"{thread}\x1fstaggered\x1f{metadata.message_id}" if staggered else thread,
        session_id,
    )


def _still_owed(group: _Group) -> list[_Pending]:
    """The buffered items whose card is still owed (see ``flush_due``).

    Same verdicts as ``handle_conversation_available``'s replay branch:
    ``CARD_POSTED`` → drop; ``CARD_SETTLED`` → drop and record the card as
    posted so later replays stay quiet; ``CARD_OWED`` or no reservation (the
    unadmitted recovery card has none) → keep. A store that cannot answer
    keeps the item: one possibly-duplicate card beats a lost wake.
    """

    from local_operator.team_bridge.store import (
        CARD_POSTED,
        CARD_SETTLED,
        Store,
        StoreRefused,
    )

    store = Store(group.config_dir)
    kept: list[_Pending] = []
    for item in group.items:
        conversation_id = item.metadata.conversation_id
        message_id = item.metadata.message_id
        try:
            status = store.body_card_status(group.owner, conversation_id, message_id)
        except StoreRefused:
            status = None
        if status == CARD_SETTLED:
            try:
                store.mark_body_card_posted(
                    group.owner, conversation_id, message_id, now=time.time()
                )
            except StoreRefused:
                pass
        if status in (CARD_POSTED, CARD_SETTLED):
            # The VERDICT, on the record: ``posted`` means a card was already
            # handed off; ``settled`` means the reservation left ``pending`` for
            # another reason (a body turn claimed it, or it completed) — two
            # different causes that used to look identical from outside, as an
            # empty transcript with no reason at all (CI shard 1). Closed enum
            # values only: never a body, an id or a label. The store is NOT
            # extended for this: its schema is a migration-versioned surface and
            # another lane is mid-change there (wake contract, hotfix lane).
            logger.info("bridge: coalesced card dropped code=card_not_owed status=%s", status)
            try:
                store.delete_hold(group.owner, item.ref_key)
            except StoreRefused:
                pass
            continue
        kept.append(item)
    return kept


def _handoff_partitions(group: _Group) -> list[list[_Pending]]:
    """Split a group by each reservation's persisted card hand-off identity.

    Keys: the bound ``peer-`` id (members of ONE earlier card), ``''`` (never
    handed off: they share one new card), and the store's sentinels for a
    legacy or unreserved row, each kept apart so ``_deliver_card`` can hold or
    pass them without contaminating another partition. Order follows the
    group's own ``conversation_seq`` order. A store that cannot answer keeps
    the whole group as ONE partition, i.e. exactly the pre-change behaviour,
    and the bind inside the hand-off then refuses or proceeds on its own read.
    """

    from local_operator.team_bridge.store import Store, StoreRefused

    by_conversation: dict[str, list[str]] = {}
    for item in group.items:
        by_conversation.setdefault(item.metadata.conversation_id, []).append(
            item.metadata.message_id
        )
    bindings: dict[tuple[str, str], str | None] = {}
    try:
        store = Store(group.config_dir)
        for conversation_id, ids in by_conversation.items():
            for mid, value in store.card_handoff_bindings(
                group.owner, conversation_id, ids
            ).items():
                bindings[(conversation_id, mid)] = value
    except StoreRefused:
        return [list(group.items)]
    parts: dict[str | None, list[_Pending]] = {}
    for item in group.items:
        key = bindings.get((item.metadata.conversation_id, item.metadata.message_id))
        parts.setdefault(key, []).append(item)
    return list(parts.values())


def _persist_hold(owner: Any, ref_key: str, config_dir: Path, *, retry_at: float) -> None:
    from local_operator.team_bridge.store import Store, StoreRefused

    try:
        Store(config_dir).persist_hold(owner, ref_key, HOLD_WAKE_COALESCING, retry_at=retry_at)
    except StoreRefused:
        pass  # fenced owner: the next live generation owns its own rows


def _persist_holds(group: _Group, *, retry_at: float) -> None:
    for item in group.items:
        _persist_hold(group.owner, item.ref_key, group.config_dir, retry_at=retry_at)


def _settle(group: _Group, *, now: float) -> None:
    from local_operator.team_bridge.store import Store, StoreRefused

    store = Store(group.config_dir)
    for item in group.items:
        try:
            store.mark_body_card_posted(
                group.owner, item.metadata.conversation_id, item.metadata.message_id, now=now
            )
        except StoreRefused:
            logger.warning("bridge: card posted, marker unrecorded code=store_refused")
        try:
            store.delete_hold(group.owner, item.ref_key)
        except StoreRefused:
            pass


#: THE connector process's coalescer. Module-level like the loop's own
#: ``_RETRY_ATTEMPTS`` scheduling aid: one connector per config dir holds the
#: singleton lock, so one buffer per process is one buffer per endpoint.
CONNECTOR_COALESCER = WakeCoalescer()


__all__ = [
    "CARD_COALESCED",
    "CONNECTOR_COALESCER",
    "DEFAULT_COALESCE_WINDOW_S",
    "HOLD_WAKE_COALESCING",
    "MAX_COALESCED_MESSAGES",
    "WakeCoalescer",
    "active_coalescer",
    "card_is_buffered",
    "merge_metadata",
]
