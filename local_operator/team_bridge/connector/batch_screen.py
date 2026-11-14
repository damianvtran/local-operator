"""Screen a connector iteration's conversation messages in batches.

WHY THIS EXISTS (2026-11-03). Agent replies are no longer capped by the hub
(agent-server #294 removed its wake caps), so the client is what bounds a
recipient owner's screening spend. Wake coalescing (``coalesce.py``) already
folds a burst into ONE card, but each message used to be screened by its own
model call, charged as its own unit of the account's hourly screening budget
(``team_bridge.screening.*``; ``team_bridge.settings.ScreeningBudget`` states
the live value), so a busy room spent the budget in minutes. This pass runs
FIRST in an iteration: it fetches each conversation message's body once, groups
the messages by conversation,
splits each group to fit the screening model (``plan_screening_batches``), and
screens every split in ONE model call that returns one verdict per message.

WHAT IT DOES NOT CHANGE. It decides nothing on its own. The per-message
handler (``handle_conversation_available``) still claims, hash-checks, admits
with that message's OWN ``body_hash`` and hands off the card, exactly as
before; it reads this pass's result for its message instead of making a model
call of its own, and reuses the body read so the hub sees one GET per
message, not two. A message this pass could not cover (an image message, a
fetch failure, a busy account turn, a stale body hash) falls through to the
handler's unchanged single-message path, whose holds are the fail-closed
answer. Who is woken is untouched: only screening ORDER and the budget UNIT
change.

ORDER. Priority messages (sent by a human, a mention of this agent, or a DM)
are screened before ambient agent chatter, and within each class the oldest
first, so a flood can never make the owner's own DM wait behind it; the
store's ambient ceiling (``Store.SCREENING_AMBIENT_BUDGET_PER_HOUR``) keeps a
reserve of the account's hour for them besides.

The results live in a context variable opened by :func:`prescreen_scope`
around ONE iteration (and one retry pass), so nothing crosses iterations: a
result is only ever consumed by the handler of the same message, the same
body hash, in the same pass.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..types import ConversationFeedRef

logger = logging.getLogger(__name__)


@dataclass
class _Prescreened:
    """One iteration's batch outcomes, keyed by (conversation, message)."""

    #: The body read the batch screened, reused by the handler.
    fetched: dict[tuple[str, str], Any] = field(default_factory=dict)
    #: ``(body_hash, ScreenResult)`` for every message a batch call covered.
    results: dict[tuple[str, str], tuple[str, Any]] = field(default_factory=dict)
    #: Model calls this pass made (observability and the measurement).
    calls: int = 0
    #: Messages whose batch this pass VOIDED (``batch_unresolved``). The
    #: handler reads it to mark the hold row ``no_batch`` (#3400 round 2).
    voided: set[tuple[str, str]] = field(default_factory=set)


_ACTIVE: contextvars.ContextVar["_Prescreened | None"] = contextvars.ContextVar(
    "team_bridge_prescreened", default=None
)


@contextlib.contextmanager
def prescreen_scope() -> Iterator[_Prescreened]:
    """Open one pass's result table; the handler reads it, nothing outlives it."""

    state = _Prescreened()
    token = _ACTIVE.set(state)
    try:
        yield state
    finally:
        _ACTIVE.reset(token)


def take_fetched(ref: ConversationFeedRef) -> Any:
    """The body read this pass already made for ``ref`` (consumed once), or None."""

    state = _ACTIVE.get()
    if state is None:
        return None
    return state.fetched.pop((ref.conversation_id, ref.message_id), None)


def take_result(ref: ConversationFeedRef, body_hash: str) -> Any:
    """This pass's screen of ``ref`` for EXACTLY ``body_hash``, or None.

    A different hash means the handler read different bytes than the batch
    screened, so the batch result is not about this body and the handler
    screens it itself.
    """

    state = _ACTIVE.get()
    if state is None:
        return None
    entry = state.results.pop((ref.conversation_id, ref.message_id), None)
    if entry is None or entry[0] != body_hash:
        return None
    from ..screening import HOLD_BATCH_UNRESOLVED

    if getattr(entry[1], "hold_code", None) == HOLD_BATCH_UNRESOLVED:
        state.voided.add((ref.conversation_id, ref.message_id))
        # The batch could not conclude this message (a neighbour's non-deliver
        # verdict, or a refused / malformed call): its verdict must come from
        # a call that saw it ALONE (#3400 S-1, M2/S-6), so the handler screens
        # it singly now, never re-batched with the same group.
        return None
    return entry[1]


def came_from_void_batch(ref: ConversationFeedRef) -> bool:
    """Whether THIS pass voided a batch that carried ``ref``.

    Such a message is screened alone from now on (#3400 round 2, R2-M1 =
    S-7): the handler marks its hold row ``no_batch`` when that single screen
    holds (most often refused for budget), so a refill can never put it back
    in a batch with the message that voided the first one.
    """

    state = _ACTIVE.get()
    return state is not None and (ref.conversation_id, ref.message_id) in state.voided


def allocation_for(ref: ConversationFeedRef, display: Mapping[str, Any]) -> dict[str, Any]:
    """The budget allocation of one message, from hub-verified facts only.

    Tier (``Store.SCREENING_TIER_*``), keyed on the sender KIND the hub
    states (``sender_kind`` on the body read), because an agent can mention
    anyone (#3400 security round 1, S-4): HUMAN_ADDRESSED for a person's DM
    or mention of this agent; ADDRESSED for an agent's; HUMAN for a person's
    line in a shared room; AMBIENT for the rest. An unknown sender kind (an
    older hub) is treated as an agent: it never reaches a person's reserve.
    THE ONE allocation source: the batch pass and the handler's single
    screen both call this, so no screen is ever charged a default tier. It
    returns the TIER and nothing else: the charged sender is the envelope's
    verified account (security round 1 S-3), so a ``sender_account_id`` here
    would be a second identity rule that could drift from it — the hint the
    connector stamps never decides who pays (round 1 S-9). Display LABELS
    never decide the tier either.
    """

    kind = display.get("sender_kind")
    conversation_kind = display.get("conversation_kind")
    from ..store import Store

    addressed = bool(ref.mention) or conversation_kind == "dm"
    human = kind == "human"
    if addressed and human:
        tier = Store.SCREENING_TIER_HUMAN_ADDRESSED
    elif addressed:
        tier = Store.SCREENING_TIER_ADDRESSED
    elif human:
        tier = Store.SCREENING_TIER_HUMAN
    else:
        tier = Store.SCREENING_TIER_AMBIENT
    return {"tier": tier}


async def prescreen_conversation_refs(
    owner: Any,
    refs: Sequence[ConversationFeedRef],
    *,
    config_dir: Path,
    access: Any,
    auth_store: Any,
    settings: Any = None,
    wire: Any = None,
) -> int:
    """Batch-screen ``refs`` into the open scope; returns the model calls made.

    Best effort by construction: every failure here leaves the message to the
    handler's own single-message path, so this pass can make screening
    cheaper but never less safe.
    """

    state = _ACTIVE.get()
    if state is None or len(refs) < 2:
        return 0
    from ..store import Store
    from . import conversation as _conversation
    from . import wake as _wake

    store = Store(config_dir)
    # The same cheap gates the handler applies first: a fenced or unengaged
    # owner would hold every one of these before any screen, so screening
    # them here would be spend with no admission behind it.
    # Read through the conversation module's own names, the very seam its
    # handler uses, so the two can never disagree on whether to screen.
    if _conversation._producer_fence(store, owner, access, settings):
        return 0
    designation = store.designated_session_for(owner)
    if designation is None or _conversation.engagement_hold(designation, config_dir):
        return 0
    client = (
        wire if wire is not None else await _conversation._wire_for(owner, config_dir, auth_store)
    )

    candidates: list[tuple[ConversationFeedRef, Any, dict[str, Any]]] = []
    for ref in refs:
        try:
            fetched = await _conversation._fetch_body(client, ref, owner, access)
        except Exception:  # noqa: BLE001 — the handler re-reads and decides
            continue
        if fetched is None:
            continue
        state.fetched[(ref.conversation_id, ref.message_id)] = fetched
        if fetched.attachments is None or fetched.attachments:
            # Images are never batched: an image verdict must come from the
            # v3 render that showed the pixels (screening module note). Nor
            # are descriptors this build cannot parse (``None``): the handler
            # holds those ``attachments_malformed`` before any screen, so a
            # batch unit spent on them would buy a receipt nothing admits.
            continue
        allocation = allocation_for(ref, fetched.display)
        try:
            sender = _conversation._allocation_sender(fetched.display)
        except ValueError:
            # The single handler records the malformed read (``sender_malformed``)
            # without spend; batching it would charge a unit for a hold.
            continue
        item = _conversation._screening_item(
            owner=owner,
            conversation_id=ref.conversation_id,
            message_id=ref.message_id,
            body=fetched.body,
            body_hash=fetched.body_hash,
            sender=sender,
            **_conversation.screening_routing(ref, fetched.display),
        )
        item["allocation"] = allocation
        candidates.append((ref, fetched, item))
    if len(candidates) < 2:
        return 0

    from ..screening import plan_screening_batches

    # Highest tier first, then oldest first (feed order is the hub's own seq
    # per conversation); one group per (tier, conversation) so a call reads one
    # room and is charged to one pool.
    def _order(entry: tuple[ConversationFeedRef, Any, dict[str, Any]]) -> tuple[int, int]:
        return (-int(entry[2]["allocation"]["tier"]), entry[0].conversation_seq)

    groups: dict[tuple[int, str], list[tuple[ConversationFeedRef, Any, dict[str, Any]]]] = {}
    for entry in sorted(candidates, key=_order):
        key = (-int(entry[2]["allocation"]["tier"]), entry[0].conversation_id)
        groups.setdefault(key, []).append(entry)
    calls = 0
    for key in sorted(groups, key=lambda k: (k[0], groups[k][0][0].conversation_seq)):
        group = groups[key]
        sizes = [len(entry[1].body.encode("utf-8", "replace")) for entry in group]
        for split in plan_screening_batches(sizes):
            chunk = [group[index] for index in split]
            if len(chunk) < 2:
                continue  # one message is the handler's ordinary single screen
            try:
                results = await _wake.screen_items(
                    owner,
                    [entry[2] for entry in chunk],
                    config_dir=config_dir,
                    auth_store=auth_store,
                    settings=settings,
                )
            except Exception:  # noqa: BLE001 — the handler screens each itself
                logger.info("bridge: batch screen skipped code=batch_failed")
                continue
            if not isinstance(results, list) or len(results) != len(chunk):
                continue  # busy turn / no default: the single path decides
            calls += 1
            for (ref, fetched, _item), result in zip(chunk, results):
                state.results[(ref.conversation_id, ref.message_id)] = (fetched.body_hash, result)
    state.calls += calls
    return calls


__all__ = [
    "allocation_for",
    "prescreen_conversation_refs",
    "prescreen_scope",
    "take_fetched",
    "take_result",
]
