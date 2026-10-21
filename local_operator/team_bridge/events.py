"""H5e — feed-ref routing: one vocabulary for what each ref means.

WHY THIS EXISTS. The events feed (§6.7) emits five typed refs; the connector
wakes on exactly one of them and the rest are "the CoS session's own reads'
business". That rule lived as an inline condition in the connector loop, which
is fine for ONE consumer — but a second consumer (the CoS refresh) must agree
about the same vocabulary, and two consumers each re-deciding what a
``request_decided`` means is how a routing table drifts. :func:`plan_feed_refs`
states the rule ONCE:

- ``message_available:message_id=…`` → a DELIVERY: screen-then-wake's input.
- ``state_changed`` / ``request_decided`` / ``request_revoked`` /
  ``request_expired`` → a REFRESH: the CoS re-reads its own state. **No new
  wake kind** — these never produce a wake row (§13.4's kind list is closed);
  the decision courier for a requester already arrives as ``message_available``.
- anything malformed/unrecognised → skipped, never guessed at (the parser's
  own strictness rule, restated here because a router that guesses is worse
  than one that drops).

The parser itself is ``connector/wake.parse_event_ref`` — one strictness
spelling, imported rather than duplicated.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from .connector.wake import parse_event_ref
from .types import (
    CONVERSATION_FEED_KIND,
    ConversationFeedRef,
    parse_conversation_ref_value,
)

logger = logging.getLogger(__name__)

#: The one ref shape that becomes a delivery (and therefore a wake).
DELIVERY_TYPE = "message_available"

#: The ref types that mean "re-read your own state" — never a delivery, never
#: a wake. The CoS consumes :attr:`FeedRefPlan.refresh` as a hint to refresh.
REFRESH_TYPES = (
    "state_changed",
    "request_decided",
    "request_revoked",
    "request_expired",
)


@dataclass(frozen=True)
class FeedRefPlan:
    """What one drained ref batch asks the client to do."""

    deliveries: tuple[str, ...]
    refresh: bool
    refresh_types: tuple[str, ...]
    skipped: tuple[str, ...]
    conversation_deliveries: tuple[ConversationFeedRef, ...] = ()


def plan_feed_refs(refs: Sequence[str]) -> FeedRefPlan:
    """Classify a drained batch. Pure; no network, no store, no wake."""

    deliveries: list[str] = []
    conversation_deliveries: list[ConversationFeedRef] = []
    refresh_types: list[str] = []
    skipped: list[str] = []
    for ref in refs:
        parsed = parse_event_ref(ref)
        if parsed is None:
            skipped.append(ref)
            continue
        event_type, kind, value = parsed
        if event_type == DELIVERY_TYPE and kind == CONVERSATION_FEED_KIND:
            # A conversation delivery is NEVER a pair delivery: it goes to the
            # conversation card path, which owns the history/read, admission
            # and mention-aware framing.
            conversation_ref = parse_conversation_ref_value(value)
            if conversation_ref is None:
                skipped.append(ref)
            else:
                conversation_deliveries.append(conversation_ref)
            continue
        if event_type == DELIVERY_TYPE and kind == "message_id":
            deliveries.append(value)
            continue
        if event_type in REFRESH_TYPES:
            refresh_types.append(event_type)
            continue
        skipped.append(ref)
    if skipped:
        # TB-FULL-1: a bounded triage signal for the drained batch — count plus
        # at most five names; resync repairs gaps, so this is hygiene, not loss.
        logger.debug(
            "bridge events: %d unrecognised refs skipped: %s",
            len(skipped),
            ", ".join(skipped[:5]),
        )
    return FeedRefPlan(
        deliveries=tuple(deliveries),
        refresh=bool(refresh_types),
        refresh_types=tuple(refresh_types),
        skipped=tuple(skipped),
        conversation_deliveries=tuple(conversation_deliveries),
    )


__all__ = ["DELIVERY_TYPE", "REFRESH_TYPES", "FeedRefPlan", "plan_feed_refs"]
