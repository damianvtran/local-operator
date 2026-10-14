"""H5e — feed-ref routing: deliveries wake, request_*/state_changed REFRESH,
nothing else is guessed at. One vocabulary for the connector and the CoS."""

from __future__ import annotations

from local_operator.team_bridge.events import plan_feed_refs

DELIVERY = "message_available:message_id=bmsg_1"
DECIDED = "request_decided:message_id=bmsg_2"
REVOKED = "request_revoked:message_id=bmsg_3"
EXPIRED = "request_expired:message_id=bmsg_4"
STATE = "state_changed:message_id=bmsg_5"


def test_deliveries_and_refreshes_split_cleanly() -> None:
    plan = plan_feed_refs([DELIVERY, DECIDED, REVOKED, EXPIRED, STATE])
    assert plan.deliveries == ("bmsg_1",)
    assert plan.refresh is True
    assert plan.refresh_types == ("request_decided", "request_revoked", "request_expired", "state_changed")
    assert plan.skipped == ()


def test_request_refs_never_produce_a_delivery() -> None:
    """The wake rule: only ``message_available:message_id`` proceeds to
    screening/wake — a settlement of OUR OWN request is a refresh, never a
    wake (no new wake kind; §13.4's kind list is closed)."""
    plan = plan_feed_refs([DECIDED, EXPIRED, STATE])
    assert plan.deliveries == ()


def test_malformed_refs_are_skipped_not_guessed() -> None:
    plan = plan_feed_refs(["garbage", "message_available:kinds=bmsg_1", "message_available:message_id="])
    assert plan.deliveries == () and plan.refresh is False
    assert len(plan.skipped) == 3


def test_a_quiet_batch_asks_for_nothing() -> None:
    plan = plan_feed_refs([])
    assert plan.deliveries == () and plan.refresh is False and plan.skipped == ()
