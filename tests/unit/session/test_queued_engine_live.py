"""``FrontendStateStore.queued_engine_live``: the clone-free capability read.

Voice mode refuses speech while a CARD is parked. A queued ask is mirrored onto
``pending_gate`` for old clients but parks nothing, so voice has to tell the two
apart. The wire's rule is that the presence of ``asks`` or ``asks_open`` IS the
capability flag; this pins the store's spelling of it.
"""

from __future__ import annotations

from typing import Any

from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
    PendingGateState,
)


def _store(**fields: Any) -> FrontendStateStore:
    return FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e1", **fields))


def test_the_engine_is_not_live_when_the_state_publishes_no_ask_fields() -> None:
    """``LOP_ASK_NONBLOCKING=0``: neither field is on the state."""
    assert _store().queued_engine_live is False


def test_asks_open_alone_makes_the_engine_live_even_at_zero() -> None:
    """Zero is a value: the empty-but-live queue."""
    assert _store(asks_open=0).queued_engine_live is True
    assert _store(asks_open=2).queued_engine_live is True


def test_a_gate_alone_never_says_the_engine_is_live() -> None:
    """An ask-kind gate with no ask fields is a BLOCKING ask, which must still block."""
    store = _store(
        pending_gate=PendingGateState(request_id="ask-1", kind="ask", title="Which one?")
    )
    assert store.pending_gate is not None
    assert store.queued_engine_live is False
