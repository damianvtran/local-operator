"""The queued aside adopt across the attach transport (decisions D3/D4).

Two facts cross the wire and each is pinned at both ends here:

* the ack's ``adopt_state`` — the owner's handle answers ``AckDetail`` with
  ``{"adopt_state": "queued" | "applied"}`` and the attach client returns it,
  reading an absent field (an owner from before the queue) as ``applied``;
* the typed refusal ``AsideAdoptRefused`` — ``error_code`` plus the closed-set
  ``error_reason`` token, rebuilt into the local sentence on the client side, so
  the desktop ladder can answer a 409 instead of the 503 owner-outage answer.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.mobile.attach_client import AttachClient
from local_operator.session.errors import (
    ASIDE_ADOPT_REFUSAL_SENTENCES,
    AsideAdoptRefused,
    admission_error,
)
from local_operator.session.runtime.server import AckDetail, _ack_frame_parts
from tests.unit.session.runtime.test_server_fork_refusal_facts import _error_frame


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", sorted(ASIDE_ADOPT_REFUSAL_SENTENCES))
async def test_each_refusal_reason_crosses_and_rebuilds_its_sentence(reason: str) -> None:
    frame = await _error_frame(AsideAdoptRefused(reason=reason))

    assert frame["error_code"] == "aside_adopt_refused"
    assert frame["error_reason"] == reason
    known = admission_error(frame["error_code"], reason=frame.get("error_reason"))
    assert isinstance(known, AsideAdoptRefused)
    assert known.reason == reason
    assert str(known) == ASIDE_ADOPT_REFUSAL_SENTENCES[reason]


def test_an_unknown_reason_degrades_to_the_generic_sentence() -> None:
    """A NEWER owner's token never renders; the fallback is true of every cause."""
    client = object.__new__(AttachClient)
    with pytest.raises(AsideAdoptRefused) as refused:
        client._raise_for_reply_error(
            {
                "op": "error",
                "req": 1,
                "error_code": "aside_adopt_refused",
                "error_reason": "a-cause-from-the-future",
                "message": "IGNORED",
            }
        )
    assert refused.value.reason == ""
    assert str(refused.value) == AsideAdoptRefused.fallback


def test_the_ack_carries_the_adopt_state() -> None:
    detail, extra = _ack_frame_parts(AckDetail("queued 1", {}, {"adopt_state": "queued"}))
    assert detail == "queued 1"
    assert extra["adopt_state"] == "queued"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ({"op": "ack", "detail": "q", "adopt_state": "queued"}, "queued"),
        ({"op": "ack", "detail": "a", "adopt_state": "applied"}, "applied"),
        # An owner from before the queue sends no field; it applied (or refused).
        ({"op": "ack", "detail": "forked 1 aside exchange(s) into the chat"}, "applied"),
    ],
)
async def test_the_attach_client_returns_the_owner_state(
    reply: dict[str, Any], expected: str
) -> None:
    client = object.__new__(AttachClient)

    async def frame(op: str, **_fields: Any) -> dict[str, Any]:
        assert op == "adopt_aside"
        return reply

    client._request_frame = frame  # type: ignore[method-assign]
    assert await client.adopt_aside([]) == expected


@pytest.mark.asyncio
async def test_the_owner_handle_reports_queued_and_applied() -> None:
    """The owner's handle wraps the session's answer in the ack field."""
    from local_operator.session.runtime.serving import ServingSessionHandle

    class _Session:
        def __init__(self, state: Any) -> None:
            self.state = state

        async def adopt_aside(self, messages: list[Any]) -> Any:
            return self.state

    for state, expected in (("queued", "queued"), ("applied", "applied"), (None, "applied")):
        # Construction-free: no ``_loop`` means the loop hop runs inline.
        handle = object.__new__(ServingSessionHandle)
        handle._session = _Session(state)  # type: ignore[assignment]
        handle._notify = lambda: None  # type: ignore[method-assign]
        ack = await handle.adopt_aside([])
        assert isinstance(ack, AckDetail)
        assert ack.fields == {"adopt_state": expected}
