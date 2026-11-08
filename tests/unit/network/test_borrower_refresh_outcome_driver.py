"""The owner's could-not-refresh answer, end to end through the failover driver.

The cells in ``test_borrower_refresh_outcome.py`` pin what the borrower ASKS and
SAYS against a scripted owner. These run the REAL owner broker (two logins on its
own ``AuthStore``, its forced refresh, its grant cascade) behind the real leg-1/2
ops, and the REAL ``stream_with_failover`` on the borrower, with a provider that
answers 401 for one bearer — the two places round 1 of #3616 found the per-key
hold wrong: it hid a working sibling login (M1/Q1), and the turn that first
heard the owner's answer still ended on the bare provider 401 (Q2).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_operator.providers.failover import ProviderError
from tests.unit.network.test_credential_limit_skip import (  # noqa: F401 — fixtures by import
    _FIXTURES,
    StubProvider,
    _ended,
    _turn,
    borrowed,
)
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    idp,
    owner,
    stub_provider,
)
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixture by import

# Fixtures registered by the imports above; kept bound so the registration holds.
_KEEP = (_FIXTURES, borrowed, owner, idp, stub_provider, devices)


class _Refuses401(StubProvider):
    """An OpenAI-compatible endpoint: 401 for the dead bearers, 200 otherwise."""

    def __init__(self, *dead: str) -> None:
        super().__init__("")
        self.dead = set(dead)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        self.seen.append(bearer)
        if bearer in self.dead:
            return httpx.Response(
                401, json={"error": {"message": "invalid api key", "type": "invalid_request"}}
            )
        chunk = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
        return httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )


@pytest.mark.asyncio
async def test_a_dead_login_the_owner_cannot_refresh_never_blocks_its_sibling(
    borrowed: Any,
) -> None:
    """#3616 M1/Q1: login A is refused and cannot be refreshed; login B serves, silently."""
    # A first for this session, the incident's shape; A has no refresh token, so the
    # owner's forced refresh of it cannot produce a new bearer.
    borrowed.owner.auth.deprioritize_credential(STUB_PROVIDER, borrowed.row_b.id)
    stub = _Refuses401("bearer-a")

    first = await _turn(borrowed.store, stub, STUB_PROVIDER, "sess-401")
    assert _ended(first), f"the sibling login never served: {first} / {stub.seen}"
    assert stub.seen[0] == "bearer-a" and stub.seen[-1] == "bearer-b", stub.seen
    assert borrowed.store.no_credential_reason(STUB_PROVIDER) is None
    assert borrowed.client.refresh_hold(STUB_PROVIDER, borrowed.row_a.id) is not None

    # Next turn: the held login is hidden from the ask, so it is not re-lent and
    # re-refused; B serves straight away.
    stub.seen.clear()
    second = await _turn(borrowed.store, stub, STUB_PROVIDER, "sess-401")
    assert _ended(second), second
    assert stub.seen == ["bearer-b"], stub.seen
    assert borrowed.relay.grant_frames[-1].get("exclude_credential_ids") == [borrowed.row_a.id]


@pytest.mark.asyncio
async def test_the_first_failing_turn_ends_on_the_owners_answer_not_the_bare_401(
    borrowed: Any,
) -> None:
    """#3616 QA Q2: the turn that hears the owner's answer shows its remedy."""
    borrowed.owner.auth.delete_credential(borrowed.row_b.id)  # A is the only login
    stub = _Refuses401("bearer-a")

    with pytest.raises(ProviderError) as raised:
        await _turn(borrowed.store, stub, STUB_PROVIDER, "sess-401")
    text = str(raised.value)
    assert "HTTP 401" not in text, text
    assert "owner-laptop" in text and f"'lop login {STUB_PROVIDER}'" in text, text
    assert stub.seen.count("bearer-a") >= 1, stub.seen
