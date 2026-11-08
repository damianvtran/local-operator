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

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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
    borrowed: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#3616 QA Q2 + design D6: the first turn says the owner's remedy, and only that.

    The first failing turn reaches the sentence through the failover hook, later
    turns through ``_no_credential_error``; both must read IDENTICALLY, and neither
    may carry the LOCAL re-login hint ("To fix: /login …" here), which the TUI and
    headless renderers add to anything that renders under the auth label.
    """
    borrowed.owner.auth.delete_credential(borrowed.row_b.id)  # A is the only login
    stub = _Refuses401("bearer-a")

    with pytest.raises(ProviderError) as raised:
        await _turn(borrowed.store, stub, STUB_PROVIDER, "sess-401")
    first = str(raised.value)
    assert "HTTP 401" not in first, first
    assert "on owner-laptop" in first and f"'lop login {STUB_PROVIDER}'" in first, first
    assert stub.seen.count("bearer-a") >= 1, stub.seen

    with pytest.raises(ProviderError) as again:
        await _turn(borrowed.store, stub, STUB_PROVIDER, "sess-401")
    assert str(again.value) == first, (str(again.value), first)

    # The two display paths that append the local auth hint, fed the turn's text.
    from rich.console import Console

    from local_operator.harness.types import AgentEndEvent
    from local_operator.headless_print import PrintRenderer
    from local_operator.providers.failover import append_auth_recovery

    assert append_auth_recovery(first, STUB_PROVIDER) == first
    sink = io.StringIO()
    PrintRenderer(Console(file=sink, width=400, highlight=False)).handle(AgentEndEvent(error=first))
    headless = sink.getvalue()
    assert first in headless and "To fix:" not in headless, headless

    from local_operator.tui.app import OperatorApp

    monkeypatch.delenv("RADIENT_API_KEY", raising=False)
    app = OperatorApp(lambda: None)  # type: ignore[arg-type]
    app._session = MagicMock(model_label=f"{STUB_PROVIDER}/claude-opus-5-5")  # noqa: SLF001
    tui = app._with_recovery_hint(first)  # noqa: SLF001
    assert tui == first, tui
