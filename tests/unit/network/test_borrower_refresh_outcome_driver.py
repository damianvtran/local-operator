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
import time
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


@pytest.mark.asyncio
async def test_a_401_after_the_owner_signed_that_row_in_again_reaches_the_owner(
    borrowed: Any, idp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#3616 QA follow-up, end to end: a re-signed-in login is reported, so the
    OWNER's remedy — never this device's own ``To fix: /login …``.

    The defect this reproduces: the owner's report window was keyed on the ROW, and a
    re-sign-in on the owner reuses the row. So the first 401 on the bearer MINTED
    after that sign-in was answered ``coalesced`` with nothing probed; the borrower got
    no owner answer, no hold, and its turn ended on ``authentication failed (HTTP
    401): invalid api key`` — to which the TUI and headless renderers append the LOCAL
    re-login remedy, sending the person to fix the wrong device. That is the class
    #3616 closed on its other paths.

    Both halves are asserted here: the owner really was asked (the probe's POST is the
    evidence, and it is refused because this IdP `still` refuses the grant), and what
    the two display surfaces make of the resulting text.
    """
    owner = borrowed.owner
    owner.auth.delete_credential(borrowed.row_b.id)  # A is the only login
    # A WITH A REFRESH TOKEN, so the owner's probe can really ask the IdP. Without one
    # the store refuses before any POST and the reply would be about a row it cannot
    # refresh at all, which is a different arm from the one under test.
    owner.auth.upsert_credential(
        STUB_PROVIDER,
        {
            "type": "oauth",
            "access": "bearer-a",
            "expires": int(time.time() * 1000) + 3_600_000,
            "refresh": "refresh-0",
            "email": "owner@example.test",
        },
    )
    # The IdP refuses the grant: the owner's probe is answered `invalid_grant`, so its
    # verdict — not the provider's, not this device's — is what the turn must carry.
    idp.invalid = True
    stub = _Refuses401("bearer-a", "bearer-a2")

    with pytest.raises(ProviderError) as lost:
        await _turn(borrowed.store, stub, STUB_PROVIDER, "sess-401")
    assert "HTTP 401" not in str(lost.value), str(lost.value)
    assert borrowed.client.refresh_hold(STUB_PROVIDER, borrowed.row_a.id) is not None
    posts_after_first = len(idp.posts)
    assert posts_after_first >= 1, "the owner never asked the IdP anything"

    # THE OPERATOR SIGNS IN AGAIN on the owner, onto the same row: same id, new
    # material, and the store's own tombstone is cleared by the interactive write.
    # The bearer written here is the one the borrower is lent next.
    owner.auth.upsert_credential(
        STUB_PROVIDER,
        {
            "type": "oauth",
            "access": "bearer-a2",
            "expires": int(time.time() * 1000) + 3_600_000,
            "refresh": "refresh-1",
            "email": "owner@example.test",
        },
    )
    assert [str(row.id) for row in owner.auth.list_credentials(STUB_PROVIDER)] == [
        str(borrowed.row_a.id)
    ], "the re-sign-in must reuse the row, which is what made the window span two logins"

    with pytest.raises(ProviderError) as after:
        await _turn(borrowed.store, stub, STUB_PROVIDER, "sess-401")
    text = str(after.value)

    # THE REPORT WAS MADE. ``coalesced`` would have probed nothing, so the POST count
    # is the discriminating number: on the defect the second turn adds zero.
    assert len(idp.posts) == posts_after_first + 1, idp.posts
    assert "bearer-a2" in stub.seen, stub.seen
    # What the person reads: the owner's own remedy, not this device's.
    assert "HTTP 401" not in text, text
    assert "on owner-laptop" in text and f"'lop login {STUB_PROVIDER}'" in text, text
    assert "To fix:" not in text, text

    # The two display paths that append the LOCAL auth hint, fed the turn's text.
    from rich.console import Console

    from local_operator.harness.types import AgentEndEvent
    from local_operator.headless_print import PrintRenderer
    from local_operator.providers.failover import append_auth_recovery

    assert append_auth_recovery(text, STUB_PROVIDER) == text
    sink = io.StringIO()
    PrintRenderer(Console(file=sink, width=400, highlight=False)).handle(AgentEndEvent(error=text))
    headless = sink.getvalue()
    assert text in headless and "To fix:" not in headless, headless

    from local_operator.tui.app import OperatorApp

    monkeypatch.delenv("RADIENT_API_KEY", raising=False)
    app = OperatorApp(lambda: None)  # type: ignore[arg-type]
    app._session = MagicMock(model_label=f"{STUB_PROVIDER}/claude-opus-5-5")  # noqa: SLF001
    assert app._with_recovery_hint(text) == text  # noqa: SLF001


@pytest.mark.asyncio
async def test_a_401_after_the_owner_signed_that_row_in_again_can_still_serve(
    borrowed: Any, idp: Any
) -> None:
    """#3616 review N2: the OTHER half of "reported promptly" — the probe works.

    The cell above asserts the owner's REFUSAL after a re-sign-in. This one asserts
    the outcome the report exists for: the operator signs in again, the next 401 on a
    borrowed bearer is reported (not coalesced), the owner's forced probe really
    rotates the token, the borrower re-borrows the fresh bearer and the turn SUCCEEDS
    with no error text at all. Coalesced, the borrower keeps the refused bearer, the
    rotation runs out and the turn fails — so this cell fails on the pre-fix keying
    for the opposite reason to the one above, which is what makes the pair cover both
    answers rather than one.
    """
    owner = borrowed.owner
    owner.auth.delete_credential(borrowed.row_b.id)  # A is the only login
    # A DEAD GRANT FIRST, so turn one really fails and the hold is the `grant_invalid`
    # kind — the only kind the owner's next lend clears (#3616 QA), which is what lets
    # the re-signed-in row be lent again inside the hold's window.
    owner.auth.upsert_credential(
        STUB_PROVIDER,
        {
            "type": "oauth",
            "access": "bearer-a",
            "expires": int(time.time() * 1000) + 3_600_000,
            "refresh": "refresh-0",
            "email": "owner@example.test",
        },
    )
    idp.invalid = True
    stub = _Refuses401("bearer-a", "bearer-a2")
    with pytest.raises(ProviderError):
        await _turn(borrowed.store, stub, STUB_PROVIDER, "sess-401")

    # THE OPERATOR SIGNS IN AGAIN, and the provider's grant is healthy afterwards:
    # that is the state a re-sign-in is supposed to produce, and the state the probe
    # must be able to turn into a served turn.
    idp.invalid = False
    owner.auth.upsert_credential(
        STUB_PROVIDER,
        {
            "type": "oauth",
            "access": "bearer-a2",
            "expires": int(time.time() * 1000) + 3_600_000,
            "refresh": idp.current_refresh,
            "email": "owner@example.test",
        },
    )
    posts_before = len(idp.posts)
    stub.seen.clear()

    events = await _turn(borrowed.store, stub, STUB_PROVIDER, "sess-401")

    assert _ended(events), f"the turn after the re-sign-in never served: {events} / {stub.seen}"
    assert len(idp.posts) == posts_before + 1, f"the owner's probe did not rotate: {idp.posts}"
    # The re-signed-in bearer first, then the one the probe minted: the borrower went
    # back to the owner for a new bearer instead of ending on the 401 it reported.
    assert stub.seen[0] == "bearer-a2", stub.seen
    assert stub.seen[-1] != "bearer-a2", stub.seen
