"""A borrowed login's rate limit, end to end through the failover driver.

The incident (2026-11-12, v0.102.2, ``lop-remote-node-2``): a remote borrower whose
only Anthropic route is a login lent by the operator's Mac asked for it while the
owner's account was resting. The owner refused ``quota_blocked`` and the turn ended:

    unknown: … 'anthropic' on damians-MacBook-Pro is rate-limited for another 35 min
    (…), so nothing was borrowed. The usual failover to another of your own logins
    is already running.

Three faults, each pinned here with the REAL borrower client, owner broker, in-process
relay and ``stream_with_failover``:

* the refusal reached the walk as ``kind="unknown"`` with no reset time, so it read
  as an unclassified failure rather than the rate limit it is — the display labelled
  it ``unknown`` and the walk benched nothing on the owner's stated reset;
* the sentence claimed a failover "already running" that nothing on the borrower
  starts — whatever runs is the turn's fallback chain, which the borrower owns;
* a configured chain must still be walked past the refusal to the next target, the
  same way a local account's quota advances it.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from local_operator.providers.failover import ProviderError, classify_provider_error
from tests.unit.network.test_credential_limit_skip import (  # noqa: F401 — fixtures by import
    _FIXTURES,
    OWNER_DEVICE,
    StubProvider,
    _ended,
    borrowed,
)
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    idp,
    owner,
    stub_provider,
)
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixture by import

_KEEP = (_FIXTURES, borrowed, owner, idp, stub_provider, devices)

#: The borrower's OWN key for the second hop: a provider it is signed in to locally,
#: the shape of the operator's ``openai`` hop on the remote node.
LOCAL_PROVIDER = "openai"
LOCAL_KEY = "local-openai-key"


def _rest_every_owner_row(borrowed: Any, *, seconds: int = 35 * 60) -> None:
    """Both of the owner's logins resting on this borrower: the incident's state."""
    until = time.time() + seconds
    for row, label in (
        (borrowed.row_a, "owner@example.test"),
        (borrowed.row_b, "second@example.test"),
    ):
        borrowed.client.state.cool_down(
            STUB_PROVIDER,
            owner_device=OWNER_DEVICE,
            credential_id=row.id,
            block_scope="",
            until=until,
            label=label,
        )


async def _turn(auth: Any, stub: StubProvider, settings: dict[str, Any]) -> list[Any]:
    import httpx

    from local_operator.harness.types import ChatRequest, ModelSpec
    from local_operator.providers.clients import OpenAICompatClient
    from local_operator.providers.failover import stream_with_failover

    request = ChatRequest(model=ModelSpec(provider=STUB_PROVIDER, model_id="claude-opus-5-5"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(stub)) as http:

        async def client_for(spec: ModelSpec) -> Any:
            return OpenAICompatClient("https://stub.invalid/v1", http_client=http)

        return [
            event
            async for event in stream_with_failover(
                request, auth, settings, client_for, session_id="sess-remote"
            )
        ]


def _settings(*hops: str) -> dict[str, Any]:
    chain = [f"{STUB_PROVIDER}/claude-opus-5-5", *hops] if hops else []
    return {
        "retry": {
            "baseDelayMs": 1,
            "maxRetries": 1,
            "fallbackChains": {"default": chain} if chain else {},
        }
    }


@pytest.mark.asyncio
async def test_a_borrowed_rate_limit_walks_on_to_the_next_chain_target(
    borrowed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner refuses ``quota_blocked``; the chain's local ``openai`` hop serves."""
    _rest_every_owner_row(borrowed)
    local = borrowed.store._local  # noqa: SLF001 — the borrower's own login table

    async def get_api_key(provider: str, *_a: Any, **_k: Any) -> str | None:
        return LOCAL_KEY if provider == LOCAL_PROVIDER else None

    monkeypatch.setattr(local, "get_api_key", get_api_key)
    stub = StubProvider("never-served")

    events = await _turn(borrowed.store, stub, _settings(f"{LOCAL_PROVIDER}/gpt-6.1-sol"))

    assert _ended(events), f"the turn stopped on the borrowed refusal: {events}"
    assert stub.seen == [LOCAL_KEY], f"the next target was never asked: {stub.seen}"


@pytest.mark.asyncio
async def test_an_exhausted_borrowed_rate_limit_is_a_quota_error_not_unknown(
    borrowed: Any,
) -> None:
    """No other route: the turn ends on an honest rate limit carrying the reset."""
    _rest_every_owner_row(borrowed)
    stub = StubProvider("never-served")

    with pytest.raises(ProviderError) as raised:
        await _turn(borrowed.store, stub, _settings())

    error = raised.value
    assert stub.seen == [], f"a bearer was sent despite the refusal: {stub.seen}"
    # The same class a local account's quota gets: the route pin, the screening
    # boundary and every display read ``quota``, never ``unknown``.
    assert error.kind == "quota", (error.kind, str(error))
    assert classify_provider_error(error) == "quota", str(error)
    # The owner's reset rides on the error, so the bench and any loop-back sweep
    # wait for it instead of re-asking a pool that is known to be resting.
    assert error.retry_after_ms and 30 * 60_000 < error.retry_after_ms <= 35 * 60_000, (
        error.retry_after_ms
    )
    text = str(error)
    # Rendered as the broker's sentence, with no "unknown"/kind label in front.
    assert text.startswith(f"'{STUB_PROVIDER}' on "), text
    assert "owner@example.test" in text and "second@example.test" in text, text
    # Never a claim that some other failover is running: it says what this turn
    # did, which with no chain configured on the borrower is stop.
    assert "already running" not in text, text
    assert "failover" not in text.lower(), text
    # The rig's borrower has no mesh identity, so the device reads "this device".
    assert text.endswith(
        "No fallback is configured on this device (retry.fallbackChains), "
        "so the turn stops here."
    ), text


@pytest.mark.asyncio
async def test_a_borrowed_rate_limit_ends_on_the_hops_it_tried(borrowed: Any) -> None:
    """A chain is configured and its hop cannot serve: the end state, not a promise.

    The hop cannot serve here (the borrower holds no ``openai`` login). The loop-back
    sweep revisits the primary (a quota refusal is recoverable), so the error the
    turn ends on says which hops were tried and that none served.
    """
    _rest_every_owner_row(borrowed)
    stub = StubProvider("never-served")

    with pytest.raises(ProviderError) as raised:
        await _turn(borrowed.store, stub, _settings(f"{LOCAL_PROVIDER}/gpt-6.1-sol"))

    text = str(raised.value)
    assert stub.seen == [], stub.seen
    assert text.endswith(
        f"No model in the fallback chain on this device served this turn "
        f"({LOCAL_PROVIDER}/gpt-6.1-sol)."
    ), text
    assert "Trying the next model" not in text, text
    assert raised.value.kind == "quota", raised.value.kind


class _ShortRateLimit:
    """The hop's own key answers ``429 retry-after: 1`` on every request."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request.headers.get("authorization", "").removeprefix("Bearer "))
        return httpx.Response(
            429,
            headers={"retry-after": "1"},
            json={"error": {"message": "slow down", "type": "rate_limit"}},
        )


@pytest.mark.asyncio
async def test_the_final_error_never_promises_a_hop_that_already_failed(
    borrowed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent review round 1, F1: the reviewer's repro.

    The hop has a local key that answers a short 429, so it is ``recoverable`` and the
    loop-back sweep revisits the primary with the hop still queued after it. The tail
    used to be written when the primary's refusal was RECORDED, so that second visit
    wrote "Trying the next model … (hop)"; the hop then failed again, and because the
    primary outranks every hop that sentence became the turn's last word.
    """
    _rest_every_owner_row(borrowed)
    local = borrowed.store._local  # noqa: SLF001 — the borrower's own login table

    async def get_api_key(provider: str, *_a: Any, **_k: Any) -> str | None:
        return LOCAL_KEY if provider == LOCAL_PROVIDER else None

    monkeypatch.setattr(local, "get_api_key", get_api_key)
    stub = _ShortRateLimit()

    with pytest.raises(ProviderError) as raised:
        await _turn(borrowed.store, stub, _settings(f"{LOCAL_PROVIDER}/gpt-6.1-sol"))

    text = str(raised.value)
    # The hop really was asked (more than once: the sweep revisits it).
    assert len(stub.seen) >= 2 and set(stub.seen) == {LOCAL_KEY}, stub.seen
    assert raised.value.kind == "quota", (raised.value.kind, text)
    assert "Trying the next model" not in text, text
    assert text.endswith(
        f"No model in the fallback chain on this device served this turn "
        f"({LOCAL_PROVIDER}/gpt-6.1-sol)."
    ), text
    # Exactly one tail, however many times the primary was visited.
    assert text.count("this turn") == 1, text


@pytest.mark.asyncio
async def test_a_switched_off_fallback_is_not_called_unconfigured(borrowed: Any) -> None:
    """Review round 1, F2: a chain exists but ``retry.modelFallback`` is off."""
    _rest_every_owner_row(borrowed)
    settings = _settings(f"{LOCAL_PROVIDER}/gpt-6.1-sol")
    settings["retry"]["modelFallback"] = False

    with pytest.raises(ProviderError) as raised:
        await _turn(borrowed.store, StubProvider("never-served"), settings)

    text = str(raised.value)
    assert "retry.fallbackChains" not in text, text
    assert text.endswith(
        "Fallback is switched off on this device (retry.modelFallback), "
        "so the turn stops here."
    ), text


def test_a_tail_never_claims_unconfigured_when_a_chain_exists() -> None:
    """Review round 1, F2: a chain whose hops a pin policy filtered out entirely."""
    from local_operator.providers.failover import (
        BORROWED_QUOTA_ATTR,
        _say_borrowed_quota_outcome,
    )

    error = ProviderError(None, "'x' on mac is rate-limited for another 5 min.", kind="quota")
    setattr(error, BORROWED_QUOTA_ATTR, True)
    _say_borrowed_quota_outcome(
        error, object(), hops_tried=(), fallback_off=None, chain_configured=True
    )
    assert error.message.endswith("No fallback model served this turn."), error.message
    assert "retry.fallbackChains" not in error.message, error.message


def test_the_pinned_cause_has_one_period() -> None:
    """Review round 1, F3: a cause ending in "." rendered "..", every arm."""
    from local_operator.providers.failover import _pinned_exhaustion_error

    reported = ProviderError(None, "'x' on mac is rate-limited for another 5 min.", kind="quota")
    for strict, hops, off in (
        (True, (), None),
        (False, (), "retry.enabled"),
        (False, (), None),
        (False, ("a/b",), None),
    ):
        text = str(
            _pinned_exhaustion_error(
                "x/m", reported, (), False, strict=strict, hops_tried=hops, fallback_off=off
            )
        )
        assert ".." not in text, text
        assert "for another 5 min." in text, text


@pytest.mark.asyncio
async def test_the_incident_card_labels_a_borrowed_quota_rate_limit(borrowed: Any) -> None:
    """QA round 1, Q1: the REAL turn-ending ``str(error)`` through the incident path.

    ``AgentEndEvent.error`` is ``str(exc)``; the session hands it to
    ``format_incident_message``, whose classifier is text-only. The borrowed sentence
    is verbatim ("… is rate-limited …", no "rate limit or quota exceeded:" label), so
    without the hyphenated marker the operator's card read ``unknown:``.
    """
    from local_operator.incidents import classify_incident, format_incident_message

    _rest_every_owner_row(borrowed)
    with pytest.raises(ProviderError) as raised:
        await _turn(borrowed.store, StubProvider("never-served"), _settings())

    raw = str(raised.value)
    assert classify_incident(raw, STUB_PROVIDER, "claude-opus-5-5").category == "rate-limit"
    card = format_incident_message(raw, STUB_PROVIDER, "claude-opus-5-5")
    assert "rate-limit:" in card and "unknown:" not in card, card


def test_the_quota_sentence_claims_no_failover_it_cannot_see() -> None:
    """Every ``quota_blocked`` arm: what happened and when, never an unseen failover."""
    from local_operator.network.credentials.messages import render_broker_error
    from local_operator.network.credentials.types import BrokerError

    for retry_after_ms, accounts in ((0, ()), (35 * 60_000, ()), (35 * 60_000, ("a@x.test",))):
        error = BrokerError(
            code="quota_blocked",
            key=STUB_PROVIDER,
            owner_device=OWNER_DEVICE,
            owner_device_name="mac",
            retry_after_ms=retry_after_ms,
        )
        text = render_broker_error(
            error, key=STUB_PROVIDER, provider=STUB_PROVIDER, owner_name="mac", accounts=accounts
        )
        assert "nothing was borrowed" in text and "free to lend" in text, text
        assert "already running" not in text and "failover" not in text.lower(), text
