"""The subscription-aware Anthropic prompt-cache TTL policy.

Every branch of ``providers/anthropic_cache_ttl.decide_cache_ttl`` and its two
producers: the wire client (credential kind, overage headers, body markers) and
``SessionStreamFn`` (role stamping). Measured rationale lives in the
policy module's docstring; these tests pin the decisions, not the numbers.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

import httpx
import pytest

from local_operator.harness.types import ChatRequest, Message, ModelSpec
from local_operator.providers import anthropic_cache_ttl as policy
from local_operator.providers.anthropic_cache_ttl import (
    cache_role,
    decide_cache_ttl,
    observe_unified_headers,
    overage_active,
)
from local_operator.providers.auth_store import OAuthAccess
from local_operator.providers.clients import AnthropicClient


@pytest.fixture(autouse=True)
def _clean_overage() -> Iterator[None]:
    policy.reset_overage_state()
    yield
    policy.reset_overage_state()


def _decide(**overrides: Any) -> str | None:
    args: dict[str, Any] = {
        "role": "main",
        "subscription": True,
        "overage": False,
        "main_min_context_tokens": 150_000,
        "subagent_min_context_tokens": 0,
        "context_size": lambda: 1_000,
    }
    args.update(overrides)
    return decide_cache_ttl(**args)


# -- roles -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("purpose", "delegated", "expected"),
    [
        ("turn", False, "main"),
        ("turn", True, "subagent"),
        ("compaction", False, "auxiliary"),
        ("naming", True, "auxiliary"),
        ("aside", False, "auxiliary"),
        ("clearance", False, "auxiliary"),
        ("supplement_render", False, "auxiliary"),
        ("isolated", False, "auxiliary"),
    ],
)
def test_cache_role(purpose: str, delegated: bool, expected: str) -> None:
    assert cache_role(purpose, delegated) == expected


# -- the decision table --------------------------------------------------------


def test_subscription_main_is_1h_at_any_size() -> None:
    """Claude Code parity: no size gate for the subscription main thread."""
    assert _decide(context_size=lambda: 10) == "1h"
    assert _decide(context_size=lambda: 900_000) == "1h"


def test_subscription_main_does_not_even_size_the_request() -> None:
    def explode() -> int:
        raise AssertionError("size must not be computed for a size-free decision")

    assert _decide(context_size=explode) == "1h"


def test_main_threshold_zero_is_still_the_global_off_switch() -> None:
    """Backwards compatibility: ``cache_ttl_1h_min_context_tokens: 0`` always
    meant "no 1h TTL"; an operator who set it must not start paying 2x writes."""
    assert _decide(main_min_context_tokens=0) is None


@pytest.mark.parametrize("role", ["subagent", "auxiliary"])
def test_subscription_non_main_defaults_to_5m(role: str) -> None:
    assert _decide(role=role, context_size=lambda: 900_000) is None


def test_subagent_threshold_opts_children_in_by_size() -> None:
    opted_in = {"role": "subagent", "subagent_min_context_tokens": 200_000}
    assert _decide(**opted_in, context_size=lambda: 199_999) is None
    assert _decide(**opted_in, context_size=lambda: 200_000) == "1h"


def test_auxiliary_ignores_every_threshold() -> None:
    assert (
        _decide(
            role="auxiliary",
            subscription=False,
            main_min_context_tokens=1,
            subagent_min_context_tokens=1,
            context_size=lambda: 900_000,
        )
        is None
    )


def test_api_key_main_keeps_the_size_threshold() -> None:
    assert _decide(subscription=False, context_size=lambda: 149_999) is None
    assert _decide(subscription=False, context_size=lambda: 150_000) == "1h"


def test_api_key_subagent_defaults_to_never() -> None:
    assert _decide(role="subagent", subscription=False, context_size=lambda: 900_000) is None


def test_overage_drops_the_subscription_main_to_the_api_key_policy() -> None:
    assert _decide(overage=True, context_size=lambda: 10_000) is None
    assert _decide(overage=True, context_size=lambda: 150_000) == "1h"


def test_size_gate_is_stateless_in_both_directions() -> None:
    """No hysteresis: a TTL change never re-writes the cached prefix (measured
    live in review round 1), so a shrink below the threshold simply returns
    the cheaper rate for new writes, and leaving overage restores 1h."""
    assert _decide(subscription=False, context_size=lambda: 160_000) == "1h"
    assert _decide(subscription=False, context_size=lambda: 140_000) is None
    assert _decide(overage=True, context_size=lambda: 100_000) is None
    assert _decide(context_size=lambda: 100_000) == "1h"


# -- overage detection ---------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "overage", "expected"),
    [
        ("allowed", None, False),
        ("allowed_warning", "allowed", False),
        ("rejected", "allowed", True),
        ("rejected", "allowed_warning", True),
        ("rejected", "rejected", False),
        ("rejected", None, False),
    ],
)
def test_overage_detection_matches_claude_code(
    status: str, overage: str | None, expected: bool
) -> None:
    """``isUsingOverage = status == "rejected" && overage-status in {allowed,
    allowed_warning}``, decoded from Claude Code v2.1.274."""
    headers = {policy.UNIFIED_STATUS_HEADER: status}
    if overage is not None:
        headers[policy.UNIFIED_OVERAGE_STATUS_HEADER] = overage
    observe_unified_headers("account:a", headers)
    assert overage_active("account:a") is expected


def test_overage_state_is_per_account_and_self_corrects() -> None:
    observe_unified_headers(
        "account:a",
        {policy.UNIFIED_STATUS_HEADER: "rejected", policy.UNIFIED_OVERAGE_STATUS_HEADER: "allowed"},
    )
    assert overage_active("account:a")
    assert not overage_active("account:b")
    # A response WITHOUT the header (proxy, API key) says nothing.
    observe_unified_headers("account:a", {})
    assert overage_active("account:a")
    # The window reset: the next subscription response re-states the truth.
    observe_unified_headers("account:a", {policy.UNIFIED_STATUS_HEADER: "allowed"})
    assert not overage_active("account:a")
    assert not overage_active("")


# -- the wire client -----------------------------------------------------------


def _request(**update: Any) -> ChatRequest:
    request = ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="claude-opus-5", context_window=1_000_000),
        system_blocks=["instructions", "inventory", "skills", "env"],
        messages=[Message.user("first"), Message.assistant("mid"), Message.user("second")],
        context_tokens_hint=2_000,
    )
    return request.model_copy(update=update)


def _markers(body: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "cache_control" in node:
                found.append(node["cache_control"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(body)
    return found


def _ttls(body: dict[str, Any]) -> set[str | None]:
    markers = _markers(body)
    assert markers, "the breakpoint policy itself must be untouched"
    return {m.get("ttl") for m in markers}


def test_client_subscription_main_small_context_is_1h_on_every_marker() -> None:
    client = AnthropicClient(cache_ttl_1h_min_context_tokens=150_000)
    body = client._build_body(_request(cache_role="main"), oauth=True)
    # ONE TTL per request: Anthropic rejects a 1h marker after a 5m one.
    assert _ttls(body) == {"1h"}


def test_client_role_defaults_from_purpose_when_unstamped() -> None:
    """A request built outside a session stream: a turn is main, an errand is
    auxiliary — the previous behaviour for scripts and one-shots."""
    client = AnthropicClient(cache_ttl_1h_min_context_tokens=150_000)
    assert _ttls(client._build_body(_request(), oauth=True)) == {"1h"}
    assert _ttls(client._build_body(_request(purpose="compaction"), oauth=True)) == {None}


def test_client_subscription_subagent_is_5m_body_byte_identical_to_api_default() -> None:
    client = AnthropicClient(cache_ttl_1h_min_context_tokens=150_000)
    body = client._build_body(
        _request(cache_role="subagent", context_tokens_hint=900_000), oauth=True
    )
    assert _ttls(body) == {None}
    assert "ttl" not in json.dumps(body)


def test_client_api_key_main_below_threshold_is_5m() -> None:
    client = AnthropicClient(cache_ttl_1h_min_context_tokens=150_000)
    assert _ttls(client._build_body(_request(cache_role="main"), oauth=False)) == {None}


def test_client_overage_account_reverts_main_to_threshold() -> None:
    client = AnthropicClient(cache_ttl_1h_min_context_tokens=150_000)
    observe_unified_headers(
        "account:acct-1",
        {policy.UNIFIED_STATUS_HEADER: "rejected", policy.UNIFIED_OVERAGE_STATUS_HEADER: "allowed"},
    )
    request = _request(cache_role="main")
    assert _ttls(client._build_body(request, oauth=True, account="account:acct-1")) == {None}
    # A different subscription is not in overage.
    assert _ttls(client._build_body(request, oauth=True, account="account:acct-2")) == {"1h"}


def test_client_subagent_threshold_reaches_the_wire() -> None:
    client = AnthropicClient(
        cache_ttl_1h_min_context_tokens=150_000, subagent_cache_ttl_1h_min_context_tokens=100_000
    )
    big = _request(cache_role="subagent", context_tokens_hint=120_000)
    assert _ttls(client._build_body(big, oauth=True)) == {"1h"}


def test_client_for_spec_threads_the_subagent_threshold() -> None:
    from local_operator.providers.clients import client_for_spec

    client = client_for_spec(
        ModelSpec(provider="anthropic", model_id="claude-opus-5"),
        anthropic_subagent_cache_ttl_1h_min_context_tokens=77_000,
    )
    assert isinstance(client, AnthropicClient)
    assert client._subagent_cache_ttl_1h_min_context_tokens == 77_000


def _oauth(account_id: str | None = "acct-9") -> OAuthAccess:
    return OAuthAccess(access_token="tok", credential_id=7, account_id=account_id, kind="oauth")


def _sse(*events: dict[str, Any]) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)


async def _drive(client: AnthropicClient, request: ChatRequest, oauth: OAuthAccess) -> None:
    async for _ in client.stream(request, None, oauth):
        pass


@pytest.mark.asyncio
async def test_stream_records_overage_from_response_headers_and_next_request_obeys() -> None:
    """The real wire path: a response carrying ``rejected`` + overage
    ``allowed`` flips the NEXT request's main-thread markers to 5m."""
    sent: list[dict[str, Any]] = []
    overage_headers = {
        "content-type": "text/event-stream",
        policy.UNIFIED_STATUS_HEADER: "rejected",
        policy.UNIFIED_OVERAGE_STATUS_HEADER: "allowed",
    }

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append(json.loads(req.content))
        return httpx.Response(
            200,
            headers=overage_headers,
            content=_sse(
                {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
                {"type": "message_stop"},
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AnthropicClient(http_client=http, cache_ttl_1h_min_context_tokens=150_000)
        request = _request(cache_role="main")
        await _drive(client, request, _oauth())
        await _drive(client, request, _oauth())
    assert _ttls(sent[0]) == {"1h"}
    assert overage_active("account:acct-9")
    assert _ttls(sent[1]) == {None}


@pytest.mark.asyncio
async def test_stream_records_overage_on_an_error_response_too() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={
                policy.UNIFIED_STATUS_HEADER: "rejected",
                policy.UNIFIED_OVERAGE_STATUS_HEADER: "allowed_warning",
            },
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "x"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AnthropicClient(http_client=http, cache_ttl_1h_min_context_tokens=150_000)
        with pytest.raises(Exception):
            await _drive(client, _request(cache_role="main"), _oauth(account_id=None))
    # No account id: keyed by credential row instead.
    assert overage_active("credential:7")


# -- SessionStreamFn stamping --------------------------------------------------


def _session_stream(session_id: str = "s-main") -> Any:
    from local_operator.model.configure import SessionStreamFn

    # No auth store: the stamping under test runs before any credential read.
    return SessionStreamFn(cast(Any, None), {}, session_id)


async def _stamped(stream: Any, request: ChatRequest) -> ChatRequest:
    """Run the stream's request preparation and capture what reaches failover."""
    captured: list[ChatRequest] = []

    async def fake_failover(req: ChatRequest, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        captured.append(req)
        if False:  # pragma: no cover - make this an async generator
            yield None

    import local_operator.providers.failover as failover

    original = failover.stream_with_failover
    failover.stream_with_failover = fake_failover  # type: ignore[assignment]
    try:
        async for _ in stream(request, None):
            pass
    except Exception:  # noqa: BLE001 - only the captured request matters
        pass
    finally:
        failover.stream_with_failover = original  # type: ignore[assignment]
    assert captured, "the request never reached the failover driver"
    return captured[0]


@pytest.mark.asyncio
async def test_session_stream_stamps_main_role() -> None:
    stream = _session_stream()
    try:
        stamped = await _stamped(stream, _request())
        assert stamped.cache_role == "main"
    finally:
        await stream.close()


@pytest.mark.asyncio
async def test_session_stream_fork_stamps_subagent() -> None:
    parent = _session_stream()
    child = parent.fork("s-child")
    try:
        stamped = await _stamped(child, _request())
        assert stamped.cache_role == "subagent"
    finally:
        await child.close()
        await parent.close()


@pytest.mark.asyncio
async def test_session_stream_errands_are_auxiliary() -> None:
    stream = _session_stream()
    try:
        for update in ({"purpose": "compaction"}, {"purpose": "naming", "isolated": True}):
            stamped = await _stamped(stream, _request(**update))
            assert stamped.cache_role == "auxiliary"
    finally:
        await stream.close()
