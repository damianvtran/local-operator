"""Rate-limit refusals from the hub AND the edge in front of it (incident 2026-11-17).

The AWS WAF rule ``rate-backend-direct`` blocked one shared address for 300 s.
Its 429 body is ``{"error": {"type", "code", "message"}}`` with no Retry-After
header, which the flat ``code``/``error`` reader turned into ``bridge_error`` /
"Radient refused the request (HTTP 429)." with no wait — so agents and the
desktop retried at once and the status discipline reported a bare
``unavailable``. These tests pin: the nested shape (and the augmented shape the
edge is about to gain), the stated wait, the tool's footer, the status
discipline keeping the last READY answer through a throttle, and the directory
cache that removes one read per named send.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import (
    STATE_READY,
    STATE_UNAVAILABLE,
    Boundary,
    BridgeRefusal,
    StatusDiscipline,
)
from local_operator.team_bridge.tool import _failure_text
from tests.unit.team_bridge.test_client import FakeRow, FakeStore, _make_client

#: The edge's exact 429 body, as captured from the aws-waf-logs-radient-prod-waf
#: block response on 2026-11-17.
WAF_300 = {
    "error": {
        "type": "rate_limit_error",
        "code": "rate_limited",
        "message": "Too many requests from this address; retry in 300 seconds.",
    }
}
WAF_60 = {
    "error": {
        "type": "rate_limit_error",
        "code": "rate_limited",
        "message": "Too many requests from this address; retry in 60 seconds.",
    }
}


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.fixture(autouse=True)
def _empty_directory_cache() -> Iterator[None]:
    tb_client._DIRECTORY_CACHE.clear()
    yield
    tb_client._DIRECTORY_CACHE.clear()


async def _refusal_for(response: httpx.Response) -> BridgeRefusal:
    client = _make_client(lambda request: response)
    with pytest.raises(BridgeRefusal) as info:
        await client.status()
    return info.value


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("body,wait", [(WAF_300, 300), (WAF_60, 60)])
async def test_the_edge_429_is_a_coded_throttle_with_its_stated_wait(body, wait):
    refusal = await _refusal_for(httpx.Response(429, json=body))
    assert refusal.code == "rate_limited"
    assert refusal.http_status == 429
    assert refusal.message == body["error"]["message"]
    assert refusal.retry_after_seconds == wait


@pytest.mark.asyncio
async def test_the_augmented_edge_shape_reads_the_same():
    """The WAF body is about to gain a top-level ``code``, ``details`` and a
    Retry-After header; every field agrees, so the reading must too."""

    body = {
        **WAF_300,
        "code": "rate_limited",
        "details": {"retry_after_seconds": 300},
    }
    refusal = await _refusal_for(httpx.Response(429, json=body, headers={"Retry-After": "300"}))
    assert refusal.code == "rate_limited"
    assert refusal.message == WAF_300["error"]["message"]
    assert refusal.retry_after_seconds == 300
    assert refusal.details == {"retry_after_seconds": 300}


@pytest.mark.asyncio
async def test_a_header_outranks_the_sentence():
    refusal = await _refusal_for(httpx.Response(429, json=WAF_300, headers={"Retry-After": "12"}))
    assert refusal.retry_after_seconds == 12


@pytest.mark.asyncio
async def test_the_hub_flat_shape_is_read_exactly_as_before():
    body = {"error": "Too many requests.", "code": "rate_limited", "details": {}}
    refusal = await _refusal_for(httpx.Response(429, json=body, headers={"Retry-After": "7"}))
    assert (refusal.code, refusal.message, refusal.retry_after_seconds) == (
        "rate_limited",
        "Too many requests.",
        7,
    )


@pytest.mark.asyncio
async def test_a_number_in_a_non_throttle_sentence_is_never_a_wait():
    body = {"error": {"code": "invalid_request", "message": "bad; retry in 5 seconds"}}
    refusal = await _refusal_for(httpx.Response(400, json=body))
    assert refusal.code == "invalid_request"
    assert refusal.retry_after_seconds is None


@pytest.mark.asyncio
async def test_a_nested_code_outside_the_grammar_is_still_bridge_error():
    body = {"error": {"code": "Rate Limited\nX", "message": "slow down"}}
    refusal = await _refusal_for(httpx.Response(429, json=body))
    assert refusal.code == "bridge_error"
    assert refusal.message == "slow down"


@pytest.mark.asyncio
async def test_the_agent_tool_names_the_throttle_and_its_wait():
    refusal = await _refusal_for(httpx.Response(429, json=WAF_300))
    text = _failure_text(refusal)
    assert text == (
        "Too many requests from this address; retry in 300 seconds. "
        "[refusal code=rate_limited retry_after_s=300]"
    )


# ---------------------------------------------------------------------------
# the status discipline
# ---------------------------------------------------------------------------


def _ready_payload() -> dict[str, Any]:
    return {
        "api_version": "1",
        "available": True,
        "tenants": [
            {
                "tenant_id": "tnt_1",
                "tenant_name": "Org",
                "eligible": True,
                "endpoint": None,
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            }
        ],
    }


class _Script:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses[min(len(self.requests), len(self.responses)) - 1]


def _discipline(script: _Script, root: Path) -> StatusDiscipline:
    return StatusDiscipline(
        root,
        base_url="https://hub.example/v1",
        transport=httpx.MockTransport(script),
        store=FakeStore(rows=[FakeRow(1, data={"access": "token-1", "account_id": "acc_1"})]),
        read_only=True,
    )


@pytest.mark.asyncio
async def test_a_throttle_keeps_the_last_ready_answer_and_names_itself(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    script = _Script(
        httpx.Response(200, json={"msg": "ok", "result": _ready_payload()}),
        httpx.Response(429, json=WAF_300),
    )
    discipline = _discipline(script, tmp_path)
    first = await discipline.check(Boundary.APP_START)
    assert first.state == STATE_READY

    # Past the success TTL, the next read is throttled by the edge.
    monkeypatch.setattr(tb_client, "STATUS_CACHE_TTL_S", 0.0)
    throttled = await discipline.check(Boundary.SURFACE_OPEN)
    assert len(script.requests) == 2
    assert throttled.state == STATE_READY
    assert throttled.payload == first.payload
    assert throttled.error is not None and throttled.error.code == "rate_limited"
    assert throttled.error.retry_after_seconds == 300


@pytest.mark.asyncio
async def test_a_throttle_is_held_for_its_named_wait_not_the_error_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    script = _Script(httpx.Response(429, json=WAF_300))
    discipline = _discipline(script, tmp_path)
    first = await discipline.check(Boundary.APP_START)
    # No READY answer yet: the state stays ``unavailable``, coded as a throttle.
    assert first.state == STATE_UNAVAILABLE
    assert first.error is not None and first.error.code == "rate_limited"

    # The plain error floor elapsing does NOT re-read: the edge named 300 s.
    monkeypatch.setattr(tb_client, "STATUS_ERROR_TTL_S", 0.0)
    again = await discipline.check(Boundary.SURFACE_OPEN)
    assert again.from_cache is True
    assert len(script.requests) == 1


# ---------------------------------------------------------------------------
# the directory cache
# ---------------------------------------------------------------------------

DIRECTORY = {"members": [{"account_id": "acc_2", "display_label": "Dana"}]}


def _directory_handler(seen: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"msg": "ok", "result": DIRECTORY})

    return handler


@pytest.mark.asyncio
async def test_a_burst_of_named_sends_reads_the_directory_once():
    seen: list[httpx.Request] = []
    client = _make_client(_directory_handler(seen))
    for _ in range(5):
        assert await client.directory_cached("tnt_1") == DIRECTORY
    assert len(seen) == 1
    # A fresh client (the tool builds one per call) shares the same entry.
    assert await _make_client(_directory_handler(seen)).directory_cached("tnt_1") == DIRECTORY
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_refresh_re_reads_and_re_primes():
    seen: list[httpx.Request] = []
    client = _make_client(_directory_handler(seen))
    await client.directory_cached("tnt_1")
    await client.directory_cached("tnt_1", refresh=True)
    await client.directory_cached("tnt_1")
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_the_cache_expires(monkeypatch: pytest.MonkeyPatch):
    seen: list[httpx.Request] = []
    client = _make_client(_directory_handler(seen))
    monkeypatch.setattr(tb_client, "DIRECTORY_CACHE_TTL_S", 0.0)
    await client.directory_cached("tnt_1")
    await client.directory_cached("tnt_1")
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_the_cache_never_crosses_accounts_or_tenants():
    seen: list[httpx.Request] = []
    first = _make_client(_directory_handler(seen))
    other_account = _make_client(
        _directory_handler(seen),
        store=FakeStore(rows=[FakeRow(1, data={"access": "token-2", "account_id": "acc_9"})]),
    )
    await first.directory_cached("tnt_1")
    await other_account.directory_cached("tnt_1")
    await first.directory_cached("tnt_2")
    assert len(seen) == 3
    assert {r.headers["authorization"] for r in seen} == {"Bearer token-1", "Bearer token-2"}


@pytest.mark.asyncio
async def test_a_mismatched_expected_account_bypasses_the_cache():
    """The captured-account fence still refuses: a cached answer read as acc_1
    is never served to a caller that expected another account."""

    seen: list[httpx.Request] = []
    client = _make_client(_directory_handler(seen))
    await client.directory_cached("tnt_1", expected_account_id="acc_1")
    with pytest.raises(BridgeRefusal) as info:
        await client.directory_cached("tnt_1", expected_account_id="acc_other")
    assert info.value.code == "account_mismatch"
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_callers_cannot_mutate_the_cached_answer():
    client = _make_client(_directory_handler([]))
    got = await client.directory_cached("tnt_1")
    got["members"].clear()
    assert (await client.directory_cached("tnt_1"))["members"]
