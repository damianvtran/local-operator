"""A transient control-plane failure must not close the relay or blame the login.

Field report (2026-11-17, lop 0.106.6, iOS TestFlight 1157): Radient's edge WAF
(`rate-backend-edge`, 2000 requests / 5 min per source IP across
api.radienthq.com) answered the connector's 10-second authorization poll with
429. `service.authorization_failure_reason` had no arm for a 429, so it fell
through to `REFUSED`; the 30-second lease lapsed, and the phone was served
`503 authorization_refused` — "The Radient login may have expired, or this
tunnel's billing may be inactive" — for a login and a plan that were both fine.

What these tests pin:

* a 429 (and a 408) is `AUTHORIZATION_UNAVAILABLE`, never `REFUSED`;
* the API client reads the wait Radient asked for (`Retry-After`, or the WAF
  body's "retry in N seconds", which the WAF cannot put in a header);
* the gateway keeps serving inside a bounded grace measured from the last
  POSITIVE check, and a definitive refusal still closes it at once;
* the poller backs off by Radient's own hint instead of re-asking every 10 s.

Synthetic ids and an isolated config dir only.
"""

from __future__ import annotations

import time
from contextlib import closing

import httpx
import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.tunnels import gateway
from local_operator.tunnels.errors import RequestFailed
from local_operator.tunnels.service import (
    POLL_SECONDS,
    authorization_failure_reason,
    classify_failure,
)
from tests.unit.test_tunnels import HOST, make_connection, make_signing_key, proof

#: The body radient-infra's WAF rule returns (`pulumi/radient_infra/waf.py`
#: `RATE_LIMIT_BODY`, quoted in local-operator-mobile PR #297): `error` is an
#: OBJECT, and there is no `Retry-After` header because the WAF cannot set one.
WAF_BODY = {
    "error": {
        "type": "rate_limit_error",
        "code": "rate_limited",
        "message": "Too many requests from this network; retry in 300 seconds.",
    }
}


def _failed(status: int) -> RequestFailed:
    return RequestFailed(f"Radient tunnel request failed (HTTP {status}).", status=status)


@pytest.mark.parametrize("status", [408, 429])
def test_a_rate_limit_is_unavailable_not_a_refusal(status):
    failure = _failed(status)
    assert authorization_failure_reason(failure) == gateway.AUTHORIZATION_UNAVAILABLE
    verdict = classify_failure(failure)
    assert verdict.kind == "transient"
    assert verdict.reason == gateway.AUTHORIZATION_UNAVAILABLE
    # Chained the way a cold start re-raises it, it still classifies the same.
    outer = ValueError("cold start failed")
    outer.__cause__ = failure
    assert authorization_failure_reason(outer) == gateway.AUTHORIZATION_UNAVAILABLE


@pytest.mark.parametrize("status", [401, 403, 409])
def test_a_definitive_4xx_still_reads_as_a_refusal(status):
    assert authorization_failure_reason(_failed(status)) == gateway.REFUSED


def test_the_unavailable_reason_has_every_surface_entry_and_no_sign_in_remedy():
    reason = gateway.AUTHORIZATION_UNAVAILABLE
    assert gateway.reason_label(reason) not in {"", reason}
    relay = gateway.relay_detail(reason)
    terminal = gateway.terminal_detail(reason, "")
    assert relay and terminal and relay != terminal
    # Not the login's fault: no sign-in remedy, a waiting kind (no button), and
    # the phone copy does not send anyone to sign in or to billing.
    assert gateway.TERMINAL_REMEDY[reason] == "lop tunnel status"
    assert gateway.remedy_kind(reason) in gateway.WAITING_KINDS
    assert "sign in" not in relay.lower() and "billing" not in relay.lower()
    assert reason in gateway.TRANSIENT_REASONS
    assert gateway.REFUSED not in gateway.TRANSIENT_REASONS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,expected",
    [
        (httpx.Response(429, json=WAF_BODY), 300.0),
        (httpx.Response(429, json=WAF_BODY, headers={"Retry-After": "42"}), 42.0),
        (httpx.Response(503, headers={"Retry-After": "7"}), 7.0),
        (httpx.Response(429, text="<html>slow down</html>"), None),
        # A hint outside the sane window is ignored, never trusted blindly.
        (httpx.Response(429, headers={"Retry-After": "999999"}), None),
        (httpx.Response(401, json=WAF_BODY), None),
    ],
)
async def test_the_client_reads_the_wait_radient_asked_for(
    tmp_path, monkeypatch, response, expected
):
    from local_operator.tunnels.api import RadientTunnels

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    with closing(AuthStore()) as store:
        row = store.upsert_credential(
            "radient", {"type": "oauth", "account_id": "acct-1", "access": "synthetic"}
        )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(RequestFailed) as raised:
            await RadientTunnels(row.id, client).request("GET", "/tunnel-1")
    assert raised.value.status == response.status_code
    assert raised.value.retry_after == expected
    # The body is read for one number only; it never reaches the message.
    assert "network" not in str(raised.value)


async def _relayed(relay: gateway.Gateway, signing_key) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=relay.app()), base_url="https://" + HOST
    ) as client:
        return await client.get("/api/sessions", headers={gateway.PROOF_HEADER: proof(signing_key)})


def _lapsed(relay: gateway.Gateway, *, seconds_since_positive: float) -> None:
    """Age the last positive check without touching the process clock.

    Patching `time.monotonic` would also move the event loop's clock, so the
    gateway's own timestamps are moved back instead: the lease (30 s) has
    already run out, exactly as it had in the incident.
    """
    relay.last_authorized_at = time.monotonic() - seconds_since_positive
    relay.authorized_until = relay.last_authorized_at + gateway.AUTHORIZATION_LEASE_SECONDS


def _origin(_request: httpx.Request) -> httpx.Response:
    # A STREAM, as a real origin answers: the gateway relays the body by
    # iterating it, and a pre-read `json=` response cannot be streamed twice.
    return httpx.Response(200, stream=httpx.ByteStream(b'{"ok":true}'))


@pytest.fixture
def signing_key():
    return make_signing_key()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        gateway.AUTHORIZATION_UNAVAILABLE,
        gateway.SERVER_UNAVAILABLE,
        gateway.UNREACHABLE,
        gateway.CONTROL_PLANE_TIMEOUT,
    ],
)
async def test_a_transient_failure_keeps_serving_inside_the_grace(signing_key, reason):
    origin = httpx.MockTransport(_origin)
    async with httpx.AsyncClient(transport=origin) as upstream:
        relay = gateway.Gateway(make_connection(signing_key), upstream, mobile_password="pw")
        relay.authorize()
        _lapsed(relay, seconds_since_positive=45)
        relay.note_authorization_failure(reason, retry_after=300)
        reply = await _relayed(relay, signing_key)
    assert reply.status_code == 200, reply.text


@pytest.mark.asyncio
async def test_the_grace_is_bounded_and_then_says_unavailable_with_retry_after(signing_key):
    origin = httpx.MockTransport(_origin)
    async with httpx.AsyncClient(transport=origin) as upstream:
        relay = gateway.Gateway(make_connection(signing_key), upstream, mobile_password="pw")
        relay.authorize()
        _lapsed(relay, seconds_since_positive=gateway.AUTHORIZATION_GRACE_SECONDS + 1)
        relay.note_authorization_failure(gateway.AUTHORIZATION_UNAVAILABLE, retry_after=120)
        reply = await _relayed(relay, signing_key)
    assert reply.status_code == 503
    assert reply.json()["reason"] == gateway.AUTHORIZATION_UNAVAILABLE
    assert reply.json()["kind"] == "wait_radient"
    # When the next check is due: honest, never longer than the hint.
    assert 1 <= int(reply.headers["Retry-After"]) <= 120


@pytest.mark.asyncio
async def test_a_definitive_refusal_closes_at_once_even_inside_the_grace(signing_key):
    origin = httpx.MockTransport(_origin)
    async with httpx.AsyncClient(transport=origin) as upstream:
        relay = gateway.Gateway(make_connection(signing_key), upstream, mobile_password="pw")
        relay.authorize()
        _lapsed(relay, seconds_since_positive=45)
        relay.note_authorization_failure(gateway.AUTHORIZATION_UNAVAILABLE, retry_after=300)
        assert (await _relayed(relay, signing_key)).status_code == 200
        relay.note_authorization_failure(gateway.REFUSED)
        reply = await _relayed(relay, signing_key)
    assert reply.status_code == 503
    assert reply.json()["reason"] == gateway.REFUSED
    assert "Retry-After" not in reply.headers


@pytest.mark.asyncio
async def test_a_transient_failure_never_extends_a_gateway_that_was_never_authorized(
    signing_key,
):
    # The grace is earned by a POSITIVE check. A cold gateway whose first poll
    # fails keeps only its boot lease; it never borrows ten minutes.
    async with httpx.AsyncClient() as upstream:
        relay = gateway.Gateway(make_connection(signing_key), upstream)
        boot = relay.authorized_until
        relay.note_authorization_failure(gateway.AUTHORIZATION_UNAVAILABLE, retry_after=300)
        assert relay.authorized_until == boot


def test_the_poller_backs_off_by_radients_hint_and_never_past_the_grace():
    from local_operator.tunnels.service import transient_poll_delay

    grace = gateway.AUTHORIZATION_GRACE_SECONDS
    # The WAF's own hint wins over the 10 s cadence.
    assert transient_poll_delay(1, 300, grace_left=grace) == 300
    # No hint: exponential from the cadence, capped.
    assert [transient_poll_delay(n, None, grace_left=grace) for n in (1, 2, 3)] == [
        POLL_SECONDS,
        2 * POLL_SECONDS,
        4 * POLL_SECONDS,
    ]
    assert transient_poll_delay(30, None, grace_left=grace) <= 300
    # A hint longer than the grace still leaves a check before the grace ends …
    assert transient_poll_delay(1, 300, grace_left=200) == 200 - POLL_SECONDS
    # … and never polls faster than the cadence.
    assert transient_poll_delay(1, 300, grace_left=5) == POLL_SECONDS
    assert transient_poll_delay(1, 1, grace_left=grace) == POLL_SECONDS
