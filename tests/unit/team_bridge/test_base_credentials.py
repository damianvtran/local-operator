"""A Team Bridge hub base carrying ``user:password@`` is refused before any request.

WHY (security round 2 on #3587, S-r2-1). A non-canonical hub is reachable
through ``RADIENT_ORG_ALLOW_NONCANONICAL_BASE``, and such a base may be spelled
with URL credentials. Two things went wrong with that spelling:

* httpx logs ``HTTP Request: POST <full url>`` at INFO, credentials included,
  and the local server did not pin httpx below INFO, so ``LOG_LEVEL=INFO``
  printed the password.
* httpx turns URL credentials into ``Authorization: Basic``, which REPLACES the
  pinned ``Bearer`` this client sets (measured on httpx 0.28.1), so such a base
  never authenticated as the signed-in account anyway — it sent the URL's
  password to the hub instead.

THE RIG is the real ``TeamBridgeClient`` over ``httpx.MockTransport`` with
httpx itself at INFO, so the request-line logger is live. The marker is built
by concatenation and is deliberately not password-shaped: a log redactor that
masked it would let the absence assertions pass without testing anything, so
every case also asserts that no ``@`` survives.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from tests.unit.team_bridge.test_client import FakeStore, _ok

_MARKER = "zz" + "secret" + "zz"


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    # The opt-in a credentialed base needs to be reachable at all: with the
    # destination guard closed the base would be refused for being
    # non-canonical, and the test would not reach the path that leaked.
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.mark.parametrize(
    ("base_url", "bare"),
    [
        (f"https://ops:{_MARKER}@hub.example/v1", "https://hub.example/v1"),
        (f"https://{_MARKER}@hub.example/v1", "https://hub.example/v1"),
        (f"https://ops:a@{_MARKER}@Hub.Example:8443/v1", "https://Hub.Example:8443/v1"),
        (f"http://ops:{_MARKER}@[2001:db8::1]:9000/v1", "http://[2001:db8::1]:9000/v1"),
        # An EMPTY userinfo still makes httpx send ``Authorization: Basic``.
        ("https://@hub.example/v1", "https://hub.example/v1"),
    ],
)
@pytest.mark.asyncio
async def test_a_credentialed_base_is_refused_and_never_logged(
    caplog: pytest.LogCaptureFixture, base_url: str, bare: str
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"features": {}})

    # Every logger, httpx included, at INFO or below: the level at which the
    # request line carried the credential.
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.INFO, logger="httpx")
    client = tb_client.TeamBridgeClient(
        None, base_url=base_url, store=FakeStore(), transport=httpx.MockTransport(handler)
    )

    with pytest.raises(tb_client.BridgeRefusal) as refused:
        await client.status()

    assert refused.value.code == "bridge_destination_refused"
    assert seen == []
    assert client.base_url == bare
    surfaced = [str(refused.value), client.base_url]
    surfaced += [record.getMessage() for record in caplog.records]
    for text in surfaced:
        assert _MARKER not in text
        # The bare base never carries ``@``; neither may any record or the refusal.
        assert "@" not in text


@pytest.mark.asyncio
async def test_a_base_without_credentials_still_reaches_the_hub() -> None:
    """The negative control: the refusal keys on userinfo, nothing broader."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"features": {}})

    client = tb_client.TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(handler),
    )
    await client.status()
    assert [str(request.url) for request in seen] == ["https://hub.example/v1/messaging/status"]
    assert seen[0].headers["Authorization"].startswith("Bearer ")
