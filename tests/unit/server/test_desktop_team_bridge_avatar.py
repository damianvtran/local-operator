"""Avatar contract over actual loopback HTTP, sharing the attachment proxy rig."""

import pytest

from tests.unit.server.test_desktop_team_bridge_attachments import PNG
from tests.unit.server.test_desktop_team_bridge_attachments import (  # noqa: F401
    proxy as attachment_proxy,
)


@pytest.fixture
def proxy(request):
    return request.getfixturevalue("attachment_proxy")


URL = "/v1/desktop/team-bridge/accounts/acc_2/avatar"
QUERY = "?v=0123456789abcdef&size=64&expected_account_id=acc_1"


def test_avatar_loopback_success(proxy):
    client, calls, _ = proxy
    response = client.get(URL + QUERY)
    assert response.status_code == 200, response.text
    assert response.content == PNG
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert calls[0][1] == "/v1/accounts/acc_2/avatar?v=0123456789abcdef&size=64"


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?v=0123456789abcdef&size=64",
        QUERY + "&size=64",
        QUERY + "&url=https://evil.test",
        QUERY.replace("size=64", "size=65"),
        QUERY.replace("0123456789abcdef", "0123456789ABCDEf"),
        QUERY.replace("acc_1", "https://evil.test"),
    ],
)
def test_avatar_closed_query(proxy, query):
    client, calls, _ = proxy
    assert client.get(URL + query).status_code == 422
    assert not calls


def test_avatar_wrong_viewer(proxy):
    client, calls, _ = proxy
    assert client.get(URL + QUERY.replace("acc_1", "acc_other")).status_code == 409
    assert not calls


def test_avatar_unsigned(proxy):
    client, calls, state = proxy
    state["store"].rows = []
    assert client.get(URL + QUERY).status_code in (401, 409)
    assert not calls


def test_avatar_desktop_auth(proxy):
    client, calls, _ = proxy
    assert client.get(URL + QUERY, headers={"authorization": "Bearer wrong"}).status_code == 401
    assert not calls


@pytest.mark.parametrize(
    "mime,body",
    [
        ("image/svg+xml", b"<svg/>"),
        ("text/html", b"oops"),
        ("image/png", b""),
        ("image/png", b"x" * (1024 * 1024 + 1)),
    ],
)
def test_avatar_bad_bytes(proxy, mime, body):
    client, _, state = proxy
    state.update(serve_type=mime, serve=body)
    assert client.get(URL + QUERY).status_code == 502


def test_avatar_upstream_refusal(proxy):
    client, _, state = proxy
    state["error"] = (404, "avatar_not_found")
    assert client.get(URL + QUERY).status_code == 404
