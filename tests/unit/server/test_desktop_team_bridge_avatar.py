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


def test_http_revalidation_survives_new_client(proxy):
    client, _, state = proxy
    state["etag"] = '"avatar-one"'
    first = client.get(URL + QUERY)
    second = client.get(URL + QUERY)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content == PNG
    assert state["statuses"] == [200, 304]
    assert state["conditionals"] == [None, '"avatar-one"']
    assert second.headers["cache-control"] == "private, no-store"


def test_unsolicited_304_refused(proxy):
    client, _, state = proxy
    state["force304"] = True
    assert client.get(URL + QUERY).status_code == 502


def test_revalidation_failure_evicts_body_and_validator(proxy):
    client, _, state = proxy
    state["etag"] = '"avatar-one"'
    assert client.get(URL + QUERY).status_code == 200
    state["error"] = (403, "avatar_forbidden")
    assert client.get(URL + QUERY).status_code == 403
    state["error"] = None
    assert client.get(URL + QUERY).status_code == 200
    assert state["conditionals"] == [None, None]


def test_account_switch_and_relogin_never_reuse_validator(proxy):
    client, _, state = proxy
    state["etag"] = '"avatar-one"'
    assert client.get(URL + QUERY).status_code == 200
    state["store"].rows[0].data["account_id"] = "acc_b"
    state["incarnation"] = "login-b"
    assert client.get(URL + QUERY.replace("acc_1", "acc_b")).status_code == 200
    state["store"].rows[0].data["account_id"] = "acc_1"
    state["incarnation"] = "login-a-again"
    assert client.get(URL + QUERY).status_code == 200
    assert state["conditionals"] == [None, None, None]


def test_signout_purges_observed_scope(proxy):
    client, _, state = proxy
    state["etag"] = '"avatar-one"'
    assert client.get(URL + QUERY).status_code == 200
    rows = state["store"].rows
    state["store"].rows = []
    assert client.get(URL + QUERY).status_code in (401, 409)
    # Even a fixture restoring the exact old incarnation cannot resurrect a
    # body after this process observed signout.
    state["store"].rows = rows
    assert client.get(URL + QUERY).status_code == 200
    assert state["conditionals"] == [None, None]


def test_memory_eviction_forces_cold_http_fetch(proxy, monkeypatch):
    from local_operator.team_bridge import avatar_cache

    monkeypatch.setattr(avatar_cache, "AVATAR_CACHE", avatar_cache.AvatarCache(max_entries=1))
    client, _, state = proxy
    state["etag"] = '"avatar-one"'
    assert client.get(URL + QUERY).status_code == 200
    assert client.get(URL + QUERY.replace("size=64", "size=256")).status_code == 200
    assert client.get(URL + QUERY).status_code == 200
    assert state["conditionals"] == [None, None, None]


@pytest.mark.parametrize("change", ["logout", "aba", "etag"])
def test_304_cannot_bypass_post_read_fence(proxy, change):
    client, _, state = proxy
    state["etag"] = '"avatar-one"'
    assert client.get(URL + QUERY).status_code == 200

    def mutate():
        if change == "logout":
            state["store"].rows = []
        elif change == "aba":
            state["incarnation"] = "login-a-again"
        else:
            state["etag"] = '"different"'

    state["on_avatar"] = mutate
    state["force304"] = True
    response = client.get(URL + QUERY)
    assert response.status_code == (502 if change == "etag" else 409)
    assert response.content != PNG
    assert state["conditionals"] == [None, '"avatar-one"']
