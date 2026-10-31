"""The agent (endpoint) picture relay over loopback HTTP.

The account relay's twin (``test_desktop_team_bridge_avatar.py``): same closed
query, same login fence, same byte bound and headers. What is specific to this
route is pinned here: the upstream path names the ENDPOINT and never an
account, the cache key cannot collide with an account's or another endpoint's,
and a version the hub does not serve under this endpoint is never answered
from bytes fetched for another.
"""

import pytest

from tests.unit.server.test_desktop_team_bridge_attachments import PNG
from tests.unit.server.test_desktop_team_bridge_attachments import (  # noqa: F401
    proxy as attachment_proxy,
)

EP = "bep_0123456789abcdef0123456789abcdef"
OTHER = "bep_fedcba9876543210fedcba9876543210"
URL = f"/v1/desktop/team-bridge/endpoints/{EP}/avatar"
QUERY = "?v=0123456789abcdef&size=64&expected_account_id=acc_1"


@pytest.fixture
def proxy(request, monkeypatch):
    from local_operator.team_bridge import avatar_cache

    monkeypatch.setattr(avatar_cache, "AVATAR_CACHE", avatar_cache.AvatarCache())
    return request.getfixturevalue("attachment_proxy")


def test_endpoint_avatar_reads_only_the_endpoint_route(proxy):
    client, calls, _ = proxy
    response = client.get(URL + QUERY)
    assert response.status_code == 200, response.text
    assert response.content == PNG
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-security-policy"] == "default-src 'none'; sandbox"
    assert [c[1] for c in calls] == [
        f"/v1/messaging/endpoints/{EP}/avatar?v=0123456789abcdef&size=64"
    ]
    # The viewer's own hub bearer, never the desktop token.
    assert calls[0][3] and calls[0][3] != "Bearer desktop-token"


@pytest.mark.parametrize(
    "path,query",
    [
        (URL, ""),
        (URL, "?v=0123456789abcdef&size=64"),  # no viewer receipt
        (URL, "?size=64&expected_account_id=acc_1"),  # missing v
        (URL, QUERY + "&size=64"),  # repeated name
        (URL, QUERY + "&url=https://evil.test"),
        (URL, QUERY + "&account_id=acc_2"),
        (URL, QUERY.replace("size=64", "size=65")),
        (URL, QUERY.replace("size=64", "size=128")),
        (URL, QUERY.replace("0123456789abcdef", "0123456789ABCDEF")),  # uppercase v
        (URL, QUERY.replace("0123456789abcdef", "0123456789abcde")),  # short v
        (URL, QUERY.replace("acc_1", "https://evil.test")),
        (URL.replace(EP, "bep.0123"), QUERY),  # path metacharacter
        (URL.replace(EP, "bep%3Fv%3D1"), QUERY),  # encoded query injection
        (URL.replace(EP, "b" * 129), QUERY),  # over the hub's id bound
    ],
)
def test_endpoint_avatar_closed_request(proxy, path, query):
    client, calls, _ = proxy
    assert client.get(path + query).status_code in (404, 422)
    assert not calls


def test_endpoint_grammar_matches_the_client():
    from local_operator.server.routes import desktop_team_bridge as route
    from local_operator.team_bridge import client

    assert route._ENDPOINT_ID.pattern == client._ENDPOINT_ID.pattern


def test_endpoint_avatar_wrong_viewer_unsigned_and_desktop_auth(proxy):
    client, calls, state = proxy
    assert client.get(URL + QUERY.replace("acc_1", "acc_other")).status_code == 409
    assert client.get(URL + QUERY, headers={"authorization": "Bearer wrong"}).status_code == 401
    state["store"].rows = []
    assert client.get(URL + QUERY).status_code in (401, 409)
    assert not calls


@pytest.mark.parametrize(
    "mime,body",
    [
        ("image/svg+xml", b"<svg/>"),
        ("text/html", b"oops"),
        ("image/webp", b"RIFF\0\0\0\0WEBP"),  # client admits it; this relay does not
        ("image/png", b""),
        ("image/png", b"x" * (1024 * 1024 + 1)),
    ],
)
def test_endpoint_avatar_bad_bytes(proxy, mime, body):
    client, _, state = proxy
    state.update(serve_type=mime, serve=body)
    assert client.get(URL + QUERY).status_code == 502


def test_stale_version_is_the_hubs_404(proxy):
    client, _, state = proxy
    state["error"] = (404, "not_found")
    response = client.get(URL + QUERY)
    assert response.status_code == 404
    assert response.content != PNG


def test_another_endpoints_version_is_never_served_under_this_id(proxy):
    """Endpoint OTHER holds version v; EP does not. A read of EP at v must
    reach the hub under EP's own path and get its 404 — never OTHER's bytes
    from the cache, and never OTHER's validator as a conditional."""

    client, calls, state = proxy
    state["etag"] = '"other-v"'
    other = f"/v1/desktop/team-bridge/endpoints/{OTHER}/avatar"
    assert client.get(other + QUERY).status_code == 200
    state["refuse_path"] = lambda path: (404, "not_found") if EP in path else None
    response = client.get(URL + QUERY)
    assert response.status_code == 404
    assert response.content != PNG
    assert [c[1].split("?")[0] for c in calls] == [
        f"/v1/messaging/endpoints/{OTHER}/avatar",
        f"/v1/messaging/endpoints/{EP}/avatar",
    ]
    assert state["conditionals"] == [None, None]


def test_endpoint_and_account_cache_keys_never_collide(proxy):
    """An account whose id equals an endpoint id, at the same version and size,
    shares no cache entry: neither read revalidates with the other's ETag."""

    from local_operator.team_bridge import avatar_cache

    client, calls, state = proxy
    state["etag"] = '"shared"'
    twin = "acc_twin"
    assert client.get(f"/v1/desktop/team-bridge/accounts/{twin}/avatar" + QUERY).status_code == 200
    assert client.get(f"/v1/desktop/team-bridge/endpoints/{twin}/avatar" + QUERY).status_code == 200
    assert state["conditionals"] == [None, None]
    entities = sorted(key[2] for key in avatar_cache.AVATAR_CACHE._entries)
    assert entities == ["acc_twin", "endpoint:acc_twin"]
    # Each now revalidates against ITS OWN entry only.
    assert client.get(f"/v1/desktop/team-bridge/endpoints/{twin}/avatar" + QUERY).status_code == 200
    assert state["conditionals"][-1] == '"shared"'
    assert [c[1].split("?")[0] for c in calls][-1] == f"/v1/messaging/endpoints/{twin}/avatar"


def test_endpoint_revalidation_and_signout_purge(proxy):
    client, _, state = proxy
    state["etag"] = '"one"'
    assert client.get(URL + QUERY).status_code == 200
    assert client.get(URL + QUERY).status_code == 200
    assert state["statuses"] == [200, 304]
    rows = state["store"].rows
    state["store"].rows = []
    assert client.get(URL + QUERY).status_code in (401, 409)
    state["store"].rows = rows
    assert client.get(URL + QUERY).status_code == 200
    assert state["conditionals"] == [None, '"one"', None]


def test_capability_is_advertised():
    from local_operator.server.features import static_feature_flags

    assert static_feature_flags()["team_bridge_endpoint_avatar"] == 1
