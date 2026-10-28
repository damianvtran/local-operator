"""Notification-avatar route over actual loopback HTTP, sharing the attachment rig.

Proves the closed query, the owner-avatar equality rule, the viewer fence, that
the only upstream read is the closed ``/accounts/{id}/avatar`` relay, that a
placeholder touches no network, and that composites are keyed under (and purged
with) the current login.
"""

import io

import pytest
from PIL import Image

from tests.unit.server.test_desktop_team_bridge_attachments import (  # noqa: F401
    proxy as attachment_proxy,
)


@pytest.fixture
def proxy(request, monkeypatch):
    from local_operator.team_bridge import avatar_cache

    monkeypatch.setattr(avatar_cache, "AVATAR_CACHE", avatar_cache.AvatarCache())
    monkeypatch.setattr(avatar_cache, "COMPOSITE_CACHE", avatar_cache.AvatarCache())
    return request.getfixturevalue("attachment_proxy")


def _jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 40, 40)).save(out, format="JPEG")
    return out.getvalue()


URL = "/v1/desktop/team-bridge/accounts/acc_2/notification-avatar"
BASE_Q = "tenant_id=tenant_a&kind=person&theme=light&size=64&initials=BT&expected_account_id=acc_1"
PHOTO_Q = BASE_Q + "&v=0123456789abcdef&avatar_account_id=acc_2"


def _png(response):
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    image = Image.open(io.BytesIO(response.content))
    assert image.size == (64, 64)
    return image


def test_photo_composite_reads_only_the_closed_avatar_route(proxy):
    client, calls, state = proxy
    state.update(serve_type="image/jpeg", serve=_jpeg())
    image = _png(client.get(f"{URL}?{PHOTO_Q}"))
    assert image.getpixel((20, 20))[:3] == pytest.approx((200, 40, 40), abs=16)
    assert [c[1] for c in calls] == ["/v1/accounts/acc_2/avatar?v=0123456789abcdef&size=64"]


def test_placeholder_touches_no_network(proxy):
    client, calls, _ = proxy
    _png(client.get(f"{URL}?{BASE_Q}"))
    _png(client.get(f"{URL}?{BASE_Q.replace('person', 'person_agent').replace('BT', '')}"))
    assert calls == []


@pytest.mark.parametrize(
    "query",
    [
        "",
        BASE_Q.replace("&expected_account_id=acc_1", ""),
        BASE_Q + "&v=0123456789abcdef",  # photo half without its account
        BASE_Q + "&avatar_account_id=acc_2",
        BASE_Q + "&kind=person",
        BASE_Q + "&url=https://evil.test/a.png",
        BASE_Q.replace("kind=person", "kind=local"),
        BASE_Q.replace("kind=person", "kind=service_agent"),
        BASE_Q.replace("theme=light", "theme=sepia"),
        BASE_Q.replace("size=64", "size=128"),
        BASE_Q.replace("initials=BT", "initials=BTX"),
        BASE_Q.replace("initials=BT", "initials=%E2%80%AEB"),
        PHOTO_Q.replace("0123456789abcdef", "0123456789ABCDEF"),
        PHOTO_Q.replace("acc_1", "https://evil.test"),
    ],
)
def test_closed_query(proxy, query):
    client, calls, _ = proxy
    assert client.get(f"{URL}?{query}").status_code == 422
    assert not calls


@pytest.mark.parametrize("kind", ["person", "person_agent"])
def test_avatar_account_must_equal_sender_owner(proxy, kind):
    client, calls, _ = proxy
    query = PHOTO_Q.replace("kind=person", f"kind={kind}").replace(
        "avatar_account_id=acc_2", "avatar_account_id=acc_3"
    )
    response = client.get(f"{URL}?{query}")
    assert response.status_code == 422
    assert not calls


def test_wrong_viewer_and_unsigned_are_refused_without_upstream(proxy):
    client, calls, state = proxy
    assert client.get(f"{URL}?{PHOTO_Q.replace('acc_1', 'acc_other')}").status_code == 409
    assert client.get(f"{URL}?{BASE_Q.replace('acc_1', 'acc_other')}").status_code == 409
    state["store"].rows = []
    assert client.get(f"{URL}?{BASE_Q}").status_code in (401, 409)
    assert not calls


def test_desktop_bearer_required(proxy):
    client, calls, _ = proxy
    response = client.get(f"{URL}?{BASE_Q}", headers={"authorization": "Bearer wrong"})
    assert response.status_code == 401
    assert not calls


@pytest.mark.parametrize(
    "mime,body",
    [("image/png", b"\x89PNG\r\n\x1a\n" + b"\0" * 30), ("image/svg+xml", b"<svg/>")],
)
def test_bad_upstream_bytes_are_refused(proxy, mime, body):
    client, _, state = proxy
    state.update(serve_type=mime, serve=body)
    assert client.get(f"{URL}?{PHOTO_Q}").status_code == 502


def test_upstream_failure_is_a_refusal_not_a_stale_image(proxy):
    client, _, state = proxy
    state.update(serve_type="image/jpeg", serve=_jpeg(), etag='"one"')
    _png(client.get(f"{URL}?{PHOTO_Q}"))
    state["error"] = (404, "avatar_not_found")
    # The composite for the old bytes is cached, but the photo path
    # revalidates first: a revoked/stale hash gets no image.
    assert client.get(f"{URL}?{PHOTO_Q}").status_code == 404


def test_composite_cache_is_keyed_by_login_and_purged_on_transition(proxy):
    from local_operator.team_bridge import avatar_cache

    client, _, state = proxy
    _png(client.get(f"{URL}?{BASE_Q}"))
    assert len(avatar_cache.COMPOSITE_CACHE._entries) == 1
    first = next(iter(avatar_cache.COMPOSITE_CACHE._entries))
    _png(client.get(f"{URL}?{BASE_Q}"))
    assert len(avatar_cache.COMPOSITE_CACHE._entries) == 1  # served from cache
    # A->B->A: same account, new login incarnation -> the old render is gone.
    state["incarnation"] = "login-a-again"
    _png(client.get(f"{URL}?{BASE_Q}"))
    assert first not in avatar_cache.COMPOSITE_CACHE._entries
    assert len(avatar_cache.COMPOSITE_CACHE._entries) == 1
    # Sign-out purges composites too.
    state["store"].rows = []
    client.get(f"{URL}?{BASE_Q}")
    assert len(avatar_cache.COMPOSITE_CACHE._entries) == 0


def test_capability_is_advertised():
    from local_operator.server.features import static_feature_flags

    assert static_feature_flags()["team_bridge_notification_avatar"] == 1
