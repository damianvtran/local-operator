"""Notification-avatar route over actual loopback HTTP, sharing the attachment rig.

Proves the closed query, the owner-avatar equality rule, the viewer fence, that
the only upstream read is the closed ``/accounts/{id}/avatar`` relay, that a
placeholder touches no network, and that core keeps no composite (the desktop
owns display retention) while still fencing each render to the current login.
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
    pixel = image.getpixel((20, 20))
    assert isinstance(pixel, tuple)
    assert pixel[:3] == pytest.approx((200, 40, 40), abs=16)
    assert [c[1] for c in calls] == ["/v1/accounts/acc_2/avatar?v=0123456789abcdef&size=64"]


def test_agent_photo_reference_is_accepted_and_never_fetched(proxy):
    """The exact shape a RELEASED desktop sends for an agent (agent review round 1, F1).

    An agent's directory row is its owner's, so desktops already in the field
    pass the OWNER's photo version with ``kind=person_agent``. The route must
    answer 200 with the glyph composite and touch no upstream route at all: if
    a later change passed ``v`` straight through to some fetch, this is the case
    that would notice.
    """

    client, calls, state = proxy
    state.update(serve_type="image/jpeg", serve=_jpeg())
    image = _png(client.get(f"{URL}?{PHOTO_Q.replace('kind=person', 'kind=person_agent')}"))
    assert calls == []
    # Not the photo it was handed: the composite is the glyph on the owner's
    # palette colour, so the red JPEG cannot be what came back.
    pixel = image.getpixel((20, 20))
    assert isinstance(pixel, tuple)
    assert pixel[:3] != pytest.approx((200, 40, 40), abs=16)


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
    "mime,body,code",
    [
        ("image/png", b"\x89PNG\r\n\x1a\n" + b"\0" * 30, "malformed_page"),
        ("image/svg+xml", b"<svg/>", "bridge_invalid_response"),
    ],
)
def test_bad_upstream_bytes_are_refused(proxy, mime, body, code):
    client, _, state = proxy
    state.update(serve_type=mime, serve=body)
    response = client.get(f"{URL}?{PHOTO_Q}")
    assert response.status_code == 502
    # A bad image keeps its own code (the relay refuses an SVG before any
    # decode); overload is the separate ``renderer_busy`` (below).
    assert response.json()["detail"]["code"] == code


def test_upstream_failure_is_a_refusal_not_a_stale_image(proxy):
    client, _, state = proxy
    state.update(serve_type="image/jpeg", serve=_jpeg(), etag='"one"')
    _png(client.get(f"{URL}?{PHOTO_Q}"))
    state["error"] = (404, "avatar_not_found")
    # Nothing derived from the old bytes is kept, and the photo path
    # revalidates first: a revoked/stale hash gets no image.
    assert client.get(f"{URL}?{PHOTO_Q}").status_code == 404


def test_each_request_renders_under_the_current_login_and_keeps_nothing(proxy):
    from local_operator.team_bridge import avatar_cache
    from local_operator.team_bridge import notification_avatar as na

    client, _, state = proxy
    first = client.get(f"{URL}?{BASE_Q}")
    _png(first)
    assert not hasattr(avatar_cache, "COMPOSITE_CACHE")
    # A->B->A: same account, new login incarnation -> still rendered, no state.
    state["incarnation"] = "login-a-again"
    again = client.get(f"{URL}?{BASE_Q}")
    _png(again)
    assert again.content == first.content
    assert not any("CACHE" in name or "IN_FLIGHT" in name for name in vars(na))
    # Sign-out refuses rather than rendering from anything retained.
    state["store"].rows = []
    signed_out = client.get(f"{URL}?{BASE_Q}")
    assert signed_out.status_code == 401
    assert signed_out.json()["detail"]["code"] == "bridge_signin_required"


def test_capability_is_advertised():
    from local_operator.server.features import static_feature_flags

    assert static_feature_flags()["team_bridge_notification_avatar"] == 1


def test_overload_is_a_transient_503_not_a_bad_image(proxy, monkeypatch):
    """A full pending queue is ``503 renderer_busy``, distinct from ``502
    malformed_page`` (QA round 2, Q2), and carries no ``Retry-After``: the
    desktop shows initials now and never retries or negative-caches it."""

    import threading

    from local_operator.team_bridge import notification_avatar as na

    client, calls, _ = proxy
    exhausted = threading.BoundedSemaphore(1)
    exhausted.acquire()
    monkeypatch.setattr(na, "_PENDING", exhausted)
    response = client.get(f"{URL}?{BASE_Q}")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "renderer_busy"
    assert "retry-after" not in response.headers
    assert not calls


AGENT_Q = (
    BASE_Q.replace("kind=person", "kind=person_agent")
    + "&v=fedcba9876543210&avatar_endpoint_id=bep_0123456789abcdef0123456789abcdef"
)


EP = "bep_0123456789abcdef0123456789abcdef"
DIRECTORY = [
    {
        "account_id": "acc_2",
        "display_label": "Damian",
        "agent": {
            "endpoint_id": EP,
            "name": "Aida",
            "state": "active",
            "avatar_url": f"/v1/messaging/endpoints/{EP}/avatar?v=fedcba9876543210",
        },
    },
    {
        "account_id": "acc_3",
        "display_label": "Bea",
        "agent": {
            "endpoint_id": "bep_bea",
            "name": "Otto",
            "state": "active",
            "avatar_url": "/v1/messaging/endpoints/bep_bea/avatar?v=b2b2b2b2b2b2b2b2",
        },
    },
]


def _without_directory(calls):
    return [c[1] for c in calls if not c[1].startswith("/v1/messaging/directory?")]


def _blue_jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 60, 210)).save(out, format="JPEG")
    return out.getvalue()


def test_agent_composite_reads_only_the_endpoint_route(proxy):
    client, calls, state = proxy
    state.update(serve_type="image/jpeg", serve=_blue_jpeg(), directory=DIRECTORY)
    response = client.get(f"{URL}?{AGENT_Q}")
    image = _png(response)
    pixel = image.getpixel((16, 32))
    assert isinstance(pixel, tuple)
    assert pixel[:3] == pytest.approx((30, 60, 210), abs=16)
    assert [c[1] for c in calls] == [
        "/v1/messaging/directory?tenant_id=tenant_a",
        "/v1/messaging/endpoints/bep_0123456789abcdef0123456789abcdef/avatar"
        "?v=fedcba9876543210&size=64",
    ]
    # The agent's own picture was drawn: no fallback signal.
    assert "x-avatar-fallback" not in response.headers


def test_agent_composite_falls_back_to_the_glyph_on_a_stale_version(proxy):
    client, calls, state = proxy
    state["directory"] = DIRECTORY
    glyph = client.get(f"{URL}?{BASE_Q.replace('kind=person', 'kind=person_agent')}")
    assert "x-avatar-fallback" not in glyph.headers  # no picture was asked for
    state["refuse_path"] = lambda path: (404, "not_found") if "/avatar?" in path else None
    fallback = client.get(f"{URL}?{AGENT_Q}")
    _png(fallback)
    assert fallback.content == glyph.content
    # F1: the desktop is told this is the glyph standing in for a picture.
    assert fallback.headers["x-avatar-fallback"] == "glyph"
    assert len(_without_directory(calls)) == 1


def test_a_composite_for_a_naming_bs_endpoint_gets_as_glyph(proxy):
    """Security review round 1, S-2. Sender acc_2's composite names Bea's agent
    endpoint (and its real current version): core finds that endpoint is not
    acc_2's in the tenant directory, draws acc_2's glyph, never reads Bea's
    picture, and signals the fallback."""

    client, calls, state = proxy
    state.update(serve_type="image/jpeg", serve=_blue_jpeg(), directory=DIRECTORY)
    glyph = client.get(f"{URL}?{BASE_Q.replace('kind=person', 'kind=person_agent')}")
    crossed = AGENT_Q.replace(EP, "bep_bea").replace("fedcba9876543210", "b2b2b2b2b2b2b2b2")
    response = client.get(f"{URL}?{crossed}")
    _png(response)
    assert response.content == glyph.content
    assert response.headers["x-avatar-fallback"] == "glyph"
    assert _without_directory(calls) == [], "B's picture was read for A"


@pytest.mark.parametrize(
    "query",
    [
        # A person never wears an endpoint picture.
        AGENT_Q.replace("kind=person_agent", "kind=person"),
        # Both photo shapes at once, or a half of either.
        AGENT_Q + "&avatar_account_id=acc_2",
        AGENT_Q.replace("&v=fedcba9876543210", ""),
        AGENT_Q.replace("fedcba9876543210", "FEDCBA9876543210"),
        AGENT_Q.replace("bep_0123456789abcdef0123456789abcdef", "bep%2F..%2Facc_2"),
        AGENT_Q.replace("bep_0123456789abcdef0123456789abcdef", "b" * 129),
    ],
)
def test_agent_picture_query_is_closed(proxy, query):
    client, calls, _ = proxy
    assert client.get(f"{URL}?{query}").status_code == 422
    assert not calls
