"""The community client against a fake ``/community/v1`` hub: every member
operation, the credential pin, the error mapping and the no-body-in-logs rule.

See ``fakehub.py`` for why this is a wire-level fake rather than mocks.
"""

from __future__ import annotations

import logging
import traceback

import httpx
import pytest

from local_operator.community.client import (
    EVENTS_TIMEOUT_S,
    MAX_RESPONSE_BYTES,
    CommunityImage,
    community_base_url,
)
from local_operator.community.errors import (
    CommunityAccountMismatch,
    CommunityBadRequest,
    CommunityConflict,
    CommunityDestinationRefused,
    CommunityForbidden,
    CommunityGuidelinesChanged,
    CommunityImagesUnavailable,
    CommunityInvalidRequest,
    CommunityInvalidResponse,
    CommunityModerationUnavailable,
    CommunityNotFound,
    CommunityPayloadTooLarge,
    CommunityRateLimited,
    CommunityRefused,
    CommunitySignInRequired,
    CommunityUnavailable,
    CommunityUnreachable,
)
from tests.unit.community.fakehub import (
    JPEG,
    KEY,
    PNG,
    FakeHub,
    FakeRow,
    FakeStore,
    make_client,
)

pytestmark = pytest.mark.asyncio

CID = "local-operator"


@pytest.fixture
def hub() -> FakeHub:
    return FakeHub()


# -- base URL ------------------------------------------------------------------


async def test_community_base_is_root_relative_on_the_api_origin() -> None:
    assert (
        community_base_url("https://api.radienthq.com/v1")
        == "https://api.radienthq.com/community/v1"
    )
    assert community_base_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080/community/v1"


async def test_requests_go_to_community_v1_with_the_pinned_bearer(hub: FakeHub) -> None:
    listed = await make_client(hub).list_communities()
    assert listed.featured[0].id == CID
    (sent,) = hub.sent
    assert sent.method == "GET" and sent.path == "/community/v1/communities"
    assert sent.headers["authorization"] == "Bearer token-1"
    assert "cookie" not in sent.headers


# -- every member operation ------------------------------------------------------


async def test_bootstrap_and_explicit_join(hub: FakeHub) -> None:
    client = make_client(hub)
    boot = await client.bootstrap(CID)
    assert boot.membership is None and boot.guidelines.version == "2026-11-01"
    joined = await client.join(CID, guidelines_version="2026-11-01", accept_guidelines=True)
    assert joined.membership is not None
    assert hub.sent[-1].path == f"/community/v1/communities/{CID}/join"
    import json

    assert json.loads(hub.sent[-1].body) == {
        "guidelines_version": "2026-11-01",
        "accept_guidelines": True,
    }


@pytest.mark.parametrize("accept", [False, None, 1, "true"])
async def test_join_without_literal_acceptance_sends_nothing(hub: FakeHub, accept) -> None:
    with pytest.raises(CommunityInvalidRequest):
        await make_client(hub).join(CID, guidelines_version="v1", accept_guidelines=accept)
    assert hub.sent == []


async def test_leave_and_second_leave_reads_as_already_left(hub: FakeHub) -> None:
    client = make_client(hub)
    assert await client.leave(CID) is False
    hub.override = lambda s: httpx.Response(404, json={"message": "community not found"})
    assert await client.leave(CID) is True


async def test_channels_history_and_their_query_bounds(hub: FakeHub) -> None:
    client = make_client(hub)
    page = await client.channels(CID, q="gen", filter="public", cursor="c1")
    assert page.items[0].name == "general"
    assert hub.sent[-1].path.endswith("/directory/channels")
    assert hub.sent[-1].params == {"q": "gen", "filter": "public", "cursor": "c1"}
    history = await client.history(CID, "conv_general", cursor="c1")
    assert history.items[0].text == "hello" and history.next_cursor == "c2"
    before = len(hub.sent)
    for bad in ({"filter": "staff"}, {"q": "x" * 101}, {"cursor": "c" * 513}):
        with pytest.raises(CommunityInvalidRequest):
            await client.channels(CID, **bad)
    assert len(hub.sent) == before


async def test_submit_streams_a_sized_multipart_with_server_side_names(hub: FakeHub) -> None:
    result = await make_client(hub).submit(
        CID,
        "conv_general",
        text="hello world",
        images=[CommunityImage("image/png", PNG), CommunityImage("image/jpeg", JPEG)],
        thread_root_id="msg_1",
        guidelines_version="2026-11-01",
        idempotency_key=KEY,
    )
    assert result.state == "published" and result.message_id == "msg_2"
    (sent,) = hub.sent
    assert sent.path == f"/community/v1/communities/{CID}/conversations/conv_general/submissions"
    assert sent.headers["idempotency-key"] == KEY
    # Sized, never chunked: a chunked upload is what an ingress buffers to disk.
    assert int(sent.headers["content-length"]) == len(sent.body)
    assert "transfer-encoding" not in sent.headers
    boundary = sent.headers["content-type"].split("boundary=", 1)[1]
    parts = sent.body.split(b"--" + boundary.encode())[1:-1]
    names = [p.split(b"\r\n\r\n", 1)[0] for p in parts]
    assert b'name="text"' in names[0] and parts[0].endswith(b"hello world\r\n")
    assert b'name="thread_root_id"' in names[1]
    assert b'name="guidelines_version"' in names[2]
    assert b'name="images"; filename="image-1.png"' in names[3]
    assert b'name="images"; filename="image-2.jpg"' in names[4]
    assert parts[3].split(b"\r\n\r\n", 1)[1] == PNG + b"\r\n"


async def test_submit_mints_a_contract_key_when_none_is_given(hub: FakeHub) -> None:
    await make_client(hub).submit(CID, "conv_general", text="hi")
    key = hub.sent[0].headers["idempotency-key"]
    import re

    assert re.fullmatch(r"[A-Za-z0-9_-]{16,64}", key)


@pytest.mark.parametrize(
    "images",
    [
        [CommunityImage("image/png", JPEG)],  # declared type is not what the bytes are
        [CommunityImage("image/gif", b"GIF89a")],
        [CommunityImage("image/png", PNG)] * 5,
        [CommunityImage("image/png", PNG + b"\x00" * (5 * 1024 * 1024))],
    ],
)
async def test_bad_images_are_refused_before_upload(hub: FakeHub, images) -> None:
    with pytest.raises(CommunityInvalidRequest):
        await make_client(hub).submit(CID, "conv_general", text="x", images=images)
    assert hub.sent == []


async def test_status_cancel_edit_delete_react_events(hub: FakeHub) -> None:
    client = make_client(hub)
    assert (await client.submission(CID, KEY)).state == "checking"
    assert (await client.cancel_submission(CID, KEY)).state == "cancelled"
    assert hub.sent[-1].path.endswith(f"/submissions/{KEY}:cancel")
    assert (
        await client.edit(CID, "msg_1", text="edited", idempotency_key=KEY)
    ).state == "published"
    assert hub.sent[-1].path.endswith("/messages/msg_1:edit")
    assert hub.sent[-1].headers["idempotency-key"] == KEY
    assert await client.delete_message(CID, "msg_1") is None
    assert hub.sent[-1].method == "DELETE"
    summary = await client.react(CID, "msg_1", emoji="👍")
    assert summary.count == 2
    batch = await client.events(CID, after="e1")
    assert batch.next == "e2" and batch.events[0].type == "conversation_changed"
    # The long-poll's read deadline outlasts the contract's 25 s hold.
    assert hub.sent[-1].timeout["read"] == EVENTS_TIMEOUT_S > 25


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.bootstrap("../admin"),
        lambda c: c.bootstrap("AB"),
        lambda c: c.history("local-operator", "conv/../x"),
        lambda c: c.delete_message("local-operator", "m?x=1"),
        lambda c: c.submission("local-operator", "short"),
    ],
)
async def test_ids_that_could_reshape_the_url_never_leave(hub: FakeHub, call) -> None:
    with pytest.raises(CommunityInvalidRequest):
        await call(make_client(hub))
    assert hub.sent == []


# -- error mapping -------------------------------------------------------------


def _answer(hub: FakeHub, status: int, body=None, headers=None) -> None:
    hub.override = lambda s: httpx.Response(status, json=body, headers=headers)


async def test_refusal_carries_only_category_and_reference(hub: FakeHub) -> None:
    _answer(
        hub,
        422,
        {
            "status": "refused",
            "category": "spam_or_promotion",
            "reference": "MOD-7K2Q-9XDF",
            "score": 0.97,
        },
    )
    with pytest.raises(CommunityRefused) as caught:
        await make_client(hub).submit(CID, "conv_general", text="buy now")
    assert caught.value.wire() == {
        "code": "refused",
        "category": "spam_or_promotion",
        "reference": "MOD-7K2Q-9XDF",
    }


async def test_unreadable_refusal_is_never_given_a_guessed_category(hub: FakeHub) -> None:
    _answer(
        hub, 422, {"status": "refused", "category": "new_category", "reference": "MOD-7K2Q-9XDF"}
    )
    with pytest.raises(CommunityInvalidResponse):
        await make_client(hub).submit(CID, "conv_general", text="x")


async def test_rate_limit_carries_retry_after_and_scope(hub: FakeHub) -> None:
    _answer(hub, 429, {"message": "slow down", "retry_after": 42, "scope": "account_burst"})
    with pytest.raises(CommunityRateLimited) as caught:
        await make_client(hub).submit(CID, "conv_general", text="x")
    assert caught.value.wire() == {
        "code": "rate_limited",
        "retry_after": 42,
        "scope": "account_burst",
    }


async def test_rate_limit_falls_back_to_the_header_and_drops_an_unknown_scope(hub: FakeHub) -> None:
    _answer(hub, 429, {"scope": "community_wide"}, headers={"Retry-After": "7"})
    with pytest.raises(CommunityRateLimited) as caught:
        await make_client(hub).channels(CID)
    assert (caught.value.retry_after, caught.value.scope) == (7, None)


async def test_guidelines_changed_names_the_new_version(hub: FakeHub) -> None:
    _answer(
        hub,
        409,
        {
            "message": "guidelines version is no longer current",
            "code": "guidelines_changed",
            "guidelines_version": "v2",
        },
    )
    with pytest.raises(CommunityGuidelinesChanged) as caught:
        await make_client(hub).join(CID, guidelines_version="v1", accept_guidelines=True)
    assert caught.value.wire() == {"code": "guidelines_changed", "guidelines_version": "v2"}


async def test_moderation_unavailable_and_images_unavailable_are_distinct(hub: FakeHub) -> None:
    client = make_client(hub)
    _answer(hub, 503, {"message": "x", "code": "moderation_unavailable", "retry_after": 30})
    with pytest.raises(CommunityModerationUnavailable) as caught:
        await client.submit(CID, "conv_general", text="x")
    assert caught.value.wire() == {"code": "moderation_unavailable", "retry_after": 30}
    for status in (403, 409, 503):
        _answer(hub, status, {"message": "x", "code": "images_unavailable"})
        with pytest.raises(CommunityImagesUnavailable):
            await client.submit(
                CID, "conv_general", text="x", images=[CommunityImage("image/png", PNG)]
            )


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (400, {"message": "bad"}, CommunityBadRequest),
        (403, {"message": "no seat"}, CommunityForbidden),
        (404, {"message": "community not found"}, CommunityNotFound),
        (409, {"message": "channel locked"}, CommunityConflict),
        (413, {"message": "too big"}, CommunityPayloadTooLarge),
        (500, {"message": "community membership is unavailable"}, CommunityUnavailable),
        (503, {"message": "down"}, CommunityUnavailable),
        (418, None, CommunityInvalidResponse),
    ],
)
async def test_status_classes(hub: FakeHub, status, body, error) -> None:
    _answer(hub, status, body)
    with pytest.raises(error) as caught:
        await make_client(hub).bootstrap(CID)
    # Never the hub's English message: only the code travels.
    assert str(caught.value) == caught.value.code


async def test_a_conflict_that_is_not_guidelines_is_not_mistaken_for_one(hub: FakeHub) -> None:
    _answer(hub, 409, {"message": "channel locked"})
    with pytest.raises(CommunityConflict) as caught:
        await make_client(hub).submit(CID, "conv_general", text="x")
    assert not isinstance(caught.value, CommunityGuidelinesChanged)


async def test_malformed_success_and_oversized_answers_are_invalid_responses(hub: FakeHub) -> None:
    client = make_client(hub)
    _answer(hub, 200, {"joined": "nope"})
    with pytest.raises(CommunityInvalidResponse):
        await client.list_communities()
    hub.override = lambda s: httpx.Response(200, content=b"x" * (MAX_RESPONSE_BYTES + 1))
    with pytest.raises(CommunityInvalidResponse):
        await client.list_communities()


async def test_transport_failure_is_unreachable(hub: FakeHub) -> None:
    def boom(sent):
        raise httpx.ConnectError("refused")

    hub.override = boom
    with pytest.raises(CommunityUnreachable):
        await make_client(hub).list_communities()


# -- credential pin --------------------------------------------------------------


async def test_one_forced_refresh_on_401_then_resend(hub: FakeHub) -> None:
    store = FakeStore(refresh={"access": "token-2", "account_id": "acc_1"})
    hub.override = lambda s: (
        httpx.Response(401, json={"message": "expired"})
        if s.headers["authorization"] == "Bearer token-1"
        else None
    )
    await make_client(hub, store).list_communities()
    assert store.forced == [1]
    assert [s.headers["authorization"] for s in hub.sent] == ["Bearer token-1", "Bearer token-2"]


async def test_the_retry_re_streams_the_whole_upload(hub: FakeHub) -> None:
    store = FakeStore(refresh={"access": "token-2", "account_id": "acc_1"})
    hub.override = lambda s: (
        httpx.Response(401) if s.headers["authorization"] == "Bearer token-1" else None
    )
    await make_client(hub, store).submit(
        CID,
        "conv_general",
        text="x",
        images=[CommunityImage("image/png", PNG)],
        idempotency_key=KEY,
    )
    first, second = hub.sent
    assert first.body == second.body and PNG in second.body
    assert first.headers["idempotency-key"] == second.headers["idempotency-key"] == KEY


async def test_a_second_401_is_the_sign_in_state_not_a_loop(hub: FakeHub) -> None:
    store = FakeStore(refresh={"access": "token-2", "account_id": "acc_1"})
    _answer(hub, 401, {"message": "no"})
    with pytest.raises(CommunitySignInRequired):
        await make_client(hub, store).list_communities()
    assert len(hub.sent) == 2 and store.forced == [1]


async def test_a_refresh_onto_another_account_never_resends(hub: FakeHub) -> None:
    store = FakeStore(refresh={"access": "token-b", "account_id": "acc_b"})
    _answer(hub, 401)
    with pytest.raises(CommunitySignInRequired):
        await make_client(hub, store).list_communities()
    assert len(hub.sent) == 1


async def test_a_captured_account_that_is_not_the_pinned_one_sends_nothing(hub: FakeHub) -> None:
    with pytest.raises(CommunityAccountMismatch):
        await make_client(hub, expected_account_id="acc_other").list_communities()
    assert hub.sent == []
    await make_client(hub, expected_account_id="acc_1").list_communities()
    assert len(hub.sent) == 1


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [
            FakeRow(1, data={"access": "a", "account_id": "x"}),
            FakeRow(2, data={"access": "b", "account_id": "y"}),
        ],
        [FakeRow(1, data={"access": "a"})],  # no identity on the row: cannot be attributed
        [FakeRow(1, data={"access": "a", "account_id": "x"}, credential_type="api_key")],
    ],
)
async def test_no_single_attributable_oauth_row_is_the_sign_in_state(hub: FakeHub, rows) -> None:
    with pytest.raises(CommunitySignInRequired):
        await make_client(hub, FakeStore(rows)).list_communities()
    assert hub.sent == []


async def test_the_resolved_row_stays_pinned_for_the_clients_lifetime(hub: FakeHub) -> None:
    store = FakeStore()
    client = make_client(hub, store)
    await client.list_communities()
    # A second login appears; the client keeps its row rather than refusing or
    # drifting onto the newcomer.
    store.rows.append(FakeRow(2, data={"access": "token-x", "account_id": "acc_x"}))
    await client.list_communities()
    assert [s.headers["authorization"] for s in hub.sent] == ["Bearer token-1", "Bearer token-1"]


async def test_destinations_the_bearer_must_not_reach(hub: FakeHub, monkeypatch) -> None:
    monkeypatch.delenv("RADIENT_ORG_ALLOW_NONCANONICAL_BASE", raising=False)
    with pytest.raises(CommunityDestinationRefused):
        await make_client(hub, base_url="https://evil.example/v1").list_communities()
    with pytest.raises(CommunityDestinationRefused):
        await make_client(hub, base_url="https://user:pw@api.radienthq.com/v1").list_communities()
    assert hub.sent == []
    monkeypatch.setenv("RADIENT_ORG_ALLOW_NONCANONICAL_BASE", "1")
    await make_client(hub, base_url="http://127.0.0.1:9/v1").list_communities()
    assert str(hub.sent[0].path) == "/community/v1/communities"


# -- bodies never logged -----------------------------------------------------------


async def test_request_and_response_bodies_never_reach_a_log(hub: FakeHub, caplog) -> None:
    secret = "SECRET-DRAFT-7f3a"
    caplog.set_level(logging.DEBUG)
    client = make_client(hub)
    await client.submit(CID, "conv_general", text=secret)
    _answer(
        hub,
        422,
        {
            "status": "refused",
            "category": "spam_or_promotion",
            "reference": "MOD-AAAA-BBBB",
            "echo": secret,
        },
    )
    with pytest.raises(CommunityRefused) as refused:
        await client.submit(CID, "conv_general", text=secret)
    # A malformed success that QUOTES the body: pydantic's error would carry it,
    # so the chain must be cut, not merely unlogged.
    _answer(hub, 200, {"items": [{"id": secret}], "next_cursor": 5})
    with pytest.raises(CommunityInvalidResponse) as invalid:
        await client.history(CID, "conv_general")
    logging.getLogger("test").exception("logged", exc_info=invalid.value)
    for record in caplog.records:
        assert secret not in record.getMessage()
    assert secret not in caplog.text
    for caught in (refused.value, invalid.value):
        assert secret not in "".join(traceback.format_exception(caught))
