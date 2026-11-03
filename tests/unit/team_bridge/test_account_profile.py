"""Self-profile writes never acquire authority from a later login.

These are synthetic transport tests, not claims of live account integration.
Independent QA additionally drives the desktop routes against a real HTTP fixture.
"""

from __future__ import annotations

import json

import httpx
import pytest

from local_operator.team_bridge.client import BridgeRefusal, TeamBridgeClient
from tests.unit.team_bridge.test_avatar import AvatarStore

pytestmark = pytest.mark.asyncio
PROFILE = {
    "account_id": "acc_1",
    "display_name": "Fixture",
    "display_label": "Fixture",
    "title": "",
    "description": "",
    "timezone": "UTC",
}


def fixture_client(tmp_path, handler):
    store = AvatarStore()
    return store, TeamBridgeClient(tmp_path, store=store, transport=httpx.MockTransport(handler))


async def scope(client):
    return (await client.account_profile_request("read", expected_account_id="acc_1"))["scope"]


async def test_partial_patch_bytes_and_normalized_result(tmp_path):
    calls = []

    def upstream(request):
        calls.append(request)
        if request.method == "PATCH":
            assert request.url.path == "/v1/account/profile"
            assert json.loads(request.content) == {"title": "Fixture title"}
        return httpx.Response(
            200,
            json={
                "result": PROFILE
                | {"title": "Fixture title", "pronouns": "not returned", "token": "never returned"}
            },
        )

    _, client = fixture_client(tmp_path, upstream)
    receipt = await scope(client)
    result = await client.account_profile_request(
        "update",
        expected_account_id="acc_1",
        expected_scope=receipt,
        fields={"title": "Fixture title"},
    )
    assert result["profile"]["title"] == "Fixture title"
    assert "pronouns" not in result["profile"] and "token" not in result["profile"]
    assert len(calls) == 2


@pytest.mark.parametrize("operation", ["update", "upload", "remove"])
async def test_stale_editor_cannot_dispatch(tmp_path, operation):
    calls = []
    store, client = fixture_client(
        tmp_path,
        lambda request: calls.append(request) or httpx.Response(200, json={"result": PROFILE}),
    )
    receipt = await scope(client)
    store.incarnation = "login-a-again"
    with pytest.raises(BridgeRefusal) as failure:
        await client.account_profile_request(
            operation,
            expected_account_id="acc_1",
            expected_scope=receipt,
            fields={"title": "x"},
            image=b"image",
        )
    assert failure.value.http_status == 409
    assert len(calls) == 1


async def test_relogin_during_refresh_never_dispatches(tmp_path):
    calls = []
    store, client = fixture_client(
        tmp_path,
        lambda request: calls.append(request) or httpx.Response(200, json={"result": PROFILE}),
    )
    receipt = await scope(client)
    original = store.ensure_oauth_fresh_or_raise

    async def refresh(*args, **kwargs):
        data = await original(*args, **kwargs)
        store.incarnation = "login-a-again"
        return data

    store.ensure_oauth_fresh_or_raise = refresh
    with pytest.raises(BridgeRefusal):
        await client.account_profile_request(
            "update", expected_account_id="acc_1", expected_scope=receipt, fields={"title": "x"}
        )
    assert len(calls) == 1


async def test_post_dispatch_switch_suppresses_success_not_commit(tmp_path):
    commits = []
    store = AvatarStore()

    def upstream(request):
        if request.method == "PATCH":
            commits.append("upstream accepted")
            store.incarnation = "login-a-again"
        return httpx.Response(200, json={"result": PROFILE})

    client = TeamBridgeClient(tmp_path, store=store, transport=httpx.MockTransport(upstream))
    receipt = await scope(client)
    with pytest.raises(BridgeRefusal):
        await client.account_profile_request(
            "update", expected_account_id="acc_1", expected_scope=receipt, fields={"title": "x"}
        )
    assert commits == ["upstream accepted"]


async def test_401_is_never_replayed(tmp_path):
    calls = []

    def upstream(request):
        calls.append(request)
        return (
            httpx.Response(200, json={"result": PROFILE})
            if request.method == "GET"
            else httpx.Response(401, json={"code": "unauthorized"})
        )

    _, client = fixture_client(tmp_path, upstream)
    receipt = await scope(client)
    with pytest.raises(BridgeRefusal):
        await client.account_profile_request(
            "update", expected_account_id="acc_1", expected_scope=receipt, fields={"title": "x"}
        )
    assert [request.method for request in calls] == ["GET", "PATCH"]


async def test_upload_fixed_multipart_and_explicit_remove_modes(tmp_path):
    calls = []

    def upstream(request):
        calls.append(request)
        if request.method == "PUT":
            assert request.url.path == "/v1/account/avatar"
            assert request.headers["content-type"].startswith("multipart/form-data;")
            assert b'name="file"' in request.content and b"fixture-image" in request.content
        if request.method == "DELETE":
            assert dict(request.url.params) in [{"mode": "uploaded"}, {"mode": "provider"}]
        return httpx.Response(200, json={"result": PROFILE})

    _, client = fixture_client(tmp_path, upstream)
    receipt = await scope(client)
    await client.account_profile_request(
        "upload", expected_account_id="acc_1", expected_scope=receipt, image=b"fixture-image"
    )
    for mode in ("uploaded", "provider"):
        await client.account_profile_request(
            "remove", expected_account_id="acc_1", expected_scope=receipt, mode=mode
        )
    assert [r.method for r in calls] == ["GET", "PUT", "DELETE", "DELETE"]


async def test_client_entry_relogin_refuses_before_dispatch(tmp_path, monkeypatch):
    calls = []
    store, client = fixture_client(
        tmp_path,
        lambda request: calls.append(request) or httpx.Response(200, json={"result": PROFILE}),
    )
    receipt = await scope(client)
    enter = httpx.AsyncClient.__aenter__

    async def swapped(http):
        result = await enter(http)
        store.incarnation = "login-a-again"
        return result

    monkeypatch.setattr(httpx.AsyncClient, "__aenter__", swapped)
    with pytest.raises(BridgeRefusal):
        await client.account_profile_request(
            "update", expected_account_id="acc_1", expected_scope=receipt, fields={"title": "x"}
        )
    assert len(calls) == 1, "No mutation may leave after the AsyncClient entry await changes login"


async def test_compressed_response_refused_before_decoder(tmp_path):
    import gzip

    _, client = fixture_client(
        tmp_path,
        lambda request: httpx.Response(
            200,
            content=gzip.compress(json.dumps({"result": PROFILE}).encode()),
            headers={"content-encoding": "gzip", "content-type": "application/json"},
        ),
    )
    with pytest.raises(BridgeRefusal) as failure:
        await scope(client)
    assert failure.value.http_status == 502


async def test_mismatched_response_account_refused(tmp_path):
    _, client = fixture_client(
        tmp_path,
        lambda request: httpx.Response(200, json={"result": PROFILE | {"account_id": "acc_other"}}),
    )
    with pytest.raises(BridgeRefusal) as failure:
        await scope(client)
    assert failure.value.http_status == 502
