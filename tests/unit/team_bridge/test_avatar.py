"""Closed avatar transport; owner incarnation survives A→B→A checks."""

from __future__ import annotations

import httpx
import pytest

from local_operator.team_bridge.client import BridgeRefusal, TeamBridgeClient
from tests.unit.team_bridge.test_client import FakeStore

pytestmark = pytest.mark.asyncio
PNG = b"\x89PNG\r\n\x1a\nfixture"


class AvatarStore(FakeStore):
    incarnation: str | None = "login-a"

    def native_login_incarnation(self, credential_id):
        return self.incarnation


async def test_avatar_success(tmp_path):
    store = AvatarStore()

    def upstream(request):
        assert request.url.path == "/v1/accounts/acc_2/avatar"
        assert dict(request.url.params) == {"v": "0123456789abcdef", "size": "64"}
        assert request.headers["authorization"] == "Bearer token-1"
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

    client = TeamBridgeClient(tmp_path, store=store, transport=httpx.MockTransport(upstream))
    assert await client.fetch_avatar(
        "acc_2", v="0123456789abcdef", size=64, expected_account_id="acc_1"
    ) == (PNG, "image/png")


async def test_avatar_relogin_during_read(tmp_path):
    store = AvatarStore()

    def upstream(request):
        store.incarnation = "login-a-again"
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

    client = TeamBridgeClient(tmp_path, store=store, transport=httpx.MockTransport(upstream))
    with pytest.raises(BridgeRefusal) as failure:
        await client.fetch_avatar(
            "acc_2", v="0123456789abcdef", size=64, expected_account_id="acc_1"
        )
    assert failure.value.http_status == 409


@pytest.mark.parametrize("change", ["logout", "swap", "aba", "missing_epoch"])
async def test_stream_owner_fence(tmp_path, change):
    store = AvatarStore()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield PNG
            if change == "logout":
                store.rows = []
            elif change == "swap":
                store.rows[0].data["account_id"] = "acc_other"
            elif change == "aba":
                store.rows[0].data["account_id"] = "acc_other"
                store.incarnation = "login-b"
                store.rows[0].data["account_id"] = "acc_1"
                store.incarnation = "login-a-again"
            else:
                store.incarnation = None
            yield b"end"

    client = TeamBridgeClient(
        tmp_path,
        store=store,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, stream=Stream(), headers={"content-type": "image/png"}
            )
        ),
    )
    with pytest.raises(BridgeRefusal) as failure:
        await client.fetch_avatar(
            "acc_2", v="0123456789abcdef", size=64, expected_account_id="acc_1"
        )
    assert failure.value.http_status == 409


@pytest.mark.parametrize(
    "target", ["../acc_2", "https://evil.test", "acc_2?x=y", "acc_2%2favatar", "acc_2#fragment"]
)
async def test_target_cannot_be_url(tmp_path, target):
    def upstream(request):
        pytest.fail("invalid target reached transport")

    client = TeamBridgeClient(
        tmp_path, store=AvatarStore(), transport=httpx.MockTransport(upstream)
    )
    with pytest.raises(ValueError):
        await client.fetch_avatar(
            target, v="0123456789abcdef", size=64, expected_account_id="acc_1"
        )


async def test_redirect_is_not_followed(tmp_path):
    calls = []

    def upstream(request):
        calls.append(request.url)
        return httpx.Response(302, headers={"location": "https://evil.test/avatar"})

    client = TeamBridgeClient(
        tmp_path, store=AvatarStore(), transport=httpx.MockTransport(upstream)
    )
    with pytest.raises(BridgeRefusal):
        await client.fetch_avatar(
            "acc_2", v="0123456789abcdef", size=64, expected_account_id="acc_1"
        )
    assert len(calls) == 1


async def test_stream_stops_at_limit(tmp_path):
    chunks = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for n in range(100):
                chunks.append(n)
                yield b"x" * 65536

    client = TeamBridgeClient(
        tmp_path,
        store=AvatarStore(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, stream=Stream(), headers={"content-type": "image/png"}
            )
        ),
    )
    with pytest.raises(BridgeRefusal) as failure:
        await client.fetch_avatar(
            "acc_2", v="0123456789abcdef", size=64, expected_account_id="acc_1"
        )
    assert failure.value.http_status == 502
    assert len(chunks) == 17


@pytest.mark.parametrize("relogin", [False, True])
async def test_real_auth_store_incarnation(tmp_path, relogin):
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(db_path=tmp_path / "auth.db")
    credential = {
        "type": "oauth",
        "access": "fixture-access",
        "refresh": "fixture-refresh",
        "expires": 9999999999999,
        "account_id": "acc_1",
    }
    row = store.upsert_credential("radient", credential)
    original = store.native_login_incarnation(row.id)

    def upstream(request):
        if relogin:
            store.upsert_credential("radient", credential)
            assert store.native_login_incarnation(row.id) != original
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

    client = TeamBridgeClient(tmp_path, store=store, transport=httpx.MockTransport(upstream))
    try:
        if relogin:
            with pytest.raises(BridgeRefusal) as failure:
                await client.fetch_avatar(
                    "acc_2", v="0123456789abcdef", size=64, expected_account_id="acc_1"
                )
            assert failure.value.http_status == 409
        else:
            assert (
                await client.fetch_avatar(
                    "acc_2", v="0123456789abcdef", size=64, expected_account_id="acc_1"
                )
            )[0] == PNG
    finally:
        store.close()


async def test_encoded_body_refused_before_read(tmp_path):
    class Unreadable(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail("compressed body must not be decoded or read")
            yield b"unreachable"

    client = TeamBridgeClient(
        tmp_path,
        store=AvatarStore(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                stream=Unreadable(),
                headers={"content-type": "image/png", "content-encoding": "gzip"},
            )
        ),
    )
    with pytest.raises(BridgeRefusal) as failure:
        await client.fetch_avatar(
            "acc_2", v="0123456789abcdef", size=64, expected_account_id="acc_1"
        )
    assert failure.value.http_status == 502


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg"])
async def test_exact_limit_and_large_thumbnail(tmp_path, mime):
    # Exactly 1 MiB, starting with the declared type's magic (S-1 sniffs it).
    magic = b"\x89PNG\r\n\x1a\n" if mime == "image/png" else b"\xff\xd8\xff"
    body = magic + b"x" * (1024 * 1024 - len(magic))

    def upstream(request):
        assert request.url.params["size"] == "256"
        return httpx.Response(200, content=body, headers={"content-type": mime})

    client = TeamBridgeClient(
        tmp_path, store=AvatarStore(), transport=httpx.MockTransport(upstream)
    )
    assert await client.fetch_avatar(
        "acc_2", v="0123456789abcdef", size=256, expected_account_id="acc_1"
    ) == (body, mime)


@pytest.mark.parametrize(
    "headers",
    [
        {"etag": '"valid"', "cache-control": "private, no-store"},
        {"etag": "unquoted"},
        {"etag": '"' + "x" * 255 + '"'},
        {},
    ],
)
async def test_noncacheable_http_entities_always_cold_fetch(tmp_path, headers):
    conditionals = []

    def upstream(request):
        conditionals.append(request.headers.get("if-none-match"))
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png", **headers})

    client = TeamBridgeClient(
        tmp_path, store=AvatarStore(), transport=httpx.MockTransport(upstream)
    )
    for _ in range(2):
        assert (
            await client.fetch_avatar(
                "acc_2", v="0123456789abcdef", size=64, expected_account_id="acc_1"
            )
        )[0] == PNG
    assert conditionals == [None, None]


async def test_transport_failure_evicts_without_stale_fallback(tmp_path):
    from local_operator.team_bridge.client import BridgeUnreachable

    conditionals = []

    def upstream(request):
        conditionals.append(request.headers.get("if-none-match"))
        if len(conditionals) == 2:
            raise httpx.ReadError("fixture connection failure")
        return httpx.Response(
            200, content=PNG, headers={"content-type": "image/png", "etag": '"valid"'}
        )

    client = TeamBridgeClient(
        tmp_path, store=AvatarStore(), transport=httpx.MockTransport(upstream)
    )
    args = {"v": "0123456789abcdef", "size": 64, "expected_account_id": "acc_1"}
    await client.fetch_avatar("acc_2", **args)
    with pytest.raises(BridgeUnreachable):
        await client.fetch_avatar("acc_2", **args)
    await client.fetch_avatar("acc_2", **args)
    assert conditionals == [None, '"valid"', None]


@pytest.mark.parametrize(
    "declared,body",
    [
        ("image/png", b"<svg xmlns='http://www.w3.org/2000/svg'><script/></svg>"),
        ("image/png", b"<!doctype html><html><script>alert(1)</script></html>"),
        ("image/jpeg", b"\x89PNG\r\n\x1a\nreally a png"),
        ("image/png", b"\xff\xd8\xffreally a jpeg"),
    ],
)
async def test_declared_type_must_match_the_bytes(tmp_path, declared, body):
    """Security review round 1, S-1: a hub label is not trusted on its own.
    Bytes whose magic contradicts the declared type are refused (502) on BOTH
    image routes, since they share ``_avatar_read``."""

    from local_operator.team_bridge.client import BridgeRefusal

    def upstream(request):
        return httpx.Response(200, content=body, headers={"content-type": declared})

    client = TeamBridgeClient(
        tmp_path, store=AvatarStore(), transport=httpx.MockTransport(upstream)
    )
    with pytest.raises(BridgeRefusal) as account:
        await client.fetch_avatar(
            "acc_2", v="0123456789abcdef", size=64, expected_account_id="acc_1"
        )
    with pytest.raises(BridgeRefusal) as endpoint:
        await client.fetch_endpoint_avatar(
            "bep_1", v="0123456789abcdef", size=64, expected_account_id="acc_1"
        )
    assert account.value.http_status == endpoint.value.http_status == 502
