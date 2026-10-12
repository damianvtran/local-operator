"""Fixture protocol evidence, not a claim of live Meta eligibility or billing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from local_operator.harness.types import AbortSignal
from local_operator.providers.auth_store import AuthStore, CredentialInvalidError
from local_operator.providers.meta_muse_credentials import REF_FIELD, delete_key
from local_operator.providers.oauth import meta_muse
from local_operator.providers.oauth.callback_server import (
    LoginCallbacks,
    LoginCancelledError,
    LoginError,
)
from local_operator.providers.registry import get_provider_definition

KEY = "LLM|1234|fixture_subscription_key"
ACCOUNT_TOKEN = "fixture_account_bearer_no_live_access"


@pytest.fixture(autouse=True)
def _never_launch_a_real_browser(monkeypatch):
    # Successful device fixtures exercise the opener, never the operator's desk.
    monkeypatch.setattr("webbrowser.open", lambda *_args, **_kwargs: True)


def fixture_transport(mint=None, *, token=None, authorization=None):
    requests = []

    def handle(request):
        requests.append(request)
        if request.url == meta_muse.AUTHORIZATION_URL:
            return httpx.Response(
                200,
                json=authorization
                or {
                    "device_code": "fixture_device_private",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://auth.meta.com/oauth/device/",
                    "interval": 1,
                    "expires_in": 60,
                },
            )
        if request.url == meta_muse.TOKEN_URL:
            return httpx.Response(
                200, json=token if token is not None else {"access_token": ACCOUNT_TOKEN}
            )
        assert request.url == meta_muse.KEY_URL
        assert request.headers["authorization"] == f"Bearer {ACCOUNT_TOKEN}"
        assert json.loads(request.content) == {"onboard": True}
        return httpx.Response(
            200,
            json=(
                mint
                if mint is not None
                else {
                    "api_key": KEY,
                    "user_id": "1234",
                    "user_email": "Fixture@Example.test",
                    "is_subs_active": True,
                    "require_payment": False,
                }
            ),
        )

    return httpx.MockTransport(handle), requests


@pytest.mark.asyncio
async def test_device_mints_distinct_key_and_reports_safe_operation_details():
    urls, notices, details = [], [], []
    transport, requests = fixture_transport()
    async with httpx.AsyncClient(transport=transport) as http:
        result = await meta_muse.login_meta_muse(
            LoginCallbacks(
                on_auth_url=lambda url, instructions=None: urls.append((url, instructions)),
                on_warning=notices.append,
                displays_warnings=True,
                on_flow_details=lambda **value: details.append(value),
            ),
            http_client=http,
        )
    assert result["access"] == KEY
    assert result["email"] == "fixture@example.test"
    assert result["source"] == "device-approval"
    assert result["type"] == "oauth"
    assert ACCOUNT_TOKEN not in json.dumps(result)
    assert "fixture_device_private" not in str(urls + notices + details)
    assert details[0]["user_code"] == "ABCD-EFGH"
    assert "unsupported by Meta" in notices[0]
    assert len(requests) == 3
    assert all(request.headers["x-api-version"] == "1.0.0" for request in requests)
    assert "client_id=1031625952748946" in requests[0].content.decode()
    assert (
        "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code"
        in requests[1].content.decode()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"is_subs_active": False},
        {"is_subs_active": "true"},
        {"require_payment": True, "is_subs_active": True},
    ],
)
async def test_entitlement_missing_or_payment_required_fails_closed(fields):
    transport, _ = fixture_transport({"api_key": KEY, "user_id": "1234", **fields})
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(LoginError, match="did not confirm") as error:
            await meta_muse.login_meta_muse(LoginCallbacks(), http_client=http)
    assert KEY not in str(error.value)
    assert ACCOUNT_TOKEN not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token",
    [
        {"error": "access_denied", "access_token": ACCOUNT_TOKEN},
        {"error": "expired_token"},
        {"error": "unknown", "secret": ACCOUNT_TOKEN},
        {},
    ],
)
async def test_token_errors_are_terminal_even_on_http_200(token):
    transport, requests = fixture_transport(token=token or {"unexpected": ACCOUNT_TOKEN})
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(LoginError) as error:
            await meta_muse.login_meta_muse(LoginCallbacks(), http_client=http)
    assert len(requests) == 2
    assert ACCOUNT_TOKEN not in str(error.value)


@pytest.mark.asyncio
async def test_untrusted_approval_url_never_reaches_host():
    transport, _ = fixture_transport(
        authorization={
            "device_code": "private",
            "user_code": "public",
            "verification_uri": "https://evil.example/",
        }
    )
    urls = []
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(LoginError, match="invalid device approval URL"):
            await meta_muse.login_meta_muse(
                LoginCallbacks(on_auth_url=urls.append), http_client=http
            )
    assert not urls


@pytest.mark.asyncio
async def test_cancel_reaps_stalled_network():
    started, stopped = asyncio.Event(), asyncio.Event()
    signal = AbortSignal()

    async def stalled(_request):
        started.set()
        try:
            await asyncio.sleep(100)
        finally:
            stopped.set()
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(stalled)) as http:
        task = asyncio.create_task(
            meta_muse.login_meta_muse(LoginCallbacks(), signal=signal, http_client=http)
        )
        await started.wait()
        signal.abort("private device code must not reach errors")
        with pytest.raises(LoginCancelledError, match="Muse Code login cancelled"):
            await asyncio.wait_for(task, 1)
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_request_timeout_reaps_network(monkeypatch):
    stopped = asyncio.Event()
    monkeypatch.setattr(meta_muse, "REQUEST_TIMEOUT_S", 0.01)

    async def stalled(_request):
        try:
            await asyncio.sleep(100)
        finally:
            stopped.set()
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(stalled)) as http:
        # Bound is injected explicitly because the function's default budget
        # is fixed at definition time; no real endpoint is contacted.
        with pytest.raises(LoginError, match="timed out"):
            await meta_muse._request(http, meta_muse.TOKEN_URL, signal=None, budget=0.01)
    assert stopped.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(302, headers={"location": "https://attacker.invalid/"}),
        httpx.Response(200, content=b"x" * (meta_muse.MAX_BODY_BYTES + 1)),
        httpx.Response(200, content=b"not-json"),
        httpx.Response(500, json={"api_key": KEY, "access_token": ACCOUNT_TOKEN}),
    ],
)
async def test_invalid_authorization_responses_never_leak_or_redirect(response):
    calls = []

    def handle(request):
        calls.append(request)
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        with pytest.raises(LoginError) as error:
            await meta_muse.login_meta_muse(LoginCallbacks(), http_client=http)
    assert len(calls) == 1
    assert KEY not in str(error.value) and ACCOUNT_TOKEN not in str(error.value)


def test_no_ambient_key_or_generic_paste_route():
    definition = get_provider_definition("meta-muse")
    assert definition is not None
    assert definition.name == "Muse Code"
    from local_operator.providers.login_catalog import picker_label

    assert picker_label("meta-muse") == "Muse Code"
    assert definition.env_keys is None
    assert definition.legacy_store_keys == ()
    assert not definition.accepts_api_key
    assert definition.login_kind == "device"


def seed(store, account="1234", key=KEY):
    return store.upsert_credential(
        "meta-muse",
        {
            "type": "oauth",
            "access": key,
            "account_id": account,
            "source": "device-approval",
            "billing_provenance": "muse-code-subscription",
            "account_access_token": ACCOUNT_TOKEN,
        },
    )


@pytest.mark.asyncio
async def test_encrypted_ref_roundtrip_account_isolation_and_logout(tmp_path: Path):
    root = tmp_path / "config"
    store = AuthStore(root / "auth.db", config_dir=root)
    other = AuthStore(tmp_path / "other" / "auth.db", config_dir=tmp_path / "other")
    try:
        row = seed(store)
        sibling = seed(store, "5678", "LLM|5678|other_fixture_key")
        unrelated = store.upsert_credential("deepseek", {"key": "unrelated", "type": "api_key"})
        assert "access" not in row.data
        assert "account_access_token" not in row.data
        assert REF_FIELD in row.data
        assert not AuthStore._needs_refresh(row.data)
        assert store._row_matches_key(row, KEY)
        assert await store.has_persisted_credential("meta-muse")
        raw = (root / "auth.db").read_bytes()
        assert KEY.encode() not in raw and ACCOUNT_TOKEN.encode() not in raw
        other_extractor = other._oauth_key_fn("meta-muse")
        assert other_extractor is not None
        assert other_extractor(row.data) == ""
        with pytest.raises(CredentialInvalidError, match="new sign-in"):
            await store._ensure_oauth_fresh(row, force=True)
        store.delete_credential(row.id)
        extractor = store._oauth_key_fn("meta-muse")
        assert extractor is not None
        assert extractor(row.data) == ""
        assert store.get_credential(sibling.id) is not None
        assert store.get_credential(unrelated.id) is not None
        store.delete_credentials_for_provider("meta-muse")
        assert not await store.has_persisted_credential("meta-muse")
        assert store.get_credential(unrelated.id) is not None
    finally:
        other.close()
        store.close()


@pytest.mark.asyncio
async def test_deleted_secret_fails_availability(tmp_path: Path):
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        row = seed(store)
        delete_key(row.data, tmp_path)
        assert not await store.has_persisted_credential("meta-muse")
        assert await store.get_api_key("meta-muse") is None
    finally:
        store.close()


def test_sql_failure_rolls_back_new_sealed_key(tmp_path: Path, monkeypatch):
    from local_operator.secrets.access import open_store

    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        existing = seed(store)
        names = {record.name for record in open_store(tmp_path).list()}

        def fail(*_args, **_kwargs):
            raise RuntimeError("fixture sql failure")

        monkeypatch.setattr(store, "_upsert_credential_metadata", fail)
        with pytest.raises(RuntimeError, match="fixture sql failure"):
            seed(store)
        assert {record.name for record in open_store(tmp_path).list()} == names
        assert store._row_matches_key(existing, KEY)
    finally:
        store.close()


@pytest.mark.parametrize("failure", ["execute", "commit", "locked-store"])
def test_failed_logout_preserves_metadata_and_sealed_key(tmp_path: Path, monkeypatch, failure):
    import sqlite3

    from local_operator.secrets.errors import BrokerUnavailable

    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        row = seed(store)
        sibling = seed(store, "5678", "LLM|5678|fixture_sibling_key")
        if failure == "execute":
            store._conn.execute(
                "CREATE TRIGGER fail_muse_logout BEFORE DELETE ON auth_credentials "
                "BEGIN SELECT RAISE(ABORT, 'fixture metadata failure'); END"
            )
            store._conn.commit()
        with monkeypatch.context() as faults:
            if failure == "commit":

                def fail_commit():
                    raise sqlite3.OperationalError("fixture commit failure")

                faults.setattr(store._conn, "commit", fail_commit)
            elif failure == "locked-store":

                def locked(*_args, **_kwargs):
                    raise BrokerUnavailable("fixture locked store")

                faults.setattr("local_operator.secrets.access.open_store", locked)
            with pytest.raises((sqlite3.Error, BrokerUnavailable)):
                store.delete_credential(row.id)
        restored = store.get_credential(row.id)
        assert restored is not None and restored.data == row.data
        assert store._row_matches_key(restored, KEY)
        assert store.get_credential(sibling.id) is not None
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("displayed", [False, True])
async def test_disclosure_capability_and_injected_browser_ownership(displayed, monkeypatch):
    warnings, urls, injected, global_opens = [], [], [], []
    monkeypatch.setattr("webbrowser.open", lambda url: global_opens.append(url))
    transport, _ = fixture_transport()
    async with httpx.AsyncClient(transport=transport) as http:
        await meta_muse.login_meta_muse(
            LoginCallbacks(
                on_warning=warnings.append,
                on_auth_url=lambda url, instructions=None: urls.append((url, instructions)),
                displays_warnings=displayed,
            ),
            http_client=http,
            open_browser=lambda url: injected.append(url),
        )
    assert injected == [urls[0][0]] and global_opens == []
    assert warnings == ([meta_muse.NOTICE] if displayed else [])
    assert (meta_muse.NOTICE in urls[0][1]) is (not displayed)
    assert "Enter code: ABCD-EFGH" in urls[0][1]


@pytest.mark.asyncio
async def test_default_local_opener_and_visible_failure_remedy(monkeypatch):
    opened, warnings = [], []

    def fail_open(url):
        opened.append(url)
        return False

    monkeypatch.setattr("webbrowser.open", fail_open)
    transport, _ = fixture_transport()
    async with httpx.AsyncClient(transport=transport) as http:
        await meta_muse.login_meta_muse(
            LoginCallbacks(on_warning=warnings.append, displays_warnings=True), http_client=http
        )
    assert opened == ["https://auth.meta.com/oauth/device/"]
    assert warnings[0] == meta_muse.NOTICE
    assert "displayed approval page" in warnings[1]
