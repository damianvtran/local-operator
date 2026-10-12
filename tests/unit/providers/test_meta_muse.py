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


def test_compensating_logout_cannot_resurrect_concurrent_success(tmp_path: Path, monkeypatch):
    import sqlite3
    import threading

    from local_operator.secrets.access import open_store
    from local_operator.secrets.store import SecretStore

    first = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    second = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    first_attempt_done, retry_allowed = threading.Event(), threading.Event()
    result: list[str] = []
    errors: list[BaseException] = []
    worker = None
    try:
        row = seed(first)
        second._conn.execute("PRAGMA busy_timeout = 0")
        original_set = SecretStore.set

        def other_logout():
            try:
                try:
                    second.delete_credential(row.id)
                    result.append("success before compensation")
                except sqlite3.OperationalError as error:
                    if "locked" not in str(error):
                        raise
                    result.append("write reservation held")
                    first_attempt_done.set()
                    assert retry_allowed.wait(10), "first logout did not release its reservation"
                    second.delete_credential(row.id)
                    result.append("success after compensation")
            except BaseException as error:
                errors.append(error)
            finally:
                first_attempt_done.set()

        def interleave_set(store, *args, **kwargs):
            nonlocal worker
            # Run the second AUTH connection after ownership was checked and
            # immediately before the compensating secret write. No sleep race:
            # timeout=0 tells us deterministically whether SQLite serializes it.
            worker = threading.Thread(target=other_logout, daemon=True)
            worker.start()
            assert first_attempt_done.wait(10), "second connection did not attempt deletion"
            return original_set(store, *args, **kwargs)

        def fail_commit():
            raise sqlite3.OperationalError("fixture first logout commit failure")

        monkeypatch.setattr(first._conn, "commit", fail_commit)
        monkeypatch.setattr(SecretStore, "set", interleave_set)
        with pytest.raises(sqlite3.OperationalError, match="fixture first logout commit failure"):
            first.delete_credential(row.id)
        retry_allowed.set()
        assert worker is not None
        worker.join(10)
        assert not worker.is_alive() and not errors
        assert first.get_credential(row.id) is None
        assert open_store(tmp_path).list() == []
        assert result == ["write reservation held", "success after compensation"]
    finally:
        retry_allowed.set()
        if worker is not None:
            worker.join(10)
        second.close()
        first.close()


def test_logout_commit_observer_failure_does_not_restore_durable_deletion(
    tmp_path: Path, monkeypatch
):
    import sqlite3

    from local_operator.secrets.access import open_store

    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        row = seed(store)
        commit = store._conn.commit

        def committed_then_failed():
            commit()
            raise sqlite3.OperationalError("fixture commit observer failure")

        monkeypatch.setattr(store._conn, "commit", committed_then_failed)
        with pytest.raises(sqlite3.OperationalError, match="fixture commit observer failure"):
            store.delete_credential(row.id)
        assert store.get_credential(row.id) is None
        assert open_store(tmp_path).list() == []
        assert not store._conn.in_transaction
    finally:
        store.close()


@pytest.mark.parametrize("fault", ["interrupt", "raise-rollback"])
def test_native_sqlite_transaction_abort_retains_account_key(tmp_path: Path, fault):
    import sqlite3

    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    native = store._conn._conn
    try:
        row = seed(store)
        armed = [False]
        if fault == "interrupt":

            def progress():
                if armed[0]:
                    armed[0] = False
                    return 1
                return 0

            def trace(sql):
                if sql.startswith("DELETE FROM auth_credentials "):
                    armed[0] = True

            native.set_progress_handler(progress, 1)
            native.set_trace_callback(trace)
        else:
            native.execute(
                "CREATE TRIGGER rollback_logout BEFORE DELETE ON auth_credentials "
                "BEGIN SELECT RAISE(ROLLBACK, 'fixture native rollback'); END"
            )
            native.commit()
        with pytest.raises(sqlite3.Error) as caught:
            store.delete_credential(row.id)
        native.set_progress_handler(None, 0)
        native.set_trace_callback(None)
        assert caught.value.sqlite_errorname == (
            "SQLITE_INTERRUPT" if fault == "interrupt" else "SQLITE_CONSTRAINT_TRIGGER"
        )
        restored = store.get_credential(row.id)
        assert restored is not None
        assert store._row_matches_key(restored, KEY)
        assert not store._conn.in_transaction
    finally:
        native.set_progress_handler(None, 0)
        native.set_trace_callback(None)
        store.close()


@pytest.mark.parametrize(
    "other_writer", ["same-ref-metadata", "replacement-login", "successful-logout"]
)
def test_transaction_loss_rechecks_immutable_owner_under_reacquired_reservation(
    tmp_path: Path, monkeypatch, other_writer
):
    import json
    import sqlite3

    from local_operator.secrets.access import open_store

    first = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    second = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        row = seed(first)
        execute = first._conn.execute
        interleaved = []
        new_key = KEY + "_replacement"

        def aborted_commit():
            first._conn.rollback()
            raise sqlite3.OperationalError("fixture transaction aborted")

        def interleave_before_reacquire(sql, *args, **kwargs):
            if sql == "BEGIN IMMEDIATE" and not interleaved:
                assert not first._conn.in_transaction
                interleaved.append(other_writer)
                if other_writer == "same-ref-metadata":
                    second._conn.execute(
                        "UPDATE auth_credentials SET data = ? WHERE id = ?",
                        (json.dumps({**row.data, "observed_metadata": "updated"}), row.id),
                    )
                    second._conn.commit()
                elif other_writer == "replacement-login":
                    seed(second, key=new_key)
                else:
                    second.delete_credential(row.id)
            return execute(sql, *args, **kwargs)

        monkeypatch.setattr(first._conn, "commit", aborted_commit)
        monkeypatch.setattr(first._conn, "execute", interleave_before_reacquire)
        with pytest.raises(sqlite3.OperationalError, match="fixture transaction aborted"):
            first.delete_credential(row.id)
        assert interleaved == [other_writer]
        current = first.get_credential(row.id)
        if other_writer == "successful-logout":
            assert current is None and open_store(tmp_path).list() == []
        else:
            assert current is not None
            expected = KEY if other_writer == "same-ref-metadata" else new_key
            assert first._row_matches_key(current, expected)
            assert len(open_store(tmp_path).list()) == 1
            if other_writer == "same-ref-metadata":
                assert current.data["observed_metadata"] == "updated"
                assert current.data["access_secret_ref"] == row.data["access_secret_ref"]
            else:
                assert current.data["access_secret_ref"] != row.data["access_secret_ref"]
        assert not first._conn.in_transaction
    finally:
        second.close()
        first.close()


def test_lost_savepoint_normalizes_rollback_and_reacquires(tmp_path: Path, monkeypatch):
    import sqlite3

    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        row = seed(store)
        execute = store._conn.execute
        normalized = []

        def fail_commit():
            raise sqlite3.OperationalError("fixture precommit failure")

        def lost_savepoint(sql, *args, **kwargs):
            if sql == "ROLLBACK TO SAVEPOINT muse_logout":
                # Cause the actual SQLite savepoint lookup error, not a fake
                # transaction flag or silently ignored rollback exception.
                execute("RELEASE SAVEPOINT muse_logout")
            if sql == "BEGIN IMMEDIATE":
                normalized.append(not store._conn.in_transaction)
            return execute(sql, *args, **kwargs)

        monkeypatch.setattr(store._conn, "commit", fail_commit)
        monkeypatch.setattr(store._conn, "execute", lost_savepoint)
        with pytest.raises(sqlite3.OperationalError, match="fixture precommit failure"):
            store.delete_credential(row.id)
        assert normalized == [True]
        current = store.get_credential(row.id)
        assert current is not None and store._row_matches_key(current, KEY)
        assert not store._conn.in_transaction
    finally:
        store.close()


@pytest.mark.parametrize("failure", ["reacquire", "sealed-restore"])
def test_incomplete_recovery_is_explicit_and_never_restores_unlocked(
    tmp_path: Path, monkeypatch, failure
):
    import sqlite3

    from local_operator.providers.auth_store import AuthStoreError
    from local_operator.secrets.errors import SecretStoreError
    from local_operator.secrets.store import SecretStore

    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        row = seed(store)
        execute, get_credential = store._conn.execute, store.get_credential
        reads = []

        def aborted_commit():
            store._conn.rollback()
            raise sqlite3.OperationalError("fixture transaction aborted")

        def read(ident):
            reads.append(ident)
            return get_credential(ident)

        def unavailable_reservation(sql, *args, **kwargs):
            if sql == "BEGIN IMMEDIATE":
                raise sqlite3.OperationalError("fixture reservation unavailable")
            return execute(sql, *args, **kwargs)

        def failed_set(*_args, **_kwargs):
            assert store._conn.in_transaction
            raise SecretStoreError("fixture restoration unavailable")

        with monkeypatch.context() as faults:
            faults.setattr(store._conn, "commit", aborted_commit)
            faults.setattr(store, "get_credential", read)
            if failure == "reacquire":
                faults.setattr(store._conn, "execute", unavailable_reservation)
            else:
                faults.setattr(SecretStore, "set", failed_set)
            with pytest.raises((AuthStoreError, SecretStoreError), match="compensation incomplete"):
                store.delete_credential(row.id)
        assert len(reads) == (2 if failure == "reacquire" else 3)
        assert store.get_credential(row.id) is not None
        assert not store._row_matches_key(row, KEY)
        assert not store._conn.in_transaction
    finally:
        store.close()
