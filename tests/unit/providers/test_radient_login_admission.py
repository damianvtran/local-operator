"""Real flow/SQLite completion shapes; HTTP is synthetic, never a live login."""

import pickle
from copy import copy, deepcopy
from dataclasses import replace
from typing import Any

import httpx
import pytest

from local_operator.env import DEFAULT_RADIENT_API_BASE_URL
from local_operator.harness.types import AbortSignal
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.oauth import radient
from local_operator.providers.oauth.callback_server import (
    LoginCancelledError,
    LoginError,
)
from local_operator.team_bridge.client import TeamBridgeClient
from tests.unit.providers import test_radient_account_retention as _retention
from tests.unit.providers.test_radient_account_retention import (
    _login,
    _stored,
    _verified_login_result,
)

store = _retention.store


@pytest.mark.asyncio
async def test_verified_profile_not_jwt_label_admits_default_after_completion(
    store: AuthStore,
) -> None:
    old = store.upsert_credential(
        "radient",
        {
            "type": "oauth",
            "access": "legacy-test-only",
            "refresh": "legacy-test-only",
            "expires": 9999999999999,
        },
    )
    result = await _verified_login_result("verified-account")
    # No admission or persistence occurs during exchange/profile verification.
    assert len(store.list_credentials("radient")) == 1
    assert _stored(store, old.id).disabled_cause is None
    row = store.complete_radient_login(result)
    assert row.data["account_id"] == "verified-account"
    assert row.data["api_endpoint"] == DEFAULT_RADIENT_API_BASE_URL
    assert _stored(store, old.id).disabled_cause == "identity_unattributed"
    default = await store.get_oauth_access("radient")
    assert default is not None and default.credential_id == row.id
    client = TeamBridgeClient(store=store, base_url=DEFAULT_RADIENT_API_BASE_URL)
    assert (await client.current_access()).account_id == "verified-account"
    with pytest.raises(LoginError):
        store.complete_radient_login(result)


@pytest.mark.asyncio
async def test_serialized_or_mutated_exchange_result_is_not_admission(store: AuthStore) -> None:
    result = await _verified_login_result()
    with pytest.raises(TypeError):
        pickle.dumps(result)
    for clone in (dict(result), copy(result), deepcopy(result)):
        with pytest.raises(LoginError):
            store.complete_radient_login(clone)
    result["account_id"] = "forged-label"
    with pytest.raises(LoginError):
        store.complete_radient_login(result)
    assert store.list_credentials("radient") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,body",
    [
        (401, {}),
        (404, {}),
        (302, {}),
        (200, {}),
        (200, {"result": []}),
        (200, {"result": {"identity": {"account_id": "a"}, "account": {"id": "b"}}}),
        (200, {"result": {"identity": {"account_id": ""}, "account": {"id": ""}}}),
        (200, {"result": {"identity": {"account_id": True}, "account": {"id": True}}}),
    ],
)
async def test_failed_or_malformed_profile_never_returns_admission(
    store: AuthStore,
    status: int,
    body: Any,
) -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"access_token": "new-test-only", "expires_in": 3600})
        assert request.headers["authorization"] == "Bearer new-test-only"
        return httpx.Response(status, json=body, headers={"location": "https://other.example/me"})

    # Even a redirect-following injected client must not forward the bearer.
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        flow = radient.RadientOAuthFlow(http_client=client)
        await flow.generate_auth_url("state", "http://localhost:54549/callback")
        with pytest.raises(LoginError):
            await flow.exchange_token("code", "state", "http://localhost:54549/callback")
    assert len(calls) == 2
    assert all(request.url.host == "api.radienthq.com" for request in calls)
    assert store.list_credentials("radient") == []


@pytest.mark.asyncio
async def test_custom_token_endpoint_has_no_inferred_hub(
    store: AuthStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(radient, "TOKEN_URL", "https://other.example/token")

    async def refuse_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError("unmapped token endpoint reached network")

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse_request)) as client:
        flow = radient.RadientOAuthFlow(http_client=client)
        await flow.generate_auth_url("state", "http://localhost:54549/callback")
        with pytest.raises(LoginError):
            await flow.exchange_token("code", "state", "http://localhost:54549/callback")
    assert store.list_credentials("radient") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["during_token", "during_profile", "after_result"])
async def test_cancelled_profile_or_late_completion_never_persists(
    store: AuthStore, when: str
) -> None:
    signal = AbortSignal()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            if when == "during_token":
                signal.abort("abandoned exchange")
            return httpx.Response(200, json={"access_token": "new-test-only", "expires_in": 3600})
        if when == "during_profile":
            signal.abort("abandoned login")
        return httpx.Response(
            200,
            json={
                "result": {
                    "identity": {"account_id": "a"},
                    "account": {"id": "a"},
                    "memberships": [],
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        flow = radient.RadientOAuthFlow(http_client=client, signal=signal)
        await flow.generate_auth_url("state", "http://localhost:54549/callback")
        with pytest.raises(LoginCancelledError):
            result = await flow.exchange_token("code", "state", "http://localhost:54549/callback")
            signal.abort("abandoned after verification")
            store.complete_radient_login(result)
    assert store.list_credentials("radient") == []


@pytest.mark.asyncio
async def test_verified_completion_preserves_explicit_disabled_row(store: AuthStore) -> None:
    prior = store.upsert_credential(
        "radient", _login("account-a", endpoint=DEFAULT_RADIENT_API_BASE_URL)
    )
    store.disable_credential(prior.id, "operator_disabled")
    completed = store.complete_radient_login(await _verified_login_result())
    assert completed.id == prior.id
    assert completed.disabled_cause == "operator_disabled"
    assert store.list_credentials("radient") == []


@pytest.mark.asyncio
async def test_expired_receipt_cannot_admit(
    store: AuthStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _verified_login_result()
    now = radient.time.monotonic()
    with monkeypatch.context() as clock:
        clock.setattr(radient.time, "monotonic", lambda: now + 120)
        with pytest.raises(LoginError, match="expired"):
            store.complete_radient_login(result)
    assert store.list_credentials("radient") == []


@pytest.mark.asyncio
async def test_refresh_cannot_switch_account_hub_or_produce_admission(store: AuthStore) -> None:
    issued = await _verified_login_result()

    async def handler(request: httpx.Request) -> httpx.Response:
        # An unverified label in refresh must not replace the /me binding.
        return httpx.Response(
            200,
            json={
                "access_token": "refreshed-test-only",
                "expires_in": 3600,
                "id_token": "eyJhbGciOiJub25lIn0.eyJzdWIiOiJvdGhlci1hY2NvdW50In0.",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        refreshed = await radient.refresh_radient_token(issued, http_client=client)
    assert refreshed["account_id"] == "account-a"
    assert refreshed["api_endpoint"] == DEFAULT_RADIENT_API_BASE_URL
    with pytest.raises(LoginError):
        store.complete_radient_login(refreshed)
    assert store.list_credentials("radient") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["identity_unattributed", "invalidated-token"])
async def test_verified_restore_releases_only_matching_automatic_state(
    store: AuthStore,
    monkeypatch: pytest.MonkeyPatch,
    cause: str,
) -> None:
    known = store.upsert_credential(
        "radient", _login("account-a", endpoint=DEFAULT_RADIENT_API_BASE_URL)
    )
    store.disable_credential(known.id, cause)
    other = store.upsert_credential("radient", _login("other-account"))
    store.disable_credential(other.id, "operator_disabled")
    notifications = []
    monkeypatch.setattr(
        store,
        "_rearm_parked_tunnel_login",
        lambda provider, row: notifications.append((row.id, row.disabled_cause)),
    )
    restored = store.complete_radient_login(await _verified_login_result())
    assert restored.id == known.id and restored.disabled_cause is None
    assert notifications == [(known.id, None)]
    assert _stored(store, other.id).disabled_cause == "operator_disabled"


@pytest.mark.asyncio
async def test_hub_is_captured_before_exchange_not_read_from_later_defaults(
    store: AuthStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.method == "POST":
            monkeypatch.setattr(radient, "TOKEN_URL", "https://other.example/token")
            monkeypatch.setenv("RADIENT_API_BASE_URL", "https://other.example/v1")
            return httpx.Response(200, json={"access_token": "fresh-test-only", "expires_in": 3600})
        assert request.headers["authorization"] == "Bearer fresh-test-only"
        assert request.extensions["timeout"]["read"] == 10.0
        return httpx.Response(
            200,
            json={
                "result": {
                    "identity": {"account_id": "verified"},
                    "account": {"id": "verified"},
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        flow = radient.RadientOAuthFlow(http_client=client)
        await flow.generate_auth_url("state", "http://localhost:54549/callback")
        result = await flow.exchange_token("code", "state", "http://localhost:54549/callback")
    row = store.complete_radient_login(result)
    assert row.data["api_endpoint"] == DEFAULT_RADIENT_API_BASE_URL
    assert requests == [
        DEFAULT_RADIENT_API_BASE_URL + "/auth/oauth/token",
        DEFAULT_RADIENT_API_BASE_URL + "/me",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["issuer", "stored_hub"])
async def test_refresh_refuses_hub_or_issuer_change_before_dispatch(
    store: AuthStore,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    result = await _verified_login_result()
    if change == "issuer":
        monkeypatch.setattr(radient, "TOKEN_URL", "https://other.example/token")
    else:
        result["api_endpoint"] = "https://other.example/v1"

    async def refuse_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError("refresh reached an unbound destination")

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse_request)) as client:
        with pytest.raises(LoginError):
            await radient.refresh_radient_token(result, http_client=client)
    assert store.list_credentials("radient") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["profile_timeout", "malformed_json", "token_redirect"])
async def test_transport_failures_and_token_redirect_never_produce_completion(
    store: AuthStore,
    failure: str,
) -> None:
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            if failure == "token_redirect":
                return httpx.Response(302, headers={"location": "https://other.example/token"})
            return httpx.Response(200, json={"access_token": "fresh-test-only", "expires_in": 3600})
        if failure == "profile_timeout":
            raise httpx.ReadTimeout("synthetic profile timeout")
        return httpx.Response(200, content=b"not-json")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        flow = radient.RadientOAuthFlow(http_client=client)
        await flow.generate_auth_url("state", "http://localhost:54549/callback")
        with pytest.raises(LoginError):
            await flow.exchange_token("code", "state", "http://localhost:54549/callback")
    assert len(requests) == (1 if failure == "token_redirect" else 2)
    assert all(request.url.host == "api.radienthq.com" for request in requests)
    assert store.list_credentials("radient") == []


def _quiet_host(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from local_operator.aida import profile
    from local_operator.providers import auth_cli, controller

    effects: list[str] = []
    monkeypatch.setattr(
        profile, "record_radient_login", lambda *args, **kwargs: effects.append("profile")
    )
    monkeypatch.setattr(
        auth_cli, "_apply_login_defaults", lambda *args, **kwargs: effects.append("defaults")
    )
    monkeypatch.setattr(auth_cli, "_invalidate_cached_listing", lambda *args: None)
    monkeypatch.setattr(auth_cli, "_invalidate_cached_usage", lambda *args: None)
    monkeypatch.setattr(controller, "_invalidate_cached_listing", lambda *args: None)
    monkeypatch.setattr(
        controller.ProviderController, "invalidate_cached_usage", lambda *args: None
    )
    return effects


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["verified", "cancel", "refused", "disabled"])
async def test_controller_completes_real_flow_only_after_final_cancel_check(
    store: AuthStore,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    from local_operator.providers import controller
    from local_operator.providers.oauth.callback_server import LoginCallbacks
    from local_operator.providers.registry import get_provider_definition

    effects = _quiet_host(monkeypatch)
    definition = get_provider_definition("radient")
    assert definition is not None
    signal = AbortSignal()

    async def login(callbacks, *, signal, **kwargs):
        result = await _verified_login_result(
            signal=signal,
            profile_status=401 if outcome == "refused" else 200,
        )
        if outcome == "cancel":
            signal.abort("host abandoned verified result")
        return result

    monkeypatch.setattr(
        controller, "get_provider_definition", lambda _: replace(definition, login=login)
    )
    legacy = store.upsert_credential(
        "radient",
        {
            "type": "oauth",
            "access": "old-test-only",
            "refresh": "old-test-only",
            "expires": 9999999999999,
        },
    )
    if outcome == "disabled":
        disabled = store.upsert_credential(
            "radient", _login("account-a", endpoint=DEFAULT_RADIENT_API_BASE_URL)
        )
        store.disable_credential(disabled.id, "operator_disabled")
    before = store.list_credentials("radient", include_disabled=True)
    host = controller.ProviderController(
        store,
        config_dir=store.db_path.parent,
        login_callbacks=lambda _: LoginCallbacks(),
    )
    if outcome != "verified":
        expected = LoginCancelledError if outcome == "cancel" else LoginError
        with pytest.raises(expected):
            await host.login("radient", signal=signal)
        if outcome == "disabled":
            current = store.list_credentials("radient", include_disabled=True)
            assert {row.id for row in current} == {row.id for row in before}
            assert all(
                row.disabled_cause == "operator_disabled" for row in current if row.id != legacy.id
            )
            assert _stored(store, legacy.id).data == legacy.data
        else:
            assert store.list_credentials("radient", include_disabled=True) == before
        assert effects == []
    else:
        text = await host.login("radient", signal=signal)
        assert "Logged in" in text
        assert "Account could not be verified" in text and f"[{legacy.id}]" in text
        assert "local-operator login radient" in text
        assert (
            store.list_credentials("radient")[0].data["api_endpoint"]
            == DEFAULT_RADIENT_API_BASE_URL
        )


@pytest.mark.parametrize("outcome", ["verified", "cancel", "refused", "disabled"])
def test_cli_completes_real_flow_only_after_final_cancel_check(
    store: AuthStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
) -> None:
    from local_operator.providers import auth_cli
    from local_operator.providers.registry import get_provider_definition

    effects = _quiet_host(monkeypatch)
    definition = get_provider_definition("radient")
    assert definition is not None

    async def login(callbacks, *, signal, **kwargs):
        result = await _verified_login_result(
            signal=signal,
            profile_status=401 if outcome == "refused" else 200,
        )
        if outcome == "cancel":
            signal.abort("host abandoned verified result")
        return result

    monkeypatch.setattr(
        auth_cli, "get_provider_definition", lambda _: replace(definition, login=login)
    )
    legacy = store.upsert_credential(
        "radient",
        {
            "type": "oauth",
            "access": "old-test-only",
            "refresh": "old-test-only",
            "expires": 9999999999999,
        },
    )
    if outcome == "disabled":
        disabled = store.upsert_credential(
            "radient", _login("account-a", endpoint=DEFAULT_RADIENT_API_BASE_URL)
        )
        store.disable_credential(disabled.id, "operator_disabled")
    before = store.list_credentials("radient", include_disabled=True)
    code = auth_cli.run_login("radient", store.db_path.parent, store)
    text = capsys.readouterr().out
    if outcome != "verified":
        assert code == (130 if outcome == "cancel" else 1)
        expected = {
            "cancel": "Login cancelled",
            "refused": "Login failed:",
            "disabled": "remains unavailable",
        }[outcome]
        assert expected in text
        assert "Logged in" not in text
        if outcome == "disabled":
            current = store.list_credentials("radient", include_disabled=True)
            assert {row.id for row in current} == {row.id for row in before}
            assert all(
                row.disabled_cause == "operator_disabled" for row in current if row.id != legacy.id
            )
            assert _stored(store, legacy.id).data == legacy.data
        else:
            assert store.list_credentials("radient", include_disabled=True) == before
        assert effects == []
    else:
        assert code == 0 and "Logged in" in text
        assert "Account could not be verified" in text and f"[{legacy.id}]" in text
        assert "local-operator login radient" in text
        assert (
            store.list_credentials("radient")[0].data["api_endpoint"]
            == DEFAULT_RADIENT_API_BASE_URL
        )
