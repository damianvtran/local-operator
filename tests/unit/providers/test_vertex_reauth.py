"""Google Cloud session control (``invalid_rapt``) on a Google Vertex sign-in.

A Google Workspace with Google Cloud session control forces a fresh sign-in
every 1-24 h; past that window Google refuses the refresh with HTTP 400
``invalid_grant`` / ``error_subtype: invalid_rapt``. These tests replay that
exact body through the real refresh, the auth store and the failover walk, and
pin the defect that started this: the turn reported ``provider failure: quota``
for what is a sign-in problem.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from local_operator.harness.types import ChatRequest, ModelSpec
from local_operator.providers import failover as failover_module
from local_operator.providers.auth_cli import format_logins
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.failover import (
    ProviderError,
    dead_grant_reason,
    settle_reason_for_target,
    stream_with_failover,
)
from local_operator.providers.oauth import vertex
from local_operator.providers.oauth.callback_server import (
    REAUTH_REASON_SESSION_POLICY,
    ConfigurationError,
    InvalidGrantError,
    ReauthRequiredError,
    RefreshNotSentError,
    is_reauth_policy_response,
)

#: The body Google's token endpoint returned, byte for byte.
INVALID_RAPT_BODY = json.dumps(
    {
        "error": "invalid_grant",
        "error_description": "reauth related error (invalid_rapt)",
        "error_uri": "https://support.google.com/a/answer/9368756",
        "error_subtype": "invalid_rapt",
    }
)


@pytest.fixture(autouse=True)
def _no_client_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the operator's env/store/config out of the client-id resolution."""
    for name in (vertex.CLIENT_ID_ENV, vertex.CLIENT_SECRET_ENV, "GOOGLE_VERTEX_TOKEN_URI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(vertex, "_stored_value", lambda _name: None)
    monkeypatch.setattr(vertex, "_configured_client_id", lambda: None)


def _rapt_transport(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(400, text=INVALID_RAPT_BODY)

    return httpx.MockTransport(handler)


def _route_refreshes_to(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport) -> None:
    """Make the refresh's own ``httpx.AsyncClient(...)`` use ``transport``.

    The store calls ``refresh_vertex_token`` without an injected client (the
    production path), so the constructor is what has to be redirected.
    """
    real = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real(*args, **kwargs)

    monkeypatch.setattr(vertex.httpx, "AsyncClient", factory)


def _seed_expired_vertex(store: AuthStore) -> int:
    row = store.upsert_credential(
        "google-vertex",
        {
            "type": "oauth",
            "auth_mode": "oauth",
            "access": "stale-access",
            "refresh": "still-valid-refresh",
            "expires": 1,  # long past: the next use must refresh
            "email": "user@example.com",
            "project_id": "proj-123",
            "location": "global",
        },
    )
    return row.id


def test_the_exact_google_body_is_recognised_and_plain_invalid_grant_is_not() -> None:
    assert is_reauth_policy_response(INVALID_RAPT_BODY)
    # Description-only spelling (no subtype field) is still Google's refusal.
    assert is_reauth_policy_response(
        json.dumps(
            {"error": "invalid_grant", "error_description": "reauth related error (invalid_rapt)"}
        )
    )
    assert not is_reauth_policy_response(
        json.dumps(
            {"error": "invalid_grant", "error_description": "Token has been expired or revoked."}
        )
    )
    # RAPT named under a different verdict is not promoted to a dead grant.
    assert not is_reauth_policy_response(
        json.dumps({"error": "invalid_request", "error_subtype": "invalid_rapt"})
    )
    assert not is_reauth_policy_response("not json invalid_rapt")


@pytest.mark.asyncio
async def test_refresh_raises_reauth_required_for_invalid_rapt() -> None:
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_rapt_transport(seen)) as http:
        with pytest.raises(ReauthRequiredError) as excinfo:
            await vertex.refresh_vertex_token(
                {"type": "oauth", "auth_mode": "oauth", "refresh": "r"}, http_client=http
            )
    # Still a dead grant for every existing handler.
    assert isinstance(excinfo.value, InvalidGrantError)
    assert excinfo.value.reason == REAUTH_REASON_SESSION_POLICY
    assert "session control" in str(excinfo.value)
    assert len(seen) == 1
    assert (
        dict(httpx.QueryParams(seen[0].content.decode()))["client_id"] == vertex.DEFAULT_CLIENT_ID
    )


@pytest.mark.asyncio
async def test_stream_reports_a_sign_in_not_quota_and_records_the_reason(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reported defect end to end: refresh refused -> walk -> the error.

    Before the fix the walk ended on ``kind="quota"`` ("rate limited, a token
    refresh failed ...; retry once the limit resets"), which the fallback
    notice rendered as ``provider failure: quota``.
    """
    seen: list[httpx.Request] = []
    _route_refreshes_to(monkeypatch, _rapt_transport(seen))
    store = AuthStore(db_path=tmp_path / "auth.db")
    row_id = _seed_expired_vertex(store)

    def unreachable(
        request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        raise AssertionError("no model request may be sent without a bearer")

    class _Client:
        async def stream(
            self, request: ChatRequest, api_key: str | None, oauth_access: Any = None
        ) -> AsyncIterator[Any]:
            async for event in unreachable(request, api_key, oauth_access):
                yield event

    async def client_for(spec: ModelSpec) -> Any:
        return _Client()

    async def no_sleep(delay_ms: int, signal: Any) -> None:
        return None

    monkeypatch.setattr(failover_module, "_abortable_sleep", no_sleep)
    request = ChatRequest(model=ModelSpec(provider="google-vertex", model_id="gemini-test"))
    with pytest.raises(ProviderError) as excinfo:
        async for _ in stream_with_failover(
            request, store, {"retry": {"enabled": True, "baseDelayMs": 1}}, client_for
        ):
            pass

    assert seen, "the refresh POST must have been made"
    error = excinfo.value
    assert error.kind == "auth", (error.kind, error.message)
    assert "quota" not in error.message.lower()
    assert "rate limit" not in error.message.lower()
    assert "/login google-vertex" in error.message
    assert "lop login google-vertex" in error.message
    assert "session control" in error.message

    assert store.grant_is_dead(row_id)
    assert store.grant_dead_reason(row_id) == REAUTH_REASON_SESSION_POLICY
    assert dead_grant_reason(store, "google-vertex") == (True, REAUTH_REASON_SESSION_POLICY)

    # The fallback notice's reason now says auth, not quota.
    state = failover_module.FailoverRouteState()
    reason = settle_reason_for_target(
        state, failover_module.FallbackTarget("anthropic/claude-test"), error
    )
    assert reason == "provider failure: auth"

    # `lop login-status` names the sign-in, not just the access-token clock.
    listing = "\n".join(format_logins(store))
    assert "google-vertex" in listing
    assert "sign-in required (Workspace policy)" in listing
    store.close()


def test_a_fresh_login_clears_the_reason_with_the_tombstone(tmp_path: Any) -> None:
    store = AuthStore(db_path=tmp_path / "auth.db")
    row_id = _seed_expired_vertex(store)
    assert store._mark_grant_dead(
        row_id, rejected_refresh_token="still-valid-refresh", reason=REAUTH_REASON_SESSION_POLICY
    )
    assert store.grant_dead_reason(row_id) == REAUTH_REASON_SESSION_POLICY
    fresh = _seed_expired_vertex(store)
    assert not store.grant_is_dead(fresh)
    assert store.grant_dead_reason(fresh) is None
    assert "sign-in required" not in "\n".join(format_logins(store))
    store.close()


def test_a_live_sibling_keeps_the_provider_out_of_the_sign_in_verdict(tmp_path: Any) -> None:
    store = AuthStore(db_path=tmp_path / "auth.db")
    row_id = _seed_expired_vertex(store)
    store._mark_grant_dead(row_id, rejected_refresh_token="still-valid-refresh")
    assert dead_grant_reason(store, "google-vertex") == (True, None)
    store.upsert_credential(
        "google-vertex",
        {
            "type": "oauth",
            "access": "a2",
            "refresh": "r2",
            "expires": None,
            "email": "b@example.com",
        },
    )
    assert dead_grant_reason(store, "google-vertex") == (False, None)
    store.close()


def test_client_id_precedence_env_store_config_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert vertex._client_id() == vertex.DEFAULT_CLIENT_ID
    monkeypatch.setattr(vertex, "_configured_client_id", lambda: "cfg.apps.googleusercontent.com")
    assert vertex._client_id() == "cfg.apps.googleusercontent.com"
    stored = {vertex.CLIENT_ID_ENV: "store.apps.googleusercontent.com"}
    monkeypatch.setattr(vertex, "_stored_value", stored.get)
    assert vertex._client_id() == "store.apps.googleusercontent.com"
    monkeypatch.setenv(vertex.CLIENT_ID_ENV, "env.apps.googleusercontent.com")
    assert vertex._client_id() == "env.apps.googleusercontent.com"


def test_custom_client_without_secret_is_refused_and_never_gets_gclouds_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert vertex._client_secret(vertex.DEFAULT_CLIENT_ID) == vertex.DEFAULT_CLIENT_SECRET
    with pytest.raises(ConfigurationError):
        vertex._client_secret("org.apps.googleusercontent.com")
    monkeypatch.setattr(vertex, "_stored_value", {vertex.CLIENT_SECRET_ENV: "org-secret"}.get)
    assert vertex._client_secret("org.apps.googleusercontent.com") == "org-secret"


def test_configured_client_id_reads_config_yml(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real reader (the autouse fixture stubs it; this test restores it)."""
    monkeypatch.undo()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text(
        "values:\n  providers:\n    google-vertex:\n"
        "      oauth_client_id: org.apps.googleusercontent.com\n"
    )
    assert vertex._configured_client_id() == "org.apps.googleusercontent.com"


@pytest.mark.asyncio
async def test_refresh_presents_the_client_that_minted_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored custom client id wins over whatever is configured now."""
    monkeypatch.setattr(vertex, "_stored_value", {vertex.CLIENT_SECRET_ENV: "org-secret"}.get)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"access_token": "new", "expires_in": 3600})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        refreshed = await vertex.refresh_vertex_token(
            {
                "type": "oauth",
                "auth_mode": "oauth",
                "refresh": "r",
                "client_id": "org.apps.googleusercontent.com",
            },
            http_client=http,
        )
    form = dict(httpx.QueryParams(seen[0].content.decode()))
    assert form["client_id"] == "org.apps.googleusercontent.com"
    assert form["client_secret"] == "org-secret"
    assert refreshed["client_id"] == "org.apps.googleusercontent.com"


@pytest.mark.asyncio
async def test_refresh_with_a_custom_client_and_no_secret_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_rapt_transport(seen)) as http:
        with pytest.raises(RefreshNotSentError):
            await vertex.refresh_vertex_token(
                {"type": "oauth", "refresh": "r", "client_id": "org.apps.googleusercontent.com"},
                http_client=http,
            )
    assert seen == []
