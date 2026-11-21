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
    sign_in_required_text,
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
    for name in (
        vertex.CLIENT_ID_ENV,
        vertex.CLIENT_SECRET_ENV,
        "GOOGLE_VERTEX_OAUTH_TOKEN_URI",
        "GOOGLE_VERTEX_TOKEN_URI",
    ):
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


#: Unicode isolate controls the display projection inserts (LRI/RLI ... PDI).
_LRI, _RLI, _PDI = "\u2066", "\u2067", "\u2069"


@pytest.mark.parametrize("reason", [REAUTH_REASON_SESSION_POLICY, None])
def test_the_urdu_notice_keeps_each_command_one_ltr_island(reason: str | None) -> None:
    """Translation-audit code MINOR (2026-11-20): in an RTL paragraph the
    leading ``/`` of ``/login`` displayed detached. Each command must be ONE
    LTR island, whole; the error text (``display=False``) stays control-free.
    """
    shown = sign_in_required_text("google-vertex", reason, locale="ur", display=True)
    assert shown.startswith(_RLI) and shown.endswith(_PDI)
    assert f"{_LRI}/login google-vertex{_PDI}" in shown
    assert f"{_LRI}lop login google-vertex{_PDI}" in shown
    assert shown.count(_LRI) == 2

    raw = sign_in_required_text("google-vertex", reason, locale="ur")
    assert not set(raw) & {_LRI, _RLI, _PDI}
    # The projection only adds controls: stripping them restores the raw copy.
    assert shown.translate({ord(c): None for c in (_LRI, _RLI, _PDI)}) == raw


@pytest.mark.parametrize("locale", ["en", "es", "fr", "hi", "ru", "vi", "zh-CN"])
def test_non_urdu_notices_are_painted_byte_identical(locale: str) -> None:
    """Only Urdu-source copy is projected. zh-CN glues the command to prose
    (``使用/login …，``), which the span validator would refuse, so it must
    never be asked to validate that copy."""
    for reason in (REAUTH_REASON_SESSION_POLICY, None):
        shown = sign_in_required_text("google-vertex", reason, locale=locale, display=True)
        assert shown == sign_in_required_text("google-vertex", reason, locale=locale)
        assert "/login google-vertex" in shown


def test_the_deferred_reauth_copy_still_renders_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``wire.providers.reauth`` is out of the eager client bundle; both render
    points read the catalogue in-process, so the user still gets their language."""
    from local_operator.i18n import catalogues, resolve
    from local_operator.providers.auth_cli import _sign_in_required_state

    assert "wire.providers.reauth" in catalogues.DEFERRED_NAMESPACES
    assert "wire.providers.reauth" not in catalogues.wire_bundle("es")["namespaces"]
    text = sign_in_required_text("google-vertex", None, locale="es", display=True)
    assert text.startswith("Tu inicio de sesión en ")
    monkeypatch.setattr(resolve, "resolve_language", lambda: "es")
    assert _sign_in_required_state(REAUTH_REASON_SESSION_POLICY) == (
        "requiere iniciar sesión (política de Workspace)"
    )


# --- Round-1 remediation (PR #4356) ------------------------------------------


def _ok_transport(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"access_token": "new", "expires_in": 3600})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["store", "env"])
async def test_a_gcloud_issued_sign_in_keeps_gclouds_secret_after_an_org_secret_is_set(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """Review M1 / QA Q3: gcloud's id paired with the org's secret is answered
    ``invalid_client`` (terminal), tombstoning a working sign-in."""
    if source == "store":
        monkeypatch.setattr(vertex, "_stored_value", {vertex.CLIENT_SECRET_ENV: "org-secret"}.get)
    else:
        monkeypatch.setenv(vertex.CLIENT_SECRET_ENV, "org-secret")
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_ok_transport(seen)) as http:
        await vertex.refresh_vertex_token({"type": "oauth", "refresh": "r"}, http_client=http)
    form = dict(httpx.QueryParams(seen[0].content.decode()))
    assert (form["client_id"], form["client_secret"]) == (
        vertex.DEFAULT_CLIENT_ID,
        vertex.DEFAULT_CLIENT_SECRET,
    )


@pytest.mark.asyncio
async def test_the_sa_token_uri_no_longer_redirects_the_user_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review m3: ``GOOGLE_VERTEX_TOKEN_URI`` is the SA ``token_uri`` check only."""
    monkeypatch.setenv("GOOGLE_VERTEX_TOKEN_URI", "http://sa.example.test/token")
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_ok_transport(seen)) as http:
        await vertex.refresh_vertex_token({"type": "oauth", "refresh": "r"}, http_client=http)
    assert str(seen[0].url) == vertex.TOKEN_URL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    ["http://127.0.0.1:8765/token", "http://localhost/token", "https://stub.example.test/t"],
)
async def test_the_oauth_refresh_override_accepts_https_or_loopback(
    monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    monkeypatch.setenv("GOOGLE_VERTEX_OAUTH_TOKEN_URI", override)
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_ok_transport(seen)) as http:
        await vertex.refresh_vertex_token({"type": "oauth", "refresh": "r"}, http_client=http)
    assert str(seen[0].url) == override


@pytest.mark.asyncio
@pytest.mark.parametrize("override", ["http://stub.example.test/token", "ftp://127.0.0.1/t"])
async def test_the_oauth_refresh_override_refuses_plain_http_off_loopback(
    monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    monkeypatch.setenv("GOOGLE_VERTEX_OAUTH_TOKEN_URI", override)
    seen: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_ok_transport(seen)) as http:
        with pytest.raises(RefreshNotSentError, match="GOOGLE_VERTEX_OAUTH_TOKEN_URI"):
            await vertex.refresh_vertex_token({"type": "oauth", "refresh": "r"}, http_client=http)
    assert seen == []


@pytest.mark.asyncio
async def test_login_refuses_a_custom_client_without_secret_before_the_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review n1 / QA Q5: refuse before consent, not at the code exchange."""
    monkeypatch.setenv(vertex.CLIENT_ID_ENV, "org.apps.googleusercontent.com")
    opened: list[str] = []
    with pytest.raises(ConfigurationError, match=vertex.CLIENT_SECRET_ENV):
        await vertex.login_vertex_oauth(open_browser=opened.append)
    assert opened == []


@pytest.mark.asyncio
async def test_a_missing_custom_secret_on_refresh_is_reported_as_configuration_not_quota(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA Q4: nothing is sent and nothing is tombstoned, and the walk now names
    the missing secret instead of "retry once the limit resets"."""
    seen: list[httpx.Request] = []
    _route_refreshes_to(monkeypatch, _ok_transport(seen))
    store = AuthStore(db_path=tmp_path / "auth.db")
    row = store.upsert_credential(
        "google-vertex",
        {
            "type": "oauth",
            "auth_mode": "oauth",
            "access": "stale",
            "refresh": "r",
            "expires": 1,
            "client_id": "org.apps.googleusercontent.com",
        },
    )

    class _Client:
        async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            raise AssertionError("no model request may be sent without a bearer")
            yield  # pragma: no cover

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
    error = excinfo.value
    assert seen == []
    assert not store.grant_is_dead(row.id)
    assert error.kind == "auth", (error.kind, error.message)
    assert vertex.CLIENT_SECRET_ENV in error.message
    assert "lop credential update" in error.message
    assert "limit resets" not in error.message
    assert dead_grant_reason(store, "google-vertex") == (False, None)
    store.close()


# --- The one-shot sign-in notice on a live session (review M2, m1; QA Q1) ----


def _dead_vertex_store(tmp_path: Any) -> tuple[AuthStore, int]:
    store = AuthStore(db_path=tmp_path / "auth.db")
    row_id = _seed_expired_vertex(store)
    assert store._mark_grant_dead(
        row_id, rejected_refresh_token="still-valid-refresh", reason=REAUTH_REASON_SESSION_POLICY
    )
    return store, row_id


def _session_for(store: AuthStore, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[str]]:
    from local_operator.i18n import resolve
    from local_operator.model.configure import SessionStreamFn

    monkeypatch.setattr(resolve, "resolve_language", lambda: "en")
    stream = SessionStreamFn(store, {}, "s-reauth")
    stream._primary_selector = "google-vertex/gemini-test"
    notices: list[str] = []

    async def collect(text: str, kind: str) -> None:
        notices.append(text)

    stream.set_notice_handler(collect)
    return stream, notices


def _sign_ins(notices: list[str]) -> list[str]:
    return [text for text in notices if "lop login google-vertex" in text]


@pytest.mark.asyncio
async def test_the_sign_in_notice_re_arms_after_the_primary_recovers(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Die -> recover -> die in ONE session: google-vertex has no usage
    fetcher, so the healthy-usage edge that used to re-arm it never comes."""
    store, row_id = _dead_vertex_store(tmp_path)
    stream, notices = _session_for(store, monkeypatch)
    fallback = failover_module.FallbackTarget("anthropic/claude-test")

    await stream._on_route_change(fallback, "provider failure: auth")
    await stream._on_route_change(fallback, "provider failure: auth")
    assert len(_sign_ins(notices)) == 1  # sticky fallback: said once

    # The user signs in again; the primary serves; the route settles back.
    _seed_expired_vertex(store)
    await stream._on_route_settle(None, "primary model recovered")
    assert "back to google-vertex/gemini-test" in notices

    # Session control expires the new sign-in hours later, same session.
    for row in store.list_credentials("google-vertex"):
        store._mark_grant_dead(
            row.id,
            rejected_refresh_token=row.data["refresh"],
            reason=REAUTH_REASON_SESSION_POLICY,
        )
    await stream._on_route_change(fallback, "provider failure: auth")
    assert len(_sign_ins(notices)) == 2
    store.close()


@pytest.mark.asyncio
async def test_the_sign_in_notice_shows_when_a_later_hop_reports_quota(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review m1: quota outranks auth in the walk's reason, so a 429 from an
    intermediate hop must not hide the dead primary's sign-in."""
    store, _row_id = _dead_vertex_store(tmp_path)
    stream, notices = _session_for(store, monkeypatch)
    await stream._on_route_change(
        failover_module.FallbackTarget("openai/gpt-test"), "provider failure: quota"
    )
    await stream._on_route_change(
        failover_module.FallbackTarget("anthropic/claude-test"), "provider failure: quota"
    )
    assert len(_sign_ins(notices)) == 1
    store.close()


@pytest.mark.asyncio
async def test_no_sign_in_notice_while_the_primary_login_is_alive(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AuthStore(db_path=tmp_path / "auth.db")
    _seed_expired_vertex(store)
    stream, notices = _session_for(store, monkeypatch)
    await stream._on_route_change(
        failover_module.FallbackTarget("anthropic/claude-test"), "provider failure: quota"
    )
    assert _sign_ins(notices) == []
    store.close()


# --- Headless (``lop exec``) rendering of the Urdu notice (QA Q2) -------------


def test_the_urdu_notice_prints_whole_commands_in_lop_exec() -> None:
    import io

    from rich.console import Console

    from local_operator import headless_print as hp
    from local_operator.harness.types import NoticeEvent

    shown = sign_in_required_text(
        "google-vertex", REAUTH_REASON_SESSION_POLICY, locale="ur", display=True
    )
    buffer = io.StringIO()
    hp.PrintRenderer(Console(file=buffer, width=400, no_color=True)).handle(NoticeEvent(text=shown))
    out = buffer.getvalue()
    assert "\\u20" not in out
    assert f"{_LRI}/login google-vertex{_PDI}" in out
    assert f"{_LRI}lop login google-vertex{_PDI}" in out


def test_keep_isolates_still_escapes_everything_but_balanced_isolates() -> None:
    from local_operator.ansi import sanitize_prompt_line

    # The default (approval prompts) is unchanged: every Cf is escaped.
    assert sanitize_prompt_line(f"{_LRI}x{_PDI}") == "\\u2066x\\u2069"
    # Balanced pairs survive, nested too.
    nested = f"{_RLI}a {_LRI}b{_PDI} c{_PDI}"
    assert sanitize_prompt_line(nested, keep_isolates=True) == nested
    # An unmatched opener would isolate the rest of the row: escaped.
    assert sanitize_prompt_line(f"a{_LRI}b", keep_isolates=True) == "a\\u2066b"
    assert sanitize_prompt_line(f"a{_PDI}b", keep_isolates=True) == "a\\u2069b"
    # Overrides, marks and joiners (Trojan source) are still escaped.
    assert (
        sanitize_prompt_line("\u202egnp.terces\u200d", keep_isolates=True)
        == "\\u202egnp.terces\\u200d"
    )
    # A length cut that severs a pair drops the orphaned opener.
    cut = sanitize_prompt_line(f"ab{_LRI}cdef{_PDI}", limit=4, keep_isolates=True)
    assert cut == "abc"


# --- ``login-status`` column alignment (design D1) ----------------------------


@pytest.mark.parametrize("locale", ["en", "zh-CN"])
def test_login_status_aligns_identity_by_display_width(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, locale: str
) -> None:
    from rich.cells import cell_len

    from local_operator.i18n import resolve

    monkeypatch.setattr(resolve, "resolve_language", lambda: locale)
    store, _dead = _dead_vertex_store(tmp_path)
    store.upsert_credential(
        "anthropic",
        {"type": "oauth", "access": "a", "refresh": "r", "expires": None, "email": "x@e.test"},
    )
    store.upsert_credential(
        "google-vertex",
        {"type": "oauth", "access": "a", "refresh": "r2", "expires": 1, "email": "z@e.test"},
    )
    store._mark_grant_dead(
        max(row.id for row in store.list_credentials("google-vertex")),
        rejected_refresh_token="r2",
    )
    rows = [line for line in format_logins(store) if "identity=" in line]
    assert len(rows) == 3
    columns = {cell_len(line[: line.index("identity=")]) for line in rows}
    assert len(columns) == 1, rows
    store.close()
