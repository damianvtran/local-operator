"""Native-mode regression controls: permission alone cannot authenticate."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.effective import (
    effective_access,
    note_resolution,
)
from local_operator.network.credentials.types import BrokerError, CredentialRef, Grant
from local_operator.network.readiness import selected_model_fact
from local_operator.providers.clients import OpenAICompatClient
from local_operator.providers.failover import (
    AuthRetryKeyState,
    _resolve_access_for_provider,
)
from tests.unit.network.test_credentials_placement import (
    NETWORK,
    OWNER,
    _point_config_at,
    placement_mod,
)


def _native() -> Grant:
    return Grant(
        access_token="synthetic-only-native-token",
        kind="bearer",
        token_expires_at_ms=0,
        grant_expires_at_ms=int((time.time() + 600) * 1000),
        credential_ref=CredentialRef(
            owner_device=OWNER,
            owner_device_name="owner",
            provider="openai",
            kind="oauth",
            credential_id=1,
        ),
        served_by=OWNER,
        identity={"org_id": "synthetic-account", "account_id": "synthetic-account"},
    )


def _source(root: Path) -> MeshCredentialClient:
    document = placement_mod.PlacementDocument(NETWORK, root=root, written_by=OWNER)
    document.declare("openai", owner_device=OWNER, provider="openai", by=OWNER)
    document.grant("openai", "recipient", scope="session", by=OWNER)
    return MeshCredentialClient(
        root=root, self_device="recipient", network_id=NETWORK, placement=document
    )


@pytest.mark.parametrize(
    "code,state",
    [
        ("owner_offline", "owner_unavailable"),
        ("no_local_credential", "withdrawn"),
        ("grant_invalid", "withdrawn"),
        ("non_portable", "device_bound"),
        ("quota_blocked", "pending"),
    ],
)
def test_truthful_states(tmp_path: Path, code: str, state: str) -> None:
    source = _source(tmp_path)
    assert (
        effective_access("openai", root=tmp_path, mesh=source).state == "eligible_borrow_unverified"
    )
    assert not effective_access("openai", root=tmp_path, mesh=source).usable
    note_resolution(tmp_path, "openai", BrokerError(code=code, owner_device=OWNER))
    assert effective_access("openai", root=tmp_path, mesh=source).state == state
    assert not effective_access("openai", root=tmp_path, mesh=source).usable
    source.close()


def test_success_is_expiry_and_recipient_bound(tmp_path: Path) -> None:
    source = _source(tmp_path)
    native = _native()
    note_resolution(tmp_path, "openai", native)
    assert effective_access("openai", root=tmp_path, mesh=source).usable
    native.grant_expires_at_ms = 1
    note_resolution(tmp_path, "openai", native)
    assert effective_access("openai", root=tmp_path, mesh=source).state == "withdrawn"
    document = source.placement
    assert document is not None
    document.revoke("openai", "recipient", by=OWNER)
    assert not effective_access("openai", root=tmp_path, mesh=source).eligible
    assert effective_access("openai", root=tmp_path, local=True).state == "local"
    source.close()


@pytest.mark.parametrize("force", [False, True])
def test_live_grant_and_old_login_refusal_do_not_short_circuit_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force: bool
) -> None:
    source = _source(tmp_path)
    source.grants.put("openai", "same-session", _native())
    source._remember("openai", BrokerError(code="no_local_credential", owner_device=OWNER))
    calls = []
    monkeypatch.setattr("local_operator.network.store.find_own_relay", lambda _root: {})

    def request(_record: Any, frame: Any, **_kw: Any) -> Any:
        calls.append(_kw)
        rotated = _native()
        rotated.access_token = f"synthetic-rotation-{len(calls)}"
        return {"op": "ack", "detail": rotated.to_detail()}

    monkeypatch.setattr("local_operator.network.relay.control_request", request)
    result = source.request_grant_sync("openai", session_id="same-session", force_refresh=force)
    assert calls and calls[0]["force_refresh"] is force
    assert isinstance(result, Grant)
    assert result.access_token != _native().access_token
    next_result = source.request_grant_sync("openai", session_id="same-session")
    assert isinstance(next_result, Grant)
    assert next_result.access_token != result.access_token
    assert len(calls) == 2  # the same recipient/session sees the owner's next rotation
    source.close()


@pytest.mark.asyncio
async def test_hot_native_access_reaches_subscription_host_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network.credentials.store import build_auth_store
    from local_operator.network.identity import mint

    _point_config_at(monkeypatch, tmp_path)
    identity = mint(tmp_path, name="synthetic-recipient")
    store = build_auth_store(tmp_path)
    assert (
        await _resolve_access_for_provider(
            store, "openai", "unchanged-session-id", AuthRetryKeyState(), None
        )
        is None
    )
    document = placement_mod.PlacementDocument(NETWORK, root=tmp_path, written_by=OWNER)
    document.declare("openai", owner_device=OWNER, provider="openai", by=OWNER)
    document.grant("openai", identity.device_id, scope="session", by=OWNER)
    document.save()

    async def grant(_self: Any, _key: str, **_fields: Any) -> Grant:
        return _native()

    monkeypatch.setattr(MeshCredentialClient, "grant_async", grant)
    access = await _resolve_access_for_provider(
        store, "openai", "unchanged-session-id", AuthRetryKeyState(), None
    )
    assert access is not None and access.kind == "oauth"
    assert store.local.list_credentials("openai") == []
    client = OpenAICompatClient(
        "https://api.openai.com/v1", oauth_base_url="https://chatgpt.com/backend-api/codex"
    )
    try:
        assert client._request_base_url(access) == "https://chatgpt.com/backend-api/codex"
        headers = client._headers(None, access)
        assert headers["chatgpt-account-id"] == "synthetic-account"
        assert headers["OpenAI-Beta"]
        # Exercise the actual streaming HTTP path, not merely the route helper.
        # Only the network destination is looped back; native endpoint selection,
        # request body, account headers and SSE normalization are the real client.
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from local_operator.harness.types import (
            ChatRequest,
            Message,
            ModelSpec,
            StreamTextDelta,
        )

        captured = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                captured.append((self.path, self.headers.get("chatgpt-account-id"), body))
                payload = (
                    'data: {"type":"response.output_text.delta","delta":"native reply"}\n\n'
                    'data: {"type":"response.completed","response":{"output":[]}}\n\n'
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                pass  # No HTTP log can accidentally include authorization.

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        monkeypatch.setattr(
            "local_operator.providers.clients.CODEX_RESPONSES_URL",
            f"http://127.0.0.1:{server.server_port}/subscription/responses",
        )
        try:
            request = ChatRequest(
                model=ModelSpec(provider="openai", model_id="gpt-6.1-sol"),
                messages=[Message.user("synthetic native probe")],
            )
            events = [event async for event in client.stream(request, None, oauth_access=access)]
            assert any(
                isinstance(event, StreamTextDelta) and event.delta == "native reply"
                for event in events
            )
            assert captured[0][0:2] == ("/subscription/responses", "synthetic-account")
            assert captured[0][2]["store"] is False
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        assert effective_access("openai", root=tmp_path).state == "effective_borrow"
        selected = selected_model_fact(
            tmp_path, "openai/gpt-6.1-sol", session_id="unchanged-session-id"
        )
        assert selected["state"] == "effective_borrow" and selected["usable"]
        assert selected["session_id"] == "unchanged-session-id"
    finally:
        await client.aclose()
        store.close()


@pytest.mark.asyncio
async def test_readiness_tool_carries_active_selector_and_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.harness.types import ToolContext
    from local_operator.network import tool

    calls = []

    async def cli(argv: list[str], _timeout: float) -> tuple[int, str, str]:
        calls.append(argv)
        return 0, '{"ok":true,"checks":[]}', ""

    monkeypatch.setattr(tool, "_run_cli", cli)
    await tool.execute_network(
        "synthetic-call",
        {"action": "ready", "peer": "synthetic-peer"},
        None,
        None,
        ToolContext(cwd=".", session_id="same-session", session_model_label="openai/gpt-6.1-sol"),
    )
    assert calls
    assert calls[0][-4:] == ["--model", "openai/gpt-6.1-sol", "--session", "same-session"]


@pytest.mark.parametrize(
    "state,usable",
    [
        ("eligible_borrow_unverified", False),
        ("effective_borrow", True),
        ("pending", False),
        ("owner_unavailable", False),
        ("withdrawn", False),
        ("device_bound", False),
        ("absent", False),
    ],
)
def test_selected_readiness_row_does_not_promote_holder_permission(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    usable: bool,
) -> None:
    from types import SimpleNamespace

    from local_operator.network import readiness, wire

    member = SimpleNamespace(device_id="synthetic-peer", name="peer")
    link = SimpleNamespace(peer_build={}, capabilities={wire.PEER_READINESS_V1})
    monkeypatch.setattr(readiness, "_reachability_rows", lambda *_args, **_kw: ([], link))
    for name in (
        "build_row",
        "operator_row",
        "git_row",
        "mcp_servers_row",
        "model_credential_row",
        "tooling_row",
    ):
        monkeypatch.setattr(readiness, name, lambda *_args, **_kw: {})
    monkeypatch.setattr(readiness, "mcp_credential_rows", lambda *_args, **_kw: [])
    monkeypatch.setattr(
        readiness,
        "_ask_readiness",
        lambda *_args, **_kw: (
            {
                "selected_model_access": {
                    "selector": "openai/gpt-6.1-sol",
                    "state": state,
                    "usable": usable,
                }
            },
            "",
        ),
    )
    # Other capability rows are separate tested seams; this matrix isolates the
    # selected row's permission-vs-bearer verdict without constructing a relay.
    server: Any = SimpleNamespace()
    viewer: Any = SimpleNamespace()
    rows = readiness._peer_checks(
        member,
        record=None,
        server=server,
        viewer=viewer,
        deadline=None,
        selector="openai/gpt-6.1-sol",
        session_id="same-session",
    )
    selected = next(row for row in rows if row.get("capability") == "selected_model_credential")
    assert selected["code"] == state and selected["ok"] is usable
    assert selected["session_id"] == "same-session"


def test_device_bound_and_absent_are_not_effective(tmp_path: Path) -> None:
    assert effective_access("kimi", root=tmp_path).state == "device_bound"
    assert effective_access("openai", root=tmp_path).state == "absent"
    assert not effective_access("kimi", root=tmp_path).usable


@pytest.mark.parametrize("provider", ["openai", "meta-muse", "openrouter", "radient"])
def test_inventory_is_trusted_provider_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    from local_operator.network.credentials.offers import _encrypted_provider_rows
    from local_operator.providers.registry import credential_file_names

    names = credential_file_names(provider)
    monkeypatch.setattr(
        "local_operator.providers.registry.stored_provider_env_keys", lambda _root: set(names)
    )
    rows = _encrypted_provider_rows(tmp_path)
    if names:
        assert any(row.provider == provider for row in rows)
        assert all(row.credential_type == "api_key" and row.data == {} for row in rows)
    else:
        assert rows == []  # OAuth-only/device-bound rows are not invented.
    monkeypatch.setattr(
        "local_operator.providers.registry.stored_provider_env_keys",
        lambda _root: {"UNREGISTERED_SECRET"},
    )
    assert _encrypted_provider_rows(tmp_path) == []


@pytest.mark.asyncio
async def test_encrypted_only_record_is_offered_and_owner_can_resolve_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from local_operator.network.credentials.offers import _strict_provider_rows
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.registry import store_provider_key
    from local_operator.secrets.store import SecretStore

    # A real encrypted database under a synthetic root, with an ephemeral key:
    # neither Keychain nor live credential material participates in this rig.
    secrets = SecretStore(os.urandom(32), tmp_path)
    secrets.initialize()
    monkeypatch.setattr("local_operator.secrets.access.open_store", lambda *_args, **_kw: secrets)
    store_provider_key("OPENROUTER_API_KEY", "synthetic-provider-material", base=tmp_path)
    secrets.set("ordinary-agent-secret", b"synthetic-agent-material")
    assert not (tmp_path / "auth.db").exists()
    rows = _strict_provider_rows(tmp_path)
    assert any(row.provider == "openrouter" and row.credential_type == "api_key" for row in rows)
    assert all(row.provider != "ordinary-agent-secret" for row in rows)
    auth = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path)
    try:
        assert auth.list_credentials("openrouter") == []
        assert await auth.get_persisted_api_key("openrouter") is not None
        access = await auth.get_oauth_access("openrouter", read_only=True)
        assert access is not None and access.kind == "api_key"
    finally:
        auth.close()
