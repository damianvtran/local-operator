"""Metadata-only catalogue over actual owner and borrower relay sockets.

All native login/IdP/Codex responses here are synthetic. No real provider call,
cache import, inference request, or bearer grant is needed for catalogue reads.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from local_operator.network.credentials import catalogue, placement
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.types import MeshRefusal
from local_operator.providers import registry
from local_operator.providers.auth_store import AuthStore
from tests.unit.network import test_credentials_real_link as helpers

mesh = helpers.mesh
stub_provider = helpers.stub_provider
idp = helpers.idp
devices = helpers.devices


@pytest.fixture(autouse=True)
def loopback_only(monkeypatch):
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create = socket.create_connection
    original_resolve = socket.getaddrinfo

    def check(host):
        try:
            if host == "localhost" or ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise RuntimeError("external network forbidden before DNS/OS")

    def connect(sock, address):
        if sock.family != socket.AF_UNIX:
            check(address[0])
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family != socket.AF_UNIX:
            check(address[0])
        return original_connect_ex(sock, address)

    def create(address, *args, **kwargs):
        check(address[0])
        return original_create(address, *args, **kwargs)

    def resolve(host, *args, **kwargs):
        check(host)
        return original_resolve(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create)
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    with pytest.raises(RuntimeError, match="external network forbidden"):
        socket.create_connection(("192.0.2.1", 443), timeout=0.1)
    with pytest.raises(RuntimeError, match="external network forbidden"):
        socket.getaddrinfo("example.invalid", 443)
    from local_operator.network.credentials import github

    monkeypatch.setattr(github, "_resolve_program", lambda _name: None)
    assert github._resolve_program("gh") is None


@pytest.fixture
def catalogue_mesh(mesh, monkeypatch):
    # Use the native OpenAI identity/cascade but a rotating loopback IdP.
    definition = registry.get_provider_definition(helpers.STUB_PROVIDER)
    assert definition is not None
    monkeypatch.setitem(
        registry._BY_ID,
        "openai",
        dataclasses.replace(definition, id="openai", store_credentials_as="openai"),
    )
    auth = AuthStore(config_dir=mesh.a.root)
    row_id = auth.upsert_credential(
        "openai",
        {
            "type": "oauth",
            "access": "fixture-stale",
            "refresh": mesh.idp.current_refresh,
            "expires": int(time.time() * 1000) - 60000,
            "account_id": "fixture-account",
        },
    )
    state = {
        "gets": 0,
        "auth_ok": False,
        "account_ok": False,
        "during_get": None,
        "expected_account": "fixture-account",
        "echo": "",
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            state["gets"] += 1
            state["auth_ok"] = (
                self.headers.get("Authorization") == f"Bearer access-{mesh.idp.access_seq}"
            )
            state["account_ok"] = (
                self.headers.get("ChatGPT-Account-ID") == state["expected_account"]
            )
            if state["during_get"] is not None:
                state["during_get"]()
            slug = "gpt-fixture-sol"
            if state["echo"] == "bearer":
                slug = f"access-{mesh.idp.access_seq}"
            elif state["echo"] == "account":
                slug = state["expected_account"]
            body = json.dumps(
                {
                    "models": [
                        {
                            "slug": slug,
                            "context_window": 200000,
                            "max_output_tokens": 8192,
                            "input_modalities": ["text"],
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    from local_operator.model import discovery

    monkeypatch.setattr(
        discovery, "OPENAI_CHATGPT_MODELS_URL", f"http://127.0.0.1:{server.server_port}/models"
    )
    try:
        yield mesh, auth, row_id, state
    finally:
        auth.close()
        server.shutdown()
        server.server_close()
        thread.join(2)


def share(mesh):
    assert (
        helpers._lop_network("credential", "share", "openai", "--with", mesh.b.identity.name) == 0
    )
    helpers._pull(mesh)
    client = MeshCredentialClient.for_this_device(mesh.b.root)
    assert client is not None
    return client


def test_metadata_real_rpc_refresh_and_native_account_header(catalogue_mesh):
    mesh, auth, row_id, state = catalogue_mesh
    client = share(mesh)
    result = client.fetch_catalogue_metadata("openai")
    assert result["models"][0]["id"] == "gpt-fixture-sol"
    assert result["models"][0]["context_window"] == 200000
    assert state["auth_ok"] and state["account_ok"]
    assert state["gets"] == len(mesh.idp.posts) == 1
    serialized = json.dumps(result)
    assert "fixture-account" not in serialized
    assert "access-" not in serialized and "refresh" not in serialized
    assert result["material_generation"] >= 1 and result["login_chain_revision"] >= 1


@pytest.mark.parametrize("withdrawal", ["revoke", "member_removed"])
def test_catalogue_revoke_during_fetch_never_returns_metadata(catalogue_mesh, withdrawal):
    mesh, auth, row_id, state = catalogue_mesh
    client = share(mesh)

    def revoke():
        with placement.mutate(
            mesh.network_id, self_device=mesh.owner, root=mesh.a.root
        ) as document:
            document.revoke("openai", mesh.borrower, by=mesh.owner)

    state["during_get"] = revoke
    with pytest.raises(MeshRefusal):
        client.fetch_catalogue_metadata("openai")
    assert state["gets"] == 1


def test_unshared_has_no_refresh_or_catalogue_http(catalogue_mesh):
    mesh, auth, row_id, state = catalogue_mesh
    client = MeshCredentialClient(
        root=mesh.b.root, self_device=mesh.borrower, network_id=mesh.network_id, relay=mesh.b
    )
    with pytest.raises(MeshRefusal):
        client.fetch_catalogue_metadata("openai")
    assert not mesh.idp.posts and state["gets"] == 0


@pytest.mark.parametrize("restriction", ["deny", "derived", "local_only", "removed", "epoch"])
def test_authority_refuses_before_refresh_or_http(catalogue_mesh, restriction):
    from local_operator.network import store
    from local_operator.network.credentials import consent
    from local_operator.network.credentials.sync import MARK_LOCAL_ONLY, SyncState

    mesh, auth, row, state = catalogue_mesh
    client = share(mesh)
    if restriction == "deny":
        consent.restrict_and_revoke(
            mesh.network_id,
            mesh.owner,
            mesh.borrower,
            {"credential_key": "openai", "account_ref": "fixture-account"},
            "fixture-deny",
            mesh.a.root,
        )
    elif restriction == "derived":
        with placement.mutate(mesh.network_id, self_device=mesh.owner, root=mesh.a.root) as doc:
            doc.revoke("openai", mesh.borrower, by=mesh.owner)
            doc.grant("openai", mesh.borrower, by=mesh.owner, provenance="consent:missing-approval")
    elif restriction == "local_only":
        sync = SyncState.load(mesh.network_id, mesh.a.root)
        sync.record_mark("openai", MARK_LOCAL_ONLY)
        sync.save()
    else:
        with store.mutate(mesh.network_id, mesh.a.root) as doc:
            if restriction == "removed":
                member = doc.member(mesh.borrower)
                assert member is not None
                member.removed_at = time.time()
                member.lifecycle = "expired"
            else:
                doc.epoch += 1
            store.save(doc, mesh.a.root)
    with pytest.raises(MeshRefusal):
        client.fetch_catalogue_metadata("openai")
    assert len(mesh.idp.posts) == state["gets"] == 0


def test_revoke_during_owner_refresh_prevents_catalogue_http(catalogue_mesh, monkeypatch):
    mesh, auth, row, state = catalogue_mesh
    client = share(mesh)
    original = registry.get_provider_definition("openai")
    assert original is not None and original.refresh_token is not None
    original_refresh = original.refresh_token

    async def refresh(creds, **kwargs):
        result = await original_refresh(creds, **kwargs)
        with placement.mutate(mesh.network_id, self_device=mesh.owner, root=mesh.a.root) as doc:
            doc.revoke("openai", mesh.borrower, by=mesh.owner)
        return result

    monkeypatch.setitem(
        registry._BY_ID, "openai", dataclasses.replace(original, refresh_token=refresh)
    )
    with pytest.raises(MeshRefusal):
        client.fetch_catalogue_metadata("openai")
    assert len(mesh.idp.posts) == 1 and state["gets"] == 0


def test_quota_blocked_account_lists_but_normal_inference_stays_blocked(catalogue_mesh):
    from local_operator.network.credentials.types import BrokerError

    mesh, auth, row, state = catalogue_mesh
    client = share(mesh)
    auth.block_credential(row.id, "openai", block_ms=60000)
    result = client.fetch_catalogue_metadata("openai")
    assert result["models"] and state["gets"] == 1
    inference = client.request_grant_sync("openai", session_id="fixture-catalogue-quota")
    assert isinstance(inference, BrokerError), "catalogue must not unblock an inference grant"
    assert inference.code == "quota_blocked"


def test_denied_row_is_not_refreshed_when_another_account_is_approved(catalogue_mesh):
    from local_operator.network.credentials import consent

    mesh, auth, row, state = catalogue_mesh
    client = share(mesh)
    auth.upsert_credential(
        "openai",
        {
            "type": "oauth",
            "access": "fixture-other-stale",
            "refresh": mesh.idp.current_refresh,
            "expires": int(time.time() * 1000) - 60000,
            "account_id": "fixture-permitted",
        },
    )
    consent.restrict_and_revoke(
        mesh.network_id,
        mesh.owner,
        mesh.borrower,
        {"credential_key": "openai", "account_ref": "fixture-account"},
        "fixture-deny",
        mesh.a.root,
    )
    with placement.mutate(mesh.network_id, self_device=mesh.owner, root=mesh.a.root) as doc:
        doc.grant("openai", mesh.borrower, by=mesh.owner)
    state["expected_account"] = "fixture-permitted"
    result = client.fetch_catalogue_metadata("openai")
    assert result["models"] and state["account_ok"]
    assert len(mesh.idp.posts) == 1
    assert auth.get_credential(row.id).updated_at == row.updated_at


def test_old_peer_is_explicitly_unsupported_without_inference_fallback(catalogue_mesh):
    from local_operator.network.types import MeshRefusal

    mesh, auth, row, state = catalogue_mesh
    client = share(mesh)
    link, _ = mesh.b._ensure_link_with_reason(mesh.owner)
    link.capabilities = tuple(c for c in link.capabilities if c != catalogue.CAPABILITY)
    with pytest.raises(MeshRefusal) as error:
        client.fetch_catalogue_metadata("openai")
    assert error.value.code == "catalogue_unsupported"
    assert len(mesh.idp.posts) == state["gets"] == 0


def _document():
    return {
        "kind": "catalogue_metadata",
        "network_id": "fixture-network",
        "owner_device": "fixture-owner",
        "credential_key": "openai",
        "provider": "openai",
        "models": [
            {
                "id": "gpt-fixture",
                "provider": "openai",
                "context_window": 200000,
                "default_context_window": 200000,
                "max_context_window": None,
                "max_tokens": 8192,
                "supports_images": None,
            }
        ],
        "account_scope": "a" * 64,
        "incarnation": "b" * 64,
        "material_generation": 1,
        "login_chain_revision": 1,
        "ttl": 60,
        "validator": "c" * 64,
    }


@pytest.mark.parametrize(
    "mutation", ["secret", "foreign", "url", "count", "bytes", "ttl", "model_secret"]
)
def test_received_metadata_rejects_unscoped_or_secret_shaped_payload(mutation):
    value = _document()
    if mutation == "secret":
        value["access_token"] = "fixture-forbidden"
    elif mutation == "foreign":
        value["models"][0]["provider"] = "radient"
    elif mutation == "url":
        value["models"][0]["endpoint_url"] = "https://invalid.test"
    elif mutation == "count":
        value["models"] *= catalogue.MAX_MODELS + 1
    elif mutation == "bytes":
        value["models"][0]["id"] = "a" * (catalogue.MAX_BYTES + 1)
    elif mutation == "ttl":
        value["ttl"] = True
    else:
        value["models"][0]["id"] = "sk-fixture-forbidden"
    with pytest.raises(ValueError):
        catalogue.validate_response(
            value, network="fixture-network", owner="fixture-owner", key="openai"
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"account_id": "caller-forged"},
        {"endpoint_url": "https://invalid.test"},
        {"force_refresh": True},
        {"exclude_credential_ids": [1]},
        {"provider": "radient"},
    ],
)
def test_closed_owner_request_rejects_caller_scope_and_auth_controls(catalogue_mesh, extra):
    mesh, auth, row, state = catalogue_mesh
    share(mesh)
    client = MeshCredentialClient.for_relay(mesh.b)
    assert client is not None
    reply = client._ask_owner_directly(
        mesh.owner,
        {
            "op": "net_broker",
            "kind": "catalogue",
            "network_id": mesh.network_id,
            "from_device": mesh.borrower,
            "key": "openai",
            "provider": "openai",
            **extra,
        },
        required_capability=catalogue.CAPABILITY,
    )
    assert reply is not None
    assert reply["detail"]["kind"] == "error"
    assert len(mesh.idp.posts) == state["gets"] == 0


def test_real_peer_catalogue_kind_is_implemented(catalogue_mesh):
    """Baseline discriminator: old owner answers unknown kind, before any IdP POST."""
    mesh, auth, row, state = catalogue_mesh
    share(mesh)
    client = MeshCredentialClient.for_relay(mesh.b)
    assert client is not None
    # A deliberately raw protocol probe (not production's old-peer fallback).
    # This is the same RPC against the old owner overlay and the new owner.
    reply = client._ask_owner_directly(
        mesh.owner,
        {
            "op": "net_broker",
            "kind": "catalogue",
            "network_id": mesh.network_id,
            "from_device": mesh.borrower,
            "key": "openai",
            "provider": "openai",
        },
    )
    assert reply is not None
    assert reply["detail"]["kind"] == "catalogue_metadata", reply["detail"].get("code")
    assert state["gets"] == len(mesh.idp.posts) == 1


@pytest.mark.parametrize("echo", ["bearer", "account"])
def test_owner_rejects_provider_echo_of_authentication_identity(catalogue_mesh, echo):
    mesh, auth, row, state = catalogue_mesh
    client = share(mesh)
    state["echo"] = echo
    with pytest.raises(MeshRefusal):
        client.fetch_catalogue_metadata("openai")
    assert state["gets"] == len(mesh.idp.posts) == 1


def test_owner_refusal_does_not_echo_native_identity(monkeypatch):
    from types import SimpleNamespace

    def refuse(*args):
        raise MeshRefusal("not_a_holder", "native fixture-account identity refused")

    monkeypatch.setattr(catalogue, "_approved_rows", refuse)
    link = SimpleNamespace(
        local_capabilities=(catalogue.CAPABILITY,), capabilities=(catalogue.CAPABILITY,)
    )
    result = catalogue.owner_catalogue(None, link, {"key": "openai", "provider": "openai"})
    assert result["kind"] == "error" and result["code"] == "not_a_holder"
    assert "fixture-account" not in json.dumps(result)
