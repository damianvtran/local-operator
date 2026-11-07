"""Plain-store consumers use owner metadata, never a copied native credential.

The paired relays/rotating IdP/Codex HTTP fixture is slice A's real transport,
not a replacement controller or record injection. No provider inference runs.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any

import pytest

from local_operator.network import store
from local_operator.network.credentials import placement
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.controller import ProviderController
from tests.unit.network import test_catalogue_metadata as transport

mesh = transport.mesh
stub_provider = transport.stub_provider
idp = transport.idp
devices = transport.devices
loopback_only = transport.loopback_only
catalogue_mesh = transport.catalogue_mesh


@pytest.fixture
def borrower(catalogue_mesh, monkeypatch):
    mesh, native, row_id, state = catalogue_mesh
    assert (
        transport.helpers._lop_network(
            "credential",
            "share",
            "openai",
            "--with",
            mesh.b.identity.name,
            "--scope",
            "device",
        )
        == 0
    )
    transport.helpers._pull(mesh)
    # Both daemons inhabit one test process. Initialise the owner's existing
    # store in its own realm BEFORE moving the viewer's ambient config to B.
    # No owner's native row is placed in B, and no catalogue is warmed here.
    from local_operator.network.credentials.owner import broker_for_relay

    broker = broker_for_relay(mesh.a)
    assert broker is not None
    assert broker._auth_store_instance()._db_path.resolve() == (mesh.a.root / "auth.db").resolve()
    plain = AuthStore(mesh.b.root / "auth.db", config_dir=mesh.b.root)
    assert plain.list_credentials("openai") == []
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(mesh.b.root))
    from local_operator.paths import config_dir

    assert config_dir() == mesh.b.root, "stop setup before any wrong-realm factory call"
    controller = ProviderController(plain, mesh.b.root)
    try:
        yield mesh, plain, controller, state
    finally:
        plain.close()


@pytest.mark.asyncio
async def test_plain_controller_imports_native_metadata_without_local_secret(borrower):
    mesh, plain, controller, state = borrower
    before = controller.initial_catalogue()
    assert not any(row.model_id == "gpt-fixture-sol" for row in before)
    entries, statuses = await controller.live_catalogue()
    native = [row for row in entries if row.provider == "openai"]
    assert [row.model_id for row in native] == ["gpt-fixture-sol"]
    assert native[0].connected and native[0].context_window == 200000
    assert statuses["openai"] == "ok"
    assert state["gets"] == 1 and state["auth_ok"] and state["account_ok"]
    assert len(mesh.idp.posts) == 1
    assert plain.list_credentials("openai") == []
    # Frame one and the canonical validator see the SAME scoped set after live
    # refresh. Neither gets a fabricated unscoped API-key listing document.
    after = controller.initial_catalogue()
    assert {row.model_id for row in after if row.provider == "openai"} == {"gpt-fixture-sol"}
    from local_operator.model.discovery import offered_model_ids

    assert offered_model_ids("openai") == {"gpt-fixture-sol"}
    files = list(store.network_root(mesh.b.root).joinpath("catalogues").glob("*.json"))
    assert len(files) == 1
    text = files[0].read_text()
    assert all(
        value not in text
        for value in ["fixture-account", "fixture-stale", "Bearer", "refresh_token"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "OpenAI"])
async def test_cold_real_owner_model_set_autofetches_without_manual_refresh(
    borrower, tmp_path, monkeypatch, provider
):
    mesh, plain, controller, state = borrower
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.model import configure
    from local_operator.model.configure import invalidate_model_info_cache
    from tests.unit.session.runtime.test_daemon_serving_plane import (
        _boot,
        _recording_stream,
    )

    original_borrow = configure._borrowed_listing_token

    def no_openai_bearer(provider, **kwargs):
        assert provider != "openai", "metadata must not borrow a bearer"
        return original_borrow(provider, **kwargs)

    monkeypatch.setattr(configure, "_borrowed_listing_token", no_openai_bearer)
    invalidate_model_info_cache()
    from concurrent.futures import ThreadPoolExecutor

    from local_operator.session.runtime import serving

    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fixture-catalogue-switch")
    monkeypatch.setattr(serving, "_switch_executor_pool", pool)
    calls = []
    session, handle, runtime = await _boot(tmp_path, _recording_stream(calls))
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        await client.connect(runtime.record, runtime.record.session_id)
        await client.set_model(provider, "gpt-fixture-sol")
        assert session.model.provider == "openai"
        assert session.model.model_id == "gpt-fixture-sol"
        assert session.model.context_window == 200000
        assert state["gets"] == 1 and state["auth_ok"] and state["account_ok"]
        assert plain.list_credentials("openai") == []
        assert calls == [], "catalogue/model mutation must not dispatch inference"
    finally:
        await client.detach()
        runtime.close()
        await session.dispose()
        pool.shutdown(wait=True, cancel_futures=True)


@pytest.mark.asyncio
async def test_explicit_phone_provider_allowlist_does_not_union_shared_sources(borrower):
    mesh, plain, controller, state = borrower
    entries, statuses = await controller.live_catalogue(providers=[])
    assert entries == [] and statuses == {}
    assert state["gets"] == 0 and len(mesh.idp.posts) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "restriction", ["session", "derived", "epoch", "removed", "untrusted", "local_only"]
)
async def test_current_machine_permission_guards_precede_metadata_io(borrower, restriction):
    mesh, plain, controller, state = borrower
    if restriction in {"session", "derived"}:
        with placement.mutate(mesh.network_id, self_device=mesh.owner, root=mesh.a.root) as doc:
            doc.revoke("openai", mesh.borrower, by=mesh.owner)
            doc.grant(
                "openai",
                mesh.borrower,
                by=mesh.owner,
                scope="session" if restriction == "session" else "device",
                provenance="consent:fixture-session-approval" if restriction == "derived" else "",
            )
        transport.helpers._pull(mesh)
    elif restriction in {"removed", "untrusted"}:
        with store.mutate(mesh.network_id, mesh.b.root) as record:
            if restriction == "untrusted":
                record.trust = "untrusted"
            else:
                member = record.member(mesh.borrower)
                assert member is not None
                member.removed_at = time.time()
                member.lifecycle = "expired"
            store.save(record, mesh.b.root)
    elif restriction == "local_only":
        from local_operator.network.credentials.sync import MARK_LOCAL_ONLY, SyncState

        sync = SyncState.load(mesh.network_id, mesh.b.root)
        sync.record_mark("openai", MARK_LOCAL_ONLY)
        sync.save()
    elif restriction == "epoch":
        # A cached receipt can never survive a changed local epoch.
        await controller.shared_catalogue("openai")
        state["gets"] = 0
        with store.mutate(mesh.network_id, mesh.b.root) as record:
            record.epoch += 1
            store.save(record, mesh.b.root)
        from local_operator.providers.shared_catalogue import cached

        assert cached(mesh.b.root) is None
        return
    expected = ([], "unsupported") if restriction in {"session", "derived"} else None
    assert await controller.shared_catalogue("openai") == expected
    assert state["gets"] == 0 and len(mesh.idp.posts) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage", ["foreign_owner", "foreign_network", "expired", "bool_epoch", "unknown_field"]
)
async def test_cache_namespace_and_lifetime_damage_never_authorizes_models(borrower, damage):
    mesh, plain, controller, state = borrower
    await controller.shared_catalogue("openai")
    (path,) = list(store.network_root(mesh.b.root).joinpath("catalogues").glob("*.json"))
    envelope = json.loads(path.read_text())
    if damage == "foreign_owner":
        envelope["document"]["owner_device"] = "fixture-foreign-owner"
    elif damage == "foreign_network":
        envelope["document"]["network_id"] = "fixture-foreign-network"
    elif damage == "expired":
        envelope["received_at"] -= 301
    elif damage == "bool_epoch":
        envelope["epoch"] = True
    else:
        envelope["document"]["token"] = "fixture-forbidden-field"
    path.write_text(json.dumps(envelope))
    from local_operator.providers.shared_catalogue import cached

    assert cached(mesh.b.root) is None
    from local_operator.model.discovery import offered_model_ids

    assert "gpt-fixture-sol" not in (offered_model_ids("openai") or set())


@pytest.mark.asyncio
async def test_older_own_relay_is_typed_unsupported_without_rpc(monkeypatch, borrower):
    mesh, plain, controller, state = borrower
    from local_operator.network.credentials.catalogue import CAPABILITY

    record = store.find_own_relay(mesh.b.root)
    assert record is not None
    older = dataclasses.replace(
        record, capabilities=[cap for cap in record.capabilities if cap != CAPABILITY]
    )
    monkeypatch.setattr(store, "find_own_relay", lambda _root=None: older)
    assert await controller.shared_catalogue("openai") == ([], "unsupported")
    assert state["gets"] == 0 and len(mesh.idp.posts) == 0


@pytest.mark.asyncio
async def test_explicit_provider_list_cannot_reuse_ambient_shared_cache(borrower):
    mesh, plain, controller, state = borrower
    await controller.shared_catalogue("openai")
    gets = state["gets"]
    entries, statuses = await controller.live_catalogue(providers={"openai"})
    assert not any(row.model_id == "gpt-fixture-sol" for row in entries)
    assert state["gets"] == gets


@pytest.mark.asyncio
async def test_native_local_credential_wins_over_prior_shared_receipt(borrower):
    mesh, plain, controller, state = borrower
    await controller.shared_catalogue("openai")
    gets = state["gets"]
    plain.upsert_credential("openai", {"type": "api_key", "key": "fixture-local-native-key"})
    assert await controller.shared_catalogue("openai") is None
    assert state["gets"] == gets
    from local_operator.model.discovery import offered_model_ids

    assert "gpt-fixture-sol" not in (offered_model_ids("openai") or set())


@pytest.mark.asyncio
async def test_public_authoring_models_uses_plain_store_and_real_owner_rpc(borrower, monkeypatch):
    mesh, plain, controller, state = borrower
    import asyncio

    from fastapi.testclient import TestClient

    from local_operator.config import ConfigManager
    from local_operator.env import get_env_config
    from local_operator.server.app import app
    from local_operator.server.routes import models

    monkeypatch.setitem(
        app.dependency_overrides, models.get_config_manager, lambda: ConfigManager(mesh.b.root)
    )
    monkeypatch.setitem(app.dependency_overrides, models.get_provider_auth_store, lambda: plain)
    monkeypatch.setattr(app.state, "env_config", get_env_config(), raising=False)
    # Like the existing authoring-route tests, do not start unrelated app
    # lifespan services; the real route/dependencies still handle this request.
    client = TestClient(app, headers={"Host": "127.0.0.1:1111"})
    try:
        response = await asyncio.to_thread(client.get, "/v1/models/?provider=openai")
    finally:
        client.close()
    assert response.status_code == 200
    assert {row["id"] for row in response.json()["result"]["models"]} == {"gpt-fixture-sol"}
    assert state["auth_ok"] and state["account_ok"] and state["gets"] == 1
    assert plain.list_credentials("openai") == []


@pytest.fixture
def changing_catalogue(borrower, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from local_operator.model import discovery

    mesh, plain, controller, original = borrower
    state = {
        "slug": "gpt-fixture-alpha",
        "window": 200000,
        "account": "fixture-account",
        "gets": 0,
        "authenticated": False,
        "during_get": None,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            state["gets"] += 1
            state["authenticated"] = (
                self.headers.get("Authorization") == f"Bearer access-{mesh.idp.access_seq}"
                and self.headers.get("ChatGPT-Account-ID") == state["account"]
            )
            if state["during_get"]:
                state["during_get"]()
            body = json.dumps(
                {
                    "models": [
                        {
                            "slug": state["slug"],
                            "context_window": state["window"],
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
    thread = threading.Thread(
        target=server.serve_forever, daemon=True, name="fixture-account-catalogue-http"
    )
    thread.start()
    monkeypatch.setattr(
        discovery, "OPENAI_CHATGPT_MODELS_URL", f"http://127.0.0.1:{server.server_port}/models"
    )
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


@pytest.mark.asyncio
async def test_account_incarnation_replacement_rekeys_info_and_removes_old_ids(
    borrower, changing_catalogue, catalogue_mesh
):
    import asyncio

    from local_operator.model.configure import build_model_spec
    from local_operator.model.discovery import offered_model_ids
    from local_operator.providers import shared_catalogue

    mesh, plain, controller, _ = borrower
    state = changing_catalogue
    await controller.shared_catalogue("openai")
    first = shared_catalogue.cached(mesh.b.root)
    assert first is not None and state["authenticated"]
    assert (
        await asyncio.to_thread(build_model_spec, "openai", state["slug"])
    ).context_window == 200000
    (old_path,) = list(store.network_root(mesh.b.root).joinpath("catalogues").glob("*.json"))
    orphan = old_path.with_name("fixture-orphan.json")
    orphan.write_bytes(old_path.read_bytes())
    # Only synthetic owner data is replaced through the native logout/login API.
    owner_auth, old_row = catalogue_mesh[1:3]
    owner_auth.delete_credential(old_row.id)
    owner_auth.upsert_credential(
        "openai",
        {
            "type": "oauth",
            "access": "fixture-new-stale",
            "refresh": mesh.idp.current_refresh,
            "expires": int(time.time() * 1000) - 60000,
            "account_id": "fixture-second-account",
        },
    )
    state["account"], state["window"] = "fixture-second-account", 300000
    second = await asyncio.to_thread(shared_catalogue.refresh, mesh.b.root, fresh=True)
    assert second is not None and state["authenticated"]
    assert second["account_scope"] != first["account_scope"]
    assert second["incarnation"] != first["incarnation"]
    assert (
        await asyncio.to_thread(build_model_spec, "openai", state["slug"])
    ).context_window == 300000
    state["slug"] = "gpt-fixture-beta"
    await asyncio.to_thread(shared_catalogue.refresh, mesh.b.root, fresh=True)
    assert offered_model_ids("openai") == {"gpt-fixture-beta"}
    assert {r.model_id for r in controller.initial_catalogue() if r.provider == "openai"} == {
        "gpt-fixture-beta"
    }
    assert state["gets"] == 3 and len(mesh.idp.posts) == 2
    assert plain.list_credentials("openai") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["before", "during_fetch"])
async def test_known_owner_denial_invalidates_prior_receipt_without_new_bearer(
    borrower, changing_catalogue, when
):
    from local_operator.model.discovery import offered_model_ids
    from local_operator.network.credentials import consent
    from local_operator.providers import shared_catalogue

    mesh, plain, controller, _ = borrower
    state = changing_catalogue
    await controller.shared_catalogue("openai")
    assert offered_model_ids("openai") == {"gpt-fixture-alpha"}

    def deny():
        consent.restrict_and_revoke(
            mesh.network_id,
            mesh.owner,
            mesh.borrower,
            {"credential_key": "openai", "account_ref": "fixture-account"},
            "fixture-b-consumer-deny",
            mesh.a.root,
        )

    if when == "before":
        deny()
        # Retain an explicit device share while the only account remains denied.
        with placement.mutate(mesh.network_id, self_device=mesh.owner, root=mesh.a.root) as doc:
            doc.grant("openai", mesh.borrower, by=mesh.owner, scope="device")
    else:
        state["during_get"] = deny
    posts = len(mesh.idp.posts)
    assert await controller.shared_catalogue("openai", fresh=True) == ([], "owner_unavailable")
    assert len(mesh.idp.posts) == posts
    assert state["gets"] == (1 if when == "before" else 2)
    assert shared_catalogue.cached(mesh.b.root) is None
    assert "gpt-fixture-alpha" not in (offered_model_ids("openai") or set())


@pytest.mark.asyncio
@pytest.mark.parametrize("restriction", ["withdrawal", "ownership"])
async def test_known_lifecycle_withdrawal_and_ownership_change_hide_current_receipt(
    borrower, restriction
):
    import hashlib

    from local_operator.model.discovery import offered_model_ids
    from local_operator.network.credentials import lifecycle
    from local_operator.providers import shared_catalogue

    mesh, plain, controller, _ = borrower
    await controller.shared_catalogue("openai")
    receipt = shared_catalogue.cached(mesh.b.root)
    assert receipt is not None
    if restriction == "withdrawal":
        contexts = lifecycle._native_contexts(mesh.a.root, "openai")
        assert contexts is not None
        (scope,) = [
            s
            for s in contexts
            if hashlib.sha256(f"{mesh.network_id}:{mesh.owner}:{s}".encode()).hexdigest()
            == receipt["account_scope"]
        ]
        observed = lifecycle.LifecycleStore(mesh.network_id, mesh.b.root).observe(
            lifecycle.Observation(
                owner=mesh.owner,
                key="openai",
                account_scope_ref=scope,
                login_chain_revision=receipt["login_chain_revision"] + 1,
                material_generation=receipt["material_generation"] + 1,
                state=lifecycle.STATE_WITHDRAWN,
            )
        )
        assert observed.applied
        assert shared_catalogue.cached(mesh.b.root) is None
        assert "gpt-fixture-sol" not in (offered_model_ids("openai") or set())
    else:
        # A changed owner namespace cannot revive an old owner's private receipt.
        with placement.mutate(mesh.network_id, self_device=mesh.borrower, root=mesh.b.root) as doc:
            entry = doc.entry("openai")
            assert entry is not None
            entry.owner_device = mesh.borrower
        assert shared_catalogue.cached(mesh.b.root) is None
    assert shared_catalogue.cached(mesh.b.root) is None
    assert "gpt-fixture-sol" not in (offered_model_ids("openai") or set())


@pytest.mark.asyncio
async def test_cold_birth_validator_prepares_real_owner_metadata(borrower):
    import asyncio

    from local_operator.server.routes.desktop_sessions import (
        DraftModel,
        _draft_model_spec,
    )

    mesh, plain, controller, state = borrower
    model = DraftModel(provider="openai", model_id="gpt-fixture-sol")
    spec = await asyncio.to_thread(_draft_model_spec, model)
    assert spec.provider == "openai" and spec.model_id == "gpt-fixture-sol"
    assert spec.context_window == 200000
    assert state["gets"] == 1 and state["auth_ok"] and state["account_ok"]
    assert plain.list_credentials("openai") == []


@pytest.mark.asyncio
async def test_controller_metadata_does_not_bypass_normal_inference_budget(
    borrower, catalogue_mesh
):
    import asyncio

    from local_operator.network.credentials.client import MeshCredentialClient
    from local_operator.network.credentials.types import BrokerError

    mesh, plain, controller, state = borrower
    owner_auth, row = catalogue_mesh[1:3]
    owner_auth.block_credential(row.id, "openai", block_ms=60000)
    listed = await controller.shared_catalogue("openai")
    assert listed is not None and listed[1] == "ok" and listed[0]
    client = MeshCredentialClient.for_this_device(mesh.b.root)
    assert client is not None
    grant = await asyncio.to_thread(
        client.request_grant_sync, "openai", session_id="fixture-b-budget"
    )
    assert isinstance(grant, BrokerError) and grant.code == "quota_blocked"
    assert state["gets"] == 1 and plain.list_credentials("openai") == []


@pytest.mark.asyncio
async def test_old_owner_capability_is_unsupported_without_token_fallback(borrower):
    from local_operator.network.credentials.catalogue import CAPABILITY

    mesh, plain, controller, state = borrower
    link, _ = mesh.b._ensure_link_with_reason(mesh.owner)
    assert link is not None
    link.capabilities = tuple(cap for cap in link.capabilities if cap != CAPABILITY)
    assert await controller.shared_catalogue("openai") == ([], "unsupported")
    assert state["gets"] == len(mesh.idp.posts) == 0
    assert plain.list_credentials("openai") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["device", "session"])
async def test_local_runtime_override_precedes_session_only_share(borrower, scope):
    mesh, plain, controller, state = borrower
    await controller.shared_catalogue("openai")
    if scope == "session":
        with placement.mutate(mesh.network_id, self_device=mesh.owner, root=mesh.a.root) as doc:
            doc.revoke("openai", mesh.borrower, by=mesh.owner)
            doc.grant("openai", mesh.borrower, by=mesh.owner, scope="session")
        transport.helpers._pull(mesh)
    plain.set_runtime_api_key("openai", "fixture-explicit-runtime-key")
    assert await controller.shared_catalogue("openai") is None
    rows, _ = await controller.live_catalogue()
    assert not any(row.model_id == "gpt-fixture-sol" for row in rows if row.provider == "openai")
    assert state["gets"] == len(mesh.idp.posts) == 1
    assert plain.list_credentials("openai") == []


@pytest.mark.asyncio
async def test_wrapped_controller_virtual_rows_do_not_turn_into_bearer_catalogue(
    borrower, monkeypatch
):
    from local_operator.network.credentials.client import MeshCredentialClient
    from local_operator.network.credentials.store import MeshAwareAuthStore

    mesh, plain, _, state = borrower
    client = MeshCredentialClient.for_this_device(mesh.b.root)
    assert client is not None
    wrapped = MeshAwareAuthStore(plain, mesh=client, config_dir=mesh.b.root)

    async def no_bearer(*args, **kwargs):
        raise AssertionError("catalogue must not request a borrowed bearer")

    monkeypatch.setattr(wrapped, "get_api_key", no_bearer)
    from typing import cast

    from local_operator.providers.controller import ControllerAuthStore

    # The real facade delegates the protocol via __getattr__, which static
    # structural checks cannot enumerate. Exercise that delegation, not a fake.
    assert callable(wrapped.get_catalogue_access) and callable(wrapped.override_keys)
    controller = ProviderController(cast(ControllerAuthStore, wrapped), mesh.b.root)
    rows, statuses = await controller.live_catalogue()
    assert {row.model_id for row in rows if row.provider == "openai"} == {"gpt-fixture-sol"}
    assert statuses["openai"] == "ok" and state["gets"] == 1
    assert plain.list_credentials("openai") == []


@pytest.mark.asyncio
async def test_pending_lifecycle_baseline_hides_a_positive_receipt(borrower):
    """An unresolved baseline outranks a receipt that predates it."""
    mesh, plain, controller, state = borrower
    assert await controller.shared_catalogue("openai") is not None
    from local_operator.model.discovery import offered_model_ids
    from local_operator.network.credentials import lifecycle
    from local_operator.providers import shared_catalogue

    assert "gpt-fixture-sol" in (offered_model_ids("openai") or set())
    marker = lifecycle.LifecycleStore(mesh.network_id, mesh.b.root).path.with_name(
        lifecycle.LIFECYCLE_OVERFLOW_FILENAME
    )
    marker.write_bytes(b"")
    try:
        # Presence-only latch: the identities it could not persist cannot be
        # reconstructed, so no receipt issued before it may be offered.
        assert shared_catalogue.cached(mesh.b.root) is None
        assert "gpt-fixture-sol" not in (offered_model_ids("openai") or set())
        assert shared_catalogue.refresh(mesh.b.root, fresh=True) is None
    finally:
        marker.unlink()


@pytest.mark.asyncio
async def test_preparation_reads_the_explicit_session_root(borrower, tmp_path, monkeypatch):
    """A session resumed under its own root must not consult the ambient one."""
    mesh, plain, controller, state = borrower
    from local_operator.providers import shared_catalogue

    ambient = tmp_path / "ambient-empty"
    ambient.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(ambient))
    # The ambient realm holds no approved share, so an ambient read cannot fetch.
    shared_catalogue.prepare_selection_sync("openai")
    assert state["gets"] == 0
    # The session root is the realm that carries the approved share.
    shared_catalogue.prepare_selection_sync("openai", mesh.b.root)
    assert state["gets"] == 1


@pytest.mark.asyncio
async def test_peer_selection_accepts_an_owner_native_id_on_a_borrow_only_device(borrower):
    """Q3: the pair the picker offers must also be selectable over the wire.

    The target holds NO local openai row — the mesh borrow is a delegate rung on
    ``MeshAwareAuthStore``, which the plain store this validator builds is not —
    so usability has to be answered by the same receipt question the picker
    asks. The cold case is the interesting one: preparation must warm the
    catalogue, then usability, then the offered-id check.
    """
    import asyncio

    from local_operator.mobile.peer_model import validate_peer_selection
    from local_operator.model.discovery import offered_model_ids

    mesh, plain, controller, state = borrower
    cold = offered_model_ids("openai")
    assert cold is not None
    assert "gpt-fixture-sol" not in cold
    assert plain.list_credentials("openai") == []
    spec = await asyncio.to_thread(validate_peer_selection, "openai", "gpt-fixture-sol")
    assert (spec.provider, spec.model_id) == ("openai", "gpt-fixture-sol")
    assert state["gets"] == 1
    warmed = offered_model_ids("openai")
    assert warmed is not None
    assert "gpt-fixture-sol" in warmed


@pytest.mark.asyncio
async def test_peer_selection_still_refuses_an_unoffered_id(borrower):
    """No widening: a borrow-only target accepts only ids the owner offers."""
    import asyncio

    from local_operator.mobile.peer_model import validate_peer_selection
    from local_operator.model.configure import ModelSelectionRefused

    mesh, plain, controller, state = borrower
    with pytest.raises(ModelSelectionRefused) as refused:
        await asyncio.to_thread(validate_peer_selection, "openai", "gpt-not-a-real-model")
    assert "not a model openai serves" in str(refused.value)


@pytest.mark.asyncio
async def test_peer_selection_translates_an_owner_transport_refusal(borrower, monkeypatch):
    """The new defect: a sleeping owner must answer through THIS module.

    ``prepare_selection_sync`` dials the owner and can raise ``MeshRefusal``,
    which is not a ``ModelSelectionRefused``; both hosts catch only the latter,
    so an escaping refusal became a bare error string with no code. Every
    sentence a sender reads is composed here, so the transport refusal is
    translated instead of leaked.
    """
    import asyncio

    from local_operator.mobile.peer_model import validate_peer_selection
    from local_operator.model.configure import ModelSelectionRefused
    from local_operator.providers import shared_catalogue

    mesh, plain, controller, state = borrower
    assert shared_catalogue.source(mesh.b.root) is not None
    monkeypatch.setattr(shared_catalogue, "refresh", lambda *a, **k: None)
    with pytest.raises(ModelSelectionRefused) as refused:
        await asyncio.to_thread(validate_peer_selection, "openai", "gpt-fixture-sol")
    assert refused.value.code == "owner_offline"
    assert "catalogue metadata unavailable" in str(refused.value)


@pytest.mark.asyncio
async def test_a_prepared_selection_skips_the_owner_when_the_caller_knows_it(borrower):
    """A locally resolvable selection must not wait on the owner at all.

    The override tiers live on a store instance that a serving session or a peer
    switch does not hold, so the caller's own answer is the only one those hosts
    can offer; preparation must honour it instead of spending an owner RPC and
    aborting when the owner device sleeps.
    """
    from local_operator.providers import shared_catalogue

    mesh, plain, controller, state = borrower
    before = state["gets"]
    shared_catalogue.prepare_selection_sync("openai", mesh.b.root, local_ready=True)
    assert state["gets"] == before
    assert shared_catalogue.cached(mesh.b.root) is None
    shared_catalogue.prepare_selection_sync("openai", mesh.b.root)
    assert state["gets"] == before + 1


@pytest.mark.asyncio
async def test_a_device_with_its_own_key_keeps_its_rows(borrower, monkeypatch):
    """The pending-owner branch must not paint EMPTY over a user's own login.

    ``approved()`` answers "is a share pending" — it must keep every exclusion
    ``source()`` applies to LOCAL authority, or a device that holds its own
    OpenAI key and an approved placement gets ``models = []``: the "no models are
    signed in yet" symptom inverted, which is the bug this PR exists to remove.
    """
    from local_operator.providers import shared_catalogue

    mesh, plain, controller, state = borrower
    await controller.shared_catalogue("openai")
    assert shared_catalogue.source(mesh.b.root) is not None
    before = controller.initial_catalogue()
    assert {row.model_id for row in before if row.provider == "openai"} == {"gpt-fixture-sol"}
    # An environment key is a local credential at the ordering rung, but this
    # rig must not depend on the operator machine's real environment: patch
    # the resolver the metadata reader consults, not the process.
    monkeypatch.setattr(
        "local_operator.providers.registry.resolve_env_key",
        lambda provider: ("fixture-own-key" if provider == "openai" else None),
    )
    try:
        # Own credential: the share is not this device's authority…
        assert shared_catalogue.source(mesh.b.root) is None
        assert shared_catalogue.approved(mesh.b.root) is False
        # …and the frame paints the user's own rows, never an empty list.
        own = controller.initial_catalogue()
        own_ids = {row.model_id for row in own if row.provider == "openai"}
        assert own_ids, "own-credential device must keep its own openai rows"
        assert "gpt-fixture-sol" not in own_ids
    finally:
        pass


@pytest.mark.asyncio
async def test_a_session_scoped_share_is_not_a_pending_machine_share(borrower):
    """Only a device-scoped share may blank the frame while its receipt is absent."""
    from local_operator.network.credentials import placement
    from local_operator.providers import shared_catalogue

    mesh, plain, controller, state = borrower
    with placement.mutate(mesh.network_id, self_device=mesh.owner, root=mesh.a.root) as doc:
        doc.revoke("openai", mesh.borrower, by=mesh.owner)
        doc.grant("openai", mesh.borrower, by=mesh.owner, scope="session")
    transport.helpers._pull(mesh)
    assert shared_catalogue.source(mesh.b.root) is None
    assert shared_catalogue.approved(mesh.b.root) is False
    rows = {row.model_id for row in controller.initial_catalogue() if row.provider == "openai"}
    assert "gpt-fixture-sol" not in rows


@pytest.mark.asyncio
async def test_a_raw_refresh_failure_answers_through_the_peer_contract(borrower, monkeypatch):
    """MINOR: OSError/ValueError out of ``refresh`` must not escape raw either."""
    import asyncio

    from local_operator.mobile.peer_model import validate_peer_selection
    from local_operator.model.configure import ModelSelectionRefused
    from local_operator.providers import shared_catalogue

    mesh, plain, controller, state = borrower

    def boom(*args, **kwargs):
        raise OSError("store unreadable")

    monkeypatch.setattr(shared_catalogue, "refresh", boom)
    with pytest.raises(ModelSelectionRefused) as refused:
        await asyncio.to_thread(validate_peer_selection, "openai", "gpt-fixture-sol")
    assert refused.value.code == "owner_offline"
    assert "catalogue metadata unavailable" in str(refused.value)


@pytest.mark.asyncio
async def test_approved_share_survives_a_latch_for_the_pending_frame(borrower):
    """MINOR-1: the frame's "pending owner" question is not ``source()``.

    ``source()`` fail-closes under a lifecycle latch, so using it to decide
    "paint nothing while the share is pending" fell through to the bundled
    static catalogue. ``approved()`` answers the structural question only and
    grants no receipt, id or authority.
    """
    from local_operator.network.credentials import lifecycle
    from local_operator.providers import shared_catalogue

    mesh, plain, controller, state = borrower
    marker = lifecycle.LifecycleStore(mesh.network_id, mesh.b.root).path.with_name(
        lifecycle.LIFECYCLE_OVERFLOW_FILENAME
    )
    marker.write_bytes(b"")
    try:
        assert shared_catalogue.source(mesh.b.root) is None
        assert shared_catalogue.approved(mesh.b.root) is True
        assert shared_catalogue.cached(mesh.b.root) is None
    finally:
        marker.unlink()
    assert shared_catalogue.source(mesh.b.root) is not None


@pytest.mark.asyncio
async def test_desktop_usable_scope_offers_the_approved_owner_model(
    borrower, tmp_path, monkeypatch
):
    """``scope=usable`` must not filter out a model the authoring path offers."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from local_operator.server.routes import desktop_catalogues

    mesh, plain, controller, state = borrower
    assert await controller.shared_catalogue("openai") is not None
    from local_operator.model.discovery import offered_model_ids

    assert "gpt-fixture-sol" in (offered_model_ids("openai") or set())

    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    token = "shared-catalogue-consumer-test-token"
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.setenv("HOME", str(mesh.b.root))
    from local_operator.config import ConfigManager

    app = FastAPI()
    app.include_router(desktop_catalogues.router)
    app.state.config_manager = ConfigManager(mesh.b.root)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        response = await client.get("/v1/desktop/models?scope=usable")
    assert response.status_code == 200, response.text
    models = response.json()["result"]["models"]
    offered = {(row["provider"], row["model_id"]) for row in models}
    assert ("openai", "gpt-fixture-sol") in offered
    # The owner row is offered as metadata, never as a local bearer.
    assert plain.list_credentials("openai") == []


@pytest.mark.asyncio
async def test_a_session_scoped_switch_names_the_owner_remedy_then_succeeds_after_it(
    borrower, monkeypatch
):
    """The remote ``/model`` refusal must name the owner's decision, exactly.

    2026-11-06: ``/model openai/gpt-6.1-sol`` on a peer holding a SESSION-scoped
    openai share answered only "catalogue needs a device-scoped share". The gate
    stays (a machine-wide catalogue is device-scoped by design); what changes is
    that the sentence carries the one command the OWNER would run, and running
    that very command is what makes the switch succeed. Nothing is widened by
    the refusal itself: no owner RPC happens before it.
    """
    import asyncio

    from local_operator.mobile.peer_model import validate_peer_selection
    from local_operator.model.configure import ModelSelectionRefused
    from local_operator.network.types import MeshRefusal
    from local_operator.providers import shared_catalogue

    mesh, plain, controller, state = borrower
    with placement.mutate(mesh.network_id, self_device=mesh.owner, root=mesh.a.root) as doc:
        doc.revoke("openai", mesh.borrower, by=mesh.owner)
        doc.grant("openai", mesh.borrower, by=mesh.owner, scope="session")
    transport.helpers._pull(mesh)

    command = (
        f"lop network credential share openai --with {mesh.borrower} "
        f"--network {mesh.network_id} --scope device"
    )
    with pytest.raises(MeshRefusal) as direct:
        shared_catalogue.prepare_selection_sync("openai", mesh.b.root)
    assert direct.value.code == "catalogue_unsupported"
    assert str(direct.value).startswith("catalogue needs a device-scoped share")
    # D1/D2: the command on its own line, delimited, and both devices named.
    assert str(direct.value).endswith(f"run on device-a:\n`{command}`")
    assert "it lends openai to every session on device-b" in str(direct.value)
    with pytest.raises(ModelSelectionRefused) as refused:
        await asyncio.to_thread(validate_peer_selection, "openai", "gpt-fixture-sol")
    assert refused.value.code == "catalogue_unsupported"
    assert command in str(refused.value)
    assert state["gets"] == 0 and len(mesh.idp.posts) == 0
    print(f"BEFORE: {refused.value}")

    # The operator's decision, typed on the OWNER exactly as the sentence spells it.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(mesh.a.root))
    assert transport.helpers._lop_network(*command.split()[2:]) == 0
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(mesh.b.root))
    transport.helpers._pull(mesh)

    spec = await asyncio.to_thread(validate_peer_selection, "openai", "gpt-fixture-sol")
    assert (spec.provider, spec.model_id) == ("openai", "gpt-fixture-sol")
    assert state["gets"] == 1
    print(f"AFTER: {spec.provider}/{spec.model_id}")


@pytest.mark.asyncio
async def test_the_readiness_fact_is_the_gates_own_first_rungs(borrower, monkeypatch):
    """MINOR-1 / Q1: ``catalogue_authority`` answers from the rungs ``_share`` uses.

    A borrow-only holder reports ``borrows``; the same device with its own openai
    key reports ``local`` — whatever its default provider — and the gate then
    refuses nothing, which is exactly when readiness must stay silent.
    """
    from local_operator.providers import shared_catalogue

    mesh, plain, controller, state = borrower
    with placement.mutate(mesh.network_id, self_device=mesh.owner, root=mesh.a.root) as doc:
        doc.revoke("openai", mesh.borrower, by=mesh.owner)
        doc.grant("openai", mesh.borrower, by=mesh.owner, scope="session")
    transport.helpers._pull(mesh)
    assert shared_catalogue.catalogue_authority(mesh.b.root) == "borrows"
    monkeypatch.setattr(
        "local_operator.providers.registry.resolve_env_key",
        lambda provider: ("fixture-own-key" if provider == "openai" else None),
    )
    assert shared_catalogue.catalogue_authority(mesh.b.root) == "local"
    shared_catalogue.prepare_selection_sync("openai", mesh.b.root)  # no refusal
    assert state["gets"] == 0


@pytest.mark.asyncio
async def test_a_selection_outlasts_an_owner_round_trip_the_listing_cap_refuses(
    borrower, tmp_path, monkeypatch
):
    """Operator report, 2026-11-07: ``catalogue metadata unavailable`` on a peer create.

    A ``--model openai/...`` create on a device-scoped borrower was refused with
    exactly that sentence and the first turn ran the device default. The
    selection reused the picker's 3 s leg-1 cap, while every surrounding bound
    (``prepare_selection``'s 6.1 s, the runtime switch's 8 s) is wider: an owner
    round trip just past 3 s failed a selection its own budget would have let
    finish. The owner here answers after 3.3 s, over the real relays.

    The picker keeps its short bound (the control half): the same slow owner
    still refuses a listing refresh rather than holding a drawn list hostage.
    """
    import time as clock
    from concurrent.futures import ThreadPoolExecutor

    from local_operator.mobile.attach_client import AttachClient
    from local_operator.model.configure import invalidate_model_info_cache
    from local_operator.network.types import MeshRefusal
    from local_operator.providers import shared_catalogue
    from local_operator.session.runtime import serving
    from tests.unit.session.runtime.test_daemon_serving_plane import (
        _boot,
        _recording_stream,
    )

    mesh, plain, controller, state = borrower
    invalidate_model_info_cache()
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fixture-slow-owner-switch")
    monkeypatch.setattr(serving, "_switch_executor_pool", pool)
    state["during_get"] = lambda: clock.sleep(3.3)
    # THE LIVE FAILURE'S OWN SITE: the borrower runtime's ``set_model`` op, the
    # call a peer create's ``_set_model_on`` makes before the first prompt.
    calls: list[int] = []
    session, handle, runtime = await _boot(tmp_path, _recording_stream(calls))
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        await client.connect(runtime.record, runtime.record.session_id)
        await client.set_model("openai", "gpt-fixture-sol")
        assert (session.model.provider, session.model.model_id) == ("openai", "gpt-fixture-sol")
    finally:
        await client.detach()
        runtime.close()
        await session.dispose()
        pool.shutdown(wait=True, cancel_futures=True)
    assert state["gets"] == 1
    document = shared_catalogue.cached(mesh.b.root)
    assert document is not None, "the selection left no receipt behind"
    assert [row["id"] for row in document["models"]] == ["gpt-fixture-sol"]

    with pytest.raises(MeshRefusal) as refused:
        shared_catalogue.refresh(mesh.b.root, fresh=True)
    assert refused.value.code == "catalogue_unavailable"


def test_the_selection_budget_sits_between_the_listing_cap_and_the_outer_wait():
    """The ordering the slow-owner fix rests on, pinned without a clock.

    Above the listing cap (else a selection fails where it should wait) and
    strictly below ``prepare_selection``'s ``SELECTION_WAIT_S`` (else the outer
    timeout races the fetch and abandons its worker mid-write). Pinned through
    the named constants, never a copy of the literal (review round 1, N2).
    """
    from local_operator.providers import shared_catalogue

    assert shared_catalogue.LISTING_FETCH_S == 3.0
    assert (
        shared_catalogue.LISTING_FETCH_S
        < shared_catalogue.SELECTION_FETCH_S
        < shared_catalogue.SELECTION_WAIT_S
    )


def test_a_selection_pays_its_lock_wait_from_its_own_budget(borrower):
    """Review round 1, M1: a lock wait plus a full fetch overran the outer wait.

    A picker refresh of the same receipt holds the per-receipt lock while a slow
    owner answers; a selection that starts meanwhile used to wait up to 3 s for
    that lock and THEN up to 5.5 s for its own fetch, 8.5 s against the 6.1 s
    ``wait_for`` around it, so the outer timeout raced the worker. The lock wait
    is now spent from the selection's own budget: the worker settles itself
    (applied or a typed refusal) inside ``SELECTION_FETCH_S``.
    """
    import threading
    import time as clock

    from local_operator.network.types import MeshRefusal
    from local_operator.providers import shared_catalogue

    mesh, plain, controller, state = borrower
    state["during_get"] = lambda: clock.sleep(3.6)
    listing_started = threading.Event()

    def listing() -> None:
        listing_started.set()
        try:
            shared_catalogue.refresh(mesh.b.root, fresh=True)
        except MeshRefusal:
            pass  # the listing's own 3 s cap; it is only here to hold the lock

    holder = threading.Thread(target=listing, daemon=True)
    holder.start()
    assert listing_started.wait(5)
    clock.sleep(0.3)  # the listing is inside its fetch, holding the lock
    started = clock.monotonic()
    try:
        # Either outcome is acceptable; what is pinned is WHEN it settles.
        shared_catalogue.prepare_selection_sync("openai", mesh.b.root)
    except MeshRefusal:
        pass
    elapsed = clock.monotonic() - started
    holder.join(10)
    assert elapsed < shared_catalogue.SELECTION_FETCH_S + 0.5, elapsed
    assert elapsed < shared_catalogue.SELECTION_WAIT_S, elapsed


@pytest.mark.asyncio
async def test_the_device_scope_remedy_survives_the_runtime_error_frame(
    borrower, tmp_path, monkeypatch
):
    """QA round 1, Q3 on #3554: #3542's remedy was cut on the create path.

    A peer create whose ``--model`` meets a SESSION-scoped share is refused by
    the borrower runtime's ``set_model`` with #3542's sentence, which ends in
    the owner's exact command. The runtime's error frame cut every message at
    400 characters, dropping ``--scope device`` and half the network id, so the
    receipt printed a command that, pasted, does something else. The real
    refusal crosses the real runtime frame here, then the relay's receipt and
    the CLI's printed lines, and the command must arrive whole and last.
    """
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from local_operator.mobile.attach_client import AttachClient
    from local_operator.model.configure import invalidate_model_info_cache
    from local_operator.network import cli as network_cli
    from local_operator.network import relay
    from local_operator.session.runtime import serving
    from tests.unit.session.runtime.test_daemon_serving_plane import (
        _boot,
        _recording_stream,
    )

    mesh, plain, controller, state = borrower
    with placement.mutate(mesh.network_id, self_device=mesh.owner, root=mesh.a.root) as doc:
        doc.revoke("openai", mesh.borrower, by=mesh.owner)
        doc.grant("openai", mesh.borrower, by=mesh.owner, scope="session")
    transport.helpers._pull(mesh)
    command = (
        f"lop network credential share openai --with {mesh.borrower} "
        f"--network {mesh.network_id} --scope device"
    )
    invalidate_model_info_cache()
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fixture-remedy-switch")
    monkeypatch.setattr(serving, "_switch_executor_pool", pool)
    calls: list[int] = []
    session, handle, runtime = await _boot(tmp_path, _recording_stream(calls))
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        await client.connect(runtime.record, runtime.record.session_id)
        with pytest.raises(RuntimeError) as refused:
            await client.set_model("openai", "gpt-fixture-sol")
    finally:
        await client.detach()
        runtime.close()
        await session.dispose()
        pool.shutdown(wait=True, cancel_futures=True)
    message = str(refused.value)
    assert len(message) <= 400, len(message)
    assert message.endswith(f"\n`{command}`"), message
    assert message.startswith("catalogue needs a device-scoped share")
    # Design D4 = QA Q4: the clause naming WHERE to run it survives the cut.
    lead = message.splitlines()[-2]
    assert "To allow it, run on " in lead and lead.endswith(":"), message

    # The relay's receipt, from that exact frame (only the dial is scripted).
    class _Dial:
        def exchange(self, frame: dict[str, object], *, timeout_s: float) -> dict[str, object]:
            return {"op": "error", "message": message}

        def close(self) -> None:
            pass

    server = relay.RelayServer(
        root=mesh.b.root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    monkeypatch.setattr(server, "_dial_owned", lambda *args, **kwargs: _Dial())
    model = server._set_model_on(  # noqa: SLF001
        "sess00000001", {"provider": "openai", "model_id": "gpt-fixture-sol"}
    )
    assert model["applied"] is False
    assert model["detail"].endswith(f"\n`{command}`"), model

    # And the receipt a person reads: the command alone, whole, on the last model line.
    monkeypatch.setattr(
        network_cli,
        "_relay_answer",
        lambda *_a, **_k: {
            "ok": True,
            "session_id": "sess00000001",
            "admitted": True,
            "detail": "",
            "model": model,
            "turn": {"state": "settled", "detail": ""},
        },
    )
    args: Any = SimpleNamespace(
        peer="device-b",
        create=True,
        name="",
        prompt="Reply with exactly: SOL_OK",
        cwd="",
        model="",
        profile="",
        agent="",
        agent_id="",
        team="",
        effort="",
        yolo=False,
        json=False,
        **{key: "" for key in ("stop", "engage", "send", "steer", "slash", "peek")},
    )
    from contextlib import redirect_stdout
    from io import StringIO

    printed = StringIO()
    with redirect_stdout(printed):
        assert network_cli._cmd_sessions(args) == 0  # noqa: SLF001
    assert f"`{command}`" in printed.getvalue().splitlines(), printed.getvalue()


def test_an_error_without_a_trailing_command_is_cut_as_before():
    """The remedy rule is structural: only a lone backticked last line is kept whole."""
    from local_operator.session.runtime.server import bounded_error_message

    plain = "x" * 500
    assert bounded_error_message(plain) == plain[:400]
    prose_tail = "reason " * 60 + "\nnot a command"
    assert bounded_error_message(prose_tail) == prose_tail[:400]
    short = "fits\n`lop do it`"
    assert bounded_error_message(short) == short
    kept = bounded_error_message("why " * 120 + "\n`lop network credential share openai`")
    assert len(kept) <= 400 and kept.endswith("…\n`lop network credential share openai`")


def test_a_cut_reason_keeps_the_lead_in_that_says_where_to_run_the_command():
    """Design D4 = QA Q4 on #3554: the cut dropped "To allow it, run on <owner>:".

    Keeping only the command left a reader with a pasteable line and no word on
    WHICH device to paste it on. The reason's last sentence, when it ends in
    ``:``, now travels with the command, and the cut falls mid-reason instead.
    """
    from local_operator.session.runtime.server import bounded_error_message

    command = "`lop network credential share openai --with dev-b --network n1 --scope device`"
    text = (
        "catalogue needs a device-scoped share: "
        + "why " * 90
        + ("is owner-a's decision. To allow it, run on owner-a:\n" + command)
    )
    kept = bounded_error_message(text)
    assert len(text) > 400 >= len(kept), (len(text), len(kept))
    assert kept.startswith("catalogue needs a device-scoped share: ")
    assert kept.endswith("… To allow it, run on owner-a:\n" + command), kept
    # A last sentence that does not introduce the command keeps the old cut.
    plain = "why " * 120 + "the end.\n" + command
    assert bounded_error_message(plain).endswith("…\n" + command)


@pytest.mark.asyncio
async def test_the_worker_spends_a_deadline_fixed_at_submission(monkeypatch, tmp_path):
    """Review m1 on #3554: the 5.5 s clock started when a pool thread picked the job up.

    ``prepare_selection``'s outer wait starts at submission, so pool queueing (or
    a caller's deadline already part-spent) let it expire while the fetch still
    thought it had its full budget. The deadline is now computed at submission
    and handed in, with the fetch's margin under whatever the outer wait is.
    """
    from local_operator.providers import shared_catalogue

    seen: list[dict[str, Any]] = []

    def worker(provider: str, root: Any, **kwargs: Any) -> None:
        seen.append(kwargs)

    monkeypatch.setattr(shared_catalogue, "prepare_selection_sync", worker)
    margin = shared_catalogue.SELECTION_WAIT_S - shared_catalogue.SELECTION_FETCH_S

    before = time.monotonic()
    await shared_catalogue.prepare_selection("openai", tmp_path)
    after = time.monotonic()
    assert "deadline" in seen[-1], seen
    assert (
        before + shared_catalogue.SELECTION_FETCH_S
        <= seen[-1]["deadline"]
        <= after + shared_catalogue.SELECTION_FETCH_S
    )

    before = time.monotonic()
    await shared_catalogue.prepare_selection("openai", tmp_path, deadline=before + 3.0)
    # Never past the outer wait minus the fetch's own margin.
    assert seen[-1]["deadline"] <= before + 3.0 - margin + 1e-6, (seen[-1], before)


def test_an_offered_only_switch_refuses_only_what_the_catalogue_rules_out(monkeypatch):
    """QA round 1, Q2 on #3554: the membership check, and its three accepting cases."""
    from local_operator.model import discovery
    from local_operator.model.configure import ModelSelectionRefused
    from local_operator.session.runtime import serving

    offered: dict[str, set[str] | None] = {"openai": {"gpt-fixture-sol"}, "openrouter": None}
    monkeypatch.setattr(
        discovery, "offered_model_ids", lambda provider, **_kw: offered.get(provider)
    )
    serving._require_offered("openai", "gpt-fixture-sol", None)  # noqa: SLF001
    # The ``<hosting>/`` spelling ``build_model_spec`` canonicalises is the same model.
    serving._require_offered("openai", "openai/gpt-fixture-sol", None)  # noqa: SLF001
    # A catalogue that cannot be enumerated offline accepts: unlooked is not absent.
    serving._require_offered("openrouter", "anything/at-all", None)  # noqa: SLF001
    with pytest.raises(ModelSelectionRefused) as refused:
        serving._require_offered("openai", "gpt-qa-nope", None)  # noqa: SLF001
    assert refused.value.code == "model_unknown"
    # The ONE sentence the desktop/mobile validator uses for the same code (D3/m-2).
    assert str(refused.value) == "'gpt-qa-nope' is not a model openai serves."


@pytest.mark.asyncio
async def test_a_remote_create_refuses_a_model_the_owner_does_not_offer(
    borrower, tmp_path, monkeypatch
):
    """QA round 1, Q2 on #3554: ``--create --model openai/<unknown>`` applied silently.

    The real borrower runtime and the real relay hop: the relay's ``set_model``
    now asks for ``offered_only``, the runtime refreshes the owner's catalogue
    and refuses an id it does not offer BEFORE switching, and the receipt says
    so in the not-applied form with the way to list what is offered. The valid
    id is the control, and a bare ``set_model`` (the local ``/model`` escape
    hatch) still applies an unlisted id as it always did.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from local_operator.mobile.attach_client import AttachClient
    from local_operator.model.configure import invalidate_model_info_cache
    from local_operator.network import relay
    from local_operator.session.runtime import serving
    from tests.unit.session.runtime.test_daemon_serving_plane import (
        _boot,
        _recording_stream,
    )

    mesh, plain, controller, state = borrower
    invalidate_model_info_cache()
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fixture-unknown-switch")
    monkeypatch.setattr(serving, "_switch_executor_pool", pool)
    calls: list[int] = []
    session, handle, runtime = await _boot(tmp_path, _recording_stream(calls))
    server = relay.RelayServer(
        root=mesh.b.root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    monkeypatch.setattr(server, "_session_record", lambda _sid: runtime.record)
    sid = runtime.record.session_id
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        started_on = f"{session.model.provider}/{session.model.model_id}"
        unknown = await asyncio.to_thread(
            server._set_model_on, sid, {"provider": "openai", "model_id": "gpt-qa-nope"}
        )
        after_unknown = f"{session.model.provider}/{session.model.model_id}"
        known = await asyncio.to_thread(
            server._set_model_on, sid, {"provider": "openai", "model_id": "gpt-fixture-sol"}
        )
        after_known = f"{session.model.provider}/{session.model.model_id}"
        await client.connect(runtime.record, sid)
        await client.set_model("openai", "gpt-qa-local-escape")
        after_bare = f"{session.model.provider}/{session.model.model_id}"
    finally:
        await client.detach()
        runtime.close()
        await session.dispose()
        pool.shutdown(wait=True, cancel_futures=True)
    label = server._own_label()  # noqa: SLF001
    assert unknown == {
        "applied": False,
        "detail": (
            "model: openai/gpt-qa-nope not applied — the session keeps its own model\n"
            "why: 'gpt-qa-nope' is not a model openai serves.\n"
            f"see what {label} offers: run /model in lop there, then create again\n"
            "a newer id than that list: /model openai/<id> in lop there takes it"
        ),
    }, unknown
    assert after_unknown == started_on != "openai/gpt-qa-nope"
    assert known == {"applied": True, "detail": ""}, known
    assert after_known == "openai/gpt-fixture-sol"
    assert after_bare == "openai/gpt-qa-local-escape"


@pytest.mark.parametrize("here", ["build-box", "studio-mini", "gpu-node-07"])
def test_a_cut_refusal_keeps_whole_sentences_and_whole_words(here):
    """#3556 round 1, design D2: the cut landed mid-word, ``it l… To allow it``.

    It also dropped the consent clause the owner is asked to accept. Measured on
    #3542's real refusal shape at three peer-name lengths: the trailing whole
    sentences (consent and lead-in) survive, the opening is cut back to a word
    and keeps its leading clause, and the whole stays within the 400 bound.
    """
    from local_operator.providers.shared_catalogue import device_scope_share_command
    from local_operator.session.runtime.server import bounded_error_message

    owner = "damians-macbook-pro"
    command = device_scope_share_command(
        "d_cf3f8de7a83b3c057fe4082566b08e63", "n_3fbdcd5287633f98ec2e1789"
    )
    head = (
        "catalogue needs a device-scoped share: "
        f"{here} borrows openai from {owner} per session, which runs turns but does not "
        f"share the model list a switch reads. Allowing model switching on {here} is "
        f"{owner}'s decision: it lends openai to every session on {here}. To allow it, "
        f"run on {owner}:"
    )
    kept = bounded_error_message(f"{head}\n`{command}`")
    assert len(head) + len(command) + 3 > 400 >= len(kept), len(kept)
    reason, command_line = kept.split("\n")
    assert command_line == f"`{command}`"
    opening, _, rest = reason.partition("… ")
    assert opening.startswith("catalogue needs a device-scoped share: ")
    # A whole word: what follows the kept opening is a space or punctuation.
    assert head.startswith(opening) and not head[len(opening)].isalnum(), opening
    assert rest == (
        f"Allowing model switching on {here} is {owner}'s decision: it lends openai to "
        f"every session on {here}. To allow it, run on {owner}:"
    ), rest
