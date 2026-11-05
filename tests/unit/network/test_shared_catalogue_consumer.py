"""Plain-store consumers use owner metadata, never a copied native credential.

The paired relays/rotating IdP/Codex HTTP fixture is slice A's real transport,
not a replacement controller or record injection. No provider inference runs.
"""
from __future__ import annotations

import dataclasses
import json
import time

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
    assert transport.helpers._lop_network(
        "credential", "share", "openai", "--with", mesh.b.identity.name,
        "--scope", "device",
    ) == 0
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
    assert all(value not in text for value in ["fixture-account", "fixture-stale", "Bearer", "refresh_token"])


@pytest.mark.asyncio
async def test_cold_real_owner_model_set_autofetches_without_manual_refresh(borrower, tmp_path, monkeypatch):
    mesh, plain, controller, state = borrower
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.model.configure import invalidate_model_info_cache
    from tests.unit.session.runtime.test_daemon_serving_plane import _boot, _recording_stream

    from local_operator.model import configure
    original_borrow = configure._borrowed_listing_token
    def no_openai_bearer(provider, **kwargs):
        assert provider != "openai", "metadata must not borrow a bearer"
        return original_borrow(provider, **kwargs)
    monkeypatch.setattr(configure, "_borrowed_listing_token", no_openai_bearer)
    invalidate_model_info_cache()
    calls = []
    session, handle, runtime = await _boot(tmp_path, _recording_stream(calls))
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        await client.connect(runtime.record, session.id)
        await client.set_model("openai", "gpt-fixture-sol")
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


@pytest.mark.asyncio
async def test_explicit_phone_provider_allowlist_does_not_union_shared_sources(borrower):
    mesh, plain, controller, state = borrower
    entries, statuses = await controller.live_catalogue(providers=[])
    assert entries == [] and statuses == {}
    assert state["gets"] == 0 and len(mesh.idp.posts) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("restriction", ["session", "derived", "epoch", "removed", "untrusted", "local_only"])
async def test_current_machine_permission_guards_precede_metadata_io(borrower, restriction):
    mesh, plain, controller, state = borrower
    if restriction in {"session", "derived"}:
        with placement.mutate(mesh.network_id, self_device=mesh.owner, root=mesh.a.root) as doc:
            doc.revoke("openai", mesh.borrower, by=mesh.owner)
            doc.grant("openai", mesh.borrower, by=mesh.owner,
                scope="session" if restriction == "session" else "device",
                provenance="consent:fixture-session-approval" if restriction == "derived" else "")
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
        from local_operator.providers.shared_catalogue import cached
        assert cached(mesh.b.root) is None
        return
    expected = ([], "unsupported") if restriction in {"session", "derived"} else None
    assert await controller.shared_catalogue("openai") == expected
    assert state["gets"] == 0 and len(mesh.idp.posts) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["foreign_owner", "foreign_network", "expired", "bool_epoch", "unknown_field"])
async def test_cache_namespace_and_lifetime_damage_never_authorizes_models(borrower, damage):
    mesh, plain, controller, state = borrower
    await controller.shared_catalogue("openai")
    path, = list(store.network_root(mesh.b.root).joinpath("catalogues").glob("*.json"))
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
    older = dataclasses.replace(record, capabilities=[cap for cap in record.capabilities if cap != CAPABILITY])
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

    monkeypatch.setitem(app.dependency_overrides, models.get_config_manager, lambda: ConfigManager(mesh.b.root))
    monkeypatch.setitem(app.dependency_overrides, models.get_provider_auth_store, lambda: plain)
    monkeypatch.setattr(app.state, "env_config", get_env_config(), raising=False)
    # Like the existing authoring-route tests, do not start unrelated app
    # lifespan services; the real route/dependencies still handle this request.
    client = TestClient(app)
    try:
        response = await asyncio.to_thread(client.get, "/v1/models/?provider=openai")
    finally:
        client.close()
    assert response.status_code == 200
    assert {row["id"] for row in response.json()["result"]["models"]} == {"gpt-fixture-sol"}
    assert state["auth_ok"] and state["account_ok"] and state["gets"] == 1
    assert plain.list_credentials("openai") == []
