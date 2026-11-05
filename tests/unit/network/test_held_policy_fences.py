"""Known restrictions fence every held slot; public continuity is not an outage.

These exercise actual paired relays and the canonical ServingSessionHandle
publisher. No real provider, bearer dump, native refresh delegation or lease
extension is needed to distinguish these security and compatibility failures.
"""

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from local_operator.network.credentials import effective, lifecycle
from local_operator.network.credentials.types import BrokerError, Grant
from tests.unit.network import test_credentials_held_offline as h

mesh = h.mesh
stub_provider = h.stub_provider
idp = h.idp
devices = h.devices
KEY = h.STUB_PROVIDER


@pytest.fixture(autouse=True)
def isolated_helpers(monkeypatch, request):
    from local_operator.network.credentials import github
    from local_operator.session.runtime import serving

    monkeypatch.setattr(github, "_resolve_program", lambda *args, **kwargs: None)
    cf = next(
        plugin
        for plugin in request.config.pluginmanager.get_plugins()
        if hasattr(plugin, "_resolve_real_store")
    )
    real = os.environ.get("TEST_REAL_STORE")
    if real:
        old = cf._resolve_real_store
        try:
            cf._resolve_real_store = lambda: Path(real)
            cf._install_real_store_guard()
        finally:
            cf._resolve_real_store = old
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fixture-held-publisher")
    monkeypatch.setattr(serving, "_switch_executor_pool", pool)
    try:
        yield
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def seed(m, sessions=("sess-A",)):
    h._share(m)
    h._pull(m)
    c = h._client(m)
    grants = []
    for session in sessions:
        grant = c.request_grant_sync(KEY, session_id=session)
        assert isinstance(grant, Grant)
        grants.append(grant)
    return c, grants


@pytest.mark.parametrize("variant", ["two_sessions", "force_then_normal"])
def test_known_policy_not_laundered(mesh, variant):
    sessions = ("sess-A", "sess-B") if variant == "two_sessions" else ("sess-A",)
    c, gs = seed(mesh, sessions)
    other = h._client(mesh)
    other.state.observe(KEY, "consent_withdrawn", retry_after_ms=300000)
    other.state.save()
    h._stop_owner(mesh)
    first = c.request_grant_sync(
        KEY, session_id="sess-A", force_refresh=variant == "force_then_normal"
    )
    second = c.request_grant_sync(KEY, session_id=sessions[-1])
    print(
        "POLICY",
        json.dumps(
            {
                "variant": variant,
                "first_is_grant": isinstance(first, Grant),
                "second_is_grant": isinstance(second, Grant),
                "second_held": getattr(second, "held_offline", False),
                "state_after_first_offline_then_second": c.state.status(KEY),
            }
        ),
    )
    assert isinstance(first, BrokerError) and first.code == "consent_withdrawn"
    assert isinstance(second, BrokerError) and second.code == "consent_withdrawn"
    assert c.grants.peek(KEY, "sess-A") is None
    assert c.grants.peek(KEY, sessions[-1]) is None


def test_persisted_lifecycle_withdrawal(mesh):
    c, gs = seed(mesh)
    scopes = lifecycle._native_contexts(mesh.a.root, KEY)
    assert scopes
    scope = next(iter(scopes))
    link, reason = mesh.a._ensure_link_with_reason(mesh.borrower)
    assert link is not None
    reply = link.request(
        {
            "op": "net_broker",
            "kind": "lifecycle_announce",
            "req": mesh.a._next_relay_req(),
            "network_id": mesh.network_id,
            "from_device": mesh.owner,
            "owner_device": mesh.owner,
            "recipient_device": mesh.borrower,
            "entries": [
                {
                    "credential_key": KEY,
                    "account_scope_ref": scope,
                    "state": "withdrawn",
                    "material_generation": 99,
                    "login_chain_revision": 99,
                }
            ],
        }
    )
    detail = reply.get("detail", reply)
    assert detail.get("kind") == "ack" and detail["entries"][0]["applied"]
    durable = lifecycle.LifecycleStore(mesh.network_id, mesh.b.root).load()
    assert (mesh.owner, KEY, scope) in durable.high_water
    h._stop_owner(mesh)
    result = c.request_grant_sync(KEY, session_id="sess-A")
    print(
        "LIFECYCLE accepted_over_authenticated_link=True durable_withdrawal=True result_is_grant="
        + str(isinstance(result, Grant))
        + " held="
        + str(getattr(result, "held_offline", False))
    )
    if not isinstance(result, BrokerError):
        raise AssertionError("persisted accepted withdrawal ignored")


@pytest.mark.asyncio
async def test_canonical_publisher_no_false_outage(mesh):
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.unit.network.test_owner_return_clears_access_band import _band, _settle
    from tests.unit.session.runtime.test_serving import _AccessSession

    c, gs = seed(mesh, ("sess-1",))
    h._stop_owner(mesh)
    session = _AccessSession(KEY + "/stub-model")
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(mesh.b.root), config_dir=mesh.b.root
    )
    states = []
    published = threading.Event()
    records = []
    fields = {
        "state",
        "effective_state",
        "owner_device",
        "owner_label",
        "observed_at",
        "last_grant_at",
        "last_grant_expires_at",
        "provider",
        "label",
    }

    def capture(phase, sequence=None):
        raw = _band(session).model_dump(mode="json")
        assert set(raw) <= fields, "unexpected public DTO fields require review before logging"
        records.append({"phase": phase, "sequence": sequence, "model_access": raw})

    capture("borrowed_initial")

    def observe_update(update):
        band = _band(session)
        states.append(band.effective_state)
        capture("canonical_publication", getattr(update, "sequence", None))
        if band.effective_state == "owner_unavailable":
            published.set()

    subscription = session._frontend_state_store.subscribe(observe_update)

    # A deterministic observer scheduling barrier exposes an allowed interleaving:
    # let the actual host publisher run before the resolution thread continues.
    # It does not modify the resolve result, network or effective receipt.
    def receipt(root, key):
        if root == str(mesh.b.root.resolve()) and key == KEY:
            access = effective.effective_access(KEY, root=mesh.b.root, mesh=c)
            if access.state == "owner_unavailable":
                published.wait(1)

    remove = effective.add_listener(receipt)
    try:
        result = await asyncio.to_thread(c.request_grant_sync, KEY, session_id="sess-1")
        await _settle(handle)
        print(
            "CANONICAL",
            json.dumps(
                {
                    "served_held": isinstance(result, Grant) and result.held_offline,
                    "published_sequence": states,
                    "final_state": _band(session).effective_state,
                    "outage_observed_by_actual_host": published.is_set(),
                }
            ),
        )
        capture("held_final")
        print(
            "PUBLIC_RECORDS",
            json.dumps(
                {
                    "source_head": os.environ.get("TEST_SOURCE_SHA", "working_delta"),
                    "records": records,
                    "not_captured": ["expired", "recovered", "withdrawn"],
                }
            ),
        )
        assert isinstance(result, Grant)
        assert "owner_unavailable" not in states, "successful held resolve published false outage"
        final = _band(session)
        assert final.effective_state == "effective_borrow"
        assert final.state == "ok"
        assert final.last_grant_at == gs[0].granted_at
        assert final.last_grant_expires_at == gs[0].grant_expires_at_ms / 1000
    finally:
        remove()
        subscription.unsubscribe()
        await handle.dispose()


def test_fresh_authorized_session_does_not_resurrect_other_old_slot(mesh):
    c, _ = seed(mesh, ("sess-A", "sess-B"))
    other = h._client(mesh)
    other.state.observe(KEY, "consent_withdrawn", owner_device=mesh.owner, retry_after_ms=1)
    other.state.save()
    # A real owner success may repair A, but cannot relicense old B. The
    # watermark survives both the retry TTL and that later successful borrow.
    fresh = c.request_grant_sync(KEY, session_id="sess-A")
    assert isinstance(fresh, Grant)
    h._stop_owner(mesh)
    old = c.request_grant_sync(KEY, session_id="sess-B")
    assert isinstance(old, BrokerError)
    assert old.code == "consent_withdrawn"
    held = c.request_grant_sync(KEY, session_id="sess-A")
    assert isinstance(held, Grant) and held.held_offline
    assert held.grant_id == fresh.grant_id


@pytest.mark.parametrize("kind", ["pending", "overflow"])
def test_known_lifecycle_baseline_restriction_cannot_hold(mesh, monkeypatch, kind):
    c, _ = seed(mesh)
    life = lifecycle.LifecycleStore(mesh.network_id, mesh.b.root)
    if kind == "overflow":
        from local_operator.network import store

        store._write_private_json(
            life.path.with_name(lifecycle.LIFECYCLE_OVERFLOW_FILENAME),
            {"reason": lifecycle.NEEDS_AUTHORITATIVE_BASELINE},
        )
    else:
        # Exercise the same process-only safety latch used when a restrictive
        # announcement cannot durably persist. This grants no new permission.
        with lifecycle._VOLATILE_GUARD:
            lifecycle._VOLATILE_BASELINES[life._volatile_key()] = lifecycle._VolatileBaseline(
                pending=True
            )
    try:
        h._stop_owner(mesh)
        denied = c.request_grant_sync(KEY, session_id="sess-A", force_refresh=True)
        assert isinstance(denied, BrokerError) and denied.code == "consent_unverifiable"
        assert c.grants.peek(KEY, "sess-A") is None
    finally:
        with lifecycle._VOLATILE_GUARD:
            lifecycle._VOLATILE_BASELINES.pop(life._volatile_key(), None)


def test_policy_watermark_is_scoped_and_survives_success(tmp_path):
    from local_operator.network.credentials.state import PlacementState

    a = PlacementState("net-A", root=tmp_path / "a")
    a.observe("openai", "consent_withdrawn", owner_device="owner-A", retry_after_ms=1)
    fence = a.held_policy_fence("openai", "owner-A")
    assert fence[0] and fence[1] == "consent_withdrawn"
    a.note_grant("openai", "new-real-owner-grant", owner_device="owner-A")
    a.save()
    assert (
        PlacementState.load("net-A", tmp_path / "a").held_policy_fence("openai", "owner-A") == fence
    )
    assert a.held_policy_fence("other-provider", "owner-A") == ("", "")
    assert a.held_policy_fence("openai", "owner-B") == ("", "")
    assert PlacementState("net-B", root=tmp_path / "a").held_policy_fence("openai") == ("", "")
    assert PlacementState("net-A", root=tmp_path / "b").held_policy_fence("openai") == ("", "")


def test_inactive_namespace_evicts_all_slots_until_fresh_owner(mesh, monkeypatch):
    import dataclasses
    import time

    from local_operator.network import store

    c, grants = seed(mesh, ("sess-A", "sess-B"))
    sibling = h._client(mesh)
    sibling_old = sibling.request_grant_sync(KEY, session_id="sess-C")
    assert isinstance(sibling_old, Grant)
    # A locally owned safe context is NOT borrowed authority from this namespace.
    # This cache-only sentinel is never served as a grant or sent over the wire.
    safe = dataclasses.replace(
        grants[0],
        credential_ref=dataclasses.replace(
            grants[0].credential_ref, owner_device=c.self_device, provider="safe-local"
        ),
    )
    c.grants.put("safe-local", "local", safe)
    h._stop_owner(mesh)
    with store.mutate(mesh.network_id, mesh.b.root) as record:
        member = record.self_member()
        assert member is not None
        member.removed_at = time.time()
        store.save(record, mesh.b.root)
    inactive = c.request_grant_sync(KEY, session_id="sess-A")
    assert isinstance(inactive, BrokerError)
    with store.mutate(mesh.network_id, mesh.b.root) as record:
        member = record.self_member()
        assert member is not None
        member.removed_at = None
        store.save(record, mesh.b.root)
    # Reactivating membership supplies no new owner grant. Neither sibling memory
    # nor a later transport retry is allowed to resurrect old authority.
    for client, slot in ((c, "sess-A"), (c, "sess-B"), (sibling, "sess-C")):
        denied = client.request_grant_sync(KEY, session_id=slot)
        assert isinstance(denied, BrokerError), "reactivation resurrected an old cached grant"
        assert client.grants.peek(KEY, slot) is None
    assert c.grants.peek("safe-local", "local") is safe
    returned = h._restart_owner(mesh, monkeypatch)
    try:
        fresh = c.request_grant_sync(KEY, session_id="sess-A")
        assert isinstance(fresh, Grant) and not fresh.held_offline
        assert fresh.grant_id != grants[0].grant_id
        assert sibling.grants.peek(KEY, "sess-C") is None
    finally:
        returned.stop()
    held = c.request_grant_sync(KEY, session_id="sess-A")
    assert isinstance(held, Grant) and held.held_offline
    assert held.grant_id == fresh.grant_id
    assert held.grant_expires_at_ms == fresh.grant_expires_at_ms
    print(
        "SAFETY inactive_denied=True reactivated_old_slots_refused=True "
        "sibling_fenced=True local_safe_preserved=True fresh_owner_then_hold=True"
    )
