"""A borrower keeps working on the grant it holds while the owner is unreachable.

THE INCIDENT (2026-11-04 09:26 EST): the operator closed the owner laptop for a
15-minute walk. The node's next request was refused ``owner_offline`` and the
session showed "cannot provide access", although it held a grant the owner had
lent minutes earlier, still inside the lease the owner set. The policy since: the
node keeps running on held credentials and fails closed only on an owner ANSWER
(revoked, denied, removed), never because the owner is unreachable.

The bounds pinned here are the whole safety argument, so every negative is real:
the dial stays first (an answer from the owner always wins), only a transport
outcome is bridged, only inside ``grant_expires_at_ms`` (never extended), never for
a forced refresh, MCP, a removed member, or after any non-transport refusal.

Real path: two relays on loopback (the repo's ``mesh`` fixture: real pairing, the
real ``credential share``/``revoke`` CLI, a rotating loopback IdP that counts
POSTs). The owner is stopped and brought back as a NEW relay on the same port with
the same identity, which is what a laptop lid does to it. Synthetic ids only.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import yaml

from local_operator.network import relay, store
from local_operator.network.credentials import effective
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.types import BrokerError, Grant
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    _lop_network,
    _pull,
    _share,
    mesh,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
    serve_shaped_relay,
)


def _client(mesh: Any) -> MeshCredentialClient:  # noqa: F811 — the fixture's name
    """ONE client for the test, as a session's store holds one for its lifetime."""
    found = MeshCredentialClient.for_this_device(mesh.b.root)
    assert found is not None, "the borrower cannot borrow: the share never reached it"
    return found


def _stop_owner(mesh: Any) -> None:  # noqa: F811 — the fixture's name
    mesh.a.stop()
    assert net_fixtures.wait_for(lambda: not any(link.alive for link in mesh.b.links.values()))


def _restart_owner(mesh: Any, monkeypatch: pytest.MonkeyPatch) -> relay.RelayServer:  # noqa: F811
    """The same device coming back: same identity, same root, same port."""
    server = serve_shaped_relay(
        mesh.a.root,
        monkeypatch,
        settings=relay.NetworkSettings(port=mesh.a.settings.port, listen_address="127.0.0.1"),
        identity=mesh.a.identity,
        audit=mesh.a.audit,
    )
    server.bind()
    server.bind_control()
    server.start()
    # WAIT FOR THE LINK, NOT THE CLOCK (review R1-m4). Both relays redial on their
    # own: the restarted owner's start-up pass seeds every member, the borrower's
    # loop retries the lost link every ``REDIAL_TICK_S``. A request sent before that
    # settles races those dials, and the link it rode could be the one replaced, so
    # the "owner wins on return" assertion intermittently met ``owner_offline``
    # and was served the hold (2 of 4 whole-file runs). The predicate is the
    # condition those assertions are about: each side holds a live link to the other.
    a_id, b_id = mesh.a.identity.device_id, mesh.b.identity.device_id
    assert net_fixtures.wait_for(
        lambda: mesh.b._link_for(a_id) is not None and server._link_for(b_id) is not None
    ), "the owner came back but no link to it settled"
    return server


def _state(mesh: Any, client: MeshCredentialClient) -> effective.EffectiveAccess:  # noqa: F811
    return effective.effective_access(STUB_PROVIDER, root=mesh.b.root, mesh=client)


def _published(
    mesh: Any, client: MeshCredentialClient, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> Any:
    """``model_access`` as the session publishes it (``serving_model_access_here``).

    The borrower's status reader builds its own placement-only client; it is pointed
    at this test's client so the reading is about this process's receipt.
    """
    from local_operator.providers.model_access import serving_model_access_here

    with monkeypatch.context() as patched:
        patched.setattr(
            MeshCredentialClient, "for_this_device", classmethod(lambda *_a, **_k: client)
        )
        return serving_model_access_here(f"{STUB_PROVIDER}/stub-model", config_dir=mesh.b.root)


def _readiness_row(claim: Any) -> dict[str, Any]:
    from types import SimpleNamespace

    from local_operator.network import readiness

    return readiness.selected_model_row(
        SimpleNamespace(device_id="synthetic-peer"),
        {
            "provider": claim.provider,
            "state": claim.effective_state,
            "usable": claim.state == "ok",
            "owner_label": "device-a",
        },
        selector=f"{STUB_PROVIDER}/stub-model",
        session_id="sess-1",
        peer_label="synthetic-peer",
    )


def _hold_end(grant: Grant, cap_s: float) -> float:
    """The borrower-side hold end (ms) for a ``cap_s`` ``offline_hold_s``.

    Measured from the receipt stamp ``GrantCache.put`` set, which is a few ms after
    ``_bounded`` capped ``grant_expires_at_ms`` — so it is never earlier.
    """
    assert grant.received_wall, "the grant was never stamped on receipt"
    return max(grant.grant_expires_at_ms, grant.received_wall * 1000 + cap_s * 1000)


@pytest.fixture
def borrowed(mesh: Any) -> tuple[MeshCredentialClient, Grant]:  # noqa: F811
    _share(mesh)
    _pull(mesh)
    client = _client(mesh)
    grant = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(grant, Grant), grant
    assert _state(mesh, client).state == "effective_borrow"
    return client, grant


def test_the_held_grant_serves_while_the_owner_is_away_and_the_owner_wins_on_return(
    mesh: Any, borrowed: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    client, grant = borrowed
    posts = len(mesh.idp.posts)
    _stop_owner(mesh)

    # K1/A1: inside the owner's lease, the next request is served the SAME grant.
    held = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(held, Grant), held
    assert held.access_token == grant.access_token
    assert held.grant_expires_at_ms == grant.grant_expires_at_ms, "the lease was extended"
    assert held.held_offline is True
    access = _state(mesh, client)
    assert (access.state, access.usable) == ("held_offline", True)
    assert access.last_grant_at == pytest.approx(grant.granted_at), "not a new resolve"
    # What the session publishes, through the readers' own path: coarse ``ok`` (no
    # outage claim) with the honest additive reason, and readiness reads it as ok.
    claim = _published(mesh, client, monkeypatch)
    assert claim is not None and (claim.state, claim.effective_state) == ("ok", "held_offline")
    row = _readiness_row(claim)
    assert row["ok"] is True and row["code"] == "held_offline", row
    assert len(mesh.idp.posts) == posts, "a held grant must not refresh anywhere"
    # The mark is never on the wire: an owner cannot tell a borrower it is "held".
    assert "held_offline" not in held.to_detail()
    assert Grant.from_detail(held.to_detail()).held_offline is False

    # A session that never borrowed holds nothing, and is told the truth.
    other = client.request_grant_sync(STUB_PROVIDER, session_id="sess-never")
    assert isinstance(other, BrokerError) and other.code == "owner_offline", other

    # A2: the owner comes back, and the very next request dials it and is lent fresh.
    owner = _restart_owner(mesh, monkeypatch)
    try:
        fresh = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
        assert isinstance(fresh, Grant), fresh
        assert fresh.held_offline is False
        assert fresh.granted_at > grant.granted_at
        assert _state(mesh, client).state == "effective_borrow"
    finally:
        owner.stop()


def test_the_lease_is_not_extended_past_its_end(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """A3: the owner stays away past the hold -> refused, owner_unavailable.

    Since the 2026-11-13 offline hold the end is the HOLD's, not the grant's: the
    borrower's own ``offline_hold_s`` caps it from receipt (here 2 s, the floor
    ``grant_ttl_s`` allows), whatever longer hold the owner stated.
    """
    (mesh.b.root / "config.yml").write_text(
        yaml.safe_dump(
            {"values": {"network": {"credentials": {"grant_ttl_s": 2.0, "offline_hold_s": 2.0}}}}
        ),
        encoding="utf-8",
    )
    _share(mesh)
    _pull(mesh)
    client = _client(mesh)
    grant = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(grant, Grant), grant
    _stop_owner(mesh)
    if time.time() * 1000 < grant.grant_expires_at_ms - 200:
        assert isinstance(client.request_grant_sync(STUB_PROVIDER, session_id="sess-1"), Grant)
    assert net_fixtures.wait_for(lambda: time.time() * 1000 > _hold_end(grant, 2.0) + 50)
    lapsed = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(lapsed, BrokerError) and lapsed.code == "owner_offline", lapsed
    access = _state(mesh, client)
    assert (access.state, access.usable) == ("owner_unavailable", False)
    claim = _published(mesh, client, monkeypatch)
    assert claim is not None and claim.state == "temporarily_unavailable", claim
    # Dropped, not merely refused once: nothing is held for a later call.
    assert client.grants.peek(STUB_PROVIDER, "sess-1") is None


def test_a_revoke_heard_while_reachable_drops_the_grant_for_good(
    mesh: Any, borrowed: Any, capsys: pytest.CaptureFixture[str]  # noqa: F811
) -> None:
    """A4: the owner's answer wins over a grant with time left, then and afterwards."""
    client, _grant = borrowed
    assert _lop_network("credential", "revoke", STUB_PROVIDER, "--from", mesh.b.identity.name) == 0
    capsys.readouterr()
    refused = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(refused, BrokerError) and refused.code == "not_a_holder", refused
    # And the owner then going away does not resurrect it: the grant had time left,
    # and the hold would serve it if the answer had not dropped it.
    _stop_owner(mesh)
    later = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(later, BrokerError), later
    assert _state(mesh, client).usable is False


def test_a_revoke_made_while_connected_reaches_an_idle_borrower_before_the_owner_sleeps(
    mesh: Any, borrowed: Any  # noqa: F811
) -> None:
    """#3922 Q2: the borrower asks for nothing between the revoke and the owner sleeping.

    ``credential revoke`` also took ``broker_credential`` off B's member row on A,
    so A's authoriser refused B's placement pull with a codeless sentence and the
    revoked document never arrived: B kept its holder row, and once A stopped the
    offline hold served the revoked grant (to the grant end on main, an hour here).
    The pull below is the borrower relay's own cadence call
    (``sync._pull_placement_unless_parked``), with no grant request in between.
    """
    from local_operator.network.credentials import sync as sync_mod

    client, _grant = borrowed
    owner_id = mesh.a.identity.device_id
    assert _lop_network("credential", "revoke", STUB_PROVIDER, "--from", mesh.b.identity.name) == 0
    # Idle: the cadence pulls; nothing asks the owner for a grant.
    sync_mod._pull_placement_unless_parked(
        mesh.b, owner_id, park_codes=sync_mod._POLICY_PULL_REFUSALS
    )
    pulled = _pull(mesh)
    assert pulled.get("refused_codes", {}).get(owner_id) == "not_a_holder", pulled
    assert client.should_borrow(STUB_PROVIDER) is False, "the revoke never reached the borrower"
    # The owner then sleeps. The held grant had an hour left; it must not serve.
    _stop_owner(mesh)
    after = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(after, BrokerError), "a revoked grant was served from the offline hold"
    assert after.code == "not_a_holder", after
    assert client.grants.peek(STUB_PROVIDER, "sess-1") is None
    assert _state(mesh, client).usable is False


def test_a_withdrawn_row_is_re_admitted_by_the_owners_next_document(tmp_path: Any) -> None:
    """The withdrawal deletes rows rather than bumping ``doc_rev`` on a copy the owner
    never saw: the owner's row at the SAME ``doc_rev`` merges again (a capability
    re-granted with no share change), where a local strip would be dropped as a replay.
    Only rows the refusing owner lent to this device are touched."""
    from local_operator.network.credentials import placement as placement_mod

    net, owner, other, me = "n_test", "d_owner", "d_other", "d_me"
    my_root = tmp_path / "me"
    with placement_mod.mutate(net, tmp_path / "owner", self_device=owner) as document:
        for key in ("lent", "not-lent"):
            document.declare(key, owner_device=owner, provider=key, by=owner)
        document.grant("lent", me, scope="device", by=owner)
        sent = document.to_json()
    with placement_mod.mutate(net, tmp_path / "other", self_device=other) as document:
        document.declare("theirs", owner_device=other, provider="theirs", by=other)
        document.grant("theirs", me, scope="device", by=other)
        theirs = document.to_json()
    for incoming, sender in ((sent, owner), (theirs, other)):
        placement_mod.merge_from_peer(
            net, incoming, from_device=sender, self_device=me, root=my_root
        )

    withdrawn = placement_mod.withdraw_rows_lent_by(net, owner, holder_device=me, root=my_root)
    assert withdrawn == ["lent"]
    mine = placement_mod.PlacementDocument.load(net, my_root, self_device=me)
    assert not mine.is_holder("lent", me)
    assert mine.entry("not-lent") is not None, "a row not lent to this device was touched"
    assert mine.is_holder("theirs", me), "another owner's row was touched"
    assert placement_mod.merge_from_peer(
        net, sent, from_device=owner, self_device=me, root=my_root
    ) == ["lent"], "the owner's unchanged row was dropped as a replay"


@pytest.mark.parametrize(
    "code",
    [
        "not_a_holder",
        "revoked",
        "grant_invalid",
        "consent_withdrawn",
        "scope_mismatch",
        "scope_unknown",
        "no_local_credential",
        "epoch_stale",
        "refresh_failed",
        "consent_unverifiable",
    ],
)
def test_any_owner_answer_drops_the_held_grant(
    mesh: Any, borrowed: Any, monkeypatch: pytest.MonkeyPatch, code: str  # noqa: F811
) -> None:
    """A6/A9 and the rest: an answer is authority; only silence is bridged.

    The owner's reply is substituted at the leg-1 control boundary so every code is
    reached; the grant being dropped is the client's own real cache.
    """
    client, _grant = borrowed

    def answered(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"op": "ack", "detail": {"kind": "error", "code": code, "message": ""}}

    # Scoped, so the ``mesh`` fixture's own environment patches stay in place.
    with monkeypatch.context() as patched:
        patched.setattr(relay, "control_request", answered)
        refused = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(refused, BrokerError) and refused.code == code, refused
    # The owner then going silent cannot bring the grant back, though it had time left.
    _stop_owner(mesh)
    after = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    # The known owner answer remains the cause; loss of connectivity cannot
    # launder it into a lesser, retry-only outage (R1-1 / Q1).
    assert isinstance(after, BrokerError) and after.code == code, after
    assert _state(mesh, client).usable is False


def test_a_refusal_another_process_heard_blocks_the_hold(
    mesh: Any, borrowed: Any  # noqa: F811
) -> None:
    """A held grant is never served after a non-transport refusal, from anywhere.

    The relay's pull or a sibling session writes the observation file; this
    process's memory never saw the answer, the durable observation did.
    """
    client, _grant = borrowed
    other = _client(mesh)
    other.state.observe(STUB_PROVIDER, "consent_withdrawn", retry_after_ms=300_000)
    other.state.save()
    _stop_owner(mesh)
    refused = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(refused, BrokerError), refused
    assert _state(mesh, client).usable is False


def test_a_forced_refresh_is_never_served_from_memory(
    mesh: Any, borrowed: Any  # noqa: F811
) -> None:
    client, _grant = borrowed
    _stop_owner(mesh)
    forced = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1", force_refresh=True)
    assert isinstance(forced, BrokerError) and forced.code == "owner_offline", forced


def test_a_removed_member_holds_nothing(mesh: Any, borrowed: Any) -> None:  # noqa: F811
    """A9: this device's own membership row says removed -> no hold, owner or not."""
    client, _grant = borrowed
    _stop_owner(mesh)
    with store.mutate(mesh.network_id, mesh.b.root) as copy:
        me = copy.self_member()
        assert me is not None
        me.removed_at = time.time()
        store.save(copy, mesh.b.root)
    refused = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(refused, BrokerError), refused
    assert _state(mesh, client).usable is False


def test_an_mcp_grant_is_never_held(mesh: Any, borrowed: Any) -> None:  # noqa: F811
    """A rotating MCP token may already be rotated by its owner: never bridged.

    A unit of the hold's guard (MCP keys borrow through ``mcp_bearer``, not this
    fixture's provider share), so it calls the guard directly with a real held grant.
    """
    from local_operator.network.credentials.types import credential_key_for_mcp

    client, grant = borrowed
    key = credential_key_for_mcp("https://mcp.example.test/mcp")
    client.grants.put(key, "sess-1", grant)
    refusal = BrokerError(code="owner_offline", key=key)
    out = client._held_or(refusal, key, "sess-1", force_refresh=False)  # noqa: SLF001
    assert out is refusal


@pytest.mark.asyncio
async def test_the_served_session_band_stays_ok_while_held_and_blocks_at_lease_end(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """The 09:26 shape on the real host publisher (#3427's republish listener).

    A real ``ServingSessionHandle`` publishes the band. With the owner away the
    request that used to flip it to ``owner_unavailable`` is now served the held
    grant, and the band stays ``ok``; once the lease is spent it says the owner is
    unavailable, honestly, rather than serving past it.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from local_operator.session.runtime import serving
    from local_operator.session.runtime.serving import ServingSessionHandle

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fixture-held-band")
    monkeypatch.setattr(serving, "_switch_executor_pool", pool)
    from tests.unit.network.test_owner_return_clears_access_band import _band, _settle
    from tests.unit.session.runtime.test_serving import _AccessSession

    (mesh.b.root / "config.yml").write_text(
        yaml.safe_dump(
            {"values": {"network": {"credentials": {"grant_ttl_s": 3.0, "offline_hold_s": 3.0}}}}
        ),
        encoding="utf-8",
    )
    effective._receipts.clear()
    _share(mesh)
    _pull(mesh)
    client = _client(mesh)
    grant = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(grant, Grant), grant
    posts = len(mesh.idp.posts)
    session = _AccessSession(f"{STUB_PROVIDER}/stub-model")
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(mesh.b.root), config_dir=mesh.b.root
    )
    try:
        assert (_band(session).effective_state, _band(session).state) == (
            "effective_borrow",
            "ok",
        )
        _stop_owner(mesh)
        held = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
        assert isinstance(held, Grant) and held.held_offline, held
        await _settle(handle)
        band = _band(session)
        assert (band.effective_state, band.state) == ("effective_borrow", "ok"), band
        assert len(mesh.idp.posts) == posts
        assert net_fixtures.wait_for(lambda: time.time() * 1000 > _hold_end(grant, 3.0) + 50)
        lapsed = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
        assert isinstance(lapsed, BrokerError) and lapsed.code == "owner_offline", lapsed
        await _settle(handle)
        band = _band(session)
        assert (band.effective_state, band.state) == (
            "owner_unavailable",
            "temporarily_unavailable",
        ), band
    finally:
        await handle.dispose()
        pool.shutdown(wait=True, cancel_futures=True)
