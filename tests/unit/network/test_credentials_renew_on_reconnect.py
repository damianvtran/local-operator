"""Renew on reconnect: an owner's return restarts an IDLE node's offline hold.

THE GAP (#3922 caveat): the 24 h hold restarts only on a fresh grant, and a borrower
got one only when its own session asked while the owner was up. The operator's
promise — "as long as I'm logging in each day it should refresh the remote
credentials" — failed for a node idle through the owner's login. Here the owner's
relay nudges each holder on link-up, the holder's relay re-asks through the ordinary
grant path, and a session adopts that renewal when the owner is away again.

Real path: two relays on loopback (the repo's ``mesh`` fixture), the owner relay
stopped and restarted (a closed and reopened lid), scaled to a 5 s hold. Negative
controls: a revoked share, a panicked network and an expired membership get no
nudge; a flapping link gets one per window; with the link-up hook neutralised the
idle session's hold lapses exactly as before (the behaviour this PR changes).
Also the two observability items: the grant receipt fields in
``placement.state.json`` and ``ready --peer <self>``. Synthetic ids only.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from local_operator.network import store
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import renew_push
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.state import PlacementState
from local_operator.network.credentials.types import BrokerError, Grant
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_held_offline import (
    _client,
    _restart_owner,
    _stop_owner,
)
from tests.unit.network.test_credentials_offline_hold import _write_config
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    _pull,
    _share,
    mesh,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
    serve_shaped_relay,
)


@pytest.fixture
def borrowed(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> tuple[MeshCredentialClient, Grant]:
    """grant_ttl 2 s, hold 5 s on both devices, no jitter, then ONE borrow by sess-1."""
    monkeypatch.setattr(renew_push, "RENEW_JITTER_S", 0.0)
    # The holder floor is 60 s; this rig's owner returns within seconds of the
    # borrow, which no daily login does. Zero it here so the RETURN is what renews.
    monkeypatch.setattr(renew_push, "HOLDER_MIN_INTERVAL_S", 0.0)
    for root in (mesh.a.root, mesh.b.root):
        _write_config(root, grant_ttl_s=2.0, offline_hold_s=5.0)
    _share(mesh)
    _pull(mesh)
    client = _client(mesh)
    grant = client.request_grant_sync(STUB_PROVIDER, session_id="sess-1", model_id="stub-model")
    assert isinstance(grant, Grant), grant
    return client, grant


def _ask(client: MeshCredentialClient) -> Grant | BrokerError:
    return client.request_grant_sync(STUB_PROVIDER, session_id="sess-1", model_id="stub-model")


def _owner_id(mesh: Any) -> str:  # noqa: F811
    return mesh.a.identity.device_id


def test_an_idle_node_is_renewed_when_the_owner_returns(
    mesh: Any, borrowed: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """The acceptance row: idle borrower, owner away and back, hold restarted unasked."""
    client, first = borrowed
    first_end = min(first.hold_until_ms, int(first.received_wall * 1000 + 5_000))
    _stop_owner(mesh)
    # Anything the borrow's own link-up fetched is forgotten: only the RETURN counts.
    renew_push.drop_held(mesh.b)
    returned_at = time.time()
    owner = _restart_owner(mesh, monkeypatch)
    try:
        # NO REQUEST FROM THE BORROWER'S SESSION: only the owner's link-up runs.
        assert net_fixtures.wait_for(
            lambda: renew_push.held_renewal(mesh.b, STUB_PROVIDER, _owner_id(mesh)) is not None
        ), "the owner came back but the idle node was never renewed"
        renewal = renew_push.held_renewal(mesh.b, STUB_PROVIDER, _owner_id(mesh))
        assert renewal is not None and renewal["grant_id"] != first.grant_id
    finally:
        owner.stop()
    assert net_fixtures.wait_for(lambda: not any(link.alive for link in mesh.b.links.values()))

    # Past the FIRST grant's hold: only the renewal's restarted window serves this.
    assert net_fixtures.wait_for(lambda: time.time() * 1000 > first_end + 200, timeout_s=15.0)
    served = _ask(client)
    assert isinstance(served, Grant), f"the renewal did not restart the hold: {served}"
    assert served.held_offline and served.grant_id == renewal["grant_id"]
    assert served.held_until_ms > first_end
    # Capped from the RELAY's receipt (after the owner returned), never re-stamped
    # by the session's adoption: the window ends <= receipt + 5 s.
    assert served.held_until_ms <= int((time.time() + 5.0) * 1000)
    assert served.held_until_ms >= int(returned_at * 1000)

    # No further contact: the renewed window ends honestly.
    assert net_fixtures.wait_for(
        lambda: time.time() * 1000 > served.held_until_ms + 100, timeout_s=15.0
    )
    lapsed = _ask(client)
    assert isinstance(lapsed, BrokerError) and lapsed.code == "owner_offline", lapsed


def test_without_the_renewal_the_idle_session_lapses_as_before(
    mesh: Any, borrowed: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Negative control: the link-up hook neutralised = the pre-PR behaviour."""
    monkeypatch.setattr(renew_push, "on_link_up", lambda *_a, **_k: None)
    client, first = borrowed
    _stop_owner(mesh)
    renew_push.drop_held(mesh.b)
    owner = _restart_owner(mesh, monkeypatch)
    try:
        time.sleep(0.5)
        assert renew_push.held_renewal(mesh.b, STUB_PROVIDER, _owner_id(mesh)) is None
    finally:
        owner.stop()
    assert net_fixtures.wait_for(lambda: not any(link.alive for link in mesh.b.links.values()))
    end = first.received_wall * 1000 + 5_000
    assert net_fixtures.wait_for(lambda: time.time() * 1000 > end + 200, timeout_s=15.0)
    lapsed = _ask(client)
    assert isinstance(lapsed, BrokerError) and lapsed.code == "owner_offline", lapsed


@pytest.fixture
def nudges(mesh: Any, monkeypatch: pytest.MonkeyPatch) -> list[str]:  # noqa: F811
    """Every renew nudge the HOLDER received (its receiver, wrapped)."""
    seen: list[str] = []
    real = renew_push.member_renew
    # Floors from the fixture's own link-ups would hide the first nudge under test.
    for server in (mesh.a, mesh.b):
        getattr(server, "_credential_renew_floors", {}).clear()

    def _spy(server: Any, link: Any, frame: Any) -> Any:
        seen.append(",".join(frame.get("keys") or []))
        return real(server, link, frame)

    monkeypatch.setattr(renew_push, "member_renew", _spy)
    return seen


def _renew(mesh: Any) -> dict[str, Any]:  # noqa: F811
    return renew_push.renew_peer(mesh.a, mesh.network_id, mesh.b.identity.device_id)


def test_a_revoked_share_gets_no_nudge(
    mesh: Any, borrowed: Any, nudges: list[str]  # noqa: F811
) -> None:
    with placement_mod.mutate(
        mesh.network_id, mesh.a.root, self_device=_owner_id(mesh)
    ) as document:
        document.revoke(STUB_PROVIDER, mesh.b.identity.device_id, by=_owner_id(mesh))
    assert _renew(mesh)["outcome"] == "skipped:nothing_lent"
    assert nudges == []


def test_a_panicked_network_gets_no_nudge(
    mesh: Any, borrowed: Any, nudges: list[str]  # noqa: F811
) -> None:
    with store.mutate(mesh.network_id, mesh.a.root) as record:
        record.trust = "untrusted"
        store.save(record, mesh.a.root)
    assert _renew(mesh)["outcome"] == "skipped:untrusted"
    assert nudges == []


def test_an_inactive_membership_gets_no_nudge(
    mesh: Any, borrowed: Any, nudges: list[str]  # noqa: F811
) -> None:
    with store.mutate(mesh.network_id, mesh.a.root) as record:
        member = record.member(mesh.b.identity.device_id)
        assert member is not None
        member.lifecycle = "expired"
        store.save(record, mesh.a.root)
    assert _renew(mesh)["outcome"] == "skipped:inactive_member"
    assert nudges == []


def test_a_flapping_link_gets_one_nudge_per_window(
    mesh: Any, borrowed: Any, nudges: list[str], monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Three link-ups through the real hook, one nudge; past the window, one more."""
    link = mesh.a._link_for(mesh.b.identity.device_id)
    assert link is not None
    for _ in range(3):
        renew_push.on_link_up(mesh.a, link)
    assert net_fixtures.wait_for(lambda: len(nudges) >= 1)
    time.sleep(1.0)  # the other two timers ran and were floored
    assert nudges == [STUB_PROVIDER], nudges
    assert _renew(mesh)["outcome"] == "rate_limited"
    # The floor is per window, not forever: once it passes, the next link-up renews.
    monkeypatch.setattr(renew_push, "RENEW_MIN_INTERVAL_S", 0.0)
    assert _renew(mesh)["outcome"] == "nudged"
    assert len(nudges) == 2


def test_the_holder_asks_only_its_authenticated_owner_for_rows_it_holds(
    mesh: Any, borrowed: Any  # noqa: F811
) -> None:
    """A nudge naming a key the sender does not own, or a forged sender, asks nothing."""
    from types import SimpleNamespace

    a_id, b_id = _owner_id(mesh), mesh.b.identity.device_id
    link = SimpleNamespace(device_id=a_id, network_id=mesh.network_id)
    forged = renew_push.member_renew(
        mesh.b, link, {"from_device": "d_other", "network_id": mesh.network_id, "keys": []}
    )
    assert forged == {"kind": "error", "code": "identity_mismatch"}
    unknown = renew_push.member_renew(
        mesh.b,
        link,
        {"from_device": a_id, "network_id": mesh.network_id, "keys": ["not-shared", 7]},
    )
    assert unknown == {"kind": "renew", "accepted": []}
    # A third member claiming the owner's row: the row names A, so B asks nothing.
    third = SimpleNamespace(device_id="d_third", network_id=mesh.network_id)
    assert renew_push.member_renew(
        mesh.b,
        third,
        {"from_device": "d_third", "network_id": mesh.network_id, "keys": [STUB_PROVIDER]},
    ) == {"kind": "renew", "accepted": []}
    assert b_id


def test_wake_step_renews_surviving_links_once_per_wake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sleep-inclusive jump of >= WAKE_GAP_S between ticks is a wake: renew once."""
    from types import SimpleNamespace

    from local_operator.network.credentials import client as client_mod

    clocks = {"mono": 100.0, "boot": 1000.0}
    monkeypatch.setattr(renew_push.time, "monotonic", lambda: clocks["mono"])
    monkeypatch.setattr(client_mod, "_sleep_inclusive_clock", lambda: clocks["boot"])
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        renew_push,
        "renew_peer",
        lambda _s, net, dev, **_k: calls.append((net, dev)) or {"outcome": "nudged"},
    )

    def _link(network: str, device: str = "d_peer") -> Any:
        return SimpleNamespace(network_id=network, device_id=device, alive=True, closing=False)

    # TWO networks shared with the same member, one link each (review m4), plus
    # another member's link that must not be renewed.
    server = SimpleNamespace(links={1: _link("n1"), 2: _link("n2"), 3: _link("n1", "d_other")})
    assert renew_push.wake_step(server, "d_peer") is None  # first sight: link-up covered it
    clocks["mono"] += 15
    clocks["boot"] += 15
    assert renew_push.wake_step(server, "d_peer") is None and calls == []
    clocks["mono"] += 15
    clocks["boot"] += 15 + 8 * 3600  # slept 8 h
    assert renew_push.wake_step(server, "d_peer") == "nudged,nudged"
    assert calls == [("n1", "d_peer"), ("n2", "d_peer")]
    clocks["mono"] += 15
    clocks["boot"] += 15
    assert renew_push.wake_step(server, "d_peer") is None and len(calls) == 2


# -- observability (mesh lane 64949) -----------------------------------------


def test_note_grant_persists_the_grant_end_and_hold(tmp_path: Any) -> None:
    state = PlacementState("net-synthetic", root=tmp_path)
    state.note_grant(
        "openai",
        "g-1",
        owner_device="d_owner",
        grant_expires_at_ms=1_000,
        hold_until_ms=9_000,
    )
    state.save()
    reloaded = PlacementState.load("net-synthetic", tmp_path)
    row = reloaded._observations["openai"]  # noqa: SLF001 — the persisted row itself
    assert row["last_grant_expires_at_ms"] == 1_000
    assert row["last_grant_hold_until_ms"] == 9_000
    assert row["last_grant_held_offline"] is False
    # A refusal after the grant keeps the receipt: that is when a lapse is dated.
    reloaded.observe("openai", "owner_offline", owner_device="d_owner")
    reloaded.note_held_serve("openai", 8_500)
    row = reloaded._observations["openai"]  # noqa: SLF001
    assert row["last_grant_expires_at_ms"] == 1_000
    assert row["last_grant_held_offline"] is True and row["last_grant_hold_until_ms"] == 8_500


def test_a_held_serve_is_dated_in_the_state_file(mesh: Any, borrowed: Any) -> None:  # noqa: F811
    client, grant = borrowed
    _stop_owner(mesh)
    assert net_fixtures.wait_for(lambda: time.time() * 1000 > grant.grant_expires_at_ms + 100)
    held = _ask(client)
    assert isinstance(held, Grant) and held.held_offline, held
    state = PlacementState.load(mesh.network_id, mesh.b.root)
    row = state._observations[STUB_PROVIDER]  # noqa: SLF001
    assert row["last_grant_expires_at_ms"] == grant.grant_expires_at_ms
    assert row["last_grant_held_offline"] is True
    assert row["last_grant_hold_until_ms"] == held.held_until_ms


def test_ready_for_this_device_answers_locally(mesh: Any) -> None:  # noqa: F811
    from local_operator.network import readiness
    from local_operator.network.types import MeshRefusal

    for token in (mesh.b.identity.device_id, mesh.b.identity.name):
        report = readiness.compose(mesh.b, peer=token)
        kinds = [check.get("check") for check in report["checks"]]
        capabilities = {check.get("capability") for check in report["checks"]}
        # This device's own capability rows (git among them), and no link rows:
        # there is no link to oneself to report on.
        assert "git_identity" in capabilities and "reachability" not in kinds, capabilities
        assert all(
            check.get("device_id") == mesh.b.identity.device_id
            for check in report["checks"]
            if check.get("check") == "readiness"
        )
    with pytest.raises(MeshRefusal) as refused:
        readiness.compose(mesh.b, peer="nobody-called-this")
    assert refused.value.code == "unknown_peer"


# -- review round 1 remediation ------------------------------------------------


def _hold_renewal_on_b(mesh: Any) -> None:  # noqa: F811
    """Owner nudges now; wait until B's relay holds the renewal."""
    for server in (mesh.a, mesh.b):
        getattr(server, "_credential_renew_floors", {}).clear()
    assert _renew(mesh)["outcome"] == "nudged"
    assert net_fixtures.wait_for(
        lambda: renew_push.held_renewal(mesh.b, STUB_PROVIDER, _owner_id(mesh)) is not None
    )


def _renewals(mesh: Any) -> list[str]:  # noqa: F811
    return renew_push.held_keys(mesh.b)


def test_a_lost_nudge_is_retried_within_the_failure_floor(
    mesh: Any, borrowed: Any, nudges: list[str], monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """M1: a nudge with no answer keeps a 30 s floor only, and the tick retries it."""
    b_id = mesh.b.identity.device_id
    link = mesh.a._link_for(b_id, network_id=mesh.network_id)
    assert link is not None
    real = link.request
    lost = {"done": False}

    def _flaky(frame: dict[str, Any], **kwargs: Any) -> Any:
        if frame.get("kind") == "renew" and not lost["done"]:
            lost["done"] = True
            raise ConnectionError("synthetic: the wake re-link dropped mid-nudge")
        return real(frame, **kwargs)

    monkeypatch.setattr(link, "request", _flaky)
    clock = {"now": 10_000.0}
    monkeypatch.setattr(renew_push, "_clock", lambda: clock["now"])
    assert _renew(mesh)["outcome"] == "failed"
    assert nudges == []
    # Flap protection still holds inside the failure floor...
    clock["now"] += renew_push.RENEW_FAILED_RETRY_S - 1
    assert _renew(mesh)["outcome"] == "rate_limited"
    # ...and once it passes, the member's own tick re-sends without any link-up.
    clock["now"] += 2
    assert renew_push.wake_step(mesh.a, b_id) == "nudged"
    assert nudges == [STUB_PROVIDER]
    # Delivered: no further retries, and the full window applies again.
    clock["now"] += renew_push.RENEW_FAILED_RETRY_S + 1
    assert renew_push.wake_step(mesh.a, b_id) is None
    assert _renew(mesh)["outcome"] == "rate_limited"


def test_an_older_peer_is_not_nudged(
    mesh: Any, borrowed: Any, nudges: list[str], monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """N3: without ``credential-renew-push-v1`` on the link, nothing is sent."""
    from local_operator.network import wire

    link = mesh.a._link_for(mesh.b.identity.device_id, network_id=mesh.network_id)
    assert link is not None
    monkeypatch.setattr(
        link, "capabilities", frozenset(link.capabilities) - {wire.CREDENTIAL_RENEW_PUSH_V1}
    )
    assert _renew(mesh)["outcome"] == "skipped:peer_lacks_renew"
    assert nudges == []


@pytest.mark.parametrize("path", ["trust", "panic"])
def test_a_trust_change_drops_the_relay_renewal_at_once(
    mesh: Any, borrowed: Any, path: str  # noqa: F811
) -> None:
    """m1: the holder's own trust change / panic forgets the renewal now, not lazily."""
    _hold_renewal_on_b(mesh)
    assert _renewals(mesh) == [STUB_PROVIDER]
    if path == "trust":
        mesh.b._ctl_trust({"network": mesh.network_id, "trust": "untrusted"})
    else:
        mesh.b._ctl_panic({"network": mesh.network_id})
    assert _renewals(mesh) == [], "the renewed login outlived the trust change"


def test_a_revoke_drops_the_relay_renewal_at_reconcile(
    mesh: Any, borrowed: Any  # noqa: F811
) -> None:
    """m1: a revoke landing through the pull drops the renewal at that reconcile."""
    _hold_renewal_on_b(mesh)
    with placement_mod.mutate(
        mesh.network_id, mesh.a.root, self_device=_owner_id(mesh)
    ) as document:
        document.revoke(STUB_PROVIDER, mesh.b.identity.device_id, by=_owner_id(mesh))
    _pull(mesh)
    assert _renewals(mesh) == [], "the revoked renewal outlived the reconcile"


def test_a_session_that_never_borrowed_is_served_and_the_owner_sees_relay_renew(
    mesh: Any, borrowed: Any  # noqa: F811
) -> None:
    """m2/N3: documented behaviour, pinned; the owner's audit names the renewal."""
    import json

    _hold_renewal_on_b(mesh)
    mesh.a.audit.flush()
    rows = [json.dumps(row) for row in mesh.a.audit.tail(500)]
    assert any(renew_push.RENEW_SESSION in row for row in rows), "owner audit cannot tell"
    _stop_owner(mesh)
    fresh = _client(mesh)
    served = fresh.request_grant_sync(
        STUB_PROVIDER, session_id="sess-never-borrowed", model_id="stub-model"
    )
    assert isinstance(served, Grant) and served.held_offline, served


def test_a_renewal_fetched_before_a_policy_change_is_not_adopted(
    mesh: Any, borrowed: Any  # noqa: F811
) -> None:
    """m3: only a renewal whose fetch-time policy stamp is still current is adopted."""
    import dataclasses

    from local_operator.network.credentials.held_policy import Stamp

    client, grant = borrowed
    current = Stamp("inc", 1, _owner_id(mesh), STUB_PROVIDER, 0)
    stale = dataclasses.replace(current, key_generation=1)  # a deny since the fetch
    renewal = dataclasses.replace(grant, grant_id="g-renewal")
    from local_operator.network.credentials.client import _sleep_inclusive_clock

    renewal.received_wall, renewal.received_boot = time.time(), _sleep_inclusive_clock()
    client._adopt_renewal(STUB_PROVIDER, "sess-m3", (renewal, stale), current)
    assert client.grants.peek(STUB_PROVIDER, "sess-m3") is None
    client._adopt_renewal(STUB_PROVIDER, "sess-m3", (renewal, current), current)
    adopted = client.grants.peek(STUB_PROVIDER, "sess-m3")
    assert adopted is not None and adopted.grant_id == "g-renewal"


def test_ready_prefers_a_peer_that_shares_this_devices_name(mesh: Any) -> None:  # noqa: F811
    """N2: a peer named like this device is still the peer; the id always means self."""
    from local_operator.network import readiness

    b_name = mesh.b.identity.name
    with store.mutate(mesh.network_id, mesh.b.root) as record:
        owner_row = record.member(_owner_id(mesh))
        assert owner_row is not None
        owner_row.name = b_name
        store.save(record, mesh.b.root)
    by_name = readiness.compose(mesh.b, peer=b_name)
    assert any(c.get("check") == "reachability" for c in by_name["checks"]), "answered for self"
    by_id = readiness.compose(mesh.b, peer=mesh.b.identity.device_id)
    assert not any(c.get("check") == "reachability" for c in by_id["checks"])
