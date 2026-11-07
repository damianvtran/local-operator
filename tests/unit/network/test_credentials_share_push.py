"""A ``credential share`` reaches a DEFAULT-INVITED borrower, over two real relays.

WHY THIS FILE EXISTS (mesh perf matrix, share-propagation finding, 5/5 >120 s). After
``lop network invite --role drive`` and ``lop network credential share … --with B``, B
never received the share: B's cadence gated its placement pull on its OWN admission-time
``self_capabilities`` (no ``broker_credential`` — ``drive`` does not carry it), while the
share grants that capability only on the OWNER's row for B, which is the row the owner's
authoriser actually reads. A manual pull worked in one round trip, so the gate was the
whole defect.

The four properties pinned here, each failing on the pre-fix build:

* the cadence pulls past the stale self-view, so the share arrives on B's own tick;
* the share verb pushes the new row to B, so it lands before the command returns;
* an UNSHARED default-invited device is still refused by the owner, and the refusal
  parks B's cadence instead of becoming a 15 s refusal loop on the owner's audit log;
* a borrower build without the push is told nothing new and still converges by pull.

Fixtures are the real-link suite's: two serve-shaped relays on loopback, the shipped
parser, a real ``AuthStore`` on the owner. The ambient config root is the OWNER's
(``AuthStore`` reads it); the borrower is driven only through root-explicit APIs.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from local_operator.network import store, wire
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import sync as cred_sync
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.types import Grant
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    RotatingIdP,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import _lop_network
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)


@pytest.fixture()
def drive_mesh(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Owner A (admin, relay up, a LIVE login) and borrower B admitted with ``--role drive``.

    Both relays listen and know each other's address, so either side can dial: the push
    is A dialling B, the cadence pull is B dialling A.
    """
    request.getfixturevalue("stub_provider")
    stub_idp: RotatingIdP = request.getfixturevalue("idp")
    server_a, server_b, host, port = request.getfixturevalue("devices")
    record, _host, _port = _pair((server_a, server_b, host, port), monkeypatch, role="drive")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    host_b, port_b = server_b.bind()
    server_b.bind_control()
    with store.mutate(record.network_id, server_b.root) as copy:
        owner_row = copy.member(server_a.identity.device_id)
        assert owner_row is not None
        owner_row.endpoints = [f"{host}:{port}"]
        store.save(copy, server_b.root)
    with store.mutate(record.network_id, server_a.root) as copy:
        borrower_row = copy.member(server_b.identity.device_id)
        assert borrower_row is not None
        borrower_row.endpoints = [f"{host_b}:{port_b}"]
        store.save(copy, server_a.root)
    from local_operator.providers.auth_store import AuthStore

    auth = AuthStore(config_dir=server_a.root)
    auth.upsert_credential(
        STUB_PROVIDER,
        {
            "type": "oauth",
            "access": "owner-live-access",
            "expires": int(time.time() * 1000) + 3_600_000,
            "refresh": stub_idp.current_refresh,
            "email": "owner@example.test",
        },
    )
    auth.close()
    yield type(
        "DriveMesh",
        (),
        {
            "a": server_a,
            "b": server_b,
            "network_id": record.network_id,
            "owner": server_a.identity.device_id,
            "borrower": server_b.identity.device_id,
            "idp": stub_idp,
        },
    )
    server_b.stop()


def _start_b(mesh: Any) -> None:
    mesh.b.start()


def _b_holds(mesh: Any) -> bool:
    document = placement_mod.PlacementDocument.load(
        mesh.network_id, mesh.b.root, self_device=mesh.borrower
    )
    return document.is_holder(STUB_PROVIDER, mesh.borrower)


def _share_json(mesh: Any, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    capsys.readouterr()
    code = _lop_network(
        "credential", "share", STUB_PROVIDER, "--with", mesh.b.identity.name, "--json"
    )
    assert code == 0
    return json.loads(capsys.readouterr().out)


def _refusals_on_owner(mesh: Any) -> int:
    mesh.a.audit.flush()
    return sum(
        1
        for row in mesh.a.audit.tail(2_000)
        if row.get("event") == "authorisation_refused"
        and row.get("actor") == mesh.borrower
        and (row.get("detail") or {}).get("op") == "net_broker"
    )


def test_the_default_invite_really_lacks_the_capability_on_its_own_row(drive_mesh: Any) -> None:
    """The precondition the defect rests on, asserted so the cells below are not vacuous."""
    from local_operator.network import definitions

    _start_b(drive_mesh)
    record = store.load(drive_mesh.network_id, drive_mesh.b.root)
    assert "broker_credential" not in record.self_capabilities
    assert (
        definitions.unholdable_capability(drive_mesh.b, drive_mesh.owner, "net_broker")
        == "broker_credential"
    )


def test_a_default_invite_share_arrives_on_the_borrowers_own_cadence(
    drive_mesh: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Push disabled: B's tick step alone must deliver the share (pre-fix: never)."""
    # Imported here, not at module level, so the pre-fix build (no such module) fails
    # this cell on the cadence assertion below rather than at collection.
    try:
        from local_operator.network.credentials import share_push
    except ImportError:
        pass
    else:
        monkeypatch.setattr(
            share_push,
            "push_share",
            lambda *a, **k: {"delivered": False, "outcome": "skipped:test_disabled"},
        )
    _start_b(drive_mesh)
    payload = _share_json(drive_mesh, capsys)
    # ``.get``: the pre-fix build has no push and no ``delivery`` field, and this cell
    # must fail there on the CADENCE, which is the defect it pins.
    assert payload.get("delivery", {}).get("delivered") is not True
    assert not _b_holds(drive_mesh), "the share arrived without the cadence under test"

    cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)

    assert _b_holds(drive_mesh), "B's cadence never pulled the share past its stale self-view"
    client = MeshCredentialClient.for_this_device(drive_mesh.b.root)
    assert client is not None and client.should_borrow(STUB_PROVIDER)


def test_a_share_lands_within_one_round_trip_of_the_push(
    drive_mesh: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B's cadence is switched off: only the push can deliver, before the verb returns."""
    monkeypatch.setattr(
        cred_sync,
        "_pull_past_stale_self_view",
        lambda *a, **k: "skipped:test_disabled",
        raising=False,
    )
    _start_b(drive_mesh)
    started = time.monotonic()
    payload = _share_json(drive_mesh, capsys)
    elapsed = time.monotonic() - started

    assert payload["delivery"] == {"delivered": True, "outcome": "pushed"}, payload
    # No wait, no tick: the row is on B's disk when the command returns.
    assert _b_holds(drive_mesh)
    # A real borrow through it (the owner still decides; this is not a local grant).
    grant = MeshCredentialClient.for_this_device(drive_mesh.b.root)
    assert grant is not None
    served = grant.request_grant_sync(STUB_PROVIDER, session_id="sess-1")
    assert isinstance(served, Grant), served
    # A loose structural bound, not a latency budget: well under one 15 s tick.
    assert elapsed < 10.0, elapsed


def test_the_human_receipt_says_the_share_was_delivered(
    drive_mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    _start_b(drive_mesh)
    capsys.readouterr()
    assert (
        _lop_network("credential", "share", STUB_PROVIDER, "--with", drive_mesh.b.identity.name)
        == 0
    )
    assert f"delivered to {drive_mesh.b.identity.name}: now" in capsys.readouterr().out


def test_an_unshared_default_invite_is_still_refused_and_parks_its_cadence(
    drive_mesh: Any,
) -> None:
    """No share: the owner refuses B's pull at its chokepoint; B learns nothing and backs off."""
    _start_b(drive_mesh)
    before = _refusals_on_owner(drive_mesh)

    first = cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
    assert first == "skipped:no_broker_credential"
    assert net_fixtures.wait_for(lambda: _refusals_on_owner(drive_mesh) == before + 1)
    assert not _b_holds(drive_mesh)
    assert MeshCredentialClient.for_this_device(drive_mesh.b.root) is None
    # And the OWNER refuses a borrow outright, whatever B believes: the ungated pull
    # widened nothing, the owner's row for B is still the enforcement point.
    asker = MeshCredentialClient.for_relay(drive_mesh.b)
    assert asker is not None
    reply = asker._ask_owner_directly(  # noqa: SLF001 — the leg-2 seam, by hand
        drive_mesh.owner,
        {
            "op": "net_broker",
            "kind": "grant",
            "network_id": drive_mesh.network_id,
            "from_device": drive_mesh.borrower,
            "key": STUB_PROVIDER,
        },
    )
    assert isinstance(reply, dict) and reply.get("op") == "error", reply
    assert "access" not in json.dumps(reply)
    before = _refusals_on_owner(drive_mesh)

    # Parked: the next ticks ask nothing, so the owner's audit log does not grow.
    for _ in range(3):
        assert (
            cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
            == "skipped:no_broker_credential"
        )
    time.sleep(0.5)
    assert _refusals_on_owner(drive_mesh) == before, "a parked cadence asked the owner again"


def test_a_share_push_clears_the_parked_refusal(
    drive_mesh: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    _start_b(drive_mesh)
    cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
    assert drive_mesh.owner in cred_sync._owner_refusals(drive_mesh.b)  # noqa: SLF001
    assert _share_json(drive_mesh, capsys)["delivery"]["delivered"] is True
    assert drive_mesh.owner not in cred_sync._owner_refusals(drive_mesh.b)  # noqa: SLF001


def test_a_borrower_without_the_push_is_not_pushed_to_and_still_converges(
    drive_mesh: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mixed versions: B's build predates ``credential-share-push-v1``.

    The owner must not send the frame (an old B would refuse it by name or merge it
    through the legacy path), must say the share is not yet delivered, and B's own pull
    must still converge.
    """
    from local_operator.network.credentials import share_push

    monkeypatch.setattr(
        wire,
        "LINK_CAPABILITIES",
        tuple(c for c in wire.LINK_CAPABILITIES if c != wire.CREDENTIAL_SHARE_PUSH_V1),
    )
    _start_b(drive_mesh)
    sent: list[dict[str, Any]] = []
    from local_operator.network import relay as relay_mod

    original = relay_mod.PeerLink.request

    def _record(self: Any, frame: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        sent.append(dict(frame))
        return original(self, frame, *args, **kwargs)

    monkeypatch.setattr(relay_mod.PeerLink, "request", _record)
    payload = _share_json(drive_mesh, capsys)
    assert payload["delivery"] == {
        "delivered": False,
        "outcome": "skipped:peer_lacks_share_push",
    }, payload
    assert not any(f.get("origin") == share_push.SHARE_PUSH_ORIGIN for f in sent)
    assert not _b_holds(drive_mesh)
    cred_sync.credentials_sync_step(drive_mesh.b, drive_mesh.owner)
    assert _b_holds(drive_mesh)


# ---------------------------------------------------------------------------
# The receiver's narrowing, without a link: what a share push may NOT do
# ---------------------------------------------------------------------------


class _Link:
    def __init__(self, device_id: str, network_id: str) -> None:
        self.device_id = device_id
        self.network_id = network_id
        self.capabilities: tuple[str, ...] = (wire.CREDENTIAL_SHARE_PUSH_V1,)
        self.local_capabilities: tuple[str, ...] = (wire.CREDENTIAL_SHARE_PUSH_V1,)


def _frame(network_id: str, sender: str, holder: str, owner: str = "") -> dict[str, Any]:
    from local_operator.network.credentials import share_push

    return {
        "op": "net_broker",
        "kind": "placement",
        "origin": share_push.SHARE_PUSH_ORIGIN,
        "from_device": sender,
        "document": {
            "schema": placement_mod.PLACEMENT_SCHEMA,
            "network_id": network_id,
            "epoch": 0,
            "written_by": sender,
            "credentials": [
                {
                    "key": STUB_PROVIDER,
                    "owner_device": owner or sender,
                    "provider": STUB_PROVIDER,
                    "kind": "oauth-rotating",
                    "doc_rev": 1,
                    "holders": [{"device": holder, "scope": "device"}],
                }
            ],
        },
    }


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("third_party_holder", "not_a_holder"),
        ("third_party_owner", "not_owner"),
        ("not_negotiated", "unsupported"),
    ],
)
def test_the_receiver_refuses_what_a_share_push_may_not_carry(
    drive_mesh: Any, case: str, expected: str
) -> None:
    from local_operator.network.credentials import share_push

    _start_b(drive_mesh)
    link = _Link(drive_mesh.owner, drive_mesh.network_id)
    holder = drive_mesh.borrower
    owner = ""
    if case == "third_party_holder":
        holder = "device-somebody-else"
    elif case == "third_party_owner":
        owner = "device-somebody-else"
    else:
        link.capabilities = ()
    detail = share_push.member_share_push(
        drive_mesh.b, link, _frame(drive_mesh.network_id, drive_mesh.owner, holder, owner)
    )
    assert detail.get("kind") == "error" and detail.get("code") == expected, detail
    assert not _b_holds(drive_mesh)
