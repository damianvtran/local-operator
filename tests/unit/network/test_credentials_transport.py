"""The broker's caller identity comes from the TRANSPORT, over a real link.

WHY TWO REAL RELAYS AND NOT A HANDLER CALL. F1 was exploitable precisely because the
transport passed the frame through unchanged: whatever the sender wrote in
``from_device`` reached the broker verbatim, and the relay's own comments say a frame
is never rewritten on the way in. A test that calls the handler with a forged dict
proves the handler checks the link it is given; only two real relays over real TCP
prove that the id the broker reads IS the one the handshake authenticated — and that
the refusal travels back to the peer as a refusal frame rather than as a closed link
or a served credential.

A SECOND PROPERTY LIVES HERE BECAUSE ONLY THIS RIG CAN SHOW IT: the placement arm is
the one that WRITES. The reviewer's P1c rewrote the owner's holder list on disk through
it, so the file is compared before and after the forged push, and a legitimate pull
from the peer's own identity is the control that says the refusal is about the claimed
identity and not about the arm being unreachable.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import pytest

from local_operator.network import relay
from local_operator.network.credentials import placement as placement_mod
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]

PROVIDER = "openai"


@pytest.fixture()
def broker_pair(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A paired B→A link, with A's own placement row for ``openai`` shared with B.

    A REALLY DECLARES THE ROW (its device id is the owner), so the request reaches a
    broker that would serve an honest caller: without that, an ``identity_mismatch``
    assertion would pass against a device that refuses everything anyway.
    """
    # ``getfixturevalue`` by name rather than a parameter of the same name: the
    # import that makes the fixture reachable would otherwise be shadowed.
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, host, port = pair_devices
    record, _host, _port = _pair(pair_devices, monkeypatch, role="admin")
    owner_device = server_a.identity.device_id
    peer_device = server_b.identity.device_id
    with placement_mod.mutate(record.network_id, server_a.root, self_device=owner_device) as doc:
        doc.declare(
            PROVIDER,
            owner_device=owner_device,
            owner_device_name=server_a.identity.name,
            provider=PROVIDER,
            by=owner_device,
        )
        doc.grant(PROVIDER, peer_device, scope="session", by=owner_device)
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    try:
        yield type(
            "BrokerPair",
            (),
            {
                "link": link,
                "server_a": server_a,
                "server_b": server_b,
                "network_id": record.network_id,
                "owner_device": owner_device,
                "peer_device": peer_device,
                "document": placement_mod.placement_path(record.network_id, server_a.root),
            },
        )
    finally:
        link.close("test")


#: Every ask carries a ``req``: ``PeerLink.request`` WAITS on a reply correlated to
#: the request it sent, and a frame without one returns ``None`` at once — which reads
#: as "the broker never answered" when the truth is that nobody was waiting.
_REQS = itertools.count(901)


def _ask(pair: Any, frame: dict[str, Any]) -> dict[str, Any]:
    req = next(_REQS)
    reply = pair.link.request({"op": "net_broker", "req": req, **frame}, timeout=30.0)
    assert reply is not None, "the broker never answered over the real link"
    assert reply.get("req") == req, reply
    return reply


def _detail(reply: dict[str, Any]) -> dict[str, Any]:
    detail = reply.get("detail")
    assert isinstance(detail, dict), reply
    return detail


@pytest.mark.parametrize("kind", ["grant", "report", "placement"])
def test_a_frame_claiming_the_owners_id_is_refused_over_a_real_link(
    broker_pair: Any, kind: str
) -> None:
    """The reviewer's P1, P1b and P1c at the boundary they were exploitable at.

    B holds ``broker_credential`` (the pairing granted it) and is a member, so the
    chokepoint lets the frame through — which is exactly the reviewer's setup. Every
    arm must answer ``identity_mismatch`` and lend nothing, because the CLAIM is the
    only thing forged: on the wire the frame is well formed and authorised.
    """
    frame: dict[str, Any] = {"kind": kind, "from_device": broker_pair.owner_device}
    if kind == "grant":
        frame.update({"key": PROVIDER, "provider": PROVIDER, "for_session": "sess-1"})
    elif kind == "report":
        frame.update({"key": PROVIDER, "failure": "invalid"})
    else:
        frame.update({"want": "push", "document": {"credentials": []}})

    before = broker_pair.document.read_bytes()
    detail = _detail(_ask(broker_pair, frame))
    assert detail["code"] == "identity_mismatch", detail
    assert broker_pair.document.read_bytes() == before
    # The owner's audit is where the operator looks, and it names the ACTUAL sender
    # rather than the id the frame claimed.
    records = [
        row
        for row in broker_pair.server_a.audit.tail(50, network_id=broker_pair.network_id)
        if row.get("event") == "credential.grant_refused"
    ]
    assert records, "a refused broker frame left no audit record on the owner"
    assert records[-1]["actor"] == broker_pair.owner_device
    assert records[-1]["detail"]["sub"] == broker_pair.peer_device, records[-1]


def test_a_forged_push_cannot_rewrite_the_owners_holder_list_on_disk(broker_pair: Any) -> None:
    """THE REVIEWER'S P1c, byte for byte, through the relay that passes frames through.

    The forged document claims the owner as its author and names a stranger as the only
    holder. Under the first version that list landed on disk, every real borrower was
    cut off, and the stranger was admitted afterwards.
    """
    document = placement_mod.PlacementDocument.load(
        broker_pair.network_id, broker_pair.server_a.root, self_device=broker_pair.owner_device
    )
    entry = document.entry(PROVIDER)
    assert entry is not None
    real_holders = sorted(row.device for row in entry.holders)
    assert real_holders == sorted([broker_pair.owner_device, broker_pair.peer_device])

    forged = {
        "schema": 1,
        "network_id": broker_pair.network_id,
        "epoch": 99,
        "written_by": broker_pair.owner_device,
        "credentials": [
            {
                "key": PROVIDER,
                "provider": PROVIDER,
                "kind": "oauth-rotating",
                "owner_device": broker_pair.owner_device,
                "owner_device_name": "owner-laptop",
                "identity_label": "",
                "holders": [
                    {
                        "device": "d_00000000000000000000000000000009",
                        "scope": "device",
                        "granted_at": 1.0,
                        "granted_by": broker_pair.owner_device,
                    }
                ],
                "doc_rev": 99,
            }
        ],
    }
    before = json.loads(broker_pair.document.read_text(encoding="utf-8"))
    detail = _detail(
        _ask(
            broker_pair,
            {
                "kind": "placement",
                "want": "push",
                "from_device": broker_pair.owner_device,
                "document": forged,
            },
        )
    )
    assert detail["code"] == "identity_mismatch", detail
    after = json.loads(broker_pair.document.read_text(encoding="utf-8"))
    assert after == before, "a forged placement push rewrote the owner's document"


def test_the_peers_own_identity_still_pulls_and_pushes(broker_pair: Any) -> None:
    """THE CONTROL: the same link, the same arm, its OWN id — served.

    Without this, "the forged frame was refused" would be satisfied by a broker that
    refused everything on this rig. The pull returns the document, and a push of a row
    the peer itself owns is merged — so the refusal above is attributable to the claimed
    identity.
    """
    pulled = _detail(
        _ask(
            broker_pair,
            {"kind": "placement", "want": "pull", "from_device": broker_pair.peer_device},
        )
    )
    assert pulled["kind"] == "placement", pulled
    assert PROVIDER in {
        row["key"] for row in pulled["document"]["credentials"] if isinstance(row, dict)
    }

    pushed = _detail(
        _ask(
            broker_pair,
            {
                "kind": "placement",
                "want": "push",
                "from_device": broker_pair.peer_device,
                "document": {
                    "epoch": 5,
                    "credentials": [
                        {
                            "key": "deepseek",
                            "provider": "deepseek",
                            "kind": "oauth-rotating",
                            "owner_device": broker_pair.peer_device,
                            "owner_device_name": broker_pair.server_b.identity.name,
                            "identity_label": "",
                            "holders": [
                                {
                                    "device": broker_pair.owner_device,
                                    "scope": "session",
                                    "granted_at": 1.0,
                                    "granted_by": broker_pair.peer_device,
                                }
                            ],
                            "doc_rev": 1,
                        }
                    ],
                },
            },
        )
    )
    assert pushed["kind"] == "ack", pushed
    document = placement_mod.PlacementDocument.load(
        broker_pair.network_id, broker_pair.server_a.root, self_device=broker_pair.owner_device
    )
    assert document.owner_of("deepseek") == broker_pair.peer_device
    # AND THE OWNER'S OWN ROW SURVIVED THE PEER'S PUSH, which is the local-write rule.
    assert document.owner_of(PROVIDER) == broker_pair.owner_device


def test_a_peer_cannot_be_served_another_of_the_owners_logins(broker_pair: Any) -> None:
    """The same class of bug one field over: the key is the owner's entry, not the claim.

    A holder of ``openai`` that names a DIFFERENT provider in the frame must be answered
    about ``openai`` — the entry the owner's own document carries — rather than being
    served whatever login the frame asked for. ``not_owner`` for a key this device does
    not own is the same refusal an unrelated key gets.
    """
    detail = _detail(
        _ask(
            broker_pair,
            {
                "kind": "grant",
                "key": "anthropic",
                "provider": PROVIDER,
                "from_device": broker_pair.peer_device,
                "for_session": "sess-1",
            },
        )
    )
    assert detail["code"] == "not_owner", detail
