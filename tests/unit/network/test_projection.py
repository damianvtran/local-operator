"""The peer catalogue, driven against a REAL relay reply.

WHY THIS FILE EXISTS, AND WHY IT IS NOT A FAKE (QA round 10, Q-R10-1). The
sidebar's remote-row producer (``session/peer_rows.py``) reads
``RelayPeerCatalog``, and that class talks to this device's own relay over the
real control socket. Every test of the producer stood a fake in for the
catalogue — ``_Catalog`` injected through ``catalog=``, or ``_Spy`` replacing
``projection.RelayPeerCatalog`` outright — so the one boundary that can be wrong
was the one boundary nothing crossed. It WAS wrong: ``_call`` handed back the
reply ENVELOPE (``{"op", "req", "detail"}``) while its two readers asked the
envelope for ``peers``/``sessions``. Both were ``None`` on every call, so
``peer_session_rows()`` returned ``()`` on a device whose peer was answering —
the ``⇄`` tier, the per-device heading, the README's screenshot and the
``/resume`` guard all inert, with a green suite.

So the load-bearing test here is the one that stands up TWO real relays, pairs
them through the product's own ceremony, and asks the catalogue for the peer's
row. Nothing between the relay's socket and the row is stubbed: the reply is
framed by ``relay._control_connection``, read by ``relay.control_request``, and
unwrapped by the production code under test.

The second property is a TIME, and it needs its own test because only a slow
peer can exercise it: the catalogue must wait out the relay's own probe budget
(``relay.LISTING_CLIENT_TIMEOUT_S``) rather than the control socket's 5 s
default. A client that gives up first returns the empty answer this file exists
to prevent, one layer down — and no functional test on a loopback peer takes
five seconds, so the deadline is asserted at the seam instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import projection, relay, store, types
from local_operator.session.peer_rows import clear_cache, peer_session_rows
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

SESSION = "8dbb1d07f3a3"

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    """The shared two-relay fixture, under a name this module's tests can take.

    Requested by NAME rather than imported into a test signature: pytest
    registers ``test_relay_e2e``'s ``devices`` fixture here by importing it, and
    a test parameter with the same name as that module-level import is a
    redefinition flake8 refuses (F811). ``test_session_plane.py`` takes the same
    alias for the same reason.
    """
    pair: Devices = request.getfixturevalue("devices")
    return pair


@pytest.fixture(autouse=True)
def _no_cached_rows() -> Any:
    """The producer caches by config root for 20 s; a test must not inherit one."""
    clear_cache()
    yield
    clear_cache()


def _seed(root: Path, session_id: str) -> None:
    """A session directory the catalogue ranks: it has a transcript.

    Written directly rather than through a create, because this file is about
    READING a listing and the cheap way to have one is to write the one file
    ``retention._ACTIVITY_FILES`` ranks on.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")


def _listen(server: relay.RelayServer) -> tuple[str, int]:
    """Give a device a listener, so the OTHER device can dial it."""
    host, port = server.bind()
    server.bind_control()
    server.start()
    return str(host), int(port)


def _dial_to(server: relay.RelayServer, record: Any, host: str, port: int) -> relay.PeerLink:
    link, reason = server.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    return link


# ---------------------------------------------------------------------------
# Q-R10-1 — the producer, against a real reply
# ---------------------------------------------------------------------------


def test_the_catalogue_reads_a_real_relay_reply(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE BLOCKER, AT THE REAL BOUNDARY: two relays, one peer row, no fakes.

    Before the unwrap fix this test failed at the FIRST assertion —
    ``catalog.peers()`` was ``[]`` against a relay that had just answered with
    the peer in it — and ``peer_session_rows()`` was ``()`` behind it, which is
    the empty sidebar QA and UX both photographed.
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    _seed(server_b.root, SESSION)
    _dial_to(server_a, record, host_b, port_b)
    try:
        catalog = projection.RelayPeerCatalog(server_a.root)

        peers = catalog.peers()
        assert [facts.device_id for facts in peers] == [server_b.identity.device_id], (
            "the catalogue read nothing off a peer that is answering — this is the "
            "reply-envelope bug, not a transport failure"
        )
        assert peers[0].reachable is True

        rows = catalog.rows()
        assert [row.session_id for row in rows] == [SESSION]
        assert rows[0].device_id == server_b.identity.device_id
        # The peer's own word for a session it holds with no runtime: the row is
        # real and re-addressed, and the catalogue invents nothing about it.
        assert rows[0].state == "stored", rows[0]

        # ... and the sidebar's producer, through the same live relay: this is
        # the surface QA drove. It is a second read of the same cache the app's
        # `/resume` guard uses, so it is asserted rather than assumed.
        produced = peer_session_rows(server_a.root)
        assert [row.id for row in produced] == [SESSION]
        assert produced[0].locality == "remote"
        assert produced[0].owner_device == server_b.identity.device_id
        assert produced[0].owner_device_name == server_b.identity.name
    finally:
        server_b.stop()


def test_the_catalogue_reports_no_peers_when_its_relay_answers_none(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CONTROL for the test above: a relay with no peer answer is an empty read.

    Without this, "rows appeared" could be satisfied by a catalogue that invents
    a row whenever it holds a relay record. A real relay on this root, no peer
    linked, and both readers empty — which is also the zero-peer frame the
    sidebar must paint on a device that has a network but nobody answering.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    server = relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    server.start()
    try:
        _seed(root, SESSION)
        catalog = projection.RelayPeerCatalog(root)
        assert catalog.peers() == []
        # The LOCAL row is still in the federated answer (it is one listing), so
        # this asserts the peer half rather than the whole list.
        assert [row.device_id for row in catalog.rows() if row.device_id] == []
        assert peer_session_rows(root) == ()
    finally:
        server.stop()


def test_the_catalogue_waits_out_the_relays_own_listing_budget(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client must outwait the server it asked, not the socket's 5 s default.

    Assorted at the seam because no loopback peer can take five seconds: the
    deadline is captured from the call the catalogue makes, and compared with
    the one home for it. On the shipped bug the value here was the control
    socket's default — the same silent-empty answer for a merely slow peer.
    """
    seen: dict[str, Any] = {}

    def _spy(record: Any, op: str, **fields: Any) -> dict[str, Any]:
        seen["op"] = op
        seen.update(fields)
        return {"op": "ack", "req": 1, "detail": {"sessions": [], "peers": {}}}

    monkeypatch.setattr(relay, "control_request", _spy)
    monkeypatch.setattr(
        store,
        "find_own_relay",
        lambda _root=None: types.PeerRecord(
            pid=os.getpid(), control_port=1, control_key="probe-key"
        ),
    )
    projection.RelayPeerCatalog(root).peers()

    assert seen["op"] == projection.OP_PEER_ROWS
    assert seen["timeout"] == relay.LISTING_CLIENT_TIMEOUT_S
    assert seen["timeout"] > 5.0, "the control socket's own default is not a listing budget"


def test_the_catalogue_reads_a_refusal_rather_than_raising(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relay that REFUSES the op is an empty read, never an exception.

    The reader is on the sidebar's poll path, so the contract is "no rows",
    and the envelope's error frame is the shape that reaches it most often
    (a device with a network but no live relay side answers this way).
    """
    monkeypatch.setattr(
        relay,
        "control_request",
        lambda *_args, **_fields: {"op": "error", "req": 1, "message": "no network"},
    )
    monkeypatch.setattr(
        store,
        "find_own_relay",
        lambda _root=None: types.PeerRecord(
            pid=os.getpid(), control_port=1, control_key="probe-key"
        ),
    )
    catalog = projection.RelayPeerCatalog(root)
    assert catalog.peers() == []
    assert catalog.rows() == []
