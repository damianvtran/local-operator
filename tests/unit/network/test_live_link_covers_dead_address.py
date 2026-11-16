"""A live link vouches for a peer whose declared address does not answer.

WHY THIS FILE EXISTS. A peer that DIALS IN (B -> A) while the direct address it
advertises is dead -- a NAT'd laptop, a node whose advertised VPC-private
address is not routable from here -- is demonstrably up: its link to A is live
and carries traffic. Yet ``ready`` and ``doctor`` on A both read the dead
address as a fault and tell the operator to check a firewall or restart the
peer's relay, so a working pair reports red. The address is not a fault while
the peer is reachable over its own link; it is information.

THE RIG: two real relays on loopback, real pairing, a real inbound link (B
dials A), and a loopback BLACK HOLE for the declared address -- a listener
whose one-slot accept queue is already full, so a SYN is dropped and the
connect times out exactly as a filtered address does (no non-loopback traffic).

Controls that must not move:
* no live link -> the dead address keeps its fault and its exact remedy;
* a REFUSED address with a live link -> unchanged (something answered there and
  refused: the address really is wrong, and the existing remedy already says
  "its link is live").
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

from local_operator.network import readiness, relay
from tests.unit.network.test_readiness_link import (
    Devices,
    _await_link_reaped,
    _await_membership_pull,
    _bind_and_start,
    _human,
    _ready_json,
    _rows,
    _set_peer_endpoints,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)


@pytest.fixture
def black_hole() -> Iterator[str]:
    """A loopback address whose connect TIMES OUT: a full one-slot accept queue."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(0)
    endpoint = f"127.0.0.1:{listener.getsockname()[1]}"
    filler = socket.socket()
    filler.settimeout(1.0)
    filler.connect(listener.getsockname())
    # Rig check: a further connect really is dropped, not refused.
    probe = socket.socket()
    probe.settimeout(0.5)
    with pytest.raises(TimeoutError):
        probe.connect(listener.getsockname())
    probe.close()
    try:
        yield endpoint
    finally:
        filler.close()
        listener.close()


def _inbound_link(
    server_a: relay.RelayServer, server_b: relay.RelayServer, record: Any, host: str, port: int
) -> None:
    """B dials A: the link A holds is real and cannot be pinned to a declared address."""
    _bind_and_start(server_b, record)
    peer_link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert peer_link is not None, reason
    link = server_a._link_for(server_b.identity.device_id)  # noqa: SLF001 — the one link seam
    assert link is not None
    _await_membership_pull(link)


def _doctor_rows(server_a: relay.RelayServer, device_id: str) -> list[dict[str, Any]]:
    return [row for row in server_a.doctor()["checks"] if row.get("device_id") == device_id]


def test_a_live_inbound_link_covers_a_dead_declared_address_in_ready_and_doctor(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    black_hole: str,
) -> None:
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, host, port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    _inbound_link(server_a, server_b, record, host, port)
    peer = server_b.identity.device_id
    _set_peer_endpoints(server_a, record, peer, [black_hole])
    try:
        _rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
        reach = _rows(payload, "reachability", device_id=peer)
        assert len(reach) == 1, reach
        row = reach[0]
        assert row["endpoint"] == black_hole
        assert row["observed"]["outcome"] == "no_answer", row
        assert row["ok"] is True, row
        assert row["observed"]["informational"] is True
        assert row["observed"]["covered_by_link"] is True
        assert row["remedies"] == []
        clause = readiness.informational_clause(row)
        assert "reachable over its live link" in clause, clause
        human = _human(capsys, "--peer", server_b.identity.name)
        assert "firewall" not in human, human
        assert "start" not in readiness.reachability_reading(row)

        mine = _doctor_rows(server_a, peer)
        assert not [r for r in mine if r.get("code") == "all_endpoints_unanswered"], mine
        dead = [
            r for r in mine if r.get("check") == "reachability" and r.get("endpoint") == black_hole
        ]
        assert len(dead) == 1 and dead[0]["ok"] is True, mine
        assert dead[0]["observed"]["covered_by_link"] is True
        assert "reachable over its live link" in readiness.informational_clause(dead[0])
    finally:
        server_b.stop()


def test_without_a_live_link_the_dead_address_keeps_its_fault_and_remedy(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    black_hole: str,
) -> None:
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    capsys.readouterr()
    _await_link_reaped(server_a, server_b.identity.device_id)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    peer = server_b.identity.device_id
    _set_peer_endpoints(server_a, record, peer, [black_hole])

    _rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
    row = _rows(payload, "reachability", device_id=peer)[0]
    assert row["ok"] is False
    assert row["observed"]["outcome"] == "no_answer"
    assert "informational" not in row["observed"]
    assert "covered_by_link" not in row["observed"]
    label = server_b.identity.name
    assert row[
        "remedies"
    ] == readiness._reachability_remedies(  # noqa: SLF001 — the one remedy home
        "no_answer", peer_label=label, observed=row["observed"]
    )
    assert "firewall" in row["remedies"][0]

    mine = _doctor_rows(server_a, peer)
    assert [r for r in mine if r.get("code") == "all_endpoints_unanswered"], mine
    reach = [
        r for r in mine if r.get("check") == "reachability" and r.get("endpoint") == black_hole
    ]
    assert reach and reach[0]["ok"] is False and "observed" not in reach[0]


def test_a_refused_address_with_a_live_link_is_unchanged(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, host, port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    _inbound_link(server_a, server_b, record, host, port)
    peer = server_b.identity.device_id
    _set_peer_endpoints(server_a, record, peer, ["127.0.0.1:1"])
    try:
        _rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
        row = _rows(payload, "reachability", device_id=peer)[0]
        assert row["observed"]["outcome"] == "refused"
        assert row["ok"] is False
        assert "covered_by_link" not in row["observed"]
        assert "its link is live" in row["remedies"][0]

        mine = _doctor_rows(server_a, peer)
        refused = [
            r
            for r in mine
            if r.get("check") == "reachability" and r.get("endpoint") == "127.0.0.1:1"
        ]
        assert refused and refused[0]["ok"] is False
    finally:
        server_b.stop()
