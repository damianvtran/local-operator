"""A live link vouches for a peer whose declared address does not answer.

WHY THIS FILE EXISTS. A peer that DIALS IN (B -> A) while the direct address it
advertises is dead -- a NAT'd laptop, a node whose advertised VPC-private
address is not routable from here -- is demonstrably up: its link to A is live
and carries traffic. Yet ``ready`` and ``doctor`` on A both read the dead
address as a fault and tell the operator to check a firewall or restart the
peer's relay, so a working pair reports red. The address is not a fault while
the peer is reachable over its own link; it is information.

THE RIG: two real relays on loopback, real pairing, a real inbound link (B
dials A). The declared address is made SILENT deterministically: the probe's
connect (``socket.socket.connect``) raises ``TimeoutError`` for that one endpoint and
passes every other connect through, so the link, the relays and every other
dial stay real. A kernel trick (a full ``listen(0)`` queue) only drops SYNs on
Linux; macOS completes the connect (#4110 round 1, R1-1), so the silence is
injected at the one seam the probe owns instead. No traffic leaves loopback.

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

#: The silent declared address. Never dialled: :func:`black_hole` answers it
#: with the timeout a filtered address produces.
DEAD = "127.0.0.1:2"


@pytest.fixture
def black_hole(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A declared address whose connect TIMES OUT, on every platform."""
    # At the connect syscall, not create_connection: the probe dials an IP
    # literal on a socket it builds itself (U3b, PR #4260), and
    # create_connection reaches this same method for hostnames.
    real = socket.socket.connect
    host, _, port = DEAD.rpartition(":")
    hits: list[float] = []

    def connect(sock: socket.socket, address: Any) -> None:
        if tuple(address)[:2] == (host, int(port)):
            hits.append(1.0)
            raise TimeoutError("timed out")
        return real(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    yield DEAD
    # Rig check: the probe really asked this address (the row is about it).
    assert hits, "the silent address was never dialled"


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


def test_a_bad_endpoint_with_a_live_link_stays_red(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A malformed declared address is a config fault, not silence (R1-5)."""
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, host, port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    _inbound_link(server_a, server_b, record, host, port)
    peer = server_b.identity.device_id
    _set_peer_endpoints(server_a, record, peer, ["127.0.0.1:not-a-port"])
    try:
        _rc, payload = _ready_json(capsys, "--peer", server_b.identity.name)
        row = _rows(payload, "reachability", device_id=peer)[0]
        assert row["observed"]["outcome"] == "bad_endpoint", row
        assert row["ok"] is False
        assert "covered_by_link" not in row["observed"]
        mine = _doctor_rows(server_a, peer)
        bad = [r for r in mine if r.get("check") == "reachability" and r.get("endpoint")]
        assert bad and all(r["ok"] is False for r in bad), mine
    finally:
        server_b.stop()


def _silent(endpoint: str) -> dict[str, Any]:
    return {
        "check": "reachability",
        "endpoint": endpoint,
        "ok": False,
        "detail": relay.DETAIL_NO_ANSWER,
        "observed": {"outcome": "no_answer"},
        "remedies": ["the firewall advice"],
    }


def test_a_same_lan_private_address_stays_red_and_an_off_lan_one_is_covered() -> None:
    """F9 counter-probe, pinned on the rule itself (R1-2)."""
    own = ["192.168.1.20"]
    same_lan, public = _silent("192.168.1.5:4097"), _silent("203.0.113.9:4097")
    rows = [same_lan, public]
    readiness.mark_covered_by_link(rows, link_live=True, own_addresses=own)
    assert same_lan["ok"] is False and same_lan["remedies"] == ["the firewall advice"]
    assert "covered_by_link" not in same_lan["observed"]
    assert public["ok"] is True and public["remedies"] == []
    assert public["observed"]["covered_by_link"] is True

    untouched = [_silent("203.0.113.9:4097")]
    readiness.mark_covered_by_link(untouched, link_live=False, own_addresses=own)
    assert untouched[0]["ok"] is False and untouched[0]["remedies"] == ["the firewall advice"]


def test_only_silence_is_covered_never_a_config_or_unproven_row() -> None:
    """The family is SILENCE: no_answer and timeouts, in both dialects (R1-5)."""
    covered = [
        _silent("203.0.113.9:1"),
        {
            "check": "reachability",
            "endpoint": "203.0.113.9:2",
            "ok": False,
            "detail": f"{relay.CONNECT_FAILED_PREFIX}TimeoutError",
        },
    ]
    kept = [
        {"check": "reachability", "endpoint": e, "ok": False, "observed": {"outcome": o}}
        for e, o in (
            ("x", "bad_endpoint"),
            ("nohost.invalid:1", "resolve_failed"),
            ("203.0.113.9:3", "no_route"),
            ("203.0.113.9:4", "not_attempted"),
            ("203.0.113.9:5", "refused"),
        )
    ] + [
        {
            "check": "reachability",
            "endpoint": "203.0.113.9:6",
            "ok": False,
            "detail": relay.DETAIL_BAD_ENDPOINT,
        },
        {
            "check": "reachability",
            "endpoint": "203.0.113.9:7",
            "ok": False,
            "detail": f"{relay.CONNECT_FAILED_PREFIX}ConnectionRefusedError",
        },
        {
            "check": "reachability",
            "endpoint": "203.0.113.9:8",
            "ok": False,
            "detail": relay.DETAIL_NOT_ATTEMPTED,
        },
    ]
    readiness.mark_covered_by_link(covered + kept, link_live=True, own_addresses=[])
    assert all(r["ok"] is True for r in covered), covered
    assert all(r["ok"] is False for r in kept), kept
