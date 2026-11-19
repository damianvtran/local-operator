"""Per-source pre-auth limits and the first-byte deadline on the mesh relay listener.

Roaming stage 4. Before this, the ONLY pre-auth bound was global
(``DEFAULT_MAX_HANDSHAKES``): one address could open eight connections, send
nothing, and hold every slot for the whole 10 s handshake budget, so a real
member's inbound dial from anywhere else was dropped at the cap. These cells
drive a REAL ``RelayServer`` over real loopback sockets:

* a flood from 127.0.0.1 holds at most the per-source cap, and a member dial from
  a SECOND source address (127.0.0.2, which Linux routes on ``lo``) still gets a
  link while the flood is held open;
* garbage is refused by the handshake as before and gives its slots back;
* a silent connection is closed at the first-byte deadline, while a peer whose
  first byte arrives late-but-inside it still completes;
* connect/close churn from one source is rate-limited without touching another;
* an established link is unaffected by a later flood from its own address.
"""

from __future__ import annotations

import selectors
import socket
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import audit as audit_mod
from local_operator.network import identity, relay
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
    serve_shaped_relay,
)

SECOND_SOURCE = "127.0.0.2"


def _second_source_available() -> bool:
    """Can this host bind a client socket to 127.0.0.2? Linux routes 127/8 on ``lo``;
    macOS only has 127.0.0.1 unless an alias is configured."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((SECOND_SOURCE, 0))
    except OSError:
        return False
    finally:
        probe.close()
    return True


needs_second_source = pytest.mark.skipif(
    not _second_source_available(),
    reason="127.0.0.2 is not bindable on this host (no 127/8 route on lo)",
)


def _connect(host: str, port: int, *, source: str = "127.0.0.1") -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((source, 0))
    sock.settimeout(5.0)
    sock.connect((host, port))
    return sock


def _closed_by_server(socks: list[socket.socket], *, within_s: float) -> set[int]:
    """Indices of ``socks`` the server closed (EOF/reset) within ``within_s``.

    A refused connection gets NO frame, so EOF is the only observable; a held one
    stays silent and never becomes readable in the window.
    """
    closed: set[int] = set()
    deadline = time.monotonic() + within_s
    with selectors.DefaultSelector() as watcher:
        for index, sock in enumerate(socks):
            watcher.register(sock, selectors.EVENT_READ, index)
        while time.monotonic() < deadline and len(closed) < len(socks):
            for key, _events in watcher.select(0.05):
                index = key.data
                if index in closed:
                    continue
                try:
                    data = socks[index].recv(1)
                except OSError:
                    data = b""
                if data == b"":
                    closed.add(index)
                    watcher.unregister(socks[index])
    return closed


def _refusals(server: relay.RelayServer, cause: str) -> list[dict[str, Any]]:
    server.audit.flush()
    return [
        row
        for row in server.audit.tail(limit=500)
        if row.get("event") == "handshake_refused" and row.get("cause") == cause
    ]


def _slots_idle(server: relay.RelayServer) -> bool:
    return (
        server._handshake_slots._value == server.settings.max_handshakes  # noqa: SLF001
        and server._preauth.inflight("127.0.0.1") == 0  # noqa: SLF001
    )


@pytest.fixture()
def lone_relay(root: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Any:
    """One serve-shaped relay on loopback with the settings a cell asks for."""
    overrides: dict[str, Any] = dict(getattr(request, "param", {}) or {})
    root_a = root / "solo"
    server = serve_shaped_relay(
        root_a,
        monkeypatch,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1", **overrides),
        identity=identity.mint(root_a, name="solo-device"),
        audit=audit_mod.AuditLog(root_a),
    )
    host, port = server.bind()
    server.bind_control()
    server.start()
    try:
        yield server, host, port
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Source normalisation and the limiter's own bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("addr", "key"),
    [
        (("127.0.0.1", 5555), "127.0.0.1"),
        (("::ffff:203.0.113.9", 5555, 0, 0), "203.0.113.9"),
        (("2001:db8:1:2:aaaa::1", 5555, 0, 0), "2001:db8:1:2::/64"),
        (("2001:db8:1:2:bbbb::7", 6666, 0, 0), "2001:db8:1:2::/64"),
        (("fe80::1%lo0", 1, 0, 0), "fe80::/64"),
        ("not-an-address", "not-an-address"),
    ],
)
def test_a_source_is_its_ip_with_mapped_v4_folded_and_v6_at_its_slash_64(
    addr: Any, key: str
) -> None:
    assert relay.preauth_source_key(addr) == key


def test_the_per_source_cap_counts_in_flight_and_frees_on_release() -> None:
    limiter = relay.PreauthLimiter(per_source=2)
    assert limiter.admit("a") == ""
    assert limiter.admit("a") == ""
    assert limiter.admit("a") == "handshake_source_cap"
    assert limiter.admit("b") == "", "another source is charged to its own budget"
    limiter.release("a")
    assert limiter.admit("a") == ""


def test_the_rate_bucket_throttles_churn_and_refills_at_its_rate() -> None:
    now = [100.0]
    limiter = relay.PreauthLimiter(per_source=99, burst=5, rate_per_s=2.0, clock=lambda: now[0])
    admitted = 0
    for _ in range(50):
        if limiter.admit("churn") == "":
            admitted += 1
            limiter.release("churn")
    assert admitted == 5, "a burst of churn admits only the bucket's burst"
    assert limiter.admit("churn") == "handshake_source_rate"
    assert limiter.admit("other") == "", "another source has its own bucket"
    now[0] += 1.0  # 2 tokens back
    assert limiter.admit("churn") == ""
    limiter.release("churn")
    assert limiter.admit("churn") == ""
    limiter.release("churn")
    assert limiter.admit("churn") == "handshake_source_rate", "refused attempts cost no debt"


def test_the_source_map_is_bounded_under_a_spray_of_sources() -> None:
    limiter = relay.PreauthLimiter(per_source=3, map_max=64)
    for n in range(5000):
        source = f"198.18.{n // 256}.{n % 256}"
        assert limiter.admit(source) == ""
        limiter.release(source)
    assert limiter.tracked_sources() <= 64


# ---------------------------------------------------------------------------
# Real sockets against a real relay
# ---------------------------------------------------------------------------


@needs_second_source
def test_an_idle_flood_from_one_source_cannot_starve_a_member_dial_from_another(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE ACCEPTANCE CELL. Default settings (8 global, 3 per source): twenty
    silent connections from 127.0.0.1, then a REAL member handshake from
    127.0.0.2 while the flood is still open. Before the fix the flood held all
    eight global slots and this dial was dropped at the cap."""
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    cap = server_a.settings.max_handshakes_per_source
    assert cap < server_a.settings.max_handshakes
    flood = [_connect(host, port) for _ in range(20)]
    try:
        closed = _closed_by_server(flood, within_s=1.0)
        held = len(flood) - len(closed)
        assert held <= cap, f"{held} silent connections from one source held slots (cap {cap})"
        assert len(closed) >= len(flood) - cap

        member = _connect(host, port, source=SECOND_SOURCE)
        link, reason = server_b.dial(
            record.network_id, host=f"{host}:{port}", epoch=record.epoch, connected=member
        )
        assert link is not None, f"a member dial from a second source was starved: {reason}"
        reply = link.request({"op": "net_catalog", "req": 7, "locality": "remote"})
        assert reply is not None and reply["op"] == "ack", reply
        rows = _refusals(server_a, "handshake_source_cap")
        assert rows, "the per-source refusal left no local audit row"
        assert len(rows) < 5, "the per-source refusal is throttled, not one row per drop"
        detail = rows[0]["detail"]
        assert detail["cause"] == "handshake_source_cap"
        assert detail["their_device"] == "" and detail["source"] == "127.0.0.1"
        assert rows[0]["detail"]["their_addr"].startswith("127.0.0.1:")
    finally:
        for sock in flood:
            sock.close()


def test_a_garbage_flood_is_refused_by_the_handshake_and_gives_its_slots_back(
    lone_relay: Any,
) -> None:
    server, host, port = lone_relay
    flood = [_connect(host, port) for _ in range(12)]
    try:
        for sock in flood:
            try:
                sock.sendall(b"\x16\x03\x01 not a hello at all\n")
            except OSError:
                pass  # already dropped at the cap
        closed = _closed_by_server(flood, within_s=5.0)
        # Every one is closed: the admitted ones by the handshake refusing garbage
        # (NOT by the first-byte timer — they sent bytes), the rest at the cap.
        assert len(closed) == len(flood)
        assert net_fixtures.wait_for(lambda: _slots_idle(server), timeout_s=5.0)
        assert not _refusals(server, "first_byte_timeout"), "garbage tripped the first-byte timer"
    finally:
        for sock in flood:
            sock.close()


@pytest.mark.parametrize("lone_relay", [{"first_byte_timeout_s": 1.0}], indirect=True)
def test_a_silent_connection_is_closed_at_the_first_byte_deadline(lone_relay: Any) -> None:
    server, host, port = lone_relay
    started = time.monotonic()
    sock = _connect(host, port)
    try:
        assert _closed_by_server([sock], within_s=4.0) == {0}
        elapsed = time.monotonic() - started
        # 1 s deadline; well under the 10 s handshake budget it used to hold.
        assert 0.8 <= elapsed < 2.5, f"closed after {elapsed:.2f}s"
        assert net_fixtures.wait_for(lambda: _slots_idle(server), timeout_s=2.0)
        rows = _refusals(server, "first_byte_timeout")
        assert rows and rows[0]["detail"]["stage"] == "first_byte", rows
    finally:
        sock.close()


@needs_second_source
def test_a_member_whose_first_byte_is_late_but_inside_the_deadline_still_links(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timer bounds silence, not a handshake: a first byte at ~1 s (default
    deadline 2 s) is followed by the ordinary handshake budget, untouched."""
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    member = _connect(host, port, source=SECOND_SOURCE)
    time.sleep(1.0)
    link, reason = server_b.dial(
        record.network_id, host=f"{host}:{port}", epoch=record.epoch, connected=member
    )
    assert link is not None, f"a late-first-byte member was cut: {reason}"
    assert not _refusals(server_a, "first_byte_timeout")


@needs_second_source
def test_churn_from_one_source_is_rate_limited_and_another_source_is_not(
    lone_relay: Any,
) -> None:
    server, host, port = lone_relay
    burst = int(relay.PREAUTH_RATE_BURST)
    for _ in range(burst * 3):
        _connect(host, port).close()
    assert net_fixtures.wait_for(
        lambda: bool(_refusals(server, "handshake_source_rate")), timeout_s=5.0
    ), "a connect/close churn past the burst was never throttled"
    # The churned source is still throttled (an idle probe is dropped at accept,
    # not held for the first-byte window) while a fresh source is held.
    churned = _connect(host, port)
    other = _connect(host, port, source=SECOND_SOURCE)
    try:
        closed = _closed_by_server([churned, other], within_s=0.5)
        assert 0 in closed, "the churning source was not throttled"
        assert 1 not in closed, "another source was throttled by someone else's churn"
    finally:
        churned.close()
        other.close()


def test_an_established_link_is_unaffected_by_a_later_flood_from_its_own_address(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    flood = [_connect(host, port) for _ in range(20)]
    try:
        _closed_by_server(flood, within_s=0.5)
        for req in range(3):
            reply = link.request({"op": "net_catalog", "req": 100 + req, "locality": "remote"})
            assert reply is not None and reply["op"] == "ack", reply
        assert link.alive
    finally:
        for sock in flood:
            sock.close()
