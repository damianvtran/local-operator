"""The passive uplink readings, ``local_offline``, the redial kick and the rows.

Incident 2026-11-18: a Mac on an IPv6-only network dialled an IPv4-only node2
and every surface blamed node2. Nothing here touches the real network: route
readers, clocks and the stop event are all scripted.
"""

from __future__ import annotations

import errno
import socket
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import redial, relay, store, uplink

#: Captured at import, before the directory's autouse fixture pins both readings
#: to ``unknown`` for every other cell (``conftest._uplink_unknown``).
_REAL_READ = uplink.read_uplink
_REAL_ENDPOINT = uplink.endpoint_has_route


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _reader(ipv4: bool | None, ipv6: bool | None) -> Any:
    return lambda family: {"ipv4": ipv4, "ipv6": ipv6}[family]


# -- classification -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("ipv4", "ipv6", "state"),
    [
        (True, True, uplink.DUAL),
        (True, False, uplink.IPV4_ONLY),
        (False, True, uplink.IPV6_ONLY),
        (False, False, uplink.OFFLINE),
        (None, True, uplink.UNKNOWN),
        (True, None, uplink.UNKNOWN),
        (False, None, uplink.UNKNOWN),
        (None, None, uplink.UNKNOWN),
    ],
)
def test_each_family_combination_classifies(ipv4: Any, ipv6: Any, state: str) -> None:
    reading = _REAL_READ(_reader(ipv4, ipv6))
    assert reading == {"state": state, "ipv4": ipv4, "ipv6": ipv6}


def test_a_reader_that_raises_reads_unknown() -> None:
    def broken(_family: str) -> bool:
        raise RuntimeError("synthetic")

    assert _REAL_READ(broken) == uplink.unknown_reading()


def test_route_lookup_sends_nothing_and_maps_errnos(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real reader only ``connect``s a datagram socket; errnos decide the answer."""
    calls: list[str] = []

    class _Sock:
        def __init__(self, family: int, kind: int) -> None:
            assert kind == socket.SOCK_DGRAM  # a datagram connect transmits nothing
            self.family = family

        def connect(self, key: Any) -> None:
            calls.append("connect")
            if self.family == socket.AF_INET6:
                raise OSError(errno.ENETUNREACH, "no route")

        def getsockname(self) -> tuple[str, int]:
            return ("192.0.2.10", 40000)

        def send(self, *_args: Any) -> None:  # pragma: no cover - must never run
            raise AssertionError("the uplink reading must not send")

        def close(self) -> None:
            pass

    monkeypatch.setattr(uplink.socket, "socket", _Sock)
    assert uplink.family_has_route("ipv4") is True
    assert uplink.family_has_route("ipv6") is False
    assert set(calls) == {"connect"}

    class _Denied(_Sock):
        def connect(self, key: Any) -> None:
            raise PermissionError(errno.EPERM, "sandbox")

    monkeypatch.setattr(uplink.socket, "socket", _Denied)
    # Not evidence of a missing route: unknown, never "offline".
    assert uplink.family_has_route("ipv4") is None


# -- per-endpoint route lookups (review R1, R2) ----------------------------------------


class _RouteTable:
    """A fake datagram socket over a scripted route table: no default route.

    Only the listed addresses route (an on-link LAN, loopback); every other
    destination — including the public route keys the device reading uses —
    fails with ``ENETUNREACH``, exactly as a host whose WAN is down.
    """

    def __init__(self, routed: set[str]) -> None:
        self.routed = routed
        self.connects: list[tuple[int, str]] = []
        table = self

        class _Sock:
            def __init__(self, family: int, kind: int) -> None:
                assert kind == socket.SOCK_DGRAM  # a datagram connect transmits nothing
                self.family = family
                self.source = ""

            def connect(self, key: Any) -> None:
                table.connects.append((self.family, key[0]))
                if key[0] not in table.routed:
                    raise OSError(errno.ENETUNREACH, "no route")
                self.source = "127.0.0.1" if key[0].startswith("127.") else "192.168.50.1"

            def getsockname(self) -> tuple[str, int]:
                return (self.source, 40000)

            def send(self, *_args: Any) -> None:  # pragma: no cover - must never run
                raise AssertionError("a route lookup must not send")

            def close(self) -> None:
                pass

        self.socket = _Sock


@pytest.fixture
def no_default_route(monkeypatch: pytest.MonkeyPatch) -> _RouteTable:
    table = _RouteTable({"192.168.50.7", "127.0.0.1", "203.0.113.5"})
    monkeypatch.setattr(uplink.socket, "socket", table.socket)
    return table


def test_lan_and_loopback_peers_have_a_route_without_a_default_route(
    no_default_route: _RouteTable,
) -> None:
    # The device reading says "no route to the internet"...
    assert _REAL_READ()["state"] == uplink.OFFLINE
    # ...yet the kernel routes the LAN and loopback peers (R1's netns repro).
    assert _REAL_ENDPOINT("192.168.50.7:4097") is True
    assert _REAL_ENDPOINT("127.0.0.1:4097") is True
    assert _REAL_ENDPOINT("16.52.146.32:4097") is False
    assert uplink.endpoints_route(["16.52.146.32:4097", "192.168.50.7:4097"], _REAL_ENDPOINT)
    # The lookup is the peer's own address and port, never a route key.
    assert (socket.AF_INET, "192.168.50.7") in no_default_route.connects


def test_loopback_lookup_on_the_real_kernel_is_routed() -> None:
    """The real syscall, no fakes: loopback is routed even in a sealed sandbox."""
    assert _REAL_ENDPOINT("127.0.0.1:4097") is True


def test_ipv4_mapped_literals_are_ipv4(no_default_route: _RouteTable) -> None:
    assert uplink.endpoint_family("[::ffff:203.0.113.5]:7777") == "ipv4"
    assert uplink.endpoint_family("::ffff:203.0.113.5") == "ipv4"
    assert uplink.endpoint_family("[2001:db8::1]:7777") == "ipv6"
    # Looked up over the IPv4 stack at its IPv4 address, where it routes.
    assert _REAL_ENDPOINT("[::ffff:203.0.113.5]:7777") is True
    assert no_default_route.connects[-1] == (socket.AF_INET, "203.0.113.5")


@pytest.mark.parametrize("endpoint", ["node2.example:4097", "", "not an endpoint"])
def test_a_hostname_or_junk_is_never_looked_up(
    endpoint: str, no_default_route: _RouteTable
) -> None:
    assert _REAL_ENDPOINT(endpoint) is None
    assert no_default_route.connects == []


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        ([False, False], False),
        ([False, True], True),
        ([False, None], None),  # one unknown endpoint: not proven
        ([], None),  # nothing to look up: not proven
    ],
)
def test_endpoints_route_needs_every_endpoint_proven(answers: list[Any], expected: Any) -> None:
    script = dict(zip([f"192.0.2.{i}:1" for i in range(len(answers))], answers))
    assert uplink.endpoints_route(list(script), script.__getitem__) is expected


def test_a_lookup_that_raises_is_not_evidence() -> None:
    def broken(_endpoint: str) -> bool:
        raise RuntimeError("synthetic")

    assert uplink.endpoints_route(["192.0.2.1:1"], broken) is None


# -- local_offline precedence --------------------------------------------------------


def _attempt(endpoint: str, detail: str = "connect_failed:OSError") -> Any:
    return relay.CandidateAttempt(
        endpoint=endpoint, connected=False, detail=detail, latency_ms=None
    )


def test_lan_and_loopback_peers_are_not_local_offline_without_a_default_route(
    no_default_route: _RouteTable,
) -> None:
    for endpoint in ("192.168.50.7:4097", "127.0.0.1:4097"):
        attempts = [_attempt(endpoint, "connect_failed:TimeoutError")]
        assert (
            relay.peer_link_state(None, "unreachable", attempts, route_reader=_REAL_ENDPOINT)
            == "address_unreachable"
        )


def test_a_peer_with_no_route_to_any_endpoint_is_local_offline(
    no_default_route: _RouteTable,
) -> None:
    attempts = [_attempt("16.52.146.32:4097"), _attempt("[2001:db8::1]:4097")]
    assert relay.peer_link_state(None, "unreachable", attempts) == "address_unreachable"
    assert (
        relay.peer_link_state(None, "unreachable", attempts, route_reader=_REAL_ENDPOINT)
        == "local_offline"
    )


@pytest.mark.parametrize(
    "endpoints",
    [
        # One endpoint routes: the failure is not ours.
        ["16.52.146.32:4097", "192.168.50.7:4097"],
        # A hostname could resolve anywhere: never local_offline.
        ["node2.example:4097"],
        ["16.52.146.32:4097", "node2.example:4097"],
    ],
)
def test_local_offline_needs_every_endpoint_proven_unroutable(
    endpoints: list[str], no_default_route: _RouteTable
) -> None:
    attempts = [_attempt(endpoint) for endpoint in endpoints]
    assert (
        relay.peer_link_state(None, "unreachable", attempts, route_reader=_REAL_ENDPOINT)
        == "address_unreachable"
    )


def test_refused_and_handshake_states_are_untouched_by_the_reading() -> None:
    refused = [
        _attempt("16.52.146.32:4097", f"{relay.CONNECT_FAILED_PREFIX}ConnectionRefusedError")
    ]
    assert relay.peer_link_state(None, "x", refused, route_reader=lambda e: False) == "relay_down"


# -- redial: advisory, baseline retries kept, per-peer kick ----------------------------


def _loop_server(
    monkeypatch: pytest.MonkeyPatch, routes: Any, clock: _Clock, ticks: int
) -> tuple[Any, list[float]]:
    """A relay whose redial loop runs ``ticks`` fake seconds; returns dial times.

    ``routes`` is the per-endpoint reader (``None`` = a relay with no reader,
    the control that is exactly today's loop) or a callable of the clock.
    """
    server: Any = object.__new__(relay.RelayServer)
    server.root = None
    server.redial = redial.RedialTracker(clock=clock, rand=lambda: 0.5)
    dial_times: list[float] = []
    waits = [0]
    if routes is not None:
        server._route_reader = lambda endpoint: routes(clock.now)

    def wait(delay: float) -> bool:
        waits[0] += 1
        clock.now += delay
        return waits[0] > ticks

    server._stop = SimpleNamespace(wait=wait, is_set=lambda: False)
    server._link_for = lambda device: None
    server._members_named = lambda device: [object()]
    server._redial_candidates = lambda member: ["192.0.2.5:4097"]

    def dial(device_id: str, *, probe_timeout_s: float) -> tuple[None, str]:
        dial_times.append(clock.now)
        return None, "unreachable"

    server._ensure_link_with_reason = dial
    monkeypatch.setattr(store, "list_networks", lambda root: [])
    return server, dial_times


def _run(monkeypatch: pytest.MonkeyPatch, routes: Any, ticks: int = 300) -> list[float]:
    clock = _Clock()
    server, dial_times = _loop_server(monkeypatch, routes, clock, ticks)
    server.redial.seed("d_peer")
    server._redial_loop()
    return [round(at - 1_000.0, 3) for at in dial_times]


@pytest.mark.parametrize("answer", [True, False, None], ids=["routed", "unroutable", "unknown"])
def test_every_peer_keeps_the_baseline_retry_whatever_the_reading(
    monkeypatch: pytest.MonkeyPatch, answer: Any
) -> None:
    """U1: the reading never pauses or skips a dial — a LAN peer (routed), a
    peer with no route to any endpoint, and an unknown reading all dial at
    exactly the control schedule (no reader: today's loop)."""
    control = _run(monkeypatch, None)
    assert len(control) >= 5  # the backoff schedule over 300 s, not one dial
    assert _run(monkeypatch, lambda now: answer) == control


def test_a_route_to_the_peer_appearing_kicks_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """U2: confirmed recovery — the first True lookup after a False one."""
    control = _run(monkeypatch, None)
    recover_at = control[-1] + 3.0  # mid-backoff: the next retry is far off
    kicked = _run(monkeypatch, lambda now: (now - 1_000.0) >= recover_at)
    assert kicked[: len(control)] == control
    # Redialled within one tick of the route appearing, not at the backoff.
    assert kicked[len(control)] == pytest.approx(recover_at + relay.REDIAL_TICK_S, abs=1.0)
    assert kicked[len(control)] < control[-1] + redial.backoff_s(len(control), rand=lambda: 0.5)


def test_a_transition_to_unknown_is_not_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """U2: unroutable -> unknown kicks nothing; the baseline schedule is unchanged."""
    control = _run(monkeypatch, None)
    flip = control[1] + 1.0
    assert _run(monkeypatch, lambda now: False if (now - 1_000.0) < flip else None) == control


def test_kick_keeps_attempts_and_makes_only_that_peer_due() -> None:
    clock = _Clock()
    tracker = redial.RedialTracker(clock=clock, rand=lambda: 0.5)
    tracker.lost("d_a", cause="timeout")
    tracker.failed("d_a", cause="timeout")
    tracker.lost("d_b", cause="timeout")
    assert tracker.due() == []
    tracker.kick("d_a")
    tracker.kick("d_unknown")  # untracked: a no-op
    assert tracker.due() == ["d_a"]
    assert tracker.snapshot("d_a")["attempts"] == 1


# -- the rows ready and doctor print first --------------------------------------------


OFFLINE = {"state": uplink.OFFLINE, "ipv4": False, "ipv6": False}
DUAL = {"state": uplink.DUAL, "ipv4": True, "ipv6": True}
V6_ONLY = {"state": uplink.IPV6_ONLY, "ipv4": False, "ipv6": True}
NO_ROUTE = lambda endpoint: False  # noqa: E731
ROUTED = lambda endpoint: True  # noqa: E731


def _reach(device: str, endpoint: str, detail: str, ok: bool = False) -> dict[str, Any]:
    return {
        "check": "reachability",
        "device_id": device,
        "device_name": "node2",
        "endpoint": endpoint,
        "ok": ok,
        "detail": detail,
    }


def test_offline_device_row_is_first_and_claims_only_the_internet_route() -> None:
    rows = uplink.uplink_checks(
        OFFLINE,
        [_reach("d_n2", "16.52.146.32:4097", "connect_failed:OSError")],
        route_reader=NO_ROUTE,
    )
    assert len(rows) == 1
    assert rows[0]["ok"] is False and rows[0]["code"] == "offline"
    assert rows[0]["detail"] == "This device has no route to the internet."
    assert rows[0]["remedies"] == ["check this device's network"]


def test_offline_device_row_never_shows_while_a_lan_peer_answers(
    no_default_route: _RouteTable,
) -> None:
    rows = uplink.uplink_checks(
        OFFLINE,
        [_reach("d_lan", "192.168.50.7:4097", "ok", ok=True)],
        route_reader=_REAL_ENDPOINT,
    )
    assert rows == []


def test_a_silent_lan_peer_without_a_default_route_isnt_answering(
    no_default_route: _RouteTable,
) -> None:
    rows = uplink.uplink_checks(
        OFFLINE,
        [_reach("d_lan", "192.168.50.7:4097", "connect_failed:TimeoutError")],
        route_reader=_REAL_ENDPOINT,
    )
    assert [row["code"] for row in rows] == ["offline", "peer_not_answering"]
    assert rows[1]["detail"] == "node2 isn't answering from this network"


def test_family_mismatch_names_the_missing_route_and_only_the_peers_addresses() -> None:
    rows = uplink.uplink_checks(
        V6_ONLY,
        [_reach("d_n2", "16.52.146.32:4097", "connect_failed:OSError")],
        route_reader=NO_ROUTE,
    )
    assert [row["code"] for row in rows] == ["ipv6_only", "local_offline"]
    assert rows[0]["ok"] is True and "remedies" not in rows[0]
    assert rows[1]["detail"] == (
        "This device's network has no IPv4 route; node2 has only IPv4 addresses"
    )
    assert rows[1]["remedies"] == ["check this device's network"]
    v4 = {"state": uplink.IPV4_ONLY, "ipv4": True, "ipv6": False}
    rows = uplink.uplink_checks(
        v4, [_reach("d_n2", "[2001:db8::1]:4097", "connect_failed:OSError")], route_reader=NO_ROUTE
    )
    assert rows[1]["detail"] == (
        "This device's network has no IPv6 route; node2 has only IPv6 addresses"
    )


def test_a_single_family_peer_with_its_own_route_is_not_blamed_on_this_device() -> None:
    """R1: v6-only device reading, but the kernel routes this IPv4 LAN peer."""
    rows = uplink.uplink_checks(
        V6_ONLY,
        [_reach("d_lan", "192.168.50.7:4097", "connect_failed:OSError")],
        route_reader=ROUTED,
    )
    assert rows == []


@pytest.mark.parametrize("reading", [DUAL, None, uplink.unknown_reading()])
def test_a_healthy_or_unknown_reading_adds_no_row(reading: Any) -> None:
    """Q-M1: no uplink row unless something is wrong."""
    rows = uplink.uplink_checks(
        reading, [_reach("d_n2", "16.52.146.32:4097", "ok", ok=True)], route_reader=ROUTED
    )
    assert rows == []


def test_a_routed_silent_peer_isnt_answering_and_no_cause_is_named() -> None:
    rows = uplink.uplink_checks(
        DUAL,
        [_reach("d_n2", "16.52.146.32:4097", "connect_failed:TimeoutError")],
        route_reader=ROUTED,
    )
    assert [row["code"] for row in rows] == ["peer_not_answering"]
    assert rows[0]["detail"] == "node2 isn't answering from this network"
    assert "remedies" not in rows[0]
    for word in ("firewall", "allowlist"):
        assert word not in rows[0]["detail"].lower()


@pytest.mark.parametrize(
    ("details", "route"),
    [
        # Resolver and local errors are not a peer staying silent (U3).
        (["connect_failed:gaierror"], ROUTED),
        (["connect_failed:OSError"], ROUTED),
        (["bad_endpoint"], ROUTED),
        # No attempt at all.
        (["not_attempted"], ROUTED),
        # One silent dial beside one resolver error: not every row is evidence.
        (["connect_failed:TimeoutError", "connect_failed:gaierror"], ROUTED),
        # A timeout to an endpoint whose route is unknown: not a routable dial.
        (["connect_failed:TimeoutError"], lambda endpoint: None),
        # The host answered.
        (["connect_failed:ConnectionRefusedError"], ROUTED),
        (["unknown_network"], ROUTED),
    ],
)
def test_isnt_answering_needs_a_timed_out_dial_to_a_routable_endpoint(
    details: list[str], route: Any
) -> None:
    peer_rows = [
        _reach("d_n2", f"16.52.146.{index}:4097", detail) for index, detail in enumerate(details)
    ]
    assert uplink.uplink_checks(DUAL, peer_rows, route_reader=route) == []


def test_ready_renders_the_uplink_rows_first_with_their_remedy() -> None:
    from local_operator.network import readiness

    reach = _reach("d_n2", "16.52.146.32:4097", "connect_failed:OSError")
    checks = uplink.uplink_checks(V6_ONLY, [reach], route_reader=NO_ROUTE) + [reach]
    lines = readiness.render_check_lines(checks)
    assert lines[0] == "ok  uplink: this device has an IPv6 route and no IPv4 route"
    assert lines[1] == (
        "FAIL uplink node2: This device's network has no IPv4 route; "
        "node2 has only IPv4 addresses"
    )
    assert lines[2] == "    - check this device's network"
    assert lines[3].startswith("FAIL reachability node2")


def _doctor(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: dict[str, Any],
    reading: Any,
) -> list[str]:
    from local_operator.network import cli

    monkeypatch.setattr(cli, "_relay_call", lambda *a, **k: payload)
    monkeypatch.setattr(cli, "_uplink_reading", lambda: dict(reading))
    monkeypatch.setattr(relay, "supervision_state", lambda: {"platform": "none"})
    monkeypatch.setattr(relay, "foreign_supervision", lambda: [])
    cli._cmd_doctor(Namespace(peer="", json=False))
    return capsys.readouterr().out.splitlines()


def _payload(endpoint: str, detail: str) -> dict[str, Any]:
    return {
        "checks": [
            {
                "check": "reachability",
                "device_id": "d_n2",
                "endpoint": endpoint,
                "ok": False,
                "detail": detail,
            }
        ]
    }


def test_doctor_puts_the_uplink_row_first(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(uplink, "endpoint_has_route", NO_ROUTE)
    out = _doctor(
        monkeypatch, capsys, _payload("16.52.146.32:4097", "connect_failed:OSError"), OFFLINE
    )
    first = next(index for index, line in enumerate(out) if "uplink" in line)
    reach = next(index for index, line in enumerate(out) if "reachability" in line)
    assert first < reach
    assert out[first] == "FAIL uplink: This device has no route to the internet."
    assert out[first + 1] == "    - check this device's network"


def test_doctor_on_a_healthy_host_prints_no_uplink_row(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-M1: dual-stack and a peer that refused — exactly today's report."""
    monkeypatch.setattr(uplink, "endpoint_has_route", ROUTED)
    out = _doctor(
        monkeypatch,
        capsys,
        _payload("16.52.146.32:4097", "connect_failed:ConnectionRefusedError"),
        DUAL,
    )
    assert not any("uplink" in line for line in out)
    assert any("reachability" in line for line in out)


def test_doctor_with_a_failing_reading_keeps_todays_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.network import cli

    payload = {"checks": [{"check": "identity", "ok": True, "detail": ""}]}
    monkeypatch.setattr(cli, "_relay_call", lambda *a, **k: payload)
    monkeypatch.setattr(uplink, "read_uplink", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(relay, "supervision_state", lambda: {"platform": "none"})
    monkeypatch.setattr(relay, "foreign_supervision", lambda: [])
    cli._cmd_doctor(Namespace(peer="", json=False))
    out = capsys.readouterr().out
    # Exactly today's report: no uplink row at all.
    assert "uplink" not in out
    assert "identity" in out
