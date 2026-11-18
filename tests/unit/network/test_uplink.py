"""The passive uplink reading, ``local_offline``, the redial pause/kick and the rows.

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

#: Captured at import, before the directory's autouse fixture pins the reading
#: to ``unknown`` for every other cell (``conftest._uplink_unknown``).
_REAL_READ = uplink.read_uplink


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


# -- local_offline precedence --------------------------------------------------------


def _attempt(endpoint: str, detail: str = "connect_failed:OSError") -> Any:
    return relay.CandidateAttempt(
        endpoint=endpoint, connected=False, detail=detail, latency_ms=None
    )


V6_ONLY = {"state": uplink.IPV6_ONLY, "ipv4": False, "ipv6": True}


def test_local_offline_wins_over_address_unreachable_for_the_peers_family() -> None:
    attempts = [_attempt("16.52.146.32:4097")]
    assert relay.peer_link_state(None, "unreachable", attempts) == "address_unreachable"
    assert (
        relay.peer_link_state(None, "unreachable", attempts, uplink_reading=V6_ONLY)
        == "local_offline"
    )


@pytest.mark.parametrize(
    ("endpoints", "reading"),
    [
        # The peer also advertises a family this device HAS a route in.
        (["16.52.146.32:4097", "[2001:db8::1]:4097"], V6_ONLY),
        # A hostname could resolve to either family: not proven.
        (["node2.example:4097"], V6_ONLY),
        # The family is unknown: no claim.
        (["16.52.146.32:4097"], {"state": "unknown", "ipv4": None, "ipv6": True}),
        # No reading at all (fail-safe).
        (["16.52.146.32:4097"], None),
    ],
)
def test_local_offline_needs_a_proven_missing_route(endpoints: list[str], reading: Any) -> None:
    attempts = [_attempt(endpoint) for endpoint in endpoints]
    assert (
        relay.peer_link_state(None, "unreachable", attempts, uplink_reading=reading)
        == "address_unreachable"
    )


def test_offline_device_marks_hostname_peers_local_offline_too() -> None:
    offline = {"state": uplink.OFFLINE, "ipv4": False, "ipv6": False}
    attempts = [_attempt("node2.example:4097")]
    assert (
        relay.peer_link_state(None, "unreachable", attempts, uplink_reading=offline)
        == "local_offline"
    )


def test_refused_and_handshake_states_are_untouched_by_the_reading() -> None:
    offline = {"state": uplink.OFFLINE, "ipv4": False, "ipv6": False}
    refused = [
        _attempt("16.52.146.32:4097", f"{relay.CONNECT_FAILED_PREFIX}ConnectionRefusedError")
    ]
    assert relay.peer_link_state(None, "x", refused, uplink_reading=offline) == "relay_down"


# -- redial pause and kick -----------------------------------------------------------


def _loop_server(
    monkeypatch: pytest.MonkeyPatch, readings: list[Any], clock: _Clock
) -> tuple[Any, list[float], list[str]]:
    server: Any = object.__new__(relay.RelayServer)
    server.root = None
    server.redial = redial.RedialTracker(clock=clock, rand=lambda: 0.5)
    waits: list[float] = []
    dials: list[str] = []
    script = iter(readings)

    def reader() -> Any:
        value = next(script)
        if isinstance(value, Exception):
            raise value
        return value

    server._uplink_reader = reader

    def wait(delay: float) -> bool:
        waits.append(delay)
        clock.now += delay
        return len(waits) > len(readings)

    server._stop = SimpleNamespace(wait=wait, is_set=lambda: False)
    server._link_for = lambda device: None
    server._members_named = lambda device: [object()]
    server._redial_candidates = lambda member: ["192.0.2.5:4097"]

    def dial(device_id: str, *, probe_timeout_s: float) -> tuple[None, str]:
        dials.append(device_id)
        return None, "unreachable"

    server._ensure_link_with_reason = dial
    monkeypatch.setattr(store, "list_networks", lambda root: [])
    return server, waits, dials


OFFLINE = {"state": uplink.OFFLINE, "ipv4": False, "ipv6": False}
DUAL = {"state": uplink.DUAL, "ipv4": True, "ipv6": True}


def test_redial_pauses_while_offline_and_kicks_on_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    server, waits, dials = _loop_server(monkeypatch, [OFFLINE, OFFLINE, DUAL], clock)
    server.redial.lost("d_peer", cause="timeout")
    # Push the retry far out, as a long outage's backoff would.
    for _ in range(6):
        server.redial.failed("d_peer", cause="timeout")
    attempts_before = server.redial.snapshot("d_peer")["attempts"]
    next_retry = server.redial.snapshot("d_peer")["next_retry_at"]
    server._redial_loop()
    # First wait is the ordinary tick; the two offline readings re-read on the
    # short bounded cadence and dial nothing.
    assert waits[:3] == [
        relay.REDIAL_TICK_S,
        relay.UPLINK_OFFLINE_TICK_S,
        relay.UPLINK_OFFLINE_TICK_S,
    ]
    # The recovered reading kicked the peer: dialled at once, long before its
    # backed-off retry time, and exactly once.
    assert dials == ["d_peer"]
    assert clock.now < next_retry
    assert server.redial.snapshot("d_peer")["attempts"] == attempts_before + 1
    assert waits[3] == relay.REDIAL_TICK_S


def test_a_reading_that_raises_redials_exactly_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    server, waits, dials = _loop_server(monkeypatch, [RuntimeError("synthetic")], clock)
    server.redial.seed("d_peer")
    server._redial_loop()
    assert dials == ["d_peer"]
    assert waits == [relay.REDIAL_TICK_S, relay.REDIAL_TICK_S]


def test_unknown_reading_never_pauses(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    server, waits, dials = _loop_server(monkeypatch, [uplink.unknown_reading()], clock)
    server.redial.seed("d_peer")
    server._redial_loop()
    assert dials == ["d_peer"]


def test_a_relay_without_a_reader_keeps_todays_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock()
    server, waits, dials = _loop_server(monkeypatch, [None], clock)
    del server._uplink_reader
    server.redial.seed("d_peer")
    server._redial_loop()
    assert dials == ["d_peer"]


def test_kick_all_keeps_attempts_and_makes_everyone_due() -> None:
    clock = _Clock()
    tracker = redial.RedialTracker(clock=clock, rand=lambda: 0.5)
    tracker.lost("d_a", cause="timeout")
    tracker.failed("d_a", cause="timeout")
    tracker.lost("d_b", cause="timeout")
    assert tracker.due() == []
    tracker.kick_all()
    assert sorted(tracker.due()) == ["d_a", "d_b"]
    assert tracker.snapshot("d_a")["attempts"] == 1


# -- the rows ready and doctor print first --------------------------------------------


def _reach(device: str, endpoint: str, detail: str, ok: bool = False) -> dict[str, Any]:
    return {
        "check": "reachability",
        "device_id": device,
        "device_name": "node2",
        "endpoint": endpoint,
        "ok": ok,
        "detail": detail,
    }


def test_offline_device_row_is_first_and_names_this_device() -> None:
    rows = uplink.uplink_checks(
        OFFLINE, [_reach("d_n2", "16.52.146.32:4097", "connect_failed:OSError")]
    )
    assert len(rows) == 1
    assert rows[0]["ok"] is False and rows[0]["code"] == "offline"
    assert rows[0]["detail"] == "This device has no network connection."
    assert rows[0]["remedies"] == ["check this device's network"]


def test_family_mismatch_names_the_missing_route_and_the_peer() -> None:
    rows = uplink.uplink_checks(
        V6_ONLY, [_reach("d_n2", "16.52.146.32:4097", "connect_failed:OSError")]
    )
    assert [row["code"] for row in rows] == ["ipv6_only", "local_offline"]
    assert rows[0]["ok"] is True
    assert rows[1]["detail"] == (
        "This device's network has no IPv4 route; node2 is reachable over IPv4 only"
    )
    assert rows[1]["remedies"] == ["check this device's network"]
    reverse = uplink.uplink_checks(
        {"state": uplink.IPV4_ONLY, "ipv4": True, "ipv6": False},
        [_reach("d_n2", "[2001:db8::7]:4097", "connect_failed:OSError")],
    )
    assert reverse[1]["detail"] == (
        "This device's network has no IPv6 route; node2 is reachable over IPv6 only"
    )


def test_a_routed_silent_peer_isnt_answering_and_no_cause_is_named() -> None:
    rows = uplink.uplink_checks(
        DUAL, [_reach("d_n2", "16.52.146.32:4097", "connect_failed:TimeoutError")]
    )
    assert rows[1]["detail"] == "node2 isn't answering from this network"
    assert "remedies" not in rows[1]
    for word in ("firewall", "allowlist"):
        assert word not in rows[1]["detail"].lower()


@pytest.mark.parametrize(
    "detail",
    ["connect_failed:ConnectionRefusedError", "unknown_network", "handshake_not_attempted: x"],
)
def test_a_peer_that_answered_gets_no_extra_row(detail: str) -> None:
    rows = uplink.uplink_checks(DUAL, [_reach("d_n2", "16.52.146.32:4097", detail)])
    assert len(rows) == 1


def test_unknown_reading_claims_nothing() -> None:
    rows = uplink.uplink_checks(
        None, [_reach("d_n2", "16.52.146.32:4097", "connect_failed:OSError")]
    )
    # No device row (it would claim nothing) and no peer row (nothing proven).
    assert rows == []


def test_ready_renders_the_uplink_row_first_with_its_remedy() -> None:
    from local_operator.network import readiness

    reach = _reach("d_n2", "16.52.146.32:4097", "connect_failed:OSError")
    checks = uplink.uplink_checks(V6_ONLY, [reach]) + [reach]
    lines = readiness.render_check_lines(checks)
    assert lines[0] == "ok  uplink: this device has an IPv6 route and no IPv4 route"
    assert lines[1] == (
        "FAIL uplink node2: This device's network has no IPv4 route; "
        "node2 is reachable over IPv4 only"
    )
    assert lines[2] == "    - check this device's network"
    assert lines[3].startswith("FAIL reachability node2")


def test_doctor_puts_the_uplink_row_first(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.network import cli

    payload = {
        "checks": [
            {
                "check": "reachability",
                "device_id": "d_n2",
                "endpoint": "16.52.146.32:4097",
                "ok": False,
                "detail": "connect_failed:OSError",
            }
        ]
    }
    monkeypatch.setattr(cli, "_relay_call", lambda *a, **k: payload)
    monkeypatch.setattr(cli, "_uplink_reading", lambda: dict(OFFLINE))
    monkeypatch.setattr(relay, "supervision_state", lambda: {"platform": "none"})
    monkeypatch.setattr(relay, "foreign_supervision", lambda: [])
    args = Namespace(peer="", json=False)
    cli._cmd_doctor(args)
    out = capsys.readouterr().out.splitlines()
    first = next(index for index, line in enumerate(out) if "uplink" in line)
    reach = next(index for index, line in enumerate(out) if "reachability" in line)
    assert first < reach
    assert out[first] == "FAIL uplink: This device has no network connection."
    assert out[first + 1] == "    - check this device's network"


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
