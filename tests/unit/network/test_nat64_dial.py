"""NAT64 dial fallback (``network.nat64``), driven by a fake resolver and route check.

No test here touches a real network: the resolver, the IPv4 route check, the
network identity and the connector are all injected, so each case states
exactly what the OS "said" and asserts what was dialled. The RFC 6052 vectors
are the RFC's own §2.4 table (192.0.2.33 under 2001:db8::/32../96).
"""

from __future__ import annotations

import errno
import ipaddress
import logging
import socket
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.network import nat64, relay

V4 = "192.0.2.33"
PUBLIC_V4 = "203.0.113.7"  # TEST-NET-3: ``is_global`` is False, so WKP tests use a real-shaped one
GLOBAL_V4 = "8.8.4.4"
PORT = 4097


class FakeResolver:
    """``getaddrinfo`` stand-in: answers by ``(host, family)``, records every call."""

    def __init__(
        self, answers: dict[tuple[str, int], list[str]] | None = None, raises: bool = False
    ):
        self.answers = answers or {}
        self.raises = raises
        self.calls: list[tuple[str, int]] = []

    def __call__(self, host: str, port: Any, family: int = 0, type: int = 0, *a: Any) -> list[Any]:
        self.calls.append((host, family))
        if self.raises:
            raise socket.gaierror(socket.EAI_NONAME, "boom")
        out = []
        for (name, fam), addresses in self.answers.items():
            if name != host or (family not in (0, socket.AF_UNSPEC) and fam != family):
                continue
            for address in addresses:
                sockaddr = (address, port, 0, 0) if fam == socket.AF_INET6 else (address, port)
                out.append((fam, socket.SOCK_STREAM, 6, "", sockaddr))
        return out


class FakeConnector:
    """Fails every IPv4 dial with ``ENETUNREACH`` (an IPv6-only host); IPv6 succeeds."""

    def __init__(self, v4_error: OSError | None = None, v6_ok: bool = True):
        self.v4_error = v4_error or OSError(errno.ENETUNREACH, "Network is unreachable")
        self.v6_ok = v6_ok
        self.dialled: list[str] = []

    def __call__(self, address: tuple[str, int], timeout: float | None = None) -> Any:
        host = address[0]
        self.dialled.append(host)
        if ":" in host:
            if self.v6_ok:
                return f"sock:{host}"
            raise OSError(errno.ECONNREFUSED, "refused")
        raise self.v4_error


@pytest.fixture(autouse=True)
def _fresh_hints() -> None:
    nat64.HINTS.clear()


def dial(
    host: str,
    resolver: FakeResolver,
    connector: FakeConnector,
    *,
    route: bool | None = False,
    key: str | None = "2001:db8:aaaa::/64",
) -> Any:
    return nat64.create_connection(
        (host, PORT),
        timeout=5.0,
        resolver=resolver,
        route_check=lambda _address: route,
        key_reader=lambda: key,
        connector=connector,
    )


def test_os_synthesis_is_dialled_first() -> None:
    synthesised = "64:ff9b::c000:221"  # what macOS returns for the literal on NAT64
    resolver = FakeResolver({(V4, socket.AF_INET6): [synthesised]})
    connector = FakeConnector()
    assert dial(V4, resolver, connector) == f"sock:{synthesised}"
    assert connector.dialled == [V4, synthesised]
    # The OS answered; discovery was never needed.
    assert ("ipv4only.arpa", socket.AF_INET6) not in resolver.calls


def test_os_answer_that_does_not_embed_the_target_is_ignored() -> None:
    # A resolver answering the literal with an arbitrary AAAA is not a synthesis.
    resolver = FakeResolver({(V4, socket.AF_INET6): ["2001:db8::dead"]})
    connector = FakeConnector()
    with pytest.raises(OSError) as raised:
        dial(V4, resolver, connector)
    assert raised.value.errno == errno.ENETUNREACH
    assert connector.dialled == [V4]


def test_discovery_well_known_prefix(caplog: pytest.LogCaptureFixture) -> None:
    resolver = FakeResolver(
        {("ipv4only.arpa", socket.AF_INET6): ["64:ff9b::c000:aa", "64:ff9b::c000:ab"]}
    )
    connector = FakeConnector()
    with caplog.at_level(logging.INFO, logger=nat64.__name__):
        assert dial(GLOBAL_V4, resolver, connector) == "sock:64:ff9b::808:404"
    assert connector.dialled == [GLOBAL_V4, "64:ff9b::808:404"]
    lines = [r.getMessage() for r in caplog.records if r.name == nat64.__name__]
    assert lines == ["nat64 dial synthesis used (prefix source: ipv4only.arpa)"]
    # Log carries the prefix SOURCE only, never the peer's address.
    assert GLOBAL_V4 not in lines[0] and "808:404" not in lines[0]


@pytest.mark.parametrize(
    ("discovered", "length", "expected"),
    [
        # RFC 6052 §2.4 table, with 192.0.0.170 discovered and 192.0.2.33 embedded.
        ("2001:db8:122:344::c000:aa", 96, "2001:db8:122:344::c000:221"),
        ("2001:db8:122:344:c0:0:aa00:0", 64, "2001:db8:122:344:c0:2:2100:0"),
        ("2001:db8:122:3c0:0:aa::", 56, "2001:db8:122:3c0:0:221::"),
        ("2001:db8:122:c000:0:aa00::", 48, "2001:db8:122:c000:2:2100::"),
        ("2001:db8:1c0:0:aa::", 40, "2001:db8:1c0:2:21::"),
        ("2001:db8:c000:aa::", 32, "2001:db8:c000:221::"),
    ],
)
def test_network_specific_prefix_lengths(discovered: str, length: int, expected: str) -> None:
    pref = nat64.prefix_from_discovery([ipaddress.IPv6Address(discovered)])
    assert pref is not None and pref.length == length
    resolver = FakeResolver({("ipv4only.arpa", socket.AF_INET6): [discovered]})
    connector = FakeConnector()
    # A network-specific prefix MAY carry a non-global IPv4 (RFC 6052 §3.1).
    assert dial(V4, resolver, connector) == f"sock:{ipaddress.IPv6Address(expected)}"


def test_well_known_prefix_refuses_non_global_ipv4() -> None:
    resolver = FakeResolver({("ipv4only.arpa", socket.AF_INET6): ["64:ff9b::c000:aa"]})
    connector = FakeConnector()
    with pytest.raises(OSError):
        dial("10.0.0.5", resolver, connector)
    assert connector.dialled == ["10.0.0.5"]


def test_hint_reused_on_same_network_and_dropped_on_change() -> None:
    resolver = FakeResolver({("ipv4only.arpa", socket.AF_INET6): ["2001:db8:64::c000:aa"]})
    dial(GLOBAL_V4, resolver, FakeConnector(), key="net-A")
    assert resolver.calls.count(("ipv4only.arpa", socket.AF_INET6)) == 1

    # Same network: the hint answers, no second discovery.
    connector = FakeConnector()
    with_hint = dial(PUBLIC_V4, resolver, connector, key="net-A")
    assert with_hint == "sock:2001:db8:64::cb00:7107"
    assert resolver.calls.count(("ipv4only.arpa", socket.AF_INET6)) == 1

    # A different network: the old prefix is NOT used; this network has none.
    resolver.answers = {}
    connector = FakeConnector()
    with pytest.raises(OSError):
        dial(PUBLIC_V4, resolver, connector, key="net-B")
    assert connector.dialled == [PUBLIC_V4]
    assert nat64.HINTS.get("net-A") is None  # dropped, not parked for later


def test_unidentifiable_network_never_caches() -> None:
    resolver = FakeResolver({("ipv4only.arpa", socket.AF_INET6): ["64:ff9b::c000:aa"]})
    dial(GLOBAL_V4, resolver, FakeConnector(), key=None)
    dial(GLOBAL_V4, resolver, FakeConnector(), key=None)
    assert resolver.calls.count(("ipv4only.arpa", socket.AF_INET6)) == 2


@pytest.mark.parametrize(
    "discovered",
    [
        "fe80::c000:aa",  # link-local
        "ff02::c000:aa",  # multicast
        "::c000:aa",  # all-zero prefix (IPv4-compatible)
        "::ffff:192.0.0.170",  # IPv4-mapped: rides the missing IPv4 stack
        "::1:c000:aa",  # under ::/96 too
        "fec0::c000:aa",  # site-local
    ],
)
def test_invalid_prefix_rejected(discovered: str) -> None:
    resolver = FakeResolver({("ipv4only.arpa", socket.AF_INET6): [discovered]})
    connector = FakeConnector()
    with pytest.raises(OSError) as raised:
        dial(GLOBAL_V4, resolver, connector)
    assert raised.value.errno == errno.ENETUNREACH
    assert connector.dialled == [GLOBAL_V4]
    assert nat64.HINTS.get("2001:db8:aaaa::/64") is None


def test_nonzero_u_octet_is_not_a_prefix() -> None:
    # /64 layout with 192.0.0.170 in place but bits 64-71 set.
    weird = ipaddress.IPv6Address("2001:db8:122:344:ffc0:0:aa00:0")
    assert nat64.prefix_from_discovery([weird]) is None


def test_ipv4_route_present_means_zero_resolver_calls() -> None:
    resolver = FakeResolver({(V4, socket.AF_INET6): ["64:ff9b::c000:221"]})
    connector = FakeConnector(v4_error=OSError(errno.ECONNREFUSED, "refused"))
    with pytest.raises(OSError) as raised:
        dial(V4, resolver, connector, route=True)
    assert raised.value.errno == errno.ECONNREFUSED
    assert resolver.calls == []
    assert connector.dialled == [V4]


def test_unknown_route_with_a_non_route_error_does_nothing() -> None:
    resolver = FakeResolver()
    connector = FakeConnector(v4_error=TimeoutError("timed out"))
    with pytest.raises(TimeoutError):
        dial(V4, resolver, connector, route=None)
    assert resolver.calls == []


def test_successful_first_dial_never_consults_anything() -> None:
    resolver = FakeResolver()
    checks: list[str] = []
    sock = nat64.create_connection(
        (V4, PORT),
        timeout=1.0,
        resolver=resolver,
        route_check=lambda a: checks.append(a) or False,
        key_reader=lambda: "k",
        connector=cast(Any, lambda address, timeout=None: "sock:v4"),
    )
    assert sock == "sock:v4" and resolver.calls == [] and checks == []


@pytest.mark.parametrize("host", ["2001:db8::5", "::1"])
def test_ipv6_target_unchanged(host: str) -> None:
    resolver = FakeResolver()
    connector = FakeConnector(v6_ok=False)
    with pytest.raises(OSError) as raised:
        dial(host, resolver, connector)
    assert raised.value.errno == errno.ECONNREFUSED
    assert connector.dialled == [host] and resolver.calls == []


def test_hostname_with_aaaa_unchanged() -> None:
    resolver = FakeResolver(
        {
            ("peer.example", socket.AF_INET): [GLOBAL_V4],
            ("peer.example", socket.AF_INET6): ["2001:db8::9"],
        }
    )
    connector = FakeConnector()
    with pytest.raises(OSError):
        dial("peer.example", resolver, connector)
    assert connector.dialled == ["peer.example"]
    assert resolver.calls == [("peer.example", socket.AF_UNSPEC)]


def test_hostname_resolving_only_to_ipv4_is_synthesised() -> None:
    resolver = FakeResolver(
        {
            ("peer.example", socket.AF_INET): [GLOBAL_V4],
            (GLOBAL_V4, socket.AF_INET6): ["64:ff9b::808:404"],
        }
    )
    connector = FakeConnector()
    assert dial("peer.example", resolver, connector) == "sock:64:ff9b::808:404"


class RaisesForLiteral(FakeResolver):
    """glibc: an IPv4 literal asked for AF_INET6 is EAI_ADDRFAMILY, never a synthesis."""

    def __call__(self, host: str, port: Any, family: int = 0, *a: Any) -> list[Any]:
        if family == socket.AF_INET6 and host[:1].isdigit():
            self.calls.append((host, family))
            raise socket.gaierror(socket.EAI_ADDRFAMILY, "Address family not supported")
        return super().__call__(host, port, family, *a)


def test_os_literal_lookup_raising_still_tries_discovery() -> None:
    # Found on the real-path rig (Linux netns, IPv6-only): the OS step raising
    # must fall through to ipv4only.arpa, not abort the fallback.
    resolver = RaisesForLiteral({("ipv4only.arpa", socket.AF_INET6): ["64:ff9b::c000:aa"]})
    assert dial(GLOBAL_V4, resolver, FakeConnector()) == "sock:64:ff9b::808:404"


def test_resolver_raising_surfaces_original_error() -> None:
    original = OSError(errno.ENETUNREACH, "Network is unreachable")
    connector = FakeConnector(v4_error=original)
    with pytest.raises(OSError) as raised:
        dial(V4, FakeResolver(raises=True), connector)
    assert raised.value is original


def test_synthesised_dial_failing_surfaces_original_error() -> None:
    original = OSError(errno.ENETUNREACH, "Network is unreachable")
    resolver = FakeResolver({(V4, socket.AF_INET6): ["64:ff9b::c000:221"]})
    connector = FakeConnector(v4_error=original, v6_ok=False)
    with pytest.raises(OSError) as raised:
        dial(V4, resolver, connector)
    assert raised.value is original


def test_hint_is_never_written_to_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import os

    writes: list[Any] = []
    real_open = builtins.open
    real_os_open = os.open

    def spy_open(file: Any, mode: str = "r", *a: Any, **k: Any) -> Any:
        if any(flag in mode for flag in "wax+"):
            writes.append(file)
        return real_open(file, mode, *a, **k)

    def spy_os_open(path: Any, flags: int, *a: Any, **k: Any) -> Any:
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            writes.append(path)
        return real_os_open(path, flags, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(os, "open", spy_os_open)
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: writes.append(a))
    monkeypatch.setattr(Path, "write_bytes", lambda *a, **k: writes.append(a))
    resolver = FakeResolver({("ipv4only.arpa", socket.AF_INET6): ["64:ff9b::c000:aa"]})
    dial(GLOBAL_V4, resolver, FakeConnector())
    dial(GLOBAL_V4, resolver, FakeConnector())  # hint path
    assert writes == []


def test_route_check_sends_nothing_and_reads_kernel_table() -> None:
    # Real syscall, no packet: TEST-NET-1 either has a route (True) or not (False/None),
    # but the call must not raise or block.
    assert nat64.ipv4_route("192.0.2.1") in (True, False, None)


def test_relay_dial_sites_use_the_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both mesh dial sites go through ``nat64.create_connection`` (one seam, not copies)."""
    seen: list[tuple[str, int]] = []

    def fake(address: tuple[str, int], timeout: float | None = None, **_: Any) -> Any:
        seen.append(address)
        raise OSError(errno.ENETUNREACH, "no route")

    monkeypatch.setattr(nat64, "create_connection", fake)
    probe = relay.probe_candidates(["192.0.2.9:4097"], deadline=None, connect_cap=1.0)
    assert ("192.0.2.9", 4097) in seen
    assert probe.sock is None and probe.winner == ""
    # ``RelayServer.dial`` needs a whole relay to drive; its source is the cheap,
    # exact proof that its own connect (the non-probe path) is the same seam.
    import inspect

    source = inspect.getsource(relay.RelayServer.dial)
    assert "nat64.create_connection(" in source
    assert "socket.create_connection(" not in source
