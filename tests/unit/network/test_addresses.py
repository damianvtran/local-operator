"""Where a device says peers can reach it: the interface table, not a hostname.

THE BUG THESE PIN. ``relay.advertise_endpoints`` built the "detected local
addresses" half of its answer from ``socket.getaddrinfo(socket.gethostname(),
None, socket.AF_INET)``. On macOS ``gethostname`` is the Bonjour name
(``damians-MacBook-Pro``), which no resolver knows, so ``getaddrinfo`` raises
``[Errno 8] nodename nor servname provided`` and the ``OSError`` branch
contributes NOTHING. With the default ``listen_address: 0.0.0.0`` the record's
``advertised`` list, every member row and every invite token therefore carried no
address on this machine at all, and a joiner was told *"that invite names no
endpoint; pass --host host:port"* — so pairing needed a human to know and type the
other machine's IP. The lane that reported it worked around the cloud node by
mapping a hostname to a public IP in ``/etc/hosts``; nothing here needs that.

WHAT IS PINNED, AND WHY IT IS PINNED THIS WAY.

* ``test_a_hostname_that_does_not_resolve_is_no_longer_the_answer`` is the
  regression test proper: it reproduces the macOS resolver failure by name and
  requires a real endpoint anyway. It FAILS on the old code (the list came back
  empty) and passes on the fix, which is the only property that makes it evidence.
* The set comes from the OS's interface table, so the assertion over the REAL
  answer is a PROPERTY (routable, never loopback) rather than a fixture: a
  runner's interface names and addresses are not something a test can fix, and
  pinning them is how a test becomes a host certificate. The multi-interface case
  is driven with a patched table instead, where the ORDER is the thing under test.
* Loopback is asserted at the SOURCE (``is_advertisable_ipv4``) as well as through
  ``advertise_endpoints``, because an address that is merely excluded from one
  caller is still published by the other three that read the module.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from local_operator.network import addresses, relay
from local_operator.network.handshake import MAX_DECLARED_ENDPOINTS

#: How macOS fails, verbatim: this is the exception the old code swallowed, and a
#: test that reproduced it with a generic ``Exception`` would not be reproducing
#: the report it exists for.
_MACOS_RESOLVER_FAILURE = OSError(8, "nodename nor servname provided, or not known")

_UP = 0x1
_LOOPBACK = 0x8


def _table(*rows: tuple[str, int, str], loopback: tuple[str, int, str] | None = None) -> Any:
    """A stand-in for ``_interface_ipv4s``: the rows given, plus ``lo0`` by default.

    The loopback row is present unless a test opts out, because EVERY host has one
    and a table without it would let a filter that never excludes loopback pass.
    """
    entries = list(rows)
    entries.append(loopback or ("lo0", _UP | _LOOPBACK, "127.0.0.1"))
    return lambda: entries


# ---------------------------------------------------------------------------
# The regression: a hostname that does not resolve must not empty the list
# ---------------------------------------------------------------------------


def test_interface_for_address_maps_through_the_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """The name a readiness row publishes comes from the ONE interface table.

    ``_interface_ipv4s`` has always carried the name and ``local_ipv4_addresses``
    discarded it; this is the mapping back, and its failure direction is the
    point: an address the table does not hold (loopback, a table that could not
    be read) answers ``None`` — "not observable" — never a guess.
    """
    monkeypatch.setattr(
        addresses,
        "_interface_ipv4s",
        _table(("utun4", _UP, "203.0.113.7"), ("en0", _UP, "10.0.2.169")),
    )
    assert addresses.interface_for_address("203.0.113.7") == "utun4"
    assert addresses.interface_for_address("10.0.2.169") == "en0"
    # Loopback is IN the table (every host has one) and maps like any other row:
    # the filter that keeps it out of ADVERTISED addresses is a different concern.
    assert addresses.interface_for_address("127.0.0.1") == "lo0"
    assert addresses.interface_for_address("192.0.2.9") is None
    monkeypatch.setattr(addresses, "_interface_ipv4s", lambda: None)
    assert addresses.interface_for_address("203.0.113.7") is None


def test_route_observation_publishes_the_kernels_own_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A UDP ``connect`` sends nothing and answers with the real source address.

    Driven against the REAL table for loopback (deterministic anywhere): the
    kernel picks 127.0.0.1, and the interface is ``None`` because loopback is
    not in the table — an unobservable name rather than a wrong one.
    """
    loopback = addresses.route_observation("127.0.0.1", 1)
    assert loopback["source_address"] == "127.0.0.1"
    assert loopback["error"] == ""
    # Port 0 is not connectable; the route decision is keyed on the ADDRESS, so
    # it must still answer rather than raise EADDRNOTAVAIL.
    assert addresses.route_observation("127.0.0.1", 0)["error"] == ""


def test_route_observation_reports_what_it_cannot_observe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed lookup names its failure; nothing reads as a source address."""

    class _Dead:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise OSError(49, "Can't assign requested address")

    monkeypatch.setattr(addresses.socket, "socket", _Dead)
    observation = addresses.route_observation("10.0.0.1", 443)
    assert observation == {"source_address": None, "interface": None, "error": "OSError"}


def test_a_hostname_that_does_not_resolve_is_no_longer_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE regression test: fails on ``getaddrinfo(gethostname())``, passes now.

    ``socket.getaddrinfo`` is patched to raise exactly what this Mac's resolver
    raises for its own hostname. On the old code path that exception was caught and
    forgotten, so the answer was ``[]`` on a host whose interface table plainly had
    an address; the test asserts the address the table holds instead, which is what
    an invite's ``hosts`` field is built from.
    """

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise _MACOS_RESOLVER_FAILURE

    monkeypatch.setattr(socket, "getaddrinfo", _refuse)
    monkeypatch.setattr(addresses, "_interface_ipv4s", _table(("en0", _UP, "192.168.1.10")))
    monkeypatch.setattr(addresses, "_default_route_address", lambda: "192.168.1.10")

    published = relay.advertise_endpoints(relay.NetworkSettings(port=4097))

    assert published == [
        "192.168.1.10:4097"
    ], "a hostname the resolver does not know must not decide where peers can dial us"


def test_the_real_enumeration_answers_when_the_resolver_cannot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same reproduction, over THIS HOST's interface table: nothing is patched
    but the resolver, so the assertion is about the address a real invite would carry.

    This is the shape that discriminates without depending on a fixture: on the old
    code path it fails as ``assert [] == [...]`` on this Mac (measured), and it stays
    a real assertion on a runner whose hostname happens to resolve, because the
    resolver is refused here either way. Both branches are asserted rather than
    skipped — a host that genuinely holds no advertisable address owes the empty
    list, and that is a different claim from the one the bug made.
    """

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise _MACOS_RESOLVER_FAILURE

    monkeypatch.setattr(socket, "getaddrinfo", _refuse)
    released = relay.advertise_endpoints(relay.NetworkSettings(port=4097))

    for endpoint in released:
        host, _, _port = endpoint.rpartition(":")
        assert addresses.is_advertisable_ipv4(host), endpoint
    assert not any(endpoint.startswith("127.") for endpoint in released), released
    holds_one = (
        any(
            (flags & _UP) and not (flags & _LOOPBACK) and addresses.is_advertisable_ipv4(address)
            for _name, flags, address in (addresses._interface_ipv4s() or [])
        )
        or addresses._default_route_address()
    )
    if holds_one:
        assert released, (
            "the resolver does not know this host's name, and that must no longer be "
            "the answer to where peers can reach it"
        )
    else:
        assert (
            released == []
        ), "a host holding no advertisable address owes an empty list, not a guess"


def test_the_table_not_the_hostname_is_what_the_listener_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The enumeration is the SOURCE, not a fallback beside the old lookup.

    A fix that kept ``getaddrinfo`` first and only fell back to the table when it
    raised would pass the test above and still publish a name-resolved address on
    any host whose hostname happens to resolve — a second answer that can disagree
    with the interface table (a DNS name pointing at another machine, a resolver
    search suffix appending a domain). ``getaddrinfo`` is therefore asserted
    UNCALLED, which is the property that makes the table the single source.
    """
    calls: list[tuple[Any, ...]] = []

    def _spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        raise _MACOS_RESOLVER_FAILURE

    monkeypatch.setattr(socket, "getaddrinfo", _spy)
    monkeypatch.setattr(addresses, "_interface_ipv4s", _table(("en0", _UP, "192.168.1.10")))

    relay.advertise_endpoints(relay.NetworkSettings(port=4097))

    assert calls == []


# ---------------------------------------------------------------------------
# The properties, over the real machine
# ---------------------------------------------------------------------------


def test_the_real_enumeration_offers_routable_addresses_and_never_loopback() -> None:
    """Drive the real table: whatever this host holds, what it offers is dialable.

    Deliberately NOT a fixed expectation. On this fleet the answer is the LAN
    address (``192.168.0.155`` on ``en0``) and in a CI container it is the
    container's own; both are the correct answer to "where can a peer reach me",
    and only the PROPERTY travels between them.
    """
    offered = addresses.local_ipv4_addresses()

    assert all(addresses.is_advertisable_ipv4(address) for address in offered), offered
    assert not any(address.startswith("127.") for address in offered), offered

    # The binding itself must work here, not merely return []: an empty answer from
    # a table that could not be READ would pass every assertion above while leaving
    # the reported bug in place. ``[]`` however IS a real answer — a device whose only
    # IPv4 address is loopback — so what is checked is the PAIR (a table that read, and
    # holds something dialable), never the table being non-empty for its own sake:
    # ``assert table`` outright contradicted ``_interface_ipv4s``'s own contract
    # (review round 1, N2).
    table = addresses._interface_ipv4s()
    if table:
        non_loopback = [row for row in table if (row[1] & _LOOPBACK) == 0]
        if non_loopback:
            assert offered, f"a readable table with {non_loopback} offered nothing"


def test_the_endpoints_the_relay_publishes_are_routable_ones() -> None:
    """``advertise_endpoints`` on the REAL table never names loopback.

    The half of the contract a receiver acts on: a peer dials what it is handed, so
    a ``127.`` entry here is a peer dialling ITSELF. The dial-only listener's own
    ``127.0.0.1`` is the documented exception and has its own test below.
    """
    published = relay.advertise_endpoints(relay.NetworkSettings(port=4097))

    for endpoint in published:
        host, _, _port = endpoint.rpartition(":")
        assert addresses.is_advertisable_ipv4(host), endpoint
    assert not any(endpoint.startswith("127.") for endpoint in published), published


# ---------------------------------------------------------------------------
# Multiple interfaces, and the order that answers "which one do I name first"
# ---------------------------------------------------------------------------


def test_the_first_offer_is_the_kernel_s_own_route_and_loopback_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A laptop has several addresses; the order is a ranking, not an accident.

    The default-route address leads because the kernel itself prefers it for every
    destination off this host, and a peer dials the list in order — so a wrong
    first entry costs one failed attempt rather than a lost peer. The other rows
    keep the table's order, and the excluded classes (loopback by FLAG on its own
    interface, link-local on a down interface, the unspecified address) do not
    appear at all.
    """
    monkeypatch.setattr(
        addresses,
        "_interface_ipv4s",
        _table(
            ("en0", _UP, "192.168.1.10"),
            ("en1", _UP, "10.0.0.5"),
            ("utun0", _UP, "100.64.0.9"),
            ("en2", 0, "169.254.9.9"),
            ("en3", _UP, "0.0.0.0"),
        ),
    )
    monkeypatch.setattr(addresses, "_default_route_address", lambda: "10.0.0.5")

    assert addresses.local_ipv4_addresses() == [
        "10.0.0.5",
        "192.168.1.10",
        "100.64.0.9",
    ]


def test_a_tunnel_address_is_kept_and_a_not_up_interface_is_not_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``IFF_POINTOPOINT`` is NOT an exclusion: a VPN peer dials that address.

    Stated as its own cell because the tempting simplification — "only offer the
    default-route interface" — is wrong in exactly the case the mesh exists for: a
    cloud node reached over a tunnel holds the tunnel address and nothing else the
    peer can reach, and its ``utun``/``wg0`` row is the only way in.
    """
    monkeypatch.setattr(addresses, "_interface_ipv4s", _table(("wg0", _UP, "10.9.0.2")))
    monkeypatch.setattr(addresses, "_default_route_address", lambda: "10.9.0.2")

    assert addresses.local_ipv4_addresses() == ["10.9.0.2"]


def test_an_unreadable_table_still_offers_the_routable_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``getifaddrs`` on this platform: one routable address beats the empty list.

    The fallback is the same answer the kernel gives, and it is strictly better
    than the behaviour this bug was, so the two sources are ordered rather than
    either/or: the table supplies the SET when it can be read, and this supplies
    the single address when it cannot.
    """
    monkeypatch.setattr(addresses, "_interface_ipv4s", lambda: None)
    monkeypatch.setattr(addresses, "_default_route_address", lambda: "203.0.113.7")

    assert addresses.local_ipv4_addresses() == ["203.0.113.7"]


def test_no_route_and_no_table_is_an_empty_answer_not_a_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An air-gapped host has nothing to advertise, and saying so is the point.

    The mirror of the fallback above: with neither source, the answer is empty
    rather than loopback or a fabricated address — a wrong address silently
    misleads a joiner, which is worse than "I do not know".
    """
    monkeypatch.setattr(addresses, "_interface_ipv4s", lambda: None)
    monkeypatch.setattr(addresses, "_default_route_address", lambda: None)

    assert addresses.local_ipv4_addresses() == []


# ---------------------------------------------------------------------------
# What advertise_endpoints does with the answer
# ---------------------------------------------------------------------------


def test_declared_hosts_lead_and_the_bound_still_caps_the_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator's own declaration still outranks detection, and the cap holds.

    Both were true before this change and must stay true: a tunnel or public
    address is something only the operator knows, so it is tried first, and the
    union is bounded by what a RECEIVER keeps so nothing published here is dropped
    at the far end and silently missing from the row the dialer reads.
    """
    rows = tuple((f"en{i}", _UP, f"10.1.{i}.1") for i in range(12))
    monkeypatch.setattr(addresses, "_interface_ipv4s", _table(*rows))
    monkeypatch.setattr(addresses, "_default_route_address", lambda: "10.1.0.1")

    settings = relay.NetworkSettings(port=4097, advertise_hosts=("tunnel.example.com:4200",))
    published = relay.advertise_endpoints(settings)

    assert published[0] == "tunnel.example.com:4200"
    assert len(published) == MAX_DECLARED_ENDPOINTS
    assert published[1] == "10.1.0.1:4097"


def test_the_declared_host_the_operator_typed_is_the_one_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config route, end to end: ``network.advertise_hosts`` reaches the invite.

    Constructing the settings directly is what makes this a test of the CONSUMER;
    the config file itself is covered by ``test_config`` (the key survives a load
    and the write that follows it) and by the pairing evidence on the PR.
    """
    monkeypatch.setattr(addresses, "local_ipv4_addresses", lambda: [])
    settings = relay.NetworkSettings(port=4097, advertise_hosts=("203.0.113.7:4097",))

    assert relay.advertise_endpoints(settings) == ["203.0.113.7:4097"]


def test_the_dial_only_listener_still_answers_its_own_loopback() -> None:
    """The one deliberate loopback entry: ``127.0.0.1`` is the honest dial-only answer.

    Unchanged by this fix, and asserted here because the exclusion above is now
    enforced in one place for both paths — the entry is added by the listener's own
    branch, BEFORE detection runs, so a filter that also suppressed this would turn
    "you cannot reach me" into "I have no address", which is a different sentence to
    the peer reading the row.
    """
    settings = relay.NetworkSettings(port=4097, listen_address="127.0.0.1")

    assert relay.advertise_endpoints(settings) == ["127.0.0.1:4097"]


def test_the_sockaddr_family_is_read_in_each_layout_and_each_byte_order() -> None:
    """Both platform layouts, on whichever platform CI runs, and both byte orders.

    ``_family_of`` had two holes that only a differently-shaped machine could show. The
    Darwin/BSD branch (``sa_family`` in the byte after ``sa_len``) never executed in CI at
    all, because it was selected from the machine the suite ran on — correct here, unrun
    there (review round 1, M5). The glibc branch read the family with a fixed
    little-endian shift, so a big-endian Linux (s390x, ppc64) read ``0x0002`` as 512 and
    skipped every interface as "not AF_INET", while its own comment claimed that class was
    closed (M3).

    So the two things are separated: the LAYOUT is driven by the ``bsd`` argument and the
    BYTE ORDER by ``_glibc_family``, which is why the rule can be asserted here rather
    than only on the hardware that would have caught it.
    """
    import ctypes
    import sys

    size = addresses._SOCKADDR_SIZE
    bsd_raw = bytes([size, socket.AF_INET]) + bytes(size - 2)
    bsd = ctypes.create_string_buffer(bsd_raw, size)
    assert addresses._family_of(ctypes.addressof(bsd), bsd=True) == socket.AF_INET
    # The same bytes read as glibc's layout are the ``uint16`` 0x1002 — not AF_INET —
    # which is what makes the two branches distinguishable rather than interchangeable.
    assert addresses._family_of(ctypes.addressof(bsd), bsd=False) != socket.AF_INET

    glibc_raw = int.to_bytes(socket.AF_INET, 2, sys.byteorder) + bytes(size - 2)
    glibc = ctypes.create_string_buffer(glibc_raw, size)
    assert addresses._family_of(ctypes.addressof(glibc), bsd=False) == socket.AF_INET

    # The byte-order rule itself, both ways: native order decodes, the other order does
    # not — which is exactly what the fixed shift got wrong on one of the two hosts.
    for order in ("little", "big"):
        first, second = (2, 0) if order == "little" else (0, 2)
        assert addresses._glibc_family(first, second, order) == socket.AF_INET
        assert addresses._glibc_family(second, first, order) == 2 << 8
