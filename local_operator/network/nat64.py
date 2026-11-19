"""NAT64 dial fallback: reach an IPv4-only peer from an IPv6-only uplink.

WHY THIS EXISTS (roaming stage 3, incident 2026-11-18). The operator's Mac
joined an IPv6-only network while node2 advertised IPv4 endpoints only. Every
dial to node2 failed instantly with "no route", although such networks almost
always run NAT64/DNS64 precisely so IPv4-only destinations stay reachable: an
IPv4 address embedded in the network's PREF64 (RFC 6052) is an IPv6 address
the NAT64 gateway translates. A peer row holds IPv4 *literals*, which DNS64
never sees, so nothing synthesised them for us. This module does, at ONE seam
(:func:`create_connection`) that every mesh dial site in ``relay.py`` calls
instead of ``socket.create_connection``.

WHEN IT ACTS — ONLY AFTER A REAL FAILURE, AND ONLY WITHOUT AN IPv4 ROUTE. The
first connect is exactly the old one, so a host with IPv4 pays nothing. The
fallback runs only when that connect failed, the target is IPv4-only (a
literal, or a name that resolves to AF_INET alone), and this host has no IPv4
route to it. The route check is a UDP ``connect`` on an unconnected socket: a
kernel table lookup that sends no packet and contacts no third party (security
review 60334 Q5, the same constraint stage 1's ``uplink`` module follows). When
the route check says a route EXISTS, nothing below runs — zero resolver calls
— because a timeout or refusal over a working IPv4 path is the peer's answer,
not a family problem. For a NAME the check first asks about an IPv4 DEFAULT
route (a fixed route key, like :func:`network_key`'s), so a host with IPv4 also
makes zero resolver calls for names; only without one is the name resolved to
find out whether it is IPv4-only (review M1).

ONE DEADLINE. Everything — first connect, route check, every ``getaddrinfo``
and the synthesised connect — shares one absolute deadline taken when the dial
started, so the fallback never makes a dial outlast the budget its caller
gave. ``getaddrinfo`` has no timeout parameter, so each lookup runs in a
short-lived daemon thread that is abandoned (not killed — it ends on its own at
the system resolver's timeout) once the deadline passes; the dial then
surfaces its ORIGINAL error (60334 S1).

HOW IT SYNTHESISES — OS FIRST. (1) ``getaddrinfo(<v4 literal>, AF_INET6)``:
macOS (and any resolver implementing RFC 7050 discovery itself) returns the
NAT64 address for a literal on a NAT64 network, so the OS's own knowledge wins.
(2) Otherwise the PREF64 is discovered by resolving ``ipv4only.arpa`` for AAAA
(RFC 7050 §3) and locating its well-known IPv4 addresses 192.0.0.170/171 at one
of the RFC 6052 prefix lengths; the IPv4 target is then embedded in it.
PREF64 from a Router Advertisement (RFC 8781) is NOT read: there is no portable
userspace API for RA options (macOS exposes none to unprivileged processes, and
Linux would need a raw ICMPv6 socket or netlink), so a resolver-visible prefix
is the only source here.

THE PREFIX IS AN UNTRUSTED, NETWORK-SCOPED HINT. It comes from whatever DNS
this network handed us, so:

* it is held in memory only, keyed by the current network's identity (the
  kernel-chosen IPv6 source /64 for an off-host route — again a no-packet
  lookup), and dropped the moment that identity changes. It is never written
  to disk and never carried to another network, so a prefix learned on a
  hostile café Wi-Fi cannot steer dials made from home;
* it is validated per RFC 6052/7050 before use (allowed lengths, a zero
  u-octet, and no synthesised address in loopback, link-local, multicast,
  unspecified, IPv4-mapped/compatible or site-local space; the well-known
  prefix 64:ff9b::/96 never carries a non-global IPv4 address, RFC 6052 §3.1);
  The same validation applies to every source — an OS answer, the hint and
  discovery alike (60334 S2);
* a prefix is cached only once a dial through it has CONNECTED, and a cached
  prefix whose dial fails is dropped, so the next attempt rediscovers instead
  of reusing a stale or wrong hint (60334 S3);
* a synthesised address is used ONLY TO DIAL. The fallback leaves the existing
  handshake checks unchanged: the socket it returns goes through the same mesh
  handshake as any other socket, and this module adds no trust of its own.

FAIL-SAFE. Anything going wrong in the fallback (a resolver that raises, no
prefix, an invalid one, the synthesised connect failing) surfaces the ORIGINAL
connect error unchanged, so every caller's ``connect_failed:<Class>`` reason
reads exactly as it did before this module existed.

A log line (``nat64 dial synthesis used (prefix source: os|ipv4only.arpa|hint)``)
records when a fallback dial succeeded; it carries no peer address beyond what
the caller's own dial audit row already holds — none at all, in fact.
"""

from __future__ import annotations

import errno
import ipaddress
import logging
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

#: The errnos that MEAN "this host has no route in that family".
_NO_ROUTE_ERRNOS = frozenset(
    code
    for code in (
        getattr(errno, "ENETUNREACH", None),
        getattr(errno, "EHOSTUNREACH", None),
    )
    if code is not None
)

#: RFC 7050 §2.2: the IPv4 addresses ``ipv4only.arpa`` is defined to hold.
_WELL_KNOWN_IPV4 = frozenset(
    {ipaddress.IPv4Address("192.0.0.170"), ipaddress.IPv4Address("192.0.0.171")}
)
_DISCOVERY_NAME = "ipv4only.arpa"

#: RFC 6052 §2.2 prefix lengths, longest first: /96 is by far the common case.
PREFIX_LENGTHS = (96, 64, 56, 48, 40, 32)

#: RFC 6052 §2.1 well-known prefix.
WELL_KNOWN_PREFIX = ipaddress.IPv6Network("64:ff9b::/96")

#: Route key for the network-identity lookup only — no packet is sent to it.
_V6_ROUTE_KEY = ("2001:4860:4860::8888", 53)

#: Route key for "does this host have an IPv4 default route at all?", asked
#: before resolving a NAME (review M1). A UDP connect to it is a kernel route
#: lookup only; no packet leaves.
_V4_ROUTE_KEY = "8.8.8.8"

SOURCE_OS = "os"
SOURCE_DISCOVERY = "ipv4only.arpa"
SOURCE_HINT = "hint"

Resolver = Callable[..., list[tuple[Any, ...]]]
RouteCheck = Callable[[str], "bool | None"]
NetworkKey = Callable[[], "str | None"]
Connector = Callable[..., socket.socket]


@dataclass(frozen=True)
class Pref64:
    """A validated NAT64 prefix: the first ``length`` bits of ``prefix``."""

    prefix: ipaddress.IPv6Address
    length: int


# --- RFC 6052 address arithmetic -------------------------------------------------


def _v4_byte_positions(length: int) -> list[int]:
    """Byte indices of the embedded IPv4 address for a prefix of ``length`` bits.

    RFC 6052 §2.2: the IPv4 bytes follow the prefix, skipping byte 8 (bits
    64-71, the reserved "u" octet) for every length shorter than /96.
    """
    return [i for i in range(length // 8, 16) if i != 8][:4]


def embed(pref: Pref64, v4: ipaddress.IPv4Address) -> ipaddress.IPv6Address:
    """The IPv4-embedded IPv6 address for ``v4`` under ``pref`` (RFC 6052 §2.2)."""
    out = bytearray(16)
    keep = pref.length // 8
    out[:keep] = pref.prefix.packed[:keep]
    for index, byte in zip(_v4_byte_positions(pref.length), v4.packed, strict=True):
        out[index] = byte
    # u-octet and suffix stay zero, as RFC 6052 requires of a synthesiser.
    return ipaddress.IPv6Address(bytes(out))


def extract(address: ipaddress.IPv6Address, length: int) -> ipaddress.IPv4Address | None:
    """The IPv4 address embedded in ``address`` at ``length``; ``None`` if malformed.

    Malformed means a non-zero u-octet (RFC 6052 §2.2: "MUST be set to zero"),
    which an honest NAT64 never produces and which is exactly the kind of
    "weird bits" a hostile resolver would have to emit to smuggle a prefix.
    """
    packed = address.packed
    if length < 96 and packed[8] != 0:
        return None
    return ipaddress.IPv4Address(bytes(packed[i] for i in _v4_byte_positions(length)))


def _usable_destination(address: ipaddress.IPv6Address) -> bool:
    """May a synthesised address be dialled at all?

    Refuses everything that would turn an untrusted prefix into a way to make
    this device connect somewhere local or special: loopback, link-local,
    multicast, unspecified, site-local, and IPv4-mapped/compatible (those are
    carried by the IPv4 stack — the very stack that has no route).
    """
    if (
        address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or address.is_site_local
        or address.ipv4_mapped is not None
    ):
        return False
    # ::/80 — an "all-zero prefix": the deprecated IPv4-compatible block (::/96)
    # and its neighbours are no NAT64's prefix, and only an attacker emits one.
    return address.packed[:10] != bytes(10)


def validate_prefix(pref: Pref64, v4: ipaddress.IPv4Address) -> bool:
    """Is ``pref`` acceptable for synthesising ``v4``? (RFC 6052 §2-3, RFC 7050 §3.)"""
    if pref.length not in PREFIX_LENGTHS:
        return False
    if pref.length < 96 and pref.prefix.packed[8] != 0:
        return False
    network = ipaddress.IPv6Network((pref.prefix, pref.length), strict=False)
    # RFC 6052 §3.1: the well-known prefix MUST NOT be used with non-global
    # IPv4 addresses; a network-specific prefix may (an operator's own NAT64).
    if network == WELL_KNOWN_PREFIX and not v4.is_global:
        return False
    return _usable_destination(embed(pref, v4))


def _prefix_of(address: ipaddress.IPv6Address, length: int) -> Pref64:
    keep = length // 8
    return Pref64(ipaddress.IPv6Address(address.packed[:keep] + bytes(16 - keep)), length)


def prefix_from_discovery(addresses: list[ipaddress.IPv6Address]) -> Pref64 | None:
    """Locate the PREF64 in ``ipv4only.arpa``'s AAAA answers (RFC 7050 §3).

    A prefix's own bits can hold the 192.0.0.170 pattern at a second length, so
    one answer may read as several candidate prefixes. RFC 7050 §3 resolves that
    with 192.0.0.171: a candidate under which BOTH well-known addresses appear
    wins; otherwise the first match (answer order, longest length first) does,
    since a network answering with only one record leaves nothing to compare.
    """
    candidates: dict[Pref64, set[ipaddress.IPv4Address]] = {}
    for address in addresses:
        for length in PREFIX_LENGTHS:
            found = extract(address, length)
            if found is not None and found in _WELL_KNOWN_IPV4:
                candidates.setdefault(_prefix_of(address, length), set()).add(found)
    for pref, seen in candidates.items():
        if seen == _WELL_KNOWN_IPV4:
            return pref
    return next(iter(candidates), None)


# --- the network-scoped, in-memory hint ------------------------------------------


class _HintCache:
    """At most ONE PREF64, bound to the network identity it was learned on.

    Memory only, by design (see the module docstring): never persisted, and
    replaced or dropped as soon as the identity it was learned under differs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: str | None = None
        self._pref: Pref64 | None = None

    def get(self, key: str | None) -> Pref64 | None:
        with self._lock:
            if key is None or key != self._key:
                # A different network (or one we cannot identify): whatever we
                # learned elsewhere is not evidence about this one.
                self._key, self._pref = None, None
                return None
            return self._pref

    def put(self, key: str | None, pref: Pref64) -> None:
        if key is None:
            return
        with self._lock:
            self._key, self._pref = key, pref

    def drop(self, key: str | None, pref: Pref64) -> None:
        """Forget ``pref`` if it is still the hint for ``key``.

        Compare-and-clear: a concurrent dial may already have learned a fresh
        prefix, and a failure seen through the old one says nothing about it.
        """
        with self._lock:
            if self._key == key and self._pref == pref:
                self._key, self._pref = None, None

    def clear(self) -> None:
        with self._lock:
            self._key, self._pref = None, None


HINTS = _HintCache()


# --- passive readings ------------------------------------------------------------


def ipv4_route(address: str) -> bool | None:
    """Does this host have an IPv4 route to ``address``? No packet is sent.

    ``False`` only on a no-route errno; any other failure is ``None`` (no
    claim), which callers treat as "do not assume the family is missing"
    unless the dial itself already failed with a no-route errno.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError:
        return None
    try:
        sock.connect((address, 9))
        return True
    except OSError as exc:
        return False if exc.errno in _NO_ROUTE_ERRNOS else None
    except (TypeError, ValueError, OverflowError):
        return None
    finally:
        sock.close()


def network_key() -> str | None:
    """An identity for the network this host's IPv6 traffic leaves on, or ``None``.

    The kernel-chosen source /64 toward an off-host IPv6 route key (a UDP
    ``connect``: a table lookup, no packet). A new network hands out a new
    prefix, so the key changes on every move between networks that matter
    here; ``None`` (no IPv6 route, unreadable) means no hint is cached at all.
    """
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    except OSError:
        return None
    try:
        sock.connect(_V6_ROUTE_KEY)
        source = sock.getsockname()[0]
    except (OSError, TypeError, ValueError):
        return None
    finally:
        sock.close()
    try:
        address = ipaddress.IPv6Address(str(source).split("%", 1)[0])
    except ValueError:
        return None
    return str(ipaddress.IPv6Network((address, 64), strict=False))


# --- the dial deadline -----------------------------------------------------------


class _DeadlineExpired(Exception):
    """The dial's budget ran out inside the fallback.

    Deliberately NOT an ``OSError``: the OS-synthesis step swallows resolver
    ``OSError``s as "the OS has nothing", and an expired budget must instead end
    the whole fallback so the caller sees its original error.
    """


def _remaining(deadline: float | None) -> float | None:
    """Seconds left before ``deadline``; raises once none are. ``None``: unbounded."""
    if deadline is None:
        return None
    left = deadline - time.monotonic()
    if left <= 0:
        raise _DeadlineExpired
    return left


def _resolve(resolver: Resolver, deadline: float | None, *args: Any) -> list[tuple[Any, ...]]:
    """``resolver(*args)``, abandoned once ``deadline`` passes (60334 S1).

    ``getaddrinfo`` takes no timeout and cannot be interrupted, so the lookup
    runs in a daemon thread and the dial stops waiting for it at the deadline.
    An abandoned thread is harmless: it holds no lock or socket of ours, writes
    only into a box nobody reads any more, and ends on its own when the system
    resolver gives up (its own resolv.conf timeout), so a broken-DNS network
    costs at most one such thread per lookup in flight — never an unbounded dial.
    A blocking dial (``timeout=None``) keeps blocking semantics: called inline.
    """
    left = _remaining(deadline)
    if left is None:
        return resolver(*args)
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = resolver(*args)
        except BaseException as exc:  # handed to the waiting dial, re-raised there
            box["error"] = exc

    thread = threading.Thread(target=run, name="nat64-getaddrinfo", daemon=True)
    thread.start()
    thread.join(left)
    if thread.is_alive():
        raise _DeadlineExpired
    if "error" in box:
        raise box["error"]
    return box["value"]


# --- synthesis -------------------------------------------------------------------


@dataclass(frozen=True)
class _Synthesis:
    address: ipaddress.IPv6Address
    source: str
    pref: Pref64


def _v6_answers(
    resolver: Resolver, host: str, port: int, deadline: float | None = None
) -> list[ipaddress.IPv6Address]:
    out: list[ipaddress.IPv6Address] = []
    for info in _resolve(resolver, deadline, host, port, socket.AF_INET6, socket.SOCK_STREAM):
        if info[0] != socket.AF_INET6:
            continue
        try:
            out.append(ipaddress.IPv6Address(str(info[4][0]).split("%", 1)[0]))
        except (ValueError, IndexError):
            continue
    return out


def _os_synthesis(answer: ipaddress.IPv6Address, v4: ipaddress.IPv4Address) -> Pref64 | None:
    """The prefix under which the OS's ``answer`` is a valid synthesis of ``v4``.

    The OS is held to the SAME rules as discovery and the hint (60334 S2):
    ``answer`` must be exactly ``v4`` embedded at an RFC 6052 length under a
    prefix ``validate_prefix`` accepts — so the well-known prefix never carries
    a non-global IPv4 here either, and an arbitrary AAAA for the literal (not a
    synthesis at all) can never redirect the dial.
    """
    for length in PREFIX_LENGTHS:
        if extract(answer, length) != v4:
            continue
        pref = _prefix_of(answer, length)
        if validate_prefix(pref, v4) and embed(pref, v4) == answer:
            return pref
    return None


def synthesise(
    v4: ipaddress.IPv4Address,
    port: int,
    *,
    resolver: Resolver,
    key: str | None,
    hints: _HintCache = HINTS,
    deadline: float | None = None,
) -> _Synthesis | None:
    """An IPv6 address reaching ``v4`` through this network's NAT64, its source and prefix.

    Raises whatever ``resolver`` raises (or :class:`_DeadlineExpired`);
    :func:`fallback` turns that into the original connect error. Caching is
    the caller's: a discovered prefix is remembered only once a dial through it
    has connected (see :func:`fallback`).
    """
    # (1) The OS.
    try:
        os_answers = _v6_answers(resolver, str(v4), port, deadline)
    except (OSError, UnicodeError):
        # glibc answers an IPv4 literal asked for AF_INET6 with EAI_ADDRFAMILY /
        # EAI_NONAME (it never synthesises; only DNS64 or macOS's resolver
        # does), so a raise here means "the OS has nothing", not "give up":
        # discovery below is exactly the case for it.
        os_answers = []
    for answer in os_answers:
        pref = _os_synthesis(answer, v4)
        if pref is not None:
            return _Synthesis(answer, SOURCE_OS, pref)
    # (2) A prefix already learned on THIS network.
    cached = hints.get(key)
    if cached is not None and validate_prefix(cached, v4):
        return _Synthesis(embed(cached, v4), SOURCE_HINT, cached)
    # (3) RFC 7050 discovery.
    pref = prefix_from_discovery(_v6_answers(resolver, _DISCOVERY_NAME, port, deadline))
    if pref is None or not validate_prefix(pref, v4):
        return None
    return _Synthesis(embed(pref, v4), SOURCE_DISCOVERY, pref)


def _ipv4_only_target(
    host: str, port: int, resolver: Resolver, deadline: float | None = None
) -> ipaddress.IPv4Address | None:
    """The IPv4 address to synthesise for, when ``host`` is IPv4-only; else ``None``."""
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        parsed = None
    if isinstance(parsed, ipaddress.IPv4Address):
        return parsed
    if parsed is not None:
        return None  # already IPv6: nothing to synthesise
    # A name. If it has ANY IPv6 answer the original connect already tried it
    # (``socket.create_connection`` walks every family), so leave it alone.
    infos = _resolve(resolver, deadline, host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    families = {info[0] for info in infos}
    if families != {socket.AF_INET}:
        return None
    return ipaddress.IPv4Address(str(infos[0][4][0]))


def create_connection(
    address: tuple[str, int],
    timeout: float | None = None,
    *,
    resolver: Resolver | None = None,
    route_check: RouteCheck | None = None,
    key_reader: NetworkKey | None = None,
    connector: Connector | None = None,
) -> socket.socket:
    """``socket.create_connection`` with a NAT64 fallback for IPv4-only targets.

    ``timeout`` bounds the WHOLE dial, fallback included: one deadline is taken
    here, before the first connect, and the fallback spends only what is left.
    The keyword hooks exist for tests (a fake resolver and route check); the
    defaults are looked up at call time so a test patching
    ``socket.create_connection`` still sees every dial.
    """
    connect = connector if connector is not None else socket.create_connection
    host, port = address
    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        return connect((host, port), timeout=timeout)
    except OSError as original:
        sock = fallback(
            host,
            port,
            original,
            timeout=timeout,
            deadline=deadline,
            resolver=resolver,
            route_check=route_check,
            key_reader=key_reader,
            connector=connect,
        )
        if sock is None:
            raise
        return sock


def fallback(
    host: str,
    port: int,
    original: OSError,
    *,
    timeout: float | None,
    deadline: float | None = None,
    resolver: Resolver | None = None,
    route_check: RouteCheck | None = None,
    key_reader: NetworkKey | None = None,
    connector: Connector | None = None,
    hints: _HintCache = HINTS,
) -> socket.socket | None:
    """The synthesised connection after ``original`` failed, or ``None`` to re-raise it.

    PUBLIC so a dial site that builds and connects its own socket (rather than
    calling :func:`create_connection`) can hand its connect error here from its
    ``except OSError`` and get the same fallback — one implementation, however
    the first connect was made. Such a site should pass ``deadline`` (a
    ``time.monotonic()`` value taken before its own connect) so the fallback
    shares the dial's budget; without one, ``timeout`` counts from this call.
    ``None`` is returned — the caller re-raises ``original`` — whenever the
    deadline passes, at any step.
    """
    resolver = resolver if resolver is not None else socket.getaddrinfo
    route_check = route_check if route_check is not None else ipv4_route
    key_reader = key_reader if key_reader is not None else network_key
    connect = connector if connector is not None else socket.create_connection
    if deadline is None and timeout is not None:
        deadline = time.monotonic() + timeout
    try:
        literal: ipaddress.IPv4Address | None = ipaddress.IPv4Address(host)
    except ValueError:
        literal = None
    key: str | None = None
    try:
        _remaining(deadline)
        if literal is None and route_check(_V4_ROUTE_KEY) is True:
            # A NAME on a host with an IPv4 default route: whatever it resolves
            # to has an IPv4 route, so the failure is the peer's answer. Asked
            # before resolving so such hosts make no extra resolver call (M1).
            return None
        # A literal is gated on its own route before any resolver call. A name
        # without an IPv4 default route must be resolved first (handing the
        # name to the route check would turn the "no-packet" lookup into a DNS
        # query); the original connect has just resolved it, so this is
        # normally a cache hit, and it runs under the deadline regardless.
        v4 = literal if literal is not None else _ipv4_only_target(host, port, resolver, deadline)
        if v4 is None:
            return None
        route = route_check(str(v4))
        if route is True:
            return None
        if route is None and original.errno not in _NO_ROUTE_ERRNOS:
            return None
        key = key_reader()
        synthesised = synthesise(
            v4, port, resolver=resolver, key=key, hints=hints, deadline=deadline
        )
        if synthesised is None:
            return None
        left = _remaining(deadline)
    except _DeadlineExpired:
        return None
    except (OSError, UnicodeError, ValueError, IndexError):
        return None
    try:
        sock = connect((str(synthesised.address), port), timeout=left)
    except OSError:
        if synthesised.source == SOURCE_HINT:
            # The cached prefix did not carry this dial: drop it so the next
            # attempt rediscovers rather than retrying a stale or wrong hint
            # forever (60334 S3 — e.g. a VPN keeping the same source /64 across
            # physical networks). A closed peer port drops it too; the price is
            # one ipv4only.arpa lookup on the next dial, under its deadline.
            hints.drop(key, synthesised.pref)
        return None
    if synthesised.source == SOURCE_DISCOVERY:
        # Remembered only once it has carried a connection, so the hint never
        # holds a prefix that has not worked on this network.
        hints.put(key, synthesised.pref)
    # Source only: the caller's audit row already names the endpoint it dialled.
    log.info("nat64 dial synthesis used (prefix source: %s)", synthesised.source)
    return sock
