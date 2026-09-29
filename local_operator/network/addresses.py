"""This device's OWN addresses, read from the OS's interface table.

WHY NOT ``socket.getaddrinfo(socket.gethostname())``, WHICH IS WHAT THIS
REPLACES. That was how :func:`local_operator.network.relay.advertise_endpoints`
found the addresses it publishes, and on macOS it is EMPTY: ``gethostname``
returns the Bonjour name (``damians-MacBook-Pro``), which is not in DNS or in
``/etc/hosts``, so ``getaddrinfo`` raises ``[Errno 8] nodename nor servname
provided`` and the caller's ``OSError`` branch contributes nothing. The damage is
not cosmetic — it is the answer to "where can peers reach me", so on a Mac with
the default ``listen_address: 0.0.0.0`` the record's ``advertised`` list, every
member row and every invite token carried no address at all, and a joiner was
told *"that invite names no endpoint; pass --host host:port"*. Pairing then
required a human to know and type the other machine's IP, which is exactly the
step this module removes. The lane that reported it worked around the cloud node
by mapping a hostname to a public IP in ``/etc/hosts`` and setting the hostname
to match; the fix must not need either.

WHY ``getifaddrs`` AND NOT A HOSTNAME. The interface table is the OS's own answer
to "what addresses do I hold", and it is the same source ``ip addr``/``ifconfig``
read. It cannot be wrong in the way a name lookup can: a hostname resolves to
whatever DNS or mDNS says, while this returns addresses that are actually
assigned and BOUND to this machine right now. ``psutil`` is not a dependency of
this project and adding one for this would be out of proportion, so the call is
made through ``ctypes`` here and nowhere else.

ROUTING IS A SEPARATE QUESTION FROM ENUMERATION, and the two are answered by two
mechanisms on purpose:

* :func:`_interface_ipv4s` (``getifaddrs``) supplies the SET — every non-loopback
  IPv4 address this device holds.
* :func:`_default_route_address` (a UDP ``connect`` to a globally routed address,
  which sends nothing) supplies the ORDER — the one address the kernel itself
  would use to reach the internet, hoisted to the front because it is the address
  a peer is most likely to be able to dial. A laptop has several addresses and the
  order is all the ranking there is: the receiver dials the list in order, so a
  wrong first entry costs one failed dial rather than a lost peer.

The route probe is ALSO the fallback for the platforms where the table cannot be
read (an unknown libc, a sandbox that refuses ``getifaddrs``): one address is a
far better answer than the empty list this bug was. It is never the only source
when the table works, so the two cannot disagree about the SET.

WHAT IS DELIBERATELY EXCLUDED, because a wrong address silently misleads a
joiner — worse than an empty list, which at least says "I do not know":
loopback (a peer dialling ``127.0.0.1`` reaches itself), ``169.254.0.0/16``
link-local (assigned when DHCP failed and reachable by nobody), unspecified
``0.0.0.0``, multicast and broadcast. IPv6 is not enumerated: endpoints are
``host:port`` strings and the handshake's ``clean_endpoints`` does not bracket a
literal, so an ``fe80::`` address could not be spelled here without a second
change to the wire format.
"""

from __future__ import annotations

import ctypes
import socket
import sys
from typing import Any

#: ``struct ifaddrs``' flag bits, identical in glibc and in Darwin/BSD. Only the
#: two this module needs are named; ``ifa_flags`` carries others (``IFF_POINTOPOINT``
#: among them, which a VPN tunnel carries and which is NOT excluded — a tunnel
#: address is the address a peer on that tunnel can dial).
_IFF_UP = 0x1
_IFF_LOOPBACK = 0x8

#: ``sizeof(struct sockaddr)`` on both platforms, and the offset of the IPv4
#: address inside it. ``sockaddr_in`` puts the address after the family and port,
#: which is 4 bytes in the BSD layout (``sa_len``, family, port) and 4 bytes in
#: glibc's (family, port) — so the OFFSET is shared even though the header is not.
_SOCKADDR_SIZE = 16
_V4_OFFSET = 4

#: BSD ``struct sockaddr`` leads with ``sa_len`` and puts the family in the
#: following byte; glibc has no length field and its family is a 16-bit integer.
#: Reading the family without this distinction reads glibc's low byte as the
#: family by luck on little-endian and its high byte as zero — i.e. it works on
#: x86/arm Linux and silently reports NOTHING on a big-endian host, which is the
#: class of bug that only appears on the machine you do not own.
_BSD_SOCKADDR = sys.platform.startswith(("darwin", "freebsd", "openbsd", "netbsd"))


class _IfAddrs(ctypes.Structure):
    """The first five fields of ``struct ifaddrs``, on glibc and on Darwin alike.

    Declaring the PREFIX rather than the whole struct is deliberate: the tail
    differs between the two (glibc has a union of broadcast/destination pointer;
    Darwin has a bare ``ifa_dstaddr``) and this module never reads it, so pinning
    a layout that is only correct on one of them would be a bug waiting for the
    other platform. These five fields are in the same order, with the same types,
    in both.
    """


_IfAddrs._fields_ = [  # type: ignore[attr-defined]
    ("ifa_next", ctypes.POINTER(_IfAddrs)),
    ("ifa_name", ctypes.c_char_p),
    ("ifa_flags", ctypes.c_uint),
    ("ifa_addr", ctypes.c_void_p),
    ("ifa_netmask", ctypes.c_void_p),
]


def _libc() -> Any:
    """``libc`` for the two symbols this module calls, or ``OSError``.

    ``CDLL(None)`` is the handle to the symbols this process already has loaded:
    on Darwin that is ``libSystem`` (which carries ``getifaddrs``) and on Linux it
    is the loader's global namespace, where glibc's are. The named fallbacks are
    for a Python linked some other way (musl, a statically linked runtime).
    """
    for name in (None, "libc.so.6", "libSystem.B.dylib", "libc.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError("no libc available for getifaddrs")


def _family_of(address: int, *, bsd: bool | None = None) -> int:
    """The ``sa_family`` of the ``sockaddr`` at ``address``, per the platform layout.

    The two layouts differ in WHERE the family sits: BSD leads with ``sa_len`` and puts
    the family in the first byte, glibc stores it as a ``uint16`` in host order.
    ``bsd`` is a parameter rather than a read of ``_BSD_SOCKADDR`` so a test can drive
    BOTH layouts on whichever platform CI runs — the Darwin branch had no test that
    executes in CI for as long as it was inferred from the machine (review round 1, M5).
    """
    raw = ctypes.string_at(address, _SOCKADDR_SIZE)
    if _BSD_SOCKADDR if bsd is None else bsd:
        return raw[1]
    return _glibc_family(raw[0], raw[1], sys.byteorder)


def _glibc_family(first: int, second: int, byteorder: str) -> int:
    """The ``uint16`` glibc stores in the first two bytes, read in HOST order.

    Split out so the RULE can be tested on any host. The read is native-order, and only
    a big-endian machine would catch the fixed little-endian shift it replaces (s390x and
    ppc64 read a family 256x the real one — `0x0002` as 512 — and then skip every
    interface as "not AF_INET"); a CI runner here is little-endian, so the test drives
    both orders through this function rather than pretending one host is both (review
    round 1, M3).
    """
    if byteorder == "little":
        return first | (second << 8)
    return (first << 8) | second


def _interface_ipv4s() -> list[tuple[str, int, str]] | None:
    """``[(name, flags, address)]`` for every non-loopback IPv4 address held.

    ``None`` means the table could not be read at all (no libc, ``getifaddrs``
    failed, the symbols are missing) — distinct from ``[]``, which means the table
    was read and this device holds no non-loopback IPv4 address. The caller treats
    them differently: ``None`` falls back to the route probe, ``[]`` is reported as
    no addresses, which is the honest answer for a device with only loopback.
    """
    try:
        libc = _libc()
        getifaddrs = libc.getifaddrs
        getifaddrs.restype = ctypes.c_int
        getifaddrs.argtypes = [ctypes.POINTER(ctypes.POINTER(_IfAddrs))]
        freeifaddrs = libc.freeifaddrs
        freeifaddrs.restype = None
        freeifaddrs.argtypes = [ctypes.POINTER(_IfAddrs)]
    except (OSError, AttributeError):
        return None

    head = ctypes.POINTER(_IfAddrs)()
    try:
        if getifaddrs(ctypes.byref(head)) != 0:
            return None
    except (OSError, ValueError):  # a refused call, not a wrong answer
        return None
    try:
        found: list[tuple[str, int, str]] = []
        node = head
        while node:
            entry = node.contents
            # A pure loopback or down interface is skipped here rather than by its
            # address, so an interface the OS itself calls loopback cannot be
            # published even if it somehow holds a routable address.
            flags = int(entry.ifa_flags)
            if entry.ifa_addr and (flags & _IFF_LOOPBACK) == 0:
                address = entry.ifa_addr
                if _family_of(address) == socket.AF_INET:
                    raw = ctypes.string_at(address + _V4_OFFSET, 4)
                    found.append(
                        (
                            (entry.ifa_name or b"").decode("utf-8", "replace"),
                            flags,
                            socket.inet_ntoa(raw),
                        )
                    )
            node = entry.ifa_next
        return found
    finally:
        # The list is libc's allocation, and a test that enumerates repeatedly
        # would leak one per call without this.
        freeifaddrs(head)


def interface_for_address(address: str) -> str | None:
    """The interface ``address`` is held on, or ``None`` when unobservable.

    :func:`_interface_ipv4s` already reads the NAME for every IPv4 the host
    holds and :func:`local_ipv4_addresses` discards it; this is the mapping back
    from an address to its interface, for the callers that PUBLISH the fact
    (the readiness rows say "this device routes to it from <addr> via <if>",
    and the interface name is the observed half of "is a tunnel or a NIC").
    ``None`` covers both "the table could not be read" and "no interface holds
    this address" — an unobservable fact reads as one, never as a guess.
    """
    entries = _interface_ipv4s()
    if not entries:
        return None
    for name, _flags, held in entries:
        if held == address:
            return name
    return None


def route_observation(host: str, port: int) -> dict[str, Any]:
    """Observed facts about the route this device would take to ``host:port``.

    A UDP ``connect`` sends no packet: it resolves a route and binds the local
    end, so ``source_address`` is the address the KERNEL itself would use —
    the same trick :func:`_default_route_address` uses, parameterised by
    destination — and the interface name comes back through
    :func:`interface_for_address`. Because it sends nothing, it is safe to run
    once per endpoint on a report path.

    ``error`` names the exception class when the lookup failed (an
    unroutable destination, a name that does not resolve); ``""`` is success.
    Both address fields are ``None``-able on purpose: the consumer publishes
    these as OBSERVED facts and must be able to say "not observable" rather
    than leave a value that reads as one.
    """
    # The route decision is keyed on the ADDRESS; the port only has to be
    # connectable. Port 0 is not (EADDRNOTAVAIL on macOS), and a caller that
    # could not parse a port at all still deserves the address facts, so a
    # non-positive port becomes discard (9).
    if port <= 0:
        port = 9
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError as exc:
        return {"source_address": None, "interface": None, "error": exc.__class__.__name__}
    try:
        sock.connect((host, port))
        address = str(sock.getsockname()[0])
    except OSError as exc:
        return {"source_address": None, "interface": None, "error": exc.__class__.__name__}
    except (TypeError, ValueError, OverflowError):
        # A garbage host or an out-of-range port: the same class of answer.
        return {"source_address": None, "interface": None, "error": "bad_endpoint"}
    finally:
        sock.close()
    return {
        "source_address": address,
        "interface": interface_for_address(address),
        "error": "",
    }


def _default_route_address() -> str | None:
    """The source address the kernel picks to reach a globally routed host.

    A UDP ``connect`` sends no packet: it resolves a route and binds the local
    end, so this is a table lookup rather than a probe of anything on the network,
    and it works with no traffic and no DNS (the address is a literal). It is the
    strongest single answer to "which of my addresses is routable" — the kernel
    itself ranks it first for every destination off this host — and it is the
    fallback for a platform where the interface table cannot be read.

    ``None`` when there is no default route at all (a sealed sandbox, an air-gapped
    host), which is a legitimate state and not an error to report.
    """
    for probe in (("1.1.1.1", 53), ("8.8.8.8", 53)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(probe)
            address = str(sock.getsockname()[0])
        except OSError:
            continue
        finally:
            sock.close()
        if is_advertisable_ipv4(address):
            return address
    return None


def is_advertisable_ipv4(text: str) -> bool:
    """May ``text`` be published as this device's address for peers to dial?

    The predicate behind the exclusion list in this module's docstring, exported so
    the test that drives the real enumeration can assert the PROPERTY over whatever
    this machine happens to hold (a runner's addresses are not something a test can
    fix) rather than pinning a fixture to one host's interface names.
    """
    try:
        socket.inet_aton(text)
    except OSError:
        return False
    octets = socket.inet_aton(text)
    first, second = octets[0], octets[1]
    if first == 127:  # loopback: a peer dialling it reaches itself
        return False
    if first == 0:  # unspecified / "this network"
        return False
    if first == 169 and second == 254:  # link-local, the no-DHCP address
        return False
    if first >= 224:  # multicast (224-239) and the limited broadcast 255.255.255.255
        return False
    return True


def local_ipv4_addresses() -> list[str]:
    """This device's own non-loopback IPv4 addresses, most dialable first.

    The one function ``relay.advertise_endpoints`` calls. Deduplicated, filtered by
    :func:`is_advertisable_ipv4`, and ordered: the default-route address first,
    then the addresses the OS reports on interfaces that are UP. An empty list is a
    real answer — a device holding only loopback has nothing a peer can dial, and
    saying so is the point (see ``advertise_endpoints``' own dial-only branch).
    """
    entries = _interface_ipv4s()
    if entries is None:
        # The table is unreadable on this platform. One routable address beats the
        # empty list that started this: it is the same answer the kernel gives, and
        # it is the best a host with no interface table can offer.
        fallback = _default_route_address()
        return [fallback] if fallback else []

    primary = _default_route_address()
    ordered: list[str] = []

    def add(address: str) -> None:
        if is_advertisable_ipv4(address) and address not in ordered:
            ordered.append(address)

    if primary:
        add(primary)
    for _name, flags, address in entries:
        if flags & _IFF_UP:
            add(address)
    return ordered
