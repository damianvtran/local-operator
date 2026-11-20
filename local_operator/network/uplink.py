"""This device's own uplink: which address families it has a route in, right now.

WHY THIS EXISTS (incident 2026-11-18): the operator's Mac joined an IPv6-only
network while node2 advertised IPv4 endpoints only. Every dial to node2 failed
with "no route", and every surface — the peer rows' ``address_unreachable``,
the redial loop's growing backoff, ``doctor``'s remedies — pointed at node2,
the one machine that had nothing wrong with it. The missing fact was about
THIS device: it had no IPv4 route at all. This module reads that fact so the
link state (``local_offline``), the redial loop (kick a peer the moment a
route to it appears) and ``ready``/``doctor`` (an ``uplink`` row first) can
name it.

TWO READINGS, TWO SCOPES (review R1, security U1). The DEVICE reading
(:func:`read_uplink`) asks about public route keys and so can only say "no
route to the internet"; it feeds the device row and nothing else. Every claim
about a PEER (``local_offline``, the redial kick) asks the kernel about that
peer's OWN literal endpoints (:func:`endpoint_has_route`): a LAN, loopback,
VPN or link-local peer keeps its route when the default route is gone, and
must keep being dialled. Both readings are ADVISORY: nothing here pauses or
skips a dial, so the redial loop's baseline backoff retry always runs.

PASSIVE ONLY — A SECURITY CONSTRAINT, NOT AN OPTIMISATION (security review,
session 60334; manager decision 79). The reading is a ROUTE TABLE LOOKUP: a
UDP ``connect`` on an unconnected datagram socket sends no packet, it only
asks the kernel which route and source address it would use. Nothing here
opens a TCP connection, sends a datagram, resolves a name or contacts any
third-party or anycast endpoint, so it is not a beacon and cannot leak this
device's presence to anyone. The literal destinations below are only route
keys for "somewhere off this host" in each family. A design bullet proposing a
periodic "cheap connect to a fixed anycast endpoint" was REJECTED on exactly
those grounds; do not add one here.

HONEST ABOUT WHAT IT CANNOT SEE. A route proves the kernel would try, not that
anything answers. So this module never claims "the internet works"; it says
which families have a route, and when a family cannot be read it says
``None`` (unknown), never "no route". A device with a route that still cannot
reach a peer is reported as "<peer> isn't answering from this network" —
never as a firewall or allowlist, because no cause beyond that is observable.

FAIL-SAFE. Every caller treats a reading that raises, or one whose answer is
``unknown`` (``None``), exactly as the code behaved before this module
existed: no ``local_offline``, no kick, no uplink claim.
"""

from __future__ import annotations

import errno
import ipaddress
import socket
from collections.abc import Callable, Iterable, Mapping
from typing import Any

#: The states a reading can take. ``unknown`` is a first-class answer: it is
#: what a sandbox, a platform without the syscall, or a reader that raised
#: reports, and every consumer reads it as "make no claim".
OFFLINE = "offline"
IPV4_ONLY = "ipv4_only"
IPV6_ONLY = "ipv6_only"
DUAL = "dual"
UNKNOWN = "unknown"

#: Route keys only — no packet is ever sent to them (see the module docstring).
#: Two per family, mirroring ``addresses._default_route_address``: a host whose
#: table routes one prefix but not the other still gets an answer.
_ROUTE_KEYS: dict[str, tuple[tuple[str, int], ...]] = {
    "ipv4": (("1.1.1.1", 53), ("8.8.8.8", 53)),
    "ipv6": (("2606:4700:4700::1111", 53), ("2001:4860:4860::8888", 53)),
}

#: The errnos that MEAN "no route in this family". Anything else (a permission
#: refusal under a sandbox, an exotic platform error) is not evidence of a
#: missing route and reads as unknown. ``EAFNOSUPPORT``/``EPROTONOSUPPORT`` at
#: socket creation mean the stack has no such family at all — no route either.
_NO_ROUTE_ERRNOS = frozenset(
    code
    for code in (
        getattr(errno, "ENETUNREACH", None),
        getattr(errno, "EHOSTUNREACH", None),
        getattr(errno, "EADDRNOTAVAIL", None),
        getattr(errno, "EAFNOSUPPORT", None),
        getattr(errno, "EPROTONOSUPPORT", None),
    )
    if code is not None
)

_FAMILIES = {"ipv4": socket.AF_INET, "ipv6": getattr(socket, "AF_INET6", None)}

RouteReader = Callable[[str], "bool | None"]


def family_has_route(family: str) -> bool | None:
    """Whether the kernel has a route off this host in ``family`` (``ipv4``/``ipv6``).

    ``True`` when a datagram ``connect`` to a route key binds a source address,
    ``False`` when it fails with a no-route errno for every key, ``None`` when
    the answer could not be read (an unexpected error is not evidence).
    """
    af = _FAMILIES.get(family)
    if af is None:
        return False if family == "ipv6" else None
    saw_no_route = False
    for key in _ROUTE_KEYS[family]:
        try:
            sock = socket.socket(af, socket.SOCK_DGRAM)
        except OSError as exc:
            if exc.errno in _NO_ROUTE_ERRNOS:
                return False
            return None
        try:
            # No packet leaves: connect() on SOCK_DGRAM is a route lookup.
            sock.connect(key)
            source = str(sock.getsockname()[0])
        except OSError as exc:
            if exc.errno in _NO_ROUTE_ERRNOS:
                saw_no_route = True
                continue
            return None
        finally:
            sock.close()
        # An unspecified source means the kernel bound nothing routable.
        if source in ("0.0.0.0", "::"):
            saw_no_route = True
            continue
        return True
    return False if saw_no_route else None


def classify(ipv4: bool | None, ipv6: bool | None) -> str:
    """The one state word for a pair of per-family answers.

    ``offline`` needs BOTH families proven route-less; one unreadable family
    makes the whole state ``unknown`` (the per-family answers still ride the
    reading, so a consumer can act on the family it does know).
    """
    if ipv4 is None or ipv6 is None:
        return UNKNOWN
    if ipv4 and ipv6:
        return DUAL
    if ipv4:
        return IPV4_ONLY
    if ipv6:
        return IPV6_ONLY
    return OFFLINE


def read_uplink(route_reader: RouteReader = family_has_route) -> dict[str, Any]:
    """One passive reading: ``{"state", "ipv4", "ipv6"}``.

    A reader that raises yields ``unknown`` with both families ``None`` — the
    fail-safe every consumer maps to "behave as before".
    """
    try:
        ipv4 = route_reader("ipv4")
        ipv6 = route_reader("ipv6")
    except Exception:  # noqa: BLE001 — a reading must never break its caller
        ipv4 = ipv6 = None
    return {"state": classify(ipv4, ipv6), "ipv4": ipv4, "ipv6": ipv6}


def unknown_reading() -> dict[str, Any]:
    """The reading a caller substitutes when it could not take one at all."""
    return {"state": UNKNOWN, "ipv4": None, "ipv6": None}


def _literal(endpoint: str) -> tuple[str, str, int] | None:
    """``(family, host, port)`` for a literal ``host:port`` / ``[v6]:port``; else ``None``.

    A hostname has no family until it resolves, and resolving would be a probe
    this module does not make, so it is ``None`` (could be either, unknown).

    IPv4-MAPPED IPv6 (``[::ffff:a.b.c.d]``, review R2) is IPv4: the kernel
    carries it over the IPv4 stack, so on an IPv4-only host it is routable and
    on an IPv6-only host it is not. Classifying it by its spelling would invert
    both answers.
    """
    text = str(endpoint or "").strip()
    port = 0
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        if rest.startswith(":") and rest[1:].isdigit():
            port = int(rest[1:])
    elif text.count(":") == 1:
        host, _, tail = text.rpartition(":")
        port = int(tail) if tail.isdigit() else 0
    else:
        host = text
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return None
    if parsed.version == 6:
        mapped = getattr(parsed, "ipv4_mapped", None)
        if mapped is not None:
            return "ipv4", str(mapped), port
        return "ipv6", host, port
    return "ipv4", host, port


def endpoint_family(endpoint: str) -> str | None:
    """``ipv4``/``ipv6`` for a literal endpoint; ``None`` for a hostname or junk."""
    literal = _literal(endpoint)
    return literal[0] if literal else None


def endpoint_has_route(endpoint: str) -> bool | None:
    """Whether the kernel has a route to THIS endpoint's own address.

    WHY PER ENDPOINT (review R1 / security U1): "no route to the public route
    keys" is not "no route to a peer". A LAN peer (``192.168.x``), a loopback
    peer, a VPN or a link-local one has its own, more specific route that
    survives a missing default route, and those peers still answer when the
    internet is gone. So the peer-level claim asks the kernel about the peer's
    own literal address — the same passive datagram ``connect`` as
    :func:`family_has_route` (a route lookup: no packet, no DNS).

    ``True``: a route and a bound source address. ``False``: a no-route errno
    (or an unspecified source). ``None``: a hostname (never resolved here), an
    unreadable family, or any other error — none of which is evidence.
    """
    literal = _literal(endpoint)
    if literal is None:
        return None
    family, host, port = literal
    af = _FAMILIES.get(family)
    if af is None:
        return False if family == "ipv6" else None
    try:
        sock = socket.socket(af, socket.SOCK_DGRAM)
    except OSError as exc:
        return False if exc.errno in _NO_ROUTE_ERRNOS else None
    try:
        # No packet leaves: connect() on SOCK_DGRAM is a route lookup. Port 0
        # is refused by some stacks, so an endpoint without one uses discard (9).
        sock.connect((host, port or 9))
        source = str(sock.getsockname()[0])
    except OSError as exc:
        return False if exc.errno in _NO_ROUTE_ERRNOS else None
    finally:
        sock.close()
    return source not in ("0.0.0.0", "::")


EndpointReader = Callable[[str], "bool | None"]


def endpoints_route(
    endpoints: Iterable[str], route_reader: EndpointReader | None = None
) -> bool | None:
    """One answer for a peer's endpoints: ``True`` any routes, ``False`` NONE do.

    ``False`` is the ``local_offline`` predicate and needs EVERY endpoint proven
    route-less: one hostname, one unreadable lookup or no endpoints at all make
    the answer ``None`` (unknown), which every caller reads as "make no claim".
    ``route_reader`` defaults to :func:`endpoint_has_route`, looked up at call
    time so a test's ``monkeypatch`` of the module attribute takes effect.
    """
    reader = route_reader if route_reader is not None else endpoint_has_route
    answers: list[bool | None] = []
    for endpoint in endpoints:
        if not str(endpoint or "").strip():
            continue
        try:
            answers.append(reader(endpoint))
        except Exception:  # noqa: BLE001 — a lookup that raises is not evidence
            answers.append(None)
    if any(answer is True for answer in answers):
        return True
    if answers and all(answer is False for answer in answers):
        return False
    return None


def no_route_for(endpoints: Iterable[str], route_reader: EndpointReader | None = None) -> bool:
    """True only when this device has NO route to ANY of the peer's own endpoints."""
    return endpoints_route(endpoints, route_reader) is False


def peer_only_family(endpoints: Iterable[str]) -> str | None:
    """The one family every literal endpoint is in, or ``None`` (mixed or a hostname)."""
    families = {endpoint_family(endpoint) for endpoint in endpoints}
    if len(families) == 1:
        (family,) = families
        return family
    return None


# -- the ``uplink`` rows ``ready`` and ``doctor`` print FIRST ----------------------
#
# WHY FIRST, AND WHY HERE: in the 2026-11-18 incident every row ``doctor``
# printed was about node2, so every remedy sent the operator to node2. A row
# about THIS device's network placed ahead of the peer rows makes the first
# remedy "check this device's network" when that is where the fault is. Built
# from the same reading the relay uses, in the CLI's own process (the same host
# as the relay), so it also covers the no-relay local fallback. The rows only
# ADD facts: every peer row the payload carried is kept byte for byte.


def _copy(message: Any) -> str:
    from local_operator.i18n.text import tr

    return tr("cli.network_uplink")(message)


def _device_words(state: str) -> str:
    from local_operator.i18n.keys import cli_network_uplink as keys

    # Only called for a whole reading: an ``unknown`` one has no device row.
    return _copy(
        {
            OFFLINE: keys.no_connection(),
            DUAL: keys.dual(),
            IPV4_ONLY: keys.ipv4_only(),
        }.get(state, keys.ipv6_only())
    )


def _answered_by_silence(row: Mapping[str, Any], route_reader: EndpointReader | None) -> bool:
    """A reachability row that IS attempt evidence of silence from a routable address.

    "<peer> isn't answering from this network" is a claim about a real dial
    (security U3): a dial to an endpoint this device HAS a route to, which then
    timed out or drew no answer. ``readiness._silent_row`` is the one reader of
    "silence" in both report dialects; a resolver error, a local error (no
    route, a bad endpoint) or a row with no attempt is not silence, and a peer
    with any such row keeps exactly the diagnostic it had before this module.
    """
    from local_operator.network import readiness

    if not readiness._silent_row(row):  # noqa: SLF001 — the shared silence reader
        return False
    # THE PRODUCER'S ATTEMPT EVIDENCE, NOT THE OUTCOME WORD (security U3, round
    # 2): ``ready`` classifies a local OSError (EADDRNOTAVAIL, EACCES, ENOBUFS)
    # and a worker the report deadline beat to its connect as ``no_answer`` too,
    # so silence needs ``probe_candidates`` to have recorded an issued connect
    # that drew nothing (``dial_silent``). ``_silent_row`` stays as it is: the
    # live-link reading also uses it and is out of this change's scope.
    observed = row.get("observed")
    evidence = row.get("dial_silent") or (
        isinstance(observed, Mapping) and observed.get("dial_silent")
    )
    if evidence is not True:
        return False
    endpoint = str(row.get("endpoint") or "")
    return endpoints_route([endpoint], route_reader) is True


def uplink_checks(
    reading: Mapping[str, Any] | None,
    checks: Iterable[Mapping[str, Any]],
    *,
    names: Mapping[str, str] | None = None,
    route_reader: EndpointReader | None = None,
) -> list[dict[str, Any]]:
    """The ``uplink`` rows: this device's own row, then one per affected peer.

    ROWS ONLY WHEN SOMETHING IS WRONG (QA Q-M1, decided): a healthy report
    reads exactly as it did before this module existed.

    - this device, ``offline`` (no route to the public route keys): ``FAIL``,
      "no route to the internet", remedy "check this device's network" —
      suppressed only when it explains nothing: some peer row is ``ok`` (a
      LAN/loopback peer answering, review R1) AND no failing peer is
      unroutable. "No route to the internet" does not contradict a LAN peer
      answering, and when a public-only peer has no route it is the ONLY row
      that names this device as the cause (UX-U1);
    - this device, one family only: an ``ok`` row naming its families, and
      only ahead of a per-peer family row it explains;
    - ``dual`` or ``unknown``: no device row;
    - a peer none of whose literal endpoints this device has a route to
      (``endpoints_route`` is ``False``, per endpoint — never inferred from
      the public-route reading), all in one family this device has no route
      in: ``FAIL`` "<family> route missing; <peer> has only <family>
      addresses", with the local remedy;
    - a peer every one of whose rows is a dial to a routable endpoint that
      timed out or drew no answer: "<peer> isn't answering from this network"
      and nothing more — no cause is observable from here, so none is named.

    ``names`` maps device ids to member names for rows that carry only an id
    (``doctor``'s); a row's own ``device_name`` (``ready``'s) wins.
    """
    from local_operator.i18n.keys import cli_network_uplink as keys

    reading = dict(reading) if reading else unknown_reading()
    state = str(reading.get("state") or UNKNOWN)
    remedy = _copy(keys.check_network())
    checks = list(checks)
    peer_rows_out: list[dict[str, Any]] = []
    by_peer: dict[str, list[Mapping[str, Any]]] = {}
    any_ok = False
    offline_explains = False
    for check in checks:
        if check.get("check") != "reachability":
            continue
        observed = check.get("observed")
        if isinstance(observed, Mapping) and observed.get("out_of_scope"):
            continue
        any_ok = any_ok or bool(check.get("ok"))
        device_id = str(check.get("device_id") or "")
        if device_id and check.get("endpoint"):
            by_peer.setdefault(device_id, []).append(check)
    for device_id, peer_rows in by_peer.items():
        if any(row.get("ok") for row in peer_rows):
            continue
        endpoints = [str(row.get("endpoint") or "") for row in peer_rows]
        label = str(peer_rows[0].get("device_name") or (names or {}).get(device_id) or device_id)
        family = peer_only_family(endpoints)
        if no_route_for(endpoints, route_reader):
            # The family sentence names a fact the DEVICE reading proves, so it
            # needs one family this device has no route in. Under ``offline``
            # the device row says it for every peer — so record that it must
            # be printed (UX-U1: skipping this peer AND dropping the device row
            # because a LAN peer answered left doctor blaming the peer). A
            # mixed-family or unproven case adds no fact. Never "isn't
            # answering": undialable.
            if state == OFFLINE:
                offline_explains = True
                continue
            if family not in ("ipv4", "ipv6"):
                continue
            if reading.get(family) is not False:
                continue
            peer_rows_out.append(
                {
                    "check": "uplink",
                    "device_id": device_id,
                    "device_name": label,
                    "endpoint": "",
                    "ok": False,
                    "code": "local_offline",
                    "detail": _copy(
                        keys.no_ipv4_for_peer(peer=label)
                        if family == "ipv4"
                        else keys.no_ipv6_for_peer(peer=label)
                    ),
                    "remedies": [remedy],
                }
            )
            continue
        if all(_answered_by_silence(row, route_reader) for row in peer_rows):
            peer_rows_out.append(
                {
                    "check": "uplink",
                    "device_id": device_id,
                    "device_name": label,
                    "endpoint": "",
                    "ok": False,
                    "code": "peer_not_answering",
                    "detail": _copy(keys.not_answering(peer=label)),
                }
            )
    device_row: dict[str, Any] = {
        "check": "uplink",
        "device_id": "",
        "endpoint": "",
        "ok": state != OFFLINE,
        "code": state,
        "detail": "",
        "observed": {"ipv4": reading.get("ipv4"), "ipv6": reading.get("ipv6")},
    }
    # Under ``offline`` the row is dropped only when a peer answered (R1) and
    # no failing peer is left that only this row explains (UX-U1).
    if state == OFFLINE and (not any_ok or offline_explains):
        device_row["detail"] = _device_words(state)
        device_row["remedies"] = [remedy]
        return [device_row, *peer_rows_out]
    explains = any(row.get("code") == "local_offline" for row in peer_rows_out)
    if state in (IPV4_ONLY, IPV6_ONLY) and explains:
        device_row["detail"] = _device_words(state)
        return [device_row, *peer_rows_out]
    return peer_rows_out
