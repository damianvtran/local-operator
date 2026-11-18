"""This device's own uplink: which address families it has a route in, right now.

WHY THIS EXISTS (incident 2026-11-18): the operator's Mac joined an IPv6-only
network while node2 advertised IPv4 endpoints only. Every dial to node2 failed
with "no route", and every surface — the peer rows' ``address_unreachable``,
the redial loop's growing backoff, ``doctor``'s remedies — pointed at node2,
the one machine that had nothing wrong with it. The missing fact was about
THIS device: it had no IPv4 route at all. This module reads that fact so the
link state (``local_offline``), the redial loop (pause while offline, kick on
recovery) and ``ready``/``doctor`` (an ``uplink`` row first) can name it.

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

FAIL-SAFE. Every caller treats a reading that raises, or one whose state is
``unknown``, exactly as the code behaved before this module existed: no pause,
no ``local_offline``, no uplink claim.
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


def endpoint_family(endpoint: str) -> str | None:
    """``ipv4``/``ipv6`` for a literal ``host:port`` (or ``[v6]:port``); else ``None``.

    A hostname has no family until it resolves, and resolving would be a probe
    this module does not make, so it reads as ``None`` (could be either).
    """
    text = str(endpoint or "").strip()
    if text.startswith("["):
        host = text[1:].split("]", 1)[0]
    elif text.count(":") == 1:
        host = text.rsplit(":", 1)[0]
    else:
        host = text
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return None
    return "ipv4" if parsed.version == 4 else "ipv6"


def no_route_for(reading: Mapping[str, Any] | None, endpoints: Iterable[str]) -> bool:
    """True only when this device has no route in ANY family ``endpoints`` use.

    The ``local_offline`` predicate. Fail-safe in every direction: no reading,
    no endpoints, or a family whose answer is unknown all return ``False``, so
    the caller keeps today's ``address_unreachable``. A hostname endpoint could
    be either family, so it counts only when BOTH families are proven absent.
    """
    if not reading:
        return False
    families: set[str] = set()
    for endpoint in endpoints:
        family = endpoint_family(endpoint)
        families.update((family,) if family else ("ipv4", "ipv6"))
    if not families:
        return False
    return all(reading.get(family) is False for family in families)


def has_route_for(reading: Mapping[str, Any] | None, endpoints: Iterable[str]) -> bool:
    """True when this device is PROVEN to have a route in some family ``endpoints`` use."""
    if not reading:
        return False
    for endpoint in endpoints:
        family = endpoint_family(endpoint)
        for candidate in (family,) if family else ("ipv4", "ipv6"):
            if reading.get(candidate) is True:
                return True
    return False


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


def _silent(detail: str) -> bool:
    """A peer-row detail that is pure silence: nothing at that address answered.

    Only these qualify for "isn't answering". A refused connection, a handshake
    refusal or a budget that ran out after an answer all prove the peer's host
    DID answer from this network, and their own rows already explain them.
    """
    if detail == "no_answer":
        return True
    return detail.startswith("connect_failed:") and "ConnectionRefusedError" not in detail


def uplink_checks(
    reading: Mapping[str, Any] | None,
    checks: Iterable[Mapping[str, Any]],
    *,
    names: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """The ``uplink`` rows: this device's own row, then one per affected peer.

    - this device: ``FAIL`` only on a proven ``offline`` (remedy: check this
      device's network), ``ok`` naming its families otherwise; an ``unknown``
      reading has NO device row at all — it would claim nothing, and a report
      whose reading failed must read exactly as it did before this existed;
    - a peer whose every endpoint is in a family this device has no route in:
      ``FAIL`` with the family sentence and the same local remedy;
    - a peer that never answered although this device HAS a route in its
      family: "<peer> isn't answering from this network" and nothing more —
      the cause is not observable from here, so none is named.

    ``names`` maps device ids to member names for rows that carry only an id
    (``doctor``'s); a row's own ``device_name`` (``ready``'s) wins.
    """
    from local_operator.i18n.keys import cli_network_uplink as keys

    reading = dict(reading) if reading else unknown_reading()
    state = str(reading.get("state") or UNKNOWN)
    remedy = _copy(keys.check_network())
    device_row: dict[str, Any] = {
        "check": "uplink",
        "device_id": "",
        "endpoint": "",
        "ok": state != OFFLINE,
        "code": state,
        "detail": "" if state == UNKNOWN else _device_words(state),
        "observed": {"ipv4": reading.get("ipv4"), "ipv6": reading.get("ipv6")},
    }
    if state == OFFLINE:
        device_row["remedies"] = [remedy]
        # The device row already says everything: no route in any family means
        # no peer row can add a fact about this device's network.
        return [device_row]
    # A partly-read uplink (one family unknown) still proves the other family's
    # answer, so the per-peer rows below may use it; only the device row waits
    # for a whole reading.
    rows = [] if state == UNKNOWN else [device_row]
    by_peer: dict[str, list[Mapping[str, Any]]] = {}
    for check in checks:
        if check.get("check") != "reachability":
            continue
        observed = check.get("observed")
        if isinstance(observed, Mapping) and observed.get("out_of_scope"):
            continue
        device_id = str(check.get("device_id") or "")
        if device_id and check.get("endpoint"):
            by_peer.setdefault(device_id, []).append(check)
    for device_id, peer_rows in by_peer.items():
        if any(row.get("ok") for row in peer_rows):
            continue
        endpoints = [str(row.get("endpoint") or "") for row in peer_rows]
        label = str(peer_rows[0].get("device_name") or (names or {}).get(device_id) or device_id)
        family = peer_only_family(endpoints)
        if no_route_for(reading, endpoints) and family in ("ipv4", "ipv6"):
            detail = _copy(
                keys.no_ipv4_for_peer(peer=label)
                if family == "ipv4"
                else keys.no_ipv6_for_peer(peer=label)
            )
            rows.append(
                {
                    "check": "uplink",
                    "device_id": device_id,
                    "device_name": label,
                    "endpoint": "",
                    "ok": False,
                    "code": "local_offline",
                    "detail": detail,
                    "remedies": [remedy],
                }
            )
            continue
        silent = all(_silent(str(row.get("detail") or "")) for row in peer_rows)
        if silent and has_route_for(reading, endpoints):
            rows.append(
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
    return rows
