"""Which API host remote compute addresses, and the one override that moves it.

REMOTE COMPUTE ONLY, ON PURPOSE. ``tunnels.config.DEFAULT_API_URL`` stays the
host the TUNNEL client is pinned to: a tunnel's origin proof is issued for the
Radient tunnel origin and its bearer is expected by production, so a variable
that moved both products would turn a debugging switch into a way to point a
live tunnel's credential somewhere else. Nothing here is imported by the
tunnels client; the default is read from that module so the two cannot drift.

WHY THE OVERRIDE EXISTS AT ALL. A staging server can only be exercised by
sending real requests to it, and the bearer on those requests is the operator's
own Radient login. So the variable is accepted only when the destination is one
a bearer may safely travel to (``https://``), with the single exception of a
loopback host, where the connection never leaves the machine and a plain
``http://127.0.0.1:PORT`` test server is the normal thing to run.

WHY IT WARNS. The host is not visible in any output of ``lop exec --remote``:
a login that is quietly going somewhere other than production looks exactly
like one going to production, which is the one thing a user must never have to
guess about. The warning names the host and never the token — it is printed
once per invocation, to stderr, so stdout stays the run id.

WHY AN UNUSABLE VALUE IS A REFUSAL, NOT A FALLBACK. Falling back silently would
send the request to production while the operator believes they are testing
staging: the worst of both outcomes, and a failure nobody looking at the output
could explain. A refusal costs one sentence and one non-zero exit.

WHY THE RAW STRING IS CHECKED BEFORE IT IS SPLIT. ``urlsplit`` is lenient in
exactly the places where leniency moves a bearer somewhere else: it STRIPS
``\t``, ``\r`` and ``\n`` before parsing, it accepts a ``%``-escape in the host,
and it reports a Cyrillic host as an ordinary hostname (httpx then quietly
punycodes it). So the value is refused for a character it may not contain at
all, and only then parsed. The authority is then matched against an explicit
pattern rather than trusted, and the parsed URL is finally handed to
``httpx.URL`` as a backstop: ``httpx`` raising on a value this module accepted
would otherwise escape as a generic CLI failure instead of the typed refusal.
"""

from __future__ import annotations

import ipaddress
import os
import re
import sys
from urllib.parse import urlsplit

import httpx

from local_operator.i18n.keys import wire_remote_compute as _wr
from local_operator.i18n.text import tr
from local_operator.remote_compute.errors import InvalidApiOverride
from local_operator.tunnels.config import DEFAULT_API_URL

#: The variable that moves the remote-compute API base off production.
ENV_VAR = "LOP_REMOTE_COMPUTE_API_URL"

#: Characters this value may never contain, anywhere, before it is parsed.
#:
#: Whitespace is here because ``urlsplit`` DROPS it (a tab or a newline inside a
#: host is silently deleted, so ``https://sta\tging.example`` parses as a host
#: the operator never wrote), and ``%`` because a percent-escape rewrites the
#: bytes of the authority. ``?`` and ``#`` are refused even with nothing after
#: them: a base with an empty query or fragment is still a base whose tail the
#: operator cannot see in the warning line.
_FORBIDDEN_RAW = re.compile(r"[\s\x00-\x1f\x7f%?#]")

#: One DNS label: no leading or trailing '-', because such a label is not a
#: hostname and the characters alone would let it through.
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"

#: ``hostname[:port]`` with the port left to the numeric range check.
_HOSTNAME_AUTHORITY = re.compile(rf"(?P<host>{_LABEL}(?:\.{_LABEL})*)(?::(?P<port>[0-9]+))?")

#: ``[ipv6][:port]``; the literal itself is validated by ``ipaddress``.
_IPV6_AUTHORITY = re.compile(r"\[(?P<host>[0-9A-Fa-f:.]+)\](?::(?P<port>[0-9]+))?")

#: The longest a hostname can be written: RFC 1035 allows 255 octets on the
#: wire, which includes a length byte per label and the root the text does not
#: carry.
_MAX_HOSTNAME = 253

#: The hosts a plain ``http://`` override may name.
#:
#: EXACT NAMES, NOT A CIDR OR A SUFFIX TEST. ``127.0.0.0/8`` and every name
#: ending in ``.localhost`` are loopback too, but a rule that admits a range is
#: a rule nobody can verify at a glance, and this one is the security boundary
#: for a bearer. Three literal hosts is the whole list.
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})

_t = tr("wire.remote_compute")

#: Hosts whose override has already been announced in this process.
#:
#: A command may build more than one client (a submit and then a watch), and
#: "once per invocation" is about the user's terminal, not about the number of
#: clients: the second warning would be noise that makes the first easier to
#: scroll past.
_announced: set[str] = set()


def _invalid() -> InvalidApiOverride:
    """The refusal, as one sentence naming the variable rather than the value.

    The value is not echoed: it can carry anything, and the one thing a reader
    needs is which variable to fix.
    """
    return InvalidApiOverride(
        f"{ENV_VAR} must be an https:// URL, or http:// on 127.0.0.1, ::1 or localhost, "
        "with no userinfo, query or fragment, no whitespace, control or non-ASCII "
        "characters and no percent-encoding."
    )


def _authority(netloc: str) -> tuple[str, int | None] | None:
    """The ``(host, port)`` an authority names EXACTLY, or ``None``.

    A full match, not a search: ``host.example:80:90``, ``host:`` and
    ``host.example.`` are all authorities ``urlsplit`` accepts and none of them
    is a base this module will use. The port range is checked here rather than
    left to ``parts.port``, whose ``ValueError`` is the generic failure the
    typed refusal exists to replace.
    """
    literal = _IPV6_AUTHORITY.fullmatch(netloc)
    if literal is not None:
        try:
            ipaddress.IPv6Address(literal["host"])
        except ValueError:
            return None
        port = _port(literal["port"])
        return None if port is None and literal["port"] is not None else (literal["host"], port)
    named = _HOSTNAME_AUTHORITY.fullmatch(netloc)
    if named is None or len(named["host"]) > _MAX_HOSTNAME:
        return None
    port = _port(named["port"])
    return None if port is None and named["port"] is not None else (named["host"], port)


def _port(text: str | None) -> int | None:
    """The port as a number, or ``None`` when it is absent or out of range.

    ``0`` is out of range on purpose: it asks the kernel for an ephemeral port,
    which is never where a staging server is listening, and a value above 65535
    cannot be a port at all. Both used to reach ``parts.port`` and raise, which
    is the generic failure the typed refusal replaces.
    """
    if text is None:
        return None
    port = int(text)
    return port if 1 <= port <= 65535 else None


def resolve_api_url() -> str:
    """The remote-compute API base, from the environment or production.

    Unset means production. A SET value is used verbatim apart from a trailing
    slash, and it must be:

    * ``https://``, or ``http://`` naming one of :data:`LOOPBACK_HOSTS` — the
      bearer authenticates the request, so a cleartext hop to anywhere but this
      machine would hand it to the network path;
    * free of userinfo, a query and a fragment — none of them belongs in a base,
      and a ``user:pass@`` authority is a credential this code would otherwise
      forward without knowing it;
    * free of whitespace, control characters, ``%`` and non-ASCII characters
      anywhere, since each of them can change how the authority is read (see
      the module docstring);
    * an authority of exactly one hostname — ASCII letters, digits, ``-`` and
      ``.``, no leading or trailing dot, no empty label — or one bracketed IPv6
      literal, with an optional port in 1-65535.

    An empty value is refused rather than read as "unset": the operator
    exported the variable, and a blank one is a mistake, not a decision.

    Returns:
        str: The base URL, without a trailing slash.

    Raises:
        InvalidApiOverride: The value is not a base this client will use. Never
            a bare ``ValueError`` from ``urlsplit``/``parts.port`` and never an
            ``httpx`` exception: the CLI maps this one type to its sentence.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return DEFAULT_API_URL
    if not raw or _FORBIDDEN_RAW.search(raw) is not None or not raw.isascii():
        raise _invalid()
    try:
        parts = urlsplit(raw)
    except ValueError:
        # ``urlsplit`` raises on an unbalanced IPv6 bracket, before any of the
        # checks below can see the value.
        raise _invalid() from None
    if (
        # ``hostname`` DROPS the userinfo, so it cannot be the test for it.
        "@" in parts.netloc
        or parts.query
        or parts.fragment
        or parts.scheme not in {"https", "http"}
    ):
        raise _invalid()
    authority = _authority(parts.netloc)
    if authority is None:
        raise _invalid()
    host, port = authority
    if parts.scheme == "http" and host.lower() not in LOOPBACK_HOSTS:
        raise _invalid()
    base = raw.rstrip("/")
    try:
        httpx.URL(base)
    except Exception:
        # ANY httpx refusal becomes this package's typed failure: a ValueError
        # escaping here reaches the CLI's generic handler, which is exactly the
        # unactionable failure the override rules exist to prevent.
        raise _invalid() from None
    # ``hostname`` drops an IPv6 literal's brackets and lowercases the name;
    # ``_authority`` matched this same netloc, so it cannot be None here.
    announced = parts.hostname if parts.hostname is not None else host
    _announce(announced.lower(), port)
    return base


def _address(host: str, port: int | None) -> str:
    """The ``host[:port]`` a warning can be checked against, bracketing IPv6.

    ``host`` arrives lowercased from the caller: the authority may have been
    written in any case, and the line has to name the host that will actually
    be contacted. The port is printed only when the value carried one, and an
    IPv6 literal is bracketed so ``::1:8123`` cannot be misread as one address.
    """
    literal = f"[{host}]" if ":" in host else host
    return literal if port is None else f"{literal}:{port}"


def _announce(host: str, port: int | None) -> None:
    """Say once, on stderr, that this login is not going to production.

    The ADDRESS (host and port) rather than the hostname alone: two local test
    servers on different ports are otherwise indistinguishable in the line, and
    the whole point is that the reader can see where the bearer went. Userinfo
    is refused above, so the address is the host and, at most, a port.

    CALLED BEFORE THE FIRST REQUEST, and for ``--remote-status`` and
    ``--remote-cancel`` as well, because those send the same bearer: the client
    resolves the base in its constructor, so the line precedes the transport on
    every command that talks to Radient.
    """
    address = _address(host, port)
    if address in _announced:
        return
    _announced.add(address)
    print(_t(_wr.api_override(host=address)), file=sys.stderr)


__all__ = ["ENV_VAR", "LOOPBACK_HOSTS", "resolve_api_url"]
