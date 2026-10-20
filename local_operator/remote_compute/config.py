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
pattern rather than trusted, every ``xn--`` label has to DECODE as punycode (an
A-label that does not is a hostname no resolver would answer for), and the
parsed URL is finally handed to ``httpx.URL`` as a backstop for the exceptions
a URL build raises: ``InvalidURL``, ``ValueError`` and ``UnicodeError`` become
the typed refusal. WHAT THE BACKSTOP DOES NOT PROMISE: any other exception from
``httpx.URL`` propagates, because that is a bug in this module rather than a
value the operator typed, and swallowing it as the override refusal would hide
it.
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
#:
#: AT MOST FIVE DIGITS, and the bound is structural rather than a preference:
#: ``int()`` refuses a decimal string above 4300 digits outright, so a longer
#: port would raise from inside ``_port`` — the generic CLI failure the typed
#: refusal exists to replace. Five digits is also the widest a port can be
#: written, so a sixth digit is not a number this code needs to read.
_HOSTNAME_AUTHORITY = re.compile(rf"(?P<host>{_LABEL}(?:\.{_LABEL})*)(?::(?P<port>[0-9]{{1,5}}))?")

#: ``[ipv6][:port]``; the literal itself is validated by ``ipaddress``, and
#: the port carries the same five-digit bound as a named host's.
_IPV6_AUTHORITY = re.compile(r"\[(?P<host>[0-9A-Fa-f:.]+)\](?::(?P<port>[0-9]{1,5}))?")

#: The longest a hostname can be written: RFC 1035 allows 255 octets on the
#: wire, which includes a length byte per label and the root the text does not
#: carry.
_MAX_HOSTNAME = 253

#: The A-label prefix, lowercased before it is compared: an A-label is
#: case-insensitive on the wire, and the decoder only recognises the lowercase
#: spelling, so ``XN--a`` has to be read as ``xn--a`` rather than passed
#: through as an ordinary name.
_ALABEL_PREFIX = "xn--"

#: The one NAME a plain ``http://`` override may carry.
#:
#: ``localhost`` is kept by name because that is how it is written and how the
#: resolver reads it; it is not an address this module can recognise. Every
#: other loopback spelling is an IP literal and is answered by ``ipaddress`` —
#: see :func:`is_loopback_host` for why a text list is the wrong rule here.
_LOOPBACK_NAME = "localhost"

_t = tr("wire.remote_compute")

#: Hosts whose override has already been announced in this process.
#:
#: A command may build more than one client (a submit and then a watch), and
#: "once per invocation" is about the user's terminal, not about the number of
#: clients: the second warning would be noise that makes the first easier to
#: scroll past.
_announced: set[str] = set()


def invalid_override() -> InvalidApiOverride:
    """The refusal, as one sentence naming the variable rather than the value.

    The value is not echoed: it can carry anything, and the one thing a reader
    needs is which variable to fix.

    PUBLIC because the refusal is not only raised where the value is parsed:
    ``client._request`` raises this same type when the URL cannot be BUILT, and
    a second sentence for the same condition would be a second vocabulary.
    """
    return InvalidApiOverride(
        f"{ENV_VAR} must be an https:// URL, or http:// on a loopback address or "
        "localhost, with no userinfo, query or fragment, no whitespace, control or "
        "non-ASCII characters and no percent-encoding."
    )


def is_loopback_host(host: str) -> bool:
    """Whether a plain ``http://`` may name this host.

    AN IP LITERAL IS ASKED OF ``ipaddress``, NOT MATCHED AS TEXT. ``::1`` has
    many spellings (``0:0:0:0:0:0:0:1``, ``::0001``) and ``127.0.0.0/8`` is
    loopback beyond ``127.0.0.1``; a list of three literal strings answered
    neither, and it refused addresses that never leave the machine. The rule
    this module means is "the connection stays local", and ``is_loopback`` is
    that question.

    ``localhost`` STAYS A NAME, because it is one: the resolver decides what it
    means, and it is the normal spelling for a local test server.

    WHAT IS STILL REFUSED, deliberately: ``0.0.0.0`` is every interface rather
    than this machine, and the short and integer forms (``127.1``,
    ``2130706433``) are not IP literals to ``ip_address`` at all — they fall
    through the name test and are refused, because a form the resolver expands
    differently from what is written is exactly what this check exists to stop.
    """
    if host.lower() == _LOOPBACK_NAME:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _decodable_alabels(host: str) -> bool:
    """Whether every ``xn--`` label in ``host`` decodes as punycode.

    AN A-LABEL IS NOT JUST A NAME WITH A PREFIX. ``xn--a`` and ``xn--0`` match
    the character rule and are not encodings of anything, so a resolver refuses
    them and ``httpx`` only finds out at build time — after this module has
    accepted the value and warned about the host. Decoding here keeps the
    refusal where the value was parsed.

    The stdlib codec is used rather than the ``idna`` package: that package is
    a transitive dependency (httpx → idna), not one this project declares, and
    a check that stops working when a transitive dependency is dropped is not a
    check. Labels are lowercased first because the codec only recognises the
    lowercase prefix and would otherwise pass ``XN--a`` through as a name.
    """
    for label in host.split("."):
        if not label.lower().startswith(_ALABEL_PREFIX):
            continue
        try:
            label.lower().encode("ascii").decode("idna")
        except (UnicodeError, ValueError):
            return False
    return True


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
    if not _decodable_alabels(named["host"]):
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
    try:
        port = int(text)
    except ValueError:
        # Unreachable while the authority patterns cap the port at five digits;
        # kept so widening either pattern cannot reintroduce a bare ``int()``
        # failure, which the CLI would report as a generic error.
        return None
    return port if 1 <= port <= 65535 else None


def resolve_api_url() -> str:
    """The remote-compute API base, from the environment or production.

    Unset means production. A SET value is used verbatim apart from a trailing
    slash, and it must be:

    * ``https://``, or ``http://`` naming a loopback address
      (:func:`is_loopback_host`: any ``127.0.0.0/8`` literal, ``::1`` in any
      spelling, or ``localhost``) — the bearer authenticates the request, so a
      cleartext hop to anywhere but this machine would hand it to the network
      path;
    * free of userinfo, a query and a fragment — none of them belongs in a base,
      and a ``user:pass@`` authority is a credential this code would otherwise
      forward without knowing it;
    * free of whitespace, control characters, ``%`` and non-ASCII characters
      anywhere, since each of them can change how the authority is read (see
      the module docstring);
    * an authority of exactly one hostname — ASCII letters, digits, ``-`` and
      ``.``, no leading or trailing dot, no empty label, every ``xn--`` label a
      decodable A-label — or one bracketed IPv6 literal, with an optional port
      in 1-65535.

    An empty value is refused rather than read as "unset": the operator
    exported the variable, and a blank one is a mistake, not a decision.

    Returns:
        str: The base URL, without a trailing slash.

    Raises:
        InvalidApiOverride: The value is not a base this client will use. Never
            a bare ``ValueError`` from ``urlsplit`` and never the ``httpx``
            exception family a URL build raises (``InvalidURL``, ``ValueError``,
            ``UnicodeError``): the CLI maps this one type to its sentence. Any
            OTHER exception is a bug in this module and propagates.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return DEFAULT_API_URL
    if not raw or _FORBIDDEN_RAW.search(raw) is not None or not raw.isascii():
        raise invalid_override()
    try:
        parts = urlsplit(raw)
    except ValueError:
        # ``urlsplit`` raises on an unbalanced IPv6 bracket, before any of the
        # checks below can see the value.
        raise invalid_override() from None
    if (
        # ``hostname`` DROPS the userinfo, so it cannot be the test for it.
        "@" in parts.netloc
        or parts.query
        or parts.fragment
        or parts.scheme not in {"https", "http"}
    ):
        raise invalid_override()
    authority = _authority(parts.netloc)
    if authority is None:
        raise invalid_override()
    host, port = authority
    if parts.scheme == "http" and not is_loopback_host(host):
        raise invalid_override()
    base = raw.rstrip("/")
    try:
        httpx.URL(base)
    except (httpx.InvalidURL, ValueError, UnicodeError):
        # The exception family a URL BUILD raises, and nothing wider: a
        # ValueError escaping here reaches the CLI's generic handler, which is
        # exactly the unactionable failure the override rules exist to prevent.
        # Anything outside this family is this module's bug, not the
        # operator's value, and must surface as one.
        raise invalid_override() from None
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


__all__ = ["ENV_VAR", "invalid_override", "is_loopback_host", "resolve_api_url"]
