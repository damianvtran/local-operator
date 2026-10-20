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
"""

from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit

from local_operator.i18n.keys import wire_remote_compute as _wr
from local_operator.i18n.text import tr
from local_operator.remote_compute.errors import InvalidApiOverride
from local_operator.tunnels.config import DEFAULT_API_URL

#: The variable that moves the remote-compute API base off production.
ENV_VAR = "LOP_REMOTE_COMPUTE_API_URL"

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
        "with no userinfo, query or fragment."
    )


def resolve_api_url() -> str:
    """The remote-compute API base, from the environment or production.

    Unset means production. A SET value is used verbatim apart from surrounding
    whitespace and a trailing slash, and it must be:

    * ``https://``, or ``http://`` naming one of :data:`LOOPBACK_HOSTS` — the
      bearer authenticates the request, so a cleartext hop to anywhere but this
      machine would hand it to the network path;
    * free of userinfo, a query and a fragment — none of them belongs in a base,
      and a ``user:pass@`` authority is a credential this code would otherwise
      forward without knowing it;
    * a host with a well-formed authority, since httpx raises on the ones that
      are not and that raise is not one of this package's typed failures.

    An empty value is refused rather than read as "unset": the operator
    exported the variable, and a blank one is a mistake, not a decision.

    Returns:
        str: The base URL, without a trailing slash.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return DEFAULT_API_URL
    value = raw.strip()
    if not value:
        raise _invalid()
    parts = urlsplit(value)
    host = parts.hostname
    if (
        not host
        # ``hostname`` DROPS the userinfo, so it cannot be the test for it.
        or "@" in parts.netloc
        or parts.netloc.endswith(":")
        or parts.query
        or parts.fragment
        or parts.scheme not in {"https", "http"}
        or (parts.scheme == "http" and host not in LOOPBACK_HOSTS)
    ):
        raise _invalid()
    try:
        parts.port
    except ValueError:
        raise _invalid() from None
    _announce(parts.netloc)
    return value.rstrip("/")


def _announce(address: str) -> None:
    """Say once, on stderr, that this login is not going to production.

    The ADDRESS (host and port) rather than the hostname alone: two local test
    servers on different ports are otherwise indistinguishable in the line, and
    the whole point is that the reader can see where the bearer went. Userinfo
    is refused above, so the authority is the host and, at most, a port.
    """
    if address in _announced:
        return
    _announced.add(address)
    print(_t(_wr.api_override(host=address)), file=sys.stderr)


__all__ = ["ENV_VAR", "LOOPBACK_HOSTS", "resolve_api_url"]
