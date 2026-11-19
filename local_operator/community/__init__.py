"""Local Operator Community: the local member client (RFC v2.4 §7.2).

WHAT THIS PACKAGE IS. The one Python client every LOCAL surface uses to reach a
public community on Team Bridge (``/community/v1`` on the agent-server's
``community`` deployment): the desktop renderer through
``server/routes/desktop_community.py``, relay web through the mobile daemon's
``/api/community/*`` allowlist, and later the TUI page and the bridge tool
operations (separate lanes). The mobile APP does not use it — it talks to the
hub directly with its own sign-in.

WHAT IT IS NOT. It never shares state with the private Team Bridge client or
receiver (RFC §7.2: "Neither shares state with the private Bridge client or
receiver"): no pooled hub connection, no status cache, no cursors, no consent
records. The only thing it reads in common is the credential store, because the
community authenticates as the same signed-in Radient account.

DARK BY DEFAULT. Every surface checks :func:`community_enabled` per request and
answers ``community_disabled`` while it is off. The switch is an environment
variable, not a ``config.yml`` key, on purpose: a key must be a ``/settings``
row (AGENTS.md "Adding a configuration key"), and a browsable row would
advertise an unreleased feature to every user before the staff pilot. The pilot
sets ``LOCAL_OPERATOR_COMMUNITY=1`` in the backend's environment; promoting it
to a registry row is the launch lane's change, with its translated copy.
"""

from __future__ import annotations

import os

#: The pilot switch. Read per call (never cached at import) so a test or a
#: respawned backend sees the value it was started with, the precedent of
#: ``references.AT_REFERENCES_ENV``.
COMMUNITY_ENV = "LOCAL_OPERATOR_COMMUNITY"

#: Off unless explicitly on: the RFC ships the community dark until the pilot.
DEFAULT_COMMUNITY_ENABLED = False

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def community_enabled() -> bool:
    """Whether the community member surfaces answer at all, RIGHT NOW.

    Fails closed: unset, empty or any value outside the truthy set is off.
    """

    value = os.environ.get(COMMUNITY_ENV)
    if value is None:
        return DEFAULT_COMMUNITY_ENABLED
    return value.strip().lower() in _TRUTHY


__all__ = ["COMMUNITY_ENV", "DEFAULT_COMMUNITY_ENABLED", "community_enabled"]
