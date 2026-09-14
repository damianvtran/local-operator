"""The session-side client for the desktop app's browser host.

The transport is NOT re-implemented here: :class:`UiHostClient` is
:class:`~local_operator.browser_bridge.backend.HostClient` pointed at this
package's discovery record. That inheritance is the whole design of this module —
the RPC shape, the 401 handling, the error mapping and the client timeout
derivation are already identical for both hosts, and they are the parts most
likely to be fixed once and forgotten in the other copy.

What IS host-specific, and therefore lives here:

* which record to read (this package's ``state``),
* what a healthy ``/health`` says (the UI host's answer carries no
  ``extension_connected`` — it has no third party to be detached from, so the
  probe instead asks "is the pid that answered the same pid the file named?"),
* and the four availability answers every browsable host must supply, because
  ``builtin`` branches on them.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from local_operator.browser_bridge.backend import (
    HEALTH_PROBE_TIMEOUT_S,
    HOST_UI,
    HostClient,
)
from local_operator.ui_browser import state as state_store


class UiHostClient(HostClient):
    """One authenticated call against the app's loopback host, or a typed error."""

    host = HOST_UI

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(state_store, root)


def ui_browser_available(root: Path | None = None) -> bool:
    """File-only "known-good right now" probe: no socket, never raises.

    File-only for the same reason the bridge's is: this runs while constructing
    every session, and a socket round-trip there would tax startup for every
    session on the machine.
    """
    try:
        return state_store.available(root)
    except Exception:  # noqa: BLE001 - session startup must not fail on discovery
        return False


def ui_browser_advertisable(root: Path | None = None) -> bool:
    """File-only gate for whether the `browser` TOOL is offered at all.

    Accepts a STALE-but-alive heartbeat, like the bridge's gate: advertising is a
    weaker commitment than executing, and hiding the tool from a host whose
    heartbeat writer stopped is how the agent ends up with no browser at all and
    no explanation for a host that is running.
    """
    try:
        return state_store.advertisable(root)
    except Exception:  # noqa: BLE001 - session startup must not fail on discovery
        return False


def ui_liveness(
    root: Path | None = None,
) -> tuple[state_store.Liveness, state_store.UiHostState | None]:
    """Classify the host from the file, never raising at a diagnostic site."""
    try:
        return state_store.liveness(root)
    except Exception:  # noqa: BLE001 - a diagnostic may never raise
        return state_store.Liveness.ABSENT, None


async def ui_browser_reachable(
    root: Path | None = None,
    *,
    classified: tuple[state_store.Liveness, state_store.UiHostState | None] | None = None,
) -> bool:
    """Availability for the BROWSER PATH: file first, socket only to acquit.

    The same contract as the bridge's, and it exists for the same reason: the
    heartbeat is a proxy that lies in both directions, so ``FRESH`` answers yes
    without a probe, ``ABSENT`` answers no without a probe, and only ``STALE``
    buys ONE bounded ``/health`` request before the host is condemned.
    """
    try:
        status, current = classified if classified is not None else state_store.liveness(root)
    except Exception:  # noqa: BLE001 - discovery must never raise at a call site
        return False
    if status is state_store.Liveness.FRESH:
        return True
    if status is not state_store.Liveness.STALE or current is None:
        return False
    return await _health_ok(current.port, current.pid)


async def _health_ok(port: int, pid: int) -> bool:
    """One bounded loopback /health probe; any failure means "not reachable".

    The probe requires the answering process to be the pid the FILE named, not
    merely HTTP 200: a stale record whose port has been recycled by another
    process would otherwise be acquitted as this host. That is the UI-host
    equivalent of the bridge's ``extension_connected`` requirement in
    :func:`~local_operator.browser_bridge.backend._health_ok`.

    ``proto`` is reported but deliberately not required to equal
    ``PROTO_VERSION``: a version-skewed host is a REAL host that can explain
    itself, so it stays reachable and the per-action path returns the typed
    ``proto_mismatch`` whose copy names the remedy.
    """
    try:
        async with httpx.AsyncClient(timeout=HEALTH_PROBE_TIMEOUT_S) as client:
            response = await client.get(f"http://127.0.0.1:{port}/health")
        if response.status_code != 200:
            return False
        body = response.json()
        return (
            isinstance(body, dict)
            and body.get("host") == state_store.HOST
            and int(body.get("pid", -1)) == pid
        )
    except Exception:  # noqa: BLE001 - unreachable, malformed, or timed out
        return False
