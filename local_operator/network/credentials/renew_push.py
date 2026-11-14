"""Renew on reconnect: the owner's return refreshes every holder's borrowed logins.

WHY THIS MODULE EXISTS. The offline hold (PR #3922, operator-approved 2026-11-13:
"as long as I'm logging in each day it should refresh the remote credentials and
allow the remote work to continue") serves a held grant for up to 24 h from its
RECEIPT, and a fresh grant restarts that window. But a borrower only received a fresh
grant when IT asked, i.e. when one of its sessions ran a turn while the owner was up.
A node idle all day therefore kept counting from its last ask, and the operator's
daily login did nothing for it. Here the owner's return is the trigger instead:

* OWNER HALF (:func:`on_link_up`, :func:`wake_step`): when a link to a member comes
  up (owner start, owner wake with dead links, a member's own reconnect), or the
  owner wakes with links that survived the sleep, the owner sends that member ONE
  ``net_broker kind=renew`` nudge naming the keys it lends it.
* HOLDER HALF (:func:`member_renew`): the holder's relay re-asks the owner for each
  named key it can hold, through the ORDINARY grant path
  (``MeshCredentialClient.serve_grant``), and keeps the fresh grant in relay memory.
  A session whose own held grant is older (or which has none) adopts it the next
  time the owner is unreachable (``client._held_or`` with ``renewal=``), so its hold
  window counts from the owner's return, not from that session's last ask.

NOTHING NEW IS LENT. The nudge carries no material and grants nothing: the token
travels in an ordinary grant reply to an ordinary grant request, which the owner's
broker authorises exactly as any other (membership, consent, placement holder,
local-only mark). Same kinds, same token material the holder would get by asking;
the owner's provider refresh runs only where it already would (its 60 s skew rule).

BOUNDS. Once per member per link-up, after a random jitter (so many nodes linking at
one wake do not all ask in the same instant), and never more than once per
:data:`RENEW_MIN_INTERVAL_S` per (member, key) — a flapping link gets one renewal per
window, not one per flap. The holder has its own floor (:data:`HOLDER_MIN_INTERVAL_S`)
so a misbehaving owner cannot make it re-ask in a loop. The owner sends nothing to a
member that is not active, on a network it does not trust (panic), or for a key it no
longer lends that member (a revoked share).

MIXED VERSIONS: only when both ends advertise ``credential-renew-push-v1``
(``wire.CREDENTIAL_RENEW_PUSH_V1``); an older peer simply keeps counting from its own
last ask. In memory only on both ends: a relay restart forgets the floors (the next
link-up renews once) and the held renewals (sessions keep their own holds).
"""

from __future__ import annotations

import logging
import random
import threading
import time
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: Minimum seconds between two renew nudges from this owner to one (member, key).
#: The operator-facing promise is "daily login refreshes the nodes" and the hold is
#: 24 h, so a renewal per five minutes is ~300x finer than needed; the floor exists
#: only so a link flapping every few seconds costs one grant per window.
RENEW_MIN_INTERVAL_S = 300.0
#: Upper bound of the random delay before a link-up nudge. A woken owner re-links
#: every member within a few seconds; spreading the asks keeps them from landing on
#: its broker (and a possible provider refresh) in the same instant.
RENEW_JITTER_S = 5.0
#: The holder's own floor per (owner, key): re-asking is cheap but not free (it may
#: mint), and a buggy or hostile owner's repeated nudges must not drive a loop.
HOLDER_MIN_INTERVAL_S = 60.0
#: A sleep-inclusive clock that ran this much further than the sleep-exclusive one
#: between two ticks means this machine slept: on macOS ``time.monotonic`` stops in
#: sleep and the definitions tick runs every 15 s, so a 60 s gap is unambiguous.
WAKE_GAP_S = 60.0
#: Bounds on the nudge frame (one row per lent key; an owner lends a handful).
MAX_RENEW_KEYS = 32

_FLOORS_ATTR = "_credential_renew_floors"
_WAKE_ATTR = "_credential_renew_wake"
_HELD_ATTR = "_credential_renewals"
_TIMERS_ATTR = "_credential_renew_timers"
_GUARD = threading.Lock()


def _clock() -> float:
    from local_operator.network.credentials.client import _sleep_inclusive_clock

    return _sleep_inclusive_clock()


def negotiated(link: Any) -> bool:
    """Both ends of THIS link advertised the renew nudge (the share-push rule)."""
    from local_operator.network import wire

    return wire.CREDENTIAL_RENEW_PUSH_V1 in getattr(
        link, "local_capabilities", ()
    ) and wire.CREDENTIAL_RENEW_PUSH_V1 in getattr(link, "capabilities", ())


def _self_device(server: Any) -> str:
    return str(getattr(getattr(server, "identity", None), "device_id", "") or "")


def _server_attr(server: Any, name: str) -> dict[Any, Any]:
    with _GUARD:
        value = getattr(server, name, None)
        if not isinstance(value, dict):
            value = {}
            setattr(server, name, value)
        return value


# ---------------------------------------------------------------------------
# Owner side
# ---------------------------------------------------------------------------


def lent_keys(server: Any, network_id: str, device_id: str) -> tuple[list[str], str]:
    """``(keys this owner lends device_id, "")`` or ``([], skip reason)``.

    Every gate here is the owner's OWN record, re-read each time: a share revoked,
    a member removed or a network panicked since the link came up gets nothing.
    """
    from local_operator.network import store
    from local_operator.network.credentials import placement as placement_mod

    root = Path(getattr(server, "root"))
    self_device = _self_device(server)
    if not (network_id and device_id and self_device) or device_id == self_device:
        return [], "bad_request"
    record = next((r for r in store.list_networks(root) if r.network_id == network_id), None)
    if record is None:
        return [], "no_network"
    if record.trust != "active":
        # A panic latches ``trust``; its promise is that this network's credentials
        # stop now, so the owner must not invite a fresh borrow either.
        return [], "untrusted"
    active = {m.device_id for m in record.active_members()}
    if self_device not in active or device_id not in active:
        return [], "inactive_member"
    document = placement_mod.PlacementDocument.load(network_id, root, self_device=self_device)
    keys = sorted(
        key
        for key in document.keys_owned_by(self_device)
        if (entry := document.entry(key)) is not None and entry.is_holder(device_id)
    )
    if not keys:
        return [], "nothing_lent"
    return keys[:MAX_RENEW_KEYS], ""


def renew_peer(
    server: Any, network_id: str, device_id: str, *, reason: str = "link_up"
) -> dict[str, Any]:
    """Nudge ``device_id`` to re-ask for the keys this device lends it. Never dials.

    Outcomes: ``nudged`` | ``skipped:<why>`` | ``rate_limited`` | ``offline`` (no live
    link: the next link-up renews) | ``failed`` (no answer) | ``refused:<code>``.
    """
    if getattr(server, "_stop", None) is not None and getattr(server._stop, "is_set", None):
        if server._stop.is_set():  # noqa: SLF001 — a stopping relay starts nothing
            return {"outcome": "skipped:stopping", "keys": []}
    keys, why = lent_keys(server, network_id, device_id)
    if why:
        return {"outcome": f"skipped:{why}", "keys": []}
    link_for = getattr(server, "_link_for", None)
    link = link_for(device_id, network_id=network_id) if link_for is not None else None
    if link is None:
        return {"outcome": "offline", "keys": []}
    if not negotiated(link):
        return {"outcome": "skipped:peer_lacks_renew", "keys": []}
    floors = _server_attr(server, _FLOORS_ATTR)
    now = _clock()
    with _GUARD:
        due = [
            key
            for key in keys
            if now - floors.get((network_id, device_id, key), -float("inf")) >= RENEW_MIN_INTERVAL_S
        ]
        # Stamped BEFORE the send: a flapping link whose nudge never lands must
        # still count against the window, or each flap would retry immediately.
        for key in due:
            floors[(network_id, device_id, key)] = now
    if not due:
        return {"outcome": "rate_limited", "keys": []}
    next_req = getattr(server, "_next_relay_req", None)
    frame = {
        "op": "net_broker",
        "kind": "renew",
        "network_id": network_id,
        "from_device": _self_device(server),
        "req": next_req() if next_req is not None else 0,
        "keys": due,
        "reason": reason,
    }
    from local_operator.network.credentials.sync import ANNOUNCE_TIMEOUT_S

    try:
        reply = link.request(frame, timeout=ANNOUNCE_TIMEOUT_S)
    except Exception:  # noqa: BLE001 — one holder's link is that holder's problem
        logger.debug("renew push: request to %s failed", device_id, exc_info=True)
        reply = None
    detail = reply.get("detail") if isinstance(reply, dict) else None
    if not isinstance(detail, dict):
        return {"outcome": "failed", "keys": due}
    if detail.get("kind") == "error":
        return {"outcome": f"refused:{detail.get('code') or 'unknown'}", "keys": due}
    return {"outcome": "nudged", "keys": due}


def _renew_later(server: Any, network_id: str, device_id: str, reason: str) -> None:
    """Run :func:`renew_peer` off the calling thread after the jitter."""

    def _run() -> None:
        try:
            outcome = renew_peer(server, network_id, device_id, reason=reason)
            logger.debug("renew push to %s (%s): %s", device_id, reason, outcome)
        except Exception:  # noqa: BLE001 — a renewal never takes the relay down
            logger.debug("renew push to %s failed", device_id, exc_info=True)

    timer = threading.Timer(random.uniform(0.0, RENEW_JITTER_S), _run)
    timer.daemon = True
    timer.name = "mesh-credential-renew"
    # Tracked so the relay's stop cancels a pending nudge (:func:`stop`) instead of
    # leaving a timer to fire into a relay that is shutting down.
    timers = _server_attr(server, _TIMERS_ATTR)
    with _GUARD:
        for done in [t for t in timers if not t.is_alive()]:
            timers.pop(done, None)
        timers[timer] = None
    timer.start()


def stop(server: Any) -> None:
    """Cancel this relay's pending link-up nudges (the slice's stop hook)."""
    timers = getattr(server, _TIMERS_ATTR, None)
    if not isinstance(timers, dict):
        return
    with _GUARD:
        pending = list(timers)
        timers.clear()
    for timer in pending:
        timer.cancel()
    for timer in pending:
        if timer is not threading.current_thread():
            timer.join(timeout=1.0)


def on_link_up(server: Any, link: Any) -> None:
    """Relay link-up hook. Runs on the admitting thread, so it only schedules.

    Covers owner START (every link is new) and an owner WAKE whose links died in
    the sleep. Fires on both ends of every link; on a device that lends nothing
    (``lent_keys`` finds nothing) the scheduled run is a cheap no-op.
    """
    device_id = str(getattr(link, "device_id", "") or "")
    network_id = str(getattr(link, "network_id", "") or "")
    if device_id and network_id and negotiated(link):
        _renew_later(server, network_id, device_id, "link_up")


def wake_step(server: Any, device_id: str) -> str | None:
    """Definitions tick step: renew members whose links SURVIVED an owner sleep.

    A short lid-close can keep TCP links alive, so no link-up fires on wake. The
    tick (every 15 s per member) compares the sleep-inclusive clock with
    ``time.monotonic`` (which stops in sleep on macOS); a gap of :data:`WAKE_GAP_S`
    or more is a wake, and every member is then renewed once, on its next visit.
    """
    from local_operator.network.credentials.client import _sleep_inclusive_clock

    state = _server_attr(server, _WAKE_ATTR)
    mono, boot = time.monotonic(), _sleep_inclusive_clock()
    with _GUARD:
        ref = state.get("ref")
        epoch = int(state.get("epoch", 0))
        if ref is not None and (boot - ref[1]) - (mono - ref[0]) >= WAKE_GAP_S:
            epoch += 1
        state["ref"] = (mono, boot)
        state["epoch"] = epoch
        seen = state.setdefault("seen", {})
        if device_id not in seen:
            # First visit: start/link-up already covered this member.
            seen[device_id] = epoch
            return None
        if seen[device_id] >= epoch:
            return None
        seen[device_id] = epoch
    link_for = getattr(server, "_link_for", None)
    link = link_for(device_id) if link_for is not None else None
    if link is None:
        # No surviving link: the link-up that brings it back renews.
        return None
    outcome = renew_peer(
        server, str(getattr(link, "network_id", "") or ""), device_id, reason="wake"
    )
    return str(outcome.get("outcome") or "") or None


# ---------------------------------------------------------------------------
# Holder side
# ---------------------------------------------------------------------------


def _holdable(key: str) -> bool:
    """The kinds the offline hold serves (``client._held_or``'s own exclusions):
    renewing a GitHub, MCP or device-bound grant would only mint for nothing."""
    from local_operator.network.credentials.github import is_github_key
    from local_operator.network.credentials.types import (
        is_device_bound_provider,
        is_mcp_key,
    )

    return not (is_mcp_key(key) or is_device_bound_provider(key) or is_github_key(key))


def member_renew(server: Any, link: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a renew nudge: re-ask the authenticated sender for the keys it lends us.

    Answers at once (the asks run on a worker thread, so the owner's request is not
    held through a provider refresh). Only keys whose row on THIS device names the
    TRANSPORT-AUTHENTICATED sender as owner and this device as a holder are asked;
    the frame's own claims are never trusted.
    """
    from local_operator.network.credentials import placement as placement_mod

    sender = str(getattr(link, "device_id", "") or "")
    network_id = str(getattr(link, "network_id", "") or "")
    self_device = _self_device(server)
    raw_keys = frame.get("keys")
    if not sender or frame.get("from_device") != sender or frame.get("network_id") != network_id:
        return {"kind": "error", "code": "identity_mismatch"}
    if not isinstance(raw_keys, list) or len(raw_keys) > MAX_RENEW_KEYS:
        return {"kind": "error", "code": "renew_invalid"}
    keys = [k for k in raw_keys if isinstance(k, str) and k]
    root = Path(getattr(server, "root"))
    document = placement_mod.PlacementDocument.load(network_id, root, self_device=self_device)
    floors = _server_attr(server, _FLOORS_ATTR)
    now = _clock()
    accepted: list[str] = []
    with _GUARD:
        for key in keys:
            entry = document.entry(key)
            if entry is None or entry.owner_device != sender or not entry.is_holder(self_device):
                continue
            if not _holdable(key):
                continue
            floor = ("held", sender, key)
            if now - floors.get(floor, -float("inf")) < HOLDER_MIN_INTERVAL_S:
                continue
            floors[floor] = now
            accepted.append(key)
    if accepted:
        worker = threading.Thread(
            target=_renew_keys,
            args=(server, sender, accepted),
            name="mesh-credential-renew-ask",
            daemon=True,
        )
        worker.start()
    return {"kind": "renew", "accepted": accepted}


def _renew_keys(server: Any, owner: str, keys: list[str]) -> None:
    from local_operator.network.credentials.client import MeshCredentialClient

    try:
        client = MeshCredentialClient.for_relay(server)
    except Exception:  # noqa: BLE001 — no client, nothing to renew
        logger.debug("renew: no relay client", exc_info=True)
        return
    if client is None:
        return
    for key in keys:
        try:
            renew_key(server, client, owner, key)
        except Exception:  # noqa: BLE001 — one key's failure is that key's
            logger.debug("renew: %s from %s failed", key, owner, exc_info=True)


def renew_key(server: Any, client: Any, owner: str, key: str) -> str:
    """Ask ``owner`` for ``key`` through the ordinary grant path; keep the answer.

    ``serve_grant`` is the exact leg-2 path a session's ask takes through this
    relay: the owner authorises it as any borrow, and ``_adopt_online`` re-checks
    this device's own policy before the grant is accepted.
    """
    from local_operator.network.credentials.client import _sleep_inclusive_clock
    from local_operator.network.credentials.types import Grant

    if client.owner_of(key) != owner:
        return "skipped:owner_changed"
    detail = client.serve_grant(key, provider=key, session_id="")
    if not isinstance(detail, dict) or detail.get("kind") != "grant":
        return f"refused:{(detail or {}).get('code') or 'unknown'}"
    grant = Grant.from_detail(detail)
    if grant.credential_ref.owner_device != owner:
        return "skipped:owner_changed"
    # The receipt stamps the hold is capped from: THIS relay's clocks, now.
    grant.received_wall = time.time()
    grant.received_boot = _sleep_inclusive_clock()
    held = _server_attr(server, _HELD_ATTR)
    with _GUARD:
        held[key] = grant
    return "renewed"


def held_renewal(server: Any, key: str, owner: str) -> dict[str, Any] | None:
    """The renewal this relay holds for ``key`` from ``owner``, as a grant detail.

    Attached to an owner-unreachable answer (``serve_grant``) so the asking session
    can adopt it (``client._held_or``). ``held_age_s`` carries the receipt age on the
    sleep-inclusive clock, so the session caps the hold from THIS relay's receipt,
    never from its own (later) one. ``None`` once the renewal's own hold has ended.
    """
    from local_operator.network.credentials import offline_hold_s
    from local_operator.network.credentials.client import _sleep_inclusive_clock

    held = getattr(server, _HELD_ATTR, None)
    if not isinstance(held, dict):
        return None
    with _GUARD:
        grant = held.get(key)
    if grant is None or grant.credential_ref.owner_device != owner:
        return None
    now_boot = _sleep_inclusive_clock()
    end = grant.hold_end_ms(offline_hold_s(Path(getattr(server, "root"))), now_boot)
    if time.time() * 1000.0 >= end:
        with _GUARD:
            if held.get(key) is grant:
                held.pop(key, None)
        return None
    detail = grant.to_detail()
    detail["held_age_s"] = max(0.0, now_boot - grant.received_boot)
    return detail


def drop_held(server: Any, key: str | None = None) -> None:
    """Forget relay-held renewals (one key, or all): a revoke or panic seen here."""
    held = getattr(server, _HELD_ATTR, None)
    if not isinstance(held, dict):
        return
    with _GUARD:
        if key is None:
            held.clear()
        else:
            held.pop(key, None)


def install() -> None:
    """Register :func:`wake_step` on the definitions syncer's tick (idempotent)."""
    from local_operator.network import definitions

    setattr(wake_step, "independent_cadence", True)
    definitions.add_tick_step(wake_step)
