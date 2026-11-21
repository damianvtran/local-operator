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
  This includes a session that never borrowed (started on the node while the
  owner is away): the share is the DEVICE's, every same-uid process could already
  borrow under it while the owner was up, and the renewal is asked under
  ``for_session=relay-renew`` so the owner's audit tells it from a session's ask.

* PRE-EXPIRY (:func:`pre_expiry_step`, G10, 2026-11-21): on a STEADY link no
  link-up fires, so the holder's relay also re-asks on its own definitions tick,
  once per token life just inside the owner's refresh skew, for keys a live local
  session borrowed. Bounds and stop conditions: the "pre-expiry renewal" section.

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
#: After a nudge that got no answer (timeout, dead link, exception), the floor for
#: those keys is moved back so they may be re-sent this many seconds later, and the
#: member's next tick retries (:func:`wake_step`). Without it one lost RPC at wake
#: cost the node its renewal until the NEXT wake (PR #3946 review M1). Two 15 s
#: ticks: a link flapping faster still costs at most one nudge per 30 s.
RENEW_FAILED_RETRY_S = 30.0
#: The ``for_session`` a relay renewal is asked under, so the OWNER's audit tells a
#: node's renewal from a session's own ask (review m2). Informational only: the
#: owner authorises the device's share, never a session id.
RENEW_SESSION = "relay-renew"
#: Bounds on the nudge frame (one row per lent key; an owner lends a handful).
MAX_RENEW_KEYS = 32

# -- the holder's pre-expiry renewal (G10) -----------------------------------
#: How far inside the OWNER's refresh skew the step asks, as a fraction of it. The
#: owner refreshes only within its skew of expiry (providers
#: ``auth_store.OAUTH_REFRESH_SKEW_MS``, MCP ``mcp.auth.REFRESH_SKEW_S``, both 60 s
#: as shipped); an ask outside it gets the SAME token back and renews nothing. So
#: the ask must land with ``0 < remaining <= skew``, and the 15 s definitions tick
#: (``sync.SYNC_TICK_S``) must visit the key inside that window at least once:
#: 0.75 * 60 s = 45 s of window against a 15 s tick gives >= 2 visits. Derived from
#: the constants at call time, never restated, so a changed skew moves it too.
PRE_EXPIRY_LEAD_FRACTION = 0.75
#: HARD holder-side cap per key (security condition Q7): at most this many step
#: asks per rolling hour, whatever the token life says. The shortest life served
#: today is ~55 min (Radient), which needs one ask an hour; four leaves room for a
#: mid-hour re-borrow and a retry without letting a short-lived token (or a
#: misbehaving owner stating one) drive a loop.
PRE_EXPIRY_MAX_ASKS_PER_HOUR = 4
PRE_EXPIRY_CAP_WINDOW_S = 3600.0
#: Borrowing processes remembered per key (a handful of sessions per device).
MAX_ASKERS_PER_KEY = 32
#: Owner answers that mean "not now", never "not yours": the step backs off to the
#: next token life and keeps the key. Anything else the owner names is a refusal
#: the step stops on (``client._HOLD_SERVES_REFUSALS`` plus the transient codes).
_BACK_OFF_CODES = frozenset(
    {
        "owner_offline",
        "owner_unreachable",
        "owner_busy",
        "rate_limited",
        "quota_blocked",
        "internal",
        "unknown",
    }
)

_FLOORS_ATTR = "_credential_renew_floors"
_WAKE_ATTR = "_credential_renew_wake"
_HELD_ATTR = "_credential_renewals"
_TIMERS_ATTR = "_credential_renew_timers"
_FAILED_ATTR = "_credential_renew_failed"
_ACTIVE_ATTR = "_credential_renew_active"
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
    failed = _server_attr(server, _FAILED_ATTR)
    if not isinstance(detail, dict):
        # NO ANSWER: nothing says the holder heard it. Keep the floor only for
        # RENEW_FAILED_RETRY_S (flap protection) and mark the member for the
        # tick's retry; a nudge that did land is harmless to repeat, because the
        # holder's own floor absorbs it.
        retry_floor = now - RENEW_MIN_INTERVAL_S + RENEW_FAILED_RETRY_S
        with _GUARD:
            for key in due:
                if floors.get((network_id, device_id, key)) == now:
                    floors[(network_id, device_id, key)] = retry_floor
            failed[(network_id, device_id)] = now
        return {"outcome": "failed", "keys": due}
    with _GUARD:
        failed.pop((network_id, device_id), None)
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
    # The renewed logins die with the relay that fetched them (review m1).
    drop_held(server)
    forget_active(server)


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
    failed = _server_attr(server, _FAILED_ATTR)
    mono, boot = time.monotonic(), _sleep_inclusive_clock()
    with _GUARD:
        # A nudge to this member that got no answer is retried here, every tick,
        # until one lands (the failure floor spaces the actual sends).
        retry = sorted(network for (network, device) in failed if device == device_id)
        ref = state.get("ref")
        epoch = int(state.get("epoch", 0))
        if ref is not None and (boot - ref[1]) - (mono - ref[0]) >= WAKE_GAP_S:
            epoch += 1
        state["ref"] = (mono, boot)
        state["epoch"] = epoch
        seen = state.setdefault("seen", {})
        woke = False
        if device_id not in seen:
            # First visit: start/link-up already covered this member.
            seen[device_id] = epoch
        elif seen[device_id] < epoch:
            seen[device_id] = epoch
            woke = True
    if woke:
        # EVERY network this member shares with us has its own link and its own
        # placement rows; renewing only the first link's network missed the rest
        # (review m4). No surviving link in a network: its link-up renews it.
        networks = sorted(
            {
                str(getattr(link, "network_id", "") or "")
                for link in list(getattr(server, "links", {}).values())
                if getattr(link, "device_id", "") == device_id
                and getattr(link, "alive", False)
                and not getattr(link, "closing", False)
            }
            - {""}
        )
        reason = "wake"
    else:
        networks, reason = retry, "retry"
    outcomes: list[str] = []
    for network in networks:
        outcome = str(renew_peer(server, network, device_id, reason=reason).get("outcome") or "")
        outcomes.append(outcome)
        if outcome not in ("failed", "rate_limited"):
            # Answered, or a skip a retry cannot change (revoked, panicked,
            # offline: the next link-up renews): stop retrying this member.
            with _GUARD:
                failed.pop((network, device_id), None)
    return ",".join(o for o in outcomes if o) or None


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
            renew_key(server, client, owner, key, trigger="link_up")
        except Exception:  # noqa: BLE001 — one key's failure is that key's
            logger.debug("renew: %s from %s failed", key, owner, exc_info=True)


def renew_key(
    server: Any, client: Any, owner: str, key: str, *, dial: bool = True, trigger: str = ""
) -> str:
    """Ask ``owner`` for ``key`` through the ordinary grant path; keep the answer.

    ``serve_grant`` is the exact leg-2 path a session's ask takes through this
    relay: the owner authorises it as any borrow, and ``_adopt_online`` re-checks
    this device's own policy before the grant is accepted. Never ``force_refresh``:
    a forced refresh is the one ask that can provoke IdP reuse detection, and the
    owner's own skew rule decides whether this ask refreshes at all.
    """
    from local_operator.network.credentials.client import _sleep_inclusive_clock
    from local_operator.network.credentials.types import Grant, is_mcp_key

    # THIS DEVICE'S OWN STANDING FIRST, each refused BY NAME and without a frame
    # (G10): a panicked/untrusted network sends nothing, a removed member and a
    # revoked holder ask nothing, and the owner would refuse them anyway.
    if not client._network_trusted():  # noqa: SLF001 — the client's own trust read
        return "refused:untrusted"
    if not client._self_membership_active():  # noqa: SLF001
        return "refused:member_not_active"
    if not client.should_borrow(key):
        return "refused:not_a_holder"
    if client.owner_of(key) != owner:
        return "skipped:owner_changed"
    # THE POLICY STAMP AT FETCH TIME (review m3), taken before the ask. A session
    # adopts the renewal only while the held policy still reads this stamp, so a
    # deny recorded after this fetch (even one later cleared) keeps it out, exactly
    # as it keeps out a grant that session fetched itself.
    from local_operator.network.credentials.held_policy import PolicyUnknown

    try:
        policy = client._policy()  # noqa: SLF001 — the client's own policy store
        with policy.transaction() as document:
            fetch_stamp = policy.stamp(document, owner, key)
            # A held-policy deny (consent withdrawn) on this device: no ask, so no
            # bearer is fetched into relay memory that no session may adopt (Q6).
            denied = policy.code(document, owner, key)
    except PolicyUnknown:
        return "skipped:policy_unknown"
    if denied:
        return f"refused:{denied}"
    detail = client.serve_grant(
        key,
        # The provider label a session's own MCP ask uses (``BrokeredBearerAuth``).
        provider="mcp-oauth" if is_mcp_key(key) else key,
        session_id=RENEW_SESSION,
        dial=dial,
        renew_trigger=trigger,
    )
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
        held[key] = (grant, fetch_stamp)
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
        entry = held.get(key)
    if entry is None:
        return None
    grant, fetch_stamp = entry
    if grant.credential_ref.owner_device != owner:
        return None
    now_boot = _sleep_inclusive_clock()
    end = grant.hold_end_ms(offline_hold_s(Path(getattr(server, "root"))), now_boot)
    if time.time() * 1000.0 >= end:
        with _GUARD:
            if held.get(key) is entry:
                held.pop(key, None)
        return None
    import dataclasses

    detail = grant.to_detail()
    detail["held_age_s"] = max(0.0, now_boot - grant.received_boot)
    detail["policy_stamp"] = dataclasses.asdict(fetch_stamp)
    return detail


def held_keys(server: Any) -> list[str]:
    """The keys this relay currently holds a renewal for (for a revoke sweep)."""
    held = getattr(server, _HELD_ATTR, None)
    if not isinstance(held, dict):
        return []
    with _GUARD:
        return list(held)


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


# ---------------------------------------------------------------------------
# Holder side: the pre-expiry renewal (G10)
# ---------------------------------------------------------------------------
#
# WHY (operator ask a-4dc5, 2026-11-21: "Yes — background renewal, revocable").
# A session's held grant ends at its TOKEN's expiry. An owner that stayed up past
# one token life while the session made no request (a long tool call, a build)
# refreshed nothing for it, so when the owner then went away the session's hold
# had already lapsed and its turn failed. The renew nudge above fires only on
# link-up/wake, never on a steady link. This step re-asks, once per token life,
# just inside the owner's refresh skew, so the relay always holds the owner's
# NEWEST access token for ``_held_or`` to adopt.
#
# NO NEW AUTHORITY: the ask is the ordinary ``grant`` under ``RENEW_SESSION`` and
# the owner authorises it exactly as any holder's (membership, consent, placement,
# local-only); ``trigger=pre_expiry`` is an audit label. SINGLE REFRESHER: only the
# owner refreshes, under its own lease/lock; no refresh token ever leaves it and
# this step never asks ``force_refresh``. LEASE CAP: the renewal is an owner-minted
# grant whose hold stays capped by its own token expiry and by this relay's receipt
# plus ``offline_hold_s`` (``Grant.hold_end_ms``); nothing here writes a hold.
# STOP CONDITIONS: no live local session that borrowed the key (pid + start time),
# a refusal by name, panic/untrusted, lost membership, not a holder, a held-policy
# deny, no live direct owner link (no dial), the per-key hourly cap. In memory only.


def _now_ms() -> int:
    return int(time.time() * 1000)


def pre_expiry_lead_ms(key: str) -> int:
    """How long before token expiry the step asks for ``key`` (see the fraction)."""
    from local_operator.network.credentials.types import is_mcp_key

    if is_mcp_key(key):
        from local_operator.mcp import auth as mcp_auth

        skew_ms = float(mcp_auth.REFRESH_SKEW_S) * 1000.0
    else:
        from local_operator.providers import auth_store

        skew_ms = float(auth_store.OAUTH_REFRESH_SKEW_MS)
    return int(skew_ms * PRE_EXPIRY_LEAD_FRACTION)


def _renewable(key: str) -> bool:
    """Provider OAuth and MCP access tokens. GitHub (its own mint/belt lifecycle,
    a non-goal) and device-bound providers (never brokered) are never renewed."""
    from local_operator.network.credentials.github import is_github_key
    from local_operator.network.credentials.types import is_device_bound_provider

    return not (is_device_bound_provider(key) or is_github_key(key))


def note_session_borrow(server: Any, key: str, owner: str, asker: Any, token_exp_ms: int) -> None:
    """Remember that a live local process just BORROWED ``key`` through this relay.

    Called only by ``serve_grant`` after the owner answered a session's ask with a
    grant (Q2: a key no local session actually borrowed is never renewed). ``asker``
    is the verified incarnation (``credentials._control_asker``); ``token_exp_ms``
    the token expiry that grant carries (``0`` for a key with none: never renewed
    here, the 24 h hold already covers it).
    """
    if not (key and owner and asker is not None and _renewable(key)):
        return
    active = _server_attr(server, _ACTIVE_ATTR)
    with _GUARD:
        entry = active.get(key)
        if entry is None or entry.get("owner") != owner:
            entry = active[key] = {"owner": owner, "askers": [], "token_exp_ms": 0}
        askers: list[Any] = entry["askers"]
        if not any(asker.same_process_as(known) for known in askers):
            askers.append(asker)
            del askers[:-MAX_ASKERS_PER_KEY]
        entry["token_exp_ms"] = max(int(entry.get("token_exp_ms") or 0), int(token_exp_ms))


def forget_active(server: Any, key: str | None = None) -> None:
    """Stop renewing (one key, or all). The next session borrow starts it again."""
    active = getattr(server, _ACTIVE_ATTR, None)
    if not isinstance(active, dict):
        return
    with _GUARD:
        if key is None:
            active.clear()
        else:
            active.pop(key, None)


def active_keys(server: Any) -> list[str]:
    """The keys this relay would currently consider renewing (tests, doctor)."""
    active = getattr(server, _ACTIVE_ATTR, None)
    if not isinstance(active, dict):
        return []
    with _GUARD:
        return list(active)


def _process_alive(identity: Any) -> bool:
    """Is that exact process incarnation still running? (pid AND start time)."""
    from local_operator.secrets.peer import process_info

    try:
        live = process_info(int(identity.pid))
    except Exception:  # noqa: BLE001 — unverifiable is gone (fail closed)
        return False
    return live is not None and identity.same_process_as(live)


def _held_token_exp_ms(server: Any, key: str, owner: str) -> int:
    held = getattr(server, _HELD_ATTR, None)
    if not isinstance(held, dict):
        return 0
    with _GUARD:
        entry = held.get(key)
    if entry is None or entry[0].credential_ref.owner_device != owner:
        return 0
    return int(entry[0].token_expires_at_ms or 0)


def _stop_key(server: Any, client: Any, owner: str, key: str, code: str) -> None:
    """A refusal BY NAME: stop renewing, drop what is held, say so (Q7/Q9)."""
    forget_active(server, key)
    drop_held(server, key)
    # ``_held_audit``: this device's audit, ids and the code only (Q9).
    client._held_audit(  # noqa: SLF001 — the client's own receipt writer
        "credential.renewal_refused", key, owner, cause="", code=code
    )


def pre_expiry_step(server: Any, owner: str) -> str | None:
    """Renew, just before expiry, the keys ``owner`` lends that a live session uses.

    One definitions-tick visit for the member ``owner`` (see the section comment
    for why and for every bound). Returns a comma-joined outcome per key asked or
    stopped, or ``None`` when there was nothing to do; never a policy-refusal
    code, so the definitions syncer never parks the member on this step's word.
    """
    stopping = getattr(server, "_stop", None)
    if stopping is not None and getattr(stopping, "is_set", lambda: False)():
        return None
    active = getattr(server, _ACTIVE_ATTR, None)
    if not isinstance(active, dict):
        return None
    with _GUARD:
        keys = [k for k, e in active.items() if e.get("owner") == owner]
    if not keys:
        return None
    from local_operator.network.credentials.client import MeshCredentialClient

    client = MeshCredentialClient.for_relay(server)
    if client is None:
        return None
    if not (client._network_trusted() and client._self_membership_active()):  # noqa: SLF001
        # PANIC / UNTRUSTED / REMOVED (Q5): send nothing, and drop what is held
        # locally. No ask is made, so there is nothing for an owner to refuse.
        for key in keys:
            forget_active(server, key)
            drop_held(server, key)
        return "stopped:untrusted_or_removed"
    # Q12: only over a LIVE, authenticated, DIRECT link to the owner. No dial, no
    # retry: an unreachable owner costs nothing until its link is back.
    link_for = getattr(server, "_link_for", None)
    link = link_for(owner, network_id=client.network_id) if link_for is not None else None
    if link is None or not getattr(link, "peer_verified", False):
        return None
    from local_operator.network.credentials.held_policy import PolicyUnknown

    outcomes: list[str] = []
    now_ms, now_boot = _now_ms(), _clock()
    for key in keys:
        with _GUARD:
            entry = active.get(key)
            askers = list(entry["askers"]) if entry else []
        if entry is None:
            continue
        if client.owner_of(key) != owner or not client.should_borrow(key):
            # Unshared / revoked here: the placement already says no; the next
            # ask would be refused ``not_a_holder`` by name, so do not make it.
            forget_active(server, key)
            drop_held(server, key)
            outcomes.append(f"{key}:stopped:not_a_holder")
            continue
        alive = [a for a in askers if _process_alive(a)]
        with _GUARD:
            if active.get(key) is entry:
                entry["askers"] = alive
        if not alive:
            # Q3: every session that borrowed this key has exited. Stop; the
            # held renewal ages out on its own hold end.
            forget_active(server, key)
            outcomes.append(f"{key}:stopped:no_live_session")
            continue
        try:
            policy = client._policy()  # noqa: SLF001 — the client's own policy store
            with policy.transaction() as document:
                denied = policy.code(document, owner, key)
        except PolicyUnknown:
            denied = "consent_unverifiable"
        if denied:
            # A consent withdrawal or held-policy deny on this device (Q6).
            forget_active(server, key)
            drop_held(server, key)
            outcomes.append(f"{key}:stopped:{denied}")
            continue
        token_exp = max(int(entry.get("token_exp_ms") or 0), _held_token_exp_ms(server, key, owner))
        if token_exp <= 0 or token_exp - now_ms > pre_expiry_lead_ms(key):
            continue
        with _GUARD:
            if entry.get("asked_for") == token_exp:
                continue  # Q7: one ask per (key, token life), whatever it answered
            # The cap's window runs on the sleep-inclusive clock: a wall-clock
            # step backwards must not reopen it.
            asks = [t for t in entry.get("asks", []) if now_boot - t < PRE_EXPIRY_CAP_WINDOW_S]
            if len(asks) >= PRE_EXPIRY_MAX_ASKS_PER_HOUR:
                entry["asks"] = asks
                continue  # Q7: the hard hourly cap
            # Stamped BEFORE the ask: an ask that never answers still spent its life.
            entry["asked_for"] = token_exp
            entry["asks"] = [*asks, now_boot]
        outcome = renew_key(server, client, owner, key, dial=False, trigger="pre_expiry")
        if outcome.startswith("refused:"):
            code = outcome.split(":", 1)[1]
            if code not in _BACK_OFF_CODES:
                _stop_key(server, client, owner, key, code)
                outcomes.append(f"{key}:stopped:{code}")
                continue
        if outcome == "renewed":
            fresh = _held_token_exp_ms(server, key, owner)
            with _GUARD:
                if active.get(key) is entry and fresh > int(entry.get("token_exp_ms") or 0):
                    entry["token_exp_ms"] = fresh
        outcomes.append(f"{key}:{outcome}")
    return ",".join(outcomes) or None


def pre_expiry_tick(server: Any, device_id: str) -> str | None:
    """The definitions-tick entry for :func:`pre_expiry_step`.

    Resolves the step through the module at call time so a test seam (the
    network conftest keeps it quiet by default, as it does the renew nudge) is
    honoured by the registered step. Failures never reach the syncer's parking.
    """
    try:
        pre_expiry_step(server, device_id)
    except Exception:  # noqa: BLE001 — a renewal never takes the cadence down
        logger.debug("pre-expiry renewal for %s failed", device_id, exc_info=True)
    return None


def install() -> None:
    """Register :func:`wake_step` and :func:`pre_expiry_tick` on the definitions
    syncer's tick (idempotent)."""
    from local_operator.network import definitions

    setattr(wake_step, "independent_cadence", True)
    definitions.add_tick_step(wake_step)
    setattr(pre_expiry_tick, "independent_cadence", True)
    definitions.add_tick_step(pre_expiry_tick)
