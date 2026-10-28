"""The item banner ladder (contract §3.5): who may raise a `high` item, once.

`normal` and `low` NEVER banner — they reach the operator through the badge and
the reel. A `high` item raises at most one banner per REVISION, machine-wide,
because every rung claims the same watermark in the store before it raises
(`NotificationStore.claim_delivery`) and a lost claim means somebody else
already told them.

PRESENCE IS NEGATIVE-ONLY HERE. The operator's presence addendum forbids
treating a focused window or fresh input as proof that a callout was seen, so
this module never writes a read mark and never suppresses a banner because
someone is "active". What presence may do is say that another channel CAN
deliver: a desktop app advertising the `update` kind raises the item through
the feed, so the runtime must not also spawn an OS banner for it.

THE GRACE IS BOUNDED AND ONE-WAY (round-1 N2/Q4). The operator asked that
something they are already looking at not ALSO raise a redundant banner. The
desktop's in-view receipt needs a 1 s dwell after a fresh input, so it can
never land before a raise that happens at the write. Every rung therefore
HOLDS a `high` revision for :data:`BANNER_GRACE_MS` after its content last
changed (`updated_at`), then claims only if the row is STILL UNREAD:
- a receipt accepted inside the grace cancels the banner;
- a receipt after it does nothing to a banner already raised;
- the grace never suppresses permanently. Expiry RAISES an unread revision,
  and a later content change bumps `revision` and re-arms it.
The runtime rung waits off-loop in its own background task, so the tool call
that wrote the item never waits on it.

ONLY A DELIVERABLE OUTCOME RAISES (round-1 R1). `created`, `updated` and
`revived` are new content the operator has not been told about; `unchanged`
is a repeat with nothing new, and `absorbed` is an update to an item the
operator DISMISSED, which contract §1.3 records "silently … no banner".

GATES AND PRIVACY (round-1 R2/Q1). Every rung applies the completion leg's
two refusals before it claims, because a claim with no raise strands the
watermark: the operator's notifications switch (`notifications_enabled`)
and the rig/foreign-HOME guard (`desktop_belongs_to_this_process`). Banner
text follows `display.notification_session_name` through `compose.
_names_allowed`: off ⇒ `APP_NAME` and the house sentence, never the
model-written title or body (contract §1.5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from local_operator.notifications.store import NotificationStore

#: How long every rung holds a `high` revision before raising it, measured from
#: the row's last CONTENT change (`updated_at`), so that an in-view receipt can
#: cancel a banner for something the operator is already looking at (N2/Q4).
#: The desktop's earliest valid receipt is the 1 s dwell after a fresh input
#: (`store.seen`); QA measured the first accepted `/seen` at 1026 ms after the
#: write. 2.5 s admits that receipt plus a tick and a heartbeat of slack, and is
#: still far below the minutes-scale delay that would make a high item late.
BANNER_GRACE_MS = 2500

#: The `notify` outcomes that carry content the operator has not been told
#: about. `unchanged` and `absorbed` never raise (R1; contract §1.3).
DELIVERABLE_ACTIONS = frozenset({"created", "updated", "revived"})

#: Rung names, as reported by :func:`deliver_item` and logged by the caller.
RUNG_PRIORITY = "priority"  # normal/low: never banners
RUNG_READ = "read"  # already acknowledged: nothing to raise
RUNG_FEED = "feed"  # the desktop app will raise it from the feed
RUNG_DETACHED = "detached"  # this process spawns the OS banner
RUNG_NONE = "none"  # nothing can raise it right now
RUNG_GATED = "gated"  # the operator's switch or the rig guard refused it
RUNG_SILENT = "silent"  # unchanged/absorbed: no new content to announce
#: A TUI is showing this conversation and the operator is active in it, so the
#: card is already on screen. The TUI's rung-3 leg stands down for that case and
#: its in-view receipt acknowledges the item; without this rung the RUNTIME
#: still raised a detached banner over a card the operator was reading (R2-2).
RUNG_TUI_VIEWER = "tui-viewer"


@dataclass(frozen=True)
class DeliveryResult:
    """What the ladder decided and what it actually did.

    ``raised`` answers "did a banner go out from HERE", never "will the
    operator see it". ``handed_off`` is the other way an item banner is on
    its way: a live desktop subscription that advertises ``update`` now owns
    the raise.
    """

    raised: bool = False
    rung: str = RUNG_NONE
    claimed: bool = False
    detail: str = ""
    handed_off: bool = False

    @property
    def spoken_for(self) -> bool:
        """Whether an item banner for this run is actually on its way (R1).

        True ONLY when this process raised one, or a deliverable revision was
        handed to an advertising desktop. A gated, silent, read or failed
        outcome is not spoken for, so it never suppresses the run's completion
        banner: suppressing one banner for another that never comes is the
        "no banner on any surface" failure round 1 measured.
        """
        return self.raised or self.handed_off


def gates_refuse() -> str:
    """The rung-independent refusals, or "" when a banner may be raised (R2/Q1).

    The completion legs' two gates in their order: the operator's switch,
    then the rig guard. A gated revision is NOT claimed, so a later process
    with the gates open (the operator turning banners back on, the real HOME)
    can still raise it.
    """
    from local_operator.tui.notify import (
        desktop_belongs_to_this_process,
        notifications_enabled,
    )

    if not notifications_enabled():
        return "notifications disabled"
    if not desktop_belongs_to_this_process():
        return "foreign HOME (rig guard)"
    return ""


def banner_text(title: str, body: str) -> tuple[str, str]:
    """``(title, body)`` for an item banner under the privacy flag (contract §1.5).

    Flag on: the item's own title and a bounded body. Flag off — or a flag
    that cannot be read, ``_names_allowed`` fails closed — the brand and the
    house sentence: nothing an agent wrote reaches a lock screen.
    """
    from local_operator.notifications.compose import _names_allowed
    from local_operator.tui.notify import APP_NAME, BODY_UPDATE, sanitize_text

    if not _names_allowed():
        return APP_NAME, BODY_UPDATE
    return sanitize_text(title, 120), sanitize_text(body, 240)


def grace_remaining_ms(row: dict[str, Any], now_ms: int | None = None) -> int:
    """How long this revision is still held for an in-view receipt (N2/Q4)."""
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    return max(0, int(row.get("updated_at") or 0) + BANNER_GRACE_MS - now_ms)


def route_item(
    config_dir: Path | str,
    *,
    action: str,
    priority: str,
    presence: Any = None,
    tui_watching: bool | None = None,
) -> DeliveryResult:
    """The write-time half: is there anything to raise, and who raises it?

    Synchronous and cheap, so the session door can decide the run's completion
    suppression before the tool call returns, while the grace hold and the
    raise happen in the background (:func:`deliver_item`). ``handed_off`` is
    set only for a deliverable revision a desktop subscription that advertises
    ``update`` will raise, read UNCACHED (R3): a 2 s-old lease could name an
    app that has already quit.

    ``tui_watching`` is the rung-3 mirror (R2-2): True when a TUI is showing
    this exact conversation AND the operator is active in it. Only an explicit
    True suppresses; ``None`` — no evidence available, an older TUI, a probe
    that failed — falls through to the normal rungs, because a missed banner is
    the failure the whole ladder exists to prevent (contract §6.3).
    """
    if str(priority) != "high":
        return DeliveryResult(rung=RUNG_PRIORITY)
    if action not in DELIVERABLE_ACTIONS:
        return DeliveryResult(rung=RUNG_SILENT, detail=action)
    refused = gates_refuse()
    if refused:
        return DeliveryResult(rung=RUNG_GATED, detail=refused)
    if presence is None:
        from local_operator.session.runtime.presence import desktop_presence

        presence = desktop_presence(Path(config_dir), cached=False)
    if tui_watching is True:
        # The card is on screen in the focused conversation. Nothing is handed
        # off and nothing raised: the TUI's in-view receipt acknowledges it.
        return DeliveryResult(rung=RUNG_TUI_VIEWER, detail="a TUI is showing it")
    if presence is not None and presence.delivers("update"):
        return DeliveryResult(rung=RUNG_FEED, detail="desktop update channel", handed_off=True)
    return DeliveryResult(rung=RUNG_DETACHED, detail="runtime will raise")


def deliver_item(
    config_dir: Path | str,
    *,
    item_id: str,
    revision: int,
    priority: str,
    action: str = "created",
    session_id: str = "",
    title: str = "",
    body: str = "",
    store: NotificationStore | None = None,
    presence: Any = None,
    notify: Callable[..., bool] | None = None,
    allow_detached: bool = True,
    hold: bool = True,
    tui_watching: bool | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> DeliveryResult:
    """Decide the rung for one written item and raise it at most once.

    BLOCKING by design when ``hold`` is set: it sleeps out the revision's
    grace before claiming, so callers run it off-loop (the session door spawns
    it as a background task and never awaits it from the tool call).

    Injected seams (``store``, ``presence``, ``notify``, ``sleep``) are for the
    tests and for a caller that already holds a live presence reading;
    production passes none of them and gets the real store, the real lease
    reader and ``tui.notify.detached_notify``.
    """
    route = route_item(
        config_dir,
        action=action,
        priority=priority,
        presence=presence,
        tui_watching=tui_watching,
    )
    if route.rung != RUNG_DETACHED:
        return route
    # ONE Path at the boundary: callers pass either spelling (the session hands
    # a str, the tests a tmp_path), and `desktop_presence` keys its cache on
    # `Path`. Passing the str through made the lease read type-unsound.
    config_dir = Path(config_dir)
    store = store or NotificationStore(config_dir)
    row = store.get(item_id)
    if row is None:
        return DeliveryResult(rung=RUNG_NONE, detail="unknown")
    # THE GRACE (N2/Q4): wait out the revision's hold, then re-read, so an
    # in-view receipt accepted meanwhile is what the checks below see.
    if hold:
        wait_ms = grace_remaining_ms(row)
        if wait_ms > 0:
            sleep(wait_ms / 1000)
        row = store.get(item_id)
        if row is None:
            return DeliveryResult(rung=RUNG_NONE, detail="unknown")
    if row.get("dismissed_at") is not None:
        return DeliveryResult(rung=RUNG_SILENT, detail="dismissed before raise")
    # STILL UNREAD? A receipt that beat the raise is the one suppression the
    # grace allows (see the module docstring). Compare against the row's own
    # current watermark, never a clock: a write and a read are ordered by
    # revision, so a stale reading cannot suppress a newer one.
    if int(row["read_rev"]) >= int(row["revision"]):
        return DeliveryResult(rung=RUNG_READ, detail="read before raise")
    if int(row["revision"]) != int(revision):
        return DeliveryResult(rung=RUNG_READ, detail="superseded before raise")
    # RE-ROUTE AFTER THE HOLD: an app that started advertising `update` while
    # this revision was held now owns it (the feed claims on its own tick), and
    # the operator may have switched banners off meanwhile.
    late = route_item(config_dir, action=action, priority=priority, presence=presence)
    if late.rung != RUNG_DETACHED:
        return late
    if not allow_detached:
        return DeliveryResult(rung=RUNG_NONE, detail="detached delivery disabled")
    if notify is None:
        from local_operator.tui.notify import detached_notify

        notify = detached_notify
    # RUNG 4: nothing else can carry it, so this process does. Claim first, so
    # a second runtime (or a feed that starts advertising between the write and
    # the claim) cannot double-banner the same revision.
    if not store.claim_delivery(item_id, "runtime"):
        return DeliveryResult(rung=RUNG_NONE, detail="already delivered")
    banner_title, banner_body = banner_text(
        title or str(row.get("title") or ""), body or str(row.get("body_md") or "")
    )
    try:
        started = bool(notify(banner_title, banner_body, session_id=session_id))
    except Exception:  # noqa: BLE001 — a banner is chrome; the item is durable
        started = False
    if started:
        return DeliveryResult(raised=True, rung=RUNG_DETACHED, claimed=True)
    # RELEASE ON A SPAWN THAT REPORTED NOTHING (contract §3.5): the claim is a
    # promise that a banner exists; an unstarted child made none, and holding
    # the watermark would stop the next surface from ever raising this
    # revision. Best-effort — a store that cannot release leaves the item
    # unread, which is the direction every unknown falls.
    try:
        store.release_delivery(item_id, revision, "runtime")
    except Exception:  # noqa: BLE001
        pass
    return DeliveryResult(rung=RUNG_NONE, detail="notifier did not start")
