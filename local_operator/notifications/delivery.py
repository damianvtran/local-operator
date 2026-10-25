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

THE GRACE IS SHORT AND ONE-WAY. A banner is pointless if the operator has
already read the item — the desktop's in-view receipt can land between the
write and the raise (the feed claims on its next tick) — so a claim requires
the row to be STILL UNREAD at raise time. That is the whole grace: it can only
suppress a banner for content the operator has already acknowledged, and it
never suppresses anything permanently, because a later content update bumps
`revision`, which makes the row unread and claimable again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from local_operator.notifications.store import NotificationStore

#: How long after a write the desktop's in-view receipt may still suppress the
#: banner by marking the item read. The raise is compared against the row's
#: CURRENT read watermark rather than a timestamp, so this is the window in
#: which the feed's 100 ms tick and a receipt can race: 2 s covers it with
#: room to spare, and a late read does not un-raise anything already raised.
BANNER_GRACE_MS = 2000

#: Rung names, as reported by :func:`deliver_item` and logged by the caller.
RUNG_PRIORITY = "priority"  # normal/low: never banners
RUNG_READ = "read"  # already acknowledged: nothing to raise
RUNG_FEED = "feed"  # the desktop app will raise it from the feed
RUNG_DETACHED = "detached"  # this process spawns the OS banner
RUNG_NONE = "none"  # nothing can raise it right now


@dataclass(frozen=True)
class DeliveryResult:
    """What the ladder decided and what it actually did.

    ``raised`` answers "did a banner go out from HERE", never "will the
    operator see it": the desktop rung hands the raise to another process. A
    caller that needs the completion-banner suppression treats ``raised`` and
    ``rung == RUNG_FEED`` alike — an item banner is on its way either way.
    """

    raised: bool = False
    rung: str = RUNG_NONE
    claimed: bool = False
    detail: str = ""

    @property
    def spoken_for(self) -> bool:
        """Whether this run's item banner is on its way (feed counts)."""
        return self.rung == RUNG_FEED or self.raised


def deliver_item(
    config_dir: Path | str,
    *,
    item_id: str,
    revision: int,
    priority: str,
    session_id: str = "",
    title: str = "",
    body: str = "",
    store: NotificationStore | None = None,
    presence: Any = None,
    notify: Callable[..., bool] | None = None,
    allow_detached: bool = True,
) -> DeliveryResult:
    """Decide the rung for one written item and raise it at most once.

    Injected seams (``store``, ``presence``, ``notify``) are for the tests and
    for a caller that already holds a live presence reading; production passes
    none of them and gets the real store, the real lease reader and
    ``tui.notify.detached_notify``.
    """
    if str(priority) != "high":
        return DeliveryResult(rung=RUNG_PRIORITY)
    store = store or NotificationStore(config_dir)
    row = store.get(item_id)
    if row is None:
        return DeliveryResult(rung=RUNG_NONE, detail="unknown")
    # STILL UNREAD? A receipt that beat the raise is the one suppression the
    # grace allows (see the module docstring). Compare against the row's own
    # current watermark, never a clock: a write and a read are ordered by
    # revision, so a stale reading cannot suppress a newer one.
    if int(row["read_rev"]) >= int(row["revision"]):
        return DeliveryResult(rung=RUNG_READ, detail="read before raise")
    if int(row["revision"]) != int(revision):
        return DeliveryResult(rung=RUNG_READ, detail="superseded before raise")
    if presence is None:
        from local_operator.session.runtime.presence import desktop_presence

        presence = desktop_presence(config_dir)
    # RUNG 2: a desktop app that advertises `update` raises it from the feed.
    # No claim here — that pass claims inside its own tick, so the two rungs
    # cannot both win the same revision.
    if presence is not None and presence.delivers("update"):
        return DeliveryResult(rung=RUNG_FEED, detail="desktop update channel")
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
    try:
        started = bool(notify(title or str(row["title"]), body, session_id=session_id))
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
