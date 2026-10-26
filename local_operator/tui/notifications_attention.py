"""The running TUI's two inbox duties: rung-3 banners and in-view receipts.

BOTH RIDE THE 1 s ATTENTION POLL that already runs (``OperatorApp
._poll_completion_attention``); neither adds a timer. Both are pure decisions
over values the app hands in, so the rules are testable without an app, and
both write only through the store's own doors.

RUNG 3 (contract §3.5): "a TUI is running on this machine" — its background
announcer raises a ``high`` item that no earlier rung delivered. CLAIM FIRST,
then raise, then release on a spawn that reported nothing: the same
``claim_delivery`` watermark the desktop feed (rung 2) and the runtime (rung 4)
claim, so one revision raises at most one banner machine-wide however many
surfaces race. A desktop advertising ``update`` owns the raise (the feed claims
on its own tick), so this leg stands down rather than racing it. ``normal`` and
``low`` never banner (``undelivered(priority="high")``). ``BURST_LIMIT`` caps a
tick; the overflow is claimed and raised as ONE digest, and handed back if the
digest itself fails.

IN-VIEW RECEIPTS (design §17.3, ``docs/ATTENTION.md`` "Items"). The TUI may mark
an item read ONLY with all of: its notify card rendered in the visible
transcript region of the focused session, no full-page mode or modal over it,
real terminal focus evidence (never Textual's startup value), and a real key or
mouse press AFTER the card became visible. The evidence goes to the store's
``seen`` door with ``surface="tui"``, which re-checks it per item (current
revision, matching anchor, input after visibility and after the item's last
change) — the TUI asserts, the store decides. A scrolled-off card, another
session, a press before render, or a newer revision all stay unread.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: The per-tick banner ceiling, the desktop feed's (``desktop_feed.BURST_LIMIT``)
#: and the completion announcer's: three at once is already an interruption.
BURST_LIMIT = 3


@dataclass(frozen=True)
class BannerTick:
    """What one rung-3 pass did, for the caller's log and the tests."""

    raised: tuple[str, ...] = ()
    digested: tuple[str, ...] = ()
    released: tuple[str, ...] = ()
    stood_down: str = ""


def announce_items(
    root: Path | str,
    *,
    raise_one: Callable[[Mapping[str, Any]], bool],
    raise_digest: Callable[[Sequence[Mapping[str, Any]]], bool],
    desktop_delivers_update: Callable[[], bool],
    store: Any = None,
    claimant: str = "tui",
    held_session_id: str = "",
    now_ms: int | None = None,
) -> BannerTick:
    """Raise every unclaimed, unread, undismissed ``high`` item once.

    ``raise_one`` / ``raise_digest`` spawn the OS banner and report whether one
    started; the app passes its existing ``detached_notify`` wrapper. A False
    or an exception hands that revision's claim back, so the next surface (or
    this one, next tick) can try again — a watermark that asserts a banner
    nobody received is the hole ``release_delivery`` exists to close.

    TWO HOLDS, both the desktop feed's own rules (T2, and ``delivery.py``):

    * GRACE. A ``high`` revision is held for ``BANNER_GRACE_MS`` after its
      content last changed, so an in-view receipt can cancel it. This leg runs
      every second and would otherwise claim on the first tick, which is faster
      than any receipt can be and also pre-empts rung 4's hold.
    * THE WATCHED SESSION. An item written by the conversation this TUI is
      SHOWING is not news to banner: the operator is looking at the card, and
      the in-view receipt is the acknowledgement that fits. Those rows are left
      UNCLAIMED (not raised, not released) so a surface that is not being
      watched — or this one once the operator moves away — still owns them.
      ``held_session_id`` is empty when the app cannot prove it is displaying
      that session, which fails toward notifying.
    """
    if desktop_delivers_update():
        # Rung 2 is live: the feed claims and raises on its own 100 ms tick.
        return BannerTick(stood_down="desktop update channel")
    if store is None:
        from local_operator.notifications.store import NotificationStore

        store = NotificationStore(root)
    from local_operator.notifications.delivery import grace_remaining_ms

    stamp = now_ms if now_ms is not None else int(time.time() * 1000)
    pending = [
        row
        for row in store.undelivered(priority="high", limit=BURST_LIMIT + 1)
        if grace_remaining_ms(row, stamp) <= 0
        and not (held_session_id and str(row.get("session_id") or "") == held_session_id)
    ]
    if not pending:
        return BannerTick()
    claimed: list[Mapping[str, Any]] = []
    for row in pending:
        try:
            if store.claim_delivery(str(row["id"]), claimant):
                claimed.append(row)
        except Exception:  # noqa: BLE001 — one unclaimable row must not mute the rest
            logger.debug("notifications: claim failed for %s", row.get("id"), exc_info=True)
    raised: list[str] = []
    released: list[str] = []
    for row in claimed[:BURST_LIMIT]:
        if _attempt(lambda r=row: raise_one(r)):
            raised.append(str(row["id"]))
        else:
            _release(store, row, claimant)
            released.append(str(row["id"]))
    overflow = claimed[BURST_LIMIT:]
    digested: list[str] = []
    if overflow:
        if _attempt(lambda: raise_digest(overflow)):
            digested = [str(row["id"]) for row in overflow]
        else:
            for row in overflow:
                _release(store, row, claimant)
                released.append(str(row["id"]))
    return BannerTick(tuple(raised), tuple(digested), tuple(released))


def _attempt(spawn: Callable[[], bool]) -> bool:
    try:
        return bool(spawn())
    except Exception:  # noqa: BLE001 — a banner is chrome; the item is durable
        logger.debug("notifications: banner spawn raised", exc_info=True)
        return False


def _release(store: Any, row: Mapping[str, Any], claimant: str) -> None:
    try:
        store.release_delivery(str(row["id"]), int(row["revision"]), claimant)
    except Exception:  # noqa: BLE001 — an unreleasable claim leaves the row unread
        logger.debug("notifications: release failed for %s", row.get("id"), exc_info=True)


@dataclass(frozen=True)
class VisibleCard:
    """One notify card on screen: its anchor and when it first became visible."""

    anchor_id: str
    visible_since_ms: int


def seen_candidates(
    rows: Iterable[Mapping[str, Any]],
    visible: Sequence[VisibleCard],
    *,
    session_id: str,
    input_at_ms: int,
) -> tuple[list[dict[str, Any]], int]:
    """Items whose card was on screen BEFORE the operator's last real input.

    Returns the ``seen`` door's item list and the visibility stamp to assert
    (the latest of the matched cards', so every claimed card was visible by
    then). An input that predates a card's visibility does not count for it —
    the press has to land on a frame that already showed the card.
    """
    by_anchor = {card.anchor_id: card for card in visible if card.anchor_id}
    items: list[dict[str, Any]] = []
    since = 0
    for row in rows:
        if str(row.get("session_id") or "") != session_id:
            continue
        anchor = str(row.get("anchor_id") or "")
        card = by_anchor.get(anchor)
        if card is None or not (0 < card.visible_since_ms <= input_at_ms):
            continue
        items.append({"id": str(row["id"]), "revision": int(row["revision"]), "anchor_id": anchor})
        since = max(since, card.visible_since_ms)
    return items, since


def submit_seen(
    root: Path | str,
    items: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
    input_at_ms: int,
    visible_since_ms: int,
) -> dict[str, Any] | None:
    """POST /v1/desktop/notifications/seen's in-process twin, ``surface="tui"``.

    The same ``notifications.items.apply_seen`` the route calls, so the TUI's
    receipt is re-checked by exactly the rules a desktop receipt is.
    """
    if not items:
        return None
    from local_operator.notifications import items as inbox

    return inbox.apply_seen(
        root,
        list(items),
        {
            "session_id": session_id,
            "input_at_ms": input_at_ms,
            "visible_since_ms": visible_since_ms,
            "surface": "tui",
        },
    )


def unread_for_session(
    root: Path | str, session_id: str, *, store: Any = None
) -> list[dict[str, Any]]:
    """This session's unread, undismissed notify rows that carry an anchor."""
    if store is None:
        from local_operator.notifications.store import NotificationStore

        store = NotificationStore(root)
    rows, _ = store.list_items(state="unread", session_id=session_id, limit=100)
    return [dict(row) for row in rows if row.get("anchor_id")]
