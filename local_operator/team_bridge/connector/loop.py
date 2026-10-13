"""The events long-poll loop + cursor/resync walk (H4b).

TRANSPORT SHAPE, pinned by §6.7/§13.3. The connector's ONLY recurring bridge
traffic is ONE outstanding ``GET /v1/messaging/events?after=<cursor>&wait=25``:
``200`` → refs are recorded and drained, ``204`` → the read returns ``None``
and the loop simply reissues. There is deliberately no periodic status/inbox
call, no plain-poll fallback, and no scheduled model wake — a missed event
costs LATENCY, never state, because the inbox list is the source of truth and
the resync walk repairs gaps (empty ``events`` + fresh cursor, or a cursor the
server no longer honours).

CURSOR. The native ``sync`` row is the only store (`record_feed_batch` writes
it, `sync_state` reads it); ``feed_seq`` is the ``after=`` value, and a
``pending_resync`` flag routes the loop into the walk before the next poll.
Refs are durable in ``feed_refs`` and drained per cursor value — crash after
commit replays nothing, crash before replays the same batch.

RESYNC WALK. Drives ``pull.inbox`` pagination to a TERMINAL page
(``next_cursor is None``); H2's witness (`finish_resync`) refuses completion
unless every page was merged through the terminal one, so an interrupted walk
can never be recorded as complete. The baseline checkpoint captured BEFORE the
walk becomes the new resume point, so arrivals during the walk are not skipped.

``RADIENT_API_BASE_URL`` is read by the CALLER at process start and its value
rides the wire client's own resolution; the loop never re-reads the
environment mid-flight (restart-scoped, §13.3).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from local_operator.team_bridge.connector.wake import (
    handle_message_available,
    parse_event_ref,
)

#: Bounded exponential backoff between loop iterations after a FAILURE (a
#: refused read, a transport error). A 204 timeout is NOT a failure — it
#: resets the backoff and reissues promptly.
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 60.0

#: Bounded walk: a page loop that never terminates is a server bug, not a
#: reason to spin — cap and surface so the next resync retries.
MAX_WALK_PAGES = 200

#: Reissue delay after a 204 timeout. Small deliberately: the server held the
#: request for the full wait window; a fresh poll is the contract's shape.
REISSUE_DELAY_S = 0.2


async def resync_walk(owner: Any, *, config_dir: Path, access: Any) -> dict[str, Any]:
    """Drive the inbox pagination to a terminal page, then complete it.

    Completion is H2's witness, not this loop's confidence: `finish_resync`
    refuses unless the LAST merged page was terminal (`walk_terminal`).
    """

    from local_operator.team_bridge import pull as _pull
    from local_operator.team_bridge.store import Store

    store = Store(config_dir)
    baseline = store.sync_state(owner)["inbox_checkpoint"]
    cursor = baseline
    pages = 0
    while True:
        page = await _pull.inbox(owner, config_dir=config_dir, access=access, after=cursor)
        pages += 1
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
        if pages >= MAX_WALK_PAGES:
            raise RuntimeError("inbox resync walk exceeded the bounded page count")
    from local_operator.team_bridge.types import InboxCursor

    _pull.finish_resync(owner, InboxCursor(baseline or ""), config_dir=config_dir)
    return {"pages": pages}


async def run_iteration(
    owner: Any,
    *,
    config_dir: Path,
    access: Any,
    auth_store: Any,
    settings: Any = None,
) -> dict[str, Any]:
    """ONE loop turn: one read, the drained refs, the walk when due.

    Returns metadata only (counts/actions) — no body, no excerpt.
    """

    from local_operator.team_bridge import pull as _pull
    from local_operator.team_bridge.store import Store

    store = Store(config_dir)
    state = store.sync_state(owner)
    after = state["feed_seq"] or None
    batch = await _pull.events(owner, config_dir=config_dir, access=access, after=after)
    if batch is None:
        return {"events": 0, "reissued": True, "resync": False, "handled": []}
    refs = _pull.drain_feed_refs(owner, config_dir=config_dir, upto_seq=batch["cursor"])
    handled: list[dict[str, Any]] = []
    for ref in refs:
        parsed = parse_event_ref(ref)
        if parsed is None:
            continue
        event_type, kind, value = parsed
        if event_type != "message_available" or kind != "message_id":
            # state_changed/request_* refs are the CoS session's own reads'
            # business; the connector only WAKES on new deliveries.
            continue
        handled.append(
            await handle_message_available(
                owner,
                value,
                config_dir=config_dir,
                auth_store=auth_store,
                access=access,
                settings=settings,
            )
        )
    resync = bool(batch["resync"]) or bool(store.sync_state(owner)["pending_resync"])
    walk: dict[str, Any] | None = None
    if resync:
        walk = await resync_walk(owner, config_dir=config_dir, access=access)
    return {
        "events": len(refs),
        "cursor": batch["cursor"],
        "reissued": False,
        "resync": resync,
        "walk": walk,
        "handled": handled,
    }


async def run_forever(
    owner: Any,
    *,
    config_dir: Path,
    auth_store: Any,
    access_provider: Any,
    settings: Any = None,
    stop: asyncio.Event | None = None,
) -> None:
    """The connector's steady state: poll, handle, back off, repeat.

    ``access_provider`` is an async callable returning the pinned OAuth
    access for this owner (the H1 client's own ``current_access`` — the
    ordinary sign-in path, never a key cascade). A refused iteration backs
    off (`BACKOFF_BASE_S` doubling to `BACKOFF_MAX_S`); a 204 resets it.
    """

    backoff = BACKOFF_BASE_S
    while stop is None or not stop.is_set():
        try:
            access = await access_provider()
            summary = await run_iteration(
                owner,
                config_dir=config_dir,
                access=access,
                auth_store=auth_store,
                settings=settings,
            )
            if summary.get("reissued"):
                backoff = BACKOFF_BASE_S
                await asyncio.sleep(REISSUE_DELAY_S)
            else:
                backoff = BACKOFF_BASE_S
        except asyncio.CancelledError:
            raise
        except Exception:
            # No content in the backoff path (privacy): the failure is
            # operational, and the next iteration re-tries the ordinary way.
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, BACKOFF_MAX_S)
