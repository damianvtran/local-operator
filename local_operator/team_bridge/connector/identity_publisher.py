"""Publish a chief-of-staff rename to the hub the moment it lands locally.

WHY THIS EXISTS (report 2026-11-17, core v0.101.38). A member renamed her
chief of staff "Aida" -> "Aubree" -> "Aubee". A #general post sent after the
second rename still carried "Aida", and the hub directory showed "Aubree" for a
while after the local setting had moved on. The hub derives every agent label
(directory, DMs, channel senders) from the endpoint's ``agent_name``, so the
labels were only as fresh as that field — and the client re-sent it ONLY at a
fresh status read (``StatusDiscipline._sync_agent_name``): at most once per
``STATUS_CACHE_TTL_S`` (300 s) and only when a bridge surface opened. The one
exception was the desktop settings page; ``/aida rename``, a session rename
synced into ``aida.name`` and the agent renaming herself all waited.

WHY THE CONNECTOR. It is the one long-lived bridge process per machine (it
holds ``ConnectorLock``) and it runs exactly while this machine owns an
enrolled endpoint — the only time there is a name to publish. Hooking every
``lop`` process instead would make N processes race the same PUT for one
rename. The settings page keeps its own immediate path; the two converge
because ``_sync_agent_name`` PUTs only when the hub's answer differs.

WHAT FIRES IT. The process config watcher (``config_watch``): every writer
lands in ``config.yml`` — ``aida.name`` for a rename, ``team_bridge.session``
when the designated chief of staff changes — and the watcher reports which
registry keys moved within its 2 s poll. Also once at start, so a rename made
while the connector was down is published when it comes back. Event-driven,
never a timer: the connector's "no periodic status call" rule holds.

RETRY. A refused/unavailable read or a failed PUT is retried after
``RETRY_DELAYS_S`` (5 s, 30 s, 120 s, 300 s), then left to the ordinary
status boundary. The last step is 300 s because the edge WAF's throttle
answers "retry in 300 seconds" (#4172), and the status discipline holds a
throttled answer for the wait it names, so a retry inside that window costs no
request. A new rename resets the schedule.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

#: The registry keys whose change can alter the published name.
IDENTITY_KEYS = frozenset({"aida.name", "team_bridge.session"})

#: Delays before each retry of a publish that did not settle (see module doc).
RETRY_DELAYS_S: tuple[float, ...] = (5.0, 30.0, 120.0, 300.0)


async def run(
    config_dir: Path,
    *,
    base_url: str,
    stop: asyncio.Event,
    publish: Callable[[], Awaitable[bool]] | None = None,
    watcher: Any = None,
    retry_delays: tuple[float, ...] = RETRY_DELAYS_S,
) -> None:
    """Publish once now, then on every identity change, until ``stop`` is set.

    ``publish`` and ``watcher`` are seams for tests; production resolves the
    shared status discipline and the process config watcher. Never raises: a
    display name must not take the connector down.
    """

    if publish is None:
        from local_operator.team_bridge.client import get_status_discipline

        discipline = get_status_discipline(config_dir, base_url)
        publish = discipline.publish_identity
    if watcher is None:
        from local_operator.config_watch import process_watcher

        watcher = process_watcher(config_dir)
    changed = asyncio.Event()
    changed.set()  # the start-up publish

    def on_change(change: Any) -> None:
        if IDENTITY_KEYS & frozenset(getattr(change, "changed_keys", ()) or ()):
            changed.set()

    unsubscribe = watcher.subscribe(on_change)
    try:
        watcher.start()
    except Exception as exc:  # noqa: BLE001 - without a watcher, start-up publish still runs
        logger.warning("bridge connector: identity watch unavailable (%s)", type(exc).__name__)
    try:
        while not stop.is_set():
            if not await _wait_either(changed, stop, None):
                return
            changed.clear()
            for delay in (*retry_delays, None):
                if await _publish_once(publish):
                    break
                if delay is None:
                    logger.info("bridge connector: identity publish left to the next status read")
                    break
                # A new change during the wait restarts the schedule with the new name.
                if not await _wait_either(changed, stop, delay) or changed.is_set():
                    break
    finally:
        unsubscribe()


async def _publish_once(publish: Callable[[], Awaitable[bool]]) -> bool:
    try:
        return bool(await publish())
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - retried; class name only (prose may carry the name)
        logger.warning("bridge connector: identity publish failed (%s)", type(exc).__name__)
        return False


async def _wait_either(changed: asyncio.Event, stop: asyncio.Event, timeout: float | None) -> bool:
    """Wait for a change, stop, or ``timeout``; ``False`` only when stopping."""

    waiters = [asyncio.ensure_future(changed.wait()), asyncio.ensure_future(stop.wait())]
    try:
        await asyncio.wait(waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for waiter in waiters:
            waiter.cancel()
    return not stop.is_set()
