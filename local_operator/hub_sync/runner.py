"""The periodic hub check/apply runner (design B3.2, B3.3).

One :class:`HubSyncRunner` per ``lop serve`` daemon, started from the server
lifespan as a single asyncio task. The routes call the SAME object
(``check_now``/``apply``), so the timer, the buttons and (through the shared
service) the CLI cannot diverge about what a check or an update is.

"AUTO ON" vs "MANUAL" BOTH KEEP CHECKING (requirement). ``hub.auto_update.*``
false means the runner still checks and records ``available`` — that is what
makes the sidebar indicator work in manual mode — and simply never merges. True
means a check that finds ``available`` additionally merges and applies iff the
outcome is a clean ``merged`` with a known baseline; anything else stays
``available`` with the reason recorded.

QUIET BY CONSTRUCTION. ``run_forever`` never raises; every tick catches per item;
nothing here logs above WARNING except a once-per-item-per-class summary. A
missing login is not a failure.

NO DOUBLE APPLY, three layers: one runner per daemon plus an ``asyncio.Lock``;
a cross-process lease so a second daemon or a CLI run skips an item another
process is merging; and the apply's own under-lock fingerprint re-verify.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from local_operator.hub_sync import service as svc
from local_operator.hub_sync import store as st
from local_operator.hub_sync.check import Link, list_links
from local_operator.hub_sync.report import ItemMergeReport
from local_operator.hub_sync.settings import HubSyncSettings

logger = logging.getLogger(__name__)

#: Startup delay so a daemon boot is never blocked or slowed by a network pass.
STARTUP_DELAY_S = 45.0
STARTUP_JITTER_S = 15.0
#: At most this many items per tick, oldest ``last_checked`` first (B3.3).
MAX_ITEMS_PER_TICK = 50
CHECK_CONCURRENCY = 2
#: Interval jitter, +-10 %.
INTERVAL_JITTER = 0.10
#: First tick after an upgrade marks items available but applies nothing
#: (``first_run_grace``, B9 risk 3): an org-agent burst must not auto-merge blind.
GRACE_FILE = ".first_run_done"


@dataclass
class TickReport:
    reason: str
    checked: int = 0
    available: int = 0
    applied: int = 0
    failed: int = 0
    credential: str = "ok"
    reports: list[ItemMergeReport] = field(default_factory=list)
    skipped: str | None = None


class HubSyncRunner:
    def __init__(
        self,
        *,
        config_manager: Any,
        auth_store: Any | None = None,
        auth_store_provider: Callable[[], Any | None] | None = None,
        env_config: Any | None = None,
        build_ctx: Callable[[], Any] | None = None,
        startup_delay: float | None = None,
    ) -> None:
        self._cm = config_manager
        self._auth_store = auth_store
        self._auth_store_provider = auth_store_provider
        self._env = env_config
        self._build_ctx = build_ctx
        self._startup_delay = (
            STARTUP_DELAY_S + random.uniform(0, STARTUP_JITTER_S)
            if startup_delay is None
            else startup_delay
        )
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._reported: set[tuple[str, str]] = set()
        self._last_credential: str | None = None
        self.last_tick: TickReport | None = None

    # -- lifecycle -----------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> bool:
        """Sleep, waking early on stop. True = stopped."""

        try:
            await asyncio.wait_for(self._stop.wait(), timeout=max(0.0, seconds))
            return True
        except asyncio.TimeoutError:
            return False

    async def run_forever(self) -> None:
        """Never raises. Cancellation propagates (lifespan shutdown)."""

        if await self._sleep(self._startup_delay):
            return
        reason: Literal["timer", "startup", "manual"] = "startup"
        while not self._stop.is_set():
            try:
                await self.tick(reason=reason)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the loop must survive anything
                logger.warning("hub update tick failed", exc_info=True)
            reason = "timer"
            try:
                minutes = HubSyncSettings.from_config(self._cm).interval_min
            except Exception:  # noqa: BLE001
                minutes = 60
            delay = minutes * 60 * random.uniform(1 - INTERVAL_JITTER, 1 + INTERVAL_JITTER)
            if await self._sleep(delay):
                return

    # -- context -----------------------------------------------------------------

    async def _context(self) -> svc.HubSyncContext:
        if self._build_ctx is not None:
            ctx = self._build_ctx()
            return await ctx if asyncio.iscoroutine(ctx) else ctx
        # Per tick, from the server's own AuthStore: a token refresh must persist
        # in the one place the login owns, and a login/logout is seen next tick.
        store = self._auth_store
        if store is None and self._auth_store_provider is not None:
            try:
                store = self._auth_store_provider()
            except Exception:  # noqa: BLE001 - fall back to a short-lived store
                store = None
        for_tenant, credential = await svc.build_clients(self._cm, store)
        return svc.HubSyncContext(
            config_dir=self._cm.config_dir,
            config_manager=self._cm,
            client_for_tenant=for_tenant,
            credential=credential,  # type: ignore[arg-type]
        )

    # -- one tick ---------------------------------------------------------------------

    async def tick(self, *, reason: Literal["timer", "startup", "manual"] = "timer") -> TickReport:
        async with self._lock:
            return await self._tick_locked(reason)

    async def _tick_locked(self, reason: str) -> TickReport:
        report = TickReport(reason=reason)
        ctx = await self._context()
        report.credential = ctx.credential
        if ctx.credential != self._last_credential:
            logger.debug("hub update credential: %s", ctx.credential)
            self._last_credential = ctx.credential

        settings = HubSyncSettings.from_config(self._cm)
        links = await asyncio.to_thread(list_links, ctx.agents(), ctx.teams())
        if not links:
            report.skipped = "no-linked-items"
            await asyncio.to_thread(self._record_tick, ctx, report)
            self.last_tick = report
            return report

        doc = ctx.store().load()
        picked = self._pick(links, doc)
        lease = st.RunnerLease(ctx.config_dir)
        if not await asyncio.to_thread(lease.acquire):
            report.skipped = "lease-held"
            self.last_tick = report
            return report
        try:
            checks = await self._check_bounded(ctx, picked)
            report.checked = len(checks)
            report.available = sum(1 for c in checks if c.verdict == "available")
            grace = await asyncio.to_thread(self._first_run_grace, ctx)
            for c in checks:
                if c.verdict != "available":
                    continue
                auto = settings.auto_for(c.kind) and not grace
                item = doc.get("items", {}).get(st.item_key(c.kind, c.local_id), {})
                if not auto or not st.auto_apply_due(item):
                    continue
                result = await asyncio.to_thread(svc.apply_one, ctx, c, auto=True)
                report.reports.append(result)
                if result.applied:
                    report.applied += 1
                elif result.error_class:
                    report.failed += 1
                    self._summarise_failure(c, result.error_class)
        finally:
            await asyncio.to_thread(lease.release)
        await asyncio.to_thread(self._record_tick, ctx, report)
        self.last_tick = report
        return report

    def _pick(self, links: Sequence[Link], doc: dict[str, Any]) -> set[str]:
        """At most :data:`MAX_ITEMS_PER_TICK` ids, oldest-checked first; 404'd items throttled."""

        items = doc.get("items", {})

        def last(link: Link) -> str:
            return str(
                items.get(st.item_key(link.kind, link.local_id), {}).get("last_checked_at") or ""
            )

        eligible = [
            link
            for link in links
            if st.check_due(items.get(st.item_key(link.kind, link.local_id), {}))
        ]
        eligible.sort(key=last)
        return {link.local_id for link in eligible[:MAX_ITEMS_PER_TICK]}

    async def _check_bounded(self, ctx: svc.HubSyncContext, ids: set[str]) -> list[Any]:
        """Checks run with concurrency 2, each off the loop (sync ``requests`` clients)."""

        if not ids:
            return []
        gate = asyncio.Semaphore(CHECK_CONCURRENCY)
        results: list[Any] = []

        async def one(kind: str) -> None:
            async with gate:
                results.extend(
                    await asyncio.to_thread(
                        svc.check_items, ctx, kinds=(kind,), only_ids=ids, record=False
                    )
                )

        await asyncio.gather(one("agent"), one("team"))
        await asyncio.to_thread(svc.record_checks, ctx, results)
        return results

    def _first_run_grace(self, ctx: svc.HubSyncContext) -> bool:
        """True exactly once per config root: the first tick after upgrade never auto-applies."""

        marker = ctx.config_dir / "hub" / GRACE_FILE
        if marker.exists():
            return False
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("1", encoding="utf-8")
        except OSError:
            return False
        return True

    def _summarise_failure(self, c: Any, cls: str) -> None:
        """One WARNING per item per class (B3.3), never a stream of them."""

        key = (f"{c.kind}:{c.local_id}", cls)
        if key in self._reported:
            return
        self._reported.add(key)
        logger.warning("hub update for %s %r not applied: %s", c.kind, c.name, cls)

    def _record_tick(self, ctx: svc.HubSyncContext, report: TickReport) -> None:
        def fold(doc: dict[str, Any]) -> None:
            doc["last_tick"] = {
                "at": st.now_iso(),
                "reason": report.reason,
                "checked": report.checked,
                "available": report.available,
                "applied": report.applied,
                "failed": report.failed,
                "credential": report.credential,
            }

        ctx.store().mutate(fold)

    # -- route/CLI entries ---------------------------------------------------------------

    async def check_now(
        self, *, kind: Literal["agent", "team", "all"] = "all", names: Sequence[str] | None = None
    ) -> list[Any]:
        """A user-requested check: applies nothing, ignores the schedule (B5.1)."""

        async with self._lock:
            ctx = await self._context()
            kinds = ("agent", "team") if kind == "all" else (kind,)
            return await asyncio.to_thread(svc.check_items, ctx, kinds=kinds, names=names)

    async def apply(
        self,
        kind: str | None,
        names: Sequence[str] | None = None,
        *,
        all_available: bool = False,
        prefer: str = "none",
        replace: str | None = None,
        acknowledge_unknown_baseline: bool = False,
        dry_run: bool = False,
    ) -> Any:
        async with self._lock:
            ctx = await self._context()
            return await asyncio.to_thread(
                svc.apply_items,
                ctx,
                kind=kind,
                names=names,
                all_available=all_available,
                prefer=prefer,  # type: ignore[arg-type]
                replace=replace,  # type: ignore[arg-type]
                acknowledge_unknown_baseline=acknowledge_unknown_baseline,
                dry_run=dry_run,
            )

    async def context(self) -> svc.HubSyncContext:
        return await self._context()
