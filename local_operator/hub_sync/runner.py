"""The periodic hub check/apply runner (design B3.2, B3.3).

One :class:`HubSyncRunner` per ``lop serve`` daemon, started from the server
lifespan as a single asyncio task. The desktop routes run their work through the
SAME object (:meth:`run_exclusive`), so the timer and the buttons queue on one
lock, and (through the shared service) the CLI cannot diverge about what a check
or an update is.

"AUTO ON" vs "MANUAL" BOTH KEEP CHECKING (requirement). ``hub.auto_update.*``
false means the runner still checks and records ``available`` — that is what
makes the sidebar indicator work in manual mode — and simply never merges. True
means a check that finds ``available`` additionally merges and applies iff the
outcome is a clean ``merged`` with a known baseline; anything else stays
``available`` with the reason recorded.

QUIET BY CONSTRUCTION. ``run_forever`` never raises; every tick catches per item;
nothing here logs above WARNING except a once-per-item-per-class summary. A
missing login is not a failure.

NO DOUBLE APPLY, three layers: one runner per daemon plus an ``asyncio.Lock``
(the timer and every route go through it); a cross-process, heartbeat-renewed
lease that EVERY writer takes — the tick here, and ``service.apply_items`` for the
routes, the CLI and the ``agent`` tool — so a second daemon or a CLI run waits
(or, for the timer, skips) instead of merging the same item; and the apply's own
under-lock fingerprint re-verify, which is what stays airtight for teams and a
narrow window for agents (design Q4).
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Literal, Mapping, Sequence, TypeVar

from local_operator.hub_sync import service as svc
from local_operator.hub_sync import store as st
from local_operator.hub_sync.check import Link, list_links
from local_operator.hub_sync.report import ItemMergeReport
from local_operator.hub_sync.settings import HubSyncSettings

logger = logging.getLogger(__name__)


def prov_fingerprint(check: Any) -> str:
    """The input-side fingerprint a prompt-too-long eligibility check compares."""

    from local_operator.hub_sync import provenance as prov

    return prov.fingerprint_snapshot(check.kind, check.local or {})


def _parse_sortable(ts: Any) -> str:
    """A stable, sortable string for a timestamp that may be missing."""

    return str(ts) if isinstance(ts, str) and ts else ""


T = TypeVar("T")

#: Startup delay so a daemon boot is never blocked or slowed by a network pass.
STARTUP_DELAY_S = 45.0
STARTUP_JITTER_S = 15.0
#: At most this many items per tick, oldest ``last_checked`` first (B3.3).
MAX_ITEMS_PER_TICK = 50
CHECK_CONCURRENCY = 2
#: Ceiling on a whole tick's CHECK phase. Each request already has its own socket
#: timeout (``check.REQUEST_TIMEOUT_S``); this is the belt for a hang those cannot
#: see. Timed-out read-only waiters may abandon the result, but the actual worker
#: lifetime remains tracked: a timeout cannot grant reload permission.
CHECK_PHASE_TIMEOUT_S = 300.0
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
        # A cancelled HTTP waiter or timed-out check does NOT stop a worker
        # thread. Private shielded futures remain owned until actual completion,
        # independently of the runner lock's state; reload never infers idle
        # merely from a released asyncio.Lock or an abandoned request task.
        self._reload_admission = asyncio.Event()
        self._reload_admission.set()
        self._dispatches: set[asyncio.Task[Any]] = set()
        self._offloads: set[asyncio.Future[Any]] = set()

    # -- lifecycle -----------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    def begin_reload(self) -> None:
        """Park future timer/manual dispatch; preserve the same runner/due state."""
        self._reload_admission.clear()

    def end_reload(self) -> None:
        """A refused handoff resumes the old runner, without cancelling work."""
        self._reload_admission.set()

    def reload_blocker(self) -> str | None:
        dispatches = sum(not task.done() for task in self._dispatches)
        workers = sum(not future.done() for future in self._offloads)
        if dispatches or workers or self._lock.locked():
            return f"hub work is active ({dispatches} dispatches, {workers} worker offloads)"
        return None

    async def _offload(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        worker = asyncio.get_running_loop().run_in_executor(None, partial(fn, *args, **kwargs))
        self._offloads.add(worker)

        def finished(future: asyncio.Future[Any]) -> None:
            self._offloads.discard(future)
            if not future.cancelled():
                future.exception()  # Observe an abandoned waiter's error too.

        worker.add_done_callback(finished)
        return await asyncio.shield(worker)

    async def _dispatch(self, operation: Callable[[], Any]) -> Any:
        async def serialized() -> Any:
            while True:
                await self._reload_admission.wait()
                async with self._lock:
                    # Reload may have fenced while this dispatch waited on the
                    # existing lock. Never admit it using the earlier gate read.
                    if not self._reload_admission.is_set():
                        continue
                    return await operation()

        task = asyncio.create_task(serialized())
        self._dispatches.add(task)

        def finished(dispatch: asyncio.Task[Any]) -> None:
            self._dispatches.discard(dispatch)
            if not dispatch.cancelled():
                dispatch.exception()

        task.add_done_callback(finished)
        return await asyncio.shield(task)

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
                minutes = HubSyncSettings.read_fresh(self._cm).interval_min
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
        await self._reload_admission.wait()
        return await self._dispatch(lambda: self._tick_locked(reason))

    async def _tick_locked(self, reason: str) -> TickReport:
        report = TickReport(reason=reason)
        ctx = await self._context()
        report.credential = ctx.credential
        if ctx.credential != self._last_credential:
            logger.debug("hub update credential: %s", ctx.credential)
            self._last_credential = ctx.credential

        settings = HubSyncSettings.read_fresh(self._cm)
        links = await self._offload(list_links, ctx.agents(), ctx.teams())
        if not links:
            report.skipped = "no-linked-items"
            await self._offload(self._record_tick, ctx, report)
            self.last_tick = report
            return report

        doc = ctx.store().load()
        policies = self._link_policies(links)
        picked = self._pick(links, doc, policies)
        lease = st.RunnerLease(ctx.config_dir)
        if not await self._offload(lease.acquire):
            report.skipped = "lease-held"
            self.last_tick = report
            return report
        try:
            try:
                checks = await asyncio.wait_for(
                    self._check_bounded(ctx, picked), timeout=CHECK_PHASE_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                report.skipped = "check-timeout"
                logger.warning("hub update check did not finish in %.0fs", CHECK_PHASE_TIMEOUT_S)
                self.last_tick = report
                return report
            report.checked = len(checks)
            report.available = sum(1 for c in checks if c.verdict == "available")
            grace = await self._offload(self._first_run_grace, ctx)
            policies = self._link_policies(links)
            for c in checks:
                if c.verdict != "available":
                    continue
                auto = settings.auto_for(c.kind) and not grace
                policy = policies.get(c.link_id) if c.link_id else None
                item = doc.get("items", {}).get(st.item_key(c.kind, c.local_id), {})
                # INBOUND applies when this LINK's subscribe is on. An explicit
                # per-link policy wins; a LEGACY link with no confirmed identity
                # has never been migrated, so the kind's creation default governs
                # (preserving a global OFF exactly once, as the addendum requires).
                if not auto:
                    continue
                if policy is not None and not policy.subscribe:
                    continue
                # A prompt-too-long item keeps a cheap SCHEDULED retry but only
                # re-sends when the input/config fingerprint moved; a plain
                # due-time check would spend tokens re-learning "still too long"
                # (and its fingerprint gate is unbounded, never retired).
                if item.get("error_class") == "prompt-too-long":
                    current_fp = prov_fingerprint(c)
                    if not st.prompt_too_long_eligible(item, current_fp):
                        continue
                elif not st.auto_apply_due(item):
                    continue
                # Independent inbound due time: a push-only link never blocks the
                # inbound cadence and a just-pulled item is not re-fetched.
                if not st.pull_due(item, settings.interval_min):
                    continue
                result = await self._offload(svc.apply_one, ctx, c, auto=True)
                report.reports.append(result)
                if result.applied:
                    report.applied += 1
                elif result.error_class:
                    report.failed += 1
                    self._summarise_failure(c, result.error_class)
        finally:
            await self._offload(lease.release)
        await self._offload(self._record_tick, ctx, report)
        self.last_tick = report
        return report

    def _link_policies(self, links: Sequence[Link]) -> dict[str, Any]:
        """Per-link policy for the links that carry a confirmed identity.

        Read FRESH from config so a TUI/CLI policy edit is seen next tick; explicit
        per-link values win and a link with no stored policy falls back to its kind's
        creation default. Local work only.
        """
        from local_operator.hub_sync import policies as pol

        try:
            values = self._cm.get_config().values
        except Exception:  # noqa: BLE001 - a bad read is treated as no explicit policy
            return {}
        out: dict[str, Any] = {}
        for link in links:
            if not link.link_id:
                continue
            try:
                out[link.link_id] = pol.link_policy_for(values, link.link_id, link.kind)
            except Exception:  # noqa: BLE001
                continue
        return out

    def _pick(
        self, links: Sequence[Link], doc: dict[str, Any], policies: Mapping[str, Any]
    ) -> dict[str, set[str]]:
        """Per-kind local ids, oldest-AND-retry-first for fairness, never a shared set.

        A link is selected when its per-link policy is interested in EITHER
        direction (``subscribe`` or ``auto_push``); a both-OFF link does no periodic
        work (manual pull/check still works). An item whose retry is due sorts ahead
        of one merely due on the interval, so a failing item is not starved by a
        long healthy queue. Keying by ``(kind, local_id)`` stops one kind's ids
        bleeding into the other.
        """

        items = doc.get("items", {})

        def interested(link: Link) -> bool:
            policy = policies.get(link.link_id) if link.link_id else None
            if policy is None:
                # A legacy link with no identity is enumerated for the check
                # (manual/pending compatibility); the apply gate decides the rest.
                return True
            return bool(policy.subscribe or policy.auto_push)

        def order(link: Link) -> tuple[int, str, str]:
            item = items.get(st.item_key(link.kind, link.local_id), {})
            retry = _parse_sortable(item.get("next_retry_at"))
            checked = _parse_sortable(item.get("last_checked_at"))
            return (0 if retry else 1, retry, checked)

        picked: dict[str, set[str]] = {"agent": set(), "team": set()}
        for kind in ("agent", "team"):
            eligible = [
                link
                for link in links
                if link.kind == kind
                and interested(link)
                and st.check_due(items.get(st.item_key(link.kind, link.local_id), {}))
            ]
            eligible.sort(key=order)
            picked[kind] = {link.local_id for link in eligible[:MAX_ITEMS_PER_TICK]}
        return picked

    async def _check_bounded(self, ctx: svc.HubSyncContext, ids: dict[str, set[str]]) -> list[Any]:
        """Checks run with concurrency 2, each off the loop (sync ``requests`` clients).

        Each kind is checked with ONLY its own due ids, so a shared bare-id set
        cannot select the wrong kind's rows.
        """

        if not any(ids.values()):
            return []
        gate = asyncio.Semaphore(CHECK_CONCURRENCY)
        results: list[Any] = []

        async def one(kind: str) -> None:
            async with gate:
                results.extend(
                    await self._offload(
                        svc.check_items,
                        ctx,
                        kinds=(kind,),
                        only_ids=ids.get(kind, set()),
                        record=False,
                    )
                    )
                )

        await asyncio.gather(one("agent"), one("team"))
        await self._offload(svc.record_checks, ctx, results)
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

    # -- route entries -------------------------------------------------------------------

    async def run_exclusive(self, fn: Callable[[svc.HubSyncContext], T]) -> T:
        """Run ``fn(ctx)`` in a worker thread while holding the runner's lock.

        The desktop routes use this for every check and apply, so a button press
        queues behind a running tick (and vice versa) instead of both computing
        the same merge. ``fn`` is synchronous and may raise; ``service.apply_items``
        additionally takes the cross-process lease, so this lock only orders the
        in-process callers. Nothing here swallows an exception: the caller maps it.
        """

        await self._reload_admission.wait()

        async def operation() -> T:
            ctx = await self._context()
            return await self._offload(fn, ctx)

        return await self._dispatch(operation)

    async def check_now(
        self, *, kind: Literal["agent", "team", "all"] = "all", names: Sequence[str] | None = None
    ) -> list[Any]:
        """A user-requested check: applies nothing, ignores the schedule (B5.1)."""

        kinds = ("agent", "team") if kind == "all" else (kind,)
        return await self.run_exclusive(lambda ctx: svc.check_items(ctx, kinds=kinds, names=names))

    async def context(self) -> svc.HubSyncContext:
        return await self._context()
