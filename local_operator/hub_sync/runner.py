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
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Literal, Mapping, Sequence, TypeVar

from local_operator.hub_sync import service as svc
from local_operator.hub_sync import store as st
from local_operator.hub_sync.budget import OperationBudget
from local_operator.hub_sync.check import Link, list_links
from local_operator.hub_sync.identity import LinkPolicy
from local_operator.hub_sync.outbox import Outbox
from local_operator.hub_sync.report import ItemMergeReport
from local_operator.hub_sync.settings import DEFAULT_CHECK_INTERVAL_MIN, HubSyncSettings

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
#: A desktop SESSION START may ask for one check, but not more often than this.
#:
#: The trigger is a freshness nicety, not the schedule: a conversation opening is
#: the moment the operator is most likely to look at the sidebar, so it is worth a
#: check -- but two conversations opened back to back are one moment, and a
#: session-per-tab UI must not turn every create into a network pass. Two minutes
#: is long enough to coalesce a burst of creates, short enough that opening one
#: conversation after a coffee still gets a fresh answer. Nothing here is a
#: correctness gate: a skipped trigger only means the regular tick answers instead.
SESSION_START_CHECK_THROTTLE_S = 120.0
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
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._cm = config_manager
        self._auth_store = auth_store
        self._auth_store_provider = auth_store_provider
        self._env = env_config
        self._build_ctx = build_ctx
        #: Monotonic seconds, injectable so a test can elapse the session-start
        #: throttle exactly instead of sleeping through it.
        self._clock = clock or time.monotonic
        self._startup_delay = (
            STARTUP_DELAY_S + random.uniform(0, STARTUP_JITTER_S)
            if startup_delay is None
            else startup_delay
        )
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._reported: set[tuple[str, str]] = set()
        self._last_credential: str | None = None
        # The policy values last read LIVE off disk. A torn read must never
        # fall back to the boot snapshot -- it predates any explicit OFF
        # written since -- so this is the only acceptable fallback; with none
        # the gate fails closed for every link (review R1).
        self._live_values: dict[str, Any] | None = None
        self._policies_trusted = True
        self.last_tick: TickReport | None = None
        # The clock reading of the last check this runner STARTED, by any route:
        # a timer tick, ``check_now``, or a session start. ``None`` means none has
        # started yet, so the first session start always checks. Read only by the
        # session-start throttle; the schedule itself is ``interval_min``.
        self._last_check_started: float | None = None
        # The in-flight session-start check. Held for two reasons: a second
        # session start must not queue a second check, and a reference keeps the
        # fire-and-forget task from being garbage-collected mid-flight.
        self._session_check_task: asyncio.Task[None] | None = None
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
                minutes = DEFAULT_CHECK_INTERVAL_MIN
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
        # A tick IS a check, so it starts the session-start window too: a
        # conversation opened right after a tick has nothing new to learn.
        self._last_check_started = self._clock()
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
            if not self._policies_trusted:
                # No readable file AND no last-good snapshot: no permission is
                # trustworthy, so nothing periodic may run (neither applies nor
                # pushes -- `_push_due` sits below this return).
                report.skipped = "policy-unreadable"
                return report
            for c in checks:
                # UP-TO-DATE items still carry an AGREEMENT (QN7): an L == R
                # check with a live revision is where the origin's lineage
                # becomes established and the (A, P) receipt is seeded, so those
                # route through the SAME apply entry point too. The apply writes
                # nothing but the baseline/receipt facts there.
                if c.verdict not in ("available", "up-to-date"):
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
                # A SUBSCRIBED-BUT-ABSENT SKILL IS CHECKED, NEVER INSTALLED
                # unattended. Installing is an explicit pull: writing a copy the
                # operator never asked for is the same class of surprise as
                # clobbering local edits, and the absent state is the recorded
                # fact that says there is nothing here to update yet.
                if c.kind == "skill" and c.content_state == "absent":
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
                if result.outcome != "up-to-date":
                    report.reports.append(result)
                if result.applied:
                    report.applied += 1
                elif result.error_class:
                    report.failed += 1
                    self._summarise_failure(c, result.error_class)
            # OUTBOUND: after the inbound pass, push the items whose per-link
            # auto_push is ON and whose push interval is due. Kept separate from
            # the inbound loop so a subscribe=OFF/auto_push=ON link still pushes
            # (without importing anything into local).
            await self._push_due(ctx, checks, policies, settings, grace, doc, report)
        finally:
            await self._offload(lease.release)
        await self._offload(self._record_tick, ctx, report)
        self.last_tick = report
        return report

    def _link_policies(self, links: Sequence[Link]) -> dict[str, Any]:
        """The EXPLICIT per-link policy of the links that carry a confirmed identity.

        Read FRESH off disk, not the manager's boot snapshot, so a TUI/CLI/daemon
        policy edit is seen next tick. Local work only.

        ONLY an explicitly stored policy is reported here. The kind-wide
        creation default (``hub.auto_update.<kind>``) is a WRITE switch: it
        decides whether an update is APPLIED, not whether the hub is read at
        all, and the tick's apply gate consults it through ``settings``. A link
        with no stored policy therefore stays in the check cadence exactly as a
        legacy link does -- "manual mode keeps checking and records what is
        available without writing it" -- while an explicitly stored both-OFF
        policy still does no periodic work. Reading the creation default as a
        subscribe default would let a WRITE switch stop the READ, and a pull
        that records an origin (which makes every pulled row an identified
        link) would then silently stop the runner from checking it.
        """
        from local_operator.hub_sync import policies as pol

        # FRESH read, not the manager's boot snapshot: a long-lived
        # ConfigManager holds the config it read at startup, so an explicit
        # policy written by the CLI/TUI/another daemon would never reach this
        # gate (observed on the real pass: subscribe=false persisted minutes
        # before a natural tick, yet the link was still picked and applied).
        # read_config_values is the same live seam HubSyncSettings.read_fresh
        # uses. None means a torn read right now -- handled below: NEVER fall
        # back to the manager's boot snapshot, whose links section predates
        # any explicit OFF written since.
        values = None
        config_dir = getattr(self._cm, "config_dir", None)
        if config_dir is not None:
            try:
                from local_operator.config import read_config_values

                values = read_config_values(config_dir)
            except Exception:  # noqa: BLE001 - never lose the gate over a config read
                values = None
        if values is None:
            # An unreadable file is NOT permission to fall back to boot
            # defaults: creation defaults would silently re-enable a link the
            # user had explicitly switched off (review R1: a torn read after an
            # observed OFF ticked checked/applied=1 1 and landed the upstream
            # text). The last LIVE snapshot is the only trustworthy fallback;
            # with no snapshot at all, fail closed for every link.
            values = self._live_values
            if values is None:
                self._policies_trusted = False
                logger.warning("hub policy file unreadable and no live snapshot; failing closed")
                return {
                    link.link_id: LinkPolicy(subscribe=False, auto_push=False)
                    for link in links
                    if link.link_id
                }
            logger.warning("hub policy file unreadable; using the last live snapshot")
        else:
            self._live_values = values
        self._policies_trusted = True
        out: dict[str, Any] = {}
        for link in links:
            if not link.link_id:
                continue
            try:
                stored = pol.stored_link_policy(values, link.link_id)
            except Exception:  # noqa: BLE001
                continue
            if stored is not None:
                out[link.link_id] = stored
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

        picked: dict[str, set[str]] = {"agent": set(), "team": set(), "skill": set()}
        for kind in ("agent", "team", "skill"):
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

        await asyncio.gather(one("agent"), one("team"), one("skill"))
        await self._offload(svc.record_checks, ctx, results)
        return results

    async def _push_due(
        self,
        ctx: svc.HubSyncContext,
        checks: Sequence[Any],
        policies: Mapping[str, Any],
        settings: HubSyncSettings,
        grace: bool,
        doc: dict[str, Any],
        report: TickReport,
    ) -> None:
        """Push authored changes for links whose auto_push is ON and push is due.

        ELIGIBILITY IS RECOMPUTED HERE, from the CURRENT baseline/config/status,
        through the SAME service rule the inventory and the policy PATCH use -- a
        hand-edited ``auto_push: true``, a revoked attestation or a deleted
        public egress grant is a NAMED persisted skip, not a wire attempt. A push
        also needs an OUTBOUND RECEIPT (A/P); without one it is a first
        publication, which is a deliberate action and never unattended. The whole
        phase is best-effort: a transport error is recorded on the item and never
        crashes the tick.
        """
        if grace:
            return
        from local_operator.hub_sync.provenance import read_baseline

        for c in checks:
            policy = policies.get(c.link_id) if c.link_id else None
            if policy is None or not policy.auto_push:
                continue
            if c.verdict == "unavailable":
                # A check whose FETCH failed knows nothing about the remote; a
                # push built on it would be a guess (and, before this guard, a
                # multi-tick source outage burned the dead-letter cap on a
                # healthy link -- round-2 R12b). Skip with the check's own named
                # reason; nothing is attempted and nothing is settled.
                await self._offload(self._mark_push_skipped, ctx, c, str(c.reason or "unavailable"))
                continue
            item = doc.get("items", {}).get(st.item_key(c.kind, c.local_id), {})
            if not st.push_due(item, settings.interval_min) or not st.auto_apply_due(item):
                continue
            # Every blocking worker we add goes through ``_offload``: the
            # reload admission counts shield-owned futures, and a raw
            # ``to_thread`` future would leave a cancelled tick's reader/writer
            # invisible to ``reload_blocker`` (the update lane's no-kill gate).
            record = await self._offload(read_baseline, ctx.config_dir, c.kind, c.local_id)
            eligibility = await self._offload(svc.push_eligibility, ctx, record)
            if not eligibility.get("can_auto_push"):
                await self._offload(
                    self._mark_push_skipped,
                    ctx,
                    c,
                    str(eligibility.get("reason") or "unsupported-precondition"),
                )
                continue
            transporter, refused = await self._offload(self._push_transport, ctx, record)
            if transporter is None:
                await self._offload(self._mark_push_skipped, ctx, c, refused or "no-credential")
                continue
            try:
                await self._push_one(ctx, c, record, transporter, report)
            except Exception:  # noqa: BLE001 - a push must never kill the tick
                logger.debug("hub push for %s %s failed", c.kind, c.local_id, exc_info=True)

    def _push_transport(
        self, ctx: svc.HubSyncContext, record: Any
    ) -> tuple[Any | None, str | None]:
        """The ONE per-link transport builder (see ``service.push_transport``)."""

        return svc.push_transport(ctx, record)

    async def _push_one(
        self,
        ctx: svc.HubSyncContext,
        c: Any,
        record: Any,
        transporter: Any,
        report: TickReport,
    ) -> None:
        """Project and push one item; record the outcome (never raises to the tick)."""

        from local_operator.hub_sync.push import push_item

        if record is None or record.outbound is None:
            # No outbound receipt: first publication is a deliberate action, not
            # an unattended one -- and the skip is PERSISTED so the surface can
            # explain why an enabled auto-push is not publishing.
            self._mark_push_skipped(ctx, c, "no-outbound-receipt")
            return
        # The revision the merge derives from: the CHECK-time remote revision
        # when the fetch exposed one, else the last ACCEPTED receipt revision
        # (the value the server itself minted at the previous successful push),
        # NEVER the stale inbound baseline -- that stale If-Match produced a
        # forever-412 after the first successful push (round-2 QN2b). The
        # transport refuses an EMPTY one; there is no unconditional-write path.
        outbound = record.outbound
        assert outbound is not None  # narrowed once; the calls below re-widen the attribute
        expected = c.remote_revision or outbound.remote_revision
        budget = OperationBudget.start("automatic", self._operation_id(c), 1)
        # A bundle push is gated on the SAME per-source attestation the inventory
        # reports: no advertised ``skill_bundles`` means the manifest would land
        # with no blob behind it, so the refusal happens before the commit.
        bundle_attested = record.identity is not None and svc._capability_bundle(
            ctx.store().load(), record.identity
        )
        # A bundle republish must store the archive BEFORE the commit: the
        # server opens the blob for the manifest's new hash. Documents have no
        # blob, so the hook is built only for the bundle form.
        bundle_upload = (
            svc.bundle_upload_hook(ctx, record.identity, record)
            if record.identity is not None and "bundle" in (record.fields or {})
            else None
        )
        outcome = await push_item(
            kind=c.kind,
            local_id=c.local_id,
            captured_local=outbound.local_fields,
            accepted=outbound.accepted_fields,
            current_local=c.local or {},
            remote_latest=c.remote or {},
            expected_revision=expected,
            transport=transporter,
            outbox=Outbox(ctx.config_dir),
            operation_id=self._operation_id(c),
            budget=budget,
            document_builder=self._document_builder(ctx, c),
            bundle_attested=bundle_attested,
            bundle_upload=bundle_upload,
        )
        report.skipped = None
        await self._offload(self._record_push, ctx, c, outcome)

    def _document_builder(self, ctx: svc.HubSyncContext, c: Any) -> Any:
        """The per-kind wire-document builder the push closes over (one per attempt)."""

        def build(fields: Mapping[str, Any]) -> Mapping[str, Any]:
            from local_operator.hub_sync.adapters import publish_document

            return publish_document(c.kind, ctx.config_dir, c.local_id, fields)

        return build

    @staticmethod
    def _operation_id(c: Any) -> str:
        return f"push-{c.kind}-{c.local_id}"

    def _mark_push_skipped(self, ctx: svc.HubSyncContext, c: Any, reason: str) -> None:
        """Persist WHY this link's push did not run, on the item (survives restarts)."""

        key = st.item_key(c.kind, c.local_id)

        def fold(doc: dict[str, Any]) -> None:
            item = doc.get("items", {}).get(key)
            if item is not None:
                st.mark_push_skipped(item, reason)

        ctx.store().mutate(fold)

    def _record_push(self, ctx: svc.HubSyncContext, c: Any, outcome: Any) -> None:
        """Persist the outcome through the ONE shared recorder (receipt + skip)."""

        svc.record_push_outcome(ctx, c, outcome)

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
        self,
        *,
        kind: Literal["agent", "team", "skill", "all"] = "all",
        names: Sequence[str] | None = None,
    ) -> list[Any]:
        """A user-requested check: applies nothing, ignores the schedule (B5.1).

        Also what the session-start trigger runs (``note_session_start``): a check
        is a check, and neither caller may merge.
        """

        kinds = ("agent", "team", "skill") if kind == "all" else (kind,)
        self._last_check_started = self._clock()
        return await self.run_exclusive(lambda ctx: svc.check_items(ctx, kinds=kinds, names=names))

    # -- session start -------------------------------------------------------------------

    def note_session_start(self) -> asyncio.Task[None] | None:
        """Ask for ONE background hub check because a desktop session just began.

        FIRE AND FORGET, AND THAT IS THE CONTRACT. The call schedules and returns;
        it never awaits the network, so a create whose hub is unreachable is
        never slowed by it, and ``_session_check_once`` swallows every failure so
        it can never surface as the create's own error. The returned task exists
        only for a caller or a test that WANTS to await it; the route ignores it.

        THROTTLED IN TWO DIMENSIONS, both checked before the task is created:

        * one check in flight at a time -- a second session start while one is
          running is coalesced into it rather than queued behind it;
        * ``SESSION_START_CHECK_THROTTLE_S`` since the last check STARTED by any
          route (timer tick, ``check_now``, or an earlier session start). The
          slot is claimed HERE, synchronously, before the check runs: claiming it
          on completion instead would let two creates in the same event-loop
          burst both see a stale stamp and both check.

        Nothing is scheduled after ``stop()`` (the daemon is going away) or with
        no running loop to own the task. ``None`` therefore means "nothing was
        scheduled", never "the check failed".

        The check deliberately runs ``check_now`` and not ``tick``: a session
        start must not merge anything. Auto-apply belongs to the scheduled tick,
        where ``first_run_grace`` and the per-link policy gates are evaluated;
        borrowing a user action to write files would be a second, quieter apply
        path. Recording the check does NOT delay the tick's apply either: the
        due-time rules key on ``last_pulled_at`` and ``next_retry_at``, not on
        ``last_checked_at`` (``store.check_due`` / ``pull_due`` / ``auto_apply_due``).
        """

        if self._stop.is_set():
            return None
        pending = self._session_check_task
        if pending is not None and not pending.done():
            return None
        now = self._clock()
        last = self._last_check_started
        if last is not None and now - last < SESSION_START_CHECK_THROTTLE_S:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no loop to own the task: nothing to schedule on
            return None
        self._last_check_started = now
        task = loop.create_task(self._session_check_once())
        self._session_check_task = task
        return task

    async def _session_check_once(self) -> None:
        """The session-start check's whole body: never raises but cancellation."""

        try:
            await self.check_now()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a session start never fails on the hub
            logger.debug("hub check for a session start failed", exc_info=True)

    async def context(self) -> svc.HubSyncContext:
        return await self._context()
