"""The scheduler: ticks, state machine, ceilings, and loss semantics (§5, §7, §11).

Everything runs against fakes — a controllable clock, a scripted check
runner, a recording deliver — so the tests assert the SCHEDULER's behaviour
rather than a session's. The session-level wiring is
``tests/unit/monitors/test_session_index.py`` and
``tests/unit/monitors/test_integration.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from local_operator.monitors import state as monitor_state
from local_operator.monitors.scheduler import CheckOutcome, MonitorScheduler
from local_operator.monitors.settings import MonitorSettings
from local_operator.monitors.spec import MonitorSpec
from tests.unit.monitors.support import DRAIN_BACKSTOP_S, drain_checks, wait_set

NOW = 1_756_000_000_000


def spec(
    monitor_id: str = "m1",
    *,
    name: str = "watch",
    tool: str = "bash",
    arguments: Mapping[str, Any] | None = None,
    every_ms: int = 60_000,
    until_at: int | None = None,
    created_at: int = NOW,
) -> MonitorSpec:
    return MonitorSpec(
        id=monitor_id,
        name=name,
        tool=tool,
        arguments=dict(arguments or {"command": "date -u"}),
        every_ms=every_ms,
        until_at=until_at,
        created_at=created_at,
    )


async def arm(scheduler: Any, command: str = "date", **extra: Any) -> dict[str, Any]:
    """Create one bash monitor through the real arm flow."""
    request: dict[str, Any] = {"tool": "bash", "arguments": {"command": command}}
    request.update(extra)
    return await scheduler.create(request, cwd="/w")


class Harness:
    def __init__(
        self,
        tmp_path: Any,
        *,
        settings: MonitorSettings | None = None,
        validate: Any = None,
        uniform: Any = None,
        index_writable: Any = None,
    ) -> None:
        self.now_ms = NOW
        self.results: list[dict[str, Any]] = []
        self.calls: list[MonitorSpec] = []
        self.deliveries: list[Any] = []
        # The lifecycle-notice sink (§D4): a notice is not a delivery, and the
        # two are recorded separately so a test can assert exactly that.
        self.notices: list[Any] = []
        self.persisted: list[list[dict[str, Any]]] = []
        self.changes = 0
        self.config_dir = tmp_path / "cfg"
        self.scheduler = MonitorScheduler(
            now=lambda: self.now_ms,
            config_dir=self.config_dir,
            session_id="sess",
            settings=settings or MonitorSettings(),
            validate=validate or (lambda tool, args: None),
            run_check=self._run_check,
            deliver=self._deliver,
            persist=self._persist,
            on_change=self._on_change,
            index_writable=index_writable,
            announce=self._announce,
            uniform=uniform or (lambda low, high: low),
        )

    def _run_check(self, monitor: MonitorSpec) -> Any:
        async def check() -> dict[str, Any]:
            self.calls.append(monitor)
            if self.results:
                return self.results.pop(0)
            return {"text": "same", "error": None}

        return check()

    async def _deliver(self, delivery: Any) -> None:
        self.deliveries.append(delivery)

    async def _announce(self, notice: Any) -> None:
        self.notices.append(notice)

    async def _persist(self, monitors: list[MonitorSpec]) -> None:
        self.persisted.append([monitor.model_dump() for monitor in monitors])

    def _on_change(self) -> None:
        self.changes += 1

    async def settle(self) -> None:
        """Await every in-flight check to completion (see ``support.drain_checks``).

        Not a loop-turn count: a delivery sink may hop to a thread, and how many
        turns that takes is the host's business, not the scheduler's.
        """
        await drain_checks(self.scheduler)

    async def pump_ripen(self, *, advance: int = 10**9) -> None:
        """Advance the clock past every due time, pump, and settle."""
        self.now_ms += advance
        await self.scheduler.pump()
        await self.settle()

    def counters(self, monitor_id: str = "m1") -> dict[str, Any]:
        found = monitor_state.read_counters(self.config_dir, "sess", monitor_id)
        assert found is not None, f"no counters for {monitor_id}"
        return found


async def rill(harness: Harness, advance_ms: int) -> None:
    """Advance the clock by EXACTLY ``advance_ms`` and run the due tick.

    ``Harness.pump_ripen`` deliberately jumps ~11.6 days, which is the right
    move for "any due time has passed" and the wrong one for anything measured
    in minutes: an unavailable episode's 30-minute stall notice and its
    24-hour disable both fall inside a single such jump, so those tests step
    the clock themselves. Kept beside the harness so the next test that
    measures an episode does not have to rediscover this.
    """
    harness.now_ms += advance_ms
    await harness.scheduler.pump()
    await harness.settle()


@pytest.fixture
def harness(tmp_path: Any) -> Iterator[Harness]:
    created = Harness(tmp_path)
    yield created
    created.scheduler.dispose()


# ---------------------------------------------------------------------------
# Arm, first check, jitter
# ---------------------------------------------------------------------------


def test_first_check_lands_in_the_first_check_window(harness: Harness) -> None:
    harness.scheduler.load([spec()])
    assert harness.scheduler.next_monitor_due_at() == NOW + 1_000  # uniform picks low
    assert harness.scheduler.index_rows()[0]["next_due_at"] == NOW + 1_000
    harness.scheduler.dispose()

    high = Harness(harness.config_dir.parent / "hi", uniform=lambda low, high: high)
    high.scheduler.load([spec()])
    assert high.scheduler.next_monitor_due_at() == NOW + 3_000
    high.scheduler.dispose()


@pytest.mark.asyncio
async def test_the_create_outcome_carries_the_first_check_instant(harness: Harness) -> None:
    """§4.5: the id and the instant ride the create receipt (F4)."""
    outcome = await harness.scheduler.create(
        {"tool": "bash", "arguments": {"command": "date -u"}}, cwd="/w"
    )
    assert outcome.get("created") is True
    assert outcome["next_due_at"] == NOW + 1_000  # uniform picks low


@pytest.mark.asyncio
async def test_jitter_is_positive_and_capped(harness: Harness) -> None:
    harness.scheduler.load([spec(every_ms=30_000)])  # cap = min(5s, 3s) = 3s
    harness.results.append({"text": "A", "error": None})
    await harness.pump_ripen()
    assert harness.scheduler.next_monitor_due_at() == harness.now_ms + 30_000  # zero jitter picked
    assert harness.persisted == []  # no persist on a quiet tick

    high = Harness(harness.config_dir.parent / "hi2", uniform=lambda low, high: high)
    high.scheduler.load([spec(every_ms=60_000)])  # cap = min(5s, 6s) = 5s
    high.results.append({"text": "A", "error": None})
    await high.pump_ripen()
    assert high.scheduler.next_monitor_due_at() == high.now_ms + 65_000
    high.scheduler.dispose()


# ---------------------------------------------------------------------------
# The tick loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_check_establishes_the_baseline_silently(harness: Harness) -> None:
    harness.scheduler.load([spec()])
    harness.results.append({"text": "line A\n", "error": None})
    await harness.pump_ripen()
    assert harness.deliveries == []
    counters = harness.counters()
    assert counters["checks"] == 1
    assert counters["last_note"] == "baseline captured"
    assert counters["content_hash"].startswith("sha256:")
    blob = monitor_state.read_snapshot(harness.config_dir, "sess", "m1")
    assert blob is not None and blob["snapshot"] == "line A"


@pytest.mark.asyncio
async def test_a_quiet_tick_delivers_nothing_and_stays_off_the_index(harness: Harness) -> None:
    harness.scheduler.load([spec()])
    harness.results.extend([{"text": "A", "error": None}, {"text": "A", "error": None}])
    await harness.pump_ripen()
    changes_before = harness.changes
    await harness.pump_ripen()
    assert harness.deliveries == []
    assert harness.counters()["checks"] == 2
    assert harness.changes == changes_before  # no index rewrite on a quiet tick


@pytest.mark.asyncio
async def test_a_change_delivers_a_bounded_delta(harness: Harness) -> None:
    harness.scheduler.load([spec()])
    harness.results.extend([{"text": "A\nB", "error": None}, {"text": "A\nB\nC", "error": None}])
    await harness.pump_ripen()
    changes_before = harness.changes
    await harness.pump_ripen()
    assert len(harness.deliveries) == 1
    delivery = harness.deliveries[0]
    assert delivery.monitor_id == "m1"
    assert delivery.changes == 1
    assert "+ C" in delivery.delta_text
    counters = harness.counters()
    assert counters["deliveries"] == 1 and counters["last_change_at"] > 0
    assert harness.changes > changes_before  # a delivery refreshes the index


@pytest.mark.asyncio
async def test_unchanged_content_never_re_surfaces(harness: Harness) -> None:
    harness.scheduler.load([spec()])
    harness.results.extend(
        [
            {"text": "A", "error": None},
            {"text": "A\nB", "error": None},
            {"text": "A\nB", "error": None},
            {"text": "A\nB", "error": None},
        ]
    )
    await harness.pump_ripen()
    await harness.pump_ripen()
    assert len(harness.deliveries) == 1
    await harness.pump_ripen()
    await harness.pump_ripen()
    assert len(harness.deliveries) == 1


@pytest.mark.asyncio
async def test_a_resume_counts_the_skipped_checks_once(harness: Harness) -> None:
    harness.scheduler.load([spec(every_ms=30_000)])
    harness.results.append({"text": "A", "error": None})
    await harness.pump_ripen()
    # Down for four intervals; the resume check supersedes the last one.
    harness.now_ms += 4 * 30_000
    harness.results.append({"text": "A\nB", "error": None})
    await harness.scheduler.pump()
    await harness.settle()
    assert len(harness.deliveries) == 1
    assert harness.deliveries[0].skipped == 3


@pytest.mark.asyncio
async def test_a_jitless_resume_that_is_merely_late_reports_zero_skips(harness: Harness) -> None:
    harness.scheduler.load([spec(every_ms=30_000)])
    harness.results.append({"text": "A", "error": None})
    await harness.pump_ripen()
    harness.now_ms += 30_000 + 500  # one interval plus a moment
    harness.results.append({"text": "B", "error": None})
    await harness.scheduler.pump()
    await harness.settle()
    assert harness.deliveries[0].skipped == 0


# ---------------------------------------------------------------------------
# Overlap and the concurrency ceiling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_due_monitor_with_a_check_in_flight_is_skipped_not_overlapped(
    tmp_path: Any,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    clock = [NOW]

    class Slow:
        def __init__(self) -> None:
            self.calls = 0
            self.deliveries: list[Any] = []

        async def run(self, monitor: MonitorSpec) -> CheckOutcome:
            self.calls += 1
            if self.calls == 1:
                return {"text": "A", "error": None}  # fast baseline
            started.set()
            await release.wait()
            return {"text": "B", "error": None}

        async def deliver(self, delivery: Any) -> None:
            self.deliveries.append(delivery)

    slow = Slow()
    scheduler = MonitorScheduler(
        now=lambda: clock[0],
        config_dir=tmp_path / "cfg",
        session_id="sess",
        settings=MonitorSettings(),
        validate=lambda tool, args: None,
        run_check=slow.run,
        deliver=slow.deliver,
        persist=lambda monitors: None,
        uniform=lambda low, high: low,
    )
    try:
        scheduler.load([spec(every_ms=30_000)])
        clock[0] += 5_000
        await scheduler.pump()
        await drain_checks(scheduler)
        assert slow.calls == 1  # baseline captured, no delivery

        clock[0] += 60_000
        await scheduler.pump()  # starts the slow second check
        await wait_set(started, "the second check to start")  # parked on ``release``
        await scheduler.pump()  # third pump while in flight: overlap
        counters = monitor_state.read_counters(tmp_path / "cfg", "sess", "m1")
        assert counters is not None and counters["skipped_overlap"] == 1
        release.set()
        await drain_checks(scheduler)
        assert slow.calls == 2  # no overlapping execution, ever
        assert len(slow.deliveries) == 1
        assert slow.deliveries[0].delta_text.count("+ B") == 1
    finally:
        scheduler.dispose()


@pytest.mark.asyncio
async def test_the_semaphore_bounds_simultaneous_checks_at_two(tmp_path: Any) -> None:
    running = 0
    peak = 0
    release = asyncio.Event()
    two_running = asyncio.Event()
    clock = [NOW]

    async def run(monitor: MonitorSpec) -> CheckOutcome:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        if running == 2:
            two_running.set()
        await release.wait()
        running -= 1
        return {"text": monitor.id, "error": None}

    scheduler = MonitorScheduler(
        now=lambda: clock[0],
        config_dir=tmp_path / "cfg",
        session_id="sess",
        settings=MonitorSettings(),
        validate=lambda tool, args: None,
        run_check=run,
        deliver=lambda delivery: None,
        persist=lambda monitors: None,
        uniform=lambda low, high: low,
    )
    try:
        scheduler.load([spec(f"m{i}") for i in (1, 2, 3)])
        clock[0] += 5_000
        await scheduler.pump()
        await wait_set(two_running, "two checks to be running at once")
        release.set()
        await drain_checks(scheduler)
        # Asserted AFTER the drain, not before the release: all three checks
        # have now run, so a ceiling that failed to hold would show as 3 here,
        # where a mid-flight look could only prove the third had not started YET.
        assert peak == 2  # the third waited for a slot
        # The deferred check is due the moment a slot frees; a later pump runs it.
        await scheduler.pump()
        await drain_checks(scheduler)
        counters = monitor_state.read_counters(tmp_path / "cfg", "sess", "m3")
        assert counters is not None and counters["checks"] == 1
        assert counters["skipped_overlap"] == 0  # deferred is not skipped
    finally:
        scheduler.dispose()


# ---------------------------------------------------------------------------
# The failure ladder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failures_back_off_exponentially_and_cap(tmp_path: Any) -> None:
    settings = dataclasses.replace(MonitorSettings(), max_consecutive_failures=99)
    harness = Harness(tmp_path, settings=settings)
    try:
        harness.scheduler.load([spec(every_ms=30_000)])
        expected = [30_000, 60_000, 120_000, 240_000, 480_000, 900_000, 900_000]
        for delay in expected:
            harness.results.append({"error": "boom", "text": None})
            await harness.pump_ripen()
            counters = harness.counters()
            assert counters["next_due_at"] == harness.now_ms + delay, counters
            assert counters["consecutive_failures"] >= 1
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_the_ladder_disables_and_names_the_reason(tmp_path: Any) -> None:
    settings = dataclasses.replace(MonitorSettings(), max_consecutive_failures=3)
    harness = Harness(tmp_path, settings=settings)
    try:
        harness.scheduler.load([spec()])
        for _ in range(3):
            harness.results.append({"error": "check timed out after 120s", "text": None})
            await harness.pump_ripen()
        counters = harness.counters()
        assert counters["disabled"] is True
        assert counters["disabled_reason"] == "check timed out after 120s"
        # No next check is due for a disabled monitor (§10.3): the counters
        # state it rather than keeping a stale instant (QA round-1 obs. 2).
        assert counters["next_due_at"] is None
        assert harness.scheduler.next_monitor_due_at() is None
        row = harness.scheduler.index_rows()[0]
        assert row["disabled"] is True and row["disabled_reason"]
        # A disabled monitor never ticks again...
        harness.results.append({"text": "later", "error": None})
        await harness.pump_ripen()
        assert len(harness.calls) == 3
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_reactivating_resets_failures_and_keeps_the_snapshot(tmp_path: Any) -> None:
    settings = dataclasses.replace(MonitorSettings(), max_consecutive_failures=2)
    harness = Harness(tmp_path, settings=settings)
    try:
        harness.scheduler.load([spec()])
        harness.results.append({"text": "A", "error": None})
        await harness.pump_ripen()  # baseline
        for _ in range(2):
            harness.results.append({"error": "boom", "text": None})
            await harness.pump_ripen()
        assert harness.counters()["disabled"] is True

        outcome = await harness.scheduler.create(
            {"tool": "bash", "arguments": {"command": "date -u"}, "every": "60s"}, cwd="/w"
        )
        assert outcome.get("reactivated") is True
        assert outcome["next_due_at"] == harness.now_ms + 1_000  # fresh counters, first check
        counters = harness.counters()
        assert counters["disabled"] is False and counters["consecutive_failures"] == 0
        # The blob survived, so the next check diffs against the old baseline.
        assert monitor_state.read_snapshot(harness.config_dir, "sess", "m1") is not None
    finally:
        harness.scheduler.dispose()


# ---------------------------------------------------------------------------
# Create/cancel ceilings (§11.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_identical_spec_is_a_duplicate_not_an_error(harness: Harness) -> None:
    first = await arm(harness.scheduler)
    assert first.get("created") is True
    again = await arm(harness.scheduler)
    assert again.get("duplicate") is True
    assert again["spec"].id == first["spec"].id


@pytest.mark.asyncio
async def test_the_cap_refuses_with_the_limit_named(tmp_path: Any) -> None:
    settings = dataclasses.replace(MonitorSettings(), max_monitors=2)
    harness = Harness(tmp_path, settings=settings)
    try:
        for i in (1, 2):
            outcome = await harness.scheduler.create(
                {"tool": "bash", "arguments": {"command": f"date -{i}"}}, cwd="/w"
            )
            assert outcome.get("created") is True
        refused = await harness.scheduler.create(
            {"tool": "bash", "arguments": {"command": "date -3"}}, cwd="/w"
        )
        assert "monitor limit reached (2 per session)" in refused["error"]
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_a_third_monitor_with_one_name_is_a_storm(harness: Harness) -> None:
    for i in (1, 2):
        outcome = await harness.scheduler.create(
            {"tool": "bash", "arguments": {"command": f"date -{i}"}, "name": "same"}, cwd="/w"
        )
        assert outcome.get("created") is True
    refused = await harness.scheduler.create(
        {"tool": "bash", "arguments": {"command": "date -3"}, "name": "same"}, cwd="/w"
    )
    assert "storm" in refused["error"]


@pytest.mark.asyncio
async def test_ids_are_never_reused_and_the_high_water_survives_a_load(tmp_path: Any) -> None:
    harness = Harness(tmp_path)
    try:
        first = await arm(harness.scheduler, "a")
        assert first["spec"].id == "m1"
        await harness.scheduler.cancel("m1")
        second = await arm(harness.scheduler, "b")
        assert second["spec"].id == "m2"
        assert harness.scheduler.next_seq == 3
    finally:
        harness.scheduler.dispose()

    # A load with the persisted mark cannot lower it (the transcript carries it).
    reopened = Harness(tmp_path)
    try:
        reopened.scheduler.load([], next_seq=7)
        third = await arm(reopened.scheduler, "c")
        assert third["spec"].id == "m7"
    finally:
        reopened.scheduler.dispose()


@pytest.mark.asyncio
async def test_an_unwritable_index_refuses_the_arm(harness: Harness) -> None:
    harness.scheduler = MonitorScheduler(
        now=lambda: harness.now_ms,
        config_dir=harness.config_dir,
        session_id="sess",
        settings=MonitorSettings(),
        validate=lambda tool, args: None,
        run_check=harness._run_check,
        deliver=harness._deliver,
        persist=harness._persist,
        index_writable=lambda: False,
        uniform=lambda low, high: low,
    )
    outcome = await arm(harness.scheduler, "a")
    assert "index cannot be written" in outcome["error"]
    assert harness.scheduler.monitors == ()


# ---------------------------------------------------------------------------
# Rate cap (§9.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_hourly_cap_holds_changes_and_names_them_next_time(tmp_path: Any) -> None:
    settings = dataclasses.replace(MonitorSettings(), max_deliveries_per_hour=1)
    harness = Harness(tmp_path, settings=settings)
    try:
        harness.scheduler.load([spec()])
        harness.results.extend(
            [
                {"text": "A", "error": None},
                {"text": "A\n1", "error": None},  # delivered
                {"text": "A\n1\n2", "error": None},  # held
                {"text": "A\n1\n2\n3", "error": None},  # held
            ]
        )
        for _ in range(4):
            # Small advances: every round is past the 60s due but stays inside
            # the one-hour delivery window the cap counts in.
            await harness.pump_ripen(advance=90_000)
        assert len(harness.deliveries) == 1
        counters = harness.counters()
        assert counters["suppressed"]["rate_cap"] == 2
        assert counters["rate_cap_held"] == 2

        # Past the window: the next change arrives and names the held two.
        harness.now_ms += 3_600_000 + 1
        harness.results.append({"text": "A\n1\n2\n3\n4", "error": None})
        await harness.pump_ripen()
        assert len(harness.deliveries) == 2
        held = harness.deliveries[1]
        assert held.held_by_cap == 2
        assert harness.counters()["rate_cap_held"] == 0
    finally:
        harness.scheduler.dispose()


# ---------------------------------------------------------------------------
# Loss semantics (§7.2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_torn_hash_is_re_adopted_quietly(harness: Harness) -> None:
    harness.scheduler.load([spec()])
    harness.results.append({"text": "A", "error": None})
    await harness.pump_ripen()  # baseline "A"
    # Rewrite the counters hash to a wrong value, leaving the blob intact.
    counters = harness.counters()
    counters["content_hash"] = "sha256:deadbeef"
    monitor_state.write_counters(harness.config_dir, "sess", "m1", counters)
    harness.results.append({"text": "A", "error": None})
    await harness.pump_ripen()
    assert harness.deliveries == []
    assert harness.counters()["content_hash"].startswith("sha256:")
    assert harness.counters()["content_hash"] != "sha256:deadbeef"


@pytest.mark.asyncio
async def test_a_lost_blob_heals_on_the_next_quiet_tick(harness: Harness) -> None:
    harness.scheduler.load([spec()])
    harness.results.append({"text": "A", "error": None})
    await harness.pump_ripen()
    monitor_state.snapshot_path(harness.config_dir, "sess", "m1").unlink()
    harness.results.append({"text": "A", "error": None})
    await harness.pump_ripen()
    assert harness.deliveries == []
    blob = monitor_state.read_snapshot(harness.config_dir, "sess", "m1")
    assert blob is not None and blob["snapshot"] == "A"


@pytest.mark.asyncio
async def test_a_lost_blob_with_changed_content_is_honestly_beyond_the_window(
    harness: Harness,
) -> None:
    harness.scheduler.load([spec()])
    harness.results.append({"text": "A", "error": None})
    await harness.pump_ripen()
    monitor_state.snapshot_path(harness.config_dir, "sess", "m1").unlink()
    harness.results.append({"text": "B", "error": None})
    await harness.pump_ripen()
    assert len(harness.deliveries) == 1
    assert "beyond the stored snapshot window" in harness.deliveries[0].delta_text


@pytest.mark.asyncio
async def test_both_files_lost_re_establishes_the_baseline_silently(harness: Harness) -> None:
    harness.scheduler.load([spec()])
    harness.results.append({"text": "A", "error": None})
    await harness.pump_ripen()
    monitor_state.remove_monitor_state(harness.config_dir, "sess", "m1")
    harness.results.append({"text": "A", "error": None})
    await harness.pump_ripen()
    assert harness.deliveries == []
    # Nothing on disk remembers a check ran: this is a fresh capture, and the
    # note says so rather than pretending otherwise.
    assert harness.counters()["last_note"] == "baseline captured"


@pytest.mark.asyncio
async def test_a_hash_and_blob_lost_under_live_counters_is_re_established(tmp_path: Any) -> None:
    harness = Harness(tmp_path)
    reopened: Harness | None = None
    try:
        harness.scheduler.load([spec()])
        harness.results.append({"text": "A", "error": None})
        await harness.pump_ripen()  # baseline
        counters = harness.counters()
        counters["content_hash"] = ""
        monitor_state.write_counters(harness.config_dir, "sess", "m1", counters)
        monitor_state.snapshot_path(harness.config_dir, "sess", "m1").unlink()

        # Loss is a PROCESS fact: a fresh scheduler re-reads the blanked
        # counters, which is the state a restart would see.
        reopened = Harness(tmp_path)
        reopened.now_ms = harness.now_ms + 10**9
        reopened.scheduler.load([spec()])
        reopened.now_ms += 5_000  # past the first-check window
        reopened.results.append({"text": "A", "error": None})
        await reopened.scheduler.pump()
        await reopened.settle()
        assert reopened.deliveries == []
        assert reopened.counters()["last_note"] == "baseline re-established"
    finally:
        if reopened is not None:
            reopened.scheduler.dispose()
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_a_lost_counters_file_rebuilds_the_hash_from_the_blob(tmp_path: Any) -> None:
    harness = Harness(tmp_path)
    reopened: Harness | None = None
    try:
        harness.scheduler.load([spec()])
        harness.results.append({"text": "A", "error": None})
        await harness.pump_ripen()
        monitor_state.counters_path(harness.config_dir, "sess", "m1").unlink()

        reopened = Harness(tmp_path)
        reopened.now_ms = harness.now_ms + 10**9
        reopened.scheduler.load([spec()])
        reopened.now_ms += 5_000  # past the first-check window
        reopened.results.append({"text": "A", "error": None})
        await reopened.scheduler.pump()
        await reopened.settle()
        assert reopened.deliveries == []  # unchanged content, healed quiet
        counters = reopened.counters()
        assert counters["content_hash"].startswith("sha256:")
        assert counters["deliveries"] == 0
    finally:
        if reopened is not None:
            reopened.scheduler.dispose()
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_a_lost_counters_file_with_a_truncated_blob_re_establishes_silently(
    tmp_path: Any,
) -> None:
    settings = dataclasses.replace(MonitorSettings(), snapshot_max_chars=5)
    harness = Harness(tmp_path, settings=settings)
    reopened: Harness | None = None
    try:
        harness.scheduler.load([spec()])
        harness.results.append({"text": "0123456789", "error": None})
        await harness.pump_ripen()
        assert harness.counters()["content_hash"]  # hash covers the FULL text
        monitor_state.counters_path(harness.config_dir, "sess", "m1").unlink()

        reopened = Harness(tmp_path, settings=settings)
        reopened.now_ms = harness.now_ms + 10**9
        reopened.scheduler.load([spec()])
        reopened.now_ms += 5_000  # past the first-check window
        reopened.results.append({"text": "01234xxxxx", "error": None})
        await reopened.scheduler.pump()
        await reopened.settle()
        # No counters and the blob is truncated: the hash cannot be rebuilt and
        # a prefix cannot diff honestly, so the baseline is re-established
        # silently (a delivery here would claim "beyond the window" evidence
        # it does not have).
        assert reopened.deliveries == []
        assert reopened.counters()["last_note"] == "baseline re-established"
    finally:
        if reopened is not None:
            reopened.scheduler.dispose()
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_an_expired_monitor_stops_ticking(harness: Harness) -> None:
    harness.scheduler.load([spec(until_at=NOW + 90_000)])
    harness.results.append({"text": "A", "error": None})
    await harness.pump_ripen(advance=1_000)
    assert len(harness.calls) == 1
    harness.now_ms = NOW + 120_000  # past until
    assert harness.scheduler.next_monitor_due_at() is None
    harness.results.append({"text": "B", "error": None})
    await harness.scheduler.pump()
    await harness.settle()
    assert len(harness.calls) == 1  # never checks past until


@pytest.mark.asyncio
async def test_cancel_removes_state_and_the_row(harness: Harness) -> None:
    harness.scheduler.load([spec()])
    harness.results.append({"text": "A", "error": None})
    await harness.pump_ripen()
    assert monitor_state.read_counters(harness.config_dir, "sess", "m1") is not None
    outcome = await harness.scheduler.cancel("m1")
    assert outcome["cancelled"] == "m1"
    assert harness.scheduler.monitors == ()
    assert monitor_state.read_counters(harness.config_dir, "sess", "m1") is None
    assert monitor_state.read_snapshot(harness.config_dir, "sess", "m1") is None


@pytest.mark.asyncio
async def test_dispose_cancels_an_in_flight_check(tmp_path: Any) -> None:
    release = asyncio.Event()
    started = asyncio.Event()
    delivered: list[Any] = []
    clock = [NOW]

    async def run(monitor: MonitorSpec) -> CheckOutcome:
        started.set()
        await release.wait()
        return {"text": "B", "error": None}

    scheduler = MonitorScheduler(
        now=lambda: clock[0],
        config_dir=tmp_path / "cfg",
        session_id="sess",
        settings=MonitorSettings(),
        validate=lambda tool, args: None,
        run_check=run,
        deliver=lambda delivery: delivered.append(delivery),
        persist=lambda monitors: None,
        uniform=lambda low, high: low,
    )
    scheduler.load([spec()])
    clock[0] += 5_000
    await scheduler.pump()
    await started.wait()
    task = next(iter(scheduler._check_tasks))
    scheduler.dispose()
    release.set()
    # The cancelled check task is what must finish; awaiting it (rather than
    # counting loop turns) is what makes "nothing was delivered" a statement
    # about a settled task and not about how far it had got.
    await asyncio.wait([task], timeout=DRAIN_BACKSTOP_S)
    assert task.done() and task.cancelled()
    assert delivered == []


@pytest.mark.asyncio
async def test_a_disabled_monitor_loaded_from_counters_stays_disabled(tmp_path: Any) -> None:
    harness = Harness(tmp_path)
    try:
        harness.scheduler.load([spec()])
        assert monitor_state.read_counters(harness.config_dir, "sess", "m1") is None
        monitor_state.write_counters(
            harness.config_dir,
            "sess",
            "m1",
            {
                "schema": 1,
                "monitor_id": "m1",
                "disabled": True,
                "disabled_reason": "boom",
                "consecutive_failures": 9,
                "next_due_at": NOW - 1,
            },
        )
        reopened = Harness(tmp_path)
        try:
            reopened.scheduler.load([spec()])
            assert reopened.scheduler.next_monitor_due_at() is None
            row = reopened.scheduler.index_rows()[0]
            assert row["disabled"] is True
        finally:
            reopened.scheduler.dispose()
    finally:
        harness.scheduler.dispose()


# ---------------------------------------------------------------------------
# §D3: transient-versus-gone, and the notices both produce
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unavailable_ticks_take_no_strike_and_resume_silently(tmp_path: Any) -> None:
    """A tool that is temporarily out of reach must not walk the failure
    ladder: five ticks of "not in this session's tool set" is how an armed
    monitor was silently disabled while its MCP server was merely reconnecting.
    """
    settings = dataclasses.replace(MonitorSettings(), max_consecutive_failures=2)
    harness = Harness(tmp_path, settings=settings)
    try:
        harness.scheduler.load([spec()])
        # Three ticks, each past the ladder's current backoff (the first check
        # of a 60 s monitor is due 60 s after the arm), and all of them inside
        # 30 minutes so the stall notice is not part of this test.
        for advance in (60_000, 60_000, 120_000):
            harness.results.append({"error": '"datadog" is disconnected', "kind": "unavailable"})
            await rill(harness, advance)
        counters = harness.counters()
        assert counters["disabled"] is False
        assert counters["consecutive_failures"] == 0
        # The diff accounting is untouched too: nothing was learned about the
        # watched thing, so it is not a check that happened.
        assert counters["checks"] == 0
        assert counters["unavailable_ticks"] == 3
        assert counters["last_note"] == "tool unavailable — retrying"
        # It keeps retrying on the capped ladder rather than leaving the timer idle.
        assert counters["next_due_at"] is not None
        # Under the stall window, so nobody was told.
        assert harness.notices == []

        harness.results.append({"text": "A", "error": None})
        await rill(harness, 240_000)
        resumed = harness.counters()
        assert resumed["unavailable_ticks"] == 0 and resumed["unavailable_since"] == 0
        assert resumed["checks"] == 1
        # Silent: the episode never announced itself, so there is nothing to
        # match with a "running again".
        assert harness.notices == []
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_resume_after_unavailability_yields_the_consolidated_delta(tmp_path: Any) -> None:
    """The baseline is NOT advanced while the tool is unreachable, so the first
    successful check after the gap diffs against the pre-gap state and reports
    everything missed as one delta with the gap counted as skipped.
    """
    harness = Harness(tmp_path)
    try:
        harness.scheduler.load([spec(every_ms=60_000)])
        harness.results.append({"text": "A", "error": None})
        await harness.pump_ripen()
        assert harness.deliveries == []
        hash_before = harness.counters()["content_hash"]
        assert hash_before

        for advance in (60_000, 60_000, 120_000):
            harness.results.append({"error": "server still connecting", "kind": "unavailable"})
            await rill(harness, advance)
        assert harness.counters()["content_hash"] == hash_before

        harness.results.append({"text": "B", "error": None})
        await rill(harness, 240_000)
        assert len(harness.deliveries) == 1
        delivery = harness.deliveries[0]
        # One line replaced, so the renderer counts the removal and the addition.
        assert delivery.changes == 2
        assert "- A" in delivery.delta_text and "+ B" in delivery.delta_text
        # The due instants that passed while the tool was unreachable are a gap
        # in the WATCH, not a change to the watched thing, and the delivery says
        # so instead of letting the reader read the gap as churn.
        assert delivery.skipped >= 1
        assert harness.counters()["content_hash"] != hash_before
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_stalled_notice_once_then_restored_notice(tmp_path: Any) -> None:
    """One notice per episode: a watch that is quietly waiting must not become
    a source of noise, but a long wait must not stay invisible either.
    """
    from local_operator.monitors.scheduler import UNAVAILABLE_STALL_MS

    harness = Harness(tmp_path)
    try:
        harness.scheduler.load([spec()])
        harness.results.append({"error": "server disconnected", "kind": "unavailable"})
        await rill(harness, 60_000)
        assert harness.notices == []

        harness.results.append({"error": "server disconnected", "kind": "unavailable"})
        await rill(harness, UNAVAILABLE_STALL_MS + 1)
        assert [notice.kind for notice in harness.notices] == ["stalled"]
        # No delivery: a notice answers no diff.
        assert harness.deliveries == []

        # Still stalled, well past the window: no second notice.
        harness.results.append({"error": "server disconnected", "kind": "unavailable"})
        await rill(harness, UNAVAILABLE_STALL_MS * 2)
        assert [notice.kind for notice in harness.notices] == ["stalled"]

        harness.results.append({"text": "same", "error": None})
        await rill(harness, 240_000)
        assert [notice.kind for notice in harness.notices] == ["stalled", "restored"]
        counters = harness.counters()
        assert counters["unavailable_notified"] is False
        assert counters["unavailable_since"] == 0
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_unavailable_past_24h_disables(tmp_path: Any) -> None:
    """ "Genuinely gone" is settled by time, not by a strike count: a server
    that reconnects comes back; one that was uninstalled does not.
    """
    from local_operator.monitors.scheduler import UNAVAILABLE_GONE_MS

    harness = Harness(tmp_path)
    try:
        harness.scheduler.load([spec()])
        harness.results.append({"error": "server disconnected", "kind": "unavailable"})
        await rill(harness, 60_000)
        assert harness.counters()["disabled"] is False

        harness.results.append({"error": "server disconnected", "kind": "unavailable"})
        await rill(harness, UNAVAILABLE_GONE_MS + 1)
        counters = harness.counters()
        assert counters["disabled"] is True
        assert counters["disabled_reason"] == "tool unavailable for 24h"
        assert counters["next_due_at"] is None
        assert [notice.kind for notice in harness.notices] == ["disabled"]
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_fatal_kind_disables_on_first_occurrence(tmp_path: Any) -> None:
    """A deterministic failure (the tool's own schema rejected the arguments)
    cannot self-heal, so five identical ticks are four wasted checks.
    """
    harness = Harness(tmp_path)
    try:
        harness.scheduler.load([spec()])
        harness.results.append(
            {"error": "invalid arguments:\n- path: Extra inputs are not permitted", "kind": "fatal"}
        )
        await harness.pump_ripen()
        counters = harness.counters()
        assert counters["disabled"] is True
        assert counters["consecutive_failures"] == 1
        assert "Extra inputs" in counters["disabled_reason"]
        assert [notice.kind for notice in harness.notices] == ["disabled"]

        # And no further checks run.
        harness.results.append({"text": "later", "error": None})
        await harness.pump_ripen()
        assert len(harness.calls) == 1
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_disable_announces_once_and_is_not_a_delivery(tmp_path: Any) -> None:
    settings = dataclasses.replace(MonitorSettings(), max_consecutive_failures=2)
    harness = Harness(tmp_path, settings=settings)
    try:
        harness.scheduler.load([spec()])
        for _ in range(2):
            harness.results.append({"error": "boom", "text": None})
            await harness.pump_ripen()
        counters = harness.counters()
        assert counters["disabled"] is True
        assert [notice.kind for notice in harness.notices] == ["disabled"]
        notice = harness.notices[0]
        assert notice.monitor_id == "m1" and notice.failures == 2
        assert "boom" in notice.detail
        # NOT a delivery: the rate window and the delivery counters are for
        # material changes.
        assert harness.deliveries == []
        assert counters["deliveries"] == 0
        assert counters["disable_notified"] is True

        # A second sweep finds nothing left to announce (the latch).
        assert await harness.scheduler.announce_unannounced_disables() == 0
        assert len(harness.notices) == 1
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_unannounced_disable_is_announced_after_reload(tmp_path: Any) -> None:
    """A disable written by an older build (or lost to a crash between the
    disable and its notice) is told exactly once on the next open.
    """
    harness = Harness(tmp_path)
    try:
        harness.scheduler.load([spec()])
        assert monitor_state.read_counters(harness.config_dir, "sess", "m1") is None
        monitor_state.write_counters(
            harness.config_dir,
            "sess",
            "m1",
            {
                "schema": 1,
                "monitor_id": "m1",
                "disabled": True,
                "disabled_reason": 'monitor can\'t watch "mcp__x": not in this tool set',
                "consecutive_failures": 5,
                "checks": 7,
                "deliveries": 0,
                "next_due_at": None,
            },
        )
        reopened = Harness(tmp_path)
        try:
            reopened.scheduler.load([spec()])
            assert await reopened.scheduler.announce_unannounced_disables() == 1
            assert [notice.kind for notice in reopened.notices] == ["disabled"]
            # Exactly once: the latch is now set on disk.
            assert await reopened.scheduler.announce_unannounced_disables() == 0
            assert reopened.counters()["disable_notified"] is True
        finally:
            reopened.scheduler.dispose()
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_a_stall_notice_that_never_reached_a_session_is_retried(tmp_path: Any) -> None:
    """R2, in the shape that DISCRIMINATES (review round 2, R9).

    The sink stays failing ACROSS the stall tick. The old code latched inside
    ``_apply_failure`` before calling the sink, so that tick left
    ``unavailable_notified`` True with nothing delivered — and the first
    success afterwards then announced a recovery the operator had never been
    told about. Asserting that takes a tick where the sink fails while the
    stall window is crossed; a test that heals the sink first passes on the old
    head and pins nothing.
    """
    from local_operator.monitors.scheduler import UNAVAILABLE_STALL_MS

    harness = Harness(tmp_path)
    delivered: list[Any] = []
    fail = {"on": True}

    async def announce(notice: Any) -> None:
        if fail["on"]:
            raise RuntimeError("sink is down")
        delivered.append(notice)

    try:
        harness.scheduler.load([spec()])
        harness.scheduler._announce = announce

        # Tick 1 — unavailable, before the stall window.
        harness.results.append({"error": "server disconnected", "kind": "unavailable"})
        await rill(harness, 60_000)
        assert delivered == []

        # Tick 2 — THE DISCRIMINATING ONE: the window is crossed while the sink
        # is failing. No notice reached anyone, so no latch may be set.
        harness.results.append({"error": "server disconnected", "kind": "unavailable"})
        await rill(harness, UNAVAILABLE_STALL_MS)
        assert delivered == []
        assert harness.counters()["unavailable_notified"] is False

        # Tick 3 — the sink answers: the episode is still open, so the notice is
        # earned again.
        fail["on"] = False
        harness.results.append({"error": "server disconnected", "kind": "unavailable"})
        await rill(harness, UNAVAILABLE_STALL_MS)
        assert [notice.kind for notice in delivered] == ["stalled"]
        assert harness.counters()["unavailable_notified"] is True

        # Tick 4 — recovery is a real transition now: the operator was told.
        harness.results.append({"text": "same", "error": None})
        await rill(harness, 240_000)
        assert [notice.kind for notice in delivered] == ["stalled", "restored"]
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_a_silent_stall_episode_never_announces_a_recovery(tmp_path: Any) -> None:
    """The other half of R2, and it is a PIN rather than a guard (round 2, R9).

    The episode crosses the stall window with the sink failing throughout: the
    old code latched anyway and emitted a phantom "running again" once the tool
    recovered. ``unavailable_notified`` is the discriminating assertion — it is
    the precondition that decides whether the recovery notice is earned.
    """
    from local_operator.monitors.scheduler import UNAVAILABLE_STALL_MS

    harness = Harness(tmp_path)
    delivered: list[Any] = []

    async def announce(notice: Any) -> None:
        raise RuntimeError("sink is down")

    try:
        harness.scheduler.load([spec()])
        harness.scheduler._announce = announce

        harness.results.append({"error": "server disconnected", "kind": "unavailable"})
        await rill(harness, 60_000)
        harness.results.append({"error": "server disconnected", "kind": "unavailable"})
        await rill(harness, UNAVAILABLE_STALL_MS)
        assert harness.counters()["unavailable_notified"] is False
        assert delivered == []

        harness.results.append({"text": "same", "error": None})
        await rill(harness, 240_000)
        assert delivered == []
    finally:
        harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_a_retro_announced_disable_keeps_the_instants_that_are_true(tmp_path: Any) -> None:
    """R3: the retro-announce used "now" as the disable instant and the strike
    count as its cause — so a legacy row read "was DISABLED at <now> after 5
    consecutive failed checks" for a 24-hour unreachable episode that charged no
    strike at all, and "after 0 consecutive failed checks" when a file carried
    no count.
    """
    from local_operator.monitors.delivery import format_monitor_notice_text

    harness = Harness(tmp_path)
    try:
        harness.scheduler.load([spec()])
        disabled_at = harness.now_ms - 600_000
        monitor_state.write_counters(
            harness.config_dir,
            "sess",
            "m1",
            {
                "schema": 1,
                "monitor_id": "m1",
                "disabled": True,
                "disabled_kind": "unreachable",
                "disabled_reason": "tool unavailable for 24h",
                "last_check_at": disabled_at,
                "checks": 7,
                "deliveries": 0,
                "next_due_at": None,
            },
        )

        reopened = Harness(tmp_path)
        try:
            reopened.scheduler.load([spec()])
            assert await reopened.scheduler.announce_unannounced_disables() == 1
            notice = reopened.notices[0]
            assert notice.kind == "disabled"
            assert notice.failure_kind == "unreachable"
            # The clock is the last check, not the announcement.
            assert notice.at_ms == disabled_at
            text = format_monitor_notice_text(notice)
            assert "consecutive failed check" not in text
            assert "stayed unreachable for 24 hours" in text
        finally:
            reopened.scheduler.dispose()
    finally:
        harness.scheduler.dispose()


def test_disabled_clause_covers_every_cause_the_counters_can_describe() -> None:
    """R15: ``disabled_clause`` renders four causes; the folded banner and the
    kept sentence were pinned through the surfaces, the other two were not.
    """
    from local_operator.monitors.store import disabled_clause

    # A raw multi-line banner is replaced by the counters' own cause.
    assert (
        disabled_clause(
            {
                "disabled_reason": "invalid arguments:\n- path: Extra inputs are not permitted",
                "consecutive_failures": 5,
            }
        )
        == "5 consecutive failed checks"
    )
    # No count left in the counters file: state the fact without a number.
    assert disabled_clause(
        {"disabled_reason": "invalid arguments:\n- x", "consecutive_failures": 0}
    ) == ("repeated failed checks")
    # The kind is used only when the reason is a raw banner, and it is the
    # closest honest cause for a 24-hour episode.
    assert (
        disabled_clause({"disabled_reason": "banner\nsecond line", "disabled_kind": "unreachable"})
        == "its tool stayed unreachable for 24 hours"
    )
    assert (
        disabled_clause({"disabled_reason": "banner\nsecond line", "disabled_kind": "fatal"})
        == "a check that cannot succeed"
    )
    # A reason that already reads as a sentence is kept, with its whitespace
    # folded — a single line is what the table cell can hold.
    assert disabled_clause({"disabled_reason": "the tool  stopped   being read-only."}) == (
        "the tool stopped being read-only."
    )
    assert disabled_clause(
        {"disabled_reason": 'monitor can\'t watch "mcp__x": it is not in this tool set.'}
    ) == ('monitor can\'t watch "mcp__x": it is not in this tool set.')
    # Multi-line is never kept, whatever it says: the table has one row line.
    assert (
        disabled_clause(
            {"disabled_reason": "not in this tool set.\nand more", "consecutive_failures": 2}
        )
        == "2 consecutive failed checks"
    )
    assert disabled_clause({}) == "repeated failed checks"
