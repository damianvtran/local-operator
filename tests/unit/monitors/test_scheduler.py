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

    async def _persist(self, monitors: list[MonitorSpec]) -> None:
        self.persisted.append([monitor.model_dump() for monitor in monitors])

    def _on_change(self) -> None:
        self.changes += 1

    async def settle(self, rounds: int = 12) -> None:
        for _ in range(rounds):
            await asyncio.sleep(0)

    async def pump_ripen(self, *, advance: int = 10**9) -> None:
        """Advance the clock past every due time, pump, and settle."""
        self.now_ms += advance
        await self.scheduler.pump()
        await self.settle()

    def counters(self, monitor_id: str = "m1") -> dict[str, Any]:
        found = monitor_state.read_counters(self.config_dir, "sess", monitor_id)
        assert found is not None, f"no counters for {monitor_id}"
        return found


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
        for _ in range(6):
            await asyncio.sleep(0)
        assert slow.calls == 1  # baseline captured, no delivery

        clock[0] += 60_000
        await scheduler.pump()  # starts the slow second check
        for _ in range(6):
            await asyncio.sleep(0)
        assert started.is_set()
        await scheduler.pump()  # third pump while in flight: overlap
        await asyncio.sleep(0)
        counters = monitor_state.read_counters(tmp_path / "cfg", "sess", "m1")
        assert counters is not None and counters["skipped_overlap"] == 1
        release.set()
        for _ in range(6):
            await asyncio.sleep(0)
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
    clock = [NOW]

    async def run(monitor: MonitorSpec) -> CheckOutcome:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
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
        for _ in range(6):
            await asyncio.sleep(0)
        assert peak == 2  # the third stayed due for the next pass
        release.set()
        for _ in range(6):
            await asyncio.sleep(0)
        # The deferred check is due the moment a slot frees; a later pump runs it.
        await scheduler.pump()
        for _ in range(6):
            await asyncio.sleep(0)
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
    scheduler.dispose()
    release.set()
    for _ in range(6):
        await asyncio.sleep(0)
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
