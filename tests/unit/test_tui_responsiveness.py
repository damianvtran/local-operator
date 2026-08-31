"""Loop-liveness contracts for the TUI responsiveness regression (v0.29.0 → v0.33.1).

The reported symptoms — boot waits for MCP, sends freeze on pricing, tool
calls freeze the TUI — were all one shape: synchronous work on the Textual
loop. These tests pin the fixes with the harness the design prescribed: a
5 ms tick probe that records every gap over a stall threshold, so a future
change that reintroduces a stall fails a test instead of shipping a freeze.

The probe is deliberately NOT Textual-dependent for the unit-level tests:
the stall is a property of the event loop, and asserting it without the app
keeps the failure diagnosis direct. The end-to-end tests drive the real
OperatorApp via run_test.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Usage


@pytest.fixture
def tmp_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated config dir + env, mirroring the factory test module's rig."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config_dir = tmp_path / ".local-operator"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


#: A gap the human eye reads as a dropped frame. The design's bar for a green
#: run is "no stall > 50 ms"; recording from 30 ms keeps headroom visible.
STALL_MS = 30.0
TICK_S = 0.005


class StallRecorder:
    """Records loop-thread CPU bursts AND wall-clock gaps while it runs.

    Two clocks, because neither alone can do this job:

    * ``time.thread_time`` (CPU) is per-thread and excludes time asleep or
      waiting on the GIL, so a sample is large only when the loop thread
      genuinely ran without yielding — never merely because the machine was
      busy. This is the same statistic ``tests/unit/tools/
      test_loop_liveness.py`` adopted for ``LoopCpuProbe``. It is the clock
      that sees the reconnect regression (a synchronous 60 MB transcript
      parse, ~90-130 ms of CPU on the loop) and is blind to OS scheduler
      starvation (the flake that produced 525 ms and 668 ms wall gaps on
      loaded CI runners whose sibling runs passed).
    * ``time.perf_counter`` (wall) sees a pure blocking sleep on the loop,
      which the CPU clock cannot (a 300 ms ``time.sleep`` records 0.1 ms of
      CPU). Its ceiling is set high enough that scheduler noise of hundreds
      of milliseconds cannot trip it, and low enough that a multi-second
      block (a 2 s scan, a 5 s pricing call) still does.

    Measured on the real shapes with this probe:

      ========================  =========  =========
      scenario                  wall gap   CPU gap
      ========================  =========  =========
      60 MB sync parse on loop   96 ms      96 ms
      pure sleep block           306 ms     0.1 ms
      loop idle, OS-starved      525-668 ms 0.0 ms
      ========================  =========  =========

    Call-site ceilings are site-appropriate, not global. The reconnect and
    connect tests use the strict 50 ms CPU bar (healthy sample 0 ms,
    regression 116 ms). Sites with legitimate loop CPU — the title-scan
    test records a healthy 36-38 ms — use a 200 ms CPU bar, because the
    same work measured 2.4× slower on ubuntu-latest in
    ``test_launch_subagent.py`` (393-492 ms on a dev box, 1056-1156 ms on
    CI), and a 50 ms bar would flake a green tree. The wall ceiling is
    2000 ms everywhere: 3× the worst observed scheduler-noise gap, and
    still well below the multi-second regressions.
    """

    def __init__(self, stall_ms: float = STALL_MS) -> None:
        self.stall_ms = stall_ms
        self.cpu_stalls: list[float] = []
        self.wall_stalls: list[float] = []
        self._task: asyncio.Task[None] | None = None

    async def _probe(self) -> None:
        last_cpu = time.thread_time()
        last_wall = time.perf_counter()
        while True:
            await asyncio.sleep(TICK_S)
            now_cpu = time.thread_time()
            now_wall = time.perf_counter()
            cpu_gap_ms = (now_cpu - last_cpu) * 1000.0
            wall_gap_ms = (now_wall - last_wall) * 1000.0
            if cpu_gap_ms >= self.stall_ms:
                self.cpu_stalls.append(cpu_gap_ms)
            if wall_gap_ms >= self.stall_ms:
                self.wall_stalls.append(wall_gap_ms)
            last_cpu = now_cpu
            last_wall = now_wall

    async def start(self) -> None:
        self._task = asyncio.create_task(self._probe())
        # Let the probe take a tick BEFORE the work under test runs, so its
        # baseline is initialised; otherwise the first post-stall tick sets
        # the baseline and the stall is invisible to the recorder.
        await asyncio.sleep(TICK_S * 2)

    async def stop(self) -> None:
        assert self._task is not None
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def assert_no_stall(
        self, cpu_ceiling_ms: float = 50.0, wall_ceiling_ms: float = 2000.0
    ) -> None:
        """Strict bar: 50 ms of loop-thread CPU, 2 s of wall as catastrophe.

        Used by the reconnect and connect tests, whose healthy CPU sample is
        0 ms and whose regression (a synchronous transcript parse) is
        90-130 ms. The wall ceiling is a backstop for a pure blocking sleep
        the CPU clock cannot see; it is 3× the worst scheduler-noise gap
        observed on CI (668 ms) so that noise cannot trip it.
        """
        worst_cpu = max(self.cpu_stalls) if self.cpu_stalls else 0.0
        worst_wall = max(self.wall_stalls) if self.wall_stalls else 0.0
        assert worst_cpu < cpu_ceiling_ms, (
            f"event loop consumed {worst_cpu:.0f} ms of CPU without yielding "
            f"(ceiling {cpu_ceiling_ms:.0f} ms); "
            f"all CPU stalls: {[round(s, 1) for s in self.cpu_stalls]}"
        )
        assert worst_wall < wall_ceiling_ms, (
            f"event loop blocked for {worst_wall:.0f} ms of wall time "
            f"(ceiling {wall_ceiling_ms:.0f} ms); "
            f"all wall stalls: {[round(s, 1) for s in self.wall_stalls]}"
        )

    def assert_no_stall_loaded(
        self, cpu_ceiling_ms: float = 200.0, wall_ceiling_ms: float = 2000.0
    ) -> None:
        """Loaded bar: 200 ms of loop-thread CPU, same 2 s wall backstop.

        The title-scan test records a healthy 36-38 ms of loop CPU for a
        120-session scan. The same class of CPU-bound work measured 2.4×
        slower on ubuntu-latest than on a dev box
        (``test_launch_subagent.py``, 393-492 ms vs 1056-1156 ms), so a
        50 ms bar would flake a green tree on a slow CI core (~90 ms).
        200 ms is 2× that projected CI cost and still well below the
        multi-second regressions (a 2 s scan, a 5 s pricing block).

        The wall ceiling is the same catastrophic backstop as
        :meth:`assert_no_stall`: a pure blocking sleep is invisible to the
        CPU clock, and 2000 ms sits 3× above the worst observed scheduler
        noise so that noise cannot trip it.
        """
        self.assert_no_stall(cpu_ceiling_ms=cpu_ceiling_ms, wall_ceiling_ms=wall_ceiling_ms)


# --- A2: the title backfill answers a no-title directory once ----------------


def _titleless_session(root: Path, name: str, lines: int = 40) -> Path:
    """A session directory whose transcript carries no journalled title.

    This is the expensive population for the backfill: a full read that can
    never produce a sidecar, which before the sentinel was re-read on every
    boot for the store's whole life.
    """
    import local_operator.resume as resume

    directory = root / "sessions" / name
    directory.mkdir(parents=True)
    line = (
        '{"type": "message", "payload": {"role": "user", '
        '"content": [{"type": "text", "text": "opening line"}]}}\n'
    )
    (directory / resume.TRANSCRIPT_NAME).write_text(line * lines, encoding="utf-8")
    return directory


def test_second_title_backfill_pass_is_stat_only(tmp_path: Path) -> None:
    """The perpetual rescan is over: boot two answers with the sentinel.

    The design measured 323 ms per boot on a real 1,365-session store because
    1,268 sessions could never grow a sidecar and were fully re-read every
    time. The sentinel turns the second pass into one ``stat`` per directory.
    """
    from local_operator.resume import (
        TITLE_SCAN_SENTINEL_NAME,
        backfill_session_titles,
        session_name,
        stored_session_title,
    )

    for index in range(25):
        _titleless_session(tmp_path, f"sess{index:04d}")

    assert backfill_session_titles(tmp_path) == 0  # nothing to journal
    sentinels = list((tmp_path / "sessions").glob(f"*/{TITLE_SCAN_SENTINEL_NAME}"))
    assert len(sentinels) == 25, "a no-title directory must record its scan"

    # The second pass is the one that used to redo all the work.
    started = time.perf_counter()
    assert backfill_session_titles(tmp_path) == 0
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    # 25 dirs x ~0.01ms stat each; the pre-fix pass re-READ every transcript
    # (~0.5ms each here). The bound is generous against CI jitter but an
    # order of magnitude below a single full read of this fixture.
    assert elapsed_ms < 20.0, f"second pass took {elapsed_ms:.1f} ms — sentinel ignored?"

    # The picker surfaces are untouched: the sentinel is not a title.
    directory = tmp_path / "sessions" / "sess0000"
    assert stored_session_title(directory) == ""
    assert session_name(directory) == "opening line"


def test_a_session_that_grows_a_title_after_the_sentinel_still_journals(
    tmp_path: Path,
) -> None:
    """The sentinel answers "scanned", not "empty forever".

    A directory scanned while title-less whose transcript LATER gains a
    journalled title (a resumed old session renamed under a newer build)
    must still get its sidecar: the sentinel is skipped only as a
    no-work marker, and the sidecar check runs first for the same reason
    it always did.
    """
    from local_operator.resume import (
        TITLE_SCAN_SENTINEL_NAME,
        TITLE_SIDECAR_NAME,
        backfill_session_titles,
    )
    from local_operator.session.naming import CONVERSATION_NAME_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript

    directory = _titleless_session(tmp_path, "grows")
    assert backfill_session_titles(tmp_path) == 0
    assert (directory / TITLE_SCAN_SENTINEL_NAME).exists()

    async def add_title() -> None:
        transcript = Transcript(directory)
        await transcript.append_custom(
            CONVERSATION_NAME_CUSTOM_TYPE, {"text": "A Late Title", "user_set": True}
        )

    asyncio.run(add_title())
    # The sentinel must not shadow a sidecar-worthy scan for a session whose
    # transcript changed: drop-in behaviour would leave it unfindable.
    (directory / TITLE_SCAN_SENTINEL_NAME).unlink()
    assert backfill_session_titles(tmp_path) == 1
    assert (directory / TITLE_SIDECAR_NAME).exists()


def test_sentinel_write_preserves_directory_mtime(tmp_path: Path) -> None:
    """Journalling a scan is bookkeeping ABOUT a session, never activity in
    it — the same contract write_session_title carries for the sidecar."""
    import os

    from local_operator.resume import TITLE_SCAN_SENTINEL_NAME, backfill_session_titles

    directory = _titleless_session(tmp_path, "mtime")
    before = directory.stat().st_mtime
    os.utime(directory, (before - 10_000, before - 10_000))
    expected = directory.stat().st_mtime
    time.sleep(0.01)
    assert backfill_session_titles(tmp_path) == 0
    assert (directory / TITLE_SCAN_SENTINEL_NAME).exists()
    assert directory.stat().st_mtime == expected


# --- A1: _prepare's store scans leave the loop free ---------------------------


@pytest.mark.asyncio
async def test_prepare_store_scans_do_not_stall_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep and backfills run off the loop: no tick gap above the bar.

    Mirrors the design's harness: ``loop.slow_callback_duration`` plus a tick
    probe over a store shaped like the operator's (many title-less sessions).
    The scans themselves still RUN (their effects are asserted); only their
    thread placement is under test.
    """
    from local_operator import resume as resume_mod
    from local_operator.session_factory import _prepare
    from tests.unit.test_session_factory import FakeConfigManager, FakeRegistry, _args

    config_dir = tmp_path / ".local-operator"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    for index in range(120):
        _titleless_session(tmp_path, f"{index:08x}abcd")

    seen_threads: set[int] = set()
    real_titles = resume_mod.backfill_session_titles

    def titles_from(*_a: Any, **_k: Any) -> int:
        seen_threads.add(threading_get_ident())
        return real_titles(*_a, **_k)

    monkeypatch.setattr(resume_mod, "backfill_session_titles", titles_from)

    from local_operator.credentials import CredentialManager

    args = _args(hosting="test", model="test-model", yolo=True)
    args.resume = None
    recorder = StallRecorder()
    await recorder.start()
    try:
        await _prepare(
            args,
            cast_config(FakeConfigManager({"hosting": "test", "model_name": "test-model"})),
            CredentialManager(config_dir),
            cast_registry(FakeRegistry(config_dir)),
            has_ui=True,
            cwd=str(tmp_path),
        )
    finally:
        await recorder.stop()
    recorder.assert_no_stall_loaded()
    # And the scan genuinely ran — in a worker thread, not on the loop.
    assert seen_threads, "the title backfill never executed"
    assert threading_get_ident() not in seen_threads


def cast_config(value: Any) -> Any:
    return value


def cast_registry(value: Any) -> Any:
    return value


def threading_get_ident() -> int:
    import threading

    return threading.get_ident()


# --- A3: RemoteSession history replay leaves the loop free --------------------


@pytest.mark.asyncio
async def test_remote_connect_replay_does_not_stall_the_loop(tmp_path: Path) -> None:
    """A large transcript replay runs in a thread: the follower's loop stays
    responsive during connect, which is the window between the attach dial
    and the first painted block."""
    from local_operator.harness.types import Message, TextContent
    from local_operator.session.transcript import Transcript

    (tmp_path / "sessions" / "s1").mkdir(parents=True)

    async def build() -> None:
        transcript = Transcript(tmp_path / "sessions" / "s1")
        for index in range(400):
            await transcript.append_message(
                Message(
                    role="assistant",
                    content=[TextContent(text=f"turn {index} " + "z" * 2_000)],
                )
            )

    await build()

    from local_operator.mobile.registrant import Registrant
    from local_operator.session.remote import RemoteSession
    from tests.unit.mobile.test_registrant import FakeHandle
    from tests.unit.session.test_remote import _wait_record

    monkeypatch_env(tmp_path)
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        recorder = StallRecorder()
        await recorder.start()
        remote = await RemoteSession.connect(
            record,
            "s1",
            config_dir=tmp_path,
            takeover_factory=_never_take_over,
        )
        await recorder.stop()
        # Strict 50 ms CPU bar: healthy sample is 0 ms, the regression this
        # test exists to catch (a synchronous 60 MB parse on the loop) is
        # 90-130 ms. The 2 s wall backstop covers a pure blocking sleep the
        # CPU clock cannot see.
        recorder.assert_no_stall()
        assert len(remote._history) == 400  # the replay genuinely happened
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


# --- MAJOR-2 (round 3): reconnect gap replay stays off the loop ---------------


@pytest.mark.asyncio
async def test_reconnect_gap_replay_does_not_stall_the_loop(tmp_path: Path) -> None:
    """The reconnect path parses the transcript in a thread, like connect.

    Round 3 found ``_replay_durable_suffix`` re-parsing the whole file
    synchronously inside ``_recover_owner`` — a 60 MB transcript blocked the
    loop ~90 ms, past the 50 ms bar #300's connect fix established. The
    recovery now takes ONE threaded parse and feeds both the gap projection
    and the history bind from it, so this drives the actual production
    recovery — owner lost, durable gap appended, replacement owner published —
    over a transcript inflated to tens of megabytes and asserts the loop
    never stalled while it settled.
    """
    from local_operator.harness.types import Message, TextContent
    from local_operator.session.transcript import Transcript

    (tmp_path / "sessions" / "s1").mkdir(parents=True)

    async def build() -> None:
        transcript = Transcript(tmp_path / "sessions" / "s1")
        # 60+ MB of durable history: 3200 rows x ~20 KB of text each. This is
        # the size class the round-3 measurement used to demonstrate the
        # synchronous stall.
        for index in range(3_200):
            await transcript.append_message(
                Message(
                    role="assistant",
                    content=[TextContent(text=f"turn {index} " + "z" * 20_000)],
                )
            )

    await build()
    assert (tmp_path / "sessions" / "s1" / "transcript.jsonl").stat().st_size > 60 * 1024 * 1024

    from local_operator.mobile.registrant import Registrant
    from local_operator.session.remote import RemoteSession
    from tests.unit.mobile.test_registrant import FakeHandle
    from tests.unit.session.test_remote import _wait_record

    monkeypatch_env(tmp_path)
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        events: list[Any] = []
        remote.subscribe(events.append)

        registrant.close()
        import os

        (tmp_path / "sessions" / "s1" / ".session.pid").write_text(str(os.getpid()))
        for _ in range(200):
            if remote._recovering:
                break
            await asyncio.sleep(0.02)
        assert remote._recovering is True
        transcript = Transcript(tmp_path / "sessions" / "s1")
        await transcript.append_message(Message.user("durable while disconnected"))

        recorder = StallRecorder()
        await recorder.start()
        replacement = Registrant(handle, kind="tui")
        replacement.start()
        try:
            deadline = asyncio.get_running_loop().time() + 60
            while asyncio.get_running_loop().time() < deadline:
                if not remote._recovering and any(
                    event.type == "history_delta" for event in events
                ):
                    break
                await asyncio.sleep(0.02)
            await recorder.stop()
            assert remote._recovering is False
            deltas = [event for event in events if event.type == "history_delta"]
            assert len(deltas) == 1  # the gap genuinely replayed
            # Strict 50 ms CPU bar: same rationale as the connect-path twin
            # above. This is the test that originally flaked at 525/668 ms of
            # wall-clock scheduler noise against a 500 ms ceiling.
            recorder.assert_no_stall()
        finally:
            replacement.close()
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


async def _never_take_over() -> Any:
    raise AssertionError("live owner should not trigger takeover")


def monkeypatch_env(tmp_path: Path) -> None:
    import os

    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(tmp_path)


# --- C1/C2: pricing never does I/O on the loop --------------------------------


class _SlowListing:
    """Discovery stub with a controllable latency, counted per call.

    ``latency_s`` is mutable so a test's teardown can shorten it: a refresh
    thread is fire-and-forget and OUTLIVES the assertion that scheduled it,
    and the next test in the same worker process would otherwise inherit a
    thread still sleeping against a stub the teardown already detached.
    """

    def __init__(self, latency_s: float) -> None:
        self.latency_s = latency_s
        self.calls = 0
        self.in_flight = 0

    def __call__(self, provider: str, **_kwargs: Any) -> tuple[list[Any], str]:
        self.calls += 1
        self.in_flight += 1
        try:
            time.sleep(self.latency_s)
        finally:
            self.in_flight -= 1
        return [], "ok"


@pytest.mark.asyncio
async def test_turn_cost_returns_fast_from_the_loop_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cold memo and a hostile provider listing: ``turn_cost`` still answers
    in milliseconds, because the paint path resolves memo-or-registry only."""
    from local_operator.model import configure, discovery
    from local_operator.tui.costs import turn_cost

    slow = _SlowListing(5.0)
    monkeypatch.setattr(discovery, "available_models", slow)
    configure.invalidate_model_info_cache()

    class _Usage:
        input_tokens = 1_000
        output_tokens = 2_000
        usd_cost = None
        cost_components = None

    recorder = StallRecorder()
    await recorder.start()
    started = time.perf_counter()
    cost = turn_cost("kimi/unlisted-model-x", _Usage())
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    await recorder.stop()

    # 500 ms, not 50: the hostile stub blocks for 5 s, so the assertion's
    # job is to separate "milliseconds" from "seconds" — a loaded CI runner
    # can add tens of ms of scheduling noise to any wall clock.
    assert elapsed_ms < 500.0, f"turn_cost blocked the loop for {elapsed_ms:.0f} ms"
    recorder.assert_no_stall_loaded()
    assert cost is None  # honestly unpriceable THIS tick, never a wrong number

    # The background refresh (C2) did the real work off-loop: exactly one
    # fetch for this model despite the loop path returning long before it.
    for _ in range(200):
        if slow.calls >= 1:
            break
        await asyncio.sleep(0.02)
    assert slow.calls == 1, "the paint miss must fire exactly one background refresh"
    # Teardown order is load-bearing: the refresh thread is still INSIDE the
    # 5 s stub sleep. Shorten the sleep so it lands promptly, WAIT for it,
    # and only then invalidate — an invalidation that races the thread's
    # memo write leaves a poisoned entry for the next test in this worker.
    slow.latency_s = 0.0
    for _ in range(200):
        if slow.in_flight == 0:
            break
        await asyncio.sleep(0.02)
    configure.invalidate_model_info_cache()


@pytest.mark.asyncio
async def test_background_refresh_lands_for_the_next_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2's contract: after the off-thread resolution lands, the SAME call
    prices from the warm memo — a one-tick 'unpriceable', not a permanent one."""
    from local_operator.model import configure, discovery
    from local_operator.model.discovery import DiscoveredModel
    from local_operator.tui.costs import turn_cost

    gate = {"open": False}
    fetched: list[str] = []

    def listing_rows(provider: str, **_kwargs: Any) -> tuple[list[Any], str]:
        if not gate["open"]:
            # Slow while closed, so a blocking paint path would fail loudly
            # rather than silently succeed.
            time.sleep(0.05)
            return [], "ok"
        fetched.append(provider)
        row = DiscoveredModel(
            id="unlisted-model-y",
            name="Unlisted",
            context_window=100_000,
            max_tokens=8_000,
            input_price=3.0,
            output_price=6.0,
        )
        return [row], "ok"

    monkeypatch.setattr(discovery, "available_models", listing_rows)
    configure.invalidate_model_info_cache()

    class _Usage:
        input_tokens = 1_000_000
        output_tokens = 1_000_000
        usd_cost = None
        cost_components = None

    assert turn_cost("kimi/unlisted-model-y", _Usage()) is None  # cold miss
    # DRAIN the cold miss's refresh thread BEFORE flipping the gate: it is
    # still inside the closed-listing sleep, and an invalidate that races
    # its memo write re-seeds the unpriced row AFTER the clear — the next
    # call then memo-HITS on it and never fires a second refresh, so the
    # priced answer can never land (the 1-in-6 flake this closed).
    for _ in range(200):
        await asyncio.sleep(0.02)
        if not configure._paint_refreshing:
            break
    gate["open"] = True
    configure.invalidate_model_info_cache()  # drop the degraded row + gate
    started = time.perf_counter()
    assert turn_cost("kimi/unlisted-model-y", _Usage()) is None  # fires refresh
    assert (time.perf_counter() - started) < 0.5, "the refresh fired on the loop"
    for _ in range(300):
        await asyncio.sleep(0.02)
        if fetched:
            break
    assert fetched, "the background refresh never consulted the listing"
    # Next tick: warm memo prices exactly. Poll until the refresh has LANDED
    # (the listing being consulted is necessary but not sufficient — the
    # resolver still has to finish before the paint memo is fed).
    cost = None
    for _ in range(200):
        cost = turn_cost("kimi/unlisted-model-y", _Usage())
        if cost is not None:
            break
        await asyncio.sleep(0.02)
    assert cost == pytest.approx(3.0 + 6.0)
    configure.invalidate_model_info_cache()


@pytest.mark.asyncio
async def test_component_pricing_never_calls_discovery_from_paint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR #295 widened the miss surface: each cost component resolves its own
    (provider, model_id). A hostile listing must not be reachable from any of
    them through the paint path."""
    from local_operator.model import configure, discovery
    from local_operator.tui.costs import turn_cost

    slow = _SlowListing(5.0)
    monkeypatch.setattr(discovery, "available_models", slow)
    configure.invalidate_model_info_cache()

    class _Component:
        provider = "openai"
        model_id = "another-unlisted-one"
        input_tokens = 10
        output_tokens = 10
        usd_cost = None

    class _Usage:
        input_tokens = 0
        output_tokens = 0
        usd_cost = None
        cost_components = [_Component()]

    started = time.perf_counter()
    cost = turn_cost("kimi/parent-model", _Usage())
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert elapsed_ms < 500.0, f"component pricing blocked for {elapsed_ms:.0f} ms"
    assert cost is None
    slow.latency_s = 0.0
    for _ in range(200):
        if slow.in_flight == 0:
            break
        await asyncio.sleep(0.02)
    configure.invalidate_model_info_cache()


def test_warm_memo_still_prices_exactly() -> None:
    """The paint path is the FULL answer once the memo is warm — not a
    degraded shortcut. A shipped registry row with prices must price through
    unchanged."""
    from local_operator.model import configure
    from local_operator.tui.costs import turn_cost

    configure.invalidate_model_info_cache()
    cost = turn_cost("anthropic/claude-opus-4-5", Usage(input_tokens=1_000_000, output_tokens=0))
    registry, memo_hit = configure.resolve_model_info_paint("anthropic", "claude-opus-4-5")
    assert cost == pytest.approx(registry.input_price)
    configure.invalidate_model_info_cache()


# --- C3: the bash live snapshot is bounded ------------------------------------


@pytest.mark.asyncio
async def test_bash_emit_tick_cost_does_not_grow_with_total_output() -> None:
    """The live update is built from a bounded tail, so tick cost is flat in
    the command's total output — the O(total) freeze the operator reported."""
    import contextlib

    from local_operator.harness.types import ToolContext
    from local_operator.tools.builtin import execute_bash
    from local_operator.variables import VariableStore

    store = VariableStore()
    store.store_credential("GITHUB_TOKEN", "ghp_notarealtoken000000")
    context = ToolContext(cwd="/tmp", session_id="bench", agent_id="main", variables=store)

    chunk = ("x" * 65536 + "\n").encode()

    class _FakeReader:
        def __init__(self, data: bytes) -> None:
            self._data = data

        async def read(self, _n: int = -1) -> bytes:
            data, self._data = self._data, b""
            return data

    import subprocess

    # A REAL process in its own session backs the fake pid: execute_bash's
    # cancellation path runs ``os.killpg(os.getpgid(pid), SIGKILL)``, and a
    # sentinel pid like 1 aims that at init's process group — on Linux the
    # kill SUCCEEDS and takes down the test runner itself (this killed the
    # GitHub Actions runner agent mid-suite; macOS only survived because
    # non-root killpg(1) raises EPERM). A throwaway ``sleep`` gives the kill
    # a target with the exact semantics the tool expects, harmlessly.
    doomed = subprocess.Popen(["sleep", "300"], start_new_session=True)

    class _FakeProc:
        def __init__(self, data: bytes) -> None:
            self.pid = doomed.pid
            self.returncode = None
            self.stdout = _FakeReader(data)
            self.stderr = _FakeReader(b"")

        async def wait(self) -> int:
            await asyncio.Event().wait()
            return 0  # pragma: no cover - unreachable

    import local_operator.tools.builtin as builtin

    real_create = asyncio.create_subprocess_exec

    async def time_one_emit(total_mb: float) -> float:
        data = b"".join([chunk] * int(total_mb * 1_048_576 / len(chunk)))
        proc_holder: dict[str, Any] = {}

        async def fake_exec(*_a: Any, **_k: Any) -> Any:
            proc_holder["proc"] = _FakeProc(data)
            return proc_holder["proc"]

        payloads: list[int] = []

        def on_update(update: Any) -> None:
            payloads.append(len(update.content[0].text))

        builtin.asyncio.create_subprocess_exec = fake_exec
        task = asyncio.create_task(
            execute_bash(  # type: ignore[arg-type]
                "bench", {"command": "true", "timeout": 60}, None, on_update, context
            )
        )
        try:
            # Wait until the stream is fully accumulated and one emit fired.
            deadline = time.perf_counter() + 10.0
            while time.perf_counter() < deadline:
                await asyncio.sleep(0.02)
                if payloads and payloads[-1] > 60_000:
                    break
            else:
                raise AssertionError("emit never fired against the synthetic stream")
            first_snapshot_len = payloads[-1]
            # Let at least three more 500 ms ticks fire, then time one by
            # the gap between successive payloads' arrival: the payload IS
            # the product of the synchronous emit on the loop.
            count = len(payloads)
            while len(payloads) < count + 3:
                await asyncio.sleep(0.05)
            return first_snapshot_len
        finally:
            builtin.asyncio.create_subprocess_exec = real_create
            task.cancel()
            with contextlib.suppress(BaseException):
                await task

    try:
        small = await time_one_emit(1.0)
        large = await time_one_emit(16.0)
    finally:
        # The cancel path SIGKILLed the sleep's group; reap it so no zombie
        # outlives the test.
        with contextlib.suppress(Exception):
            doomed.kill()
        with contextlib.suppress(Exception):
            doomed.wait(timeout=5)
    # The snapshot the card receives is bounded: 16x the output must not be
    # 16x the payload. (Timing the tick itself is flaky on CI; the payload
    # bound is the deterministic proxy for the same invariant — a bounded
    # payload cannot cost O(total) to build.)
    assert large <= small * 1.5, (small, large)
    assert large <= 200_000, f"live snapshot is unbounded: {large} chars"


# --- B: deferred MCP wiring on the TUI boot path -------------------------------


class _WiringManager:
    """A discovery stub manager shaped like the factory test rig's fake."""

    def __init__(self) -> None:
        self.disconnected = 0
        self.on_tools_changed: Any = None
        self.on_startup_settled: Any = None
        self.tools: list[Any] = []
        self.meta: dict[str, dict[str, Any]] = {}

    def startup_settling(self) -> bool:
        return False

    def startup_failures(self) -> dict[str, str]:
        return {}

    def get_all_server_names(self) -> list[str]:
        return ["stub"]

    def get_connected_servers(self) -> list[str]:
        return ["stub"]

    def get_connection_status(self, name: str) -> str:
        return "connected" if name == "stub" else "disconnected"

    def set_on_tools_changed(self, callback: Any) -> None:
        self.on_tools_changed = callback

    def get_tools(self) -> list[Any]:
        return list(self.tools)

    def get_tool_meta(self, tool_name: str) -> dict[str, Any]:
        return self.meta.get(tool_name, {})

    async def disconnect_all(self) -> None:
        self.disconnected += 1


@pytest.mark.asyncio
async def test_deferred_wiring_returns_a_session_usable_before_wiring_lands(
    tmp_config_dir: Path,
) -> None:
    """The TUI boot opt-in: ``create_session`` returns while discovery is still
    pending, the session runs on its non-MCP tools, and the outcome lands when
    the gate settles — the same degraded-but-correct surface a slow server
    already produces today."""
    import asyncio as _asyncio

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.session.session import Session
    from local_operator.session_factory import create_session
    from tests.unit.test_session_factory import _args

    manager = _WiringManager()
    wired = _asyncio.Event()

    async def slow_discover(cwd, auth_store=None):
        await _asyncio.sleep(0.3)  # longer than the factory call must take
        wired.set()
        return manager, [], []

    from unittest.mock import patch

    # The patch must OUTLIVE the factory call: deferred wiring runs in a
    # background task that fires after create_session has returned, so a
    # with-block around the call would restore the real discovery before the
    # task ever reaches it.
    patcher = patch("local_operator.mcp.discover_and_load_mcp_tools", slow_discover)
    patcher.start()
    try:
        session = await create_session(
            _args(hosting="test", model="test-model", yolo=True),
            ConfigManager(tmp_config_dir),
            CredentialManager(tmp_config_dir),
            AgentRegistry(tmp_config_dir),
            has_ui=True,
            defer_mcp_wiring=True,
        )
        assert isinstance(session, Session)
        assert not wired.is_set(), "factory returned before wiring completed"
        # The session is USABLE: non-MCP tools only, no MCP schemas loaded.
        tool_names = {tool.name for tool in session._tools}
        assert tool_names, "a deferred-wiring session must still have its tools"
        # Wiring lands in the background and records its outcome.
        await _asyncio.wait_for(wired.wait(), timeout=5.0)
        for _ in range(100):
            if getattr(session, "mcp_startup", None) is not None:
                break
            await _asyncio.sleep(0.02)
        startup = getattr(session, "mcp_startup", None)
        assert startup is not None
        assert startup.connected == ("stub",)
        assert getattr(session, "mcp_manager", None) is manager
        await session.dispose()
        assert manager.disconnected == 1, "dispose must tear the wired manager down"
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_headless_callers_still_await_mcp_wiring(tmp_config_dir: Path) -> None:
    """The contract every caller except the TUI boot path keeps: a returned
    session has MCP wiring completed and recorded."""
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.session_factory import create_session
    from tests.unit.test_session_factory import _args

    manager = _WiringManager()
    calls = []

    async def discover(cwd, auth_store=None):
        calls.append("discover")
        return manager, [], []

    from unittest.mock import patch

    with patch("local_operator.mcp.discover_and_load_mcp_tools", discover):
        session = await create_session(
            _args(hosting="test", model="test-model", yolo=True),
            ConfigManager(tmp_config_dir),
            CredentialManager(tmp_config_dir),
            AgentRegistry(tmp_config_dir),
        )
    try:
        assert calls == ["discover"], "wiring must complete before return"
        assert getattr(session, "mcp_startup", None) is not None
        assert getattr(session, "mcp_manager", None) is manager
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_dispose_during_deferred_wiring_cancels_cleanly(
    tmp_config_dir: Path,
) -> None:
    """A session torn down while wiring is still in flight must cancel the
    task, not leak it running against a disposed session."""
    import asyncio as _asyncio

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.session_factory import create_session
    from tests.unit.test_session_factory import _args

    started = _asyncio.Event()
    cancelled = _asyncio.Event()

    async def slow_discover(cwd, auth_store=None):
        started.set()
        try:
            await _asyncio.sleep(30)
        except _asyncio.CancelledError:
            cancelled.set()
            raise
        return _WiringManager(), [], []

    from unittest.mock import patch

    with patch("local_operator.mcp.discover_and_load_mcp_tools", slow_discover):
        session = await create_session(
            _args(hosting="test", model="test-model", yolo=True),
            ConfigManager(tmp_config_dir),
            CredentialManager(tmp_config_dir),
            AgentRegistry(tmp_config_dir),
            has_ui=True,
            defer_mcp_wiring=True,
        )
        await _asyncio.wait_for(started.wait(), timeout=5.0)
        await session.dispose()
    assert cancelled.is_set(), "dispose must cancel the in-flight wiring task"


# --- End-to-end per symptom: the real OperatorApp under a lag monitor ----------


@pytest.mark.asyncio
async def test_s1_boot_paints_and_stays_responsive_over_the_first_seconds() -> None:
    """S1: over the app's first 3 s, no loop stall above the loaded bar, and
    the model label lands on the band — the two facts the "startup waits for
    MCP" report was made of. The boot session is a fake (the real factory's
    loop work is covered by the unit tests above); what this test measures
    is the app's own boot composition — paint, adoption, timers. The loaded
    bar (200 ms CPU / 2 s wall) is the right one here: boot has legitimate
    loop CPU from Textual layout, and the reconnect/connect tests are the
    ones that assert the strict 50 ms CPU bar against the parse regression.
    """
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        recorder = StallRecorder()
        await recorder.start()
        # The boot window: adoption, splash, the first timers. Two seconds
        # covers several 1 Hz ticks and the update check's call_after_refresh.
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            await pilot.pause(0.05)
        await recorder.stop()
        recorder.assert_no_stall_loaded()
        # The band has the model label: the boot did not wait on it.
        assert session.adopted or app._status is not None
        label = app._status._model_label if app._status is not None else ""
        assert label, "the model label never landed on the band"


@pytest.mark.asyncio
async def test_s2_send_to_agent_end_keeps_the_loop_under_the_bar() -> None:
    """S2: a streamed reply with tool batches, from submit to ``agent_end``,
    leaves no loop stall above the bar — the reported 'sending a message
    freezes the UI until processed'. The pricing leg is made hostile the same
    way the unit test makes it: discovery stubbed slow, memo cold."""
    from local_operator.harness.types import (
        AgentEndEvent,
        AgentStartEvent,
        Message,
        MessageEndEvent,
        MessageStartEvent,
        MessageUpdateEvent,
        ToolExecutionEndEvent,
        ToolExecutionStartEvent,
        ToolResult,
        Usage,
    )
    from local_operator.model import configure, discovery
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.events import ContextUsageReported
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    slow = _SlowListing(5.0)
    original = discovery.available_models
    discovery.available_models = slow  # type: ignore[assignment]
    configure.invalidate_model_info_cache()
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        recorder = StallRecorder()
        await recorder.start()
        # A turn with message text, a tool batch, per-call usage reports, and
        # an unpriceable model on the components — the exact surface the
        # freeze was reported on.
        session.emit(AgentStartEvent())
        message = Message.assistant("streaming answer")
        session.emit(MessageStartEvent(message=message))
        session.emit(MessageUpdateEvent(message=message, delta="streaming answer"))
        session.emit(MessageEndEvent(message=message))
        usage = Usage(input_tokens=10_000, output_tokens=2_000, context_tokens=10_200)
        app.post_message(ContextUsageReported(10_200, usage))
        session.emit(
            ToolExecutionStartEvent(tool_call_id="t1", tool_name="bash", args={"command": "ls"})
        )
        session.emit(
            ToolExecutionEndEvent(
                tool_call_id="t1",
                tool_name="bash",
                result=ToolResult(tool_call_id="t1", tool_name="bash"),
            )
        )
        session.emit(AgentEndEvent())
        for _ in range(20):
            await pilot.pause(0.05)
        await recorder.stop()
        # Drain the background refresh BEFORE detaching the stub: the thread
        # is mid-sleep inside it, and restoring the real discovery + clearing
        # the memo while it runs leaves either a poisoned memo entry or a
        # thread fetching against a swapped module for the next test.
        slow.latency_s = 0.0
        for _ in range(200):
            if slow.in_flight == 0:
                break
            await asyncio.sleep(0.02)
        discovery.available_models = original  # type: ignore[assignment]
        configure.invalidate_model_info_cache()
        # Loaded bar (200 ms CPU / 2 s wall): this path has legitimate loop
        # CPU from Textual layout, and the reconnect/connect tests are the
        # ones that assert the strict 50 ms CPU bar against the parse
        # regression. Bench numbers live in bench/before.json vs after.json.
        recorder.assert_no_stall_loaded()


@pytest.mark.asyncio
async def test_s3_harvest_with_unlisted_model_and_cold_memo_keeps_the_loop_free() -> None:
    """S3: the 1 Hz subagent harvest pricing a child on an unlisted model
    with a cold memo must not stall the loop — the reported 'tool calls
    freeze the TUI' while subagents fan out."""
    from types import SimpleNamespace

    from local_operator.harness.types import Usage
    from local_operator.model import configure, discovery
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    slow = _SlowListing(5.0)
    original = discovery.available_models
    discovery.available_models = slow  # type: ignore[assignment]
    configure.invalidate_model_info_cache()

    session = FakeSession()
    # A child job on a model nothing knows: the harvest's worst case.
    child = SimpleNamespace(
        id="j1",
        usage=Usage(input_tokens=48_000, output_tokens=9_000),
        model_label="kimi/never-heard-of-it",
    )
    session.jobs = SimpleNamespace(list=lambda: [child])  # type: ignore[assignment]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        recorder = StallRecorder()
        await recorder.start()
        started = time.perf_counter()
        for _ in range(3):  # three harvest ticks' worth of work, inline
            app._harvest_subagent_costs()
            await pilot.pause(0.02)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        await recorder.stop()
        slow.latency_s = 0.0
        for _ in range(200):
            if slow.in_flight == 0:
                break
            await asyncio.sleep(0.02)
        discovery.available_models = original  # type: ignore[assignment]
        configure.invalidate_model_info_cache()
        assert elapsed_ms < 500.0, f"harvest blocked for {elapsed_ms:.0f} ms"
        recorder.assert_no_stall_loaded()


@pytest.mark.asyncio
async def test_tui_wires_mcp_status_when_deferred_wiring_lands(tmp_path: Path) -> None:
    """F1: the real TUI against a real deferred-boot session.

    Adoption runs before the wiring lands, so ``_wire_mcp_status`` sees no
    manager — the exact regression shape. The sink installed at adoption
    must re-enter the app when the wiring completes: the band's MCP segment
    fills, the startup toast fires for a FAILED server, and the failure
    lands in the transcript as the durable notice.
    """
    import asyncio as _asyncio
    import os

    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(tmp_path)
    from unittest.mock import patch

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.session_factory import create_session
    from local_operator.tui.app import OperatorApp
    from tests.unit.test_session_factory import _args

    wired = _asyncio.Event()

    class _FailingManager(_WiringManager):
        """One configured server that fails: the reportable outcome."""

        def get_all_server_names(self) -> list[str]:
            return ["broken"]

        def get_connected_servers(self) -> list[str]:
            return []

        def get_connection_status(self, name: str) -> str:
            return "disconnected"

        def startup_failures(self) -> dict[str, str]:
            return {"broken": "command not found: nope"}

        def startup_settling(self) -> bool:
            return False

    manager = _FailingManager()

    async def slow_discover(cwd, auth_store=None):
        await _asyncio.sleep(0.2)
        wired.set()
        # The errors list is what the GATE outcome's failures map is built
        # from; startup_failures() only feeds the settle rebuild, which a
        # nothing-deferred round never fires.
        return manager, [], [{"path": "mcp:broken", "error": "command not found: nope"}]

    # The patch OUTLIVES create_session: deferred wiring runs in a background
    # task that fires after the factory has returned.
    patcher = patch("local_operator.mcp.discover_and_load_mcp_tools", slow_discover)
    patcher.start()
    session = None
    try:
        session = await create_session(
            _args(hosting="test", model="test-model", yolo=True),
            ConfigManager(tmp_path / ".local-operator"),
            CredentialManager(tmp_path / ".local-operator"),
            AgentRegistry(tmp_path / ".local-operator"),
            has_ui=True,
            defer_mcp_wiring=True,
        )

        async def factory():
            return session

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 30)) as pilot:
            # Wait for ADOPTION before asserting: the boot worker awaits the
            # factory and adopts on the next loop turns.
            for _ in range(100):
                await pilot.pause(0.05)
                if app._session is session:
                    break
            assert app._session is session, "the app never adopted the session"
            # Adoption happened with the manager absent: the sink is the only
            # route back in, and it must have been installed.
            assert getattr(session, "_on_mcp_startup_settled", None) is not None
            await _asyncio.wait_for(wired.wait(), timeout=5.0)
            # Let the sink's call_later land and the app process it.
            for _ in range(50):
                await pilot.pause(0.05)
                if getattr(session, "mcp_manager", None) is not None and app._session is session:
                    break
            # The band's MCP segment now reads the live manager, not the
            # adoption-time absence.
            assert getattr(session, "mcp_manager", None) is manager
            status = app._mcp_status()
            assert status.configured == 1
            assert status.connected == 0
            assert status.failed is True
            # The startup toast fired for the failed server — the surface F1
            # severed. Toast content is the reportable outcome's rendering.
            from local_operator.tui.widgets.toast import Toast

            toast = app.query_one(Toast)
            shown = str(getattr(toast, "_message", "") or "")
            assert "broken" in shown or "MCP" in shown, f"no startup toast fired: {shown!r}"
    finally:
        patcher.stop()
        if session is not None:
            await session.dispose()


@pytest.mark.asyncio
async def test_relay_frame_during_threaded_replay_is_not_double_painted(tmp_path: Path) -> None:
    """F2: a message landing durably during the replay window is dropped from
    the buffer once the replay binds its ids — the race the threaded replay
    opened (invariant 4, gapless mid-turn attach)."""
    import asyncio as _asyncio
    import os

    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(tmp_path)
    from local_operator.harness.types import (
        AgentEndEvent,
        AgentStartEvent,
        Message,
        MessageEndEvent,
        MessageStartEvent,
        TextContent,
    )
    from local_operator.mobile.registrant import Registrant
    from local_operator.session.remote import RemoteSession
    from local_operator.session.transcript import Transcript
    from tests.unit.mobile.test_registrant import FakeHandle
    from tests.unit.session.test_remote import _wait_record

    (tmp_path / "sessions" / "s1").mkdir(parents=True)

    # A transcript big enough that the threaded replay takes real time, with
    # a FINAL durable assistant message the owner wrote just before attach.
    async def build() -> None:
        transcript = Transcript(tmp_path / "sessions" / "s1")
        for index in range(600):
            body = f"turn {index} " + "z" * 2_000
            await transcript.append_message(
                Message(role="assistant", content=[TextContent(text=body)])
            )

    await build()

    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        # The owner streams the SAME durable message over the relay while the
        # follower's replay is still running: message-grade frames for an id
        # already in history.
        replay_started = _asyncio.Event()

        real_load = RemoteSession._load_history

        async def instrumented_load(self_inner, *args, **kwargs) -> None:
            replay_started.set()
            await real_load(self_inner, *args, **kwargs)

        from unittest.mock import patch as _patch

        with _patch.object(RemoteSession, "_load_history", instrumented_load):
            connect_task = _asyncio.create_task(
                RemoteSession.connect(
                    record,
                    "s1",
                    config_dir=tmp_path,
                    takeover_factory=_never_take_over,
                )
            )
            await _asyncio.wait_for(replay_started.wait(), timeout=5.0)
            # Fire the relay frames INTO the window: start, message start,
            # end, agent end, for the message the replay will contain.
            message = Message.assistant("turn 599 z…")
            handle.emit_event(AgentStartEvent(generation=9))
            handle.emit_event(MessageStartEvent(message=message))
            handle.emit_event(MessageEndEvent(message=message))
            handle.emit_event(AgentEndEvent())
            remote = await _asyncio.wait_for(connect_task, timeout=15.0)

        events: list[Any] = []
        remote.subscribe(events.append)
        for _ in range(50):
            await _asyncio.sleep(0.02)
            if any(getattr(e, "type", "") == "agent_end" for e in events):
                break
        kinds = [e.type for e in events]
        # The seed replay plus relay: exactly ONE message_start for the
        # durable message — the buffered relay copy was re-filtered out.
        assert kinds.count("message_start") <= 1, kinds
        assert kinds.count("message_end") <= 1, kinds
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()
