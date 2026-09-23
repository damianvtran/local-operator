"""Keep-alive after the last viewer leaves (design-runtime-prewarm §5).

**The problem this covers.** A runtime was disposable the instant its last
viewer detached: ``_should_exit`` held for the 3 s drain and the process left.
So closing a conversation and re-opening it a minute later paid a COLD SPAWN
(measured on this fleet at 1.1-1.5 s of wall time and ~0.9-1.0 s of CPU) for
session state that was already in a live process's memory — and a re-open is
what a user does constantly.

**What is asserted here, in the order the mechanism runs.**

1. The policy: ``_drain_window_s`` gives a runtime a viewer has LEFT the
   keep-alive window, and only that population. A runtime nobody has looked at
   (an ``exec``, a wake delivery, a phone-only session) keeps the ordinary
   drain, which is the whole reason the window is keyed on the RECORD's
   ``detached_at`` rather than on "is this runtime idle".
2. Both keys are read through the registry's own path, at the moment a window
   is drawn — the ``#576`` failure (a key written nested and read flat) is
   invisible from every angle except this one.
3. The window is honoured by the reaper on a FAKE CLOCK, so a 300 s residency
   costs no wall time, and the build-refresh check still fires INSIDE it (see
   ``_reaper``'s own docstring: the drain loop is where that check runs for
   exactly this case).
4. The machine-wide LRU: with six detached idle records, the two least recently
   detached give up their windows and the four newest keep theirs — ordered by
   ``detached_at``, not by pid and not by insertion.
5. Every way the LRU can fail to answer keeps the runtime, including the one
   that matters — a scan that cannot see our own record is not evidence that we
   are outside the cap.
6. ``detached_at`` is stamped by the server at the viewer's departure, cleared
   when one returns, and survives the round trip to the discovery record on
   disk, where an older reader drops it rather than refusing the record.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime import registry
from local_operator.session.runtime.process import (
    DEFAULT_GRACE_S,
    DEFAULT_KEEP_ALIVE_MAX,
    DEFAULT_KEEP_ALIVE_SECONDS,
    KEEP_ALIVE_SCAN_S,
    _drain_window_s,
    _keep_alive_candidates,
    _keep_alive_max,
    _keep_alive_seconds,
    _keep_alive_victim,
    _reaper,
)
from local_operator.session.runtime.types import SessionRecord

_REAL_ASYNCIO = asyncio
_REAL_TIME = time

#: A pid no process on any platform this runs on can hold, for the ``stale``
#: arm of the candidate filter — the record is published, the owner is gone.
_DEAD_PID = 4_000_000


# -- fakes ---------------------------------------------------------------------


def _record(pid: int = 0, **overrides: Any) -> SessionRecord:
    values: dict[str, Any] = {
        "pid": pid,
        "kind": "daemon",
        "session_id": "keepalive01",
        "conversation_name": "keep-alive",
        "cwd": "/tmp",
        "model_label": "test/model",
        "control_port": 0,
        "control_key": "k",
    }
    values.update(overrides)
    return SessionRecord(**values)


class FakeRuntime:
    """The runtime as the reaper sees it: a record, a viewer count, idle work."""

    def __init__(
        self,
        *,
        pid: int = 4242,
        detached_at: float | None = None,
        attaches: int = 0,
        boot: Any = None,
    ) -> None:
        self._record = _record(pid=pid, detached=detached_at is not None)
        self._record.detached_at = detached_at
        self._attaches = attaches
        self._boot_build = boot
        self.retiring: list[tuple[str, str, bool, str]] = []
        self.closed = False

    def attach_clients(self) -> int:
        return self._attaches

    async def announce_retiring(
        self, reason: str, *, to: str = "", draining: bool = False, leaving: str = ""
    ) -> None:
        self.retiring.append((reason, to, draining, leaving))

    async def aclose(self) -> None:
        self.closed = True


class FakeHandle:
    def __init__(self, *, busy: bool = False) -> None:
        self._busy = busy
        self.disposed = False
        self.denied = False

    def is_busy(self) -> bool:
        return self._busy

    def next_wake_due_at(self) -> int | None:
        return None

    def may_refresh(self) -> str:
        return "busy" if self._busy else ""

    def _deny_pending_gates(self) -> None:
        self.denied = True

    async def dispose(self) -> None:
        self.disposed = True


class Clock:
    """A monotonic clock the test moves, so a 300 s window costs no wall time."""

    def __init__(self, start: float = 50_000.0) -> None:
        self.now = start

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeTime:
    """``process.time`` with a movable ``monotonic``; everything else delegated."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.now

    def __getattr__(self, name: str) -> Any:
        return getattr(_REAL_TIME, name)


class FakeAsyncio:
    """``process.asyncio`` whose ``sleep`` advances the fake clock by its argument."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def __getattr__(self, name: str) -> Any:
        return getattr(_REAL_ASYNCIO, name)

    async def sleep(self, seconds: float) -> None:
        self._clock.advance(seconds)
        await _REAL_ASYNCIO.sleep(0)


async def _pump_until(
    clock: Clock,
    seconds: float,
    *,
    start: float,
    until: Callable[[], bool] | None = None,
    limit_turns: int = 20_000,
) -> None:
    """Drive loop turns until the fake clock has advanced ``seconds``.

    Bounded by LOOP TURNS rather than by a wall-clock deadline (AGENTS.md,
    "Wait on the event, never on the clock"): the fake clock only moves when the
    reaper's own sleep does, so a turn count is the honest bound — and the limit
    exists so a reaper that stopped sleeping fails the test instead of hanging.

    ``until`` is why the bound has to be a predicate and not just a duration: a
    reaper that has LEFT no longer sleeps, so a pump waiting for the clock to
    reach a value past its exit would spin to the limit. Every caller passes the
    condition it expects to become true, and then asserts it did.
    """
    for _ in range(limit_turns):
        if until is not None and until():
            return
        if clock.now - start >= seconds:
            return
        await _REAL_ASYNCIO.sleep(0)
    raise AssertionError(
        f"the reaper stopped sleeping: {seconds}s of fake time needed, "
        f"{clock.now - start:.1f}s elapsed in {limit_turns} loop turns"
    )


@pytest.fixture
def keep_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped policy, stated explicitly rather than inherited from config."""
    monkeypatch.setattr(child_mod, "_keep_alive_seconds", lambda: float(DEFAULT_KEEP_ALIVE_SECONDS))
    monkeypatch.setattr(child_mod, "_keep_alive_max", lambda: DEFAULT_KEEP_ALIVE_MAX)


# -- 1. the window predicate ---------------------------------------------------


def test_a_never_viewed_runtime_keeps_the_ordinary_drain(keep_alive: None) -> None:
    """``exec``, a wake delivery, a phone-only session: nobody looked, so 3 s.

    This is the assertion that keeps the keep-alive from becoming a residency
    leak: the population it must NOT touch is every runtime spawned for one job,
    which is exactly the population the 3 s drain was written for.
    """
    assert _drain_window_s(DEFAULT_GRACE_S, FakeRuntime()) == (DEFAULT_GRACE_S, 0)


def test_a_viewed_runtime_gets_the_keep_alive_window(keep_alive: None) -> None:
    window_s, cap = _drain_window_s(DEFAULT_GRACE_S, FakeRuntime(detached_at=1.0))
    assert window_s == float(DEFAULT_KEEP_ALIVE_SECONDS)
    assert cap == DEFAULT_KEEP_ALIVE_MAX


def test_the_keep_alive_never_shortens_an_operators_drain(keep_alive: None) -> None:
    """``LOP_SESSION_GRACE_S`` widened by an operator still wins.

    The keep-alive exists to make a watched runtime outlive an unwatched one; a
    version that replaced a 600 s drain with 300 s would make it leave FIRST.
    """
    window_s, cap = _drain_window_s(600.0, FakeRuntime(detached_at=1.0))
    assert window_s == 600.0
    assert cap == DEFAULT_KEEP_ALIVE_MAX


def test_a_zero_window_disables_the_keep_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch, and it takes the registry scan with it (cap 0)."""
    monkeypatch.setattr(child_mod, "_keep_alive_seconds", lambda: 0.0)
    assert _drain_window_s(DEFAULT_GRACE_S, FakeRuntime(detached_at=1.0)) == (DEFAULT_GRACE_S, 0)


def test_a_broken_record_reads_as_never_viewed(keep_alive: None) -> None:
    """A foreign type where a timestamp belongs must not reach arithmetic."""

    class Rotten:
        detached_at = "yesterday"

    class Runtime:
        _record = Rotten()

    assert _drain_window_s(DEFAULT_GRACE_S, Runtime()) == (DEFAULT_GRACE_S, 0)


# -- 2. both keys, through the registry's own path -------------------------------


def _write(key: str, value: Any, root: Path) -> None:
    """Write one setting the way the product writes it (``/settings``, the CLI)."""
    settings_io.write_setting(ConfigManager(root), settings_io.BY_KEY[key], value)


def test_the_keys_are_read_through_the_path_the_registry_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write through ``settings_io``, read through the consumer.

    A consumer reading the flat key ``"runtime.keep_alive_seconds"`` instead of
    the nested tuple would read a key nothing ever writes and fall back to its
    default forever — the ``#576`` reaper toggle, which looked correct from
    every angle but the value's route to the code.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _write("runtime.keep_alive_seconds", 45, tmp_path)
    _write("runtime.keep_alive_max", 2, tmp_path)
    assert _keep_alive_seconds() == 45.0
    assert _keep_alive_max() == 2
    # And the value is where the page says it is, not at the top level.
    assert ConfigManager(tmp_path).get_config().values["runtime"]["keep_alive_seconds"] == 45


def test_the_defaults_are_the_shipped_numbers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    assert _keep_alive_seconds() == float(DEFAULT_KEEP_ALIVE_SECONDS) == 300.0
    assert _keep_alive_max() == DEFAULT_KEEP_ALIVE_MAX == 4


def test_an_unreadable_config_uses_the_defaults_and_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config file that cannot be parsed must not decide residency.

    Falling back to the default (rather than to 0) is deliberate: the ordinary
    case is a machine whose config is fine, and reading "unreadable" as "turn
    the feature off" would silently give the operator a different product the
    first time a YAML file was hand-broken.
    """
    (tmp_path / "config.yml").write_text("values: [this is not a mapping\n", encoding="utf-8")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    assert _keep_alive_seconds() == float(DEFAULT_KEEP_ALIVE_SECONDS)
    assert _keep_alive_max() == DEFAULT_KEEP_ALIVE_MAX


def test_zero_seconds_is_read_as_off_and_not_as_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``0`` is the documented opt-out, so it must not be "refused as invalid"."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _write("runtime.keep_alive_seconds", 0, tmp_path)
    assert _keep_alive_seconds() == 0.0


def test_a_cap_of_zero_is_read_as_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cap of 0 would evict every window the instant it opened.

    So 0 reads as 1 rather than as "no runtimes": "do not keep anything warm"
    has a spelling that says so (``keep_alive_seconds: 0``), and a COUNT of zero
    silently meaning the same thing is how a knob stops being readable.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # Bypass the registry's own minimum (1) to state the consumer's floor, which
    # is the half that has to hold against a hand-edited config file.
    (tmp_path / "config.yml").write_text(
        "values:\n  runtime:\n    keep_alive_max: 0\n", encoding="utf-8"
    )
    assert _keep_alive_max() == 1


def test_the_settings_registry_declares_both_keys() -> None:
    """The registry half of "Adding a configuration key" (AGENTS.md).

    The consumer defaults are asserted by ``test_settings_io``'s own
    ``_consumer_defaults`` sweep; what is pinned here is that both keys are
    discoverable at all — a key that exists only in the code that reads it can
    only be set by someone who already knows it exists.
    """
    for key, default, minimum in (
        ("runtime.keep_alive_seconds", DEFAULT_KEEP_ALIVE_SECONDS, 0),
        ("runtime.keep_alive_max", DEFAULT_KEEP_ALIVE_MAX, 1),
    ):
        setting = settings_io.BY_KEY[key]
        assert setting.section == "runtime"
        assert setting.default == default
        assert setting.minimum == minimum
        assert setting.path[0] == "runtime", "path and key namespace must agree"


# -- 3. the reaper on a fake clock ------------------------------------------------


def _fake_clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(child_mod, "time", FakeTime(clock))
    monkeypatch.setattr(child_mod, "asyncio", FakeAsyncio(clock))
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 1.0)  # one fake second per tick
    monkeypatch.setattr(child_mod, "_build_changed", lambda _boot: None)
    return clock


@pytest.mark.asyncio
async def test_the_keep_alive_window_is_honoured_and_the_build_check_still_fires(
    monkeypatch: pytest.MonkeyPatch, keep_alive: None
) -> None:
    """300 s of fake time, then the ordinary idle exit — on a 3 s base grace.

    The negative half is the load-bearing one: the runtime must still be
    resident long past the 3 s it would have left on before this change. The
    positive half is what stops a residency policy from becoming a leak: an exit
    that never arrives fails this test too.
    """
    clock = _fake_clock(monkeypatch)
    runtime = FakeRuntime(detached_at=time.time() - 1.0)
    monkeypatch.setattr(child_mod, "_keep_alive_candidates", lambda: [(999.0, runtime._record.pid)])
    handle = FakeHandle()
    stop = asyncio.Event()
    start = clock.now
    task = asyncio.ensure_future(_reaper(handle, runtime, stop))

    await _pump_until(clock, 100.0, start=start, until=stop.is_set)
    assert not stop.is_set(), "the runtime left inside its keep-alive window"
    assert not handle.disposed

    await _pump_until(clock, 295.0, start=start, until=stop.is_set)
    assert not stop.is_set(), "the window ended at 295 s, not 300"

    await _pump_until(clock, 310.0, start=start, until=stop.is_set)
    assert stop.is_set(), "the keep-alive never ended: the window is a leak"
    assert handle.disposed and runtime.closed
    assert await task is True


@pytest.mark.asyncio
async def test_a_viewer_that_comes_and_goes_inside_one_tick_still_starts_the_window(
    monkeypatch: pytest.MonkeyPatch, keep_alive: None
) -> None:
    """The window belongs to the STATE, not to the sampling interval.

    The drain's deadline is drawn from the state at draw time, so a viewer that
    attaches and disposes inside one ``REAP_CHECK_S`` tick never makes
    ``_should_exit`` go false — the tick that would have cancelled the drain saw
    a viewer, and by the next one the viewer is gone. Left alone, that runtime
    exits on the 3 s grace it was drawn with, having just been looked at.
    Measured before this guard: ~1 run in 8 of a re-open bench took the cold
    spawn instead of the live attach, with the record's stamp present the whole
    time.

    Here the stamp appears one fake second into a drain drawn WITHOUT one, and
    the assertion is that the runtime then holds the keep-alive window rather
    than the 3 s it started with.
    """
    clock = _fake_clock(monkeypatch)
    # A BASE GRACE WIDER THAN ONE TICK, so the drain drawn before the viewer can
    # still be running when the state changes: one fake second per tick, and the
    # first draw lands at t=1.
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "10")
    runtime = FakeRuntime()  # no viewer yet: the drain is drawn on the base grace
    monkeypatch.setattr(child_mod, "_keep_alive_candidates", lambda: [(999.0, runtime._record.pid)])
    handle = FakeHandle()
    stop = asyncio.Event()
    start = clock.now
    task = asyncio.ensure_future(_reaper(handle, runtime, stop))

    await _pump_until(clock, 2.0, start=start, until=stop.is_set)
    assert not stop.is_set(), "the base grace ended before the state change"
    # A viewer arrives, is seen, and leaves — all inside one tick, AFTER the
    # drain was drawn on the pre-viewer state (which is the point: the tick that
    # would have cancelled that drain never saw the viewer).
    runtime._record.detached_at = time.time()
    runtime._record.detached = True

    await _pump_until(clock, 30.0, start=start, until=stop.is_set)
    assert not stop.is_set(), "the runtime left on the grace it drew before the viewer arrived"

    await _pump_until(clock, 320.0, start=start, until=stop.is_set)
    assert stop.is_set(), "the re-drawn keep-alive window never ended"
    assert await task is True


@pytest.mark.asyncio
async def test_the_build_refresh_fires_inside_the_keep_alive_window(
    monkeypatch: pytest.MonkeyPatch, keep_alive: None
) -> None:
    """The reaper's drain loop is where the refresh check must keep running.

    ``refresh_check`` runs on both loop levels precisely so a minutes-long grace
    does not starve it (see ``_reaper``'s own docstring); the keep-alive is the
    longest grace in this codebase, so a runtime sitting inside it is the case
    that would starve if this change had got the structure wrong. Here the build
    moves under the runtime inside the window and the runtime retires — long
    before the window would have ended.
    """
    from local_operator.update import BuildStamp

    new_build = BuildStamp(version="9.9.9", source_ref="deadbeef1234567")
    clock = _fake_clock(monkeypatch)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 5.0)
    monkeypatch.setenv("LOP_BUILD_STAGGER_S", "0.5")
    monkeypatch.setattr(child_mod, "_build_changed", lambda _boot: new_build)

    runtime = FakeRuntime(detached_at=time.time() - 1.0, boot=BuildStamp(version="1.0.0"))
    monkeypatch.setattr(child_mod, "_keep_alive_candidates", lambda: [(999.0, runtime._record.pid)])
    handle = FakeHandle()
    stop = asyncio.Event()
    start = clock.now
    task = asyncio.ensure_future(_reaper(handle, runtime, stop))

    await _pump_until(clock, 60.0, start=start, until=stop.is_set)
    assert stop.is_set(), "a newer build on disk did not end the keep-alive window"
    assert clock.now - start < 300.0, "it waited out the window instead of refreshing"
    assert runtime.retiring and runtime.retiring[0][0] == "stale-build", runtime.retiring
    assert handle.disposed and runtime.closed
    assert await task is True


@pytest.mark.asyncio
async def test_an_unviewed_runtime_still_exits_on_the_base_grace(
    monkeypatch: pytest.MonkeyPatch, keep_alive: None
) -> None:
    """The other side of the same mechanism: no stamp, no window, 3 s."""
    clock = _fake_clock(monkeypatch)
    runtime = FakeRuntime()  # nobody ever attached
    handle = FakeHandle()
    stop = asyncio.Event()
    start = clock.now
    task = asyncio.ensure_future(_reaper(handle, runtime, stop))

    await _pump_until(clock, DEFAULT_GRACE_S + 5.0, start=start, until=stop.is_set)
    assert stop.is_set(), "an unwatched runtime held a keep-alive window"
    assert await task is True


@pytest.mark.asyncio
async def test_the_lru_is_not_consulted_for_an_unviewed_runtime(
    monkeypatch: pytest.MonkeyPatch, keep_alive: None
) -> None:
    """No viewer ever left, so no scan: the registry is not read at all."""
    clock = _fake_clock(monkeypatch)
    scans: list[int] = []

    def candidates() -> list[tuple[float, int]]:
        scans.append(1)
        return []

    monkeypatch.setattr(child_mod, "_keep_alive_candidates", candidates)
    runtime = FakeRuntime()
    handle = FakeHandle()
    stop = asyncio.Event()
    start = clock.now
    task = asyncio.ensure_future(_reaper(handle, runtime, stop))
    await _pump_until(clock, DEFAULT_GRACE_S + 5.0, start=start, until=stop.is_set)
    assert await task is True
    assert scans == []


@pytest.mark.asyncio
async def test_a_lru_victim_leaves_before_its_window_ends(
    monkeypatch: pytest.MonkeyPatch, keep_alive: None
) -> None:
    """The cap's whole purpose: the fleet settles at N, not at "as many as opened"."""
    clock = _fake_clock(monkeypatch)
    runtime = FakeRuntime(detached_at=1.0)
    # Four more recently detached peers, so this one is fifth of five.
    monkeypatch.setattr(
        child_mod,
        "_keep_alive_candidates",
        lambda: [(1.0, runtime._record.pid)] + [(10.0 + i, 9000 + i) for i in range(4)],
    )
    handle = FakeHandle()
    stop = asyncio.Event()
    start = clock.now
    task = asyncio.ensure_future(_reaper(handle, runtime, stop))

    # The first scan lands at KEEP_ALIVE_SCAN_S, so a few fake seconds is all it
    # takes — and the assertion is that it is far short of the window.
    await _pump_until(clock, KEEP_ALIVE_SCAN_S + 5.0, start=start, until=stop.is_set)
    assert stop.is_set(), "the least recently detached runtime kept its window"
    assert clock.now - start < 60.0, "it waited out the whole window first"
    assert handle.disposed and runtime.closed
    assert await task is True


# -- 4. the LRU itself, with six fake records -------------------------------------


def _victim(stamp: float, cap: int = 4, pid: int = 4242) -> bool:
    return _keep_alive_victim(FakeRuntime(pid=pid, detached_at=stamp), cap)


def test_six_detached_records_keep_the_four_most_recent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Six candidates, cap 4: the four most recently detached keep their windows.

    Stamps ascend with detach order, so 60..30 are the four newest and 20 and 10
    are the two that must give up. EVERY record is asked in turn, with OUR pid
    moved onto it, so the test states the whole partition rather than one row of
    it — and so the runtime always appears in the population it is judged
    against.
    """
    stamps = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    def population_with_ours(index: int) -> list[tuple[float, int]]:
        return [(stamp, 4242 if i == index else 1000 + i) for i, stamp in enumerate(stamps)]

    for index, stamp in enumerate(stamps):
        monkeypatch.setattr(
            child_mod, "_keep_alive_candidates", lambda i=index: population_with_ours(i)
        )
        # cap 4 keeps indices 2..5: the records detached most recently.
        assert _victim(stamp) is (index < 2), f"the record detached at {stamp} was mis-ranked"


def test_the_order_is_detached_at_and_not_pid_or_insertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sorting by anything else evicts the conversation the user just left.

    The pids here run OPPOSITE to the stamps, so a pid-ordered implementation —
    or one that keeps the list in scan order — passes every other test in this
    file and fails this one.
    """
    population = [(100.0, 1), (300.0, 2), (200.0, 4242), (400.0, 4)]
    monkeypatch.setattr(child_mod, "_keep_alive_candidates", lambda: list(population))
    # Cap 2 keeps stamps 400 and 300.
    assert _victim(200.0, cap=2) is True, "pid 4242 was ranked by pid, not by stamp"
    assert _victim(400.0, cap=2, pid=4) is False
    assert _victim(100.0, cap=2, pid=1) is True


def test_a_boundary_tie_is_broken_by_pid_rather_than_by_luck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A total order, so the same population always answers the same way.

    Stamps tie, so the pid decides: the order is ``(detached_at, pid)``
    descending, which means the LARGER pid is kept. That is arbitrary and it is
    the point — the alternative is an order that depends on scan order or dict
    iteration, where the same six records give different answers on different
    scans. The design accepts the resulting N±1 (see ``DEFAULT_KEEP_ALIVE_MAX``)
    precisely because the tie-break cannot be made meaningful.
    """
    population = [(500.0, 7), (500.0, 8), (500.0, 4242)]
    monkeypatch.setattr(child_mod, "_keep_alive_candidates", lambda: list(population))
    assert _victim(500.0, cap=2, pid=7) is True, "the tie must resolve deterministically"
    assert _victim(500.0, cap=2, pid=8) is False
    assert _victim(500.0, cap=2, pid=4242) is False


def test_a_runtime_that_cannot_see_its_own_record_keeps_its_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead instrument returns a reading, not an error.

    Our own record missing from the scan (not yet published, unreadable, reaped)
    says nothing about where we sit in the LRU. Exiting on it would turn a
    moment of unreadability into an eviction — and the cost of the other
    direction is bounded, because the peers that CAN see themselves still
    preempt themselves.
    """
    monkeypatch.setattr(child_mod, "_keep_alive_candidates", lambda: [(10.0, 9001), (9.0, 9002)])
    assert _victim(1.0) is False


def test_a_registry_that_will_not_read_keeps_the_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> list[tuple[float, int]]:
        raise OSError("run dir gone")

    monkeypatch.setattr(child_mod, "_keep_alive_candidates", boom)
    assert _victim(1.0) is False


def test_a_runtime_with_no_stamp_is_never_a_victim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(child_mod, "_keep_alive_candidates", lambda: [(10.0, 9001), (9.0, 9002)])
    assert _keep_alive_victim(FakeRuntime(pid=4242), 1) is False


# -- 5. the candidate population, read from a real registry ------------------------


def test_candidates_are_detached_idle_live_records_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The filter, against the real ``registry.scan`` over a real run directory.

    Four published records, three of which must not hold a cap slot: one being
    WATCHED (``detached=False``), one BUSY, and one whose owner is gone
    (``stale``). Only the live detached idle one may come back — and it must come
    back with its stamp read off the FILE, which is the round trip that also
    proves the field is serialized at all.

    One record per pid, because the run directory is KEYED by pid: four records
    sharing a pid would be one file. Liveness is stubbed at the registry's own
    probe (``registry.pid_alive``) rather than arranged with four real child
    processes — the classification is the registry's subject, and what is under
    test here is the FILTER over its verdicts.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    alive = {9001, 9002, 9003}
    monkeypatch.setattr(registry, "pid_alive", lambda pid, *, check_zombie=False: pid in alive)

    def publish(pid: int, stamp: float, **flags: Any) -> None:
        flags.setdefault("detached", True)
        record = _record(pid=pid, **flags)
        record.detached_at = stamp
        registry.publish(record, tmp_path)

    publish(9001, 111.0)  # the one that counts
    publish(9002, 222.0, detached=False)  # somebody is watching it
    publish(9003, 333.0, busy=True)  # work in flight
    publish(_DEAD_PID, 444.0)  # the owner is gone

    assert _keep_alive_candidates() == [(111.0, 9001)]


# -- 6. the record field, end to end ----------------------------------------------


def test_an_older_reader_drops_the_stamp_rather_than_refusing_the_record() -> None:
    """The compatibility claim, verified here rather than taken on trust.

    ``from_json`` filters to ``SessionRecord``'s own fields, which is what makes
    a mixed-version fleet safe: an older runtime writes a record with no
    ``detached_at`` and drops the key a newer one wrote, and neither refuses the
    other's file. A field that instead moved ``PROTOCOL_VERSION`` would make
    every older peer refuse a record it can in fact use.
    """
    payload = _record(pid=7).to_json()
    payload["detached_at"] = 1234.5
    payload["a_field_from_the_future"] = {"nested": True}
    parsed = SessionRecord.from_json(payload)
    assert parsed.detached_at == 1234.5
    assert parsed.pid == 7

    without = _record(pid=7).to_json()
    without.pop("detached_at")
    older = SessionRecord.from_json(without)
    assert older.detached_at is None
    assert older.detached is False


@pytest.mark.asyncio
async def test_the_server_stamps_on_the_departure_and_clears_it_on_a_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write side of ``detached_at``, through the production server.

    A real Session under the production ``RuntimeServer``, with the surface
    reader swapped for the two states a viewer can be in. What is asserted:
    nothing is stamped for a runtime nobody has looked at; the stamp appears at
    the 1->0 transition; it is CLEARED when a viewer comes back (a watched
    runtime must not be a keep-alive candidate); and the value reaches the
    discovery record on disk through the publisher, not only the in-memory
    object.
    """
    from local_operator.session.runtime.registry import RecordPublisher
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.e2e.harness import ScriptedStream, build_session, text_turn

    directory = tmp_path / "sessions" / "keepalive"
    directory.mkdir(parents=True, exist_ok=True)
    session = build_session(directory, ScriptedStream([text_turn("ok")]))
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    try:
        # A server that is never started has no publisher; wire one so the
        # republish path writes a real record — the same object the serve thread
        # installs (``RecordPublisher(self._record, self._config_root)``).
        server._config_root = tmp_path
        server._publisher = RecordPublisher(server._record, tmp_path)

        surfaces: set[str] = set()
        monkeypatch.setattr(server, "_visible_attach_surfaces", lambda: set(surfaces))

        server._republish_detached()  # deduped: nobody has ever attached
        assert server._record.detached_at is None

        surfaces = {"terminal"}  # a viewer arrives
        server._republish_detached()
        assert server._record.detached_at is None, "a watched runtime is not a candidate"

        surfaces = set()  # and leaves
        server._republish_detached()
        stamped = server._record.detached_at
        assert stamped is not None and stamped > 0

        def on_disk() -> SessionRecord:
            found = [
                record
                for record, _state in registry.scan(tmp_path, reap=False)
                if record.pid == server._record.pid
            ]
            assert found, "the server's record was not published"
            return found[0]

        assert on_disk().detached_at == stamped, "the stamp did not reach the record file"

        surfaces = {"terminal"}  # a viewer returns
        server._republish_detached()
        assert server._record.detached_at is None, "the stamp must be cleared on attach"
        assert on_disk().detached_at is None
    finally:
        await session.dispose()
