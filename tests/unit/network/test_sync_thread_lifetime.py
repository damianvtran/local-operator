"""The sync slice's threads stop with the relay, and a stopped relay is collectable.

THE DEFECT (green-main stabilisation ledger, ``main-full-suite-green-1109``):
``sync.install`` registered the slice's ops but no ``on_stop`` hook, so the
``mesh-sync-watch`` and ``mesh-replica-refresh`` threads that a first push / pull
starts kept ticking after ``RelayServer.stop()``. Each holds its relay, and the
module registries hold each thread, so a stopped relay stayed pinned for the life
of the process: a slow leak on a long-running ``lop network serve`` and a
cross-test isolation source in CI.

Real path: a relay built the way ``lop network serve`` builds one (no ``root=``,
``serve_shaped_relay``), started, with both threads started by the slice's own
lazy starters, then stopped.
"""

from __future__ import annotations

import gc
import threading
import weakref
from pathlib import Path

import pytest

from local_operator.network import identity, sync
from tests.unit.network.test_relay_e2e import serve_shaped_relay

#: How long ``stop()`` may take to leave both threads dead. The threads wait on an
#: Event between ticks, so a stop is seen at once; the bound is the slice's own join.
STOP_BOUND_S = 5.0


def _started_relay(root: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    server = serve_shaped_relay(root, monkeypatch, identity=identity.mint(root, name="device-a"))
    server.start()
    return server


def _sync_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name in ("mesh-sync-watch", "mesh-replica-refresh")
    ]


def test_stop_ends_both_sync_threads_and_frees_the_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _started_relay(tmp_path, monkeypatch)
    watcher = sync.ensure_watcher(server)  # type: ignore[arg-type]
    refresher = sync.ensure_refresher(server)  # type: ignore[arg-type]
    assert watcher.is_alive() and refresher.is_alive()
    ref = weakref.ref(server)

    server.stop()  # type: ignore[attr-defined]

    # ``stop()``'s return is the moment the device is quiet (relay ``on_stop``
    # contract): no grace loop here, the hooks run inline.
    assert not watcher.is_alive(), "mesh-sync-watch outlived RelayServer.stop()"
    assert not refresher.is_alive(), "mesh-replica-refresh outlived RelayServer.stop()"
    del server, watcher, refresher
    gc.collect()
    assert ref() is None, "a stopped relay is still referenced"


def test_a_stopped_relay_is_collectable_after_its_threads_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collectability on its own, so it is not hidden behind the thread assertion.

    Even with both threads told to stop and joined by hand, the module registries
    (``_watchers``/``_refreshers``, keyed by ``id(server)``) held each thread and
    each thread its ``_server``: the relay was pinned for the process lifetime.
    """
    server = _started_relay(tmp_path, monkeypatch)
    watcher = sync.ensure_watcher(server)  # type: ignore[arg-type]
    refresher = sync.ensure_refresher(server)  # type: ignore[arg-type]
    ref = weakref.ref(server)
    server.stop()  # type: ignore[attr-defined]
    for thread in (watcher, refresher):
        thread.stop()
        thread.join(STOP_BOUND_S)
    del server, watcher, refresher, thread
    gc.collect()
    held_by = (
        sorted({type(r).__name__ for r in gc.get_referrers(ref())}) if ref() is not None else []
    )
    assert ref() is None, f"a stopped relay is still referenced by {held_by}"


def test_a_relay_whose_threads_never_started_stops_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the hook is a no-op for the common relay that never pushed or pulled."""
    server = _started_relay(tmp_path, monkeypatch)
    before = set(_sync_threads())
    ref = weakref.ref(server)
    server.stop()  # type: ignore[attr-defined]
    assert set(_sync_threads()) <= before
    del server
    gc.collect()
    assert ref() is None


def test_a_second_relay_keeps_its_threads_when_another_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registries are per relay: stopping one must not reach into another."""
    first = _started_relay(tmp_path / "a", monkeypatch)
    second = _started_relay(tmp_path / "b", monkeypatch)
    try:
        sync.ensure_watcher(first)  # type: ignore[arg-type]
        kept = sync.ensure_watcher(second)  # type: ignore[arg-type]
        kept_refresher = sync.ensure_refresher(second)  # type: ignore[arg-type]
        first.stop()  # type: ignore[attr-defined]
        assert kept.is_alive() and kept_refresher.is_alive()
        # And the survivor is still the one its starter hands back: nothing re-made it.
        assert sync.ensure_watcher(second) is kept  # type: ignore[arg-type]
    finally:
        second.stop()  # type: ignore[attr-defined]
    assert not kept.is_alive() and not kept_refresher.is_alive()


def test_a_thread_that_will_not_stop_is_reported_and_stays_findable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The bounded join: a survivor is named, not hidden, and its handle is kept.

    Popping the registry entry before knowing the thread stopped would leave a
    running thread nothing can reach (the same rule ``rollouts.stop_driver`` and
    ``definitions.stop_syncer`` follow).
    """
    server = _started_relay(tmp_path, monkeypatch)
    watcher = sync.ensure_watcher(server)  # type: ignore[arg-type]
    watcher.stop()
    watcher.join(STOP_BOUND_S)
    release = threading.Event()
    stuck = threading.Thread(target=release.wait, name="mesh-sync-watch", daemon=True)
    stuck.stop = lambda: None  # type: ignore[attr-defined]  # ignores its stop
    stuck.start()
    # A plain swap, NOT ``monkeypatch.setitem``: its teardown would put the original
    # entry back and re-pin this relay for the rest of the process (the leak under
    # test). The survivor is the registry's entry until ``stop()`` reaps it below.
    with sync._watcher_lock:  # noqa: SLF001
        sync._watchers[id(server)] = stuck  # type: ignore[assignment]  # noqa: SLF001
    try:
        with caplog.at_level("WARNING", logger=sync.logger.name):
            sync.stop_threads(server, timeout=0.2)  # type: ignore[arg-type]
            sync.stop_threads(server, timeout=0.2)  # type: ignore[arg-type]
        assert sync._watchers.get(id(server)) is stuck  # noqa: SLF001
        named = [r for r in caplog.records if "did not stop" in r.getMessage()]
        assert len(named) == 1, caplog.text  # once per survivor episode
    finally:
        release.set()
        stuck.join(STOP_BOUND_S)
        server.stop()  # type: ignore[attr-defined]
    # The survivor died, so the next stop reaped its entry.
    assert id(server) not in sync._watchers  # noqa: SLF001


def test_a_push_after_stop_does_not_restart_a_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late push/pull (a handler in flight as ``stop()`` ran) must not re-open it."""
    server = _started_relay(tmp_path, monkeypatch)
    ref = weakref.ref(server)
    server.stop()  # type: ignore[attr-defined]
    before = set(_sync_threads())
    sync.mark_replica_available(server, "synthetic-session")  # type: ignore[arg-type]
    late = sync.ensure_watcher(server)  # type: ignore[arg-type]
    assert not late.is_alive()
    assert set(_sync_threads()) <= before
    assert id(server) not in sync._watchers and id(server) not in sync._refreshers  # noqa: SLF001
    del server, late
    gc.collect()
    assert ref() is None


def test_a_finished_sync_thread_can_be_joined_and_queried() -> None:
    """3.12: an Event named ``_stop`` shadowed ``Thread._stop()``; join/is_alive raised."""
    server = type("S", (), {"root": None})()
    for cls in (sync.SyncWatcher, sync.ReplicaRefresher):
        thread = cls(server, settings=sync.SyncSettings())  # type: ignore[arg-type]
        thread.start()
        thread.stop()
        thread.join(STOP_BOUND_S)
        assert thread.is_alive() is False


# #3719 round 1 follow-ups -------------------------------------------------


@pytest.mark.parametrize("kind", ["watcher", "refresher"])
def test_a_thread_that_outlives_the_stop_frees_the_relay_when_it_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """R1-1: a tick wedged past the stop's join must not pin the relay forever.

    A ``_push`` to a wedged peer can run past ``SYNC_THREAD_STOP_JOIN_S`` (links close
    only after the hooks run), so ``stop_threads`` keeps the survivor's entry, and on
    the normal path nothing calls ``stop()`` again. RED before the fix: once the tick
    returned and the thread exited, its registry entry (and the once-per-episode
    warning row) still held the thread, the thread its ``_server``.
    """
    monkeypatch.setattr(
        sync.SyncSettings,
        "from_config",
        classmethod(lambda cls, root=None: sync.SyncSettings(tick_s=0.02)),
    )
    entered = threading.Event()
    release = threading.Event()

    def _wedged(self: object, *args: object, **kwargs: object) -> list[tuple[str, str]]:
        entered.set()
        release.wait(STOP_BOUND_S * 4)
        return []

    cls = sync.SyncWatcher if kind == "watcher" else sync.ReplicaRefresher
    monkeypatch.setattr(cls, "tick", _wedged)
    table = sync._watchers if kind == "watcher" else sync._refreshers  # noqa: SLF001
    server = _started_relay(tmp_path, monkeypatch)
    start = sync.ensure_watcher if kind == "watcher" else sync.ensure_refresher
    thread = start(server)  # type: ignore[arg-type]
    assert entered.wait(STOP_BOUND_S), "the wedged tick never ran"
    ref = weakref.ref(server)
    # THE ONE REAL STOP, its join shortened through the module (the relay's hook
    # looks ``stop_threads`` up at call time). No second stop follows: on the
    # normal path nothing calls one, which is the finding.
    real_stop_threads = sync.stop_threads
    monkeypatch.setattr(
        sync, "stop_threads", lambda srv, timeout=0.1: real_stop_threads(srv, timeout=timeout)
    )
    try:
        server.stop()  # type: ignore[attr-defined]
        assert thread.is_alive(), "the wedge did not hold: this proves nothing"
        assert table.get(id(server)) is thread, "the survivor's entry must be kept while alive"
    finally:
        release.set()
        thread.join(STOP_BOUND_S)
    assert not thread.is_alive()
    del server, thread
    gc.collect()
    held_by = (
        sorted({type(r).__name__ for r in gc.get_referrers(ref())}) if ref() is not None else []
    )
    assert ref() is None, f"the relay stayed pinned after its survivor exited: {held_by}"


def test_a_starter_on_a_stopped_relay_reads_no_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-2: the unstarted, unregistered stand-in is built from defaults.

    It exists so a late caller's ``.mark()`` is harmless; reading ``config.yml`` from
    disk for an instance that never ticks was wasted I/O on every late push.
    """
    server = _started_relay(tmp_path, monkeypatch)
    server.stop()  # type: ignore[attr-defined]
    reads: list[object] = []
    real = sync.SyncSettings.from_config

    def _counting(cls: type, root: object = None) -> object:
        reads.append(root)
        return real(root)  # type: ignore[arg-type]

    monkeypatch.setattr(sync.SyncSettings, "from_config", classmethod(_counting))

    watcher = sync.ensure_watcher(server)  # type: ignore[arg-type]
    refresher = sync.ensure_refresher(server)  # type: ignore[arg-type]

    assert reads == [], f"a stand-in read config {len(reads)} time(s)"
    assert not watcher.is_alive() and not refresher.is_alive()
    refresher.mark("abc", "available")  # still harmless
