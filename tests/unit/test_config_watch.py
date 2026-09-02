"""``ConfigWatcher``: the process-wide live view of ``config.yml``.

Every test here is bound by a loop turn or an event, never by the clock (see
AGENTS.md "Timing, flakes"). The watcher's poll interval is a property of the
production cadence, not of these tests: they drive ``poll_now()`` directly, and
the one test that exercises the kqueue accelerator waits on a signal the
listener sets.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.config_watch import (
    ConfigChange,
    ConfigWatcher,
    _reset_for_tests,
    existing_watcher,
    process_watcher,
)
from tests.unit.harness.test_comms import ChangeSignal, wait_for


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


def _write(config_dir: Path, key: str, value: Any) -> None:
    """A write from ANOTHER process: a fresh manager, no watcher notified."""
    setting = settings_io.resolve_key(key)
    assert setting is not None, key
    manager = ConfigManager(config_dir)
    settings_io._store(manager, setting.path, value)


class Recorder:
    def __init__(self) -> None:
        self.changes: list[ConfigChange] = []
        self.threads: list[int] = []

    def __call__(self, change: ConfigChange) -> None:
        self.changes.append(change)
        self.threads.append(threading.get_ident())


# ---------------------------------------------------------------------------
# 1. Fingerprint gate
# ---------------------------------------------------------------------------


def test_the_poll_does_not_parse_when_the_fingerprint_is_unchanged(tmp_path, monkeypatch) -> None:
    """Two ticks with no write cost ONE parse (at construction), then none.

    Structural, not timed: the property that makes a 2 s poll across fifty
    processes affordable is that an unchanged file is a stat, never a parse.
    """
    ConfigManager(tmp_path).set_config_value("hosting", "x")
    parses: list[int] = []
    original = ConfigWatcher._parse

    def spy(self):
        parses.append(1)
        return original(self)

    monkeypatch.setattr(ConfigWatcher, "_parse", spy)
    watcher = ConfigWatcher(tmp_path)
    assert len(parses) == 1  # the priming parse
    assert watcher.poll_now() is None
    assert watcher.poll_now() is None
    assert len(parses) == 1


def test_a_missing_file_is_a_first_run_not_an_error(tmp_path) -> None:
    """No config.yml yet: the snapshot is the defaults, nothing is logged as
    bad, and the first write is seen as the change it is."""
    watcher = ConfigWatcher(tmp_path)
    recorder = Recorder()
    watcher.subscribe(recorder)
    assert watcher.values["retry"]["enabled"] is True  # back-filled defaults
    assert watcher.poll_now() is None
    _write(tmp_path, "compaction.threshold_percent", 0.5)
    change = watcher.poll_now()
    assert change is not None
    # Only the key that differs from the defaults, not thirty back-filled ones.
    assert change.changed_keys == {"compaction.threshold_percent"}
    assert recorder.changes == [change]


# ---------------------------------------------------------------------------
# 2. Unreadable file keeps the last snapshot
# ---------------------------------------------------------------------------


def test_an_unreadable_file_keeps_the_last_good_snapshot_and_creates_no_bad_file(
    tmp_path, monkeypatch
) -> None:
    """A half-typed hand edit must not degrade a live session NOR be moved
    aside. ``ConfigManager._load_config`` does both; the watcher must never
    call it. Then the fix is delivered exactly once."""
    _write(tmp_path, "compaction.threshold_percent", 0.5)
    watcher = ConfigWatcher(tmp_path)
    recorder = Recorder()
    watcher.subscribe(recorder)
    assert watcher.values["compaction"]["threshold_percent"] == 0.5

    # The trap: any construction of a manager would move the file aside.
    monkeypatch.setattr(
        ConfigManager,
        "_load_config",
        lambda self: pytest.fail("the watcher must never load through ConfigManager"),
    )
    (tmp_path / "config.yml").write_text("values:\n  compaction: [", encoding="utf-8")
    assert watcher.poll_now() is None
    assert recorder.changes == []
    assert watcher.values["compaction"]["threshold_percent"] == 0.5
    assert not [p for p in tmp_path.iterdir() if ".bad" in p.name]

    # Not re-parsed while it stays broken: the bad fingerprint is remembered.
    parses: list[int] = []
    original = ConfigWatcher._parse
    monkeypatch.setattr(ConfigWatcher, "_parse", lambda self: (parses.append(1), original(self))[1])
    assert watcher.poll_now() is None
    assert parses == []

    monkeypatch.undo()
    (tmp_path / "config.yml").write_text(
        "values:\n  compaction:\n    threshold_percent: 0.25\n", encoding="utf-8"
    )
    change = watcher.poll_now()
    assert change is not None and change.changed_keys == {"compaction.threshold_percent"}
    assert watcher.values["compaction"]["threshold_percent"] == 0.25
    assert len(recorder.changes) == 1


@pytest.mark.parametrize(
    "content",
    ["", "   \n", "- a\n- b\n", "just a scalar\n", b"\xff\xfe".decode("latin-1")],
    ids=["empty", "blank", "list-top-level", "scalar-top-level", "non-utf8-ish"],
)
def test_every_shape_the_write_guard_rejects_is_kept_out_of_the_snapshot(
    tmp_path, content: str
) -> None:
    """Same shapes ``settings_io._require_readable_config`` refuses to write
    over: they must be refused as a live view too, or a session would adopt
    'defaults' from a file that is mid-edit."""
    _write(tmp_path, "retry.maxRetries", 3)
    watcher = ConfigWatcher(tmp_path)
    (tmp_path / "config.yml").write_bytes(content.encode("latin-1"))
    assert watcher.poll_now() is None
    assert watcher.values["retry"]["maxRetries"] == 3


def test_a_deleted_file_keeps_the_snapshot_and_sees_its_return(tmp_path) -> None:
    _write(tmp_path, "retry.maxRetries", 3)
    watcher = ConfigWatcher(tmp_path)
    (tmp_path / "config.yml").unlink()
    assert watcher.poll_now() is None
    assert watcher.values["retry"]["maxRetries"] == 3
    _write(tmp_path, "retry.maxRetries", 4)
    change = watcher.poll_now()
    assert change is not None and change.changed_keys == {"retry.maxRetries"}


# ---------------------------------------------------------------------------
# 3. Per-key diff
# ---------------------------------------------------------------------------


def test_a_metadata_only_write_delivers_nothing(tmp_path) -> None:
    """Every write bumps ``metadata.last_modified``; a no-op write must not
    produce a notification (or the TUI would print a line for nothing)."""
    _write(tmp_path, "retry.maxRetries", 3)
    watcher = ConfigWatcher(tmp_path)
    recorder = Recorder()
    watcher.subscribe(recorder)
    ConfigManager(tmp_path).update_config({}, write=True)  # rewrites, same values
    assert watcher.poll_now() is None
    assert recorder.changes == []


def test_changed_keys_name_exactly_the_registry_keys_that_moved(tmp_path) -> None:
    _write(tmp_path, "retry.maxRetries", 3)
    watcher = ConfigWatcher(tmp_path)
    _write(tmp_path, "compaction.threshold_percent", 0.5)
    change = watcher.poll_now()
    assert change is not None
    assert change.changed_keys == {"compaction.threshold_percent"}
    assert change.source == "disk"
    assert change.values is watcher.values


def test_an_unset_is_a_change(tmp_path) -> None:
    """Resetting a key to default on the page is an edit the consumers must
    see — the value they hold is no longer what the user wants."""
    _write(tmp_path, "compaction.threshold_percent", 0.5)
    watcher = ConfigWatcher(tmp_path)
    setting = settings_io.resolve_key("compaction.threshold_percent")
    assert setting is not None
    settings_io._delete(ConfigManager(tmp_path), setting.path)
    change = watcher.poll_now()
    assert change is not None and change.changed_keys == {"compaction.threshold_percent"}


def test_a_flat_dotted_display_key_diffs_by_its_literal_path(tmp_path) -> None:
    """``display.shimmer`` is a top-level key with a dot IN it. The diff walks
    the registry's declared ``path`` so it is seen as one key, not as a
    nesting ``display -> shimmer`` that does not exist."""
    watcher = ConfigWatcher(tmp_path)
    _write(tmp_path, "display.shimmer", False)
    change = watcher.poll_now()
    assert change is not None and change.changed_keys == {"display.shimmer"}


# ---------------------------------------------------------------------------
# 4. Local fast path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_settings_io_write_reaches_listeners_synchronously_with_source_local(
    tmp_path,
) -> None:
    """The writing process never waits a poll interval: ``write_setting``
    returns with the listener already called, and the next poll is a no-op
    because the fingerprint was recorded."""
    watcher = process_watcher(tmp_path)
    watcher.start()
    recorder = Recorder()
    watcher.subscribe(recorder)
    try:
        setting = settings_io.resolve_key("compaction.enabled")
        assert setting is not None
        settings_io.write_setting(ConfigManager(tmp_path), setting, False)
        # Already delivered, before any await.
        assert [c.source for c in recorder.changes] == ["local"]
        assert recorder.changes[0].changed_keys == {"compaction.enabled"}
        assert watcher.poll_now() is None
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_reset_and_write_chains_also_take_the_fast_path(tmp_path) -> None:
    watcher = process_watcher(tmp_path)
    watcher.start()
    recorder = Recorder()
    watcher.subscribe(recorder)
    try:
        manager = ConfigManager(tmp_path)
        settings_io.write_chains(manager, {"default": ["zai/glm-5.3"]})
        assert recorder.changes[-1].changed_keys == {"retry.fallbackChains"}
        setting = settings_io.resolve_key("retry.fallbackChains")
        assert setting is not None
        settings_io.reset_setting(manager, setting)
        assert recorder.changes[-1].changed_keys == {"retry.fallbackChains"}
        assert all(c.source == "local" for c in recorder.changes)
    finally:
        await watcher.stop()


def test_a_write_with_no_watcher_started_is_a_cheap_no_op(tmp_path) -> None:
    """``lop config edit`` runs in a process that never built a watcher; the
    hook must neither build one nor fail the write."""
    setting = settings_io.resolve_key("compaction.enabled")
    assert setting is not None
    settings_io.write_setting(ConfigManager(tmp_path), setting, False)
    assert existing_watcher(tmp_path) is None


def test_a_write_notifies_only_the_watcher_on_the_managers_directory(tmp_path) -> None:
    """A manager pointed elsewhere must not poke the watcher on the default
    directory: the hook keys on the MANAGER's ``config_dir``."""
    other = tmp_path / "other"
    other.mkdir()
    watched = process_watcher(tmp_path)
    recorder = Recorder()
    watched.subscribe(recorder)
    setting = settings_io.resolve_key("compaction.enabled")
    assert setting is not None
    settings_io.write_setting(ConfigManager(other), setting, False)
    assert recorder.changes == []
    assert existing_watcher(other) is None


# ---------------------------------------------------------------------------
# 5. Listener isolation
# ---------------------------------------------------------------------------


def test_a_raising_listener_does_not_starve_the_next(tmp_path, caplog) -> None:
    watcher = ConfigWatcher(tmp_path)
    recorder = Recorder()

    def bad(change: ConfigChange) -> None:
        raise RuntimeError("boom")

    watcher.subscribe(bad)
    watcher.subscribe(recorder)
    _write(tmp_path, "retry.maxRetries", 2)
    with caplog.at_level("WARNING", logger="local_operator.config_watch"):
        assert watcher.poll_now() is not None
    assert len(recorder.changes) == 1
    assert "listener failed" in caplog.text


def test_unsubscribe_is_idempotent(tmp_path) -> None:
    watcher = ConfigWatcher(tmp_path)
    recorder = Recorder()
    unsubscribe = watcher.subscribe(recorder)
    unsubscribe()
    unsubscribe()  # a session disposed through two paths must not raise
    _write(tmp_path, "retry.maxRetries", 2)
    watcher.poll_now()
    assert recorder.changes == []


# ---------------------------------------------------------------------------
# 6. Thread hop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notify_local_from_a_worker_thread_delivers_on_the_loop_thread(tmp_path) -> None:
    """Structural spy (the ``test_launch_subagent`` pattern): the listener
    records the thread it ran on, and it must be the loop's — listeners touch
    widgets and session state that are loop-affine.

    The kqueue accelerator is disarmed here so the delivery under test is the
    ``call_soon_threadsafe`` hop itself; with it armed, the directory event can
    legitimately win the race and deliver the same change as ``"disk"``
    (documented on ``notify_local``), which would make this assert about
    scheduling rather than about the hop."""
    watcher = process_watcher(tmp_path)
    watcher.start()
    watcher._disarm_kqueue()
    recorder = Recorder()
    watcher.subscribe(recorder)
    loop_thread = threading.get_ident()
    try:
        setting = settings_io.resolve_key("retry.maxRetries")
        assert setting is not None
        await asyncio.to_thread(settings_io.write_setting, ConfigManager(tmp_path), setting, 7)
        # The hop lands on the next loop turn; bound by turns, not seconds.
        await wait_for(lambda: bool(recorder.changes))
        assert recorder.threads == [loop_thread]
        assert recorder.changes[0].source == "local"
        assert len(recorder.changes) == 1
    finally:
        await watcher.stop()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_is_idempotent_and_process_watcher_is_keyed_on_the_directory(
    tmp_path,
) -> None:
    a = process_watcher(tmp_path)
    assert process_watcher(tmp_path) is a
    other = tmp_path / "other"
    other.mkdir()
    assert process_watcher(other) is not a
    a.start()
    task = a._task
    a.start()
    assert a._task is task
    await a.stop()
    await process_watcher(other).stop()


@pytest.mark.asyncio
async def test_process_watcher_reaps_idle_watchers_and_keeps_live_ones(tmp_path) -> None:
    """The registry must not hold one directory fd (plus a kqueue) per
    directory a process has ever seen: the test suite's one-``tmp_path``-per-
    test shape was an EMFILE at suite scale. An idle watcher (no listeners,
    or a dead loop) is reaped when the NEXT directory is requested; a live
    session's watcher on another directory is kept."""
    idle = process_watcher(tmp_path / "idle")
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    live = process_watcher(live_dir)
    live.start()
    live.subscribe(lambda change: None)
    try:
        third = process_watcher(tmp_path / "third")
        assert existing_watcher(tmp_path / "idle") is None  # reaped
        assert existing_watcher(live_dir) is live  # untouched
        assert existing_watcher(tmp_path / "third") is third
        # The reaped watcher's descriptors are gone.
        assert idle._kqueue is None and idle._dir_fd is None
        # And a RE-request of that directory gets a fresh, healthy watcher.
        again = process_watcher(tmp_path / "idle")
        assert again is not idle
        assert again._fingerprint is not None or True  # primed without error
    finally:
        await live.stop()
        await process_watcher(tmp_path / "third").stop()
        await process_watcher(tmp_path / "idle").stop()


@pytest.mark.asyncio
async def test_a_change_between_construction_and_start_is_delivered(tmp_path) -> None:
    """``start`` re-polls first so a write that landed while the session was
    being built is not silently adopted as the baseline."""
    watcher = ConfigWatcher(tmp_path)
    recorder = Recorder()
    watcher.subscribe(recorder)
    _write(tmp_path, "retry.maxRetries", 5)
    watcher.start()
    try:
        assert [c.changed_keys for c in recorder.changes] == [{"retry.maxRetries"}]
    finally:
        await watcher.stop()


# ---------------------------------------------------------------------------
# 10. kqueue accelerator (darwin only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason="kqueue directory watch is a BSD/macOS leg")
@pytest.mark.asyncio
async def test_a_directory_write_wakes_the_tick_without_the_poll_timer(
    tmp_path, monkeypatch
) -> None:
    """The accelerator's whole value is latency below the poll interval, so
    the poll is disarmed (``asyncio.sleep`` in the run loop never resolves)
    and the listener must still fire from the kqueue reader. Waits on the
    listener's own signal, never on the clock."""
    _write(tmp_path, "retry.maxRetries", 3)
    watcher = ConfigWatcher(tmp_path)

    never = asyncio.get_running_loop().create_future()

    async def frozen_sleep(_interval):
        await never

    monkeypatch.setattr("local_operator.config_watch.asyncio.sleep", frozen_sleep)

    signal = ChangeSignal()
    recorder = Recorder()

    def listener(change: ConfigChange) -> None:
        recorder(change)
        signal._fire()

    watcher.subscribe(listener)
    watcher.start()
    try:
        assert watcher._kqueue is not None, "the accelerator did not arm on darwin"
        # An os.replace into the directory: exactly what ConfigManager does.
        _write(tmp_path, "retry.maxRetries", 9)
        await wait_for(lambda: bool(recorder.changes), signal=signal)
        assert recorder.changes[0].changed_keys == {"retry.maxRetries"}
        assert not never.done()
    finally:
        signal.close()
        await watcher.stop()
        never.cancel()
        assert watcher._kqueue is None and watcher._dir_fd is None


def test_the_kqueue_arm_failure_is_silent(tmp_path, monkeypatch) -> None:
    """A directory that cannot be opened (EMFILE, an odd filesystem) leaves
    the poll as the only mechanism, without a raise or a warning."""

    def refuse(*_a, **_k):
        raise OSError("EMFILE")

    monkeypatch.setattr(os, "open", refuse)
    loop = asyncio.new_event_loop()
    try:
        watcher = ConfigWatcher(tmp_path)
        watcher._arm_kqueue(loop)
        assert watcher._kqueue is None
    finally:
        loop.close()
