"""Bounded lock and existing writer race evidence, never an operator config."""

from __future__ import annotations

import multiprocessing
import threading
from pathlib import Path

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager, read_config_values
from local_operator.config_writer import (
    ConfigWriteLockError,
    ConfigWriteLockTimeout,
    config_write_lock,
)


def _hold_process(root, ready, release) -> None:
    with config_write_lock(Path(root)):
        ready.set()
        release.wait(10)


def test_nested_same_root_is_reentrant_and_distinct_roots_independent(tmp_path: Path) -> None:
    root = tmp_path / "profile"
    with config_write_lock(root, wait=0):
        with config_write_lock(root / ".." / "profile", wait=0):
            with config_write_lock(tmp_path / "other-profile", wait=0):
                pass
    with config_write_lock(root, wait=0):
        pass


def test_other_thread_times_out_instead_of_deadlocking(tmp_path: Path) -> None:
    caught = []

    def contender() -> None:
        try:
            with config_write_lock(tmp_path, wait=0):
                pass
        except ConfigWriteLockTimeout as error:
            caught.append(error.code)

    with config_write_lock(tmp_path):
        thread = threading.Thread(target=contender)
        thread.start()
        thread.join(5)
        assert not thread.is_alive()
    assert caught == ["config-write-busy"]
    with config_write_lock(tmp_path, wait=0):
        pass


def test_kernel_lock_contends_across_processes_and_recovers(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_hold_process, args=(str(tmp_path), ready, release))
    process.start()
    try:
        assert ready.wait(10)
        with pytest.raises(ConfigWriteLockTimeout):
            with config_write_lock(tmp_path, wait=0):
                pass
        # Sharing HOME/install does not merge separate config lock domains.
        with config_write_lock(tmp_path / "another-profile", wait=0):
            pass
    finally:
        release.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0
    with config_write_lock(tmp_path, wait=0):
        pass


def test_symlink_lock_refused_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("untouched")
    (tmp_path / ".config-write.lock").symlink_to(target)
    with pytest.raises(ConfigWriteLockError):
        with config_write_lock(tmp_path):
            pass
    assert target.read_text() == "untouched"


def test_memory_only_config_update_creates_no_lock_or_file(tmp_path: Path) -> None:
    root = tmp_path / "profile"
    manager = ConfigManager(root)
    manager.update_config({"language": "en"}, write=False)
    assert not (root / ".config-write.lock").exists()
    assert not (root / "config.yml").exists()


def test_language_and_nested_hub_writes_preserve_each_other(tmp_path: Path, monkeypatch) -> None:
    first, second = ConfigManager(tmp_path), ConfigManager(tmp_path)
    first.set_config_value("untouched", {"value": "keep"})
    ready, release, language_started = threading.Event(), threading.Event(), threading.Event()
    original_reload = settings_io._reload_before_write
    paused = False
    errors = []

    def pause_first_reload(manager) -> None:
        nonlocal paused
        original_reload(manager)
        if manager is first and not paused:
            paused = True
            ready.set()
            assert release.wait(10)

    monkeypatch.setattr(settings_io, "_reload_before_write", pause_first_reload)

    def hub_write() -> None:
        try:
            settings_io._store(first, ("hub", "links", "synthetic-link", "subscribe"), False)
        except BaseException as error:
            errors.append(error)

    def language_write() -> None:
        try:
            language_started.set()
            settings_io.write_setting(second, settings_io.BY_KEY["language"], "en")
        except BaseException as error:
            errors.append(error)

    hub = threading.Thread(target=hub_write)
    language = threading.Thread(target=language_write)
    hub.start()
    try:
        assert ready.wait(10)
        language.start()
        assert language_started.wait(10)
    finally:
        release.set()
        hub.join(10)
        if language.ident is not None:
            language.join(10)
    assert not errors
    assert not hub.is_alive() and not language.is_alive()
    values = read_config_values(tmp_path)
    assert values is not None
    assert values["hub"]["links"]["synthetic-link"]["subscribe"] is False
    assert values["language"] == "en"
    assert values["untouched"] == {"value": "keep"}


def test_settings_callbacks_run_after_config_lock_is_released(tmp_path: Path, monkeypatch) -> None:
    manager = ConfigManager(tmp_path)
    results = []

    def callback(_manager) -> None:
        def other_thread() -> None:
            try:
                with config_write_lock(tmp_path, wait=0):
                    results.append("unlocked")
            except ConfigWriteLockTimeout:
                results.append("locked")

        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join(5)
        assert not worker.is_alive()

    monkeypatch.setattr(settings_io, "_notify_watcher", callback)
    settings_io.write_setting(manager, settings_io.BY_KEY["language"], "en")
    assert results == ["unlocked"]


def test_invalid_language_and_corrupt_config_do_not_mutate_anything(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path)
    manager.set_config_value("untouched", {"value": "keep"})
    before = manager.config_file.read_bytes()
    with pytest.raises(ValueError):
        settings_io.write_setting(manager, settings_io.BY_KEY["language"], "not-a-language")
    assert manager.config_file.read_bytes() == before
    manager.config_file.write_text("values: [broken")
    corrupt = manager.config_file.read_bytes()
    with pytest.raises(settings_io.ConfigUnreadableError):
        settings_io.write_setting(manager, settings_io.BY_KEY["language"], "en")
    assert manager.config_file.read_bytes() == corrupt
