"""The shared temp-then-replace writer (``local_operator/atomic_files.py``).

The class under test: a temp named from ``os.getpid()`` alone is shared by every
thread of a process, so two threads publishing one target write the SAME temp;
the first ``os.replace`` consumes it and the second raises
``FileNotFoundError`` (or publishes a file the other thread is still filling).
The race is made deterministic with a ``threading.Barrier`` inside
``os.replace``: both writers have finished writing their temp before either
renames it -- no sleeps, no retry loops.
"""

from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator import atomic_files
from local_operator.atomic_files import (
    atomic_symlink,
    atomic_write_bytes,
    atomic_write_text,
    discard_temp,
    unique_temp,
)

_REAL_REPLACE = os.replace


def _gate_replace(monkeypatch: pytest.MonkeyPatch, parties: int = 2) -> None:
    """Hold every ``os.replace`` until ``parties`` writers have reached it."""
    barrier = threading.Barrier(parties, timeout=10)

    def gated(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        barrier.wait()
        _REAL_REPLACE(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", gated)


def _race(writers: list[Callable[[], Any]]) -> list[Any]:
    """Run each writer on its own thread; return results/exceptions in order."""
    results: list[Any] = [None] * len(writers)

    def run(index: int) -> None:
        try:
            results[index] = writers[index]()
        except BaseException as error:  # noqa: BLE001 — the assertion reads it
            results[index] = error

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(writers))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive(), "a writer never returned"
    return results


def _pid_only_write(target: Path, text: str) -> None:
    """The shape every migrated site had: a temp named from the pid alone."""
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def _leftovers(directory: Path, target: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p != target)


def test_a_pid_only_temp_loses_a_same_process_race(tmp_path, monkeypatch) -> None:
    """NEGATIVE CONTROL: the old naming fails under the very barrier the helper passes.

    Proves the harness discriminates -- if this ever passes, the race test below
    is no longer exercising the race.
    """
    target = tmp_path / "doc.json"
    _gate_replace(monkeypatch)
    results = _race([lambda: _pid_only_write(target, "A"), lambda: _pid_only_write(target, "B")])
    assert any(isinstance(r, FileNotFoundError) for r in results), results


def test_two_threads_writing_one_target_both_land_whole(tmp_path, monkeypatch) -> None:
    target = tmp_path / "doc.json"
    payloads = [json.dumps({"writer": name, "pad": name * 4096}) for name in ("a", "b")]
    _gate_replace(monkeypatch)
    results = _race([lambda p=p: atomic_write_text(target, p) for p in payloads])
    assert results == [None, None]
    assert target.read_text(encoding="utf-8") in payloads
    assert _leftovers(tmp_path, target) == []


@pytest.mark.parametrize(
    "module_name",
    ["local_operator.code_requests.ledger", "local_operator.code_requests.cache"],
)
def test_a_migrated_writer_survives_a_same_process_race(tmp_path, monkeypatch, module_name) -> None:
    """A real call site, RED on the pid-only base (the loser returned False)."""
    import importlib

    write_json = importlib.import_module(module_name)._write_json
    target = tmp_path / "entry.json"
    payloads = [{"writer": name} for name in ("a", "b")]
    _gate_replace(monkeypatch)
    results = _race([lambda p=p: write_json(target, p) for p in payloads])
    assert results == [True, True]
    assert json.loads(target.read_text(encoding="utf-8")) in payloads
    assert _leftovers(tmp_path, target) == []


def test_temp_names_are_siblings_and_distinct_per_write(tmp_path) -> None:
    target = tmp_path / "sub" / "doc.json"
    names: set[Path] = set()
    lock = threading.Lock()

    def mint() -> None:
        for _ in range(50):
            with lock:
                names.add(unique_temp(target))

    _race([mint, mint, mint])
    assert len(names) == 150
    for name in names:
        assert name.parent == target.parent  # same directory: os.replace stays a rename
        assert name.name.startswith(f"doc.json.{os.getpid()}.")
        assert name.name.endswith(".tmp")


def test_hidden_and_marker_shapes(tmp_path) -> None:
    target = tmp_path / "current"
    assert unique_temp(target, hidden=True).name.startswith(f".current.{os.getpid()}.")
    # ``update._sweep_staging_links`` globs ``current.tmp-*``: the marker keeps that shape.
    staged = unique_temp(target, marker="tmp")
    assert list(tmp_path.glob("current.tmp-*")) == []
    staged.write_text("x", encoding="utf-8")
    assert list(tmp_path.glob("current.tmp-*")) == [staged]


def test_a_failed_replace_leaves_no_temp_and_the_target_untouched(tmp_path, monkeypatch) -> None:
    target = tmp_path / "doc.json"
    target.write_text("old", encoding="utf-8")

    def refuse(src: Any, dst: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="No space"):
        atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert _leftovers(tmp_path, target) == []


def test_a_failed_write_leaves_no_temp(tmp_path, monkeypatch) -> None:
    target = tmp_path / "doc.bin"

    def refuse(fd: int) -> None:
        raise OSError(5, "I/O error")

    monkeypatch.setattr(os, "fsync", refuse)
    with pytest.raises(OSError, match="I/O error"):
        atomic_write_bytes(target, b"payload", fsync=True)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("umask", [0o022, 0o077, 0o000])
def test_mode_holds_from_creation_whatever_the_umask(tmp_path, monkeypatch, umask) -> None:
    """A secret is 0600 from the first byte: the TEMP already has the mode."""
    target = tmp_path / "secret.json"
    seen: list[int] = []

    def inspect(src: Any, dst: Any) -> None:
        seen.append(stat.S_IMODE(os.stat(src).st_mode))
        _REAL_REPLACE(src, dst)

    monkeypatch.setattr(os, "replace", inspect)
    previous = os.umask(umask)
    try:
        atomic_write_text(target, "{}", mode=0o600)
    finally:
        os.umask(previous)
    assert seen == [0o600]
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_no_mode_matches_write_text(tmp_path) -> None:
    previous = os.umask(0o022)
    try:
        atomic_write_text(tmp_path / "a.txt", "x")
        (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / "a.txt").stat().st_mode) == stat.S_IMODE(
        (tmp_path / "b.txt").stat().st_mode
    )


def test_a_read_only_mode_still_writes(tmp_path) -> None:
    """``update._rebind_scripts`` republishes entries whose mode may lack owner-write."""
    target = tmp_path / "script"
    atomic_write_bytes(target, b"#!/bin/sh\n", mode=0o555)
    assert target.read_bytes() == b"#!/bin/sh\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o555


def test_fsync_only_when_asked(tmp_path, monkeypatch) -> None:
    calls: list[int] = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd)))
    atomic_write_bytes(tmp_path / "a", b"x")
    assert calls == []
    atomic_write_bytes(tmp_path / "b", b"x", fsync=True)
    assert len(calls) == 1


def test_text_is_written_verbatim(tmp_path) -> None:
    target = tmp_path / "t.txt"
    atomic_write_text(target, "line\nżółw\n")
    assert target.read_bytes() == "line\nżółw\n".encode("utf-8")


def test_atomic_symlink_replaces_an_existing_link(tmp_path) -> None:
    first, second = tmp_path / "gen1", tmp_path / "gen2"
    first.mkdir()
    second.mkdir()
    link = tmp_path / "current"
    atomic_symlink(link, first, target_is_directory=True)
    atomic_symlink(link, second, target_is_directory=True)
    assert os.readlink(link) == str(second)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["current", "gen1", "gen2"]


def test_neither_writer_can_displace_a_directory(tmp_path) -> None:
    """The session-deletion guard's argument for allow-listing these calls once."""
    directory = tmp_path / "session"
    directory.mkdir()
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(OSError):
        atomic_write_text(directory, "x")
    with pytest.raises(OSError):
        atomic_symlink(directory, tmp_path)
    assert (directory / "transcript.jsonl").read_text(encoding="utf-8") == "{}\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["session"]


def test_discard_temp_never_raises(tmp_path) -> None:
    discard_temp(tmp_path / "absent.tmp")
    directory = tmp_path / "dir"
    directory.mkdir()
    discard_temp(directory)  # unlink refuses a directory; swallowed, not removed
    assert directory.is_dir()


def test_library_atomic_is_the_same_implementation() -> None:
    from local_operator.library import atomic

    assert atomic.unique_temp is atomic_files.unique_temp
