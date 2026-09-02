"""The session as the index's single writer: after every persist, on every
open, self-healing, and never able to break wake persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from local_operator.harness.types import ModelSpec, StreamEndEvent
from local_operator.harness.wake import WakeSchedule
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.wakes import install as wake_install
from local_operator.wakes import store as wake_store
from tests.unit.session.test_session import ScriptedStream

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


def _open(tmp_path: Path, session_id: str = "sess") -> Session:
    return Session(
        model=MODEL,
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=Transcript(tmp_path / session_id),
        system_blocks_provider=lambda: [],
        cwd="/work/here",
    )


def _schedule(sid: str = "w1", *, due: int | None = None) -> WakeSchedule:
    # The default due time MUST be future-derived. These tests assert on the
    # index file surviving between set_wake_schedules and dispose, and a
    # schedule that is already overdue arms the scheduler's MIN_ARM_MS timer
    # (~25 ms): when that timer wins the race against dispose()'s awaits, the
    # pump fires the one-shot, retires it, and persists the now-EMPTY list —
    # which removes the index file the test is about to read. Nothing here
    # asserts firing behaviour, so an hour out makes every test in this file
    # deterministic instead of racing dispose on a loaded host.
    if due is None:
        due = int(time.time() * 1000) + 3_600_000
    return WakeSchedule(id=sid, message="check in", next_due_at=due, created_at=1_700_000_000_000)


@pytest.mark.asyncio
async def test_persist_writes_index_entry_matching_transcript(
    tmp_path: Path, config_dir: Path
) -> None:
    session = _open(tmp_path)
    try:
        await session.set_wake_schedules([_schedule()])
        entry = wake_store.read_entry(config_dir, session.session_id)
        assert entry is not None
        assert entry["cwd"] == "/work/here"
        transcript_rows = session._transcript.latest_custom("wake_schedules")
        assert transcript_rows is not None
        # The index is a projection of the transcript entry: same rows, same shape.
        assert entry["schedules"] == transcript_rows["schedules"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_cancelling_last_wake_removes_entry(tmp_path: Path, config_dir: Path) -> None:
    session = _open(tmp_path)
    try:
        await session.set_wake_schedules([_schedule()])
        assert wake_store.entry_path(config_dir, session.session_id).exists()
        await session.set_wake_schedules([])
        assert not wake_store.entry_path(config_dir, session.session_id).exists()
        assert wake_store.read_index(config_dir) == {}
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_open_rewrites_deleted_entry_from_transcript(
    tmp_path: Path, config_dir: Path
) -> None:
    session = _open(tmp_path)
    await session.set_wake_schedules([_schedule()])
    await session.dispose()
    path = wake_store.entry_path(config_dir, "sess")
    path.unlink()
    assert not path.exists()

    reopened = _open(tmp_path)
    try:
        assert path.exists(), "open must rebuild the index from the transcript"
        entry = wake_store.read_entry(config_dir, "sess")
        assert entry is not None and entry["schedules"][0]["id"] == "w1"
    finally:
        await reopened.dispose()


@pytest.mark.asyncio
async def test_open_with_no_schedules_removes_stale_entry(tmp_path: Path, config_dir: Path) -> None:
    # A stale file for a session whose transcript carries no wakes (e.g. a
    # hand copy, or a schema that emptied) is removed on open.
    wake_store.write_entry(config_dir, "sess", cwd="/stale", schedules=[_schedule()])
    session = _open(tmp_path)
    try:
        assert not wake_store.entry_path(config_dir, "sess").exists()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_stopped_at_preserved_on_persist_and_cleared_on_open(
    tmp_path: Path, config_dir: Path
) -> None:
    session = _open(tmp_path)
    await session.set_wake_schedules([_schedule()])
    path = wake_store.entry_path(config_dir, "sess")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["stopped_at"] = 1_700_000_100_000
    path.write_text(json.dumps(data), encoding="utf-8")

    # A persist from the live session keeps the marker (the stop is not
    # this session's to undo mid-flight). The second schedule is also
    # future-derived — and later than w1's — for the same reason as the
    # default: an overdue fixture here races dispose the same way, and this
    # test reads the transcript rows back after a reopen.
    await session.set_wake_schedules(
        [_schedule("w1"), _schedule("w2", due=int(time.time() * 1000) + 3_900_000)]
    )
    entry = wake_store.read_entry(config_dir, "sess")
    assert entry is not None and entry["stopped_at"] == 1_700_000_100_000
    assert [row["id"] for row in entry["schedules"]] == ["w1", "w2"]
    await session.dispose()

    # Opening the session is what un-stops it.
    reopened = _open(tmp_path)
    try:
        entry = wake_store.read_entry(config_dir, "sess")
        assert entry is not None and "stopped_at" not in entry
        assert [row["id"] for row in entry["schedules"]] == ["w1", "w2"]
    finally:
        await reopened.dispose()


@pytest.mark.asyncio
async def test_index_failure_never_breaks_wake_persistence(
    tmp_path: Path, config_dir: Path, monkeypatch, caplog
) -> None:
    def boom(*args, **kwargs):  # noqa: ANN001, ANN202
        raise OSError("read-only filesystem")

    monkeypatch.setattr(wake_store, "write_entry", boom)
    session = _open(tmp_path)
    try:
        with caplog.at_level("WARNING"):
            await session.set_wake_schedules([_schedule()])
        # The transcript (source of truth) and the live scheduler both took it.
        assert session._transcript.latest_custom("wake_schedules") is not None
        assert [s.id for s in session.wake_scheduler.schedules] == ["w1"]
        assert any("wake index" in rec.getMessage() for rec in caplog.records)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_install_hook_called_only_for_non_empty_persist(
    tmp_path: Path, config_dir: Path, monkeypatch
) -> None:
    calls: list[Path] = []

    def fake_install(root: Path) -> wake_install.InstallOutcome:
        calls.append(root)
        return wake_install.InstallOutcome(installed=False, reason="stub")

    monkeypatch.setattr(wake_install, "ensure_supervisor_installed", fake_install)
    session = _open(tmp_path)
    try:
        await session.set_wake_schedules([_schedule()])
        assert calls == [config_dir]
        await session.set_wake_schedules([])
        assert calls == [config_dir], "an empty persist must not try to install"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_install_hook_failure_never_breaks_wake_persistence(
    tmp_path: Path, config_dir: Path, monkeypatch, caplog
) -> None:
    def boom(root: Path) -> wake_install.InstallOutcome:
        raise RuntimeError("launchctl exploded")

    monkeypatch.setattr(wake_install, "ensure_supervisor_installed", boom)
    session = _open(tmp_path)
    try:
        with caplog.at_level("WARNING"):
            await session.set_wake_schedules([_schedule()])
        assert [s.id for s in session.wake_scheduler.schedules] == ["w1"]
        assert wake_store.read_entry(config_dir, "sess") is not None
        assert any("install hook failed" in rec.getMessage() for rec in caplog.records)
    finally:
        await session.dispose()
