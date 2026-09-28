"""The session as the index's single writer: after every persist, on every
open, self-healing — and the delivery path. The ``wakes/test_session_index``
contract, mirrored for monitors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.harness.types import CustomMessage, ModelSpec, StreamEndEvent
from local_operator.monitors import store as monitor_store
from local_operator.monitors.delivery import MonitorDelivery
from local_operator.monitors.spec import MonitorSpec
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
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


def _spec(sid: str = "m1", *, due: int | None = None) -> MonitorSpec:
    # The due time is only a fixture input when a test wants one; the
    # scheduler's own first-check delay (1-3 s) keeps a freshly loaded
    # monitor comfortably inside the test's lifetime.
    return MonitorSpec(
        id=sid,
        name="watch",
        tool="bash",
        arguments={"command": "date -u"},
        every_ms=60_000,
        created_at=1_756_000_000_000,
    )


@pytest.mark.asyncio
async def test_persist_writes_index_entry_matching_transcript(
    tmp_path: Path, config_dir: Path
) -> None:
    session = _open(tmp_path)
    try:
        await session.set_monitor_schedules([_spec()])
        entry = monitor_store.read_entry(config_dir, session.session_id)
        assert entry is not None
        assert entry["cwd"] == "/work/here"
        details = session._transcript.latest_custom("monitor_schedules")
        assert details is not None
        # The index is a projection of the transcript entry: every SPEC field
        # round-trips, and the index row carries runtime health on top (§10.2
        # — the transcript stays the spec's source of truth). The high-water
        # sequence rides the transcript entry.
        transcript_row = details["monitors"][0]
        index_row = entry["monitors"][0]
        for key, value in transcript_row.items():
            assert index_row[key] == value, key
        assert {"checks", "next_due_at", "disabled"} <= set(index_row)
        assert details["next_seq"] == session._monitors.next_seq
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_cancelling_last_monitor_removes_entry(tmp_path: Path, config_dir: Path) -> None:
    session = _open(tmp_path)
    try:
        await session.set_monitor_schedules([_spec()])
        assert monitor_store.entry_path(config_dir, session.session_id).exists()
        await session.set_monitor_schedules([])
        assert not monitor_store.entry_path(config_dir, session.session_id).exists()
        assert monitor_store.read_index(config_dir) == {}
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_open_rewrites_deleted_entry_from_transcript(
    tmp_path: Path, config_dir: Path
) -> None:
    session = _open(tmp_path)
    await session.set_monitor_schedules([_spec()])
    await session.dispose()
    path = monitor_store.entry_path(config_dir, "sess")
    path.unlink()
    assert not path.exists()

    reopened = _open(tmp_path)
    try:
        assert path.exists(), "open must rebuild the index from the transcript"
        entry = monitor_store.read_entry(config_dir, "sess")
        assert entry is not None and entry["monitors"][0]["id"] == "m1"
    finally:
        await reopened.dispose()


@pytest.mark.asyncio
async def test_open_with_no_monitors_removes_stale_entry(tmp_path: Path, config_dir: Path) -> None:
    # A stale file for a session whose transcript carries no monitors (a hand
    # copy, or a schema that emptied) is removed on open.
    monitor_store.write_entry(config_dir, "sess", cwd="/stale", monitors=[{"id": "m1"}])
    session = _open(tmp_path)
    try:
        assert not monitor_store.entry_path(config_dir, "sess").exists()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_stopped_at_preserved_on_persist_and_cleared_on_open(
    tmp_path: Path, config_dir: Path
) -> None:
    session = _open(tmp_path)
    await session.set_monitor_schedules([_spec()])
    path = monitor_store.entry_path(config_dir, "sess")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["stopped_at"] = 1_700_000_100_000
    path.write_text(json.dumps(data), encoding="utf-8")

    # A persist from the live session keeps the marker: an in-flight update is
    # not the stop's to undo.
    await session.set_monitor_schedules([_spec("m1"), _spec("m2")])
    entry = monitor_store.read_entry(config_dir, "sess")
    assert entry is not None and entry["stopped_at"] == 1_700_000_100_000
    assert [row["id"] for row in entry["monitors"]] == ["m1", "m2"]
    await session.dispose()

    # Opening the session is what un-stops it.
    reopened = _open(tmp_path)
    try:
        entry = monitor_store.read_entry(config_dir, "sess")
        assert entry is not None and "stopped_at" not in entry
        assert [row["id"] for row in entry["monitors"]] == ["m1", "m2"]
    finally:
        await reopened.dispose()


@pytest.mark.asyncio
async def test_index_failure_never_breaks_monitor_persistence(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(monitor_store, "write_entry", boom)
    session = _open(tmp_path)
    try:
        await session.set_monitor_schedules([_spec()])
        # The transcript append is what may fail the request; it didn't, and
        # the rows are live.
        details = session._transcript.latest_custom("monitor_schedules")
        assert details is not None and details["monitors"][0]["id"] == "m1"
        assert session._monitors.monitors[0].id == "m1"
        assert session._monitor_index_write_failed is True
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_arming_is_refused_while_the_index_cannot_be_written(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    session = _open(tmp_path)
    try:
        monkeypatch.setattr(monitor_store, "write_entry", boom)
        await session.set_monitor_schedules([_spec()])
        assert session._monitor_index_write_failed is True
        outcome = await session._monitors.create(
            {"tool": "bash", "arguments": {"command": "date -u"}}, cwd="/work/here"
        )
        assert "error" in outcome
        assert "index cannot be written" in outcome["error"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_delivery_in_a_busy_session_is_courtesy_queued(
    tmp_path: Path, config_dir: Path
) -> None:
    session = _open(tmp_path)
    try:
        events: list[object] = []

        async def record(event: object) -> None:
            events.append(event)

        session._emit = record  # type: ignore[method-assign]
        session._is_streaming = True
        delivery = MonitorDelivery(
            monitor_id="m1",
            name="watch",
            tool="bash",
            changes=2,
            checks=7,
            skipped=1,
            delta_text="+2/-0 changed lines\n+ x",
            at_ms=1_756_000_000_000,
        )
        await session._deliver_monitor(delivery)
        # The receipt event carries the full text for a front end; it is
        # emitted BEFORE the turn spawn (here: before the queue put).
        assert len(events) == 1
        event = events[0]
        assert getattr(event, "type", "") == "monitor_delta"
        assert getattr(event, "monitor_id", "") == "m1"
        message = session._steering_queue.get_nowait()
        assert isinstance(message, CustomMessage)
        assert message.custom_type == "monitor_prompt"
        assert message.attribution == "user"
        assert message.details["monitor_id"] == "m1"
        assert "1 skipped while the session was down" in message.details["text"]
        assert "fired while you were already working" in message.details["text"]
        assert session._courtesy_wake_count == 1
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_idle_delivery_spawns_a_turn(tmp_path: Path, config_dir: Path) -> None:
    session = _open(tmp_path)
    try:
        events: list[object] = []
        spawned: list[object] = []

        async def record(event: object) -> None:
            events.append(event)

        def capture(coro: object) -> None:
            spawned.append(coro)

        session._emit = record  # type: ignore[method-assign]
        session._spawn_background = capture  # type: ignore[method-assign]
        delivery = MonitorDelivery(
            monitor_id="m1",
            name="watch",
            tool="bash",
            changes=1,
            checks=2,
            skipped=0,
            delta_text="+1/-0 changed lines\n+ x",
            at_ms=1_756_000_000_000,
        )
        await session._deliver_monitor(delivery)
        assert len(events) == 1
        assert len(spawned) == 1  # a turn, not a queue put
        for coro in spawned:
            coro.close()  # type: ignore[attr-defined]
        assert session._steering_queue.empty()
        assert session._courtesy_wake_count == 0
    finally:
        await session.dispose()
