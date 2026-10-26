"""R6-1: the rung-4 stand-down, on the REAL runtime path.

The reviewer's finding was that `_tui_viewer_active` read
`registrant.tui_input_age_s` and no such reader existed, so the probe answered
`None` in production and the runtime still bannered a conversation the operator
was reading. Round 2's test injected `tui_watching=True` and never reached the
reader, which is exactly why it passed while production did not.

This test drives the whole chain instead:

    real AttachClient / raw control socket
      -> the real `activity` control op on a real `RuntimeServer`
      -> `RuntimeServer.tui_input_age_s()`
      -> `ServingSessionHandle._tui_viewer_active()` (the production probe)
      -> `delivery.route_item(...)`'s decision

No `tui_watching` is injected anywhere. The negative arms are the production
answers this must keep: no report at all, a stale report, and a viewer that has
declared it is showing something else all leave the answer at `None`, which the
ladder treats as "raise the banner" (contract §6.3).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from tests.unit.session.runtime.test_server import (
    FakeHandle,
    _dial,
    _until,
    _wait_record,
)


def _send(writer: object, frame: dict[str, object]) -> None:
    writer.write(json.dumps(frame).encode() + b"\n")  # type: ignore[attr-defined]


def _make_handle(tmp_path: Path, runtime: object) -> Any:
    """A real `ServingSessionHandle` registered against ``runtime``.

    Built the way the serving tests build it (`install_gates=False` skips the
    approval machinery this question has nothing to do with). The handle's
    ``_registrant`` IS the live `RuntimeServer`, so the probe reads the real
    connection table rather than a stub.
    """
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(tmp_path / "session", ScriptedStream([]))
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    handle._registrant = runtime  # type: ignore[attr-defined]
    return handle


@pytest.mark.asyncio
async def test_a_fresh_activity_report_on_a_displaying_viewer_suppresses_rung_four(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POSITIVE: attached + fresh input on THIS session => no rung-4 banner."""
    from local_operator.session.runtime.server import RuntimeServer

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        handle = _make_handle(tmp_path, runtime)

        # Before any report: unknown, and unknown notifies.
        assert runtime.tui_input_age_s() is None
        assert handle._tui_viewer_active() is None

        _send(
            writer,
            {"op": "activity", "req": 1, "input_at_ms": int(time.time() * 1000) - 500},
        )
        await writer.drain()
        await _until(reader, "ack", 1)

        age = runtime.tui_input_age_s()
        assert age is not None and age < 120.0, age
        assert handle._tui_viewer_active() is True

        # And the ladder uses it: no detached banner for this item. The two
        # GATES are pinned open first, or `route_item` answers "gated" before it
        # ever consults the watched-session arm — which would make this test pass
        # while proving nothing about the probe (round 2's mistake, one layer up).
        from local_operator.notifications import delivery
        from local_operator.tui import notify as tui_notify

        monkeypatch.setattr(tui_notify, "notifications_enabled", lambda: True)
        monkeypatch.setattr(tui_notify, "desktop_belongs_to_this_process", lambda *a, **k: True)
        route = delivery.route_item(
            tmp_path,
            action="created",
            priority="high",
            tui_watching=handle._tui_viewer_active(),
        )
        assert route.rung == delivery.RUNG_TUI_VIEWER, route
        writer.close()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_no_activity_report_leaves_the_answer_unknown(tmp_path: Path) -> None:
    """NEGATIVE 1: an attached viewer that never reported input => notify."""
    from local_operator.session.runtime.server import RuntimeServer

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        handle = _make_handle(tmp_path, runtime)
        assert runtime.attach_clients() == 1
        assert runtime.tui_input_age_s() is None
        assert handle._tui_viewer_active() is None
        writer.close()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_stale_activity_report_is_not_active(tmp_path: Path) -> None:
    """NEGATIVE 2: an edge older than the 120 s window => notify."""
    from local_operator.session.runtime.server import RuntimeServer

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        handle = _make_handle(tmp_path, runtime)
        _send(
            writer,
            {"op": "activity", "req": 1, "input_at_ms": int(time.time() * 1000) - 400_000},
        )
        await writer.drain()
        await _until(reader, "ack", 1)
        age = runtime.tui_input_age_s()
        assert age is not None and age > 120.0, age
        assert handle._tui_viewer_active() is False, "stale input is not active"
        writer.close()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_a_viewer_showing_another_session_is_not_a_watched_session(
    tmp_path: Path,
) -> None:
    """NEGATIVE 3: fresh input from a viewer that switched away => notify."""
    from local_operator.session.runtime.server import RuntimeServer

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        handle = _make_handle(tmp_path, runtime)
        # The multiplexed terminal says it is NOT showing this conversation, and
        # then keeps typing in whatever it switched to.
        _send(writer, {"op": "viewer_watch", "req": 1, "displaying": False})
        await writer.drain()
        await _until(reader, "ack", 1)
        _send(
            writer,
            {"op": "activity", "req": 2, "input_at_ms": int(time.time() * 1000) - 200},
        )
        await writer.drain()
        await _until(reader, "ack", 2)
        assert runtime.tui_input_age_s() is None, "an away viewer reports nothing for us"
        assert handle._tui_viewer_active() is None
        writer.close()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_the_activity_op_carries_a_timestamp_and_nothing_else(tmp_path: Path) -> None:
    """SECURITY: the op refuses anything but a positive integer millisecond.

    The wire must never be able to carry keystroke content for a routing
    decision, so the field is type-checked and the connection keeps one integer.
    """
    from local_operator.session.runtime.server import RuntimeServer

    runtime = RuntimeServer(FakeHandle(), kind="tui")
    runtime.start()
    try:
        record = await _wait_record()
        reader, writer = await _dial(record, client="attach")
        for bad in ("1700000000000", 0, -5, 1.5, True):
            _send(writer, {"op": "activity", "req": 9, "input_at_ms": bad})
            await writer.drain()
            frame = await _until(reader, "error", 9)
            assert "input_at_ms" in json.dumps(frame), (bad, frame)
        assert runtime.tui_input_age_s() is None, "no refused frame was stored"
        writer.close()
    finally:
        runtime.close()
