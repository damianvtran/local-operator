"""The TUI auto-registers with the mobile control plane when a session is
adopted — this pins that contract: record published, control socket answers,
slash commands land through the app's own dispatch, and unmount unpublishes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.mobile.tui_handle import (
    TuiSessionHandle,
    _DetailChangedDuringHydration,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.mark.asyncio
async def test_tui_same_id_concurrent_steers_cross_thread_once() -> None:
    class Session(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.steer_calls: list[tuple[str, str | None]] = []

        def steer(self, text, images=None, *, message_id=None):  # noqa: ANN001, ANN202
            self.steer_calls.append((text, message_id))

    class App:
        def __init__(self, session) -> None:  # noqa: ANN001
            self._session = session

        def call_from_thread(self, callback) -> None:  # noqa: ANN001
            # The callback is the Textual loop's atomic admission section. This
            # fake runs it inline while concurrent bridge tasks contend for it.
            callback()

    session = Session()
    handle = TuiSessionHandle(App(session))  # type: ignore[arg-type]
    receipts = await asyncio.gather(
        handle.steer("correction", command_id="same-id"),
        handle.steer("correction", command_id="same-id"),
    )

    assert receipts == ["steering queued", "already admitted"]
    assert session.steer_calls == [("correction", "same-id")]
    assert [row.text for row in handle._fold.projection.transcript] == ["correction"]


@pytest.mark.asyncio
async def test_tui_stalled_steers_apply_owner_loop_backpressure() -> None:
    class Session(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.steer_calls: list[str] = []

        def steer(self, text, images=None, *, message_id=None):  # noqa: ANN001, ANN202
            assert isinstance(message_id, str)
            self.steer_calls.append(message_id)

    class App:
        def __init__(self, session) -> None:  # noqa: ANN001
            self._session = session

        def call_from_thread(self, callback) -> None:  # noqa: ANN001
            callback()

    session = Session()
    handle = TuiSessionHandle(App(session))  # type: ignore[arg-type]
    for index in range(32):
        assert await handle.steer(str(index), command_id=f"id-{index}") == "steering queued"
    assert await handle.steer("duplicate", command_id="id-0") == "already admitted"
    with pytest.raises(RuntimeError, match=r"steering queue is full \(32\)"):
        await handle.steer("overflow", command_id="overflow")
    assert len(session.steer_calls) == 32


@pytest.mark.asyncio
async def test_nested_child_detail_events_refresh_after_warm(monkeypatch) -> None:
    class Comms:
        def __init__(self) -> None:
            self.listener = None
            self.child = SimpleNamespace(
                job_id="nested",
                label="nested",
                session_dir=Path("/tmp/nested"),
                parent_job_id="parent",
                session_id="nested",
                prompt="",
                effort="",
                agent_role="task",
                launch_message_id=None,
            )

        def roster(self):  # noqa: ANN201
            return []

        def nodes(self):  # noqa: ANN201
            return [self.child]

        def job(self, job_id):  # noqa: ANN001, ANN201
            return None

        def node(self, job_id):  # noqa: ANN001, ANN201
            return self.child if job_id == "nested" else None

        def subscribe_detail_changes(self, listener):  # noqa: ANN001, ANN201
            self.listener = listener
            return lambda: None

    comms = Comms()
    session = FakeSession()
    setattr(session, "_subagent_comms", comms)
    app = SimpleNamespace(_session=session)
    handle = TuiSessionHandle(app)  # type: ignore[arg-type]
    invalidated: list[str] = []
    monkeypatch.setattr(handle, "_invalidate_subagent_detail", invalidated.append)
    handle.subscribe(lambda: None)
    assert invalidated == ["nested"]  # initial warm
    assert comms.listener is not None
    comms.listener("nested")
    assert invalidated == ["nested", "nested"]  # later nested mutation


@pytest.mark.asyncio
async def test_dirty_hydration_retries_without_later_event(monkeypatch) -> None:
    node = SimpleNamespace(job_id="nested", session_dir=Path("/tmp/nested"))
    comms = SimpleNamespace(node=lambda job_id: node)
    session = FakeSession()
    setattr(session, "_subagent_comms", comms)
    handle = TuiSessionHandle(SimpleNamespace(_session=session))  # type: ignore[arg-type]
    handle._detail_generations["nested"] = 1
    calls = 0

    async def fake_to_thread(fn, session_dir):  # noqa: ANN001, ANN202
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _DetailChangedDuringHydration
        return (1, 1), []

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(handle._fold, "set_subagent_hydrated_details", lambda *args: True)
    await handle._hydrate_subagent_detail("nested")
    assert calls == 2


@pytest.mark.asyncio
async def test_tui_auto_registers_and_answers_control() -> None:
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        for _ in range(50):
            if app._mobile_registrant is not None:
                break
            await pilot.pause(0.1)
        assert app._mobile_registrant is not None, "mobile registrant never started"

        from local_operator.session.runtime import registry

        # The registrant EXISTING and its record being on disk are two
        # different async steps: publication is a thread hop, so on a slow
        # runner the scan lands in between and reads an empty store. Poll for
        # the state the assertion is about (CI shard 0, 3.12) rather than
        # betting the first scan wins the race.
        records = registry.scan()
        for _ in range(50):
            if records:
                break
            await pilot.pause(0.1)
            records = registry.scan()
        assert records, "no discovery record published"
        record, state = records[0]
        assert state == "live"
        assert record.kind == "tui"

        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=5)
        frame = json.loads(line)
        assert frame["op"] == "projection"
        assert frame["data"]["model_label"] == record.model_label

        writer.write(
            json.dumps({"op": "slash", "req": 1, "command": "goal", "args": "test goal"}).encode()
            + b"\n"
        )
        await writer.drain()
        acked = None
        for _ in range(10):
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            frame = json.loads(line)
            if frame.get("op") == "ack":
                acked = frame
                break
        assert acked is not None and "goal" in acked["detail"]
        writer.close()

    assert not registry.scan(), "record was not unpublished on exit"


def test_the_started_hook_is_wired_and_reseeded_on_rebind(tmp_path) -> None:
    """Q2: ``TuiSessionHandle`` must wire ``session._publish_session_started``
    the way ``ServingSessionHandle`` does, or a TUI-owned ``kind="tui"`` record
    stays ``started=False`` forever. Rebind must RE-wire it on the new session
    object and re-seed the registrant's bit from the new session's own durable
    history: a ``/new`` (no message rows) drops back to False — the composer
    window — while a ``/resume`` (message rows on disk) reads True."""

    class _Session(FakeSession):
        # The attribute ServingSessionHandle/TuiSessionHandle wire onto a real
        # Session; the fake does not carry it, so the test stands it up.
        def __init__(self, session_id: str, transcript_rows: list[str]) -> None:  # noqa: ANN001
            super().__init__()
            self._publish_session_started: Any = None
            self._sid = session_id
            path = tmp_path / session_id / "transcript.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(row + "\n" for row in transcript_rows))
            self.transcript_path = path

        # FakeSession pins a constant "sess"; the rebind path copies this
        # onto the projection, so the test needs per-session values.
        @property
        def session_id(self) -> str:
            return self._sid

    class _App:
        def __init__(self, session) -> None:  # noqa: ANN001
            self._session = session

    class _Registrant:
        def __init__(self) -> None:
            self.resets: list[bool] = []
            self.started_calls: list[bool] = []

        def reset_record_started(self, started: bool) -> None:
            self.resets.append(started)

        def set_record_started(self, started: bool) -> None:
            self.started_calls.append(started)

    fresh = _Session("fresh-id", [])
    resumed = _Session(
        "resumed-id",
        [
            '{"id":"m1","ts":1,"type":"custom","payload":{"custom_type":"title"}}',
            '{"id":"m2","ts":2,"type":"message","payload":{"role":"user"}}',
        ],
    )
    registrant = _Registrant()
    handle = TuiSessionHandle(_App(fresh))  # type: ignore[arg-type]
    handle._registrant = registrant  # type: ignore[attr-defined]

    # Wired at construction: the session's hook reaches the handle. Bound
    # methods are compared with == (each attribute access mints a new object).
    assert fresh._publish_session_started == handle._publish_session_started
    fresh._publish_session_started()
    assert registrant.started_calls == [True]

    # /resume: message history on disk re-seeds the bit True even though THIS
    # process has not run a turn for it.
    handle._app = _App(resumed)  # type: ignore[assignment]
    handle.rebind()
    assert resumed._publish_session_started == handle._publish_session_started
    assert registrant.resets == [True]

    # /new after that working conversation: no message rows, bit drops False.
    newer = _Session("newer-id", ['{"id":"t1","ts":3,"type":"custom","payload":{}}'])
    handle._app = _App(newer)  # type: ignore[assignment]
    handle.rebind()
    assert newer._publish_session_started == handle._publish_session_started
    assert registrant.resets == [True, False]

    # QA Q4: a transcript whose only message rows are quiet-dialled peer
    # notes (``peer_message`` CustomMessages, persisted without a turn) is
    # NOT durable history — the bit must stay False on rebind to it.
    noted = _Session(
        "noted-id",
        [
            '{"id":"p1","ts":4,"type":"message","payload":{"kind":"custom",'
            '"custom_type":"peer_message","attribution":"user","details":{"text":"hi"}}}',
        ],
    )
    handle._app = _App(noted)  # type: ignore[assignment]
    handle.rebind()
    assert registrant.resets == [True, False, False]
