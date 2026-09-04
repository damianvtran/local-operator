"""The opt-in control surface for headless ``exec`` runs.

Covers the composition root (``exec_control``), the ``cancel`` op's two modes
on the real socket, and the lifecycle ordering ``run_print_mode`` now owns —
close-and-announce BEFORE dispose, so a short run that ends inside the first
heartbeat still leaves no record behind.

The socket tests run against a REAL RuntimeServer for the same reason
``test_server.py`` does: the failure modes here (a record that outlives the
run, a cancel routed to the wrong rung) only exist on a live connection.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime import registry
from local_operator.session.runtime.exec_control import (
    EXEC_RECORD_KIND,
    maybe_start_exec_control,
    start_exec_control,
)


class FakeSession:
    """The slice of Session the owned handle and the runtime touch here."""

    def __init__(self, *, graceful: bool = True) -> None:
        self.session_id = "exec-sess"
        self.model_label = "test/model"
        self.effective_model_label = "test/model"
        self.model = None
        self.conversation_name = ""
        self.is_streaming = False
        self.disposed = False
        self.aborts: list[str] = []
        self.graceful_cancels: list[str] = []
        self._handlers: list[Any] = []
        # A session predating the boundary-cancel seam has no callable there.
        # Shadowing the class method with None on the instance is exactly what
        # the handle's ``callable(getattr(...))`` probe tests for, and unlike
        # ``del`` it works against a method defined on the class.
        if not graceful:
            self.request_graceful_cancel = None  # type: ignore[assignment]

        from local_operator.harness.jobs import AsyncJobManager

        self.jobs = AsyncJobManager()

    # -- the seams the handle reads at construction ---------------------------
    def set_approval_handler(self, handler) -> None:  # noqa: ANN001
        self._approval_handler = handler

    def set_ask_handler(self, handler) -> None:  # noqa: ANN001
        self._ask_handler = handler

    def subscribe(self, handler):  # noqa: ANN001, ANN201
        self._handlers.append(handler)
        return lambda: self._handlers.remove(handler)

    def history(self) -> list[Any]:
        return []

    # -- what the control ops drive -------------------------------------------
    def abort(self, reason: str = "") -> None:
        self.aborts.append(reason)

    def request_graceful_cancel(self, reason: str = "cancelled") -> None:
        self.graceful_cancels.append(reason)

    async def dispose(self) -> None:
        self.disposed = True


@pytest.fixture()
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the record directory at a tmp dir.

    Never the operator's real config dir: these tests publish live records, and
    one escaping into ``~/.local-operator/run/mobile`` would show up in the
    developer's own ``lop sessions`` as a session that does not exist. The env
    var is enough because ``paths.config_dir`` re-reads it on every call.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


async def _dial(record) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:  # noqa: ANN001
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
    writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
    await writer.drain()
    return reader, writer


async def _request(reader, writer, frame: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN001
    frame = {"req": "t1", **frame}
    writer.write(json.dumps(frame).encode() + b"\n")
    await writer.drain()
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        assert line, "socket closed before a reply"
        got = json.loads(line)
        if got.get("op") in ("ack", "error") and got.get("req") == "t1":
            return got


@pytest.mark.asyncio
async def test_start_publishes_an_exec_record(isolated_config: Path) -> None:
    session = FakeSession()
    control = await start_exec_control(session, cwd="/tmp")
    try:
        # The record's kind is the whole point: `lop sessions` and the daemon
        # can now tell a supervised one-shot from a terminal a human owns.
        records = registry.scan(isolated_config)
        assert [(r.kind, r.session_id) for r, _ in records] == [(EXEC_RECORD_KIND, "exec-sess")]
        assert control.port > 0
        # The endpoint line names the record rather than carrying the key: the
        # 0600 file IS the authorization model, so the credential must not be
        # copied into a supervisor's log.
        assert control.session_id in control.endpoint_line
        assert str(control.port) in control.endpoint_line
        assert control.record_path in control.endpoint_line
        assert control.runtime.record.control_key not in control.endpoint_line
    finally:
        await control.aclose()


@pytest.mark.asyncio
async def test_aclose_unpublishes_the_record(isolated_config: Path) -> None:
    """A run shorter than one heartbeat still cleans up after itself."""
    control = await start_exec_control(FakeSession(), cwd="/tmp")
    assert registry.scan(isolated_config)
    await control.aclose()
    assert registry.scan(isolated_config) == []


@pytest.mark.asyncio
async def test_cancel_defaults_to_the_tool_boundary(isolated_config: Path) -> None:
    """The default mode must never cut a running tool.

    This is the guarantee a supervisor cancelling mid-``git push`` depends on,
    so the assertion is on WHICH session method ran: ``request_graceful_cancel``
    sets a sticky flag the loop honours at the post-tool boundary, while
    ``abort`` fires the AbortSignal and cancels the tool task outright.
    """
    session = FakeSession()
    control = await start_exec_control(session, cwd="/tmp")
    try:
        reader, writer = await _dial(control.runtime.record)
        reply = await _request(reader, writer, {"op": "cancel"})
        assert reply["op"] == "ack"
        assert "boundary" in reply["detail"]
        assert session.graceful_cancels == ["cancelled by supervisor"]
        assert session.aborts == []
        writer.close()
    finally:
        await control.aclose()


@pytest.mark.asyncio
async def test_cancel_immediate_routes_to_abort(isolated_config: Path) -> None:
    """``mode: immediate`` is the explicit opt-in to cutting a tool in half."""
    session = FakeSession()
    control = await start_exec_control(session, cwd="/tmp")
    try:
        reader, writer = await _dial(control.runtime.record)
        reply = await _request(reader, writer, {"op": "cancel", "mode": "immediate"})
        assert reply["op"] == "ack"
        assert session.aborts and session.graceful_cancels == []
        writer.close()
    finally:
        await control.aclose()


@pytest.mark.asyncio
async def test_cancel_rejects_an_unknown_mode(isolated_config: Path) -> None:
    """A typo must not silently resolve to either rung.

    The two modes differ in whether external side effects get cut, so guessing
    is worse than refusing on the one op whose reason to exist is that
    distinction.
    """
    session = FakeSession()
    control = await start_exec_control(session, cwd="/tmp")
    try:
        reader, writer = await _dial(control.runtime.record)
        reply = await _request(reader, writer, {"op": "cancel", "mode": "soft"})
        assert reply["op"] == "error"
        assert "graceful" in reply["message"]
        assert session.aborts == [] and session.graceful_cancels == []
        writer.close()
    finally:
        await control.aclose()


@pytest.mark.asyncio
async def test_cancel_on_a_session_without_the_seam_errors(isolated_config: Path) -> None:
    """An older session says so rather than silently ignoring the cancel."""
    session = FakeSession(graceful=False)
    control = await start_exec_control(session, cwd="/tmp")
    try:
        reader, writer = await _dial(control.runtime.record)
        reply = await _request(reader, writer, {"op": "cancel"})
        assert reply["op"] == "error"
        assert "boundary" in reply["message"]
        writer.close()
    finally:
        await control.aclose()


@pytest.mark.asyncio
async def test_steer_reaches_the_session(isolated_config: Path) -> None:
    """The existing control vocabulary works unchanged over an exec runtime."""
    session = FakeSession()
    steered: list[str] = []

    def _steer(text: str, images: Any = None, **kwargs: Any) -> None:
        steered.append(text)

    session.steer = _steer  # type: ignore[attr-defined]
    session.has_admitted_command = lambda cid: False  # type: ignore[attr-defined]
    control = await start_exec_control(session, cwd="/tmp")
    try:
        reader, writer = await _dial(control.runtime.record)
        reply = await _request(reader, writer, {"op": "steer", "text": "change course"})
        assert reply["op"] == "ack"
        assert steered == ["change course"]
        writer.close()
    finally:
        await control.aclose()


@pytest.mark.asyncio
async def test_stop_op_aborts_rather_than_disposing_under_the_run(
    isolated_config: Path,
) -> None:
    """`lop stop` on an exec run ends the turn; the run's own teardown does
    the disposing, in the ordering ``aclose`` owns."""
    session = FakeSession()
    control = await start_exec_control(session, cwd="/tmp")
    try:
        reader, writer = await _dial(control.runtime.record)
        reply = await _request(reader, writer, {"op": "stop"})
        assert reply["op"] == "ack"
        assert session.aborts == ["stopped by supervisor"]
        # The session must NOT have been torn down out from under the prompt
        # that is still running.
        assert session.disposed is False
        writer.close()
    finally:
        await control.aclose()


@pytest.mark.asyncio
async def test_maybe_start_is_a_no_op_when_disabled(isolated_config: Path) -> None:
    session = FakeSession()
    assert await maybe_start_exec_control(session, enabled=False, cwd="/tmp") is None
    # No record, and nothing torn down: an ordinary exec run is unaffected.
    assert registry.scan(isolated_config) == []
    assert session.disposed is False


@pytest.mark.asyncio
async def test_maybe_start_disposes_and_raises_when_it_cannot_start(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A supervisor that asked to be able to steer must not get an agent it
    cannot stop — the failure is fatal, and the built session is released."""
    session = FakeSession()

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("no port")

    monkeypatch.setattr(
        "local_operator.session.runtime.server.RuntimeServer.start_in_process",
        boom,
    )
    with pytest.raises(OSError):
        await maybe_start_exec_control(session, enabled=True, cwd="/tmp")
    assert session.disposed is True
