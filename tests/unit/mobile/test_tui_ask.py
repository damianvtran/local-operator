"""A TUI ``ask`` picker is answerable from the phone.

The TUI-side bridge used to project no ask card and refuse every remote answer
(``TuiSessionHandle.ask_answer`` was a stub). These tests pin the fixed
contract against a REAL ``OperatorApp`` running under Textual ``run_test`` with
its live mobile handle attached:

- mounting an ``ask`` picker projects a ``kind="ask"`` PendingRequest carrying
  the first question's prompt + option labels (:meth:`note_ask_pending`);
- a phone ``ask_answer`` resolves the LIVE picker's future through its own
  settle path, takes the terminal card down, and returns ``{question.id:
  [value]}`` to the parked tool call;
- exactly one answer wins each race: phone-first takes the terminal card down,
  terminal-first makes a late phone answer report "no longer waiting" \u2014 never
  a double-resolve;
- when the picker settles by any route the phone card clears
  (:meth:`note_ask_settled`).

``ask_answer`` is contracted to run on the REGISTRANT's own loop/thread and hop
onto Textual via ``call_from_thread`` \u2014 which refuses a same-thread call \u2014 so
the answer is driven through the real loopback control socket exactly as the
daemon does (the shape ``test_tui_bridge`` uses), not by calling the handle on
the app's own thread. That also makes these tests true end-to-end evidence:
projection SSE-equivalent snapshot in, ``ask_answer`` control op out.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.mobile.tui_handle import TuiSessionHandle
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _question(secret: bool = False) -> AskQuestion:
    if secret:
        return AskQuestion(
            id="OPENAI_API_KEY",
            question="Paste your OpenAI API key",
            secret=True,
        )
    return AskQuestion(
        id="stale",
        question="What should happen to the stale rows?",
        options=[
            AskOption(label="Drop them", description="nothing reads the column"),
            AskOption(label="Backfill", description="slower, keeps history"),
        ],
    )


async def _wait_for_handle(app: OperatorApp, pilot) -> TuiSessionHandle:
    """The mobile bridge starts a few pilot ticks after adoption; wait for it."""
    for _ in range(50):
        handle = app._mobile_handle
        if isinstance(handle, TuiSessionHandle):
            return handle
        await pilot.pause(0.05)
    raise AssertionError("mobile handle never attached")


async def _mount_ask(
    app: OperatorApp, pilot, question: AskQuestion
) -> "asyncio.Task[dict[str, list[str]] | None]":
    asked = asyncio.create_task(app.request_user_choice([question]))
    for _ in range(4):
        await pilot.pause()
    assert app._ask_screen is not None, "ask picker never mounted"
    return asked


def _pending_request_id(handle: TuiSessionHandle) -> str:
    pending = handle._fold.projection.pending
    assert pending is not None, "no pending ask projected"
    return pending.request_id


class _Control:
    """An authenticated control connection to the TUI's registrant socket \u2014 a
    stand-in for the daemon, on its OWN thread's loop the way the real client
    runs (so ``ask_answer`` hops back onto Textual as designed)."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer
        self._req = 0

    @classmethod
    async def connect(cls, control_port: int, control_key: str) -> "_Control":
        reader, writer = await asyncio.open_connection("127.0.0.1", control_port)
        writer.write(json.dumps({"key": control_key}).encode() + b"\n")
        await writer.drain()
        return cls(reader, writer)

    async def read_frame(self, timeout: float = 5.0) -> dict[str, object]:
        line = await asyncio.wait_for(self._reader.readline(), timeout=timeout)
        return json.loads(line)

    async def latest_projection(self, timeout: float = 5.0) -> dict[str, object]:
        """Drain to the most recent projection frame available right now."""
        data: dict[str, object] | None = None
        while True:
            try:
                frame = await asyncio.wait_for(self._reader.readline(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            parsed = json.loads(frame)
            if parsed.get("op") == "projection":
                data = parsed["data"]
                timeout = 0.3  # keep draining any already-queued repaints, briefly
        assert data is not None, "no projection frame arrived"
        return data

    async def send(self, op: str, **fields) -> dict[str, object]:
        self._req += 1
        req = self._req
        self._writer.write(json.dumps({"op": op, "req": req, **fields}).encode() + b"\n")
        await self._writer.drain()
        # The registrant answers with ack|error for our req, interleaved with
        # projection pushes; skip the pushes and return the reply for this req.
        for _ in range(20):
            frame = await self.read_frame()
            if frame.get("req") == req and frame.get("op") in ("ack", "error"):
                return frame
        raise AssertionError(f"no ack/error for {op}")

    def close(self) -> None:
        self._writer.close()


async def _control_for(app: OperatorApp, pilot) -> _Control:
    for _ in range(50):
        if app._mobile_registrant is not None:
            break
        await pilot.pause(0.05)
    assert app._mobile_registrant is not None, "mobile registrant never started"
    from local_operator.mobile import registry

    records = registry.scan()
    assert records, "no discovery record published"
    record, _state = records[0]
    return await _Control.connect(record.control_port, record.control_key)


@pytest.mark.asyncio
async def test_tui_ask_projects_a_pending_ask_to_the_phone() -> None:
    """Mounting the picker pushes a kind=ask card carrying the question and its
    option labels \u2014 the projection JSON the phone renders from."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        handle = await _wait_for_handle(app, pilot)
        asked = await _mount_ask(app, pilot, _question())

        projection = handle._fold.projection
        pending = projection.pending
        assert pending is not None
        assert pending.kind == "ask"
        assert pending.title == "What should happen to the stale rows?"
        assert pending.options == ["Drop them", "Backfill"]
        # The wire is JSON; option labels are strings, not AskOption objects.
        assert pending.to_json()["options"] == ["Drop them", "Backfill"]

        # Clean up the parked call.
        app._settle_ask_picker()
        for _ in range(4):
            await pilot.pause()
        await asyncio.wait_for(asked, 2)


@pytest.mark.asyncio
async def test_phone_answer_resolves_the_picker_and_clears_pending() -> None:
    """A phone ask_answer (over the real control socket) settles the LIVE
    picker: the tool call returns the chosen value keyed by question.id, the
    terminal card comes down, and the projection clears its pending card."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        handle = await _wait_for_handle(app, pilot)
        control = await _control_for(app, pilot)
        try:
            asked = await _mount_ask(app, pilot, _question())
            picker = app._ask_screen
            request_id = _pending_request_id(handle)

            reply = await control.send("ask_answer", request_id=request_id, value="Backfill")
            assert reply["op"] == "ack" and reply["detail"] == "answered"
            for _ in range(4):
                await pilot.pause()

            answer = await asyncio.wait_for(asked, 2)
            assert answer == {"stale": ["Backfill"]}
            assert picker is not None and not picker.is_attached
            assert handle._fold.projection.pending is None
        finally:
            control.close()


@pytest.mark.asyncio
async def test_terminal_answer_first_makes_a_late_phone_answer_fail() -> None:
    """Terminal wins the race: once the picker settles from the terminal, a late
    phone answer reports "no longer waiting" rather than double-resolving."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        handle = await _wait_for_handle(app, pilot)
        control = await _control_for(app, pilot)
        try:
            asked = await _mount_ask(app, pilot, _question())
            request_id = _pending_request_id(handle)

            # The terminal answers: settle the picker exactly as the app's Enter
            # path does, on the Textual thread.
            picker = app._ask_screen
            assert picker is not None
            picker.settle({"stale": ["Drop them"]})
            for _ in range(4):
                await pilot.pause()
            answer = await asyncio.wait_for(asked, 2)
            assert answer == {"stale": ["Drop them"]}
            assert handle._fold.projection.pending is None

            # A phone answer arriving now must not double-resolve: the socket
            # returns an error frame carrying the "no longer waiting" message.
            reply = await control.send("ask_answer", request_id=request_id, value="Backfill")
            assert reply["op"] == "error"
            assert "no longer waiting" in str(reply["message"])
        finally:
            control.close()


@pytest.mark.asyncio
async def test_phone_answer_first_takes_the_terminal_card_down() -> None:
    """Phone wins the race: the terminal picker comes down because the phone
    answer resolves the very future the terminal awaits, and a second phone
    answer for the same request then fails."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        handle = await _wait_for_handle(app, pilot)
        control = await _control_for(app, pilot)
        try:
            asked = await _mount_ask(app, pilot, _question())
            picker = app._ask_screen
            request_id = _pending_request_id(handle)

            reply = await control.send("ask_answer", request_id=request_id, value="Drop them")
            assert reply["op"] == "ack"
            for _ in range(4):
                await pilot.pause()
            answer = await asyncio.wait_for(asked, 2)
            assert answer == {"stale": ["Drop them"]}
            assert picker is not None and not picker.is_attached

            reply = await control.send("ask_answer", request_id=request_id, value="Backfill")
            assert reply["op"] == "error" and "no longer waiting" in str(reply["message"])
        finally:
            control.close()


@pytest.mark.asyncio
async def test_secret_ask_projects_a_free_text_card_without_leaking_the_value() -> None:
    """A secret question has no options, so it projects with an empty option
    list (the phone shows a paste field); the pasted value resolves the picker
    and never appears in the projection."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        handle = await _wait_for_handle(app, pilot)
        control = await _control_for(app, pilot)
        try:
            asked = await _mount_ask(app, pilot, _question(secret=True))

            pending = handle._fold.projection.pending
            assert pending is not None
            assert pending.kind == "ask"
            assert pending.options == []
            request_id = pending.request_id

            reply = await control.send("ask_answer", request_id=request_id, value="sk-supersecret")
            assert reply["op"] == "ack"
            for _ in range(4):
                await pilot.pause()
            answer = await asyncio.wait_for(asked, 2)
            assert answer == {"OPENAI_API_KEY": ["sk-supersecret"]}
            # The secret never rode the projection.
            assert handle._fold.projection.pending is None
        finally:
            control.close()
