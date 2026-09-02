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
import os

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
    """Connect to the app's control socket once its record is really published.

    Waits on the DISCOVERY RECORD, not on the registrant object, and the
    distinction is the whole bug this replaced. ``RuntimeServer.start`` spawns a
    daemon thread and returns; the record is written by ``_serve`` on that
    thread. So ``app._mobile_registrant is not None`` means "registration was
    requested", not "the record is on disk" — and the old helper waited for the
    first and then immediately asserted the second, which is only true when
    the thread happens to win the race.

    On an idle machine it always does, which is why this reads as solid: 5/5
    passes unloaded and 206/206 for the module at ``-n4``. CI runs ``-n auto``
    (14 workers), and there it fails as ``no discovery record published`` —
    observed on two unrelated PRs whose diffs touch no mobile code, and once
    passing and failing on the SAME head. Polling the condition the assertion
    actually depends on removes the gap entirely.

    The bound is a deadlock guard rather than a timing assumption: it is
    reached only when nothing ever publishes.
    """
    from local_operator.session.runtime import registry

    records = []
    for _ in range(200):
        records = registry.scan()
        if records:
            break
        await pilot.pause(0.05)
    # Reported against the registrant when there is one, because "started but
    # never published" and "never started" are different failures and the
    # second is the more likely one to introduce. Checked only on the failure
    # path: once a record exists the registrant necessarily started, so
    # asserting it first would have read as though it were still part of the
    # wait — the exact misreading that produced the original bug (round 1, F4).
    if not records:
        assert app._mobile_registrant is not None, "mobile registrant never started"
        raise AssertionError("registrant started but published no discovery record")
    # This process's own record, not merely the first one scanned: ``scan`` is a
    # global discovery read, so indexing it assumes nothing else has published.
    # That holds under the suite's per-test config dir today, but matching on
    # our own pid states the requirement rather than depending on it, and turns
    # a stray record into a legible failure instead of a connection to someone
    # else's socket (round 1, F6). Pid rather than control port because the port
    # is stamped on the registrant's thread when the listener binds, so reading
    # it here would reintroduce exactly the cross-thread race this helper fixes.
    mine = os.getpid()
    record = next((r for r, _state in records if r.pid == mine), None)
    assert record is not None, (
        f"no discovery record for this process (pid {mine}); "
        f"scan saw pids {[r.pid for r, _ in records]}"
    )
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
        assert [o.label for o in pending.options] == ["Drop them", "Backfill"]
        # U3: option consequence lines ride the wire so the phone user decides
        # with the same information the terminal shows.
        assert [o.description for o in pending.options] == [
            "nothing reads the column",
            "slower, keeps history",
        ]
        assert pending.secret is False
        assert (pending.question_index, pending.question_total) == (0, 1)
        # The whole card must be JSON-serializable (asdict -> dict).
        wire = pending.to_json()
        assert wire["options"] == [
            {"label": "Drop them", "description": "nothing reads the column"},
            {"label": "Backfill", "description": "slower, keeps history"},
        ]
        assert wire["secret"] is False
        json.dumps(projection.to_json())

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
            # returns an error frame with the human "already answered" message (U4).
            reply = await control.send("ask_answer", request_id=request_id, value="Backfill")
            assert reply["op"] == "error"
            assert "already answered" in str(reply["message"])
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
            assert reply["op"] == "error" and "already answered" in str(reply["message"])
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
            # D1/U2: the secret flag rides the wire so the phone masks the
            # paste field — but the VALUE never does.
            assert pending.secret is True
            wire = handle._fold.projection.to_json()
            assert wire["pending"]["secret"] is True
            assert "sk-" not in json.dumps(wire)
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


@pytest.mark.asyncio
async def test_multi_question_ask_advances_instead_of_truncating() -> None:
    """U1: a phone answer to Q1 of a two-question ask must advance the picker
    to Q2 (re-projecting it) rather than settling the whole card and dropping
    Q2. Only after the last question is answered does the tool call resolve,
    and it resolves with BOTH answers."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        handle = await _wait_for_handle(app, pilot)
        control = await _control_for(app, pilot)
        try:
            q1 = AskQuestion(
                id="env",
                question="Which environment?",
                options=[
                    AskOption(label="prod", description="the live one"),
                    AskOption(label="staging", description="safe to break"),
                ],
            )
            q2 = AskQuestion(
                id="confirm",
                question="Confirm the deploy?",
                options=[
                    AskOption(label="yes", description="ship it"),
                    AskOption(label="no", description="hold"),
                ],
            )
            asked = asyncio.create_task(app.request_user_choice([q1, q2]))
            for _ in range(4):
                await pilot.pause()

            # Q1 is up, with the "1 of 2" position on the wire.
            pending = handle._fold.projection.pending
            assert pending is not None
            assert pending.title == "Which environment?"
            assert (pending.question_index, pending.question_total) == (0, 2)
            request_id = pending.request_id

            # Answer Q1 from the phone: the card must NOT settle — it advances.
            reply = await control.send("ask_answer", request_id=request_id, value="prod")
            assert reply["op"] == "ack"
            for _ in range(4):
                await pilot.pause()
            assert not asked.done(), "the tool call settled after only Q1 (U1 truncation)"

            # Q2 is now projected under the SAME request id, position 2 of 2.
            pending = handle._fold.projection.pending
            assert pending is not None
            assert pending.title == "Confirm the deploy?"
            assert (pending.question_index, pending.question_total) == (1, 2)
            assert pending.request_id == request_id

            # Answer Q2: now the whole card settles with both answers.
            reply = await control.send("ask_answer", request_id=request_id, value="yes")
            assert reply["op"] == "ack"
            for _ in range(4):
                await pilot.pause()
            answer = await asyncio.wait_for(asked, 2)
            assert answer == {"env": ["prod"], "confirm": ["yes"]}
            assert handle._fold.projection.pending is None
        finally:
            control.close()


@pytest.mark.asyncio
async def test_terminal_advance_reprojects_and_stale_phone_tap_is_safe() -> None:
    """U8: when the TERMINAL advances a multi-question ask, the phone must be
    re-projected to the new question, and a phone tap carrying the OLD question
    index must never be recorded against the question the terminal moved to.

    Reproduces the corruption on the pre-fix code (a phone answer against the
    stale question was silently keyed to the advanced question) and proves it is
    now closed by BOTH halves: re-projection on terminal advance, and the
    question-index guard on ask_answer.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        handle = await _wait_for_handle(app, pilot)
        control = await _control_for(app, pilot)
        try:
            q1 = AskQuestion(
                id="env",
                question="Which environment?",
                options=[
                    AskOption(label="prod", description="the live one"),
                    AskOption(label="staging", description="safe to break"),
                ],
            )
            q2 = AskQuestion(
                id="confirm",
                question="Confirm the deploy?",
                options=[AskOption(label="yes"), AskOption(label="no")],
            )
            asked = asyncio.create_task(app.request_user_choice([q1, q2]))
            for _ in range(4):
                await pilot.pause()

            pending = handle._fold.projection.pending
            assert pending is not None
            assert pending.title == "Which environment?"
            assert pending.question_index == 0
            request_id = pending.request_id

            # The TERMINAL answers Q1 by pressing Enter on the selected option
            # (prod is index 0 and preselected). This advances the picker to Q2.
            picker = app._ask_screen
            assert picker is not None
            await pilot.press("enter")
            for _ in range(6):
                await pilot.pause()
            assert picker.question_index == 1, "terminal Enter did not advance the picker"

            # Part 1 (re-project on terminal advance): the phone now shows Q2,
            # not the stale Q1 — the desync U8 described is gone.
            pending = handle._fold.projection.pending
            assert pending is not None
            assert pending.title == "Confirm the deploy?"
            assert pending.question_index == 1
            assert pending.request_id == request_id

            # Part 2 (index guard): a phone tap that still carries the OLD index
            # (0) — a tap in flight before the re-projection landed — must be
            # REJECTED, never recorded against Q2. This is the exact corruption
            # from the U8 repro: tapping 'staging' (a Q1 option) must not become
            # the answer to 'Confirm the deploy?'.
            reply = await control.send(
                "ask_answer", request_id=request_id, value="staging", question_index=0
            )
            assert reply["op"] == "error"
            assert "moved on" in str(reply["message"])
            assert not asked.done(), "a stale-index tap must not settle the card"

            # A correctly-indexed phone answer to the CURRENT question lands.
            reply = await control.send(
                "ask_answer", request_id=request_id, value="yes", question_index=1
            )
            assert reply["op"] == "ack"
            for _ in range(4):
                await pilot.pause()
            answer = await asyncio.wait_for(asked, 2)
            # Each value keyed to the RIGHT question: prod->env (terminal),
            # yes->confirm (phone). 'staging' never entered the map.
            assert answer == {"env": ["prod"], "confirm": ["yes"]}
        finally:
            control.close()
