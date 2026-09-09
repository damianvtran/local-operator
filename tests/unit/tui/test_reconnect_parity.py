"""Reconnect parity: a recovered transcript must equal a fresh boot of it.

Review round 3 (MAJOR-1/U7/D1): the durable gap replay collapsed every row
into the assistant path — a user prompt entered from another frontend painted
as agent speech, tool execution became prose, images and custom rows vanished.
The fix routes the gap through the SAME settled-history renderer a cold
resume uses, and the only honest proof is the comparison itself: drive a real
``OperatorApp`` over the production registrant socket through a disconnect
that spans a full interleaving (user+image, assistant prose + tool call, tool
result + image, custom peer row), then boot a second app cold on the same
transcript and assert the block classes and order are identical.

Semantic block-class assertions, deliberately: design round 3 required that
reconnect evidence not rely on text content alone, because the text survives
misattribution — the block type is what carries the speaker.

Block classes alone were not enough. The two paths this file compares are two
DIFFERENT functions — ``app._settle_painted_tool_card`` for a card already on
screen, ``session_presentation.replay_tool_call`` for a cold boot — and a
change that taught only one of them to restore ``provider_payload.duration_s``
left the reconnected row painting a blank duration column beside a cold boot
showing ``7.2s``, with the class signature identical and this test green. So
the parity assertion now covers the ToolCard's rendered duration cell as well
(review round 1, MAJOR-1/D2/Q1).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    CustomMessage,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.transcript import Transcript
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import ToolStarted
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import TranscriptView
from tests.unit.session.runtime.test_server import FakeHandle
from tests.unit.session.test_remote import _never_take_over, _wait_record

# One transparent 1x1 PNG: enough for the image pipeline without a real
# screenshot in the fixture.
_PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBg"
    "AAAABQABh6FO1AAAAABJRU5ErkJggg=="
)


def _remote_factory(remote: RemoteSession) -> Any:
    """An app factory over the production RemoteSession (not a FakeSession)."""

    async def factory() -> RemoteSession:
        return remote

    return factory


#: The interval the gap's tool result carries on disk. Both settle paths must
#: read it back, so it is asserted rather than merely present.
_GAP_DURATION_S = 7.25

_DURATION_CALL_ID = "call-duration-1"


def _duration_call_row() -> Message:
    """The assistant turn that ISSUES the call, persisted BEFORE the gap.

    Order matters for which settle path the result takes. In production the
    call row lands durably when the model emits it, so a terminal that was
    connected then already has the card on screen and the gap carries only the
    RESULT — which is what routes it through ``_settle_painted_tool_card``.
    Putting the call row in the gap instead would replay it as a fresh card and
    exercise the cold path twice, testing nothing about the divergence.
    """
    return Message(
        role="assistant",
        content=[TextContent(text="reading the file")],
        tool_calls=[ToolCall(id=_DURATION_CALL_ID, name="read", arguments={"path": "/tmp/x"})],
    )


def _duration_result_row() -> Message:
    """The durable result that landed during the gap, carrying its interval."""
    return _with_payload(
        Message.tool_result(
            ToolResult(
                tool_call_id=_DURATION_CALL_ID,
                tool_name="read",
                content=[TextContent(text="file contents")],
                duration_s=_GAP_DURATION_S,
            )
        ),
        {"details": None, "useless": False, "duration_s": _GAP_DURATION_S},
    )


def _gap_rows() -> list[Any]:
    """The interleaving that reconnect must survive with roles intact."""
    call = ToolCall(id="call-parity-1", name="read", arguments={"path": "/tmp/x"})
    return [
        Message.user(
            "durable while disconnected",
            images=[ImageContent(data=_PNG_1X1, mime_type="image/png")],
        ),
        Message(
            role="assistant",
            content=[TextContent(text="answer while disconnected")],
            tool_calls=[call],
        ),
        _with_payload(
            Message.tool_result(
                ToolResult(
                    tool_call_id="call-parity-1",
                    tool_name="read",
                    content=[
                        TextContent(text="tool output"),
                        ImageContent(data=_PNG_1X1, mime_type="image/png"),
                    ],
                    duration_s=_GAP_DURATION_S,
                )
            ),
            # Same payload shape the harness writes, so this leg crosses the
            # duration restore rather than only the blank-column case.
            {"details": None, "useless": False, "duration_s": _GAP_DURATION_S},
        ),
        CustomMessage(
            custom_type="peer_message",
            attribution="system",
            details={"body": "note from a peer", "sender": {"session_id": "s2"}},
        ),
    ]


async def _boot(app: OperatorApp, pilot: Any) -> None:
    for _ in range(120):
        await pilot.pause()
        if app._session is not None:
            return
    raise RuntimeError("app did not boot")


def _with_payload(message: Message, payload: dict[str, Any]) -> Message:
    """Attach the ``provider_payload`` the harness writes beside a tool result.

    ``Message.tool_result`` does not carry it — ``harness/loop.py`` sets it on
    the message after construction — so the fixture reproduces that shape here
    rather than asserting against a row no real transcript holds.
    """
    message.provider_payload = payload
    return message


def _block_signature(app: OperatorApp) -> list[str]:
    view = app.query_one(TranscriptView)
    return [type(block).__name__ for block in view.blocks()]


def _duration_cells(app: OperatorApp) -> list[float | None]:
    """The restored interval on every ToolCard, in transcript order."""
    view = app.query_one(TranscriptView)
    return [block._duration for block in view.blocks() if isinstance(block, ToolCard)]


async def _fresh_boot_signature(tmp_path: Path) -> list[str]:
    """Boot a cold follower app on the finished transcript; return its blocks."""
    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        app = OperatorApp(_remote_factory(remote))
        async with app.run_test(size=(118, 32)) as pilot:
            await _boot(app, pilot)
            for _ in range(5):
                await pilot.pause()
            return _block_signature(app)
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


async def _fresh_boot_durations(tmp_path: Path) -> list[float | None]:
    """Boot a cold follower app on the finished transcript; return its
    restored tool durations — the cold-resume side of the parity."""
    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        app = OperatorApp(_remote_factory(remote))
        async with app.run_test(size=(118, 32)) as pilot:
            await _boot(app, pilot)
            for _ in range(5):
                await pilot.pause()
            return _duration_cells(app)
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_reconnect_paints_gap_rows_with_fresh_boot_block_parity(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    transcript = Transcript(tmp_path / "sessions" / "s1")
    await transcript.append_message(Message.user("visible before disconnect"))

    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    reconnect_signature: list[str] = []
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        app = OperatorApp(_remote_factory(remote))
        async with app.run_test(size=(118, 32)) as pilot:
            await _boot(app, pilot)
            for _ in range(5):
                await pilot.pause()
            before = _block_signature(app)
            assert before.count("UserBlock") == 1

            registrant.close()
            (tmp_path / "sessions" / "s1" / ".session.pid").write_text(str(os.getpid()))
            for _ in range(100):
                if remote._recovering:
                    break
                await asyncio.sleep(0.02)
            assert remote._recovering is True
            for row in _gap_rows():
                await transcript.append_message(row)

            replacement = RuntimeServer(handle, kind="tui")
            replacement.start()
            try:
                deadline = asyncio.get_running_loop().time() + 15
                while asyncio.get_running_loop().time() < deadline:
                    await pilot.pause()
                    if not remote._recovering and len(remote.history()) == 5:
                        break
                    await asyncio.sleep(0.02)
                assert remote._recovering is False
                for _ in range(10):
                    await pilot.pause()
                reconnect_signature = _block_signature(app)
            finally:
                replacement.close()
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()

    # The reconnect painted every native shape exactly once: the recovered
    # user row is a UserBlock with its ImageBlock, the assistant turn is
    # prose plus a ToolCard carrying its result image, and the custom row is
    # a PeerMessageBlock — none of them assistant speech.
    assert reconnect_signature.count("UserBlock") == 2
    assert reconnect_signature.count("AssistantBlock") == 1
    assert reconnect_signature.count("ToolCard") == 1
    assert reconnect_signature.count("ImageBlock") == 2
    assert reconnect_signature.count("PeerMessageBlock") == 1

    # And the whole surface equals a cold boot of the same transcript: the
    # conversation reads identically whether or not this terminal lived
    # through the gap.
    fresh_signature = await _fresh_boot_signature(tmp_path)
    assert reconnect_signature == fresh_signature


@pytest.mark.asyncio
async def test_a_settled_painted_card_restores_the_same_duration_a_cold_boot_shows(
    tmp_path: Path, monkeypatch
) -> None:
    """The duration column is part of "reads identically", and it is the one
    part the block-class signature above cannot see.

    ``replay_history`` settles a tool row through one of TWO functions, chosen
    by whether a card for that call is already painted: ``replay_tool_call``
    for a cold boot, ``_settle_painted_tool_card`` for a terminal that painted
    the row live and then lost the connection. Teaching only the first to read
    ``provider_payload.duration_s`` made the same transcript row show ``7.2s``
    on one path and blank on the other while every block class matched, which
    is the divergence this file exists to catch (review round 1,
    MAJOR-1/D2/Q1).

    So this drives the painted-card route specifically: paint the call live
    through the production ``on_tool_started``, retire it the way the
    disconnect handler does, then let the reconnect settle it from the durable
    result — and compare the restored interval against a cold boot of the same
    bytes.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    transcript = Transcript(tmp_path / "sessions" / "s1")
    await transcript.append_message(Message.user("visible before disconnect"))
    # The call is durable BEFORE the gap; only its result lands during it.
    await transcript.append_message(_duration_call_row())

    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    settled_durations: list[float | None] = []
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        app = OperatorApp(_remote_factory(remote))
        async with app.run_test(size=(118, 32)) as pilot:
            await _boot(app, pilot)
            for _ in range(5):
                await pilot.pause()

            # Paint the call LIVE, so the reconnect finds a card on screen and
            # routes its result through `_settle_painted_tool_card` rather than
            # mounting a fresh row.
            app.on_tool_started(
                ToolStarted(
                    ToolExecutionStartEvent(
                        tool_call_id=_DURATION_CALL_ID,
                        tool_name="read",
                        args={"path": "/tmp/x"},
                    )
                )
            )
            await pilot.pause()
            assert app._painted_tool_card(_DURATION_CALL_ID) is not None

            registrant.close()
            (tmp_path / "sessions" / "s1" / ".session.pid").write_text(str(os.getpid()))
            for _ in range(100):
                if remote._recovering:
                    break
                await asyncio.sleep(0.02)
            assert remote._recovering is True
            # What the disconnect handler does to a live card: mark it
            # interrupted and retire it out of `_tool_cards`, leaving it
            # mounted for the recovered result to settle.
            app._retire_live_tool_cards()
            await transcript.append_message(_duration_result_row())

            replacement = RuntimeServer(handle, kind="tui")
            replacement.start()
            try:
                deadline = asyncio.get_running_loop().time() + 15
                while asyncio.get_running_loop().time() < deadline:
                    await pilot.pause()
                    if not remote._recovering and len(remote.history()) == 3:
                        break
                    await asyncio.sleep(0.02)
                assert remote._recovering is False
                for _ in range(10):
                    await pilot.pause()
                settled_durations = _duration_cells(app)
            finally:
                replacement.close()
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()

    # The settled card carries the persisted interval, not a blank and not a
    # clock started when this terminal painted the row.
    assert settled_durations == [pytest.approx(_GAP_DURATION_S)], settled_durations

    # And it agrees with the other path over the identical transcript.
    cold_durations = await _fresh_boot_durations(tmp_path)
    assert settled_durations == cold_durations
