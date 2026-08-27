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
    ToolResult,
)
from local_operator.mobile.registrant import Registrant
from local_operator.session.remote import RemoteSession
from local_operator.session.transcript import Transcript
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import TranscriptView
from tests.unit.mobile.test_registrant import FakeHandle
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
        Message.tool_result(
            ToolResult(
                tool_call_id="call-parity-1",
                tool_name="read",
                content=[
                    TextContent(text="tool output"),
                    ImageContent(data=_PNG_1X1, mime_type="image/png"),
                ],
            )
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


def _block_signature(app: OperatorApp) -> list[str]:
    view = app.query_one(TranscriptView)
    return [type(block).__name__ for block in view.blocks()]


async def _fresh_boot_signature(tmp_path: Path) -> list[str]:
    """Boot a cold follower app on the finished transcript; return its blocks."""
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
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


@pytest.mark.asyncio
async def test_reconnect_paints_gap_rows_with_fresh_boot_block_parity(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    transcript = Transcript(tmp_path / "sessions" / "s1")
    await transcript.append_message(Message.user("visible before disconnect"))

    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
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

            replacement = Registrant(handle, kind="tui")
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
