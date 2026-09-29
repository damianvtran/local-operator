"""The stale-frame sweep at the session boundary: gated, guarded, journalled.

C4 of the fleet token-efficiency audit: resident screenshot frames replay on
every call (91.7% of them sat outside a newest-1 window fleet-wide), and the
pruning primitive existed but sessions never called it — only the evaluation
runner's shared pass did. These tests pin the wiring: a session that has
CAPTURED browser views sweeps stale frames at its turn boundary and journals
the blanking; a session that has not is left byte-identical; the newest
frames and the user's own pastes survive.

``_plan_compaction`` is the real path (the prune runs on it before the
trigger math); the model is a 1M-context one so the trigger stays below
threshold and the only effect under test is the sweep.
"""

from __future__ import annotations

import pytest

from local_operator.compaction.pruning import STALE_FRAME_NOTICE
from local_operator.harness.types import (
    AgentMessage,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
    ToolResult,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

#: 1M-context model: the gate under test is the SWEEP, not the trigger.
BIG_MODEL = ModelSpec(provider="test", model_id="opus-like", context_window=1_000_000)


class ScriptedStream:
    """A stream that only ever answers one text reply; no tool calls."""

    def __init__(self) -> None:
        self.requests: list[object] = []

    def __call__(self, request, signal):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="reply")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(tmp_path) -> Session:
    return Session(
        model=BIG_MODEL,
        stream_fn=ScriptedStream(),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
    )


def _image_read(call_id: str, path: str) -> Message:
    """A tool row shaped exactly like the loop converts a read image result."""
    return Message.tool_result(
        ToolResult(
            tool_call_id=call_id,
            tool_name="read",
            content=[
                TextContent(text=f"Image {path} (640x480 png, 12 KB)"),
                ImageContent(data="aGVsbG8=", mime_type="image/png"),
            ],
            details={"path": path, "mime_type": "image/png"},
        )
    )


def _browser_capture(call_id: str, path: str) -> Message:
    """A browser screenshot result — the capture the gate keys on.

    No pixels: a capture only WRITES a file; the pixels enter context when the
    file is read (the tool row carries the path and byte count, never an
    image block — that is the shape both browser hosts return).
    """
    return Message.tool_result(
        ToolResult(
            tool_call_id=call_id,
            tool_name="browser",
            content=[
                TextContent(
                    text=(
                        f"Screenshot of Example (https://example.test/) saved to "
                        f"{path} (2048 bytes)."
                    )
                )
            ],
            details={"path": path, "bytes": 2048},
        )
    )


def _frame_messages(messages: list[AgentMessage]) -> list[Message]:
    """The frame-bearing subset, in order — the sweep's unit of counting."""
    return [
        message
        for message in messages
        if isinstance(message, Message)
        and any(isinstance(block, ImageContent) for block in message.content)
    ]


@pytest.mark.asyncio
async def test_a_browser_capturing_session_sweeps_stale_frames(tmp_path):
    """The wiring: stale frames fold, newest frames survive, and the browser
    capture row itself is untouched (it is the gate's evidence, not a frame)."""
    session = make_session(tmp_path)
    shots = [_image_read(f"r{i}", f"/shots/{i}.png") for i in range(4)]
    capture = _browser_capture("b1", "/tmp/browser-shot.png")
    session._context.messages.extend(
        [
            shots[0],
            Message.assistant("one"),
            shots[1],
            Message.assistant("two"),
            capture,
            shots[2],
            Message.assistant("three"),
            shots[3],
            Message.assistant("four"),
        ]
    )
    # Both conditions hold — four resident frames and a captured browser
    # view — asserted so a fixture drift cannot make this test vacuous.
    assert len(_frame_messages(session._context.messages)) == 4

    outcome = await session._plan_compaction(respect_threshold=True)

    assert getattr(outcome, "ran", None) is False  # below threshold; the prune still ran
    # The two oldest frames are folded to the notice, captions kept.
    for stale in shots[:2]:
        assert not any(isinstance(block, ImageContent) for block in stale.content)
        assert STALE_FRAME_NOTICE in stale.text
        assert "Image /shots/" in stale.text
        assert (stale.provider_payload or {}).get("pruned") is True
    # The newest two frames (the active comparison window) survive intact.
    for survivor in shots[2:]:
        assert any(isinstance(block, ImageContent) for block in survivor.content)
        assert STALE_FRAME_NOTICE not in survivor.text
    # The capture row was never a candidate.
    assert "Screenshot of Example" in capture.text
    await session.dispose()


@pytest.mark.asyncio
async def test_a_session_without_browser_captures_is_untouched(tmp_path):
    """The gate: frames without any captured browser view stay byte-identical."""
    session = make_session(tmp_path)
    shots = [_image_read(f"r{i}", f"/shots/{i}.png") for i in range(4)]
    session._context.messages.extend(list(shots))

    await session._plan_compaction(respect_threshold=True)

    for shot in shots:
        assert any(
            isinstance(block, ImageContent) for block in shot.content
        ), "a session that never captured a browser view must not be swept"
        assert (shot.provider_payload or {}).get("pruned") is None
    await session.dispose()


@pytest.mark.asyncio
async def test_browser_use_without_a_capture_is_untouched(tmp_path):
    """Browsing alone does not make a session sweep-eligible: a goto/click is
    not a view of a surface that a later capture supersedes."""
    session = make_session(tmp_path)
    shots = [_image_read(f"r{i}", f"/shots/{i}.png") for i in range(4)]
    go = Message.tool_result(
        ToolResult(
            tool_call_id="b1",
            tool_name="browser",
            content=[TextContent(text="Opened browser surface e5: https://example.test/")],
            details={"surface_id": "e5", "url": "https://example.test/", "title": "Example"},
        )
    )
    session._context.messages.extend([go, *shots])

    await session._plan_compaction(respect_threshold=True)

    for shot in shots:
        assert any(isinstance(block, ImageContent) for block in shot.content)
    await session.dispose()


@pytest.mark.asyncio
async def test_the_sweep_is_journalled_for_resume(tmp_path):
    """A blanked frame must not come back on resume: the prune journal records
    it with the folded text, exactly as the tool-output prune's does."""
    session = make_session(tmp_path)
    shots = [_image_read(f"r{i}", f"/shots/{i}.png") for i in range(3)]
    session._context.messages.extend([_browser_capture("b1", "/tmp/cap.png"), *shots])

    await session._plan_compaction(respect_threshold=True)

    pruned = [entry for entry in session._transcript.entries() if entry.type == "prune"]
    targets = {entry.payload["target"] for entry in pruned}
    assert shots[0].id in targets
    notices = [e.payload["notice"] for e in pruned if e.payload["target"] == shots[0].id]
    assert notices and STALE_FRAME_NOTICE in notices[0]
    assert "Image /shots/0.png" in notices[0]
    await session.dispose()
