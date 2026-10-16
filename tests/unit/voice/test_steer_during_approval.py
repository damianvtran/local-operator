"""V7, runtime half: a steer that lands while an approval is parked is not its answer.

The voice utterance route admits a steer while the session is streaming, and a
parked approval card sits INSIDE a streaming turn. This drives the real owner
objects (a real ``Session`` on the ``test`` provider over a real transcript, and
the real ``ServingSessionHandle`` whose approval gate the desktop answers
through) and only scripts the provider stream:

1. The model asks for ``write``; the gate parks an approval card.
2. "yes" is steered in, exactly the owner op the voice route reaches.
3. The card is STILL pending and its future is NOT resolved: the steer did not
   answer it. The steer is queued, not yet in the transcript.
4. The operator denies on the card (the desktop's ``/answers`` path).
5. The tool did not run, and the steered "yes" folds in AFTER the gate settled,
   as an ordinary user row the model reads at the next boundary.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from local_operator.harness.types import (
    ChatRequest,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import build_write_tool
from tests.e2e.harness import TEST_MODEL

DEADLOCK_GUARD_S = 20.0


async def _until(predicate) -> None:
    """Wait on the state the code under test publishes; the bound is a deadlock guard."""
    async with asyncio.timeout(DEADLOCK_GUARD_S):
        while not predicate():
            await asyncio.sleep(0.005)


def _user_rows(directory: Path) -> list[str]:
    rows = []
    path = directory / "transcript.jsonl"
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        payload = entry.get("payload") or {}
        if payload.get("role") == "user":
            rows.append(
                # A text block is stored WITHOUT its ``type`` (``exclude_defaults``
                # drops the default "text"), so read ``text`` off every block.
                "".join(block.get("text", "") for block in payload.get("content", []))
            )
    return rows


@pytest.mark.asyncio
async def test_a_steer_during_a_pending_approval_is_queued_not_an_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    directory = tmp_path / "config" / "sessions" / "c3c3c3c3c3c3"
    target = workspace / "never-written.txt"

    # Keyed on the REQUEST, not on call order: the session makes side calls
    # (conversation naming) beside the turn, and an order-indexed tape would be
    # consumed by them. Only a tool-bearing request whose newest row is the
    # opening prompt asks for the write; every other call answers in text.
    turn_requests: list[ChatRequest] = []

    async def stream(request: ChatRequest, signal: object = None):
        if not request.tools:
            yield StreamTextDelta(delta="A title")
            yield StreamEndEvent(stop_reason="stop")
            return
        turn_requests.append(request)
        newest = request.messages[-1]
        if newest.role == "user" and newest.text == "write the file":
            yield StreamTextDelta(delta="Writing the file.")
            yield StreamToolCallDelta(
                index=0,
                id="call_write_1",
                name="write",
                argument_delta=json.dumps({"path": str(target), "content": "x"}),
            )
            yield StreamEndEvent(stop_reason="toolUse")
            return
        yield StreamTextDelta(delta="Understood, not written.")
        yield StreamEndEvent(stop_reason="stop")

    session = Session(
        model=TEST_MODEL,
        stream_fn=stream,
        tools=[build_write_tool()],
        transcript=Transcript(directory),
        system_blocks_provider=lambda *_args: [],
        # The gate ARMED: the whole point is a turn that parks on a card.
        yolo=False,
        cwd=str(workspace),
    )
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(workspace), auto_approve=False
    )
    try:
        # ``prompt`` acknowledges on admission; the turn then runs on its own.
        await handle.prompt("write the file", command_id="prompt-1")
        await _until(lambda: handle._fold.projection.pending is not None)
        card = handle._fold.projection.pending
        assert card is not None and card.kind == "approval"
        gate_future = handle._pending_futures[card.request_id]
        assert session.is_streaming, "a parked approval sits inside a streaming turn"

        # The voice route's owner op while streaming: a STEER.
        receipt = await handle.steer("yes", command_id="voice-utterance-1", input_mode="dictated")
        assert receipt  # the handle acknowledged the queue insertion
        for _ in range(20):
            await asyncio.sleep(0)

        # (3) Not an answer: the card is still the same pending card, and its
        # future is unresolved. The steer is queued, not yet a transcript row.
        assert handle._fold.projection.pending is not None
        assert handle._fold.projection.pending.request_id == card.request_id
        assert not gate_future.done(), "a steer resolved the approval gate"
        assert "yes" not in _user_rows(directory)
        assert not target.exists()

        # (4) The operator denies on the card: the only path that settles it.
        await handle.approval_answer(card.request_id, False, False)
        await _until(lambda: _user_rows(directory) == ["write the file", "yes"])
        await _until(lambda: not session.is_streaming)

        # (5) The denial held (nothing written), and the steered "yes" folded
        # in after the gate settled, as an ordinary user row the model saw.
        assert gate_future.done() and gate_future.result() is False
        assert not target.exists(), "the steered 'yes' approved the write"
        assert _user_rows(directory) == ["write the file", "yes"]
        # What the model read on the call after the card settled: the denial as
        # the tool's result, THEN the steered words as the operator's next row.
        after = [(m.role, m.text) for m in turn_requests[-1].messages]
        assert after[-2][0] == "tool" and "denied" in after[-2][1].lower(), after
        assert after[-1] == ("user", "yes"), after
    finally:
        await handle.dispose()
