"""RemoteSession over the production loopback registrant socket (protocol v4)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentStartEvent,
    Message,
    ToolExecutionStartEvent,
)
from local_operator.mobile import registry
from local_operator.mobile.registrant import Registrant
from local_operator.mobile.types import AskOptionWire, PendingRequest
from local_operator.session.remote import RemoteSession
from tests.unit.mobile.test_registrant import FakeHandle


async def _wait_record(root: Path) -> registry.SessionRecord:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        found = registry.scan(root)
        if found and found[0][1] == "live":
            return found[0][0]
        await asyncio.sleep(0.02)
    raise AssertionError("registrant never published a live record")


async def _never_take_over() -> Any:
    raise AssertionError("live owner should not trigger takeover")


@pytest.mark.asyncio
async def test_remote_session_rehydrates_seed_then_streams_concrete_events(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        # Join in the middle of a live tool: these events precede the socket,
        # so only attach_sync can rebuild them on the follower.
        handle.emit_event(AgentStartEvent(generation=6))
        handle.emit_event(
            ToolExecutionStartEvent(
                tool_call_id="t1",
                tool_name="bash",
                args={"command": "pytest -q"},
                intent="Running tests",
            )
        )
        await asyncio.sleep(0.05)
        remote = await RemoteSession.connect(
            record,
            "s1",
            config_dir=tmp_path,
            takeover_factory=_never_take_over,
        )
        events = []
        remote.subscribe(events.append)
        assert [event.type for event in events] == [
            "agent_start",
            "tool_execution_start",
        ]
        assert remote.is_streaming is True

        from local_operator.harness.types import NoticeEvent

        handle.emit_event(NoticeEvent(text="after join", kind="info"))
        for _ in range(50):
            if any(event.type == "notice" for event in events):
                break
            await asyncio.sleep(0.02)
        assert events[-1].type == "notice"
        assert events[-1].text == "after join"
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_multi_question_ask_advances_same_request_across_two_followers(
    tmp_path: Path, monkeypatch
) -> None:
    """Question position is identity even when a different TUI answers next."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    first_remote = second_remote = None
    never = asyncio.Event()
    try:
        record = await _wait_record(tmp_path)
        first_remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        second_remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )

        async def answer_first(questions):  # noqa: ANN001, ANN202
            question = questions[0]
            if question.question == "Choose environment":
                return {question.id: ["prod"]}
            await never.wait()
            return {}

        async def answer_second(questions):  # noqa: ANN001, ANN202
            question = questions[0]
            if question.question == "Confirm deploy":
                return {question.id: ["yes"]}
            await never.wait()
            return {}

        first_remote.set_ask_handler(answer_first)
        second_remote.set_ask_handler(answer_second)
        answers = []
        registrant.set_pending(
            PendingRequest(
                request_id="ask-shared",
                kind="ask",
                title="Choose environment",
                options=[AskOptionWire(label="prod"), AskOptionWire(label="staging")],
                question_index=0,
                question_total=2,
            )
        )
        for _ in range(100):
            answers = [call for call in handle.calls if call[0] == "ask_answer"]
            if answers:
                break
            await asyncio.sleep(0.02)
        assert [call[1] for call in answers] == [("ask-shared", "prod", 0)]

        # The owner keeps one request id for the picker and only advances its
        # position. Before the fix both followers treated this as the Q1 replay.
        registrant.set_pending(
            PendingRequest(
                request_id="ask-shared",
                kind="ask",
                title="Confirm deploy",
                options=[AskOptionWire(label="yes"), AskOptionWire(label="no")],
                question_index=1,
                question_total=2,
            )
        )
        for _ in range(100):
            answers = [call for call in handle.calls if call[0] == "ask_answer"]
            if len(answers) == 2:
                break
            await asyncio.sleep(0.02)
        assert [call[1] for call in answers] == [
            ("ask-shared", "prod", 0),
            ("ask-shared", "yes", 1),
        ]
    finally:
        if first_remote is not None:
            await first_remote.dispose()
        if second_remote is not None:
            await second_remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_remote_prompt_steer_and_approval_route_to_owner(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record,
            "s1",
            config_dir=tmp_path,
            takeover_factory=_never_take_over,
        )
        remote.subscribe(lambda event: None)
        await remote.prompt("from follower")
        assert handle.calls[-1][0:2] == ("prompt", ("from follower",))

        handle.emit_event(AgentStartEvent(generation=1))
        await asyncio.sleep(0.05)
        remote.steer_message(Message.user("steer now", id="11111111-1111-4111-8111-111111111111"))
        for _ in range(50):
            if any(call[0] == "steer" for call in handle.calls):
                break
            await asyncio.sleep(0.02)
        assert [call[1][0] for call in handle.calls if call[0] in {"prompt", "steer"}] == [
            "from follower",
            "steer now",
        ]

        decisions: list[tuple[str, str]] = []

        async def approve(tool: str, detail: str) -> bool:
            decisions.append((tool, detail))
            return True

        remote.set_approval_handler(approve)
        registrant.set_pending(
            PendingRequest(
                request_id="approve-1",
                kind="approval",
                title="bash",
                detail="run pytest",
            )
        )
        for _ in range(100):
            if any(call[0] == "approval_answer" for call in handle.calls):
                break
            await asyncio.sleep(0.02)
        assert decisions == [("bash", "run pytest")]
        assert [call for call in handle.calls if call[0] == "approval_answer"][-1][1] == (
            "approve-1",
            True,
            False,
        )
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()
