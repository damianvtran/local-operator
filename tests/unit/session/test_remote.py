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
from local_operator.mobile.types import AskOptionWire, PendingRequest
from local_operator.session.frontend_state import FRONTEND_CAPABILITY
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from tests.unit.session.runtime.test_server import FakeHandle


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
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        # Join in the middle of a live tool: these events precede the socket,
        # so only the canonical v5 snapshot can rebuild them on the follower.
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
    registrant = RuntimeServer(handle, kind="tui")
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
async def test_remote_aside_runs_on_owner_without_joining_transcript(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        deltas: list[str] = []
        answer = await remote.complete_aside(
            [Message.user("Why this approach?")],
            on_delta=deltas.append,
        )
        assert answer == "aside answer"
        assert deltas == ["aside answer"]
        assert handle.calls[-1][0] == "complete_aside"
        assert remote.history() == []
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_remote_prompt_steer_and_approval_route_to_owner(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
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


@pytest.mark.asyncio
async def test_remote_slash_returns_typed_result_rendered_by_invoker(
    tmp_path: Path, monkeypatch
) -> None:
    """A routed slash returns the owner's typed outcome, not a transport receipt."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        outcome = await remote.route_shared_slash("goal", "ship it")
        # The typed SlashResult dict crosses the socket; the invoker renders it.
        assert isinstance(outcome, dict)
        assert outcome["kind"] == "notice"
        assert outcome["text"] == "owner ran /goal"
        assert handle.calls[-1][0] == "run_slash_authoritative"
        assert handle.calls[-1][1][0:2] == ("goal", "ship it")
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_remote_adopt_aside_and_cancel_route_to_owner(tmp_path: Path, monkeypatch) -> None:
    """Ctrl+F fork and double-Esc cancel reach the owner's real operations."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        # /btw Ctrl+F: the fork routes the pair to the owner's adopt_aside.
        pair = [
            Message.user("q", id="aaaaaaa1-1111-4111-8111-111111111111"),
            Message.assistant("a", id="aaaaaaa2-1111-4111-8111-111111111111"),
        ]
        await remote.adopt_aside(pair)
        assert handle.calls[-1][0] == "adopt_aside"
        forwarded = handle.calls[-1][1][0]
        assert isinstance(forwarded, list) and len(forwarded) == 2

        # Double-Esc: the synchronous method issues the authoritative op and
        # the owner's confirmed count resolves through the callback.
        resolved: list[int] = []
        remote.set_cancel_resolution(resolved.append)
        optimistic = remote.cancel_subagents()
        assert optimistic == 0  # no canonical jobs staged => nothing offered
        for _ in range(100):
            if resolved:
                break
            await asyncio.sleep(0.02)
        assert resolved == [2]  # the owner's REAL count, not a guessed zero
        assert any(call[0] == "cancel_subagents_count" for call in handle.calls)
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_a_refused_sync_leaves_the_viewer_cold_and_holds_no_connection(
    tmp_path: Path, monkeypatch
) -> None:
    """#573's viewer half: a rejected bind must not leave a connected client.

    An owner whose canonical state names ANOTHER session (a fork serving its
    parent's checkpoint before the restore re-stamped it) is refused by
    ``_install_frontend``. Before this, ``_dial`` had already installed the
    client, so the facade stayed "bound": ``is_cold`` said False, RPCs still
    reached the owner, no state was ever installed, and every recovery pass
    dialled ANOTHER connection on top — the runtime's attach cap evicted them
    in bursts. The refusal must leave exactly what a failed engage leaves:
    a cold viewer and zero attach clients on the runtime.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    # The owner is hosting "s1" (welcome identity passes) but its canonical
    # state was restored from a checkpoint stamped with another session.
    handle._frontend.mutate(session_id="parent000001")
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record(tmp_path)
        with pytest.raises(ConnectionError, match="belongs to another session"):
            await RemoteSession.connect(
                record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
            )
        # Cold path: the same refusal through ``_ensure_bound`` on a viewer
        # that will keep living (the TUI's mount engage).
        viewer = await RemoteSession.cold(
            "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never_take_over
        )
        try:
            with pytest.raises(ConnectionError, match="belongs to another session"):
                await viewer._bind_to(record)
            assert viewer.is_cold is True, "a refused bind must not leave the facade bound"
            # Two refusals, zero leaked sockets: give the server a few ticks
            # to observe the closes.
            for _ in range(50):
                if registrant.attach_clients() == 0:
                    break
                await asyncio.sleep(0.02)
            assert registrant.attach_clients() == 0
            # And no recovery loop was started by our own close.
            await asyncio.sleep(0.1)
            assert viewer._recovery_task is None
        finally:
            await viewer.dispose()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_a_failed_redial_in_recovery_leaves_no_stale_runtime_identity(
    tmp_path: Path, monkeypatch
) -> None:
    """The recovery loop's dial-failure branch must leave no trace of the try.

    ``_dial`` stamps ``_runtime_pid`` from the record on ENTRY, before it can
    know the dial will fail, and installs ``self._client`` only as its LAST
    statement (it closes the client it built on every raise path). So what a
    failed redial leaks is not a socket — it is the IDENTITY: the branch
    ``continue``d straight to the next pass, so a viewer that never bound kept
    reporting ``runtime_pid`` as the pid it merely tried, where the property
    promises None while cold, and ``_go_cold`` does not clear it either.

    That pid is not cosmetic. ``take_unannounced_cleanup`` compares it against
    the pid that wrote the cleanup record to decide WHICH terminal announces a
    cleanup, so a stale match makes this viewer claim a notice belonging to a
    runtime it never attached to, and the terminal that should have printed it
    stays blank (review round 1, F3).

    Driven through the real loop: a live-looking record whose control port
    refuses is exactly what recovery sees while a runtime is going away.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    viewer = await RemoteSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never_take_over
    )
    try:
        dead = registry.SessionRecord(
            pid=424242,
            kind="tui",
            session_id="s1",
            conversation_name="gone",
            cwd=str(tmp_path),
            model_label="test/model",
            control_port=1,  # nothing listens here, so `_dial` raises
            control_key="k",
            protocol=5,
            capabilities=[FRONTEND_CAPABILITY],
        )

        # Feed the loop that unreachable record instead of the registry, then
        # let it take a few passes and go cold on its own deadline.
        monkeypatch.setattr(
            "local_operator.session.remote.find_owner_record",
            lambda *_a, **_k: (dead, None),
        )
        monkeypatch.setattr("local_operator.session.remote.COLD_FALLBACK_S", 0.4)
        await viewer._recover_owner()

        assert viewer._client is None, "a failed dial installs no client"
        assert viewer.is_cold is True
        assert (
            viewer.runtime_pid is None
        ), "a viewer that never bound must not report the pid it failed to dial"
        assert viewer._frontend_future is None, "the abandoned sync wait must not survive"
    finally:
        await viewer.dispose()
