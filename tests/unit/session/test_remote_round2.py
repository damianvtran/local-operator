"""Round-2 review regressions: lifecycle dedupe, gap replay, cancel failure.

BLOCKER-1 (review round 2): dedupe by message id alone dropped the UPDATE and
END of every live message, so a follower's assistant rows froze at their
start event. These tests drive ``_on_wire_event`` with every lifecycle
interleaving and assert each legitimate beat arrives exactly once while true
replays (history/seed/reconnect re-advertising a completed row) stay dropped.

U6 (UX round 2): rows that became durable while the follower was disconnected
must be PAINTED after reconnect, not merely loaded into ``history()``.

MAJOR-1/U7/D1 (review round 3): the gap goes out as ONE typed
``history_delta`` carrying the settled rows verbatim — never per-row
``message_end`` events, which are a live assistant contract and painted every
role as assistant prose.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentStartEvent,
    CustomMessage,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    NoticeEvent,
    TextContent,
    ToolCall,
    ToolResult,
)
from local_operator.mobile.registrant import Registrant
from local_operator.session.remote import RemoteSession
from local_operator.session.transcript import Transcript
from tests.unit.mobile.test_registrant import FakeHandle
from tests.unit.session.test_remote import _never_take_over, _wait_record


def _bare_remote(tmp_path: Path) -> RemoteSession:
    """A connected-shape facade without the socket, for wire-event unit tests."""
    remote = RemoteSession(config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over)
    remote._ready_for_events = True
    return remote


@pytest.mark.asyncio
async def test_live_message_lifecycle_flows_start_updates_end() -> None:
    """BLOCKER-1: every phase of one live message reaches the subscriber."""
    remote = _bare_remote(Path("/tmp/r2-lifecycle"))
    events: list[str] = []
    remote.subscribe(lambda event: events.append(event.type))
    message = Message.assistant("")
    remote._on_wire_event(MessageStartEvent(message=message).model_dump(mode="json"))
    # Deltas are INCREMENTAL (the field contract is "UIs should append"), so
    # two updates for one in-flight message are two legitimate beats at the
    # same phase rank — the exact case BLOCKER-1 dropped.
    remote._on_wire_event(MessageUpdateEvent(message=message, delta="hel").model_dump(mode="json"))
    remote._on_wire_event(MessageUpdateEvent(message=message, delta="lo").model_dump(mode="json"))
    settled = Message.assistant("hello")
    settled.id = message.id
    remote._on_wire_event(MessageEndEvent(message=settled).model_dump(mode="json"))
    assert events == [
        "message_start",
        "message_update",
        "message_update",
        "message_end",
    ]


@pytest.mark.asyncio
async def test_completed_message_replay_is_dropped_but_new_messages_flow() -> None:
    """A replayed lifecycle for a COMPLETED message is a duplicate; a new id is not."""
    remote = _bare_remote(Path("/tmp/r2-replay"))
    events: list[tuple[str, str]] = []
    remote.subscribe(lambda event: events.append((event.type, getattr(event, "delta", "") or "")))
    message = Message.assistant("")
    remote._on_wire_event(MessageStartEvent(message=message).model_dump(mode="json"))
    remote._on_wire_event(MessageUpdateEvent(message=message, delta="hi").model_dump(mode="json"))
    settled = Message.assistant("hi")
    settled.id = message.id
    remote._on_wire_event(MessageEndEvent(message=settled).model_dump(mode="json"))
    # The same lifecycle re-advertised (a reconnect seed, a history replay):
    # nothing may paint twice.
    remote._on_wire_event(MessageStartEvent(message=message).model_dump(mode="json"))
    remote._on_wire_event(MessageUpdateEvent(message=message, delta="hi").model_dump(mode="json"))
    remote._on_wire_event(MessageEndEvent(message=settled).model_dump(mode="json"))
    # A genuinely new message with the same shape flows normally.
    second = Message.assistant("")
    remote._on_wire_event(MessageStartEvent(message=second).model_dump(mode="json"))
    remote._on_wire_event(MessageUpdateEvent(message=second, delta="next").model_dump(mode="json"))
    settled_second = Message.assistant("next")
    settled_second.id = second.id
    remote._on_wire_event(MessageEndEvent(message=settled_second).model_dump(mode="json"))
    assert events == [
        ("message_start", ""),
        ("message_update", "hi"),
        ("message_end", ""),
        ("message_start", ""),
        ("message_update", "next"),
        ("message_end", ""),
    ]


@pytest.mark.asyncio
async def test_seeded_completed_row_claims_its_id_against_a_late_end() -> None:
    """A sync-seeded start that already carries its text owns the row (M4).

    The snapshot's live fold keeps completed mid-join rows as message_start
    entries with their full content; a snapshot taken just after the end
    event would otherwise repaint the row when the durable/relayed end lands.
    """
    remote = _bare_remote(Path("/tmp/r2-seed"))
    events: list[str] = []
    remote.subscribe(lambda event: events.append(event.type))
    durable_looking = Message.assistant("already complete")
    remote._on_wire_event(MessageStartEvent(message=durable_looking).model_dump(mode="json"))
    remote._on_wire_event(MessageEndEvent(message=durable_looking).model_dump(mode="json"))
    assert events == ["message_start"]


@pytest.mark.asyncio
async def test_interleaved_messages_each_deliver_their_full_lifecycle() -> None:
    """Two concurrent message ids interleave without either losing a beat."""
    remote = _bare_remote(Path("/tmp/r2-interleave"))
    events: list[tuple[str, str]] = []
    remote.subscribe(
        lambda event: events.append(
            (event.type, str(getattr(getattr(event, "message", None), "id", "")))
        )
    )
    first = Message.assistant("", id="aaaaaaaa-1111-4111-8111-111111111111")
    second = Message.assistant("", id="bbbbbbbb-2222-4111-8111-222222222222")
    remote._on_wire_event(MessageStartEvent(message=first).model_dump(mode="json"))
    remote._on_wire_event(MessageStartEvent(message=second).model_dump(mode="json"))
    remote._on_wire_event(MessageUpdateEvent(message=first, delta="one").model_dump(mode="json"))
    remote._on_wire_event(MessageUpdateEvent(message=second, delta="two").model_dump(mode="json"))
    end_first = Message.assistant("one", id=first.id)
    end_second = Message.assistant("two", id=second.id)
    remote._on_wire_event(MessageEndEvent(message=end_first).model_dump(mode="json"))
    remote._on_wire_event(MessageEndEvent(message=end_second).model_dump(mode="json"))
    assert events == [
        ("message_start", first.id),
        ("message_start", second.id),
        ("message_update", first.id),
        ("message_update", second.id),
        ("message_end", first.id),
        ("message_end", second.id),
    ]


@pytest.mark.asyncio
async def test_reconnect_paints_rows_that_became_durable_during_the_gap(
    tmp_path: Path, monkeypatch
) -> None:
    """U6: a durable row committed during a disconnect paints exactly once."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    transcript = Transcript(tmp_path / "sessions" / "s1")
    await transcript.append_message(Message.user("visible before disconnect"))
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        events: list[Any] = []
        remote.subscribe(events.append)
        assert [event.type for event in events] == []
        assert len(remote.history()) == 1

        # Owner socket dies; the owner appends a completed turn while the
        # follower is detached, then the replacement owner comes up. The
        # registrant itself never writes the session claim marker (the real
        # Session owns it), so the test writes it: without a live owner pid
        # the recovery loop tries TAKEOVER, not reattach, and this test's
        # never-take-over factory would spin forever.
        registrant.close()
        import os

        (tmp_path / "sessions" / "s1" / ".session.pid").write_text(str(os.getpid()))
        for _ in range(100):
            if remote._recovering:
                break
            await asyncio.sleep(0.02)
        assert remote._recovering is True
        gap_user = Message.user("durable while disconnected")
        await transcript.append_message(gap_user)
        gap_assistant = Message.assistant("answer while disconnected")
        await transcript.append_message(gap_assistant)

        replacement = Registrant(handle, kind="tui")
        replacement.start()
        try:
            # The transient redial can land a client whose sync then fails;
            # the stable end state is a settled recovery with the gap rows
            # BOTH loaded and painted — the client check alone races the
            # retry loop (a failed dial clears the flag before re-arming).
            deadline = asyncio.get_running_loop().time() + 15
            while asyncio.get_running_loop().time() < deadline:
                deltas = [event for event in events if event.type == "history_delta"]
                if not remote._recovering and len(remote.history()) == 3 and deltas:
                    break
                await asyncio.sleep(0.02)
            assert remote._recovering is False
            # history() carries the gap rows AND the transcript painted them:
            # ONE typed history delta carrying the gap rows verbatim — role,
            # id and order preserved — with the pre-disconnect row absent.
            # Per-row message_end replay is the round-3 regression (MAJOR-1):
            # it repainted user rows as assistant speech.
            deltas = [event for event in events if event.type == "history_delta"]
            assert len(deltas) == 1
            assert not any(event.type == "message_end" for event in events)
            replayed = deltas[0].messages
            assert [(m.role, str(m.id)) for m in replayed] == [
                ("user", gap_user.id),
                ("assistant", gap_assistant.id),
            ]
            assert len(remote.history()) == 3
            assert str(remote.history()[-2].id) == gap_user.id
            assert str(remote.history()[-1].id) == gap_assistant.id
        finally:
            replacement.close()
    finally:
        if remote is not None:
            await remote.dispose()


@pytest.mark.asyncio
async def test_failed_cancel_resolution_preserves_failure_sentinel(
    tmp_path: Path, monkeypatch
) -> None:
    """MAJOR-2: a failed remote cancellation resolves ``-1``, never a success."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()

    async def broken_cancel() -> int:
        raise RuntimeError("socket lost mid-cancel")

    handle.cancel_subagents_count = broken_cancel  # type: ignore[assignment]
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        resolved: list[int] = []
        remote.set_cancel_resolution(resolved.append)
        remote.cancel_subagents()
        for _ in range(100):
            if resolved:
                break
            await asyncio.sleep(0.02)
        # The transport failure must reach the resolver as the failure
        # sentinel so the app can render "could not confirm", never the
        # optimistic offered count as a confirmed success.
        assert resolved == [-1]
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_partial_and_successful_cancel_counts_resolve_authoritatively(
    tmp_path: Path, monkeypatch
) -> None:
    """A partial stop (fewer than offered) and a clean stop both report REAL."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        # The fake owner confirms 2 — a PARTIAL stop relative to any larger
        # offer must surface as 2, and a matching offer as the same 2. The
        # follower's job is to relay the owner's number untouched.
        resolved: list[int] = []
        remote.set_cancel_resolution(resolved.append)
        remote.cancel_subagents()
        for _ in range(100):
            if resolved:
                break
            await asyncio.sleep(0.02)
        assert resolved == [2]
        assert any(call[0] == "cancel_subagents_count" for call in handle.calls)
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_reconnect_replays_each_gap_row_once_even_across_two_cycles(
    tmp_path: Path, monkeypatch
) -> None:
    """The replay filter is idempotent: a quiet reconnect adds nothing."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    transcript = Transcript(tmp_path / "sessions" / "s1")
    await transcript.append_message(Message.user("row one"))
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        events: list[Any] = []
        remote.subscribe(events.append)

        registrant.close()
        import os

        (tmp_path / "sessions" / "s1" / ".session.pid").write_text(str(os.getpid()))
        for _ in range(100):
            if remote._recovering:
                break
            await asyncio.sleep(0.02)
        await transcript.append_message(Message.assistant("row two in gap"))
        replacement = Registrant(handle, kind="tui")
        replacement.start()
        try:
            deadline = asyncio.get_running_loop().time() + 15
            while asyncio.get_running_loop().time() < deadline:
                if (
                    not remote._recovering
                    and [event.type for event in events].count("history_delta") == 1
                ):
                    break
                await asyncio.sleep(0.02)
            assert remote._recovering is False
            first_cycle = [event.type for event in events]
            assert first_cycle.count("history_delta") == 1

            # A SECOND disconnect with no new durable rows replays nothing.
            replacement.close()
            for _ in range(100):
                if remote._recovering:
                    break
                await asyncio.sleep(0.02)
            third = Registrant(handle, kind="tui")
            third.start()
            try:
                deadline = asyncio.get_running_loop().time() + 15
                while asyncio.get_running_loop().time() < deadline:
                    if not remote._recovering and remote._client is not None:
                        # A settle grace: the failed-dial retry window must
                        # have passed before "no repaint" is meaningful.
                        await asyncio.sleep(0.3)
                        if not remote._recovering:
                            break
                    await asyncio.sleep(0.02)
                assert remote._recovering is False
                assert [event.type for event in events].count("history_delta") == 1
            finally:
                third.close()
        finally:
            replacement.close()
    finally:
        if remote is not None:
            await remote.dispose()


@pytest.mark.asyncio
async def test_non_message_events_are_never_deduped() -> None:
    """Notices and other id-less events always pass the replay filter."""
    remote = _bare_remote(Path("/tmp/r2-notice"))
    events: list[str] = []
    remote.subscribe(lambda event: events.append(event.type))
    for _ in range(3):
        remote._on_wire_event(NoticeEvent(text="same text", kind="info").model_dump(mode="json"))
    assert events == ["notice", "notice", "notice"]


# One transparent 1x1 PNG, enough for ImageContent round-trips without a real
# screenshot in the fixture.
_PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBg"
    "AAAABQABh6FO1AAAAABJRU5ErkJggg=="
)


@pytest.mark.asyncio
async def test_reconnect_gap_delta_preserves_every_native_row_shape(
    tmp_path: Path, monkeypatch
) -> None:
    """MAJOR-1/U7/D1: the durable gap survives reconnect with roles intact.

    Production socket, full interleaving: a user prompt WITH an image, an
    assistant turn carrying prose plus a tool call, the call's tool result
    (with its own image), and a custom peer-message row all become durable
    during the disconnect. The replay must hand them over verbatim in ONE
    typed history delta — the shape the app's settled renderer consumes —
    with no role collapsed into assistant speech and no per-row live events.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    transcript = Transcript(tmp_path / "sessions" / "s1")
    await transcript.append_message(Message.user("visible before disconnect"))
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        events: list[Any] = []
        remote.subscribe(events.append)

        registrant.close()
        import os

        (tmp_path / "sessions" / "s1" / ".session.pid").write_text(str(os.getpid()))
        for _ in range(100):
            if remote._recovering:
                break
            await asyncio.sleep(0.02)
        assert remote._recovering is True

        gap_user = Message.user(
            "user gap", images=[ImageContent(data=_PNG_1X1, mime_type="image/png")]
        )
        await transcript.append_message(gap_user)
        call = ToolCall(id="call-gap-1", name="read", arguments={"path": "/tmp/x"})
        gap_assistant = Message(
            role="assistant",
            content=[TextContent(text="reading the file")],
            tool_calls=[call],
        )
        await transcript.append_message(gap_assistant)
        gap_result = Message.tool_result(
            ToolResult(
                tool_call_id="call-gap-1",
                tool_name="read",
                content=[
                    TextContent(text="tool output"),
                    ImageContent(data=_PNG_1X1, mime_type="image/png"),
                ],
            )
        )
        await transcript.append_message(gap_result)
        gap_custom = CustomMessage(
            custom_type="peer_message",
            attribution="system",
            details={"body": "note from a peer", "sender": {"session_id": "s2"}},
        )
        await transcript.append_message(gap_custom)

        replacement = Registrant(handle, kind="tui")
        replacement.start()
        try:
            deadline = asyncio.get_running_loop().time() + 15
            while asyncio.get_running_loop().time() < deadline:
                deltas = [event for event in events if event.type == "history_delta"]
                if not remote._recovering and deltas:
                    break
                await asyncio.sleep(0.02)
            assert remote._recovering is False
            deltas = [event for event in events if event.type == "history_delta"]
            assert len(deltas) == 1
            # No role-blind live replay: the round-3 defect emitted one
            # message_end per row and every row painted as assistant prose.
            assert not any(event.type == "message_end" for event in events)
            replayed = deltas[0].messages
            assert [str(getattr(m, "id", "")) for m in replayed] == [
                gap_user.id,
                gap_assistant.id,
                gap_result.id,
                gap_custom.id,
            ]
            user_row, assistant_row, result_row, custom_row = replayed
            assert user_row.role == "user"
            assert any(isinstance(block, ImageContent) for block in user_row.content)
            assert assistant_row.role == "assistant"
            assert assistant_row.text == "reading the file"
            assert [c.id for c in assistant_row.tool_calls] == ["call-gap-1"]
            assert result_row.role == "tool"
            assert result_row.tool_call_id == "call-gap-1"
            assert any(isinstance(block, ImageContent) for block in result_row.content)
            assert getattr(custom_row, "custom_type", None) == "peer_message"
            # A second quiet cycle replays nothing: the ids are claimed.
            assert len(remote.history()) == 5
        finally:
            replacement.close()
    finally:
        if remote is not None:
            await remote.dispose()


@pytest.mark.asyncio
async def test_routed_slash_during_recovery_refuses_in_user_language(
    tmp_path: Path, monkeypatch
) -> None:
    """MINOR-1/U8: a gap slash answers in user vocabulary, never transport terms."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        registrant.close()
        import os

        (tmp_path / "sessions" / "s1" / ".session.pid").write_text(str(os.getpid()))
        for _ in range(100):
            if remote._recovering:
                break
            await asyncio.sleep(0.02)
        assert remote._recovering is True
        with pytest.raises(ConnectionError) as excinfo:
            await remote.route_shared_slash("goal", "")
        text = str(excinfo.value)
        assert "reconnecting" in text
        assert "/goal" in text
        assert "attach" not in text.lower()
    finally:
        if remote is not None:
            await remote.dispose()


@pytest.mark.asyncio
async def test_prompt_during_recovery_still_queues_and_delivers(
    tmp_path: Path, monkeypatch
) -> None:
    """The gap prompt contract is unchanged: chat waits on the owner, slash refuses.

    Guards the asymmetry the U8 fix makes deliberate — the refusal above must
    not leak into the prompt path, whose queued delivery is the designed
    behavior (verified end to end in UX round 3).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        registrant.close()
        import os

        (tmp_path / "sessions" / "s1" / ".session.pid").write_text(str(os.getpid()))
        for _ in range(100):
            if remote._recovering:
                break
            await asyncio.sleep(0.02)
        assert remote._recovering is True
        prompt_task = asyncio.create_task(remote.prompt("queued during gap"))
        await asyncio.sleep(0.2)
        assert not prompt_task.done()  # waiting on _owner_ready, not failing
        replacement = Registrant(handle, kind="tui")
        replacement.start()
        try:
            await asyncio.wait_for(prompt_task, timeout=15)
            assert any(call[0] == "prompt" for call in handle.calls)
        finally:
            replacement.close()
    finally:
        if remote is not None:
            await remote.dispose()


@pytest.mark.asyncio
async def test_reconnect_delivers_gap_delta_before_live_frames_buffered_mid_parse(
    tmp_path: Path, monkeypatch
) -> None:
    """MAJOR-1 (review round 4): durable gap rows paint ABOVE the in-flight turn.

    The replacement owner is mid-turn and streaming; a live relay frame lands
    during the threaded transcript parse. Before the fix the buffer became
    [live frame…, delta] and ``_finish_sync`` front-inserted the seed, so
    delivery order was seed, live frame, THEN the durable gap — the recovered
    rows painted below the in-flight turn, which no cold boot of the same
    transcript can look like. The delta must sit at the buffer's head
    (positionally, never by timing) so delivered order is history_delta,
    then the seeded in-flight turn, then the buffered live frame.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    transcript = Transcript(tmp_path / "sessions" / "s1")
    await transcript.append_message(Message.user("visible before disconnect"))
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    parse_opened = asyncio.Event()
    release_parse = asyncio.Event()
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        events: list[Any] = []
        remote.subscribe(events.append)

        registrant.close()
        import os

        (tmp_path / "sessions" / "s1" / ".session.pid").write_text(str(os.getpid()))
        for _ in range(100):
            if remote._recovering:
                break
            await asyncio.sleep(0.02)
        assert remote._recovering is True

        gap_user = Message.user("durable while disconnected")
        await transcript.append_message(gap_user)
        gap_assistant = Message.assistant("answer while disconnected")
        await transcript.append_message(gap_assistant)

        # Instrument the threaded parse so a live relay frame lands AFTER the
        # socket is up and BEFORE the delta is emitted — the exact window
        # MAJOR-1 reproduced through the production recovery path.
        real_read = RemoteSession._read_transcript

        async def gated_read(self_inner: RemoteSession) -> Any:
            if self_inner is not remote or parse_opened.is_set():
                return await real_read(self_inner)
            parse_opened.set()
            await release_parse.wait()
            return await real_read(self_inner)

        replacement_handle = FakeHandle()
        replacement = Registrant(replacement_handle, kind="tui")
        replacement.start()
        try:
            from unittest.mock import patch as _patch

            with _patch.object(RemoteSession, "_read_transcript", gated_read):
                # The replacement owner is mid-turn (streaming generation 3)
                # and emits one live frame the moment the follower's reader
                # task is attached but the parse has not finished.
                for _ in range(200):
                    if parse_opened.is_set():
                        break
                    await asyncio.sleep(0.02)
                assert parse_opened.is_set()
                replacement_handle.emit_event(AgentStartEvent(generation=3))
                replacement_handle.emit_event(NoticeEvent(text="mid-parse live frame", kind="info"))
                await asyncio.sleep(0.05)
                release_parse.set()

                deadline = asyncio.get_running_loop().time() + 15
                while asyncio.get_running_loop().time() < deadline:
                    types = [event.type for event in events]
                    if not remote._recovering and "history_delta" in types and "notice" in types:
                        break
                    await asyncio.sleep(0.02)

            assert remote._recovering is False
            types = [event.type for event in events]
            delta_index = types.index("history_delta")
            notice_index = types.index("notice")
            # The durable gap delta is delivered BEFORE the live frame that
            # landed mid-parse — durable rows paint above the in-flight turn.
            assert delta_index < notice_index
            deltas = [event for event in events if event.type == "history_delta"]
            assert [str(m.id) for m in deltas[0].messages] == [gap_user.id, gap_assistant.id]
        finally:
            replacement.close()
    finally:
        if remote is not None:
            await remote.dispose()


@pytest.mark.asyncio
async def test_reconnect_tool_result_only_gap_settles_painted_card(
    tmp_path: Path, monkeypatch
) -> None:
    """MINOR-1 (review round 4): a results-only gap is delivered, not dropped.

    The tool call painted LIVE before the disconnect; the owner recorded the
    result and then died. Before the fix the gap replay early-returned on a
    results-only gap, so the delta never emitted and the painted card stayed
    ``interrupted`` forever while ``history()`` carried the real output. The
    gap must emit ONE delta carrying the result so the settled renderer can
    resolve it onto the painted card.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    transcript = Transcript(tmp_path / "sessions" / "s1")
    await transcript.append_message(Message.user("visible before disconnect"))
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        events: list[Any] = []
        remote.subscribe(events.append)

        # The call's assistant row is durable BEFORE the disconnect (so the
        # gap replay does not re-claim it), but its live-paint id must be in
        # the painted set — which the connect's history bind already seeded.
        call = ToolCall(id="call-live-1", name="read", arguments={"path": "/tmp/x"})
        assistant_row = Message(
            role="assistant",
            content=[TextContent(text="reading the file")],
            tool_calls=[call],
        )
        await transcript.append_message(assistant_row)
        # Re-load so the assistant row's id joins _history_ids / painted set,
        # mimicking a follower that painted the call live before the drop.
        await remote._load_history()
        painted_ids = {str(m.id) for m in remote.history()}

        registrant.close()
        import os

        (tmp_path / "sessions" / "s1" / ".session.pid").write_text(str(os.getpid()))
        for _ in range(100):
            if remote._recovering:
                break
            await asyncio.sleep(0.02)
        assert remote._recovering is True

        # The owner records ONLY the tool result during the gap, then dies.
        gap_result = Message.tool_result(
            ToolResult(
                tool_call_id="call-live-1",
                tool_name="read",
                content=[TextContent(text="real tool output")],
            )
        )
        await transcript.append_message(gap_result)
        assert str(gap_result.id) not in painted_ids

        replacement = Registrant(handle, kind="tui")
        replacement.start()
        try:
            deadline = asyncio.get_running_loop().time() + 15
            while asyncio.get_running_loop().time() < deadline:
                deltas = [event for event in events if event.type == "history_delta"]
                if not remote._recovering and deltas:
                    break
                await asyncio.sleep(0.02)
            assert remote._recovering is False
            # The results-only gap is NOT dropped: ONE delta carries the
            # settled result so the painted card can be resolved.
            deltas = [event for event in events if event.type == "history_delta"]
            assert len(deltas) == 1
            replayed = deltas[0].messages
            assert [str(getattr(m, "id", "")) for m in replayed] == [gap_result.id]
            assert replayed[0].role == "tool"
            assert replayed[0].tool_call_id == "call-live-1"
            assert str(gap_result.id) in {str(m.id) for m in remote.history()}
        finally:
            replacement.close()
    finally:
        if remote is not None:
            await remote.dispose()


@pytest.mark.asyncio
async def test_compact_now_during_recovery_refuses_in_user_language(
    tmp_path: Path, monkeypatch
) -> None:
    """NIT-1 (review round 4): /compact during recovery never leaks transport terms."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _wait_record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        registrant.close()
        import os

        (tmp_path / "sessions" / "s1" / ".session.pid").write_text(str(os.getpid()))
        for _ in range(100):
            if remote._recovering:
                break
            await asyncio.sleep(0.02)
        assert remote._recovering is True
        outcome = await remote.compact_now()
        assert outcome.ran is False
        assert outcome.reason == "unavailable"
        assert "reconnecting" in outcome.detail
        assert "attach" not in outcome.detail.lower()
    finally:
        if remote is not None:
            await remote.dispose()
