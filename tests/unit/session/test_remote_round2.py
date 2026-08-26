"""Round-2 review regressions: lifecycle dedupe, gap replay, cancel failure.

BLOCKER-1 (review round 2): dedupe by message id alone dropped the UPDATE and
END of every live message, so a follower's assistant rows froze at their
start event. These tests drive ``_on_wire_event`` with every lifecycle
interleaving and assert each legitimate beat arrives exactly once while true
replays (history/seed/reconnect re-advertising a completed row) stay dropped.

U6 (UX round 2): rows that became durable while the follower was disconnected
must be PAINTED after reconnect, not merely loaded into ``history()``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    NoticeEvent,
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
                ends = [event for event in events if event.type == "message_end"]
                if not remote._recovering and len(remote.history()) == 3 and len(ends) == 2:
                    break
                await asyncio.sleep(0.02)
            assert remote._recovering is False
            # history() carries the gap rows AND the transcript painted them:
            # one settled message event per gap row, in durable order, none
            # repeated, and the pre-disconnect row not replayed.
            ends = [event for event in events if event.type == "message_end"]
            painted_ids = [str(event.message.id) for event in ends]
            assert painted_ids == [gap_user.id, gap_assistant.id]
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
                    and [event.type for event in events].count("message_end") == 1
                ):
                    break
                await asyncio.sleep(0.02)
            assert remote._recovering is False
            first_cycle = [event.type for event in events]
            assert first_cycle.count("message_end") == 1

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
                assert [event.type for event in events].count("message_end") == 1
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
