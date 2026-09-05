"""Desktop stream algebra, resource bounds and durable retry invariants."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.types import Message
from local_operator.server.routes.desktop_sessions import Answer, Command, Image, Prompt
from local_operator.server.utils import desktop_sessions as module
from local_operator.server.utils.desktop_receipts import (
    DesktopReceipts,
    ReceiptConflict,
)
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.transcript import Transcript, read_transcript_page


@pytest.mark.asyncio
async def test_replay_receipts_precede_snapshot_even_when_snapshot_is_newer(tmp_path):
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        bridge.publish("event", {"type": "steering_delivered", "command_id": "semantic"})
        bridge.publish("event", {"type": "agent_end"})
        sub = bridge.subscribe()
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=0)
        assert (await anext(stream))["type"] == "open"
        assert (await anext(stream))["payload"]["command_id"] == "semantic"
        assert (await anext(stream))["payload"]["type"] == "agent_end"
        assert (await anext(stream))["type"] == "snapshot"
        await stream.aclose()
    assert bridge.remote is None and bridge.users == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", [-1, 999])
async def test_outside_retained_range_requires_snapshot(tmp_path, monkeypatch, cursor):
    monkeypatch.setattr(module, "REPLAY_COUNT", 2)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        for n in range(3):
            bridge.publish("event", {"value": n})
        stream = bridge.events(bridge.subscribe(), epoch=bridge.epoch, after_seq=cursor)
        assert (await anext(stream))["payload"]["gap"]
        assert (await anext(stream))["type"] == "snapshot"
        await stream.aclose()


@pytest.mark.asyncio
async def test_reopening_after_last_detach_invalidates_receipt_epoch(tmp_path):
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        old_epoch = bridge.epoch
        bridge.publish("event", {"type": "agent_end"})
    async with pool.session(sid) as reopened:
        assert reopened is bridge and reopened.epoch != old_epoch
        assert reopened.sequence == 0 and not reopened.replay
        stream = bridge.events(bridge.subscribe(), epoch=old_epoch, after_seq=1)
        assert (await anext(stream))["payload"]["gap"]
        await stream.aclose()


@pytest.mark.asyncio
async def test_slow_subscriber_overflow_is_explicit_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "REPLAY_BYTES", 400)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        slow = bridge.subscribe()
        slow.visible = slow.can_notify = True
        stream = bridge.events(slow, epoch=bridge.epoch, after_seq=0)
        await anext(stream)
        await anext(stream)
        for _ in range(20):
            bridge.publish("event", {"text": "x" * 200})
        assert slow.overflow and not slow.visible and not slow.can_notify
        assert slow.queue.qsize() == 1 and slow.queued_bytes == 0
        assert bridge.replay_bytes <= 400
        assert (await anext(stream))["type"] == "gap"
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert not bridge.subscribers


@pytest.mark.asyncio
async def test_watch_aggregation_does_not_resurrect_an_expired_viewer(tmp_path, monkeypatch):
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        remote = bridge.remote
        writes = []

        async def record(**kwargs):
            writes.append(kwargs)

        bridge.remote = cast(Any, SimpleNamespace(is_cold=False, update_desktop_watch=record))
        try:
            monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 100))
            expired = bridge.subscribe()
            expired.visible, expired.expires = True, 99
            notifier = bridge.subscribe()
            notifier.can_notify, notifier.expires = True, 110
            await bridge.refresh_watch()
            assert writes[-1] == {"visible": False, "can_notify": True}
            notifier.expires = 99
            await bridge.refresh_watch()
            assert writes[-1] == {"visible": False, "can_notify": False}
            bridge.subscribers.clear()
            with pytest.raises(KeyError):
                await bridge.watch(expired.id, visible=True, can_notify=True)
        finally:
            bridge.remote = remote


@pytest.mark.asyncio
async def test_the_last_lease_to_expire_is_still_reported_to_the_owner(tmp_path, monkeypatch):
    """Expiring the FINAL lease must recompute presence before the loop ends.

    `_expire_watches` returned as soon as no live lease remained, which left
    the owner holding whatever presence the previous pass asserted -- visible
    and notifiable -- for the rest of the session, because nothing else
    recomputes it once the loop is gone. The expiry that ends the loop is
    exactly the one the owner needs to hear about.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        remote = bridge.remote
        writes: list[dict[str, Any]] = []

        async def record(**kwargs):
            writes.append(kwargs)

        bridge.remote = cast(Any, SimpleNamespace(is_cold=False, update_desktop_watch=record))
        try:
            now = 100.0
            monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now))
            watcher = bridge.subscribe()
            watcher.visible, watcher.can_notify = True, True
            watcher.expires = 100.5
            await bridge.refresh_watch()
            assert writes[-1] == {"visible": True, "can_notify": True}

            # Time passes the only lease's TTL, and the expiry loop runs out.
            now = 101.0
            await bridge._expire_watches()

            assert writes[-1] == {
                "visible": False,
                "can_notify": False,
            }, "the owner was left believing a watcher is present after its lease expired"
        finally:
            bridge.remote = remote


@pytest.mark.asyncio
async def test_a_stream_lease_is_released_even_if_the_body_is_never_consumed(tmp_path):
    """A response whose generator never runs must not strand an acquired bridge.

    The bridge is acquired BEFORE the response exists, so an invalid session is
    a JSON error rather than a 200 with a broken stream. That leaves the
    release owed by something other than the generator: a client that
    disconnects between headers and body never iterates it, and the session
    would stay attached for the life of the process.
    """
    from local_operator.server.routes.desktop_sessions import events

    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    request = cast(Any, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())))
    request.app.state.desktop_sessions = pool

    response = await events(sid, request, epoch=None, after_seq=0)
    bridge = pool.bridges[sid]
    assert bridge.users == 1, "the stream did not acquire the bridge"

    # The body is DISCARDED without ever being iterated; Starlette still runs
    # the response's background task, which is what must return the lease.
    assert response.background is not None
    await response.background()
    assert bridge.users == 0, "an unconsumed stream leaked its bridge lease"

    # Idempotent: the generator's own teardown may still run afterwards.
    await response.background()
    assert bridge.users == 0


@pytest.mark.asyncio
async def test_active_bridge_is_never_evicted_or_duplicated(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "BRIDGE_COUNT", 1)
    pool = DesktopSessions(tmp_path)
    first, second = await pool.create(str(tmp_path)), await pool.create(str(tmp_path))
    async with pool.session(first) as active:
        async with pool.session(first) as shared:
            assert shared is active and shared.users == 2
        with pytest.raises(ValueError, match="Too many"):
            async with pool.session(second):
                pytest.fail("active entry was evicted")
        assert active.users == 1
    async with pool.session(second) as other:
        assert other.session_id == second
        assert first not in pool.bridges


@pytest.mark.asyncio
async def test_receipts_survive_adapter_restart_and_reject_changed_body(tmp_path):
    calls = []

    async def op():
        calls.append(True)
        return {"result": "real receipt"}

    first = DesktopReceipts(tmp_path)
    assert await first.run("s:id", {"argument": "one"}, op) == {"result": "real receipt"}
    replacement = DesktopReceipts(tmp_path)
    assert (await replacement.run("s:id", {"argument": "one"}, op))["replayed"]
    with pytest.raises(ReceiptConflict, match="different input"):
        await replacement.run("s:id", {"argument": "two"}, op)
    assert len(calls) == 1
    assert first.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_a_waiting_control_does_not_block_another_sessions_admission(tmp_path):
    receipts = DesktopReceipts(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_control():
        entered.set()
        await release.wait()
        return {"finished": True}

    async def other_admission():
        assert not release.is_set()
        return {"admitted": True}

    slow = asyncio.create_task(receipts.run("first:id", {"control": 1}, slow_control))
    try:
        await asyncio.wait_for(entered.wait(), 30)
        result = await asyncio.wait_for(
            receipts.run("second:id", {"prompt": 1}, other_admission), 30
        )
        assert result["admitted"]
    finally:
        release.set()
        await asyncio.wait_for(slow, 30)
    assert not receipts.locks


@pytest.mark.asyncio
async def test_interrupted_control_is_indeterminate_not_reexecuted(tmp_path):
    receipts = DesktopReceipts(tmp_path)
    calls = []

    async def interrupted():
        calls.append(True)
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await receipts.run("s:id", {"op": "control"}, interrupted)
    with pytest.raises(ReceiptConflict, match="indeterminate"):
        await DesktopReceipts(tmp_path).run("s:id", {"op": "control"}, interrupted)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_owner_idempotent_admission_can_resume_indeterminate_receipt(tmp_path):
    receipts = DesktopReceipts(tmp_path)

    async def interrupted():
        raise ConnectionError()

    with pytest.raises(ConnectionError):
        await receipts.run("s:id", {"text": "prompt"}, interrupted, retry_safe=True)

    async def owner_duplicate():
        return {"duplicate": True, "status": "admitted"}

    assert (await receipts.run("s:id", {"text": "prompt"}, owner_duplicate, retry_safe=True))[
        "duplicate"
    ]


@pytest.mark.asyncio
async def test_snapshot_history_has_inclusive_authoritative_boundary(tmp_path):
    transcript = Transcript(tmp_path)
    first = await transcript.append_message(Message.user("first"))
    second = await transcript.append_message(Message.assistant("second"))
    await transcript.append_message(Message.user("later"))
    page = read_transcript_page(tmp_path, through_id=second.id)
    assert [row.id for row in page.entries] == [first.id, second.id]
    assert not page.reconciled
    missing = read_transcript_page(tmp_path, through_id="evicted")
    assert missing.reconciled and not missing.entries
    assert [row.id for row in read_transcript_page(tmp_path, before_id=second.id).entries] == [
        first.id
    ]
    with pytest.raises(ValueError):
        read_transcript_page(tmp_path, before_id=first.id, through_id=second.id)


@pytest.mark.parametrize(
    "fields",
    [
        {"request_id": "bad", "text": "hello"},
        {"request_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "text": "/settings"},
        {"request_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "text": ""},
    ],
)
def test_invalid_prompts_are_rejected_before_owner_binding(fields):
    with pytest.raises(ValueError):
        Prompt.model_validate(fields)


@pytest.mark.parametrize("op", ["prompt", "steer"])
def test_canonical_wire_accepts_image_only_without_invented_text(op):
    from local_operator.mobile.types import ContinuationCommand, validate_control_frame

    payload = {
        "op": op,
        "command_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "session_id": "123456abcdef",
        "text": "",
        "images": [{"data_b64": "aW1hZ2U=", "mime_type": "image/png"}],
    }
    validate_control_frame(payload)
    assert ContinuationCommand.from_json(payload).text == ""
    for images in ([], [{}], [{"data_b64": ""}], [{"data_b64": 1}]):
        with pytest.raises(ValueError):
            validate_control_frame({**payload, "images": images})
        with pytest.raises(ValueError):
            ContinuationCommand.from_json({**payload, "images": images})
    with pytest.raises(ValueError):
        validate_control_frame({"op": "peer_message", "text": "", "images": payload["images"]})


def test_route_response_models_publish_the_real_canonical_contract():
    from local_operator.server.app import app

    schema = app.openapi()
    expected = {
        ("/v1/desktop/sessions", "get"): "SessionList",
        ("/v1/desktop/sessions", "post"): "CreatedSession",
        ("/v1/desktop/sessions/{session_id}", "get"): "SessionSnapshot",
        ("/v1/desktop/sessions/{session_id}/history", "get"): "HistoryPage",
        ("/v1/desktop/sessions/{session_id}/messages", "post"): "MessageAdmission",
        ("/v1/desktop/sessions/{session_id}/commands", "post"): "CommandReceipt",
        ("/v1/desktop/sessions/{session_id}/answers", "post"): "AnswerReceipt",
        ("/v1/desktop/sessions/{session_id}/watch", "post"): "WatchReceipt",
    }
    for (path, method), name in expected.items():
        response = schema["paths"][path][method]["responses"]["200"]
        ref = response["content"]["application/json"]["schema"]["$ref"]
        envelope = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
        result = envelope["properties"]["result"]
        assert name in str(result), (path, result)


def test_command_and_answer_shapes_are_closed():
    with pytest.raises(ValueError):
        Command(request_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", command="goal extra")
    with pytest.raises(ValueError):
        Answer.model_validate({"epoch": "epoch", "request_id": "request", "approved": "true"})
    with pytest.raises(ValueError):
        Answer(epoch="epoch", request_id="request", value="answer")
    with pytest.raises(ValueError):
        Image(data_b64="not base64", mime_type="image/png")
