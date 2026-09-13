"""Desktop stream algebra, resource bounds and durable retry invariants."""

import asyncio
import base64
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.harness.types import Message
from local_operator.resume import ORIGIN_FORK, ORIGIN_SUBAGENT, mark_session_origin
from local_operator.server.routes import capabilities, desktop_sessions
from local_operator.server.routes.desktop_sessions import Answer, Command, Image, Prompt
from local_operator.server.utils import desktop_sessions as module
from local_operator.server.utils.desktop_receipts import (
    DesktopReceipts,
    ReceiptConflict,
)
from local_operator.server.utils.desktop_sessions import (
    DesktopSessions,
    SubagentChildUnavailable,
)
from local_operator.session.runtime import registry
from local_operator.session.transcript import (
    ENTRY_MESSAGE,
    TRANSCRIPT_FILENAME,
    Transcript,
    TranscriptEntry,
    read_transcript_page,
)


async def _armed_warm(bridge: Any) -> asyncio.Task[None]:
    """The engage task the lease-warm loop starts, after its first step.

    The lease-driven warm is ARMED by `/watch` rather than started inside it,
    because the retry and its backoff have to live in one place and that place
    is the loop -- so the task appears one event-loop turn after the heartbeat
    that armed it. Everything the ordering promises is unchanged: the presence
    record still lands before the engage, and `/watch` still returns without
    awaiting either.
    """
    for _ in range(64):
        if bridge.warm_task is not None:
            return bridge.warm_task
        await asyncio.sleep(0)
    raise AssertionError("a live visible lease did not start a warm")


async def _until(predicate: Callable[[], bool], *, why: str, timeout: float = 30.0) -> None:
    """Poll until ``predicate`` holds, or fail with ``why`` after ``timeout``.

    A LOAD-TOLERANT bound on the state the assertion is about, rather than a
    fixed count of sleeps or a wait on an attempt COUNT. The loop records an
    attempt (the fake bind appends it) before it awaits the engage and updates
    its pace, so a poll on the attempt count can read the pace field inside that
    window and fail for scheduling reasons rather than for the rule under test
    (QA round 2, Q1: 6 failures in 21 isolated runs).

    ``time`` here is the REAL clock: these tests patch the MODULE's ``time``,
    not this module's, so the bound does not move with the frozen clock.
    """
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, why
        await asyncio.sleep(0.01)


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
        (
            "/v1/desktop/sessions/{session_id}/children/{child_id}/transcript",
            "get",
        ): "ChildTranscriptPage",
        ("/v1/desktop/sessions/{session_id}/history", "get"): "HistoryPage",
        ("/v1/desktop/sessions/{session_id}/messages", "post"): "MessageAdmission",
        ("/v1/desktop/sessions/{session_id}/commands", "post"): "CommandReceipt",
        ("/v1/desktop/sessions/{session_id}/answers", "post"): "AnswerReceipt",
        ("/v1/desktop/sessions/{session_id}/watch", "post"): "WatchReceipt",
        ("/v1/desktop/sessions/{session_id}/warm", "post"): "WarmReceipt",
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


@pytest.mark.asyncio
async def test_only_a_typed_actionable_error_reaches_the_user(tmp_path):
    """`errors()` echoes a ConnectionError's text only when its TYPE vouches for it.

    The relay used to echo `str(error)` for EVERY ConnectionError on the
    strength of a docstring claiming they were limited to the vetted startup
    reasons. Nothing enforced that, and `attach_client` raises bare
    ConnectionErrors from a dozen places, so the renderer was shown an internal
    control port and another session's id verbatim -- painted as "The message
    was not sent: ..." (review round 2, MAJOR-1).

    Vettedness cannot be recovered from message text, so it rides the type.
    """
    from fastapi import HTTPException

    from local_operator.server.routes.desktop_sessions import errors
    from local_operator.session.runtime.launch import ActionableConnectionError

    generic = "Session owner is unavailable. Reconnect and reconcile before retrying."

    async def relay(error: BaseException) -> str:
        with pytest.raises(HTTPException) as raised:
            async with errors():
                raise error
        assert raised.value.status_code == 503
        return str(raised.value.detail)

    leaky = [
        ConnectionError(
            "owner socket unreachable: [Errno 61] Connect call failed ('127.0.0.1', 54321)"
        ),
        ConnectionError("owner moved to another conversation (abc123secretsession)"),
        ConnectionError("owner replied 'refused', not its state"),
        ConnectionError("owner runs protocol v1; attach needs >= 2"),
    ]
    for error in leaky:
        detail = await relay(error)
        assert detail == generic, detail
        # The specific values from the reproduction must be absent, not merely
        # reworded: these are an internal port and another session's identifier.
        assert "54321" not in detail
        assert "abc123secretsession" not in detail
        assert "127.0.0.1" not in detail

    # The vetted sentence still survives -- suppressing it would re-break the
    # "no model provider configured" case this relay exists to report (QA Q1).
    vetted = (
        "No model provider is configured yet. Connect one in Settings > Providers, "
        "then send the message again."
    )
    assert await relay(ActionableConnectionError(vetted)) == vetted


@pytest.mark.asyncio
async def test_served_list_order_is_the_catalogs_rank(tmp_path):
    """The HTTP surface inherits `rank_entries`, wake key included.

    `DesktopSessions.list` orders through `load_catalog` -> `rank_entries`, so
    every ordering key the sidebar gains lands on this endpoint too -- the wake
    key among them. That is easy to miss because the mobile daemon has its OWN
    sort and is genuinely untouched by the same change, so "the catalog decides"
    holds for one remote surface and not the other. Asserting the served order
    IS `rank_entries` keeps the desktop half honest without restating the ladder
    here: if the two ever diverge this fails, whichever one moved.
    """
    from local_operator.session.catalog import load_catalog, rank_entries
    from local_operator.wakes import store as wake_store

    # Cold sessions only, so the tier ties and the wake key is what decides.
    # The armed session is the OLDEST, which is exactly where birth date alone
    # would sort it last.
    for session_id, created in [("newest", 300), ("middle", 200), ("armed", 100)]:
        path = tmp_path / "sessions" / session_id
        path.mkdir(parents=True)
        (path / "created_at.json").write_text(str(created))
        (path / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(tmp_path)}))
    wake_store.write_entry(
        tmp_path,
        "armed",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "next_due_at": 10**12}],
    )

    served = [row["id"] for row in await DesktopSessions(tmp_path).list(50)]
    catalog = load_catalog(tmp_path, limit=50)
    assert served == [entry.id for entry in catalog]
    # Re-ranked from a SHUFFLED input, so this pins the sort rather than merely
    # agreeing that two calls returned the same list.
    assert served == [entry.id for entry in rank_entries(catalog[::-1])]
    # And concretely: the armed row leads despite being the oldest.
    assert served == ["armed", "newest", "middle"]


@pytest.mark.asyncio
async def test_durable_attachment_is_readable_without_starting_an_owner(tmp_path):
    """A digest from a history row resolves to bytes on a cold conversation.

    This is the read that makes durable images renderable at all: `/history`
    serves rows verbatim, and an image block over the externalisation floor is a
    digest with the payload stripped, so a reader that cannot resolve a digest
    can only ever know an image WAS there.

    It deliberately does NOT go through `DesktopSessions.session`. A finished
    conversation's screenshot must be readable without starting an owner
    process, which is the same argument `acknowledge_attention` already makes
    for a read receipt -- and the assertion that no bridge was created is what
    keeps a later refactor from quietly acquiring one per image.
    """
    from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore

    raw = b"\x89PNG\r\n\x1a\nnot-a-real-png-but-real-bytes"
    store = AttachmentStore(tmp_path / ATTACHMENTS_DIRNAME)
    ref = store.put(base64.b64encode(raw).decode("ascii"), "image/png")
    assert ref is not None

    session = tmp_path / "sessions" / "0123456789ab"
    session.mkdir(parents=True)
    (session / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(tmp_path)}))

    pool = DesktopSessions(tmp_path)
    data, mime_type = await pool.attachment("0123456789ab", ref.digest)
    assert data == raw and mime_type == "image/png"
    assert pool.bridges == {}

    # A miss is ordinary, not a fault: the store's own contract is that an
    # interrupted write or a hand-pruned store degrades to a placeholder, and
    # `errors()` maps KeyError to 404 rather than letting it reach a 500.
    with pytest.raises(KeyError):
        await pool.attachment("0123456789ab", "f" * 32)
    # A well-formed id whose directory is simply absent. This reaches the
    # `is_dir()` check ONLY -- `ffffffffffff` already satisfies `SESSION_ID`,
    # so it says nothing about either gate. The two tests below carry those.
    with pytest.raises(KeyError):
        await pool.attachment("ffffffffffff", ref.digest)


@pytest.mark.asyncio
async def test_attachment_refuses_a_session_id_that_is_not_the_session_shape(tmp_path):
    """The shape guard rejects before a path is built, not after.

    `self.root / "sessions" / session_id` turns the id straight into a path, so
    an id carrying `..` would climb out of the sessions namespace and read a
    sibling directory's marker. Unlike the digest, the session id has no
    route-declaration pattern -- `SESSION_ID.fullmatch` inside the read IS the
    whole gate, which is why it needs a case that a merely-absent directory
    cannot satisfy: this id is one `is_dir()` alone would also reject, so the
    assertion is that the ESCAPE never happens rather than that the read
    missed. The planted marker is what distinguishes the two -- with the guard
    removed the traversal resolves onto a real directory.
    """
    from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore

    raw = b"\x89PNG\r\n\x1a\ntraversal-target"
    ref = AttachmentStore(tmp_path / ATTACHMENTS_DIRNAME).put(
        base64.b64encode(raw).decode("ascii"), "image/png"
    )
    assert ref is not None
    # A real directory one level above the sessions namespace, so a guard-free
    # read would find `is_dir()` true and `is_user_session()` true and serve.
    outside = tmp_path / "sessions" / ".." / "elsewhere"
    outside.mkdir(parents=True)
    pool = DesktopSessions(tmp_path)

    with pytest.raises(KeyError):
        await pool.attachment("../elsewhere", ref.digest)
    with pytest.raises(KeyError):
        await pool.attachment("0123456789AB", ref.digest)
    assert outside.is_dir(), "the traversal target must exist, or this proves nothing"


@pytest.mark.asyncio
async def test_attachment_refuses_a_subagent_origin_session(tmp_path):
    """A delegated run's screenshots are not the desktop surface's to serve.

    A subagent session is a machine's own work the user never opened, and the
    desktop surface does not list it anywhere else either. `is_user_session` is
    the ONLY thing keeping this route out of those conversations, and it is a
    one-token edit away from removal -- so the case is an existing directory
    that differs from the passing one in nothing but its origin marker,
    written by the production `mark_session_origin` rather than hand-forged.

    Canaried in both directions on purpose: the same digest and an identically
    built directory WITHOUT the marker must serve, or a passing assertion here
    would only mean the fixture was broken.
    """
    from local_operator.resume import ORIGIN_SUBAGENT, mark_session_origin
    from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore

    raw = b"\x89PNG\r\n\x1a\nsubagent-screenshot"
    ref = AttachmentStore(tmp_path / ATTACHMENTS_DIRNAME).put(
        base64.b64encode(raw).decode("ascii"), "image/png"
    )
    assert ref is not None

    def make(session_id: str) -> Any:
        directory = tmp_path / "sessions" / session_id
        directory.mkdir(parents=True)
        (directory / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(tmp_path)}))
        return directory

    users = make("0123456789ab")
    child = make("abcdef012345")
    mark_session_origin(child, ORIGIN_SUBAGENT, label="review", agent="reviewer")
    pool = DesktopSessions(tmp_path)

    # The marker is the only difference, so the user session must still serve.
    data, _ = await pool.attachment("0123456789ab", ref.digest)
    assert data == raw and users.is_dir()
    with pytest.raises(KeyError):
        await pool.attachment("abcdef012345", ref.digest)


def test_attachment_digest_shape_is_enforced_by_the_route_declaration():
    """Traversal is unreachable by construction, not by a handler check.

    The store turns a digest straight into `<root>/<digest>.bin`, so the only
    safe place for the constraint is the path declaration: FastAPI rejects a
    non-matching path before the handler runs, and no later edit inside the
    handler can route around it. Asserting on the published schema rather than
    on a string literal is what keeps this true if the annotation moves.
    """
    from local_operator.server.app import app

    schema = app.openapi()
    path = "/v1/desktop/sessions/{session_id}/attachments/{digest}"
    digest = next(
        parameter
        for parameter in schema["paths"][path]["get"]["parameters"]
        if parameter["name"] == "digest"
    )
    assert digest["schema"]["pattern"] == r"^[a-f0-9]{32}$"
    # And the response is raw bytes rather than the CRUD envelope: a JSON
    # envelope has nowhere to put an image.
    assert "application/json" not in schema["paths"][path]["get"]["responses"]["200"].get(
        "content", {}
    )


@pytest.mark.asyncio
async def test_attachment_route_sets_no_cache_control_of_its_own(tmp_path):
    """Caching belongs to the boundary, and this asserts the layer that owns it.

    An earlier draft set `public, max-age=31536000, immutable` on this route.
    That header never reached the wire, because `managed_desktop_boundary`
    overwrites `Cache-Control` for everything under `/v1/desktop/` -- which is
    exactly why an assertion made THROUGH the middleware cannot pin this down:
    the effective value is `no-store` on the fixed tree AND on a tree where the
    route re-adds `immutable`, so such a test passes on both and detects
    nothing. The regression is a claim the route makes about caching, so the
    assertion has to read the route's OWN response before anything rewrites it.

    Calling the endpoint function directly is what makes that possible. The
    wire-level companion below keeps the effective header covered too; neither
    assertion substitutes for the other.
    """
    from local_operator.server.routes.desktop_sessions import attachment
    from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore

    raw = b"\x89PNG\r\n\x1a\nuncached"
    ref = AttachmentStore(tmp_path / ATTACHMENTS_DIRNAME).put(
        base64.b64encode(raw).decode("ascii"), "image/png"
    )
    assert ref is not None
    session = tmp_path / "sessions" / "0123456789ab"
    session.mkdir(parents=True)
    (session / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(tmp_path)}))

    state = SimpleNamespace(
        desktop_sessions=DesktopSessions(tmp_path),
        config_manager=SimpleNamespace(config_dir=tmp_path),
    )
    request = cast(Any, SimpleNamespace(app=SimpleNamespace(state=state)))
    response = await attachment("0123456789ab", ref.digest, request)

    assert response.body == raw
    assert "cache-control" not in response.headers
    # And the header the route DOES own is on that same pre-middleware response.
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.asyncio
async def test_attachment_bytes_are_not_cached_by_any_shared_cache(tmp_path, monkeypatch):
    """The digest is content-addressed, and the response is still `no-store`.

    An `immutable` header would be correct about the BYTES and wrong about the
    RESPONSE: `managed_desktop_boundary` marks everything under `/v1/desktop/`
    no-store because it is bearer-gated session data. This is the WIRE half --
    it proves the boundary is in force for this path, which is what makes the
    route's silence above the correct behaviour rather than an omission.
    """
    from fastapi.testclient import TestClient

    from local_operator.server.app import app
    from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path))
    raw = b"\x89PNG\r\n\x1a\nbytes"
    ref = AttachmentStore(tmp_path / ATTACHMENTS_DIRNAME).put(
        base64.b64encode(raw).decode("ascii"), "image/png"
    )
    assert ref is not None
    session = tmp_path / "sessions" / "0123456789ab"
    session.mkdir(parents=True)
    (session / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(tmp_path)}))
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)

    with TestClient(app) as client:
        response = client.get(
            f"/v1/desktop/sessions/0123456789ab/attachments/{ref.digest}",
            headers={"Authorization": "Bearer token"},
        )
    assert response.status_code == 200
    assert response.content == raw
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
async def test_stored_mime_is_allowlisted_before_it_becomes_a_header(tmp_path):
    """A sidecar cannot choose this response's `Content-Type`.

    The store records the mime its CALLER supplied and never verifies the
    sidecar: `transcript._externalize_attachments` copies `block["mime_type"]`
    verbatim with no allowlist of its own. No ingress puts a non-image mime in
    the store today, but that is a property of callers upstream, and this
    boundary is what pays if one changes -- so it is asserted HERE rather than
    assumed there.

    Three cases, each a real failure rather than a hypothetical:

    - `text/html` and `image/svg+xml` round-trip out of the store and, unfixed,
      are served as active content from an authenticated local port. `svg+xml`
      is in `routes/static.py`'s broader list and deliberately NOT in this one.
    - A CRLF-bearing mime is not merely wrong, it is unserveable: h11 rejects
      the header and the client gets no response at all, which contradicts this
      route's documented "404, never 500".

    Every one must degrade to opaque bytes while the BODY still arrives intact
    -- the fix is a refusal to label, not a refusal to serve.
    """
    from local_operator.server.routes.desktop_sessions import (
        ATTACHMENT_FALLBACK_MIME,
        attachment,
    )
    from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore

    store = AttachmentStore(tmp_path / ATTACHMENTS_DIRNAME)
    session = tmp_path / "sessions" / "0123456789ab"
    session.mkdir(parents=True)
    (session / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(tmp_path)}))
    state = SimpleNamespace(
        desktop_sessions=DesktopSessions(tmp_path),
        config_manager=SimpleNamespace(config_dir=tmp_path),
    )
    request = cast(Any, SimpleNamespace(app=SimpleNamespace(state=state)))

    hostile = {
        "text/html": b"<script>alert(document.domain)</script>",
        "image/svg+xml": b"<svg xmlns='http://www.w3.org/2000/svg'><script/></svg>",
        "image/png\r\nX-Injected: yes": b"\x89PNG\r\n\x1a\ncrlf",
        "application/x-msdownload": b"MZ\x90\x00executable",
    }
    for mime, raw in hostile.items():
        ref = store.put(base64.b64encode(raw).decode("ascii"), mime)
        assert ref is not None
        # The store really did keep the hostile value -- otherwise this test
        # would be asserting against an input that never reaches the boundary.
        assert store.get(ref.digest) == (base64.b64encode(raw).decode("ascii"), mime)
        response = await attachment("0123456789ab", ref.digest, request)
        assert response.media_type == ATTACHMENT_FALLBACK_MIME, mime
        assert "\r" not in response.headers["content-type"], mime
        assert "html" not in response.headers["content-type"], mime
        # Refusing the label must not corrupt the bytes.
        assert response.body == raw, mime

    # And the allowlisted types still pass through untouched, or the fix would
    # be a blanket downgrade that breaks every real screenshot.
    for mime in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        raw = b"\x89PNG\r\n\x1a\n" + mime.encode("ascii")
        ref = store.put(base64.b64encode(raw).decode("ascii"), mime)
        assert ref is not None
        response = await attachment("0123456789ab", ref.digest, request)
        assert response.media_type == mime
        assert response.body == raw


async def _seed_searchable_session(root: Path, session_id: str, *, opener: str, body: str) -> None:
    """A real canonical session on disk, written through the real transcript.

    Async rather than an ``asyncio.run`` wrapper because every caller already
    runs inside the event loop the test client owns.
    """
    from local_operator.harness.types import Message, TextContent
    from local_operator.session.transcript import Transcript

    session = root / "sessions" / session_id
    session.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(session)
    for role, text in (("user", opener), ("assistant", body)):
        await transcript.append_message(
            Message(role=cast(Any, role), content=[TextContent(text=text)])
        )


@pytest.mark.asyncio
async def test_search_finds_a_session_by_what_was_said_in_it(tmp_path, monkeypatch):
    """The whole point of the route: a session whose opener says nothing about
    the subject it became is still findable by the subject.

    Driven over real loopback HTTP through the app, because the thing being
    verified is the WIRE contract the desktop chat search consumes — a unit call
    into the search module would not exercise the route, its auth gate, or the
    response model.
    """
    from fastapi.testclient import TestClient

    from local_operator.server.app import app
    from local_operator.session.session_search import RANK_BODY

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path))
    await _seed_searchable_session(
        tmp_path,
        "aaaa1111",
        opener="hey can you look at this thing",
        body="The retention sweep is evicting live session directories.",
    )
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)

    with TestClient(app) as client:
        denied = client.get("/v1/desktop/sessions/search?q=retention")
        assert denied.status_code in (401, 403)
        response = client.get(
            "/v1/desktop/sessions/search?q=retention",
            headers={"Authorization": "Bearer token"},
        )

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["query"] == "retention"
    assert [row["id"] for row in result["sessions"]] == ["aaaa1111"]
    # The row's own name does not contain the query, so the answer says why it
    # surfaced rather than leaving the client to guess.
    assert result["sessions"][0]["body_match"] is True
    assert result["sessions"][0]["rank"] == RANK_BODY
    assert result["sessions"][0]["name"] == "hey can you look at this thing"


@pytest.mark.asyncio
async def test_search_is_declared_before_the_session_id_route(tmp_path, monkeypatch):
    """FastAPI matches in declaration order, so a parent route declared first
    would swallow ``/v1/desktop/sessions/search`` and answer with a session
    snapshot (a 404 for an id that does not exist) instead of search results.
    Pinned on the routing table itself, not on one request's outcome: a later
    edit that reorders the handlers is what this catches."""
    from fastapi.testclient import TestClient

    from local_operator.server.app import _iter_routes, app

    paths = [
        path
        for path, _methods in _iter_routes(app.routes)
        if path.startswith("/v1/desktop/sessions")
    ]
    assert paths.index("/v1/desktop/sessions/search") < paths.index(
        "/v1/desktop/sessions/{session_id}"
    )

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path))
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    with TestClient(app) as client:
        response = client.get(
            "/v1/desktop/sessions/search?q=", headers={"Authorization": "Bearer token"}
        )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["sessions"] == []


@pytest.mark.asyncio
async def test_an_oversized_or_invalid_query_is_refused_without_echoing_it(tmp_path, monkeypatch):
    """The query is bounded at the boundary, and the refusal must not quote the
    rejected input back: this route sits under `/v1/desktop/`, where pydantic's
    default 422 body would echo whatever the caller sent."""
    from fastapi.testclient import TestClient

    from local_operator.server.app import app

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path))
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    secret_looking = "x" * 300

    with TestClient(app) as client:
        too_long = client.get(
            f"/v1/desktop/sessions/search?q={secret_looking}",
            headers={"Authorization": "Bearer token"},
        )
        bad_limit = client.get(
            "/v1/desktop/sessions/search?q=ok&limit=501",
            headers={"Authorization": "Bearer token"},
        )

    assert too_long.status_code == 422
    assert secret_looking not in too_long.text
    assert bad_limit.status_code == 422


@pytest.mark.asyncio
async def test_a_warm_whose_engage_fails_is_still_a_success_for_the_caller(tmp_path, monkeypatch):
    """R3: an engage failure must never reach a user who has only typed.

    The warm is fired speculatively from the renderer's composer, so any
    non-2xx it can produce becomes an error banner triggered BY TYPING. The
    state at return time is honestly "an engage was started"; whether that
    engage then dies of a missing provider or a refused dial is the SEND's
    problem to report, and the send still reports it through its own
    ConnectionError ladder.
    """
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from local_operator.server.routes import desktop_sessions as routes

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "warm-token")
    app = FastAPI()
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    app.include_router(routes.router)
    sid = await pool.create(str(tmp_path))

    failures: list[BaseException] = []

    async def exploding_bind(*, foreground: bool = True) -> None:
        error = ConnectionError("no runtime for this test")
        failures.append(error)
        raise error

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer warm-token"},
    ) as client:
        async with pool.session(sid) as bridge:
            assert bridge.remote is not None
            monkeypatch.setattr(bridge.remote, "_ensure_bound", exploding_bind)
            response = await client.post(f"/v1/desktop/sessions/{sid}/warm", json={})
            assert response.status_code == 200, response.text
            assert response.json()["result"]["state"] == "warming"
            task = bridge.warm_task
            assert task is not None
            # The task must SWALLOW it, not merely fail out of band: an
            # unretrieved exception would also surface as a warning the
            # operator has to read.
            await task
            assert task.exception() is None
    assert failures, "the engage was never actually attempted"
    await pool.close()


@pytest.mark.asyncio
async def test_detaching_a_bridge_cancels_the_warm_it_started(tmp_path):
    """R5: a warm must not outlive the facade it was started against.

    An engage landing after `dispose()` holds a freshly spawned runtime
    resident with no viewer left to release it — the TUI shipped exactly this
    leak once, where a session swap's engage kept the old runtime up for the
    process's life.
    """
    entered = asyncio.Event()

    async def never_finishes(*, foreground: bool = True) -> None:
        entered.set()
        await asyncio.sleep(3600)

    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        bridge.remote._ensure_bound = never_finishes  # type: ignore[method-assign]
        assert await bridge.warm() == "warming"
        task = bridge.warm_task
        assert task is not None
        await asyncio.wait_for(entered.wait(), timeout=10)
    # Leaving the context detaches the last user, which must take the warm with
    # it rather than leaving it parked on the loop.
    assert task.cancelled() or task.done()
    assert bridge.warm_task is None
    await pool.close()


@pytest.mark.asyncio
async def test_a_warm_on_an_already_engaged_session_starts_nothing(tmp_path):
    """The cheap path: an engaged viewer answers `warm` without a task at all.

    Pins the short-circuit rather than the lock behind it, because this is the
    common case once a session is live and the renderer keeps firing warms.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        # `is_cold` is three field reads; making the viewer look bound is
        # enough to exercise the branch without a real runtime. A plain class
        # rather than SimpleNamespace because `dispose()` tests the client for
        # set membership, and SimpleNamespace defines __eq__ and so is
        # unhashable.

        class BoundClient:
            connected = True

            def close(self) -> None:
                pass

        bridge.remote._client = BoundClient()  # type: ignore[assignment]
        bridge.remote._ready_for_events = True
        assert await bridge.warm() == "warm"
        assert bridge.warm_task is None
    await pool.close()


@pytest.mark.asyncio
async def test_an_unknown_session_is_a_404_rather_than_a_warm(tmp_path, monkeypatch):
    """The two refusals the route DOES keep, so the 200 rule is not read as
    "this route can never fail". An id that names nothing is not an engage
    that failed, it is a call that was never admissible."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from local_operator.server.routes import desktop_sessions as routes

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "warm-token")
    app = FastAPI()
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    app.include_router(routes.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer warm-token"},
    ) as client:
        missing = await client.post("/v1/desktop/sessions/aaaaaaaaaaaa/warm", json={})
        # `extra="forbid"` is what makes an invented option a named 422 rather
        # than a silently ignored field (R12's backend half).
        extra = await client.post("/v1/desktop/sessions/aaaaaaaaaaaa/warm", json={"eager": True})
    assert missing.status_code == 404, missing.text
    assert extra.status_code == 422, extra.text
    await app.state.desktop_sessions.close()


@pytest.mark.asyncio
async def test_the_warm_route_returns_while_the_engage_is_still_running(tmp_path, monkeypatch):
    """R2: the whole point — the response must NOT wait for the spawn.

    Awaiting the engage inside the handler would not remove the ~1.15 s cold
    cost, it would relocate it from the send onto a request the renderer fires
    while the user is still typing. Pinned structurally: the engage is parked
    on an event this test controls, so a handler that awaited it could not
    return at all, and the assertion is that the response arrived anyway with
    the bind lock still held.
    """
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from local_operator.server.routes import desktop_sessions as routes

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "warm-token")
    app = FastAPI()
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    app.include_router(routes.router)
    sid = await pool.create(str(tmp_path))

    entered = asyncio.Event()
    release = asyncio.Event()

    async def parked_bind(*, foreground: bool = True) -> None:
        entered.set()
        await release.wait()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer warm-token"},
    ) as client:
        async with pool.session(sid) as bridge:
            assert bridge.remote is not None
            monkeypatch.setattr(bridge.remote, "_ensure_bound", parked_bind)
            response = await asyncio.wait_for(
                client.post(f"/v1/desktop/sessions/{sid}/warm", json={}), timeout=10
            )
            assert response.status_code == 200, response.text
            assert response.json()["result"]["state"] == "warming"
            # The engage is demonstrably still running at the moment the
            # caller already has its answer.
            await asyncio.wait_for(entered.wait(), timeout=10)
            task = bridge.warm_task
            assert task is not None and not task.done()
            release.set()
            await task
    await pool.close()


@pytest.mark.asyncio
async def test_a_second_warm_during_an_engage_starts_no_second_task(tmp_path):
    """Idempotence at the bridge: a renderer firing repeatedly costs one task.

    The lock is what makes a duplicate SAFE; this check is what makes it free.
    """
    entered = asyncio.Event()
    release = asyncio.Event()
    binds = 0

    # Parked INSIDE the lock rather than replacing `_ensure_bound` wholesale:
    # `engage_in_flight` reads `_bind_lock`, so a stub that skipped the real
    # acquisition would make the predicate answer False and the test would
    # pass for the wrong reason.
    async def parked_bind(*, foreground: bool) -> None:
        nonlocal binds
        binds += 1
        entered.set()
        await release.wait()

    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        bridge.remote._bind_under_lock = parked_bind  # type: ignore[method-assign]
        assert await bridge.warm() == "warming"
        first = bridge.warm_task
        await asyncio.wait_for(entered.wait(), timeout=10)
        # `engage_in_flight` reads the bind lock, which `_ensure_bound` is
        # holding while parked, so the second call must recognise it.
        assert bridge.remote.engage_in_flight
        assert await bridge.warm() == "warming"
        assert bridge.warm_task is first, "a second warm must not replace the task"
        release.set()
        await first  # type: ignore[arg-type]
    assert binds == 1
    await pool.close()


@pytest.mark.asyncio
async def test_a_warm_survives_its_own_request_while_a_subscriber_holds_the_bridge(tmp_path):
    """The lifetime rule, stated both ways, because it surprised the design.

    A bridge is reference-counted and `_detach()` cancels an in-flight warm, so
    a warm issued while NOBODY else holds the bridge is cancelled the moment
    its own request releases -- the warm request is itself the last user. That
    is correct: a spawn must not outlive the facade it was started against.

    It is also exactly why the cancel is not a bug in the feature. The renderer
    warms from a composer that lives inside a mounted `SessionPanel`, which
    holds an events subscription, so the real caller always has a second user
    on the bridge and the engage survives to be found by the send.

    Pinned in BOTH directions because the two halves argue with each other: a
    future reader who sees only the first half deletes the cancel and
    reintroduces the leak; one who sees only the second assumes the warm is
    unconditionally durable and moves the renderer's warm outside the panel.
    """
    entered = asyncio.Event()
    release = asyncio.Event()

    async def parked_bind(*, foreground: bool = True) -> None:
        entered.set()
        await release.wait()

    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))

    # WITHOUT another holder: the warm dies with its request.
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        bridge.remote._ensure_bound = parked_bind  # type: ignore[method-assign]
        assert await bridge.warm() == "warming"
        alone = bridge.warm_task
        assert alone is not None
        await asyncio.wait_for(entered.wait(), timeout=10)
    assert alone.cancelled() or alone.done()

    entered.clear()
    # WITH a holder, as a subscribed panel always is: the warm outlives it.
    holder = pool.session(sid)
    held = await holder.__aenter__()
    try:
        assert held.remote is not None
        held.remote._ensure_bound = parked_bind  # type: ignore[method-assign]
        async with pool.session(sid) as requester:
            assert await requester.warm() == "warming"
            survivor = requester.warm_task
            assert survivor is not None
            await asyncio.wait_for(entered.wait(), timeout=10)
        assert not survivor.done(), "a held bridge must not cancel the warm"
    finally:
        release.set()
        await holder.__aexit__(None, None, None)
    await pool.close()


@pytest.mark.asyncio
async def test_a_second_warm_never_orphans_the_first_task(tmp_path):
    """Review round 1, MINOR-1: the bridge must reference every task it starts.

    `engage_in_flight` samples the facade's bind lock, which says nothing about
    whether THIS BRIDGE already owns a warm task that has not reached the lock
    yet. A second request resumed out of `acquire()` ahead of the first task's
    first step therefore passed the predicate, and assigning `warm_task` again
    dropped the first task's only reference -- reviewer's repro reported 2 tasks
    started with the bridge holding the second, the first escaping `_detach()`'s
    cancel.

    Driven at the seam rather than over HTTP: the hazard is the guard, and
    scheduling two real requests to interleave at exactly that point is not
    something a test can make deterministic.
    """
    started: list[asyncio.Task[None]] = []
    release = asyncio.Event()

    async def parked_warm() -> None:
        # Never reaches the bind lock, which is the whole point: the second
        # caller must be refused by the bridge's own bookkeeping, not by a
        # lock the first task has not taken.
        await release.wait()

    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        bridge.remote.warm_runtime = parked_warm  # type: ignore[method-assign]
        real_create_task = asyncio.create_task

        def recording_create_task(coro):
            task = real_create_task(coro)
            started.append(task)
            return task

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(asyncio, "create_task", recording_create_task)
            first = await bridge.warm()
            second = await bridge.warm()
            assert (first, second) == ("warming", "warming")
            assert len(started) == 1, f"a second warm started another task: {len(started)}"
            assert bridge.warm_task is started[0]
            release.set()
            await started[0]
            # A SETTLED warm must not wedge the path: the engage may have
            # failed, leaving the viewer cold, and the next keystroke has to
            # be free to try again.
            release.clear()
            assert await bridge.warm() == "warming"
            assert len(started) == 2 and bridge.warm_task is started[1]
            release.set()
            await started[1]
    await pool.close()


@pytest.mark.asyncio
async def test_the_desktop_event_payload_still_carries_inline_base64(tmp_path):
    """The out-of-repo renderer is the one consumer we cannot change.

    The runtime now externalizes an oversized image on the live wire, leaving the
    same ``{"attachment": <digest>}`` block the durable transcript writes. The
    viewer resolves it inside its wire callback — BEFORE the bridge's
    ``model_dump`` — so the published payload here is the exact inline-base64
    shape Electron already consumes. Resolution anywhere later (or not at all)
    would publish the raw reference instead, and this test is what stands in for
    a renderer no test in this repository can read.
    """
    import base64

    from local_operator.session.attached import AttachedSession
    from local_operator.session.attachments import AttachmentStore

    raw = bytes(range(256)) * 16
    stored = base64.b64encode(raw).decode("ascii")
    ref = AttachmentStore(tmp_path / "attachments").put(stored, "image/png")
    assert ref is not None

    bridge = module.DesktopSessionBridge(tmp_path, "s1", str(tmp_path))
    remote = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=module._no_takeover
    )
    remote._ready_for_events = True
    bridge.remote = remote
    remote.subscribe(bridge._event)

    remote._on_wire_event(
        {
            "type": "tool_execution_end",
            "tool_call_id": "call_image",
            "tool_name": "screenshot",
            "is_error": False,
            "result": {
                "tool_call_id": "call_image",
                "tool_name": "screenshot",
                "content": [
                    {"type": "text", "text": "PAGE"},
                    {"type": "image", "attachment": ref.digest, "mime_type": "image/png"},
                ],
                "is_error": False,
            },
        }
    )

    frames = [frame for frame, _ in bridge.replay]
    assert frames, "the bridge published nothing"
    payload = frames[-1]["payload"]
    assert payload["type"] == "tool_execution_end"
    block = payload["result"]["content"][1]
    assert block["data"] == stored
    assert "attachment" not in block


@pytest_asyncio.fixture
async def draft_api(tmp_path: Path, monkeypatch):
    """A minimal app over THIS test's config root, shared by the preview tests.

    Deliberately not the shared ``test_app_client``: that one carries the legacy
    chat surface, and the property under test is what the preview route does to
    the filesystem, so the root has to be the test's own ``tmp_path`` and the
    session pool has to close before the assertions about disk state run.
    """
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "draft-preview-test")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.include_router(desktop_sessions.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer draft-preview-test"},
    ) as client:
        yield client, tmp_path
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


@pytest.mark.asyncio
async def test_a_draft_preview_resolves_without_creating_a_session(draft_api) -> None:
    """A new-conversation pane gets its readings without costing a session.

    The pane has no session to cold-GET, and creating one at pane open would
    leave a visible empty row in the sidebar for every draft the user abandons.
    So the preview answers the question the first send will ask — which model will
    run — while writing nothing: no directory, no marker, no runtime record.
    """
    client, root = draft_api
    config = ConfigManager(config_dir=root)
    config.update_config({"hosting": "anthropic", "model_name": "claude-opus-5"})

    result = await client.post(
        "/v1/desktop/sessions/preview",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root)},
    )
    assert result.status_code == 200
    snapshot = result.json()["result"]["frontend"]["snapshot"]
    # The resolution a real cold open would produce for this session.
    assert snapshot["session_id"] == ""
    model = snapshot["selected_model"]
    assert model["provider"] == "anthropic" and model["model_id"] == "claude-opus-5"
    assert snapshot["effective_model"]["model_id"] == "claude-opus-5"
    # Nothing has been sent, so nothing may be claimed about it.
    assert snapshot["context_tokens"] is None
    assert snapshot["cumulative_parent_cost"] is None

    # No session record: neither the durable directory nor a runtime lease.
    assert not (root / "sessions").exists(), "a draft pane must not create a session"
    assert registry.scan(root) == []

    capabilities = await client.get("/v1/capabilities")
    assert (
        capabilities.json()["result"]["features"]["draft_preview"] == 1
    ), "the strip that renders a draft is gated on this key, so it must be published"


@pytest.mark.asyncio
async def test_a_draft_preview_does_not_go_through_the_create_path(draft_api, monkeypatch) -> None:
    """The route cannot be "fixed" by creating the record it is previewing.

    Pins the mechanism rather than the observation: the directory assertion above
    would also pass if the route created a session and cleaned it up, and this
    makes any use of the create path an immediate failure.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(
        {"hosting": "anthropic", "model_name": "claude-opus-5"}
    )

    async def _boom(*args, **kwargs):
        raise AssertionError("a preview must not create a session")

    monkeypatch.setattr(DesktopSessions, "create", _boom)

    result = await client.post(
        "/v1/desktop/sessions/preview",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root)},
    )
    assert result.status_code == 200
    assert result.json()["result"]["frontend"]["snapshot"]["selected_model"]["model_id"] == (
        "claude-opus-5"
    )


@pytest.mark.asyncio
async def test_a_preview_refuses_a_working_directory_that_does_not_exist(draft_api) -> None:
    """m2: one body answers the same way on both routes.

    A cwd that cannot be created cannot host a session, so a preview describing one
    would publish readings for a session the first send could never create — the
    same refusal the design states for an unresolvable profile, and now the same
    shared admission (`resolve_working_directory`) `create` applies.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(
        {"hosting": "anthropic", "model_name": "claude-opus-5"}
    )
    missing = root / "no-such-directory"

    preview = await client.post(
        "/v1/desktop/sessions/preview",
        json={"request_id": str(uuid.uuid4()), "cwd": str(missing)},
    )
    create = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "cwd": str(missing)},
    )

    assert preview.status_code == 409, "a preview must not describe a session that cannot exist"
    assert create.status_code == 409
    assert preview.json() == create.json(), "the two routes must answer the same body the same way"
    assert not missing.exists(), "neither route may create the directory it was refused"
    assert not (root / "sessions").exists()


# -- the child transcript route (design § 9.1) --------------------------------
#
# A new READ PATH ACROSS A TRUST BOUNDARY, so these tests are about what the
# route REFUSES at least as much as about the rows it returns. Two ids arrive
# from a renderer that holds an absolute `session_dir` on the wire and must
# never be able to submit one; the route, not the caller, proves membership.

PARENT_ID = "0123456789ab"
CHILD_ID = "abcdef012345"
UNNAMED_CHILD_ID = "001122334455"

#: Every way the containment proof can fail, one per clause. Parametrised so
#: the transcript route and the attachment route cannot drift apart: a gate
#: with two callers is a gate that has to hold for both.
REFUSAL_CASES = [
    "child-id-is-not-an-id",
    "child-id-is-a-path",
    "parent-unknown",
    "parent-is-a-subagent",
    "child-unknown",
    "child-is-the-users-own-conversation",
    "child-is-a-fork",
    "parent-never-named-it",
    "roster-points-outside-the-sessions-root",
]


def subagent_child(
    root: Path,
    *,
    parent_id: str = PARENT_ID,
    child_id: str = CHILD_ID,
    origin: str | None = ORIGIN_SUBAGENT,
) -> tuple[Path, Path]:
    """A parent conversation directory plus one child directory on disk.

    The child's origin marker is written by the PRODUCTION writer
    (`mark_session_origin`) because that marker is a containment fact this
    route reads — a hand-rolled copy in the fixture could drift from the one
    the launcher writes and every refusal test would pass vacuously.
    """
    sessions = root / "sessions"
    parent_dir = sessions / parent_id
    parent_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(root)}))
    child_dir = sessions / child_id
    child_dir.mkdir(parents=True, exist_ok=True)
    if origin is not None:
        mark_session_origin(child_dir, origin, label="reviewer")
    return parent_dir, child_dir


def name_child(parent_dir: Path, *session_dirs: Path, job_id: str = "job-1") -> None:
    """Record children on the parent's roster through the PRODUCTION writer.

    `_write_roster_sidecar` is what `Session._persist_subagent_roster` calls on
    every roster move, so the containment proof is exercised against the store
    shape a runtime actually writes. `session_dir` is the field a record
    carries (there is no `session_id`), and it is the whole basis of the
    ownership check.
    """
    from local_operator.session.session import (
        SUBAGENT_ROSTER_SIDECAR,
        _write_roster_sidecar,
    )

    _write_roster_sidecar(
        parent_dir / SUBAGENT_ROSTER_SIDECAR,
        {
            "version": 1,
            "generation": len(session_dirs),
            "jobs": [],
            "accounting": [],
            "records": [
                {"job_id": f"{job_id}-{index}", "label": "reviewer", "session_dir": str(directory)}
                for index, directory in enumerate(session_dirs)
            ],
        },
    )


def uncontained_pair(tmp_path: Path, case: str) -> tuple[DesktopSessions, str, str]:
    """The pool, parent id and child id for ONE refusal case.

    Each case starts from a real, contained pair and breaks exactly one clause,
    so a refusal cannot be an accident of a fixture that never looked readable.
    """
    parent_dir, child_dir = subagent_child(tmp_path)
    name_child(parent_dir, child_dir)
    pool = DesktopSessions(tmp_path)
    if case == "child-id-is-not-an-id":
        return pool, PARENT_ID, "not-an-id"
    if case == "child-id-is-a-path":
        # What a renderer holding `session_dir` would send if the route ever
        # took a path: an absolute directory, and one that escapes upward.
        return pool, PARENT_ID, str(child_dir)
    if case == "parent-unknown":
        return pool, "ffffffffffff", CHILD_ID
    if case == "parent-is-a-subagent":
        mark_session_origin(parent_dir, ORIGIN_SUBAGENT, label="child-too")
        return pool, PARENT_ID, CHILD_ID
    if case == "child-unknown":
        return pool, PARENT_ID, UNNAMED_CHILD_ID
    if case == "child-is-the-users-own-conversation":
        # On disk, user-owned (an ABSENT marker means the user, see
        # `session_origin`) — and named by the roster, so only the origin
        # clause can refuse it.
        (child_dir / "origin.json").unlink()
        return pool, PARENT_ID, CHILD_ID
    if case == "child-is-a-fork":
        mark_session_origin(child_dir, ORIGIN_FORK, parent=PARENT_ID)
        return pool, PARENT_ID, CHILD_ID
    if case == "parent-never-named-it":
        name_child(parent_dir, tmp_path / "sessions" / UNNAMED_CHILD_ID)
        return pool, PARENT_ID, CHILD_ID
    if case == "roster-points-outside-the-sessions-root":
        elsewhere = tmp_path / "elsewhere" / CHILD_ID
        elsewhere.mkdir(parents=True)
        mark_session_origin(elsewhere, ORIGIN_SUBAGENT, label="reviewer")
        name_child(parent_dir, elsewhere)
        return pool, PARENT_ID, CHILD_ID
    raise AssertionError(f"unhandled refusal case {case!r}")


@pytest.mark.asyncio
async def test_child_route_returns_the_childs_raw_rows_in_the_parents_envelope(tmp_path):
    """§ 9.1: `read_transcript_page`'s rows, verbatim, plus the derived state.

    Verbatim is the requirement that lets the renderer fold a child's page
    through the same reducer as the parent's history, so the assertion is on
    identity with the child's OWN reader — ids, timestamps, types and payloads
    — and not on a rendering of them (`peek` is the counter-example: it drops
    compaction and bookkeeping rows and flattens messages into strings).
    """
    parent_dir, child_dir = subagent_child(tmp_path)
    transcript = Transcript(child_dir)
    ids = [
        (await transcript.append_message(Message.user(f"child row {index}"))).id
        for index in range(3)
    ]
    name_child(parent_dir, child_dir)

    result = await DesktopSessions(tmp_path).child_transcript(PARENT_ID, CHILD_ID)

    expected = [
        json.loads(row.to_json()) for row in read_transcript_page(child_dir, limit=100).entries
    ]
    assert result == {
        "entries": expected,
        "has_more": False,
        "cursor_missing": False,
        "state": "ready",
    }
    assert [row["id"] for row in result["entries"]] == ids
    assert all(set(row) == {"id", "ts", "type", "payload"} for row in result["entries"])


@pytest.mark.asyncio
async def test_child_route_pages_backwards_and_reports_a_vanished_cursor(tmp_path):
    """The envelope's paging rules hold on the CHILD's file, not the parent's."""
    parent_dir, child_dir = subagent_child(tmp_path)
    transcript = Transcript(child_dir)
    ids = [
        (await transcript.append_message(Message.user(f"child row {index}"))).id
        for index in range(5)
    ]
    name_child(parent_dir, child_dir)
    pool = DesktopSessions(tmp_path)

    tail = await pool.child_transcript(PARENT_ID, CHILD_ID, limit=2)
    assert [row["id"] for row in tail["entries"]] == ids[-2:]
    assert tail["has_more"] is True
    assert tail["cursor_missing"] is False

    older = await pool.child_transcript(PARENT_ID, CHILD_ID, before_id=ids[-2], limit=2)
    assert [row["id"] for row in older["entries"]] == ids[-4:-2]
    assert older["has_more"] is True

    # A compaction replaces the JSONL atomically, so a cursor can vanish
    # between two reads; `/history`'s answer to that is the current tail plus
    # `cursor_missing`, and a child page must not invent a second one.
    transcript.path.write_text(
        TranscriptEntry("replacement", 1.0, ENTRY_MESSAGE, {"role": "user"}).to_json() + "\n"
    )
    replaced = await pool.child_transcript(PARENT_ID, CHILD_ID, before_id=ids[0], limit=100)

    assert replaced["cursor_missing"] is True
    assert [row["id"] for row in replaced["entries"]] == ["replacement"]
    assert replaced["state"] == "ready"


@pytest.mark.asyncio
async def test_child_route_distinguishes_pending_ready_and_gone(tmp_path):
    """Three states, three different answers, and only the filesystem knows.

    `pending` and `gone` are the two absences a reader must not conflate: one
    promises rows that may still arrive, the other is final. Neither is an
    error — the roster still names the child, so the read is legal and the
    ABSENCE is the answer.
    """
    parent_dir, child_dir = subagent_child(tmp_path)
    name_child(parent_dir, child_dir)
    pool = DesktopSessions(tmp_path)

    # The directory exists and nothing has been appended yet: the child may
    # still speak, so this is `pending` — and a caller that paged into the
    # absent file is told to reconcile, exactly as `/history` does.
    assert await pool.child_transcript(PARENT_ID, CHILD_ID) == {
        "entries": [],
        "has_more": False,
        "cursor_missing": False,
        "state": "pending",
    }
    paged = await pool.child_transcript(PARENT_ID, CHILD_ID, before_id="evicted")
    assert paged["cursor_missing"] is True and paged["state"] == "pending"

    # A transcript file that EXISTS with no rows is `ready`: an empty page and
    # an unwritten child are different facts, and only one of them is worth
    # re-probing.
    (child_dir / TRANSCRIPT_FILENAME).write_text("")
    ready = await pool.child_transcript(PARENT_ID, CHILD_ID)
    assert ready == {
        "entries": [],
        "has_more": False,
        "cursor_missing": False,
        "state": "ready",
    }

    # The directory itself gone is final, and still an ANSWER rather than a
    # refusal: the parent's own roster is the evidence the child existed.
    shutil.rmtree(child_dir)
    gone = await pool.child_transcript(PARENT_ID, CHILD_ID)
    assert gone == {
        "entries": [],
        "has_more": False,
        "cursor_missing": False,
        "state": "gone",
    }
    # A caller that paged into a transcript which is no longer there is told to
    # reconcile, exactly as the `pending` branch and `/history` do — the three
    # answers to one question have to agree (review round 1, R1-3).
    paged_gone = await pool.child_transcript(PARENT_ID, CHILD_ID, before_id="evicted")
    assert paged_gone["cursor_missing"] is True
    assert paged_gone["state"] == "gone"


@pytest.mark.parametrize("case", REFUSAL_CASES)
@pytest.mark.asyncio
async def test_child_transcript_refuses_an_uncontained_pair(tmp_path, case):
    pool, session_id, child_id = uncontained_pair(tmp_path, case)
    with pytest.raises(SubagentChildUnavailable):
        await pool.child_transcript(session_id, child_id)


@pytest.mark.parametrize("case", REFUSAL_CASES)
@pytest.mark.asyncio
async def test_child_attachment_refuses_the_same_uncontained_pairs(tmp_path, case):
    """The media route carries the identical gate, not a weaker one."""
    pool, session_id, child_id = uncontained_pair(tmp_path, case)
    with pytest.raises(SubagentChildUnavailable):
        await pool.child_attachment(session_id, child_id, "f" * 32)


@pytest.mark.asyncio
async def test_child_attachment_serves_bytes_for_a_contained_child(tmp_path):
    from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore

    parent_dir, child_dir = subagent_child(tmp_path)
    name_child(parent_dir, child_dir)
    raw = b"\x89PNG\r\n\x1a\nchild-media"
    ref = AttachmentStore(tmp_path / ATTACHMENTS_DIRNAME).put(
        base64.b64encode(raw).decode("ascii"), "image/png"
    )
    assert ref is not None

    data, mime_type = await DesktopSessions(tmp_path).child_attachment(
        PARENT_ID, CHILD_ID, ref.digest
    )

    assert data == raw and mime_type == "image/png"
    with pytest.raises(KeyError):
        await DesktopSessions(tmp_path).child_attachment(PARENT_ID, CHILD_ID, "f" * 32)


def test_child_attachment_digest_shape_is_enforced_by_the_route_declaration():
    """Same traversal gate as the parent's route: a declared path pattern.

    FastAPI answers a non-matching digest with 422 before the handler runs, so
    a digest can never be a filename this code builds.
    """
    from local_operator.server.app import app

    parameters = app.openapi()["paths"][
        "/v1/desktop/sessions/{session_id}/children/{child_id}/attachments/{digest}"
    ]["get"]["parameters"]
    digest = next(parameter for parameter in parameters if parameter["name"] == "digest")
    assert digest["required"] is True
    assert digest["schema"]["pattern"] == "^[a-f0-9]{32}$"


@pytest.mark.asyncio
async def test_child_transcript_route_is_bearer_gated_and_no_store(tmp_path, monkeypatch):
    """The boundary, on the wire: auth first, then the rows, then no-store.

    This is the REAL-HTTP half — the route, the `require_desktop` dependency
    and `managed_desktop_boundary` together. The adapter tests above prove the
    containment proof; this proves a refusal REACHES a caller as the contract's
    `404 child_not_found` rather than as a 500 or a bare 404 with no code.
    """
    from fastapi.testclient import TestClient

    from local_operator.server.app import app

    parent_dir, child_dir = subagent_child(tmp_path)
    transcript = Transcript(child_dir)
    ids = [
        (await transcript.append_message(Message.user(f"child row {index}"))).id
        for index in range(2)
    ]
    name_child(parent_dir, child_dir)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path))
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    path = f"/v1/desktop/sessions/{PARENT_ID}/children/{CHILD_ID}/transcript"

    with TestClient(app) as client:
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
        forbidden = client.get(
            path,
            headers={"Authorization": "Bearer token", "Origin": "https://evil.example"},
        )
        assert forbidden.status_code == 403
        headers = {"Authorization": "Bearer token"}
        response = client.get(path + "?limit=1", headers=headers)
        refused = client.get(
            f"/v1/desktop/sessions/{PARENT_ID}/children/{UNNAMED_CHILD_ID}/transcript",
            headers=headers,
        )
        not_a_child = client.get(
            f"/v1/desktop/sessions/{PARENT_ID}/children/{CHILD_ID}/transcript?limit=501",
            headers=headers,
        )
        long_cursor = client.get(path + "?before_id=" + "a" * 129, headers=headers)

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    body = response.json()["result"]
    assert body["state"] == "ready"
    assert [row["id"] for row in body["entries"]] == ids[-1:]
    assert body["has_more"] is True

    # A 404 whose code is readable: "I cannot read that pair", retryable on the
    # next pulse when the roster snapshot catches up.
    assert refused.status_code == 404
    assert refused.json()["detail"]["code"] == "child_not_found"
    # Limits and cursors are the app's own bugs, never a user state.
    assert not_a_child.status_code == 422
    assert long_cursor.status_code == 422


@pytest.mark.asyncio
async def test_child_attachment_route_is_bearer_gated_and_no_store(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from local_operator.server.app import app
    from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore

    parent_dir, child_dir = subagent_child(tmp_path)
    name_child(parent_dir, child_dir)
    raw = b"\x89PNG\r\n\x1a\nchild-media-wire"
    ref = AttachmentStore(tmp_path / ATTACHMENTS_DIRNAME).put(
        base64.b64encode(raw).decode("ascii"), "image/png"
    )
    assert ref is not None
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path))
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    path = f"/v1/desktop/sessions/{PARENT_ID}/children/{CHILD_ID}/attachments/{ref.digest}"

    with TestClient(app) as client:
        assert client.get(path).status_code == 401
        headers = {"Authorization": "Bearer token"}
        response = client.get(path, headers=headers)
        refused = client.get(
            f"/v1/desktop/sessions/{PARENT_ID}/children/{UNNAMED_CHILD_ID}"
            f"/attachments/{ref.digest}",
            headers=headers,
        )
        bad_digest = client.get(
            f"/v1/desktop/sessions/{PARENT_ID}/children/{CHILD_ID}/attachments/not-a-digest",
            headers=headers,
        )

    assert response.status_code == 200
    assert response.content == raw
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert refused.status_code == 404
    assert refused.json()["detail"]["code"] == "child_not_found"
    assert bad_digest.status_code == 422


async def legacy_roster(parent_dir: Path, *session_dirs: Path) -> None:
    """Name children the way a PRE-sidecar runtime did: one transcript entry.

    No sidecar, deliberately: that is the shape whose lack of a fork guard let a
    fork inherit the original's children (review round 1, R1-1).
    """
    from local_operator.session.session import SUBAGENT_ROSTER_CUSTOM_TYPE

    await Transcript(parent_dir).append_custom(
        SUBAGENT_ROSTER_CUSTOM_TYPE,
        {
            "version": 1,
            "generation": len(session_dirs),
            "jobs": [],
            "records": [
                {"job_id": f"legacy-{index}", "label": "reviewer", "session_dir": str(directory)}
                for index, directory in enumerate(session_dirs)
            ],
        },
    )


@pytest.mark.asyncio
async def test_a_fork_cannot_read_the_originals_children(tmp_path):
    """A fork inherits the parent's TRANSCRIPT, so the roster needs its fork guard.

    ``fork_session`` clones ``transcript.jsonl`` and leaves
    ``subagent-roster.v1.json`` behind (``fork.EXCLUDED_SIDECARS``), so a legacy
    ``subagent_roster`` entry rides into the fork VERBATIM — and this reader used
    to accept it, letting the fork read the ORIGINAL's children through its own
    route (review round 1, R1-1). The guard is the one every other reader of that
    entry applies: accept it only when it was appended AFTER this session's fork
    boundary.
    """
    from local_operator.fork import fork_session

    parent_dir, child_dir = subagent_child(tmp_path)
    await Transcript(child_dir).append_message(Message.user("child row"))
    await legacy_roster(parent_dir, child_dir)
    pool = DesktopSessions(tmp_path)

    # Positive control: the conversation that actually launched the child reads it.
    assert (await pool.child_transcript(PARENT_ID, CHILD_ID))["state"] == "ready"

    fork_id = await asyncio.to_thread(fork_session, tmp_path, PARENT_ID)
    assert (tmp_path / "sessions" / fork_id / "origin.json").is_file()
    with pytest.raises(SubagentChildUnavailable):
        await pool.child_transcript(fork_id, CHILD_ID)

    # A fork's OWN roster is still honoured — re-stamping the sidecar is what a
    # current runtime does on its first roster move, so the legacy fallback is
    # only what a fresh fork rides until then.
    name_child(tmp_path / "sessions" / fork_id, child_dir)
    assert (await pool.child_transcript(fork_id, CHILD_ID))["state"] == "ready"


@pytest.mark.asyncio
async def test_child_route_refuses_a_child_directory_that_escapes_the_store(tmp_path):
    """The id cannot be a path, but the DIRECTORY it names can be a link.

    ``sessions/`` is writable by anything running as the user, so
    ``sessions/<12-hex>`` pointing outside the store would take the read — and
    the origin check, which follows the link — with it while every id-shaped
    clause above still passed (review round 1, R1-2). The target is resolved and
    must stay inside the store, the same gate ``session/cleanup.py`` and the
    legacy chat workspace route apply.
    """
    parent_dir, existing_child_dir = subagent_child(tmp_path)
    # The link must OCCUPY the id's own path: that is the shape this refuses,
    # and the fixture's real child directory has to make way for it.
    shutil.rmtree(existing_child_dir)
    outside = tmp_path / "outside" / CHILD_ID
    outside.mkdir(parents=True)
    mark_session_origin(outside, ORIGIN_SUBAGENT, label="reviewer")
    (outside / TRANSCRIPT_FILENAME).write_text(
        TranscriptEntry("sneaky", 1.0, ENTRY_MESSAGE, {"role": "user"}).to_json() + "\n"
    )
    link = tmp_path / "sessions" / CHILD_ID
    link.symlink_to(outside, target_is_directory=True)
    name_child(parent_dir, link)

    with pytest.raises(SubagentChildUnavailable):
        await DesktopSessions(tmp_path).child_transcript(PARENT_ID, CHILD_ID)


@pytest.mark.asyncio
async def test_child_route_refuses_a_non_directory_at_the_child_path(tmp_path):
    """``gone`` means the directory is ABSENT; a file is not a child session.

    A path that exists and is not a directory answered ``gone`` before this,
    which reports a deletion that never happened and tells the reader the
    absence is final (review round 1, R1-5).
    """
    parent_dir, child_dir = subagent_child(tmp_path)
    shutil.rmtree(child_dir)
    (tmp_path / "sessions" / CHILD_ID).write_text("not a session")
    name_child(parent_dir, child_dir)

    with pytest.raises(SubagentChildUnavailable):
        await DesktopSessions(tmp_path).child_transcript(PARENT_ID, CHILD_ID)


@pytest.mark.parametrize("limit", [0, -1, 501, 5000])
@pytest.mark.asyncio
async def test_child_transcript_refuses_a_limit_outside_the_page_ceiling(tmp_path, limit):
    """The adapter guards its own argument, because a route is not its only caller.

    The wire bound is the route's ``Query`` (a 422, asserted over HTTP above);
    this keeps a direct caller — a test, a future internal one — from asking for
    an unbounded page, and both read the same ``CHILD_PAGE_LIMIT`` so the number
    exists once (review round 1, R1-6).
    """
    parent_dir, child_dir = subagent_child(tmp_path)
    await Transcript(child_dir).append_message(Message.user("child row"))
    name_child(parent_dir, child_dir)

    with pytest.raises(ValueError):
        await DesktopSessions(tmp_path).child_transcript(PARENT_ID, CHILD_ID, limit=limit)


@pytest.mark.asyncio
async def test_a_visible_lease_warms_the_runtime_and_records_presence_first(tmp_path, monkeypatch):
    """B1: a window LOOKING at a session starts its runtime, off the request path.

    The ordering half is the load-bearing one rather than a style choice.
    `_ensure_bound`'s dial re-asserts whatever presence the facade last
    recorded (TTL-bounded), and a runtime whose viewer was never asserted
    judges itself unwatched and idle-exits about 3 s after the bind
    (`DEFAULT_GRACE_S`) -- so the renderer's next 15 s heartbeat starts
    another one. A spawn per heartbeat is worse than the stall this removes,
    which is why the lease is recorded BEFORE the engage is scheduled.

    The envelope half pins that the warm is the ordinary BACKGROUND bind: a
    foreground envelope would claim the 15 s budget nobody is waiting on and,
    worse, announce itself as a user-visible caller and preempt itself
    (`warm_runtime`'s docstring).
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        order: list[tuple[Any, ...]] = []

        async def record_watch(*, visible: bool, can_notify: bool) -> None:
            order.append(("presence", visible, can_notify))

        async def record_engage(*, foreground: bool = True) -> None:
            order.append(("engage", foreground))

        monkeypatch.setattr(bridge.remote, "update_desktop_watch", record_watch)
        monkeypatch.setattr(bridge.remote, "_ensure_bound", record_engage)
        watcher = bridge.subscribe()
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        task = await _armed_warm(bridge)
        await asyncio.wait_for(task, timeout=10)
        assert order == [("presence", True, True), ("engage", False)], order
    await pool.close()


@pytest.mark.asyncio
async def test_a_hidden_or_notify_only_lease_creates_no_runtime(tmp_path, monkeypatch):
    """The other half of B1's boundary: delivery reachability is not attention.

    Term 3 still counts `visible or can_notify` to PRESERVE a runtime that
    exists, and that policy is untouched here -- but a lease nobody is looking
    at must not CREATE one, or every hidden window in the app would pin ~283 MB
    per session it happens to have open. The visible lease at the end is the
    positive control: the gate is a gate, not a dead path.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    engages: list[bool] = []

    async def record_engage(*, foreground: bool = True) -> None:
        engages.append(foreground)

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        monkeypatch.setattr(bridge.remote, "_ensure_bound", record_engage)
        hidden = bridge.subscribe()
        await bridge.watch(hidden.id, visible=False, can_notify=True)
        assert bridge.warm_task is None, "a hidden notifiable lease created a runtime"
        assert bridge.lease_warm_task is None, "a hidden lease armed the warm retry"
        never = bridge.subscribe()
        await bridge.watch(never.id, visible=False, can_notify=False)
        assert bridge.warm_task is None, "a lease with neither term created a runtime"
        assert bridge.lease_warm_task is None, "a lease with neither term armed a retry"
        watcher = bridge.subscribe()
        await bridge.watch(watcher.id, visible=True, can_notify=False)
        task = await _armed_warm(bridge)
        await asyncio.wait_for(task, timeout=10)
    assert engages == [False]
    await pool.close()


@pytest.mark.asyncio
async def test_an_expired_lease_stops_warming_and_releases_presence(tmp_path, monkeypatch):
    """B1 hands the session back exactly as it found it.

    The runtime this change creates leaves through the SAME machinery as
    before: the lease expires (`WATCH_TTL`), term 3 stops counting the desktop
    client, and the existing idle drain reaps it. What the bridge owes is the
    other end of that bargain -- after expiry nothing re-creates the process
    and nothing keeps asserting presence for it.

    The lease is expired by moving ONE subscription's deadline into the past
    rather than by replacing the module's `time` for the module under test: a
    whole-module clock also freezes the retry loop's own comparisons, so the
    next `loop.time()`-based one would silently escape it. The assertion that
    matters is on the pair the facade was last recorded with, which is the same
    thing the fake clock bought.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    writes: list[dict[str, Any]] = []
    engages: list[bool] = []

    async def record_watch(*, visible: bool, can_notify: bool) -> None:
        writes.append({"visible": visible, "can_notify": can_notify})

    async def record_engage(*, foreground: bool = True) -> None:
        engages.append(foreground)

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        monkeypatch.setattr(bridge.remote, "update_desktop_watch", record_watch)
        monkeypatch.setattr(bridge.remote, "_ensure_bound", record_engage)
        watcher = bridge.subscribe()
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        task = await _armed_warm(bridge)
        await asyncio.wait_for(task, timeout=10)
        assert writes == [{"visible": True, "can_notify": True}], writes

        # The window stops heartbeating (closed, killed, navigated away) and the
        # expiry loop runs out. The COLD half of the bargain: nothing
        # re-creates the process this change started, and there is no runtime
        # left holding a presence it no longer has -- including no STALE
        # presence: a cold facade is told the live aggregate on every beat, so
        # the `visible=True` this lease recorded is withdrawn here rather than
        # re-asserted by the next dial (H1).
        watcher.expires = time.monotonic() - 1.0
        await bridge._expire_watches()
        assert engages == [False], "the expired lease started a second engage"
        assert writes[-1] == {
            "visible": False,
            "can_notify": False,
        }, "a viewer with no runtime was left asserted at after its lease expired"

        # A hidden notifiable subscriber on the same bridge, live and fresh,
        # re-warms nothing either -- which is the state the session was in
        # before the visible lease ever arrived. It is still recorded: the pair
        # is the DESIRED presence either way.
        hidden = bridge.subscribe()
        hidden.can_notify, hidden.expires = True, time.monotonic() + 10
        await bridge.refresh_watch()
        assert engages == [False], "a notify-only lease created a runtime"
        assert writes[-1] == {"visible": False, "can_notify": True}, writes
    await pool.close()


@pytest.mark.asyncio
async def test_an_expired_lease_is_released_from_the_owner_that_was_warm(tmp_path, monkeypatch):
    """The other state the same expiry must handle: a runtime that IS up.

    With a bound viewer there is nothing to create -- so the assertion is that
    nothing is, and that the presence which was holding term 3 is WITHDRAWN on
    expiry. That withdrawal is what makes the runtime reapable through the
    existing drain rather than resident for the life of the process, and it is
    the one thing a warm must never change.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        original = bridge.remote
        writes: list[dict[str, Any]] = []

        async def record_watch(*, visible: bool, can_notify: bool) -> None:
            writes.append({"visible": visible, "can_notify": can_notify})

        bridge.remote = cast(Any, SimpleNamespace(is_cold=False, update_desktop_watch=record_watch))
        try:
            now = 200.0
            monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now))
            watcher = bridge.subscribe()
            await bridge.watch(watcher.id, visible=True, can_notify=True)
            assert bridge.warm_task is None, "a bound session was warmed again"
            assert writes[-1] == {"visible": True, "can_notify": True}

            now = 200.0 + module.WATCH_TTL + 1
            await bridge._expire_watches()
            assert writes[-1] == {
                "visible": False,
                "can_notify": False,
            }, "the owner was left believing a watcher is present after its lease expired"
        finally:
            bridge.remote = original
    await pool.close()


@pytest.mark.asyncio
async def test_a_command_arriving_during_the_warm_shares_one_engage(tmp_path):
    """A click that beats the warm must not deadlock, and must not spawn twice.

    Both callers take the SAME `_bind_lock` -- the warm as the ordinary
    background engage, the command's `bind_runtime()` as the foreground one
    that announces itself and preempts. Pinned structurally rather than on a
    stopwatch: the background bind is parked INSIDE the lock, so a command that
    did not queue on it would get past the park and the engage count would be
    2, and a command that queued on something else could never finish.
    """
    entered = asyncio.Event()
    release = asyncio.Event()
    engages: list[bool] = []

    async def parked_bind(*, foreground: bool) -> None:
        engages.append(foreground)
        if not foreground:
            entered.set()
            await release.wait()

    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        bridge.remote._bind_under_lock = parked_bind  # type: ignore[method-assign]
        watcher = bridge.subscribe()
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        await asyncio.wait_for(entered.wait(), timeout=10)

        command = asyncio.create_task(bridge.remote.bind_runtime())
        # The announcement is set BEFORE the acquire, so waiting on it is
        # waiting for the queue rather than for a duration.
        await asyncio.wait_for(bridge.remote._foreground_arrived.wait(), timeout=10)
        assert not command.done(), "the command did not wait for the warm's bind"
        assert engages == [False], "the command started a second engage"

        release.set()
        await asyncio.wait_for(command, timeout=10)
        assert engages == [False, True]
    await pool.close()


@pytest.mark.asyncio
async def test_repeated_visible_heartbeats_do_not_stack_warms(tmp_path):
    """The renderer re-asserts the lease every 15 s; each one costs at most one.

    A heartbeat during an engage must return without touching `warm_task` --
    the second task would escape `_detach()`'s cancel and duplicate the spawn
    the first one is already performing.
    """
    entered = asyncio.Event()
    release = asyncio.Event()
    engages: list[bool] = []

    async def parked_bind(*, foreground: bool) -> None:
        engages.append(foreground)
        entered.set()
        await release.wait()

    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        bridge.remote._bind_under_lock = parked_bind  # type: ignore[method-assign]
        watcher = bridge.subscribe()
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        first = await _armed_warm(bridge)
        await asyncio.wait_for(entered.wait(), timeout=10)
        for _ in range(3):
            await bridge.watch(watcher.id, visible=True, can_notify=True)
            assert bridge.warm_task is first, "a heartbeat during the engage stacked a task"
        release.set()
        await asyncio.wait_for(first, timeout=10)
    assert engages == [False]
    await pool.close()


@pytest.mark.asyncio
async def test_the_watch_route_returns_while_the_warm_is_still_binding(tmp_path, monkeypatch):
    """The non-blocking half, driven through the real route.

    `/watch` is on a 15 s heartbeat that also drives the sidebar's live
    markers, so a handler that awaited the spawn would move the cold cost onto
    the heartbeat instead of removing it. Parked engage plus a bounded wait, so
    a handler that awaited it could not answer at all.

    The bridge is held for the whole test on purpose: in production the SSE
    subscription is the second user that keeps an in-flight warm alive, and a
    warm with no holder is cancelled when its own request releases.
    """
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from local_operator.server.routes import desktop_sessions as routes

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "warm-token")
    app = FastAPI()
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    app.include_router(routes.router)
    sid = await pool.create(str(tmp_path))

    entered = asyncio.Event()
    release = asyncio.Event()

    async def parked_bind(*, foreground: bool = True) -> None:
        entered.set()
        await release.wait()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer warm-token"},
    ) as client:
        async with pool.session(sid) as bridge:
            assert bridge.remote is not None
            monkeypatch.setattr(bridge.remote, "_ensure_bound", parked_bind)
            subscription = bridge.subscribe()
            response = await asyncio.wait_for(
                client.post(
                    f"/v1/desktop/sessions/{sid}/watch",
                    json={
                        "subscription_id": subscription.id,
                        "visible": True,
                        "can_notify": True,
                    },
                ),
                timeout=10,
            )
            assert response.status_code == 200, response.text
            assert response.json()["result"] == {"lease_seconds": 45}
            await asyncio.wait_for(entered.wait(), timeout=10)
            task = bridge.warm_task
            assert task is not None and not task.done()
            release.set()
            await task
    await pool.close()


@pytest.mark.asyncio
async def test_a_cold_facade_is_told_the_live_aggregate_not_the_last_heartbeat(
    tmp_path, monkeypatch
):
    """H1: the pair a cold facade holds for its NEXT dial is the CURRENT one.

    The record taken before a warm is what makes the runtime it starts count
    its viewer from the first tick, and `_dial` re-asserts the same record when
    that runtime comes up. So a cold facade must be told the live aggregate on
    every beat: hide the window (or let the lease lapse) during the ~1 s a spawn
    takes, and a facade that skipped the write leaves `visible=True` standing,
    which the dial then asserts for a viewer who has gone -- one idle runtime
    (~82 MB) held for up to the next beat (15 s) plus the runtime-side lease and
    the 3 s drain.

    The engage here is recorded rather than performed, so the facade stays cold
    for the second half; what is under test is the write, not the bind.
    """
    writes: list[dict[str, Any]] = []
    engages: list[bool] = []

    async def record_watch(*, visible: bool, can_notify: bool) -> None:
        writes.append({"visible": visible, "can_notify": can_notify})

    async def record_engage(*, foreground: bool = True) -> None:
        engages.append(foreground)

    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        monkeypatch.setattr(bridge.remote, "update_desktop_watch", record_watch)
        monkeypatch.setattr(bridge.remote, "_ensure_bound", record_engage)
        watcher = bridge.subscribe()
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        assert writes == [{"visible": True, "can_notify": True}], writes

        # The window goes to the background while the engage it just armed is
        # still in flight. The runtime it starts must not be told a viewer is
        # looking when one is not.
        await bridge.watch(watcher.id, visible=False, can_notify=True)
        assert writes[-1] == {"visible": False, "can_notify": True}, writes
    await pool.close()


@pytest.mark.asyncio
async def test_a_failing_warm_is_paced_rather_than_retried_every_heartbeat(tmp_path, monkeypatch):
    """MAJOR-1: an unattended retry must not spawn a child on every beat.

    The renderer heartbeats `/watch` every 15 s for as long as a window is
    focused, so a bind that cannot start (credential gone, an MCP hang, an
    unwritable config dir) used to be re-attempted -- a real child spawn each
    time -- once per beat, indefinitely, with no user behind it and nothing on
    any surface to say so. The retry loop keeps the intent but charges an
    attempt that actually ran a doubling backoff.

    The module clock is frozen so the assertion is about the rule (a heartbeat
    inside the backoff re-engages nothing) rather than about how fast this
    machine can sleep; the clock is then advanced by hand, which is the only
    thing that stands in for 30 s of real time, in heartbeat-sized steps so the
    45 s lease stays live across each one (a retry that stopped because the
    lease LAPSED would prove nothing about the pace).

    The ceiling and the lease bound are asserted here too, because "paced" is
    only half of what this finding asks for: an unattended retry is only safe if
    the pace stops growing and if a lease that stops being renewed ends it.
    """
    attempts: list[bool] = []

    async def exploding_bind(*, foreground: bool = True) -> None:
        attempts.append(foreground)
        raise ConnectionError("no runtime")

    now = 100.0

    def clock() -> float:
        return now

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=clock))
    # `raising=False` so a tree without the knob reports the BEHAVIOUR this
    # test is about rather than the missing attribute: the polling pace is
    # what makes these tests fast, not what they assert.
    monkeypatch.setattr(module, "_LEASE_WARM_POLL_S", 0.01, raising=False)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        monkeypatch.setattr(bridge.remote, "_ensure_bound", exploding_bind)
        watcher = bridge.subscribe()
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        first = await _armed_warm(bridge)
        await asyncio.wait_for(first, timeout=10)
        # WAIT ON THE PACE, NOT ON THE ATTEMPT COUNT. The fake bind appends the
        # attempt before the loop awaits the task and charges the pace, so
        # `attempts == [False]` is true for a scheduling window before
        # `warm_backoff_s` is -- which is the race QA round 2, Q1 caught here.
        await _until(
            lambda: bridge.warm_backoff_s == module._LEASE_WARM_BACKOFF_S,
            why="the first attempt that ran was not paced",
        )
        assert attempts == [False], attempts

        # Three heartbeats, all inside the backoff: the deadline cannot arrive
        # on a frozen clock, so every one of them must be answered with nothing.
        for _ in range(3):
            await bridge.watch(watcher.id, visible=True, can_notify=True)
            await asyncio.sleep(0.05)
        assert attempts == [False], "a heartbeat inside the backoff re-engaged"
        assert bridge.warm_backoff_s == module._LEASE_WARM_BACKOFF_S, bridge.warm_backoff_s

        # The intent is still there when the backoff expires, with no user
        # action and no heartbeat in between -- and the second attempt grows the
        # pace rather than resetting it. The predicate is the FIELD the pace is
        # asserted on, so the poll cannot land between an attempt and its charge.
        now = 100.0 + module._LEASE_WARM_BACKOFF_S + 1
        await _until(
            lambda: bridge.warm_backoff_s == 2 * module._LEASE_WARM_BACKOFF_S,
            why="the backoff never re-engaged the intent",
        )
        assert attempts == [False, False], attempts

        async def advance(seconds: float) -> None:
            """Move the clock in heartbeat-sized steps, renewing as a window does.

            The loop re-asks the lease on every pass, so a single clock jump
            past `WATCH_TTL` would end the retries for the wrong reason -- the
            lease lapsing rather than the pace holding. Each step is therefore a
            genuine heartbeat, which is also the strongest form of the assertion
            above: MANY beats inside one backoff, and none of them may re-engage
            the intent or reset the pace it landed in.
            """
            nonlocal now
            while seconds > 0:
                step = min(15.0, seconds)
                now += step
                seconds -= step
                await bridge.watch(watcher.id, visible=True, can_notify=True)

        # TWO more attempts reach and hold the ceiling: 60 -> 120 -> 120, the
        # second of them being the clamp taking `min(2 * 120, 120)` instead of
        # doubling on to 240. Read off the loop's own `min()` that would be an
        # assertion by inspection; the pace is observed here instead.
        #
        # `warm_not_before` is the field that moves for BOTH of them (the clamped
        # attempt leaves `warm_backoff_s` at the value it already had), and the
        # loop writes it AFTER the pace in one synchronous block, so waiting on
        # it is waiting for the attempt AND its charge -- even when the attempt
        # lands in the middle of `advance` rather than after it, which is the
        # second race QA round 2, Q1 reproduced.
        for _ in range(2):
            prior = bridge.warm_not_before
            await advance(bridge.warm_backoff_s + 1)
            await _until(
                lambda: bridge.warm_not_before > prior,
                why="the backoff never re-engaged the intent",
            )
        assert attempts == [False] * 4, attempts
        assert (
            bridge.warm_backoff_s == module._LEASE_WARM_BACKOFF_CAP_S
        ), f"the backoff is not clamped at the ceiling: {bridge.warm_backoff_s}"

        # AND THE LEASE ENDS IT. The pace is waited out in poll-sized slices
        # rather than one long sleep for exactly this case: a lease withdrawn
        # during the 120 s wait must end the retries within a slice, not after
        # the wait.
        watcher.expires = module.time.monotonic() - 1.0
        await _until(
            lambda: bridge.lease_warm_task is not None and bridge.lease_warm_task.done(),
            why="the retry loop outlived the lease whose intent it was holding",
        )
        assert len(attempts) == 4, "a withdrawn lease kept re-engaging"
    await pool.close()


@pytest.mark.asyncio
async def test_a_lease_driven_warm_survives_the_facades_recovery_window(tmp_path, monkeypatch):
    """QA Q1: a lease's intent must outlive one refuted attempt.

    A viewer that has just lost its runtime sits in owner recovery for up to
    `COLD_FALLBACK_S` (~9.4 s measured end to end on this path), and
    `_ensure_bound` returns at its own `_recovering` guard -- no error, no
    engage, nothing started and nothing to report. The desktop panel's own shape
    is `visible` -> reaped -> `visible` again a fraction of a second later, i.e.
    INSIDE that window; with one attempt per `/watch` the attempt was simply
    lost, and the user's next command paid the cold bind this feature exists to
    remove while the renderer's next beat was 15 s away.

    `_bind_under_lock` is the patch point rather than `_ensure_bound`, because
    the guard under test IS the real `_ensure_bound`: a fake one would record an
    engage the real recovery never makes, and the test would pass for the wrong
    reason. Nothing binds, so the second half is about the loop re-asking on its
    own, with no further heartbeat.
    """
    engages: list[bool] = []

    async def record_bind(*, foreground: bool) -> None:
        engages.append(foreground)

    # `raising=False` so a tree without the knob reports the BEHAVIOUR this
    # test is about rather than the missing attribute: the polling pace is
    # what makes these tests fast, not what they assert.
    monkeypatch.setattr(module, "_LEASE_WARM_POLL_S", 0.01, raising=False)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        bridge.remote._bind_under_lock = record_bind  # type: ignore[method-assign]
        bridge.remote._recovering = True
        watcher = bridge.subscribe()
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        await asyncio.sleep(0.2)
        assert engages == [], "an attempt was made while recovery owned the dial"

        # Recovery releases the facade -- what `_give_up_recovery` ends in, minus
        # the flags this test does not exercise.
        bridge.remote._recovering = False
        for _ in range(200):
            if engages:
                break
            await asyncio.sleep(0.01)
        assert engages == [False], "the lease's intent did not survive the recovery window"
    await pool.close()


@pytest.mark.asyncio
async def test_a_lease_driven_warm_survives_an_in_flight_bind(tmp_path, monkeypatch):
    """H2: `engage_in_flight` is a reason to wait, not a reason to forget.

    `warm()` answers "warming" and starts NOTHING when the facade's bind lock is
    held -- right for the route caller, whose own send joins that bind, but the
    lease-driven warm has no send behind it: before the retry loop, a beat
    landing in a foreign holder's window (another subscriber's `attach_existing`
    takes the same lock across a thread hop) warmed nothing, and a click in the
    following 15 s paid the full cold spawn.

    The lock is taken directly here, which is the narrowest way to produce the
    state `warm()` samples, and released with no heartbeat in between.
    """
    engages: list[bool] = []

    async def record_bind(*, foreground: bool) -> None:
        engages.append(foreground)

    # `raising=False` so a tree without the knob reports the BEHAVIOUR this
    # test is about rather than the missing attribute: the polling pace is
    # what makes these tests fast, not what they assert.
    monkeypatch.setattr(module, "_LEASE_WARM_POLL_S", 0.01, raising=False)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        bridge.remote._bind_under_lock = record_bind  # type: ignore[method-assign]
        await bridge.remote._bind_lock.acquire()
        try:
            watcher = bridge.subscribe()
            await bridge.watch(watcher.id, visible=True, can_notify=True)
            await asyncio.sleep(0.2)
            assert engages == [], "a warm engaged beside a bind already in flight"
            assert bridge.warm_task is None
        finally:
            bridge.remote._bind_lock.release()
        for _ in range(200):
            if engages:
                break
            await asyncio.sleep(0.01)
        assert engages == [False], "the lease's intent did not survive the wait"
    await pool.close()


@pytest.mark.asyncio
async def test_a_deliberately_stopped_session_is_not_warmed_by_a_visible_lease(
    tmp_path, monkeypatch
):
    """Reviewer round-2 MAJOR-1: a focused viewer must not resurrect a stop.

    `_recover_runtime` already refuses on `session_was_stopped()` -- "it is what
    keeps the takeover from resurrecting a session a kill switch just ended"
    (`session/attached.py`) -- and the desktop stop's own copy promises that
    `/resume` re-opens a stopped conversation. Without the same guard the
    lease-driven warm did it for free: the user stops a session in the focused
    window, the runtime exits, and the next beat (<= 15 s, ~82 MB idle) starts a
    fresh runtime for the session they just ended, clearing the `stopped_at`
    marker the stop wrote.

    Both halves of the contract are pinned here: the beat that must engage
    nothing while the session is stopped, and the explicit resume that must
    behave exactly as before.
    """
    engages: list[bool] = []

    async def record_engage(*, foreground: bool = True) -> None:
        engages.append(foreground)

    monkeypatch.setattr(module, "_LEASE_WARM_POLL_S", 0.01, raising=False)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        monkeypatch.setattr(bridge.remote, "_ensure_bound", record_engage)
        # Exactly the shape `request_stop` leaves (and the one the reviewer
        # reproduced): the flag is set BEFORE the op is sent, so
        # `await remote.session_was_stopped()` answers True from the facade
        # alone, without consulting the wake store.
        bridge.remote._deliberate_stop = True
        watcher = bridge.subscribe()
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        await asyncio.sleep(0.2)
        assert engages == [], "a deliberately stopped session was warmed"
        assert bridge.warm_task is None
        assert bridge.lease_warm_task is None

        # The resume: `_finish_sync` clears the flag on the sync that re-attaches
        # the viewer, and the next visible beat must warm as it always has.
        bridge.remote._deliberate_stop = False
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        task = await _armed_warm(bridge)
        await asyncio.wait_for(task, timeout=10)
        assert engages == [False], engages
    await pool.close()


@pytest.mark.asyncio
async def test_a_stop_ends_the_lease_warm_that_was_already_retrying(tmp_path, monkeypatch):
    """The other seam of MAJOR-1: a loop armed before the stop must honour it.

    A loop already pacing an attempt is a warm that is still trying, and the
    per-pass re-ask is the only place a stop landing mid-pace can end it --
    otherwise the loop's next attempt resurrects the session the user just
    ended, seconds after they ended it. The pace goes with the intent too: a
    later `/resume` is a user action and must not wait out a deadline no live
    intent is holding.
    """
    attempts: list[bool] = []

    async def exploding_bind(*, foreground: bool = True) -> None:
        attempts.append(foreground)
        raise ConnectionError("no runtime")

    monkeypatch.setattr(module, "_LEASE_WARM_POLL_S", 0.01, raising=False)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        monkeypatch.setattr(bridge.remote, "_ensure_bound", exploding_bind)
        watcher = bridge.subscribe()
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        first = await _armed_warm(bridge)
        await asyncio.wait_for(first, timeout=10)
        await _until(
            lambda: bridge.warm_backoff_s == module._LEASE_WARM_BACKOFF_S,
            why="the attempt that ran was not paced",
        )
        assert attempts == [False], attempts

        bridge.remote._deliberate_stop = True
        await _until(
            lambda: bridge.lease_warm_task is not None and bridge.lease_warm_task.done(),
            why="the retry loop outlived the deliberate stop",
        )
        assert attempts == [False], "a stopped session was re-attempted"
        assert bridge.warm_backoff_s == 0.0, "the stopped intent kept its pace"

        # And the next beat does not ARM a fresh loop for the stopped session --
        # the arm guard, not the loop's re-ask, is what answers here. It is
        # checked by identity: `lease_warm_task` keeps the last loop once it has
        # finished (a done task is not None), so a new object is the shape a
        # missing guard takes.
        stopped_loop = bridge.lease_warm_task
        assert stopped_loop is not None and stopped_loop.done()
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        await asyncio.sleep(0.2)
        assert bridge.lease_warm_task is stopped_loop, "a stopped session was re-armed"
        assert attempts == [False], attempts
    await pool.close()


@pytest.mark.asyncio
async def test_a_runtime_that_boots_then_dies_is_paced_not_respawned_each_beat(
    tmp_path, monkeypatch
):
    """Reviewer round-2 MINOR-1: the pace follows the ATTEMPT, not its outcome.

    A runtime that comes up and then dies -- a late boot failure, an OOM, a
    build-stamp restart gone wrong -- passes the loop's post-await `is_cold`
    check. Charging the pace only on the cold side of that check left
    `warm_backoff_s` at 0.0 and the renderer's next beat started another child
    (reproduced: 4 beats -> 4 attempts), which is round-1 MAJOR-1's unattended
    spawn loop one failure shape over.

    `_ensure_bound` stands in for the spawn: it makes the viewer BOUND through
    the real `is_cold` predicate (the `_client` that property reads), and the
    test then kills it exactly as a crash does -- `_client` back to None --
    before the next beat. The clock is frozen and advanced by hand, as in the
    pacing pin, so the assertions are about the rule rather than machine speed.
    """
    attempts: list[bool] = []
    now = 100.0

    def clock() -> float:
        return now

    class BoundClient:
        """Only what `is_cold` and the presence RPC read."""

        connected = True

        def close(self) -> None:
            pass

        async def desktop_watch(self, *, visible: bool, can_notify: bool) -> None:
            pass

    async def boot_then_die(*, foreground: bool = True) -> None:
        attempts.append(foreground)
        remote._client = BoundClient()  # type: ignore[assignment]
        remote._ready_for_events = True

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(module, "_LEASE_WARM_POLL_S", 0.01, raising=False)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        # Bound once, as a local: the closure below cannot see the narrowing the
        # assertion above gives `bridge.remote`.
        remote = bridge.remote
        monkeypatch.setattr(remote, "_ensure_bound", boot_then_die)
        watcher = bridge.subscribe()
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        await _until(
            lambda: bridge.warm_backoff_s == module._LEASE_WARM_BACKOFF_S,
            why="an attempt that left the viewer bound was not charged",
        )
        assert attempts == [False], attempts

        # The crash, then three heartbeats inside the pace: the old behaviour
        # was a real spawn per beat, and the deadline cannot arrive on a frozen
        # clock, so nothing here may attempt.
        remote._client = None
        for _ in range(3):
            await bridge.watch(watcher.id, visible=True, can_notify=True)
            await asyncio.sleep(0.05)
        assert attempts == [False], "a beat inside the pace re-spawned the runtime"
        assert bridge.warm_backoff_s == module._LEASE_WARM_BACKOFF_S, bridge.warm_backoff_s

        # The intent survives the pace and the second attempt is charged too
        # (30 -> 60), so a session that keeps crashing keeps backing off rather
        # than settling into a per-beat cadence.
        now += module._LEASE_WARM_BACKOFF_S + 1
        await _until(
            lambda: bridge.warm_backoff_s == 2 * module._LEASE_WARM_BACKOFF_S,
            why="the paced intent never re-engaged",
        )
        assert attempts == [False, False], attempts
    await pool.close()


@pytest.mark.asyncio
async def test_a_fresh_intent_does_not_inherit_a_withdrawn_ones_pace(tmp_path, monkeypatch):
    """QA round-2 Q2: an abandoned intent's pace must not charge the next one.

    `_clear_warm_backoff` promises a fresh cold period starts from the base, but
    the loop exits on a withdrawn lease without clearing, so a viewer returning
    12 s into a 30 s pace paid the remaining 15.9 s before its first child
    appeared (measured; up to the 120 s ceiling on a longer failure). The pace
    belongs to the intent that earned it, and a withdrawn lease ends that
    intent: on the frozen clock a wrongly inherited pace shows up as an attempt
    that never happens at all.
    """
    attempts: list[bool] = []
    now = 500.0

    def clock() -> float:
        return now

    async def exploding_bind(*, foreground: bool = True) -> None:
        attempts.append(foreground)
        raise ConnectionError("no runtime")

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(module, "_LEASE_WARM_POLL_S", 0.01, raising=False)
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        monkeypatch.setattr(bridge.remote, "_ensure_bound", exploding_bind)
        watcher = bridge.subscribe()
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        first = await _armed_warm(bridge)
        await asyncio.wait_for(first, timeout=10)
        await _until(
            lambda: bridge.warm_backoff_s == module._LEASE_WARM_BACKOFF_S,
            why="the attempt that ran was not paced",
        )

        # The window leaves, well inside the 30 s pace. The beat itself drops
        # the pace (it is the first with no live visible lease), so the clear
        # does not wait for the loop to notice.
        watcher.expires = module.time.monotonic() - 1.0
        await bridge.refresh_watch()
        assert bridge.warm_backoff_s == 0.0, "a withdrawn intent kept its pace"

        # It comes back 12 s later, and the fresh intent engages at once.
        now += 12.0
        watcher.expires = module.time.monotonic() + module.WATCH_TTL
        await bridge.watch(watcher.id, visible=True, can_notify=True)
        await _until(
            lambda: bridge.warm_backoff_s == module._LEASE_WARM_BACKOFF_S,
            why="the fresh intent waited out the abandoned one's deadline",
        )
        assert len(attempts) == 2, attempts
    await pool.close()
