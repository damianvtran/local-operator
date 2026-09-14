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
from typing import Any, Callable, Literal, cast

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


async def _armed_warm(bridge: Any) -> asyncio.Task[None]:
    """The engage task the lease-warm loop starts, after its first step.

    The lease-driven warm is ARMED by `/watch` rather than started inside it,
    because the retry and its backoff have to live in one place and that place
    is the loop -- so the task appears after the heartbeat that armed it, once
    the loop has taken its first pass. Everything the ordering promises is
    unchanged: the presence record still lands before the engage, and `/watch`
    still returns without awaiting either.

    WAITED ON WITH A REAL-CLOCK BOUND, NOT A TURN COUNT. This used to spin 64
    `sleep(0)` turns, which is a budget in event-loop turns rather than in time
    -- and the loop's first pass now hops to a worker thread for the
    deliberate-stop marker, so 64 turns can expire while that thread runs
    (measured: this helper went red under CI's 4-worker shard while the same
    file passed 6/6 locally at default parallelism).
    """
    await _until(
        lambda: bridge.warm_task is not None,
        why="a live visible lease did not start a warm",
    )
    assert bridge.warm_task is not None
    return bridge.warm_task


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
@pytest.mark.parametrize("sigil", ["?", "#"])
async def test_the_read_only_probe_never_clips_the_truncated_path(tmp_path, sigil):
    """A config root is arbitrary user data, and ``?``/``#`` must stay DATA (R11).

    The probe hands its path to SQLite through a ``file:`` URI, where both of those
    characters are delimiters. Interpolated raw they truncate the filename — so the
    open landed on the path up to the sigil, CREATED a 0-byte file there, and then
    answered ``False`` for a key the store does hold. Both halves are asserted
    below because either alone passes on the defect: the absent key answering
    ``False`` is also the broken probe's answer, and only the recorded key can tell
    a genuine read from one that opened the wrong file.
    """
    root = tmp_path / f"root{sigil}odd"
    receipts = DesktopReceipts(root)

    async def op():
        return {"result": "real receipt"}

    assert receipts.recorded("s:id") is False, "an absent store records nothing"
    assert not receipts.path.exists(), "the probe created the store it only reads"

    await receipts.run("s:id", {"op": "create"}, op)
    assert receipts.path.exists()
    assert receipts.recorded("s:id") is True, "the probe must read the INTENDED file"
    assert receipts.recorded("other:id") is False
    # Everything before the sigil is a different path, and the unescaped shape
    # opened (and created) exactly that.
    assert not (tmp_path / "root").exists(), "the probe wrote at a truncated path"


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


# -- the draft's chosen model and reasoning effort (draft_selection) ----------------
#
# A new-conversation pane renders the identity the FIRST turn will use. These
# tests hold the two halves of that promise together: the readings the pane
# shows (preview), and the session actually being BORN on the same choice
# (create's marker, the cold viewer, and the engage).

#: A choice that differs from the configured default in BOTH halves, so a test
#: that accidentally reads config cannot pass. ``deepseek-flash`` is a shipped
#: registry row, so resolution needs no network and the refusal path below is
#: exercised by the ids that are NOT in the shipped catalogue.
CHOSEN_MODEL = {"provider": "deepseek", "model_id": "deepseek-flash", "reasoning_effort": "max"}
CONFIGURED = {"hosting": "anthropic", "model_name": "claude-sonnet-5"}


def draft_model(provider: str = "deepseek", model_id: str = "deepseek-flash", effort=None):
    return {"provider": provider, "model_id": model_id, "reasoning_effort": effort}


@pytest.mark.asyncio
async def test_a_draft_preview_resolves_the_requested_selection(draft_api) -> None:
    """The pane's readings are the ones the first turn will get, not config's.

    Both halves matter and they come from different sources: the IDENTITY from
    the requested pair, the SPEC (context window, effort ladder) from the same
    metadata resolver a real cold open uses. A preview that showed the config
    default would hand the user a chip naming a model that never answers.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)

    result = await client.post(
        "/v1/desktop/sessions/preview",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "model": CHOSEN_MODEL},
    )
    assert result.status_code == 200
    snapshot = result.json()["result"]["frontend"]["snapshot"]
    model = snapshot["selected_model"]
    assert model["provider"] == "deepseek" and model["model_id"] == "deepseek-flash"
    assert (
        model["reasoning_effort"] == "max"
    ), "the pane must show the chosen LEVEL, not the model's seed"
    assert snapshot["effective_model"]["reasoning_effort"] == "max"
    # The spec the first turn gets: the real ladder and window, not the default's.
    assert tuple(model["reasoning_efforts"]) == ("none", "low", "high", "max")
    assert model["context_window"] == 1000000
    assert not (root / "sessions").exists(), "a preview still costs nothing durable"


@pytest.mark.asyncio
async def test_a_preview_that_omits_the_selection_answers_an_explicit_null(draft_api) -> None:
    """``model`` is OPTIONAL: omitting the field answers exactly the explicit ``null``.

    The invariant pinned is *omitted ≡ explicit ``null``* — the spelling a client
    that always sends the key would produce. It is NOT byte-identity with the
    pre-feature payload, and does not claim to be: what DID change, deliberately
    and in one direction only, is the SPEC the pane publishes, which is now the
    configured pair resolved through its own metadata so the effort LADDER and
    LEVEL are answered instead of being left empty. That is the operator's own
    report (an empty ladder hides the strip's effort chip and makes the picker
    unreachable on every new conversation), and the frame the UI swaps in at send
    answers the same fields — see
    ``test_an_unpicked_draft_answers_the_ladder_and_level_it_will_run_at``.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)

    omitted = await client.post(
        "/v1/desktop/sessions/preview", json={"request_id": str(uuid.uuid4()), "cwd": str(root)}
    )
    explicit_null = await client.post(
        "/v1/desktop/sessions/preview",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "model": None},
    )

    assert omitted.status_code == explicit_null.status_code == 200
    assert omitted.json() == explicit_null.json()
    snapshot = omitted.json()["result"]["frontend"]["snapshot"]
    model = snapshot["selected_model"]
    assert model["model_id"] == "claude-sonnet-5", "the configured identity, untouched"
    assert model["reasoning_efforts"], "the ladder is answered, not left empty"
    assert model["reasoning_effort"] == "high", "the seeded rung, since no level is configured"
    assert not (root / "sessions").exists(), "omitting the field is still free"


#: Every way a draft selection can be refused, one per clause of the contract.
#: ``effort_unsupported`` appears twice on purpose: a laddered model that does
#: not offer the level, and a model with NO ladder, which must not accept one at
#: all (offering ``none`` to a non-reasoning model is the same claim as offering
#: ``high``).
REFUSED_DRAFT_MODELS = [
    (draft_model("nope-provider", "whatever"), "provider_unknown"),
    (draft_model("anthropic", "claude-opus-9"), "model_unknown"),
    (draft_model("deepseek", "deepseek-flash", "turbo"), "effort_unsupported"),
    (draft_model("anthropic", "claude-opus-5", "xhighz"), "effort_unsupported"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model,code", REFUSED_DRAFT_MODELS)
async def test_a_draft_selection_the_engine_could_not_serve_is_refused(
    draft_api, model, code
) -> None:
    """One 422 per way a pick can fail, and NOTHING durable on either route.

    Refusing rather than degrading is the point: the client rendered a choice
    the user made, so answering with a different model is the disagreement this
    feature exists to remove. The refusal must also land BEFORE the create
    route's receipt claim — a claim is a durable write, and a claim left by a
    refused request answers its own retry with "outcome indeterminate".
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)
    request_id = str(uuid.uuid4())

    preview = await client.post(
        "/v1/desktop/sessions/preview",
        json={"request_id": request_id, "cwd": str(root), "model": model},
    )
    created = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": request_id, "cwd": str(root), "model": model},
    )

    assert preview.status_code == 422, preview.text
    assert created.status_code == 422, created.text
    assert preview.json()["detail"]["code"] == code
    assert created.json()["detail"]["code"] == code
    assert not (root / "sessions").exists(), "a refusal must not create a draft"

    # The same request id is still UNCLAIMED, so the corrected retry is the
    # create it should be rather than a conflict over a claim nobody finished.
    retry = await client.post(
        "/v1/desktop/sessions", json={"request_id": request_id, "cwd": str(root)}
    )
    assert retry.status_code == 200, retry.text
    assert retry.json()["result"]["session_id"]


@pytest.mark.asyncio
async def test_an_effort_on_a_model_with_no_ladder_is_refused(draft_api) -> None:
    """A model that offers no levels must not accept one.

    Separate from the parametrised ladder case because the LADDER is empty here
    rather than merely missing the level: accepting ``none`` would put a "do not
    reason" claim on a band for a turn that cannot express it.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)

    result = await client.post(
        "/v1/desktop/sessions/preview",
        json={
            "request_id": str(uuid.uuid4()),
            "cwd": str(root),
            "model": draft_model("ollama", "llama3", "low"),
        },
    )
    assert result.status_code == 422, result.text
    assert result.json()["detail"]["code"] == "effort_unsupported"


def _marker(root: Path, session_id: str) -> dict[str, Any]:
    return json.loads((root / "sessions" / session_id / "desktop.json").read_text())


@pytest.mark.asyncio
async def test_a_created_drafts_choice_is_stored_additively(draft_api) -> None:
    """The marker gains the choice WITHOUT changing the document it already was.

    Additivity is the whole backwards-compatibility contract: every reader of
    ``desktop.json`` in this tree reads keys it knows, and a marker written by
    any earlier build must keep loading. So the same create is run twice — with
    and without a selection — and the no-selection document is asserted to be
    exactly the one that build wrote, key for key.
    """
    _client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)
    pool = DesktopSessions(root)

    chosen = await pool.create(str(root), model=dict(CHOSEN_MODEL))
    plain = await pool.create(str(root))

    stored = _marker(root, chosen)
    assert stored == {"version": 1, "cwd": str(root.resolve()), "model": CHOSEN_MODEL}
    assert _marker(root, plain) == {"version": 1, "cwd": str(root.resolve())}


@pytest.mark.asyncio
async def test_a_created_drafts_choice_is_what_a_snapshot_reports(draft_api) -> None:
    """The stored choice reaches the COLD VIEWER a real open builds.

    This is the path the pane takes the moment it stops being a draft: the
    route writes the marker, and the first snapshot of that session is answered
    by an ``AttachedSession.cold`` synthesised from it. A choice that survived
    create but not the open would be a chip that names one model while the first
    turn runs another.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)

    created = await client.post(
        "/v1/desktop/sessions",
        json={
            "request_id": str(uuid.uuid4()),
            "cwd": str(root),
            "model": dict(CHOSEN_MODEL),
        },
    )
    assert created.status_code == 200, created.text
    session_id = created.json()["result"]["session_id"]

    snapshot = await client.get(f"/v1/desktop/sessions/{session_id}")
    assert snapshot.status_code == 200, snapshot.text
    state = snapshot.json()["result"]["payload"]["frontend"]["snapshot"]
    model = state["selected_model"]
    assert model["provider"] == "deepseek" and model["model_id"] == "deepseek-flash"
    assert model["reasoning_effort"] == "max"
    assert state["effective_model"]["model_id"] == "deepseek-flash"
    # The spec a cold frame carries comes from the same synth the preview used,
    # so the draft chip and the first real frame cannot disagree.
    assert tuple(model["reasoning_efforts"]) == ("none", "low", "high", "max")


@pytest.mark.asyncio
async def test_the_conversations_own_selection_outranks_the_one_it_was_born_on(draft_api) -> None:
    """A conversation the user later switched must NOT be dragged back.

    The birth choice is only ever a seed. Once the session's own journal carries
    a selection — a v2 row, written by its leased owner — that row IS the
    answer, whatever the marker still says. This is the failure mode the
    override exists to avoid: a resume that re-applied the birth model would
    silently undo every switch the user made.
    """
    from local_operator.session.model_selection import SELECTED_MODEL_CUSTOM_TYPE

    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)
    created = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "model": dict(CHOSEN_MODEL)},
    )
    session_id = created.json()["result"]["session_id"]
    await Transcript(root / "sessions" / session_id).append_custom(
        SELECTED_MODEL_CUSTOM_TYPE,
        {"version": 2, "selector": "anthropic/claude-opus-5", "effort": "low"},
    )

    snapshot = await client.get(f"/v1/desktop/sessions/{session_id}")
    model = snapshot.json()["result"]["payload"]["frontend"]["snapshot"]["selected_model"]
    assert (model["provider"], model["model_id"]) == ("anthropic", "claude-opus-5")
    assert model["reasoning_effort"] == "low"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored",
    [
        draft_model("anthropic", "claude-opus-9", "high"),
        draft_model("vanished-provider", "claude-opus-5", "high"),
    ],
)
async def test_a_marker_naming_a_vanished_model_degrades_to_the_default(draft_api, stored) -> None:
    """A stored PAIR that no longer resolves must cost a default, never an open.

    A marker outlives the catalogue that produced it: a provider can be removed
    from the registry, an id can be retired. Neither may fail a resume — the
    conversation still opens, on the configured default, which is exactly
    today's behaviour for a session with no choice.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)
    session_id = await DesktopSessions(root).create(str(root))
    marker = root / "sessions" / session_id / "desktop.json"
    marker.write_text(json.dumps({"version": 1, "cwd": str(root), "model": stored}))

    snapshot = await client.get(f"/v1/desktop/sessions/{session_id}")
    assert snapshot.status_code == 200, snapshot.text
    model = snapshot.json()["result"]["payload"]["frontend"]["snapshot"]["selected_model"]
    assert (model["provider"], model["model_id"]) == ("anthropic", "claude-sonnet-5")


@pytest.mark.asyncio
async def test_a_marker_naming_a_retired_level_clamps_within_the_model(draft_api) -> None:
    """A level the ladder no longer offers lands on the nearest rung it does.

    Clamped rather than refused or dropped, and NOT escalated to the config
    default: the model is still served, the conversation is still this user's
    choice, and ``resolve_effort_in`` is the same clamp the owner applies when a
    carried level outlives its route. ``deepseek-flash``'s table default is
    ``high``, so the unrankable word degrades there rather than to ``None``.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)
    session_id = await DesktopSessions(root).create(str(root))
    marker = root / "sessions" / session_id / "desktop.json"
    marker.write_text(
        json.dumps(
            {
                "version": 1,
                "cwd": str(root),
                "model": draft_model("deepseek", "deepseek-flash", "turbo"),
            }
        )
    )

    snapshot = await client.get(f"/v1/desktop/sessions/{session_id}")
    assert snapshot.status_code == 200, snapshot.text
    model = snapshot.json()["result"]["payload"]["frontend"]["snapshot"]["selected_model"]
    assert model["model_id"] == "deepseek-flash"
    assert model["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_a_pick_that_names_no_level_does_not_invent_one(draft_api) -> None:
    """Choosing a MODEL is not choosing a LEVEL (review round 1, R1).

    ``build_model_spec`` seeds the model's own default rung — ``high`` for
    ``deepseek-flash`` — and a seed that reaches the marker becomes a PIN: the
    plane exports it as ``LOP_MOBILE_CHILD_EFFORT`` and the first turn takes it
    over the machine's configured ``model_effort``. A level nobody chose must not
    silently replace the one they configured, so the marker records ``null`` while
    the pane and the launch both resolve the machine's configured level — exactly
    what a session with no pick at all resolves. The pair is still the user's; only
    the level is theirs to leave alone.

    The route half and the resolver half are BOTH asserted, because either alone
    re-pins the seed: the route would store it, or the resolver would seed it back
    from ``build_model_spec`` on the way in.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config({**CONFIGURED, "model_effort": "low"})
    session_id = await DesktopSessions(root).create(str(root), model=draft_model())

    stored = _marker(root, session_id)["model"]
    assert stored == {
        "provider": "deepseek",
        "model_id": "deepseek-flash",
        "reasoning_effort": None,
    }, "a level the user never named must not be stored as if they had"

    birth = module.draft_birth_selection(root, session_id)
    assert birth is not None
    assert (birth.provider, birth.model_id) == ("deepseek", "deepseek-flash")
    # The machine's configured level, CLAMPED into the pick's ladder — not the
    # model's seeded rung ("high") and not an absent one. The seed matters: it is
    # what the owner's pair-only model RPC would reseat the conversation on, so a
    # null level here would be the R1 defect arriving by another road.
    assert birth.reasoning_effort == "low"
    assert tuple(birth.reasoning_efforts) == ("none", "low", "high", "max")

    snapshot = await client.get(f"/v1/desktop/sessions/{session_id}")
    assert snapshot.status_code == 200, snapshot.text
    model = snapshot.json()["result"]["payload"]["frontend"]["snapshot"]["selected_model"]
    assert (model["provider"], model["model_id"]) == ("deepseek", "deepseek-flash")
    assert model["reasoning_effort"] == "low"
    assert model["context_window"] == 1000000

    # The DRAFT PREVIEW must agree with the frame the UI swaps it for, from the
    # same resolution: a pane reading "no level" while the first turn runs at the
    # configured one is the flicker ``finishDraft`` makes visible.
    preview = await client.post(
        "/v1/desktop/sessions/preview",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "model": draft_model()},
    )
    assert preview.status_code == 200, preview.text
    pane = preview.json()["result"]["frontend"]["snapshot"]["selected_model"]
    assert pane["reasoning_effort"] == model["reasoning_effort"] == "low"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document",
    [
        "not json at all",
        '["a", "list"]',
        '{"version": 1, "model": "deepseek/deepseek-flash"}',
        '{"version": 1, "model": {"provider": 7, "model_id": "deepseek-flash"}}',
        '{"version": 1, "model": {"provider": "deepseek", "model_id": ""}}',
        '{"version": 1}',
    ],
)
async def test_a_marker_this_code_cannot_read_costs_the_choice_not_the_session(
    draft_api, document
) -> None:
    """The marker is untrusted input, and an unreadable one must not fail an open.

    A hand edit, an interrupted write or an older build can shape ``desktop.json``
    arbitrarily, and the value read out of it decides which model the child
    runtime runs on. ``stored_draft_model`` re-checks every field for that reason;
    this pins the OBSERVABLE consequence: no birth seed, the session still opens,
    and it opens on the configured default — today's behaviour for a session with
    no choice (review round 1, R3).
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)
    session_id = await DesktopSessions(root).create(str(root))
    (root / "sessions" / session_id / "desktop.json").write_text(document)

    assert module.draft_birth_selection(root, session_id) is None
    snapshot = await client.get(f"/v1/desktop/sessions/{session_id}")
    assert snapshot.status_code == 200, snapshot.text
    model = snapshot.json()["result"]["payload"]["frontend"]["snapshot"]["selected_model"]
    assert (model["provider"], model["model_id"]) == ("anthropic", "claude-sonnet-5")


@pytest.mark.asyncio
async def test_a_marker_that_cannot_be_READ_AT_ALL_also_costs_only_the_choice(draft_api) -> None:
    """The other unreadable-marker shape: a path that is not a readable file.

    ``read_desktop_marker`` swallows ``OSError`` (a directory where the document
    should be, a permission the owner lost, a marker deleted mid-write) and answers
    ``None`` rather than propagating. The same rule as the malformed documents
    above has to hold on this path too, because it is the one a half-written marker
    takes.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)
    session_id = await DesktopSessions(root).create(str(root))
    marker = root / "sessions" / session_id / "desktop.json"
    marker.unlink()
    marker.mkdir()

    assert module.draft_birth_selection(root, session_id) is None
    snapshot = await client.get(f"/v1/desktop/sessions/{session_id}")
    assert snapshot.status_code == 200, snapshot.text
    model = snapshot.json()["result"]["payload"]["frontend"]["snapshot"]["selected_model"]
    assert (model["provider"], model["model_id"]) == ("anthropic", "claude-sonnet-5")


@pytest.mark.asyncio
async def test_a_marker_with_an_unreadable_level_keeps_the_pair_and_drops_the_level(
    draft_api,
) -> None:
    """A readable PAIR with an unusable level is not a discarded choice.

    The level is the only field allowed to disappear: the pair still resolves, so
    the conversation still opens on the model the user picked, at the level the
    machine resolves — and NOT at the unreadable value, nor at the model's seed.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)
    session_id = await DesktopSessions(root).create(str(root))
    (root / "sessions" / session_id / "desktop.json").write_text(
        json.dumps(
            {
                "version": 1,
                "cwd": str(root),
                "model": draft_model("deepseek", "deepseek-flash", 7),
            }
        )
    )

    birth = module.draft_birth_selection(root, session_id)
    assert birth is not None and birth.model_id == "deepseek-flash"
    # ``CONFIGURED`` names no ``model_effort``, so the machine's resolution IS the
    # model's own default rung — the level a launch that named no level would use.
    # The unreadable 7 is nowhere in it.
    assert birth.reasoning_effort == "high"
    snapshot = await client.get(f"/v1/desktop/sessions/{session_id}")
    model = snapshot.json()["result"]["payload"]["frontend"]["snapshot"]["selected_model"]
    assert model["model_id"] == "deepseek-flash"
    assert model["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_both_routes_answer_the_same_refusal_for_a_body_bad_in_two_ways(draft_api) -> None:
    """One body, one answer, whichever route is asked (review round 1, R4).

    Both routes are documented as answering the same refusals; that is only true
    if they also agree on WHICH one a body bad in two ways gets. So the same body —
    a working directory that does not exist AND a provider that does not — is sent
    to both, and the create route must also leave the request unclaimed: a refusal
    that writes the receipt would answer its own retry with "outcome
    indeterminate" instead of the refusal.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)
    request_id = str(uuid.uuid4())
    body = {
        "request_id": request_id,
        "cwd": str(root / "does-not-exist"),
        "model": draft_model("no-such-provider", "no-such-model", "high"),
    }

    created = await client.post("/v1/desktop/sessions", json=body)
    previewed = await client.post("/v1/desktop/sessions/preview", json=body)
    assert created.status_code == 409, created.text
    assert previewed.status_code == 409, previewed.text
    assert created.json()["detail"] == previewed.json()["detail"]

    # Unclaimed: the corrected retry of the SAME id is a create, not a conflict.
    corrected = await client.post(
        "/v1/desktop/sessions", json={"request_id": request_id, "cwd": str(root)}
    )
    assert corrected.status_code == 200, corrected.text
    assert corrected.json()["result"]["replayed"] is False


@pytest.mark.asyncio
async def test_a_replayed_create_answers_its_receipt_even_if_the_cwd_is_gone(
    draft_api, tmp_path
) -> None:
    """A retry of a request that SUCCEEDED is answered, not refused (round 2, R7).

    The four admissions run above the receipt claim so a refusal cannot claim a
    request id. The cost of putting them there is that they must NOT run for a
    request whose first attempt already created the session: the client the
    at-most-once contract exists for is the one whose response was lost, and
    answering it with "choose an existing working directory" turns a success into
    a failure. The probe is what keeps the claim's meaning — a recorded key is
    answered from its receipt, admissions and all.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)
    workdir = tmp_path / "work"
    workdir.mkdir()
    request_id = str(uuid.uuid4())
    body = {"request_id": request_id, "cwd": str(workdir)}

    first = await client.post("/v1/desktop/sessions", json=body)
    assert first.status_code == 200, first.text
    session_id = first.json()["result"]["session_id"]

    workdir.rmdir()
    replay = await client.post("/v1/desktop/sessions", json=body)
    assert replay.status_code == 200, replay.text
    assert replay.json()["result"]["session_id"] == session_id
    assert replay.json()["result"]["replayed"] is True


@pytest.mark.asyncio
async def test_a_refusal_at_the_model_admission_writes_nothing(draft_api) -> None:
    """The model admission runs BEFORE the target's registry build (round 2, R8).

    ``validate_target`` needs an ``AgentRegistry``, whose constructor materialises
    ``<config>/agents``. Running the target admission first therefore made a body
    refused at the MODEL step — which writes nothing at all — leave a directory
    behind, while the docstring and the docs both said a refusal writes nothing.
    Ordering the pure reads ahead of it is what makes that sentence true; both
    routes keep the same order, so they still answer the same refusal (R4).
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)
    assert not (root / "agents").exists()
    body = {
        "request_id": str(uuid.uuid4()),
        "cwd": str(root),
        "target": {"kind": "agent", "name": "manager"},
        "model": {"provider": "no-such-provider", "model_id": "x"},
    }

    created = await client.post("/v1/desktop/sessions", json=body)
    previewed = await client.post("/v1/desktop/sessions/preview", json=body)
    assert created.status_code == 422, created.text
    assert previewed.status_code == 422, previewed.text
    assert created.json()["detail"]["code"] == "provider_unknown"
    assert created.json()["detail"] == previewed.json()["detail"]
    assert not (root / "agents").exists(), "a refusal left <config>/agents behind"


def test_the_previews_model_read_does_not_materialise_a_config_directory(tmp_path) -> None:
    """``_configured_effort_without_writing`` reads the config WITHOUT writing it.

    The reader is named for the constraint it carries (round 2, Q-R2-3):
    ``ConfigManager``'s loader mkdirs the directory it is pointed at, and the
    preview is documented as side-effect free. Through HTTP the directory always
    exists, so the write was unreachable by accident rather than by construction;
    a root that does not exist has no configured level, which is the honest
    answer.
    """
    root = tmp_path / "absent" / "deeper"
    spec = desktop_sessions._preview_birth_model(
        root, desktop_sessions.DraftModel(provider="deepseek", model_id="deepseek-flash")
    )
    assert spec.model_id == "deepseek-flash"
    assert not root.exists(), "the config read created the directory it was pointed at"


@pytest.mark.asyncio
async def test_a_pick_with_no_level_matches_the_cold_frame_when_nothing_is_configured(
    draft_api,
) -> None:
    """The two readers agree when the machine configures NO level (round 2, R6).

    With no ``model_effort`` the level a birth RUNS at is the model's own seeded
    rung, and that is the case round 1 broke: the preview resolved through the
    seed-CLEARED spec (the marker's value) instead of the seeded one, so it said
    "no level" while the cold frame and the first turn said ``high``. ``CONFIGURED``
    carries no ``model_effort`` deliberately — with one set the defect is
    invisible, which is why the round-1 guard beside this one could not see it.
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)
    created = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "model": draft_model()},
    )
    assert created.status_code == 200, created.text
    session_id = created.json()["result"]["session_id"]

    stored = _marker(root, session_id)["model"]
    assert stored["reasoning_effort"] is None, "the marker stores the CHOICE, not the level"

    frame = (await client.get(f"/v1/desktop/sessions/{session_id}")).json()["result"]
    frame_model = frame["payload"]["frontend"]["snapshot"]["selected_model"]
    pane = await client.post(
        "/v1/desktop/sessions/preview",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "model": draft_model()},
    )
    pane_model = pane.json()["result"]["frontend"]["snapshot"]["selected_model"]

    assert frame_model["reasoning_effort"] == "high"
    assert pane_model["reasoning_effort"] == frame_model["reasoning_effort"]
    assert pane_model["reasoning_efforts"] == frame_model["reasoning_efforts"]


@pytest.mark.asyncio
async def test_an_unpicked_draft_answers_the_ladder_and_level_it_will_run_at(draft_api) -> None:
    """A draft that chose NO model still answers its effort reading (round 2).

    The operator's report: on a fresh conversation the effort reading was hidden and
    the picker unreachable, because the pane's ``selected_model`` carried an empty
    ladder and no level — a config-only projection — and the desktop strip gates its
    effort chip and its picker on exactly those fields. The ladder is a
    MODEL-derived field (``ModelSpec.reasoning_efforts``), so the draft's own
    resolution can answer it, and it must: the frame the UI swaps in at send answers
    it too, or the reading changes under the user.

    The WINDOW is deliberately not asserted equal here: it is model-derived in both
    readings, but a real cold open may apply ACCOUNT metadata and a window the plan
    scopes, and this op must not read account metadata (see the module docstring).
    """
    client, root = draft_api
    ConfigManager(config_dir=root).update_config(
        {"hosting": "deepseek", "model_name": "deepseek-flash"}
    )
    body = {"request_id": str(uuid.uuid4()), "cwd": str(root)}

    pane = await client.post("/v1/desktop/sessions/preview", json=body)
    pane_model = pane.json()["result"]["frontend"]["snapshot"]["selected_model"]
    assert (pane_model["provider"], pane_model["model_id"]) == ("deepseek", "deepseek-flash")
    assert pane_model["reasoning_efforts"], "the ladder is what gates the strip's effort chip"
    assert tuple(pane_model["reasoning_efforts"]) == ("none", "low", "high", "max")
    assert pane_model["reasoning_effort"] == "high"

    created = await client.post("/v1/desktop/sessions", json=body)
    session_id = created.json()["result"]["session_id"]
    frame = (await client.get(f"/v1/desktop/sessions/{session_id}")).json()["result"]
    frame_model = frame["payload"]["frontend"]["snapshot"]["selected_model"]
    assert frame_model["reasoning_efforts"] == pane_model["reasoning_efforts"]
    assert frame_model["reasoning_effort"] == pane_model["reasoning_effort"]


@pytest.mark.asyncio
async def test_the_cold_viewer_is_told_the_stored_choice_and_only_for_a_new_conversation(
    draft_api, monkeypatch
) -> None:
    """Spy on the ONE call that turns the marker into a birth sample.

    Three cases, and the two negative ones are the ones that carry the risk: a
    session with no stored choice must engage exactly as it did before this
    feature existed (``initial_model=None``, no override), and a session whose
    journal already answers the question must NOT be handed a birth sample at
    all — the override is what would drag a switched conversation back.
    """
    from local_operator.session.model_selection import SELECTED_MODEL_CUSTOM_TYPE

    client, root = draft_api
    ConfigManager(config_dir=root).update_config(CONFIGURED)
    seen: list[dict[str, Any]] = []
    real = module.AttachedSession.cold

    async def spy(cls, session_id, **kwargs):
        seen.append(kwargs)
        return await real.__func__(cls, session_id, **kwargs)

    monkeypatch.setattr(module.AttachedSession, "cold", classmethod(spy))

    chosen = await DesktopSessions(root).create(str(root), model=dict(CHOSEN_MODEL))
    plain = await DesktopSessions(root).create(str(root))
    switched = await DesktopSessions(root).create(str(root), model=dict(CHOSEN_MODEL))
    await Transcript(root / "sessions" / switched).append_custom(
        SELECTED_MODEL_CUSTOM_TYPE,
        {"version": 2, "selector": "anthropic/claude-opus-5", "effort": "low"},
    )
    for session_id in (chosen, plain, switched):
        assert (await client.get(f"/v1/desktop/sessions/{session_id}")).status_code == 200

    assert len(seen) == 3, seen
    birth = seen[0]["initial_model"]
    assert birth is not None and (birth.provider, birth.model_id) == ("deepseek", "deepseek-flash")
    assert birth.reasoning_effort == "max", "the chosen LEVEL must ride with the pair"
    assert seen[0]["model_selection_override"] is True
    assert seen[1]["initial_model"] is None
    assert seen[1]["model_selection_override"] is False
    assert seen[2]["initial_model"] is None, "the journal owns this conversation's selection"
    assert seen[2]["model_selection_override"] is False


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
        await _until(
            lambda: bool(engages),
            why="the lease's intent did not survive the recovery window",
        )
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
        await _until(
            lambda: bool(engages),
            why="the lease's intent did not survive the wait",
        )
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


# ---------------------------------------------------------------------------
# Moving a live session's working directory (POST .../working-directory)
# ---------------------------------------------------------------------------


class MoveClient:
    """The attach client's move-relevant surface, and nothing else.

    The two refusals that matter for a move are both decided from state a real
    client holds (a live socket, an ack that is not ``retiring``), so this double
    is what lets the route's tests reach them without spawning a runtime — the
    same shape ``tests/unit/session/test_remote_move.py`` uses one layer down,
    lifted here because these tests are about what the ROUTE and the BRIDGE do
    with each answer.
    """

    def __init__(self, answer: str = "retiring") -> None:
        self.answer = answer
        self.ops: list[str] = []
        # ``is_cold`` and ``move_will_wait`` both read this.
        self.connected = True

    async def retire_now(self) -> str:
        self.ops.append("retire_now")
        return self.answer

    def close(self) -> None:
        pass


class LegacyMoveClient:
    """A runtime from before the wire grew ``retire_now``.

    A class rather than a ``SimpleNamespace`` for the reason the warm tests
    record: ``dispose()`` tests the client for set membership, and a
    ``SimpleNamespace`` defines ``__eq__`` and so is unhashable.
    """

    connected = True

    def close(self) -> None:
        pass


def _bind_move_client(bridge: Any, client: object, *, idle: bool = True) -> None:
    """Make the bridge's facade look bound, and idle unless told otherwise.

    ``runtime_idle`` is replaced rather than arranged, because the boundary
    between "idle" and "working" is the RUNTIME's (it re-checks ``may_refresh``
    on its own side); what the route owes is a faithful mapping of each answer.
    """
    bridge.remote._client = client
    bridge.remote._ready_for_events = True
    bridge.remote.runtime_idle = lambda: idle  # type: ignore[method-assign]


def _move_body(cwd: str, request_id: str | None = None) -> dict[str, str]:
    return {"request_id": request_id or str(uuid.uuid4()), "cwd": cwd}


def _marker_path(root: Path, session_id: str) -> Path:
    return root / "sessions" / session_id / module.DESKTOP_MARKER_NAME


@pytest_asyncio.fixture
async def move_api(tmp_path: Path, monkeypatch):
    """A minimal app over THIS test's config root, for the move route.

    The same shape as ``draft_api`` above (its own ``tmp_path``, every ``CMUX_*``
    stripped, the pool closed after the client): the property under test is what
    a move does to the FILESYSTEM and to the bridge's own fields, so the root has
    to be the test's own and the pool has to outlive the requests rather than be
    reconstructed per call. ``app.state.desktop_sessions`` is how a test reaches
    the BRIDGE — the durability and re-engage claims are about what it does with
    fields it owns, not about the JSON the route returned.
    """
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "move-test-token")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    app.include_router(desktop_sessions.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer move-test-token"},
    ) as client:
        yield client, app, tmp_path.resolve()
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


@pytest.mark.asyncio
async def test_a_bound_move_retires_the_runtime_and_answers_the_new_directory(move_api) -> None:
    """The bound path: the runtime has to go, and the receipt says where it went.

    ``will_wait`` is asserted True because this is the case it exists to
    describe — the runtime has to be asked to retire, which takes seconds — and
    a MOVED session whose receipt said ``will_wait: False`` while it repaid a
    spawn would be the hint lying about the one thing it is read for.
    """
    client, app, root = move_api
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        double = MoveClient()
        _bind_move_client(bridge, double)
        request = _move_body(str(after))
        response = await client.post(f"/v1/desktop/sessions/{sid}/working-directory", json=request)

        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["outcome"] == "rebound"
        assert result["cwd"] == str(after)
        assert result["label"] == str(after).replace(str(root), "~", 1)
        assert result["will_wait"] is True
        assert double.ops == ["retire_now"], "the runtime was not asked to leave"
        assert bridge.remote.cwd == str(after), "the viewer still works in the old tree"


@pytest.mark.asyncio
async def test_a_cold_move_is_a_field_assignment_and_answers_cold(move_api, monkeypatch) -> None:
    """The common case — `lop` opens cold — is a field assignment, and the
    assertion that matters is not the return value: 92 green tests were once
    green while the runtime came up in the OLD directory, because they asserted
    ``_cwd`` rather than what the next engage was HANDED."""
    from local_operator.session.runtime import launch

    client, app, root = move_api
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))

    spawns: list[str] = []

    async def record_engage(session_id, cwd, work, **kwargs):
        spawns.append(str(cwd))

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None and bridge.remote.is_cold
        response = await client.post(
            f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(after))
        )
        result = response.json()["result"]
        assert result["outcome"] == "cold"
        assert result["will_wait"] is False
        assert bridge.remote.cwd == str(after)

        # The spawn the successor would make, driven for real: `engage_runtime`
        # is where the directory stops being a field and becomes a process.
        monkeypatch.setattr(launch, "engage_runtime", record_engage)
        try:
            await asyncio.wait_for(bridge.remote._ensure_bound(), timeout=10)
        except Exception:  # noqa: BLE001 — the engage's own outcome is not under test
            pass

    assert spawns == [str(after)], "the next engage was handed the wrong directory"


@pytest.mark.asyncio
async def test_a_move_rewrites_the_desktop_marker(move_api) -> None:
    """The durability claim of §2.4: the marker is what ``locate()`` reads after
    a server restart or a bridge eviction, so a move that leaves it naming the
    old directory is a session that silently resumes in the old tree."""
    client, app, root = move_api
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))
    marker = _marker_path(root, sid)
    assert json.loads(marker.read_text()) == {"version": 1, "cwd": str(before)}

    response = await client.post(
        f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(after))
    )

    assert response.status_code == 200, response.text
    assert json.loads(marker.read_text()) == {"version": 1, "cwd": str(after)}
    assert (marker.stat().st_mode & 0o777) == 0o600


@pytest.mark.asyncio
async def test_an_evicted_bridge_respawns_in_the_moved_directory(move_api, monkeypatch) -> None:
    """Both stale copies at once, which is why the route writes both.

    The BRIDGE field is the half the marker cannot cover: a re-``acquire()`` of a
    bridge still in the pool passes ``cwd=self.cwd`` to ``cold(cwd=…)``, and a
    field that was only ever set in ``__init__`` would spawn the successor in the
    directory the session LEFT. The marker is the half the bridge field cannot
    cover: an EVICTED bridge is rebuilt from ``locate()``, which reads the file.
    """
    monkeypatch.setattr(module, "BRIDGE_COUNT", 1)
    client, app, root = move_api
    before, after, other = root / "before", root / "after", root / "other"
    before.mkdir()
    after.mkdir()
    other.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))

    async with pool.session(sid):
        response = await client.post(
            f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(after))
        )
        assert response.status_code == 200, response.text

    # Released but retained: the bridge's own field, read back before anything
    # can rebuild it from disk.
    assert pool.bridges[sid].cwd == str(after), "the bridge field still names the old tree"

    other_sid = await pool.create(str(other))
    async with pool.session(other_sid):
        assert sid not in pool.bridges, "no eviction happened; this test proves nothing"

    async with pool.session(sid) as bridge:
        assert bridge.cwd == str(after), "the rebuilt bridge read the old directory"
        assert bridge.remote is not None and bridge.remote.cwd == str(
            after
        ), "the successor would be engaged in the old directory"


@pytest.mark.asyncio
async def test_moving_to_the_directory_youre_already_in_is_a_no_op(move_api) -> None:
    """The receipt the TUI prints ("already in ~/x"), and the property that makes
    a RETRY safe: a move that had already landed must not retire a second time,
    which is what lets the route declare ``retry_safe=True``."""
    client, app, root = move_api
    before = root / "before"
    before.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))
    marker = _marker_path(root, sid)
    untouched = marker.read_bytes()

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        double = MoveClient()
        _bind_move_client(bridge, double)
        response = await client.post(
            f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(before))
        )
        result = response.json()["result"]

        assert response.status_code == 200, response.text
        assert result["outcome"] == "unchanged"
        assert result["cwd"] == str(before)
        assert double.ops == [], "a no-op retired the runtime"
        assert bridge.remote.cwd == str(before)
    assert marker.read_bytes() == untouched, "a no-op rewrote the durable marker"


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["../sibling", "~/project"])
async def test_a_relative_or_tilde_path_resolves_against_the_SESSIONS_directory(
    move_api, typed
) -> None:
    """`resolve_working_directory` is the WRONG resolver here and the difference is
    the whole test: it resolves against this process's cwd, so ``/move ../sibling``
    would mean the sibling of the SERVER's directory. ``expand_path`` resolves
    against the session's, which is what the band shows and what the TUI does.
    ``~`` is the same rule's other half, and its ``label`` proves the backend
    renders it home-aware rather than echoing the typed text.
    """
    client, app, root = move_api
    start, sibling, project = root / "x" / "y", root / "x" / "sibling", root / "project"
    start.mkdir(parents=True)
    sibling.mkdir(parents=True)
    project.mkdir()
    expected = sibling if typed == "../sibling" else project
    pool = app.state.desktop_sessions
    sid = await pool.create(str(start))

    response = await client.post(
        f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(typed)
    )

    result = response.json()["result"]
    assert response.status_code == 200, response.text
    assert result["cwd"] == str(expected)
    assert result["label"] == str(expected).replace(str(root), "~", 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "sentence"),
    [
        ("absent", "no such directory: "),
        ("file", "not a directory: "),
        ("unreadable", "cannot enter "),
    ],
)
async def test_a_rejected_target_is_refused_with_409_and_moves_nothing(
    move_api, case, sentence
) -> None:
    """The three rejections told apart, because they call for three different next
    moves — and the state assertion afterwards is the point: a refusal must leave
    all three copies (the viewer's ``_cwd``, the bridge field, the marker) agreeing
    on the OLD directory. A 409 that had already rewritten the marker would make
    the NEXT server start resume in a directory the user was refused."""
    client, app, root = move_api
    before = root / "before"
    before.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))
    marker = _marker_path(root, sid)
    untouched = marker.read_bytes()

    if case == "absent":
        target = root / "nowhere"
    elif case == "file":
        target = root / "a-file"
        target.write_text("not a directory")
    else:
        target = root / "locked"
        target.mkdir()
        target.chmod(0o000)

    try:
        async with pool.session(sid) as bridge:
            assert bridge.remote is not None
            response = await client.post(
                f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(target))
            )

            assert response.status_code == 409, response.text
            assert sentence in response.json()["detail"], response.json()
            assert bridge.remote.cwd == str(before)
            assert bridge.cwd == str(before)
    finally:
        if case == "unreadable":
            target.chmod(0o700)
    assert marker.read_bytes() == untouched, "a refused move rewrote the durable marker"


@pytest.mark.asyncio
async def test_a_busy_session_is_refused_409_not_503(move_api) -> None:
    """The load-bearing mapping of this route.

    ``errors()`` answers a bare ``RuntimeError`` with 503 and "Session owner is
    unavailable. Reconnect and reconcile before retrying." A mid-turn session is
    not an unreachable backend — it is a HEALTHY session declining — and telling
    the user to reconnect would be advice about a problem they do not have.
    """
    client, app, root = move_api
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        double = MoveClient()
        _bind_move_client(bridge, double, idle=False)
        response = await client.post(
            f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(after))
        )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == (
            "this session is working right now — /move again when the turn finishes"
        )
        assert double.ops == []
        assert bridge.remote.cwd == str(before)


@pytest.mark.asyncio
async def test_a_runtime_that_keeps_itself_rolls_the_marker_back(move_api) -> None:
    """Work can arrive between this viewer's idle read and the runtime's own
    re-check, so the runtime is the authority and its reason is the receipt.

    The marker is asserted BYTE-identically: the durable copy is what a later
    server start reads, and leaving it at the refused directory is a move the
    user was told did not happen, silently applied on the next restart.
    """
    client, app, root = move_api
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))
    marker = _marker_path(root, sid)
    untouched = marker.read_bytes()

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        double = MoveClient(answer="kept: a background job started")
        _bind_move_client(bridge, double)
        response = await client.post(
            f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(after))
        )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "could not move: a background job started"
        assert bridge.cwd == str(before)
        assert bridge.remote.cwd == str(before)
    assert marker.read_bytes() == untouched, "the marker was left at the refused directory"


@pytest.mark.asyncio
async def test_a_version_skewed_runtime_gets_the_vetted_sentence(move_api) -> None:
    """A runtime too old to know ``retire_now``: refused in the sentence written
    for it rather than moved anyway and left in the old directory."""
    client, app, root = move_api
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        _bind_move_client(bridge, LegacyMoveClient())
        response = await client.post(
            f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(after))
        )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == (
            "this session's runtime is too old to be moved; /reload first"
        )
        assert bridge.remote.cwd == str(before)


@pytest.mark.asyncio
async def test_a_move_during_owner_recovery_answers_503(move_api) -> None:
    """The one refusal the route deliberately LEAVES to the ladder.

    A recovering viewer is chasing a successor that will bind at whatever
    directory the owner's record names, so a "cold move" reported here would be
    undone the moment that bind lands. The reconnect banner is the right surface
    for "the owner is reconnecting", which is why this stays a 503 with the
    ladder's sentence rather than a 409 with the session's.
    """
    client, app, root = move_api
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))
    marker = _marker_path(root, sid)
    untouched = marker.read_bytes()

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        bridge.remote._recovering = True
        response = await client.post(
            f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(after))
        )

        assert response.status_code == 503, response.text
        assert response.json()["detail"] == (
            "Session owner is unavailable. Reconnect and reconcile before retrying."
        )
        assert bridge.remote.cwd == str(before)
    assert marker.read_bytes() == untouched, "a refusing viewer moved the durable copy"


@pytest.mark.asyncio
async def test_a_retried_move_with_the_same_request_id_replays_the_first_receipt(move_api) -> None:
    """A lost response must not cost a SECOND retire.

    ``retry_safe=True`` is only safe because the re-run is a no-op — so the
    assertion that matters is not "replayed: true" but that the runtime was asked
    to retire exactly once.
    """
    client, app, root = move_api
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))
    request = _move_body(str(after))

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        double = MoveClient()
        _bind_move_client(bridge, double)
        first = await client.post(f"/v1/desktop/sessions/{sid}/working-directory", json=request)
        second = await client.post(f"/v1/desktop/sessions/{sid}/working-directory", json=request)

        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert "replayed" in second.json()["result"], (first.json(), second.json(), double.ops)
        assert second.json()["result"]["cwd"] == first.json()["result"]["cwd"]
        assert double.ops == ["retire_now"], "the retry retired the runtime a second time"


@pytest.mark.asyncio
async def test_a_failed_move_does_not_spend_its_request_id(move_api) -> None:
    """A refused move leaves the receipt row UNRESOLVED rather than claiming an
    outcome, so the same request can be retried once the reason for the refusal
    is gone — here, once the turn that was running has finished.

    The retry is the SAME request id and the SAME body, because that is what a
    retry IS: the fingerprint is the body's hash, so a client that changed its
    mind about the directory is a new request with a new id, and it gets the
    honest ``ReceiptConflict`` rather than a replay of an answer to a question it
    no longer asks.
    """
    client, app, root = move_api
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))
    request = _move_body(str(after))

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        double = MoveClient()
        _bind_move_client(bridge, double, idle=False)
        refused = await client.post(f"/v1/desktop/sessions/{sid}/working-directory", json=request)
        assert refused.status_code == 409, refused.text

        bridge.remote.runtime_idle = lambda: True  # type: ignore[method-assign]
        retried = await client.post(f"/v1/desktop/sessions/{sid}/working-directory", json=request)

        assert retried.status_code == 200, retried.text
        assert retried.json()["result"]["cwd"] == str(after)
        assert (
            retried.json()["result"]["replayed"] is False
        ), "a refused move recorded an outcome the retry then replayed"
        assert double.ops == ["retire_now"]


@pytest.mark.asyncio
async def test_a_retiring_frame_reengages_the_successor_on_the_desktop_bridge(tmp_path) -> None:
    """The §2.5 gap, and the test that would have caught it.

    ``retiring`` means "a successor is owed; engage one" for a build refresh and
    for a move alike. The TUI answers it with a refresh callback; the bridge
    installed NONE, so a retired runtime left the desktop viewer cold and the
    chip on the OLD directory until the user's next send happened to engage —
    which is the difference between the chip settling in a second and settling
    whenever the user next types.

    Driven through the real frame path (``_on_disconnected(RETIRING_REASON)``,
    which is what the client's pump delivers when the runtime announces
    ``retiring``), and the engage is the ordinary BACKGROUND one: a successor
    nobody asked for must not claim a foreground envelope.
    """
    from local_operator.mobile.attach_client import RETIRING_REASON

    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    async with pool.session(sid) as bridge:
        remote = bridge.remote
        assert remote is not None
        engages: list[bool] = []

        async def record_engage(*, foreground: bool = True) -> None:
            engages.append(foreground)

        remote._ensure_bound = record_engage  # type: ignore[method-assign]
        remote._on_disconnected(RETIRING_REASON)

        task = bridge.warm_task
        assert task is not None, "the retiring frame left the viewer cold with no engage"
        await asyncio.wait_for(task, timeout=10)
        assert remote.is_cold, "the retire frame must leave the viewer cold for its successor"
    assert engages == [False], "the successor engage must be the background envelope"
    await pool.close()


@pytest.mark.asyncio
async def test_the_move_route_is_advertised_by_session_move_only(move_api) -> None:
    """Its OWN key, and nothing else moves.

    A renderer that does not see ``session_move`` keeps its read-only
    working-directory chip — the EXISTING surface, working unchanged — which is
    the rule every other key in this map states. Bumping ``commands`` would hide
    the palette (which renders fine without this route) to guard a chip, and
    ``session_catalogue`` versions warming, not moving.
    """
    client, _app, _root = move_api
    response = await client.get("/v1/capabilities")
    features = response.json()["result"]["features"]

    assert response.status_code == 200, response.text
    assert features["session_move"] == 1
    assert features["commands"] == 1
    assert features["session_catalogue"] == 3


@pytest.mark.asyncio
async def test_the_command_endpoint_still_presents_move_rather_than_running_it(move_api) -> None:
    """``/move`` is a PRESENTATION request, on both forms, and it stays one.

    Adding ``move`` to ``OWNER_COMMANDS`` would route it to the runtime's slash
    dispatcher, which has no ``move`` branch and answers a user with a 200 notice
    about "this machine's configuration" — a false statement about the command
    and a dead end that looks like success. So the backend must keep answering
    the native action and must NOT execute anything: the renderer owns both the
    picker and the typed-path call.
    """
    from local_operator.server.utils.desktop_commands import OWNER_COMMANDS

    client, app, root = move_api
    before = root / "before"
    before.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))

    assert "move" not in OWNER_COMMANDS

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        double = MoveClient()
        _bind_move_client(bridge, double)
        bare = await client.post(
            f"/v1/desktop/sessions/{sid}/commands",
            json={"request_id": str(uuid.uuid4()), "command": "move"},
        )
        typed = await client.post(
            f"/v1/desktop/sessions/{sid}/commands",
            json={
                "request_id": str(uuid.uuid4()),
                "command": "move",
                "args": str(root / "after"),
            },
        )

        for response in (bare, typed):
            assert response.status_code == 200, response.text
            action = response.json()["result"]["result"]
            assert action["kind"] == "native_action"
            assert action["destination"] == "session.move"
        assert double.ops == [], "the command endpoint executed the move"
        assert bridge.remote.cwd == str(before)


@pytest.mark.asyncio
async def test_the_catalogue_keeps_its_own_mtime_for_a_session_with_a_transcript(move_api) -> None:
    """A move rewrites ``desktop.json``, and rewriting it must not reorder a real
    session in the sidebar: the marker's mtime is a DRAFT fallback for a directory
    with no transcript, so a session that has one keeps the mtime it already had.
    """
    from local_operator.session.catalog import load_catalog

    client, app, root = move_api
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))

    transcript = root / "sessions" / sid / TRANSCRIPT_FILENAME
    transcript.write_text(
        json.dumps({"type": ENTRY_MESSAGE, "role": "user", "content": "hello"}) + "\n"
    )
    old = time.time() - 3600
    os.utime(transcript, (old, old))

    response = await client.post(
        f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(after))
    )
    assert response.status_code == 200, response.text

    rows = {entry.row.id: entry.row for entry in load_catalog(app.state.config_manager.config_dir)}
    assert sid in rows, "the session fell out of the catalogue"
    assert rows[sid].mtime == pytest.approx(
        old, abs=1.0
    ), "the move's marker rewrite reordered a real session"


@pytest.mark.asyncio
async def test_a_move_keeps_the_drafts_stored_model(move_api) -> None:
    """A move changes ``cwd`` and NOTHING else.

    The marker is also where a draft's chosen model is stored (upstream's
    ``DRAFT_MODEL_KEY``), and a move rewrites the whole document. A writer that
    reproduced the file from its own arguments would silently discard the choice
    the new-conversation pane made — the first turn would then be born on a
    different model than the strip showed.
    """
    client, app, root = move_api
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(
        str(before),
        model={"provider": "anthropic", "model_id": "claude-opus-5", "reasoning_effort": "high"},
    )
    marker = _marker_path(root, sid)
    assert json.loads(marker.read_text())["model"]["model_id"] == "claude-opus-5"

    response = await client.post(
        f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(after))
    )

    assert response.status_code == 200, response.text
    stored = json.loads(marker.read_text())
    assert stored["cwd"] == str(after)
    assert stored["model"] == {
        "provider": "anthropic",
        "model_id": "claude-opus-5",
        "reasoning_effort": "high",
    }, "the move discarded the draft's chosen model"


@pytest.mark.asyncio
async def test_a_latched_daemon_gets_no_successor_from_the_retire_frame(tmp_path) -> None:
    """The retire frame must not spawn into a daemon that is being REPLACED.

    A build update latches the daemon (``server/retire.py``): it has told its
    clients to leave and its successor is on the way, so a runtime started here
    would be one whose viewer follows it onto a dead address — the refusal
    ``warm`` states, and the reason it asks ``assert_admitting`` before anything
    else. The retire-frame caller is not a route and has no named 503 to
    compose, so it declines silently: the app reconnects to whatever replaces
    the daemon.

    A MOVE is the other case, and the one this callback exists for — the daemon
    is healthy, only the session's runtime went, and the successor is owed by
    this frame alone. Both directions are pinned, so neither can be "fixed" into
    the other.
    """
    from local_operator.mobile.attach_client import RETIRING_REASON
    from local_operator.session.attached import AttachedSession

    bridge = module.DesktopSessionBridge(tmp_path, "s1", str(tmp_path), retiring=lambda: True)
    remote = await AttachedSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=module._no_takeover
    )
    bridge.remote = remote
    remote.set_refresh_callback(bridge._on_runtime_retired)

    remote._on_disconnected(RETIRING_REASON)

    assert bridge.warm_task is None, "a latched daemon's retire frame started a runtime"
    assert remote.is_cold


@pytest.mark.asyncio
async def test_concurrent_move_waits_for_refusal_rollback(move_api) -> None:
    client, app, root = move_api
    before, refused, accepted = (root / name for name in ("before", "refused", "accepted"))
    for directory in (before, refused, accepted):
        directory.mkdir()
    sid = await app.state.desktop_sessions.create(str(before))
    entered, release, queued = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class ObservedLock(asyncio.Lock):
        async def acquire(self) -> Literal[True]:
            if self.locked():
                queued.set()
            return await super().acquire()

    class RefuseFirst(MoveClient):
        async def retire_now(self) -> str:
            self.ops.append("retire_now")
            if len(self.ops) == 1:
                entered.set()
                await release.wait()
                return "kept: busy"
            return "retiring"

    async with app.state.desktop_sessions.session(sid) as bridge:
        bridge.move_lock = ObservedLock()
        _bind_move_client(bridge, RefuseFirst())
        # Both requests may acquire the bridge before either starts retiring.
        # Enter the shared transaction directly to pin that admitted ordering.
        first = asyncio.create_task(module.move_session(bridge, str(refused)))
        await asyncio.wait_for(entered.wait(), 5)
        second = asyncio.create_task(module.move_session(bridge, str(accepted)))
        try:
            await asyncio.wait_for(queued.wait(), 5)
            # The second request has reached the transaction lock, but cannot
            # write its marker until the first request has restored its own.
            assert json.loads(_marker_path(root, sid).read_text())["cwd"] == str(refused)
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="could not move: busy"):
            await first
        second_response = await second
        assert second_response.cwd == str(accepted)
        assert bridge.remote is not None
        assert bridge.cwd == bridge.remote.cwd == str(accepted)
        assert json.loads(_marker_path(root, sid).read_text())["cwd"] == str(accepted)
        assert await _published_cwd(client, sid) == str(accepted)


async def _published_cwd(client: Any, session_id: str) -> str:
    """The ``cwd`` a renderer reads out of the session's published state."""
    payload = (await client.get(f"/v1/desktop/sessions/{session_id}")).json()["result"]["payload"]
    return payload["frontend"]["snapshot"]["cwd"]


@pytest.mark.asyncio
async def test_a_move_is_published_in_the_state_the_chip_reads(move_api) -> None:
    """The STREAM is what the chip shows, so the published state must move too.

    Asserting the marker and the receipt alone is what let the stale-stream
    defect through: both named the new directory while the published
    ``frontend.cwd`` still named the old one, and the renderer's own rule is
    that the stream is authoritative — so it kept showing a directory the
    session had left.
    """
    client, app, root = move_api
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))
    # Keep the viewer acquired, as an open desktop event stream does. Releasing
    # it between requests rebuilds a cold snapshot from the already-correct
    # marker and would conceal the stale in-memory publication this tests.
    async with pool.session(sid):
        assert await _published_cwd(client, sid) == str(before)
        response = await client.post(
            f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(after))
        )
        assert response.status_code == 200, response.text
        assert response.json()["result"]["cwd"] == str(after)
        assert await _published_cwd(client, sid) == str(
            after
        ), "the receipt moved but the stream the chip reads did not"


@pytest.mark.asyncio
async def test_a_move_through_a_symlink_to_the_same_directory_is_a_no_op(move_api) -> None:
    """``/tmp/x`` and ``/private/tmp/x`` are ONE directory.

    Comparing spellings alone missed it (macOS's ``/tmp`` is the ordinary case),
    so a user typing the other name of the directory they were already in paid a
    full retire-and-respawn for a move that went nowhere — and got the rebuild
    notice on screen for it. The receipt answers with the SESSION's own spelling,
    which is what the frontend state stream reports, so the renderer's
    reconciliation has nothing to disagree with.
    """
    client, app, root = move_api
    real, link = root / "real", root / "link"
    real.mkdir()
    link.symlink_to(real)
    pool = app.state.desktop_sessions
    sid = await pool.create(str(real))

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        double = MoveClient()
        _bind_move_client(bridge, double)
        response = await client.post(
            f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(link))
        )
        result = response.json()["result"]

        assert response.status_code == 200, response.text
        assert (
            result["outcome"] == "unchanged"
        ), "the same directory through a symlink retired the runtime"
        assert result["cwd"] == str(real)
        assert double.ops == [], "a no-op retired the runtime"
    assert json.loads(_marker_path(root, sid).read_text())["cwd"] == str(real)


@pytest.mark.asyncio
async def test_a_bound_move_publishes_the_moved_directory_at_once(move_api) -> None:
    """The window between the retire and the successor's bind is the chip's whole world.

    A bound move returns after the OLD runtime has been asked to leave and before
    any successor has published: for those seconds (and for as long as the engage
    takes, or for good if it fails) the only state any surface can read is the one
    the viewer holds. Left alone it named the directory the session had LEFT —
    the receipt, the marker and a real `bash pwd` all named the new one, and the
    stream the renderer trusts named the old one (QA Q1 on the desktop move).
    """
    client, app, root = move_api
    before, after = root / "before", root / "after"
    before.mkdir()
    after.mkdir()
    pool = app.state.desktop_sessions
    sid = await pool.create(str(before))

    async with pool.session(sid) as bridge:
        assert bridge.remote is not None
        _bind_move_client(bridge, MoveClient())
        assert await _published_cwd(client, sid) == str(before)
        outgoing = bridge.remote.frontend_state

        response = await client.post(
            f"/v1/desktop/sessions/{sid}/working-directory", json=_move_body(str(after))
        )

        assert response.status_code == 200, response.text
        assert response.json()["result"]["outcome"] == "rebound"
        assert await _published_cwd(client, sid) == str(
            after
        ), "the move retired the runtime and left the stream naming the old directory"
        # A final owner delta can already be in flight when retire is accepted.
        # Local publication must not consume the sequence reserved for that delta.
        bridge.remote._on_frontend_update(
            {
                "epoch": outgoing.epoch,
                "sequence": outgoing.sequence + 1,
                "changes": {"conversation_title": "Final owner update"},
            }
        )
        assert bridge.remote.frontend_state.cwd == str(after)
        assert bridge.remote.frontend_state.conversation_title == "Final owner update"
