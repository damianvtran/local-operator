"""Desktop stream algebra, resource bounds and durable retry invariants."""

import asyncio
import base64
import json
from pathlib import Path
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
