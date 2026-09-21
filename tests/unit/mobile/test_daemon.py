"""End-to-end over the real sockets: a fake registrant publishes a record,
the daemon adopts it, projections flow, control requests round-trip, and the
HTTP gate holds. These tests use the repo's config-dir isolation (conftest
sets LOCAL_OPERATOR_CONFIG_DIR to a tmp path)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.mobile.daemon import (
    MobileDaemon,
    SessionEntry,
    SessionTable,
    _dial,
    build_app,
)
from local_operator.mobile.types import (
    PROJECTION_TRANSCRIPT_LIMIT,
    SessionProjection,
    SessionRecord,
    SubagentRow,
    TodoItem,
    TodoPhase,
    TranscriptEntry,
)
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer


class FakeHandle:
    """The minimal SessionHandle: static projection, echo answers."""

    def __init__(self) -> None:
        self._projection = SessionProjection(
            session_id="s1",
            pid=0,
            kind="tui",
            conversation_name="fake session",
            cwd="/tmp",
            model_label="anthropic/claude-opus-5",
        )
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    @property
    def session_projection_seed(self) -> SessionProjection:
        return self._projection

    def subscribe(self, on_projection):  # noqa: ANN001, ANN202
        return lambda: None

    async def _record(self, name: str, *args, **kwargs) -> str:  # noqa: ANN202
        self.calls.append((name, args, kwargs))
        return f"{name} ok"

    async def prompt(self, text, images=None, command_id=None):  # noqa: ANN001, ANN202
        return await self._record("prompt", text)

    async def steer(self, text, images=None, command_id=None):  # noqa: ANN001  # noqa: ANN202
        return await self._record("steer", text, command_id=command_id)

    async def abort(self):  # noqa: ANN202
        return await self._record("abort")

    async def set_model(self, provider, model_id):  # noqa: ANN001, ANN202
        return await self._record("set_model", provider, model_id)

    async def set_effort(self, effort):  # noqa: ANN001, ANN202
        return await self._record("set_effort", effort)

    async def slash(self, command, args):  # noqa: ANN001, ANN202
        return await self._record("slash", command, args)

    async def new_conversation(self):  # noqa: ANN202
        return await self._record("new_conversation")

    async def resume_session(self, session_id):  # noqa: ANN001, ANN202
        return await self._record("resume_session", session_id)

    async def approval_answer(self, request_id, approved, remember):  # noqa: ANN001, ANN202
        return await self._record("approval_answer", request_id, approved, remember)

    async def ask_answer(self, request_id, value, question_index=None):  # noqa: ANN001, ANN202
        return await self._record("ask_answer", request_id, value)

    async def refresh(self) -> None:
        pass


@pytest.mark.asyncio
async def test_registrant_publishes_and_daemon_adopts() -> None:
    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    try:
        # Wait for the record to appear.
        deadline = asyncio.get_running_loop().time() + 5
        record = None
        while asyncio.get_running_loop().time() < deadline:
            found = registry.scan()
            if found:
                record, state = found[0]
                if state == "live":
                    break
            await asyncio.sleep(0.1)
        assert record is not None
        assert record.control_port > 0

        daemon = MobileDaemon(port=0, password="pw")
        entry = SessionEntry(record)
        daemon.table.entries[record.pid] = entry
        dial = asyncio.ensure_future(_dial(daemon, entry))
        try:
            for _ in range(50):
                if entry.projection is not None:
                    break
                await asyncio.sleep(0.1)
            assert entry.projection is not None
            assert entry.projection.conversation_name == "fake session"

            reply = await daemon.request(record.pid, "prompt", text="hello")
            assert reply["op"] == "ack"
            assert handle.calls[-1][0] == "prompt"

            command_id = "12345678-1234-4678-9234-567812345678"
            reply = await daemon.request(
                record.pid,
                "steer",
                command_id=command_id,
                text="parent instruction",
            )
            # Field-wise rather than whole-frame: ack frames gain additive
            # fields over time (``duplicate`` for command idempotency), and an
            # equality assertion here would fail every such addition while
            # claiming to test the steer receipt.
            assert reply["op"] == "ack"
            assert reply["detail"] == "steer ok"
            assert handle.calls[-1] == (
                "steer",
                ("parent instruction",),
                {"command_id": command_id},
            )

            reply = await daemon.request(record.pid, "set_effort", effort="high")
            assert "set_effort ok" in reply["detail"]
        finally:
            dial.cancel()
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_dial_skips_oversized_frame_and_delivers_the_next() -> None:
    """An oversized control frame must degrade to "drop this one frame", not
    wedge the session on the durable fold forever.

    ``StreamReader.readline`` DRAINS an over-limit line (clears the buffer)
    before raising ``ValueError``, so ``_dial``'s ``except ValueError: continue``
    already recovers — this pins that contract. The real guard against ever
    reaching this path is fix #1 (subagent transcripts no longer ride the wire),
    but a single future oversized frame must still leave the connection usable so
    the NEXT normal projection lands and the phone keeps updating live. A
    regression that reintroduced huge frames, OR a swap to ``readuntil`` (which
    would NOT drain and would re-raise on the same bytes forever), fails here.
    """

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # Consume the daemon's auth line, then push one frame past the 1 MB
        # limit followed by a valid projection the daemon must still adopt.
        await reader.readline()
        writer.write(b"{" + b"x" * (2 << 20) + b"}\n")
        good = {
            "op": "projection",
            "data": {"session_id": "s-oversize", "pid": 4321, "conversation_name": "recovered"},
        }
        writer.write(json.dumps(good).encode() + b"\n")
        await writer.drain()
        # Close the fixture's side explicitly: the daemon never closes
        # mid-stream, and a writer left open keeps the handler's socket
        # transport alive after the test body finishes — Python 3.12's
        # teardown then waits on that leaked transport forever (3.13+
        # reap it, which is why only the 3.12 CI job hung).
        writer.close()
        with contextlib.suppress(ConnectionResetError):
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    record = SessionRecord(
        pid=4321,
        kind="tui",
        session_id="s-oversize",
        conversation_name="recovered",
        cwd="/tmp",
        model_label="test/model",
        control_port=port,
        control_key="secret",
    )
    daemon = MobileDaemon(port=0, password="pw")
    entry = SessionEntry(record)
    daemon.table.entries[record.pid] = entry
    dial = asyncio.ensure_future(_dial(daemon, entry))
    try:
        # Poll is bounded (50 x 0.05 s = 2.5 s), so a functional regression
        # here fails the asserts instead of hanging the suite; the historical
        # 3.12 hang was teardown, below, not this wait.
        for _ in range(50):
            if entry.projection is not None:
                break
            await asyncio.sleep(0.05)
        # The oversized frame was skipped; the following normal frame arrived.
        assert entry.projection is not None
        assert entry.projection.conversation_name == "recovered"
    finally:
        dial.cancel()
        # A cancelled task that is never awaited leaves the dial's reader/
        # socket transport un-reaped; Python 3.12's asyncio teardown waits on
        # it forever (3.13+ tolerate the leak). Await the cancellation to
        # completion before tearing the server down.
        with contextlib.suppress(asyncio.CancelledError):
            await dial
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_wrong_key_is_rejected_silently() -> None:
    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    try:
        deadline = asyncio.get_running_loop().time() + 5
        record = None
        while asyncio.get_running_loop().time() < deadline:
            found = registry.scan()
            if found:
                record = found[0][0]
                break
            await asyncio.sleep(0.1)
        assert record is not None

        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        writer.write(json.dumps({"key": "wrong"}).encode() + b"\n")
        await writer.drain()
        # The registrant closes without a reply: reading yields EOF.
        data = await asyncio.wait_for(reader.read(), timeout=5)
        assert data == b""
        writer.close()
    finally:
        registrant.close()


def test_http_gate_and_login_flow() -> None:
    daemon = MobileDaemon(port=0, password="pw123")
    app = build_app(daemon)
    client = TestClient(app, follow_redirects=False)

    assert client.get("/healthz").status_code == 200
    assert client.get("/api/sessions").status_code == 401
    root = client.get("/")
    assert root.status_code == 303
    assert root.headers["location"] == "/login"

    bad = client.post("/login", data={"password": "nope"})
    assert bad.status_code == 401

    good = client.post("/login", data={"password": "pw123"})
    assert good.status_code == 303
    assert "lop_mobile" in good.headers["set-cookie"]

    authed = client.get("/api/sessions")
    assert authed.status_code == 200
    # The listing carries the durable-read marker beside the rows (present on
    # every frame, empty when everything was read). Asserted in full rather than
    # by key so a field appearing here is a decision this test sees.
    assert authed.json() == {"sessions": [], "degraded": []}

    logout = client.get("/logout")
    assert logout.status_code == 303
    assert logout.headers["location"] == "/login"
    assert logout.headers["clear-site-data"] == '"storage"'
    assert "lop_mobile=" in logout.headers["set-cookie"]


def test_login_page_clears_private_storage_without_relying_on_header() -> None:
    """U2: the WebKit-safe cleanup path. Every logout/401/expiry lands on the
    server-rendered login page, whose inline script clears the private storage
    prefixes in the page's own engine — so cleanup does not depend on the
    ``Clear-Site-Data`` header that WebKit may ignore."""
    daemon = MobileDaemon(port=0, password="pw123")
    client = TestClient(build_app(daemon), follow_redirects=False)
    body = client.get("/login").text
    # The script must remove exactly the two private prefixes and nothing else
    # (theme and other preferences survive), matching web/src/private-storage.ts.
    assert "localStorage.removeItem(key)" in body
    assert '"lo-mobile-command:"' in body
    assert '"lo-mobile-draft:"' in body
    # It must not blanket-clear storage, which would wipe non-private prefs.
    assert "localStorage.clear()" not in body


def test_subagent_summary_detail_and_child_history_are_isolated(tmp_path, monkeypatch) -> None:
    """Root repaints stay light while the selected child pages its own file."""
    from local_operator.harness.types import Message
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    root_dir = cfg / "sessions" / "root-session"
    child_dir = cfg / "sessions" / "child-session"
    root_dir.mkdir(parents=True)
    child_dir.mkdir(parents=True)
    asyncio.run(Transcript(root_dir).append_message(Message.user("root-only", id="root-row")))
    asyncio.run(Transcript(child_dir).append_message(Message.user("child-only", id="child-row")))

    daemon = MobileDaemon(port=0, password="pw123")
    projection = SessionProjection(session_id="root-session", pid=9, version=7)
    projection.subagents = [
        SubagentRow(
            job_id="child-job",
            label="child",
            session_id="child-session",
            transcript=[TranscriptEntry(id="child-row", kind="user", text="child-only")],
            todos=[TodoPhase(name="Todos", items=[TodoItem(text="verify")])],
        )
    ]
    summary = daemon.capture_subagent_details(projection)
    assert projection.subagents[0].transcript[0].text == "child-only"
    assert projection.subagents[0].todos[0].items[0].text == "verify"
    assert summary.subagents[0].transcript == []
    assert summary.subagents[0].todos == []

    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    detail = client.get("/api/sessions/root-session/agents/child-job")
    assert detail.status_code == 200
    assert detail.json()["version"] == 7
    assert [entry["text"] for entry in detail.json()["transcript"]] == ["child-only"]
    history = client.get(
        "/api/sessions/root-session/agents/child-job/history", params={"limit": 10}
    )
    assert history.status_code == 200
    assert [entry["id"] for entry in history.json()["entries"]] == ["child-row"]
    assert "root-row" not in str(history.json())
    assert client.get("/api/sessions/root-session/agents/not-related").status_code == 404


def test_runtime_hosted_roster_routes_the_child_history_end_to_end(tmp_path, monkeypatch) -> None:
    """A runtime-hosted session must publish its children's session dirs.

    ``/agents/{job}/history`` resolves a child through the folded roster's
    ``session_id``, and the ONLY writer of that field is
    ``ProjectionFold.set_subagent_details``. Since the viewer/runtime split the
    phone's sessions are hosted by ``ServingSessionHandle``, which mirrored the
    TUI handle's state push but not its roster push -- and the event path can
    never learn a child's session dir (``SubagentStartEvent`` carries no session
    id), so every runtime-hosted session published ``session_id: null`` for
    every child and the route 404'd for all of them. The summary/detail test
    above HAND-BUILDS that row, which is why nothing caught it; this drives the
    real producer and walks it out to the HTTP route.
    """
    from local_operator.harness.comms import SubagentComms
    from local_operator.harness.types import Message, SubagentStartEvent
    from local_operator.mobile.daemon import _projection_frame
    from local_operator.mobile.types import _projection_from_json
    from local_operator.session.runtime.serving import ServingSessionHandle
    from local_operator.session.transcript import Transcript
    from tests.unit.harness.test_comms import FakeChild, FakeJobs, FakeParent
    from tests.unit.session.runtime.test_serving import FakeSession

    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    child_dir = cfg / "sessions" / "child-session"
    grandchild_dir = cfg / "sessions" / "grandchild-session"
    for directory, text, row_id in (
        (child_dir, "child-only", "child-row"),
        (grandchild_dir, "grandchild-only", "grandchild-row"),
    ):
        directory.mkdir(parents=True)
        asyncio.run(Transcript(directory).append_message(Message.user(text, id=row_id)))

    jobs = FakeJobs()
    jobs.add("child-job", status="running")
    jobs.add("grandchild-job", status="running")
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    comms.record_launch("child-job", "child", prompt="Inspect it.")
    comms.attach("child-job", FakeChild(), child_dir)  # type: ignore[arg-type]
    comms.record_launch("grandchild-job", "grandchild", parent_job_id="child-job")
    comms.attach("grandchild-job", FakeChild(), grandchild_dir)  # type: ignore[arg-type]

    async def build() -> tuple[SessionProjection, FakeSession]:
        session = FakeSession()
        session.session_id = "root-session"
        session._subagent_comms = comms  # type: ignore[attr-defined]
        handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
        handle.subscribe(lambda: None)
        # The attach seed alone must publish the roster: a settled child sends
        # no further event, so an event-only push leaves it unroutable for as
        # long as the session stays quiet.
        return handle.session_projection_seed, session

    projection, session = asyncio.run(build())
    rows = {row.job_id: row.session_id for row in projection.subagents}
    # Nested children live only in the shared registry, so the walk must reach
    # them too -- the phone opens a grandchild from the roster's Children.
    assert rows == {"child-job": "child-session", "grandchild-job": "grandchild-session"}

    record = SessionRecord(
        pid=9,
        kind="daemon",
        session_id="root-session",
        conversation_name="root",
        cwd=str(tmp_path),
        model_label="",
        control_port=1,
        control_key="k",
        started_at=0.0,
        heartbeat_at=0.0,
    )
    # The same boundary the daemon's dial loop crosses: the registrant
    # serializes the frame and the daemon rebuilds it before capturing.
    incoming = _projection_from_json(json.loads(json.dumps(_projection_frame(projection))), record)
    daemon = MobileDaemon(port=0, password="pw123")
    daemon.capture_subagent_details(incoming, record=record)

    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    history = client.get(
        "/api/sessions/root-session/agents/child-job/history", params={"limit": 10}
    )
    assert history.status_code == 200
    assert [entry["text"] for entry in history.json()["entries"]] == ["child-only"]
    nested = client.get(
        "/api/sessions/root-session/agents/grandchild-job/history", params={"limit": 10}
    )
    assert nested.status_code == 200
    assert [entry["text"] for entry in nested.json()["entries"]] == ["grandchild-only"]

    # A live child announcing itself through the root event stream must keep
    # the same row (the event path rebuilds it without a session id).
    session.emit(SubagentStartEvent(job_id="child-job", label="child"))
    assert {row.job_id: row.session_id for row in projection.subagents} == rows


def test_retained_summary_recapture_preserves_rich_detail_and_monotonic_version() -> None:
    """Wake/reconnect may recapture only the already-stripped retained summary."""
    daemon = MobileDaemon(port=0, password="pw123")
    projection = SessionProjection(session_id="root-session", pid=9, version=7)
    projection.subagents = [
        SubagentRow(
            job_id="child-job",
            label="child",
            prompt="secret prompt",
            launch_message_id="subagent-launch:child-job",
            status="completed",
            result_text="full result",
            transcript=[TranscriptEntry(id="child-row", kind="assistant", text="full reply")],
            todos=[TodoPhase(name="Work", items=[TodoItem(text="ship", status="done")])],
        )
    ]

    summary = daemon.capture_subagent_details(projection)
    assert summary is daemon.session_projections["root-session"]
    assert summary is not projection
    assert summary.subagents[0].prompt == ""

    # The exact retained object is reused by wake and reconnect paths. Repeated
    # capture must be idempotent instead of treating stripped empties as updates.
    recaptured = daemon.capture_subagent_details(summary)
    stale = SessionProjection(session_id="root-session", pid=9, version=5)
    stale.subagents = [SubagentRow(job_id="child-job", label="stale child")]
    recaptured = daemon.capture_subagent_details(stale)
    detail = daemon.subagent_details[("root-session", "child-job")]
    assert recaptured.version == 7
    assert detail["version"] == 7
    assert detail["prompt"] == "secret prompt"
    assert detail["launch_message_id"] == "subagent-launch:child-job"
    assert detail["result_text"] == "full result"
    assert detail["transcript"][0]["text"] == "full reply"
    assert detail["todos"][0]["items"][0]["text"] == "ship"


def test_new_process_generation_supersedes_high_version_and_rejects_late_old_frame() -> None:
    daemon = MobileDaemon(port=0, password="pw123")
    old_record = SessionRecord(
        pid=101,
        kind="tui",
        session_id="root-session",
        conversation_name="old",
        cwd="/tmp",
        model_label="old-model",
        control_port=4101,
        control_key="old-registration",
        started_at=100.0,
        heartbeat_at=101.0,
    )
    old = SessionProjection(
        session_id="root-session",
        pid=101,
        version=40,
        transcript=[TranscriptEntry(id="old-root", kind="assistant", text="old root")],
        todos=[TodoPhase(name="Old", items=[TodoItem(text="old todo")])],
    )
    old.subagents = [
        SubagentRow(
            job_id="child-job",
            label="old child",
            status="running",
            prompt="old prompt",
            transcript=[TranscriptEntry(id="old-child", kind="assistant", text="old child")],
            todos=[TodoPhase(name="Old child", items=[TodoItem(text="old child todo")])],
        )
    ]
    assert daemon.capture_subagent_details(old, record=old_record).version == 40

    # started_at + registration key distinguishes process birth even if the OS
    # reuses the PID; its low ProjectionFold counter must still advance the
    # daemon epoch and rematerialize every detail-only field from this owner.
    new_record = SessionRecord(
        pid=101,
        kind="tui",
        session_id="root-session",
        conversation_name="new",
        cwd="/tmp",
        model_label="new-model",
        control_port=4202,
        control_key="new-registration",
        started_at=200.0,
        heartbeat_at=201.0,
    )
    new = SessionProjection(
        session_id="root-session",
        pid=101,
        version=1,
        transcript=[TranscriptEntry(id="new-root", kind="assistant", text="new root")],
        todos=[TodoPhase(name="New", items=[TodoItem(text="new todo", status="done")])],
    )
    new.subagents = [
        SubagentRow(
            job_id="child-job",
            label="new child",
            status="completed",
            prompt="new prompt",
            result_text="new result",
            transcript=[TranscriptEntry(id="new-child", kind="assistant", text="new child")],
            todos=[TodoPhase(name="New child", items=[TodoItem(text="new child todo")])],
        )
    ]
    current = daemon.capture_subagent_details(new, record=new_record)
    assert current.version == 41
    assert [row.id for row in current.transcript] == ["new-root"]
    assert current.todos[0].items[0].text == "new todo"
    detail = daemon.subagent_details[("root-session", "child-job")]
    assert detail["version"] == 41
    assert detail["label"] == "new child"
    assert detail["status"] == "completed"
    assert detail["prompt"] == "new prompt"
    assert detail["result_text"] == "new result"
    assert detail["transcript"][0]["id"] == "new-child"
    assert detail["todos"][0]["items"][0]["text"] == "new child todo"

    late = daemon.capture_subagent_details(old, record=old_record)
    assert late is current
    assert daemon.subagent_details[("root-session", "child-job")]["label"] == "new child"


def test_scan_replaces_process_state_when_registration_reuses_pid(monkeypatch) -> None:
    daemon = MobileDaemon(port=0, password="pw123")
    old_record = SessionRecord(
        pid=101,
        kind="tui",
        session_id="root-session",
        conversation_name="old",
        cwd="/tmp",
        model_label="old-model",
        control_port=4101,
        control_key="old-registration",
        started_at=100.0,
    )
    old_entry = SessionEntry(old_record)
    old_entry.ended = True
    old_entry.degraded = True
    daemon.table.entries[101] = old_entry
    new_record = SessionRecord(
        pid=101,
        kind="tui",
        session_id="root-session",
        conversation_name="new",
        cwd="/tmp",
        model_label="new-model",
        control_port=4202,
        control_key="new-registration",
        started_at=200.0,
    )
    monkeypatch.setattr(registry, "scan", lambda: [(new_record, "live")])
    dialed: list[SessionEntry] = []

    async def fake_dial(_daemon, entry):  # noqa: ANN001, ANN202
        dialed.append(entry)

    monkeypatch.setattr("local_operator.mobile.daemon._dial", fake_dial)
    asyncio.run(daemon._scan_once())

    replacement = daemon.table.entries[101]
    assert replacement is not old_entry
    assert replacement.record.control_key == "new-registration"
    assert replacement.ended is False
    assert replacement.degraded is False
    assert dialed == [replacement]


def test_terminal_fold_advances_epoch_and_blocks_late_live_frame() -> None:
    daemon = MobileDaemon(port=0, password="pw123")
    record = SessionRecord(
        pid=101,
        kind="tui",
        session_id="root-session",
        conversation_name="live",
        cwd="/tmp",
        model_label="model",
        control_port=4101,
        control_key="registration",
        started_at=100.0,
    )
    live = SessionProjection(session_id="root-session", pid=101, version=40, streaming=True)
    daemon.capture_subagent_details(live, record=record)
    durable = SessionProjection(
        session_id="root-session",
        pid=0,
        version=1,
        ended=True,
        transcript=[TranscriptEntry(id="durable", kind="assistant", text="settled")],
    )
    terminal = daemon.capture_subagent_details(durable, record=record, terminal=True)
    assert terminal.version == 41
    assert terminal.ended is True
    assert daemon.capture_subagent_details(live, record=record) is terminal


def test_live_generation_epoch_survives_payload_eviction_pressure() -> None:
    """A browser-observed epoch must outlive bounded route payload eviction."""
    from local_operator.mobile.daemon import (
        MAX_RETAINED_SESSION_PROJECTIONS,
        SessionEntry,
    )

    daemon = MobileDaemon(port=0, password="pw123")
    old_record = SessionRecord(
        pid=101,
        kind="tui",
        session_id="root-session",
        cwd="/tmp",
        model_label="old",
        conversation_name="root",
        heartbeat_at=10,
        control_port=4101,
        control_key="old-key",
        started_at=10,
    )
    new_record = SessionRecord(
        pid=102,
        kind="tui",
        session_id="root-session",
        cwd="/tmp",
        model_label="new",
        conversation_name="root",
        heartbeat_at=20,
        control_port=4102,
        control_key="new-key",
        started_at=20,
    )
    daemon.table.entries[new_record.pid] = SessionEntry(new_record)

    old = SessionProjection(session_id="root-session", pid=101, version=40)
    assert daemon.capture_subagent_details(old, record=old_record).version == 40
    replacement = SessionProjection(session_id="root-session", pid=102, version=1)
    assert daemon.capture_subagent_details(replacement, record=new_record).version == 41

    for index in range(MAX_RETAINED_SESSION_PROJECTIONS):
        daemon.capture_subagent_details(
            SessionProjection(session_id=f"pressure-{index:03d}", pid=200 + index, version=1)
        )
    assert "root-session" not in daemon.session_projections
    assert daemon._projection_generations["root-session"].epoch == 41

    next_replacement = SessionProjection(session_id="root-session", pid=102, version=2)
    assert daemon.capture_subagent_details(next_replacement, record=new_record).version == 42
    late_old = SessionProjection(session_id="root-session", pid=101, version=999)
    assert (
        daemon.capture_subagent_details(late_old, record=old_record)
        is daemon.session_projections["root-session"]
    )
    assert daemon.session_projections["root-session"].version == 42

    daemon.session_projections.pop("root-session")
    from local_operator.mobile.daemon import _StaleProjection

    with pytest.raises(_StaleProjection):
        daemon.capture_subagent_details(late_old, record=old_record)
    daemon.table.entries[new_record.pid].ended = True
    daemon._prune_projection_generation("root-session")
    assert "root-session" not in daemon._projection_generations


def test_evicted_payload_reconstructs_under_epoch_while_late_old_frame_stays_fenced(
    tmp_path, monkeypatch
) -> None:
    """The lifecycle contract's hardest case, end to end.

    A high daemon epoch is superseded by a low-version replacement owner; the
    only route payload is then evicted under cache pressure while a live SSE
    subscriber keeps the generation ledger alive. Detail and history must still
    reconstruct from durable disk (re-admitted at the retained monotonic epoch,
    NOT fenced to HTTP 500), and a genuine late frame from the OLD process must
    still be fenced. This is the single documented reconciliation the R9 pass
    replaced case-by-case eviction patches with.
    """
    from local_operator.harness.types import Message
    from local_operator.mobile.daemon import MAX_RETAINED_SESSION_PROJECTIONS
    from local_operator.session.session import SUBAGENT_ROSTER_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    root_dir = cfg / "sessions" / "root-session"
    child_dir = cfg / "sessions" / "child-session"
    root_dir.mkdir(parents=True)
    child_dir.mkdir(parents=True)
    (child_dir / "origin.json").write_text('{"origin":"subagent"}')
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    asyncio.run(Transcript(root_dir).append_message(Message.user("root", id="root-row")))
    asyncio.run(Transcript(child_dir).append_message(Message.assistant("reply", id="child-row")))
    asyncio.run(
        Transcript(root_dir).append_custom(
            SUBAGENT_ROSTER_CUSTOM_TYPE,
            {
                "jobs": [{"id": "child-job", "status": "completed", "label": "child"}],
                "records": [
                    {
                        "job_id": "child-job",
                        "label": "child",
                        "prompt": "durable prompt",
                        "session_dir": str(child_dir),
                        "outcome": "completed",
                        "result_text": "done",
                    }
                ],
            },
        )
    )

    daemon = MobileDaemon(port=0, password="pw123")
    old_record = SessionRecord(
        pid=101,
        kind="tui",
        session_id="root-session",
        conversation_name="root",
        cwd="/tmp",
        model_label="old",
        control_port=4101,
        control_key="old-key",
        started_at=10.0,
    )
    new_record = SessionRecord(
        pid=102,
        kind="tui",
        session_id="root-session",
        conversation_name="root",
        cwd="/tmp",
        model_label="new",
        control_port=4102,
        control_key="new-key",
        started_at=20.0,
    )
    # A live SSE subscriber is the route owner that intentionally keeps the
    # generation ledger alive past payload eviction (the F1 scenario).
    daemon.table.session_subscribers["root-session"] = {asyncio.Queue()}

    # High epoch, then a low-version replacement owner supersedes it.
    high = SessionProjection(session_id="root-session", pid=101, version=50)
    assert daemon.capture_subagent_details(high, record=old_record).version == 50
    replacement = SessionProjection(session_id="root-session", pid=102, version=1)
    assert daemon.capture_subagent_details(replacement, record=new_record).version == 51

    # Evict the route payload under pressure; the subscriber keeps the ledger.
    for index in range(MAX_RETAINED_SESSION_PROJECTIONS):
        daemon.capture_subagent_details(
            SessionProjection(session_id=f"pressure-{index:03d}", pid=200 + index, version=1)
        )
    assert "root-session" not in daemon.session_projections
    assert daemon._projection_generations["root-session"].epoch == 51

    # Durable detail/history now rebuild instead of fencing to a 500, and the
    # rematerialized payload carries the retained monotonic epoch.
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    detail = client.get("/api/sessions/root-session/agents/child-job")
    assert detail.status_code == 200
    assert detail.json()["prompt"] == "durable prompt"
    assert detail.json()["version"] == 51
    history = client.get(
        "/api/sessions/root-session/agents/child-job/history", params={"limit": 10}
    )
    assert history.status_code == 200
    assert [row["id"] for row in history.json()["entries"]] == ["child-row"]

    # A genuine late frame from the OLD process is still fenced: reconstruction
    # rebuilt the payload but never reopened the superseded generation.
    late_old = SessionProjection(session_id="root-session", pid=101, version=999)
    assert (
        daemon.capture_subagent_details(late_old, record=old_record)
        is daemon.session_projections["root-session"]
    )
    assert daemon.session_projections["root-session"].version == 51


def test_subagent_detail_merge_accepts_lifecycle_updates_and_terminal_clearing() -> None:
    daemon = MobileDaemon(port=0, password="pw123")
    first = SessionProjection(session_id="root-session", pid=9, version=2)
    first.subagents = [
        SubagentRow(
            job_id="child-job",
            label="child",
            status="failed",
            error_text="first failure",
            prompt="original prompt",
            transcript=[TranscriptEntry(id="old", kind="assistant", text="old reply")],
        )
    ]
    daemon.capture_subagent_details(first)

    resumed = SessionProjection(session_id="root-session", pid=9, version=3)
    resumed.subagents = [
        SubagentRow(
            job_id="child-job",
            label="child renamed",
            status="running",
            progress="trying again",
            transcript=[TranscriptEntry(id="new", kind="assistant", text="new reply")],
        )
    ]
    daemon.capture_subagent_details(resumed)
    detail = daemon.subagent_details[("root-session", "child-job")]
    assert detail["version"] == 3
    assert detail["label"] == "child renamed"
    assert detail["status"] == "running"
    assert detail["progress"] == "trying again"
    assert detail["error_text"] == ""
    assert detail["prompt"] == "original prompt"
    assert [row["id"] for row in detail["transcript"]] == ["new"]

    completed = SessionProjection(session_id="root-session", pid=9, version=4)
    completed.subagents = [
        SubagentRow(job_id="child-job", label="child renamed", status="completed")
    ]
    daemon.capture_subagent_details(completed)
    detail = daemon.subagent_details[("root-session", "child-job")]
    assert detail["version"] == 4
    assert detail["status"] == "completed"
    assert detail["result_text"] == ""
    assert detail["error_text"] == ""
    assert [row["id"] for row in detail["transcript"]] == ["new"]


def test_every_published_subagent_resolves_beyond_legacy_256_limit() -> None:
    """A rendered roster row must never lead to a deterministic detail 404."""
    daemon = MobileDaemon(port=0, password="pw123")
    projection = SessionProjection(session_id="root-session", pid=9, version=11)
    projection.subagents = [
        SubagentRow(job_id=f"job-{index:03d}", label=f"child {index}") for index in range(300)
    ]
    daemon.capture_subagent_details(projection)

    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    assert len(projection.subagents) == 300
    assert client.get("/api/sessions/root-session/agents/job-000").status_code == 200
    assert client.get("/api/sessions/root-session/agents/job-256").status_code == 200
    assert client.get("/api/sessions/root-session/agents/job-299").status_code == 200


def test_retained_projection_routes_survive_more_than_16_root_sessions() -> None:
    """Projection and detail ownership cannot diverge at the old cache boundary."""
    daemon = MobileDaemon(port=0, password="pw123")
    for index in range(20):
        projection = SessionProjection(session_id=f"root-{index:02d}", pid=index, version=index)
        projection.subagents = [
            SubagentRow(job_id="child", label=f"child {index}", session_id=f"child-{index:02d}")
        ]
        daemon.capture_subagent_details(projection)

    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    assert "root-00" in daemon.session_projections
    assert "root-19" in daemon.session_projections
    assert client.get("/api/sessions/root-00/agents/child").status_code == 200
    assert client.get("/api/sessions/root-19/agents/child").status_code == 200


def test_projection_and_detail_evict_as_one_bounded_unit() -> None:
    """The only allowed dead route is one no retained projection advertises."""
    from local_operator.mobile.daemon import MAX_RETAINED_SESSION_PROJECTIONS

    daemon = MobileDaemon(port=0, password="pw123")
    for index in range(MAX_RETAINED_SESSION_PROJECTIONS + 1):
        projection = SessionProjection(session_id=f"root-{index:03d}", pid=index)
        projection.subagents = [SubagentRow(job_id="child", label="child")]
        daemon.capture_subagent_details(projection)

    assert len(daemon.session_projections) == MAX_RETAINED_SESSION_PROJECTIONS
    # Unowned generation entries leave with their payload cache unit; active or
    # subscribed routes are the only entries allowed to outlive this bound.
    assert len(daemon._projection_generations) == MAX_RETAINED_SESSION_PROJECTIONS
    assert "root-000" not in daemon.session_projections
    assert "root-000" not in daemon._projection_generations
    assert ("root-000", "child") not in daemon.subagent_details
    for session_id, projection in daemon.session_projections.items():
        assert all(
            (session_id, row.job_id) in daemon.subagent_details for row in projection.subagents
        )


def test_durable_subagent_routes_reconstruct_after_daemon_restart(tmp_path, monkeypatch) -> None:
    """Restart/reconnect rebuilds detail and child history from durable lineage."""
    from local_operator.harness.types import Message
    from local_operator.session.session import SUBAGENT_ROSTER_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    root_dir = cfg / "sessions" / "root-session"
    child_dir = cfg / "sessions" / "child-session"
    root_dir.mkdir(parents=True)
    child_dir.mkdir(parents=True)
    (child_dir / "origin.json").write_text('{"origin":"subagent"}')
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    asyncio.run(Transcript(root_dir).append_message(Message.user("root", id="root-row")))
    asyncio.run(Transcript(child_dir).append_message(Message.user("oldest", id="child-oldest")))
    asyncio.run(
        Transcript(child_dir).append_message(Message.assistant("newest", id="child-newest"))
    )
    asyncio.run(
        Transcript(root_dir).append_custom(
            SUBAGENT_ROSTER_CUSTOM_TYPE,
            {
                "jobs": [{"id": "child-job", "status": "completed", "label": "child"}],
                "records": [
                    {
                        "job_id": "child-job",
                        "label": "child",
                        "prompt": "inspect durable state",
                        "session_dir": str(child_dir),
                        "outcome": "completed",
                        "result_text": "done",
                    }
                ],
            },
        )
    )

    daemon = MobileDaemon(port=0, password="pw123")
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    detail = client.get("/api/sessions/root-session/agents/child-job")
    assert detail.status_code == 200
    assert detail.json()["prompt"] == "inspect durable state"
    # The detail payload no longer embeds the child transcript — it is fetched
    # lazily from the /history route below so a full repaint never carries an
    # unbounded child transcript past the daemon's 1 MB control-frame limit.
    assert detail.json()["transcript"] == []
    history = client.get("/api/sessions/root-session/agents/child-job/history", params={"limit": 1})
    assert history.status_code == 200
    assert [row["id"] for row in history.json()["entries"]] == ["child-newest"]
    # The full child transcript is still reachable through paging.
    full = client.get("/api/sessions/root-session/agents/child-job/history", params={"limit": 10})
    assert [row["id"] for row in full.json()["entries"]] == ["child-oldest", "child-newest"]

    restarted = MobileDaemon(port=0, password="pw123")
    restarted_client = TestClient(build_app(restarted), follow_redirects=False)
    restarted_client.post("/login", data={"password": "pw123"})
    assert restarted_client.get("/api/sessions/root-session/agents/child-job").status_code == 200
    assert (
        restarted_client.get(
            "/api/sessions/root-session/agents/child-job/history",
            params={"before": "child-newest", "limit": 1},
        ).json()["entries"][0]["id"]
        == "child-oldest"
    )


def test_durable_fold_bounds_prompt_and_outcome_on_the_wire(tmp_path, monkeypatch) -> None:
    """A durable rebuild must not reintroduce the unbounded frame the live caps
    prevent. ``_durable_projection`` rebuilds a roster from the persisted record,
    whose prompt/result/error fields are unbounded on disk; the wire row must
    compact them the same way the live fold does, or a restart/reconnect of a
    deep-roster session re-wedges with the identical oversized-frame symptom.
    """
    from local_operator.harness.types import Message
    from local_operator.mobile.daemon import _durable_projection
    from local_operator.mobile.projection import (
        SUBAGENT_OUTCOME_CHARS,
        SUBAGENT_PROMPT_PREVIEW_CHARS,
    )
    from local_operator.session.session import SUBAGENT_ROSTER_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    root_dir = cfg / "sessions" / "root-session"
    child_dir = cfg / "sessions" / "child-session"
    root_dir.mkdir(parents=True)
    child_dir.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    asyncio.run(Transcript(root_dir).append_message(Message.user("root", id="root-row")))
    asyncio.run(
        Transcript(root_dir).append_custom(
            SUBAGENT_ROSTER_CUSTOM_TYPE,
            {
                "jobs": [{"id": "child-job", "status": "completed", "label": "child"}],
                "records": [
                    {
                        "job_id": "child-job",
                        "label": "child",
                        "prompt": "P" * 50_000,
                        "session_dir": str(child_dir),
                        "outcome": "completed",
                        "result_text": "R" * 50_000,
                    }
                ],
            },
        )
    )

    projection = _durable_projection("root-session")
    assert projection is not None
    row = projection.subagents[0]
    assert len(row.prompt) <= SUBAGENT_PROMPT_PREVIEW_CHARS
    assert len(row.result_text) <= SUBAGENT_OUTCOME_CHARS


def test_durable_fold_keeps_failed_child_error_text_generous(tmp_path, monkeypatch) -> None:
    """A FAILED child's ``error_text`` must survive to the phone in full.

    ``error_text`` is ``str(exc)`` from the parent runner and is NEVER written
    into the child transcript, so the phone's lazy /history fetch cannot recover
    it: the durable wire value is the only copy the Outcome panel can render.
    Capping it at the 200-char ``result_text`` preview would truncate the
    failure tail everywhere on the phone with no recovery path (F1), so the
    durable fold must carry it generously (``SUBAGENT_ERROR_CHARS``). This pins
    that the error tail is not clipped to the result preview length, and that a
    multi-line trace keeps its line breaks.
    """
    from local_operator.harness.types import Message
    from local_operator.mobile.daemon import _durable_projection
    from local_operator.mobile.projection import (
        SUBAGENT_ERROR_CHARS,
        SUBAGENT_OUTCOME_CHARS,
    )
    from local_operator.session.session import SUBAGENT_ROSTER_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    root_dir = cfg / "sessions" / "root-session"
    child_dir = cfg / "sessions" / "child-session"
    root_dir.mkdir(parents=True)
    child_dir.mkdir(parents=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    # A realistic multi-line provider failure well past the 200-char result cap.
    error = "Traceback (most recent call last):\n" + "\n".join(
        f"  frame {i}: boom in module_{i}" for i in range(60)
    )
    assert len(error) > SUBAGENT_OUTCOME_CHARS
    asyncio.run(Transcript(root_dir).append_message(Message.user("root", id="root-row")))
    asyncio.run(
        Transcript(root_dir).append_custom(
            SUBAGENT_ROSTER_CUSTOM_TYPE,
            {
                "jobs": [{"id": "child-job", "status": "failed", "label": "child"}],
                "records": [
                    {
                        "job_id": "child-job",
                        "label": "child",
                        "session_dir": str(child_dir),
                        "outcome": "failed",
                        "error_text": error,
                    }
                ],
            },
        )
    )

    projection = _durable_projection("root-session")
    assert projection is not None
    row = projection.subagents[0]
    assert row.status == "failed"
    # NOT truncated to the 200-char result preview: the whole failure tail rides.
    assert len(row.error_text) > SUBAGENT_OUTCOME_CHARS
    assert len(row.error_text) <= SUBAGENT_ERROR_CHARS
    assert row.error_text.count("\n") > 1  # multi-line structure preserved
    assert "frame 59" in row.error_text  # the tail survives, not just the head


def test_http_command_requires_auth_and_rejects_empty_steer_before_dispatch() -> None:
    daemon = MobileDaemon(port=0, password="pw123")
    record = SessionRecord(
        pid=123,
        kind="tui",
        session_id="root-session",
        conversation_name="root",
        cwd="/tmp",
        model_label="fixture",
        control_port=1,
        control_key="fixture",
    )
    daemon.table.entries[record.pid] = SessionEntry(record)
    client = TestClient(build_app(daemon), follow_redirects=False)
    payload = {
        "op": "steer",
        "command_id": "12345678-1234-4678-9234-567812345678",
        "text": "parent instruction",
    }
    unauthorized = client.post("/api/sessions/root-session/command", json=payload)
    assert unauthorized.status_code == 401

    client.post("/login", data={"password": "pw123"})
    empty = client.post(
        "/api/sessions/root-session/command",
        json={**payload, "text": "   "},
    )
    assert empty.status_code == 422
    assert empty.json() == {"error": "text must be a non-empty string"}


def test_unknown_session_command_is_a_409() -> None:
    daemon = MobileDaemon(port=0, password="pw123")
    app = build_app(daemon)
    client = TestClient(app, follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    reply = client.post("/api/sessions/424242/command", json={"op": "abort"})
    assert reply.status_code == 409
    # A prompt to an unknown/malformed route must not reach continuation child
    # construction; only a proven durable user conversation can wake a host.
    prompt = {"op": "prompt", "command_id": "unknown-1", "text": "hello"}
    assert client.post("/api/sessions/424242/command", json=prompt).status_code == 409
    assert client.post("/api/sessions/%2E%2E%2Foutside/command", json=prompt).status_code in (
        404,
        409,
    )


def test_spawn_dir_gate_allows_home_and_tmp_only(tmp_path, monkeypatch) -> None:
    """The phone may start a session anywhere under home or in the system temp
    dir (a common scratch root), and nowhere else — the gate guards against a
    fat-fingered/traversed path, not against the owner."""
    from pathlib import Path

    from local_operator.mobile import daemon as daemon_mod
    from local_operator.mobile.daemon import _spawn_dir_allowed

    # pytest's tmp_path already lives UNDER the real system temp dir, so pin an
    # explicit fake tmp and home under it and assert against those bounds —
    # otherwise "outside" would still be a child of the real /tmp and allowed.
    home = tmp_path / "home"
    home.mkdir()
    fake_tmp = tmp_path / "scratch"
    fake_tmp.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr(daemon_mod, "_tmp_dir", lambda: str(fake_tmp.resolve()))

    # Under home: allowed. Home itself: allowed.
    sub = home / "projects"
    sub.mkdir()
    assert _spawn_dir_allowed(home.resolve())
    assert _spawn_dir_allowed(sub.resolve())

    # The (fake) temp dir and a child of it: allowed.
    assert _spawn_dir_allowed(fake_tmp.resolve())
    child = fake_tmp / "work"
    child.mkdir()
    assert _spawn_dir_allowed(child.resolve())

    # Somewhere neither under home nor tmp: refused.
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert not _spawn_dir_allowed(outside.resolve())


def test_directories_endpoint_offers_tmp(monkeypatch) -> None:
    daemon = MobileDaemon(port=0, password="pw123")
    app = build_app(daemon)
    client = TestClient(app, follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    body = client.get("/api/directories").json()
    assert "home" in body
    assert "recent" in body
    # /tmp is offered as a scratch start dir beside home.
    assert body.get("tmp")


def test_previous_command_validation_is_bounded_without_side_effects(tmp_path, monkeypatch) -> None:
    """Malformed authenticated continuation input never reaches child startup."""
    import asyncio as _asyncio

    from local_operator.harness.types import Message
    from local_operator.mobile import attach_client
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    directory = cfg / "sessions" / "previous-invalid"
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    _asyncio.run(transcript.append_message(Message.user("existing", id="existing")))
    called = False

    async def should_not_start(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal called
        called = True
        raise AssertionError("invalid input spawned a continuation")

    monkeypatch.setattr(attach_client, "continue_command", should_not_start)
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")))
    client.post("/login", data={"password": "pw123"})
    invalid = [
        {"op": "prompt", "command_id": "not-a-uuid", "text": "hello"},
        {"op": "prompt", "text": "hello"},
        {"op": "prompt", "command_id": "12345678-1234-5678-1234-567812345678", "text": []},
        {
            "op": "prompt",
            "command_id": "12345678-1234-5678-1234-567812345678",
            "text": "hello",
            "images": {},
        },
    ]
    for payload in invalid:
        response = client.post("/api/sessions/previous-invalid/command", json=payload)
        assert response.status_code in (400, 422)
        assert response.headers["content-type"].startswith("application/json")
        assert "error" in response.json()
    assert not called
    assert [message.id for message in transcript.build_llm_history()] == ["existing"]


def test_failed_wake_recapture_preserves_cached_child_detail(tmp_path, monkeypatch) -> None:
    """A retained summary may be republished before wake construction fails."""
    import asyncio as _asyncio

    from local_operator.harness.types import Message
    from local_operator.mobile import attach_client
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    directory = cfg / "sessions" / "previous-rich"
    directory.mkdir(parents=True)
    _asyncio.run(Transcript(directory).append_message(Message.user("existing", id="existing")))

    daemon = MobileDaemon(port=0, password="pw123")
    projection = SessionProjection(session_id="previous-rich", pid=9, version=7, ended=True)
    projection.subagents = [
        SubagentRow(
            job_id="child-job",
            label="child",
            prompt="secret prompt",
            result_text="full result",
            transcript=[TranscriptEntry(id="child-row", kind="assistant", text="full reply")],
            todos=[TodoPhase(name="Work", items=[TodoItem(text="verify", status="done")])],
        )
    ]
    daemon.capture_subagent_details(projection)

    async def fail(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise ConnectionError("daemon restarted")

    monkeypatch.setattr(attach_client, "continue_command", fail)
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    response = client.post(
        "/api/sessions/previous-rich/command",
        json={
            "op": "prompt",
            "command_id": "12345678-1234-5678-1234-567812345678",
            "text": "wake again",
        },
    )
    assert response.status_code == 502
    detail = client.get("/api/sessions/previous-rich/agents/child-job")
    assert detail.status_code == 200
    assert detail.json()["prompt"] == "secret prompt"
    assert detail.json()["result_text"] == "full result"
    assert detail.json()["transcript"][0]["text"] == "full reply"
    assert detail.json()["todos"][0]["items"][0]["text"] == "verify"
    assert daemon.session_projections["previous-rich"].ended is False


def test_previous_continuation_transport_failure_is_non_2xx(tmp_path, monkeypatch) -> None:
    """Provider/child/socket failures return an error and never a false ACK."""
    import asyncio as _asyncio

    from local_operator.harness.types import Message
    from local_operator.mobile import attach_client
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    directory = cfg / "sessions" / "previous-failure"
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    _asyncio.run(transcript.append_message(Message.user("existing", id="existing")))

    async def fail(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise ConnectionError("daemon restarted")

    monkeypatch.setattr(attach_client, "continue_command", fail)
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")))
    client.post("/login", data={"password": "pw123"})
    response = client.post(
        "/api/sessions/previous-failure/command",
        json={
            "op": "prompt",
            "command_id": "12345678-1234-5678-1234-567812345678",
            "text": "retry me",
        },
    )
    assert response.status_code == 502
    assert response.json() == {"error": "daemon restarted"}
    history = transcript.build_llm_history()
    assert [message.id for message in history] == ["existing"]


def test_previous_history_pages_full_durable_transcript_once(tmp_path, monkeypatch) -> None:
    """A Previous route pages beyond the projection cap without a live host.

    The first request anchors at the retained tail's oldest row, then each
    cursor walks backwards. Concatenating the pages with that tail must recover
    every folded row exactly once and in chronological order.
    """
    import asyncio as _asyncio

    from local_operator.harness.types import Message
    from local_operator.mobile.daemon import _durable_projection
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    session_id = "durable-history"
    directory = cfg / "sessions" / session_id
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    expected_ids: list[str] = []
    for turn in range(PROJECTION_TRANSCRIPT_LIMIT + 25):
        user = Message.user(f"user {turn}", id=f"u-{turn:03d}")
        assistant = Message.assistant(f"answer {turn}", id=f"a-{turn:03d}")
        _asyncio.run(transcript.append_message(user))
        _asyncio.run(transcript.append_message(assistant))
        expected_ids.extend([user.id, assistant.id])

    daemon = MobileDaemon(port=0, password="pw123")
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    projection = _durable_projection(session_id)
    assert projection is not None
    tail_ids = [entry.id for entry in projection.transcript]
    assert len(tail_ids) == PROJECTION_TRANSCRIPT_LIMIT

    pages: list[list[str]] = []
    # The projection pins the opener at index 0 and keeps the chronological tail
    # after it; the web view therefore anchors history at the first tail row.
    before = tail_ids[1]
    while True:
        response = client.get(
            f"/api/sessions/{session_id}/history", params={"before": before, "limit": 17}
        )
        assert response.status_code == 200
        body = response.json()
        page = [entry["id"] for entry in body["entries"]]
        pages.insert(0, page)
        if not body["has_more"]:
            break
        assert page
        before = page[0]

    recovered = [entry_id for page in pages for entry_id in page] + tail_ids[1:]
    # The pinned opener overlaps the oldest page and is de-duplicated by the
    # web merge, exactly as it is here.
    recovered = list(dict.fromkeys([tail_ids[0], *recovered]))
    assert recovered == expected_ids
    assert len(recovered) == len(set(recovered))


def test_previous_history_rejects_unknown_traversal_and_subagent(tmp_path, monkeypatch) -> None:
    """Durability does not broaden the route beyond human-owned sessions."""
    import asyncio as _asyncio

    from local_operator.harness.types import Message
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    subagent = cfg / "sessions" / "subagent-session"
    subagent.mkdir(parents=True)
    _asyncio.run(Transcript(subagent).append_message(Message.user("hidden", id="hidden")))
    (subagent / "origin.json").write_text('{"origin":"subagent"}')

    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")))
    client.post("/login", data={"password": "pw123"})
    assert client.get("/api/sessions/unknown/history").status_code == 404
    assert client.get("/api/sessions/subagent-session/history").status_code == 404
    # Encoded slash must never turn the public identifier into a filesystem path.
    assert client.get("/api/sessions/%2E%2E%2Foutside/history").status_code in (404, 400)


def test_image_bytes_reads_attachment_from_transcript(tmp_path, monkeypatch) -> None:
    """The image endpoint's helper decodes the Nth image block of a message
    from the on-disk transcript — the lazy source the phone fetches pixels
    from. Index counts IMAGE blocks only, matching _image_refs."""
    import asyncio as _asyncio
    import base64

    from local_operator.harness.types import ImageContent, Message
    from local_operator.mobile.daemon import _image_bytes
    from local_operator.mobile.types import SessionRecord
    from local_operator.session.transcript import Transcript

    cfg = tmp_path / "config"
    cfg.mkdir()
    # _image_bytes imports config_dir from local_operator.paths at call time,
    # so patching the source module is what redirects it to the fake config.
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    session_id = "sess-img"
    directory = cfg / "sessions" / session_id
    directory.mkdir(parents=True)
    raw = b"\x89PNG\r\n\x1a\nHELLO"
    message = Message.user(
        "look",
        [ImageContent(data=base64.b64encode(raw).decode(), mime_type="image/png")],
    )
    transcript = Transcript(directory)
    _asyncio.run(transcript.append_message(message))

    record = SessionRecord(
        pid=1,
        kind="daemon",
        session_id=session_id,
        conversation_name="",
        cwd=str(tmp_path),
        model_label="",
        control_port=0,
        control_key="k",
    )
    found = _image_bytes(record, message.id, 0)
    assert found is not None
    data, mime = found
    assert data == raw
    assert mime == "image/png"

    # Out-of-range index and unknown entry both miss cleanly.
    assert _image_bytes(record, message.id, 1) is None
    assert _image_bytes(record, "nope", 0) is None


def test_slash_catalogue_excludes_terminal_chrome() -> None:
    daemon = MobileDaemon(port=0, password="pw123")
    names = [c["name"] for c in daemon.slash_commands()]
    assert "model" in names
    assert "effort" in names
    assert "resume" in names
    # TUI chrome never leaves the terminal.
    assert "exit" not in names
    assert "quit" not in names
    assert "clear" not in names


def test_oversized_control_frames_report_the_rate_not_each_frame(caplog, monkeypatch) -> None:
    """One line per skipped frame buried the relay log; the RATE is the signal.

    Field measurement on the operator's machine: 6,104,351
    ``oversized control frame`` records were 78% of a 420 MB ``mobile.log``, and
    the records that mattered (a stalled runtime, the MCP client, the schedule)
    were unreadable past them. Every skipped frame does cost a session its live
    projection, so the count must not be lost — only its per-frame line.

    A single frame is still reported on its own, because a one-off is a real
    event and a reader should see it immediately. A flood collapses to one line
    per window carrying the window count and a monotonic total, so a producer
    that regresses and then stops is still accounted for by the next line.
    """
    import logging

    from local_operator.mobile import daemon as daemon_module

    clock = {"now": 1_000.0}
    monkeypatch.setattr(daemon_module.time, "monotonic", lambda: clock["now"])
    counter = daemon_module._OversizedControlFrames()

    with caplog.at_level(logging.WARNING, logger=daemon_module.logger.name):
        counter.note(7742)
        assert "first oversized control frame from pid 7742" in caplog.text

        caplog.clear()
        for _ in range(5_000):
            counter.note(7742)
        assert caplog.text == "", "a flood must not be one log line per frame"

        clock["now"] += daemon_module.OVERSIZED_CONTROL_WINDOW_S + 0.1
        counter.note(7742)
        assert caplog.text.count("oversized control frame") == 1, caplog.text
        assert "5001 oversized control frames from pid 7742" in caplog.text
        assert "5002 this process" in caplog.text

        # A different session's flood is accounted for separately, so one noisy
        # child cannot hide another's count.
        caplog.clear()
        counter.note(9911)
        assert "first oversized control frame from pid 9911" in caplog.text


def test_the_phone_list_carries_the_drain_so_its_row_can_say_it() -> None:
    """UX round 2, U8: a signalled runtime also STREAMS, so the list said "busy".

    The phone's row ladder is driven by ``streaming``, which is true of a
    draining runtime exactly as it is of an ordinary working one — so the list
    the operator reads on a phone could not tell "finishing a turn somebody
    asked it to finish" from "working". The record's phrase is carried as its
    own additive field; a client that does not know it renders exactly as
    before, which is what makes this safe to ship before the card's own
    treatment of it.
    """
    from local_operator.session.runtime.types import LEAVING_ON_SIGNAL, SessionRecord

    def record(session_id: str = "s-drain", pid: int = 4321, **extra: Any) -> SessionRecord:
        return SessionRecord(
            pid=pid,
            kind="tui",
            session_id=session_id,
            conversation_name="draining",
            cwd="/tmp",
            model_label="test/model",
            control_port=1,
            control_key="k",
            **extra,
        )

    daemon = MobileDaemon(port=0, password="pw")
    draining = record(leaving=LEAVING_ON_SIGNAL, busy=True)
    daemon.table.entries[draining.pid] = SessionEntry(draining)
    rows = daemon.table._merge_summaries({})
    assert rows[0]["leaving"] == LEAVING_ON_SIGNAL, rows[0]

    # An ordinary busy session carries nothing, so the field means something.
    plain = record("s-plain", pid=4322, busy=True)
    daemon.table.entries[plain.pid] = SessionEntry(plain)
    daemon.table.entries.pop(draining.pid)
    rows = daemon.table._merge_summaries({})
    assert rows[0]["leaving"] == "", rows[0]

    # And a record written by an OLDER runtime (no such field) is empty rather
    # than missing: this list is served on a host mid-upgrade.
    old = record("s-old", pid=4323)
    del old.leaving  # type: ignore[attr-defined]
    daemon.table.entries.pop(plain.pid)
    daemon.table.entries[old.pid] = SessionEntry(old)
    rows = daemon.table._merge_summaries({})
    assert rows[0]["leaving"] == "", rows[0]


# --- the phone's listing: membership, so an unreadable store is not an empty one --


def _listing_rows(cfg, *session_ids: str):
    """One user-visible session per id, with the activity the scan requires.

    Built through the catalogue suite's own helpers so this file and that one
    agree about what a listable session IS, rather than this one growing a
    second opinion about it.
    """
    from tests.unit.session.test_catalog_read_failures import _store

    return _store(cfg, *session_ids)


@pytest.mark.asyncio
async def test_a_phone_listing_keeps_the_rows_it_read_when_the_store_goes_unreadable(
    tmp_path, monkeypatch
) -> None:
    """The wipe this repairs, on the surface whose client replaces the whole list.

    The phone's home screen does ``sessions = payload.sessions`` — MEMBERSHIP,
    not a merge — so a durable half that answers "zero conversations" for a
    store it could not read empties the operator's history with nothing to say
    why. The rows it already had are the true answer; they are what this serves.

    The failure is injected at the REAL seam (``os.scandir`` of the store), and
    the second read is forced through ``invalidate_summaries_cache`` — the path
    a structural change takes — because that is the state where the fresh cache
    is gone and only the last listing that was actually read can answer.
    """
    import errno
    import os

    from tests.unit.session.test_catalog_read_failures import _failing_open

    cfg = tmp_path / "config"
    store = _listing_rows(cfg, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    real_scandir = os.scandir

    table = SessionTable()
    first = await table.summaries()
    assert {row["session_id"] for row in first} == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert table.listing_degraded() == []

    _failing_open(monkeypatch, store, OSError(errno.EIO, "Input/output error"))
    table.invalidate_summaries_cache()
    second = await table.summaries()

    assert {row["session_id"] for row in second} == {
        "aaaaaaaaaaaa",
        "bbbbbbbbbbbb",
    }, "an unreadable store must not be published as an empty conversation list"
    assert table.listing_degraded() == ["sessions"]

    # And the healing is real: once the read works again the marker clears, so a
    # client keyed on it cannot latch a stale "couldn't refresh".
    #
    # The seam is healed by hand rather than with ``monkeypatch.undo()``: undo
    # drops EVERY patch on this fixture, including the config-dir isolation this
    # suite runs under, and the next read would then walk the operator's real
    # store.
    monkeypatch.setattr(os, "scandir", real_scandir)
    table.invalidate_summaries_cache()
    third = await table.summaries()
    assert {row["session_id"] for row in third} == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert table.listing_degraded() == []


@pytest.mark.asyncio
async def test_a_cold_listing_against_an_unreadable_store_says_so(tmp_path, monkeypatch) -> None:
    """Nothing to serve is still not a verdict about the operator's conversations.

    A daemon that has never read the store has no rows to keep, so the frame is
    empty — but the marker rides with it, which is what lets a client say
    "couldn't read" instead of rendering "no conversations" over a read that
    never happened.
    """
    import errno

    from tests.unit.session.test_catalog_read_failures import _failing_open

    cfg = tmp_path / "config"
    store = _listing_rows(cfg, "aaaaaaaaaaaa")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    _failing_open(monkeypatch, store, OSError(errno.EACCES, "Permission denied"))

    table = SessionTable()
    rows = await table.summaries()

    assert rows == []
    assert table.listing_degraded() == ["sessions"]


@pytest.mark.asyncio
async def test_the_store_failure_is_a_ttl_paced_retry_not_a_rescan_per_repaint(
    tmp_path, monkeypatch
) -> None:
    """A store that stays unreadable must not be rescanned on every repaint.

    The live-projection push path repaints the list ~30x/s. If a FAILED attempt
    left the timestamp unset, every one of those would re-walk a store that
    just failed to walk, on the single daemon loop — the starvation the TTL
    cache exists to prevent, reintroduced by the error path.
    """
    import errno

    from local_operator import resume as resume_module
    from tests.unit.session.test_catalog_read_failures import _failing_open

    cfg = tmp_path / "config"
    store = _listing_rows(cfg, "aaaaaaaaaaaa")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    _failing_open(monkeypatch, store, OSError(errno.EIO, "Input/output error"))

    calls = {"n": 0}
    real_rows = resume_module.recent_session_rows

    def counting_rows(directory, limit=None, *, strict=False):
        calls["n"] += 1
        return real_rows(directory, limit, strict=strict)

    monkeypatch.setattr(resume_module, "recent_session_rows", counting_rows)

    table = SessionTable()
    await table.summaries()
    assert calls["n"] == 1
    calls["n"] = 0
    for _ in range(30):
        table.notify_list_changed()
        await table.summaries()
    assert calls["n"] == 0, "a failed read must back off for the TTL, not retry per repaint"


def test_the_listing_route_publishes_the_marker_beside_the_rows(tmp_path, monkeypatch) -> None:
    """The wire half: the marker and the rows travel together, on both transports.

    ``/api/sessions`` and the ``sessions`` event frame are built by ONE function
    (``_list_frame``) because the phone's home screen reads the SSE one, and a
    marker present on only one of two spellings of the same answer is a marker
    that screen never sees. Pinned on the JSON route, which is the same payload.
    """
    import errno

    from tests.unit.session.test_catalog_read_failures import _failing_open

    cfg = tmp_path / "config"
    store = _listing_rows(cfg, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    daemon = MobileDaemon(port=0, password="pw123")
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})

    healthy = client.get("/api/sessions").json()
    assert {row["session_id"] for row in healthy["sessions"]} == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert healthy["degraded"] == []

    _failing_open(monkeypatch, store, OSError(errno.EMFILE, "Too many open files"))
    daemon.table.invalidate_summaries_cache()
    broken = client.get("/api/sessions").json()

    assert {row["session_id"] for row in broken["sessions"]} == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert broken["degraded"] == ["sessions"]


@pytest.mark.asyncio
async def test_a_deferred_attention_read_degrades_the_listing_instead_of_raising(
    tmp_path, monkeypatch
) -> None:
    """The read the incident's own chain ends in, at the call site that had no arm.

    ``SessionTable._build`` reads the attention store AFTER the durable scan, and
    that read was the one listing read in the tree without an ``except`` (round-2
    review MINOR-A / round-2 QA Q-1): a lock that outlasted the store's retry
    budget raised ``AttentionReadDeferred`` straight out of ``summaries``, so the
    phone's ``GET /api/sessions`` answered 500 while its neighbours degraded. The
    arms asserted here are what make it answer: the rows are kept, their last
    TRUE marks are kept, and the failure is named in the list the marker already
    travels in.

    Injected at the CLASS, which pins THIS call site rather than the store: an
    edit that removes the ``except`` re-raises out of ``summaries`` and fails
    here. The store's own retry and its typed verdict are pinned separately in
    ``tests/unit/session/test_attention_lock_contention.py``.
    """
    from local_operator.session.attention import AttentionReadDeferred, AttentionStore

    cfg = tmp_path / "config"
    _listing_rows(cfg, "aaaaaaaaaaaa")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    # A REAL unread completion, so "the marks were kept" below is about a mark
    # that existed rather than about a default that happens to match one.
    AttentionStore().publish("session/aaaaaaaaaaaa", str(uuid.uuid4()), "answer", "complete")

    table = SessionTable()
    healthy = await table.summaries()
    assert [row["unseen"] for row in healthy] == [True], healthy
    assert table.listing_degraded() == []

    real_state_many = AttentionStore.state_many

    def defers(_store, _conversations):
        raise AttentionReadDeferred("attention store stayed busy through 2 attempts")

    monkeypatch.setattr(AttentionStore, "state_many", defers)
    table.invalidate_summaries_cache()
    degraded = await table.summaries()

    assert [row["session_id"] for row in degraded] == ["aaaaaaaaaaaa"]
    assert [row["unseen"] for row in degraded] == [True], (
        "a deferred read must keep the last TRUE marks: every completion reading "
        "as already seen is the confident negative this store exists to stop"
    )
    assert table.listing_degraded() == ["attention"]

    # And the marker does not latch — the next successful read clears it, which
    # is what lets a client's "couldn't refresh" go away on its own. The seam is
    # healed by hand rather than with ``monkeypatch.undo()``, which would drop
    # the config-dir isolation patch too and walk the operator's real store.
    monkeypatch.setattr(AttentionStore, "state_many", real_state_many)
    table.invalidate_summaries_cache()
    healed = await table.summaries()
    assert [row["unseen"] for row in healed] == [True]
    assert table.listing_degraded() == []


def test_the_listing_route_serves_a_deferred_attention_read(tmp_path, monkeypatch) -> None:
    """The wire half: past the budget the list route degrades instead of 500ing.

    ``GET /api/sessions`` answered 500 @10.8 s on the previous head once a lock
    outlasted the store's retry budget, because ``_list_frame`` -> ``summaries``
    let ``AttentionReadDeferred`` escape to uvicorn. A route may not lose the
    rows it DID read to a read behind them: it answers 200 with those rows and
    names the decoration, the shape the durable half already uses on this frame.
    """
    from local_operator.session.attention import AttentionReadDeferred, AttentionStore

    cfg = tmp_path / "config"
    _listing_rows(cfg, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    daemon = MobileDaemon(port=0, password="pw123")
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})

    healthy = client.get("/api/sessions").json()
    assert healthy["degraded"] == []

    def defers(_store, _conversations):
        raise AttentionReadDeferred("attention store stayed busy through 2 attempts")

    monkeypatch.setattr(AttentionStore, "state_many", defers)
    daemon.table.invalidate_summaries_cache()
    response = client.get("/api/sessions")

    assert response.status_code == 200
    body = response.json()
    assert {row["session_id"] for row in body["sessions"]} == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert body["degraded"] == ["attention"]


def test_the_history_route_names_the_store_it_could_not_read(tmp_path, monkeypatch) -> None:
    """The phone's other conversation list, and the same rule.

    ``/api/sessions/past`` is a second listing of the operator's conversations,
    so a store it cannot walk may not reach it as "there are none" either. Its
    previous shape was worse than the home listing's: ``except Exception:
    return []`` laundered EVERY failure into an empty history, bugs included.

    Whose list this is, stated correctly because the first draft of this
    docstring was not: the shipped history screen is ``mobile/web/src/screens/
    past-sessions.tsx`` and it renders ``searchSessions(query)``, i.e.
    ``/api/sessions/search`` — pinned separately below. ``api.ts``'s
    ``getPastSessions`` helper has no call site anywhere in the tree, so this
    route is a listing with no shipped renderer of its own; it is still a
    listing, and the rule is about the wire, not about who reads it today.
    """
    import errno

    from tests.unit.session.test_catalog_read_failures import _failing_open

    cfg = tmp_path / "config"
    store = _listing_rows(cfg, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    daemon = MobileDaemon(port=0, password="pw123")
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})

    healthy = client.get("/api/sessions/past").json()
    assert {row["id"] for row in healthy["sessions"]} == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert healthy["degraded"] == []

    _failing_open(monkeypatch, store, OSError(errno.EACCES, "Permission denied"))
    broken = client.get("/api/sessions/past").json()

    assert broken["sessions"] == []
    assert broken["degraded"] == ["sessions"]


def test_a_non_store_failure_on_the_history_route_is_not_laundered(tmp_path, monkeypatch) -> None:
    """The half of the removal that the store test above cannot pin.

    ``_past_sessions`` dropped its ``except Exception: return []`` because it
    swallowed defects along with ``EACCES``, and the test above asserts only the
    GRACEFUL half (``[], ["sessions"]``). Without this pin a future refactor can
    put the broad catch back and stay green, which is how the laundering
    returned the first time. A failure that is not the store read reaches the
    request as a failure.
    """
    cfg = tmp_path / "config"
    _listing_rows(cfg, "aaaaaaaaaaaa")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    class _NotAStoreRead(Exception):
        """A defect in the row builder: exactly what may not be laundered."""

    def exploding_recent_session_rows(config_dir, limit=None, **kwargs):
        raise _NotAStoreRead("row builder bug")

    # Patched at the module the route imports FROM at call time, so the real
    # store never has to be broken to reach this arm.
    monkeypatch.setattr("local_operator.resume.recent_session_rows", exploding_recent_session_rows)

    daemon = MobileDaemon(port=0, password="pw123")
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})

    with pytest.raises(_NotAStoreRead):
        client.get("/api/sessions/past")


def test_the_search_route_names_the_store_it_could_not_read(tmp_path, monkeypatch) -> None:
    """The listing the shipped history screen ACTUALLY renders, and the marker.

    ``mobile/web/src/screens/past-sessions.tsx`` runs ``searchSessions("")`` on
    mount and then replaces its whole list with ``r.sessions`` — membership, not
    a merge — so this route's empty answer is the same claim the two listing
    routes had to stop making. Round 2 found it answering ``200`` with
    ``keys=['query', 'sessions']`` and no marker at all for a store it could not
    read, which is how the screen came to say "no past sessions yet" about a
    history nobody had read.

    The marker is the phone's own ``DEGRADED_DURABLE_LISTING`` word, the same
    one the home listing and the history route use, so a renderer keys on one
    vocabulary. Recording what the SCREEN should do with it (a "couldn't load"
    state instead of that empty text) is a user-visible design round of its own
    and is deferred on the PR rather than guessed at here.
    """
    import errno

    from tests.unit.session.test_catalog_read_failures import _failing_open

    cfg = tmp_path / "config"
    store = _listing_rows(cfg, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)

    daemon = MobileDaemon(port=0, password="pw123")
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})

    healthy = client.get("/api/sessions/search", params={"q": ""}).json()
    assert {row["id"] for row in healthy["sessions"]} == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert healthy["degraded"] == []

    # A store that READS, queried for something absent: still no marker. Or the
    # marker would mean "this search found nothing", and every miss on a healthy
    # phone would render as a failed read.
    miss = client.get("/api/sessions/search", params={"q": "zzzz-no-such-session"}).json()
    assert miss["sessions"] == []
    assert miss["degraded"] == []

    _failing_open(monkeypatch, store, OSError(errno.EACCES, "Permission denied"))
    broken = client.get("/api/sessions/search", params={"q": ""}).json()

    assert broken["sessions"] == []
    assert broken["degraded"] == ["sessions"], (
        "the screen the operator actually looks at must be able to tell an "
        "unreadable store from an empty one"
    )
