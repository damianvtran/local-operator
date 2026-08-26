"""End-to-end over the real sockets: a fake registrant publishes a record,
the daemon adopts it, projections flow, control requests round-trip, and the
HTTP gate holds. These tests use the repo's config-dir isolation (conftest
sets LOCAL_OPERATOR_CONFIG_DIR to a tmp path)."""

from __future__ import annotations

import asyncio
import json

import pytest
from starlette.testclient import TestClient

from local_operator.mobile import registry
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, _dial, build_app
from local_operator.mobile.registrant import Registrant
from local_operator.mobile.types import (
    PROJECTION_TRANSCRIPT_LIMIT,
    SessionProjection,
    SessionRecord,
    SubagentRow,
    TodoItem,
    TodoPhase,
    TranscriptEntry,
)


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
    registrant = Registrant(handle, kind="tui")
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
            assert reply == {"op": "ack", "req": reply["req"], "detail": "steer ok"}
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
        for _ in range(50):
            if entry.projection is not None:
                break
            await asyncio.sleep(0.05)
        # The oversized frame was skipped; the following normal frame arrived.
        assert entry.projection is not None
        assert entry.projection.conversation_name == "recovered"
    finally:
        dial.cancel()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_wrong_key_is_rejected_silently() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
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
    assert authed.json() == {"sessions": []}

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
