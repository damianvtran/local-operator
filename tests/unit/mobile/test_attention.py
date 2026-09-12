"""Exercise the real relay route, including stale receipt and passive SSE cases."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from starlette.testclient import TestClient

from local_operator.mobile.daemon import MobileDaemon, _projection_frame, build_app
from local_operator.mobile.types import SessionProjection
from local_operator.paths import config_dir
from local_operator.session.attention import AttentionStore


def test_mobile_requires_exact_observed_token_and_never_reads_on_subscription() -> None:
    sid = "abcdef123456"
    directory = config_dir() / "sessions" / sid
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("")
    store = AttentionStore()
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    store.publish(f"session/{sid}", a, "answer-a", "complete")
    daemon = MobileDaemon(port=0, password="isolated-test")
    row = SimpleNamespace(mtime=1e30)
    daemon.table._attention_states = store.state_many([f"session/{sid}"])
    assert daemon.table._is_unseen(sid, row, None)
    daemon.table.session_subscribers[sid] = {asyncio.Queue(maxsize=1)}
    assert daemon.table._is_unseen(sid, row, None)
    before = (config_dir() / "attention.db").stat().st_mtime_ns
    _projection_frame(SessionProjection(session_id=sid, pid=12345))
    assert (config_dir() / "attention.db").stat().st_mtime_ns == before
    client = TestClient(build_app(daemon), follow_redirects=False)
    route = f"/api/sessions/{sid}/seen"
    assert client.post(route, json={"completion_token": a}).status_code == 401
    client.post("/login", data={"password": "isolated-test"})
    assert client.post(route).status_code == 422
    assert client.post(route, json={"completion_token": 123}).status_code == 422
    assert client.post(route, json={"completion_token": str(uuid.uuid4())}).status_code == 409
    assert (
        client.post("/api/sessions/deadbeef1234/seen", json={"completion_token": a}).status_code
        == 404
    )
    store.publish(f"session/{sid}", b, "answer-b", "complete")
    response = client.post(route, json={"completion_token": a})
    assert response.status_code == 200
    assert response.json()["attention"]["unseen"]
    assert response.json()["attention"]["completion_token"] == b
    assert client.post(route, json={"completion_token": b}).json()["attention"]["unseen"] is False
    assert client.post(route, json={"completion_token": a}).json()["attention"]["unseen"] is False
    assert not AttentionStore(config_dir() / "attention.db").state(f"session/{sid}")["unseen"]


@pytest.mark.asyncio
async def test_summary_receipts_are_one_batch_off_the_event_loop(monkeypatch) -> None:
    from local_operator.mobile.daemon import SessionTable

    store = AttentionStore()
    ids = [f"session-{index}" for index in range(40)]
    for session_id in ids:
        store.publish(f"session/{session_id}", str(uuid.uuid4()), "result", "complete")
    table = SessionTable()

    async def durable_rows():
        return dict.fromkeys(ids)

    monkeypatch.setattr(table, "_refresh_durable_rows", durable_rows)
    monkeypatch.setattr(
        table, "_merge_summaries", lambda rows: list(table._attention_states.values())
    )
    original = sqlite3.connect
    threads: list[int] = []

    def connect(*args, **kwargs):
        threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    rows = await table.summaries()
    assert len(rows) == 40
    assert len(threads) == 1
    assert threads[0] != threading.get_ident()
    for session_id in ids:
        assert table._is_unseen(session_id, None, None)
    assert len(threads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["file_created", "partial_transaction"])
async def test_first_publication_is_safe_for_batch_revision_and_authenticated_list(
    monkeypatch, stage
) -> None:
    from local_operator.harness.types import Message, TextContent
    from local_operator.session.transcript import Transcript

    sid = "fedcba654321"
    transcript = Transcript(config_dir() / "sessions" / sid)
    await transcript.append_message(
        Message(role="assistant", content=[TextContent(text="Completed result")])
    )
    path = config_dir() / "attention.db"
    created, release = threading.Event(), threading.Event()
    original_touch, original_connect = Path.touch, sqlite3.connect

    def pause() -> None:
        created.set()
        if not release.wait(10):
            raise RuntimeError("test did not release the first publisher")

    def touch(self: Path, *args: Any, **kwargs: Any) -> None:
        original_touch(self, *args, **kwargs)
        if self == path and stage == "file_created":
            pause()

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = original_connect(*args, **kwargs)
        if stage == "partial_transaction" and threading.current_thread().name.startswith(
            "attention-writer"
        ):

            def trace(sql: str) -> None:
                if "CREATE TABLE" in sql.upper() and "receipts" in sql:
                    pause()

            conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(Path, "touch", touch)
    monkeypatch.setattr(sqlite3, "connect", connect)
    daemon = MobileDaemon(port=0, password="isolated-first-publication")
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "isolated-first-publication"})
    store = AttentionStore(path)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="attention-writer") as pool:
        publisher = pool.submit(
            store.publish, f"session/{sid}", str(uuid.uuid4()), "result", "complete"
        )
        try:
            assert created.wait(10)
            assert store.state_many([f"session/{sid}"])[f"session/{sid}"]["unseen"] is False
            # Three terms since supersession became detectable: the third is a
            # heal counter that a store this empty has never bumped.
            assert store.revision() == (0, 0, 0)
            response = client.get("/api/sessions")
            assert response.status_code == 200
            assert response.json()["sessions"][0]["unseen"] is False
        finally:
            release.set()
            publisher.result(timeout=10)
    assert store.state(f"session/{sid}")["unseen"] is True


@pytest.mark.parametrize(
    "final_size,neighbor_size,complete",
    [(1_000_000, 0, False), (5000, 1_000_000, False), (20, 1_000_000, True)],
)
def test_completion_end_completeness_survives_runtime_and_relay_caps(
    final_size, neighbor_size, complete
) -> None:
    from local_operator.mobile.types import TranscriptEntry, _projection_from_json
    from local_operator.session.runtime.server import RuntimeServer

    ending = "TRUE_FINAL_RESULT_END"
    original = "x" * final_size + ending
    projection = SessionProjection(
        session_id="cap-session",
        pid=0,
        attention={
            "conversation_id": "session/cap-session",
            "completion_token": str(uuid.uuid4()),
            "anchor_id": "final",
            "kind": "complete",
            "unseen": True,
            "revision": [1, 0],
        },
        transcript=[
            TranscriptEntry(id="neighbor", kind="assistant", text="n" * neighbor_size),
            TranscriptEntry(id="final", kind="assistant", text=original),
        ],
    )
    runtime = RuntimeServer(
        cast(Any, SimpleNamespace(session_projection_seed=projection)), kind="tui"
    )
    payload = runtime._projection_payload()["data"]
    received = _projection_from_json(payload, runtime._record)
    relayed = _projection_frame(received)
    row = next(row for row in relayed["transcript"] if row["id"] == "final")
    assert row["text_complete"] is complete
    assert (ending in row["text"]) is complete
    assert relayed["attention"]["anchor_id"] == row["id"]
    assert projection.transcript[-1].text == original
    assert projection.transcript[-1].text_complete


@pytest.mark.asyncio
async def test_cold_receipt_repaint_keeps_the_existing_relay_epoch(monkeypatch) -> None:
    daemon = MobileDaemon(port=0, password="isolated-cold-repaint")
    daemon._attention_bootstrapped = True
    sid = "cold-session"
    previous = daemon.capture_subagent_details(SessionProjection(session_id=sid, pid=0, version=80))
    prior_version = previous.version
    cold = SessionProjection(session_id=sid, pid=0, version=1)
    monkeypatch.setattr("local_operator.mobile.daemon._durable_projection", lambda session_id: cold)
    monkeypatch.setattr("local_operator.mobile.daemon.registry.scan", lambda *args, **kwargs: [])
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
    daemon.table.session_subscribers[sid] = {queue}
    store = AttentionStore()
    token = str(uuid.uuid4())
    store.publish(f"session/{sid}", token, "result", "complete")
    store.acknowledge(f"session/{sid}", token)
    await daemon._scan_once()
    frame = queue.get_nowait()
    assert frame["version"] == prior_version
    assert frame["attention"]["completion_token"] == token
    assert not frame["attention"]["unseen"]


def test_relay_only_cap_and_unknown_old_owner_never_claim_complete() -> None:
    from local_operator.mobile.types import TranscriptEntry, _projection_from_json
    from local_operator.session.runtime.server import RuntimeServer

    projection = SessionProjection(
        session_id="cap-session",
        pid=0,
        transcript=[TranscriptEntry(id="final", kind="assistant", text="f" * 5000 + "TRUE_END")],
    )
    runtime = RuntimeServer(
        cast(Any, SimpleNamespace(session_projection_seed=projection)), kind="tui"
    )
    payload = runtime._projection_payload()["data"]
    assert payload["transcript"][0]["text_complete"]
    received = _projection_from_json(payload, runtime._record)
    received.transcript.insert(
        0, TranscriptEntry(id="neighbor", kind="assistant", text="n" * 1_000_000)
    )
    row = _projection_frame(received)["transcript"][-1]
    assert not row["text_complete"] and "TRUE_END" not in row["text"]
    payload["transcript"][0].pop("text_complete")
    legacy = _projection_from_json(payload, runtime._record)
    assert not legacy.transcript[0].text_complete
    assert not _projection_frame(legacy)["transcript"][0]["text_complete"]


def test_legacy_metadata_mtime_is_not_a_completion(tmp_path: Path) -> None:
    daemon = MobileDaemon(port=0, password="isolated-test")
    assert not daemon.table._is_unseen("no-outcome", SimpleNamespace(mtime=1e30), None)


@pytest.mark.parametrize(
    "kind,reason,expected_text,expected_severity",
    [
        (
            "error",
            "the runtime was terminated while this turn was running",
            "Stopped with an error — the runtime was terminated while this turn was running",
            "error",
        ),
        ("error", "", "Stopped with an error", "error"),
        ("interrupted", "the session was stopped by the user", "Interrupted", "info"),
    ],
)
def test_the_phone_notice_carries_the_tier_the_row_deserves(
    kind: str, reason: str, expected_text: str, expected_severity: str
) -> None:
    """D4: the phone flattened a cut-off into the routine receipt tier.

    ``NoticeRow`` picks its glyph and ink from ``details.severity``, and the
    frame carried an empty ``details`` — so a cut-off rendered as the same dim
    ``·`` as ``Interrupted``, indistinguishable from a routine receipt, while
    the TUI painted the same event in danger ink. The tier is derived in
    ``harness/rows.py`` (both surfaces) rather than at each serialization site.
    """
    projection = SessionProjection(
        session_id="sev-session",
        pid=0,
        attention={
            "conversation_id": "session/sev-session",
            "completion_token": str(uuid.uuid4()),
            "anchor_id": "completion-t",
            "kind": kind,
            "reason": reason,
            "unseen": True,
            "revision": [1, 0],
        },
    )
    row = _projection_frame(projection)["transcript"][-1]
    assert row["text"] == expected_text
    assert row["details"] == {"severity": expected_severity}


def _end_frame(
    *,
    kind: str | None,
    cause: str = "",
    stop_reason: str = "",
    cut_off: bool = False,
    streaming: bool = False,
) -> dict[str, Any]:
    """One frame through the daemon's real serialization boundary."""
    attention: dict[str, Any] = {
        "conversation_id": "session/end-session",
        "completion_token": str(uuid.uuid4()),
        "anchor_id": "completion-t",
        "kind": kind,
        "reason": "the session's runtime stopped answering while this turn was running",
        "cause": cause,
        "unseen": True,
        "revision": [1, 0],
    }
    projection = SessionProjection(
        session_id="end-session",
        pid=0,
        attention=attention,
        stop_reason=stop_reason,
        cut_off=cut_off,
        streaming=streaming,
    )
    return _projection_frame(projection)


@pytest.mark.parametrize(
    "kind,cause,expected_cut_off",
    [
        # A cut-off: the taxonomy's `error` kind. The cause is what tells the
        # button which WORD to use, never whether the affordance is offered.
        ("error", "runtime-killed", True),
        ("error", "owner-lost", True),
        # No cause at all is still a cut-off: that is the row whose notice
        # already says the cause could not be determined.
        ("error", "", True),
        # ...and the ONE deliberate token stays deliberate even when a writer
        # mismatches it onto `error`, so the button cannot call a stop a cut-off.
        ("error", "user-stop", False),
        # A deliberate stop from the phone: no button-less dead end either.
        ("interrupted", "user-stop", False),
    ],
)
def test_the_phone_frame_fills_the_end_the_runtime_could_not_send(
    kind: str, cause: str, expected_cut_off: bool
) -> None:
    """U1: the resume affordance needs the field, and only the fold wrote it.

    ``ProjectionFold`` sets ``stop_reason``/``cut_off`` from a folded
    ``AgentEndEvent``, and a runtime that stops mid-turn never emits one — the
    follower's socket just closes. So D7's word was unreachable on a real phone:
    ``composer.tsx`` gates the whole button on ``stop_reason === "aborted"``.
    The durable outcome is where the end DOES exist for those arms, so the frame
    fills the missing field from the same record its notice is built from.
    """
    frame = _end_frame(kind=kind, cause=cause)
    assert frame["stop_reason"] == "aborted"
    assert frame["cut_off"] is expected_cut_off
    # One record decides both, so the button and the sentence above it agree.
    assert frame["transcript"][-1]["text"].startswith(
        "Stopped with an error" if kind == "error" else "Interrupted"
    )


def test_the_phone_frame_never_overrides_an_abort_the_fold_saw() -> None:
    """FILL, never override — the FOLD's own abort outranks a durable record.

    A record may describe an earlier turn than the one the fold last saw, and
    the fold is the only party that saw an end event for the current one. So a
    folded ``aborted`` end keeps its word and its flag in both directions: a
    deliberate stop the fold classified stays ``cut_off=False``, and a cut-off
    it classified stays ``True``. And a frame with nothing to offer (a live
    turn, no outcome, a completion) is left exactly as the fold produced it.
    """
    # The fold saw the deliberate stop: the record's older error must not
    # relabel the user's own act as a cut-off.
    deliberate = _end_frame(
        kind="error", cause="runtime-killed", stop_reason="aborted", cut_off=False
    )
    assert deliberate["stop_reason"] == "aborted"
    assert deliberate["cut_off"] is False
    # ...and the other way round: the fold saw the cut-off.
    folded_cut_off = _end_frame(
        kind="interrupted", cause="user-stop", stop_reason="aborted", cut_off=True
    )
    assert folded_cut_off["stop_reason"] == "aborted"
    assert folded_cut_off["cut_off"] is True
    # A live turn banners nothing either: the suppression the notice already
    # applies is the same condition, so the field cannot fill ahead of it.
    live = _end_frame(kind="error", cause="runtime-killed", streaming=True)
    assert live["stop_reason"] == ""
    # ...and a store with no outcome at all has no end to offer.
    empty = _end_frame(kind=None)
    assert empty["stop_reason"] == ""
    # A completion is not a dead end either: no resume affordance is claimed
    # for a turn that finished with nothing outstanding in the store.
    done = _end_frame(kind="complete")
    assert done["stop_reason"] == ""


def test_a_stop_after_a_completed_turn_still_offers_the_way_back() -> None:
    """``completed`` is not an END for this purpose, and requiring it be empty
    withheld the button from a deliberate stop.

    A session that finished a turn and then had the NEXT one stopped from the
    phone leaves exactly this pair: ``stop_reason='completed'`` from the earlier
    fold, and an ``interrupted`` outcome from the turn the fold never saw end
    (the phone's stop path disposes the runtime, so no ``AgentEndEvent`` reaches
    the follower). A completed turn publishes ``kind='complete'``, so a store
    carrying an error or an interruption is describing the LATEST turn, not the
    one the fold finished.
    """
    stopped = _end_frame(kind="interrupted", cause="user-stop", stop_reason="completed")
    assert stopped["stop_reason"] == "aborted"
    assert stopped["cut_off"] is False
    cut_off = _end_frame(kind="error", cause="owner-lost", stop_reason="completed")
    assert cut_off["stop_reason"] == "aborted"
    assert cut_off["cut_off"] is True


@pytest.mark.asyncio
async def test_a_daemon_discovered_kill_names_the_runtime_it_found_dead() -> None:
    """MINOR-1 on the daemon's own path, not on the classifier in isolation.

    The discovery branch classifies the death `registry.scan` just reported —
    and that same scan UNLINKS the record. The daemon therefore hands the record
    it is holding to the classification; without it the phone's notice for a
    daemon-owned SIGKILL was the no-evidence sentence ("the cause could not be
    determined") even though the pid had just been proved dead, and the design
    round measured exactly that sentence on a real frame (D6).

    Driven through `_scan_once`, the daemon's real 2 s pass, because the bug was
    an ORDERING between two of its own calls and a classifier-level test cannot
    see it.
    """
    import json
    import uuid as _uuid

    from local_operator.session.runtime.types import RUN_DIRNAME, SessionRecord
    from local_operator.session.transcript import Transcript

    session_id = "discovered1"
    directory = config_dir() / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    token = str(_uuid.uuid4())
    await Transcript(directory).append_custom(
        "attention_started", {"conversation_id": f"session/{session_id}", "token": token}
    )
    dead_pid = 2**22 + 13
    run = config_dir() / RUN_DIRNAME
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{dead_pid}.json").write_text(
        json.dumps(
            SessionRecord(
                pid=dead_pid,
                kind="daemon",
                session_id=session_id,
                conversation_name=session_id,
                cwd=str(directory),
                model_label="m",
                control_port=1,
                control_key="k",
                version="1.2.3",
                source_ref="abcdef0",
            ).to_json()
        ),
        encoding="utf-8",
    )

    daemon = MobileDaemon(port=0, password="pw")
    # The BOOT sweep is a different path with its own ordering (it classifies up
    # to 100 recent directories before this loop's scan runs), and it is not what
    # this test is about. Production reaches the discovery branch on every pass
    # after the first; pre-marking is how the test lands on that pass.
    daemon._attention_bootstrapped = True
    await daemon._scan_once()

    state = AttentionStore().state(f"session/{session_id}")
    assert (state["kind"], state["cause"]) == ("error", "runtime-killed"), state
    assert f"pid {dead_pid}" in state["reason"], state["reason"]
    assert not (run / f"{dead_pid}.json").exists(), "the record was not reaped"
