"""The phone list's ``last_user_at``: the Running section's order clock.

WHY: the mobile drawer orders running conversations the way the desktop sidebar
does (``chat-list-sections.ts``: ``last_user_at``, falling back to
``created_at``), and until this field rode ``/api/sessions`` the phone could only
use the fallback. These cells pin the wire contract the mobile client parses:

* the key is on EVERY row (live, ended or durable-only);
* its value is the person's own last message in epoch seconds, read from the
  transcript, for a running row;
* it is ``None`` — never a minted epoch — for a running row with no user message,
  and for a row that is not running (only running rows pay the transcript read).

Driven through the real app's ``/api/sessions`` route, not ``_merge_summaries``
alone: the value is resolved off the loop by the listing build, so a merge-only
test would not exercise the path a phone reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import SessionProjection, SessionRecord

USER_TS = 1_763_000_000.25
REPLY_TS = USER_TS + 30.0


def _record(session_id: str, pid: int, *, busy: bool) -> SessionRecord:
    return SessionRecord(
        pid=pid,
        kind="tui",
        session_id=session_id,
        conversation_name=session_id,
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="k",
        busy=busy,
    )


def _transcript(session_dir: Path, rows: list[dict[str, object]]) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )


def _message(role: str, text: str, ts: float, entry_id: str) -> dict[str, object]:
    return {
        "id": entry_id,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": role, "content": [{"text": text}]},
    }


@pytest.fixture
def listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[MobileDaemon, Path]:
    cfg = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))
    return MobileDaemon(port=0, password="pw123"), cfg / "sessions"


def _rows(daemon: MobileDaemon) -> dict[str, dict[str, object]]:
    client = TestClient(build_app(daemon), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    body = client.get("/api/sessions").json()
    return {row["session_id"]: row for row in body["sessions"]}


def _live(daemon: MobileDaemon, record: SessionRecord, *, streaming: bool = False) -> None:
    entry = SessionEntry(record)
    projection = SessionProjection(
        session_id=record.session_id, pid=record.pid, kind="tui", conversation_name="x"
    )
    projection.streaming = streaming
    entry.projection = projection
    daemon.table.entries[record.pid] = entry


def test_a_running_row_carries_the_person_s_last_message_time(listing) -> None:
    daemon, sessions = listing
    # The assistant's LATER reply must not move the clock: that is the whole
    # reason this field exists instead of ``mtime``.
    _transcript(
        sessions / "s-user",
        [
            _message("user", "hi", USER_TS, "u" * 24),
            _message("assistant", "yo", REPLY_TS, "a" * 24),
        ],
    )
    _live(daemon, _record("s-user", 101, busy=True))

    row = _rows(daemon)["s-user"]
    assert row["last_user_at"] == USER_TS
    assert isinstance(row["last_user_at"], float)


def test_a_streaming_projection_counts_as_running(listing) -> None:
    daemon, sessions = listing
    _transcript(sessions / "s-stream", [_message("user", "go", USER_TS, "u" * 24)])
    _live(daemon, _record("s-stream", 102, busy=False), streaming=True)

    assert _rows(daemon)["s-stream"]["last_user_at"] == USER_TS


def test_unknown_is_null_never_absent(listing) -> None:
    daemon, sessions = listing
    # Running, but the person has not sent anything yet (assistant row only).
    _transcript(sessions / "s-none", [_message("assistant", "ready", REPLY_TS, "a" * 24)])
    _live(daemon, _record("s-none", 103, busy=True))
    # Idle resident runtime WITH a user message: not running, so it does not pay
    # the transcript read and reports unknown.
    _transcript(sessions / "s-idle", [_message("user", "hi", USER_TS, "u" * 24)])
    _live(daemon, _record("s-idle", 104, busy=False))

    rows = _rows(daemon)
    for session_id in ("s-none", "s-idle"):
        assert "last_user_at" in rows[session_id], rows[session_id]
        assert rows[session_id]["last_user_at"] is None, rows[session_id]
