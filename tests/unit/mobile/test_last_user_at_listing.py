"""The phone list's ``last_user_at``: the Running section's order clock.

WHY: the mobile drawer orders running conversations the way the desktop sidebar
does (``chat-list-sections.ts``: ``last_user_at``, falling back to
``created_at``), and until this field rode ``/api/sessions`` the phone could only
use the fallback. These cells pin the wire contract the mobile client parses:

* the key is on EVERY row (live, ended or durable-only);
* its value is the person's own last message in epoch seconds, read from the
  transcript, for a running row;
* it is ``None`` — never a minted epoch — for a running row with no user message
  or an unreadable transcript, and for a row that is not running (only running
  rows pay the transcript read);
* remote (peer) rows carry the peer's claim, or ``None`` when it is absent or
  not a usable number.

Driven through the real app's ``/api/sessions`` route, not ``_merge_summaries``
alone: the value is resolved off the loop by the listing build, so a merge-only
test would not exercise the path a phone reads.
"""

from __future__ import annotations

import json
import os
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
    # An ENDED entry that was busy, and a durable-only directory with no runtime:
    # both have a user message on disk, neither is live, so neither pays the
    # read -- the "EVERY row" claim in the module docstring covers these too.
    _transcript(sessions / "s-ended", [_message("user", "hi", USER_TS, "u" * 24)])
    _live(daemon, _record("s-ended", 105, busy=True))
    daemon.table.entries[105].ended = True
    _transcript(sessions / "s-durable", [_message("user", "hi", USER_TS, "u" * 24)])

    rows = _rows(daemon)
    for session_id in ("s-none", "s-idle", "s-ended", "s-durable"):
        assert "last_user_at" in rows[session_id], rows[session_id]
        assert rows[session_id]["last_user_at"] is None, rows[session_id]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file anyway")
def test_an_unreadable_transcript_is_null(listing) -> None:
    """A running row whose transcript cannot be opened reports ``None``, and the
    listing still answers: ``last_user_at`` swallows the ``OSError`` rather than
    failing the whole page for one row (review round 1, MINOR-1).
    """
    daemon, sessions = listing
    _transcript(sessions / "s-locked", [_message("user", "hi", USER_TS, "u" * 24)])
    transcript = sessions / "s-locked" / "transcript.jsonl"
    _live(daemon, _record("s-locked", 106, busy=True))
    transcript.chmod(0)
    try:
        row = _rows(daemon)["s-locked"]
    finally:
        transcript.chmod(0o600)
    assert "last_user_at" in row, row
    assert row["last_user_at"] is None, row


def test_a_remote_row_carries_the_peer_s_claim_or_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote rows ride the SAME response and file into the same Active bins
    (``mobile/mesh.py``: "one list"), so they must carry the same clock or a
    mixed Running section orders local rows by ``last_user_at`` and remote rows
    by the ``created_at`` fallback. Mirrors ``test_desktop_mesh.py``'s cell with
    one deliberate difference: this route's contract is "always present, null
    when unknown", where the desktop omits the key.
    """
    from typing import Any, cast

    from local_operator.mobile import mesh as mobile_mesh
    from local_operator.session import peer_rows as peer_rows_mod
    from local_operator.session.peer_rows import SessionRow

    claimed = SessionRow("b" * 12, 1789400000.0, "build box chat", last_user_at=1789400111.5)
    bare = SessionRow("c" * 12, 1789400000.0, "no claim")
    bool_claim = SessionRow("d" * 12, 1789400000.0, "old peer", last_user_at=cast(Any, True))
    garbage = SessionRow("e" * 12, 1789400000.0, "odd peer", last_user_at=cast(Any, "abc"))
    monkeypatch.setattr(
        peer_rows_mod,
        "peer_session_rows",
        lambda root=None, **kwargs: (claimed, bare, bool_claim, garbage),
    )

    rows = {row["session_id"]: row for row in mobile_mesh.remote_session_rows(tmp_path, pinned=())}

    assert rows["b" * 12]["last_user_at"] == 1789400111.5
    for session_id in ("c" * 12, "d" * 12, "e" * 12):
        assert "last_user_at" in rows[session_id], rows[session_id]
        assert rows[session_id]["last_user_at"] is None, rows[session_id]
