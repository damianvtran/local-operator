"""Session-health receipts reach the phone (mobile UX batch 2, U7).

The daemon half of the fix. An ended or degraded session used to serialize as
live on every wire path a phone observes: the list summary carried no flag at
all, and the projection's own ``ended``/``degraded`` fields were never True on
any path the phone could reach (the fold resets both; ``grep`` found no
non-test consumer in ``web/src``). Three places now make them true, and each
is pinned here where it is produced:

* the LIST summary (``_merge_summaries``) — the row's ended/degraded chip;
* the terminal repaint a death pushes at subscribers (``_scan_once``) — the
  session view's resume affordance and its "not answering" strip;
* the seed frame an opening session stream serves (``api_session_events``).

A durable-only row keeps reporting False for both: the daemon has not observed
that conversation end, and claiming it would be guessing.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from starlette.requests import Request

from local_operator.harness.types import Message
from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import SessionRecord
from local_operator.session.runtime import registry
from local_operator.session.transcript import Transcript

PASSWORD = "pw123"


def _record(session_id: str, pid: int) -> SessionRecord:
    return SessionRecord(
        pid=pid,
        kind="tui",
        session_id=session_id,
        conversation_name=session_id,
        cwd="/tmp",
        model_label="model",
        control_port=4100 + pid,
        control_key=f"key-{pid}",
        started_at=100.0,
    )


def test_summaries_carry_the_ended_and_degraded_receipts() -> None:
    daemon = MobileDaemon(port=0, password=PASSWORD)

    ended = SessionEntry(_record("gone", 101))
    ended.ended = True
    daemon.table.entries[101] = ended

    degraded = SessionEntry(_record("sick", 102))
    degraded.degraded = True
    daemon.table.entries[102] = degraded

    rows = {
        row["session_id"]: row
        for row in daemon.table._merge_summaries({"gone": None, "sick": None, "durable": None})
    }
    # The ended entry can only return through its durable row — and the receipt
    # rides it.
    assert rows["gone"]["ended"] is True
    assert rows["gone"]["degraded"] is False
    # A live entry whose dial is down says so; it is not ended.
    assert rows["sick"]["degraded"] is True
    assert rows["sick"]["ended"] is False
    # Nothing registered since boot: neither receipt is the daemon's to claim.
    assert rows["durable"]["ended"] is False
    assert rows["durable"]["degraded"] is False


def test_a_death_push_carries_the_ended_receipt(tmp_path, monkeypatch) -> None:
    """The scan PROVED this pid dead; the terminal repaint says so (U7)."""
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)
    directory = config / "sessions" / "dead-session"
    directory.mkdir(parents=True)
    asyncio.run(Transcript(directory).append_message(Message.user("hello", id="u1")))

    daemon = MobileDaemon(port=0, password=PASSWORD, dial_registrants=False)
    daemon.table.entries[101] = SessionEntry(_record("dead-session", 101))
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=8)
    daemon.table.session_subscribers["dead-session"] = {queue}

    monkeypatch.setattr(registry, "scan", lambda: [(_record("dead-session", 101), "stale")])
    asyncio.run(daemon._scan_once())

    frame = queue.get_nowait()
    assert frame["session_id"] == "dead-session"
    assert frame["ended"] is True
    assert frame["degraded"] is False


async def _seed_payload(app, session_id: str) -> dict:
    """The first SSE frame an opening session stream serves."""
    endpoint = next(
        route.endpoint
        for route in app.routes
        if route.path == "/api/sessions/{session_id:str}/events"
    )
    response = await endpoint(
        Request(
            {
                "type": "http",
                "method": "GET",
                "path": f"/api/sessions/{session_id}/events",
                "path_params": {"session_id": session_id},
                "query_string": b"",
                "headers": [
                    (b"host", b"fixture"),
                    (b"cookie", f"{COOKIE_NAME}={sign_cookie(PASSWORD)}".encode()),
                ],
                "scheme": "http",
                "server": ("fixture", 80),
            }
        )
    )
    try:
        frame = await anext(response.body_iterator)
    finally:
        await response.body_iterator.aclose()
    line = next(line for line in frame.splitlines() if line.startswith("data:"))
    return json.loads(line[len("data:") :].strip())


@pytest.mark.asyncio
async def test_the_seed_frame_carries_ended_or_degraded(tmp_path, monkeypatch) -> None:
    """Opening a stream on a non-live session seeds the receipt (U7)."""
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)
    for session_id in ("dead-session", "sick-session"):
        directory = config / "sessions" / session_id
        directory.mkdir(parents=True)
        await Transcript(directory).append_message(Message.user("hello", id=f"u-{session_id}"))

    daemon = MobileDaemon(port=0, password=PASSWORD, dial_registrants=False)
    app = build_app(daemon)

    # (a) The session is over: an ended entry is the receipt, no live entry
    # exists, and the rebuilt seed frame says so — this is what flips the
    # session view to "ended, offering resume".
    ended = SessionEntry(_record("dead-session", 101))
    ended.ended = True
    daemon.table.entries[101] = ended
    payload = await _seed_payload(app, "dead-session")
    assert payload["ended"] is True

    # (b) The session is alive but its dial is down: the entry knows, and the
    # rebuilt frame must carry it.
    sick = SessionEntry(_record("sick-session", 102))
    sick.degraded = True
    daemon.table.entries[102] = sick
    payload = await _seed_payload(app, "sick-session")
    assert payload["degraded"] is True
    assert payload["ended"] is False
