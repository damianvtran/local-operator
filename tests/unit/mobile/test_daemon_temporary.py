"""Temporary chats on the phone relay (contract §2.3, frozen wire).

Real routes over a real store: the list excludes temporary chats unless asked —
on BOTH transports (``GET /api/sessions`` and the ``/api/sessions/events``
SSE the home screen reads) — Past and search never show them, the toggle and
start body mark them, a pin is refused, a deleted one answers its own 404, and
``features.temporary_chats`` is republished.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from local_operator.harness.types import Message
from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.session import temporary
from local_operator.session.transcript import Transcript

PASSWORD = "pw123"
TEMP = "aaaaaaaaaaaa"
PLAIN = "bbbbbbbbbbbb"


def _turn(directory: Path, text: str) -> None:
    transcript = Transcript(directory)
    asyncio.run(transcript.append_message(Message.user(text, id=f"u-{directory.name}")))
    asyncio.run(transcript.append_message(Message.assistant("ok", id=f"a-{directory.name}")))


@pytest.fixture
def relay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "config"
    for session_id in (TEMP, PLAIN):
        directory = cfg / "sessions" / session_id
        directory.mkdir(parents=True)
        _turn(directory, f"needle words {session_id}")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))
    assert temporary.write_mark(cfg, TEMP, True, by="relay").ok
    daemon = MobileDaemon(port=0, password=PASSWORD)
    app = build_app(daemon)
    client = TestClient(app, follow_redirects=False)
    client.post("/login", data={"password": PASSWORD})
    return client, cfg, daemon, app


def _ids(rows: list[dict[str, Any]]) -> set[str]:
    return {row["session_id"] for row in rows}


def test_the_list_excludes_temporary_chats_unless_asked(relay) -> None:
    client, _cfg, _daemon, _app = relay
    rows = client.get("/api/sessions").json()["sessions"]
    assert _ids(rows) == {PLAIN}
    assert all(row["temporary"] is None for row in rows)
    both = client.get("/api/sessions?include_temporary=1").json()["sessions"]
    assert _ids(both) == {TEMP, PLAIN}
    by_id = {row["session_id"]: row for row in both}
    assert by_id[TEMP]["temporary"]["idle_minutes"] == 30
    assert by_id[TEMP]["temporary"]["hold"] is None
    assert by_id[PLAIN]["temporary"] is None


async def _first_sessions_frame(app: Any, query: bytes) -> dict[str, Any]:
    endpoint = next(
        r for r in app.routes if getattr(r, "path", "") == "/api/sessions/events"
    ).endpoint
    response = await endpoint(
        Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/api/sessions/events",
                "query_string": query,
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


def test_the_sse_list_takes_the_flag_too(relay) -> None:
    """The home screen reads the SSE stream: without the flag there, the
    relay's Temporary group would stay empty however the GET answered."""
    _client, _cfg, _daemon, app = relay

    async def both() -> tuple[dict[str, Any], dict[str, Any]]:
        plain = await _first_sessions_frame(app, b"")
        asked = await _first_sessions_frame(app, b"include_temporary=1")
        return plain, asked

    plain, asked = asyncio.run(both())
    assert _ids(plain["sessions"]) == {PLAIN}
    assert _ids(asked["sessions"]) == {TEMP, PLAIN}


def test_the_feature_is_republished(relay) -> None:
    client, *_ = relay
    frame = client.get("/api/sessions").json()
    assert frame["capabilities"]["features"]["temporary_chats"] == 1


def test_past_and_search_never_show_a_temporary_chat(relay) -> None:
    client, *_ = relay
    past = client.get("/api/sessions/past").json()
    past_rows = past.get("sessions", past) if isinstance(past, dict) else past
    assert TEMP not in json.dumps(past_rows)
    found = client.get("/api/sessions/search", params={"q": "needle"})
    assert found.status_code == 200, found.text
    assert TEMP not in found.text


def test_the_toggle_keeps_and_marks(relay) -> None:
    client, cfg, *_ = relay
    kept = client.post(f"/api/sessions/{TEMP}/temporary", json={"temporary": False})
    assert kept.status_code == 200, kept.text
    assert kept.json() == {"session_id": TEMP, "temporary": None}
    assert _ids(client.get("/api/sessions").json()["sessions"]) == {TEMP, PLAIN}
    marked = client.post(f"/api/sessions/{PLAIN}/temporary", json={"temporary": True})
    assert marked.status_code == 200, marked.text
    assert marked.json()["temporary"]["idle_minutes"] == 30
    assert temporary.is_marked(cfg, PLAIN)
    assert (
        client.post(f"/api/sessions/{PLAIN}/temporary", json={"temporary": "yes"}).status_code
        == 422
    )


def test_making_a_pinned_chat_temporary_is_refused(relay) -> None:
    client, cfg, *_ = relay
    assert client.post(f"/api/sessions/{PLAIN}/pin", json={"pinned": True}).status_code == 200
    refused = client.post(f"/api/sessions/{PLAIN}/temporary", json={"temporary": True})
    assert refused.status_code == 409
    assert refused.json()["code"] == "session_pinned"


def test_pinning_a_temporary_chat_is_refused(relay) -> None:
    client, *_ = relay
    refused = client.post(f"/api/sessions/{TEMP}/pin", json={"pinned": True})
    assert refused.status_code == 409
    assert refused.json()["code"] == "session_temporary"
    assert refused.json()["error"]


def test_a_deleted_temporary_chat_answers_its_own_404(relay) -> None:
    client, cfg, *_ = relay
    gone = "cccccccccccc"
    now = time.time()
    (cfg / "sessions" / temporary.TOMBSTONES_NAME).write_text(
        json.dumps([{"id": gone, "deleted_at": now}])
    )
    for response in (
        client.get(f"/api/sessions/{gone}/history"),
        client.post(f"/api/sessions/{gone}/temporary", json={"temporary": False}),
        client.post("/api/sessions/resume", json={"session_id": gone}),
        client.get(f"/api/sessions/{gone}/events"),
    ):
        assert response.status_code == 404, response.text
        assert response.json()["code"] == "session_temporary_deleted"
        assert response.json()["deleted_at"] == now
    plain_404 = client.get("/api/sessions/dddddddddddd/history")
    assert plain_404.status_code == 404
    assert "code" not in plain_404.json()


def test_start_with_temporary_marks_the_new_session(relay, monkeypatch) -> None:
    client, cfg, daemon, _app = relay

    async def fake_spawn(cwd, *, resume=None, **_kwargs):  # noqa: ANN001, ANN202
        (cfg / "sessions" / resume).mkdir(parents=True, exist_ok=True)
        return 4242

    monkeypatch.setattr(daemon, "spawn_session", fake_spawn)
    response = client.post(
        "/api/sessions/start", json={"cwd": str(cfg), "temporary": True}
    )
    assert response.status_code == 200, response.text
    assert temporary.is_marked(cfg, response.json()["session_id"])
    plain = client.post("/api/sessions/start", json={"cwd": str(cfg)})
    assert not temporary.is_marked(cfg, plain.json()["session_id"])


def test_a_keyed_start_carries_temporary_in_its_intent() -> None:
    from local_operator.mobile.start_receipts import StartRefusal, canonical_request

    key = "8b0f4c8e-3f7a-4f43-9b1a-2a7f0c1d2e3f"
    _, plain = canonical_request({"create_request_id": key, "cwd": "/x"})
    _, temp = canonical_request({"create_request_id": key, "cwd": "/x", "temporary": True})
    _, off = canonical_request({"create_request_id": key, "cwd": "/x", "temporary": False})
    assert "temporary" not in plain and plain == off
    assert json.loads(temp)["temporary"] == "1"
    with pytest.raises(StartRefusal):
        canonical_request({"create_request_id": key, "temporary": "yes"})
