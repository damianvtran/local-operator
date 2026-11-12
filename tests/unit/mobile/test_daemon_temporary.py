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
    response = client.post("/api/sessions/start", json={"cwd": str(cfg), "temporary": True})
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


def test_the_daemon_hosts_the_temporary_sweeper(tmp_path, monkeypatch) -> None:
    """The lifespan really starts it (a swallowed NameError once made it a no-op)."""
    from local_operator.session import cleanup

    started: list[tuple[Path, str]] = []
    stopped: list[bool] = []
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)
    monkeypatch.setattr(
        cleanup,
        "start_temporary_sweeper",
        lambda _manager, root, *, actor: started.append((root, actor)) or True,
    )
    monkeypatch.setattr(cleanup, "stop_temporary_sweeper", lambda: stopped.append(True))
    with TestClient(build_app(MobileDaemon(port=0, password=PASSWORD))):
        assert started == [(tmp_path, "phone-daemon")]
    assert stopped == [True]


def _live_entry(daemon: MobileDaemon, session_id: str, pid: int = 4747) -> None:
    from local_operator.mobile.daemon import SessionEntry
    from local_operator.mobile.registry import SessionRecord

    record = SessionRecord(
        pid=pid,
        kind="tui",
        session_id=session_id,
        conversation_name=f"session {session_id}",
        cwd="/synthetic",
        model_label="fixture",
        control_port=1,
        control_key="fixture",
    )
    daemon.table.entries[pid] = SessionEntry(record)


def _reap(cfg: Path, session_id: str, now: float) -> None:
    """The temporary pass's own order: tombstone first, then the removal."""
    import shutil

    (cfg / "sessions" / temporary.TOMBSTONES_NAME).write_text(
        json.dumps([{"id": session_id, "deleted_at": now}])
    )
    shutil.rmtree(cfg / "sessions" / session_id)


def test_a_chat_this_daemon_still_holds_answers_the_tombstone_first(relay) -> None:
    # Core #3832 round 1, C-1: every session route consulted the tombstone
    # only when it found NO live entry, so the daemon that hosted the chat kept
    # serving it after the pass removed it.
    client, cfg, daemon, _app = relay
    _live_entry(daemon, TEMP)
    now = time.time()
    _reap(cfg, TEMP, now)
    # One request at a time, asserted before the next: on the round-1 code the
    # SSE route SERVED the reaped chat (a stream that never ends), so it goes
    # last and the earlier reads fail first.
    for method, url, body in (
        ("get", f"/api/sessions/{TEMP}/history", None),
        ("get", f"/api/sessions/{TEMP}/checkpoints", None),
        ("post", f"/api/sessions/{TEMP}/seen", {}),
        ("post", f"/api/sessions/{TEMP}/pin", {"pinned": True}),
        ("post", f"/api/sessions/{TEMP}/temporary", {"temporary": False}),
        ("get", f"/api/sessions/{TEMP}/events", None),
    ):
        response = getattr(client, method)(url, **({"json": body} if body is not None else {}))
        assert response.status_code == 404, (url, response.status_code, response.text[:200])
        assert response.json()["code"] == "session_temporary_deleted"
        assert response.json()["deleted_at"] == now
        assert not (cfg / "sessions" / TEMP).exists(), "a route recreated the directory"
        assert all(e.record.session_id != TEMP for e in daemon.table.entries.values())
        _live_entry(daemon, TEMP)  # the next route meets a held chat again
    # Unauthenticated callers learn nothing about which ids were deleted.
    anonymous = TestClient(_app, follow_redirects=False)
    assert anonymous.get(f"/api/sessions/{TEMP}/history").status_code == 401


def _wrapped_session_routes(app) -> list[Any]:
    from starlette.routing import Route

    return [
        route
        for route in app.routes
        if isinstance(route, Route) and route.path.startswith("/api/sessions/{session_id:str}")
    ]


def test_an_anonymous_caller_gets_each_route_s_own_refusal_and_no_tombstone_read(
    relay, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The C-1 wrapper answered every unauthenticated request with the bare
    # ``gate`` refusal, stripping the route family's refusal policy (sound:
    # nosniff + CORP; supplements: nosniff + no-store). It now hands the
    # refused request to the route's own gate. Pinned on EVERY wrapped route,
    # against a tombstoned id: 401 each time, and the tombstone is never read,
    # so a probe cannot tell a deleted temporary chat from any other id.
    _client, cfg, _daemon, app = relay
    _reap(cfg, TEMP, time.time())
    reads: list[str] = []
    real_deleted_at = temporary.deleted_at

    def counting_deleted_at(*args: Any, **kwargs: Any) -> Any:
        reads.append(str(args[1:]))
        return real_deleted_at(*args, **kwargs)

    monkeypatch.setattr(temporary, "deleted_at", counting_deleted_at)
    anonymous = TestClient(app, follow_redirects=False)
    routes = _wrapped_session_routes(app)
    assert len(routes) >= 15, "the census must see the session routes"
    for route in routes:
        url = (
            route.path.replace("{session_id:str}", TEMP)
            .replace("{job_id:str}", "3f9c1a7e5b20")
            .replace("{digest:str}", "0" * 32)
            .replace("{job:str}", "3f9c1a7e5b20")
        )
        assert "{" not in url, url
        method = sorted(set(route.methods or {"GET"}) - {"HEAD"})[0].lower()
        response = getattr(anonymous, method)(url)
        assert response.status_code == 401, (route.path, response.status_code)
    assert reads == [], "an anonymous request read the tombstone"
    # The families whose refusals carry a policy keep it through the wrapper.
    sound = anonymous.get(f"/api/sessions/{TEMP}/sound", params={"digest": "0" * 64})
    assert sound.headers["x-content-type-options"] == "nosniff"
    assert sound.headers["cross-origin-resource-policy"] == "same-origin"


def test_every_wrapped_session_route_gates_before_anything_else(relay) -> None:
    # The invariant the wrapper's anonymous path relies on: each wrapped
    # endpoint's FIRST statement is ``denied = gate(request)`` and its second
    # returns on a refusal, so handing it a refused request runs nothing else.
    import ast
    import inspect
    import textwrap

    _client, _cfg, _daemon, app = relay
    for route in _wrapped_session_routes(app):
        endpoint = inspect.unwrap(route.endpoint)
        tree = ast.parse(textwrap.dedent(inspect.getsource(endpoint)))
        body = tree.body[0].body  # type: ignore[attr-defined]
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        first, second = body[0], body[1]
        assert ast.unparse(first) == "denied = gate(request)", (route.path, ast.unparse(first))
        assert isinstance(second, ast.If), route.path
        assert ast.unparse(second.test) == "denied is not None", route.path
        assert len(second.body) == 1 and isinstance(second.body[0], ast.Return), route.path


def test_an_expiry_elsewhere_repaints_the_list_without_the_chat(relay) -> None:
    # C-1 + C-10 on the list stream (``api_list_events``): an expiry by any
    # host's sweeper rings the list, and the chat the table still held is gone
    # from the frame it publishes.
    _client, cfg, daemon, app = relay
    _live_entry(daemon, TEMP)

    async def run() -> tuple[set[str], set[str], int, int]:
        await daemon.refresh_temporary_if_changed()  # first pass: the baseline
        before = _ids((await _first_sessions_frame(app, b"include_temporary=1"))["sessions"])
        revision = daemon.table.list_revision
        _reap(cfg, TEMP, time.time())
        await daemon.refresh_temporary_if_changed()
        after = _ids((await _first_sessions_frame(app, b"include_temporary=1"))["sessions"])
        return before, after, revision, daemon.table.list_revision

    before, after, revision, rung = asyncio.run(run())
    assert TEMP in before
    assert rung > revision, "the expiry rang the list stream"
    assert TEMP not in after and PLAIN in after
    assert not (cfg / "sessions" / TEMP).exists()


def test_a_mark_or_keep_from_another_surface_rings_the_list(relay) -> None:
    # C-10: a mark restores its directory mtime, so only the store-level stamp
    # tells an idle phone that a chat turned temporary or was kept.
    _client, cfg, daemon, _app = relay

    async def ring_after(change) -> bool:  # noqa: ANN001
        await daemon.refresh_temporary_if_changed()
        revision = daemon.table.list_revision
        change()
        await daemon.refresh_temporary_if_changed()
        return daemon.table.list_revision > revision

    assert asyncio.run(ring_after(lambda: temporary.write_mark(cfg, TEMP, False, by="desktop")))
    assert asyncio.run(ring_after(lambda: temporary.write_mark(cfg, PLAIN, True, by="tui")))
    assert not asyncio.run(ring_after(lambda: None)), "no change, no repaint"


def test_temporary_rows_ride_on_top_of_the_hundred_row_window(tmp_path: Path) -> None:
    # C-11 (review R1-m2): a temporary chat used to push the oldest ordinary
    # chat off the phone's 100-row home list.
    import os

    from local_operator.mobile.daemon import DURABLE_WINDOW_ROWS, _durable_window_rows

    base = time.time() - 10_000
    ordinary = [f"{n:012x}" for n in range(DURABLE_WINDOW_ROWS + 1)]
    temps = [f"f{n:011x}" for n in range(3)]
    for index, session_id in enumerate(ordinary + temps):
        directory = tmp_path / "sessions" / session_id
        directory.mkdir(parents=True)
        transcript = directory / "transcript.jsonl"
        transcript.write_text('{"type":"message"}\n')
        stamp = base + index
        os.utime(transcript, (stamp, stamp))
    for session_id in temps:  # the newest rows
        assert temporary.write_mark(tmp_path, session_id, True, by="relay").ok
    rows = _durable_window_rows(tmp_path)
    plain = [row.id for row in rows if not row.temporary]
    assert len(plain) == DURABLE_WINDOW_ROWS
    assert {row.id for row in rows if row.temporary} == set(temps)
    assert ordinary[0] not in plain and ordinary[1] in plain


def test_a_keyed_start_marks_before_spawning_and_a_replay_never_re_marks(
    relay, monkeypatch
) -> None:
    # C-9 (review R1-m4): the mark is written before the runtime spawns, and
    # a replay that arrives after a Keep from another surface leaves it kept.
    client, cfg, daemon, _app = relay
    seen_marked: list[bool] = []

    async def fake_spawn(cwd, *, resume=None, **_kwargs):  # noqa: ANN001, ANN202
        seen_marked.append(temporary.is_marked(cfg, resume))
        return 4243

    async def recover(_sid):  # noqa: ANN001, ANN202
        return 4243

    monkeypatch.setattr(daemon, "spawn_session", fake_spawn)
    monkeypatch.setattr(daemon, "recover_started_session", recover)
    monkeypatch.setattr(
        "local_operator.mobile.daemon._resolve_start_model",
        lambda provider, model: ("fixture", "fixture-model"),
    )
    body = {"create_request_id": "0" * 31 + "1", "cwd": str(cfg), "temporary": True}
    first = client.post("/api/sessions/start", json=body)
    assert first.status_code == 200, first.text
    session_id = first.json()["session_id"]
    assert seen_marked == [True], "the runtime was spawned before the mark landed"
    assert temporary.write_mark(cfg, session_id, False, by="desktop").ok  # kept elsewhere
    replay = client.post("/api/sessions/start", json=body)
    assert replay.status_code == 200, replay.text
    assert replay.json()["session_id"] == session_id
    assert not temporary.is_marked(cfg, session_id), "the replay re-marked a kept chat"
    assert len(seen_marked) == 1
