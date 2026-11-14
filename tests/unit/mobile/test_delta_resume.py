"""Delta resume (mobile background-sync design, Phase 2a + 2b) on the relay.

Two additive contracts a returning phone uses to avoid re-downloading what it
already holds:

- **2b, forward history** — ``GET /api/sessions/{id}/history?after=<row id>``
  answers the rows strictly NEWER than that row, oldest first, with
  ``has_more``; a cursor the relay cannot place is ``409 {"reset": true}`` and
  never an empty 200 (which a forward reader would take as "already current").
- **2a, seed elision** — ``?have=<digest>`` on both SSE streams. Absent: the
  legacy byte-identical seed. Present: frames carry ``digest``, and a matching
  ``have`` turns the seed into a tiny ``unchanged`` event.

Both are advertised as relay capabilities (``history_after``, ``seed_digest``)
on the list frame, because an older relay silently ignores both parameters.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from local_operator.harness.types import Message
from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.mobile.daemon import (
    MobileDaemon,
    _frame_digest,
    _projection_frame,
    _projection_seed_event,
    build_app,
)
from local_operator.session.transcript import Transcript

PASSWORD = "pw123"
SESSION_ID = "delta-session"


def _write_session(config: Path, turns: int, *, session_id: str = SESSION_ID) -> list[str]:
    """A plain durable conversation of ``turns`` user/assistant pairs; returns
    the row ids in journal order."""
    directory = config / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    ids: list[str] = []

    async def run() -> None:
        for turn in range(turns):
            user = Message.user(f"user {turn}", id=f"u-{turn:03d}")
            answer = Message.assistant(f"answer {turn}", id=f"a-{turn:03d}")
            await transcript.append_message(user)
            await transcript.append_message(answer)
            ids.extend([user.id, answer.id])

    asyncio.run(run())
    return ids


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    return cfg


def _client() -> TestClient:
    client = TestClient(
        build_app(MobileDaemon(port=0, password=PASSWORD, dial_registrants=False)),
        follow_redirects=False,
    )
    client.post("/login", data={"password": PASSWORD})
    return client


# --- 2b: forward history -----------------------------------------------------


def test_after_returns_only_newer_rows_in_order(config: Path) -> None:
    ids = _write_session(config, 6)
    body = _client().get(f"/api/sessions/{SESSION_ID}/history", params={"after": ids[3]}).json()
    assert [row["id"] for row in body["entries"]] == ids[4:]
    assert body["has_more"] is False
    assert "reset" not in body


def test_after_at_the_head_is_an_empty_but_valid_page(config: Path) -> None:
    """The phone is already current: 200, no rows, no reset."""
    ids = _write_session(config, 3)
    response = _client().get(f"/api/sessions/{SESSION_ID}/history", params={"after": ids[-1]})
    assert response.status_code == 200
    assert response.json() == {"entries": [], "has_more": False}


def test_after_an_unknown_id_is_a_409_reset_never_an_empty_page(config: Path) -> None:
    _write_session(config, 3)
    response = _client().get(f"/api/sessions/{SESSION_ID}/history", params={"after": "no-such-row"})
    assert response.status_code == 409
    body = response.json()
    assert body["reset"] is True
    assert "entries" not in body


def test_after_pages_forward_by_limit_until_has_more_is_false(config: Path) -> None:
    ids = _write_session(config, 10)
    client = _client()
    cursor = ids[0]
    served: list[str] = []
    requests = 0
    while True:
        requests += 1
        body = client.get(
            f"/api/sessions/{SESSION_ID}/history", params={"after": cursor, "limit": 3}
        ).json()
        page = [row["id"] for row in body["entries"]]
        assert len(page) <= 3
        served.extend(page)
        if not body["has_more"]:
            break
        assert len(page) == 3, "a page short of the limit cannot claim more"
        cursor = page[-1]
    assert served == ids[1:]
    assert requests == 7  # 19 rows / 3 per page, rounded up


def test_after_and_before_together_are_refused(config: Path) -> None:
    ids = _write_session(config, 3)
    response = _client().get(
        f"/api/sessions/{SESSION_ID}/history",
        params={"after": ids[1], "before": ids[3]},
    )
    assert response.status_code == 400


def test_after_on_a_session_with_no_transcript_resets(config: Path) -> None:
    """A durable route whose journal is gone must not answer "current"."""
    from local_operator.mobile.daemon import _history_page

    page, has_more = _history_page("never-written", None, 10, after="u-000")
    assert getattr(page, "reset", "") and page == [] and has_more is False


def test_after_a_compaction_prefix_row_resets(config: Path) -> None:
    """A cursor naming a preserved turn re-injected at the render's head.

    Those rows are journal rows from BELOW the cut, so "the rows after it in the
    render" would hand back older kept-window rows as newer. The relay must
    answer reset instead.
    """
    directory = config / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    transcript = Transcript(directory)

    async def run() -> None:
        for i in range(3):
            await transcript.append_message(Message.user(f"old user {i}", id=f"old-u{i}"))
            await transcript.append_message(Message.assistant(f"old answer {i}", id=f"old-a{i}"))
        cut = Message.user("kept opens here", id="kept-u0")
        await transcript.append_message(cut)
        await transcript.append_compaction(
            "summary",
            cut.id,
            1000,
            preserved_user_turns=[{"id": "old-u2", "text": "old user 2"}],
            preserved_turns_cap=40_000,
        )
        await transcript.append_message(Message.assistant("kept answer", id="kept-a0"))

    asyncio.run(run())
    client = _client()
    # Negative control: the preserved row IS in the render, so only the prefix
    # rule (not a plain lookup miss) can produce the reset below.
    from local_operator.mobile.daemon import _durable_fold_cache

    render_ids = [row.id for row in _durable_fold_cache().load(directory).render]
    assert "old-u2" in render_ids
    response = client.get(f"/api/sessions/{SESSION_ID}/history", params={"after": "old-u2"})
    assert response.status_code == 409 and response.json()["reset"] is True
    # A kept-window row still pages forward normally.
    kept = client.get(f"/api/sessions/{SESSION_ID}/history", params={"after": "kept-u0"}).json()
    assert [row["id"] for row in kept["entries"]] == ["kept-a0"]


def test_a_compaction_folded_incrementally_records_its_prefix(config: Path) -> None:
    """The tail path (``_rebuild_history_after_compaction``) — a live session
    compacting while the cache is warm — must record the prefix too, or the
    forward cursor would splice after a preserved turn on exactly the sessions
    a phone is watching."""
    from local_operator.mobile.daemon import _durable_fold_cache

    _write_session(config, 3)
    directory = config / "sessions" / SESSION_ID
    cache = _durable_fold_cache()
    assert cache.load(directory).prefix_ids == frozenset()
    transcript = Transcript(directory)

    async def run() -> None:
        await transcript.append_compaction(
            "summary",
            "u-002",
            1000,
            preserved_user_turns=[{"id": "u-001", "text": "user 1"}],
            preserved_turns_cap=40_000,
        )
        await transcript.append_message(Message.user("after", id="u-new"))

    asyncio.run(run())
    state = cache.load(directory)
    assert "u-001" in state.prefix_ids
    response = _client().get(f"/api/sessions/{SESSION_ID}/history", params={"after": "u-001"})
    assert response.status_code == 409


def test_history_without_after_is_unchanged(config: Path) -> None:
    """Legacy reads keep their shape: no ``reset`` key, newest page."""
    ids = _write_session(config, 3)
    body = _client().get(f"/api/sessions/{SESSION_ID}/history", params={"limit": 2}).json()
    assert [row["id"] for row in body["entries"]] == ids[-2:]
    assert set(body) == {"entries", "has_more"}


# --- 2a: seed elision -------------------------------------------------------


def test_digest_is_deterministic_and_ignores_clock_fields() -> None:
    frame = {"b": [1, 2], "a": "x", "version": 3, "activity_started_s": 1.5}
    reordered = {"activity_started_s": 9.0, "version": 7, "a": "x", "b": [1, 2]}
    clock = ("version", "activity_started_s")
    assert _frame_digest(frame, clock) == _frame_digest(reordered, clock)
    assert _frame_digest(frame, clock) != _frame_digest({**frame, "a": "y"}, clock)
    assert len(_frame_digest(frame, clock)) == 32


def _events_request(path: str, query: bytes, path_params: dict[str, str] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "path_params": path_params or {},
            "query_string": query,
            "headers": [
                (b"host", b"fixture"),
                (b"cookie", f"{COOKIE_NAME}={sign_cookie(PASSWORD)}".encode()),
            ],
            "scheme": "http",
            "server": ("fixture", 80),
        }
    )


async def _first_event(app: Any, route: str, path: str, query: bytes, **params: str) -> str:
    endpoint = next(r.endpoint for r in app.routes if getattr(r, "path", "") == route)
    response = await endpoint(_events_request(path, query, params))
    try:
        return await anext(response.body_iterator)
    finally:
        await response.body_iterator.aclose()


def _parse(event: str) -> tuple[str, dict[str, Any]]:
    name = next(line for line in event.splitlines() if line.startswith("event:"))
    data = next(line for line in event.splitlines() if line.startswith("data:"))
    return name[len("event:") :].strip(), json.loads(data[len("data:") :].strip())


SESSION_ROUTE = "/api/sessions/{session_id:str}/events"
LIST_ROUTE = "/api/sessions/events"


@pytest.mark.asyncio
async def test_session_seed_with_and_without_have(config: Path) -> None:
    # A worker thread: ``_write_session`` drives its own event loop.
    await asyncio.to_thread(_write_session, config, 4)
    daemon = MobileDaemon(port=0, password=PASSWORD, dial_registrants=False)
    app = build_app(daemon)
    path = f"/api/sessions/{SESSION_ID}/events"
    try:
        legacy = await _first_event(app, SESSION_ROUTE, path, b"", session_id=SESSION_ID)
        name, frame = _parse(legacy)
        assert name == "projection" and "digest" not in frame

        opted = await _first_event(app, SESSION_ROUTE, path, b"have=", session_id=SESSION_ID)
        name, stamped = _parse(opted)
        assert name == "projection"
        digest = stamped.pop("digest")
        # The stamped frame is the legacy frame plus ``digest`` and nothing else
        # (the clock fields may legitimately differ between two serializations).
        clock = ("version", "activity_started_s")
        assert {k: v for k, v in stamped.items() if k not in clock} == {
            k: v for k, v in frame.items() if k not in clock
        }

        match = await _first_event(
            app, SESSION_ROUTE, path, f"have={digest}".encode(), session_id=SESSION_ID
        )
        name, unchanged = _parse(match)
        assert name == "unchanged"
        assert unchanged["digest"] == digest
        assert set(unchanged) <= {"digest", "version", "activity_started_s"}
        assert len(match) < len(opted) // 4

        stale = await _first_event(
            app,
            SESSION_ROUTE,
            path,
            b"have=0123456789abcdef0123456789abcdef",
            session_id=SESSION_ID,
        )
        name, full = _parse(stale)
        assert name == "projection" and full["digest"] == digest
    finally:
        await daemon.close_phone_views()


def test_seed_event_without_have_is_byte_identical_to_the_legacy_event() -> None:
    from local_operator.mobile.daemon import _sse

    frame = {"session_id": "s", "version": 2, "transcript": [{"id": "x"}]}
    assert _projection_seed_event(frame, None) == _sse("projection", frame)


@pytest.mark.asyncio
async def test_list_seed_with_and_without_have(config: Path) -> None:
    app = build_app(MobileDaemon(port=0, password=PASSWORD, dial_registrants=False))
    legacy = await _first_event(app, LIST_ROUTE, LIST_ROUTE, b"")
    name, frame = _parse(legacy)
    assert name == "sessions" and "digest" not in frame

    opted = await _first_event(app, LIST_ROUTE, LIST_ROUTE, b"have=")
    name, stamped = _parse(opted)
    assert name == "sessions"
    digest = stamped.pop("digest")
    assert stamped == frame

    match = await _first_event(app, LIST_ROUTE, LIST_ROUTE, f"have={digest}".encode())
    assert _parse(match) == ("unchanged", {"digest": digest})

    stale = await _first_event(app, LIST_ROUTE, LIST_ROUTE, b"have=stale")
    name, full = _parse(stale)
    assert name == "sessions" and full["digest"] == digest


def test_capabilities_advertise_both_contracts(config: Path) -> None:
    capabilities = _client().get("/api/sessions").json()["capabilities"]
    assert capabilities["history_after"] == {"available": True}
    assert capabilities["seed_digest"] == {"available": True}


def test_projection_frame_digest_matches_what_the_stream_sends(config: Path) -> None:
    """The digest a phone holds is the one ``_frame_digest`` computes over the
    capped frame — the frame it actually received."""
    from local_operator.mobile.daemon import _durable_projection

    _write_session(config, 2)
    projection = _durable_projection(SESSION_ID)
    assert projection is not None
    frame = _projection_frame(projection)
    _, stamped = _parse(_projection_seed_event(frame, ""))
    assert stamped["digest"] == _frame_digest(frame, ("version", "activity_started_s"))
