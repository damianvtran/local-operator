"""Conversation tags on the phone wire: ``/api/tags`` and per-session assignment.

WHAT THIS FILE PINS. The mobile relay is a THIRD client of the durable tag
store the desktop app and the TUI already share (``session.session_tags``): it
projects the store onto the session list, the list SSE frame and the past/search
rows, advertises that its routes exist through ``capabilities.session_tags``,
and accepts one desired-state assignment per conversation. Those are four
independent ways to be half-right, so every cell is asserted against the REAL
``build_app(MobileDaemon)`` under the isolated config root conftest installs --
the same construction ``test_projects_routes.py`` uses -- rather than against a
mocked projection.

WHY THE ALWAYS-PRESENT rule is tested on untagged rows too. A client cannot
tell "this conversation has no tags" from "this relay is too old to send tags"
when the key is absent, so absence would silently force every client to keep a
second code path forever. Empty is ``[]``/``None`` and it is asserted on the
list, the stream, past and search.

The store itself (validation sentences, identity reuse, orphan pruning) has its
own suite; this file asserts what the PHONE is told -- status codes, shapes and
the exact store sentences crossing the seam unchanged.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.requests import Request
from starlette.testclient import TestClient

from local_operator.harness.types import Message
from local_operator.mobile import mesh as mobile_mesh
from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.paths import config_dir
from local_operator.session import session_tags
from local_operator.session.transcript import Transcript

PASSWORD = "pw123"
SESSION_A = "aaaa11112222"
SESSION_B = "bbbb33334444"
REMOTE_SESSION = "dddd77778888"


def _login() -> TestClient:
    """A logged-in client over the real daemon app, under the test's HOME."""
    client = TestClient(build_app(MobileDaemon(port=0, password=PASSWORD)), follow_redirects=False)
    assert client.post("/login", data={"password": PASSWORD}).status_code in (200, 303)
    return client


def _seed_session(session_id: str, text: str = "alpha conversation") -> None:
    """A durable user conversation: a directory with a real transcript turn.

    Built through ``Transcript`` rather than by writing a JSON line by hand, so
    the row the listing serves is the store's own parse of a conversation, not
    a fixture the route could only agree with by construction.
    """
    directory = config_dir() / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    asyncio.run(transcript.append_message(Message.user(text, id="u-000")))
    asyncio.run(transcript.append_message(Message.assistant("ack", id="a-000")))


def _catalogue(*names: str) -> list[dict[str, str]]:
    return session_tags.write_categories(config_dir(), [(None, name) for name in names])


def _rows(client: TestClient) -> dict[str, dict[str, Any]]:
    listed = client.get("/api/sessions")
    assert listed.status_code == 200
    return {row["session_id"]: row for row in listed.json()["sessions"]}


def _put_tags(client: TestClient, session_id: str, tags: list[str], primary: str | None):
    return client.put(f"/api/sessions/{session_id}/tags", json={"tags": tags, "primary": primary})


async def _first_sessions_frame(app: Any, query: bytes) -> dict[str, Any]:
    """The first ``sessions`` frame off the real SSE endpoint.

    The endpoint object is driven directly (the helper ``test_mesh_relay`` uses)
    because an infinite EventSource is not a request a sync TestClient can
    finish: this reads ONE frame and closes the iterator, which is the whole
    contract under test.
    """
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


# ---------------------------------------------------------------------------
# The gate and the capability
# ---------------------------------------------------------------------------


def test_the_tag_routes_sit_behind_the_login_and_origin_gates() -> None:
    anonymous = TestClient(
        build_app(MobileDaemon(port=0, password=PASSWORD)), follow_redirects=False
    )
    assert anonymous.get("/api/tags").status_code == 401
    assert _put_tags(anonymous, SESSION_A, [], None).status_code == 401

    client = _login()
    assert client.get("/api/tags").status_code == 200
    # The same cross-origin mutation gate as the pin route, which is what makes
    # a sibling tunnel subdomain's simple POST land as a refusal, not a write.
    cross_site = client.put(
        f"/api/sessions/{SESSION_A}/tags",
        json={"tags": [], "primary": None},
        headers={"origin": "https://evil.example"},
    )
    assert cross_site.status_code == 403


def test_the_list_frame_advertises_the_routes_this_build_serves() -> None:
    client = _login()
    capabilities = client.get("/api/sessions").json()["capabilities"]
    assert capabilities["session_tags"] == {"available": True}
    # The distinction that keeps the two keys honest: ``features`` is core's
    # contract and is untouched; the relay's capability is a separate fact.
    assert capabilities["features"]["session_tags"] == 1


# ---------------------------------------------------------------------------
# Rows: always present on every local listing path
# ---------------------------------------------------------------------------


def test_untagged_rows_carry_empty_fields_on_list_stream_past_and_search() -> None:
    _seed_session(SESSION_A)
    client = _login()

    row = _rows(client)[SESSION_A]
    assert row["tags"] == [] and row["primary"] is None

    past = client.get("/api/sessions/past").json()["sessions"]
    assert [(r["tags"], r["primary"]) for r in past] == [([], None)]

    found = client.get("/api/sessions/search?q=alpha").json()["sessions"]
    assert found and [(r["tags"], r["primary"]) for r in found] == [([], None)]

    frame = asyncio.run(
        _first_sessions_frame(build_app(MobileDaemon(port=0, password=PASSWORD)), b"")
    )
    assert [(r["tags"], r["primary"]) for r in frame["sessions"]] == [([], None)]


def test_tag_rows_carry_canonical_names_on_list_stream_past_and_search() -> None:
    _seed_session(SESSION_A, text="alpha conversation")
    review, ops = _catalogue("Review", "Ops")
    session_tags.set_entity_tags(
        config_dir(), "session", SESSION_A, [ops["id"], review["id"]], review["id"]
    )
    expected = [
        {"id": ops["id"], "name": "Ops"},
        {"id": review["id"], "name": "Review"},
    ]

    client = _login()
    row = _rows(client)[SESSION_A]
    assert row["tags"] == expected and row["primary"] == review["id"]

    past = {r["id"]: r for r in client.get("/api/sessions/past").json()["sessions"]}
    assert past[SESSION_A]["tags"] == expected and past[SESSION_A]["primary"] == review["id"]

    found = client.get("/api/sessions/search?q=alpha").json()["sessions"]
    assert found[0]["tags"] == expected and found[0]["primary"] == review["id"]

    frame = asyncio.run(
        _first_sessions_frame(build_app(MobileDaemon(port=0, password=PASSWORD)), b"")
    )
    streamed = {r["session_id"]: r for r in frame["sessions"]}[SESSION_A]
    assert streamed["tags"] == expected and streamed["primary"] == review["id"]


def test_remote_rows_are_listed_untagged_and_the_write_is_refused(monkeypatch) -> None:
    """A peer's row has no local assignment file: empty, and not assignable.

    The row is decorated EXPLICITLY empty in ``_list_frame`` -- see its comment
    -- so a client's render rule is one shape for both planes. The relay never
    offers the edit; a caller that sends it anyway gets the same 404 an unknown
    local id earns, because no directory here can hold the assignment.
    """
    monkeypatch.setattr(
        mobile_mesh,
        "remote_session_rows",
        lambda *args, **kwargs: [
            {"session_id": REMOTE_SESSION, "section": "active", "pinned": False}
        ],
    )
    client = _login()
    listed = client.get("/api/sessions?include_peers=true").json()["sessions"]
    remote = next(row for row in listed if row["session_id"] == REMOTE_SESSION)
    assert remote["tags"] == [] and remote["primary"] is None

    review = _catalogue("Review")
    refused = _put_tags(client, REMOTE_SESSION, [review[0]["id"]], None)
    assert refused.status_code == 404
    assert refused.json() == {"error": "unknown session"}


# ---------------------------------------------------------------------------
# Assignment: desired state, canonical read-back, atomic refusals
# ---------------------------------------------------------------------------


def test_assignment_round_trip_preserves_order_names_and_clears() -> None:
    _seed_session(SESSION_A)
    review, ops = _catalogue("Review", "Ops")
    client = _login()

    assigned = _put_tags(client, SESSION_A, [ops["id"], review["id"]], review["id"])
    assert assigned.status_code == 200
    assert assigned.json() == {
        "session_id": SESSION_A,
        "tags": [
            {"id": ops["id"], "name": "Ops"},
            {"id": review["id"], "name": "Review"},
        ],
        "primary": review["id"],
    }

    # The assignment the list serves is the store's, not a route-local copy.
    row = _rows(client)[SESSION_A]
    assert row["tags"] == assigned.json()["tags"] and row["primary"] == review["id"]

    # Assignment HISTORY owns the order: a desired state naming the tags the
    # other way round cannot reorder a conversation's existing assignment.
    replay = _put_tags(client, SESSION_A, [review["id"], ops["id"]], ops["id"])
    assert [tag["id"] for tag in replay.json()["tags"]] == [ops["id"], review["id"]]
    assert replay.json()["primary"] == ops["id"]

    cleared = _put_tags(client, SESSION_A, [], None)
    assert cleared.status_code == 200
    assert cleared.json() == {"session_id": SESSION_A, "tags": [], "primary": None}
    assert session_tags.read_entity_tags(config_dir(), "session", SESSION_A) == {
        "tags": [],
        "primary": None,
    }
    row = _rows(client)[SESSION_A]
    assert row["tags"] == [] and row["primary"] is None


def test_store_refusals_are_422_and_leave_the_store_byte_identical() -> None:
    _seed_session(SESSION_A)
    tags = _catalogue(*[f"Tag {index}" for index in range(9)])
    ids = [tag["id"] for tag in tags]
    client = _login()
    assert _put_tags(client, SESSION_A, [ids[0]], ids[0]).status_code == 200

    path = config_dir() / "tags" / "session" / f"{SESSION_A}.json"
    before = path.read_bytes()
    revision = session_tags.read_tags_revision(config_dir())

    cases = [
        (["ffffffffff"], None, "Unknown category id"),
        ([ids[0], ids[0]], None, "At most 8 distinct tags are allowed"),
        (ids, ids[0], "At most 8 distinct tags are allowed"),
        ([ids[0]], ids[1], "Primary must be one of the assigned tags or null"),
    ]
    for tag_ids, primary, sentence in cases:
        refused = _put_tags(client, SESSION_A, tag_ids, primary)
        assert refused.status_code == 422, (tag_ids, primary)
        assert refused.json() == {"error": sentence}
        assert path.read_bytes() == before, "a 422 must not touch the assignment file"

    malformed = _put_tags(client, SESSION_A, "not-a-list", None)  # type: ignore[arg-type]
    assert malformed.status_code == 422
    assert set(malformed.json()) == {"error"}
    assert path.read_bytes() == before
    assert session_tags.read_tags_revision(config_dir()) == revision


def test_unknown_and_malformed_ids_are_the_same_404() -> None:
    client = _login()
    for session_id in ("0123456789ab", "not-a-session-id", "ZZZZZZZZZZZZ", REMOTE_SESSION):
        refused = _put_tags(client, session_id, [], None)
        assert refused.status_code == 404, session_id
        assert refused.json() == {"error": "unknown session"}


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


def test_the_catalogue_is_read_only_and_counts_distinct_conversations() -> None:
    review, ops = _catalogue("Review", "Ops")
    _seed_session(SESSION_A)
    _seed_session(SESSION_B)
    session_tags.set_entity_tags(config_dir(), "session", SESSION_A, [review["id"]], review["id"])
    session_tags.set_entity_tags(config_dir(), "session", SESSION_B, [review["id"]], review["id"])
    # An assignment below a DELETED conversation is not a conversation.
    session_tags.set_entity_tags(
        config_dir(), "session", "cccc55556666", [review["id"]], review["id"]
    )

    client = _login()
    assert client.get("/api/tags").json() == {
        "tags": [
            {"id": review["id"], "name": "Review", "order": 0, "conversations": 2},
            {"id": ops["id"], "name": "Ops", "order": 1, "conversations": 0},
        ]
    }
    # Read-only by construction: there is no phone-side create/rename verb.
    assert client.put("/api/tags", json={"tags": []}).status_code == 405


# ---------------------------------------------------------------------------
# The change marker: external writes reach the list without a per-tick scan
# ---------------------------------------------------------------------------


def _warm(daemon: MobileDaemon) -> dict[str, dict[str, Any]]:
    snapshot = daemon.table.tags_snapshot()
    # The daemon's own scan-tick seeding: nothing has moved since the warm read.
    asyncio.run(daemon.refresh_tags_if_changed())
    return snapshot


def test_an_external_write_wakes_the_stream_and_invalidates_the_merge() -> None:
    _seed_session(SESSION_A)
    review = _catalogue("Review")[0]
    daemon = MobileDaemon(port=0, password=PASSWORD, dial_registrants=False)
    warm = _warm(daemon)
    assert warm == {}

    daemon.table._summaries_cache = [{"session_id": SESSION_A}]
    daemon.table._summaries_at = 1.0
    queue: asyncio.Queue[None] = asyncio.Queue()
    daemon.table.list_subscribers.add(queue)

    # A write from the DESKTOP's writer -- never this daemon's route.
    session_tags.set_entity_tags(config_dir(), "session", SESSION_A, [review["id"]], review["id"])
    asyncio.run(daemon.refresh_tags_if_changed())

    assert daemon.table._summaries_cache is None, "the merge cache must be dropped"
    assert daemon.table._summaries_at == 0.0
    assert not queue.empty(), "the list stream must be woken for another surface's write"
    assert daemon.table.tags_snapshot()[SESSION_A] == {
        "tags": [{"id": review["id"], "name": "Review"}],
        "primary": review["id"],
    }


def test_an_unchanged_store_costs_no_scan_and_no_repaint() -> None:
    _seed_session(SESSION_A)
    review = _catalogue("Review")[0]
    session_tags.set_entity_tags(config_dir(), "session", SESSION_A, [review["id"]], review["id"])
    daemon = MobileDaemon(port=0, password=PASSWORD, dial_registrants=False)
    snapshot = daemon.table.tags_snapshot()
    # Seed the daemon's published marker the way its first scan tick would,
    # before a client is watching; the repaint under test is the SECOND pass.
    asyncio.run(daemon.refresh_tags_if_changed())

    queue: asyncio.Queue[None] = asyncio.Queue()
    daemon.table.list_subscribers.add(queue)
    asyncio.run(daemon.refresh_tags_if_changed())

    assert daemon.table.tags_snapshot() is snapshot, "an unchanged marker must reuse the cache"
    assert queue.empty(), "an unchanged marker must not repaint"
    assert daemon.table.tags_snapshot()[SESSION_A]["tags"] == [
        {"id": review["id"], "name": "Review"}
    ]


def test_a_no_op_write_does_not_move_the_marker() -> None:
    _seed_session(SESSION_A)
    review = _catalogue("Review")[0]
    session_tags.set_entity_tags(config_dir(), "session", SESSION_A, [review["id"]], review["id"])
    daemon = MobileDaemon(port=0, password=PASSWORD, dial_registrants=False)
    snapshot = daemon.table.tags_snapshot()
    before = session_tags.read_tags_revision(config_dir())

    session_tags.set_entity_tags(config_dir(), "session", SESSION_A, [review["id"]], review["id"])

    assert session_tags.read_tags_revision(config_dir()) == before
    asyncio.run(daemon.refresh_tags_if_changed())
    assert daemon.table.tags_snapshot() is snapshot


def test_an_older_store_without_the_marker_is_still_detected() -> None:
    """A store written before this build has no ``tags/revision`` file.

    Such a writer cannot bump a token it does not know about, so the marker
    falls back to the filesystem fingerprint of the two write sites. The write
    here is an OLD store's write -- a same-directory replace with no marker
    update -- and the watcher must still notice it without a 2 s sleep or a
    directory walk (the fingerprint is stats only).
    """
    import os

    _seed_session(SESSION_A)
    review = _catalogue("Review")[0]
    path = config_dir() / "tags" / "session" / f"{SESSION_A}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    (config_dir() / "tags" / "revision").unlink()

    daemon = MobileDaemon(port=0, password=PASSWORD, dial_registrants=False)
    assert _warm(daemon) == {}
    queue: asyncio.Queue[None] = asyncio.Queue()
    daemon.table.list_subscribers.add(queue)

    scratch = path.with_name(".legacy.tmp")
    scratch.write_text(json.dumps({"version": 1, "tags": [review["id"]], "primary": review["id"]}))
    os.replace(scratch, path)

    asyncio.run(daemon.refresh_tags_if_changed())
    assert not queue.empty()
    assert daemon.table.tags_snapshot()[SESSION_A]["tags"] == [
        {"id": review["id"], "name": "Review"}
    ]


def test_the_body_shape_is_exactly_two_keys() -> None:
    """Review round 1, MINOR-1: absent/extra keys are 422, never a default.

    ``body.get("primary")`` made a malformed request a real re-primary: the
    conversation's stored primary was overwritten with ``None`` and the client
    was told it succeeded. The desktop model refuses both shapes, and the
    relay's own ``_TAG_BODY_ERROR`` already promised both keys.
    """
    _seed_session(SESSION_A)
    review, ops = _catalogue("Review", "Ops")
    client = _login()
    assert _put_tags(client, SESSION_A, [review["id"]], review["id"]).status_code == 200
    path = config_dir() / "tags" / "session" / f"{SESSION_A}.json"
    before = path.read_bytes()
    revision = session_tags.read_tags_revision(config_dir())

    malformed = (
        {"tags": [review["id"], ops["id"]]},
        {"tags": [], "primary": None, "archived": True},
    )
    for body in malformed:
        refused = client.put(f"/api/sessions/{SESSION_A}/tags", json=body)
        assert refused.status_code == 422, body
        assert refused.json() == {
            "error": "tags (a list of category ids) and primary (an id or null) are required"
        }
        assert path.read_bytes() == before, "a refused body must not touch the assignment"
        assert session_tags.read_tags_revision(config_dir()) == revision


def test_a_live_entry_without_a_directory_is_not_admitted(monkeypatch) -> None:
    """Review round 1, MINOR-2: a live entry alone must not admit a write.

    The tag store prunes no orphans (unlike ``read_pins``), so admitting an id
    with no ``sessions/<id>/`` directory stored an assignment no surface lists.
    """
    monkeypatch.setattr(
        "local_operator.mobile.daemon._entry_for_session", lambda *args: object()
    )
    review = _catalogue("Review")[0]
    orphan = "eeee99990000"
    client = _login()
    refused = client.put(
        f"/api/sessions/{orphan}/tags", json={"tags": [review["id"]], "primary": review["id"]}
    )
    assert refused.status_code == 404
    assert refused.json() == {"error": "unknown session"}
    assert not (config_dir() / "tags" / "session" / f"{orphan}.json").exists()

    _seed_session(SESSION_A)
    accepted = _put_tags(client, SESSION_A, [review["id"]], review["id"])
    assert accepted.status_code == 200


def test_an_older_writer_that_does_not_bump_is_still_detected() -> None:
    """Review round 1, MINOR-3: stats are compared even when a token exists.

    An older build replaces an assignment without knowing the token; a
    revision-only marker would make that write invisible forever after the
    first marker-aware write created the file.
    """
    import os

    _seed_session(SESSION_A)
    review = _catalogue("Review")[0]
    assert session_tags.read_tags_revision(config_dir()) is not None
    session_tags.set_entity_tags(config_dir(), "session", SESSION_A, [], None)
    daemon = MobileDaemon(port=0, password=PASSWORD, dial_registrants=False)
    assert _warm(daemon)[SESSION_A] == {"tags": [], "primary": None}

    queue: asyncio.Queue[None] = asyncio.Queue()
    daemon.table.list_subscribers.add(queue)

    path = config_dir() / "tags" / "session" / f"{SESSION_A}.json"
    scratch = path.with_name(".legacy-writer.tmp")
    scratch.write_text(json.dumps({"version": 1, "tags": [review["id"]], "primary": review["id"]}))
    os.replace(scratch, path)  # the OLD writer's write: no marker bump

    asyncio.run(daemon.refresh_tags_if_changed())
    assert not queue.empty(), "a token-unaware assignment replace must still wake the list"
    assert daemon.table.tags_snapshot()[SESSION_A]["tags"] == [
        {"id": review["id"], "name": "Review"}
    ]


def test_a_concurrent_read_does_not_spend_the_wake() -> None:
    """QA round 1, Q-1: a read that rebuilds the projection must not swallow it.

    ``GET /api/sessions`` (or past/search) advances the table's cache token;
    the watcher must compare against what it last PUBLISHED, or an open
    SSE-only client is never told about the external write.
    """
    _seed_session(SESSION_A)
    review = _catalogue("Review")[0]
    daemon = MobileDaemon(port=0, password=PASSWORD, dial_registrants=False)
    assert _warm(daemon) == {}

    queue: asyncio.Queue[None] = asyncio.Queue()
    daemon.table.list_subscribers.add(queue)

    session_tags.set_entity_tags(config_dir(), "session", SESSION_A, [review["id"]], review["id"])
    # A concurrent request path rebuilds the projection first, consuming the
    # cache token -- exactly the read that used to suppress the wake.
    daemon.table.tags_snapshot()

    asyncio.run(daemon.refresh_tags_if_changed())
    assert not queue.empty(), "the read must not spend the watcher's wake"
    assert daemon.table.tags_snapshot()[SESSION_A]["tags"] == [
        {"id": review["id"], "name": "Review"}
    ]


def test_the_scan_tick_wakes_the_stream(monkeypatch) -> None:
    """Review round 1, MINOR-5: the wiring itself, not just the helper.

    Removing ``await self.refresh_tags_if_changed()`` from ``_scan_once`` must
    fail here: the tick is what makes an external write reach an idle phone.
    """
    from local_operator.session import attention

    _seed_session(SESSION_A)
    review = _catalogue("Review")[0]
    daemon = MobileDaemon(port=0, password=PASSWORD, dial_registrants=False)
    daemon._attention_bootstrapped = True
    monkeypatch.setattr(attention.AttentionStore, "revision", lambda self: "stable")
    daemon._attention_revision = "stable"

    # The first tick seeds the published marker; no subscriber exists yet.
    asyncio.run(daemon._scan_once())

    queue: asyncio.Queue[None] = asyncio.Queue()
    daemon.table.list_subscribers.add(queue)
    daemon.table._summaries_cache = [{"session_id": SESSION_A}]
    daemon.table._summaries_at = 1.0

    session_tags.set_entity_tags(config_dir(), "session", SESSION_A, [review["id"]], review["id"])
    asyncio.run(daemon._scan_once())

    assert not queue.empty(), "the scan tick must wake the list for an external write"
    assert daemon.table._summaries_cache is None, "the merge cache must be dropped"
    assert daemon.table._summaries_at == 0.0
