"""The relay's inbox wire (contract §5.4): the reel, one item, and the two writes.

The phone gets the SAME answers as the desktop routes, through thin adapters over
``notifications.items`` — so what is pinned here is the wire the phone reads: the
cookie gate, the ``observed`` token contract that makes a bulk write safe, the
``notifications`` block riding the list payload, and the per-row liveness the asks
wire already taught a client to expect.

The isolated config root is the one ``conftest`` installs; every id is synthetic
and no runtime is opened.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.notifications.store import NotificationStore
from local_operator.paths import config_dir

SID = "feed00000001"


def _client() -> TestClient:
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _store() -> NotificationStore:
    return NotificationStore(config_dir())


def test_the_inbox_routes_need_the_cookie() -> None:
    anonymous = TestClient(
        build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False
    )
    for method, path in (
        ("GET", "/api/notifications"),
        ("GET", "/api/notifications/n_aaaaaaaaaaaaaaaa"),
        ("PATCH", "/api/notifications"),
        ("POST", "/api/notifications/seen"),
    ):
        response = anonymous.request(method, path, json={})
        assert response.status_code == 401, (method, path)


def test_list_detail_patch_and_seen_round_trip() -> None:
    # A row is ``durable`` only over a conversation directory that survives a
    # restart, so the fixture creates one rather than asserting a flag about a
    # session that does not exist.
    # Durable means "a conversation a reader can still open": the desktop store
    # index reads a session dir, and the relay's own check (the one this flag
    # must agree with) additionally requires a transcript. So the fixture writes
    # the one the handler looks for rather than asserting on a bare directory.
    session_dir = Path(str(config_dir())) / "sessions" / SID
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "transcript.jsonl").write_text(
        '{"id":"e1","ts":1,"type":"message",'
        '"payload":{"kind":"message","role":"user","content":[{"text":"go"}]}}\n'
    )
    client = _client()
    item = _store().notify(
        SID, title="Outside PR needs review", body_md="b", priority="high", anchor_id="call_relay"
    )
    page = client.get("/api/notifications").json()
    assert [row["id"] for row in page["items"]] == [item.item_id]
    assert page["observed"] == [{"id": item.item_id, "revision": 1}]
    assert page["waiting_unread_count"] == 0
    row = page["items"][0]
    # The two flags the asks wire established: a durable row is not necessarily a
    # row whose conversation can answer right now.
    assert row["durable"] is True and row["runtime_live"] is False

    detail = client.get(f"/api/notifications/{item.item_id}")
    assert detail.status_code == 200
    assert detail.json()["item"]["title"] == "Outside PR needs review"
    assert client.get("/api/notifications/n_aaaaaaaaaaaaaaaa").status_code == 404
    assert client.get("/api/notifications/not-an-id").status_code == 422

    read = client.patch(
        "/api/notifications",
        json={"op": "read", "ids": [{"id": item.item_id, "revision": 1}]},
    ).json()
    assert read["changed"] == [item.item_id] and read["unread_count"] == 0

    # D11 (supersedes R6's blanket refusal): the phone's OWN receipt is
    # accepted — `native` surface, touch/press/scroll edges — and re-checked
    # against the store exactly like the desktop's. Round 1's negative is kept
    # below: a receipt that labels itself `desktop` is refused.
    seen_item = _store().notify(SID, title="in view", anchor_id="call_view")
    time.sleep(1.3)
    visible_since = int(time.time() * 1000) - 1_100
    input_at = int(time.time() * 1000)
    time.sleep(1.1)
    accepted = client.post(
        "/api/notifications/seen",
        json={
            "items": [{"id": seen_item.item_id, "revision": 1, "anchor_id": "call_view"}],
            "evidence": {
                "surface": "native",
                "session_id": SID,
                "input_kind": "touch",
                "input_at_ms": input_at,
                "visible_since_ms": visible_since,
            },
        },
    )
    assert accepted.status_code == 200, accepted.text
    body = accepted.json()
    assert body["changed"] == [seen_item.item_id], body
    row = _store().get(seen_item.item_id)
    assert row is not None
    assert row["read_via"] == "in_view" and row["read_surface"] == "native"
    assert row["read_rev"] == 1


def test_the_relay_refuses_a_receipt_that_claims_another_surface() -> None:
    """R6's negative, kept: this door is the native one, and only it (D11)."""
    client = _client()
    item = _store().notify(SID, title="not for a desktop claim", anchor_id="call_x")
    now = int(time.time() * 1000)
    for surface in ("desktop", "tui", ""):
        refusal = client.post(
            "/api/notifications/seen",
            json={
                "items": [{"id": item.item_id, "revision": 1, "anchor_id": "call_x"}],
                "evidence": {
                    "surface": surface,
                    "session_id": SID,
                    "input_kind": "touch",
                    "input_at_ms": now,
                    "visible_since_ms": now - 5_000,
                },
            },
        )
        assert refusal.status_code == 422, (surface, refusal.text)
        assert refusal.json()["code"] == "relay_surface_mismatch", surface
    row = _store().get(item.item_id)
    assert row is not None and row["read_via"] is None and row["read_rev"] == 0


def test_the_relay_refuses_a_focus_edge_and_stale_evidence() -> None:
    """D12 on the phone, and the desktop's per-item re-checks (D13)."""
    client = _client()
    item = _store().notify(SID, title="edges", anchor_id="call_e")
    now = int(time.time() * 1000)

    def receipt(**evidence: Any) -> Any:
        return client.post(
            "/api/notifications/seen",
            json={
                "items": [{"id": item.item_id, "revision": 1, "anchor_id": "call_e"}],
                "evidence": {
                    "surface": "native",
                    "session_id": SID,
                    "input_kind": "touch",
                    "input_at_ms": now,
                    "visible_since_ms": now - 5_000,
                    **evidence,
                },
            },
        )

    assert receipt(input_kind="focus").status_code == 422, "a focus gain is not an edge"
    # Stale visibility: before this row's own update.
    stale = receipt(visible_since_ms=now - 600_000)
    assert stale.status_code == 200 and stale.json()["changed"] == []
    assert stale.json()["skipped"][0]["reason"] == "stale_evidence"
    # Wrong revision, and the wrong session's claim. Both arrive as SKIPPED with
    # their own reason: a receipt is ANSWERED, never rejected wholesale, so the
    # phone can tell "this is not mine" from "this moved since I painted it".
    wrong_revision = client.post(
        "/api/notifications/seen",
        json={
            "items": [{"id": item.item_id, "revision": 2, "anchor_id": "call_e"}],
            "evidence": {
                "surface": "native",
                "session_id": SID,
                "input_kind": "touch",
                "input_at_ms": now,
                "visible_since_ms": now - 5_000,
            },
        },
    )
    assert wrong_revision.status_code == 200, wrong_revision.text
    assert wrong_revision.json()["skipped"][0]["reason"] == "stale_revision"
    other = receipt(session_id="0" * 12)
    assert other.status_code == 200, other.text
    assert other.json()["skipped"][0]["reason"] == "unknown"
    row = _store().get(item.item_id)
    assert row is not None and row["read_rev"] == 0


def test_the_relay_advertises_the_native_seen_door() -> None:
    """D11: native turns its receipt feed on from this key, never by probing.

    Cold: the capability payload's ``features`` dict IS ``_feature_flags()``
    (``/api/sessions`` is the door that carries it, and that door dials
    runtimes), so the fact is asserted where it is defined. The live wire is
    proven against a real ``lop mobile serve`` in the PR's evidence section.
    """
    from local_operator.mobile import daemon as daemon_module

    assert daemon_module._feature_flags()["notifications_seen"] == 1


def test_bulk_writes_need_the_observed_tokens_and_never_widen() -> None:
    client = _client()
    store = _store()
    first = store.notify(SID, title="first", priority="high")
    page = client.get("/api/notifications").json()
    # A bulk write with no observations is refused, not silently widened to the
    # whole inbox (the desktop route's contract, mirrored).
    for body in (
        {"op": "read", "all": True},
        {"op": "read", "all": True, "through_seq": page["high_water"]},
        {"op": "dismiss", "session_id": SID, "through_seq": page["high_water"]},
    ):
        assert client.patch("/api/notifications", json=body).status_code == 422, body
    # A revision written AFTER the list stays unread: the tokens bound the write.
    store.notify(SID, title="late", priority="high")
    marked = client.patch(
        "/api/notifications",
        json={
            "op": "read",
            "all": True,
            "through_seq": page["high_water"],
            "observed": page["observed"],
        },
    ).json()
    assert marked["changed"] == [first.item_id]
    assert marked["unread_count"] == 1


def test_the_list_payload_carries_the_notifications_block() -> None:
    """The badge rides `/api/sessions`, and reads clear half of it.

    ``waiting_unread_count`` is the ask half of the badge; ``waiting_count`` is
    every pending ask. Reading clears the first and must not remove the second,
    because an agent is still blocked on that question.
    """
    from tests.unit.notifications.test_inbox import _open_ask

    _open_ask(config_dir(), SID, "pending")
    _store().notify(SID, title="one", priority="high")
    block = _client().get("/api/sessions").json()["notifications"]
    assert block["unread_count"] == 1
    assert block["waiting_count"] == 1 and block["waiting_unread_count"] == 1

    client = _client()
    page = client.get("/api/notifications").json()
    client.patch(
        "/api/notifications",
        json={
            "op": "read",
            "all": True,
            "through_seq": page["high_water"],
            "observed": page["observed"],
        },
    )
    after = _client().get("/api/sessions").json()["notifications"]
    assert after["unread_count"] == 0
    assert after["waiting_count"] == 1 and after["waiting_unread_count"] == 0


def test_the_scan_doorbell_wakes_the_list_when_the_inbox_moves() -> None:
    """The badge rides the list stream, so the store must ring its bell.

    Exercised through the REAL scan pass (not the helper in isolation): the
    first pass seeds the fingerprint and publishes, a second pass with nothing
    written publishes nothing, and a write publishes again. The import of the
    store's path helper lives inside that pass, which is exactly where a wrong
    name survives every lint and only shows up here — this cell is why.
    """
    import asyncio

    daemon = MobileDaemon(port=0, password="pw123")
    queue: asyncio.Queue[None] = asyncio.Queue(maxsize=4)
    daemon.table.list_subscribers.add(queue)

    def scan() -> int:
        asyncio.run(daemon._scan_once())
        count = 0
        while not queue.empty():
            queue.get_nowait()
            count += 1
        return count

    assert scan() >= 1, "the first pass seeds and publishes"
    assert scan() == 0, "a quiet pass repaints nothing"
    _store().notify(SID, title="doorbell", priority="high")
    assert scan() == 1, "a write wakes the list exactly once"


# -- round 1 remediation: transport parity (R7, Q5) --------------------------

#: The bodies the desktop route refuses with 422 (reviewer R7, QA R23-R31/R33),
#: and the queries it refuses with 422. Each is listed as the client would send
#: it, so the test reads as the wire contract rather than as a model API.
BAD_PATCH_BODIES: list[tuple[str, dict[str, object]]] = [
    (
        "observed revision 0",
        {
            "op": "read",
            "all": True,
            "through_seq": 3,
            "observed": [{"id": "n_aaaaaaaaaaaaaaaa", "revision": 0}],
        },
    ),
    ("malformed id in ids", {"op": "read", "ids": [{"id": "n_BAD", "revision": 1}]}),
    (
        "malformed id in observed",
        {"op": "read", "all": True, "through_seq": 3, "observed": [{"id": "n_BAD", "revision": 1}]},
    ),
    ("negative through_seq", {"op": "read", "session_id": SID, "through_seq": -5, "observed": []}),
    (
        "malformed session_id",
        {"op": "read", "session_id": "../etc", "through_seq": 3, "observed": []},
    ),
]

BAD_LIST_QUERIES: list[tuple[str, dict[str, str]]] = [
    ("limit 0", {"limit": "0"}),
    ("limit 500", {"limit": "500"}),
    ("negative through_seq", {"through_seq": "-5"}),
    ("cursor of 200 characters", {"cursor": "9" * 200}),
]


def _models() -> Any:
    from local_operator.server.routes import desktop_notifications

    return desktop_notifications


def test_the_relay_refuses_what_the_desktop_route_refuses() -> None:
    """R7: ONE validation for both transports.

    The relay used to answer 200 where the desktop answered 422 for these nine
    requests. What is asserted is the parity itself: the same model the desktop
    route validates with rejects each one, and the relay's wire refuses it.
    """
    from pydantic import ValidationError

    models = _models()
    client = _client()
    for label, body in BAD_PATCH_BODIES:
        with pytest.raises(ValidationError):
            models.NotificationsPatch.model_validate(body)
        response = client.patch("/api/notifications", json=body)
        assert response.status_code == 422, (label, response.status_code, response.text)
    for label, query in BAD_LIST_QUERIES:
        with pytest.raises(ValidationError):
            models.NotificationsQuery.model_validate(query)
        response = client.get("/api/notifications", params=query)
        assert response.status_code == 422, (label, response.status_code, response.text)
    # The bodies the shared validator ACCEPTS still work over the wire, so the
    # test above is not passing because the relay refuses everything.
    assert client.get("/api/notifications", params={"limit": "5"}).status_code == 200


def test_the_relay_detail_carries_the_transcript_anchor(tmp_path: Path) -> None:
    """R7: ``anchor_excerpt`` reached the desktop detail and not the relay's.

    Without it the phone's detail always painted ``anchor: null`` for a call the
    desktop could show in context.
    """
    import json as jsonlib

    session_dir = Path(str(config_dir())) / "sessions" / SID
    session_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "type": "message",
        "payload": {
            "role": "tool",
            "tool_call_id": "call_anchor",
            "content": [{"text": "Notified (id n_fixture)."}],
        },
    }
    (session_dir / "transcript.jsonl").write_text(jsonlib.dumps(row) + "\n")
    item = _store().notify(SID, title="with anchor", priority="high", anchor_id="call_anchor")
    detail = _client().get(f"/api/notifications/{item.item_id}").json()
    assert detail["anchor"] == {
        "id": "call_anchor",
        "excerpt": "Notified (id n_fixture).",
    }


def test_an_unreadable_store_answers_the_desktops_refusal() -> None:
    """Q5: a store failure is a classified refusal, never a bare 500.

    The phone needs the code to word it (``store_unavailable``), and the raw
    Python message -- which names a filesystem path -- must not be echoed.
    """
    store = _store()
    item = store.notify(SID, title="x", priority="high")
    client = _client()
    store.path.chmod(0o000)
    try:
        response = client.get("/api/notifications")
        detail = client.get(f"/api/notifications/{item.item_id}")
    finally:
        store.path.chmod(0o600)
    assert response.status_code == 500, response.text
    body = response.json()
    assert body["code"] == "store_unavailable"
    assert body["error"] and "sqlite" not in body["error"].lower()
    assert detail.status_code == 500 and detail.json()["code"] == "store_unavailable"


# -- round 1 remediation: transport parity (R7, Q5) --------------------------


#: The cursor is bounded in LENGTH and in MAGNITUDE. The length bound alone was
#: the round-2 defect: 79 digits fit, and sqlite could not bind them.
CURSOR_SHAPES: list[tuple[str, str]] = [
    ("9 x 200 characters", "9" * 200),
    ("9 x 79 characters", "9" * 79),
    ("9 x 40 characters", "9" * 40),
    ("int64 max + 1", "9223372036854775808.n_aaaaaaaaaaaaaaaa"),
    ("negative, out of range", "-" + "9" * 30),
]


def test_an_out_of_range_cursor_is_refused_on_the_wire() -> None:
    """QA round 2: a magnitude bound, not a longer length bound.

    Each shape below answered 500 with ``OverflowError: Python int too large to
    convert to SQLite INTEGER`` (store.py, ``list_items``) on the previous head;
    the length bound refused only the first. A NON-NUMERIC cursor stays the
    documented "from the top" case and must keep answering 200.
    """
    client = _client()
    for label, cursor in CURSOR_SHAPES:
        response = client.get("/api/notifications", params={"cursor": cursor})
        assert response.status_code == 422, (label, response.status_code, response.text)
    # The boundary itself is a real stamp: 2**63 - 1 is bindable.
    boundary = client.get(
        "/api/notifications", params={"cursor": "9223372036854775807.n_aaaaaaaaaaaaaaaa"}
    )
    assert boundary.status_code == 200, boundary.text
    # ...and malformed-by-contract still means "from the top", never an error.
    top = client.get("/api/notifications", params={"cursor": "not-a-cursor"})
    assert top.status_code == 200, top.text


def test_the_store_refuses_an_unbindable_stamp_without_raising() -> None:
    """The second line of defence: no caller can hand sqlite an out-of-range stamp."""
    from local_operator.notifications.items import list_page
    from local_operator.notifications.store import NotificationStore, _decode_cursor

    for _, cursor in CURSOR_SHAPES:
        assert _decode_cursor(cursor) is None, cursor
    assert _decode_cursor("1.n_aaaaaaaaaaaaaaaa") == (1, "n_aaaaaaaaaaaaaaaa")
    # And the page itself is unaffected by the value that used to kill it.
    NotificationStore(config_dir()).notify(SID, title="x", priority="high")
    page = list_page(config_dir(), cursor="9" * 79)
    assert page["items"], "an unbindable cursor reads from the top rather than raising"
