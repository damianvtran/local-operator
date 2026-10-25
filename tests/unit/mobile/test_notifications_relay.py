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

    seen_item = _store().notify(SID, title="in view", anchor_id="call_view")
    # The receipt is EVIDENCE, and the store re-checks it: the desktop rung
    # waits a second after the visibility edge, and no edge may predate the
    # row's last content change. So the fixture makes the row, waits out the
    # dwell, and then claims an input edge that happened after the row existed
    # -- the real ordering, not a timestamp picked to pass.
    time.sleep(1.3)
    visible_since = int(time.time() * 1000) - 1_100
    input_at = int(time.time() * 1000)
    # ...and the receipt itself lands one dwell later, which is the rule the
    # store reads: visibility, then a fresh input edge, then a second of it.
    time.sleep(1.1)
    receipt = client.post(
        "/api/notifications/seen",
        json={
            "items": [{"id": seen_item.item_id, "revision": 1, "anchor_id": "call_view"}],
            "evidence": {
                "surface": "desktop",
                "session_id": SID,
                "input_at_ms": input_at,
                "visible_since_ms": visible_since,
            },
        },
    ).json()
    assert receipt["changed"] == [seen_item.item_id], receipt


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
