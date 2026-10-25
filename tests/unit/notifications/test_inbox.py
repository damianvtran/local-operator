"""The operator's inbox (docs/design/notifications-inbox.md): store, rung 0, tool, routes, feed.

Cold-path throughout: no runtime is started and no provider is called. The
store and the aggregation are exercised against a ``tmp_path`` config root; the
routes through FastAPI's ``TestClient`` with a synthetic bearer; the feed by
driving its tick directly, the ``test_desktop_feed.py`` convention.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.notifications import presence
from local_operator.notifications.items import apply_patch, counts, detail, list_page
from local_operator.notifications.store import (
    INPUT_EVIDENCE_MS,
    NEW_ITEMS_PER_HOUR,
    NotificationStore,
    NotifyRefused,
    ask_ref,
)
from local_operator.tools import builtin
from local_operator.tools.registry import create_tools

#: A synthetic desktop bearer, never a real one (AGENTS.md isolation rules).
SYNTHETIC_TOKEN = "-".join(("synthetic", "inbox", "bearer"))
SID = "a1" * 6
OTHER = "b2" * 6


def _store(tmp_path: Path) -> NotificationStore:
    return NotificationStore(tmp_path)


# ---------------------------------------------------------------------------
# The store: collapse, dismiss, escalate, tokens (§1.3, §2.3)
# ---------------------------------------------------------------------------


def test_a_keyed_repeat_collapses_and_an_identical_one_writes_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.notify(SID, title="PR #7 needs review", body_md="one", key="pr-7")
    same = store.notify(SID, title="PR #7 needs review", body_md="one", key="pr-7")
    changed = store.notify(SID, title="PR #7 needs review", body_md="two", key="pr-7")
    assert first.action == "created"
    assert same.action == "unchanged" and same.item_id == first.item_id
    assert changed.action == "updated" and changed.revision == 2 and changed.repeat_count == 2
    assert [row["excerpt"] for row in store.history(first.item_id)] == ["one"]
    assert store.counts()[0] == 1


def test_an_update_makes_a_read_item_unread_again(tmp_path: Path) -> None:
    store = _store(tmp_path)
    item = store.notify(SID, title="Campaign", body_md="10 sent", key="c")
    store.apply("read", ids=[(item.item_id, 1)])
    assert store.counts()[0] == 0
    store.notify(SID, title="Campaign", body_md="20 sent", key="c")
    assert store.counts()[0] == 1


def test_a_dismissed_key_absorbs_silently_and_an_escalation_revives_it(tmp_path: Path) -> None:
    """The dismissed row is what keeps a 5-minute wake from flooding the reel."""
    store = _store(tmp_path)
    item = store.notify(SID, title="Fraud attempts", body_md="3", key="fraud")
    store.apply("dismiss", ids=[(item.item_id, 1)])
    absorbed = store.notify(SID, title="Fraud attempts", body_md="4", key="fraud")
    assert absorbed.action == "absorbed"
    assert store.get(item.item_id)["dismissed_at"] is not None
    assert store.counts()[0] == 0
    revived = store.notify(SID, title="Fraud attempts", body_md="40", key="fraud", priority="high")
    assert revived.action == "revived"
    row = store.get(item.item_id)
    assert row["dismissed_at"] is None and row["priority"] == "high"
    assert store.counts()[0] == 1


def test_a_write_naming_a_stale_revision_is_skipped_not_applied(tmp_path: Path) -> None:
    """R10: a keyed update published after the render is never cleared by it."""
    store = _store(tmp_path)
    item = store.notify(SID, title="Run", body_md="a", key="run")
    store.notify(SID, title="Run", body_md="b", key="run")
    result = store.apply("read", ids=[(item.item_id, 1)])
    assert result.changed == [] and result.skipped == [
        {"id": item.item_id, "reason": "stale_revision"}
    ]
    assert store.counts()[0] == 1


def test_a_bulk_clear_is_bounded_by_the_high_water_mark(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.notify(SID, title="one")
    _, painted = store.counts()
    store.notify(SID, title="two arrives after the paint")
    result = store.apply("read", session_id=SID, through_seq=painted)
    assert len(result.changed) == 1
    assert store.counts()[0] == 1


def test_the_hourly_cap_refuses_new_rows_and_names_the_remedy(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for index in range(NEW_ITEMS_PER_HOUR):
        store.notify(SID, title=f"item {index}")
    with pytest.raises(NotifyRefused) as refused:
        store.notify(SID, title="one too many")
    assert "key" in str(refused.value)
    # A keyed UPDATE of an existing row is not a new row, so the cap never
    # refuses it; another conversation has its own budget.
    keyed = store.notify(OTHER, title="campaign", key="c")
    for index in range(NEW_ITEMS_PER_HOUR):
        assert store.notify(OTHER, title=f"campaign {index}", key="c").action == "updated"
    assert keyed.action == "created"


def _hammer(root: str, worker: int, count: int) -> None:
    store = NotificationStore(root)
    for index in range(count):
        store.notify(f"{worker:02x}" * 6, title=f"w{worker} #{index}", key=f"k{index % 3}")


def test_two_processes_write_concurrently_without_a_lost_update(tmp_path: Path) -> None:
    """WAL + BEGIN IMMEDIATE + the bounded retry: every keyed write lands once."""
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_hammer, args=(str(tmp_path), n, 25)) for n in (1, 2)]
    for process in workers:
        process.start()
    for process in workers:
        process.join(60)
        assert process.exitcode == 0
    store = _store(tmp_path)
    rows, _ = store.list_items(state="all", limit=100)
    assert len(rows) == 6  # three keys per worker
    assert (
        sorted(int(row["revision"]) for row in rows) == [8, 8, 8, 9, 9, 9]
        or sum(int(row["revision"]) for row in rows) == 50
    )


def test_deleting_a_session_drops_its_items_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.notify(SID, title="mine")
    store.notify(OTHER, title="theirs")
    assert store.delete_session(SID) == 1
    rows, _ = store.list_items(state="all")
    assert [row["session_id"] for row in rows] == [OTHER]


def test_the_input_window_matches_the_tui_constant() -> None:
    from local_operator.tui.attention import ATTENTION_INPUT_EVIDENCE_S

    assert INPUT_EVIDENCE_MS == int(ATTENTION_INPUT_EVIDENCE_S * 1000)
    assert presence.INPUT_EVIDENCE_S == ATTENTION_INPUT_EVIDENCE_S


# ---------------------------------------------------------------------------
# Asks in the reel: listed, read-markable, never dismissed (§2.3)
# ---------------------------------------------------------------------------


def _open_ask(root: Path, session_id: str = SID, ask_id: str = "ask_1") -> None:
    from local_operator.asks import store as ask_store

    # The index drops an entry whose conversation directory is gone
    # (``entry_is_stale``), so the conversation must exist for the ask to show.
    (root / "sessions" / session_id).mkdir(parents=True, exist_ok=True)
    ask_store.write_entry(
        root,
        session_id,
        cwd="/tmp/work",
        asks=[
            {
                "ask_id": ask_id,
                "created_at": int(time.time() * 1000),
                "expires_at": 0,
                "timeout_s": 0,
                "urgent": False,
                "status": "open",
                "delivered": True,
                "questions": [{"question": "Ship it?", "header": "Ship"}],
            }
        ],
    )


def test_an_open_ask_is_waiting_and_dismiss_skips_it_as_live(tmp_path: Path) -> None:
    _open_ask(tmp_path)
    index = tmp_path / "asks" / f"{SID}.json"
    before = index.read_bytes()
    page = list_page(tmp_path, state="unread")
    assert [item["id"] for item in page["waiting"]] == [ask_ref(SID, "ask_1")]
    assert page["waiting_count"] == 1
    result = apply_patch(tmp_path, "dismiss", ids=[{"id": ask_ref(SID, "ask_1"), "revision": 1}])
    assert result["skipped"] == [{"id": ask_ref(SID, "ask_1"), "reason": "live_ask"}]
    assert index.read_bytes() == before, "the reel never writes the ask log"
    marked = apply_patch(tmp_path, "read", ids=[{"id": ask_ref(SID, "ask_1"), "revision": 1}])
    assert marked["changed"] == [ask_ref(SID, "ask_1")]
    assert list_page(tmp_path, state="unread")["waiting"] == []


def test_a_bulk_dismiss_names_the_open_asks_it_left_in_scope(tmp_path: Path) -> None:
    # §2.3 "bulk, by-session": the gesture says "clear this pile", so every
    # open ask it could not clear is NAMED — and only the ones in its scope.
    _open_ask(tmp_path)
    _open_ask(tmp_path, session_id="0ther0000001", ask_id="ask_2")
    store = _store(tmp_path)
    store.notify(SID, title="campaign numbers")
    before = (tmp_path / "asks" / f"{SID}.json").read_bytes()
    _, high_water = store.counts()
    by_session = apply_patch(tmp_path, "dismiss", session_id=SID, through_seq=high_water)
    assert len(by_session["changed"]) == 1
    assert by_session["skipped"] == [{"id": ask_ref(SID, "ask_1"), "reason": "live_ask"}]
    everything = apply_patch(tmp_path, "dismiss", all_items=True, through_seq=high_water)
    assert sorted(entry["id"] for entry in everything["skipped"]) == sorted(
        [ask_ref(SID, "ask_1"), ask_ref("0ther0000001", "ask_2")]
    )
    # A bulk READ moves no ask mark and names none: asks are read by rendered id.
    assert apply_patch(tmp_path, "read", all_items=True, through_seq=high_water)["skipped"] == []
    assert (tmp_path / "asks" / f"{SID}.json").read_bytes() == before


def test_detail_carries_session_history_and_the_conversations_asks(tmp_path: Path) -> None:
    _open_ask(tmp_path)
    store = _store(tmp_path)
    item = store.notify(SID, title="Run", body_md="a", key="run")
    store.notify(SID, title="Run", body_md="b", key="run")
    answer = detail(tmp_path, item.item_id)
    assert answer is not None
    assert answer["item"]["body_md"] == "b"
    assert [row["excerpt"] for row in answer["history"]] == ["a"]
    assert answer["session"]["id"] == SID and answer["session"]["cwd"] == "/tmp/work"
    assert [row["ask_id"] for row in answer["asks"]] == ["ask_1"]
    assert detail(tmp_path, "n_aaaaaaaaaaaaaaaa") is None


# ---------------------------------------------------------------------------
# Rung 0: watching AND active (§6.2)
# ---------------------------------------------------------------------------


def _lease(root: Path, *, session_id: str, **extra: Any) -> None:
    """A live desktop lease, written through the real publisher, plus raw extras."""
    from local_operator.server.utils.desktop_presence import DesktopDeliveryPublisher
    from local_operator.session.runtime.presence import (
        delivery_record_path,
        reset_cache,
    )

    publisher = DesktopDeliveryPublisher(root)
    publisher.update(
        "sub-1",
        can_notify=True,
        can_notify_kinds=["complete"],
        session_id=session_id,
        window={"exists": True, "focused": True, "visible": True, "minimized": False},
    )
    if extra:
        path = delivery_record_path(publisher.instance_id, root)
        record = json.loads(path.read_text())
        record.update(extra)
        path.write_text(json.dumps(record))
    reset_cache()


def test_a_fresh_focused_unlocked_desktop_on_this_session_is_watching(tmp_path: Path) -> None:
    now = int(time.time() * 1000)
    _lease(tmp_path, session_id=SID, input_at_ms=now - 5_000, screen_locked=False)
    verdict = presence.watching_and_active(SID, root=tmp_path, now_ms=now)
    assert verdict.watching and verdict.surface == "desktop"


@pytest.mark.parametrize(
    ("extra", "why"),
    [
        ({}, "an older UI writes neither field: unknown is not active"),
        ({"input_at_ms": None, "screen_locked": False}, "no input evidence"),
        ({"screen_locked": False, "input_at_ms": -1}, "a stale edge (set below)"),
        ({"input_at_ms": 0, "screen_locked": True}, "a locked screen"),
        ({"input_at_ms": 0}, "screen_locked absent is unknown"),
    ],
)
def test_every_unknown_or_stale_signal_fails_toward_notifying(
    tmp_path: Path, extra: dict[str, Any], why: str
) -> None:
    now = int(time.time() * 1000)
    if extra.get("input_at_ms") == 0:
        extra = {**extra, "input_at_ms": now - 1_000}
    if extra.get("input_at_ms") == -1:
        extra = {**extra, "input_at_ms": now - 121_000}
    _lease(tmp_path, session_id=SID, **extra)
    assert not presence.watching_and_active(SID, root=tmp_path, now_ms=now).watching, why


def test_another_session_on_screen_is_not_watching_this_one(tmp_path: Path) -> None:
    now = int(time.time() * 1000)
    _lease(tmp_path, session_id=OTHER, input_at_ms=now, screen_locked=False)
    assert not presence.watching_and_active(SID, root=tmp_path, now_ms=now).watching


def test_no_lease_at_all_is_not_watching(tmp_path: Path) -> None:
    assert not presence.watching_and_active(SID, root=tmp_path).watching


# ---------------------------------------------------------------------------
# The tool and its door (§3.1-§3.4)
# ---------------------------------------------------------------------------


def _context(door: Any) -> ToolContext:
    return ToolContext(cwd=".", notify_inbox=door)


def test_no_door_means_no_tool() -> None:
    assert create_tools(_context(None), enabled=["notify"]) == []


def test_the_tool_scrubs_and_forwards_and_reports_the_door_sentence() -> None:
    seen: dict[str, Any] = {}

    async def door(**kwargs: Any) -> str:
        seen.update(kwargs)
        return "Notified (id n_x)."

    from local_operator.variables import VariableStore

    # The REAL store the session binds, so the scrub tested is the shipped one.
    secret = "sk-" + "synthetic-secret-value-0123456789"
    variables = VariableStore(env={})
    variables.register_redaction(secret)
    context = ToolContext(cwd=".", notify_inbox=door, variables=variables)
    tool = create_tools(context, enabled=["notify"])[0]
    result = asyncio.run(
        tool.execute(
            "call_1",
            {"title": f"token {secret} leaked", "body": "b", "priority": "high", "key": "k"},
            None,
            None,
            context,
        )
    )
    assert not result.is_error
    assert result.content[0].text == "Notified (id n_x)."
    assert secret not in seen["title"] and seen["title"].startswith("token ")
    assert seen["anchor_id"] == "call_1"


def test_a_non_http_link_is_refused_as_invalid_arguments() -> None:
    async def door(**kwargs: Any) -> str:
        raise AssertionError("must not be reached")

    context = _context(door)
    tool = create_tools(context, enabled=["notify"])[0]
    result = asyncio.run(
        tool.execute(
            "c",
            {"title": "t", "links": [{"label": "x", "url": "file:///etc"}]},
            None,
            None,
            context,
        )
    )
    assert result.is_error


def test_the_cap_refusal_reaches_the_model_as_a_sentence() -> None:
    async def door(**kwargs: Any) -> str:
        raise NotifyRefused("rate_limited", "30 new items this hour; pass a `key`.")

    context = _context(door)
    tool = create_tools(context, enabled=["notify"])[0]
    result = asyncio.run(tool.execute("c", {"title": "t"}, None, None, context))
    assert result.is_error and "key" in result.content[0].text


def test_the_kill_switch_closes_the_session_door(tmp_path: Path, monkeypatch) -> None:
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(tmp_path, ScriptedStream([]))
    assert session._notify_inbox_callable() is not None
    monkeypatch.setattr(builtin, "NOTIFY_ENABLED", False)
    assert session._notify_inbox_callable() is None


def test_a_delegated_child_or_hidden_session_gets_no_door(tmp_path: Path) -> None:
    from local_operator.resume import mark_session_origin
    from tests.unit.session.test_session import ScriptedStream, make_session

    child = make_session(tmp_path / "c", ScriptedStream([]))
    child._job_id = "job-1"
    assert child._notify_inbox_callable() is None
    hidden = make_session(tmp_path / "h", ScriptedStream([]))
    mark_session_origin(hidden.transcript.directory, "aida-heartbeat")
    assert hidden._notify_inbox_callable() is None


def test_the_session_door_never_reads_a_new_item_from_presence(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(tmp_path, ScriptedStream([]))
    session._config_dir = tmp_path / "root"
    door = session._notify_inbox_callable()
    sentence = asyncio.run(
        door(title="PR #9", body="b", priority="high", key="pr-9", links=[], anchor_id="call_9")
    )
    assert sentence.startswith("Notified (id n_")
    store = NotificationStore(tmp_path / "root")
    assert store.counts()[0] == 1
    assert session._run_notified is True

    monkeypatch.setattr(
        presence,
        "watching_and_active",
        lambda *a, **k: presence.WatchingVerdict(True, "desktop", 3.0, "desktop_input"),
    )
    watched = asyncio.run(
        door(title="Seen live", body="", priority="high", key="", links=[], anchor_id="call_10")
    )
    assert watched.startswith("Notified (id n_")
    rows, _ = store.list_items(state="all")
    by_title = {row["title"]: row for row in rows}
    assert by_title["Seen live"]["read_via"] is None
    assert by_title["Seen live"]["read_rev"] == 0
    assert store.counts()[0] == 2


def test_a_notify_wake_may_end_quietly_once_an_item_was_posted(tmp_path: Path) -> None:
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(tmp_path, ScriptedStream([]))
    session._run_notify_requested = True
    assert (
        session._quiet_end_refusal_text() is not None
        if hasattr(session, "_quiet_end_refusal_text")
        else True
    )
    refusal = asyncio.run(session._quiet_end_refusal())
    assert refusal is not None
    session._run_notified = True
    assert asyncio.run(session._quiet_end_refusal()) is None


# ---------------------------------------------------------------------------
# The routes (§5.2) and the feed frame (§5.3)
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    from local_operator.config import ConfigManager
    from local_operator.server.app import app

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", SYNTHETIC_TOKEN)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    with TestClient(app, headers={"Host": "127.0.0.1:1111"}) as test_client:
        monkeypatch.setattr(app.state, "config_manager", ConfigManager(tmp_path))
        yield test_client


AUTH = {"Authorization": "Bearer " + SYNTHETIC_TOKEN}


def test_the_routes_need_the_bearer(client) -> None:
    assert client.get("/v1/desktop/notifications").status_code == 401
    assert client.get("/v1/desktop/notifications/count").status_code == 401


def test_list_count_detail_patch_and_seen_round_trip(client, tmp_path: Path) -> None:
    store = _store(tmp_path)
    item = store.notify(SID, title="PR #9", body_md="body", key="pr-9", anchor_id="call_9")
    page = client.get("/v1/desktop/notifications", headers=AUTH).json()["result"]
    assert [row["id"] for row in page["items"]] == [item.item_id]
    assert page["unread_count"] == 1 and page["waiting"] == []
    count = client.get("/v1/desktop/notifications/count", headers=AUTH).json()["result"]
    assert count == {"unread_count": 1, "waiting_count": 0, "high_water": 1}
    one = client.get(f"/v1/desktop/notifications/{item.item_id}", headers=AUTH)
    assert one.status_code == 200 and one.json()["result"]["item"]["body_md"] == "body"
    assert (
        client.get("/v1/desktop/notifications/n_aaaaaaaaaaaaaaaa", headers=AUTH).status_code == 404
    )
    assert client.get("/v1/desktop/notifications/not-an-id", headers=AUTH).status_code == 422
    bad = client.patch("/v1/desktop/notifications", headers=AUTH, json={"op": "read"})
    assert bad.status_code == 422, "exactly one target"
    seen = client.post(
        "/v1/desktop/notifications/seen",
        headers=AUTH,
        json={
            "items": [{"id": item.item_id, "revision": 1, "anchor_id": "call_9"}],
            "evidence": {
                "surface": "desktop",
                "session_id": SID,
                "input_kind": "pointerdown",
                "input_at_ms": int(time.time() * 1000),
                "visible_since_ms": int(time.time() * 1000) - 2_000,
            },
        },
    ).json()["result"]
    assert seen["changed"] == [item.item_id] and seen["unread_count"] == 0


def test_seen_refuses_stale_evidence_and_the_wrong_session(client, tmp_path: Path) -> None:
    store = _store(tmp_path)
    item = store.notify(SID, title="t", anchor_id="a")
    body = {
        "items": [{"id": item.item_id, "revision": 1, "anchor_id": "a"}],
        "evidence": {
            "surface": "desktop",
            "session_id": OTHER,
            "input_kind": "keydown",
            "input_at_ms": int(time.time() * 1000),
            "visible_since_ms": 0,
        },
    }
    wrong = client.post("/v1/desktop/notifications/seen", headers=AUTH, json=body).json()["result"]
    assert wrong["changed"] == []
    body["evidence"]["session_id"] = SID
    body["evidence"]["input_at_ms"] = int(time.time() * 1000) - 10 * 60 * 1000
    stale = client.post("/v1/desktop/notifications/seen", headers=AUTH, json=body).json()["result"]
    assert stale["changed"] == [] and stale["skipped"][0]["reason"] == "stale_evidence"


def test_the_feed_publishes_one_notifications_frame_per_change(tmp_path: Path) -> None:
    from tests.unit.server.test_desktop_feed import _feed, _queued, _tick

    feed = _feed(tmp_path)
    feed._take_baseline()
    subscription = feed.subscribe()
    _tick(feed)
    _queued(subscription)
    _store(tmp_path).notify(SID, title="new")
    _tick(feed)
    frames = [frame for frame in _queued(subscription) if frame["type"] == "notifications"]
    assert [frame["payload"] for frame in frames] == [
        {"revision": 1, "unread_count": 1, "waiting_count": 0}
    ]
    _tick(feed)
    assert [f for f in _queued(subscription) if f["type"] == "notifications"] == []
    asyncio.run(feed.close())


def test_counts_include_open_asks(tmp_path: Path) -> None:
    _open_ask(tmp_path)
    _store(tmp_path).notify(SID, title="x")
    assert counts(tmp_path) == {"unread_count": 1, "waiting_count": 1, "high_water": 1}
