"""One topic, one inbox entry: an open ask and a same-topic ``notify`` (``asks/dedupe.py``).

Cold-path: no runtime, no provider. The rule is pinned as pure functions (the
threshold's near-miss pairs), then through the two session doors over a
``tmp_path`` config root, then on the payloads every surface reads — the desktop
route and the phone relay.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import dedupe
from local_operator.asks import store as ask_store
from local_operator.notifications.items import detail, list_page
from local_operator.notifications.store import (
    MERGED_INTO_ASK_VIA,
    READ_VIA,
    NotificationStore,
    ask_ref,
)

ASK_QUESTION = "Drop the staging orders table before the migration?"
ASK_HEADER = "Staging table"
SAME_TITLE = "Waiting on you: drop the staging orders table"
OTHER_TITLE = "Nightly backup finished: 42 GB in 7 minutes"


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------


def test_the_threshold_near_miss_pairs_are_pinned() -> None:
    """One pair just UNDER the threshold and one just OVER — exact fractions.

    Ask tokens: {deploy, billing, service, production, tonight}. "hotfix" adds a
    sixth token to the union (3/6 = 0.5, merged: the boundary is inclusive);
    "hotfix canary" adds a seventh (3/7 ~ 0.43, delivered). If the threshold or
    the tokenizer moves, one of these two flips and this cell says which.
    """
    ask = "Deploy the billing service to production tonight?"
    over = "Deploy billing service hotfix"
    under = "Deploy billing service hotfix canary"
    assert dedupe.THRESHOLD == 0.5
    assert dedupe.similarity(over, "", ask) == pytest.approx(3 / 6)
    assert dedupe.similarity(under, "", ask) == pytest.approx(3 / 7)
    assert dedupe.same_topic(over, "", ask) is True
    assert dedupe.same_topic(under, "", ask) is False


def test_a_long_body_cannot_dilute_a_same_topic_title() -> None:
    """The title alone is one of the two readings, so prose cannot hide it."""
    body = " ".join(f"detail{n}" for n in range(60))
    assert dedupe.same_topic(SAME_TITLE, body, f"{ASK_HEADER} {ASK_QUESTION}")


def test_stopwords_and_short_tokens_carry_no_topic() -> None:
    assert dedupe.tokens("Do you want the fix to go out now?") == frozenset({"want", "fix"})
    assert dedupe.similarity("", "", "anything") == 0.0


def test_matching_ask_honours_status_window_and_relay() -> None:
    now = 10_000_000
    base = {"questions": [{"header": ASK_HEADER, "question": ASK_QUESTION}]}
    fresh = {**base, "ask_id": "a1", "status": "open", "created_at": now - 1000}
    stale = {**base, "ask_id": "a2", "status": "open", "created_at": now - dedupe.WINDOW_MS - 1}
    settled = {**base, "ask_id": "a3", "status": "answered", "created_at": now}
    relay = {**base, "ask_id": "a4", "status": "open", "created_at": now, "relay": {"x": 1}}
    assert dedupe.matching_ask([stale, settled, relay], title=SAME_TITLE, body="", now=now) is None
    match = dedupe.matching_ask([stale, fresh, settled, relay], title=SAME_TITLE, body="", now=now)
    assert match is not None and match["ask_id"] == "a1"


def test_the_merged_via_is_not_a_read_any_surface_may_mint() -> None:
    assert dedupe.MERGED_VIA == MERGED_INTO_ASK_VIA == "merged_into_ask"
    assert MERGED_INTO_ASK_VIA not in READ_VIA


# ---------------------------------------------------------------------------
# The fold and the store
# ---------------------------------------------------------------------------


def _queued(ask_id: str, at: int) -> dict[str, Any]:
    return {
        "v": 1,
        "kind": ask_store.EVENT_QUEUED,
        "at": at,
        "ask_id": ask_id,
        "expires_at": at + 3_600_000,
        "timeout_s": 3600,
        "questions": [{"id": "q", "header": ASK_HEADER, "question": ASK_QUESTION, "options": []}],
    }


def test_context_rides_the_fold_and_settles_with_the_ask() -> None:
    """A context row is not a status event: the ask settles, and the text with it."""
    at = 1_000_000
    events = [
        _queued("a1", at),
        {"kind": ask_store.EVENT_CONTEXT, "ask_id": "a1", "at": at + 1, "title": "T", "body": "B"},
    ]
    [record] = ask_store.fold(events, at + 2)
    assert record["status"] == "open"
    assert record["context"] == [{"title": "T", "body": "B", "source": "notify", "at": at + 1}]
    row = ask_store.pending_row(record)
    assert row["context"][0]["body"] == "B"
    settled = events + [{"kind": ask_store.EVENT_WITHDRAWN, "ask_id": "a1", "at": at + 3}]
    [done] = ask_store.fold(settled, at + 4)
    assert done["status"] == "withdrawn"
    assert not ask_store.is_outstanding(done["status"])
    # An ask with nothing folded keeps the exact shape older builds wrote.
    [plain] = ask_store.fold([_queued("a2", at)], at + 1)
    assert "context" not in plain and "context" not in ask_store.pending_row(plain)


def test_merge_into_ask_reads_and_dismisses_token_bound(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    item = store.notify("s1", title=SAME_TITLE, key="k")
    store.notify("s1", title=SAME_TITLE + " (updated)", key="k")  # revision 2
    assert store.merge_into_ask([(item.item_id, 1)]) == [], "a newer revision stays unread"
    assert store.merge_into_ask([(item.item_id, 2)]) == [item.item_id]
    row = store.get(item.item_id)
    assert row is not None
    assert row["read"] and row["read_via"] == MERGED_INTO_ASK_VIA
    assert row["dismissed_at"] is not None and row["dismissed_via"] == MERGED_INTO_ASK_VIA
    assert store.counts()[0] == 0
    # Already merged: a second call changes nothing and nothing reappears.
    assert store.merge_into_ask([(item.item_id, 2)]) == []
    assert store.list_items(state="all")[0] == []


# ---------------------------------------------------------------------------
# The two session doors
# ---------------------------------------------------------------------------


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.unit.session.test_session import ScriptedStream, make_session

    root = tmp_path / "root"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    # A real-shaped (12-hex) id: the detail routes validate ask refs strictly.
    sess = make_session(tmp_path, ScriptedStream([]), session_id="d1e2d3d4e5f6")
    sess._config_dir = root

    async def _answer(*_a: Any, **_k: Any) -> Any:  # pragma: no cover — never parked
        return None

    sess._ask_user = _answer  # mounts the queued engine (asks/policy default on)
    queue = sess.ask_queue()
    assert queue is not None
    # A test-only handle on the config root the doors write under.
    setattr(sess, "_test_root", root)
    return sess


def _notify(sess: Any, title: str, body: str = "") -> str:
    door = sess._notify_inbox_callable()
    assert door is not None

    async def run() -> str:
        return await door(
            title=title, body=body, priority="normal", key="", links=[], anchor_id="c"
        )

    return asyncio.run(run())


def _ask(sess: Any) -> dict[str, Any]:
    questions = [{"id": "q", "header": ASK_HEADER, "question": ASK_QUESTION, "options": []}]
    outcome = sess._enqueue_ask(questions, 3600, tool_call_id="call_ask")
    assert outcome["ok"], outcome
    return outcome


def _open_record(sess: Any) -> dict[str, Any]:
    [record] = sess.ask_queue().open_records()
    return record


def test_same_topic_notify_after_an_ask_is_suppressed_and_folded(session: Any) -> None:
    _ask(session)
    sentence = _notify(session, SAME_TITLE, "Rows: 1.2M; backup taken at 09:10.")
    assert sentence == (
        "Not sent: your open ask already notifies the user; its text was added to that ask."
    )
    store = NotificationStore(session._test_root)
    assert store.list_items(state="all")[0] == [], "no second inbox item"
    assert _open_record(session)["context"][0]["body"] == "Rows: 1.2M; backup taken at 09:10."


def test_a_different_topic_notify_in_the_same_turn_is_delivered(session: Any) -> None:
    _ask(session)
    sentence = _notify(session, OTHER_TITLE, "All shards verified.")
    assert sentence.startswith("Notified (id n_")
    assert NotificationStore(session._test_root).counts()[0] == 1
    assert "context" not in _open_record(session)


def test_a_same_topic_ask_after_a_notify_merges_the_update(session: Any) -> None:
    sentence = _notify(session, SAME_TITLE, "The table holds 1.2M rows.")
    assert sentence.startswith("Notified (id n_")
    store = NotificationStore(session._test_root)
    assert store.counts()[0] == 1
    _ask(session)
    assert store.counts()[0] == 0, "the update left Unread"
    [row] = store.list_items(state="dismissed")[0]
    assert row["read_via"] == row["dismissed_via"] == MERGED_INTO_ASK_VIA
    context = _open_record(session)["context"]
    assert context[0]["source"] == "merged" and context[0]["body"] == "The table holds 1.2M rows."


def test_a_different_topic_ask_leaves_the_update_alone(session: Any) -> None:
    _notify(session, OTHER_TITLE)
    _ask(session)
    assert NotificationStore(session._test_root).counts()[0] == 1
    assert "context" not in _open_record(session)


def test_outside_the_window_both_are_delivered(
    session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ask, then a same-topic update 11 minutes later: news, not a restatement."""
    queue = session.ask_queue()
    real = queue._now
    _ask(session)
    monkeypatch.setattr(queue, "_now", lambda: real() + dedupe.WINDOW_MS + 60_000)
    assert _notify(session, SAME_TITLE).startswith("Notified (id n_")


def test_an_update_older_than_the_window_is_not_merged(session: Any) -> None:
    store = NotificationStore(session._test_root)
    old = store.notify(
        session.session_id,
        title=SAME_TITLE,
        now=ask_store.now_ms() - dedupe.WINDOW_MS - 60_000,
    )
    _ask(session)
    row = store.get(old.item_id)
    assert row is not None and not row["read"] and row["dismissed_at"] is None


def test_merged_updates_stay_settled_when_the_ask_settles(session: Any) -> None:
    """(c): nothing merged reappears once the ask is withdrawn."""
    _notify(session, SAME_TITLE, "context")
    outcome = _ask(session)
    store = NotificationStore(session._test_root)
    queue = session.ask_queue()
    queue.withdraw(outcome["details"]["ask_id"], reason="moot")
    assert queue.open_records() == []
    assert store.counts()[0] == 0
    assert store.list_items(state="all")[0] == []
    page = list_page(session._test_root, state="unread")
    assert page["items"] == [] and page["waiting"] == []


# ---------------------------------------------------------------------------
# The payloads: desktop route and phone relay
# ---------------------------------------------------------------------------


def test_the_folded_context_rides_the_ask_item_and_its_detail(session: Any) -> None:
    _ask(session)
    _notify(session, SAME_TITLE, "Backup taken at 09:10.")
    root = session._test_root
    [waiting] = list_page(root, state="unread")["waiting"]
    assert waiting["ask"]["context"] == [
        {
            "title": SAME_TITLE,
            "body_md": "Backup taken at 09:10.",
            "at": waiting["ask"]["context"][0]["at"],
        }
    ]
    answer = detail(root, waiting["id"])
    assert answer is not None
    assert answer["item"]["body_md"] == f"**{SAME_TITLE}**\n\nBackup taken at 09:10."


def test_an_ask_without_context_keeps_its_detail_shape(session: Any) -> None:
    _ask(session)
    root = session._test_root
    [waiting] = list_page(root, state="unread")["waiting"]
    assert waiting["ask"]["context"] == []
    answer = detail(root, waiting["id"])
    assert answer is not None and "body_md" not in answer["item"]


def test_the_desktop_route_serves_the_folded_context(session: Any, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from local_operator.config import ConfigManager
    from local_operator.server.app import app

    _ask(session)
    _notify(session, SAME_TITLE, "Backup taken at 09:10.")
    token = "-".join(("synthetic", "dedupe", "bearer"))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    auth = {"Authorization": "Bearer " + token}
    with TestClient(app, headers={"Host": "127.0.0.1:1111"}) as client:
        monkeypatch.setattr(app.state, "config_manager", ConfigManager(session._test_root))
        page = client.get("/v1/desktop/notifications", headers=auth).json()["result"]
        [waiting] = page["waiting"]
        assert waiting["ask"]["context"][0]["body_md"] == "Backup taken at 09:10."
        assert page["items"] == []
        one = client.get(f"/v1/desktop/notifications/{waiting['id']}", headers=auth)
        assert one.status_code == 200
        assert "Backup taken at 09:10." in one.json()["result"]["item"]["body_md"]


def test_the_relay_serves_the_folded_context(session: Any) -> None:
    from starlette.testclient import TestClient

    from local_operator.mobile.daemon import MobileDaemon, build_app

    _ask(session)
    _notify(session, SAME_TITLE, "Backup taken at 09:10.")
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    page = client.get("/api/notifications").json()
    [waiting] = page["waiting"]
    assert waiting["id"] == ask_ref(session.session_id, waiting["ask"]["ask_id"])
    assert waiting["ask"]["context"][0]["body_md"] == "Backup taken at 09:10."
    one = client.get(f"/api/notifications/{waiting['id']}").json()
    assert "Backup taken at 09:10." in one["item"]["body_md"]
