"""One topic, one inbox entry: an open ask and a restating ``notify`` (``asks/dedupe.py``).

Cold-path: no runtime, no provider. The rule is pinned as pure functions over
the review's and QA's real pairs, then through the two session doors over a
``tmp_path`` config root, then on the payloads every surface reads.

EVERY ASK FIXTURE IS BUILT FROM ``AskQuestion(...).model_dump()`` (QA Q1,
#3094): round 1 fed the rule a ``header`` key the tool's model forbids, so the
tests exercised a shape no real ask has.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import dedupe
from local_operator.asks import store as ask_store
from local_operator.harness.types import AskOption, AskQuestion
from local_operator.notifications.items import detail, list_page
from local_operator.notifications.store import (
    MERGED_INTO_ASK_VIA,
    READ_VIA,
    NotificationStore,
    ask_ref,
)


def _question(text: str, *options: str, qid: str = "q") -> dict[str, Any]:
    return AskQuestion(
        id=qid, question=text, options=[AskOption(label=label) for label in options]
    ).model_dump()


ASK_QUESTION = "Drop the staging orders table before the migration?"
ASK_OPTIONS = ("Drop it", "Keep it")
ASK_ID = "drop_staging_orders"
#: A realistic exact repetition; paraphrases deliberately do not merge.
SAME_TITLE = ASK_QUESTION
SAME_BODY = ASK_QUESTION
OTHER_TITLE = "Nightly backup finished: 42 GB in 7 minutes"


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------

#: The reviewer's probe (review 6475197793) and QA's cases (6475414560):
#: (ask question, notify title, notify body, priority, merges?).
PAIRS = [
    # Additional reviewer round-3 probes; retain these as explicit regressions.
    (
        "Fraud campaign Q4: block IP range?",
        "Fraud campaign Q4: 3 new attempts",
        "",
        "normal",
        False,
    ),
    ("Merge PR #3094?", "Merged #3094, but CI is red", "", "normal", False),
    (
        "Deploy release 1.2.3?",
        "Deploy release 1.2.3 only after the freeze lifts",
        "",
        "normal",
        False,
    ),
    ("Delete branch feat/x?", "Should I also delete feat/y?", "", "normal", False),
    ("Delete branch feat/x?", "Branch feat/x nicht löschen", "", "normal", False),
    ("Delete branch feat/x?", "Ne pas supprimer branch feat/x", "", "normal", False),
    ("Delete branch feat/x?", "Delete branch feat/x? Don’t", "", "normal", False),
    ("Deploy release 1.2.3?", "Deploy release 1.2.3 tomorrow", "", "normal", False),
    ("Delete branch feat/x?", "Branch feat/x already deleted", "", "normal", False),
    ("Delete branch feat/x?", "Keep branch feat/x", "", "normal", False),
    (
        "Deploy core to production?",
        "Core production deploy failed",
        "rollback triggered; 502s for 4 min",
        "high",
        False,
    ),
    ("Deploy core to production?", "Core production deploy failed", "", "normal", False),
    (
        "Merge PR #3094 (notify dedupe)?",
        "PR #3094 notify dedupe: CI failed",
        "shard 3 red",
        "normal",
        False,
    ),
    ("Release local-operator v1.2.3?", "local-operator v1.2.3 released", "", "normal", False),
    (
        "Approve spend on campaign Spring Launch?",
        "Campaign Spring Launch: CTR 4.2%",
        "",
        "normal",
        False,
    ),
    ("Merge PR #3094?", "CI failed on #3094", "", "normal", False),
    (
        "Block IP range 10.0.0.0/8?",
        "Fraud attempts: 3 new",
        "from IP range 10.0.0.0/8",
        "normal",
        False,
    ),
    ("Release local-operator v1.2.3?", "local-operator v1.2.3 publish failed", "", "normal", False),
    (
        "Should I roll back the production deploy of the billing service "
        "that failed health checks?",
        "Production deploy rollback decision",
        "Production deploy of billing service failed health checks; rollback recommended.",
        "normal",
        False,
    ),
    ("Delete branch feat/x?", "Waiting on you: branch feat/x", "", "normal", False),
    (
        "Should I drop the staging orders table now?",
        "Drop staging orders table decision",
        "Should I drop the staging orders table now? Waiting on your decision.",
        "normal",
        False,
    ),
    (
        "Drop the staging orders table before the migration?",
        "Waiting on you: drop the staging orders table",
        "Decision needed before the migration.",
        "normal",
        False,
    ),
    ("Delete branch feat/x?", "Do not delete branch feat/x", "", "normal", False),
    ("Deploy core to production?", "Deploy core to production? No - wait", "", "normal", False),
    ("Delete branch feat/x?", "Don't delete branch feat/x", "", "normal", False),
    ("Delete branch feat/x?", "Never delete branch feat/x", "", "normal", False),
    ("Deploy core to production?", "Hold deploy core to production", "", "normal", False),
    ("Deploy core to production?", "Cancel the core production deploy", "", "normal", False),
    ("Deploy core to production?", "Deploy core to production instead", "", "normal", False),
    ("Delete branch feat/x?", "Delete branch feat/x unless CI is green", "", "normal", False),
    ("Delete branch feat/x?", "Delete branch feat/x only after review", "", "normal", False),
    ("Delete branch feat/x?", "Which branch: feat/x?", "", "normal", False),
    ("Delete branch feat/x?", "Delete branch feat/x? Should I keep feat/y", "", "normal", False),
    ("Deploy core to production?", "Rollback core production deploy", "", "normal", False),
    ("Delete branch feat/x?", "Delete branch feat/x, but keep the tag", "", "normal", False),
    (
        "Should I delete branch feat/x?",
        "Waiting on you: should I delete branch feat/x",
        "",
        "normal",
        False,
    ),
    ("Delete branch feat/x?", "Ne supprimez pas la branche feat/x", "", "normal", False),
    ("Delete branch feat/x?", "Waiting on you: branch feat/x", "", "high", False),
    ("Delete branch feat/x?", "Waiting on branch", "", "normal", False),
    ("Do not delete branch feat/x until backup?", "Delete branch feat/x now", "", "normal", False),
    (
        "Delete branch feat/x but not feat/y?",
        "Delete branch feat/y but not feat/x",
        "",
        "normal",
        False,
    ),
    ("Copy source to target?", "Copy target to source?", "", "normal", False),
    ("Deploy core after backup?", "Deploy core?", "", "normal", False),
    ("Set x < y?", "Set y < x?", "", "normal", False),
    ("Set x <= y?", "Set x >= y?", "", "normal", False),
    ("Allocate 5 GB?", "Allocate 5 MB?", "", "normal", False),
    ("Allocate 5 GB?", "Allocate 50 GB?", "", "normal", False),
    ("Delete branch feat/X?", "Delete branch feat/x?", "", "normal", False),
    ("Delete branch feat/x?", "Delete branch feat/y?", "", "normal", False),
    ("Set key K?", "Set key K?", "", "normal", False),
    (
        "Delete branch feat/x?",
        "Delete branch feat/x?",
        "Do not delete branch feat/x",
        "normal",
        False,
    ),
    (
        "Delete branch feat/x?",
        "Delete branch feat/x?",
        "https://example.org/rollback",
        "normal",
        False,
    ),
    ("Copy A  B?", "Copy A B?", "", "normal", False),
    ("Delete branch feat/x?", "Delete branch feat/x", "", "normal", False),
    (
        "Drop the staging orders table before the migration?",
        "Drop the staging orders table before the migration?",
        "",
        "normal",
        True,
    ),
    (
        "Drop the staging orders table before the migration?",
        "Drop the staging orders table before the migration?",
        "Drop the staging orders table before the migration?",
        "normal",
        True,
    ),
    (
        "Drop the staging orders table before the migration?",
        "  Drop the staging orders table before the migration?  ",
        "\nDrop the staging orders table before the migration?\n",
        "normal",
        True,
    ),
    (
        "Drop the staging orders table before the migration?",
        "Drop the staging orders table before the migration?",
        "Drop the staging orders table before the migration?",
        "high",
        False,
    ),
]


@pytest.mark.parametrize(("ask", "title", "body", "priority", "merges"), PAIRS)
def test_only_an_update_that_adds_nothing_is_merged(
    ask: str, title: str, body: str, priority: str, merges: bool
) -> None:
    record = {"questions": [_question(ask, "Yes", "No")]}
    assert dedupe.repeats_question(title, body, record, priority=priority) is merges


def test_only_individual_question_sentences_can_match() -> None:
    record = {
        "questions": [
            _question("Deploy core?", "Yes", "No", qid="deploy_core"),
            _question("Keep backup?", "Keep", "Delete", qid="backup"),
        ]
    }
    for text in ("Yes", "deploy_core", "Deploy core? Keep backup?"):
        assert not dedupe.repeats_question(text, "", record, priority="normal")
    assert dedupe.repeats_question("Keep backup?", "Keep backup?", record, priority="normal")


def test_matching_ask_honours_status_window_relay_and_priority() -> None:
    now = 10_000_000
    base = {"questions": [_question(ASK_QUESTION, *ASK_OPTIONS, qid=ASK_ID)]}
    fresh = {**base, "ask_id": "a1", "status": "open", "created_at": now - 1000}
    stale = {**base, "ask_id": "a2", "status": "open", "created_at": now - dedupe.WINDOW_MS - 1}
    settled = {**base, "ask_id": "a3", "status": "answered", "created_at": now}
    relay = {**base, "ask_id": "a4", "status": "open", "created_at": now, "relay": {"x": 1}}
    kw: dict[str, Any] = {"title": SAME_TITLE, "body": SAME_BODY, "now": now}
    assert dedupe.matching_ask([stale, settled, relay], **kw) is None
    match = dedupe.matching_ask([stale, fresh, settled, relay], **kw)
    assert match is not None and match["ask_id"] == "a1"
    assert dedupe.matching_ask([fresh], priority="high", **kw) is None


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
        "questions": [_question(ASK_QUESTION, *ASK_OPTIONS, qid=ASK_ID)],
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


def test_merge_into_ask_reads_token_bound_and_never_dismisses(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    item = store.notify("s1", title=SAME_TITLE, key="k")
    store.notify("s1", title=SAME_TITLE + " (updated)", key="k")  # revision 2
    assert store.merge_into_ask([(item.item_id, 1)]) == [], "a newer revision stays unread"
    assert store.merge_into_ask([(item.item_id, 2)]) == [item.item_id]
    row = store.get(item.item_id)
    assert row is not None
    assert row["read"] and row["read_via"] == MERGED_INTO_ASK_VIA
    assert row["dismissed_at"] is None, "D1: stays in All for surfaces without context"
    assert store.counts()[0] == 0
    # Already merged: a second call changes nothing.
    assert store.merge_into_ask([(item.item_id, 2)]) == []
    assert [r["id"] for r in store.list_items(state="all")[0]] == [item.item_id]


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


def _notify(sess: Any, title: str, body: str = "", *, priority: str = "normal") -> str:
    door = sess._notify_inbox_callable()
    assert door is not None

    async def run() -> str:
        return await door(
            title=title, body=body, priority=priority, key="", links=[], anchor_id="c"
        )

    return asyncio.run(run())


def _ask(sess: Any) -> dict[str, Any]:
    questions = [_question(ASK_QUESTION, *ASK_OPTIONS, qid=ASK_ID)]
    outcome = sess._enqueue_ask(questions, 3600, tool_call_id="call_ask")
    assert outcome["ok"], outcome
    return outcome


def _open_record(sess: Any) -> dict[str, Any]:
    [record] = sess.ask_queue().open_records()
    return record


def test_a_restating_notify_after_an_ask_is_suppressed_and_folded(session: Any) -> None:
    outcome = _ask(session)
    ask_id = outcome["details"]["ask_id"]
    sentence = _notify(session, SAME_TITLE, SAME_BODY)
    assert sentence == dedupe.SUPPRESSED_TEXT.format(ask_id=ask_id)
    assert sentence.startswith(f"Not posted: open ask {ask_id} already says this")
    store = NotificationStore(session._test_root)
    assert store.list_items(state="all")[0] == [], "no second inbox item"
    assert _open_record(session)["context"][0]["body"] == SAME_BODY


def test_an_outcome_on_the_same_subject_is_delivered(session: Any) -> None:
    """B1 through the real door: a failure about the asked-about thing is news."""
    _ask(session)
    sentence = _notify(session, "Staging orders table drop failed", "lock timeout")
    assert sentence.startswith("Notified (id n_")
    assert "context" not in _open_record(session)


@pytest.mark.parametrize(
    ("question", "options", "title"),
    [
        ("Delete branch feat/x?", ("Delete it", "Keep it"), "Do not delete branch feat/x"),
        ("Deploy core to production?", ("Yes", "No"), "Deploy core to production? No - wait"),
    ],
)
def test_a_negated_or_held_update_is_delivered_through_the_door(
    session: Any, question: str, options: tuple[str, ...], title: str
) -> None:
    """B1 round 2, through the real notify door (not only the predicate)."""
    session._enqueue_ask([_question(question, *options)], 3600, tool_call_id="c")
    sentence = _notify(session, title)
    assert sentence.startswith("Notified (id n_"), sentence
    assert "context" not in _open_record(session)


def test_a_high_priority_restatement_is_still_delivered(session: Any) -> None:
    _ask(session)
    sentence = _notify(session, SAME_TITLE, SAME_BODY, priority="high")
    assert sentence.startswith("Notified (id n_")
    assert "context" not in _open_record(session)


def test_a_different_topic_notify_in_the_same_turn_is_delivered(session: Any) -> None:
    _ask(session)
    sentence = _notify(session, OTHER_TITLE, "All shards verified.")
    assert sentence.startswith("Notified (id n_")
    assert NotificationStore(session._test_root).counts()[0] == 1
    assert "context" not in _open_record(session)


def test_a_restating_ask_after_a_notify_merges_the_update_read_not_dismissed(
    session: Any,
) -> None:
    sentence = _notify(session, SAME_TITLE, SAME_BODY)
    assert sentence.startswith("Notified (id n_")
    store = NotificationStore(session._test_root)
    assert store.counts()[0] == 1
    _ask(session)
    assert store.counts()[0] == 0, "the update left Unread"
    # D1: still reachable in All on surfaces that do not draw an ask's context.
    [row] = store.list_items(state="all")[0]
    assert row["read_via"] == MERGED_INTO_ASK_VIA
    assert row["dismissed_at"] is None and row["dismissed_via"] is None
    context = _open_record(session)["context"]
    assert context[0]["source"] == "merged" and context[0]["body"] == SAME_BODY


def test_a_different_topic_ask_leaves_the_update_alone(session: Any) -> None:
    _notify(session, OTHER_TITLE)
    _ask(session)
    assert NotificationStore(session._test_root).counts()[0] == 1
    assert "context" not in _open_record(session)


def test_outside_the_window_both_are_delivered(
    session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ask, then a restating update 11 minutes later: delivered."""
    queue = session.ask_queue()
    real = queue._now
    _ask(session)
    monkeypatch.setattr(queue, "_now", lambda: real() + dedupe.WINDOW_MS + 60_000)
    assert _notify(session, SAME_TITLE, SAME_BODY).startswith("Notified (id n_")


def test_an_update_older_than_the_window_is_not_merged(session: Any) -> None:
    store = NotificationStore(session._test_root)
    old = store.notify(
        session.session_id,
        title=SAME_TITLE,
        body_md=SAME_BODY,
        now=ask_store.now_ms() - dedupe.WINDOW_MS - 60_000,
    )
    _ask(session)
    row = store.get(old.item_id)
    assert row is not None and not row["read"] and row["dismissed_at"] is None


def test_an_ask_settled_after_the_match_refuses_the_context_and_delivers(
    session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1: the ask is answered elsewhere between the match and the append.

    ``matching_ask`` is wrapped to settle the ask right after it matched, which
    is exactly the window a phone answer lands in. The locked re-check must
    refuse, and the update must be delivered rather than attached to a settled
    question nobody will reopen.
    """
    outcome = _ask(session)
    ask_id = outcome["details"]["ask_id"]
    queue = session.ask_queue()
    real_match = dedupe.matching_ask

    def match_then_settle(*args: Any, **kwargs: Any) -> Any:
        found = real_match(*args, **kwargs)
        queue.withdraw(ask_id, reason="moot")
        return found

    monkeypatch.setattr(dedupe, "matching_ask", match_then_settle)
    sentence = _notify(session, SAME_TITLE, SAME_BODY)
    assert sentence.startswith("Notified (id n_")
    [record] = [r for r in queue.records() if r["ask_id"] == ask_id]
    assert record["status"] == "withdrawn" and "context" not in record
    assert queue.add_context(ask_id, title="t", body="b", source="notify") is False


def test_a_contended_lock_refuses_the_context_and_delivers(session: Any) -> None:
    """M1 round 2: a lock held past the retry budget is NOT exclusive, so refuse.

    Another fd holds ``flock`` on the log for the whole call (longer than
    ``_NonBlockingLock``'s ~95 ms budget). ``add_context`` must refuse and the
    door must deliver the update, rather than append on an unproven read.
    """
    import fcntl
    import os

    _ask(session)
    queue = session.ask_queue()
    holder = os.open(ask_store.asks_log_path(queue.session_dir), os.O_RDWR)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX)
        sentence = _notify(session, SAME_TITLE, SAME_BODY)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)
    assert sentence.startswith("Notified (id n_"), sentence
    assert "context" not in _open_record(session)


def test_a_settle_landing_between_attach_and_recheck_delivers_the_update(
    session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1 round 2: a settling writer appends UNLOCKED after its retry budget.

    Simulated exactly: the moment ``add_context`` writes its context row, an
    ``answered`` row is appended with the plain ``os.write`` a lock-starved
    ``append_event`` uses. The post-append re-check must see the settle and
    return False, so the update is delivered too (duplicate, never a loss).
    """
    import json
    import os

    outcome = _ask(session)
    ask_id = outcome["details"]["ask_id"]
    queue = session.ask_queue()
    log = ask_store.asks_log_path(queue.session_dir)
    real_write = os.write

    def write_then_settle(fd: int, data: bytes) -> int:
        written = real_write(fd, data)
        if b'"kind":"context"' in data:
            settle = {
                "v": 1,
                "kind": ask_store.EVENT_ANSWERED,
                "at": ask_store.now_ms(),
                "ask_id": ask_id,
                "answers": {"q": ["Drop it"]},
            }
            other = os.open(log, ask_store.append_flags())
            try:
                real_write(other, json.dumps(settle).encode() + b"\n")
            finally:
                os.close(other)
        return written

    monkeypatch.setattr(os, "write", write_then_settle)
    sentence = _notify(session, SAME_TITLE, SAME_BODY)
    monkeypatch.setattr(os, "write", real_write)
    assert sentence.startswith("Notified (id n_"), sentence
    [record] = [r for r in queue.records() if r["ask_id"] == ask_id]
    assert record["status"] == "answered"
    assert NotificationStore(session._test_root).counts()[0] == 1, "delivered, not lost"


def test_the_context_cap_is_counted_not_silent(session: Any) -> None:
    outcome = _ask(session)
    queue = session.ask_queue()
    ask_id = outcome["details"]["ask_id"]
    for n in range(ask_store.CONTEXT_MAX_ENTRIES + 2):
        assert queue.add_context(ask_id, title=f"t{n}", body="", source="notify")
    record = _open_record(session)
    assert [c["title"] for c in record["context"]] == ["t2", "t3", "t4", "t5", "t6"]
    assert record["context_dropped"] == 2
    [waiting] = list_page(session._test_root, state="unread")["waiting"]
    assert waiting["ask"]["context_dropped"] == 2


def test_folded_text_settles_with_the_ask(session: Any) -> None:
    """(c): once the ask settles nothing it absorbed is listed or reappears."""
    outcome = _ask(session)
    _notify(session, SAME_TITLE, SAME_BODY)
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
    _notify(session, SAME_TITLE, SAME_BODY)
    root = session._test_root
    [waiting] = list_page(root, state="unread")["waiting"]
    assert waiting["ask"]["context"] == [
        {
            "title": SAME_TITLE,
            "body_md": SAME_BODY,
            "at": waiting["ask"]["context"][0]["at"],
        }
    ]
    answer = detail(root, waiting["id"])
    assert answer is not None
    assert answer["item"]["body_md"] == f"**{SAME_TITLE}**\n\n{SAME_BODY}"


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
    _notify(session, SAME_TITLE, SAME_BODY)
    token = "-".join(("synthetic", "dedupe", "bearer"))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    auth = {"Authorization": "Bearer " + token}
    with TestClient(app, headers={"Host": "127.0.0.1:1111"}) as client:
        monkeypatch.setattr(app.state, "config_manager", ConfigManager(session._test_root))
        page = client.get("/v1/desktop/notifications", headers=auth).json()["result"]
        [waiting] = page["waiting"]
        assert waiting["ask"]["context"][0]["body_md"] == SAME_BODY
        assert page["items"] == []
        one = client.get(f"/v1/desktop/notifications/{waiting['id']}", headers=auth)
        assert one.status_code == 200
        assert SAME_BODY in one.json()["result"]["item"]["body_md"]


def test_the_relay_serves_the_folded_context(session: Any) -> None:
    from starlette.testclient import TestClient

    from local_operator.mobile.daemon import MobileDaemon, build_app

    _ask(session)
    _notify(session, SAME_TITLE, SAME_BODY)
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    page = client.get("/api/notifications").json()
    [waiting] = page["waiting"]
    assert waiting["id"] == ask_ref(session.session_id, waiting["ask"]["ask_id"])
    assert waiting["ask"]["context"][0]["body_md"] == SAME_BODY
    one = client.get(f"/api/notifications/{waiting['id']}").json()
    assert SAME_BODY in one["item"]["body_md"]


@pytest.mark.parametrize(
    ("question", "title"),
    [
        ("Do not delete branch feat/x until backup?", "Delete branch feat/x now"),
        ("Delete branch feat/x but not feat/y?", "Delete branch feat/y but not feat/x"),
    ],
)
@pytest.mark.parametrize("notify_first", [False, True])
def test_round3_polarity_failures_deliver_in_both_orders(
    session: Any, question: str, title: str, notify_first: bool
) -> None:
    if notify_first:
        assert _notify(session, title).startswith("Notified (id n_")
    outcome = session._enqueue_ask([_question(question, "Yes", "No")], 3600, tool_call_id="r3")
    assert outcome["ok"]
    if not notify_first:
        assert _notify(session, title).startswith("Notified (id n_")
    [row] = NotificationStore(session._test_root).list_items(state="unread")[0]
    assert row["title"] == title and not row["read"] and not row["dismissed"]
    assert "context" not in _open_record(session)


def test_exact_title_with_contradictory_body_delivers(session: Any) -> None:
    _ask(session)
    assert _notify(session, SAME_TITLE, "Do not drop the table.").startswith("Notified (id n_")
    assert "context" not in _open_record(session)
