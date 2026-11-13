"""Inbox read cost and the session-id grammar (notifications list/detail perf PR).

Opening Notifications and clicking between items were slow because every
request rebuilt the project registry (a parse of every ``projects/*.json``),
read the open asks twice, and the app paged the whole store to count rows per
conversation. These tests pin the fixes: one shared registry that still sees a
changed project, one open-asks read per request, one grouped ``session_counts``
read matching the app's old walk, and a session-id grammar that admits the
core's own pseudo-session (``network-credentials``) and nothing arbitrary.

Cold-path throughout, ``tmp_path`` config roots, a synthetic bearer.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from local_operator.notifications import items as inbox
from local_operator.notifications.store import (
    NETWORK_CREDENTIALS_SESSION_ID,
    PSEUDO_SESSION_IDS,
    NotificationStore,
)

SYNTHETIC_TOKEN = "-".join(("synthetic", "inbox", "perf", "bearer"))
AUTH = {"Authorization": "Bearer " + SYNTHETIC_TOKEN}
URL = "/v1/desktop/notifications"
SID = "a1" * 6
OTHER = "b2" * 6
PSEUDO = NETWORK_CREDENTIALS_SESSION_ID


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


def _open_ask(root: Path, session_id: str = SID, ask_id: str = "ask_1") -> None:
    from local_operator.asks import store as ask_store

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


# ---------------------------------------------------------------------------
# One shared project registry, still fresh
# ---------------------------------------------------------------------------


def test_the_project_registry_is_built_once_across_requests_and_sees_a_new_link(
    tmp_path: Path, monkeypatch
) -> None:
    from local_operator import projects

    built = 0
    real_init = projects.ProjectRegistry.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal built
        built += 1
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(projects.ProjectRegistry, "__init__", counting_init)
    item = NotificationStore(tmp_path).notify(SID, title="PR #1", body_md="b")
    first = inbox.list_page(tmp_path, state="all")
    assert first["items"][0]["projects"] == []
    assert built == 1
    inbox.list_page(tmp_path, state="unread")
    inbox.detail(tmp_path, item.item_id)
    inbox.list_page(tmp_path, state="all", limit=10)
    # Before the fix every request (and the detail) constructed its own.
    assert built == 1, built

    # A peer process links the session: a SEPARATE registry instance publishes
    # (temp-write + os.replace), which moves projects/'s mtime. The shared one
    # must answer with the link on the very next request, inside the 5 s
    # refresh interval, without being rebuilt.
    writer = projects.ProjectRegistry(tmp_path)
    writer.create_project(projects.ProjectEdit(name="perf-proj", title="Perf"), sessions=[SID])
    built_after_writer = built
    page = inbox.list_page(tmp_path, state="all")
    assert page["items"][0]["projects"] == [
        {"key": "perf-proj", "name": "Perf", "status": "active"}
    ]
    one = inbox.detail(tmp_path, item.item_id)
    assert one is not None and one["projects"][0]["key"] == "perf-proj"
    assert built == built_after_writer, "a changed project file needs no rebuild"


def test_a_failed_registry_build_is_not_cached(tmp_path: Path, monkeypatch) -> None:
    from local_operator import projects

    NotificationStore(tmp_path).notify(SID, title="x")
    real_init = projects.ProjectRegistry.__init__
    calls = 0

    def failing_once(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("transient")
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(projects.ProjectRegistry, "__init__", failing_once)
    # Degrades to "no projects" (context, not the item) and retries next time.
    assert inbox.list_page(tmp_path, state="all")["items"][0]["projects"] == []
    inbox.list_page(tmp_path, state="all")
    assert calls == 2


# ---------------------------------------------------------------------------
# One open-asks read per request
# ---------------------------------------------------------------------------


def test_a_list_page_and_a_patch_read_the_open_asks_once(tmp_path: Path, monkeypatch) -> None:
    _open_ask(tmp_path)
    store = NotificationStore(tmp_path)
    item = store.notify(SID, title="t")
    reads = 0
    real = inbox.open_asks

    def counted(*args, **kwargs):
        nonlocal reads
        reads += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(inbox, "open_asks", counted)
    page = inbox.list_page(tmp_path, state="unread")
    assert reads == 1, reads  # was 2: badge figures + waiting group
    assert page["waiting_count"] == 1 and len(page["waiting"]) == 1
    reads = 0
    result = inbox.apply_patch(tmp_path, "read", ids=[{"id": item.item_id, "revision": 1}])
    assert reads == 1, reads  # was 2: open refs + badge
    assert result["changed"] == [item.item_id] and result["waiting_count"] == 1


# ---------------------------------------------------------------------------
# session_counts: the app's whole-store walk as one grouped read
# ---------------------------------------------------------------------------


def _ui_walk(client, through_seq: int) -> dict[str, int]:
    """The app's ``readSessionDismissCounts``, verbatim in rule."""
    counts: dict[str, int] = {}
    cursor = None
    while True:
        params = {"state": "all", "source": "notify", "limit": 100, "through_seq": through_seq}
        if cursor:
            params["cursor"] = cursor
        page = client.get(URL, headers=AUTH, params=params).json()["result"]
        for row in page["items"]:
            if not row["dismissed"] and row["source"] != "ask":
                counts[row["session"]["id"]] = counts.get(row["session"]["id"], 0) + 1
        cursor = page["next_cursor"]
        if not cursor:
            return counts


def test_session_counts_match_the_app_walk_and_ignore_page_filters(client, tmp_path: Path) -> None:
    _open_ask(tmp_path)  # an open ask is never counted
    store = NotificationStore(tmp_path)
    store.notify(SID, title="unread one", priority="low")
    read = store.notify(SID, title="read one")
    store.apply("read", ids=[(read.item_id, 1)])
    gone = store.notify(SID, title="dismissed")
    store.apply("dismiss", ids=[(gone.item_id, 1)])
    for n in range(3):
        store.notify(OTHER, title=f"other {n}", priority="high")
    store.notify(PSEUDO, title="repair", priority="high")
    only_dismissed = store.notify("c3" * 6, title="only dismissed")
    store.apply("dismiss", ids=[(only_dismissed.item_id, 1)])

    # Filters and paging on the page do NOT narrow the counts.
    answer = client.get(
        URL,
        headers=AUTH,
        params={
            "include_session_counts": "true",
            "state": "unread",
            "priority": "high",
            "limit": 1,
        },
    ).json()["result"]
    expected = {
        SID: {"undismissed": 2, "unread": 1},
        OTHER: {"undismissed": 3, "unread": 3},
        PSEUDO: {"undismissed": 1, "unread": 1},
    }
    assert answer["session_counts"] == expected
    walk = _ui_walk(client, answer["high_water"])
    assert {sid: c["undismissed"] for sid, c in answer["session_counts"].items()} == walk

    # Absent unless asked for: the field is additive.
    assert "session_counts" not in client.get(URL, headers=AUTH).json()["result"]


def test_session_counts_are_taken_at_the_responses_high_water(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    store.notify(SID, title="before")
    _, high_water = store.counts()
    store.notify(SID, title="after the boundary")
    page = inbox.list_page(
        tmp_path, state="all", through_seq=high_water, include_session_counts=True
    )
    assert page["high_water"] == high_water
    assert page["session_counts"] == {SID: {"undismissed": 1, "unread": 1}}


def test_session_counts_on_a_store_that_does_not_exist_is_empty(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    assert store.session_counts() == {}
    assert not store.path.exists()


# ---------------------------------------------------------------------------
# Session-id grammar: 12 hex + the allowlisted core pseudo-sessions only
# ---------------------------------------------------------------------------


def test_the_repair_writer_files_under_an_allowlisted_pseudo_session() -> None:
    from local_operator.network.credentials import repair_inbox

    assert repair_inbox.SESSION_ID in PSEUDO_SESSION_IDS


def test_dismiss_all_from_a_pseudo_session_is_accepted(client, tmp_path: Path) -> None:
    """Reproduces "Invalid desktop operation": this PATCH was a 422 before."""
    store = NotificationStore(tmp_path)
    repair = store.notify(PSEUDO, title="Repair the mesh credential")
    kept = store.notify(SID, title="unrelated")
    page = client.get(URL, headers=AUTH, params={"state": "all", "session_id": PSEUDO})
    assert page.status_code == 200, page.text
    listed = page.json()["result"]
    assert [row["id"] for row in listed["items"]] == [repair.item_id]
    response = client.patch(
        URL,
        headers=AUTH,
        json={
            "op": "dismiss",
            "session_id": PSEUDO,
            "through_seq": listed["high_water"],
            "observed": listed["observed"],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["changed"] == [repair.item_id]
    assert store.get(kept.item_id)["dismissed"] is False
    preview = client.get(
        URL,
        headers=AUTH,
        params={"preview": "true", "state": "all", "count_session_ids": [PSEUDO, SID]},
    )
    assert preview.status_code == 200, preview.text


@pytest.mark.parametrize(
    "junk",
    [
        "network-credentialsx",
        "xnetwork-credentials",
        "NETWORK-CREDENTIALS",
        "network",
        "a1a1a1a1a1a1a1",
        "a1a1a1a1a1",
        "../etc",
        "network-credentials\n",
    ],
)
def test_other_session_ids_stay_refused(client, junk: str) -> None:
    assert client.get(URL, headers=AUTH, params={"session_id": junk}).status_code == 422
    assert (
        client.get(
            URL, headers=AUTH, params={"preview": "true", "count_session_ids": [junk]}
        ).status_code
        == 422
    )
    body = {"op": "dismiss", "session_id": junk, "through_seq": 1, "observed": []}
    assert client.patch(URL, headers=AUTH, json=body).status_code == 422


def test_the_relay_query_model_shares_the_grammar_and_the_counts_flag() -> None:
    from pydantic import ValidationError

    from local_operator.server.routes.desktop_notifications import NotificationsQuery

    query = NotificationsQuery.model_validate(
        {"session_id": PSEUDO, "include_session_counts": "true"}
    )
    assert query.session_id == PSEUDO and query.include_session_counts is True
    with pytest.raises(ValidationError):
        NotificationsQuery.model_validate({"session_id": "network-credentialsx"})
