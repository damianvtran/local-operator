"""Real production routes and SQLite state; no runtime/model or operator roots.

Large fixtures exercise one server-side request beyond the old 500-id limit.
Other cases pin snapshot races, receipt replay, ABA-safe Undo and live-ask
separation. All request bodies go through the production schema and auth gate.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_operator.notifications import bulk
from local_operator.notifications.store import NotificationStore, ask_ref, new_item_id
from local_operator.server.routes.desktop_notifications import router
from tests.unit.notifications.test_inbox import (
    AUTH,
    OTHER,
    SID,
    SYNTHETIC_TOKEN,
    _open_ask,
)

URL = "/v1/desktop/notifications"


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", SYNTHETIC_TOKEN)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    app = FastAPI()
    app.include_router(router)
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    with TestClient(app, headers={"Host": "127.0.0.1:1111", **AUTH}) as test_client:
        yield test_client


def seed(root: Path, count: int = 1, *, session: str = SID, priority: str = "normal") -> list[str]:
    store = NotificationStore(root)
    first = store.notify(session, title="Bulk fixture", priority=priority, key="")
    ids = [first.item_id]

    # Populate the volume fixture in one transaction without relaxing production
    # publish rate limits or dispatching hundreds of separate fixture writes.
    def insert(conn: sqlite3.Connection) -> None:
        source = dict(conn.execute("SELECT * FROM items WHERE id=?", (first.item_id,)).fetchone())
        for _ in range(count - 1):
            row = {**source, "id": new_item_id(), "seq": store._next_seq(conn)}
            conn.execute(
                "INSERT INTO items("
                + ",".join(row)
                + ") VALUES ("
                + ",".join("?" for _ in row)
                + ")",
                list(row.values()),
            )
            ids.append(row["id"])

    store._write(insert, "fixture")
    return ids


def preview(client, **filters):
    response = client.get(URL, params={"preview": "true", **filters})
    assert response.status_code == 200, response.text
    return response.json()["result"]["selection"]


def patch(client, **body):
    response = client.patch(URL, json=body)
    assert response.status_code == 200, response.text
    assert len(response.content) < 20_000
    return response.json()["result"]


def test_whole_filtered_result_exclusions_and_undo_beyond_pages(client, tmp_path):
    ids = seed(tmp_path, 625)
    unrelated = seed(tmp_path, 3, session=OTHER, priority="low")
    store = NotificationStore(tmp_path)
    selection = preview(
        client, kind="updates", state="all", session_id=SID, count_session_ids=[SID, OTHER]
    )
    assert selection["eligible_count"] == 625
    assert selection["eligible_unread_count"] == 625
    assert selection["session_counts"][OTHER]["eligible_count"] == 0
    assert client.get(URL, params={"limit": 100}).json()["result"]["next_cursor"]
    result = patch(client, op="dismiss", selection_token=selection["token"], excluded_ids=[ids[0]])
    assert result["changed_count"] == 624
    assert result["excluded_count"] == 1
    assert result["unread_delta"] == -624
    assert result["session_counts"][SID] == {"changed_count": 624, "unread_delta": -624}
    # Missing fixture rows must fail these assertions rather than weaken them.
    excluded = store.get(ids[0])
    assert excluded is not None and not excluded["dismissed"]
    assert all(
        (item_row := store.get(item)) is not None and not item_row["dismissed"]
        for item in unrelated
    )
    assert store.counts()[0] == 4
    undo = patch(client, op="undo", operation_token=result["operation_token"])
    assert undo["changed_count"] == 624
    assert undo["unread_delta"] == 624
    assert store.counts()[0] == 628


@pytest.mark.parametrize(
    "kind,priority,state,session,expected",
    [
        ("all", "any", "all", None, 6),
        ("updates", "high", "all", None, 2),
        ("updates", "normal_up", "all", None, 4),
        ("updates", "any", "unread", None, 5),
        ("asks", "any", "all", None, 0),
        ("all", "normal_up", "unread", SID, 1),
        ("all", "high", "all", OTHER, 1),
    ],
)
def test_each_filter_and_combinations(client, tmp_path, kind, priority, state, session, expected):
    ids = []
    for sid in (SID, OTHER):
        for rank in ("high", "normal", "low"):
            ids.extend(seed(tmp_path, session=sid, priority=rank))
    NotificationStore(tmp_path).apply("read", ids=[(ids[0], 1)])
    params = {"kind": kind, "bulk_priority": priority, "state": state}
    if session:
        params["session_id"] = session
    selection = preview(client, **params)
    assert selection["eligible_count"] == expected
    result = patch(client, op="dismiss", selection_token=selection["token"])
    assert result["changed_count"] == expected


def test_new_content_and_state_revisions_are_not_swept_in(client, tmp_path):
    store = NotificationStore(tmp_path)
    keyed = store.notify(SID, title="old", key="key")
    ids = seed(tmp_path, 2)
    selection = preview(client, kind="updates", state="all")
    store.notify(SID, title="new", key="key")
    store.apply("read", ids=[(ids[0], 1)])
    fresh = seed(tmp_path)[0]
    result = patch(client, op="dismiss", selection_token=selection["token"])
    assert result["changed_count"] == 1 and result["skipped_count"] == 2
    for item in (keyed.item_id, ids[0], fresh):
        row = store.get(item)
        assert row is not None and not row["dismissed"]
    dismissed = store.get(ids[1])
    assert dismissed is not None and dismissed["dismissed"]


def test_replay_binds_op_and_exclusions_even_after_undo(client, tmp_path):
    ids = seed(tmp_path, 3)
    token = preview(client)["token"]
    body = {"op": "dismiss", "selection_token": token, "excluded_ids": [ids[0], ids[1]]}
    first = patch(client, **body)
    replay = patch(client, **{**body, "excluded_ids": [ids[1], ids[0], ids[0]]})
    assert replay == {**first, "replayed": True}
    for changed in ({"op": "read"}, {"excluded_ids": []}):
        response = client.patch(URL, json={**body, **changed})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "selection_operation_conflict"
    undo = patch(client, op="undo", operation_token=first["operation_token"])
    again = patch(client, op="undo", operation_token=first["operation_token"])
    assert again == {**undo, "replayed": True}
    patch(client, **body)
    excluded = NotificationStore(tmp_path).get(ids[2])
    assert excluded is not None and not excluded["dismissed"]


def test_undo_restores_read_state_but_never_intervening_writes(client, tmp_path):
    ids = seed(tmp_path, 4)
    store = NotificationStore(tmp_path)
    store.apply("read", ids=[(ids[0], 1)])
    selection = preview(client, state="all")
    result = patch(client, op="dismiss", selection_token=selection["token"])
    store.apply("read", ids=[(ids[1], 1)])
    # A state ABA cycle is not content revision and must still block Undo.
    store.apply("undismiss", ids=[(ids[2], 1)])
    store.apply("dismiss", ids=[(ids[2], 1)])
    undo = patch(client, op="undo", operation_token=result["operation_token"])
    assert undo["changed_count"] == 2 and undo["skipped_count"] == 2
    restored, intervening, skipped, written = (store.get(item) for item in ids)
    assert restored is not None and restored["read"] and not restored["dismissed"]
    assert intervening is not None and intervening["dismissed"]
    assert skipped is not None and skipped["dismissed"]
    assert written is not None and not written["read"] and not written["dismissed"]


@pytest.mark.parametrize("op", ["read", "unread", "dismiss", "undismiss"])
def test_operations_and_partial_receipts_exclude_asks(client, tmp_path, op):
    _open_ask(tmp_path)
    ask_path = tmp_path / "asks" / f"{SID}.json"
    before_ask = ask_path.read_bytes()
    ids = seed(tmp_path)
    store = NotificationStore(tmp_path)
    if op == "unread":
        store.apply("read", ids=[(ids[0], 1)])
    if op == "undismiss":
        store.apply("dismiss", ids=[(ids[0], 1)])
    before = store.get(ids[0])
    result = patch(
        client,
        op=op,
        ids=[{"id": ids[0], "revision": 1}, {"id": ask_ref(SID, "ask_1"), "revision": 1}],
        through_seq=store.counts()[1],
        receipt=True,
    )
    assert result["changed_count"] == 1 and result["skipped_ask_count"] == 1
    patch(client, op="undo", operation_token=result["operation_token"])
    after = store.get(ids[0])
    assert before is not None and after is not None
    assert all(before[key] == after[key] for key in bulk._STATE)
    assert ask_path.read_bytes() == before_ask


def test_read_asks_stay_in_unread_preview_and_never_change(client, tmp_path):
    _open_ask(tmp_path)
    store = NotificationStore(tmp_path)
    store.mark_refs([ask_ref(SID, "ask_1")], read=True)
    seed(tmp_path, 2)
    selection = preview(client, state="unread", bulk_priority="normal_up")
    assert selection["eligible_count"] == 2 and selection["skipped_ask_count"] == 1
    assert preview(client, kind="updates")["skipped_ask_count"] == 0
    assert preview(client, session_id=OTHER)["skipped_ask_count"] == 0
    result = patch(client, op="unread", selection_token=selection["token"])
    assert result["changed_count"] == 0
    assert result["waiting_unread_count"] == 0


def test_refresh_keeps_frozen_boundary_and_expiry(client, tmp_path):
    ids = seed(tmp_path, 4)
    selection = preview(client)
    seed(tmp_path, 2)
    refreshed = preview(client, selection_token=selection["token"], excluded_ids=[ids[0]])
    assert refreshed["eligible_count"] == 3
    for key in ("token", "through_seq", "expires_at"):
        assert refreshed[key] == selection[key]
    assert (
        client.get(
            URL, params={"preview": True, "selection_token": selection["token"], "kind": "all"}
        ).status_code
        == 422
    )


def test_expiry_wrong_scope_and_store_do_not_mutate(client, tmp_path, monkeypatch):
    seed(tmp_path)
    selection = preview(client)
    other_store = NotificationStore(tmp_path / "other")
    with pytest.raises(bulk.SelectionError):
        bulk.apply(other_store, "dismiss", scope="other", selection_token=selection["token"])
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "rotated-synthetic-token")
    result = client.patch(
        URL,
        headers={"Authorization": "Bearer rotated-synthetic-token"},
        json={"op": "dismiss", "selection_token": selection["token"]},
    )
    assert result.status_code == 409
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", SYNTHETIC_TOKEN)
    monkeypatch.setattr(bulk, "now_ms", lambda: selection["expires_at"])
    assert (
        client.patch(URL, json={"op": "dismiss", "selection_token": selection["token"]}).status_code
        == 409
    )
    assert NotificationStore(tmp_path).counts()[0] == 1


def test_limits_fail_closed_without_partial_writes(client, tmp_path, monkeypatch):
    seed(tmp_path, 3)
    monkeypatch.setattr(bulk, "MAX_ROWS", 2)
    assert client.get(URL, params={"preview": True}).status_code == 413
    monkeypatch.setattr(bulk, "MAX_ROWS", 3)
    selection = preview(client)
    monkeypatch.setattr(bulk, "MAX_ROWS", 2)
    assert (
        client.patch(URL, json={"op": "dismiss", "selection_token": selection["token"]}).status_code
        == 429
    )
    assert NotificationStore(tmp_path).counts()[0] == 3
    monkeypatch.setattr(bulk, "MAX_TOKENS", 1)
    assert client.get(URL, params={"preview": True}).status_code == 429


@pytest.mark.parametrize(
    "query",
    [
        {"kind": "bad"},
        {"bulk_priority": "low"},
        {"state": "bad"},
        {"session_id": "../private"},
        {"priority": "high"},
        {"source": "notify"},
        {"cursor": "next"},
        {"count_session_ids": [SID] * 51},
    ],
)
def test_invalid_filters_fail(client, query):
    assert client.get(URL, params={"preview": True, **query}).status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"op": "bad", "all": True},
        {"op": "undo", "ids": []},
        {"op": "dismiss", "selection_token": "x" * 43, "all": True},
        {"op": "dismiss", "selection_token": "x" * 43, "excluded_ids": ["n_" + "a" * 16] * 501},
        {"op": "dismiss", "operation_token": "x" * 43},
        {"op": "dismiss", "ids": [{"id": "n_" + "a" * 16, "revision": 1}], "receipt": True},
    ],
)
def test_invalid_write_targets_fail(client, body):
    assert client.patch(URL, json=body).status_code == 422


def test_partial_receipt_checks_watermark_and_revision(client, tmp_path):
    ids = seed(tmp_path, 2)
    store = NotificationStore(tmp_path)
    water = store.counts()[1]
    store.apply("read", ids=[(ids[0], 1)])
    result = patch(
        client,
        op="dismiss",
        ids=[{"id": ids[0], "revision": 1}, {"id": ids[1], "revision": 2}],
        through_seq=water,
        receipt=True,
    )
    assert result["changed_count"] == 0 and result["skipped_count"] == 2
    unchanged = store.get(ids[0])
    assert unchanged is not None and not unchanged["dismissed"]


def test_auth_and_legacy_contract_unchanged(client, tmp_path):
    ids = seed(tmp_path)
    assert (
        client.get(
            URL, headers={"Authorization": "Bearer wrong"}, params={"preview": True}
        ).status_code
        == 401
    )
    assert (
        client.patch(
            URL,
            headers={"Authorization": "Bearer wrong"},
            json={"op": "dismiss", "selection_token": "x" * 43},
        ).status_code
        == 401
    )
    result = patch(client, op="read", ids=[{"id": ids[0], "revision": 1}])
    assert result["changed"] == [ids[0]] and "operation_token" not in result
    assert len(json.dumps(result)) < 1000


def test_v1_store_upgrades_without_losing_rows(client, tmp_path):
    from local_operator.notifications import store as module

    store = NotificationStore(tmp_path)
    with sqlite3.connect(store.path) as conn:
        for statement in module._SCHEMA:
            if "bulk_" not in statement:
                conn.execute(statement)
        conn.execute("PRAGMA user_version=1")
    ids = seed(tmp_path)
    selection = preview(client)
    assert selection["eligible_count"] == 1
    upgraded = store.get(ids[0])
    assert upgraded is not None and upgraded["revision"] == 1
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM bulk_tokens").fetchone()[0] == 1


def test_failed_receipt_write_rolls_back_all_state(client, tmp_path):
    ids = seed(tmp_path, 2)
    selection = preview(client)
    store = NotificationStore(tmp_path)
    # A disk/schema failure after item mutation must not orphan that mutation.
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "CREATE TRIGGER fail_receipt BEFORE INSERT ON bulk_rows "
            "BEGIN SELECT RAISE(ABORT, 'synthetic receipt failure'); END"
        )
    response = client.patch(URL, json={"op": "dismiss", "selection_token": selection["token"]})
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "store_unavailable"
    assert all(
        (item_row := store.get(item)) is not None and not item_row["dismissed"] for item in ids
    )
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT op FROM bulk_tokens").fetchone()[0] is None
        assert conn.execute("SELECT count(*) FROM bulk_rows").fetchone()[0] == 0


def test_shared_relay_validator_refuses_desktop_only_targets():
    from pydantic import ValidationError

    from local_operator.server.routes.desktop_notifications import NotificationsPatch

    with pytest.raises(ValidationError):
        NotificationsPatch.model_validate({"op": "dismiss", "selection_token": "x" * 43})
    with pytest.raises(ValidationError):
        NotificationsPatch.model_validate(
            {
                "op": "dismiss",
                "receipt": True,
                "ids": [{"id": "n_" + "a" * 16, "revision": 1}],
                "through_seq": 1,
            }
        )


def test_concurrent_replay_commits_one_receipt(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    ids = seed(tmp_path, 12)
    store = NotificationStore(tmp_path)
    filters = {"state": "unread", "kind": "updates", "bulk_priority": "any"}
    selection = bulk.preview(store, filters, scope="synthetic-scope", skipped_ask_count=0)

    def submit():
        return bulk.apply(
            store, "dismiss", scope="synthetic-scope", selection_token=selection["token"]
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(lambda _: submit(), range(2)))
    assert sorted(answer["replayed"] for answer in answers) == [False, True]
    assert all(answer["changed_count"] == len(ids) for answer in answers)
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM bulk_rows").fetchone()[0] == 12


def test_preview_ask_scope_and_maximum_count_payload(client, tmp_path):
    _open_ask(tmp_path, session_id=SID)
    _open_ask(tmp_path, session_id=OTHER)
    seed(tmp_path)
    sessions = [format(index, "012x") for index in range(50)]
    selected = preview(client, kind="all", session_id=SID, count_session_ids=sessions)
    assert selected["skipped_ask_count"] == 1
    assert len(selected["session_counts"]) == 50
    assert len(json.dumps(selected)) < 5000
    assert client.get(URL, params={"preview": True, "bulk_priorty": "high"}).status_code == 422


def recount(client, token, excluded_ids=None, **headers):
    body = {"op": "preview", "selection_token": token}
    if excluded_ids is not None:
        body["excluded_ids"] = excluded_ids
    return client.patch(URL, json=body, headers=headers or None)


def test_body_recount_matches_get_and_never_mutates_or_extends(client, tmp_path, monkeypatch):
    """QA Q1: PATCH ``op:"preview"`` is the body form of the GET recount."""
    ids = seed(tmp_path, 4)
    selection = preview(client)
    seed(tmp_path, 2)
    before = NotificationStore(tmp_path).counts()
    by_body = recount(client, selection["token"], [ids[0], ids[1]])
    assert by_body.status_code == 200, by_body.text
    by_query = client.get(
        URL,
        params={
            "preview": True,
            "selection_token": selection["token"],
            "excluded_ids": [ids[0], ids[1]],
        },
    )
    assert by_body.json()["result"] == by_query.json()["result"]
    refreshed = by_body.json()["result"]["selection"]
    assert refreshed["eligible_count"] == 2
    for key in ("token", "through_seq", "expires_at"):
        assert refreshed[key] == selection[key]
    assert NotificationStore(tmp_path).counts() == before
    with sqlite3.connect(NotificationStore(tmp_path).path) as conn:
        assert conn.execute("SELECT count(*) FROM bulk_rows").fetchone()[0] == 0
        assert conn.execute("SELECT op FROM bulk_tokens").fetchone()[0] is None
    # Not a replay and not a commit: the same token still applies afterwards.
    applied = patch(client, op="dismiss", selection_token=selection["token"])
    assert applied["changed_count"] == 4 and applied["replayed"] is False
    assert recount(client, selection["token"]).json()["detail"] == {
        "code": "selection_already_committed"
    }


#: A rotated desktop bearer: valid auth, different receipt scope.
ROTATED = "-".join(("rotated", "synthetic", "bearer"))


def test_body_recount_keeps_auth_scope_and_expiry(client, tmp_path, monkeypatch):
    seed(tmp_path)
    selection = preview(client)
    assert recount(client, selection["token"], Authorization="Bearer wrong").status_code == 401
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", ROTATED)
    foreign = recount(client, selection["token"], Authorization="Bearer " + ROTATED)
    assert foreign.status_code == 409
    assert foreign.json()["detail"] == {"code": "selection_unavailable"}
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", SYNTHETIC_TOKEN)
    assert recount(client, "A" * 43).json()["detail"] == {"code": "selection_unavailable"}
    monkeypatch.setattr(bulk, "now_ms", lambda: selection["expires_at"])
    expired = recount(client, selection["token"])
    assert expired.status_code == 409
    assert expired.json()["detail"] == {"code": "selection_unavailable"}


@pytest.mark.parametrize(
    "body",
    [
        {"op": "preview"},
        {"op": "preview", "operation_token": "x" * 43},
        {"op": "preview", "selection_token": "x" * 43, "all": True},
        {"op": "preview", "selection_token": "x" * 43, "receipt": True},
        {"op": "preview", "selection_token": "x" * 43, "excluded_ids": ["bad"]},
        {"op": "preview", "selection_token": "x" * 43, "excluded_ids": ["n_" + "a" * 16] * 501},
        {"op": "preview", "ids": [{"id": "n_" + "a" * 16, "revision": 1}]},
    ],
)
def test_body_recount_refuses_mixed_or_missing_targets(client, body):
    assert client.patch(URL, json=body).status_code == 422


def test_relay_validator_refuses_preview_op():
    from pydantic import ValidationError

    from local_operator.server.routes.desktop_notifications import NotificationsPatch

    with pytest.raises(ValidationError):
        NotificationsPatch.model_validate({"op": "preview", "selection_token": "x" * 43})


def test_unbindable_receipt_watermark_is_422_not_500(client, tmp_path):
    """Review F3: SQLite cannot bind ``seq<=10**20``; refuse it at the edge."""
    (item,) = seed(tmp_path)
    body = {"op": "read", "ids": [{"id": item, "revision": 1}], "receipt": True}
    assert client.patch(URL, json={**body, "through_seq": 10**20}).status_code == 422
    assert client.patch(URL, json={**body, "through_seq": 2**63 - 1}).status_code == 200
    # The legacy path (shared with the relay) keeps clamping, unchanged.
    legacy = {"op": "read", "ids": [{"id": item, "revision": 1}], "through_seq": 10**20}
    assert client.patch(URL, json=legacy).status_code == 200


def test_relay_does_not_advertise_desktop_only_selection(monkeypatch):
    """Review F1: the relay answers ``preview`` with a legacy first page."""
    from local_operator.mobile import daemon as daemon_module
    from local_operator.server.features import static_feature_flags

    monkeypatch.setattr(daemon_module, "_FEATURE_FLAGS_MEMO", None)
    assert static_feature_flags()["notification_bulk_selection"] == 1
    flags = daemon_module._feature_flags()
    assert "notification_bulk_selection" not in flags
    assert flags["notification_contract"] == 1
