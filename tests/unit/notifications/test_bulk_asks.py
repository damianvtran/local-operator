"""V2 frozen ask notification reads through production routes and real ask logs.

In-process HTTP only: the shared-machine live-rig hold prohibits new socket
servers. These tests do not claim independent real-path or rendered UI proof.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import store as ask_store
from local_operator.notifications import bulk
from local_operator.notifications.store import NotificationStore, ask_ref
from tests.unit.asks.test_store import _answered, _queued
from tests.unit.notifications.test_bulk import URL
from tests.unit.notifications.test_bulk import client as _route_client
from tests.unit.notifications.test_bulk import patch, preview, recount, seed
from tests.unit.notifications.test_inbox import OTHER, SID

# Reuse the production-route fixture without starting another live app rig.
client = _route_client


def queue(root: Path, count: int = 1, *, session: str = SID, start: int = 0, deadline: int = 0):
    directory = ask_store.session_dir(root, session)
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(start, start + count):
        assert ask_store.append_event(
            directory,
            _queued(
                f"ask_{index}",
                at=int(time.time() * 1000),
                expires_at=deadline or int(time.time() * 1000) + 86_400_000,
                timeout_s=0,
            ),
        )
    publish_index(root, session)
    return [ask_ref(session, f"ask_{index}") for index in range(start, start + count)]


def publish_index(root: Path, session: str = SID):
    directory = ask_store.session_dir(root, session)
    records = ask_store.fold(
        ask_store.read_events(directory), ask_store.now_ms(), session_id=session
    )
    ask_store.write_entry(
        root, session, cwd=str(root), asks=[ask_store.pending_row(r) for r in records]
    )


def row(store: NotificationStore, item_id: str) -> dict[str, Any]:
    """``store.get`` for a row the test just seeded; narrows the Optional for pyright."""
    found = store.get(item_id)
    assert found is not None, item_id
    return found


def log_bytes(root: Path, session: str = SID):
    return ask_store.asks_log_path(ask_store.session_dir(root, session)).read_bytes()


def select(client, **filters):
    return preview(client, include_ask_reads="true", **filters)


def test_frozen_all_item_read_beyond_page_and_new_ask_exclusion(client, tmp_path):
    refs = queue(tmp_path, 125)
    updates = seed(tmp_path, 4)
    selection = select(client, state="all", count_session_ids=[SID])
    assert selection["selection_version"] == 2
    assert selection["eligible_count"] == 129
    assert selection["eligible_ask_count"] == 125
    assert selection["eligible_update_count"] == 4
    assert selection["operation_counts"]["read"] == {"eligible_count": 129, "skipped_ask_count": 0}
    assert selection["operation_counts"]["dismiss"] == {
        "eligible_count": 4,
        "skipped_ask_count": 125,
    }
    new_ref = queue(tmp_path, start=125)[0]
    before = log_bytes(tmp_path)
    result = patch(client, op="read", selection_token=selection["token"], excluded_ids=[refs[0]])
    assert result["changed_count"] == 128
    assert result["changed_ask_count"] == 124 and result["changed_update_count"] == 4
    assert result["waiting_count"] == 126 and result["waiting_unread_count"] == 2
    assert result["waiting_unread_delta"] == -124 and result["unread_delta"] == -4
    assert result["excluded_count"] == 1
    assert result["session_counts"][SID]["changed_ask_count"] == 124
    marks = NotificationStore(tmp_path).ref_states([refs[0], new_ref])
    assert not any(mark.get("read_at") is not None for mark in marks.values())
    assert all(row(NotificationStore(tmp_path), item)["read"] for item in updates)
    assert log_bytes(tmp_path) == before
    undo = patch(client, op="undo", operation_token=result["operation_token"])
    assert undo["changed_count"] == 128 and undo["waiting_unread_delta"] == 124
    assert undo["waiting_count"] == 126 and undo["waiting_unread_count"] == 126
    assert log_bytes(tmp_path) == before
    assert len(json.dumps(result)) < 2000


def test_v1_token_stays_update_only_on_v2_server(client, tmp_path):
    queue(tmp_path)
    seed(tmp_path)
    old = preview(client)
    assert "selection_version" not in old and old["eligible_count"] == 1
    result = patch(client, op="read", selection_token=old["token"])
    assert result["changed_count"] == 1 and result["skipped_ask_count"] == 1
    assert "changed_ask_count" not in result and result["waiting_unread_count"] == 1


@pytest.mark.parametrize("op", ["dismiss", "undismiss"])
def test_v2_dismiss_never_acknowledges_or_settles_asks(client, tmp_path, op):
    refs = queue(tmp_path, 2)
    ids = seed(tmp_path)
    if op == "undismiss":
        NotificationStore(tmp_path).apply("dismiss", ids=[(ids[0], 1)])
    before = log_bytes(tmp_path)
    selection = select(client, state="all" if op == "dismiss" else "dismissed")
    result = patch(client, op=op, selection_token=selection["token"])
    assert result["changed_update_count"] == 1 and result["changed_ask_count"] == 0
    assert result["waiting_count"] == 2 and result["waiting_unread_count"] == 2
    assert result["skipped_ask_count"] == (2 if op == "dismiss" else 0)
    patch(client, op="undo", operation_token=result["operation_token"])
    assert NotificationStore(tmp_path).ref_states(refs) == {}
    assert log_bytes(tmp_path) == before


def test_read_asks_remain_in_unread_selection_and_unread_is_reversible(client, tmp_path):
    refs = queue(tmp_path, 3)
    store = NotificationStore(tmp_path)
    store.mark_refs(refs[:2], read=True)
    selection = select(client, state="unread", kind="asks")
    assert selection["eligible_count"] == 3 and selection["eligible_unread_count"] == 1
    assert selection["eligible_unread_ask_count"] == 1 and selection["eligible_update_count"] == 0
    before = store.ref_states(refs)
    logs = log_bytes(tmp_path)
    result = patch(client, op="unread", selection_token=selection["token"])
    assert result["changed_ask_count"] == 2 and result["skipped_count"] == 1
    assert result["waiting_count"] == 3 and result["waiting_unread_delta"] == 2
    undo = patch(client, op="undo", operation_token=result["operation_token"])
    assert undo["waiting_unread_delta"] == -2
    after = store.ref_states(refs)
    assert [after[r]["read_at"] for r in refs[:2]] == [before[r]["read_at"] for r in refs[:2]]
    assert log_bytes(tmp_path) == logs


def test_settled_log_beats_stale_index_and_concurrent_marks_are_skipped(client, tmp_path):
    refs = queue(tmp_path, 3)
    selection = select(client)
    directory = ask_store.session_dir(tmp_path, SID)
    ask_store.append_event(directory, _answered("ask_0", at=ask_store.now_ms()))
    # Intentionally do NOT publish the settled index: exercise crash-window safety.
    NotificationStore(tmp_path).mark_refs([refs[1]], read=True)
    before = log_bytes(tmp_path)
    result = patch(client, op="read", selection_token=selection["token"])
    assert result["changed_ask_count"] == 1 and result["skipped_count"] == 2
    assert refs[0] not in NotificationStore(tmp_path).ref_states(refs)
    assert log_bytes(tmp_path) == before


def test_undo_skips_settled_and_aba_changed_refs(client, tmp_path):
    refs = queue(tmp_path, 3)
    selection = select(client)
    result = patch(client, op="read", selection_token=selection["token"])
    ask_store.append_event(
        ask_store.session_dir(tmp_path, SID), _answered("ask_0", at=ask_store.now_ms())
    )
    store = NotificationStore(tmp_path)
    store.mark_refs([refs[1]], read=False)
    store.mark_refs([refs[1]], read=True)
    before = log_bytes(tmp_path)
    undo = patch(client, op="undo", operation_token=result["operation_token"])
    assert undo["changed_ask_count"] == 1 and undo["skipped_count"] == 2
    assert store.ref_states([refs[0], refs[1]])[refs[0]]["read_at"] is not None
    assert store.ref_states([refs[2]])[refs[2]]["read_at"] is None
    assert log_bytes(tmp_path) == before


def test_expired_ask_is_not_acknowledged(client, tmp_path, monkeypatch):
    deadline = ask_store.now_ms() + 10_000
    refs = queue(tmp_path, deadline=deadline)
    token = select(client)["token"]
    monkeypatch.setattr(ask_store, "now_ms", lambda: deadline + 1)
    result = patch(client, op="read", selection_token=token)
    assert result["changed_count"] == 0 and result["skipped_count"] == 1
    assert NotificationStore(tmp_path).ref_states(refs) == {}


def test_recount_keeps_scope_op_choice_and_replay_are_fenced(client, tmp_path):
    refs = queue(tmp_path, 2)
    token = select(client)
    queue(tmp_path, start=2)
    refreshed = preview(client, selection_token=token["token"], excluded_ids=[refs[0]])
    assert refreshed["eligible_ask_count"] == 1
    assert all(refreshed[key] == token[key] for key in ("token", "through_seq", "expires_at"))
    body = {"op": "unread", "selection_token": token["token"], "excluded_ids": [refs[0]]}
    first = patch(client, **body)
    assert first["changed_ask_count"] == 0
    assert patch(client, **body)["replayed"]
    assert client.patch(URL, json={**body, "op": "read"}).status_code == 409
    assert client.patch(URL, json={**body, "excluded_ids": []}).status_code == 409
    assert (
        client.get(
            URL,
            params={"preview": True, "selection_token": token["token"], "include_ask_reads": True},
        ).status_code
        == 422
    )


def test_v2_body_recount_matches_get_and_subtracts_ask_exclusions(client, tmp_path):
    """The PATCH op:"preview" recount (#3103 Q1) must carry v2 ask selections too.

    Ask refs are the long ids (~81 bytes) that pushed the GET form over h11's
    head limit, so the body form is the one a v2 client actually uses.
    """
    refs = queue(tmp_path, 3)
    updates = seed(tmp_path, 2)
    token = select(client, state="all", count_session_ids=[SID])
    queue(tmp_path, start=3)  # arrives after the freeze: must never enter
    excluded = [refs[0], updates[0]]
    response = recount(client, token["token"], excluded)
    assert response.status_code == 200, response.text
    body = response.json()["result"]["selection"]
    assert body == preview(client, selection_token=token["token"], excluded_ids=excluded)
    assert body["selection_version"] == 2
    assert body["eligible_ask_count"] == 2 and body["eligible_update_count"] == 1
    assert body["eligible_count"] == 3 and body["skipped_ask_count"] == 0
    assert all(body[key] == token[key] for key in ("token", "through_seq", "expires_at"))
    # Read metadata belongs to receipts; a recount would silently drop it.
    refused = client.patch(
        URL,
        json={"op": "preview", "selection_token": token["token"], "reconcile_ids": [refs[1]]},
    )
    assert refused.status_code == 422
    # Still uncommitted: the token applies afterwards with the same exclusions.
    applied = patch(client, op="read", selection_token=token["token"], excluded_ids=excluded)
    assert applied["changed_ask_count"] == 2 and applied["changed_update_count"] == 1


def test_update_and_mark_receipts_roll_back_together(client, tmp_path):
    refs = queue(tmp_path)
    ids = seed(tmp_path)
    token = select(client)["token"]
    store = NotificationStore(tmp_path)
    before = log_bytes(tmp_path)
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "CREATE TRIGGER fail_ask_receipt BEFORE UPDATE ON bulk_ask_refs "
            "WHEN NEW.post_seq IS NOT NULL BEGIN SELECT RAISE(ABORT, 'synthetic'); END"
        )
    response = client.patch(URL, json={"op": "read", "selection_token": token})
    assert response.status_code == 500
    assert not row(store, ids[0])["read"] and store.ref_states(refs) == {}
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM bulk_rows").fetchone()[0] == 0
        assert conn.execute("SELECT op FROM bulk_tokens").fetchone()[0] is None
    assert log_bytes(tmp_path) == before


def test_combined_capacity_filter_scopes_and_expiry_cleanup(client, tmp_path, monkeypatch):
    queue(tmp_path, 2)
    queue(tmp_path, session=OTHER)
    seed(tmp_path, 2)
    monkeypatch.setattr(bulk, "MAX_ROWS", 4)
    assert client.get(URL, params={"preview": True, "include_ask_reads": True}).status_code == 413
    selected = select(client, session_id=SID, kind="asks", bulk_priority="normal_up")
    assert selected["eligible_count"] == 2
    another = select(client, session_id=SID, kind="asks")
    assert another["eligible_count"] == 2
    assert (
        client.get(
            URL, params={"preview": True, "include_ask_reads": True, "kind": "asks"}
        ).status_code
        == 429
    )
    monkeypatch.setattr(bulk, "now_ms", lambda: another["expires_at"])
    fresh = select(client, kind="asks")
    assert fresh["eligible_count"] == 3
    with sqlite3.connect(NotificationStore(tmp_path).path) as conn:
        assert conn.execute("SELECT count(*) FROM bulk_ask_refs").fetchone()[0] == 3


def test_ask_capability_and_preview_flag_validation(client):
    from local_operator.server.features import static_feature_flags

    assert static_feature_flags()["notification_bulk_ask_read"] == 1
    assert client.get(URL, params={"include_ask_reads": True}).status_code == 422
    assert (
        client.get(URL, params={"preview": True, "include_ask_reads": "wrong"}).status_code == 422
    )


def test_settlement_racing_acknowledgement_leaves_only_a_harmless_mark(
    client, tmp_path, monkeypatch
):
    refs = queue(tmp_path)
    token = select(client)["token"]
    original = NotificationStore.write_ref
    raced_log = []

    def settle_then_write(conn, ref, read_at):
        assert ask_store.append_event(
            ask_store.session_dir(tmp_path, SID), _answered("ask_0", at=ask_store.now_ms())
        )
        publish_index(tmp_path)
        raced_log.append(log_bytes(tmp_path))
        return original(conn, ref, read_at)

    with monkeypatch.context() as race:
        race.setattr(NotificationStore, "write_ref", staticmethod(settle_then_write))
        result = patch(client, op="read", selection_token=token)
    assert result["changed_ask_count"] == 1
    assert result["waiting_count"] == 0 and result["waiting_unread_count"] == 0
    assert NotificationStore(tmp_path).ref_states(refs)[refs[0]]["read_at"] is not None
    undo = patch(client, op="undo", operation_token=result["operation_token"])
    assert undo["changed_count"] == 0 and undo["skipped_count"] == 1
    assert log_bytes(tmp_path) == raced_log[0]


def test_ask_receipts_keep_auth_expiry_and_lifecycle_fences(client, tmp_path, monkeypatch):
    refs = queue(tmp_path)
    token = select(client)
    before = log_bytes(tmp_path)
    with pytest.raises(bulk.SelectionError, match="selection_unavailable"):
        bulk.apply(
            NotificationStore(tmp_path), "read", selection_token=token["token"], scope="wrong"
        )
    monkeypatch.setattr(bulk, "now_ms", lambda: token["expires_at"])
    assert (
        client.patch(URL, json={"op": "read", "selection_token": token["token"]}).status_code == 409
    )
    assert NotificationStore(tmp_path).ref_states(refs) == {}
    assert log_bytes(tmp_path) == before


def test_changed_question_is_not_acknowledged_from_an_old_snapshot(client, tmp_path):
    refs = queue(tmp_path)
    token = select(client)["token"]
    entry = ask_store.read_entry(tmp_path, SID)
    assert entry is not None
    entry["asks"][0]["questions"][0]["question"] = "Revised question"
    ask_store.write_entry(tmp_path, SID, cwd=str(tmp_path), asks=entry["asks"])
    before = log_bytes(tmp_path)
    result = patch(client, op="read", selection_token=token)
    assert result["changed_count"] == 0 and result["skipped_count"] == 1
    assert NotificationStore(tmp_path).ref_states(refs) == {}
    assert log_bytes(tmp_path) == before


def test_schema2_selection_survives_schema3_migration(client, tmp_path):
    from local_operator.notifications import store as module

    store = NotificationStore(tmp_path)
    with sqlite3.connect(store.path) as conn:
        for statement in module._SCHEMA:
            if "bulk_ask_refs" not in statement:
                conn.execute(statement)
        conn.execute("PRAGMA user_version=2")
        conn.execute(
            "INSERT INTO bulk_tokens(token,scope,expires_at,filters_json,through_seq,"
            "eligible_count,skipped_ask_count) VALUES (?,?,?,?,?,?,?)",
            (
                "x" * 43,
                "old-scope",
                ask_store.now_ms() + 600_000,
                json.dumps({"state": "all", "kind": "all", "bulk_priority": "any"}),
                0,
                0,
                0,
            ),
        )
    result = bulk.apply(store, "read", scope="old-scope", selection_token="x" * 43)
    assert result["changed_count"] == 0 and "selection_version" not in result
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM bulk_ask_refs").fetchone()[0] == 0


def test_bounded_reconcile_preserves_deep_skipped_revision_and_scope(client, tmp_path):
    store = NotificationStore(tmp_path)
    deep = store.notify(SID, title="Deep old", body_md="original", key="deep")
    ids = seed(tmp_path, 120)
    other = seed(tmp_path, session=OTHER)[0]
    selected = select(client, session_id=SID, count_session_ids=[SID, OTHER], state="all")
    store.notify(SID, title="Deep revised", body_md="private long body " * 500, key="deep")
    missing = "n_" + "a" * 16
    body = {
        "op": "read",
        "selection_token": selected["token"],
        "reconcile_ids": [deep.item_id, other, missing],
    }
    result = patch(client, **body)
    assert result["changed_count"] == 120 and result["skipped_count"] == 1
    assert result["session_counts"][SID]["unread_count"] == 1
    assert result["session_counts"][OTHER]["unread_count"] == 1
    assert result["counts_high_water"] == store.counts()[1]
    snapshot = result["reconciliation"]
    assert snapshot["absent_ids"] == [missing]
    rows = {row["id"]: row for row in snapshot["items"]}
    assert rows[deep.item_id]["revision"] == 2 and rows[deep.item_id]["read"] is False
    assert rows[deep.item_id]["seq"] > selected["through_seq"]
    assert rows[other]["read"] is False  # read metadata did not broaden the action
    assert "body_md" not in rows[deep.item_id] and len(rows[deep.item_id]["excerpt"]) < 500
    # Reconciliation is not replay identity; refresh requested metadata while
    # retaining the old operation aggregate fence and original mutation receipt.
    store.apply("read", ids=[(deep.item_id, 2)])
    replay = patch(client, **{**body, "reconcile_ids": [deep.item_id]})
    assert replay["replayed"] and replay["counts_high_water"] == result["counts_high_water"]
    assert replay["session_counts"][SID]["unread_count"] == 1
    fresh = replay["reconciliation"]
    assert fresh["high_water"] > replay["counts_high_water"]
    assert fresh["items"][0]["read"] is True
    assert fresh["items"][0]["seq"] > rows[deep.item_id]["seq"]
    # A client already holding fresh.seq can recognize and refuse the older
    # snapshot. The actual facade's compare/apply implementation lives in #1471.
    assert rows[deep.item_id]["seq"] < fresh["items"][0]["seq"]
    assert all(row(store, item)["read"] for item in ids)


def test_undo_reconciliation_preserves_intervening_write_and_ask_projection(client, tmp_path):
    refs = queue(tmp_path)
    ids = seed(tmp_path, 2)
    selected = select(client, count_session_ids=[SID])
    result = patch(client, op="read", selection_token=selected["token"])
    store = NotificationStore(tmp_path)
    store.apply("dismiss", ids=[(ids[0], 1)])
    undo = patch(
        client, op="undo", operation_token=result["operation_token"], reconcile_ids=[*ids, *refs]
    )
    assert undo["skipped_count"] == 1 and undo["changed_count"] == 2
    assert undo["session_counts"][SID]["unread_count"] == 1
    rows = {row["id"]: row for row in undo["reconciliation"]["items"]}
    assert rows[ids[0]]["dismissed"] is True and rows[ids[0]]["read"] is True
    assert rows[ids[1]]["read"] is False and rows[refs[0]]["read"] is False
    assert rows[refs[0]]["ask"]["status"] == "open"
    assert "questions" not in rows[refs[0]] and "answers" not in rows[refs[0]]["ask"]


def test_reconciliation_bounds_and_invalid_token_fail_before_read(client, tmp_path, monkeypatch):
    from local_operator.notifications import reconcile

    refs = queue(tmp_path)
    selected = select(client)
    assert (
        client.patch(
            URL,
            json={"op": "read", "selection_token": selected["token"], "reconcile_ids": refs * 501},
        ).status_code
        == 422
    )
    assert (
        client.patch(
            URL, json={"op": "read", "ids": [{"id": refs[0], "revision": 1}], "reconcile_ids": refs}
        ).status_code
        == 422
    )

    def forbidden(*args, **kwargs):
        pytest.fail("expired/unknown receipt disclosed reconciliation data")

    monkeypatch.setattr(reconcile, "read_rows", forbidden)
    assert (
        client.patch(
            URL, json={"op": "read", "selection_token": "x" * 43, "reconcile_ids": refs}
        ).status_code
        == 409
    )
    monkeypatch.setattr(bulk, "now_ms", lambda: selected["expires_at"])
    assert (
        client.patch(
            URL, json={"op": "read", "selection_token": selected["token"], "reconcile_ids": refs}
        ).status_code
        == 409
    )


def test_reconciliation_items_include_read_waiting_row_and_settled_absence(client, tmp_path):
    refs = queue(tmp_path)
    selected = select(client, kind="asks")
    body = {"op": "read", "selection_token": selected["token"], "reconcile_ids": refs}
    result = patch(client, **body)
    row = result["reconciliation"]["items"][0]
    assert row["id"] == refs[0] and row["source"] == "ask" and row["read"] is True
    assert row["ask"]["status"] == "open" and "answers" not in row["ask"]
    assert "waiting" not in result["reconciliation"]
    assert result["waiting_count"] == 1 and result["waiting_unread_count"] == 0
    ask_store.append_event(
        ask_store.session_dir(tmp_path, SID), _answered("ask_0", at=ask_store.now_ms())
    )
    # Deliberately leave the projection stale: a committed settled log must not
    # be returned as a pending row merely because the writer crashed mid-publish.
    before = log_bytes(tmp_path)
    replay = patch(client, **body)
    assert replay["replayed"]
    assert replay["reconciliation"]["items"] == []
    assert replay["reconciliation"]["absent_ids"] == refs
    assert log_bytes(tmp_path) == before
