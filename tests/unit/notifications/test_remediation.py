"""Round-one regressions: observation scope, ask invalidation, secrets and dwell.

These exercise real SQLite writes, the production HTTP router, AskQueue/index,
and the registry tool. Synthetic values only; no provider or user data.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from local_operator.harness.types import ToolContext
from local_operator.notifications.items import apply_patch, list_page
from local_operator.notifications.store import NotificationStore, ask_ref
from local_operator.tools.registry import create_tools
from local_operator.variables import VariableStore
# Bind the fixture here: late plugin registration can leave it scoped to
# test_inbox when that module was already collected in the same CI shard.
# The explicit re-export preserves its pytest name without global plugin state.
from tests.unit.notifications.test_inbox import (
    AUTH,
    OTHER,
    SID,
    _open_ask,
    _run_awaitable,
    client as client,
)


@pytest.mark.parametrize("op", ["read", "dismiss"])
def test_writer_after_page_read_cannot_acknowledge_new_revision(tmp_path, monkeypatch, op):
    store = NotificationStore(tmp_path)
    item = store.notify(SID, title="revision one", key="r1")
    original = NotificationStore.list_items

    def interleaved(self, **kwargs):
        result = original(self, **kwargs)
        writer = threading.Thread(target=lambda: store.notify(SID, title="revision two", key="r1"))
        writer.start()
        writer.join()
        return result

    monkeypatch.setattr(NotificationStore, "list_items", interleaved)
    page = list_page(tmp_path)
    assert page["items"][0]["revision"] == 1
    assert page["high_water"] == 1
    changed = apply_patch(
        tmp_path, op, all_items=True, through_seq=page["high_water"], observed=page["observed"]
    )
    assert changed["changed"] == []
    assert changed["skipped"] == [{"id": item.item_id, "reason": "stale_revision"}]
    row = store.get(item.item_id)
    assert row is not None and row["revision"] == 2
    assert row["read_rev"] == 0 and row["dismissed_at"] is None


def test_true_all_enumerates_more_than_100_without_late_refs_or_revisions(tmp_path):
    store = NotificationStore(tmp_path)
    for i in range(125):
        store.notify(f"{i // 25 + 1:012x}", title=f"result {i}", key=f"k{i}")
    _open_ask(tmp_path, SID, "first")
    first = list_page(tmp_path, limit=50)
    boundary = first["high_water"]
    observed = list(first["observed"])
    raced = first["items"][0]
    store.notify(raced["session"]["id"], title="late revision", key=raced["key"])
    late = store.notify(OTHER, title="late item")
    _open_ask(tmp_path, OTHER, "late")
    cursor = first["next_cursor"]
    while cursor:
        page = list_page(tmp_path, cursor=cursor, limit=50, through_seq=boundary)
        assert page["high_water"] == boundary and "waiting" not in page
        observed.extend(page["observed"])
        cursor = page["next_cursor"]
    assert len(observed) == 126  # 125 rendered revisions + the first ask
    index_before = (tmp_path / "asks" / f"{SID}.json").read_bytes()
    result = apply_patch(tmp_path, "read", all_items=True, through_seq=boundary, observed=observed)
    assert len(result["changed"]) == 125
    assert result["skipped"] == [{"id": raced["id"], "reason": "stale_revision"}]
    assert store.counts()[0] == 2  # late new item and raced revision
    assert late.item_id not in result["changed"]
    assert ask_ref(OTHER, "late") not in store.ref_states([ask_ref(OTHER, "late")])
    assert (tmp_path / "asks" / f"{SID}.json").read_bytes() == index_before
    reread = list_page(tmp_path, state="all", session_id=SID)
    unread = apply_patch(
        tmp_path,
        "unread",
        session_id=SID,
        through_seq=reread["high_water"],
        observed=reread["observed"],
    )
    assert ask_ref(SID, "first") in unread["changed"]
    assert (tmp_path / "asks" / f"{SID}.json").read_bytes() == index_before


def test_filtered_page_never_widens_bulk_scope(tmp_path):
    store = NotificationStore(tmp_path)
    included = store.notify(SID, title="high", priority="high")
    excluded = store.notify(SID, title="normal")
    _open_ask(tmp_path)
    page = list_page(tmp_path, source="notify", priority="high", limit=1)
    result = apply_patch(
        tmp_path,
        "dismiss",
        all_items=True,
        through_seq=page["high_water"],
        observed=page["observed"],
    )
    assert result["changed"] == [included.item_id] and result["skipped"] == []
    row = store.get(excluded.item_id)
    assert row is not None and row["dismissed_at"] is None


def test_bulk_http_requires_observations(client):
    refused = client.patch(
        "/v1/desktop/notifications",
        headers=AUTH,
        json={"op": "read", "all": True, "through_seq": 0},
    )
    assert refused.status_code == 422
    empty = client.patch(
        "/v1/desktop/notifications",
        headers=AUTH,
        json={"op": "read", "all": True, "through_seq": 0, "observed": []},
    )
    assert empty.status_code == 200 and empty.json()["result"]["changed"] == []


@pytest.mark.asyncio
async def test_same_count_ask_replacement_emits_and_unchanged_tick_does_not(tmp_path, monkeypatch):
    from tests.unit.server.test_desktop_feed import _feed

    _open_ask(tmp_path, SID, "one")
    feed = _feed(tmp_path)
    frames = []
    monkeypatch.setattr(
        feed, "_publish", lambda kind, payload, **kw: frames.append((kind, payload))
    )
    await feed._maybe_emit_notifications()
    frames.clear()
    _open_ask(tmp_path, SID, "two")
    feed._asks_probed_at = 0
    await feed._maybe_emit_notifications()
    assert len(frames) == 1 and frames[0][0] == "notifications"
    assert frames[0][1]["waiting_count"] == 1
    assert list_page(tmp_path)["waiting"][0]["id"] == ask_ref(SID, "two")
    frames.clear()
    monkeypatch.setattr(feed, "_notifications_state", lambda: pytest.fail("unchanged tick read"))
    await feed._maybe_emit_notifications()
    assert frames == []


def test_registered_secret_urls_are_dropped_before_store(tmp_path):
    secret = "synthetic-not-a-real-secret-42"
    variables = VariableStore(env={})
    variables.register_redaction(secret)
    store = NotificationStore(tmp_path)

    async def door(**kw):
        result = store.notify(SID, title=kw["title"], body_md=kw["body"], links=kw["links"])
        return result.item_id

    context = ToolContext(cwd=str(tmp_path), notify_inbox=door, variables=variables)
    tool = create_tools(context, enabled=["notify"])[0]
    encoded = "".join(f"%{ord(c):02x}" for c in secret)
    links = [
        {"label": "unsafe", "url": u}
        for u in (
            f"https://example.invalid/?token={secret}",
            f"https://{secret}@example.invalid/",
            f"https://example.invalid/{secret}/report",
            f"https://example.invalid/{encoded}",
            f"https://example.invalid/#{secret}",
            "https://user:password@example.invalid/",
        )
    ]  # schema permits five at a time
    for start in (0, 5):
        result = _run_awaitable(
            tool.execute(
                "call", {"title": "probe", "links": links[start : start + 5]}, None, None, context
            )
        )
        assert not result.is_error
    rows, _ = store.list_items(state="all")
    assert all(json.loads(row["links_json"]) == [] for row in rows)
    assert secret not in json.dumps(rows) and encoded not in json.dumps(rows)


@pytest.mark.parametrize("visible_delta,input_delta", [(60000, -2000), (-1000, -2000), (-3000, 0)])
def test_seen_rejects_future_visibility_input_before_visible_and_zero_dwell(
    client,
    tmp_path,
    visible_delta,
    input_delta,
    monkeypatch,
):
    stamp = int(time.time() * 1000)
    monkeypatch.setattr("local_operator.notifications.store.now_ms", lambda: stamp)
    store = NotificationStore(tmp_path)
    row = store.notify(SID, title="receipt", anchor_id="anchor", now=stamp - 5000)
    body = {
        "items": [{"id": row.item_id, "revision": 1, "anchor_id": "anchor"}],
        "evidence": {
            "surface": "desktop",
            "session_id": SID,
            "input_kind": "keydown",
            "input_at_ms": stamp + input_delta,
            "visible_since_ms": stamp + visible_delta,
        },
    }
    response = client.post("/v1/desktop/notifications/seen", headers=AUTH, json=body)
    assert response.status_code == 200
    assert response.json()["result"]["skipped"][0]["reason"] == "stale_evidence"
    assert store.counts()[0] == 1
    body["evidence"].update(input_at_ms=stamp - 1500, visible_since_ms=stamp - 3000)
    positive = client.post("/v1/desktop/notifications/seen", headers=AUTH, json=body)
    assert positive.json()["result"]["changed"] == [row.item_id]
    assert store.counts()[0] == 0


def test_reading_ask_clears_badge_not_pending_group(tmp_path):
    from local_operator.asks import store as ask_store
    from local_operator.notifications.items import counts

    _open_ask(tmp_path)
    page = list_page(tmp_path)
    assert page["waiting_count"] == page["waiting_unread_count"] == 1
    receipt = apply_patch(
        tmp_path, "read", all_items=True, through_seq=page["high_water"], observed=page["observed"]
    )
    assert receipt["waiting_count"] == 1 and receipt["waiting_unread_count"] == 0
    assert counts(tmp_path)["waiting_unread_count"] == 0
    # The pending group survives a read; only settling the ask removes it.
    assert len(list_page(tmp_path, state="all")["waiting"]) == 1
    ask_store.write_entry(tmp_path, SID, cwd="/tmp/work", asks=[])
    assert counts(tmp_path)["waiting_count"] == 0


def test_undo_uses_original_revision_not_mutation_high_water(tmp_path):
    store = NotificationStore(tmp_path)
    item = store.notify(SID, title="first", key="undo")
    token = {"id": item.item_id, "revision": 1}
    dismissed = apply_patch(tmp_path, "dismiss", ids=[token])
    assert dismissed["changed"] == [item.item_id]
    store.notify(SID, title="changed after dismiss", key="undo")
    undo = apply_patch(tmp_path, "undismiss", ids=[token])
    assert undo["changed"] == []
    assert undo["skipped"] == [{"id": item.item_id, "reason": "stale_revision"}]
    current = store.get(item.item_id)
    assert current is not None and current["dismissed_at"] is not None


def test_normal_priority_filter_excludes_high_asks(tmp_path):
    _open_ask(tmp_path)
    page = list_page(tmp_path, priority="normal")
    assert page["waiting"] == [] if "waiting" in page else True
    assert page["observed"] == []


@pytest.mark.asyncio
async def test_factory_prompt_uses_session_notify_membership(tmp_path):
    from local_operator.session.transcript import Transcript
    from local_operator.session_factory import (
        _KnowledgeHooks,
        _make_system_blocks_provider,
    )
    from tests.unit.session.test_session import ScriptedStream, make_session

    provider = _make_system_blocks_provider([], Transcript(tmp_path / "prompt"), _KnowledgeHooks())
    assert "Before ending a turn, `notify`" not in "\n".join(await provider())
    session = make_session(tmp_path, ScriptedStream([]), system_blocks_provider=provider)
    blocks = await provider()
    assert any(tool.name == "notify" for tool in session._tools)
    assert "Before ending a turn, `notify`" in "\n".join(blocks)
    session.declare_one_shot_exit()
    assert "Before ending a turn, `notify`" not in "\n".join(await provider())


def test_detail_anchor_reads_archived_transcript(tmp_path):
    import gzip

    from local_operator.server.routes.desktop_notifications import anchor_excerpt

    row = {
        "type": "message",
        "payload": {
            "role": "tool",
            "tool_call_id": "archived-anchor",
            "content": [{"text": "Notified (id n_fixture)."}],
        },
    }
    archive = tmp_path / "transcript.jsonl.gz"
    archive.write_bytes(gzip.compress((json.dumps(row) + "\n").encode()))
    assert anchor_excerpt(tmp_path, "archived-anchor") == "Notified (id n_fixture)."
