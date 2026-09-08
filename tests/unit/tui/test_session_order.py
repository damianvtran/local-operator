"""The sidebar and phone keep immutable targets within each outcome category."""

from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

import pytest

from local_operator.mobile.daemon import SessionEntry, SessionTable
from local_operator.mobile.types import SessionProjection, SessionRecord
from local_operator.resume import SessionRow
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_catalog import CatalogEntry, rank_entries
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_session_sidebar import _quiesce_sidebar_refresh


def entries():
    rows = []
    for category, kind, busy, pending in [
        ("gate", "complete", True, "ask"),
        ("done", "complete", False, None),
        ("error", "error", False, None),
        ("interrupted", "interrupted", False, None),
        ("busy", "error", True, None),
        ("idle", "", False, None),
    ]:
        for suffix, created in [("old", 1), ("b", 2), ("a", 2)]:
            sid = f"{category}-{suffix}"
            rows.append(
                CatalogEntry(
                    SessionRow(
                        sid,
                        100 - created,
                        sid,
                        live_state="busy" if busy else "idle",
                        pending=pending,
                        created_at=created,
                    ),
                    unseen=bool(kind),
                    completion_kind=kind,
                )
            )
    return rows


def expected():
    return [
        f"{kind}-{suffix}"
        for kind in ("gate", "done", "error", "interrupted", "busy", "idle")
        for suffix in ("a", "b", "old")
    ]


def test_durable_catalog_loads_birth_even_when_transcript_activity_changes(tmp_path):
    import json

    from local_operator.tui.session_catalog import cached_session_rows

    for sid, created in [("old", 10), ("new", 20)]:
        directory = tmp_path / "sessions" / sid
        directory.mkdir(parents=True)
        (directory / "created_at.json").write_text(str(created))
        (directory / "transcript.jsonl").write_text(
            json.dumps(
                {
                    "id": sid,
                    "ts": 1,
                    "type": "message",
                    "payload": {"role": "user", "content": [{"type": "text", "text": sid}]},
                }
            )
            + "\n"
        )
    for tick in range(4):
        os.utime(tmp_path / "sessions" / "old" / "transcript.jsonl", (9999 + tick, 9999 + tick))
        rows = cached_session_rows(tmp_path)
        assert {row.id: row.created_at for row in rows} == {"old": 10, "new": 20}
        assert [e.id for e in rank_entries([CatalogEntry(row) for row in rows])] == ["new", "old"]


def test_category_birth_id_order_ignores_activity_and_input_order():
    rows = entries()
    assert [e.id for e in rank_entries(rows)] == expected()
    for tick in range(8):
        rows = [
            replace(e, row=e.row._replace(mtime=(index + tick) * 777))
            for index, e in enumerate(reversed(rows))
        ]
        assert [e.id for e in rank_entries(rows)] == expected()
    assert all(e.active for e in rows)
    cold = CatalogEntry(SessionRow("cold", 9999, "Viewed history", created_at=3))
    assert not cold.active
    assert rank_entries([cold, *rows])[-1] == cold


def test_mobile_uses_same_categories_birth_and_ties():
    table = SessionTable()
    durable = {}
    for index, e in enumerate(entries()):
        entry = SessionEntry(
            SessionRecord(
                pid=900000 + index,
                kind="tui",
                session_id=e.id,
                conversation_name=e.id,
                cwd="/synthetic",
                model_label="demo",
                control_port=1,
                control_key="test",
            )
        )
        entry.projection = SessionProjection(
            session_id=e.id, pid=entry.record.pid, kind="tui", streaming=e.row.live_state == "busy"
        )
        if e.row.pending:
            from local_operator.mobile.types import PendingRequest

            entry.projection.pending = PendingRequest(
                kind="ask", request_id="test", title="Question"
            )
        table.entries[entry.record.pid] = entry
        table._attention_states[f"session/{e.id}"] = {"unseen": e.unseen, "kind": e.completion_kind}
        durable[e.id] = e.row
    for tick in range(8):
        durable = {
            sid: row._replace(mtime=tick * 1000 + i)
            for i, (sid, row) in enumerate(reversed(list(durable.items())))
        }
        for entry in table.entries.values():
            entry.record.heartbeat_at += 1000
            entry.record.started_at += 1000
        assert [r["session_id"] for r in table._merge_summaries(durable)] == expected()


@pytest.mark.asyncio
async def test_activity_refresh_keeps_mouse_target_and_keyboard_cursor(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda self: None)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 45)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        _quiesce_sidebar_refresh(app)
        sidebar = app._session_sidebar
        rows = entries()
        sidebar.set_entries(rows)
        sidebar.cursor_id = "busy-b"
        await pilot.pause()
        target_y = next(
            y
            for y in range(sidebar.size.height)
            if (entry := sidebar._entry_at(y)) is not None and entry.id == "busy-b"
        )
        for tick in range(5):
            rows = [
                replace(e, row=e.row._replace(mtime=1000 * tick + i))
                for i, e in enumerate(reversed(rows))
            ]
            sidebar.set_entries(rows)
            await pilot.pause()
            assert sidebar.cursor_id == "busy-b"
            target = sidebar._entry_at(target_y)
            assert target is not None and target.id == "busy-b"
        selected = []
        with patch.object(app._sidebar_navigation, "select", side_effect=selected.append):
            await pilot.click("#session-sidebar", offset=(8, target_y))
            await pilot.pause()
        assert selected == ["busy-b"]
