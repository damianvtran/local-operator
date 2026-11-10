"""The desktop routes that page a write/edit call's complete, redacted diff.

Contract (frozen UI v1): ``GET /v1/desktop/sessions/{sid}/tool-calls/{call}/diff``
and its child twin under ``/children/{child}/``. Ids only -- the client never
sends a digest. Unknown/foreign session, call or non-write/edit call: one 404.
Bad paging: 422. Unservable-but-captured states: 200 with no rows. Every
response is ``no-store`` + ``nosniff`` and bearer-gated.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.tools import diff_archive
from tests.unit.server.test_desktop_sessions import (
    CHILD_ID,
    PARENT_ID,
    UNNAMED_CHILD_ID,
    name_child,
    subagent_child,
)

ROWS = ["--- ", "+++ ", "@@ -0,0 +1,3 @@", "+first", "+middle", "+final"]


def _record(
    tmp_path: Path, session_dir: Path, call_id: str, *, tool: str = "write", rows=ROWS
) -> dict[str, Any]:
    """An archived artifact plus the persisted tool row that names it.

    The artifact is written through the production writer under the same
    config root the host reads, bound to ``session_dir``'s own id.
    """
    from local_operator.tools.spill import SpillStore

    store = SpillStore(tmp_path / diff_archive.DIFF_ARCHIVE_DIRNAME)
    header = json.dumps(
        {
            "v": 1,
            "kind": "diff-archive",
            "session_id": session_dir.name,
            "call_id": call_id,
            "tool": tool,
            "rows": len(rows),
        }
    )
    meta = store.write("\n".join([header, *(json.dumps(r) for r in rows)]), tool_name=tool)
    assert meta is not None
    desc = diff_archive.descriptor("archived", rows=len(rows), digest=meta.digest)
    payload = {
        "role": "tool",
        "tool_call_id": call_id,
        "tool_name": tool,
        "provider_payload": {
            "details": {"diff": ROWS[:4], "diff_truncated_rows": 2, "diff_full": desc}
        },
    }
    with (session_dir / "transcript.jsonl").open("a") as handle:
        handle.write(
            json.dumps({"id": f"m-{call_id}", "ts": 0, "type": "message", "payload": payload})
            + "\n"
        )
    return desc


@pytest.mark.asyncio
async def test_parent_and_child_adapters_page_and_refuse_uniformly(tmp_path):
    parent_dir, child_dir = subagent_child(tmp_path)
    name_child(parent_dir, child_dir)
    _record(tmp_path, parent_dir, "p-1")
    _record(tmp_path, child_dir, "c-1")
    pool = DesktopSessions(tmp_path)
    page = await pool.tool_call_diff(PARENT_ID, "p-1", start=0, limit=2)
    assert page["rows"] == ROWS[:2] and page["next_start"] == 2 and page["total_rows"] == 6
    child = await pool.child_tool_call_diff(PARENT_ID, CHILD_ID, "c-1", start=4, limit=200)
    assert child["rows"] == ROWS[4:] and child["next_start"] is None
    assert child["hunk_header"] == "@@ -0,0 +1,3 @@"
    # Wrong scope: the child's call through the parent, the parent's through the child.
    for coro in (
        pool.tool_call_diff(PARENT_ID, "c-1"),
        pool.child_tool_call_diff(PARENT_ID, CHILD_ID, "p-1"),
        pool.tool_call_diff(CHILD_ID, "c-1"),  # a subagent is not a user session
        pool.tool_call_diff("ffffffffffff", "p-1"),
        pool.tool_call_diff("../" + PARENT_ID, "p-1"),
        pool.tool_call_diff(PARENT_ID, "nope"),
    ):
        with pytest.raises(KeyError):
            await coro
    with pytest.raises(Exception) as refused:
        await pool.child_tool_call_diff(PARENT_ID, UNNAMED_CHILD_ID, "c-1")
    assert not isinstance(refused.value, AssertionError)


@pytest.mark.asyncio
async def test_routes_are_bearer_gated_no_store_and_follow_the_contract(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from local_operator.server.app import app

    parent_dir, child_dir = subagent_child(tmp_path)
    name_child(parent_dir, child_dir)
    _record(tmp_path, parent_dir, "p-1")
    _record(tmp_path, child_dir, "c-1")
    _record(tmp_path, parent_dir, "b-1", tool="bash")
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path))
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    parent = f"/v1/desktop/sessions/{PARENT_ID}/tool-calls/p-1/diff"
    child = f"/v1/desktop/sessions/{PARENT_ID}/children/{CHILD_ID}/tool-calls/c-1/diff"
    auth = {"Authorization": "Bearer token"}

    with TestClient(app, headers={"Host": "127.0.0.1:1111"}) as client:
        assert client.get(parent).status_code == 401
        assert client.get(child).status_code == 401
        ok = client.get(parent + "?start=0&limit=200", headers=auth)
        assert ok.status_code == 200, ok.text
        assert ok.headers["cache-control"] == "no-store"
        assert ok.headers["x-content-type-options"] == "nosniff"
        result = ok.json()["result"]
        assert result == {
            "state": "archived",
            "start": 0,
            "total_rows": 6,
            "rows": ROWS,
            "hunk_header": None,
            "next_start": None,
            "clipped_rows": [],
        }
        child_ok = client.get(child + "?start=3&limit=1", headers=auth).json()["result"]
        assert child_ok["rows"] == ["+first"] and child_ok["next_start"] == 4
        missing = [
            f"/v1/desktop/sessions/{PARENT_ID}/tool-calls/nope/diff",
            f"/v1/desktop/sessions/{PARENT_ID}/tool-calls/c-1/diff",
            f"/v1/desktop/sessions/{PARENT_ID}/tool-calls/b-1/diff",
            "/v1/desktop/sessions/ffffffffffff/tool-calls/p-1/diff",
            f"/v1/desktop/sessions/{PARENT_ID}/children/{CHILD_ID}/tool-calls/p-1/diff",
            f"/v1/desktop/sessions/{PARENT_ID}/children/{UNNAMED_CHILD_ID}/tool-calls/c-1/diff",
        ]
        statuses = {path: client.get(path, headers=auth).status_code for path in missing}
        assert set(statuses.values()) == {404}, statuses
        for query in ("start=-1", "limit=0", "limit=201", "start=6", "start=x"):
            bad = client.get(parent + "?" + query, headers=auth)
            assert bad.status_code == 422, (query, bad.text)
        long_id = client.get(
            f"/v1/desktop/sessions/{PARENT_ID}/tool-calls/{'a' * 257}/diff", headers=auth
        )
        assert long_id.status_code == 422
        # Eviction is an honest 200 answer, not a 404 or a false "not recorded".
        for path in (tmp_path / diff_archive.DIFF_ARCHIVE_DIRNAME).iterdir():
            path.unlink()
        evicted = client.get(parent, headers=auth).json()["result"]
        assert evicted["state"] == "evicted" and evicted["rows"] == []


def test_openapi_declares_both_routes_with_the_page_model():
    from local_operator.server.app import app

    paths = app.openapi()["paths"]
    for path in (
        "/v1/desktop/sessions/{session_id}/tool-calls/{call_id}/diff",
        "/v1/desktop/sessions/{session_id}/children/{child_id}/tool-calls/{call_id}/diff",
    ):
        params = {p["name"]: p for p in paths[path]["get"]["parameters"]}
        assert params["limit"]["schema"]["maximum"] == 200
        assert params["call_id"]["schema"]["maxLength"] == 256
        assert "digest" not in params
