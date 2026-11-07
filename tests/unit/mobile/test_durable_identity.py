"""Model, effort and cwd reach the phone for sessions with no attached runtime.

The operator's report: the phone showed ``n/a`` for model, directory and effort
on any session whose runtime was not attached. The list row and the rebuilt
session detail both read those fields from the LIVE projection or the discovery
record only, and both die with the process. The facts are durable already — the
journal's ``selected_model`` row (model + effort, written at birth and on every
``/model``/``/effort``) and the frontend checkpoint / ``desktop.json`` marker /
wake index (cwd) — so these tests pin that the daemon reads them:

* ``_durable_identity`` — the reader, including an old row with no effort;
* ``_merge_summaries`` — the list row carries ``effort`` and falls back to the
  journal for model/cwd, while a live projection's own answer still wins;
* ``_durable_projection`` — the detail of an ended session matches its row;
* ``_refresh_durable_rows`` — the listing resolves identities off the loop.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message
from local_operator.mobile import daemon as daemon_module
from local_operator.mobile.daemon import (
    MobileDaemon,
    SessionEntry,
    _durable_identity,
    _durable_projection,
    _DurableIdentity,
)
from local_operator.mobile.types import SessionProjection, SessionRecord
from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
from local_operator.session.model_selection import SELECTED_MODEL_CUSTOM_TYPE
from local_operator.session.transcript import Transcript
from local_operator.wakes import store as wake_store

PASSWORD = "pw123"


@pytest.fixture
def config(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: root)
    return root


def _session(
    config: Path,
    session_id: str,
    *,
    selection: dict[str, Any] | None = None,
    checkpoint_cwd: str | None = None,
) -> Path:
    directory = config / "sessions" / session_id
    directory.mkdir(parents=True)

    async def write() -> None:
        transcript = Transcript(directory)
        await transcript.append_message(Message.user("hello", id=f"u-{session_id}"))
        if selection is not None:
            await transcript.append_custom(SELECTED_MODEL_CUSTOM_TYPE, selection)
        if checkpoint_cwd is not None:
            await transcript.append_custom(
                FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"state": {"cwd": checkpoint_cwd}}
            )

    asyncio.run(write())
    return directory


def _v2(selector: str, effort: str | None) -> dict[str, Any]:
    return {"version": 2, "selector": selector, "effort": effort, "fast": False, "boot": selector}


def test_identity_reads_model_effort_and_checkpoint_cwd(config) -> None:
    directory = _session(
        config,
        "s-ended",
        selection=_v2("anthropic/claude-opus-5-5", "medium"),
        checkpoint_cwd="/work/project",
    )
    assert _durable_identity(directory) == _DurableIdentity(
        model_label="anthropic/claude-opus-5-5", cwd="/work/project", effort="medium"
    )


def test_identity_follows_a_later_effort_change(config) -> None:
    """The NEWEST selection row is the effective one (an ``/effort`` appends)."""
    directory = _session(config, "s-switch", selection=_v2("anthropic/claude-opus-5-5", "low"))
    assert _durable_identity(directory).effort == "low"
    asyncio.run(
        Transcript(directory).append_custom(
            SELECTED_MODEL_CUSTOM_TYPE, _v2("anthropic/claude-opus-5-5", "high")
        )
    )
    # The memo is keyed on the journal's stat, so the append is seen at once.
    assert _durable_identity(directory).effort == "high"


def test_an_old_row_without_effort_still_names_the_model(config) -> None:
    """A row an older build wrote (no ``effort`` key) loads with effort ``None``."""
    directory = _session(
        config, "s-legacy", selection={"version": 2, "selector": "openai/gpt-5", "boot": None}
    )
    identity = _durable_identity(directory)
    assert identity.model_label == "openai/gpt-5"
    assert identity.effort is None


def test_identity_is_empty_not_guessed_when_nothing_is_recorded(config) -> None:
    directory = _session(config, "s-bare")
    assert _durable_identity(directory) == _DurableIdentity()


def test_cwd_ladder_marker_then_wake_index(config) -> None:
    marked = _session(config, "s-marked", checkpoint_cwd="/from/checkpoint")
    (marked / "desktop.json").write_text(json.dumps({"cwd": "/from/marker"}))
    assert _durable_identity(marked).cwd == "/from/marker"

    waked = _session(config, "s-waked")
    # Through the store's own writer, so the fixture carries its real schema.
    wake_store.write_entry(config, "s-waked", cwd="/from/wakes", schedules=[{"id": "w1"}])
    assert _durable_identity(waked).cwd == "/from/wakes"


def _record(session_id: str, pid: int) -> SessionRecord:
    return SessionRecord(
        pid=pid,
        kind="tui",
        session_id=session_id,
        conversation_name=session_id,
        cwd="/live/cwd",
        model_label="live/model",
        control_port=4100 + pid,
        control_key=f"key-{pid}",
        started_at=100.0,
    )


def test_list_rows_fall_back_to_the_journal_and_carry_effort() -> None:
    daemon = MobileDaemon(port=0, password=PASSWORD)
    table = daemon.table
    table._durable_identities = {
        "ended": _DurableIdentity("anthropic/claude-opus-5-5", "/work/ended", "medium"),
        "live": _DurableIdentity("stale/model", "/stale", "low"),
    }
    gone = SessionEntry(_record("ended", 101))
    gone.ended = True
    table.entries[101] = gone
    live = SessionEntry(_record("live", 102))
    # A live projection whose model has no ladder: "" is a real answer, so the
    # (older) journal rung must not leak onto the row.
    live.projection = SessionProjection(
        session_id="live", pid=102, cwd="/live/cwd", model_label="local/llama", effort=""
    )
    table.entries[102] = live

    rows = {row["session_id"]: row for row in table._merge_summaries({"ended": None, "live": None})}
    assert rows["ended"]["model_label"] == "anthropic/claude-opus-5-5"
    assert rows["ended"]["cwd"] == "/work/ended"
    assert rows["ended"]["effort"] == "medium"
    assert rows["live"]["model_label"] == "local/llama"
    assert rows["live"]["cwd"] == "/live/cwd"
    assert rows["live"]["effort"] is None


def test_a_row_with_no_identity_says_null_effort() -> None:
    daemon = MobileDaemon(port=0, password=PASSWORD)
    rows = daemon.table._merge_summaries({"durable": None})
    assert rows[0]["effort"] is None
    assert rows[0]["model_label"] == ""


def test_ended_session_detail_matches_its_row(config) -> None:
    _session(
        config,
        "s-detail",
        selection=_v2("anthropic/claude-opus-5-5", "medium"),
        checkpoint_cwd="/work/detail",
    )
    projection = _durable_projection("s-detail")
    assert projection is not None
    assert projection.model_label == "anthropic/claude-opus-5-5"
    assert projection.model_selector == "anthropic/claude-opus-5-5"
    assert projection.cwd == "/work/detail"
    assert projection.effort == "medium"


def test_the_listing_refresh_resolves_identities(config) -> None:
    _session(
        config,
        "s-listed",
        selection=_v2("anthropic/claude-opus-5-5", "medium"),
        checkpoint_cwd="/work/listed",
    )
    daemon = MobileDaemon(port=0, password=PASSWORD)
    rows = asyncio.run(daemon.table._refresh_durable_rows())
    assert rows is not None and "s-listed" in rows
    merged = daemon.table._merge_summaries(rows)
    row = next(r for r in merged if r["session_id"] == "s-listed")
    assert (row["model_label"], row["cwd"], row["effort"]) == (
        "anthropic/claude-opus-5-5",
        "/work/listed",
        "medium",
    )


def test_an_unreadable_store_costs_the_identity_not_the_row(config, monkeypatch) -> None:
    directory = _session(config, "s-broken", selection=_v2("a/b", "low"))

    def boom(_directory):
        raise RuntimeError("damaged")

    monkeypatch.setattr(daemon_module, "_read_durable_identity", boom)
    # A fresh directory key: the memo has never seen this path.
    assert _durable_identity(directory) == _DurableIdentity()
