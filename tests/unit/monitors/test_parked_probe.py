"""The parked-session monitor probe: quiet ticks stay parked, anything else engages.

``monitors/parked.py`` runs a parked session's due checks OUTSIDE the session
with the session's own tool factories, read-only gate and hash pipeline. It may
commit only a quiet tick; a change (or anything it cannot judge) must leave the
counters and snapshot untouched so the engaged session delivers the delta
against the unchanged baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.monitors import parked
from local_operator.monitors import state as monitor_state
from local_operator.monitors import store as monitor_store
from local_operator.monitors.diff import content_hash
from local_operator.monitors.scheduler import normalize_result
from local_operator.monitors.settings import read_monitor_settings
from local_operator.monitors.spec import MonitorSpec

SID = "probe0000001"
NOW = 1_800_000_000_000


def _row(tool: str, arguments: dict[str, Any], cwd: str) -> dict[str, Any]:
    return {
        "id": "m1",
        "name": "watch",
        "tool": tool,
        "arguments": arguments,
        "every_ms": 60_000,
        "until_at": None,
        "cwd": cwd,
        "created_at": 1,
        "next_due_at": NOW - 1,
        "disabled": False,
    }


def _arm(root: Path, row: dict[str, Any], *, parked_at: bool = True) -> None:
    monitor_store.write_entry(
        root,
        SID,
        cwd=row["cwd"],
        monitors=[row],
        preserve={"parked_at": NOW - 10} if parked_at else None,
    )


async def _baseline(root: Path, row: dict[str, Any]) -> str:
    """The hash the in-session scheduler would have stored: same tool, same pipeline."""
    from local_operator.harness.types import AbortSignal, ToolContext
    from local_operator.session.session import _monitor_comparable_text

    spec = MonitorSpec.model_validate({k: row[k] for k in MonitorSpec.model_fields if k in row})
    tool = parked._build_tool(spec.tool)
    result = await tool.execute(
        "t",
        dict(spec.arguments),
        AbortSignal(),
        None,
        ToolContext(cwd=spec.cwd, monitor_check=True),
    )
    text, _blind, _handle = _monitor_comparable_text(result.text, result.details)
    return content_hash(normalize_result(str(text), spec, read_monitor_settings()))


@pytest.mark.asyncio
async def test_quiet_tick_advances_counters_and_a_change_engages_without_writing(
    tmp_path: Path,
) -> None:
    watched = tmp_path / "watched.txt"
    watched.write_text("thread: 3 replies\n", encoding="utf-8")
    row = _row("read", {"path": str(watched)}, str(tmp_path))
    _arm(tmp_path, row)

    # No baseline yet: only the session may establish one.
    first = await parked.probe_session(tmp_path, SID, now_ms=NOW)
    assert first["verdict"] == "engage" and "no baseline" in first["why"]

    counters = {
        "schema": 1,
        "monitor_id": "m1",
        "next_due_at": NOW - 1,
        "content_hash": await _baseline(tmp_path, row),
        "checks": 4,
        "last_check_at": NOW - 60_000,
    }
    monitor_state.write_counters(tmp_path, SID, "m1", counters)

    quiet = await parked.probe_session(tmp_path, SID, now_ms=NOW)
    assert quiet == {"verdict": "quiet", "why": "unchanged", "checked": 1}
    after = monitor_state.read_counters(tmp_path, SID, "m1")
    assert after is not None
    assert after["checks"] == 5 and after["last_check_at"] == NOW
    assert NOW + 60_000 <= after["next_due_at"] <= NOW + 65_000
    assert after["content_hash"] == counters["content_hash"]

    # Not due any more: nothing runs.
    assert (await parked.probe_session(tmp_path, SID, now_ms=NOW + 1))["checked"] == 0

    # A change: engage, and the counters are exactly as the quiet tick left them
    # (the session will diff against the unchanged baseline and deliver).
    watched.write_text("thread: 4 replies\n", encoding="utf-8")
    later = after["next_due_at"] + 1
    changed = await parked.probe_session(tmp_path, SID, now_ms=later)
    assert changed["verdict"] == "engage" and changed["why"] == "m1: changed"
    assert monitor_state.read_counters(tmp_path, SID, "m1") == after
    assert not monitor_state.snapshot_exists(tmp_path, SID, "m1")


@pytest.mark.asyncio
async def test_a_tool_that_needs_a_session_engages(tmp_path: Path) -> None:
    # web_search stays session-only by decision (``parked.SESSION_ONLY_TOOLS``);
    # MCP rows are probed now (``test_parked_probe_mcp.py``).
    row = _row("web_search", {"query": "x"}, str(tmp_path))
    _arm(tmp_path, row)
    monitor_state.write_counters(
        tmp_path,
        SID,
        "m1",
        {"schema": 1, "monitor_id": "m1", "next_due_at": NOW - 1, "content_hash": "abc"},
    )
    outcome = await parked.probe_session(tmp_path, SID, now_ms=NOW)
    assert outcome["verdict"] == "engage"
    assert "cannot run outside a session" in outcome["why"]


@pytest.mark.asyncio
async def test_a_call_the_readonly_gate_refuses_is_never_run(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    row = _row("bash", {"command": f"touch {marker}"}, str(tmp_path))
    _arm(tmp_path, row)
    monitor_state.write_counters(
        tmp_path,
        SID,
        "m1",
        {"schema": 1, "monitor_id": "m1", "next_due_at": NOW - 1, "content_hash": "abc"},
    )
    outcome = await parked.probe_session(tmp_path, SID, now_ms=NOW)
    assert outcome["verdict"] == "engage" and "no longer read-only" in outcome["why"]
    assert not marker.exists()


@pytest.mark.asyncio
async def test_an_entry_that_is_not_parked_is_left_to_the_session(tmp_path: Path) -> None:
    row = _row("read", {"path": str(tmp_path)}, str(tmp_path))
    _arm(tmp_path, row, parked_at=False)
    assert (await parked.probe_session(tmp_path, SID, now_ms=NOW))["verdict"] == "engage"


@pytest.mark.asyncio
async def test_a_monitor_mid_failure_episode_engages(tmp_path: Path) -> None:
    watched = tmp_path / "w.txt"
    watched.write_text("x", encoding="utf-8")
    row = _row("read", {"path": str(watched)}, str(tmp_path))
    _arm(tmp_path, row)
    monitor_state.write_counters(
        tmp_path,
        SID,
        "m1",
        {
            "schema": 1,
            "monitor_id": "m1",
            "next_due_at": NOW - 1,
            "content_hash": "abc",
            "consecutive_failures": 2,
        },
    )
    outcome = await parked.probe_session(tmp_path, SID, now_ms=NOW)
    assert outcome["verdict"] == "engage" and "episode" in outcome["why"]


@pytest.mark.asyncio
async def test_a_session_committing_during_the_check_wins_the_quiet_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-2: the probe never writes its stale counters over a reopened session's commit."""
    watched = tmp_path / "watched.txt"
    watched.write_text("thread: 3 replies\n", encoding="utf-8")
    row = _row("read", {"path": str(watched)}, str(tmp_path))
    _arm(tmp_path, row)
    baseline = await _baseline(tmp_path, row)
    monitor_state.write_counters(
        tmp_path,
        SID,
        "m1",
        {
            "schema": 1,
            "monitor_id": "m1",
            "next_due_at": NOW - 1,
            "content_hash": baseline,
            "checks": 4,
            "last_check_at": NOW - 60_000,
        },
    )
    # What the reopened session's scheduler commits mid-check: a delivered
    # change (new hash), and the open-time rewrite cleared ``parked_at``.
    session_commit = {
        "schema": 1,
        "monitor_id": "m1",
        "next_due_at": NOW + 60_000,
        "content_hash": "the-session-delivered-a-change",
        "checks": 5,
        "last_check_at": NOW - 5,
    }
    real_build = parked._build_tool

    def _build(name: str) -> Any:
        tool = real_build(name)
        real_execute = tool.execute

        async def _execute(*args: Any, **kwargs: Any) -> Any:
            result = await real_execute(*args, **kwargs)
            monitor_state.write_counters(tmp_path, SID, "m1", dict(session_commit))
            return result

        monkeypatch.setattr(tool, "execute", _execute)
        return tool

    monkeypatch.setattr(parked, "_build_tool", _build)
    outcome = await parked.probe_session(tmp_path, SID, now_ms=NOW)
    assert outcome["verdict"] == "quiet" and outcome["why"] == "unchanged"
    # The session's commit survives byte-for-byte: nothing was written over it.
    assert monitor_state.read_counters(tmp_path, SID, "m1") == session_commit

    # Same race, but the entry stopped being parked (the session reopened)
    # while its counters happened to be untouched: also no write.
    monitor_state.write_counters(
        tmp_path,
        SID,
        "m1",
        {**session_commit, "content_hash": baseline, "next_due_at": NOW - 1},
    )
    before = monitor_state.read_counters(tmp_path, SID, "m1")

    def _build_unpark(name: str) -> Any:
        tool = real_build(name)
        real_execute = tool.execute

        async def _execute(*args: Any, **kwargs: Any) -> Any:
            result = await real_execute(*args, **kwargs)
            monitor_store.write_entry(tmp_path, SID, cwd=row["cwd"], monitors=[row])
            return result

        monkeypatch.setattr(tool, "execute", _execute)
        return tool

    monkeypatch.setattr(parked, "_build_tool", _build_unpark)
    await parked.probe_session(tmp_path, SID, now_ms=NOW)
    assert monitor_state.read_counters(tmp_path, SID, "m1") == before
