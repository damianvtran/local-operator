"""The desktop open snapshot carries SLIM job rows; deltas still restore full rows.

WHY THIS FILE EXISTS. The desktop relay ends the stream on any record over 8 MiB
(``MAX_FRAME_BYTES``, ``local-operator-ui`` ``src/main/desktop-stream.ts``), and
the open snapshot's roster grew ~3.3 KB per job: ~6.6 MB at 2,000 jobs, so a
long-lived orchestrator was a few thousand jobs from a conversation the app
could never open. The bridge now keeps only the fields the renderer reads
(``DESKTOP_JOB_ROW_KEYS``). These cells pin the three things that make that
safe and bounded:

* the slim row renders exactly what the full row renders (the Python mirror of
  ``deriveChild``'s reads below);
* the runtime socket / TUI payload (``sync_wire_payload``) is untouched;
* a slim row's first ``job_deltas`` upsert carries the FULL bounded row, so the
  app's replace-by-id merge restores every field;

plus the byte guards: bytes per job, and 10,000 jobs inside 8 MiB.
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import store as asks_store
from local_operator.server.utils import desktop_sessions as module
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendSync,
    FrontendUpdate,
    _job_summary,
    bound_delta_rows_in_place,
    sync_wire_payload,
)

#: Bytes per job of the slim desktop ``frontend`` payload on the realistic
#: roster below, measured 2026-11-15 at 2,000 jobs: 1,595,637 bytes / 2,000 =
#: 798 (the full rows measured 6,287,383 bytes, 3,144 per job). The guard allows
#: +20% so an intended field addition fits, while re-admitting any one of the
#: dropped heavyweights (``descendant_usage`` alone is ~1 KB a row) fails it.
MEASURED_BYTES_PER_JOB = 798
BYTES_PER_JOB_CEILING = int(MEASURED_BYTES_PER_JOB * 1.2)
#: The relay's per-record cap (``MAX_FRAME_BYTES`` in the app).
RELAY_RECORD_CAP = 8 * 1024 * 1024

#: Fields the desktop app never reads and the slim row must not carry.
DROPPED = (
    "agent",
    "intent",
    "effort",
    "restored",
    "started_at",
    "cut_off_cause",
    "requested_model_label",
    "direct_cost_knowledge",
    "trajectory",
    "trajectory_length",
    "descendant_usage",
    "output_tail",
    "output_seq",
    "session_dir",
    "attempt_aliases",
    "todos",
    "roster_released",
)


@pytest.fixture(autouse=True)
def _fresh_snapshot_pool() -> Any:
    module.shutdown_snapshot_pool(wait=True)
    yield
    module.shutdown_snapshot_pool(wait=True)


def _usage(index: int, components: int = 3) -> dict[str, Any]:
    return {
        "provider": "anthropic",
        "model_id": "claude-opus-5-5",
        "account_ref": "acct-main",
        "billing_basis": "metered",
        "input_tokens": 41_000 + index,
        "output_tokens": 3_200,
        "cache_read_tokens": 120_000,
        "cache_write_tokens": 8_000,
        "reasoning_tokens": 900,
        "context_tokens": 52_000 + index,
        "cost_components": [
            {
                "provider": "anthropic",
                "model_id": f"claude-opus-{part}",
                "account_ref": f"acct-{part}",
                "input_tokens": 1000 + index,
                "output_tokens": 200 + part,
            }
            for part in range(components)
        ],
    }


def _job(index: int, **overrides: Any) -> dict[str, Any]:
    """A long orchestrator's child row: full usage, descendants, prompt, result."""
    running = index % 20 == 0
    launch = f"msg-{uuid.UUID(int=index + 10**9).hex}"
    row = {
        "id": f"job-{index:05d}",
        "type": "task",
        "status": "running" if running else "completed",
        "label": f"Implement slice {index}: audit the roster bounds",
        "agent": "coder",
        "intent": "Implementing the bounded slice",
        "agent_role": "coder",
        "effort": "high",
        "latest_details": {"progress": f"step {index}: running the targeted tests"},
        "result_text": f"Result paragraph {index}. " * 60,
        "model_label": "anthropic/claude-opus-5-5",
        "requested_model_label": "anthropic/claude-opus-5-5",
        "context_window": 1_000_000,
        "usage": _usage(index),
        "direct_cost": 0.4321,
        "start_time": 1_760_000_000.0 + index,
        "started_at": 1_760_000_000.5 + index,
        "settled_at": None if running else 1_760_000_100.0 + index,
        "trajectory_length": 120,
        "descendant_usage": [_usage(index, 1) for _ in range(2)],
        "prompt": f"Brief line {index} explaining the task in detail. " * 25,
        "output_tail": "tail " * 40,
        "output_seq": 42,
        "session_dir": f"/sessions/{uuid.UUID(int=index).hex[:12]}",
        "session_id": uuid.UUID(int=index).hex[:12],
        "launch_message_id": launch,
        "launch_prompts": {launch: "Launch prompt " * 30},
        "attempt_aliases": [f"job-{index:05d}-a{k}" for k in range(2)],
    }
    row.update(overrides)
    return row


def _state(count: int, *, cwd: str = "/w", jobs: list[dict[str, Any]] | None = None) -> Any:
    return FrontendSessionState.model_validate(
        {
            "session_id": "s1",
            "epoch": "e1",
            "cwd": cwd,
            "jobs": jobs if jobs is not None else [_job(i) for i in range(count)],
        }
    )


def _runtime_payload(state: Any) -> dict[str, Any]:
    """What the runtime socket and the TUI followers receive (``server.py``)."""
    return sync_wire_payload(
        FrontendSync(
            epoch=state.epoch,
            sequence=state.sequence,
            snapshot=state,
            live_cursor=state.history_cursor,
        )
    )


def _desktop_frontend(state: Any) -> dict[str, Any]:
    return module._wire_state_of(module._WireCapture(reference=state, attention={}))


def _empty(value: Any) -> bool:
    return value is None or value is False or value in ("", {}, [])


_OPEN = {"running", "pausing", "queued", "starting", "paused"}


def _rendered(row: dict[str, Any]) -> dict[str, Any]:
    """The inputs ``deriveChild`` (``run-detail-model.ts``) reads from a row.

    A mirror of its reads, NOT of its output: ``wireText``/``wireNumber``/
    ``toWireStringMap`` all read an absent key as its empty value, so both
    normalize to ``None`` here; ``usage`` contributes ``context_tokens`` and
    ``input_tokens`` only; ``latest_details`` is drawn on an open row (activity)
    and read on any ``music`` row (the music card).
    """
    keys = module.DESKTOP_JOB_ROW_KEYS
    seen = {key: (None if _empty(row.get(key)) else row.get(key)) for key in keys}
    usage = row.get("usage") or {}
    seen["usage"] = (usage.get("context_tokens"), usage.get("input_tokens"))
    status = str(row.get("status") or "running").lower()
    if not (row.get("queued") is True or status in _OPEN or row.get("type") == "music"):
        seen["latest_details"] = None
    return seen


def _bridge(tmp_path: Path, state: Any) -> Any:
    bridge = module.DesktopSessionBridge(tmp_path, "s1", str(tmp_path))
    remote = AttachedSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=module._no_takeover,
        surface="desktop",
    )
    remote._install_frontend(state)
    bridge.remote = remote
    (tmp_path / asks_store.SESSIONS_DIRNAME / "s1").mkdir(parents=True, exist_ok=True)
    return bridge


@pytest.mark.asyncio
async def test_the_open_snapshot_carries_slim_rows_that_render_the_same(tmp_path: Path) -> None:
    state = _state(200, cwd=str(tmp_path))
    full = {row["id"]: row for row in _runtime_payload(state)["snapshot"]["jobs"]}
    bridge = _bridge(tmp_path, state)

    frame = await bridge.snapshot()

    rows = frame["payload"]["frontend"]["snapshot"]["jobs"]
    assert [row["id"] for row in rows] == list(full), "the slim roster keeps every row, in order"
    for row in rows:
        assert set(row) <= module.DESKTOP_JOB_ROW_KEYS, set(row) - module.DESKTOP_JOB_ROW_KEYS
        assert set(row["usage"]) <= module.DESKTOP_JOB_USAGE_KEYS
        assert _rendered(row) == _rendered(full[row["id"]]), row["id"]
    settled = [row for row in rows if row["status"] == "completed"]
    assert settled and all("latest_details" not in row for row in settled)
    running = [row for row in rows if row["status"] == "running"]
    assert running and all(row["latest_details"] for row in running)


def test_the_runtime_and_tui_payload_still_carries_full_rows() -> None:
    """``sync_wire_payload`` is shared with the runtime socket and the TUI followers.

    They read ``descendant_usage``, ``session_dir``, ``trajectory_length``,
    ``launch_*`` and ``attempt_aliases``; slimming belongs to the desktop bridge,
    after this function, on its own copy.
    """
    state = _state(40)
    before = _runtime_payload(state)

    desktop = _desktop_frontend(state)

    after = _runtime_payload(state)
    assert after == before, "building the desktop frame changed the shared payload"
    row = after["snapshot"]["jobs"][1]
    for key in ("descendant_usage", "session_dir", "trajectory_length", "attempt_aliases"):
        assert row.get(key), key
    assert {"output_tokens", "cache_read_tokens"} <= set(row["usage"])
    assert "descendant_usage" not in desktop["snapshot"]["jobs"][1]


def test_a_slim_row_is_restored_by_its_first_delta(tmp_path: Path, monkeypatch) -> None:
    """P3: snapshot slim row, then a delta full row; the merged row IS the full row."""
    frames: list[dict[str, Any]] = []
    bridge = module.DesktopSessionBridge(tmp_path, "s1", cwd=str(tmp_path), retiring=lambda: False)
    monkeypatch.setattr(bridge, "publish", lambda kind, payload, **kw: frames.append(payload))
    state = _state(60)
    roster = _desktop_frontend(state)["snapshot"]["jobs"]
    active = state.jobs[20]
    assert active.status == "running" and "descendant_usage" not in roster[20]

    changed = active.model_copy(update={"latest_details": {"progress": "step 2"}})
    bridge._frontend(
        FrontendUpdate(
            epoch="e1",
            sequence=2,
            changes={"streaming": False},
            job_deltas={"upserted": [_job_summary(changed)], "roster_size": len(roster)},
        )
    )

    wire = frames[-1]["job_deltas"]["upserted"][0]
    expected = [copy.deepcopy(_job_summary(changed))]
    bound_delta_rows_in_place(expected, roster_size=len(roster))
    assert wire == expected[0], "the delta row must be the FULL bounded row, never slimmed"
    # The app's merge (``job-deltas.ts``) REPLACES the row by id.
    merged = [wire if row["id"] == wire["id"] else row for row in roster]
    assert merged[20] == wire
    assert merged[20]["descendant_usage"] and merged[20]["session_dir"]
    assert _rendered(merged[20]) == _rendered(wire)


def test_a_full_roster_replacement_frame_is_slim(tmp_path: Path, monkeypatch) -> None:
    """A ``changes["jobs"]`` frame is swapped for ``state()``'s roster: slim too."""
    bridge = _bridge(tmp_path, _state(30, cwd=str(tmp_path)))
    frames: list[dict[str, Any]] = []
    monkeypatch.setattr(bridge, "publish", lambda kind, payload, **kw: frames.append(payload))

    bridge._frontend(
        FrontendUpdate(
            epoch="e1",
            sequence=2,
            changes={"jobs": [_job_summary(job) for job in bridge.remote.frontend_state.jobs]},
        )
    )

    rows = frames[-1]["changes"]["jobs"]
    assert len(rows) == 30
    assert all(set(row) <= module.DESKTOP_JOB_ROW_KEYS for row in rows)


@pytest.mark.parametrize(
    ("overrides", "keeps_details"),
    [
        ({"status": "completed"}, False),
        ({"status": "failed", "error_text": "boom"}, False),
        ({"status": "cancelled"}, False),
        ({"status": "interrupted"}, False),
        ({"status": "running"}, True),
        ({"status": "completed", "queued": True}, True),
        # The music card reads ``latest_details.music`` on a SETTLED row.
        ({"status": "cancelled", "type": "music"}, True),
        # A word the app has not been taught is not assumed settled.
        ({"status": "some-future-word"}, True),
    ],
)
def test_latest_details_is_dropped_only_where_it_is_never_drawn(
    overrides: dict[str, Any], keeps_details: bool
) -> None:
    row = _job(1, latest_details={"progress": "x", "music": {"tracks": []}}, **overrides)

    slim = module.slim_desktop_job_row(row)

    assert ("latest_details" in slim) is keeps_details
    assert _rendered(slim) == _rendered(row)


def test_empty_values_are_omitted_but_zero_readings_are_kept() -> None:
    row = _job(1, error_text="", parent_job_id=None, queued=False, direct_cost=0.0)
    row["usage"]["context_tokens"] = 0

    slim = module.slim_desktop_job_row(row)

    assert {"error_text", "parent_job_id", "queued"}.isdisjoint(slim)
    assert slim["direct_cost"] == 0.0 and slim["usage"]["context_tokens"] == 0
    assert module.slim_desktop_job_row({"id": "", "type": "", "status": ""}) == {
        "id": "",
        "type": "",
        "status": "",
    }


def test_bytes_per_job_stay_within_the_measured_budget() -> None:
    jobs = 2_000
    size = len(json.dumps(_desktop_frontend(_state(jobs)), separators=(",", ":")))

    assert size / jobs <= BYTES_PER_JOB_CEILING, (size, size / jobs)
    assert size <= 2 * 1024 * 1024, size


async def _seed_cold_session(root: Path, session_id: str, jobs: int) -> None:
    """A journal shaped like a long session's tail plus a checkpoint holding ``jobs`` rows.

    100 user/assistant rows of ~2 KB each: an ordinary open-frame page (the real
    fixture's page measured 206,856 B). The checkpoint row is the newest row, as
    the runtime's turn-end checkpoint is; the open frame never serves it.
    """
    from local_operator.harness.types import Message, TextContent
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True)
    (directory / "created_at.json").write_text(str(time.time()))
    transcript = Transcript(directory)
    await transcript.append_custom(
        "selected_model",
        {"version": 2, "selector": "test/mock", "effort": None, "boot": "test/mock"},
    )
    turns = []
    for turn in range(50):
        turns.append(Message(role="user", content=[TextContent(text=f"q{turn} " + "x" * 2000)]))
        turns.append(
            Message(
                role="assistant",
                content=[TextContent(text=f"a{turn} " + "y" * 2000)],
                stop_reason="stop",
            )
        )
    await transcript.append_messages(turns)
    # The checkpoint must name THIS session, or the cold open refuses it.
    state = _state(jobs, cwd=str(root)).model_copy(update={"session_id": session_id})
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {"checkpoint_id": "c1", "state": state.model_dump(mode="json")},
    )


async def _open_record_bytes(tmp_path: Path, monkeypatch, jobs: int) -> tuple[int, int, int]:
    """The WHOLE open record a negotiating app receives: frontend + open-frame page.

    ``open_frame=1`` is what the current app sends on ``/events`` once the core
    advertises ``open_frame`` (``server/features.py``); that page never serves
    the checkpoint row (``open_frame.strip_entry``).
    """
    root = tmp_path / "home" / ".local-operator"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    await _seed_cold_session(root, "c01d00000002", jobs)
    pool = module.DesktopSessions(root)
    try:
        async with pool.session("c01d00000002", read=True, allow_draft=True) as bridge:
            # Through the stream, as the app opens: the snapshot it yields is the
            # one built after the cold facade restored the checkpoint.
            sub = bridge.subscribe()
            stream = bridge.events(sub, epoch=None, after_seq=0, open_frame=True)
            try:
                frame = None
                async for candidate in stream:
                    if candidate["type"] == "snapshot":
                        frame = candidate
                        break
            finally:
                await stream.aclose()
        assert frame is not None, "the stream ended without a snapshot"
    finally:
        await pool.close()
    payload = frame["payload"]
    assert len(payload["frontend"]["snapshot"]["jobs"]) == jobs, "the roster must be restored"
    assert payload["history"]["entries"], "the guard must measure a page with rows on it"
    record = len(("data: " + json.dumps(frame, separators=(",", ":")) + "\n\n").encode())
    page = len(json.dumps(payload["history"], separators=(",", ":")))
    return record, page, jobs


@pytest.mark.asyncio
async def test_the_open_record_at_two_thousand_jobs_is_under_two_mib(tmp_path, monkeypatch):
    record, page, _ = await _open_record_bytes(tmp_path, monkeypatch, 2_000)

    assert page < 400_000, f"the fixture's page should be an ordinary one; {page}"
    assert record <= 2 * 1024 * 1024, record


@pytest.mark.asyncio
async def test_the_open_record_at_ten_thousand_jobs_fits_the_relay_cap(tmp_path, monkeypatch):
    record, _, _ = await _open_record_bytes(tmp_path, monkeypatch, 10_000)

    assert record <= RELAY_RECORD_CAP, record
