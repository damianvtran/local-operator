"""Presence follows the parked tool, not the parent's generic busy flag.

These use synthetic sessions and temporary registry files. The integration cell
executes a real wait with real AsyncJobManager events and the actual publisher,
then reads the record through the catalogue and info listing while it is parked.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from rich.cells import cell_len

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.network.projection import PeerRow
from local_operator.network.types import row_needs_claim
from local_operator.resume import SessionRow
from local_operator.server.utils import desktop_mesh
from local_operator.session.catalog import (
    CatalogEntry,
    decorate_rows,
    delegating_label,
    waiting_label,
)
from local_operator.session.peer_rows import peer_session_rows
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import PROTOCOL_VERSION, SessionRecord
from local_operator.tools.builtin import (
    _session_row_brief,
    _sessions_info_body,
    execute_wait,
)
from local_operator.tui.widgets.session_picker import WAITING_MARKER, row_state_mark

SID = "c" * 12


def record(**values):  # noqa: ANN003, ANN201
    return SessionRecord(
        pid=os.getpid(),
        kind="exec",
        session_id=SID,
        conversation_name="synthetic presence",
        cwd="/synthetic",
        model_label="test/model",
        control_port=0,
        control_key="synthetic",
        **values,
    )


def test_record_additive_defaults_round_trip_and_heartbeat(tmp_path: Path) -> None:
    original = record(waiting_on=2, jobs_running=3)
    assert original.protocol == PROTOCOL_VERSION
    assert SessionRecord.from_json(original.to_json()) == original
    old = original.to_json()
    old.pop("waiting_on")
    old.pop("jobs_running")
    assert SessionRecord.from_json(old).waiting_on == 0
    assert SessionRecord.from_json(old).jobs_running is None
    publisher = registry.RecordPublisher(original, root=tmp_path)
    try:
        publisher.heartbeat(conversation_name="changed")
        found = registry.scan(tmp_path)[0][0]
        assert (found.waiting_on, found.jobs_running) == (2, 3)
        assert found.conversation_name == "changed"
    finally:
        publisher.close()


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"pending": "approval", "waiting_on": 2}, "approval"),
        ({"pending": "approval", "live_state": "wedged", "waiting_on": 2}, "wedged"),
        ({"live_state": "busy", "waiting_on": 2}, "waiting"),
        ({"live_state": "busy"}, "busy"),
    ],
)
def test_status_precedence(fields, expected) -> None:  # noqa: ANN001
    entry = CatalogEntry(SessionRow(SID, 0, "test", **fields))
    assert entry.status_code == expected
    if expected == "wedged":
        assert "Approval" not in entry.status
    if expected == "waiting":
        assert entry.status == "Waiting on 2 jobs"


@pytest.mark.parametrize(
    ("jobs", "subagents", "expected"),
    [
        (1, 0, "Waiting on 1 job"),
        (2, 1, "Waiting on 2 jobs · 1 subagent running"),
        (1, 2, "Waiting on 1 job · 2 subagents running"),
    ],
)
def test_waiting_words(jobs: int, subagents: int, expected: str) -> None:
    assert waiting_label(jobs, subagents) == expected


@pytest.mark.parametrize(
    ("running", "queued", "jobs", "expected"),
    [
        (0, 0, 1, "1 job running"),
        (0, 0, 2, "2 jobs running"),
        (1, 2, 1, "1 job running · 1 subagent running · 2 queued"),
        (0, 1, 1, "1 job running · 1 subagent queued"),
    ],
)
def test_delegating_jobs(running: int, queued: int, jobs: int, expected: str) -> None:
    row = SessionRow(
        SID, 0, "test", jobs_running=jobs, subagents_running=running, subagents_queued=queued
    )
    counts = row.delegating
    assert counts is not None
    assert counts == (running, queued, jobs)
    assert delegating_label(*counts) == expected
    assert CatalogEntry(row).status_code == "delegating"
    assert CatalogEntry(row).status == expected


def test_waiting_marker_has_one_cell_and_precedes_busy() -> None:
    assert cell_len(WAITING_MARKER) == 1
    assert row_state_mark(SessionRow(SID, 0, "test", live_state="busy", waiting_on=2), 0) == (
        WAITING_MARKER,
        "accent",
    )
    assert (
        row_state_mark(SessionRow(SID, 0, "test", live_state="wedged", waiting_on=2), 0)[0]
        != WAITING_MARKER
    )


@pytest.mark.parametrize("state", ["stored", "wedged", "stale"])
def test_non_live_pending_is_not_a_claim(state: str) -> None:
    assert row_needs_claim(state=state, pending="approval") is None
    assert (
        PeerRow.from_json({"state": state, "pending": "approval"}, device_id="d_test").pending
        is None
    )


def test_wedged_catalogue_drops_phantom_gate(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    claimed = record(pending="approval", waiting_on=2)
    monkeypatch.setattr(registry, "scan", lambda *_: [(claimed, "wedged")])
    row = decorate_rows(tmp_path, [SessionRow(SID, 0, "test")])[0]
    assert row.pending is None
    assert CatalogEntry(row).status_code == "wedged"


def test_peer_live_row_wins_over_stored_and_carries_presence() -> None:
    data = dict(
        session_id=SID,
        state="live",
        busy=True,
        detached=True,
        waiting_on=2,
        jobs_running=3,
        subagents_running=1,
        subagents_queued=4,
    )
    live = PeerRow.from_json(data, device_id="d_test")
    stored = PeerRow.from_json({**data, "state": "stored", "waiting_on": 0}, device_id="d_test")
    assert live.to_row_json()["jobs_running"] == 3
    assert live.to_record().waiting_on == 2
    assert PeerRow.from_json({"state": "live"}, device_id="d_test").jobs_running is None
    facts = SimpleNamespace(
        device_id="d_test", name="peer", reachable=True, reason="", network_id="n_test"
    )
    catalog = SimpleNamespace(peers=lambda: [facts], rows=lambda: [stored, live])
    rows = peer_session_rows(catalog=catalog)
    assert len(rows) == 1
    row = rows[0]
    assert (
        row.live_state,
        row.waiting_on,
        row.jobs_running,
        row.subagents_running,
        row.subagents_queued,
    ) == ("busy", 2, 3, 1, 4)
    assert CatalogEntry(row).status_code == "waiting"


def test_remote_desktop_presence_and_active(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    from local_operator.session import peer_rows

    row = SessionRow(
        SID,
        0,
        "test",
        live_state="busy",
        waiting_on=2,
        jobs_running=3,
        subagents_running=1,
        subagents_queued=4,
    )
    monkeypatch.setattr(desktop_mesh, "has_any_network", lambda _: True)
    monkeypatch.setattr(peer_rows, "peer_session_rows", lambda _: [row])
    emitted = desktop_mesh.remote_session_rows(tmp_path, pins=set())[0]
    assert emitted["active"] is True
    assert emitted["live_state"] == "busy"
    assert emitted["status"] == {
        "code": "waiting",
        "label": "Waiting on 2 jobs · 1 subagent running",
    }
    assert [
        emitted[key]
        for key in ("waiting_on", "jobs_running", "subagents_running", "subagents_queued")
    ] == [2, 3, 1, 4]
    row = row._replace(live_state="")
    assert desktop_mesh.remote_session_rows(tmp_path, pins=set())[0]["active"] is True


@pytest.mark.parametrize("reachable", [False, True])
def test_peer_count_no_claim_versus_empty(
    tmp_path: Path, monkeypatch, reachable: bool
) -> None:  # noqa: ANN001
    from local_operator.server.models.desktop_mesh import PeerRow as PeerResponse
    from local_operator.session import peer_rows

    monkeypatch.setattr(desktop_mesh, "has_any_network", lambda _: True)
    monkeypatch.setattr(desktop_mesh, "_networks", lambda _: [])
    monkeypatch.setattr(peer_rows, "peer_session_rows", lambda _: [])
    monkeypatch.setattr(
        desktop_mesh,
        "_listing_call",
        lambda *_: [
            {"device_id": "d_test", "name": "peer", "reachable": reachable},
        ],
    )
    emitted = desktop_mesh.peer_catalogue(tmp_path)["peers"][0]
    assert emitted["session_count"] == (0 if reachable else None)
    assert PeerResponse.model_validate(emitted).session_count == (0 if reachable else None)


def test_info_text_and_unreachable_remote_row() -> None:
    local = {"session_id": SID, "state": "live", "waiting_on": 1}
    assert "waiting on 1 job" in _sessions_info_body(local, {})
    remote = {
        **local,
        "locality": "remote",
        "busy": True,
        "peer": {"device_id": "d_test", "reachable": False},
    }
    text = _session_row_brief(remote)
    assert "link down; status unknown" in text
    assert "[live]" not in text and ", busy" not in text
    assert "waiting on" not in _sessions_info_body(remote, {})


def test_serving_probe_excludes_task_rows_and_preserves_unknown() -> None:
    handle = object.__new__(ServingSessionHandle)
    handle._session = SimpleNamespace(
        _waiting_on=2,
        jobs=SimpleNamespace(
            list=lambda: [
                SimpleNamespace(status="running", type="task"),
                SimpleNamespace(status="running", type="bash"),
                SimpleNamespace(status="completed", type="eval"),
            ]
        ),
    )
    assert handle.waiting_on() == 2
    assert handle.job_counts() == 1
    handle._session.jobs = None
    assert handle.job_counts() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_kind", ["settled", "timeout", "cancel", "abort", "steer", "disappeared"]
)
async def test_wait_hook_clears_every_park_exit(
    exit_kind: str, monkeypatch
) -> None:  # noqa: ANN001
    manager = AsyncJobManager()
    release = asyncio.Event()
    entered = asyncio.Event()
    notes = []
    updates = []

    async def runner(job_id, signal, progress):  # noqa: ANN001, ANN202
        await release.wait()
        return "done"

    job_id = manager.register("bash", "synthetic wait", runner)
    context = ToolContext(jobs=manager, note_waiting=notes.append)
    signal = AbortSignal() if exit_kind in {"abort", "steer"} else None

    def update(event):  # noqa: ANN001, ANN202
        updates.append(event.details["waiting_on"])
        entered.set()

    task = asyncio.ensure_future(
        execute_wait(
            "synthetic",
            {"job_id": [job_id, job_id], "wait_ms": 25 if exit_kind == "timeout" else 5000},
            signal,
            update,
            context,
        )
    )
    try:
        await entered.wait()
        assert notes == [1]
        if exit_kind == "settled":
            release.set()
        elif exit_kind == "abort":
            assert signal is not None
            signal.abort("synthetic abort")
            task.cancel()
        elif exit_kind in {"cancel", "steer"}:
            task.cancel()
        elif exit_kind == "disappeared":
            monkeypatch.setattr(manager, "get", lambda _: None)
            release.set()
        if exit_kind in {"cancel", "abort"}:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            if exit_kind == "disappeared":
                assert "disappeared" in result.text
            elif exit_kind == "steer":
                assert result.details is not None
                assert result.details["interrupted_by"] == "cancelled"
        assert notes == [1, 0]
        assert updates == [1, 0]
    finally:
        release.set()
        await manager.dispose()


@pytest.mark.asyncio
async def test_real_park_publishes_waiting_record_catalogue_and_info(tmp_path: Path) -> None:
    """Drive real job/wait, serving probes, setters and registry, not a static row."""
    from local_operator.info.collect import session_rows
    from tests.e2e.harness import ScriptedStream, build_session, text_turn

    directory = tmp_path / "sessions" / SID
    directory.mkdir(parents=True)
    session = build_session(directory, ScriptedStream([text_turn("synthetic reply")]))
    manager = session.jobs
    release = asyncio.Event()
    entered = asyncio.Event()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="exec")
    publisher = registry.RecordPublisher(record(busy=True, detached=True), root=tmp_path)
    server._publisher = publisher

    async def runner(job_id, signal, progress):  # noqa: ANN001, ANN202
        await release.wait()
        return "real background work settled"

    job_id = manager.register("bash", "synthetic real park", runner)
    context = session._build_tool_context()

    def update(event):  # noqa: ANN001, ANN202
        handle._publish_busy()
        entered.set()

    task = asyncio.ensure_future(
        execute_wait("synthetic", {"job_id": job_id, "wait_ms": 5000}, None, update, context)
    )
    try:
        await entered.wait()
        saved = registry.scan(tmp_path)[0][0]
        assert (saved.waiting_on, saved.jobs_running) == (1, 1)
        decorated = decorate_rows(tmp_path, [SessionRow(SID, 0, "synthetic")])[0]
        assert CatalogEntry(decorated).status_code == "waiting"
        assert CatalogEntry(decorated).status == "Waiting on 1 job"
        assert session_rows(tmp_path)[0]["waiting_on"] == 1
        release.set()
        result = await task
        assert "settled" in result.text
        assert registry.scan(tmp_path)[0][0].waiting_on == 0
        assert session._waiting_on == 0
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await session.dispose()
        publisher.close()


def test_presence_setters_dedupe() -> None:
    server = object.__new__(RuntimeServer)
    server._waiting_on = 0
    server._jobs_running = None
    server._republish = Mock()
    server.set_waiting(0)
    server.set_jobs(None)
    server._republish.assert_not_called()
    server.set_waiting(2)
    server.set_jobs(1)
    assert server._republish.call_count == 2
    server.set_waiting(2)
    server.set_jobs(1)
    assert server._republish.call_count == 2


@pytest.mark.asyncio
async def test_heartbeat_floor_uses_same_presence_probes(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    """Execute the actual heartbeat loop once, not a transcription of its body."""
    from local_operator.session.runtime import server as server_module
    from tests.e2e.harness import ScriptedStream, build_session, text_turn

    directory = tmp_path / "sessions" / SID
    directory.mkdir(parents=True)
    session = build_session(directory, ScriptedStream([text_turn("synthetic reply")]))
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="exec")
    publisher = registry.RecordPublisher(record(detached=True), root=tmp_path)
    server._publisher = publisher
    session._note_waiting(2)
    server._waiting_on = 99
    server._jobs_running = 99
    monkeypatch.setattr(server_module, "HEARTBEAT_INTERVAL_S", 0.01)
    original = publisher.heartbeat
    writes = []

    def publish(**values):  # noqa: ANN003, ANN202
        writes.append(values)
        original(**values)
        if "beat_lag_s" in values:
            server._closed.set()

    monkeypatch.setattr(publisher, "heartbeat", publish)
    try:
        await asyncio.wait_for(server._heartbeat_loop(), timeout=2)
        saved = registry.scan(tmp_path)[0][0]
        assert (saved.waiting_on, saved.jobs_running) == (2, 0)
        assert writes[-1]["waiting_on"] == 2
        assert writes[-1]["jobs_running"] == 0
    finally:
        publisher.close()
        await session.dispose()


@pytest.mark.parametrize("state", ["live", "wedged", "stale"])
def test_feed_row_matches_catalogue_pending_and_presence(
    state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.server.utils.desktop_feed import DesktopFeed

    feed = object.__new__(DesktopFeed)
    monkeypatch.setattr(feed, "_user_session", lambda _: True)
    feed._records = {SID: (record(pending="approval", waiting_on=2, jobs_running=3), state)}
    feed._wake_index = {}
    feed._monitor_index = {}
    row = feed._row_for(SID)
    assert row is not None
    assert row.live_state == {"live": "attached", "wedged": "wedged", "stale": ""}[state]
    assert row.pending == ("approval" if state == "live" else None)
    assert row.waiting_on == (0 if state == "stale" else 2)
    assert row.jobs_running == (None if state == "stale" else 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("observer", ["context", "update", "both"])
@pytest.mark.parametrize("fail_when", ["entry", "clear"])
@pytest.mark.parametrize("outcome", ["normal", "timeout", "abort", "cancel", "settled"])
async def test_wait_observer_failures_preserve_base_outcome(
    observer: str, fail_when: str, outcome: str
) -> None:
    """Compare healthy and failing observers, including an already settled job.

    Each observer fails independently; both must still receive entry and clear.
    The abort cases cancel a real parked tool after setting its abort signal,
    so a failure in finally cannot quietly replace CancelledError with a result.
    """

    async def probe(failing: bool) -> tuple[Any, list[int], list[int]]:
        manager = AsyncJobManager()
        entered = asyncio.Event()
        release = asyncio.Event()
        notes: list[int] = []
        updates: list[int] = []

        async def runner(job_id, signal, progress):  # noqa: ANN001, ANN202
            await release.wait()
            return "synthetic work settled"

        job_id = manager.register("bash", "observer failure control", runner)
        if outcome == "settled":
            release.set()
            await manager.settled_event(job_id).wait()
        signal = AbortSignal() if outcome == "abort" else None

        def note(count: int) -> None:
            notes.append(count)
            if (
                failing
                and observer in {"context", "both"}
                and ((count > 0) == (fail_when == "entry"))
            ):
                raise RuntimeError("synthetic context observer failure")

        def update(event: Any) -> None:
            count = event.details["waiting_on"]
            updates.append(count)
            entered.set()
            if (
                failing
                and observer in {"update", "both"}
                and ((count > 0) == (fail_when == "entry"))
            ):
                raise RuntimeError("synthetic update observer failure")

        context = ToolContext(jobs=manager, note_waiting=note)
        task = asyncio.ensure_future(
            execute_wait(
                "synthetic",
                {"job_id": job_id, "wait_ms": 10 if outcome == "timeout" else 5000},
                signal,
                update,
                context,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            if outcome in {"normal", "settled"}:
                release.set()
            elif outcome in {"abort", "cancel"}:
                if signal is not None:
                    signal.abort("synthetic abort")
                task.cancel()
            if outcome in {"abort", "cancel"}:
                with pytest.raises(asyncio.CancelledError):
                    await task
                result: Any = "CancelledError"
            else:
                completed = await task
                details = dict(completed.details or {})
                details.pop("job_id", None)
                result = (completed.text.replace(job_id, "JOB"), completed.is_error, details)
            return result, notes, updates
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await manager.dispose()

    base = await probe(False)
    broken = await probe(True)
    assert broken == base
    assert broken[1:] == ([1, 0], [1, 0])


def test_stale_registry_record_cannot_shadow_relay_stored_fallback(tmp_path: Path) -> None:
    """Exercise the FIRST real reaping scan, not a post-reap empty registry."""
    from local_operator.network import relay
    from local_operator.session.cleanup import mark_store
    from local_operator.session.peer_rows import _live_state

    server = relay.RelayServer(
        root=tmp_path, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    directory = tmp_path / "sessions" / SID
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    mark_store(directory.parent)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=5)
    registry.publish(
        replace(record(busy=True, waiting_on=2, jobs_running=3, pending="approval"), pid=child.pid),
        root=tmp_path,
    )
    emitted = next(row for row in server.local_session_rows() if row["session_id"] == SID)
    assert emitted["state"] == "stored"
    assert emitted.get("waiting_on", 0) == 0
    assert emitted.get("jobs_running") is None
    assert emitted.get("pending") is None
    assert emitted["busy"] is False
    assert registry.scan(tmp_path) == [], "the first scan actually reaped the dead record"
    peer = PeerRow.from_json(emitted, device_id="d_synthetic")
    cold = SessionRow(SID, 0, "synthetic", live_state=_live_state(peer))
    assert CatalogEntry(cold).status == "Recent"
    # A mixed-version peer may still emit the old stale shape with work claims.
    # Both the state and counts must be cold: otherwise waiting/delegating can
    # resurrect a dead pid, or a wedged label can falsely claim it is alive.
    peer = PeerRow.from_json(
        {
            "state": "stale",
            "busy": True,
            "waiting_on": 2,
            "jobs_running": 3,
            "subagents_running": 1,
            "subagents_queued": 1,
            "pending": "approval",
        },
        device_id="d_synthetic",
    )
    assert _live_state(peer) == ""
    cold = SessionRow(
        SID,
        0,
        "synthetic",
        live_state=_live_state(peer),
        pending=peer.pending,
        waiting_on=peer.waiting_on,
        jobs_running=peer.jobs_running,
        subagents_running=peer.subagents_running,
        subagents_queued=peer.subagents_queued,
    )
    assert CatalogEntry(cold).status == "Recent"
    assert cold.delegating is None


def test_local_first_reaping_scan_is_cold(tmp_path: Path) -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=5)
    registry.publish(
        replace(record(busy=True, waiting_on=2, jobs_running=3, pending="approval"), pid=child.pid),
        root=tmp_path,
    )
    row = decorate_rows(tmp_path, [SessionRow(SID, 0, "synthetic")])[0]
    assert row.live_state == ""
    assert row.pending is None
    assert row.waiting_on == 0
    assert row.jobs_running is None
    assert CatalogEntry(row).status == "Recent"
    assert CatalogEntry(row).active is False
    assert registry.scan(tmp_path) == []
