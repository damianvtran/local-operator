from __future__ import annotations

import asyncio
import copy

import pytest

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import ToolContext
from local_operator.music import jobs as music_jobs
from local_operator.tools import music_tool
from local_operator.tools.registry import create_tools


@pytest.mark.asyncio
async def test_real_registry_detaches_and_explicit_cancel_retains_one_observer(monkeypatch):
    entered = asyncio.Event()
    observed_again = asyncio.Event()
    actions = []
    records = []
    disposed = False
    manager = AsyncJobManager()

    class Client:
        def __init__(self, **kwargs):
            pass

        async def request(self, action, **kwargs):
            actions.append(action)
            if action == "models":
                return {
                    "models": [
                        {"id": "radient/music", "enabled": True, "operations": {"compose": True}}
                    ]
                }
            if action == "generate":
                return {
                    "tracks": [
                        {
                            "id": "paid-child",
                            "settled": False,
                            "status": "IN_PROGRESS",
                            "variant_index": 0,
                        }
                    ]
                }
            if action == "cancel":
                return {"request_id": "paid-child", "requested": True, "vendor_stopped": False}
            if action == "status":
                if actions.count("status") > 1:
                    observed_again.set()
                entered.set()
                await asyncio.Event().wait()
            raise AssertionError(action)

        async def close(self):
            pass

    async def journal(record):
        records.append(copy.deepcopy(record))

    context = ToolContext(
        jobs=manager,
        journal_music_job=journal,
        session_id="synthetic-session",
        session_disposed=lambda: disposed,
    )
    monkeypatch.setattr(music_tool, "radient_available", lambda root: True)
    monkeypatch.setattr(music_tool, "MusicClient", Client)
    original_runner = music_jobs.runner
    monkeypatch.setattr(
        music_tool,
        "runner",
        lambda ctx, record: original_runner(ctx, record, client_factory=Client),
    )
    # The cancel handler's successor must use the same offline transport too.
    monkeypatch.setattr(
        music_jobs,
        "runner",
        lambda ctx, record: original_runner(ctx, record, client_factory=Client),
    )
    tools = create_tools(context, enabled=["generate_music", "modify_music", "music_status"])
    assert {tool.name for tool in tools} == {"generate_music", "modify_music", "music_status"}
    result = await music_tool.execute_generate_music(
        "call-id",
        {"prompt": "Piano", "duration_ms": 3000},
        None,
        None,
        context,
    )
    assert not result.is_error
    assert result.details is not None
    job_id = result.details["job_id"]
    job = manager.get(job_id)
    assert job is not None
    assert job.type == "music"
    await asyncio.wait_for(entered.wait(), timeout=2)
    assert await manager.cancel(job_id)
    await asyncio.wait_for(observed_again.wait(), timeout=2)
    assert actions.count("generate") == 1
    assert actions.count("cancel") == 1
    assert records[-1]["cancellation"][0]["vendor_stopped"] is False
    assert len([job for job in manager.list() if job.status == "running"]) == 1
    disposed = True
    await manager.dispose()
    assert actions.count("cancel") == 1


def test_reduced_hosts_do_not_advertise_unusable_music_tools(monkeypatch):
    monkeypatch.setattr(music_tool, "radient_available", lambda root: True)
    assert music_tool.build_generate_music_tool(ToolContext()) is None
    assert music_tool.build_modify_music_tool(ToolContext()) is None
    assert music_tool.build_music_status_tool(ToolContext()) is None


@pytest.mark.asyncio
async def test_real_journal_recovery_uses_newest_snapshot(tmp_path, monkeypatch):
    from local_operator.session.transcript import Transcript, read_custom_entries

    transcript = Transcript(tmp_path / "synthetic-session")
    for state, watcher in [("registered", "old"), ("running", "current"), ("completed", "final")]:
        await transcript.append_custom(
            music_jobs.CUSTOM_TYPE,
            {
                "job": "paid-job",
                "state": state,
                "watcher": watcher,
                "cancellation_requested": state != "registered",
                "artifacts": [state],
            },
        )
    entries = read_custom_entries(transcript.directory, music_jobs.CUSTOM_TYPE)
    assert [entry.payload["details"]["state"] for entry in entries] == [
        "completed",
        "running",
        "registered",
    ]
    latest = music_jobs.read_records(transcript.directory)["paid-job"]
    assert latest["watcher"] == "final"
    assert latest["cancellation_requested"] is True
    assert latest["artifacts"] == ["completed"]
    monkeypatch.setattr(
        music_jobs, "register_observer", lambda *args: pytest.fail("terminal job re-armed")
    )
    await music_jobs.recover(ToolContext(session_dir=str(transcript.directory)))


@pytest.mark.asyncio
async def test_all_failed_receipt_stays_terminal_on_real_journal_recovery(tmp_path, monkeypatch):
    from local_operator.session.transcript import Transcript, read_custom_entries

    transcript = Transcript(tmp_path / "synthetic-session")

    async def journal(record):
        await transcript.append_custom(music_jobs.CUSTOM_TYPE, record)

    class Client:
        def __init__(self, **kwargs):
            pass

        async def request(self, *args, **kwargs):
            pytest.fail("settled failure must not resubmit or poll")

        async def close(self):
            pass

    context = ToolContext(session_dir=str(transcript.directory), journal_music_job=journal)
    record = {
        "job": "failed-job",
        "state": "running",
        "tracks": [
            {
                "id": "paid-child",
                "settled": True,
                "status": "FAILED",
                "error_type": "provider_failed",
            }
        ],
    }
    with pytest.raises(RuntimeError, match="music_no_playable_results"):
        await music_jobs.runner(context, record, client_factory=Client)(
            "watcher", None, lambda report: None
        )
    # Assert the actual newest append independently of the reader regression.
    newest = read_custom_entries(transcript.directory, music_jobs.CUSTOM_TYPE)[0]
    assert newest.payload["details"]["state"] == "failed"
    assert music_jobs.read_records(transcript.directory)["failed-job"]["state"] == "failed"
    monkeypatch.setattr(
        music_jobs, "register_observer", lambda *args: pytest.fail("settled failure re-armed")
    )
    await music_jobs.recover(context)
