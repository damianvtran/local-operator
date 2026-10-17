"""Agent tool: preview-first contract, path refusal and real import smoke."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.harness_import.jobs import JOBS
from local_operator.harness_import.service import HarnessImportService
from local_operator.harness_import.types import ImportResult
from local_operator.session.transcript import Transcript
from local_operator.tools.harness_import_tool import _PREVIEWS, execute_harness_import
from local_operator.tools.registry import TOOL_BUILDERS

FIXTURES = Path(__file__).resolve().parents[1] / "harness_import" / "fixtures"
SOURCE = "omp"
SESSION_ID = "omp-synthetic"


@pytest.fixture
def import_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config = tmp_path / "config"
    home.mkdir()
    config.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    for name in (
        "CLAUDE_CONFIG_DIR",
        "CODEX_HOME",
        "PI_CODING_AGENT_DIR",
        "PI_CONFIG_DIR",
        "OMP_PROFILE",
        "PI_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    agent = home / ".omp" / "agent"
    agent.mkdir(parents=True)
    shutil.copytree(FIXTURES / "omp" / "sessions", agent / "sessions")
    JOBS.jobs.clear()
    _PREVIEWS.clear()
    return home, config


def _context() -> ToolContext:
    return ToolContext(cwd=".", session_id="agent-import-test")


async def _call(args: dict[str, Any]):
    return await execute_harness_import("call-1", args, None, None, _context())


def _text(result) -> str:
    return result.text


def _details(result) -> dict[str, Any]:
    assert result.details is not None
    return result.details


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_no_paths(value: Any, *, allow: set[str] | None = None) -> None:
    allow = allow or set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in allow:
                continue
            _assert_no_paths(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_paths(item)
    elif isinstance(value, str):
        for marker in ("/Users/", "/private/", "/tmp/", "~/.omp", "~/.claude", "~/.codex"):
            assert marker not in value, value


@pytest.mark.asyncio
async def test_tool_registers_with_no_path_parameter():
    tool = TOOL_BUILDERS["harness_import"](_context())
    assert tool is not None
    assert tool.name == "harness_import"
    assert "preview first" in tool.description
    assert "path" not in tool.parameters["properties"]
    tier = tool.call_approval_tier
    assert tier is not None
    assert tier({"op": "sources"}) == "read"
    assert tier({"op": "import"}) == "write"


@pytest.mark.asyncio
async def test_sources_and_sessions_write_nothing(import_home):
    home, config = import_home
    before = sorted(path.relative_to(home) for path in home.rglob("*"))
    sources = await _call({"op": "sources"})
    assert not sources.is_error
    assert any(row["source"] == SOURCE for row in _details(sources)["sources"])
    sessions = await _call({"op": "sessions", "source": SOURCE})
    assert not sessions.is_error
    assert _details(sessions)["sessions"][0]["source_id"] == SESSION_ID
    _assert_no_paths(_details(sources))
    _assert_no_paths(_details(sessions))
    after = sorted(path.relative_to(home) for path in home.rglob("*"))
    assert before == after
    assert not (config / "sessions").exists()


@pytest.mark.asyncio
async def test_preview_writes_nothing_and_mints_a_token(import_home):
    home, config = import_home
    preview = await _call({"op": "preview", "source": SOURCE})
    assert not preview.is_error
    assert _details(preview)["preview_token"].startswith("prev-")
    assert _details(preview)["total_sessions"] == 1
    assert _details(preview)["message_counts_estimated"] is True
    assert _details(preview)["sessions"][0]["source_id"] == SESSION_ID
    assert "Nothing has been imported" in _text(preview)
    _assert_no_paths(_details(preview))
    assert not (config / "sessions").exists()
    assert not (config / "imports").exists()


@pytest.mark.asyncio
async def test_import_requires_confirm_selection_and_preview(import_home):
    home, config = import_home
    no_confirm = await _call({"op": "import", "source": SOURCE, "all": True})
    assert no_confirm.is_error and "confirm=true" in _text(no_confirm)

    confirm = await _call({"op": "import", "source": SOURCE, "confirm": True})
    assert confirm.is_error and "no default-all" in _text(confirm)

    preview = await _call({"op": "preview", "source": SOURCE})
    token = _details(preview)["preview_token"]
    no_token = await _call({"op": "import", "source": SOURCE, "all": True, "confirm": True})
    assert no_token.is_error and "preview_token" in _text(no_token)

    wrong = await _call(
        {
            "op": "import",
            "source": SOURCE,
            "session_ids": {SOURCE: ["different"]},
            "confirm": True,
            "preview_token": token,
        }
    )
    assert wrong.is_error and "differ from the previewed" in _text(wrong)
    assert not (config / "sessions").exists()


@pytest.mark.asyncio
async def test_path_argument_is_refused(import_home):
    result = await _call({"op": "sessions", "source": SOURCE, "path": "/tmp/anything"})
    assert result.is_error
    assert "invalid arguments" in _text(result)


@pytest.mark.asyncio
async def test_explicit_import_replay_and_source_immutability(import_home):
    home, config = import_home
    source = home / ".omp" / "agent" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    before_hash = _sha(source)
    preview = await _call(
        {
            "op": "preview",
            "source": SOURCE,
            "session_ids": {SOURCE: [SESSION_ID]},
        }
    )
    token = _details(preview)["preview_token"]
    started = await _call(
        {
            "op": "import",
            "source": SOURCE,
            "session_ids": {SOURCE: [SESSION_ID]},
            "confirm": True,
            "preview_token": token,
        }
    )
    assert not started.is_error
    job_id = _details(started)["job_id"]
    job = JOBS.get(job_id)
    assert job is not None and job.task is not None
    await job.task
    status = await _call({"op": "status", "job_id": job_id})
    assert not status.is_error
    result = _details(status)["result"]
    assert result["state"] == "done"
    target_id = result["sessions"][0]["target_session_id"]
    target = config / "sessions" / target_id
    assert target.is_dir()
    history = Transcript(target).build_llm_history()
    assert len(history) == 5
    assert _sha(source) == before_hash
    assert not list(source.parent.glob("*.jsonl-shm"))
    _assert_no_paths(_details(status), allow={"instruction_review_path", "memory_review_path"})

    second_preview = await _call(
        {
            "op": "preview",
            "source": SOURCE,
            "session_ids": {SOURCE: [SESSION_ID]},
        }
    )
    second = await _call(
        {
            "op": "import",
            "source": SOURCE,
            "session_ids": {SOURCE: [SESSION_ID]},
            "confirm": True,
            "preview_token": _details(second_preview)["preview_token"],
        }
    )
    second_job = JOBS.get(_details(second)["job_id"])
    assert second_job is not None and second_job.task is not None
    await second_job.task
    second_status = await _call({"op": "status", "job_id": _details(second)["job_id"]})
    assert _details(second_status)["result"]["sessions"][0]["outcome"] == "skipped_already_imported"
    assert _sha(source) == before_hash


@pytest.mark.asyncio
async def test_tool_cancel_reaches_the_running_job(import_home, monkeypatch):
    """A fresh service per call must not cancel a throwaway event.

    The fake import only returns after the service instance OWNING the job sees
    its cancellation event. The old wiring passed a new service to cancel, so
    this test would hang and time out instead of reporting cancelled.
    """
    home, config = import_home

    async def slow_import(self, request, progress=None):
        while not self._cancel.is_set():
            await asyncio.sleep(0.01)
        return ImportResult(job_id="slow", state="cancelled")

    monkeypatch.setattr(HarnessImportService, "import_sessions", slow_import)
    preview = await _call({"op": "preview", "source": SOURCE})
    started = await _call(
        {
            "op": "import",
            "source": SOURCE,
            "all": True,
            "confirm": True,
            "preview_token": _details(preview)["preview_token"],
        }
    )
    job_id = _details(started)["job_id"]
    job = JOBS.get(job_id)
    assert job is not None and job.task is not None
    cancelled = await _call({"op": "cancel", "job_id": job_id})
    assert not cancelled.is_error
    await asyncio.wait_for(job.task, timeout=2)
    status = await _call({"op": "status", "job_id": job_id})
    assert _details(status)["state"] == "cancelled"
    assert not (config / "sessions").exists()


@pytest.mark.asyncio
async def test_jobs_are_scoped_to_the_calling_agent_session(import_home, monkeypatch):
    home, config = import_home

    async def slow_import(self, request, progress=None):
        while not self._cancel.is_set():
            await asyncio.sleep(0.01)
        return ImportResult(job_id="slow", state="cancelled")

    monkeypatch.setattr(HarnessImportService, "import_sessions", slow_import)
    preview = await _call({"op": "preview", "source": SOURCE})
    started = await _call(
        {
            "op": "import",
            "source": SOURCE,
            "all": True,
            "confirm": True,
            "preview_token": _details(preview)["preview_token"],
        }
    )
    job_id = _details(started)["job_id"]
    other = ToolContext(cwd=".", session_id="another-agent")
    hidden = await execute_harness_import(
        "call-2", {"op": "status", "job_id": job_id}, None, None, other
    )
    assert hidden.is_error and "no import job" in _text(hidden)
    hidden_cancel = await execute_harness_import(
        "call-3", {"op": "cancel", "job_id": job_id}, None, None, other
    )
    assert hidden_cancel.is_error
    owner_cancel = await _call({"op": "cancel", "job_id": job_id})
    assert not owner_cancel.is_error
    job = JOBS.get(job_id)
    assert job is not None and job.task is not None
    await asyncio.wait_for(job.task, timeout=2)


@pytest.mark.asyncio
async def test_preview_redacts_titles_and_strips_private_glyphs(import_home):
    home, config = import_home
    body = home / ".omp" / "agent" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    lines = body.read_text(encoding="utf-8").splitlines()
    title = {"type": "title", "title": "\U000f0016 ghp_" + "A" * 36, "v": 1}
    lines[0] = json.dumps(title)
    body.write_text("\n".join(lines) + "\n", encoding="utf-8")
    preview = await _call({"op": "preview", "source": SOURCE})
    sessions = await _call({"op": "sessions", "source": SOURCE})
    for result in (preview, sessions):
        row = _details(result)["sessions"][0]
        assert "ghp_" not in row["title"]
        assert "\U000f0016" not in row["title"]
        assert "[redacted]" in row["title"] or row["title"].startswith("synthetic")


@pytest.mark.asyncio
async def test_all_and_session_ids_are_refused_together(import_home):
    ambiguous = await _call(
        {
            "op": "preview",
            "source": SOURCE,
            "session_ids": {SOURCE: [SESSION_ID]},
            "all": True,
        }
    )
    assert ambiguous.is_error and "never both" in _text(ambiguous)
    preview = await _call(
        {
            "op": "preview",
            "source": SOURCE,
            "session_ids": {SOURCE: [SESSION_ID]},
        }
    )
    result = await _call(
        {
            "op": "import",
            "source": SOURCE,
            "session_ids": {SOURCE: [SESSION_ID]},
            "all": True,
            "confirm": True,
            "preview_token": _details(preview)["preview_token"],
        }
    )
    assert result.is_error
    assert "never both" in _text(result)


@pytest.mark.asyncio
async def test_import_reports_the_instruction_review_path(import_home):
    home, config = import_home
    shutil.copytree(FIXTURES / "claude", home / ".claude")
    (home / ".claude" / "CLAUDE.md").write_text(
        "# Synthetic instructions\n- Use fake text only.\n", encoding="utf-8"
    )
    (home / ".claude" / "memory").mkdir()
    (home / ".claude" / "memory" / "notes.md").write_text(
        "- synthetic untethered work state\n", encoding="utf-8"
    )
    preview = await _call(
        {
            "op": "preview",
            "source": "claude",
            "session_ids": {"claude": ["sess-claude"]},
        }
    )
    assert not preview.is_error
    started = await _call(
        {
            "op": "import",
            "source": "claude",
            "session_ids": {"claude": ["sess-claude"]},
            "confirm": True,
            "preview_token": _details(preview)["preview_token"],
        }
    )
    job = JOBS.get(_details(started)["job_id"])
    assert job is not None and job.task is not None
    await job.task
    status = await _call({"op": "status", "job_id": _details(started)["job_id"]})
    review_path = _details(status)["result"]["instruction_review_path"]
    assert review_path.endswith("instructions/claude.md")
    assert Path(review_path).is_file()
    memory_path = _details(status)["result"]["memory_review_path"]
    assert memory_path.endswith("memory-review-claude.jsonl")
    assert Path(memory_path).is_file()
    assert "Instructions saved for review at" in _text(status)
    assert "Untethered memory saved for review at" in _text(status)


@pytest.mark.asyncio
async def test_discovery_and_preview_make_no_network_calls(import_home, monkeypatch):
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("harness_import must not open a network connection")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert not (await _call({"op": "sources"})).is_error
    assert not (await _call({"op": "sessions", "source": SOURCE})).is_error
    assert not (await _call({"op": "preview", "source": SOURCE})).is_error


@pytest.mark.asyncio
async def test_preview_and_sessions_strip_paths_from_titles(import_home):
    home, config = import_home
    body = home / ".omp" / "agent" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    lines = body.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(
        {"type": "title", "title": "Investigate /Users/damian/secret/file.txt", "v": 1}
    )
    body.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for result in (
        await _call({"op": "preview", "source": SOURCE}),
        await _call({"op": "sessions", "source": SOURCE}),
    ):
        title = _details(result)["sessions"][0]["title"]
        assert "/Users/" not in title
        assert "<path>" in title


@pytest.mark.asyncio
async def test_cancel_in_the_queued_window_is_preserved(import_home):
    """Cancel before the task first runs must survive into import_sessions.

    This calls JOBS.cancel synchronously after start, so the event is set while
    the job is still queued. The old service cleared it at import_sessions()
    entry and imported; the fix clears only at JOBS.start().
    """
    home, config = import_home
    preview = await _call({"op": "preview", "source": SOURCE})
    started = await _call(
        {
            "op": "import",
            "source": SOURCE,
            "all": True,
            "confirm": True,
            "preview_token": _details(preview)["preview_token"],
        }
    )
    job_id = _details(started)["job_id"]
    job = JOBS.get(job_id)
    assert job is not None and job.task is not None
    cancelled = JOBS.cancel(job_id, session_key="agent-import-test")
    assert cancelled["cancelled"] is True
    await job.task
    status = await _call({"op": "status", "job_id": job_id})
    assert _details(status)["state"] == "cancelled"
    assert not (config / "sessions").exists()
