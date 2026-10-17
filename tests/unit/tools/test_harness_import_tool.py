"""Agent tool: preview-first contract, path refusal and real import smoke."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.harness_import.jobs import JOBS
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
    assert tool.name == "harness_import"
    assert "preview first" in tool.description
    assert "path" not in tool.parameters["properties"]
    assert tool.call_approval_tier({"op": "sources"}) == "read"
    assert tool.call_approval_tier({"op": "import"}) == "write"


@pytest.mark.asyncio
async def test_sources_and_sessions_write_nothing(import_home):
    home, config = import_home
    before = sorted(path.relative_to(home) for path in home.rglob("*"))
    sources = await _call({"op": "sources"})
    assert not sources.is_error
    assert any(row["source"] == SOURCE for row in sources.details["sources"])
    sessions = await _call({"op": "sessions", "source": SOURCE})
    assert not sessions.is_error
    assert sessions.details["sessions"][0]["source_id"] == SESSION_ID
    _assert_no_paths(sources.details)
    _assert_no_paths(sessions.details)
    after = sorted(path.relative_to(home) for path in home.rglob("*"))
    assert before == after
    assert not (config / "sessions").exists()


@pytest.mark.asyncio
async def test_preview_writes_nothing_and_mints_a_token(import_home):
    home, config = import_home
    preview = await _call({"op": "preview", "source": SOURCE})
    assert not preview.is_error
    assert preview.details["preview_token"].startswith("prev-")
    assert preview.details["total_sessions"] == 1
    assert preview.details["message_counts_estimated"] is True
    assert preview.details["sessions"][0]["source_id"] == SESSION_ID
    assert "Nothing has been imported" in _text(preview)
    _assert_no_paths(preview.details)
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
    token = preview.details["preview_token"]
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
    token = preview.details["preview_token"]
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
    job_id = started.details["job_id"]
    job = JOBS.get(job_id)
    assert job is not None and job.task is not None
    await job.task
    status = await _call({"op": "status", "job_id": job_id})
    assert not status.is_error
    result = status.details["result"]
    assert result["state"] == "done"
    target_id = result["sessions"][0]["target_session_id"]
    target = config / "sessions" / target_id
    assert target.is_dir()
    history = Transcript(target).build_llm_history()
    assert len(history) == 5
    assert _sha(source) == before_hash
    assert not list(source.parent.glob("*.jsonl-shm"))
    _assert_no_paths(status.details, allow={"instruction_review_path", "memory_review_path"})

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
            "preview_token": second_preview.details["preview_token"],
        }
    )
    second_job = JOBS.get(second.details["job_id"])
    assert second_job is not None and second_job.task is not None
    await second_job.task
    second_status = await _call({"op": "status", "job_id": second.details["job_id"]})
    assert second_status.details["result"]["sessions"][0]["outcome"] == "skipped_already_imported"
    assert _sha(source) == before_hash


@pytest.mark.asyncio
async def test_status_and_cancel_are_shared_with_routes(import_home):
    home, config = import_home
    preview = await _call({"op": "preview", "source": SOURCE})
    token = preview.details["preview_token"]
    started = await _call(
        {
            "op": "import",
            "source": SOURCE,
            "all": True,
            "confirm": True,
            "preview_token": token,
        }
    )
    job_id = started.details["job_id"]
    cancelled = await _call({"op": "cancel", "job_id": job_id})
    assert not cancelled.is_error
    job = JOBS.get(job_id)
    assert job is not None and job.task is not None
    await job.task
    status = await _call({"op": "status", "job_id": job_id})
    assert status.details["state"] in ("cancelled", "done")
    assert JOBS.get(job_id) is job


@pytest.mark.asyncio
async def test_import_reports_the_instruction_review_path(import_home):
    home, config = import_home
    shutil.copytree(FIXTURES / "claude", home / ".claude")
    (home / ".claude" / "CLAUDE.md").write_text(
        "# Synthetic instructions\n- Use fake text only.\n", encoding="utf-8"
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
            "preview_token": preview.details["preview_token"],
        }
    )
    job = JOBS.get(started.details["job_id"])
    assert job is not None and job.task is not None
    await job.task
    status = await _call({"op": "status", "job_id": started.details["job_id"]})
    review_path = status.details["result"]["instruction_review_path"]
    assert review_path.endswith("instructions/claude.md")
    assert Path(review_path).is_file()
    assert "Instructions saved for review at" in _text(status)


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
