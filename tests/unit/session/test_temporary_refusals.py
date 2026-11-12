"""What a temporary chat refuses outside the session plane (contract §5).

A project link (store, desktop and relay mappings, the agent's tool), a memory
write that would outlive the chat, automatic capture, and the ``lop exec``
flag combinations. Each cell pairs the refusal with the ordinary chat it must
NOT affect.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.memory_capture import auto_capture_skipped
from local_operator.projects import ProjectEdit, ProjectRegistry, TemporarySessionError
from local_operator.session import temporary

TEMP = "a1b2c3d4e5f6"
PLAIN = "0f1e2d3c4b5a"


def _store(root: Path) -> None:
    for session_id in (TEMP, PLAIN):
        (root / "sessions" / session_id).mkdir(parents=True, exist_ok=True)
    assert temporary.write_mark(root, TEMP, True, by="tui").ok


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def test_the_store_refuses_to_link_or_create_with_a_temporary_chat(tmp_path: Path) -> None:
    _store(tmp_path)
    registry = ProjectRegistry(tmp_path)
    with pytest.raises(TemporarySessionError) as raised:
        registry.create_project(ProjectEdit(name="alpha"), sessions=[TEMP])
    assert raised.value.code == "session_temporary"
    project = registry.create_project(ProjectEdit(name="alpha"), sessions=[PLAIN])
    with pytest.raises(TemporarySessionError):
        registry.link_session(project.id, TEMP)
    with pytest.raises(TemporarySessionError):
        registry.link_session(project.id, TEMP, role="coordination")
    assert temporary.write_mark(tmp_path, TEMP, False, by="tui").ok  # Keep and link
    linked, changed = registry.link_session(project.id, TEMP)
    assert changed and TEMP in linked.sessions


def test_desktop_and_relay_map_the_refusal_to_409_session_temporary() -> None:
    from local_operator.mobile import projects as relay_projects
    from local_operator.server.routes import desktop_projects

    desktop = desktop_projects._refusal(TemporarySessionError(TEMP), "alpha")
    assert desktop.status_code == 409
    assert desktop.detail["code"] == "session_temporary"
    assert TEMP in desktop.detail["text"]
    relay = relay_projects._refusal(TemporarySessionError(TEMP))
    assert (relay.status, relay.code) == (409, "session_temporary")


@pytest.mark.asyncio
async def test_the_agents_link_is_refused_with_the_ask_instruction(tmp_path: Path) -> None:
    from local_operator.tools.project_tool import execute_project

    _store(tmp_path)
    registry = ProjectRegistry(tmp_path)
    context = ToolContext(cwd=".", session_id=PLAIN, project_registry=registry)
    created = await execute_project("tc", {"op": "create", "name": "alpha"}, None, None, context)
    assert not created.is_error, created.text
    refused = await execute_project(
        "tc", {"op": "link", "name": "alpha", "session_id": TEMP}, None, None, context
    )
    assert refused.is_error
    assert "temporary chat" in refused.text and "`ask`" in refused.text
    assert "do not retry" in refused.text


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def _memory_context(root: Path, session_id: str) -> ToolContext:
    return ToolContext(
        session_id=session_id,
        memory_config_dir=str(root),
        session_dir=str(root / "sessions" / session_id),
    )


@pytest.mark.asyncio
async def test_a_global_fact_is_refused_in_a_temporary_chat_and_session_scope_is_not(
    tmp_path: Path,
) -> None:
    from local_operator.tools.memory_tools import TEMPORARY_MEMORY_REFUSAL, execute_fact_memory

    _store(tmp_path)
    refused = await execute_fact_memory(
        "a",
        {"op": "store", "scope": "global", "text": "prefers tea over coffee"},
        context=_memory_context(tmp_path, TEMP),
    )
    assert refused.is_error
    assert TEMPORARY_MEMORY_REFUSAL in refused.content[0].text
    session_scoped = await execute_fact_memory(
        "b",
        {"op": "store", "scope": "session", "text": "prefers tea over coffee"},
        context=_memory_context(tmp_path, TEMP),
    )
    assert not session_scoped.is_error, session_scoped.content[0].text
    ordinary = await execute_fact_memory(
        "c",
        {"op": "store", "scope": "global", "text": "prefers tea over coffee"},
        context=_memory_context(tmp_path, PLAIN),
    )
    assert not ordinary.is_error, ordinary.content[0].text


@pytest.mark.asyncio
async def test_a_path_based_global_write_is_refused_too(tmp_path: Path) -> None:
    import json

    from local_operator.tools.memory_tools import TEMPORARY_MEMORY_REFUSAL, execute_memory

    _store(tmp_path)
    refused = await execute_memory(
        "a",
        "write",
        {"path": "memory://global", "content": json.dumps({"title": "t", "body": "b"})},
        _memory_context(tmp_path, TEMP),
    )
    assert refused.is_error
    assert TEMPORARY_MEMORY_REFUSAL in refused.content[0].text


def test_auto_capture_skips_a_temporary_chat_only(tmp_path: Path) -> None:
    _store(tmp_path)
    assert auto_capture_skipped(tmp_path / "sessions" / TEMP) is True
    assert auto_capture_skipped(tmp_path / "sessions" / PLAIN) is False
    assert auto_capture_skipped(None) is False
    # A damaged mark is not a temporary chat: capture stays on.
    (tmp_path / "sessions" / PLAIN / temporary.MARK_NAME).write_text("{not json")
    assert auto_capture_skipped(tmp_path / "sessions" / PLAIN) is False


# ---------------------------------------------------------------------------
# lop exec --temporary
# ---------------------------------------------------------------------------


def _exec_args(**fields: Any) -> argparse.Namespace:
    base = {"temporary": True, "resume": None, "workstream": False}
    return argparse.Namespace(**{**base, **fields})


def test_exec_temporary_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.agent_shell import AGENT_SHELL_ENV
    from local_operator.cli import _exec_temporary_refusal

    monkeypatch.delenv(AGENT_SHELL_ENV, raising=False)
    assert _exec_temporary_refusal(_exec_args()) is None
    assert _exec_temporary_refusal(_exec_args(temporary=False, resume="x")) is None
    assert "--resume" in (_exec_temporary_refusal(_exec_args(resume="abc")) or "")
    assert "--workstream" in (_exec_temporary_refusal(_exec_args(workstream=True)) or "")
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")
    assert "agent's shell" in (_exec_temporary_refusal(_exec_args()) or "")


def test_exec_temporary_reaches_the_detached_worker() -> None:
    from local_operator.exec_startup import STARTUP_FIELDS

    assert "temporary" in STARTUP_FIELDS
