"""`lop exec --resume` starts a session in its RECORDED directory, not its caller's.

A viewer-less move records its destination and stops, and the tool's own reply says
the next thing to start the session starts it there — naming `lop exec` among the
ways in. Before this, exec built the session at the PROCESS's cwd, so the successor
booted somewhere else and its boot reconcile settled a real move as `superseded`
(QA round 1, Q1). This drives the exec composition root and reads the ``cwd`` it
hands ``create_session``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator import exec_mode, session_factory
from local_operator.config import ConfigManager
from local_operator.exec_mode import ExecArgs
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.session import cwd_move

SID = "abc123def456"


@pytest.fixture()
def exec_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "config"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(CONFIG_DIR_ENV, str(root))
    ConfigManager(root).update_config({"hosting": "test", "model_name": "m"})
    directory = root / "sessions" / SID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "seed"}]}}\n',
        encoding="utf-8",
    )
    return root


def _capture_create_session(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    async def fake_create_session(args: Any, *rest: Any, **kwargs: Any) -> Any:
        seen.append({"args": args, **kwargs})
        return object()

    monkeypatch.setattr(session_factory, "create_session", fake_create_session)
    return seen


@pytest.mark.asyncio
async def test_a_resume_starting_at_the_source_starts_in_the_destination(
    exec_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = tmp_path / "src", tmp_path / "dest"
    source.mkdir()
    destination.mkdir()
    intent = cwd_move.enqueue_self_move(
        exec_root, SID, requested_cwd=str(destination), effective_cwd=str(source), owner_pid=1
    )
    assert cwd_move.accept_handoff(exec_root, SID, intent) is not None
    seen = _capture_create_session(monkeypatch)
    factory = exec_mode._make_default_session_factory(ExecArgs(resume=SID))
    monkeypatch.chdir(source)
    await factory()
    assert seen, "the factory built a session"
    assert seen[0]["cwd"] == str(
        destination.resolve()
    ), "exec honours the recorded move instead of the process cwd"


@pytest.mark.asyncio
async def test_a_resume_with_no_record_starts_at_the_callers_directory(
    exec_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    seen = _capture_create_session(monkeypatch)
    factory = exec_mode._make_default_session_factory(ExecArgs(resume=SID))
    monkeypatch.chdir(source)
    await factory()
    assert seen[0]["cwd"] == str(source.resolve()), "no record, no redirect"


@pytest.mark.asyncio
async def test_a_fresh_run_is_untouched(
    exec_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    seen = _capture_create_session(monkeypatch)
    factory = exec_mode._make_default_session_factory(ExecArgs())
    monkeypatch.chdir(source)
    await factory()
    assert seen[0]["cwd"] == str(source.resolve())


@pytest.mark.asyncio
async def test_a_destination_that_vanished_refuses_the_run(
    exec_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = tmp_path / "src", tmp_path / "dest"
    source.mkdir()
    destination.mkdir()
    intent = cwd_move.enqueue_self_move(
        exec_root, SID, requested_cwd=str(destination), effective_cwd=str(source), owner_pid=1
    )
    cwd_move.accept_handoff(exec_root, SID, intent)
    destination.rmdir()
    seen = _capture_create_session(monkeypatch)
    factory = exec_mode._make_default_session_factory(ExecArgs(resume=SID))
    monkeypatch.chdir(source)
    with pytest.raises(RuntimeError) as refused:
        await factory()
    assert "can no longer be entered" in str(refused.value), str(refused.value)
    assert seen == [], "and nothing was constructed in the wrong tree"
    kept = cwd_move.read_move_intent(exec_root, SID)
    assert kept is not None and kept.stage == cwd_move.STAGE_HANDOFF
