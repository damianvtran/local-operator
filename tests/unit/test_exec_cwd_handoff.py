"""`lop exec --resume` starts a session in its RECORDED directory, not its caller's.

A viewer-less move records its destination and stops, and the tool's own reply says
the next thing to start the session starts it there — naming `lop exec` among the
ways in. Before this, exec built the session at the PROCESS's cwd, so the successor
booted somewhere else and its boot reconcile settled a real move as `superseded`
(QA round 1, Q1). This drives the exec composition root and reads the ``cwd`` it
hands ``create_session``.
"""

from __future__ import annotations

import os
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


@pytest.mark.asyncio
async def test_the_exec_runtime_publishes_the_destination_and_that_boot_commits(
    exec_root: Path, tmp_path: Path
) -> None:
    """BOTH REAL STEPS IN ONE PROCESS: the record exec publishes, then the reconcile.

    The round-1 cell stopped one step short: it proved the SESSION was built at the
    destination, not that the RECORD says so. ``run_session`` passed ``os.getcwd()``
    to ``start_exec_control``, which becomes the published record's cwd, and the
    only writer of ``committed`` reconciles against the RECORD — so a correct start
    published the caller's directory, the move was settled ``superseded``, and the
    next start went back to the source (agent review round 2, MAJOR; QA round 2).

    No patched ``create_session`` here: a real ``start_exec_control`` publishes a
    real record, its runtime's own boot reconcile settles the marker, and a SECOND
    start is checked to stay in the destination.
    """
    from local_operator.exec_session import session_cwd
    from local_operator.session import cwd_move
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.exec_control import start_exec_control
    from tests.unit.session.runtime.test_exec_control import FakeSession

    source, destination = tmp_path / "src", tmp_path / "dest"
    source.mkdir()
    destination.mkdir()
    intent = cwd_move.enqueue_self_move(
        exec_root, SID, requested_cwd=str(destination), effective_cwd=str(source), owner_pid=1
    )
    cwd_move.accept_handoff(exec_root, SID, intent)

    session = FakeSession()
    session.session_id = SID
    # What `create_session(cwd=...)` sets on a real Session; the double predates
    # the field, so it is attached rather than declared.
    setattr(session, "_cwd", str(destination))
    control = await start_exec_control(session, cwd=session_cwd(session), supervised=False)
    try:
        records = [r for r, _ in registry.scan(exec_root) if r.pid == os.getpid()]
        assert records, "the exec runtime published a record"
        assert records[0].cwd == str(destination.resolve()), "the record names the destination"
        _wait_for_stage(cwd_move, exec_root, cwd_move.STAGE_COMMITTED)
        # AND THE STATUS NO LONGER CLAIMS A CANCELLATION: it is the committed
        # sentence, not "the move was cancelled; this session stays in <src>".
        rendered = cwd_move_stage_sentence(cwd_move, exec_root)
        assert "cancelled" not in rendered, rendered
    finally:
        await control.aclose()

    # A SECOND START STAYS IN THE DESTINATION.
    again = FakeSession()
    again.session_id = SID
    setattr(again, "_cwd", cwd_move.spawn_cwd_for_session(exec_root, SID, str(source)))
    second = await start_exec_control(again, cwd=session_cwd(again), supervised=False)
    try:
        records = [r for r, _ in registry.scan(exec_root) if r.pid == os.getpid()]
        assert records and records[0].cwd == str(destination.resolve())
        final = cwd_move.read_move_intent(exec_root, SID)
        assert final is not None and final.stage == cwd_move.STAGE_COMMITTED
    finally:
        await second.aclose()


def _wait_for_stage(cwd_move_module: Any, root: Path, stage: str, *, ticks: int = 100) -> None:
    import time

    for _ in range(ticks):
        stored = cwd_move_module.read_move_intent(root, SID)
        if stored is not None and stored.stage == stage:
            return
        time.sleep(0.05)
    stored = cwd_move_module.read_move_intent(root, SID)
    raise AssertionError(f"stage never reached {stage}: {stored}")


def cwd_move_stage_sentence(cwd_move_module: Any, root: Path) -> str:
    from local_operator.tools.builtin import _sessions_move_state_sentence

    stored = cwd_move_module.read_move_intent(root, SID)
    assert stored is not None
    return _sessions_move_state_sentence(
        {
            "move_stage": stored.stage,
            "move_destination": stored.destination_cwd,
            "move_source": stored.source_cwd,
        }
    )
