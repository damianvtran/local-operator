"""New sessions follow current; same-process helpers and developers stay pinned."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from local_operator import interpreter, update


def _generation(tmp_path: Path, name: str) -> Path:
    root = tmp_path / "lop" / "generations" / name / "tools" / "local-operator"
    binary = root / "bin" / "python3"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    return root


def test_new_session_follows_current_without_changing_parent(tmp_path, monkeypatch):
    old = _generation(tmp_path, "old")
    new = _generation(tmp_path, "new")
    stable = tmp_path / "lop"
    (stable / "current").symlink_to(old.parents[1])
    monkeypatch.setattr(update, "stable_root", lambda: stable)
    monkeypatch.setattr(sys, "prefix", str(old))
    monkeypatch.setattr(sys, "executable", str(old / "bin" / "python3"))
    before = interpreter.session_python_argv("-m", "local_operator.cli")
    assert before[0] == str(old / "bin" / "python3")
    (stable / "current").unlink()
    (stable / "current").symlink_to(new.parents[1])
    assert interpreter.session_python_argv("-m", "local_operator.cli") == [
        str(new / "bin" / "python3"),
        "-P",
        "-m",
        "local_operator.cli",
    ]
    # Helpers belonging to a busy old session must not switch code underneath it.
    assert interpreter.python_argv("-c", "pass")[0] == str(old / "bin" / "python3")
    assert sys.prefix == str(old)
    assert sys.executable == before[0]


@pytest.mark.parametrize("kind", ["editable", "legacy"])
def test_non_generation_install_stays_own(tmp_path, monkeypatch, kind):
    new = _generation(tmp_path, "new")
    stable = tmp_path / "lop"
    (stable / "current").symlink_to(new.parents[1])
    own = tmp_path / kind / ".venv"
    monkeypatch.setattr(update, "stable_root", lambda: stable)
    monkeypatch.setattr(sys, "prefix", str(own))
    monkeypatch.setattr(sys, "executable", str(own / "bin" / "python"))
    assert interpreter.session_python_argv("-m", "local_operator.cli")[0] == sys.executable


@pytest.mark.parametrize("state", ["missing", "dangling", "nonexecutable"])
def test_unavailable_current_preserves_old_interpreter(tmp_path, monkeypatch, state):
    old = _generation(tmp_path, "old")
    stable = tmp_path / "lop"
    if state != "missing":
        target = stable / "generations" / "absent"
        if state == "nonexecutable":
            new = _generation(tmp_path, "new")
            (new / "bin" / "python3").chmod(0o644)
            target = new.parents[1]
        (stable / "current").symlink_to(target)
    monkeypatch.setattr(update, "stable_root", lambda: stable)
    monkeypatch.setattr(sys, "prefix", str(old))
    monkeypatch.setattr(sys, "executable", str(old / "bin" / "python3"))
    assert interpreter.session_python_argv("-m", "local_operator.cli")[0] == sys.executable


@pytest.mark.asyncio
async def test_sessions_tool_launches_current_cli(tmp_path, monkeypatch):
    from local_operator.tools import builtin

    old = _generation(tmp_path, "old")
    new = _generation(tmp_path, "new")
    stable = tmp_path / "lop"
    (stable / "current").symlink_to(new.parents[1])
    monkeypatch.setattr(update, "stable_root", lambda: stable)
    monkeypatch.setattr(sys, "prefix", str(old))
    monkeypatch.setattr(sys, "executable", str(old / "bin" / "python3"))
    child = AsyncMock()
    child.returncode = 0
    child.communicate.return_value = (b"{}", b"")
    spawn = AsyncMock(return_value=child)
    monkeypatch.setattr(builtin.asyncio, "create_subprocess_exec", spawn)
    assert await builtin._sessions_launch(["exec", "--background", "test"], {}, str(tmp_path)) == (
        0,
        "",
    )
    assert spawn.call_args.args[:4] == (
        str(new / "bin" / "python3"),
        "-P",
        "-m",
        "local_operator.cli",
    )


@pytest.mark.asyncio
async def test_sessions_launch_executes_selected_image(tmp_path, monkeypatch):
    """Real subprocess boundary, not merely an assertion about argv construction."""
    from local_operator.tools import builtin

    old = _generation(tmp_path, "old")
    new = _generation(tmp_path, "new")
    stable = tmp_path / "lop"
    pointer = stable / "current"
    pointer.symlink_to(old.parents[1])
    for root, marker in ((old, "old"), (new, "new")):
        (root / "bin" / "python3").write_text(f"#!/bin/sh\necho {marker} >&2\n")
    monkeypatch.setattr(update, "stable_root", lambda: stable)
    monkeypatch.setattr(sys, "prefix", str(old))
    monkeypatch.setattr(sys, "executable", str(old / "bin" / "python3"))
    assert await builtin._sessions_launch(["--version"], {}, str(tmp_path)) == (0, "old\n")
    pointer.unlink()
    pointer.symlink_to(new.parents[1])
    assert await builtin._sessions_launch(["--version"], {}, str(tmp_path)) == (0, "new\n")
    assert sys.prefix == str(old)


@pytest.mark.parametrize("branded", [False, True])
def test_background_worker_brands_selected_generation(tmp_path, monkeypatch, branded):
    from unittest.mock import MagicMock

    from local_operator import exec_mode, procname

    old = _generation(tmp_path, "old")
    new = _generation(tmp_path, "new")
    stable = tmp_path / "lop"
    (stable / "current").symlink_to(new.parents[1])
    monkeypatch.setattr(update, "stable_root", lambda: stable)
    monkeypatch.setattr(sys, "prefix", str(old))
    monkeypatch.setattr(sys, "executable", str(old / "bin" / "python3"))
    monkeypatch.setattr(exec_mode, "resolve_hosting_model_dry", lambda _: ("test", "mock"))
    monkeypatch.setattr(exec_mode, "logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(exec_mode, "_process_generation", lambda _: None)
    spawn = MagicMock()
    spawn.return_value.pid = 123456
    monkeypatch.setattr(exec_mode, "spawn_disclaimed", spawn)
    identity = MagicMock(
        return_value=(
            "Local Operator [exec]" if branded else str(new / "bin" / "python3"),
            str(new / "bin" / "Local Operator") if branded else None,
        )
    )
    monkeypatch.setattr(procname, "spawn_identity_for_interpreter", identity)
    monkeypatch.setattr(
        procname, "spawn_identity", MagicMock(side_effect=AssertionError("old image"))
    )
    assert exec_mode._spawn_background("test", exec_mode.ExecArgs(background=True)) == 0
    assert identity.call_args.args[:2] == (procname.LABEL_EXEC, str(new / "bin" / "python3"))
    assert spawn.call_args.args[0][0] == identity.return_value[0]
    assert spawn.call_args.kwargs["executable"] == identity.return_value[1]
