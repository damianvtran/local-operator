"""Cold-resume guards: full TUI uses the shared session factory; exec refuses."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_operator.cli import main as cli_main


@pytest.fixture
def config(tmp_path: Path, monkeypatch) -> Path:
    cfg = tmp_path / ".local-operator"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))
    return cfg


def _own(config: Path, session_id: str, pid: int) -> None:
    d = config / "sessions" / session_id
    d.mkdir(parents=True, exist_ok=True)
    # resume_dir requires a transcript to consider the session resumable.
    (d / "transcript.jsonl").write_text("")
    (d / ".session.pid").write_text(str(pid))


def test_exec_resume_owned_refuses_exit_1(config: Path, monkeypatch, capsys) -> None:
    sleeper_pid = os.getppid()  # a live pid that is not this process
    _own(config, "sess-owned", sleeper_pid)
    monkeypatch.setattr(
        "sys.argv",
        ["local-operator", "exec", "--resume", "sess-owned", "do the thing"],
    )
    code = cli_main()
    assert code == 1
    err = capsys.readouterr().err
    assert "already open in another process" in err
    assert str(sleeper_pid) in err


def test_exec_resume_unowned_proceeds_to_exec(config: Path, monkeypatch) -> None:
    # No marker: the guard must not interfere. Patch the preflight AND run_exec
    # to observe the call rather than standing up a provider.
    _own(config, "sess-free", os.getpid())  # owner == self: not "another process"
    seen: list[tuple[object, object]] = []

    def fake_run_exec(command, args):  # noqa: ANN001
        seen.append((command, args))
        return 0

    monkeypatch.setattr(
        "local_operator.exec_mode.resolve_hosting_model_dry",
        lambda a: ("anthropic", "claude-x"),
        raising=False,
    )
    import local_operator.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_preflight_api_key", lambda *a, **k: None)
    monkeypatch.setattr("local_operator.exec_mode.run_exec", fake_run_exec)
    monkeypatch.setattr("sys.argv", ["local-operator", "exec", "--resume", "sess-free", "task"])
    code = cli_main()
    assert code == 0
    assert seen and seen[0][0] == "task"


def test_standalone_attach_modules_are_deleted() -> None:
    """Cold resume cannot regress to the projection screen or exit-75 shim."""
    import importlib.util

    assert importlib.util.find_spec("local_operator.cli_attach") is None
    assert importlib.util.find_spec("local_operator.tui.attach_screen") is None


def test_startup_import_weight_unchanged() -> None:
    """The CLI startup guard: importing cli must not pull Textual or the
    mobile package's heavy half (attach imports are lazy on the owned branch)."""
    import subprocess
    import sys

    code = (
        "import sys, local_operator.cli; "
        "bad = [m for m in sys.modules if m.startswith('textual') "
        "or m.startswith('local_operator.mobile.attach_client') "
        "or m.startswith('local_operator.session.remote')]; "
        "print('LEAKED:' + ','.join(bad) if bad else 'CLEAN')"
    )
    repo_root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=repo_root
    )
    assert out.stdout.strip() == "CLEAN", out.stdout
