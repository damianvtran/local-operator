"""The CLI's owned-resume guards: interactive --resume attaches, exec --resume
refuses exit 1, unowned paths are untouched."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_operator import cli_attach
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


def test_interactive_resume_owned_runs_attach_path(config: Path, monkeypatch) -> None:
    sleeper_pid = os.getppid()
    _own(config, "sess-owned2", sleeper_pid)
    called: list[tuple[object, object, object]] = []

    def fake_attach(cfg, session_id, owner):  # noqa: ANN001
        called.append((cfg, session_id, owner))
        return 0

    monkeypatch.setattr(cli_attach, "run_owned_resume_attach", fake_attach, raising=False)
    # cli imports the function lazily inside main; patch the module attribute
    # the lazy import resolves against.
    import local_operator.cli_attach as mod

    monkeypatch.setattr(mod, "run_owned_resume_attach", fake_attach)
    monkeypatch.setattr("sys.argv", ["local-operator", "--resume", "sess-owned2"])
    code = cli_main()
    assert code == 0
    assert called and called[0][1] == "sess-owned2"
    assert called[0][2] == sleeper_pid


def test_owned_attach_refusal_copy_when_no_record(config: Path, capsys) -> None:
    """The attach gate's graceful degradation: an owner with no record gets
    today's refusal line and exit 1."""
    sleeper_pid = os.getppid()
    _own(config, "sess-old", sleeper_pid)
    code = cli_attach.run_owned_resume_attach(config, "sess-old", sleeper_pid)
    assert code == 1
    err = capsys.readouterr().err
    assert "already open in another process" in err
    assert "watch and steer" in err


def test_resume_here_relaunch_preserves_session_id(config: Path, monkeypatch) -> None:
    """Exit 75 from the attach app means the owner died; relaunch the SAME
    transcript now that its claim is safe to take."""
    from local_operator.mobile.types import SessionRecord

    owner = os.getppid()
    record = SessionRecord(
        pid=owner,
        kind="tui",
        session_id="sess-relaunch",
        conversation_name="",
        cwd="/tmp",
        model_label="",
        control_port=1,
        control_key="k",
    )
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_owner_record",
        lambda cfg, sid: (record, owner),
    )
    monkeypatch.setattr("local_operator.tui.attach_screen.run_attach_app", lambda *a: 75)
    seen: list[list[str]] = []
    monkeypatch.setattr("subprocess.call", lambda argv: seen.append(list(argv)) or 0)
    assert cli_attach.run_owned_resume_attach(config, "sess-relaunch", owner) == 0
    assert seen[0][-2:] == ["--resume", "sess-relaunch"]


def test_startup_import_weight_unchanged() -> None:
    """The CLI startup guard: importing cli must not pull Textual or the
    mobile package's heavy half (attach imports are lazy on the owned branch)."""
    import subprocess
    import sys

    code = (
        "import sys, local_operator.cli; "
        "bad = [m for m in sys.modules if m.startswith('textual') "
        "or m.startswith('local_operator.mobile.attach_client') "
        "or m.startswith('local_operator.tui.attach_screen')]; "
        "print('LEAKED:' + ','.join(bad) if bad else 'CLEAN')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd="/tmp/lop-attach"
    )
    assert out.stdout.strip() == "CLEAN", out.stdout
