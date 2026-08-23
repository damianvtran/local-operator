"""Process-relaunch helper: argv rewrite and OS backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from local_operator.reexec import RestartPlan, make_plan, plan_argv, replace_self


def test_plan_argv_no_resume() -> None:
    assert plan_argv(["lop", "--model", "gpt-4o"]) == ["lop", "--model", "gpt-4o"]


def test_plan_argv_injects_resume() -> None:
    assert plan_argv(["lop"], resume_id="abc123") == ["lop", "--resume", "abc123"]


def test_plan_argv_replaces_existing_resume() -> None:
    assert plan_argv(["lop", "--resume", "old", "--debug"], resume_id="new") == [
        "lop",
        "--debug",
        "--resume",
        "new",
    ]


def test_plan_argv_strips_resume_when_none() -> None:
    assert plan_argv(["lop", "--resume", "old", "--tui"]) == ["lop", "--tui"]


def test_plan_argv_preserves_model_and_hosting() -> None:
    argv = ["lop", "--hosting", "anthropic", "--model", "claude", "--train"]
    assert plan_argv(argv, resume_id="s1") == [
        "lop",
        "--hosting",
        "anthropic",
        "--model",
        "claude",
        "--train",
        "--resume",
        "s1",
    ]


def test_plan_argv_equals_form_resume() -> None:
    assert plan_argv(["lop", "--resume=old"], resume_id="new") == ["lop", "--resume", "new"]


def test_make_plan_records_resume_id() -> None:
    plan = make_plan(["lop"], resume_id="sess")
    assert plan.resume_id == "sess"
    assert plan.argv[-2:] == ["--resume", "sess"]


def test_replace_self_posix_execs_once() -> None:
    plan = RestartPlan(argv=["lop", "--resume", "s"], resume_id="s")
    with (
        patch("local_operator.reexec.sys.platform", "darwin"),
        patch("local_operator.reexec.os.execvpe") as execvpe,
    ):
        replace_self(plan)
        execvpe.assert_called_once()
        args, kwargs = execvpe.call_args
        assert args[0] == "lop"
        assert args[1] == ["lop", "--resume", "s"]
        assert "PATH" in args[2] or args[2] is not None


def test_replace_self_windows_popen_then_exit() -> None:
    from local_operator.reexec import _replace_windows

    with (
        patch("local_operator.reexec.subprocess.Popen") as popen,
        patch("local_operator.reexec.os._exit") as exit_,
    ):
        popen.return_value = MagicMock()
        _replace_windows(["lop.exe"], {"PATH": "/bin"})
        popen.assert_called_once()
        assert popen.call_args.args[0] == ["lop.exe"]
        assert popen.call_args.kwargs["close_fds"] is True
        exit_.assert_called_once_with(0)


def test_replace_self_dispatches_windows_backend() -> None:
    plan = RestartPlan(argv=["lop.exe"], resume_id=None)
    with (
        patch("local_operator.reexec._on_windows", return_value=True),
        patch("local_operator.reexec._replace_windows") as win,
        patch("local_operator.reexec._replace_posix") as posix,
    ):
        replace_self(plan)
        win.assert_called_once()
        posix.assert_not_called()
