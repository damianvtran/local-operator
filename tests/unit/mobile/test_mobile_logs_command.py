"""`lop mobile logs` must show BOTH writers of the relay log, and keep showing them.

Why this is a regression guard rather than a nicety: the daemon and its session
runtime children write separate files on purpose — the daemon's `mobile.log` is a
launchd ``StandardOutPath`` whose fd can never be reopened, and bounding a file
means RENAMING it, so a runtime that rotated the daemon's log would move the
daemon's stream (and every sibling runtime's) out of the file this command reads.
Measured on the operator's machine after a single rename of that path: nine
runtime children held the renamed inode while only the daemon held the fresh
`mobile.log`.

Two files per writer class is only acceptable while one command still shows both,
so the argv this command builds is the contract under test. The follow flag is
part of it: ``-f`` follows the fd it opened, so it (a) never reads a file created
after it started — the normal state on a machine whose daemons are up but whose
runtimes have all exited — and (b) goes blind the first time a bounded runtime log
rotates, because bounding means renaming. Both were measured against the system
``tail`` in review; ``-F`` retries by name and reopens on rename, which is why it
is the flag, and why a not-yet-existing path is passed rather than omitted.
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

from local_operator.cli import build_cli_parser, mobile_command
from local_operator.paths import CONFIG_DIR_ENV, log_dir, runtime_log_path


def _args(*extra: str) -> argparse.Namespace:
    return build_cli_parser().parse_args(["mobile", "logs", *extra])


def test_logs_reads_the_daemon_and_the_runtimes_in_one_tail(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    daemon_log = log_dir() / "mobile.log"
    runtime_log = runtime_log_path()
    daemon_log.parent.mkdir(parents=True, exist_ok=True)
    daemon_log.write_text("daemon\n", encoding="utf-8")
    runtime_log.write_text("runtime\n", encoding="utf-8")

    with patch("subprocess.call", return_value=0) as call:
        assert mobile_command(_args("--lines", "7", "-f")) == 0

    argv = call.call_args.args[0]
    assert argv[:3] == ["tail", "-n", "7"]
    assert "-F" in argv, "follow must retry by name, so a rotation cannot blind it"
    assert "-f" not in argv
    assert str(daemon_log) in argv, "the daemon's own log must still be read"
    assert str(runtime_log) in argv, "the runtimes' log must be read too"
    assert argv.index(str(daemon_log)) < argv.index(str(runtime_log))


def test_logs_passes_a_runtime_file_that_does_not_exist_yet(tmp_path, monkeypatch) -> None:
    """Under `--follow` the not-yet-created file must still be named: `-F` retries
    a missing path, so the follow picks the file up the moment a runtime creates
    it — which omitting the path would make impossible."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    daemon_log = log_dir() / "mobile.log"
    daemon_log.parent.mkdir(parents=True, exist_ok=True)
    daemon_log.write_text("daemon\n", encoding="utf-8")
    assert not runtime_log_path().exists()

    with patch("subprocess.call", return_value=0) as call:
        assert mobile_command(_args("--follow")) == 0

    argv = call.call_args.args[0]
    assert str(daemon_log) in argv
    assert str(runtime_log_path()) in argv


def test_logs_without_follow_omits_a_missing_file_rather_than_erroring(
    tmp_path, monkeypatch
) -> None:
    """A plain `tail` does NOT tolerate a missing operand: it warns and exits 1,
    which is what shipped for one revision and would have made `lop mobile logs`
    fail on every freshly booted machine (measured through the real CLI: exit 1
    and `tail: …/logs/runtime.log: No such file or directory`). Only the following
    branch may pass a path that may not exist."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    daemon_log = log_dir() / "mobile.log"
    daemon_log.parent.mkdir(parents=True, exist_ok=True)
    daemon_log.write_text("daemon\n", encoding="utf-8")
    assert not runtime_log_path().exists()

    with patch("subprocess.call", return_value=0) as call:
        assert mobile_command(_args("--lines", "2")) == 0

    argv = call.call_args.args[0]
    assert str(daemon_log) in argv
    assert str(runtime_log_path()) not in argv
    assert "-F" not in argv and "-f" not in argv


def test_logs_with_no_log_files_at_all_does_not_read_stdin(tmp_path, monkeypatch, capsys) -> None:
    """`tail` with no operand reads STDIN, so an empty file list would hang the
    command on a machine whose daemon has not written yet. It must say so instead."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    assert not log_dir().exists()

    with patch("subprocess.call", return_value=0) as call:
        assert mobile_command(_args()) == 0

    call.assert_not_called()
    assert "no log files yet" in capsys.readouterr().out
