"""`lop mobile logs` must show BOTH writers of the relay log, not one tail.

Why this is a regression guard rather than a nicety: the daemon and its session
runtime children write separate files on purpose — the daemon's `mobile.log` is a
launchd ``StandardOutPath`` whose fd can never be reopened, and bounding a file
means RENAMING it, so a runtime that rotated the daemon's log would move the
daemon's stream (and every sibling runtime's) out of the file this command reads.
Measured on the operator's machine after a single rename of that path: nine
runtime children held the renamed inode while only the daemon held the fresh
`mobile.log`.

Two files per writer class is only acceptable while one command still shows both,
so the argv this command builds is the contract under test — including the case
where the runtime file does not exist yet, which is the normal state on a machine
whose daemons are running but whose runtimes are all idle or not yet started.
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
    assert "-f" in argv
    assert str(daemon_log) in argv, "the daemon's own log must still be read"
    assert str(runtime_log) in argv, "the runtimes' log must be read too"
    assert argv.index(str(daemon_log)) < argv.index(str(runtime_log))


def test_logs_tolerates_a_machine_with_no_runtime_file_yet(tmp_path, monkeypatch) -> None:
    """Before any runtime has started the file does not exist, and `tail` errors
    on a missing path — so it must simply be left out rather than passed."""
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    daemon_log = log_dir() / "mobile.log"
    daemon_log.parent.mkdir(parents=True, exist_ok=True)
    daemon_log.write_text("daemon\n", encoding="utf-8")

    with patch("subprocess.call", return_value=0) as call:
        assert mobile_command(_args()) == 0

    argv = call.call_args.args[0]
    assert str(daemon_log) in argv
    assert str(runtime_log_path()) not in argv
