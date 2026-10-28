"""The desktop bearer leaves the environment at serve start-up (operator ask a-772b).

``server/desktop.py`` (``TOKEN_FD_ENV`` and ``adopt_spawn_bearer``) states why: macOS
keeps a process's exec-time environment readable by same-user ``ps -E`` for the life of
the process, and QA on #3002 read the daemon's bearer that way from an agent's ``bash``
tool child. These tests pin the in-process half of the contract:

* the descriptor channel is read once, wins over the environment, and both names leave
  ``os.environ``;
* a stdin descriptor is retired onto ``/dev/null`` rather than closed;
* an app that still passes the environment variable gets a daemon that re-execs itself
  onto a pipe -- same interpreter, same argv, the value gone from the new environment;
* only a real launch of this product re-execs (never pytest, never an embedder);
* the serve reload hands the bearer on by pipe, never by environment;
* an OPTIMIZED launch (``python -O -m local_operator.cli serve``) reaches the same
  exec-time guarantee as a plain one -- the launch predicate is what decides that,
  so ``-O``/``-OO`` are launch flags, not an unknown option.

The exec-time half (``ps -E`` on a real daemon, a runtime and a tool child) cannot be
asserted from inside the test process; the PR carries that probe as evidence.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.server import desktop

TOKEN = "synthetic-bearer-channel-test"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``_SPAWN_BEARER`` is process-wide module state; monkeypatch restores it so one
    # test's adoption can never open the plane for the next.
    monkeypatch.setattr(desktop, "_SPAWN_BEARER", None)
    monkeypatch.setattr(desktop, "_CLAIMED", None)
    monkeypatch.delenv(desktop.TOKEN_ENV, raising=False)
    monkeypatch.delenv(desktop.TOKEN_FD_ENV, raising=False)


def _pipe_with(payload: bytes) -> int:
    read_end, write_end = os.pipe()
    os.write(write_end, payload)
    os.close(write_end)
    return read_end


def _no_relaunch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop, "_relaunchable_argv", lambda: None)


def test_the_descriptor_is_adopted_and_both_names_leave_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_relaunch(monkeypatch)
    fd = _pipe_with(TOKEN.encode() + b"\n")
    monkeypatch.setenv(desktop.TOKEN_FD_ENV, str(fd))
    monkeypatch.setenv(desktop.TOKEN_ENV, "env-loses-env-loses")
    desktop.adopt_spawn_bearer()
    assert desktop.desktop_posture().token == TOKEN
    assert desktop.TOKEN_FD_ENV not in os.environ
    assert desktop.TOKEN_ENV not in os.environ
    with pytest.raises(OSError):
        os.fstat(fd)  # consumed and closed: a child cannot inherit the read end


def _capture_exec(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub a relaunchable launch and record the ``execve`` it makes (then fail it)."""
    argv = [sys.executable, "-c", "from local_operator.cli import main; main()", "serve"]
    monkeypatch.setattr(desktop, "_relaunchable_argv", lambda: argv)
    seen: dict[str, Any] = {"calls": 0}

    def capture(path: str, args: list[str], env: dict[str, str]) -> None:
        seen["calls"] += 1
        seen.update(path=path, argv=args, env=env)
        named = env.get(desktop.TOKEN_FD_ENV)
        seen["delivered"] = os.read(int(named), 4096) if named else None
        raise OSError("stubbed exec")

    monkeypatch.setattr(os, "execve", capture)
    return seen


def test_both_names_set_still_relaunches_after_a_good_descriptor_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#3039 review M-1 / QA Q1: the env value must leave the exec-time block here too."""
    seen = _capture_exec(monkeypatch)
    monkeypatch.setenv(desktop.TOKEN_FD_ENV, str(_pipe_with(TOKEN.encode() + b"\n")))
    monkeypatch.setenv(desktop.TOKEN_ENV, "env-value-the-app-forgot-to-drop")
    desktop.adopt_spawn_bearer()
    assert seen["calls"] == 1
    assert desktop.TOKEN_ENV not in seen["env"]
    assert "env-value-the-app-forgot-to-drop" not in "".join(seen["env"].values())
    # The descriptor's bearer -- not the env one -- is what the successor is handed.
    assert seen["delivered"] == TOKEN.encode() + b"\n"
    assert desktop.desktop_posture().token == TOKEN


@pytest.mark.parametrize("named", ["987654", "not-a-number", "EMPTY"])
def test_both_names_set_still_relaunches_after_a_failed_descriptor_read(
    monkeypatch: pytest.MonkeyPatch, named: str
) -> None:
    """A failed read closes the plane AND still scrubs the env value by re-exec."""
    seen = _capture_exec(monkeypatch)
    if named == "EMPTY":
        named = str(_pipe_with(b""))
    monkeypatch.setenv(desktop.TOKEN_FD_ENV, named)
    monkeypatch.setenv(desktop.TOKEN_ENV, TOKEN)
    desktop.adopt_spawn_bearer()
    assert seen["calls"] == 1
    assert desktop.TOKEN_ENV not in seen["env"]
    assert TOKEN not in "".join(seen["env"].values())
    # Nothing adopted: the successor gets no descriptor either, so its plane is closed.
    assert desktop.TOKEN_FD_ENV not in seen["env"]
    assert desktop.desktop_posture().enabled is False


def test_the_size_cap_without_a_newline_is_a_failure_not_a_truncation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """#3039 review m-2 / QA Q3: 4096 bytes and no newline adopts nothing."""
    read_end, write_end = os.pipe()
    os.write(write_end, b"A" * 5000)  # under the 64 KiB pipe buffer, so one write fits
    os.close(write_end)
    monkeypatch.setenv(desktop.TOKEN_FD_ENV, str(read_end))
    with caplog.at_level("WARNING", logger=desktop.logger.name):
        desktop.adopt_spawn_bearer()
    assert desktop.desktop_posture().enabled is False
    assert "no newline" in caplog.text


@pytest.mark.parametrize("payload", [b"", b"\n", b"short-prefix\n", b"0123456789"])
def test_an_empty_or_torn_token_is_refused_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, payload: bytes
) -> None:
    """m-2 (a writer that died after N bytes) and Q4 (an empty pipe was silent)."""
    monkeypatch.setenv(desktop.TOKEN_FD_ENV, str(_pipe_with(payload)))
    with caplog.at_level("WARNING", logger=desktop.logger.name):
        desktop.adopt_spawn_bearer()
    assert desktop.desktop_posture().enabled is False
    assert "desktop plane stays closed" in caplog.text


def test_a_token_at_the_minimum_length_is_adopted(monkeypatch: pytest.MonkeyPatch) -> None:
    token = "x" * desktop._TOKEN_MIN_CHARS
    monkeypatch.setenv(desktop.TOKEN_FD_ENV, str(_pipe_with(token.encode() + b"\n")))
    desktop.adopt_spawn_bearer()
    assert desktop.desktop_posture().token == token


def test_eof_without_a_newline_still_delivers_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(desktop.TOKEN_FD_ENV, str(_pipe_with(TOKEN.encode())))
    desktop.adopt_spawn_bearer()
    assert desktop.desktop_posture().token == TOKEN


@pytest.mark.parametrize("value", ["not-a-number", "987654"])
def test_a_bad_descriptor_leaves_the_plane_closed(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Fail closed: the app then reads 401 and reports the daemon unpaired."""
    monkeypatch.setenv(desktop.TOKEN_FD_ENV, value)
    desktop.adopt_spawn_bearer()
    assert desktop.desktop_posture().enabled is False


def test_a_stdin_descriptor_is_pointed_at_dev_null_not_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing fd 0 would let the next ``open`` take the number "stdin" names."""
    saved = os.dup(0)
    try:
        os.dup2(_pipe_with(TOKEN.encode() + b"\n"), 0)
        monkeypatch.setenv(desktop.TOKEN_FD_ENV, "0")
        desktop.adopt_spawn_bearer()
        assert desktop.desktop_posture().token == TOKEN
        assert os.read(0, 16) == b""  # /dev/null: open, empty
    finally:
        os.dup2(saved, 0)
        os.close(saved)


def test_the_environment_fallback_is_adopted_and_scrubbed_when_no_relaunch_is_possible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_relaunch(monkeypatch)
    monkeypatch.setenv(desktop.TOKEN_ENV, TOKEN)
    desktop.adopt_spawn_bearer()
    assert desktop.desktop_posture().token == TOKEN
    assert desktop.TOKEN_ENV not in os.environ  # so no child is handed it


def test_the_environment_fallback_relaunches_onto_a_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The compatibility path an unchanged desktop app takes: one self-exec, same argv."""
    argv = [sys.executable, "-c", "from local_operator.cli import main; main()", "serve"]
    monkeypatch.setattr(desktop, "_relaunchable_argv", lambda: argv)
    seen: dict[str, Any] = {}

    def capture(path: str, args: list[str], env: dict[str, str]) -> None:
        seen.update(path=path, argv=args, env=env)
        seen["delivered"] = os.read(int(env[desktop.TOKEN_FD_ENV]), 4096)
        raise OSError("stubbed exec")  # the failure branch must clean up the pipe

    monkeypatch.setattr(os, "execve", capture)
    monkeypatch.setenv(desktop.TOKEN_ENV, TOKEN)
    desktop.adopt_spawn_bearer()
    assert seen["path"] == sys.executable
    assert seen["argv"] == argv
    assert desktop.TOKEN_ENV not in seen["env"]
    assert TOKEN not in "".join(seen["env"].values())
    assert seen["delivered"] == TOKEN.encode() + b"\n"
    with pytest.raises(OSError):
        os.fstat(int(seen["env"][desktop.TOKEN_FD_ENV]))  # closed after the failed exec
    # A failed exec keeps the daemon serving with the bearer it adopted.
    assert desktop.desktop_posture().token == TOKEN


def test_the_successor_takes_the_descriptor_branch_and_never_relaunches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loop-free by construction: the successor's environment has no ``TOKEN_ENV``."""
    monkeypatch.setattr(os, "execve", lambda *a: pytest.fail("a descriptor launch re-exec'd"))
    monkeypatch.setattr(desktop, "_relaunchable_argv", lambda: [sys.executable, "x"])
    monkeypatch.setenv(desktop.TOKEN_FD_ENV, str(_pipe_with(TOKEN.encode() + b"\n")))
    desktop.adopt_spawn_bearer()
    assert desktop.desktop_posture().token == TOKEN


@pytest.mark.parametrize(
    ("orig_argv", "own_launch", "expected"),
    [
        # The desktop app's launch shape (owned-serve-launch.ts SERVE_ENTRYPOINT).
        (["/py", "-c", "from local_operator.cli import main; main()", "serve"], False, True),
        # A real ``lop serve`` / ``-m local_operator.cli`` launch.
        (["/py", "/bin/lop", "serve"], True, True),
        # pytest, an embedder, some other ``-c`` program: never replace the caller.
        (["/py", "-m", "pytest", "tests"], False, False),
        (["/py", "-c", "import something_else", "serve"], False, False),
        (["/py"], False, False),
    ],
)
def test_only_a_real_launch_relaunches(
    monkeypatch: pytest.MonkeyPatch, orig_argv: list[str], own_launch: bool, expected: bool
) -> None:
    from local_operator import procname

    monkeypatch.setattr(sys, "orig_argv", orig_argv)
    monkeypatch.setattr(procname, "is_own_launch", lambda: own_launch)
    assert (desktop._relaunchable_argv() is not None) is expected


def test_the_reload_hands_the_bearer_on_by_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """``server/reload.py`` builds its successor's environment from this handoff."""
    monkeypatch.setattr(desktop, "_SPAWN_BEARER", TOKEN)
    env, fd = desktop.spawn_bearer_handoff()
    try:
        assert fd is not None
        assert set(env) == {desktop.TOKEN_FD_ENV}
        assert os.get_inheritable(fd) is True
        assert os.read(fd, 4096) == TOKEN.encode() + b"\n"
    finally:
        if fd is not None:
            os.close(fd)


def test_no_bearer_means_no_handoff() -> None:
    assert desktop.spawn_bearer_handoff() == ({}, None)


@pytest.mark.skipif(os.name != "posix", reason="the exec-time environment is a POSIX read")
@pytest.mark.parametrize("optimize", ["-O", "-OO"])
def test_an_optimized_launch_leaves_no_bearer_in_the_exec_time_environment(
    tmp_path: Path, optimize: str
) -> None:
    """The real ``-O -m local_operator.cli serve`` shape, measured on the daemon.

    ``procname.is_own_launch`` decides whether ``adopt_spawn_bearer`` re-execs, and an
    unknown leading option answers "not our launch" -- so ``-O`` used to leave the
    bearer in the daemon's exec-time block for its whole life, which ``ps eww`` shows
    (#3022 code review round 5, MAJOR). This starts a REAL daemon the way an operator
    would and reads it the way a model child would: ``ps eww`` on the pid.
    """

    import socket
    import subprocess
    import time
    import urllib.error
    import urllib.request

    port_probe = socket.socket()
    port_probe.bind(("127.0.0.1", 0))
    port = port_probe.getsockname()[1]
    port_probe.close()

    home = tmp_path / "home"
    config = tmp_path / "config"
    home.mkdir()
    config.mkdir()
    token = "synthetic-optimized-launch-" + "q" * 16
    root = Path(__file__).resolve().parents[3]
    env = {
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(config),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONPATH": str(root),
        "GIT_CONFIG_SYSTEM": "/dev/null",
        desktop.TOKEN_ENV: token,
    }
    log = tmp_path / "serve.log"
    with log.open("wb") as sink:
        process = subprocess.Popen(
            [
                sys.executable,
                optimize,
                "-m",
                "local_operator.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            stdout=sink,
            stderr=subprocess.STDOUT,
            cwd=str(tmp_path),
        )
    try:
        deadline = time.monotonic() + 90
        healthy = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                    healthy = True
                    break
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        assert healthy, log.read_text()[-4000:]

        # ``ps eww <pid>`` is what a model's bash child can run; ``-p`` on the pid we
        # spawned also proves the re-exec kept it (an exec, not a new process).
        listing = subprocess.run(
            ["ps", "eww", "-p", str(process.pid)], capture_output=True, text=True, timeout=30
        ).stdout
        assert token not in listing, "the bearer is in the daemon's exec-time environment"
        assert process.poll() is None, "the re-exec must keep the pid"

        def status(headers: dict[str, str]) -> int:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/desktop/team-bridge/status", headers=headers
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as answer:
                    return answer.status
            except urllib.error.HTTPError as refusal:
                return refusal.code

        assert status({"Authorization": f"Bearer {token}"}) == 200
        assert status({}) == 401
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
