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


#: A descriptor number past ``FD_SETSIZE`` (1024), the ceiling ``select()`` cannot cross.
_HIGH_FD = 1100


@pytest.mark.skipif(os.name != "posix", reason="the bounded wait and F_DUPFD are POSIX-only")
def test_a_descriptor_numbered_past_fd_setsize_is_still_adopted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bearer pipe can arrive at any number; ``select()`` refused >= 1024 (CI 44047538630).

    A long-lived pytest worker -- or a busy serve handing its bearer to a reload successor --
    returns pipe numbers past 1024 from ``os.pipe()``. The wait must not care. The number is
    pinned with ``F_DUPFD`` rather than by opening a thousand files, so the test is exact.
    macOS starts processes with a soft RLIMIT_NOFILE of 256, below the target, so the soft
    limit is raised (never past the hard limit) for this test and restored after it.
    """
    import fcntl
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = _HIGH_FD + 16
    if hard != resource.RLIM_INFINITY and hard < needed:
        pytest.skip(f"hard RLIMIT_NOFILE {hard} cannot hold descriptor {_HIGH_FD}")
    if soft != resource.RLIM_INFINITY and soft < needed:
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
    try:
        low = _pipe_with(TOKEN.encode() + b"\n")
        try:
            high = fcntl.fcntl(low, fcntl.F_DUPFD, _HIGH_FD)
        finally:
            os.close(low)
        assert high >= _HIGH_FD
        monkeypatch.setenv(desktop.TOKEN_FD_ENV, str(high))
        desktop.adopt_spawn_bearer()
        assert desktop.desktop_posture().token == TOKEN
        # Consumed and closed, like any other bearer descriptor.
        with pytest.raises(OSError):
            os.fstat(high)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


@pytest.mark.skipif(os.name != "posix", reason="only POSIX bounds the wait")
def test_a_writer_that_never_writes_times_out_with_the_plane_closed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The bounded wait (now ``poll``) still gives up: boot never hangs on a silent spawner."""
    monkeypatch.setattr(desktop, "_TOKEN_FD_TIMEOUT_S", 0.05)
    read_end, write_end = os.pipe()  # write end held open and silent
    try:
        monkeypatch.setenv(desktop.TOKEN_FD_ENV, str(read_end))
        desktop.adopt_spawn_bearer()
    finally:
        os.close(write_end)
    assert desktop.desktop_posture().enabled is False
    assert "delivered nothing in time" in caplog.text


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
@pytest.mark.parametrize(
    ("label", "leading", "as_script"),
    [
        # The shapes an argv-based "is this our launch" question got wrong, each
        # of which left the bearer in the daemon's exec-time block (#3022 code
        # review rounds 5 and 6). The decision is now made from what is running,
        # so none of these can move it.
        #
        # ``-S`` and ``-IO`` are deliberately absent HERE and live in
        # ``test_procname`` as predicate facts: neither can be measured against a
        # running daemon. ``-S`` skips site-packages, so the app cannot import its
        # dependencies at all (a real ``lop -S`` is equally dead), and ``-I`` both
        # ignores ``PYTHONPATH`` and stops ``-m`` from putting the working directory
        # on the path, so a local run would exercise whatever the environment has
        # installed rather than this tree.
        ("-O", ["-O"], False),
        ("-OO", ["-OO"], False),
        ("-Ou", ["-Ou"], False),
        ("-q", ["-q"], False),
        ("-b", ["-b"], False),
        ("-Es", ["-Es"], False),
        ("-sP", ["-sP"], False),
        ("-uO", ["-uO"], False),
        ("-R", ["-R"], False),
        ("-mlocal_operator.cli", [], False),  # attached -m, substituted below
        ("-sP console script", ["-sP"], True),
        ("console script", [], True),
    ],
)
def test_a_recognised_launch_leaves_no_bearer_in_the_exec_time_environment(
    tmp_path: Path, label: str, leading: list[str], as_script: bool
) -> None:
    """Each real launch shape, measured on a REAL daemon the way a model child could.

    ``procname.is_own_launch`` decides whether ``adopt_spawn_bearer`` re-execs, and
    the answer comes from ``__main__`` now -- the module for a ``-m`` launch, the
    script for a console one -- so interpreter flags cannot defeat it. This starts
    the launch an operator would and reads it with ``ps eww``.
    """

    import json
    import socket
    import subprocess
    import time
    import urllib.error
    import urllib.request

    # Port selection: the child binds ``--port 0`` ITSELF and reports the port it
    # actually got in its serve record. The previous shape bound port 0 here,
    # read the number, CLOSED the socket and passed it as ``--port``: a
    # bind/close/reuse window in which any other process could take the port, so
    # ``/health`` and the bearer/``ps``/auth assertions could be answered by (or
    # about) a process that is not the child (CI saw ``[Errno 98]`` on a
    # busy port). ``serve`` outside ``--reload`` binds before uvicorn starts, so
    # there is no probe/close on this path; the lifespan then publishes
    # ``<config>/run/serve/<pid>.json`` with ``port`` (the port ACTUALLY bound)
    # and ``instance_id`` (also returned by ``/health``). The record, not the
    # log, is the channel: the log line goes to a block-buffered file sink.
    # Same mechanism as tests/e2e/test_serve_build_announcement.py. This is a
    # startup wait only -- there is deliberately no retry/fresh-port on a bind
    # failure; the child would exit and the loop below fails with the log tail.
    #
    # Collision control: the test OWNS a listening ephemeral socket for the whole
    # case. The kernel never hands a listening port out twice, so the child's
    # port must differ from it; asserted below as documentation of the control.
    incumbent = socket.socket()
    incumbent.bind(("127.0.0.1", 0))
    incumbent.listen(1)
    incumbent_port = incumbent.getsockname()[1]

    root = Path(__file__).resolve().parents[3]
    home = tmp_path / "home"
    config = tmp_path / "config"
    home.mkdir()
    config.mkdir()
    token = "[redacted]" + "q" * 16

    if as_script:
        # The console-script shape without depending on an installed entry point:
        # ``__main__`` is a script FILE named ``lop``, which is exactly what
        # ``bin/lop`` is to the interpreter.
        entry = tmp_path / "lop"
        entry.write_text("import sys\n\nfrom local_operator.cli import main\n\nsys.exit(main())\n")
        target = [str(entry)]
    elif leading == [] and label.startswith("-m"):
        target = ["-mlocal_operator.cli"]
    else:
        target = [*leading, "-m", "local_operator.cli"]

    env = {
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(config),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONPATH": str(root),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        desktop.TOKEN_ENV: token,
    }
    log = tmp_path / "serve.log"
    try:
        with log.open("wb") as sink:
            process = subprocess.Popen(
                [sys.executable, *target, "serve", "--host", "127.0.0.1", "--port", "0"],
                env=env,
                stdout=sink,
                stderr=subprocess.STDOUT,
                # ``-Es`` ignores every PYTHON* variable, so the package has to resolve
                # from the working directory for that shape: ``-m`` prepends it.
                cwd=str(root),
            )
    except BaseException:
        incumbent.close()
        raise
    started = process.pid
    record_path = config / "run" / "serve" / f"{started}.json"
    try:
        deadline = time.monotonic() + 90
        healthy = False
        port = 0
        record: dict[str, Any] = {}  # heterogeneous serve-record JSON (str/int/None)
        # Bound before the loop: ``healthy`` gates every read of ``body``, but the
        # type checker cannot see that; a no-response run still fails closed on
        # the ``assert healthy`` below.
        body: dict[str, Any] = {}
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            if not port:
                # Only "not published yet" / "being staged" are transient. The
                # record holds the access token: never print it.
                try:
                    record = json.loads(record_path.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    time.sleep(0.5)
                    continue
                port = int(record.get("port") or 0)
                if not port:
                    time.sleep(0.5)
                    continue
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as answer:
                    body = json.loads(answer.read())
                healthy = True
                break
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        assert healthy, f"[{label}] " + log.read_text()[-4000:]
        record_pid = record.get("pid")
        assert record_pid == started, f"[{label}] the record is not the child's"
        assert port != incumbent_port, f"[{label}] child shares the owned incumbent's port"
        # Identity, not just 200: the answer must come from THIS child.
        # ``/health`` nests its payload under ``result``. Compare plain strings so
        # a failure never has pytest repr the record (it holds the access token).
        published_id = str(record.get("instance_id") or "")
        answered_id = str((body.get("result") or {}).get("instance_id") or "")
        assert published_id and answered_id == published_id, (
            f"[{label}] /health instance_id does not match the child's serve record "
            f"(port={port}, pid={started}, record_has_id={bool(published_id)})"
        )

        listing = subprocess.run(
            ["ps", "eww", "-p", str(started)], capture_output=True, text=True, timeout=30
        ).stdout
        assert token not in listing, f"[{label}] the bearer is in the exec-time environment"
        assert process.poll() is None, f"[{label}] the re-exec must keep the pid"

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
        incumbent.close()
        assert process.pid == started, "an exec must not hand the pid to another process"
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
