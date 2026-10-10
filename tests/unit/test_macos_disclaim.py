"""``macos_disclaim``: the disclaimed spawn, its fallback, and the chain evidence.

The stakes here are the 2026-10-09 incident's: a runtime that LOOKS detached
(own session, ppid 1) but still sits in the desktop app's macOS responsibility
scope is a runtime a UI force-quit can sweep. So the tests pin the POSIX spawn
shape (setsid + CLOEXEC + disclaim + stdio file actions), the chain that makes
a later unattributed signal readable, and — just as load-bearing — the fallback
that keeps every other host on exactly today's behaviour.

Nothing here spawns a process except the real-path test, which runs ``/bin/sh``
for about a second on darwin only.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator import macos_disclaim
from local_operator.macos_disclaim import (
    ENV_SPAWN_CHAIN,
    DisclaimedProcess,
    spawn_disclaimed,
)

DARWIN_ONLY = pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only mechanism")


@pytest.fixture(autouse=True)
def _reset_fallback_flag() -> Any:
    """The loud-once flag is per-process module state; every test starts clean."""
    macos_disclaim._fallback_logged = False
    yield
    macos_disclaim._fallback_logged = False


class _FakeLibc:
    """Records the call sequence the module makes; writes a pid back like the real one."""

    def __init__(self, pid: int = 4242) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.pid = pid
        self.spawn_path: bytes | None = None
        self.argv: list[bytes] = []
        self.envp: list[bytes] = []

    def posix_spawnattr_init(self, attr: Any) -> int:
        self.calls.append(("attr_init",))
        return 0

    def posix_spawnattr_destroy(self, attr: Any) -> int:
        self.calls.append(("attr_destroy",))
        return 0

    def posix_spawnattr_setflags(self, attr: Any, flags: int) -> int:
        self.calls.append(("setflags", flags))
        return 0

    def posix_spawn_file_actions_init(self, fa: Any) -> int:
        self.calls.append(("fa_init",))
        return 0

    def posix_spawn_file_actions_destroy(self, fa: Any) -> int:
        self.calls.append(("fa_destroy",))
        return 0

    def posix_spawn_file_actions_adddup2(self, fa: Any, src: int, dst: int) -> int:
        self.calls.append(("adddup2", src, dst))
        return 0

    def posix_spawn_file_actions_addopen(
        self, fa: Any, fd: int, path: bytes, flags: int, mode: int
    ) -> int:
        self.calls.append(("addopen", fd, path, flags))
        return 0

    def posix_spawn_file_actions_addinherit_np(self, fa: Any, fd: int) -> int:
        self.calls.append(("addinherit", fd))
        return 0

    def posix_spawn_file_actions_addchdir_np(self, fa: Any, path: bytes) -> int:
        self.calls.append(("addchdir", path))
        return 0

    def responsibility_spawnattrs_setdisclaim(self, attr: Any, disclaim: bool) -> int:
        self.calls.append(("setdisclaim", bool(disclaim)))
        return 0

    def posix_spawn(
        self,
        pid_ref: Any,
        path: bytes,
        fa: Any,
        attr: Any,
        argv: Any,
        envp: Any,
    ) -> int:
        self.calls.append(("spawn",))
        self.spawn_path = bytes(path)
        self.argv = [bytes(a) for a in argv if a]
        self.envp = [bytes(a) for a in envp if a]
        pid_ref._obj.value = self.pid
        return 0


def _install_fake_libc(monkeypatch: pytest.MonkeyPatch, fake: _FakeLibc) -> None:
    monkeypatch.setattr(macos_disclaim, "_load_libc", lambda: fake)
    monkeypatch.setattr(macos_disclaim, "_fallback_reason", lambda: None)


def test_posix_spawn_shape_maps_stdio_and_disclaims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One spawn through the POSIX path: flags, disclaim, stdio file actions, argv/env."""
    fake = _FakeLibc(pid=4242)
    _install_fake_libc(monkeypatch, fake)
    log_path = tmp_path / "out.log"
    with open(log_path, "wb") as handle:
        handle_fd = handle.fileno()
        proc = spawn_disclaimed(
            ["/bin/echo", "hi"],
            executable="/bin/echo",
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    assert isinstance(proc, DisclaimedProcess)
    assert proc.pid == 4242
    assert ("setflags", 0x0400 | 0x4000) in fake.calls
    assert ("setdisclaim", True) in fake.calls
    assert ("addopen", 0, b"/dev/null", os.O_RDWR) in fake.calls
    assert ("adddup2", handle_fd, 1) in fake.calls
    assert ("adddup2", 1, 2) in fake.calls
    assert fake.spawn_path == b"/bin/echo"
    assert fake.argv == [b"/bin/echo", b"hi"]
    chain = json.loads(
        next(e.split(b"=", 1)[1] for e in fake.envp if e.startswith(b"LOP_SPAWN_CHAIN="))
    )
    assert isinstance(chain, list) and chain[0]["pid"] == os.getpid()


def test_posix_spawn_shape_keeps_pass_fds_and_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``pass_fds``/``cwd``/``DEVNULL`` stdout are the standby's and nobody else's."""
    fake = _FakeLibc(pid=7)
    _install_fake_libc(monkeypatch, fake)
    spawn_disclaimed(
        ["/bin/true"],
        cwd=str(tmp_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        pass_fds=(9,),
    )
    assert ("addinherit", 9) in fake.calls
    assert ("addchdir", os.fsencode(str(tmp_path))) in fake.calls
    assert ("addopen", 1, b"/dev/null", os.O_WRONLY) in fake.calls
    assert ("addopen", 2, b"/dev/null", os.O_WRONLY) in fake.calls


def test_pipe_stdio_falls_back_instead_of_misreading_minus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``PIPE`` is the int -1; refusing it early is what keeps ``dup2(-1)`` out.

    The check order is load-bearing: without the explicit refusal this call
    would silently ask for ``dup2(-1, 1)`` and spawn a child attached to
    nothing (or fail with EBADF and be read as an environment bug).
    """
    fake = _FakeLibc()
    _install_fake_libc(monkeypatch, fake)
    calls: list[dict[str, Any]] = []

    def record_fallback(argv: Any, **kwargs: Any) -> str:
        calls.append({"argv": list(argv), **kwargs})
        return "POPEN"

    monkeypatch.setattr(macos_disclaim, "_fallback_popen", record_fallback)
    result = spawn_disclaimed(["/bin/true"], stdout=subprocess.PIPE)
    assert result == "POPEN"
    assert calls and calls[0]["stdout"] is subprocess.PIPE
    assert not any(call[0] == "spawn" for call in fake.calls)


def test_close_fds_false_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLibc()
    _install_fake_libc(monkeypatch, fake)
    monkeypatch.setattr(macos_disclaim, "_fallback_popen", lambda *a, **k: "POPEN")
    assert spawn_disclaimed(["/bin/true"], close_fds=False) == "POPEN"
    assert not any(call[0] == "spawn" for call in fake.calls)


def test_missing_symbols_fall_back_loudly_once(
    monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """The fallback is never silent, and never per-spawn noisy.

    ``loudly once`` is the contract: an operator (or this repo's next agent)
    must be able to see from a single log read that this host lacks the lever,
    and a serve's log must not carry one line per runtime spawn.
    """
    monkeypatch.setattr(macos_disclaim, "_fallback_reason", lambda: "missing symbol: test")
    calls: list[Any] = []
    monkeypatch.setattr(
        macos_disclaim, "_fallback_popen", lambda argv, **kwargs: calls.append(list(argv)) or "POP"
    )
    with caplog.at_level(logging.WARNING, logger="local_operator.macos_disclaim"):
        assert spawn_disclaimed(["/bin/true"]) == "POP"
        assert spawn_disclaimed(["/bin/true"]) == "POP"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "missing symbol: test" in warnings[0].getMessage()
    assert len(calls) == 2


def test_attribute_errnos_fall_back_and_other_errors_raise(
    monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """An OS refusing our flags degrades; a missing binary is still the caller's error."""
    monkeypatch.setattr(macos_disclaim, "_fallback_reason", lambda: None)
    monkeypatch.setattr(
        macos_disclaim,
        "_spawn_via_posix_spawn",
        lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EINVAL, "bad flags")),
    )
    monkeypatch.setattr(macos_disclaim, "_fallback_popen", lambda *a, **k: "POP")
    with caplog.at_level(logging.WARNING, logger="local_operator.macos_disclaim"):
        assert spawn_disclaimed(["/bin/true"]) == "POP"
    assert any("refused the spawn flags" in r.getMessage() for r in caplog.records)

    monkeypatch.setattr(
        macos_disclaim,
        "_spawn_via_posix_spawn",
        lambda *a, **k: (_ for _ in ()).throw(OSError(errno.ENOENT, "no binary")),
    )
    monkeypatch.setattr(macos_disclaim, "_fallback_logged", False)
    with pytest.raises(FileNotFoundError):
        spawn_disclaimed(["/does/not/exist"])


@DARWIN_ONLY
def test_real_spawn_is_disclaimed_own_session_and_carries_the_chain(tmp_path: Path) -> None:
    """The real path, exercised: a live child, its own session, the chain in env.

    This is the one test that actually calls ``posix_spawn`` with the real
    symbols; it is skipped on hosts where the lever is absent (the fallback is
    covered above) and on non-macOS kernels entirely.
    """
    if macos_disclaim._fallback_reason() is not None:
        pytest.skip("responsibility disclaim unavailable on this host")
    out = tmp_path / "out.txt"
    # The child is GATED on a marker file rather than sleeping a fixed time.
    # A fixed sleep races the box: the parent asserts ``os.getsid`` right after
    # the spawn, and a scheduling stall longer than the child's life would make
    # that probe run after the child has been reaped (observed once under fleet
    # load). A gated child stays alive until the test says go, however long the
    # parent stalls — so the only way it is ever gone is a real spawn failure.
    #
    # ``printf '%s'``, NOT ``echo``: the chain JSON carries ancestor command
    # lines verbatim, and they can contain backslashes; sh's ``echo`` processes
    # escape sequences and mangles the JSON at the first ``\\`` (a launcher
    # whose own command line carried a ``\|`` was enough — measured, char 2230).
    # The product reads this variable with ``os.environ``, never through a
    # shell; only a shell consumer needs the escape-safe spelling.
    gate = tmp_path / "release"
    with open(out, "wb") as handle:
        proc = spawn_disclaimed(
            [
                "/bin/sh",
                "-c",
                f"printf 'CHAIN=%s\\n' \"$LOP_SPAWN_CHAIN\";"
                f" while [ ! -e {shlex.quote(str(gate))} ];"
                " do sleep 0.2; done; printf 'DONE\\n'",
            ],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    try:
        assert isinstance(proc, DisclaimedProcess)
        assert proc.pid > 1
        # Own session: the whole point of the setsid flag surviving the rework.
        assert os.getsid(proc.pid) == proc.pid
        gate.write_text("go")
        assert proc.wait(timeout=30) == 0
        assert proc.poll() == 0
    finally:
        try:
            os.kill(proc.pid, 9)
        except ProcessLookupError:
            pass
    body = out.read_text()
    assert "DONE" in body
    line = next(ln for ln in body.splitlines() if ln.startswith("CHAIN="))
    chain = json.loads(line.split("=", 1)[1])
    assert isinstance(chain, list) and chain and chain[0]["pid"] == os.getpid()


def test_wait_timeout_and_signal_kill(tmp_path: Path) -> None:
    """``wait`` honors a timeout, ``kill`` delivers, ``poll`` reports signal death."""
    if macos_disclaim._fallback_reason() is not None:
        pytest.skip("responsibility disclaim unavailable on this host")
    with open(tmp_path / "kill.log", "wb") as handle:
        proc = spawn_disclaimed(["/bin/sleep", "30"], stdout=handle, stderr=handle)
    try:
        assert isinstance(proc, DisclaimedProcess)
        assert proc.poll() is None
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=0.2)
        proc.kill()
        assert proc.wait(timeout=10) == -9
        assert proc.poll() == -9
    finally:
        try:
            os.kill(proc.pid, 9)
        except ProcessLookupError:
            pass


def test_chain_facts_from_env_and_prepend_and_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_SPAWN_CHAIN, json.dumps([{"pid": 5, "argv0": "ancestor"}]))
    facts = macos_disclaim.spawn_chain_facts()
    assert facts == [{"pid": 5, "argv0": "ancestor"}]
    chain = macos_disclaim._chain_for_child()
    assert chain is not None and chain[0]["pid"] == os.getpid()
    assert {"pid": 5, "argv0": "ancestor"} in chain

    monkeypatch.delenv(ENV_SPAWN_CHAIN, raising=False)
    monkeypatch.setattr(
        macos_disclaim,
        "_walk_parent_chain",
        lambda *args, **kwargs: [{"pid": 42, "argv0": "parent"}],
    )
    walked = macos_disclaim._chain_for_child()
    assert walked == [
        {"pid": os.getpid(), "argv0": macos_disclaim._self_entry()["argv0"]},
        {"pid": 42, "argv0": "parent"},
    ]


def test_env_with_chain_drops_stale_value_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(macos_disclaim, "_chain_for_child", lambda: None)
    base = {ENV_SPAWN_CHAIN: "[garbage]", "KEEP": "1"}
    out = macos_disclaim._env_with_chain(base)
    assert ENV_SPAWN_CHAIN not in out and out["KEEP"] == "1"


def test_parse_spawn_chain_accepts_both_spellings_and_refuses_others() -> None:
    assert macos_disclaim.parse_spawn_chain('[{"pid": 5, "argv0": "x"}]') == [
        {"pid": 5, "argv0": "x"}
    ]
    assert macos_disclaim.parse_spawn_chain([{"pid": 5, "argv0": "x"}]) == [
        {"pid": 5, "argv0": "x"}
    ]
    assert macos_disclaim.parse_spawn_chain("not json") is None
    assert macos_disclaim.parse_spawn_chain(None) is None
    assert macos_disclaim.parse_spawn_chain({"pid": 5}) is None
    assert macos_disclaim.parse_spawn_chain("[1, 2]") is None


def test_snapshot_spawn_chain_marks_liveness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        ENV_SPAWN_CHAIN,
        json.dumps(
            [
                {"pid": os.getpid(), "argv0": "self"},
                {"pid": 2_147_483_600, "argv0": "/Applications/X.app/Contents/MacOS/X"},
            ]
        ),
    )
    snapshot = macos_disclaim.snapshot_spawn_chain()
    assert snapshot is not None
    assert snapshot[0]["alive"] is True
    assert snapshot[1]["alive"] is False

    monkeypatch.delenv(ENV_SPAWN_CHAIN, raising=False)
    assert macos_disclaim.snapshot_spawn_chain() is None
