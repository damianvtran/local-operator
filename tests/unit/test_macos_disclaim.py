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
import signal
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


def _expected_restored_bits() -> int:
    """The ``POSIX_SPAWN_SETSIGDEF`` mask Popen's ``restore_signals`` parity needs.

    CPython's ``_Py_RestoreSignals`` resets SIGPIPE always and SIGXFZ/SIGXFSZ
    where the platform defines them; Darwin's ``sigset_t`` is a 32-bit mask with
    bit ``signal - 1``.
    """
    bits = 0
    for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):
        signum = getattr(signal, name, None)
        if signum is not None:
            bits |= 1 << (signum - 1)
    return bits


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

    def posix_spawnattr_setsigdefault(self, attr: Any, sigset: Any) -> int:
        self.calls.append(("setsigdefault", sigset._obj.value))
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
    assert ("setflags", 0x0400 | 0x4000 | 0x0004) in fake.calls
    assert ("setdisclaim", True) in fake.calls
    assert ("setsigdefault", _expected_restored_bits()) in fake.calls
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
    ancestor = next(entry for entry in chain if entry["pid"] == 5)
    assert ancestor["argv0"] == "ancestor"
    assert ancestor["alive_at_spawn"] in (True, False, None)

    monkeypatch.delenv(ENV_SPAWN_CHAIN, raising=False)
    monkeypatch.setattr(
        macos_disclaim,
        "_walk_parent_chain",
        lambda *args, **kwargs: [{"pid": 2_147_483_600, "argv0": "parent"}],
    )
    walked = macos_disclaim._chain_for_child()
    assert walked == [
        {
            "pid": os.getpid(),
            "argv0": macos_disclaim._self_entry()["argv0"],
            "alive_at_spawn": True,
        },
        {"pid": 2_147_483_600, "argv0": "parent", "alive_at_spawn": False},
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


def test_snapshot_spawn_chain_marks_arrival_liveness_and_keeps_the_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        ENV_SPAWN_CHAIN,
        json.dumps(
            [
                {"pid": os.getpid(), "argv0": "self", "alive_at_spawn": True},
                {
                    "pid": 2_147_483_600,
                    "argv0": "/Applications/X.app/Contents/MacOS/X",
                    "alive_at_spawn": True,
                },
            ]
        ),
    )
    snapshot = macos_disclaim.snapshot_spawn_chain()
    assert snapshot is not None
    # The arrival reading is added, the recorded reading is PRESERVED: the
    # renderer's gate needs both (design round 1, D1).
    assert snapshot[0]["alive_now"] is True
    assert snapshot[0]["alive_at_spawn"] is True
    assert snapshot[1]["alive_now"] is False
    assert snapshot[1]["alive_at_spawn"] is True

    monkeypatch.delenv(ENV_SPAWN_CHAIN, raising=False)
    assert macos_disclaim.snapshot_spawn_chain() is None


def test_pid_liveness_delegates_to_procstate_and_validates_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1-1: the probe is ``procstate``'s, because ``os.kill(pid, 0)`` KILLS on Windows.

    The delegation is pinned by identity: a fake platform probe on ``procstate``
    must be what a valid pid reaches, and nothing may reach any probe for an
    argument that is not a positive int (those answer None directly). If this
    function ever re-inlines ``os.kill``, the fake is not called and this fails.
    """
    from local_operator import procstate

    seen: list[int] = []

    def fake_liveness(pid: int) -> Any:
        seen.append(pid)
        return "DELEGATED"

    monkeypatch.setattr(procstate, "pid_liveness", fake_liveness)
    assert macos_disclaim._pid_liveness(4321) == "DELEGATED"
    assert seen == [4321]
    for bad in (True, False, 0, -3, "123", None, 1.5):
        assert macos_disclaim._pid_liveness(bad) is None
    assert seen == [4321]


def test_recorded_commands_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """R1-3: a ``ps`` command column is unbounded (measured 26,921 chars); the
    recorded chain keeps a bounded prefix per entry, which the ``.app`` readers
    still match and which stops a secret-bearing command line riding whole into
    every child's environment and every boot record."""
    long_command = "x" * 40_000
    monkeypatch.setattr(macos_disclaim, "_process_table", lambda: {77: (1, long_command)})
    macos_disclaim._walk_cache.clear()
    chain = macos_disclaim._walk_parent_chain(77, cap=4)
    (bounded,) = [entry["argv0"] for entry in chain]
    # Cut to the budget AND marked as cut (R2-N3): a reader of the boot record
    # can tell "the command ended here" from "the record ended here".
    assert len(bounded) == macos_disclaim.MAX_ENTRY_CHARS
    assert bounded.endswith("…")
    assert bounded[:-1] == long_command[: macos_disclaim.MAX_ENTRY_CHARS - 1]
    # Within budget is untouched and unmarked.
    assert macos_disclaim._bound_command("x" * macos_disclaim.MAX_ENTRY_CHARS) == "x" * (
        macos_disclaim.MAX_ENTRY_CHARS
    )


def test_parse_spawn_chain_bounds_commands_and_preserves_liveness() -> None:
    raw = json.dumps(
        [{"pid": 5, "argv0": "y" * 10_000, "alive_at_spawn": True, "alive_now": False}]
    )
    assert macos_disclaim.parse_spawn_chain(raw) == [
        {
            "pid": 5,
            "argv0": "y" * (macos_disclaim.MAX_ENTRY_CHARS - 1) + "…",
            "alive_at_spawn": True,
            "alive_now": False,
        }
    ]
    # A reading that is neither a bool nor None is no evidence — normalised to
    # None, never a reason to drop an otherwise valid entry.
    assert macos_disclaim.parse_spawn_chain(
        [{"pid": 5, "argv0": "x", "alive_at_spawn": "yes"}]
    ) == [{"pid": 5, "argv0": "x", "alive_at_spawn": None}]
    assert macos_disclaim.parse_spawn_chain([{"pid": 5, "argv0": "x"}]) == [
        {"pid": 5, "argv0": "x"}
    ]


def test_chain_for_child_records_liveness_at_record_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D1: the "was alive earlier" reading is taken when the chain is written.

    Fresh for every member: an inherited reading (taken at the spawner's own
    spawn) is replaced by a probe at THIS record, because the renderer gates on
    "alive when the chain was recorded for this child".
    """
    monkeypatch.delenv(ENV_SPAWN_CHAIN, raising=False)
    monkeypatch.setattr(
        macos_disclaim,
        "_walk_parent_chain",
        lambda *args, **kwargs: [{"pid": 2_147_483_600, "argv0": "was-here"}],
    )
    chain = macos_disclaim._chain_for_child()
    assert chain is not None
    assert chain[0] == {
        "pid": os.getpid(),
        "argv0": macos_disclaim._self_entry()["argv0"],
        "alive_at_spawn": True,
    }
    assert chain[1] == {"pid": 2_147_483_600, "argv0": "was-here", "alive_at_spawn": False}

    # An inherited reading that no longer describes reality is replaced by the
    # fresh probe: this entry names THIS process (alive), recorded as dead.
    monkeypatch.setenv(
        ENV_SPAWN_CHAIN,
        json.dumps([{"pid": os.getpid(), "argv0": "anc", "alive_at_spawn": False}]),
    )
    refreshed = macos_disclaim._chain_for_child()
    assert refreshed is not None
    assert refreshed[1]["alive_at_spawn"] is True


def test_e2big_sheds_the_chain_and_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """R1-3: the chain must never fail a spawn the platform would otherwise start."""
    monkeypatch.setattr(macos_disclaim, "_fallback_reason", lambda: None)
    attempts: list[dict[str, str]] = []

    def fake_spawn(*args: Any, **kwargs: Any) -> int:
        attempts.append(dict(kwargs["env"]))
        if len(attempts) == 1:
            raise OSError(errno.E2BIG, "too big")
        return 4242

    monkeypatch.setattr(macos_disclaim, "_spawn_via_posix_spawn", fake_spawn)
    proc = spawn_disclaimed(["/bin/true"])
    assert isinstance(proc, DisclaimedProcess) and proc.pid == 4242
    assert len(attempts) == 2
    assert ENV_SPAWN_CHAIN in attempts[0] and ENV_SPAWN_CHAIN not in attempts[1]


def test_e2big_after_shedding_raises_like_popen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second E2BIG means the caller's env is over the limit — surface it."""
    monkeypatch.setattr(macos_disclaim, "_fallback_reason", lambda: None)
    calls: list[int] = []

    def always_too_big(*args: Any, **kwargs: Any) -> int:
        calls.append(1)
        raise OSError(errno.E2BIG, "too big")

    monkeypatch.setattr(macos_disclaim, "_spawn_via_posix_spawn", always_too_big)
    with pytest.raises(OSError) as excinfo:
        spawn_disclaimed(["/bin/true"])
    assert excinfo.value.errno == errno.E2BIG
    assert len(calls) == 2


def test_e2big_is_shed_on_the_popen_fallback_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """R2-1: the fallback receives the same chain-bearing env, so it gets the same shed."""
    monkeypatch.setattr(macos_disclaim, "_fallback_reason", lambda: "forced for the test")
    monkeypatch.setattr(macos_disclaim, "_log_fallback", lambda reason: None)
    attempts: list[dict[str, str]] = []
    sentinel = object()

    def fake_fallback(*args: Any, **kwargs: Any) -> Any:
        attempts.append(dict(kwargs["env"]))
        if len(attempts) == 1:
            raise OSError(errno.E2BIG, "too big")
        return sentinel

    monkeypatch.setattr(macos_disclaim, "_fallback_popen", fake_fallback)
    assert spawn_disclaimed(["/bin/true"]) is sentinel
    assert len(attempts) == 2
    assert ENV_SPAWN_CHAIN in attempts[0] and ENV_SPAWN_CHAIN not in attempts[1]

    # A second E2BIG is the caller's own env, and surfaces like Popen's would.
    attempts.clear()

    def always_too_big(*args: Any, **kwargs: Any) -> Any:
        attempts.append(dict(kwargs["env"]))
        raise OSError(errno.E2BIG, "too big")

    monkeypatch.setattr(macos_disclaim, "_fallback_popen", always_too_big)
    with pytest.raises(OSError) as excinfo:
        spawn_disclaimed(["/bin/true"])
    assert excinfo.value.errno == errno.E2BIG
    assert len(attempts) == 2


def test_chain_annotation_does_not_write_into_the_memoised_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2-3: ``alive_at_spawn`` belongs to one spawn's chain, never the lineage memo."""
    monkeypatch.delenv(ENV_SPAWN_CHAIN, raising=False)
    monkeypatch.setattr(
        macos_disclaim, "_process_table", lambda: {os.getppid(): (1, "/Applications/X.app/x")}
    )
    macos_disclaim._walk_cache.clear()
    try:
        chain = macos_disclaim._chain_for_child()
        assert chain is not None and all("alive_at_spawn" in entry for entry in chain)
        memo = macos_disclaim._walk_parent_chain(os.getppid(), cap=macos_disclaim.MAX_CHAIN - 1)
        assert memo and all("alive_at_spawn" not in entry for entry in memo)
    finally:
        macos_disclaim._walk_cache.clear()


def test_a_pid_past_pid_t_reads_as_unknown_not_as_an_exception() -> None:
    """R2-N2: ``os.kill`` raises ``OverflowError`` past 2**31-1; a corrupt env must not."""
    assert macos_disclaim._pid_liveness(2**31) is None
    assert macos_disclaim._pid_liveness(2**63) is None


def test_a_disclaimed_child_gets_popens_signal_defaults(tmp_path: Path) -> None:
    """R1-4: ``restore_signals`` parity, observed on a real child.

    Python ignores SIGPIPE at startup, and an ignored disposition survives a
    raw ``execve``. A child that has SIGPIPE ignored survives its own
    ``kill -PIPE`` and keeps going; one with the default dies of it — so ``sh``
    alone shows which dispositions the spawn restored. The disclaimed child
    must match ``Popen(restore_signals=True)``; the no-restore control proves
    the discriminator works in this environment.
    """
    if macos_disclaim._fallback_reason() is not None:
        pytest.skip("responsibility disclaim unavailable on this host")
    argv = ["/bin/sh", "-c", "kill -PIPE $$; printf SURVIVED"]

    control = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        restore_signals=False,
    )
    control_out = control.communicate()[0]
    if control.returncode != 0 or b"SURVIVED" not in control_out:
        pytest.skip("the parent does not ignore SIGPIPE here; no discrimination to make")

    reference = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        restore_signals=True,
    )
    reference_out = reference.communicate()[0]
    assert reference.returncode == -signal.SIGPIPE and b"SURVIVED" not in reference_out

    out_path = tmp_path / "sigpipe.txt"
    with open(out_path, "wb") as handle:
        proc = spawn_disclaimed(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    try:
        assert isinstance(proc, DisclaimedProcess)
        assert proc.wait(timeout=30) == -signal.SIGPIPE
    finally:
        # ``kill()`` is guarded on ``returncode``: after a successful ``wait()``
        # the pid is reaped and may already belong to someone else, so a bare
        # ``os.kill(proc.pid, …)`` here could hit an unrelated process (R2-N1).
        proc.kill()
    assert "SURVIVED" not in out_path.read_text()
