"""The runtime standby: adopted over a private inherited channel, refused when anything moved.

WHAT IS PINNED, AND HOW. ``session/runtime/standby.py`` exists because a cold
engage pays ~1.1 s of import CPU before any session work, which at this host's
load is most of a 3-4 s "starting…". Its value is speed; its RISKS are (a) that
the operator capability is handed to a process this console did not start, (b)
that a session is served by an interpreter that no longer matches what a cold
spawn would run, and (c) that the sweep cannot recognise the adopted process. So
the tests below pin, against REAL standby processes with only the warm stubbed:

* the channel: a private socketpair this process forked — no path exists to bind,
  and a same-uid impostor that binds the path the FIRST revision of this module
  used receives nothing (``test_an_impostor_listener_receives_no_capability``,
  which fails on that revision);
* the handover: the capability reaches the adopted runtime on the console's own
  descriptor, the requester's environment replaces the warmer's whole, the
  process becomes a countable runtime, and its boot record names the session
  BEFORE construction starts;
* the guards: a moved ``config.yml``, a moved loaded module, a differing
  interpreter or a differing warm-sensitive environment each refuse, and stale
  ones exit rather than waiting;
* the lifecycle: one standby per process, exit on idle, exit when the console
  goes away, and no directory or socket left anywhere;
* the loud failure: an adopted runtime whose construction raises exits NON-ZERO
  with the reason in the capture file, exactly as a forked child does.

No timing is asserted (AGENTS.md "Timing, flakes"): the saving is measured by
``scripts/bench_standby_engage.py`` and stated in the PR, never calibrated here.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from local_operator.harness import approval
from local_operator.session.runtime import reclaim, standby
from local_operator.session.runtime.types import HOST_RUN_DIRNAME, RUNTIME_MODULE

pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(socket, "send_fds"),
    reason="the standby needs SCM_RIGHTS; other platforms keep the cold spawn",
)

#: The rendezvous path the FIRST revision of this module used, written out here
#: as a literal on purpose: it is the artifact the impostor test attacks, and it
#: must not be reachable through the module any more.
RETIRED_RENDEZVOUS = ("run/standby", "standby.sock")


def test_the_module_words_are_the_same_length() -> None:
    # Adoption rewrites the module word IN PLACE, which cannot grow a string.
    assert len(standby.STANDBY_MODULE) == len(RUNTIME_MODULE)
    assert len(b"[standby]") == len(b"[session]")


def test_the_standby_argv_is_not_a_runtime_to_the_census() -> None:
    row = (
        "4242 1 01:00:00 00:00:01 Local Operator [standby] id=-------- "
        f"-P -m {standby.STANDBY_MODULE}"
    )
    assert reclaim.parse_process_row(row) is None
    adopted = row.replace("[standby]", "[session]").replace(standby.STANDBY_MODULE, RUNTIME_MODULE)
    assert reclaim.parse_process_row(adopted) is not None


def test_no_rendezvous_path_exists_in_the_module() -> None:
    """R1-1, structurally: there is nothing to bind, so nothing to authenticate.

    The first revision reached one standby per MACHINE over a path any same-uid
    process could bind first, and handed the adopter the operator capability. The
    fix is not a check on that path — a same-uid impostor's pid, environment and
    argv are all as real as the standby's, and ``secrets/peer.py`` records that
    same-uid peers are identifiable only by LINEAGE. The fix is that no such path
    exists: the channel is a socketpair this process inherited from its own fork.
    """
    import ast

    source = Path(standby.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    # CODE only: the docstring NAMES the retired path to explain why it is gone,
    # and a prose mention is the opposite of a bindable socket.
    docstrings = {
        node.body[0].value.value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    code = "\n".join(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    )
    assert "AF_UNIX" in source, "the channel should still be a POSIX socketpair"
    # The four spellings of the retired design, each of which is what an impostor
    # needed: a socket path, the directory it lived in, the helper that built it,
    # and the lock that arbitrated who owned it.
    for gone in ("standby.sock", "STANDBY_DIRNAME", "socket_path", "LOCK_EX", "gettempdir"):
        assert gone not in code, f"{gone!r} is back in the standby: a path an impostor can bind"
    # ...and nothing on disk either: no module-level function that would build one.
    assert not hasattr(standby, "socket_path")
    assert not hasattr(standby, "standby_dir")


def test_the_suite_runs_with_standbys_disabled() -> None:
    """The conftest gate: ``cli.main()`` for ``serve``/the TUI is driven all over
    the suite, and each would otherwise leave a warmed interpreter behind it."""
    assert standby.disabled()


def test_warming_is_off_unless_a_host_enables_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spawned: list[Any] = []
    monkeypatch.delenv(standby.DISABLE_ENV, raising=False)
    monkeypatch.setattr(standby, "_WARMING", [False])
    monkeypatch.setattr(standby, "_spawn_standby", lambda *a: spawned.append(a))
    # The warm is for THIS console's own root, so the probe engine is told the
    # console's root is the temporary one.
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)
    standby.ensure_warm(tmp_path, sys.executable)
    assert spawned == []
    # And a runtime child never warms, even in a process that enabled it: a
    # runtime that warmed spares would make every session a warmer.
    monkeypatch.setattr(standby, "_WARMING", [True])
    monkeypatch.setenv("LOP_MOBILE_CHILD_RESUME", "abc")
    standby.ensure_warm(tmp_path, sys.executable)
    assert spawned == []
    monkeypatch.delenv("LOP_MOBILE_CHILD_RESUME")
    standby.ensure_warm(tmp_path, sys.executable)
    assert len(spawned) == 1
    # ...and a root this console does not serve is never warmed, because the
    # child would resolve the console's own root from its inherited environment.
    standby.ensure_warm(tmp_path / "elsewhere", sys.executable)
    assert len(spawned) == 1


def test_a_root_that_is_not_a_real_path_is_never_warmed(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    monkeypatch.delenv(standby.DISABLE_ENV, raising=False)
    monkeypatch.setattr(standby, "_WARMING", [False])
    spawned: list[Any] = []
    monkeypatch.setattr(standby, "warm_in_background", lambda *a: spawned.append(a))
    standby.enable_warming(MagicMock())
    standby.enable_warming(Path("relative/root"))
    assert spawned == [] and standby._WARMING == [False]


def test_one_standby_per_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A second warm while one is alive is refused: this process holds at most one."""
    monkeypatch.delenv(standby.DISABLE_ENV, raising=False)
    monkeypatch.setattr(standby, "_WARMING", [True])

    class _Child:
        pid = 4321

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr(standby.subprocess, "Popen", lambda *a, **k: _Child())
    standby.reset_for_tests()
    try:
        first = standby._spawn_standby(tmp_path, sys.executable)
        monkeypatch.setattr(standby, "_WARM", [first])
        spawned: list[Any] = []
        monkeypatch.setattr(standby, "_spawn_standby", lambda *a: spawned.append(a))
        standby.ensure_warm(tmp_path, sys.executable)
        assert spawned == []
    finally:
        standby.reset_for_tests()


def test_the_disable_switch_turns_off_adoption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(standby.DISABLE_ENV, "1")
    assert standby.try_adopt(tmp_path, sys.executable, {}, tmp_path / "c", None) is None
    assert standby.adoption_possible() is False


def test_no_standby_means_the_ordinary_cold_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``try_adopt`` answering None must leave ``_spawn_runtime`` exactly as before."""
    from local_operator.session.runtime import launch as launch_module

    calls: list[Any] = []
    monkeypatch.setattr(standby, "try_adopt", lambda *a, **k: calls.append(a) or None)

    class _Child:
        pid = 99999

        def poll(self) -> None:
            return None

    monkeypatch.delenv(standby.DISABLE_ENV, raising=False)
    monkeypatch.setattr(launch_module.subprocess, "Popen", lambda argv, **kw: _Child())
    monkeypatch.setattr(launch_module, "open_operator_cap_handoff", _FakeHandoff)
    process = launch_module._spawn_runtime("sess-cold01", str(tmp_path), defer_materialise=True)
    capture = getattr(process, "lop_capture_path", None)
    if capture is not None:
        capture.unlink(missing_ok=True)
    assert isinstance(process, _Child)
    # No standby was available, so the engage never even built an adoption
    # handoff: ``adoption_possible`` answered False and try_adopt was not called.
    assert calls == []


class _FakeHandoff:
    """A REAL handoff, so the theft test can tell a refusal from a failure.

    ``OperatorCapHandoff``'s essential part is a live descriptor pair: the child
    end rides ``pass_fds`` into the spawn and the capability value is written into
    the OTHER end once the child exists. A double holding a made-up fd number (7)
    would make the first revision's ``send_fds`` fail with ``EBADF``, its adoption
    fall back to a cold spawn, and this whole test pass against the very revision
    it exists to reproduce — measured, and the reason these are real sockets.
    """

    def __init__(self) -> None:
        self.parent, self.child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.argv = ["--operator-fd", str(self.child.fileno())]
        self.pass_fds = (self.child.fileno(),)
        self.close_fds = True
        self.delivered: list[bytes] = []
        self.closed = False

    def deliver(self, cap: bytes) -> None:
        self.delivered.append(cap)
        try:
            self.parent.sendall(cap)
        finally:
            self.close()

    def close(self) -> None:
        self.closed = True
        self.parent.close()
        self.child.close()


# ---------------------------------------------------------------------------
# The impostor: the R1-1 reproduction, and it must fail against the first revision
# ---------------------------------------------------------------------------

#: Binds the retired rendezvous path, waits for a connection and reports what it
#: got. A console that trusts that path hands it the capability descriptor.
#: Binds EVERY path the first revision could have reached and reports what it got.
#:
#: Two paths, because that revision had two: the natural one under the root, and
#: a ``$TMPDIR/lop-standby-<uid>-<digest>`` fallback it used whenever the natural
#: path was too long for ``sun_path`` (104 bytes on macOS). EVERY ``tmp_path`` in
#: this suite is long enough to take the fallback, which is why an impostor
#: listening only on the natural path would see nothing and this test would pass
#: against the very revision it exists to reproduce — measured.
#: Binds the rendezvous path THE FIRST REVISION WOULD HAVE COMPUTED, and reports
#: what it got. That rule had two candidates: the natural path under the root, and
#: a ``$TMPDIR/lop-standby-<uid>-<digest>`` fallback chosen whenever the natural one
#: was too long for ``sun_path`` (104 bytes on macOS). Getting this wrong makes the
#: test pass for the wrong reason in both directions — an impostor listening only
#: on the natural path sees nothing under a long pytest ``tmp_path`` (measured: the
#: bind fails with ``AF_UNIX path too long``), and one listening only on the
#: fallback misses the root the engage actually uses.
_IMPOSTOR = textwrap.dedent("""
    import hashlib, json, os, select, socket, sys, tempfile
    from pathlib import Path
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    natural_dir = root / "run" / "standby"
    natural_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(natural_dir).encode("utf-8")).hexdigest()[:12]
    uid = os.getuid() if hasattr(os, "getuid") else 0
    fallback_dir = Path(tempfile.gettempdir()) / ("lop-standby-%d-%s" % (uid, digest))
    natural = natural_dir / "standby.sock"
    # The retired rule, verbatim: 103 is the longest ``sun_path`` payload the
    # first revision would use (macOS allows 104 bytes including the NUL).
    path = natural if len(str(natural)) <= 103 else fallback_dir / "standby.sock"
    if path.parent != natural_dir:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
    if path.exists():
        path.unlink()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    os.chmod(path, 0o600)
    listener.listen(1)
    report = {"bound": str(path), "connections": 0, "cap_len": 0}
    # Ready to be contacted; the marker goes out BEFORE the wait, so the test
    # knows the listener is bound rather than waiting out its own deadline.
    Path(sys.argv[3]).write_text("ready")
    ready, _w, _x = select.select([listener], [], [], 25.0)
    if ready:
        conn, _ = listener.accept()
        with conn:
            report["connections"] = 1
            conn.settimeout(5.0)
            try:
                _marker, fds, _flags, _address = socket.recv_fds(conn, 1, 1)
                if fds:
                    report["cap_len"] = len(os.read(fds[0], 64))
            except OSError:
                pass
            payload = json.dumps({"ok": True, "pid": os.getpid()}).encode()
            try:
                conn.sendall(len(payload).to_bytes(4, "big") + payload)
            except OSError:
                pass
    listener.close()
    # Clean up after ourselves, including the fallback directory when this rig had
    # to create it: the retired design leaked one of these per root (R1-5), and a
    # test that reproduces the leak must not add to it.
    try:
        path.unlink()
    except OSError:
        pass
    if path.parent != natural_dir:
        try:
            path.parent.rmdir()
        except OSError:
            pass
    out.write_text(json.dumps(report))
    """)


def test_an_impostor_listener_receives_no_capability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R1-1: a same-uid listener on the retired path must get nothing.

    Against the first revision this test FAILS: ``launch._spawn_runtime`` connects
    to whatever is listening, sends it the child end of the capability handoff and
    calls ``remember_operator_cap`` with the LISTENER'S pid — so the listener holds
    the value the console will accept for authority-increasing requests, and the
    console has bound that value to it (issue #1310). Here the same attempt is
    made against the current module and the listener must come away with neither.
    """
    from local_operator.session.runtime import launch as launch_module

    root = tmp_path / "cfg"
    root.mkdir()
    report = tmp_path / "impostor.json"
    ready = tmp_path / "impostor-ready"
    errors = tmp_path / "impostor.err"
    impostor = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        [sys.executable, "-c", _IMPOSTOR, str(root), str(report), str(ready)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        # Kept, not discarded: the only thing that makes an early exit from this
        # rig diagnosable is its own traceback.
        stderr=errors.open("wb"),
    )
    try:
        deadline = time.monotonic() + 20
        while not ready.exists():
            why = errors.read_text() if errors.exists() else ""
            assert impostor.poll() is None, f"the impostor could not bind: {why}"
            assert time.monotonic() < deadline, f"the impostor never bound: {why}"
            time.sleep(0.02)

        approval.reset_operator_caps_for_tests()
        monkeypatch.delenv(standby.DISABLE_ENV, raising=False)
        # THE ROOT THE ENGAGE USES is the one in the PROCESS environment: this is
        # the isolated root a real engage would carry, and it is what decides which
        # rendezvous path the first revision would have consulted. Without it the
        # attempt goes looking under the suite's own config dir, finds nothing and
        # the test passes for the wrong reason — measured.
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
        # ``raising=False``: the first revision had no per-process warm state to
        # clear (it discovered a socket by path), and this test must RUN against
        # that revision rather than error out in setup — its assertions are the
        # reproduction.
        monkeypatch.setattr(standby, "_WARM", [None], raising=False)
        handoffs: list[_FakeHandoff] = []

        def fake_handoff() -> _FakeHandoff:
            made = _FakeHandoff()
            handoffs.append(made)
            return made

        class _Child:
            pid = 99999

            def poll(self) -> None:
                return None

        monkeypatch.setattr(launch_module, "open_operator_cap_handoff", fake_handoff)
        monkeypatch.setattr(launch_module.subprocess, "Popen", lambda argv, **kw: _Child())
        process = launch_module._spawn_runtime("sess-thief1", str(root), defer_materialise=True)
        capture = getattr(process, "lop_capture_path", None)
        if capture is not None:
            capture.unlink(missing_ok=True)

        # The listener's own account: it was never connected to, so nothing of
        # this session's spawn — least of all a descriptor — reached it.
        assert impostor.wait(timeout=40) == 0
        seen = json.loads(report.read_text())
        # The retired path was bound and never reached: nothing of this session's
        # spawn, least of all a descriptor, travelled anywhere.
        assert seen["bound"].endswith("standby.sock"), seen
        assert seen["connections"] == 0, seen
        assert seen["cap_len"] == 0, seen
        # And the console never bound a capability to it. This is the assertion
        # that fails on the first revision, where the listener's pid was recorded.
        assert approval.operator_cap_for(impostor.pid) is None
        assert approval.operator_cap_for(99999) is not None, "the cold child got no capability"
        assert isinstance(process, _Child)
    finally:
        if impostor.poll() is None:
            impostor.kill()
            impostor.wait(timeout=10)
        approval.reset_operator_caps_for_tests()


def test_a_failed_adoption_does_not_reuse_the_handoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R1-4: the cold fallback must get a FRESH handoff, not the attempted one.

    If it reused it, two processes would hold one socketpair: whichever reads the
    32 bytes first wins, lease arbitration then decides which SURVIVES, and the
    survivor can be the one without the capability — an authority-increasing
    request would be refused for the whole session.
    """
    from local_operator.session.runtime import launch as launch_module

    monkeypatch.delenv(standby.DISABLE_ENV, raising=False)
    monkeypatch.setattr(standby, "adoption_possible", lambda: True)
    monkeypatch.setattr(standby, "try_adopt", lambda *a, **k: None)
    monkeypatch.setattr(standby, "reset_for_tests", lambda: None)
    approval.reset_operator_caps_for_tests()

    handoffs: list[_FakeHandoff] = []

    def fake_handoff() -> _FakeHandoff:
        made = _FakeHandoff()
        handoffs.append(made)
        return made

    passes: dict[str, Any] = {}

    class _Child:
        pid = 99998

        def poll(self) -> None:
            return None

    def fake_popen(argv: list[str], **kwargs: Any) -> _Child:
        passes["pass_fds"] = kwargs.get("pass_fds")
        return _Child()

    monkeypatch.setattr(launch_module, "open_operator_cap_handoff", fake_handoff)
    monkeypatch.setattr(launch_module.subprocess, "Popen", fake_popen)
    process = launch_module._spawn_runtime("sess-fresh1", str(tmp_path), defer_materialise=True)
    capture = getattr(process, "lop_capture_path", None)
    if capture is not None:
        capture.unlink(missing_ok=True)
    try:
        # Two handoffs were built and populated exactly once each: the attempt's
        # was closed WITHOUT the capability in it, the cold path's got it.
        assert len(handoffs) == 2
        attempted, used = handoffs
        assert attempted.delivered == [] and attempted.closed
        assert used.delivered and used.closed
        assert passes["pass_fds"] == used.pass_fds
        assert approval.operator_cap_for(99998) is not None
    finally:
        approval.reset_operator_caps_for_tests()


# ---------------------------------------------------------------------------
# A real standby process, with the warm stubbed
# ---------------------------------------------------------------------------

#: Runs the real ``standby.main()`` with ``_warm`` replaced and the scheduler band
#: left alone, so everything between the inherited descriptor and the runtime
#: entry point is production code. ``--probe`` makes it report what the adopted
#: process actually holds instead of running the runtime; without it the REAL
#: ``_become_runtime`` runs, which is what the loud-failure test needs.
_STANDBY_DRIVER = textwrap.dedent("""
    import json, os, subprocess, sys
    from pathlib import Path
    from local_operator.session.runtime import standby
    standby._warm = lambda: None
    # The band is the host scheduler's business, not this test's: at load 100+ a
    # background-band process is starved for minutes, which is weather, not logic.
    standby._background_priority = lambda on: None
    if "--probe" in sys.argv:
        out = Path(os.environ["STANDBY_PROBE_OUT"])
        def become(fd):
            cap = b""
            if fd is not None and fd >= 0:
                cap = os.read(fd, 64)
                os.close(fd)
            ps = subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(os.getpid())],
                                capture_output=True, text=True).stdout.strip()
            out.write_text(json.dumps({
                "pid": os.getpid(), "cap": cap.hex(), "cwd": os.getcwd(), "ps": ps,
                "env": {k: v for k, v in os.environ.items() if k.startswith(("LOP_", "PROBE_"))},
                "boots": sorted(
                    p.name
                    for p in Path(
                        os.environ["LOCAL_OPERATOR_CONFIG_DIR"]
                    ).glob("run/host/*.json")
                ),
            }))
            return 0
        standby._become_runtime = become
    sys.exit(standby.main())
    """)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tmp_path / "cfg"
    config.mkdir()
    (config / "config.yml").write_text("values: {}\n", encoding="utf-8")
    monkeypatch.delenv(standby.DISABLE_ENV, raising=False)
    return config


class _Standby:
    """One driver process plus the console's end of its private channel."""

    def __init__(self, proc: subprocess.Popen[bytes], sock: socket.socket, root: Path) -> None:
        self.proc = proc
        self.sock = sock
        self.root = root

    @property
    def pid(self) -> int:
        return self.proc.pid

    def ready(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        self.sock.settimeout(0.1)
        while time.monotonic() < deadline:
            assert self.proc.poll() is None, f"standby exited early rc={self.proc.returncode}"
            try:
                byte = self.sock.recv(1)
            except (TimeoutError, socket.timeout):
                continue
            if byte == standby._READY:
                return
            raise AssertionError(f"standby reported {byte!r}")
        raise AssertionError("standby never warmed")

    def request(self, **overrides: Any) -> dict[str, Any] | None:
        """One adoption attempt, framed exactly as ``try_adopt`` frames it."""
        (self.root / "capture.log").touch()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_", "HERDR_"))}
        # The root, exactly as an isolated production engage carries it: the
        # child's identity comes from this environment (``paths.config_dir``).
        env["LOCAL_OPERATOR_CONFIG_DIR"] = str(self.root)
        env.update(
            {
                "LOP_MOBILE_CHILD_RESUME": "abcdef123456",
                "LOP_MOBILE_CHILD_CWD": str(self.root),
                "PROBE_ONLY_IN_REQUESTER": "yes",
            }
        )
        env.update(overrides.pop("env", {}))
        payload = {
            "op": "adopt",
            "interpreter": str(overrides.pop("interpreter", sys.executable)),
            "root": str(overrides.pop("root", self.root)),
            "env": env,
            "cwd": str(self.root),
            "capture": str(overrides.pop("capture", self.root / "capture.log")),
            "has_cap_fd": bool(overrides.pop("has_cap_fd", True)),
        }
        body = json.dumps(payload).encode()
        payload_bytes = len(body).to_bytes(4, "big") + body
        self.sock.settimeout(5.0)
        # ONE message, exactly as ``try_adopt`` sends it: the frame and the
        # descriptor together, so the child's first ``recv_fds`` gets both.
        socket.send_fds(self.sock, [payload_bytes], [overrides.pop("cap_fd", self._cap_fd[0])])
        try:
            return _read_frame(self.sock)
        except (TimeoutError, socket.timeout):
            return None

    #: The child end of this standby's own capability handoff, replaced per use.
    _cap_fd: tuple[int, int] = (0, 0)


def _read_frame(sock: socket.socket) -> dict[str, Any]:
    """One length-prefixed reply. EOF raises, so a dead child cannot spin this."""
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("the standby closed the channel")
        header += chunk
    size = int.from_bytes(header, "big")
    body = b""
    while len(body) < size:
        chunk = sock.recv(size - len(body))
        if not chunk:
            raise ConnectionError("the standby closed the channel")
        body += chunk
    return json.loads(body.decode())


@pytest.fixture
def started(tmp_path: Path) -> Iterator[dict[str, Any]]:
    """Track every driver this test starts, and reap them by exact pid."""
    made: list[_Standby] = []
    yield {"made": made}
    for item in made:
        if item.proc.poll() is None:
            item.proc.send_signal(signal.SIGTERM)
            try:
                item.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                item.proc.kill()
                item.proc.wait(timeout=10)
        item.sock.close()


def _base_env(root: Path, out: Path) -> dict[str, str]:
    return {
        **{k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_", "HERDR_"))},
        "LOCAL_OPERATOR_CONFIG_DIR": str(root),
        "LOCAL_OPERATOR_NO_NOTIFICATIONS": "1",
        "LOCAL_OPERATOR_NO_DESKTOP_LAUNCH": "1",
        "STANDBY_PROBE_OUT": str(out),
    }


def _start(
    root: Path,
    tmp_path: Path,
    started: dict[str, Any],
    *,
    probe: bool = True,
    label: str = "[standby]",
) -> _Standby:
    """Fork a driver the way ``_spawn_standby`` does: private socketpair, inherited."""
    console_end, child_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    argv0 = f"Local Operator {label} id=--------"
    argv = [argv0, "-P", "-c", _STANDBY_DRIVER, "--standby-fd", str(child_end.fileno())]
    if probe:
        argv.append("--probe")
    # The module word production passes with ``-m``, so the adoption rename
    # (in-place, same length) has something to find. The driver's own code has
    # already been handed over by ``-c``; these two words are argv shape only.
    argv += ["-m", standby.STANDBY_MODULE]
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        argv,
        executable=sys.executable,
        env=_base_env(root, tmp_path / "probe.json"),
        pass_fds=(child_end.fileno(),),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    child_end.close()
    item = _Standby(proc, console_end, root)
    # A fresh capability handoff for this standby, whose child end is sent by
    # SCM_RIGHTS on each request — exactly what the spawn path passes.
    item._handoff = _new_handoff()
    item._cap_fd = (item._handoff[0].fileno(), item._handoff[1].fileno())
    started["made"].append(item)
    return item


def _new_handoff() -> tuple[socket.socket, socket.socket]:
    """One capability handoff. The SOCKETS are returned, not their numbers: a
    fileno whose socket has been collected is a closed descriptor (EBADF)."""
    return socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)


def _wait_json(path: Path, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while not path.exists() or not path.read_text():
        assert time.monotonic() < deadline, "adopted process never reported"
        time.sleep(0.02)
    return json.loads(path.read_text())


def test_adoption_hands_over_the_whole_spawn(
    root: Path, tmp_path: Path, started: dict[str, Any]
) -> None:
    out = tmp_path / "probe.json"
    item = _start(root, tmp_path, started)
    # A fresh socketpair standing in for the console's capability handoff.
    reader, writer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    item._cap_fd = (reader.fileno(), writer.fileno())
    item.ready()

    # INVISIBLE before adoption: no census row, no record, no boot record.
    assert item.pid not in {p.pid for p in reclaim.runtime_processes()}
    assert not (root / HOST_RUN_DIRNAME).exists()
    assert not (root / "sessions").exists()

    reply = item.request()
    writer.sendall(b"\x07" * 32)
    writer.close()
    assert reply is not None and reply["ok"] is True and reply["pid"] == item.pid
    report = _wait_json(out)

    # The capability travelled on the passed descriptor, byte for byte.
    assert report["cap"] == ("07" * 32)
    # The requester's environment replaced the warmer's, WHOLE.
    assert report["env"]["LOP_MOBILE_CHILD_RESUME"] == "abcdef123456"
    assert report["env"]["PROBE_ONLY_IN_REQUESTER"] == "yes"
    # ...and ps now names a runtime of that session, which the census counts.
    assert f"-m {RUNTIME_MODULE}" in report["ps"]
    assert "[session] id=abcdef12" in report["ps"]
    assert reclaim.parse_process_row(f"1 1 01:00 00:01 {report['ps']}") is not None
    # R1-3: the boot record exists BEFORE construction, so the sweep can date and
    # attribute this process from the instant it starts building a session.
    # The boot record is named by PID (``registry.publish``), and it is written
    # BEFORE construction: that is what gives the sweep a session id and a fresh
    # age for a process whose interpreter is minutes older than its session.
    assert f"{item.pid}.json" in report["boots"], report["boots"]
    fleet = reclaim.read_fleet(root)
    assert fleet.boots[item.pid]["session_id"] == "abcdef123456"
    assert fleet.boot_starts[item.pid] > 0
    assert item.proc.wait(timeout=30) == 0


def test_the_young_rung_uses_the_boot_record() -> None:
    """R1-3, the unit half: an old process with a young boot record is YOUNG.

    An adopted runtime's interpreter started up to ``IDLE_REAP_S`` before the
    session it constructs, so its ``ps`` age says "long-lived" while its session is
    seconds old — and the young rung is the one rung that covers the whole
    construction window. Without this, a sweep could signal a runtime that has not
    published yet.
    """
    from local_operator.session.runtime.reclaim import (
        Fleet,
        RuntimeProcess,
        SocketEvidence,
        effective_age,
        verdict,
    )

    now = 1_800_000_000.0
    process = RuntimeProcess(
        pid=4242,
        parent_pid=1,
        age_s=900.0,
        cpu_s=0.0,
        command=f"/usr/bin/python3 -P -m {RUNTIME_MODULE}",
    )
    adopted = Fleet(
        root=Path("/nonexistent-root"),
        records={},
        boots={4242: {"pid": 4242, "session_id": "s1"}},
        viewers=[],
        sockets=SocketEvidence(),
        boot_starts={4242: now - 1.0},
    )
    assert effective_age(process, adopted, now=now) == 1.0
    decision = verdict(process, adopted, min_age_s=300.0, now=now)
    assert decision.refusal == reclaim.REFUSAL_YOUNG, decision
    # A forked runtime has no boot record yet at this point, and keeps the reading
    # the process table gives.
    forked = Fleet(
        root=Path("/nonexistent-root"),
        records={},
        boots={},
        viewers=[],
        sockets=SocketEvidence(),
    )
    assert effective_age(process, forked, now=now) == 900.0
    assert verdict(process, forked, min_age_s=300.0, now=now).refusal != reclaim.REFUSAL_YOUNG


def test_the_standby_exits_when_its_console_goes_away(
    root: Path, tmp_path: Path, started: dict[str, Any]
) -> None:
    """R1-6: the channel's EOF is the console's death, and it is immediate.

    A standby is detached by design (it must outlive a terminal teardown once it
    is a runtime), so nothing else would end it: without this, a rig that keeps its
    root and its host would keep ~140 MB for the whole idle window.
    """
    item = _start(root, tmp_path, started)
    item.ready()
    assert item.proc.poll() is None
    item.sock.close()
    assert item.proc.wait(timeout=30) == 0


def test_an_adopted_runtime_fails_loudly(
    root: Path, tmp_path: Path, started: dict[str, Any]
) -> None:
    """R1-2: a construction failure in an ADOPTED runtime is as loud as a forked one.

    Both routes are driven against the same unconfigured root (no provider), and
    both must exit non-zero with the actionable reason in the capture file that
    ``engage_runtime`` reads. The first revision caught the exception, logged it at
    DEBUG and returned 0 with an empty capture, so the user lost the "connect a
    provider" message whenever the candidate that died had been adopted.
    """
    from local_operator.interpreter import SAFE_PATH_FLAG

    def run(capture: Path, adopted: bool) -> tuple[int, str]:
        handle = capture.open("wb")
        if not adopted:
            proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
                [sys.executable, SAFE_PATH_FLAG, "-m", RUNTIME_MODULE],
                env={
                    **_base_env(root, tmp_path / "probe.json"),
                    "LOP_MOBILE_CHILD_RESUME": "cold00000001",
                    "LOP_MOBILE_CHILD_CWD": str(root),
                },
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            code = proc.wait(timeout=180)
        else:
            item = _start(root, tmp_path, started, probe=False)
            reader, writer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            item._cap_fd = (reader.fileno(), writer.fileno())
            item.ready()
            reply = item.request(capture=str(capture))
            assert reply is not None and reply["ok"] is True
            writer.sendall(b"\x07" * 32)
            writer.close()
            code = item.proc.wait(timeout=180)
            item.sock.close()
        handle.close()
        return code, capture.read_text(errors="replace")

    cold = tmp_path / "cold.log"
    adopted = tmp_path / "adopted.log"
    cold_code, cold_text = run(cold, adopted=False)
    adopted_code, adopted_text = run(adopted, adopted=True)

    from local_operator.session.runtime.launch import _spawn_failure_reason

    assert cold_code != 0, "the forked child is expected to fail on an unconfigured root"
    assert adopted_code == cold_code, (adopted_code, cold_code)
    assert cold_text.strip(), "the forked child's reason should be in its capture"
    assert adopted_text.strip(), "an adopted failure left an empty capture (R1-2)"
    # THE PROPERTY THAT MATTERS is not "some text is there" but that the console
    # reads the SAME actionable sentence out of both captures: this is the reader
    # ``engage_runtime`` uses to tell the user "connect a provider" rather than
    # retrying a generic failure.
    cold_line, cold_action = _spawn_failure_reason(cold)
    adopted_line, adopted_action = _spawn_failure_reason(adopted)
    assert adopted_line == cold_line, (adopted_line, cold_line)
    assert "HostingNotConfiguredError" in adopted_line
    assert adopted_action == cold_action != "", (adopted_action, cold_action)


def test_the_rendezvous_directory_is_never_created(
    root: Path, tmp_path: Path, started: dict[str, Any]
) -> None:
    """R1-5, and the reason it is structural: no path exists to leave behind.

    The first revision put a fallback rendezvous directory under ``$TMPDIR``
    (``lop-standby-<uid>-<digest>``) for any root too deep for ``sun_path`` — every
    ``tmp_path`` on this machine — unlinked the socket but never the directory, and
    left one per root.
    """
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    before = sorted(p.name for p in tmpdir.iterdir())
    item = _start(root, tmp_path, started)
    item.ready()
    reader, writer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    item._cap_fd = (reader.fileno(), writer.fileno())
    assert item.request() is not None
    writer.sendall(b"\x07" * 32)
    writer.close()
    item.proc.wait(timeout=30)
    assert sorted(p.name for p in tmpdir.iterdir()) == before == []
    # ``run/host`` is the boot record this adoption wrote on purpose (R1-3); what
    # must not exist is the retired rendezvous directory beside it.
    assert not (root / "run" / RETIRED_RENDEZVOUS[0]).exists()
    if (root / "run").exists():
        assert sorted(p.name for p in (root / "run").iterdir()) == ["host"]


@pytest.mark.parametrize("what", ["config", "tree"])
def test_a_moved_input_refuses_and_retires_the_standby(
    what: str, root: Path, tmp_path: Path, started: dict[str, Any]
) -> None:
    item = _start(root, tmp_path, started)
    item.ready()
    target = Path(standby.__file__)
    st = target.stat()
    if what == "config":
        # The product's own writer shape: a replace, i.e. a new inode.
        fresh = root / "config.yml.new"
        fresh.write_text("values: {hosting: other}\n", encoding="utf-8")
        os.replace(fresh, root / "config.yml")
    else:
        # A loaded module of THIS tree moved: bump the mtime of a file the
        # standby imported (``standby.py`` itself), then put it back.
        os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    try:
        reply = item.request()
        assert reply is not None and reply["ok"] is False, reply
        # Stale, so it leaves rather than waiting: nothing will ever adopt it.
        assert item.proc.wait(timeout=30) == 0
    finally:
        if what == "tree":
            os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_a_differing_requester_is_declined_and_the_standby_keeps_waiting(
    root: Path, tmp_path: Path, started: dict[str, Any]
) -> None:
    item = _start(root, tmp_path, started)
    item.ready()
    reply = item.request(env={"LANG": "xx_XX.WEIRD-8"})
    assert reply is not None and reply["ok"] is False, reply
    other_venv = tmp_path / "other" / "bin" / "python3"
    reply = item.request(interpreter=str(other_venv))
    assert reply is not None and reply["ok"] is False, reply
    # Declined, not retired: it still serves a matching requester.
    assert item.proc.poll() is None
