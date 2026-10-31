"""The session host (design-surface-independence slice 2): rules R3 and R4, and the relay.

WHAT THIS FILE PROVES, by execution rather than by reading:

* **R4 — a host-absent engage never forks locally.** ``engage_runtime`` is driven
  through the real engage loop and the real ``_spawn_runtime`` dispatcher with
  the host route selected, behind a refusing double at each rung of the cold-start
  ladder (installer refuses, platform unsupported, nobody listening, host answers
  "no"). Every spawn primitive — ``subprocess.Popen`` itself, ``spawn_disclaimed``
  at both import sites, and the local route — is replaced by a counter that FAILS
  the test if reached. The assertion is a keyed ``RuntimeStartupError`` and a
  count of zero. :func:`test_the_r4_check_goes_red_when_a_local_fallback_is_added`
  runs the SAME check against a mutant dispatcher that falls back, and requires it
  to fail: a passing test proves nothing until it can fail.
* **The capability relay (§4).** A real ``SessionHost`` on a real unix socket
  receives the child end of a socketpair by ``SCM_RIGHTS``; the value the surface
  later writes into the parent end is readable only through the descriptor the
  host handed on, and the host's own copy is closed before the request returns.
* **No double spawn (§3).** N concurrent requests for one session produce exactly
  one fork and N identical pids.
* **R3 (unit half).** The host's exit path signals nothing, and the module has no
  stop vocabulary at all (AST pin). The scripted half — a real host, a real
  runtime, SIGTERM the host — is ``tests/e2e/test_session_host_e2e.py``.

The launchd wiring is deliberately not driven here: it addresses the operator's
real ``gui/<uid>`` domain, which a test must never do (``launchd.is_own_plist``).
"""

from __future__ import annotations

import ast
import asyncio
import fcntl
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator import macos_disclaim, supervisors
from local_operator.harness.approval import mint_operator_cap, open_operator_cap_handoff
from local_operator.session.runtime import (
    host_install,
    host_service,
    launch,
    session_host,
)
from local_operator.session.runtime.launch import (
    RuntimeStartupError,
    WarmErrand,
    engage_runtime,
)

ROOT = Path(__file__).resolve().parents[4]
RUNTIME_DIR = ROOT / "local_operator" / "session" / "runtime"


# ---------------------------------------------------------------------------
# A host on a real socket, with a stand-in fork
# ---------------------------------------------------------------------------


class _Child:
    """Minimal ``Popen`` stand-in: a pid and a ``wait`` the host's reaper can park on."""

    def __init__(self, pid: int, *, exits: threading.Event | None = None, code: int = 0) -> None:
        self.pid = pid
        self._exits = exits
        self._code = code

    def wait(self, timeout: float | None = None) -> int:
        if self._exits is None:
            threading.Event().wait(timeout)
            raise subprocess.TimeoutExpired("child", timeout or 0)
        if not self._exits.wait(timeout):
            raise subprocess.TimeoutExpired("child", timeout or 0)
        return self._code


class _Rig:
    def __init__(self, root: Path, host: host_service.SessionHost, stop: threading.Event) -> None:
        self.root, self.host, self.stop = root, host, stop
        self.thread: threading.Thread | None = None


@pytest.fixture
def make_host(tmp_path: Path) -> Iterator[Callable[..., _Rig]]:
    rigs: list[_Rig] = []

    def build(spawner: Any = None) -> _Rig:
        root = tmp_path / f"store{len(rigs)}"
        root.mkdir()
        stop = threading.Event()
        kwargs = {"spawner": spawner} if spawner is not None else {}
        handler = host_service.SessionHost(root, **kwargs)
        rig = _Rig(root, handler, stop)
        rig.thread = threading.Thread(
            target=host_service.serve, args=(root, stop), kwargs={"host": handler}, daemon=True
        )
        rig.thread.start()
        deadline = time.monotonic() + 10
        while not session_host.ping(root, timeout_s=0.5):
            assert time.monotonic() < deadline, "the host never started answering"
            time.sleep(0.05)
        rigs.append(rig)
        return rig

    yield build
    for rig in rigs:
        rig.stop.set()
        assert rig.thread is not None
        rig.thread.join(timeout=10)


def _capture(tmp_path: Path) -> Path:
    path = tmp_path / f"capture-{time.monotonic_ns()}.log"
    path.write_bytes(b"")
    path.chmod(0o600)
    return path


def _ask(rig: _Rig, tmp_path: Path, session_id: str = "sess00000001", **kw: Any) -> Any:
    handoff = open_operator_cap_handoff()
    try:
        return session_host.request_spawn(
            rig.root,
            session_id=session_id,
            cwd=str(tmp_path),
            env={"PATH": "/usr/bin"},
            interpreter=sys.executable,
            capture=_capture(tmp_path),
            cap_fd=handoff.pass_fds[0] if handoff.pass_fds else None,
            **kw,
        )
    finally:
        handoff.close()


# ---------------------------------------------------------------------------
# R4 — a host-absent engage never forks locally
# ---------------------------------------------------------------------------


class _Forks:
    """Counts every way a runtime could be started, and fails the test if one is."""

    def __init__(self) -> None:
        self.popen: list[Any] = []
        self.disclaimed: list[Any] = []
        self.local: list[Any] = []

    @property
    def total(self) -> int:
        return len(self.popen) + len(self.disclaimed) + len(self.local)


def _arm_fork_counters(monkeypatch: pytest.MonkeyPatch) -> _Forks:
    forks = _Forks()
    real_init = subprocess.Popen.__init__

    def counting_init(self: Any, args: Any, *a: Any, **k: Any) -> None:
        forks.popen.append(args)
        raise AssertionError(f"R4: subprocess.Popen reached: {args!r}")

    def counting_disclaimed(*a: Any, **k: Any) -> Any:
        forks.disclaimed.append(a)
        raise AssertionError("R4: spawn_disclaimed reached")

    def counting_local(*a: Any, **k: Any) -> Any:
        forks.local.append(a)
        raise AssertionError("R4: the local spawn route reached")

    monkeypatch.setattr(subprocess.Popen, "__init__", counting_init)
    monkeypatch.setattr(launch, "spawn_disclaimed", counting_disclaimed)
    monkeypatch.setattr(macos_disclaim, "spawn_disclaimed", counting_disclaimed)
    monkeypatch.setattr(launch, "_spawn_runtime_local", counting_local)
    del real_init
    return forks


def _engage(tmp_path: Path) -> Any:
    return asyncio.run(
        engage_runtime(
            "r4session0001",
            str(tmp_path),
            WarmErrand(),
            config_dir=tmp_path,
            deadline_s=8.0,
        )
    )


def _assert_engage_refuses_without_forking(tmp_path: Path, forks: _Forks) -> RuntimeStartupError:
    """The R4 check itself, shared with its negative control."""
    with pytest.raises(RuntimeStartupError) as raised:
        _engage(tmp_path)
    assert forks.total == 0, (
        f"a host-absent engage forked: Popen={forks.popen} disclaimed={forks.disclaimed} "
        f"local={forks.local}"
    )
    # A KEYED sentence, in the active locale, vetted for display (``actionable``).
    assert raised.value.actionable
    assert raised.value.actionable == str(raised.value)
    return raised.value


@pytest.fixture
def host_route(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from local_operator import memory_guard

    monkeypatch.setenv(session_host.SPAWN_AUTHORITY_ENV, "1")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # Admission shells out to read memory; it is not what is under test and its
    # subprocess would make the zero-Popen assertion about the machine, not the code.
    monkeypatch.setattr(
        memory_guard,
        "admission_check",
        lambda _what: memory_guard.Admission(True, 0, 0, 0, ""),
    )


def test_installer_refusal_never_forks_locally(
    host_route: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    forks = _arm_fork_counters(monkeypatch)
    asked: list[Path] = []

    def refusing(root: Path) -> host_install.HostInstallOutcome:
        asked.append(root)
        return host_install.HostInstallOutcome(False, "unavailable", "refused by the double")

    monkeypatch.setattr(host_install, "ensure_host_installed", refusing)
    error = _assert_engage_refuses_without_forking(tmp_path, forks)
    assert session_host.unavailable_sentence("unavailable") == str(error)
    assert len(asked) == 1, "one engage must not ask the installer twice"


def test_an_unsupported_platform_never_forks_locally(
    host_route: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    forks = _arm_fork_counters(monkeypatch)
    # The REAL installer, with no supervisor on this "machine".
    monkeypatch.setattr(supervisors, "supervisor", lambda: None)
    error = _assert_engage_refuses_without_forking(tmp_path, forks)
    assert session_host.unavailable_sentence("unsupported") == str(error)


def test_a_host_nobody_answers_never_forks_locally(
    host_route: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    forks = _arm_fork_counters(monkeypatch)
    monkeypatch.setattr(
        host_install, "ensure_host_installed", lambda root: host_install.HostInstallOutcome(True)
    )
    monkeypatch.setattr(session_host, "CONNECT_DEADLINE_S", 0.6)
    monkeypatch.setattr(
        session_host.request_spawn, "__kwdefaults__", {"connect_deadline_s": 0.6}, raising=False
    )
    started = time.monotonic()
    error = _assert_engage_refuses_without_forking(tmp_path, forks)
    assert session_host.unavailable_sentence("unavailable") == str(error)
    # Bounded: the connect ladder is inside the engage's own deadline.
    assert time.monotonic() - started < 6.0


def test_a_host_that_refuses_the_fork_never_forks_locally(
    host_route: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_host: Any
) -> None:
    calls: list[Any] = []

    def refusing_spawner(request: Any, cap_fd: Any) -> Any:
        calls.append(request.session_id)
        raise OSError("no fork for you")

    rig = make_host(refusing_spawner)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(rig.root))
    forks = _arm_fork_counters(monkeypatch)
    error = _assert_engage_refuses_without_forking(rig.root, forks)
    assert session_host.unavailable_sentence("refused") == str(error)
    assert calls == ["r4session0001"], "exactly one ask reached the host, and it was not retried"


def test_the_r4_check_goes_red_when_a_local_fallback_is_added(
    host_route: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NEGATIVE CONTROL (design §8: "R4 must go red if a local fallback is added")."""
    forks = _arm_fork_counters(monkeypatch)
    monkeypatch.setattr(
        host_install,
        "ensure_host_installed",
        lambda root: host_install.HostInstallOutcome(False, "unavailable", "down"),
    )
    real_hosted = launch._spawn_runtime_hosted

    def mutant(*args: Any, **kwargs: Any) -> Any:
        try:
            return real_hosted(*args, **kwargs)
        except RuntimeStartupError:
            return launch._spawn_runtime_local(*args, **kwargs)  # the fallback R4 forbids

    monkeypatch.setattr(launch, "_spawn_runtime_hosted", mutant)
    with pytest.raises(AssertionError, match="R4: the local spawn route reached|forked"):
        _assert_engage_refuses_without_forking(tmp_path, forks)
    assert forks.local, "the mutant reached the local route and the counter saw it"


def test_the_default_authority_is_local_so_no_surface_loses_its_spawn_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(session_host.SPAWN_AUTHORITY_ENV, raising=False)
    assert session_host.spawn_authority() == session_host.AUTHORITY_LOCAL
    monkeypatch.setenv(session_host.SPAWN_AUTHORITY_ENV, "1")
    assert session_host.spawn_authority() == session_host.AUTHORITY_HOST
    monkeypatch.setenv(session_host.SPAWN_AUTHORITY_ENV, "0")
    assert session_host.spawn_authority() == session_host.AUTHORITY_LOCAL


# ---------------------------------------------------------------------------
# The capability relay — a descriptor, never a value, never held
# ---------------------------------------------------------------------------


def test_the_capability_travels_as_a_descriptor_and_the_host_never_holds_it(
    make_host: Any, tmp_path: Path
) -> None:
    received: list[int] = []
    child_ends: list[int] = []

    def spawner(request: Any, cap_fd: Any) -> Any:
        received.append(cap_fd)
        # A real fork gives the child its OWN duplicate; the host closes its copy.
        child_ends.append(os.dup(cap_fd))
        return _Child(os.getpid())

    rig = make_host(spawner)
    cap = mint_operator_cap()
    handoff = open_operator_cap_handoff()
    try:
        process = session_host.request_spawn(
            rig.root,
            session_id="capsession01",
            cwd=str(tmp_path),
            env={},
            interpreter=sys.executable,
            capture=_capture(tmp_path),
            cap_fd=handoff.pass_fds[0],
        )
        # The reply is out and the value does not exist anywhere yet: nothing was
        # readable through the child's end before the surface wrote it.
        child = socket.socket(fileno=child_ends[0])
        child.setblocking(False)
        with pytest.raises(BlockingIOError):
            child.recv(32)
        child.setblocking(True)
        handoff.deliver(cap)
        assert child.recv(32) == cap, "the surface's value reaches the runtime's end"
        child.close()
    finally:
        handoff.close()
    # The host's copy of the received descriptor is closed once the request is
    # done ("no capability outlives the request in flight"). Poll: the host closes
    # it in the connection thread's finally, after the reply.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            fcntl.fcntl(received[0], fcntl.F_GETFD)
        except OSError:
            break
        time.sleep(0.02)
    else:
        raise AssertionError("the host still holds the capability descriptor after the request")
    assert process.lop_coalesced is False


def test_a_request_larger_than_a_socket_buffer_arrives_whole_with_its_descriptor(
    make_host: Any, tmp_path: Path
) -> None:
    """The environment travels wholesale; a 300 KB one must not tear the frame."""
    seen: dict[str, Any] = {}

    def spawner(request: Any, cap_fd: Any) -> Any:
        seen["env"] = request.env
        seen["cap_fd_open"] = cap_fd is not None
        return _Child(os.getpid())

    rig = make_host(spawner)
    big = {f"BIG_{i}": "x" * 1000 for i in range(300)}
    handoff = open_operator_cap_handoff()
    try:
        session_host.request_spawn(
            rig.root,
            session_id="bigenvsess001",
            cwd=str(tmp_path),
            env=big,
            interpreter=sys.executable,
            capture=_capture(tmp_path),
            cap_fd=handoff.pass_fds[0],
        )
    finally:
        handoff.close()
    assert seen["env"] == big
    assert seen["cap_fd_open"] is True


def test_the_host_never_reads_a_descriptor_it_was_handed() -> None:
    """API-surface pin: the host module has no way to read the relayed capability."""
    tree = ast.parse((RUNTIME_DIR / "host_service.py").read_text())
    banned_calls = {"read", "readv", "pread", "fromfd", "recv_into"}
    seen: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if name in banned_calls:
                seen.append(f"{name}@{node.lineno}")
            if name == "socket" and any(k.arg == "fileno" for k in node.keywords):
                seen.append(f"socket(fileno=)@{node.lineno}")
    assert not seen, f"the host reads a received descriptor: {seen}"


def test_the_hosted_route_registers_the_cap_against_the_hosts_pid_and_only_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_operator.harness import approval

    approval.reset_operator_caps_for_tests()
    delivered: list[bytes] = []
    handoffs: list[Any] = []
    real_open = launch.open_operator_cap_handoff

    def spy_open() -> Any:
        h = real_open()
        real_write = h._writer
        h._writer = lambda data: (delivered.append(data), real_write(data))[1]
        handoffs.append(h)
        return h

    monkeypatch.setattr(launch, "open_operator_cap_handoff", spy_open)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    answers = iter([False, True])

    def fake_request(root: Path, **kw: Any) -> Any:
        proc = session_host.HostedProcess(4242, coalesced=next(answers), connection=None)
        return proc

    monkeypatch.setattr(session_host, "request_spawn", fake_request)
    first = launch._spawn_runtime_hosted("s1", str(tmp_path), defer_materialise=False)
    assert first.pid == 4242 and len(delivered) == 1
    assert approval.operator_cap_for(4242) == delivered[0]
    approval.reset_operator_caps_for_tests()
    second = launch._spawn_runtime_hosted("s1", str(tmp_path), defer_materialise=False)
    # COALESCED: this engage's descriptor reached no runtime, so nothing is delivered
    # and nothing is remembered (fail-closed).
    assert second.pid == 4242 and len(delivered) == 1
    assert approval.operator_cap_for(4242) is None
    assert all(h.closed for h in handoffs)


# ---------------------------------------------------------------------------
# No double spawn (§3)
# ---------------------------------------------------------------------------


def test_n_contenders_for_one_session_fork_exactly_one_runtime(
    make_host: Any, tmp_path: Path
) -> None:
    forked: list[str] = []
    gate = threading.Event()

    def spawner(request: Any, cap_fd: Any) -> Any:
        forked.append(request.session_id)
        return _Child(40_000 + len(forked), exits=gate)

    rig = make_host(spawner)
    pids: list[int] = []
    flags: list[bool] = []
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def contend() -> None:
        try:
            barrier.wait(5)
            p = _ask(rig, tmp_path, "contended001")
            pids.append(p.pid)
            flags.append(p.lop_coalesced)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=contend) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    gate.set()
    assert not errors, errors
    assert forked == ["contended001"], f"exactly one fork, got {forked}"
    assert len(set(pids)) == 1 and len(pids) == 8
    assert sorted(flags) == [False] + [True] * 7
    stats = rig.host.stats()
    assert stats["forks"] == 1 and stats["coalesced"] == 7


def test_different_sessions_each_get_their_own_runtime(make_host: Any, tmp_path: Path) -> None:
    gate = threading.Event()
    forked: list[str] = []

    def spawner(request: Any, cap_fd: Any) -> Any:
        forked.append(request.session_id)
        return _Child(50_000 + len(forked), exits=gate)

    rig = make_host(spawner)
    a = _ask(rig, tmp_path, "sessiona0001")
    b = _ask(rig, tmp_path, "sessionb0001")
    gate.set()
    assert a.pid != b.pid and sorted(forked) == ["sessiona0001", "sessionb0001"]


# ---------------------------------------------------------------------------
# The death fact (§2: "the reply must carry the child's exit status")
# ---------------------------------------------------------------------------


def test_a_candidate_that_dies_is_reported_with_its_exit_status(
    make_host: Any, tmp_path: Path
) -> None:
    def spawner(request: Any, cap_fd: Any) -> Any:
        return subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])

    rig = make_host(spawner)
    process = _ask(rig, tmp_path)
    assert process.wait(timeout=15) == 7
    assert process.poll() == 7


def test_a_live_candidate_reads_as_running(make_host: Any, tmp_path: Path) -> None:
    gate = threading.Event()
    rig = make_host(lambda request, cap_fd: _Child(os.getpid(), exits=gate))
    process = _ask(rig, tmp_path)
    assert process.poll() is None
    gate.set()
    assert process.wait(timeout=10) == 0


def test_the_proxy_has_no_way_to_signal_a_runtime() -> None:
    for name in ("kill", "terminate", "send_signal"):
        assert not hasattr(session_host.HostedProcess, name), name


# ---------------------------------------------------------------------------
# Validation: what the host will not do
# ---------------------------------------------------------------------------


def test_the_host_refuses_what_it_should_not_act_on(make_host: Any, tmp_path: Path) -> None:
    rig = make_host(lambda r, c: _Child(os.getpid(), exits=threading.Event()))
    base = {
        "op": "spawn",
        "v": session_host.PROTOCOL_VERSION,
        "root": str(rig.root),
        "session_id": "validate0001",
        "interpreter": sys.executable,
        "env": {},
        "cwd": str(tmp_path),
        "capture": str(_capture(tmp_path)),
        "has_cap_fd": False,
    }
    from local_operator.session.runtime.standby import _recv, _send

    def ask(**changes: Any) -> dict[str, Any]:
        sock = session_host._connect(session_host.socket_path(rig.root), 2.0)
        try:
            _send(sock, {**base, **changes})
            reply = _recv(sock)
            assert isinstance(reply, dict)
            return reply
        finally:
            sock.close()

    link = tmp_path / "link.log"
    link.symlink_to(base["capture"])
    cases = {
        "another store": {"root": str(tmp_path / "elsewhere")},
        "a path in the session id": {"session_id": "../x"},
        "a relative interpreter": {"interpreter": "python3"},
        "a symlinked capture": {"capture": str(link)},
        "a missing capture": {"capture": str(tmp_path / "absent.log")},
        "a promised descriptor that never arrived": {"has_cap_fd": True},
        "an unknown version": {"v": 99},
        "an unknown op": {"op": "stop-everything"},
    }
    for label, change in cases.items():
        reply = ask(**change)
        assert reply["ok"] is False, f"{label}: {reply}"
    assert rig.host.stats()["forks"] == 0


# ---------------------------------------------------------------------------
# R3 (unit half) — the host's exit path signals nothing
# ---------------------------------------------------------------------------


def test_the_hosts_exit_path_signals_nothing(
    make_host: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[Any] = []
    monkeypatch.setattr(os, "kill", lambda *a: sent.append(("kill", a)))
    monkeypatch.setattr(os, "killpg", lambda *a: sent.append(("killpg", a)))
    gate = threading.Event()
    rig = make_host(lambda request, cap_fd: _Child(os.getpid(), exits=gate))
    _ask(rig, tmp_path)
    rig.stop.set()
    assert rig.thread is not None
    rig.thread.join(10)
    assert not rig.thread.is_alive()
    assert sent == [], f"the host signalled on its way out: {sent}"
    assert not session_host.socket_path(rig.root).exists(), "the socket is unlinked on exit"
    gate.set()


def test_the_host_has_no_stop_vocabulary() -> None:
    """R3, structurally: nothing in the host or its installer can end a runtime."""
    banned = {"kill", "killpg", "terminate", "send_signal", "sweep"}
    for module in ("host_service.py", "host_install.py", "session_host.py"):
        tree = ast.parse((RUNTIME_DIR / module).read_text())
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in banned:
                    hits.append(f"{name}@{node.lineno}")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                if any("owned_runtimes" in n for n in names):
                    hits.append(f"import owned_runtimes@{node.lineno}")
        assert not hits, f"{module}: {hits}"


def test_the_surface_half_cannot_fork_by_construction() -> None:
    """R4, structurally: ``session_host`` imports nothing that starts a process."""
    tree = ast.parse((RUNTIME_DIR / "session_host.py").read_text())
    forbidden = {"subprocess", "macos_disclaim", "launch", "multiprocessing"}
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] in forbidden]
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[-1] in forbidden or any(a.name in forbidden for a in node.names):
                offenders.append(module)
    # ``subprocess`` appears once, for TimeoutExpired in ``HostedProcess.wait``.
    offenders = [o for o in offenders if o != "subprocess"]
    assert not offenders, offenders
    calls = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", "") in {"Popen", "run", "spawn_disclaimed", "fork", "system"}
    ]
    assert not calls, f"session_host.py starts a process at lines {calls}"


def test_the_dispatcher_has_no_fallthrough_from_the_host_route() -> None:
    """The host branch is a ``return`` inside no ``try`` and the local call lives in ``else``."""
    tree = ast.parse((RUNTIME_DIR / "launch.py").read_text())
    dispatcher = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_spawn_runtime"
    )
    assert not [n for n in ast.walk(dispatcher) if isinstance(n, ast.Try)], "no try in the seam"
    branch = next(n for n in dispatcher.body if isinstance(n, ast.If))
    assert isinstance(branch.body[-1], ast.Return)
    assert "_spawn_runtime_hosted" in ast.dump(branch.body[-1])
    assert "_spawn_runtime_local" not in ast.dump(ast.Module(branch.body, []))
    assert "_spawn_runtime_local" in ast.dump(ast.Module(branch.orelse, []))
    hosted = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_spawn_runtime_hosted"
    )
    # Identifiers actually REFERENCED, not string constants: the return annotation
    # and the ``cast`` target legitimately spell ``Popen``.
    names = {n.id for n in ast.walk(hosted) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(hosted) if isinstance(n, ast.Attribute)
    }
    for name in ("_spawn_runtime_local", "spawn_disclaimed", "Popen", "_spawn_runtime"):
        assert name not in names, f"the hosted route references {name}"


def test_both_routes_hand_a_runtime_the_same_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One builder: the host route cannot drift from the local one on the env scrubs."""
    monkeypatch.setenv("LOP_MOBILE_CHILD_PROVIDER", "stale")
    monkeypatch.setenv("LOP_SESSION_ENGAGED", "1")
    monkeypatch.setenv("LOP_WAKE_SUPERVISOR_OWNER", "tok")
    seen: dict[str, dict[str, str]] = {}

    class _Proc:
        pid = 1

        def poll(self) -> None:
            return None

        def wait(self) -> int:
            return 0

    def fake_disclaimed(argv: Any, **kw: Any) -> Any:
        seen["local"] = dict(kw["env"])
        return _Proc()

    monkeypatch.setattr(launch, "spawn_disclaimed", fake_disclaimed)
    from local_operator.session.runtime import standby

    monkeypatch.setattr(standby, "adoption_possible", lambda: False)
    launch._spawn_runtime_local("envsession01", str(tmp_path), defer_materialise=True, warm=True)

    def fake_request(root: Path, **kw: Any) -> Any:
        seen["hosted"] = dict(kw["env"])
        return session_host.HostedProcess(7, coalesced=True, connection=None)

    monkeypatch.setattr(session_host, "request_spawn", fake_request)
    launch._spawn_runtime_hosted("envsession01", str(tmp_path), defer_materialise=True, warm=True)
    local_env = {k: v for k, v in seen["local"].items() if k != "LOP_SPAWN_CHAIN"}
    assert local_env == seen["hosted"]
    assert "LOP_MOBILE_CHILD_PROVIDER" not in seen["hosted"]
    assert seen["hosted"]["LOP_SESSION_ENGAGED"] == "1"
    assert seen["hosted"]["LOP_MOBILE_CHILD_RESUME"] == "envsession01"


# ---------------------------------------------------------------------------
# The unit
# ---------------------------------------------------------------------------


def test_the_plist_is_a_sibling_of_the_wake_supervisor_not_an_extension(tmp_path: Path) -> None:
    from local_operator.wakes import install as wakes_install

    plist = host_install.render_plist(tmp_path)
    assert plist["Label"] == host_install.LABEL != wakes_install.LABEL
    # A crashed host restarts; a cleanly-exited one (election loser, SIGTERM) stays down.
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    # NOT the wake supervisor's self-heal interval: this unit must never self-retire
    # on an empty index, and nothing about it is a schedule.
    assert "StartInterval" not in plist
    # Stopping the job must not be able to reach a runtime (R3).
    assert plist["AbandonProcessGroup"] is True
    assert plist["EnvironmentVariables"] == {"LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path)}
    args = plist["ProgramArguments"]
    assert isinstance(args, list) and host_install.HOST_MODULE in args


def test_the_installer_refuses_a_sandboxed_home_without_touching_launchd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.LAUNCHCTL)
    monkeypatch.setattr(host_install, "_launchctl", lambda *a: calls.append(a))
    outcome = host_install.ensure_host_installed(tmp_path)
    assert not outcome.ok and outcome.code == "unavailable"
    assert calls == [], "a sandboxed run must never address the operator's launchd domain"
    # The refusal comes BEFORE the plist write too: a store that is not the
    # default one must not repoint the shared unit (2026-10-08), and the plist
    # lives under the redirected HOME here, so its absence is observable.
    assert not host_install.plist_path().exists()


def test_a_machine_without_launchd_is_reported_unsupported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.SYSTEMCTL)
    outcome = host_install.ensure_host_installed(tmp_path)
    assert (outcome.ok, outcome.code) == (False, "unsupported")


# ---------------------------------------------------------------------------
# Election
# ---------------------------------------------------------------------------


def test_a_second_host_for_the_same_store_exits_cleanly_without_serving(
    make_host: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = make_host()
    monkeypatch.setattr(host_service, "LOCK_WAIT_S", 0.3)
    started = time.monotonic()
    code = host_service.serve(rig.root, threading.Event())
    assert code == 0, "a loser exits 0 so KeepAlive{SuccessfulExit:false} does not loop it"
    assert time.monotonic() - started < 5
    assert session_host.ping(rig.root), "the first host is undisturbed"
