"""A serve daemon replacing its own process image onto the current build.

What is asserted here is the CONTRACT the rest of the machine leans on, and each
assertion names the failure it prevents:

* a refusal (never a guess) when there is no socket to carry, no pointer to
  follow, or nothing to move onto — because a reload that guessed would either
  drop the listener or cut work for no gain;
* the drain is NARROW — it waits for a runtime being started and does NOT wait
  for the standing app relays, which never end while the app is open, so gating
  on them would be a reload that never fires where it is needed;
* ``execve`` is handed the fd as INHERITABLE and the new generation's
  interpreter, with ``-P`` — three ways to lose the port silently, all of them
  cheap to get wrong and invisible from outside the process.

The end-to-end proof — a real daemon adopting a real socket and coming back on a
new build with its runtimes still up — is not a unit test and is recorded on the
pull request, not here.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.interpreter import SAFE_PATH_FLAG
from local_operator.server import registry as serve_registry
from local_operator.server import reload as serve_reload


class FakeApp:
    """What the reload touches: ``state``, and nothing else.

    A stand-in rather than the module-level singleton, for the reason
    ``test_serve_retire`` gives for its own: the flag and the announcement are
    per-boot, and a shared app would leak them between tests.
    """

    def __init__(self, **state: Any) -> None:
        self.state = SimpleNamespace(**state)


class FakePool:
    """A desktop pool that reports one fixed reason, or raises when asked."""

    def __init__(self, reason: str | None = None, *, raises: bool = False) -> None:
        self.reason = reason
        self.raises = raises
        self.reads = 0

    def reload_blocker(self) -> str | None:
        self.reads += 1
        if self.raises:
            raise RuntimeError("the probe could not be read")
        return self.reason


def _app(
    *,
    fd: int | None = 7,
    announced: tuple[str, int] | None = ("127.0.0.1", 53421),
    pool: Any = None,
) -> FakeApp:
    return FakeApp(
        **{
            serve_reload.LISTENER_FD_ATTR: fd,
            serve_registry.ANNOUNCED_STATE_ATTR: announced,
            "desktop_sessions": pool,
        }
    )


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Control what the pointer says, without touching a real install.

    The three ``update`` readers the reload uses are patched together on purpose:
    a test that patched only the interpreter would pass while the refusal
    comparison read the developer's own ``sys.prefix`` and answered "you are
    already there" on a machine where the two happened to agree.
    """
    state: dict[str, Any] = {
        "interpreter": Path("/opt/gen/20260918T124715Z-0.59.0/tools/local-operator/bin/python3"),
        "current": Path("/opt/gen/20260918T124715Z-0.59.0/tools/local-operator"),
        "loaded": "/opt/gen/20260918T000828Z-0.56.14/tools/local-operator",
    }

    def _interpreter() -> Path | None:
        return state["interpreter"]

    def _current() -> Path | None:
        return state["current"]

    def _loaded() -> str:
        return state["loaded"]

    monkeypatch.setattr("local_operator.update.current_interpreter", _interpreter)
    monkeypatch.setattr("local_operator.update.current_install_root", _current)
    monkeypatch.setattr("local_operator.update.process_install_root", _loaded)
    return state


def _watch(app: FakeApp, **kwargs: Any) -> serve_reload.ReloadWatch:
    return serve_reload.ReloadWatch(app, **kwargs)  # type: ignore[arg-type]


def test_plan_refuses_a_boot_with_no_listener_of_its_own(install: dict[str, Any]) -> None:
    """A ``--reload`` child's port is its supervisor's, so there is nothing to carry.

    Signalling one would exec a successor that then cannot bind — the reload
    would turn a working dev server into a dead port.
    """
    with pytest.raises(serve_reload.ReloadRefusal, match="no listening socket"):
        _watch(_app(fd=None)).plan()


def test_plan_refuses_a_daemon_that_was_never_announced(install: dict[str, Any]) -> None:
    """No announced address means nothing to put in the record or the banner."""
    with pytest.raises(serve_reload.ReloadRefusal, match="never told which address"):
        _watch(_app(announced=None)).plan()


def test_plan_refuses_when_the_pointer_resolves_to_nothing(
    install: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No pointer, no build to move onto — ``lop update`` has not run yet."""
    monkeypatch.setattr("local_operator.update.current_interpreter", lambda: None)
    with pytest.raises(serve_reload.ReloadRefusal, match="pointer resolves to nothing"):
        _watch(_app()).plan()


def test_plan_refuses_to_move_a_daemon_onto_the_build_it_is_running(
    install: dict[str, Any],
) -> None:
    """The no-op that would cut every client's stream for no gain at all.

    The comparison is through symlinks because the two answers genuinely differ in
    shape: a daemon's ``sys.prefix`` is the concrete generation and the pointer is
    a symlink to it.
    """
    install["current"] = Path(install["loaded"])
    with pytest.raises(serve_reload.ReloadRefusal, match="already running the build"):
        _watch(_app()).plan()


def test_plan_names_the_new_interpreter_and_keeps_the_socket(install: dict[str, Any]) -> None:
    plan = _watch(_app(fd=11)).plan()
    assert plan.interpreter == install["interpreter"]
    assert plan.target_root == install["current"]
    assert plan.loaded_root == Path(install["loaded"])
    assert plan.listener_fd == 11
    assert (plan.host, plan.port) == ("127.0.0.1", 53421)


@pytest.mark.asyncio
async def test_drain_returns_immediately_when_no_runtime_is_being_started(
    install: dict[str, Any],
) -> None:
    """The ordinary case, and the one the feature lives or dies on.

    A loaded app holds a relay and a watch lease for as long as a window is open,
    so a drain that waited for those would never return on the machine this
    exists for. ``reload_blocker`` reports neither, which is what this asserts.
    """
    pool = FakePool(None)
    await _watch(_app(pool=pool)).drain()
    assert pool.reads == 1


@pytest.mark.asyncio
async def test_drain_refuses_after_the_budget_and_names_the_term(
    install: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reload that cannot drain keeps serving, and says why.

    The alternative — exec anyway — is the one outcome the fail-closed rule
    exists to prevent: it would cut the spawn handshake the budget was sized for.
    """
    monkeypatch.setattr(serve_reload, "DRAIN_BUDGET_S", 0.05)
    monkeypatch.setattr(serve_reload, "DRAIN_POLL_S", 0.01)
    pool = FakePool("a runtime being started for session abc")
    with pytest.raises(serve_reload.ReloadRefusal, match="a runtime being started"):
        await _watch(_app(pool=pool)).drain()


@pytest.mark.asyncio
async def test_an_unreadable_drain_probe_is_not_a_reason_to_wait(install: dict[str, Any]) -> None:
    """A probe that raises must not be the reason a reload never happens.

    Same rule ``retire.in_flight`` states for its own probes, in the direction
    that keeps the daemon alive rather than the one that keeps it stale.
    """
    await _watch(_app(pool=FakePool(raises=True))).drain()


@pytest.mark.asyncio
async def test_a_refused_reload_clears_the_pending_flag_and_keeps_serving(
    install: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag is intent, and an intent that was refused must not linger.

    A flag left set would make every later check re-attempt a reload that cannot
    happen, and the daemon's record would advertise a capability it cannot use.
    """
    app = _app(fd=None)
    watch = _watch(app)
    watch.request()
    assert watch.pending is True
    task = asyncio.create_task(watch.run())
    await asyncio.sleep(0.05)
    assert watch.pending is False
    assert not task.done()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def test_exec_hands_over_the_fd_the_new_interpreter_and_a_safe_path(
    install: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three silent ways to lose the port, asserted in one place.

    A non-inheritable fd, an ``execve`` onto this process's own interpreter, or a
    missing ``-P`` each produces a daemon that looks healthy from the outside and
    either serves the old code or has no listener at all.
    """
    seen: dict[str, Any] = {"inherit": []}
    monkeypatch.setattr(os, "set_inheritable", lambda fd, flag: seen["inherit"].append((fd, flag)))
    monkeypatch.setattr(
        os, "execve", lambda path, argv, env: seen.update(path=path, argv=argv, env=env)
    )
    plan = _watch(_app(fd=13)).plan()
    serve_reload._exec(plan)
    # The fd is made inheritable for the exec and put BACK afterwards, because a
    # stubbed exec leaves this process serving (serve-reload review round 1's
    # `set_inheritable` nit).
    assert seen["inherit"] == [(13, True), (13, False)]
    assert seen["path"] == str(plan.interpreter)
    assert seen["argv"] == [
        str(plan.interpreter),
        SAFE_PATH_FLAG,
        "-m",
        "local_operator.cli",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "53421",
        "--listener-fd",
        "13",
    ]
    assert seen["env"]["PATH"] == os.environ["PATH"]


@pytest.mark.asyncio
async def test_install_arms_the_handler_and_the_signal_sets_the_flag(
    install: dict[str, Any],
) -> None:
    """The capability is a SIGNAL, so its whole contract is that one arrives."""
    app = _app()
    watch = serve_reload.install(app)  # type: ignore[arg-type]
    assert watch is not None
    assert serve_reload.is_reloadable(app) is True
    os.kill(os.getpid(), serve_reload.RELOAD_SIGNAL)
    await asyncio.sleep(0.05)
    assert watch.pending is True


def test_exec_ignores_the_signal_across_the_replace(
    install: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """serve-reload R1-1: the successor must be DEAF until it is ready to listen.

    For the successor's whole boot the last published record is still live and
    still advertising ``reloadable``, while the process it names has not reached
    ``add_signal_handler`` — so SIGUSR1 is still at its default disposition of
    terminate. Review reproduced the consequence end to end: a second request
    1.45 s after the first left the pid gone and the port dead, with nothing
    logged. ``SIG_IGN`` survives ``execve``, which is what closes the window.
    """
    import signal as signal_mod

    seen: dict[str, Any] = {}

    def _capture(path: str, argv: list[str], env: dict[str, str]) -> None:
        # Read INSIDE the exec call: this is the disposition the successor
        # inherits, which is the only place the claim can be checked.
        seen["disposition"] = signal_mod.getsignal(serve_reload.RELOAD_SIGNAL)

    monkeypatch.setattr(os, "set_inheritable", lambda fd, flag: None)
    monkeypatch.setattr(os, "execve", _capture)
    before = signal_mod.getsignal(serve_reload.RELOAD_SIGNAL)
    serve_reload._exec(_watch(_app(fd=17)).plan())
    assert seen["disposition"] == signal_mod.SIG_IGN
    # And this process is NOT left deaf: the exec was stubbed, so it kept
    # serving, and a daemon that ignored the signal forever could never be
    # asked to reload again.
    assert signal_mod.getsignal(serve_reload.RELOAD_SIGNAL) == before


def test_a_build_that_cannot_import_refuses_the_reload(
    install: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """serve-reload R1-5: the one refusal that has to happen BEFORE the exec to be worth anything.

    Every other refusal keeps the daemon serving what it loaded. An exec into a
    build that cannot start leaves a dead daemon, a dead port and no process left
    to log it, so the target is asked to import its own CLI first.
    """
    import subprocess as subprocess_mod

    def _broke(argv: list[str], **kwargs: Any) -> Any:
        return subprocess_mod.CompletedProcess(
            argv, 1, stdout="", stderr="ModuleNotFoundError: no module named 'local_operator'"
        )

    monkeypatch.setattr(subprocess_mod, "run", _broke)
    with pytest.raises(serve_reload.ReloadRefusal, match="cannot import its own CLI"):
        serve_reload._smoke(Path("/opt/gen/new/bin/python3"))


def test_a_smoke_check_that_could_not_run_proceeds(
    install: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "We could not ask" is not "the build is bad".

    The same rule ``retire`` applies to its own probes: a machine where the probe
    cannot run is a machine where refusing would strand the daemon on the old
    build for that reason alone.
    """
    import subprocess as subprocess_mod

    def _cannot_run(argv: list[str], **kwargs: Any) -> Any:
        raise OSError("no such file or directory")

    monkeypatch.setattr(subprocess_mod, "run", _cannot_run)
    serve_reload._smoke(Path("/opt/gen/new/bin/python3"))


def test_an_adopted_listener_keeps_its_address_family() -> None:
    """serve-reload R1-3: ``fromfd`` needs the right family or it misreads the address bytes.

    A hardcoded ``AF_INET`` reinterprets an IPv6 listener's bytes as IPv4 — the
    daemon still serves, which is exactly why review found it by reading log
    lines (`::24:b503:100:0:61963` for a peer that an ordinary bind reports as
    `::1:62246`) rather than by a failure. This asserts the family the reload
    would have used against a REAL bound IPv6 socket.
    """
    import socket as socket_mod

    from local_operator.cli import adopt_serve_socket

    try:
        donor = socket_mod.socket(socket_mod.AF_INET6, socket_mod.SOCK_STREAM)
        donor.bind(("::1", 0))
    except OSError:  # pragma: no cover - no IPv6 on this host
        pytest.skip("this host has no usable IPv6 loopback")
    try:
        donor.set_inheritable(True)
        # ``detach`` rather than handing over `.fileno()`: the helper CLOSES the fd
        # it is given (it is the inherited one, and the dup it makes is what
        # survives), so a Python socket object still tracking that fd would close
        # it a second time. Detaching says "the helper owns this now".
        adopted = adopt_serve_socket(donor.detach(), "::1")
        try:
            assert adopted.family == socket_mod.AF_INET6
            assert adopted.getsockname()[0] == "::1"
        finally:
            adopted.close()
    finally:
        donor.close()


def test_a_spawn_that_arrives_during_the_smoke_is_still_waited_for(
    install: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """serve-reload R3-1: the drain does not cover the off-loop smoke, so it is re-asked.

    Moving the smoke to a thread made the pre-exec phase longer than the wait that
    guards it, and the term the wait exists for — a runtime being SPAWNED, whose
    ~1.2 s handshake is the one cut this module calls unrecoverable — can begin in
    that window. A first drain that returned at t=0 says nothing about t=0.3.

    The blocker answers "clear" once and "busy" thereafter, which is exactly a
    spawn arriving during the smoke: with only the first drain, this test's
    `_exec` would be reached and would fail.
    """
    reloader = _watch(_app(fd=19))
    answers = {"n": 0}

    def blocker() -> str | None:
        answers["n"] += 1
        return None if answers["n"] == 1 else "a runtime is being spawned"

    monkeypatch.setattr(reloader, "blocker", blocker)
    # Zero budgets so the single-poll refusal is immediate instead of ten seconds.
    monkeypatch.setattr(serve_reload, "DRAIN_BUDGET_S", 0.0)
    monkeypatch.setattr(serve_reload, "_smoke", lambda interpreter: None)
    monkeypatch.setattr(
        serve_reload, "_exec", lambda plan: pytest.fail("exec'd with a spawn in flight")
    )
    with pytest.raises(serve_reload.ReloadRefusal, match="being spawned"):
        asyncio.run(reloader.perform())
    assert answers["n"] >= 2


def test_install_refuses_a_boot_with_no_listener(install: dict[str, Any]) -> None:
    """``None`` is a refusal the record then publishes, not a failure to report."""
    assert serve_reload.install(_app(fd=None)) is None  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_install_refuses_a_uvicorn_reload_child(
    install: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dev-mode child must not advertise a capability it will not honour.

    Its port belongs to uvicorn's supervisor, so a reload there would leave the
    parent accepting on a socket with nothing behind it.
    """
    monkeypatch.setattr(serve_registry, "is_reload_child", lambda app=None: True)
    assert serve_reload.install(_app()) is None  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_the_watch_returns_when_the_daemon_is_stopping(install: dict[str, Any]) -> None:
    """A teardown must not wait for a request that will never come."""
    stop = asyncio.Event()
    watch = _watch(_app(), stop=stop)
    task = asyncio.create_task(watch.run())
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)


def test_a_dead_reload_task_is_reported_and_a_cancelled_one_is_not(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A task that died leaves a record advertising a capability nothing serves.

    Cancellation is the ordinary teardown and must stay silent, or every ordinary
    exit of every daemon logs an error.
    """

    async def _die() -> None:
        raise RuntimeError("boom")

    async def _drive() -> None:
        dead = asyncio.create_task(_die())
        cancelled = asyncio.create_task(asyncio.sleep(30))
        cancelled.cancel()
        await asyncio.gather(dead, return_exceptions=True)
        await asyncio.gather(cancelled, return_exceptions=True)
        with caplog.at_level("ERROR", logger="local_operator.server.reload"):
            serve_reload.observe_reload(dead)
            serve_reload.observe_reload(cancelled)

    asyncio.run(_drive())
    messages = [record.message for record in caplog.records]
    assert len(messages) == 1
    assert "reload task died" in messages[0]
