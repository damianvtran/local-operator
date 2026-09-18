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
    seen: dict[str, Any] = {}
    monkeypatch.setattr(os, "set_inheritable", lambda fd, flag: seen.update(fd=fd, flag=flag))
    monkeypatch.setattr(
        os, "execve", lambda path, argv, env: seen.update(path=path, argv=argv, env=env)
    )
    plan = _watch(_app(fd=13)).plan()
    serve_reload._exec(plan)
    assert seen["fd"] == 13 and seen["flag"] is True
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
