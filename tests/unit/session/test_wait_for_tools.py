"""Typed required-tool readiness: the owning session's ``wait_for_tools`` contract.

WHY THIS FILE EXISTS. MCP discovery is asynchronous by design -- a server
deferred past the 250 ms startup gate connects on a background continuation --
so reading the inventory right after ``open_session`` can catch it empty, and a
prompt that wins that race answers every call with "Tool not found" (measured:
~10 ms after open on the pilot). ``Session.wait_for_tools`` is the sanctioned
gate; these tests pin its contract against CONTROLLED delayed fixtures rather
than timing assumptions:

* already-present names return ready WITHOUT consulting a manager (the
  built-in fast path, including while unrelated servers run slow);
* a delayed arrival is gated: the raw read races, the typed wait returns
  ready, and the first request built after it sees the tool;
* settled-without-the-name / no-manager / manager-failure never read ready
  (missing, missing, pending respectively);
* a timeout reports pending WITHOUT cancelling discovery, and a later fresh
  call can still return ready;
* cancellation propagates and leaves discovery alive;
* the inventory is never widened, and names are validated before any wait.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.session.session import Session
from local_operator.session.tool_readiness import ToolReadiness


class _DuckSession:
    """The live-inventory + manager surface ``Session.wait_for_tools`` touches."""

    def __init__(self, names: tuple[str, ...] = (), manager: object | None = None) -> None:
        self.names = list(names)
        self.mcp_manager = manager

    @property
    def tool_inventory(self) -> tuple[str, ...]:
        return tuple(self.names)


class _ExplodingManager:
    """Fails loudly if consulted -- the fast path must not wait on anything."""

    def __init__(self) -> None:
        self.calls = 0

    async def wait_settled(self, timeout_s: float) -> bool:
        self.calls += 1
        raise AssertionError("wait_settled must not be called on the fast path")


class _SettledManager:
    """Settles immediately, reporting success or failure; never adds tools."""

    def __init__(self, settle_result: bool = True, error: Exception | None = None) -> None:
        self.calls = 0
        self.timeouts: list[float] = []
        self.settle_result = settle_result
        self.error = error

    async def wait_settled(self, timeout_s: float) -> bool:
        self.calls += 1
        self.timeouts.append(timeout_s)
        if self.error is not None:
            raise self.error
        return self.settle_result


class _DelayedManager:
    """The controlled delayed-MCP fixture.

    The tool lands after ``delay_s`` on an arrival task that ``wait_settled``
    watches -- and, exactly like the real manager, a timeout stops WATCHING
    the arrival without cancelling it, so discovery keeps moving.
    """

    def __init__(self, duck: _DuckSession, name: str, delay_s: float) -> None:
        self.duck = duck
        self.name = name
        self.delay_s = delay_s
        self.calls = 0
        self.arrived = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def _schedule(self) -> None:
        if self._task is not None:
            return

        async def _arrive() -> None:
            await asyncio.sleep(self.delay_s)
            if self.name not in self.duck.names:
                self.duck.names.append(self.name)
            self.arrived.set()

        self._task = asyncio.get_running_loop().create_task(_arrive())

    async def wait_settled(self, timeout_s: float) -> bool:
        self.calls += 1
        self._schedule()
        try:
            await asyncio.wait_for(self.arrived.wait(), timeout=max(0.0, timeout_s))
        except TimeoutError:
            return False
        return True


@pytest.mark.asyncio
async def test_already_present_reads_ready_without_a_manager_call() -> None:
    manager = _ExplodingManager()
    duck = _DuckSession(("bash",), manager)
    readiness = await Session.wait_for_tools(duck, ["bash"])
    assert readiness == ToolReadiness("ready", ("bash",), ("bash",), (), None)
    assert manager.calls == 0


@pytest.mark.asyncio
async def test_delayed_discovery_race_is_gated_by_the_wait() -> None:
    duck = _DuckSession()
    manager = _DelayedManager(duck, "mcp__episode_actions", 0.05)
    duck.mcp_manager = manager
    # The raw race the gap class describes: an immediate read is empty even
    # though the tool is on its way.
    assert duck.tool_inventory == ()
    readiness = await Session.wait_for_tools(duck, ["mcp__episode_actions"], timeout_s=1.0)
    assert readiness.status == "ready"
    assert readiness.discovery_settled is True
    assert readiness.present == ("mcp__episode_actions",)
    # The first request built after this gate sees the tool.
    assert "mcp__episode_actions" in duck.tool_inventory


@pytest.mark.asyncio
async def test_settled_without_the_name_is_missing() -> None:
    manager = _SettledManager(settle_result=True)
    duck = _DuckSession(("bash",), manager)
    readiness = await Session.wait_for_tools(duck, ["mcp__gone"], timeout_s=0.5)
    assert readiness.status == "missing"
    assert readiness.discovery_settled is True
    assert readiness.missing == ("mcp__gone",)


@pytest.mark.asyncio
async def test_missing_without_a_manager_is_missing_not_pending() -> None:
    duck = _DuckSession(("bash",), None)
    readiness = await Session.wait_for_tools(duck, ["mcp__gone"])
    assert readiness.status == "missing"
    assert readiness.discovery_settled is None


@pytest.mark.asyncio
async def test_manager_failure_is_never_ready() -> None:
    manager = _SettledManager(error=RuntimeError("reduced manager"))
    duck = _DuckSession((), manager)
    readiness = await Session.wait_for_tools(duck, ["mcp__x"], timeout_s=0.5)
    assert readiness.status == "pending"
    assert readiness.discovery_settled is None


@pytest.mark.asyncio
async def test_timeout_is_pending_and_a_later_wait_still_reads_ready() -> None:
    duck = _DuckSession()
    manager = _DelayedManager(duck, "mcp__late", 0.15)
    duck.mcp_manager = manager
    readiness = await Session.wait_for_tools(duck, ["mcp__late"], timeout_s=0.01)
    assert readiness.status == "pending"
    assert readiness.discovery_settled is False
    # The timeout must not have cancelled discovery: the continuation lands.
    await asyncio.wait_for(manager.arrived.wait(), timeout=1.0)
    fresh = await Session.wait_for_tools(duck, ["mcp__late"], timeout_s=1.0)
    assert fresh.status == "ready"


@pytest.mark.asyncio
async def test_cancellation_propagates_and_leaves_discovery_alive() -> None:
    duck = _DuckSession()
    manager = _DelayedManager(duck, "mcp__slow", 0.05)
    duck.mcp_manager = manager
    task = asyncio.create_task(Session.wait_for_tools(duck, ["mcp__slow"], timeout_s=30.0))
    await asyncio.sleep(0)  # let the task enter the manager wait
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.calls == 1
    # Discovery untouched: it completes later and a fresh wait reads ready.
    await asyncio.wait_for(manager.arrived.wait(), timeout=1.0)
    fresh = await Session.wait_for_tools(duck, ["mcp__slow"], timeout_s=1.0)
    assert fresh.status == "ready"


@pytest.mark.asyncio
async def test_names_are_deduped_in_caller_order() -> None:
    duck = _DuckSession(("a", "b"))
    readiness = await Session.wait_for_tools(duck, ["b", "a", "b"])
    assert readiness.required == ("b", "a")
    assert readiness.present == ("b", "a")


@pytest.mark.asyncio
async def test_inventory_is_never_widened() -> None:
    manager = _SettledManager(settle_result=True)
    duck = _DuckSession(("bash",), manager)
    before = duck.tool_inventory
    readiness = await Session.wait_for_tools(duck, ["mcp__typo"], timeout_s=0.5)
    assert readiness.status == "missing"
    assert duck.tool_inventory == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("required", "timeout_s"),
    [
        (["ok"], -1.0),
        (["ok"], float("nan")),
        (["ok"], float("inf")),
        ([""], 1.0),
        (["ok", 123], 1.0),
        ("bash", 1.0),
        (None, 1.0),
    ],
)
async def test_invalid_inputs_are_refused_before_any_wait(
    required: object, timeout_s: float
) -> None:
    manager = _ExplodingManager()
    duck = _DuckSession(("ok",), manager)
    with pytest.raises(ValueError):
        await Session.wait_for_tools(duck, required, timeout_s=timeout_s)  # type: ignore[arg-type]
    assert manager.calls == 0
