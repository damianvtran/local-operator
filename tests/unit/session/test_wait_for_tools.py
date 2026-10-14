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

The doubles are deliberately TYPED: ``_SessionDouble`` is a real ``Session``
subclass (never engine-``__init__``-ed -- ``build`` seeds only the two
attributes the method under test reads), and the manager fakes subclass
``McpManager`` so the tests exercise the production receiver types rather
than duck-typed stand-ins the static gate would flag.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.harness.types import AgentTool
from local_operator.mcp.manager import McpManager
from local_operator.session.session import Session


class _SessionDouble(Session):
    """A typed ``Session`` double for the readiness surface.

    Built via :meth:`build` so the engine constructor never runs: only the
    live ``_tools`` inventory and ``mcp_manager`` -- exactly what
    ``wait_for_tools`` reads -- are seeded. Seeds use ``model_construct``
    because ``AgentTool.execute`` is a required field this surface never
    touches.
    """

    @classmethod
    def build(
        cls, names: tuple[str, ...] = (), manager: McpManager | None = None
    ) -> _SessionDouble:
        session = cls.__new__(cls)
        session._tools = [AgentTool.model_construct(name=name) for name in names]
        session.mcp_manager = manager
        return session

    def seed_tool(self, name: str) -> None:
        """Append a name to the LIVE inventory (the delayed arrival fixture)."""

        self._tools.append(AgentTool.model_construct(name=name))


class _ExplodingManager(McpManager):
    """Fails loudly if consulted -- the fast path must not wait on anything."""

    def __init__(self) -> None:
        self.calls = 0

    async def wait_settled(self, timeout_s: float) -> bool:
        self.calls += 1
        raise AssertionError("wait_settled must not be called on the fast path")


class _SettledManager(McpManager):
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


class _DelayedManager(McpManager):
    """The controlled delayed-MCP fixture.

    The tool lands after ``delay_s`` on an arrival task that ``wait_settled``
    watches -- and, exactly like the real manager, a timeout stops WATCHING
    the arrival without cancelling it, so discovery keeps moving.
    """

    def __init__(self, session: _SessionDouble, name: str, delay_s: float) -> None:
        self.session = session
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
            self.session.seed_tool(self.name)
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
    session = _SessionDouble.build(("bash",), manager)
    readiness = await session.wait_for_tools(["bash"])
    assert readiness.status == "ready"
    assert readiness.discovery_settled is None
    assert readiness.present == ("bash",)
    assert readiness.missing == ()
    assert manager.calls == 0


@pytest.mark.asyncio
async def test_delayed_discovery_is_gated_by_the_wait() -> None:
    session = _SessionDouble.build()
    manager = _DelayedManager(session, "mcp__episode_actions", 0.05)
    session.mcp_manager = manager
    # The raw race the gap class describes: an immediate read is empty even
    # though the tool is on its way.
    assert session.tool_inventory == ()
    readiness = await session.wait_for_tools(["mcp__episode_actions"], timeout_s=1.0)
    assert readiness.status == "ready"
    assert readiness.discovery_settled is True
    assert readiness.present == ("mcp__episode_actions",)
    # The first request built after this gate sees the tool.
    assert "mcp__episode_actions" in session.tool_inventory


@pytest.mark.asyncio
async def test_settled_without_the_name_is_missing() -> None:
    manager = _SettledManager(settle_result=True)
    session = _SessionDouble.build(("bash",), manager)
    readiness = await session.wait_for_tools(["mcp__gone"], timeout_s=0.5)
    assert readiness.status == "missing"
    assert readiness.discovery_settled is True
    assert readiness.missing == ("mcp__gone",)
    assert manager.timeouts == [0.5]


@pytest.mark.asyncio
async def test_missing_without_a_manager_is_missing_not_pending() -> None:
    session = _SessionDouble.build(("bash",))
    readiness = await session.wait_for_tools(["mcp__gone"])
    assert readiness.status == "missing"
    assert readiness.discovery_settled is None


@pytest.mark.asyncio
async def test_manager_failure_is_never_ready() -> None:
    manager = _SettledManager(error=RuntimeError("reduced manager"))
    session = _SessionDouble.build((), manager)
    readiness = await session.wait_for_tools(["mcp__x"], timeout_s=0.5)
    assert readiness.status == "pending"
    assert readiness.discovery_settled is None


@pytest.mark.asyncio
async def test_timeout_is_pending_and_a_later_wait_still_reads_ready() -> None:
    session = _SessionDouble.build()
    manager = _DelayedManager(session, "mcp__late", 0.15)
    session.mcp_manager = manager
    readiness = await session.wait_for_tools(["mcp__late"], timeout_s=0.01)
    assert readiness.status == "pending"
    assert readiness.discovery_settled is False
    # The timeout must not have cancelled discovery: the continuation lands.
    await asyncio.wait_for(manager.arrived.wait(), timeout=1.0)
    fresh = await session.wait_for_tools(["mcp__late"], timeout_s=1.0)
    assert fresh.status == "ready"


@pytest.mark.asyncio
async def test_cancellation_propagates_and_leaves_discovery_alive() -> None:
    session = _SessionDouble.build()
    manager = _DelayedManager(session, "mcp__slow", 0.05)
    session.mcp_manager = manager
    task = asyncio.create_task(session.wait_for_tools(["mcp__slow"], timeout_s=30.0))
    for _ in range(100):  # let the task enter the manager wait
        if manager.calls == 1:
            break
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.calls == 1
    # Discovery untouched: it completes later and a fresh wait reads ready.
    await asyncio.wait_for(manager.arrived.wait(), timeout=1.0)
    fresh = await session.wait_for_tools(["mcp__slow"], timeout_s=1.0)
    assert fresh.status == "ready"


@pytest.mark.asyncio
async def test_names_are_deduped_in_caller_order() -> None:
    session = _SessionDouble.build(("a", "b"))
    readiness = await session.wait_for_tools(["b", "a", "b"])
    assert readiness.required == ("b", "a")
    assert readiness.present == ("b", "a")


@pytest.mark.asyncio
async def test_inventory_is_never_widened() -> None:
    manager = _SettledManager(settle_result=True)
    session = _SessionDouble.build(("bash",), manager)
    before = session.tool_inventory
    readiness = await session.wait_for_tools(["mcp__typo"], timeout_s=0.5)
    assert readiness.status == "missing"
    assert session.tool_inventory == before


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
async def test_invalid_inputs_are_refused_before_any_wait(required: Any, timeout_s: float) -> None:
    manager = _ExplodingManager()
    session = _SessionDouble.build(("ok",), manager)
    with pytest.raises(ValueError):
        await session.wait_for_tools(required, timeout_s=timeout_s)
    assert manager.calls == 0
