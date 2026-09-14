"""The runtime child wires MCP OFF its pre-record path, and still reports.

The TUI's first frame is a viewer waiting on an owner record that a separate
process publishes. Everything that process does before ``RecordPublisher`` runs
sits between the user and a bound session, and MCP discovery is the largest
piece of it that has no business being there: it dials every configured server,
runs the 250 ms gate, and on a machine with an authenticating or hanging server
can spend a connect timeout before the gate defers it — for tools MCP itself
deliberately does not advertise until the model asks for them.

``defer_mcp_wiring`` is the mechanism for exactly this, written when the TUI
built its own in-process session. After the viewer/runtime split the process
whose boot the first frame waits on is the spawned child, and it was the one
caller not opting in — so the deferred branch was unreachable in production.
These tests pin the opt-in and the ordering it buys, which is the half a later
change could silently undo.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from tests.unit.test_session_factory import FakeMcpManager

#: Upper bound on an awaited event, never a budget to sleep through.
GUARD_S = 20.0

#: Loop TURNS used to assert something has NOT happened yet. A turn count rather
#: than a sleep for the reason ``AGENTS.md`` gives: a silent window measured in
#: seconds is a bet on machine load, and on this project's hosts the work being
#: waited out is exactly the work that stretches under load.
NEGATIVE_TURNS = 200

BROKEN_SERVER = "broken"
BROKEN_ERROR = "command not found: definitely-not-installed"


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Config, home and cwd out of the way of the developer's real ones."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


async def _wait_for_record(config_dir: Path, session_id: str, timeout: float = GUARD_S) -> Any:
    from local_operator.session.runtime import registry

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for record, _state in registry.scan(config_dir):
            if getattr(record, "session_id", "") == session_id:
                return record
        await asyncio.sleep(0.02)
    raise AssertionError(f"no record published for {session_id} within {timeout}s")


@pytest.mark.asyncio
async def test_the_runtime_child_asks_for_deferred_mcp_wiring(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one caller whose boot a user waits on must opt in.

    Asserted on the CALL rather than on a downstream effect, because the call is
    the opt-in and a future refactor that drops the flag would leave every other
    test here green: the eager branch is still correct, it is merely back on the
    critical path.
    """
    from local_operator import session_factory
    from local_operator.session.runtime.serving import spawn_owned_session

    calls: list[dict[str, Any]] = []
    real_create = session_factory.create_session

    async def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return await real_create(*args, **kwargs)

    async def no_wiring(session: Any, tools: Any, cwd: str, **kwargs: Any) -> None:
        # Returning None is the manager-less degrade; this test is about the
        # flag, and letting the deferral actually happen keeps it fast.
        await asyncio.sleep(0)
        return None

    monkeypatch.setattr(session_factory, "create_session", spy)
    monkeypatch.setattr(session_factory, "wire_mcp_into_session", no_wiring)

    handle = await asyncio.wait_for(
        spawn_owned_session(
            asyncio.get_running_loop(),
            cwd=str(isolated_config),
            provider="test",
            model_id="mock",
        ),
        timeout=GUARD_S,
    )
    try:
        assert calls, "spawn_owned_session never reached the composition root"
        assert calls[0].get("defer_mcp_wiring") is True, (
            "the runtime child must wire MCP off its pre-record path: the whole "
            "point of the viewer/runtime split is that nothing a user waits on "
            "lives behind an integration"
        )
        # Unchanged, and named here because the deferral's announcement routing
        # depends on it: the child has no full-screen terminal to write over, so
        # its MCP failures stay on stderr (the capture file) as well as in the
        # recorded outcome.
        assert calls[0].get("has_ui") is False
    finally:
        await handle._session.dispose()


@pytest.mark.asyncio
async def test_the_record_is_published_without_waiting_for_mcp_wiring(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record does not wait on MCP, proved by holding the wiring open.

    A deterministic ordering proof rather than a timing one: the wiring is
    blocked on an event the test controls, so the record's presence and the
    wiring's unfinished state are observed in the same instant. On the eager
    path this test cannot even reach its first assertion — ``create_session``
    awaits the wiring, so ``spawn_owned_session`` never returns.
    """
    from local_operator import session_factory
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import spawn_owned_session

    entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_wiring(session: Any, tools: Any, cwd: str, **kwargs: Any) -> None:
        entered.set()
        await release.wait()
        return None

    monkeypatch.setattr(session_factory, "wire_mcp_into_session", gated_wiring)

    handle = await asyncio.wait_for(
        spawn_owned_session(
            asyncio.get_running_loop(),
            cwd=str(isolated_config),
            provider="test",
            model_id="mock",
        ),
        timeout=GUARD_S,
    )
    session = handle._session
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    try:
        assert await asyncio.wait_for(entered.wait(), timeout=GUARD_S)
        assert not release.is_set(), "the wiring gate must still be closed"

        record = await _wait_for_record(isolated_config, session.session_id)
        assert record.session_id == session.session_id
        assert not release.is_set(), (
            "the record was published while MCP discovery was still held open — "
            "which is the change under test"
        )
    finally:
        release.set()
        server.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_the_deferred_round_still_reports_its_outcome_to_a_subscriber(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report is not lost by moving the wiring: it rides the state push.

    This is the half a reviewer should attack, and it is a real risk rather than
    a theoretical one: a full-TUI viewer learns MCP state from the frontend-state
    PUSH, and the seed it gets at attach is snapshotted at subscribe time. A
    viewer that binds before the wiring has finished therefore knows nothing
    about MCP — which is exactly what deferring invites — so the outcome has to
    reach it afterwards.

    It does: ``attach_mcp_dispose`` refreshes the frontend state once the manager
    exists (and the settle callback re-reports when a round drains late). This
    pins that hop by subscribing a handler the way a viewer's socket does, and
    asserts the failure arrives AFTER the subscription.
    """
    from local_operator import session_factory
    from local_operator.session_factory import create_session

    entered = asyncio.Event()
    release = asyncio.Event()
    real_wire = session_factory.wire_mcp_into_session

    async def gated_wiring(session: Any, tools: Any, cwd: str, **kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        return await real_wire(session, tools, cwd, **kwargs)

    async def fake_discover(cwd: str, auth_store: Any = None) -> Any:
        return (
            FakeMcpManager(configured=[BROKEN_SERVER], connected=[]),
            [],
            [{"path": f"mcp:{BROKEN_SERVER}", "error": BROKEN_ERROR}],
        )

    monkeypatch.setattr(session_factory, "wire_mcp_into_session", gated_wiring)
    monkeypatch.setattr("local_operator.mcp.discover_and_load_mcp_tools", fake_discover)

    import argparse

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.session_factory import await_store_maintenance_for_tests

    args = argparse.Namespace(
        hosting="test",
        model="mock",
        agent_name=None,
        agent_id=None,
        yolo=True,
        train=False,
    )
    session = await create_session(
        args,
        ConfigManager(isolated_config),
        CredentialManager(isolated_config),
        AgentRegistry(isolated_config),
        has_ui=False,
        cwd=str(isolated_config),
        defer_mcp_wiring=True,
    )
    try:
        # Off the boot path: the caller has the session and the wiring has not
        # run. This is the property the record's early publication rests on.
        assert getattr(session, "mcp_startup", None) is None
        assert await asyncio.wait_for(entered.wait(), timeout=GUARD_S)

        pushed: list[Any] = []
        subscription = cast(Any, session).subscribe_frontend(pushed.append)
        # The seed a viewer binds with, snapshotted at subscribe time. It carries
        # no MCP outcome — the wiring has not finished — which is precisely the
        # race the report has to survive.
        assert subscription.sync.snapshot.mcp_startup is None

        release.set()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + GUARD_S
        while loop.time() < deadline:
            if any("mcp_startup" in update.changes for update in pushed):
                break
            await asyncio.sleep(0.02)
        # Asserted on the pushed DELTA, because that is the payload a viewer's
        # socket relays — proving the outcome reached a subscriber, not merely
        # that it was recorded on the session.
        reported = [
            update.changes["mcp_startup"] for update in pushed if "mcp_startup" in update.changes
        ]
        assert reported, (
            "the MCP outcome never reached a subscriber that was watching before "
            "the wiring finished; a viewer bound this early would see nothing"
        )
        assert reported[-1]["failures"] == {BROKEN_SERVER: BROKEN_ERROR}
        # And the record itself is unchanged: same failures, same bare server key.
        startup = cast(Any, session).mcp_startup
        assert startup is not None
        assert startup.failures == {BROKEN_SERVER: BROKEN_ERROR}
    finally:
        release.set()
        await await_store_maintenance_for_tests()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_degradation_arm_also_pushes_the_outcome_to_a_subscriber(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1, R2 — the arms WITHOUT a manager are the ones that lost it.

    ``attach_mcp_dispose`` refreshes the frontend store, and it only runs when
    ``wire_mcp_into_session`` returned a manager. Discovery raising and the MCP
    layer failing to import both return ``None``, so on the deferred path the
    child recorded an outcome that no viewer could ever read: a viewer bound
    before the wiring kept the empty snapshot it was seeded with. The same code
    on the eager path told that viewer correctly, because the record waited for
    the wiring and the seed carried the outcome — so this is a regression the
    deferral introduced, not a pre-existing gap.
    """

    async def exploding_discovery(cwd: str, auth_store: Any = None) -> Any:
        raise RuntimeError("discovery exploded")

    monkeypatch.setattr("local_operator.mcp.discover_and_load_mcp_tools", exploding_discovery)

    import argparse

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.session_factory import (
        await_store_maintenance_for_tests,
        create_session,
    )

    args = argparse.Namespace(
        hosting="test",
        model="mock",
        agent_name=None,
        agent_id=None,
        yolo=True,
        train=False,
    )
    session = await create_session(
        args,
        ConfigManager(isolated_config),
        CredentialManager(isolated_config),
        AgentRegistry(isolated_config),
        has_ui=False,
        cwd=str(isolated_config),
        defer_mcp_wiring=True,
    )
    try:
        pushed: list[Any] = []
        subscription = cast(Any, session).subscribe_frontend(pushed.append)
        assert subscription.sync.snapshot.mcp_startup is None

        loop = asyncio.get_running_loop()
        deadline = loop.time() + GUARD_S
        while loop.time() < deadline:
            if any("mcp_startup" in update.changes for update in pushed):
                break
            await asyncio.sleep(0.02)
        reported = [
            update.changes["mcp_startup"] for update in pushed if "mcp_startup" in update.changes
        ]
        assert reported, (
            "the degradation arm recorded an outcome no viewer could read: no push "
            "carried mcp_startup after discovery raised"
        )
        assert reported[-1]["failures"] == {"discovery": "discovery exploded"}
    finally:
        await await_store_maintenance_for_tests()
        await session.dispose()


@pytest.mark.asyncio
async def test_the_in_process_path_wires_mcp_without_any_publisher(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The TUI's own Session never publishes a record, and must still wire MCP.

    This is the failure mode a publication gate invites, and it is not
    hypothetical: the TUI builds an in-process Session with
    ``defer_mcp_wiring=True`` and there is no ``RuntimeServer`` anywhere in that
    process to open a latch. The gate is therefore OPT-IN — ``None`` means "no
    publisher", and the deferred task is dispatched ungated exactly as it was
    before the parameter existed — which this pins by settling
    ``session.mcp_startup`` with no gate and no server in sight.
    """
    from local_operator.session_factory import (
        await_store_maintenance_for_tests,
        create_session,
    )

    async def fake_discover(cwd: str, auth_store: Any = None) -> Any:
        return (
            FakeMcpManager(configured=[BROKEN_SERVER], connected=[]),
            [],
            [{"path": f"mcp:{BROKEN_SERVER}", "error": BROKEN_ERROR}],
        )

    monkeypatch.setattr("local_operator.mcp.discover_and_load_mcp_tools", fake_discover)

    import argparse

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager

    args = argparse.Namespace(
        hosting="test",
        model="mock",
        agent_name=None,
        agent_id=None,
        yolo=True,
        train=False,
    )
    session = await create_session(
        args,
        ConfigManager(isolated_config),
        CredentialManager(isolated_config),
        AgentRegistry(isolated_config),
        has_ui=False,
        cwd=str(isolated_config),
        defer_mcp_wiring=True,
    )
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + GUARD_S
        while loop.time() < deadline:
            if getattr(session, "mcp_startup", None) is not None:
                break
            await asyncio.sleep(0.02)
        startup = getattr(session, "mcp_startup", None)
        assert startup is not None, (
            "an ungated deferred session never settled its MCP outcome: the "
            "in-process/TUI path has no publisher to open a gate, so a gate that "
            "applied to it would mean MCP was never wired at all"
        )
        assert startup.failures == {BROKEN_SERVER: BROKEN_ERROR}
    finally:
        await await_store_maintenance_for_tests()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_gated_wiring_parks_until_the_latch_is_set(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate means the wiring WAITS, it does not merely get dispatched early.

    Deterministic, not timed: the loop is given many turns with the latch closed
    and the wiring must still not have run, which is the ordering the whole
    change rests on — the task's first instruction is a synchronous SDK import
    that would otherwise take the loop during ``process.amain``'s inbox drain,
    before the record exists.
    """
    from local_operator import session_factory
    from local_operator.session_factory import (
        await_store_maintenance_for_tests,
        create_session,
    )

    entered = asyncio.Event()
    gate = asyncio.Event()

    async def spy(session: Any, tools: Any, cwd: str, **kwargs: Any) -> Any:
        entered.set()
        return None

    monkeypatch.setattr(session_factory, "wire_mcp_into_session", spy)

    import argparse

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager

    args = argparse.Namespace(
        hosting="test",
        model="mock",
        agent_name=None,
        agent_id=None,
        yolo=True,
        train=False,
    )
    session = await create_session(
        args,
        ConfigManager(isolated_config),
        CredentialManager(isolated_config),
        AgentRegistry(isolated_config),
        has_ui=False,
        cwd=str(isolated_config),
        defer_mcp_wiring=True,
        mcp_publication_gate=gate,
    )
    try:
        for _ in range(NEGATIVE_TURNS):
            await asyncio.sleep(0)
        assert not entered.is_set(), (
            "the deferred wiring ran while its publication latch was closed — a "
            "gated session would then pay the MCP SDK import in front of its own "
            "record, which is the defect the latch exists to prevent"
        )

        gate.set()
        assert await asyncio.wait_for(
            entered.wait(), timeout=GUARD_S
        ), "setting the latch must release the wiring"
    finally:
        await await_store_maintenance_for_tests()
        await session.dispose()


@pytest.mark.asyncio
async def test_the_runtime_child_gates_its_wiring_on_publication(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child's wiring starts AFTER its record exists, in the real spawn path.

    ``spawn_owned_session`` is the only spawn site whose runtime publishes, so it
    is the only one that hands out a latch. The ordering is asserted across the
    same boundary production uses: the engine's boot (here: just loop turns)
    runs with the record not yet published, then ``start_in_process`` publishes
    and the wiring starts.
    """
    from local_operator import session_factory
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import spawn_owned_session

    entered = asyncio.Event()

    async def spy(session: Any, tools: Any, cwd: str, **kwargs: Any) -> Any:
        entered.set()
        return None

    monkeypatch.setattr(session_factory, "wire_mcp_into_session", spy)

    handle = await asyncio.wait_for(
        spawn_owned_session(
            asyncio.get_running_loop(),
            cwd=str(isolated_config),
            provider="test",
            model_id="mock",
        ),
        timeout=GUARD_S,
    )
    session = handle._session
    server = RuntimeServer(handle, kind="daemon")
    try:
        gate = handle.mcp_publication_gate
        assert gate is not None, (
            "the runtime child did not pass a publication latch, so its deferred "
            "wiring is ungated and can run inside its pre-publication window"
        )
        assert not gate.is_set()

        # The production boot between construction and publication: the drain
        # and async_init. Those awaits are where the task used to start.
        for _ in range(NEGATIVE_TURNS):
            await asyncio.sleep(0)
        assert not entered.is_set(), (
            "the wiring started before the record was published, so its SDK "
            "import is inside the window the user is waiting through"
        )

        await server.start_in_process()
        assert gate.is_set(), "publication must open the latch"
        assert await asyncio.wait_for(
            entered.wait(), timeout=GUARD_S
        ), "the published record must release the wiring"
    finally:
        server.close()
        await session.dispose()
