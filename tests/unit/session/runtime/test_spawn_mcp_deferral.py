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
