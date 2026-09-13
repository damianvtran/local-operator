"""The runtime child's record leaves before its MCP wiring, and MCP still reports.

**Two properties, and the second is the one that is easy to get wrong.** The
structural half is that the owner record — the thing every viewer binds to —
must not be published behind MCP discovery, because pagers, OAuth handshakes and
a hanging server's connect timeout all live in that work and none of them is
allowed to decide when a session becomes reachable. The second half is that the
MCP FAILURE REPORT must survive the move: on this path the report rides the
frontend-state push, so it needs a viewer that is already bound when the wiring
finishes — which is exactly what happens now that the record comes first.

The first assertion is the deferral itself. The second is the honest version of
"the report still lands": measured against a real socket, with the wiring held
open so the ordering is deterministic instead of a race between a ~300 ms
wiring pass and the viewer's dial.

Discovery is the only thing doubled (``FakeMcpManager`` + a failing server
entry), the same double the factory's own tests use, because a real broken
``.mcp.json`` would make the failure timing host-dependent. ``spawn_owned_session``,
``RuntimeServer``, ``ServingSessionHandle``, the socket and ``AttachedSession``
are the production objects.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_session_factory import FakeMcpManager

pytestmark = pytest.mark.e2e

#: Upper bound on an awaited event, never a budget to sleep through.
GUARD_S = 20.0

#: The failure the doubled discovery reports. Named like a real one — a server
#: whose configured command does not exist is the commonest broken setup and is
#: terminal at the 250 ms gate, so nothing here waits out a connect timeout.
BROKEN_SERVER = "broken"
BROKEN_ERROR = "command not found: definitely-not-installed"


async def _never_take_over() -> Any:
    """A viewer must never take a session over in these tests."""
    raise AssertionError("a viewer must never take over a session")


async def _wait_for_record(config_dir: Path, session_id: str, timeout: float = GUARD_S) -> Any:
    """The record the runtime publishes, once it is discoverable."""
    from local_operator.session.runtime import registry

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for record, _state in registry.scan(config_dir):
            if getattr(record, "session_id", "") == session_id:
                return record
        await asyncio.sleep(0.05)
    raise AssertionError(f"no record published for {session_id} within {timeout}s")


async def _wait_for(predicate: Any, timeout: float = GUARD_S) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return predicate()


@pytest.mark.asyncio
async def test_the_record_is_published_before_mcp_wiring_and_mcp_still_reports(
    headless_tui_env: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A viewer binds while MCP discovery is still running, and still hears the failure.

    On the branch before this one the first assertion cannot even be reached:
    ``create_session`` awaits the wiring, so a held-open wiring pass means
    ``spawn_owned_session`` never returns and no record ever exists — which is
    the defect, stated as a test.
    """
    from local_operator import session_factory
    from local_operator.session.attached import AttachedSession
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import spawn_owned_session

    wiring_entered = asyncio.Event()
    release_wiring = asyncio.Event()
    real_wire = session_factory.wire_mcp_into_session

    async def gated_wire(session: Any, tools: Any, cwd: str, **kwargs: Any) -> Any:
        """The production wiring, held open so the ordering is deterministic.

        Everything past the gate is the real function: the outcome record, the
        ``_deferred_boot`` sink hop and the manager's settle callback all run as
        they do in production.
        """
        wiring_entered.set()
        await release_wiring.wait()
        return await real_wire(session, tools, cwd, **kwargs)

    async def fake_discover(cwd: str, auth_store: Any = None) -> Any:
        return (
            FakeMcpManager(configured=[BROKEN_SERVER], connected=[]),
            [],
            [{"path": f"mcp:{BROKEN_SERVER}", "error": BROKEN_ERROR}],
        )

    monkeypatch.setattr(session_factory, "wire_mcp_into_session", gated_wire)
    monkeypatch.setattr("local_operator.mcp.discover_and_load_mcp_tools", fake_discover)

    handle = await asyncio.wait_for(
        spawn_owned_session(
            asyncio.get_running_loop(),
            cwd=str(workspace),
            provider="test",
            model_id="mock",
        ),
        timeout=GUARD_S,
    )
    # ``ServingSessionHandle`` holds the session privately (it is the owner's own
    # object, reached only through the handle in production); the test needs it to
    # read the record's identity and to dispose cleanly.
    session = handle._session
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    viewer = None
    try:
        assert await asyncio.wait_for(wiring_entered.wait(), timeout=GUARD_S)
        assert not release_wiring.is_set(), "the wiring gate must still be closed"

        # ---- the deferral -------------------------------------------------
        record = await _wait_for_record(headless_tui_env, session.session_id)
        viewer = await asyncio.wait_for(
            AttachedSession.connect(
                record,
                session.session_id,
                config_dir=headless_tui_env,
                takeover_factory=_never_take_over,
            ),
            timeout=GUARD_S,
        )
        assert not release_wiring.is_set(), (
            "the viewer bound while MCP discovery was still held open, and the "
            "record existed first — that is the change under test"
        )
        # Nothing to report yet, and that is the point: the bind did not wait
        # for a round that has not run.
        assert getattr(viewer, "mcp_startup", None) is None

        # ---- the report survives the move ---------------------------------
        release_wiring.set()
        assert await _wait_for(
            lambda: getattr(viewer, "mcp_startup", None) is not None
        ), "the MCP outcome never reached a viewer that was bound before wiring finished"
        outcome = viewer.mcp_startup
        assert outcome is not None, "the outcome arrived but read back as None"
        assert outcome.failures == {BROKEN_SERVER: BROKEN_ERROR}, (
            "the failure the runtime recorded is not what the viewer was told: "
            f"{outcome.failures!r}"
        )
    finally:
        release_wiring.set()
        if viewer is not None:
            with contextlib.suppress(Exception):
                await viewer.dispose()
        server.close()
        with contextlib.suppress(Exception):
            await session.dispose()
