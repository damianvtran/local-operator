"""The viewer→runtime attach, over a real socket, end to end.

**This file exists because the whole suite stayed green through a total
product outage.** Round 1 QA and UX both found, against the real binary, that
no message could be sent in any session on this branch: `RuntimeServer`
advertises `tui_state_v1` only when its handle has `subscribe_frontend`, that
method lived only on the owner-path class this PR deletes, so every runtime
published `capabilities: []` and hung up on every viewer.

189 unit tests and 4 e2e tests passed anyway, for two structural reasons:

* `tests/unit/session/runtime/test_server.py`'s stub handle DECLARES
  `subscribe_frontend`, so the server tests exercise a capability the
  production handle did not have; and
* `tests/e2e/test_tui_e2e.py` injects an in-process `Session` straight into
  `OperatorApp`, so it never performs an attach at all.

Both are reasonable in isolation and together they left the seam this release
is *about* with no coverage. So the rule these tests follow is: **the
production handle class, the production server, a real loopback socket, and
the production `RemoteSession` client.** Nothing here may substitute a stub
for the object under test — a stub that declares the capability is precisely
what hid the defect.

Kept out of `tests/unit` deliberately: it binds a socket and drives two
asyncio components against each other, which is the e2e stage's job.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime import registry
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from tests.e2e.harness import ScriptedStream, build_session, text_turn

pytestmark = pytest.mark.e2e


async def _never_take_over() -> Any:
    raise AssertionError("a viewer must never take over a session")


async def _wait_for_record(config_dir: Path, session_id: str, timeout: float = 10.0) -> Any:
    """The record the runtime publishes, once it is discoverable."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        for record, _state in registry.scan(config_dir):
            if getattr(record, "session_id", "") == session_id:
                return record
        await asyncio.sleep(0.05)
    raise AssertionError(f"no record published for {session_id} within {timeout}s")


async def _runtime(
    directory: Path, replies: list[str]
) -> tuple[Any, OwnedSessionHandle, RuntimeServer]:
    """A real Session behind the production handle behind the production server."""

    # ScriptedStream is the harness's own provider double: one async
    # generator per model call, the same shape the session's real provider
    # contract has. A bare list is not iterable with `async for`.
    stream = ScriptedStream([text_turn(reply) for reply in replies] or [text_turn("ok")])
    session = build_session(directory, stream)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    return session, handle, server


@pytest.mark.asyncio
async def test_the_runtime_advertises_the_full_tui_capability(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The one-line regression guard for the round-1 blocker.

    `capabilities: []` is what refused every viewer. Asserted on the RECORD
    the server publishes rather than on the handle, because the record is what
    a client reads and `server.py` gates the handshake on it.
    """
    directory = headless_tui_env / "sessions" / "capsess00001"
    directory.mkdir(parents=True)
    session, _handle, server = await _runtime(directory, [])
    await server.start_in_process()
    try:
        record = await _wait_for_record(headless_tui_env, session.session_id)
        assert "tui_state_v1" in record.capabilities, (
            "the runtime must advertise the full-TUI capability; without it "
            "server.py hangs up on every viewer and no message can be sent"
        )
    finally:
        server.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_viewer_attaches_over_a_real_socket_and_runs_a_turn(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The full path the product depends on: attach, prompt, stream, persist.

    This is the test whose absence let the outage ship. It performs the exact
    handshake `RemoteSession` performs in production (`frontend_state=True`),
    against the exact handle the runtime builds, and then drives a turn to a
    durable transcript row.
    """
    from local_operator.session.remote import RemoteSession

    directory = headless_tui_env / "sessions" / "attachsess01"
    directory.mkdir(parents=True)
    session, _handle, server = await _runtime(directory, ["Hello from the runtime."])
    await server.start_in_process()
    viewer = None
    try:
        record = await _wait_for_record(headless_tui_env, session.session_id)

        # The production client, asking for what the production viewer asks
        # for. Before the fix this raised ConnectionError("owner closed the
        # connection") right here.
        viewer = await RemoteSession.connect(
            record,
            session.session_id,
            config_dir=headless_tui_env,
            takeover_factory=_never_take_over,
        )

        assert viewer.frontend_state is not None, "the attach carried no state seed"

        await viewer.prompt("hello there")

        # The turn ran on the RUNTIME and reached its durable transcript,
        # which is the property a viewer cannot fake: the reply is not in the
        # viewer's memory, it is on disk in the runtime's session directory.
        transcript_path = directory / "transcript.jsonl"
        for _ in range(200):
            if transcript_path.exists() and "Hello from the runtime." in transcript_path.read_text(
                encoding="utf-8"
            ):
                break
            await asyncio.sleep(0.05)
        else:  # pragma: no cover - only on a real regression
            written = (
                transcript_path.read_text(encoding="utf-8")
                if transcript_path.exists()
                else "<absent>"
            )
            raise AssertionError(f"the runtime never wrote the assistant reply; got {written}")
    finally:
        if viewer is not None:
            await viewer.dispose()
        server.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_the_event_relay_reaches_an_attached_viewer(
    headless_tui_env: Path, workspace: Path
) -> None:
    """`subscribe_events` is the other half of the capability.

    Its absence is quieter than the handshake failure and would survive a fix
    that only restored `subscribe_frontend`: the attach succeeds, and then
    nothing ever streams. Asserted on events the VIEWER received, so it fails
    if the relay is wired but not delivering.
    """
    from local_operator.session.remote import RemoteSession

    directory = headless_tui_env / "sessions" / "relaysess001"
    directory.mkdir(parents=True)
    session, _handle, server = await _runtime(directory, ["Streamed reply."])
    await server.start_in_process()
    viewer = None
    try:
        record = await _wait_for_record(headless_tui_env, session.session_id)
        viewer = await RemoteSession.connect(
            record,
            session.session_id,
            config_dir=headless_tui_env,
            takeover_factory=_never_take_over,
        )
        seen: list[Any] = []
        viewer.subscribe(seen.append)

        await viewer.prompt("say something")

        for _ in range(200):
            if seen:
                break
            await asyncio.sleep(0.05)
        assert seen, "no AgentEvent reached the viewer; the v4 relay is dead"
    finally:
        if viewer is not None:
            await viewer.dispose()
        server.close()
        await session.dispose()
