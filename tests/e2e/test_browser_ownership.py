"""Real Session → authenticated HTTP → bundled extension; Chrome is disposable.

The extension CI installs the Node fixture dependencies and runs this module.
Python-only TUI jobs skip it rather than downloading any browser engine.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import subprocess
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.browser_bridge import state
from local_operator.browser_bridge.protocol import PROTO_VERSION
from local_operator.harness.types import (
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.session_lease import acquire_session_lease
from local_operator.tools.builtin import execute_browser
from tests.e2e.watchdog import bounded

EXTENSION = Path(__file__).resolve().parents[2] / "extension"


@pytest.fixture(autouse=True)
def bound_protocol_run() -> Iterator[None]:
    with bounded(60, "browser ownership protocol fixture"):
        yield


@pytest.fixture
def protocol_peer(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[int, str]]:
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    node = shutil.which("node")
    if not node or not (EXTENSION / "node_modules" / "esbuild").exists():
        pytest.skip(
            "run pnpm install in extension/ for disposable protocol E2E (no browser needed)"
        )
    key = secrets.token_urlsafe(32)
    env = {**os.environ, "BROWSER_FIXTURE_KEY": key}
    process = subprocess.Popen(
        [node, "tests/fixtures/ownership-server.mjs"],
        cwd=EXTENSION,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    try:
        greeting = process.stdout.readline()
        assert greeting, "disposable extension fixture failed to start"
        port = json.loads(greeting)["port"]
        state.publish(
            state.BridgeState(
                pid=process.pid,
                port=port,
                session_key=key,
                proto=PROTO_VERSION,
                extension_connected=True,
                paired=True,
            ),
            headless_tui_env,
        )
        yield port, key
    finally:
        process.terminate()
        process.wait(timeout=10)


def _control(peer: tuple[int, str], **changes: Any) -> dict[str, Any]:
    port, key = peer
    response = httpx.post(
        f"http://127.0.0.1:{port}/fixture",
        headers={"X-Bridge-Key": key},
        json=changes,
    )
    response.raise_for_status()
    return response.json()


def _session(root: Path) -> Session:
    directory = root / "sessions" / "synthetic-browser"
    lease = acquire_session_lease(directory)

    def stream(_request: Any, _signal: Any) -> Any:
        async def events() -> Any:
            if False:
                yield None

        return events()

    session = Session(
        model=ModelSpec(provider="test", model_id="synthetic", context_window=1000),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: [],
    )
    session.add_dispose_hook(lease.release)
    return session


@pytest.mark.asyncio
async def test_failed_navigation_then_completion_uses_real_protocol(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    session = _session(headless_tui_env)
    try:
        port, _key = protocol_peer
        denied = httpx.post(f"http://127.0.0.1:{port}/rpc", json={})
        assert denied.status_code == 401
        _control(protocol_peer, faults={"navigation": True})
        failed = await execute_browser(
            "synthetic-fail",
            {"action": "open", "url": "https://example.test/"},
            None,
            None,
            session._build_tool_context(),
        )
        assert failed.is_error
        assert _control(protocol_peer)["tabs"] == 0
        _control(protocol_peer, faults={"navigation": False})
        opened = await execute_browser(
            "synthetic-open",
            {"action": "open", "url": "https://example.test/"},
            None,
            None,
            session._build_tool_context(),
        )
        assert not opened.is_error, opened.text
        assert _control(protocol_peer)["tabs"] == 1
        finished = await session.finish_browser_scope(
            scope_id=session.session_id, generation=session.browser_generation, outcome="completed"
        )
        assert finished.state == "closed"
        assert _control(protocol_peer)["tabs"] == 0
        assert session._browser.resource.record["terminal"] == "completed"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_failed_close_retention_resume_and_release(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    session = _session(headless_tui_env)
    try:
        opened = await execute_browser(
            "synthetic-open",
            {"action": "open", "url": "https://example.test/"},
            None,
            None,
            session._build_tool_context(),
        )
        assert not opened.is_error, opened.text
        _control(protocol_peer, faults={"remove": True})
        closed = await execute_browser(
            "synthetic-close",
            {"action": "close"},
            None,
            None,
            session._build_tool_context(),
        )
        assert closed.is_error and "retained" in closed.text
        assert session._browser.surface_id
        retained = await execute_browser(
            "synthetic-retain",
            {"action": "retain", "text": "pending login"},
            None,
            None,
            session._build_tool_context(),
        )
        assert not retained.is_error, retained.text
    finally:
        await session.dispose()
    assert _control(protocol_peer)["tabs"] == 1
    _control(protocol_peer, faults={"remove": False}, restartWorker=True)
    resumed = _session(headless_tui_env)
    try:
        recovered = await execute_browser(
            "synthetic-recover",
            {"action": "recover"},
            None,
            None,
            resumed._build_tool_context(),
        )
        assert not recovered.is_error, recovered.text
        assert resumed._browser.surface_id
        assert _control(protocol_peer)["tabs"] == 1
        # A stale child must still publish its outcome: generation lookup is
        # pure and the finalizer returns unresolved without touching its successor.
        stale = await session.finish_browser_scope(
            scope_id=session.session_id,
            generation=session.browser_generation,
            outcome="failed",
        )
        assert stale.state == "unresolved"
        assert _control(protocol_peer)["tabs"] == 1
        assert not resumed._browser.resource.record.get("terminal")
        released = await execute_browser(
            "synthetic-release",
            {"action": "release"},
            None,
            None,
            resumed._build_tool_context(),
        )
        assert not released.is_error, released.text
        assert (
            await resumed.finish_browser_scope(
                scope_id=resumed.session_id,
                generation=resumed.browser_generation,
                outcome="completed",
            )
        ).state == "closed"
        assert _control(protocol_peer)["tabs"] == 0
    finally:
        await resumed.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
async def test_child_finalizes_before_terminal_handoff(
    protocol_peer: tuple[int, str],
    headless_tui_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    from local_operator.harness import subagent

    calls = 0
    opened = asyncio.Event()
    terminal_tab_counts: list[int] = []
    allocated_tab_counts: list[int] = []
    publish = subagent._publish_terminal_outcome

    async def observed_publish(*args: Any, **kwargs: Any) -> Any:
        terminal_tab_counts.append(_control(protocol_peer)["tabs"])
        return await publish(*args, **kwargs)

    monkeypatch.setattr(subagent, "_publish_terminal_outcome", observed_publish)

    async def stream(request: Any, signal: Any = None) -> AsyncIterator[Any]:
        nonlocal calls
        if not any("BROWSER_CHILD" in getattr(message, "text", "") for message in request.messages):
            yield StreamTextDelta(delta="Acknowledged")
            yield StreamEndEvent(stop_reason="stop")
            return
        calls += 1
        if calls == 1:
            yield StreamToolCallDelta(
                index=0,
                id="synthetic-child-open",
                name="browser",
                argument_delta=json.dumps(
                    {
                        "action": "open",
                        "url": "https://example.test/",
                        "i": "Opening isolated fixture",
                    }
                ),
            )
            yield StreamEndEvent(stop_reason="toolUse")
            return
        allocated_tab_counts.append(_control(protocol_peer)["tabs"])
        opened.set()
        if outcome == "failed":
            raise RuntimeError("controlled child failure after browser allocation")
        if outcome == "cancelled":
            await asyncio.Event().wait()
        yield StreamTextDelta(delta="Completed child")
        yield StreamEndEvent(stop_reason="stop")

    async def approve(*_args: Any, **_kwargs: Any) -> bool:
        return True

    owner = Session(
        model=ModelSpec(provider="test", model_id="synthetic", context_window=100000),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(headless_tui_env / "sessions" / "synthetic-parent"),
        system_blocks_provider=lambda *_: [],
        yolo=True,
        cwd=str(headless_tui_env),
        request_approval=approve,
    )
    try:
        job_id = owner._launch_subagent("browser-child", "BROWSER_CHILD")
        if outcome == "cancelled":
            await asyncio.wait_for(opened.wait(), 20)
            assert _control(protocol_peer)["tabs"] == 1
            await owner.jobs.cancel(job_id)
        await asyncio.wait_for(owner.jobs.settled_event(job_id).wait(), 25)
        row = owner.jobs.get(job_id)
        assert row is not None and row.status == outcome
        assert calls >= 2, "child must actually execute the browser tool"
        assert allocated_tab_counts == [1]
        assert terminal_tab_counts == [0]
        assert _control(protocol_peer)["tabs"] == 0
    finally:
        await owner.dispose()


@pytest.mark.asyncio
async def test_old_extension_is_actionable_before_any_allocation(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    session = _session(headless_tui_env)
    try:
        _control(protocol_peer, oldExtension=True)
        result = await execute_browser(
            "synthetic-old",
            {"action": "open", "url": "https://example.test/"},
            None,
            None,
            session._build_tool_context(),
        )
        assert result.is_error
        assert "Update the extension" in result.text
        assert _control(protocol_peer)["tabs"] == 0
    finally:
        await session.dispose()


def _unleased_session(root: Path) -> Session:
    """A child as ``_build_child_session`` builds one: claimed, never leased.

    The distinction is the whole point of this guard. Round 2 established that
    an in-process child reuses ONE generation per session so a second live
    instance cannot fence the incumbent; the leased ``_session`` above rotates
    its generation on resume and therefore cannot observe anything that depends
    on the generation staying the same.
    """
    from local_operator.session.retention import claim_session

    directory = root / "sessions" / "synthetic-child"
    claim_session(directory)

    def stream(_request: Any, _signal: Any) -> Any:
        async def events() -> Any:
            if False:
                yield None

        return events()

    return Session(
        model=ModelSpec(provider="test", model_id="synthetic", context_window=1000),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: [],
    )


@pytest.mark.asyncio
async def test_resumed_unleased_child_can_open_again_across_the_bridge(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    """A finalized subagent must be able to browse on a later run.

    This crosses the WIRE deliberately. The unit guard for the same invariant
    stops at ``allocate()``, which is a local sidecar write, so it could not
    observe that the extension keeps its own copy of the terminal intent and
    used to clear it only when the generation string changed — which the
    unleased path never does. Both halves of the fence have to be retired, and
    only a real ``open`` through the fixture proves it: the tab count is the
    assertion, because a refusal here still returns a well-formed error.
    """
    first = _unleased_session(headless_tui_env)
    try:
        opened = await execute_browser(
            "child-run1-open",
            {"action": "open", "url": "https://example.test/"},
            None,
            None,
            first._build_tool_context(),
        )
        assert not opened.is_error, opened.text
        assert _control(protocol_peer)["tabs"] == 1
        finished = await first.finish_browser_scope(
            scope_id=first.session_id,
            generation=first.browser_generation,
            outcome="completed",
        )
        assert finished.state == "closed"
        assert _control(protocol_peer)["tabs"] == 0
    finally:
        await first.dispose()

    # `hub op='resume'` relaunches the child over its own transcript directory.
    resumed = _unleased_session(headless_tui_env)
    try:
        assert resumed.browser_generation == first.browser_generation, "B1 reuses the generation"
        reopened = await execute_browser(
            "child-run2-open",
            {"action": "open", "url": "https://example.test/"},
            None,
            None,
            resumed._build_tool_context(),
        )
        assert not reopened.is_error, reopened.text
        # Run 2 allocated a REAL tab; the round-2 head reported 0 here.
        assert _control(protocol_peer)["tabs"] == 1
    finally:
        await resumed.dispose()
