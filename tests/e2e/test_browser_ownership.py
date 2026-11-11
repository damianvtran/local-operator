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
from tests.e2e.harness import NO_NOTIFY_ENV
from tests.e2e.watchdog import bounded

EXTENSION = Path(__file__).resolve().parents[2] / "extension"

#: The version the disposable peer reports. Below the floor named by
#: `OWNERSHIP_MIN_EXTENSION_VERSION` on purpose (`0.1.9` as of review R1-1):
#: `oldExtension=true` makes every `owner_*` method throw a bare `internal`,
#: which for a PRE-OWNERSHIP build is literally true (it has no such methods),
#: so the two halves of the fixture agree about what peer it is. Kept as a
#: literal rather than imported so the fixture keeps pinning a version that is
#: genuinely below any floor the runtime may move to.
EXTENSION_VERSION = "0.1.8"


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
    env = {**os.environ, "BROWSER_FIXTURE_KEY": key, **NO_NOTIFY_ENV}
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
                # The peer's OWN reported identity, exactly as a real `hello`
                # stamps it. Load-bearing since the version-skew change: the
                # runtime decides whether a bare `internal` from `owner_*` means
                # "this build predates ownership" (degrade and keep working) or
                # "this worker has stopped answering" (the OFF/ON remedy) from
                # these two fields, and it reads them from the DISCOVERY FILE
                # because a session-side consumer may not open a socket.
                extension_version=EXTENSION_VERSION,
                extension_proto=PROTO_VERSION,
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
async def test_old_extension_is_driven_in_legacy_mode(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    """An extension without the ownership lifecycle is DRIVEN, not refused.

    This replaces `test_old_extension_is_actionable_before_any_allocation`,
    which asserted the opposite: an error whose text said "Update the
    extension" and no tab allocated. That WAS the defect — the live store build
    was 0.1.10 while the store refused uploads, so the instruction could not be
    acted on and the tool was simply dead. The session now degrades to the
    capability-only legacy path, allocates the tab, and records the redacted
    marker that says so.
    """
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
        assert not result.is_error, result.text
        assert "Update the extension" not in result.text
        assert _control(protocol_peer)["tabs"] == 1, "an older extension must still drive a tab"
        # The record's own statement of the mode, which `lop browser tabs`
        # renders as legacy mode. Named here because it is the difference a
        # support conversation turns on: "it worked" versus "ownership proven".
        assert session._browser.resource.record["ownership"] == "unavailable"
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


@pytest.mark.asyncio
async def test_serialized_extension_refusals_reach_python_recovery(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    """Real TS producer -> fixture JSON HTTP -> BridgeClient -> recover mapping.

    A producer exception is not serialization proof. This existing protocol
    peer uses in-memory Chrome, not a native/installed-extension acceptance rig.
    """
    from local_operator.browser_bridge.backend import BridgeClient, BridgeError
    from local_operator.browser_bridge.protocol import ErrorCode
    from local_operator.browser_bridge.resources import (
        BrowserOwnershipError,
        BrowserResource,
    )

    resource = BrowserResource(headless_tui_env / "sessions" / "wire-sid", "wire-sid")
    resource.initialize()
    client = BridgeClient(headless_tui_env)
    owner = resource.params()
    opened = await client.call("open", {**owner, "url": "https://example.test/"})
    assert opened["tab"].startswith("bridge:")
    for changed, marker, copy in (
        ({"requester": "session:attacker", "resumed_scope": True}, "foreign", "different owner"),
        (
            {
                "owner_generation": "unknown-g",
                "previous_generation": "",
                "previous_generations": [],
                "resumed_scope": True,
            },
            "stale",
            "generation is stale",
        ),
    ):
        with pytest.raises(BridgeError) as wire:
            await client.call("owner_recover", {**owner, **changed})
        assert wire.value.code is ErrorCode.OWNER_REFUSED
        assert wire.value.data["refusal"] == marker

        class RefusalTransport:
            async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
                return await client.call(method, {**params, **changed})

        # Only transport selection is replaced; recovery gets an actual decoded
        # HTTP envelope from the TS producer, never a synthesized BridgeError.
        original_client = resource.client
        resource.client = lambda: RefusalTransport()  # type: ignore[method-assign]
        try:
            with pytest.raises(BrowserOwnershipError, match=copy):
                await resource.recover()
        finally:
            resource.client = original_client  # type: ignore[method-assign]
    with pytest.raises(BridgeError) as wrong_tab:
        await client.call(
            "open", {**owner, "tab": "bridge:999:foreign", "url": "https://example.test/"}
        )
    assert wrong_tab.value.data["refusal"] == "foreign"
    assert _control(protocol_peer)["tabs"] == 1
    await client.call("owner_finish", {**owner, "outcome": "completed"})
    assert _control(protocol_peer)["tabs"] == 0


# --- Child → parent browser handoff (2026-11-11, session 363d303f3e48) -------
#
# A subagent opened a tab for the operator to sign in, said "the tab is open",
# and its teardown closed it — twice. These run a REAL child through
# `_launch_subagent` against the real extension ownership code, so the handoff
# is proven over the same owner_* wire the hosts enforce, not a stub of it.


def _child_browser_stream(
    peer: tuple[int, str], *, retain: str, final: str
) -> tuple[Any, list[int]]:
    """A child that opens one tab, optionally retains it, then finishes."""
    calls = 0
    tab_counts: list[int] = []

    async def stream(request: Any, signal: Any = None) -> AsyncIterator[Any]:
        nonlocal calls
        if not any("BROWSER_CHILD" in getattr(message, "text", "") for message in request.messages):
            yield StreamTextDelta(delta="Acknowledged")
            yield StreamEndEvent(stop_reason="stop")
            return
        calls += 1
        steps: list[dict[str, Any]] = [
            {"action": "open", "url": "https://example.test/", "i": "Opening sign-in page"}
        ]
        if retain:
            steps.append({"action": "retain", "text": retain, "i": "Holding tab for human"})
        if calls <= len(steps):
            yield StreamToolCallDelta(
                index=0,
                id=f"synthetic-child-{calls}",
                name="browser",
                argument_delta=json.dumps(steps[calls - 1]),
            )
            yield StreamEndEvent(stop_reason="toolUse")
            return
        tab_counts.append(_control(peer)["tabs"])
        yield StreamTextDelta(delta=final)
        yield StreamEndEvent(stop_reason="stop")

    return stream, tab_counts


def _parent(root: Path, stream: Any) -> Session:
    async def approve(*_args: Any, **_kwargs: Any) -> bool:
        return True

    return Session(
        model=ModelSpec(provider="test", model_id="synthetic", context_window=100000),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(root / "sessions" / "synthetic-parent"),
        system_blocks_provider=lambda *_: [],
        yolo=True,
        cwd=str(root),
        request_approval=approve,
    )


async def _run_child(owner: Session) -> Any:
    job_id = owner._launch_subagent("browser-child", "BROWSER_CHILD")
    await asyncio.wait_for(owner.jobs.settled_event(job_id).wait(), 25)
    row = owner.jobs.get(job_id)
    assert row is not None and row.status == "completed", row
    return row


@pytest.mark.asyncio
async def test_retained_child_tab_is_handed_to_parent_which_closes_it(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    stream, during = _child_browser_stream(
        protocol_peer, retain="pending human sign-in", final="The tab is open for you."
    )
    owner = _parent(headless_tui_env, stream)
    try:
        row = await _run_child(owner)
        assert during == [1]
        # The child ended; the tab did NOT (the reported bug closed it here).
        assert _control(protocol_peer)["tabs"] == 1
        handed = owner._browser.handed
        assert len(handed) == 1
        label = ":".join(handed[0].surface_id.split(":")[:2])
        # Authoritative fate line, appended by the runner after the settle.
        assert f"Browser: handed {label} to parent {owner.session_id}" in row.result_text
        assert handed[0].surface_id not in row.result_text, "capability must not leak"
        # The parent's listing marks it as its own, with where it came from.
        listing = await execute_browser(
            "parent-tabs", {"action": "tabs"}, None, None, owner._build_tool_context()
        )
        assert not listing.is_error, listing.text
        assert "(yours — handed from child" in listing.text
        assert "pending human sign-in" in listing.text
        # And the parent can close it.
        closed = await execute_browser(
            "parent-close",
            {"action": "close", "tab": label},
            None,
            None,
            owner._build_tool_context(),
        )
        assert not closed.is_error, closed.text
        assert f"Browser: closed {label}" in closed.text
        assert _control(protocol_peer)["tabs"] == 0
        assert owner._browser.handed == []
    finally:
        await owner.dispose()


@pytest.mark.asyncio
async def test_handed_child_tab_is_closed_by_parent_dispose(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    stream, _during = _child_browser_stream(
        protocol_peer, retain="pending human sign-in", final="Handed."
    )
    owner = _parent(headless_tui_env, stream)
    try:
        await _run_child(owner)
        assert _control(protocol_peer)["tabs"] == 1
    finally:
        await owner.dispose()
    # The parent was the last owner: its teardown closes what it was handed.
    assert _control(protocol_peer)["tabs"] == 0


@pytest.mark.asyncio
async def test_unretained_child_tab_is_closed_and_fate_says_so(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    stream, during = _child_browser_stream(
        protocol_peer, retain="", final="The tab is open for the human to sign in."
    )
    owner = _parent(headless_tui_env, stream)
    try:
        row = await _run_child(owner)
        assert during == [1]
        assert _control(protocol_peer)["tabs"] == 0
        assert owner._browser.handed == []
        # The child's prose claims "open"; the harness line contradicts it.
        assert "\n\nBrowser: closed bridge:" in row.result_text
        assert "handed" not in row.result_text
    finally:
        await owner.dispose()


@pytest.mark.asyncio
async def test_pending_approval_child_tab_is_handed(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    """A pending `request_access` writes retention "pending site approval".

    The fixture peer serves no approval flow, so the retention the tool writes
    on a `pending` access state is set directly on a real child's record after
    a real open; everything after that is the production settle.
    """
    from local_operator.harness import subagent

    parent = _parent(headless_tui_env, _child_browser_stream(protocol_peer, retain="", final="")[0])
    child = Session(
        model=ModelSpec(provider="test", model_id="synthetic", context_window=1000),
        stream_fn=lambda *_: None,
        tools=[],
        transcript=Transcript(headless_tui_env / "sessions" / "synthetic-child"),
        system_blocks_provider=lambda *_: [],
        parent_session=parent,
        parent_session_id=parent.session_id,
    )
    try:
        opened = await execute_browser(
            "child-open",
            {"action": "open", "url": "https://example.test/"},
            None,
            None,
            child._build_tool_context(),
        )
        assert not opened.is_error, opened.text
        # The subagent is told its tab's fate at the moment it opens one.
        assert f"retain' it, which hands it to your parent session {parent.session_id}" in (
            opened.text
        )
        child._browser.resource.record["retention"] = "pending site approval"
        fates = await subagent._finish_child_browser(
            child, None, "job", "completed", parent_session=parent
        )
        assert len(fates) == 1 and fates[0].startswith("Browser: handed bridge:")
        assert parent._browser.handed[0].reason == "pending site approval"
        await child.dispose()
        assert _control(protocol_peer)["tabs"] == 1, "child dispose must not close a handed tab"
    finally:
        await parent.dispose()
    assert _control(protocol_peer)["tabs"] == 0


@pytest.mark.asyncio
async def test_retain_in_child_names_parent_handoff(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    parent = _parent(headless_tui_env, _child_browser_stream(protocol_peer, retain="", final="")[0])
    child = Session(
        model=ModelSpec(provider="test", model_id="synthetic", context_window=1000),
        stream_fn=lambda *_: None,
        tools=[],
        transcript=Transcript(headless_tui_env / "sessions" / "synthetic-child"),
        system_blocks_provider=lambda *_: [],
        parent_session=parent,
        parent_session_id=parent.session_id,
    )
    try:
        for args in (
            {"action": "open", "url": "https://example.test/"},
            {"action": "retain", "text": "pending login"},
        ):
            result = await execute_browser("c", args, None, None, child._build_tool_context())
            assert not result.is_error, result.text
        assert f"HANDED to your parent session {parent.session_id}" in result.text
        assert "never as 'left open'" in result.text
    finally:
        await child.dispose()
        await parent.dispose()


# --- Remediation round 1 (PR #3821): running children at teardown, nested
# chains, failed/cancelled paths, and the handed-close handle rules. ---------
#
# One scripted model for every session in a test, keyed by the SCRIPT_<name>
# marker in the session's prompt, so a parent, child and grandchild can each
# run their own steps over the real subagent runner. A ``"block"`` step parks
# the session mid-task (until it is cancelled) and sets ``blocked[name]``;
# ``("await", name)`` waits for another session to reach its block; a
# ``("wait_for_job",)`` step waits on the job id the previous ``task`` returned;
# ``"raise"`` fails the child's stream.


def _scripted_stream(scripts: dict[str, list[Any]], final: dict[str, str]) -> tuple[Any, Any]:
    import re

    steps_taken: dict[str, int] = {}
    blocked = {name: asyncio.Event() for name in scripts}

    def _marker(request: Any) -> str:
        for message in request.messages:
            text = getattr(message, "text", "") or ""
            for name in scripts:
                if f"SCRIPT_{name}" in text and f"SCRIPT_{name}_" not in text:
                    return name
        return ""

    def _last_job_id(request: Any) -> str:
        for message in reversed(request.messages):
            found = re.search(r"\b([0-9a-f]{12})\b", str(getattr(message, "content", "")))
            if found:
                return found.group(1)
        return ""

    async def stream(request: Any, signal: Any = None) -> AsyncIterator[Any]:
        name = _marker(request)
        if not name:
            yield StreamTextDelta(delta="Acknowledged")
            yield StreamEndEvent(stop_reason="stop")
            return
        index = steps_taken.get(name, 0)
        steps_taken[name] = index + 1
        steps = scripts[name]
        step = steps[index] if index < len(steps) else None
        while isinstance(step, tuple) and step[0] == "await":
            await asyncio.wait_for(blocked[step[1]].wait(), 20)
            index += 1
            steps_taken[name] = index + 1
            step = steps[index] if index < len(steps) else None
        if step == "block":
            blocked[name].set()
            await asyncio.Event().wait()
        if step == "raise":
            raise RuntimeError(f"synthetic failure in {name}")
        if isinstance(step, tuple) and step[0] == "wait_for_job":
            step = {"job_id": _last_job_id(request), "wait_ms": 20000}
            tool = "wait"
        elif isinstance(step, dict) and "prompt" in step:
            tool = "task"
        else:
            tool = "browser"
        if isinstance(step, dict):
            yield StreamToolCallDelta(
                index=0, id=f"synthetic-{name}-{index}", name=tool, argument_delta=json.dumps(step)
            )
            yield StreamEndEvent(stop_reason="toolUse")
            return
        yield StreamTextDelta(delta=final.get(name, f"{name} done."))
        yield StreamEndEvent(stop_reason="stop")

    return stream, blocked


_OPEN = {"action": "open", "url": "https://example.test/", "i": "Opening sign-in page"}
#: For a direct ``execute_browser`` call: ``i`` is stripped by the agent loop,
#: never by the executor, whose params forbid extras.
_OPEN_ARGS = {"action": "open", "url": "https://example.test/"}


def _retain(reason: str) -> dict[str, Any]:
    return {"action": "retain", "text": reason, "i": "Holding tab for human"}


async def _settle(owner: Session, job_id: str) -> Any:
    await asyncio.wait_for(owner.jobs.settled_event(job_id).wait(), 25)
    return owner.jobs.get(job_id)


@pytest.mark.asyncio
async def test_running_child_tab_is_closed_when_parent_disposes(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    """R1: dispose cancels a still-running child AFTER the first handed pass.

    The cancelled child hands its retained tab to the parent on its way out;
    before the fix nothing read the parent's ``handed`` list again, so the tab
    stayed open, owned by nobody `lop browser cleanup` would settle.
    """
    stream, blocked = _scripted_stream(
        {"C": [_OPEN, _retain("pending human sign-in"), "block"]}, {}
    )
    owner = _parent(headless_tui_env, stream)
    try:
        owner._launch_subagent("browser-child", "SCRIPT_C")
        await asyncio.wait_for(blocked["C"].wait(), 20)
        assert _control(protocol_peer)["tabs"] == 1
    finally:
        await owner.dispose()
    assert _control(protocol_peer)["tabs"] == 0, "running child's handed tab stranded"
    assert owner._browser.handed == []


@pytest.mark.asyncio
async def test_grandchild_running_when_child_finishes_climbs_to_parent(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    """R1 nested: the mid child's dispose cancels the leaf, which hands up late.

    The late arrival must climb to the parent (still alive, so it owns the
    tab), not be closed by the mid child or stranded in its list.
    """
    stream, _blocked = _scripted_stream(
        {
            "MID": [{"label": "leaf", "prompt": "SCRIPT_LEAF"}, ("await", "LEAF")],
            "LEAF": [_OPEN, _retain("leaf sign-in"), "block"],
        },
        {"MID": "Mid finished while its leaf waits."},
    )
    owner = _parent(headless_tui_env, stream)
    try:
        job_id = owner._launch_subagent("mid", "SCRIPT_MID")
        row = await _settle(owner, job_id)
        assert row is not None and row.status == "completed", row
        handed = owner._browser.handed
        assert [item.reason for item in handed] == ["leaf sign-in"], handed
        assert handed[0].from_session != row.id
        assert _control(protocol_peer)["tabs"] == 1, "the human's tab must survive"
    finally:
        await owner.dispose()
    assert _control(protocol_peer)["tabs"] == 0


@pytest.mark.asyncio
async def test_finished_grandchild_tab_chains_up_with_fate_lines(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    """R2: grandchild → child → parent, each hop reporting a fate line."""
    stream, _blocked = _scripted_stream(
        {
            "MID": [{"label": "leaf", "prompt": "SCRIPT_LEAF"}, ("wait_for_job",)],
            "LEAF": [_OPEN, _retain("leaf sign-in")],
        },
        {"MID": "Mid done.", "LEAF": "Leaf: tab is open."},
    )
    owner = _parent(headless_tui_env, stream)
    try:
        job_id = owner._launch_subagent("mid", "SCRIPT_MID")
        row = await _settle(owner, job_id)
        assert row is not None and row.status == "completed", row
        assert f"to parent {owner.session_id}" in row.result_text
        assert row.result_text.count("Browser: handed bridge:") == 1
        assert [item.reason for item in owner._browser.handed] == ["leaf sign-in"]
        assert _control(protocol_peer)["tabs"] == 1
        label = ":".join(owner._browser.handed[0].surface_id.split(":")[:2])
        closed = await execute_browser(
            "c", {"action": "close", "tab": label}, None, None, owner._build_tool_context()
        )
        assert not closed.is_error and closed.text.startswith(f"Browser: closed {label}")
        assert _control(protocol_peer)["tabs"] == 0
    finally:
        await owner.dispose()


@pytest.mark.asyncio
async def test_parent_with_own_and_handed_tab_lists_and_matches_handles(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    """R2 listing + R3 handle shape + Q3 no-match message."""
    stream, _blocked = _scripted_stream({"C": [_OPEN, _retain("child sign-in")]}, {})
    owner = _parent(headless_tui_env, stream)
    try:
        own = await execute_browser(
            "p-open", dict(_OPEN_ARGS), None, None, owner._build_tool_context()
        )
        assert not own.is_error, own.text
        job_id = owner._launch_subagent("browser-child", "SCRIPT_C")
        assert (await _settle(owner, job_id)).status == "completed"
        assert _control(protocol_peer)["tabs"] == 2
        listing = await execute_browser(
            "p-tabs", {"action": "tabs"}, None, None, owner._build_tool_context()
        )
        lines = [
            line
            for line in listing.text.splitlines()
            if line.startswith("bridge:") and "(yours" in line
        ]
        assert len(lines) == 2, listing.text
        assert sum("(yours)" in line for line in lines) == 1
        assert sum("handed from child" in line and "child sign-in" in line for line in lines) == 1
        # R3: a bare or one-char ellipsis is not a redacted listing token.
        for bogus in ("…", "b…", "bridge:…", "bridge:999"):
            missed = await execute_browser(
                "p-close",
                {"action": "close", "tab": bogus},
                None,
                None,
                owner._build_tool_context(),
            )
            assert missed.is_error
            # Q3: names the mismatch and the handles that do exist.
            assert f"No handed tab matches {bogus!r}" in missed.text, missed.text
        assert _control(protocol_peer)["tabs"] == 2, "a non-matching handle closed a tab"
        # The listing's own redacted token does match.
        handed_line = next(line for line in lines if "handed from child" in line)
        token = handed_line.split(" ", 1)[0]
        assert token.endswith("…")
        closed = await execute_browser(
            "p-close", {"action": "close", "tab": token}, None, None, owner._build_tool_context()
        )
        assert not closed.is_error, closed.text
        assert _control(protocol_peer)["tabs"] == 1
        assert owner._browser.surface_id, "the parent's own tab is untouched"
    finally:
        await owner.dispose()
    assert _control(protocol_peer)["tabs"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("steps", "fate"),
    [
        ([_OPEN, _retain("pending human sign-in"), "raise"], "Browser: handed bridge:"),
        ([_OPEN, "raise"], "Browser: closed bridge:"),
    ],
    ids=["retained", "unretained"],
)
async def test_failed_child_fate_line_reaches_parent_wait(
    protocol_peer: tuple[int, str], headless_tui_env: Path, steps: list[Any], fate: str
) -> None:
    """Q1: the job ROW (what `wait`/`jobs` read) carries the fate line."""
    from local_operator.tools.builtin import execute_wait

    stream, _blocked = _scripted_stream({"C": steps}, {})
    owner = _parent(headless_tui_env, stream)
    try:
        job_id = owner._launch_subagent("browser-child", "SCRIPT_C")
        row = await _settle(owner, job_id)
        assert row is not None and row.status == "failed", row
        assert "synthetic failure in C" in row.error_text
        assert fate in row.error_text, row.error_text
        waited = await execute_wait(
            "w", {"job_id": job_id, "wait_ms": 1000}, None, None, owner._build_tool_context()
        )
        assert fate in waited.text, waited.text
    finally:
        await owner.dispose()
    assert _control(protocol_peer)["tabs"] == 0


@pytest.mark.asyncio
async def test_cancelled_child_retained_tab_is_handed_then_closed(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    """R2 cancelled: an explicit cancel still hands the human's tab up."""
    stream, blocked = _scripted_stream(
        {"C": [_OPEN, _retain("pending human sign-in"), "block"]}, {}
    )
    owner = _parent(headless_tui_env, stream)
    try:
        job_id = owner._launch_subagent("browser-child", "SCRIPT_C")
        await asyncio.wait_for(blocked["C"].wait(), 20)
        assert await owner.jobs.cancel(job_id)
        row = await _settle(owner, job_id)
        assert row is not None and row.status == "cancelled", row
        assert [item.reason for item in owner._browser.handed] == ["pending human sign-in"]
        assert _control(protocol_peer)["tabs"] == 1
    finally:
        await owner.dispose()
    assert _control(protocol_peer)["tabs"] == 0


@pytest.mark.asyncio
async def test_child_open_result_drops_top_level_left_open_reminder(
    protocol_peer: tuple[int, str], headless_tui_env: Path
) -> None:
    """Q2: one result must not say both 'leave it open' and 'never left open'."""
    from local_operator.tools import builtin

    parent = _parent(headless_tui_env, _child_browser_stream(protocol_peer, retain="", final="")[0])
    child = Session(
        model=ModelSpec(provider="test", model_id="synthetic", context_window=1000),
        stream_fn=lambda *_: None,
        tools=[],
        transcript=Transcript(headless_tui_env / "sessions" / "synthetic-child"),
        system_blocks_provider=lambda *_: [],
        parent_session=parent,
        parent_session_id=parent.session_id,
    )
    try:
        opened = await execute_browser(
            "c", dict(_OPEN_ARGS), None, None, child._build_tool_context()
        )
        assert not opened.is_error, opened.text
        assert "never as 'left open'" in opened.text
        assert builtin._BROWSER_OPEN_CLEANUP_REMINDER not in opened.text
        top = await execute_browser("p", dict(_OPEN_ARGS), None, None, parent._build_tool_context())
        assert builtin._BROWSER_OPEN_CLEANUP_REMINDER in top.text, "top-level text unchanged"
    finally:
        await child.dispose()
        await parent.dispose()
