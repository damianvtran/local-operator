"""A viewer that WATCHED tool calls run must keep their measured durations.

Regression guard for the defect PR #823 shipped past: durations were blank on
every already-settled tool card after resuming through the sidebar, while the
newest card still showed a time.

WHAT IS BEING PROVEN
--------------------
A viewer attached over a runtime socket (what the SIDEBAR attaches to) keeps
each completed tool result in ``AttachedSession._live_history``, built by
``_remember_live`` via ``Message.tool_result(event.result)``. That constructor
copies content/ids/is_error and NOT ``provider_payload`` — so the measured
``duration_s`` that rode the wire on ``event.result.duration_s`` is dropped.

``display_history_window()`` returns those live rows, and every re-render of
history (a sidebar switch -> ``_adopt_session``, and the keystroke route's
``PreparedReplay.prepare``) replays them through ``replay_tool_call``, which
reads ``provider_payload["duration_s"]`` and finds nothing. Result: blank
column. ``details`` and ``useless`` are dropped on the same line, so the tool
diff badge (``+1``) disappears from those cards too.

WHY EVERY EARLIER TEST STAYED GREEN
-----------------------------------
The state this needs is a viewer that WATCHED TURNS RUN, so ``_live_history``
is populated. A viewer attached to a QUIESCENT owner renders durations fine
(nothing shadows the durable rows), and a cold local resume never builds a
live row at all — those are the paths #823 was verified on. Reaching the bug
therefore requires a real ``ServingSessionHandle`` + ``RuntimeServer`` +
``AttachedSession(display_window=True)`` executing a REAL tool, which is what
``_live_runtime`` stands up: the duration here is MEASURED by the harness, not
fixtured, so a test that fakes the payload cannot pass in its place.

And the assertion is the DURATION COLUMN itself. ``test_reconnect_parity.py``
compared block classes, so it stayed green while the column diverged — the
``✓``-shaped signature is identical whether or not the interval survives.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session.attached import AttachedSession
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.tools.builtin import build_write_tool
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.tool_card import ToolCard
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    seed_transcript,
    text_turn,
    tool_call_turn,
)
from tests.unit.session.test_remote import _never_take_over


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    # A headless pilot must never touch the operator's real multiplexer.
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda _s: None)
    monkeypatch.setattr(OperatorApp, "_start_terminal_title", lambda _s: None)
    monkeypatch.setattr(OperatorApp, "_start_multiplexer_broadcast", lambda _s: None)
    monkeypatch.setattr(OperatorApp, "_start_herdr_reporter", lambda _s: None)
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)


def _cells(app: OperatorApp) -> list[float | None]:
    # ``_transcript_view()``, NOT ``query_one(TranscriptView)``. The sidebar
    # gesture this file exercises leaves the OUTGOING conversation's view in
    # the DOM beside the adopted one, and ``query_one`` returns the first in
    # DOM order — the transcript being left behind. The assertion would then
    # silently grade the wrong conversation, which is exactly the class of
    # false green this file exists to prevent.
    view = app._transcript_view()
    return [b._duration for b in view.blocks() if isinstance(b, ToolCard)]


async def _boot(app, pilot) -> None:
    for _ in range(400):
        await pilot.pause()
        if app._session is not None and app._transcript_view().blocks():
            for _ in range(10):
                await pilot.pause()
            return
    raise AssertionError("app never booted")


async def _live_runtime(tmp_path: Path, n: int):
    """A REAL runtime socket serving a session that RUNS n timed tool calls."""
    config = tmp_path / "config"
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config / "sessions" / "synthetic-dur"
    await seed_transcript(
        directory, [Message(id="u0", role="user", content=[TextContent(text="start")])]
    )
    # The session's first model request is its own internal one (conversation
    # naming), so the scripted tool turns start at index 1.
    turns: list[Any] = [text_turn("filler")]
    for i in range(n):
        turns.append(
            tool_call_turn(
                text=f"step {i}",
                tool_name="write",
                tool_call_id=f"c{i}",
                arguments={"path": str(tmp_path / f"o{i}.txt"), "content": f"{i}\n"},
            )
        )
        turns.append(text_turn(f"ok {i}"))
    # Spare turns: the session may issue its own extra model requests
    # (naming/retries); running off the end of the script is a harness
    # IndexError, not a product failure.
    turns.extend(text_turn("spare") for _ in range(6))

    session = build_session(
        directory, ScriptedStream(turns), tools=[build_write_tool()], cwd=tmp_path
    )
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    remote = await AttachedSession.connect(
        server._record,
        directory.name,
        config_dir=config,
        takeover_factory=_never_take_over,
        display_window=True,
    )
    return session, handle, server, remote, directory


async def _run_turns(handle, remote, tmp_path: Path, n: int, pilot=None) -> None:
    """Drive n real tool turns; wait on the tool's OWN side effect, not a flag."""
    await handle.prompt("warmup")
    for _ in range(400):
        await asyncio.sleep(0.01)
        if pilot is not None:
            await pilot.pause()
        if not remote.frontend_state.streaming:
            break
    for i in range(n):
        await handle.prompt(f"go {i}")
        target = tmp_path / f"o{i}.txt"
        for _ in range(800):
            await asyncio.sleep(0.01)
            if pilot is not None:
                await pilot.pause()
            if target.exists() and not remote.frontend_state.streaming:
                break
        assert target.exists(), f"tool call {i} never executed"
    await asyncio.sleep(0.3)


@pytest.mark.asyncio
async def test_live_tool_row_keeps_its_measured_duration(tmp_path) -> None:
    """UNIT-LEVEL: the losing boundary itself.

    ``duration_s`` arrives on the wire as ``event.result.duration_s`` and must
    survive into the live row's ``provider_payload``, because that row is what
    ``display_history_window()`` serves to every replay.
    """
    session, handle, server, remote, _ = await _live_runtime(tmp_path, 2)
    try:
        await _run_turns(handle, remote, tmp_path, 2)
        live_tools = [
            (k, m) for k, m in remote._live_history.items() if getattr(m, "role", "") == "tool"
        ]
        assert live_tools, "no live tool rows were retained"
        for key, message in live_tools:
            payload = getattr(message, "provider_payload", None) or {}
            assert isinstance(payload, dict), f"{key}: provider_payload missing entirely"
            duration = payload.get("duration_s")
            assert isinstance(duration, (int, float)) and duration >= 0, (
                f"{key}: measured duration was dropped building the live row "
                f"(provider_payload={payload!r})"
            )
            # ``details`` rides the SAME construction and was lost with it, so
            # the tool card's diff badge vanished on this path as well. The
            # write tool always reports its line counts, so an absent key here
            # is the same defect rather than a tool that had nothing to say.
            assert payload.get("details"), (
                f"{key}: harness details were dropped building the live row "
                f"(provider_payload={payload!r})"
            )
    finally:
        await remote.dispose()
        server.close()
        await handle.dispose()


@pytest.mark.asyncio
async def test_sidebar_resume_renders_durations_on_a_watched_session(tmp_path) -> None:
    """END TO END: the operator's exact gesture, in the real app.

    Watch tool calls run, then re-render history the way a SIDEBAR switch does
    (``_adopt_session`` -> ``_render_resumed_history``) and assert the DURATION
    COLUMN, not the block classes — the assertion ``test_reconnect_parity.py``
    was missing when this shipped.
    """
    session, handle, server, remote, _ = await _live_runtime(tmp_path, 3)

    async def factory():
        return remote

    app = OperatorApp(factory)
    try:
        async with app.run_test(size=(118, 40)) as pilot:
            await _boot(app, pilot)
            await _run_turns(handle, remote, tmp_path, 3, pilot=pilot)
            for _ in range(20):
                await pilot.pause()

            live_cells = _cells(app)
            assert live_cells, "no tool cards painted live"
            assert all(
                c is not None for c in live_cells
            ), f"live painting already lost durations: {live_cells}"

            # THE SIDEBAR GESTURE: re-adopt the same session, which is what
            # SessionSidebar.Selected -> _adopt_session does on a switch.
            app._adopt_session(remote)
            for _ in range(25):
                await pilot.pause()

            resumed = _cells(app)
            assert resumed, "resume painted no tool cards"
            blank = [i for i, c in enumerate(resumed) if c is None]
            assert not blank, (
                f"{len(blank)}/{len(resumed)} tool cards lost their duration on the "
                f"sidebar resume path: {resumed} (live frame was {live_cells})"
            )
    finally:
        await remote.dispose()
        server.close()
        await handle.dispose()
