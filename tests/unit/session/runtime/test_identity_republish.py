"""The discovery record follows a model switch on the next push, not the heartbeat.

`lop sessions` reads the RECORD. Only the 15 s heartbeat copied the model and
title into it, so after `/model` the listing named the old model for up to a
heartbeat — and indefinitely on a TUI, where the switch emitted no event to
move the projection either (design D5; the other half is covered in
``tests/unit/session/test_active_route.py``).

These drive the REAL ``Session`` behind the production handle and server, so the
seam under test is the one production runs; the record writer is only wrapped,
to count writes, and every assertion reads the published file.
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from local_operator.harness.types import ModelChangeEvent, ModelSpec
from local_operator.headless_print import PrintRenderer
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session, text_turn

NEW_MODEL = ModelSpec(provider="deepseek", model_id="deepseek-flash", context_window=128_000)


class _WriteLog(list[dict[str, Any]]):
    """Every ``heartbeat`` update the REAL publisher was asked to write. The
    write itself still happens, so the file `lop sessions` scans is real."""


@asynccontextmanager
async def _rig(
    directory: Path,
) -> AsyncIterator[tuple[Any, ServingSessionHandle, RuntimeServer, _WriteLog]]:
    """A real Session, the production handle, and a LISTENING in-process server
    with its real publisher — the ``_serve`` prologue installs the projection
    subscription exactly as production does. No client ever dials, so this is
    the detached owner."""
    directory.mkdir(parents=True, exist_ok=True)
    session = build_session(directory, ScriptedStream([text_turn("reply")]))
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    try:
        real = server._publisher
        assert real is not None
        writes = _WriteLog()
        heartbeat = real.heartbeat

        def counted(**updates: Any) -> None:
            writes.append(updates)
            heartbeat(**updates)

        real.heartbeat = counted  # type: ignore[method-assign]
        yield session, handle, server, writes
    finally:
        await server.aclose()


def _on_disk(server: RuntimeServer) -> dict[str, Any]:
    return json.loads(server.record_path.read_text())


async def _until(predicate: Any, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "timed out waiting for the record"
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_detached_owner_republishes_the_new_model_on_the_push_tick(tmp_path: Path) -> None:
    """No client is attached, so ``_push`` takes its no-recipients return — the
    identity republish must run before it, or an idle detached session (the
    herdr case) keeps its old model in `lop sessions` until the heartbeat."""
    async with _rig(tmp_path / "s") as (session, _handle, server, publisher):
        assert server._projection_recipients() == []
        assert _on_disk(server)["model_label"] == "test/e2e-model"

        session.set_model(NEW_MODEL, explicit=True)
        # Far inside the 15 s heartbeat: the push tick is what carried it.
        await _until(lambda: _on_disk(server)["model_label"] == "deepseek/deepseek-flash")

        identity_writes = [w for w in publisher if "model_label" in w]
        assert identity_writes == [
            {
                "session_id": session.session_id,
                "model_label": "deepseek/deepseek-flash",
                "conversation_name": "",
            }
        ]


@pytest.mark.asyncio
async def test_a_new_session_id_moves_with_its_title(tmp_path: Path) -> None:
    """Review R1-1. After ``/resume`` or ``/new`` on a TUI host the projection
    carries a new id AND a new title; a republish that carried only the title
    paired the new conversation's name with the OLD id until the heartbeat, and
    an id copied from `lop sessions` then resumes the wrong conversation."""
    async with _rig(tmp_path / "s") as (_session, handle, server, _writes):
        handle._projection.session_id = "synthnewid01"
        handle._fold.set_state(conversation_name="the resumed conversation")
        await server._push()
        record = _on_disk(server)
        assert (record["session_id"], record["conversation_name"]) == (
            "synthnewid01",
            "the resumed conversation",
        )


@pytest.mark.asyncio
async def test_an_unchanged_identity_does_not_rewrite_the_record(tmp_path: Path) -> None:
    """``_push`` runs on every coalesced tick — ~20 Hz while streaming — so the
    steady cost must be a comparison, and a write only on a change."""
    async with _rig(tmp_path / "s") as (_session, _handle, server, publisher):
        for _ in range(10):
            await server._push()
        assert publisher == []


@pytest.mark.asyncio
async def test_a_title_change_reaches_the_record_on_the_push_tick(tmp_path: Path) -> None:
    """The title rides the same comparison: a rename is the other identity
    field the listing prints, and it had the same heartbeat-only lag."""
    async with _rig(tmp_path / "s") as (_session, handle, server, publisher):
        handle._fold.set_state(conversation_name="triage the flaky shard")
        await server._push()
        assert _on_disk(server)["conversation_name"] == "triage the flaky shard"
        assert len(publisher) == 1


@pytest.mark.asyncio
async def test_a_failing_record_write_does_not_break_the_push(tmp_path: Path) -> None:
    """The heartbeat still corrects the record within 15 s; a raise here would
    cost every attached viewer its repaint."""
    async with _rig(tmp_path / "s") as (_session, handle, server, _writes):
        real = server._publisher
        assert real is not None
        saved = real.heartbeat

        def broken(**_updates: Any) -> None:
            raise OSError("disk full")

        real.heartbeat = broken  # type: ignore[method-assign]
        try:
            handle._fold.set_state(model_label="deepseek/deepseek-flash")
            await server._push()  # must not raise
        finally:
            real.heartbeat = saved  # type: ignore[method-assign]


@pytest.mark.parametrize(
    ("event", "line"),
    [
        (
            ModelChangeEvent(
                provider="deepseek", model_id="deepseek-flash", reason="model switched"
            ),
            "switched to deepseek/deepseek-flash",
        ),
        (
            ModelChangeEvent(provider="anthropic", model_id="claude-opus-5", reason="recovered"),
            "back to anthropic/claude-opus-5",
        ),
        (
            # Review R2-1: a route edge that FORGOT its reason still prints. The
            # skip keys on ``context_metadata`` alone, so a future emitter's
            # omission degrades to the plain verb instead of vanishing.
            ModelChangeEvent(provider="anthropic", model_id="claude-opus-5"),
            "back to anthropic/claude-opus-5",
        ),
        (
            ModelChangeEvent(provider="zai", model_id="glm-5.3", is_fallback=True),
            "fell back to zai/glm-5.3",
        ),
        (
            # Every production pin carries its cause (``RouteState.activate``).
            ModelChangeEvent(
                provider="zai", model_id="glm-5.3", is_fallback=True, reason="provider failure"
            ),
            "fell back to zai/glm-5.3",
        ),
    ],
)
def test_headless_output_names_each_route_edge(event: ModelChangeEvent, line: str) -> None:
    """A deliberate switch is not a recovery: "back to" would claim a model the
    run never left."""
    buffer = io.StringIO()
    console = Console(file=buffer, no_color=True, highlight=False, width=100)
    PrintRenderer(json_mode=False, console=console).handle(event)
    assert buffer.getvalue().splitlines() == [line]


@pytest.mark.parametrize("pinned", [False, True], ids=["unpinned", "under-a-fallback-pin"])
def test_headless_output_prints_no_route_line_for_a_metadata_refresh(
    tmp_path: Path, pinned: bool
) -> None:
    """Design D1 / review R1-2. ``_refresh_context_metadata`` re-announces the
    model in force (``context_metadata=True``, no reason) at turn start and end
    whenever the resolved window differs. Printed, that read as a recovery that
    never happened ("back to <primary>") or, under a pin, as the fallback being
    taken again ("fell back to …"). Driven through a REAL session and turn, so
    the events are the ones production emits, rewritten by ``_emit``.
    """
    from local_operator.providers.failover import FallbackTarget

    class _WindowStream(ScriptedStream):
        route_handler: Any = None

        def set_route_handler(self, handler: Any) -> None:
            self.route_handler = handler

        def resolve_context_model(self, spec: ModelSpec) -> ModelSpec:
            return spec.model_copy(update={"context_window": 400_000})

    async def turn() -> tuple[str, int]:
        directory = tmp_path / "s"
        directory.mkdir()
        stream = _WindowStream([text_turn("hi")])
        session = build_session(directory, stream)
        buffer = io.StringIO()
        renderer = PrintRenderer(
            json_mode=False, console=Console(file=buffer, no_color=True, width=120)
        )
        if pinned:
            await stream.route_handler(FallbackTarget("zai/glm-5.3", None), "provider failure")
        refreshes: list[Any] = []
        session.subscribe(
            lambda e: (
                refreshes.append(e)
                if isinstance(e, ModelChangeEvent) and e.context_metadata
                else None
            )
        )
        session.subscribe(renderer.handle)
        try:
            await session.prompt("hi")
            await _until(lambda: bool(refreshes))
        finally:
            await session.dispose()
        return buffer.getvalue(), len(refreshes)

    out, refreshes = asyncio.run(turn())
    assert refreshes, "the rig never produced a metadata refresh, so it proves nothing"
    route_lines = [
        line
        for line in out.splitlines()
        if line.startswith(("back to", "fell back to", "switched to"))
    ]
    assert route_lines == [], route_lines
