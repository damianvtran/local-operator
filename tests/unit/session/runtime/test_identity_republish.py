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
            {"model_label": "deepseek/deepseek-flash", "conversation_name": ""}
        ]


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
            ModelChangeEvent(provider="zai", model_id="glm-5.3", is_fallback=True),
            "fell back to zai/glm-5.3",
        ),
    ],
)
def test_headless_output_names_a_deliberate_switch(event: ModelChangeEvent, line: str) -> None:
    """A deliberate switch is not a recovery: "back to" would claim a model the
    run never left."""
    buffer = io.StringIO()
    console = Console(file=buffer, no_color=True, highlight=False, width=100)
    PrintRenderer(json_mode=False, console=console).handle(event)
    assert buffer.getvalue().splitlines() == [line]
