"""A peer `lop send` landing on a HIDDEN session paints on the first reveal.

The reported bug: an inbound ``peer_message`` CustomMessage appended while a
session sat parked in the sidebar was invisible until a SECOND reload. The
transcript showed the agent answering the note but the ``PeerMessageBlock``
itself never mounted on the first reveal, on BOTH a cold resume and a warm
session-to-session swap.

The mechanism is that a settled peer append moves NONE of the viewer's
freshness signals: ``history_generation`` bumps only on compaction/prune, and
the ``PeerMessageDeliveredEvent`` carries no ``_MESSAGE_PHASE`` lifecycle (it is
one settled row, not a streaming beat). So the hidden viewer's
``history_message_count`` stayed frozen — which meant neither
``_sidebar_presentation_current``'s count guard rejected the stale cached
presentation, nor ``_commit_sidebar_session``'s ``total > history_size`` delta
projection fired. The fix (``RemoteSession._remember_live``) files the settled
row keyed by its persisted id so the count advances.

This drives the full assembled path — a real owner behind a real socket, the
production sidebar select, and the TUI's replay — asserting the block appears
on the FIRST switch back, and is not painted twice.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import PeerMessageBlock
from tests.e2e.harness import (
    ScriptedStream,
    assistant_message,
    build_session,
    seed_transcript,
    text_turn,
    user_message,
    wait_for_adoption,
)
from tests.unit.harness.test_comms import DEADLOCK_GUARD_S


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Headless apps must never rename the caller's real multiplexer workspace.
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_terminal_title", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_multiplexer_broadcast", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_herdr_reporter", lambda _self: None)


@asynccontextmanager
async def live_owners(
    config: Path, ids: list[str], monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Callable[[str | None], Awaitable[RemoteSession]]]:
    """Real ``RuntimeServer`` owners behind one sidebar, one per id.

    Mirrors ``test_sidebar_live_tool_card.live_owners`` minus the parking tool
    and turn-driving stream: these sessions stay IDLE, because the receipt being
    tested is the mailbox record-only path — no turn runs at all.
    """
    from local_operator.session.runtime import registry

    def publish(record: Any, root: Path | None = None) -> Path:
        directory = registry.run_dir(root)
        record.heartbeat_at = time.time()
        handle, path = tempfile.mkstemp(dir=directory, prefix=".x.", suffix=".tmp")
        with os.fdopen(handle, "w") as stream:
            json.dump(record.to_json(), stream)
        target = directory / f"{record.pid}-{record.session_id}.json"
        os.replace(path, target)
        return target

    monkeypatch.setattr(registry, "publish", publish)
    monkeypatch.setattr(registry, "unpublish", lambda pid, root=None: None)

    servers: dict[str, RuntimeServer] = {}
    handles: list[OwnedSessionHandle] = []
    try:
        for session_id in ids:
            directory = config / "sessions" / session_id
            await seed_transcript(
                directory,
                [
                    user_message(f"{session_id} question"),
                    assistant_message(f"{session_id} saved answer"),
                ],
            )
            owner = build_session(directory, ScriptedStream([text_turn("idle")]), cwd=config)
            handle = OwnedSessionHandle(owner, asyncio.get_running_loop(), cwd=str(config))
            handles.append(handle)
            server = RuntimeServer(handle, kind="daemon")
            await server.start_in_process()
            servers[session_id] = server

        def find(_directory: Path, session_id: str) -> tuple[Any, Any]:
            server = servers.get(session_id)
            return (server._record, server._record.pid) if server else (None, None)

        async def never() -> Any:
            raise AssertionError("view navigation must never take execution ownership")

        async def resume(session_id: str | None) -> RemoteSession:
            assert session_id is not None
            return await RemoteSession.connect(
                servers[session_id]._record,
                session_id,
                config_dir=config,
                takeover_factory=never,
                display_window=True,
            )

        monkeypatch.setattr("local_operator.mobile.attach_client.find_owner_record", find)
        resume.servers = servers  # type: ignore[attr-defined]
        yield resume
    finally:
        for server in servers.values():
            await server.aclose()
        for handle in handles:
            await handle.dispose()


async def _switch(app: OperatorApp, session_id: str, pilot: Any) -> None:
    """The production sidebar path: select, then await the connection it opens."""
    await asyncio.wait_for(app._sidebar_navigation.select(session_id), DEADLOCK_GUARD_S)
    connection = app._interaction.connection_task
    if connection is not None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(connection), DEADLOCK_GUARD_S)
    for _ in range(40):
        await pilot.pause()


def _peer_blocks(app: OperatorApp) -> list[PeerMessageBlock]:
    return [b for b in app._transcript_view().blocks() if isinstance(b, PeerMessageBlock)]


@pytest.mark.asyncio
async def test_peer_row_on_a_hidden_session_paints_on_the_first_reveal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Visit, leave, deliver a `lop send`, then click back: the block is there.

    The exact operator repro: the hidden session receives the note while it is
    parked, and the FIRST return must already show it. Asserted twice — that the
    viewer's count advanced (the signal the reveal consults) and that exactly
    one ``PeerMessageBlock`` mounted (no double-paint against the live receipt,
    which a hidden session never paints).
    """
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    ids = ["home00", "peer01"]
    async with live_owners(config, ids, monkeypatch) as resume:
        app = OperatorApp(lambda: resume("home00"), resume_factory=resume)
        async with app.run_test(size=(120, 36)) as pilot:
            await wait_for_adoption(app, pilot)
            app._set_sidebar_open(True)
            if app._sidebar_timer is not None:
                app._sidebar_timer.pause()  # no background prewarm races

            # Visit the peer conversation, then leave it hidden — the state the
            # bug needs: a connected, parked source with a cached presentation.
            await _switch(app, "peer01", pilot)
            await _switch(app, "home00", pilot)

            owner = resume.servers["peer01"]._handle._session  # type: ignore[attr-defined]
            await owner.receive_peer_message(
                "gates are green",
                sender={"pid": 4242, "conversation_name": "peer-send design"},
            )

            # The receipt crosses the socket asynchronously; wait until the
            # hidden viewer has filed the settled row before the reveal, or the
            # assertion measures a delivery race instead of the paint path.
            # `cast`, because the protocol the app types its sources against
            # does not declare the viewer-only `_live_history` this wait reads.
            viewer = cast(Any, app._sidebar_sources["peer01"].session)
            for _ in range(200):
                await pilot.pause()
                if any(
                    getattr(row, "custom_type", None) == "peer_message"
                    for row in viewer._live_history.values()
                ):
                    break
            else:
                raise AssertionError("the hidden viewer never filed the settled peer row")

            await _switch(app, "peer01", pilot)

            blocks = _peer_blocks(app)
            assert (
                len(blocks) == 1
            ), f"expected exactly one peer receipt on the first reveal, got {len(blocks)}"
            assert blocks[0].text() == "gates are green"
