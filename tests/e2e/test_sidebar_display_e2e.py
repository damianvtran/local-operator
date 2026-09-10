"""A responsive viewport is independent of the runtime's canonical sync.

Use real authenticated owner sockets and real OperatorApp composition. The
held subscribe event is the assertion: display must finish before it is ever
released, rather than winning a race against an arbitrary latency threshold.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tui.app import (
    ASIDE_PLACEHOLDER,
    SIDEBAR_CONNECT_ATTEMPTS,
    OperatorApp,
    ToolCard,
)
from local_operator.tui.session_interaction import SessionDraft
from local_operator.tui.widgets.copy_picker import CopyPickerScreen
from tests.e2e.harness import (
    ScriptedStream,
    assistant_message,
    build_session,
    seed_transcript,
    user_message,
    wait_for_adoption,
)
from tests.unit.tui.test_app_pilot import _renderable_plain, _set_editor_line


def visible_text(app: OperatorApp) -> str:
    return "\n".join(
        _renderable_plain(getattr(block, "renderable", ""))
        for block in app._transcript_view().blocks()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "switch_away, fail_sync, restore_anchor, wrapped, move_reader",
    [
        (False, False, False, False, False),
        (True, False, False, False, False),
        (False, True, False, False, False),
        (False, False, True, False, False),
        (True, False, True, False, False),
        (False, False, True, True, False),
        (False, False, True, False, True),
    ],
)
async def test_saved_view_switches_while_authenticated_owner_sync_is_held(
    headless_tui_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    switch_away: bool,
    fail_sync: bool,
    restore_anchor: bool,
    wrapped: bool,
    move_reader: bool,
):
    config = headless_tui_env
    servers = {}
    released = asyncio.Event()
    entered = asyncio.Event()
    anchor_id = ""
    aborts = []
    initial_content = "waiting historical anchor" if wrapped else "waiting saved answer"
    local_modes = not (switch_away or fail_sync or restore_anchor)
    original_await = RemoteSession._await_frontend

    async def await_frontend(remote, future, *, timeout, preempt=None):
        # Expire an AUTHENTICATED, deliberately held sync deterministically.
        # No healthy work competes against this test clock: release is closed.
        if remote.session_id == "waiting" and fail_sync:
            await entered.wait()
            timeout = 0
        return await original_await(remote, future, timeout=timeout, preempt=preempt)

    monkeypatch.setattr(RemoteSession, "_await_frontend", await_frontend)
    try:
        for sid in ("origin", "waiting", "neighbour"):
            directory = config / "sessions" / sid
            messages = [user_message(sid + " question"), assistant_message(sid + " saved answer")]
            if sid == "waiting" and restore_anchor:
                anchor = user_message("waiting historical anchor")
                anchor_id = anchor.id
                messages = (
                    [anchor]
                    + [
                        assistant_message(
                            f"Historical row {i} " + ("wrapped content " * 35 if wrapped else "")
                        )
                        for i in range(20 if wrapped else 200)
                    ]
                    + messages
                )
            await seed_transcript(directory, messages)
            owner = build_session(directory, ScriptedStream([]), cwd=config)
            handle = OwnedSessionHandle(owner, asyncio.get_running_loop(), cwd=str(config))
            if sid == "waiting":
                original_abort = owner.abort

                def record_abort(*args, **kwargs):
                    aborts.append(args)
                    return original_abort(*args, **kwargs)

                monkeypatch.setattr(owner, "abort", record_abort)
                original = handle.subscribe_frontend

                async def held_subscribe(*args, **kwargs):
                    entered.set()
                    await released.wait()
                    return await original(*args, **kwargs)

                monkeypatch.setattr(handle, "subscribe_frontend", held_subscribe)
            server = RuntimeServer(handle, kind="daemon")
            await server.start_in_process()
            servers[sid] = server

        def find(_directory, sid):
            server = servers.get(sid)
            return (server._record, server._record.pid) if server else (None, None)

        async def never():
            raise AssertionError("view navigation must never take execution ownership")

        async def resume(sid):
            return await RemoteSession.connect(
                servers[sid]._record,
                sid,
                config_dir=config,
                takeover_factory=never,
                display_window=True,
            )

        app = OperatorApp(lambda: resume("origin"), resume_factory=resume)
        with (
            patch("local_operator.mobile.attach_client.find_runtime_record", find),
            patch.object(OperatorApp, "_check_for_update", lambda self: None),
        ):
            async with app.run_test(size=(120, 36)) as pilot:
                await wait_for_adoption(app, pilot)
                # A provider placeholder may be adopted by tool NAME, but only
                # inside its own presentation. An outgoing bash row must never
                # become the incoming conversation's bash call.
                outgoing_card = ToolCard("pending-origin", "bash", {}, "Origin command")
                app._composing_cards["pending-origin"] = outgoing_card
                app._append_block(outgoing_card)
                if restore_anchor:
                    await app._sidebar_drafts.put(
                        "waiting", SessionDraft(following_tail=False, scroll_anchor_id=anchor_id)
                    )
                task = app._sidebar_navigation.select("waiting")
                await asyncio.wait_for(task, 10)
                await asyncio.wait_for(entered.wait(), 10)
                source = app._interaction
                assert not released.is_set()
                assert isinstance(app._session, RemoteSession)
                assert app._session.session_id == "waiting"
                assert initial_content in visible_text(app)
                assert app._sidebar_gate_surface_ready(source)
                assert app._adopt_composing_card("new-call", "bash") is None
                assert outgoing_card not in app._transcript_view().blocks()
                assert source.display_only and app.composer_submission_blocked()
                assert app._status is not None
                assert "Saved" in app._status.render_text(120).plain
                editor = app._editor()
                await pilot.press("ctrl+c")
                editor.text = "Keep this unsent draft"
                await pilot.press("enter")
                assert editor.text == "Keep this unsent draft"
                assert not app._session.frontend_state.streaming

                if local_modes:
                    _set_editor_line(editor, "/copy")
                    await pilot.press("escape", "enter")
                    assert isinstance(app.screen, CopyPickerScreen)
                    await pilot.press("escape")
                    editor.text = "Keep this unsent draft"
                    await pilot.press("f8")
                    assert app._aside_is_open()
                    assert editor.placeholder == ASIDE_PLACEHOLDER
                if move_reader:
                    await pilot.press("ctrl+end")
                    await pilot.pause()
                saved_view = app._transcript_view()
                saved_name = app._status.render_text(120).plain
                connection = source.connection_task
                # A slow source does not serialize the next local selection.
                if switch_away:
                    await asyncio.wait_for(app._sidebar_navigation.select("neighbour"), 10)
                    assert "neighbour saved answer" in visible_text(app)
                    assert source.draft.text == "Keep this unsent draft"
                    await asyncio.wait_for(app._sidebar_navigation.select("waiting"), 10)
                    assert app._transcript_view() is saved_view
                    assert source.connection_task is connection
                    assert app._status.render_text(120).plain == saved_name
                    assert not released.is_set()
                    assert initial_content in visible_text(app)
                    await asyncio.wait_for(app._sidebar_navigation.select("neighbour"), 10)
                if fail_sync:
                    assert source.connection_task is not None
                    # DRAINED TO EXHAUSTION, not awaited once. A failed connect
                    # is no longer terminal on its first attempt: a momentarily
                    # unreachable owner is retried up to
                    # `SIDEBAR_CONNECT_ATTEMPTS` times, each round a NEW task
                    # re-armed from the previous one's `finally`, and only a
                    # spent budget reaches the user. `fail_sync` holds the sync
                    # broken for every one of those attempts, so the chain runs
                    # the full budget and then latches — which is exactly the
                    # terminal state this block goes on to assert, reached the
                    # way the user now reaches it.
                    for _ in range(SIDEBAR_CONNECT_ATTEMPTS + 2):
                        task = source.connection_task
                        if task is None:
                            break
                        await asyncio.wait_for(asyncio.shield(task), 10)
                        await pilot.pause()
                        if source.connection_error or source.connection_task is task:
                            break
                    assert source.connection_error == "the runtime is not responding"
                    assert source.display_only and app.composer_submission_blocked()
                    assert initial_content in visible_text(app)
                    assert editor.text == "Keep this unsent draft"
                    # "Reconnect failed", not "Connection unavailable": the
                    # terminal copy now says the app already tried, since by the
                    # time it is shown a whole retry budget has been spent.
                    assert "Reconnect failed" in app._status.render_text(80).plain
                    assert "retry" in app._status.render_text(80).plain
                    fail_sync = False
                    released.set()
                    app._start_sidebar_connection(source)
                else:
                    released.set()
                assert source.connection_task is not None
                await asyncio.wait_for(asyncio.shield(source.connection_task), 10)
                if switch_away:
                    assert isinstance(app._session, RemoteSession)
                    assert app._session.session_id == "neighbour"
                    assert "waiting saved answer" not in visible_text(app)
                    # Returning uses its own canonical state and controller.
                    await asyncio.wait_for(app._sidebar_navigation.select("waiting"), 10)
                    await pilot.pause()
                    if source.connection_task is not None:
                        await asyncio.wait_for(asyncio.shield(source.connection_task), 10)
                if local_modes:
                    assert app._aside_is_open()
                    assert editor.placeholder == ASIDE_PLACEHOLDER
                    await pilot.press("escape")
                assert app._interaction is source
                assert editor.text == "Keep this unsent draft"
                assert not source.display_only
                if restore_anchor and not move_reader:
                    assert "waiting historical anchor" in visible_text(app)
                    assert not source.draft.following_tail
                    assert source.draft.scroll_anchor_id == anchor_id
                elif move_reader:
                    assert "waiting saved answer" in visible_text(app)
                    assert source.draft.following_tail
                else:
                    assert initial_content in visible_text(app)
                assert not app.composer_submission_blocked()
                await pilot.pause()
                assert aborts == [], "an offline shortcut queued a delayed owner abort"
                if local_modes:
                    missing = config / "sessions" / "deleted-target"
                    assert not missing.exists()
                    await asyncio.wait_for(app._sidebar_navigation.select("deleted-target"), 10)
                    assert app._interaction is source
                    assert editor.text == "Keep this unsent draft"
                    assert "waiting saved answer" in visible_text(app)
                    assert "no longer available" in visible_text(app)
                    assert not missing.exists()
    finally:
        released.set()
        for server in servers.values():
            await server.aclose()
