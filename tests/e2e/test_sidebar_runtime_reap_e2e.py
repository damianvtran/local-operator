"""Switching through the sidebar must not strand one runtime per conversation.

The operator's report: clicking conversations in the sidebar to "check" them
left every one of those runtimes resident, and an empty ``Untitled
conversation`` sat there too. Measured on the reporting host: ONE TUI process
held 15 established attach sockets, one per session ever clicked, and 15
runtime children (~1.9 GB RSS) that had been idle for half an hour.

These tests drive the REAL app against REAL runtime servers and assert on
process-level facts (is the viewer's socket still connected, did the runtime
retire) rather than on widget state, because the leak is invisible on screen —
the conversation the user is looking at is correct either way.

THE COMPRESSED TIMEOUT BELOW IS DELIBERATE — do not "restore" a real-time
budget. These tests were written during diagnosis, before the fix was designed,
so they originally expected release to be IMMEDIATE. The shipped fix releases a
parked viewer after ``SIDEBAR_IDLE_RELEASE_S`` (5 minutes in production) so that
switching back to a recent conversation stays a warm cache hit; waiting that out
here would make the stage take five minutes per test.

What is asserted is unchanged and is the whole acceptance criterion: after the
clock expires, a conversation the user merely visited holds NO attach client, so
its runtime can reap. Only the wall-clock is compressed, by patching the module
constant the sweep re-reads every tick. Loosening or deleting those assertions
would defeat the purpose of the file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tui import app as app_module
from local_operator.tui.app import OperatorApp
from tests.e2e.harness import (
    ScriptedStream,
    assistant_message,
    build_session,
    seed_transcript,
    user_message,
    wait_for_adoption,
)


def _compress_idle_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the production reap path on a sub-second clock.

    The sweep re-reads both constants on every tick, so patching the module
    attributes exercises the real timer, the real predicate and the real
    release worker — nothing about the mechanism is stubbed, only the wait.
    """
    monkeypatch.setattr(app_module, "SIDEBAR_IDLE_RELEASE_S", 0.2)
    monkeypatch.setattr(app_module, "SIDEBAR_IDLE_SWEEP_S", 0.05)


async def _stand_up(config: Path, ids: list[str]) -> dict[str, RuntimeServer]:
    """One real runtime server per conversation, each with durable history."""
    servers: dict[str, RuntimeServer] = {}
    for sid in ids:
        directory = config / "sessions" / sid
        await seed_transcript(
            directory,
            [user_message(f"{sid} question"), assistant_message(f"{sid} saved answer")],
        )
        owner = build_session(directory, ScriptedStream([]), cwd=config)
        handle = OwnedSessionHandle(owner, asyncio.get_running_loop(), cwd=str(config))
        server = RuntimeServer(handle, kind="daemon")
        await server.start_in_process()
        servers[sid] = server
    return servers


@pytest.mark.asyncio
async def test_visited_conversations_do_not_hold_their_runtimes_open(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Click through five conversations; only the current one may stay attached.

    ``attach_clients()`` is the runtime's own count of interactive viewers and
    is exactly the term ``process._should_exit`` reads as "somebody is looking
    at this" — so a runtime a user merely glanced at reporting 1 here is the
    leak itself, in the units the reaper cares about.
    """
    config = headless_tui_env
    ids = ["origin", "alpha", "beta", "gamma", "delta"]
    servers = await _stand_up(config, ids)
    _compress_idle_clock(monkeypatch)

    def find(_directory, sid):
        server = servers.get(sid)
        return (server._record, server._record.pid) if server else (None, None)

    async def never():
        raise AssertionError("a sidebar viewer never takes over a session")

    async def resume(sid):
        return await RemoteSession.connect(
            servers[sid]._record,
            sid,
            config_dir=config,
            takeover_factory=never,
            display_window=True,
        )

    app = OperatorApp(lambda: resume("origin"), resume_factory=resume)
    try:
        with (
            patch("local_operator.mobile.attach_client.find_runtime_record", find),
            patch.object(OperatorApp, "_check_for_update", lambda self: None),
        ):
            async with app.run_test(size=(120, 36)) as pilot:
                await wait_for_adoption(app, pilot)
                for sid in ids[1:]:
                    await asyncio.wait_for(app._sidebar_navigation.select(sid), 15)
                    await pilot.pause()
                # Let every release worker the switches queued run to
                # completion; the app releases sources from workers, not
                # inline, so an immediate assertion would race them.
                for _ in range(40):
                    await pilot.pause()
                    await asyncio.sleep(0.05)

                current = str(getattr(app._session, "session_id", ""))
                assert current == "delta", "the last click is the visible conversation"

                # `attach_clients()` unconditionally: the runtime has no
                # `_handle_attach_count`, so the defensive branch this replaced
                # was dead code that only cost a type error.
                held = {sid: server.attach_clients() for sid, server in servers.items()}
                stranded = {sid: count for sid, count in held.items() if count and sid != current}
                assert not stranded, (
                    "conversations the user merely visited still hold an attached "
                    f"viewer, so their runtimes can never reap: {stranded} "
                    f"(all counts: {held})"
                )
    finally:
        for server in servers.values():
            await server.aclose()


@pytest.mark.asyncio
async def test_an_empty_conversation_visited_and_left_retires_its_runtime(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``Untitled conversation`` case: nothing durable, so nothing to keep.

    An empty conversation is still REACHABLE while parked — the user can click
    back onto the ``/new`` they just left — so it waits out the same clock as
    any other source rather than being disposed at park time. What makes it
    special is the offer made when that clock expires: the release path asks
    the runtime to retire, and a runtime with nothing durable accepts, so it
    goes immediately instead of waiting for the residency drain.

    The judgment stays the RUNTIME's (``retire_if_unused`` -> ``is_pristine``):
    only it can see an armed wake, a peer message that just landed, or a second
    attached viewer.
    """
    config = headless_tui_env
    servers = await _stand_up(config, ["origin"])
    _compress_idle_clock(monkeypatch)

    # The empty one: a real runtime whose session has no durable row at all.
    empty_dir = config / "sessions" / "untitled"
    empty_dir.mkdir(parents=True, exist_ok=True)
    owner = build_session(empty_dir, ScriptedStream([]), cwd=config)
    handle = OwnedSessionHandle(owner, asyncio.get_running_loop(), cwd=str(config))
    empty = RuntimeServer(handle, kind="daemon")
    await empty.start_in_process()
    servers["untitled"] = empty

    assert handle.is_pristine(), "the fixture must actually be an empty session"

    def find(_directory, sid):
        server = servers.get(sid)
        return (server._record, server._record.pid) if server else (None, None)

    async def never():
        raise AssertionError("a sidebar viewer never takes over a session")

    async def resume(sid):
        return await RemoteSession.connect(
            servers[sid]._record,
            sid,
            config_dir=config,
            takeover_factory=never,
            display_window=True,
        )

    app = OperatorApp(lambda: resume("origin"), resume_factory=resume)
    try:
        with (
            patch("local_operator.mobile.attach_client.find_runtime_record", find),
            patch.object(OperatorApp, "_check_for_update", lambda self: None),
        ):
            async with app.run_test(size=(120, 36)) as pilot:
                await wait_for_adoption(app, pilot)
                await asyncio.wait_for(app._sidebar_navigation.select("untitled"), 15)
                await pilot.pause()
                await asyncio.wait_for(app._sidebar_navigation.select("origin"), 15)
                for _ in range(40):
                    await pilot.pause()
                    await asyncio.sleep(0.05)

                assert empty.attach_clients() == 0, (
                    "an empty conversation the user looked at and left still holds "
                    "a viewer, so its runtime never retires"
                )
    finally:
        for server in servers.values():
            await server.aclose()
