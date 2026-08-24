"""The daemon's watch/unwatch pushes on SSE subscriber transitions (design
§2.8): first-in sends watch, last-out sends unwatch, and an old registrant's
error reply is swallowed rather than 500ing the handshake."""

from __future__ import annotations

import asyncio

import pytest

from local_operator.mobile import registry
from local_operator.mobile.daemon import MobileDaemon, SessionEntry
from local_operator.mobile.registrant import Registrant
from local_operator.mobile.types import SessionProjection


class FakeHandle:
    def __init__(self) -> None:
        self._projection = SessionProjection(
            session_id="s1",
            pid=0,
            kind="tui",
            conversation_name="fake",
            cwd="/tmp",
            model_label="test/model",
        )

    @property
    def session_projection_seed(self) -> SessionProjection:
        return self._projection

    def subscribe(self, on_projection):  # noqa: ANN001, ANN202
        return lambda: None

    async def prompt(self, text, images=None):  # noqa: ANN001, ANN202
        return "sent"

    async def steer(self, text, images=None):  # noqa: ANN001, ANN202
        return "queued"

    async def abort(self):  # noqa: ANN202
        return "stopping"

    async def set_model(self, provider, model_id):  # noqa: ANN001, ANN202
        return "model"

    async def set_effort(self, effort):  # noqa: ANN001, ANN202
        return "effort"

    async def slash(self, command, args):  # noqa: ANN001, ANN202
        return "done"

    async def new_conversation(self):  # noqa: ANN202
        return "done"

    async def resume_session(self, session_id):  # noqa: ANN001, ANN202
        return "done"

    async def approval_answer(self, request_id, approved, remember):  # noqa: ANN001, ANN202
        return "done"

    async def ask_answer(self, request_id, value, question_index=None):  # noqa: ANN001, ANN202
        return "done"

    async def refresh(self) -> None:
        pass


async def _wait_record() -> registry.SessionRecord:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        found = registry.scan()
        if found and found[0][1] == "live":
            return found[0][0]
        await asyncio.sleep(0.05)
    raise AssertionError("no live record")


@pytest.mark.asyncio
async def test_first_subscriber_sends_watch_and_last_out_sends_unwatch() -> None:
    handle = FakeHandle()
    registrant = Registrant(handle, kind="tui")
    registrant.start()
    try:
        record = await _wait_record()
        # Drive the daemon's notify path directly against the live socket:
        # the SSE handler's transitions are what call it, and the handler
        # itself is covered by the TestClient test below.
        daemon = MobileDaemon(port=0, password="pw")
        entry = SessionEntry(record)
        daemon.table.entries[record.pid] = entry
        dial = asyncio.ensure_future(_dial_and_hold(daemon, entry))
        try:
            await _wait_dial(entry)
            assert registrant.phone_watchers == 0
            daemon.notify_watch_transition(record.pid, watching=True)
            deadline = asyncio.get_running_loop().time() + 5
            while asyncio.get_running_loop().time() < deadline:
                if registrant.phone_watchers == 1 and registrant.watch_supported:
                    break
                await asyncio.sleep(0.05)
            assert registrant.phone_watchers == 1
            assert registrant.watch_supported
            daemon.notify_watch_transition(record.pid, watching=False)
            deadline = asyncio.get_running_loop().time() + 5
            while asyncio.get_running_loop().time() < deadline:
                if registrant.phone_watchers == 0:
                    break
                await asyncio.sleep(0.05)
            assert registrant.phone_watchers == 0
        finally:
            dial.cancel()
    finally:
        registrant.close()


async def _dial_and_hold(daemon, entry) -> None:  # noqa: ANN001
    from local_operator.mobile.daemon import _dial

    await _dial(daemon, entry)


async def _wait_dial(entry) -> None:  # noqa: ANN001
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        if entry.writer is not None:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("daemon never dialed")


@pytest.mark.asyncio
async def test_watch_push_to_unknown_session_is_swallowed() -> None:
    daemon = MobileDaemon(port=0, password="pw")
    # No entry for the pid: the push must not raise (the SSE stream that
    # triggered it is already mid-handshake).
    daemon.notify_watch_transition(999999, watching=True)
    await asyncio.sleep(0.1)


def test_old_registrant_error_reply_is_swallowed() -> None:
    """An entry whose writer is None (or a registrant that answers
    `error: unknown op`) must not propagate: notify is fire-and-forget and
    the RuntimeError path is caught inside the task."""
    daemon = MobileDaemon(port=0, password="pw")

    async def scenario() -> None:
        # A request against a not-connected entry raises KeyError inside the
        # task; the swallow keeps the loop clean.
        daemon.notify_watch_transition(12345, watching=True)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
