"""The away window: a snapshot page must never be a lossy view of the journal.

THE REPORT. In a conversation running a turn, the user sends a STEERING message
mid-turn, switches to another conversation, and switches back. Every durable row
written after the steer — the tool calls between the steer and the return, and
the assistant rows among them — never appears in the restored transcript; only
frames arriving after the return are painted ("I only start getting new messages
past that point"). Reported as "often times", so a timing-dependent boundary
rather than a missing feature.

THE MECHANISM, and it is a BOUND rather than a race. ``DesktopSessionBridge.
snapshot`` reads its transcript page with ``through_id=<frontend history_cursor>``.
That cursor is ``transcript.entries()[-1].id`` as of the owning store's last
``refresh_from_session`` (``frontend_state.py``, the ``history_cursor``
derivation) — a frontend REFRESH watermark, which a turn advances only at its
message/tool/turn boundaries, not on every append. The page read is a JOURNAL
read. Bounding one source's read by another source's watermark is what loses
rows: any row durable past the watermark is outside the page, and because the
bound row is still on disk ``read_transcript_page`` reports no ``cursor_missing``
— so the client's reducer, which reconciles through ``/history`` only for an
empty page or a missing cursor, accepts the short page as complete and never
learns the rows exist. The bound is not needed for pairing either: the client
merges a page durable-wins BY ID and reads neither cursor.

Two shapes of the same defect are pinned here, because the fix is one bound:

* the MIRROR goes stale while the turn keeps writing — the reported shape, and
  the reason the boundary lands on the steer (the drain is one of the refreshes
  that a viewer receives, so it is the last row a stale mirror knows);
* the OWNER goes away between two visits, leaving a previously-installed viewer
  cold while a successor's rows land — reachable without any staleness at all,
  and it needs no patched behaviour to reproduce.

The third test is the shape the client itself drives every time it is reopened
(fresh visitor, fresh dial): complete, ordered, and idempotent, which is also
the no-duplicate guard for the fix.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AgentTool, Message, TextContent, ToolResult
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.frontend_state import FrontendStateStore
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import (
    ScriptedStream,
    assistant_message,
    build_session,
    seed_transcript,
    text_turn,
    tool_call_turn,
    user_message,
)

#: A tool that parks until the test releases it, one event per call id. A long
#: ``wait`` in miniature: the point is a turn that is genuinely in flight, not a
#: stub that returns before anything can be observed.
PARKING_TOOL = "await_job"

#: Bound on an awaited state, never a budget to sleep through.
SETTLE_S = 30.0


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a headless boot touch the operator's live surfaces."""
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key, raising=False)


def _parking_tool(releases: dict[str, asyncio.Event], entered: list[str]) -> AgentTool:
    async def execute(
        call_id: str, _args: Any, _signal: Any, _on_update: Any, _context: Any
    ) -> ToolResult:
        entered.append(call_id)
        await releases[call_id].wait()
        return ToolResult(
            tool_call_id=call_id,
            tool_name=PARKING_TOOL,
            content=[TextContent(text=f"{call_id} finished")],
        )

    return AgentTool(
        name=PARKING_TOOL,
        label="Await",
        description="Awaits a background job.",
        parameters={
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
        execute=execute,
    )


async def _until(predicate: Any, why: str, timeout: float = SETTLE_S) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)
    assert predicate(), why


class _Bridge:
    """One real owner behind a real runtime server, and one real bridge.

    The owner is reached through the bridge's own attach path
    (``AttachedSession.cold`` -> ``attach_existing`` -> the discovery record), so
    a snapshot is taken exactly as the HTTP route takes it rather than through a
    stand-in for one.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.releases: dict[str, asyncio.Event] = {
            "call-1": asyncio.Event(),
            "call-2": asyncio.Event(),
        }
        self.entered: list[str] = []
        self.pool: DesktopSessions | None = None
        self.session_id = ""
        self.owner: Any = None
        self.handle: ServingSessionHandle | None = None
        self.server: RuntimeServer | None = None

    async def start(self, script: list[list[Any]], monkeypatch: pytest.MonkeyPatch) -> None:
        (self.root / "sessions").mkdir(parents=True, exist_ok=True)
        self.pool = DesktopSessions(self.root)
        self.session_id = await self.pool.create(str(self.root))
        self.directory = self.root / "sessions" / self.session_id
        await seed_transcript(
            self.directory,
            [user_message("first question"), assistant_message("first answer")],
        )
        await self.start_owner(script, monkeypatch)

    async def start_owner(self, script: list[list[Any]], monkeypatch: pytest.MonkeyPatch) -> None:
        """Bring up the owner this bridge's viewer dials, and point discovery at it."""
        self.owner = build_session(
            self.directory,
            ScriptedStream(script),
            tools=[_parking_tool(self.releases, self.entered)],
            cwd=self.root,
        )
        self.handle = ServingSessionHandle(
            self.owner, asyncio.get_running_loop(), cwd=str(self.root)
        )
        self.server = RuntimeServer(self.handle, kind="daemon")
        await self.server.start_in_process()
        assert self.server._record is not None

        def find(*_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
            server = self.server
            record = server._record if server is not None else None
            return (record, record.pid) if record is not None else (None, None)

        monkeypatch.setattr("local_operator.mobile.attach_client.find_runtime_record", find)

    async def stop_owner(self) -> None:
        """The owner goes away — a retirement, or a lap of the runtime's lease."""
        if self.server is not None:
            await self.server.aclose()
            self.server = None
        if self.handle is not None:
            with contextlib.suppress(Exception):
                await self.handle.dispose()
            self.handle = None

    @property
    def disk_ids(self) -> list[str]:
        """Every durable row id on the transcript, in order."""
        return [entry.id for entry in self.owner.transcript.entries()]

    async def visit(self) -> dict[str, Any]:
        """One visit's snapshot: acquire the bridge, read, release.

        Releasing is what a closed SSE stream does, and it DISPOSES the viewer
        (``DesktopSessionBridge._detach``), so a later visit re-dials and takes a
        fresh frontend sync exactly as a reopened conversation does.
        """
        assert self.pool is not None
        async with self.pool.session(self.session_id) as bridge:
            return await bridge.snapshot()

    @staticmethod
    def page_ids(snapshot: dict[str, Any]) -> list[str]:
        return [entry["id"] for entry in snapshot["payload"]["history"]["entries"]]

    async def prompt(self, text: str) -> asyncio.Task[Any]:
        return asyncio.create_task(self.owner.prompt(text))

    async def finish(self, task: asyncio.Task[Any], *call_ids: str) -> None:
        for call_id in call_ids:
            self.releases[call_id].set()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(task, SETTLE_S)

    async def dispose(self) -> None:
        for release in self.releases.values():
            release.set()
        await self.stop_owner()


@pytest.mark.asyncio
async def test_a_stale_mirror_after_a_steer_leaves_no_durable_row_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reported flow: steer mid-turn, then the mirror stops advancing.

    The steer is what makes the boundary visible rather than what causes it: the
    drain touches the owner's store, so the steer row is the LAST row a viewer
    that then stops receiving canonical updates ever learns about. Everything
    written after it — the tool call and the assistant row that carry the
    answer — is durable past the watermark, which is the exact "after the
    steering message" boundary the operator reported.

    The stale mirror is modelled by withholding the source refreshes that feed
    it, which is the half the client cannot see; the rows themselves are written
    by the real turn pipeline, and the assertion is on what the returning page
    delivers.
    """
    bridge = _Bridge(tmp_path)
    await bridge.start(
        [
            tool_call_turn(
                text="Waiting for the job.",
                tool_name=PARKING_TOOL,
                tool_call_id="call-1",
                arguments={"job_id": "7a73c97ffc54"},
            ),
            tool_call_turn(
                text="Now waiting on the second job.",
                tool_name=PARKING_TOOL,
                tool_call_id="call-2",
                arguments={"job_id": "91c0d6a1b2e3"},
            ),
        ],
        monkeypatch,
    )
    task: asyncio.Task[Any] | None = None
    try:
        task = await bridge.prompt("run the job")
        await _until(lambda: bridge.entered == ["call-1"], "the first job never started")
        steer = Message.user("actually, also check the logs")
        bridge.owner.steer_message(steer)
        bridge.releases["call-1"].set()
        await _until(
            lambda: bridge.entered == ["call-1", "call-2"], "the turn never reached the second job"
        )
        await _until(
            lambda: steer.id in bridge.disk_ids and bridge.owner.frontend_state.streaming,
            "the steer was never drained into a running turn",
        )

        # FROM HERE THE VIEWER LEARNS NOTHING FURTHER. The owner keeps writing.
        monkeypatch.setattr(FrontendStateStore, "refresh_from_session", lambda self, *a, **k: None)

        bridge.releases["call-2"].set()
        await _until(lambda: task.done(), "the turn never finished")
        on_disk = bridge.disk_ids
        after_the_steer = on_disk[on_disk.index(steer.id) + 1 :]
        assert after_the_steer, "the steer must be followed by real durable rows"

        returned = await bridge.visit()
        delivered = bridge.page_ids(returned)
        missing = [row for row in after_the_steer if row not in delivered]
        assert not missing, (
            "rows written after the steering message are missing from the returning page:"
            f" {missing} (on disk: {on_disk}, delivered: {delivered})"
        )
    finally:
        if task is not None:
            await bridge.finish(task, "call-2")
        await bridge.dispose()


@pytest.mark.asyncio
async def test_an_owner_lost_between_visits_leaves_no_durable_row_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No staleness needed: the owner goes away and a successor writes the rows.

    The installed viewer outlives its owner (the desktop surface is deliberately
    ``_can_go_cold``: it never takes work over, it waits to be re-dialled), so its
    frontend mirror is frozen at whatever row it last saw. The bridge here stays
    HELD across the away window — one long-lived read, the shape a stream that
    outlives a runtime replacement has — so the viewer is the installed one
    rather than a fresh facade. Rows a successor writes after that are durable
    past the frozen bound, and the page is short with ``cursor_missing: False``,
    so nothing tells the client to reconcile.
    """
    bridge = _Bridge(tmp_path)
    await bridge.start(
        [
            tool_call_turn(
                text="Waiting for the job.",
                tool_name=PARKING_TOOL,
                tool_call_id="call-1",
                arguments={"job_id": "7a73c97ffc54"},
            ),
        ],
        monkeypatch,
    )
    assert bridge.pool is not None
    try:
        # ONE hold, spanning the away window: acquire, read, leave the lease in
        # place while the owner is replaced underneath it.
        async with bridge.pool.session(bridge.session_id) as held:
            first = await held.snapshot()
            frozen_at = first["payload"]["frontend"]["snapshot"]["history_cursor"]
            assert frozen_at == bridge.disk_ids[-1], "the first read must end at the tail"

            # The away window. The owner that the viewer mirrors goes away, and a
            # successor runs a turn — the shape a scheduled wake, a TUI resume or
            # a second window produces while the desktop app is elsewhere.
            await bridge.stop_owner()
            await _until(
                lambda: held.remote is not None and held.remote.is_cold,
                "the viewer never noticed its owner going away",
            )
            await bridge.start_owner([text_turn("written while the client was away")], monkeypatch)
            await bridge.owner.prompt("a question asked while nobody watched")
            on_disk = bridge.disk_ids
            written_after = on_disk[on_disk.index(frozen_at) + 1 :]
            assert written_after, "the successor must have written something"

            returned = await held.snapshot()
            delivered = bridge.page_ids(returned)
            assert not returned["payload"]["history"][
                "cursor_missing"
            ], "the loss this test is about is SILENT: the bound row is still on disk"
            missing = [row for row in written_after if row not in delivered]
            assert not missing, (
                "durable rows written while the client was away are missing from the page:"
                f" {missing} (on disk: {on_disk}, delivered: {delivered})"
            )
    finally:
        await bridge.dispose()


@pytest.mark.asyncio
async def test_a_reopened_conversation_delivers_the_tail_once_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh visitor: complete, in transcript order, and duplicate-free.

    The shape a reopened conversation actually drives, and the no-duplicate
    guard for the fix as well: the client merges a page durable-wins BY ID, so
    an id repeated inside one page — or a second read growing it — is the
    regression a careless "re-paint on return" fix would introduce.
    """
    bridge = _Bridge(tmp_path)
    await bridge.start([text_turn("written while the client was away")], monkeypatch)
    try:
        first = await bridge.visit()
        assert first["payload"]["history"]["entries"], "the first visit read nothing"
        assert (
            bridge.page_ids(first) == bridge.disk_ids
        ), "the first visit must deliver the whole durable transcript"
        await bridge.owner.prompt("a question asked while nobody watched")
        on_disk = bridge.disk_ids
        returned = await bridge.visit()
        delivered = bridge.page_ids(returned)
        assert len(set(delivered)) == len(delivered), f"the page repeats a row: {delivered}"
        assert delivered == [
            row for row in on_disk if row in set(delivered)
        ], f"the page is not in transcript order: {delivered} vs {on_disk}"
        assert delivered == on_disk, f"the page must carry every durable row: {on_disk}"
        assert bridge.page_ids(await bridge.visit()) == delivered, "a second read grew the page"
    finally:
        await bridge.dispose()
