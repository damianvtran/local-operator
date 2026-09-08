"""Real styled TUI geometry with a controlled host-probe boundary.

These are render-policy tests, not evidence that the OS focused a real window.
Native focus evidence is captured separately using the read-only host protocol.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest
from textual.events import AppBlur, AppFocus
from textual.screen import Screen

from local_operator.harness.types import Message, TextContent
from local_operator.session.attention import AttentionStore
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class ReceiptSession(FakeSession):
    def __init__(self, path: Path, *, long: bool = False) -> None:
        super().__init__()
        self.store = AttentionStore(path)
        self.token = str(uuid.uuid4())
        text = "Finished result\n\n" + (
            "A long result line\n\n" * 90 if long else "The final answer is visible."
        )
        self.result = Message(role="assistant", content=[TextContent(text=text)])
        self._history = [self.result]
        self.store.publish("session/sess", self.token, self.result.id, "complete")

    async def refresh_attention(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.state, "session/sess")

    async def acknowledge_attention(self, token: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.acknowledge, "session/sess", token)


@pytest.mark.asyncio
async def test_default_focus_is_not_proof_and_rendered_focus_acknowledges(
    tmp_path, monkeypatch
) -> None:
    session = ReceiptSession(tmp_path / "attention.db")
    probes: list[bool] = []
    monkeypatch.setattr(
        "local_operator.tui.attention.terminal_is_foreground", lambda: probes.append(True) or True
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        # Boot history is mounted by a worker, then laid out on a later frame.
        # Wait on the actual geometry, not a fixed wall-clock delay.
        for _ in range(50):
            await pilot.pause()
            if app._completion_anchor_visible(session.result.id):
                break
        assert app._completion_anchor_visible(session.result.id)
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["unseen"]
        assert not probes
        app.on_app_focus(AppFocus())
        await app._poll_completion_attention()
        assert not session.store.state("session/sess")["unseen"]
        count = len(probes)
        await app._poll_completion_attention()
        assert len(probes) == count


@pytest.mark.asyncio
@pytest.mark.parametrize("follow", [False, True])
@pytest.mark.parametrize("already_read", [False, True])
async def test_old_failure_is_not_inserted_at_a_new_retry_tail(
    tmp_path, follow, already_read
) -> None:
    from local_operator.harness.types import StreamEndEvent
    from local_operator.paths import config_dir
    from local_operator.session.remote import RemoteSession
    from local_operator.session.runtime.owned import OwnedSessionHandle
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.tui.widgets.transcript import NoticeBlock, UserBlock
    from tests.unit.session.test_session import make_session

    calls = 0
    started, release = asyncio.Event(), asyncio.Event()

    async def stream(request, signal):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield StreamEndEvent(stop_reason="error", error="Old failure")
            return
        started.set()
        await release.wait()
        # A quiet settled retry leaves the older unread outcome authoritative;
        # it still must not acquire a newly appended historical-error marker.
        yield StreamEndEvent(stop_reason="stop")

    session = make_session(tmp_path, stream)
    await session.prompt("Old request")
    old = await session.refresh_attention()
    if already_read:
        await session.acknowledge_attention(old["completion_token"])
    runtime = RuntimeServer(
        OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path)), kind="daemon"
    )
    await runtime.start_in_process()
    retry = asyncio.create_task(session.prompt("New retry is running"))
    await asyncio.wait_for(started.wait(), 10)
    source: Any = session

    async def never(*args, **kwargs):
        raise AssertionError("live fixture must not take over")

    try:
        if follow:
            source = await RemoteSession.connect(
                runtime._record, session.session_id, config_dir=config_dir(), takeover_factory=never
            )

        async def factory():
            return source

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(50):
                await pilot.pause()
                if app._session is source and len(app._transcript_view().query(UserBlock)) > 0:
                    break
            assert source.is_streaming
            await app._poll_completion_attention()
            await pilot.pause()
            assert not any(
                block.completion_anchor_id == old["anchor_id"]
                for block in app._transcript_view().query(NoticeBlock)
            )
            release.set()
            await retry
            for _ in range(50):
                await pilot.pause()
                if not source.is_streaming:
                    break
            await app._poll_completion_attention()
            assert not any(
                block.completion_anchor_id == old["anchor_id"]
                for block in app._transcript_view().query(NoticeBlock)
            )
    finally:
        release.set()
        await asyncio.gather(retry, return_exceptions=True)
        if follow:
            await source.dispose()
        await runtime.aclose()
        await session.dispose()


@pytest.mark.asyncio
async def test_overlay_scrollback_and_blur_do_not_acknowledge(tmp_path, monkeypatch) -> None:
    session = ReceiptSession(tmp_path / "attention.db", long=True)
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(50):
            await pilot.pause()
            if app._completion_anchor_visible(session.result.id):
                break
        assert app._completion_anchor_visible(session.result.id)
        app.on_app_focus(AppFocus())
        app._transcript_view().scroll_home(animate=False)
        await pilot.pause()
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["unseen"]
        app._transcript_view().scroll_end(animate=False)
        app.push_screen(Screen())
        await pilot.pause()
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["unseen"]
        app.on_app_blur(AppBlur())
        app.pop_screen()
        await pilot.pause()
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["unseen"]


class InterruptSession(FakeSession):
    """An attention-backed fake with nothing published until a test says so.

    ``ReceiptSession`` publishes a COMPLETE outcome in its constructor. The
    duplicate-row defect needs the opposite order — the app paints its own
    live row first, and the durable ``interrupted`` outcome is published
    afterwards, exactly as a real session does (it mints
    ``completion-<token>`` when it publishes, which is strictly after the turn
    has ended on screen).
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.store = AttentionStore(path)
        self.identity = "session/interrupted"

    def publish_interrupted(self) -> str:
        token = str(uuid.uuid4())
        self.store.publish(self.identity, token, f"completion-{token}", "interrupted")
        return token

    async def refresh_attention(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.state, self.identity)

    async def acknowledge_attention(self, token: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.acknowledge, self.identity, token)


async def _interrupt_a_turn(app: OperatorApp, pilot: Any) -> None:
    """Run a turn to the point where the app has painted its own abort row."""
    from local_operator.tui.events import TurnEnded, TurnStarted
    from local_operator.tui.widgets.editor import Editor

    for _ in range(200):
        if app._session is not None:
            break
        await pilot.pause()
        await asyncio.sleep(0.01)
    editor = app.query_one(Editor)
    editor.focus()
    editor.text = "run the long job"
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    app.post_message(TurnStarted())
    await pilot.pause()
    app.post_message(TurnEnded(True, None))
    await pilot.pause()
    await pilot.pause()


def _notice_texts(app: OperatorApp) -> list[str]:
    from local_operator.tui.widgets.transcript import NoticeBlock

    return [
        block._text for block in app._transcript_view().blocks() if isinstance(block, NoticeBlock)
    ]


@pytest.mark.asyncio
async def test_one_interruption_is_stated_once(tmp_path) -> None:
    """Two producers, one outcome, one row.

    ``_finalize_turn`` announces the abort live; the attention poller reads the
    same interruption back out of the durable store a tick later and appended a
    SECOND row, because its only dedupe is ``completion_anchor_id`` and the
    live row cannot carry one (the anchor does not exist until the session
    publishes). The user saw ``! interrupted`` above ``· Interrupted``.
    """
    from local_operator.tui.widgets.transcript import NoticeBlock

    session = InterruptSession(tmp_path / "attention.db")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _interrupt_a_turn(app, pilot)
        assert _notice_texts(app) == ["interrupted"], "the turn states its own outcome"

        # Now the session publishes that same interruption durably.
        token = session.publish_interrupted()
        await app._poll_completion_attention()
        await pilot.pause()

        assert _notice_texts(app) == ["interrupted"], "and it is not restated"
        # The live row ADOPTED the anchor rather than being shadowed by a
        # second one, which is what lets looking at it mark the outcome read.
        anchor = session.store.state(session.identity)["anchor_id"]
        assert [
            block.completion_anchor_id
            for block in app._transcript_view().query(NoticeBlock)
            if block.completion_anchor_id
        ] == [anchor]
        assert token


@pytest.mark.asyncio
async def test_an_interruption_this_app_never_painted_still_gets_a_row(tmp_path) -> None:
    """The poller keeps the case it exists for: a session that stopped away.

    Proves the fix suppresses a DUPLICATE, not the attention notice itself —
    with no live row to adopt, a user returning to the session must still be
    told it was interrupted.
    """
    session = InterruptSession(tmp_path / "attention.db")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        # No turn ran in THIS app; the outcome was produced elsewhere.
        session.publish_interrupted()
        await app._poll_completion_attention()
        await pilot.pause()

        assert _notice_texts(app) == ["Interrupted"]


@pytest.mark.asyncio
async def test_a_later_outcome_never_adopts_an_earlier_turns_row(tmp_path) -> None:
    """The held row is turn-scoped: a new turn makes it unadoptable.

    Otherwise a second interruption would stamp its anchor onto the FIRST
    turn's row — marking an old outcome read while the new one gets no row at
    all.
    """
    from local_operator.tui.events import TurnStarted

    session = InterruptSession(tmp_path / "attention.db")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _interrupt_a_turn(app, pilot)
        assert _notice_texts(app) == ["interrupted"]

        # A NEW turn opens, and only then does an outcome get published.
        app.post_message(TurnStarted())
        await pilot.pause()
        session.publish_interrupted()
        await app._poll_completion_attention()
        await pilot.pause()

        assert _notice_texts(app) == ["interrupted", "Interrupted"]


@pytest.mark.asyncio
async def test_the_adopted_row_can_be_acknowledged(tmp_path, monkeypatch) -> None:
    """Adoption stamps a real anchor, so looking at the row marks it read.

    Not incidental to the duplicate fix but the other half of it. The receipt
    is cleared by `_completion_anchor_visible` finding the ANCHORED block in
    the viewport; suppressing the poller's row without stamping the live one
    would leave an interruption the user is looking at permanently unseen, and
    the sidebar flagging a session whose outcome is on screen.
    """
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    session = InterruptSession(tmp_path / "attention.db")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _interrupt_a_turn(app, pilot)
        session.publish_interrupted()
        app.on_app_focus(AppFocus())
        await app._poll_completion_attention()
        await pilot.pause()

        anchor = session.store.state(session.identity)["anchor_id"]
        assert app._completion_anchor_visible(anchor), "the adopted row is the anchor"
        # Wait on the acknowledgement the poll publishes, not on a clock.
        for _ in range(20):
            await app._poll_completion_attention()
            await pilot.pause()
            if not session.store.state(session.identity)["unseen"]:
                break
        assert not session.store.state(session.identity)["unseen"]
