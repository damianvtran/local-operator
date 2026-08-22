"""Owned-session handle behaviours that have no terminal to fall back on:
full-auto approval, headless conversation naming, and the concurrent-approval
queue wired through the fold.

These are the phone-started-session equivalents of things the TUI's
OperatorApp does (adopt ``tool_approval_mode: auto`` at boot, run the naming
worker). The handle is exercised directly with a minimal fake session so the
tests stay off the real provider and event loop machinery.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.mobile.owned import OwnedSessionHandle


class FakeSession:
    """The slice of Session the OwnedSessionHandle touches in these tests."""

    def __init__(self) -> None:
        self.session_id = "sess-1"
        self.model_label = "test/model"
        self.model = None
        self.conversation_name = ""
        self.is_streaming = False
        self._handlers: list = []
        self._named: list[tuple[str, bool]] = []
        self._complete_calls: list[tuple[str, str]] = []
        # The naming call's reply must be wrapped in the <title> tag the
        # parser expects; a bare string is treated as a model that ignored the
        # format and discarded.
        self.title_reply = "<title>A Neat Title</title>"

    # -- naming seams ----------------------------------------------------------
    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        self.conversation_name = text
        self._named.append((text, user_set))
        return text

    async def complete_once(self, system: str, prompt: str) -> str:
        self._complete_calls.append((system, prompt))
        return self.title_reply

    # -- gate registration -----------------------------------------------------
    def set_approval_handler(self, handler) -> None:
        self._approval_handler = handler

    def set_ask_handler(self, handler) -> None:
        self._ask_handler = handler

    # -- subscribe/selectors the handle reads at construction ------------------
    def subscribe(self, handler):
        # Capture the handler so a test can drive the fold with real events,
        # the way the live session's event stream does.
        self._handlers.append(handler)

        def _unsub() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return _unsub

    def emit(self, event) -> None:
        for handler in list(getattr(self, "_handlers", [])):
            handler(event)

    def history(self):  # pragma: no cover - not exercised here
        return []

    @property
    def reasoning_effort(self):  # pragma: no cover
        return "auto"


def make_handle(auto_approve: bool = False) -> tuple[OwnedSessionHandle, FakeSession]:
    # The handle records whichever loop it is built on; inside an async test
    # that is the running loop, and the sync construction test never awaits so
    # a fresh loop is fine. get_event_loop_policy().get_event_loop() avoids the
    # "no current event loop" deprecation of the bare accessor.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    session = FakeSession()
    handle = OwnedSessionHandle(session, loop, cwd="/tmp", auto_approve=auto_approve)
    return handle, session


def test_default_conversation_name_is_empty_not_a_placeholder() -> None:
    """A fresh mobile session shows nothing until named, so the phone's own
    fallback ("untitled" / cwd) applies — never a frozen "mobile session"."""
    handle, _ = make_handle()
    assert handle.session_projection_seed.conversation_name == ""


@pytest.mark.asyncio
async def test_full_auto_approves_inline_without_a_card() -> None:
    """With the owner's saved default at full-auto, the gate answers True
    inline and never parks a pending card — matching the TUI's auto mode."""
    handle, _ = make_handle(auto_approve=True)
    gate = handle._approval_gate
    approved = await gate("bash", "rm -rf build/")
    assert approved is True
    # No card was ever queued.
    assert handle._fold.projection.pending is None
    assert handle._fold.projection.pending_count == 0


@pytest.mark.asyncio
async def test_ask_mode_parks_a_card_then_resolves() -> None:
    """Without full-auto, the gate queues a card and blocks until answered —
    and a second concurrent gate queues behind it rather than overwriting."""
    handle, _ = make_handle(auto_approve=False)
    gate = handle._approval_gate

    first = asyncio.ensure_future(gate("bash", "one"))
    second = asyncio.ensure_future(gate("write", "two"))
    await asyncio.sleep(0)  # let both gates enqueue

    assert handle._fold.projection.pending_count == 2
    front = handle._fold.projection.pending
    assert front is not None and front.title == "bash"

    # Answer the front; the second card surfaces, count drops.
    await handle.approval_answer(front.request_id, True, False)
    assert await first is True
    await asyncio.sleep(0)
    assert handle._fold.projection.pending_count == 1
    nxt = handle._fold.projection.pending
    assert nxt is not None and nxt.title == "write"

    await handle.approval_answer(nxt.request_id, False, False)
    assert await second is False
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is None


@pytest.mark.asyncio
async def test_naming_worker_stores_a_title_once() -> None:
    """The first substantive prompt names an unnamed session; a low-signal
    opener does not consume the one attempt."""
    handle, session = make_handle()

    # Low-signal opener: skipped, latch not spent.
    handle._maybe_name_conversation("hi")
    assert handle._name_requested is False

    handle._maybe_name_conversation("please refactor the billing importer")
    assert handle._name_requested is True
    # Let the background naming task run.
    for _ in range(5):
        await asyncio.sleep(0)
    assert session.conversation_name == "A Neat Title"
    assert session._named == [("A Neat Title", False)]

    # A second prompt does not re-name.
    handle._maybe_name_conversation("and now the invoices too")
    for _ in range(5):
        await asyncio.sleep(0)
    assert session._named == [("A Neat Title", False)]


@pytest.mark.asyncio
async def test_agent_end_clears_streaming_despite_stale_is_streaming() -> None:
    """Regression: the phone stayed pinned to "in progress" after a turn ended.

    The session emits ``AgentEndEvent`` while its ``is_streaming`` flag is
    STILL True — the flag clears only in the turn's ``finally`` block, after
    the event has been emitted and folded. The per-event ``_refresh_state``
    used to re-read that stale True and overwrite the fold's correct
    ``streaming=False``; because the end event is the turn's last event, no
    later push ever corrected it and the session list shimmered forever.

    This drives the real handler wiring (subscribe → emit) and asserts the
    projection settles to ``streaming=False`` even though ``is_streaming`` is
    left True, exactly as the live session leaves it at the emit point.
    """
    from local_operator.harness.types import AgentEndEvent, AgentStartEvent

    handle, session = make_handle()
    handle.subscribe(lambda: None)

    # Turn starts: the session marks itself streaming and emits the start.
    session.is_streaming = True
    session.emit(AgentStartEvent(generation=1))
    assert handle._fold.projection.streaming is True

    # Turn ends: the session emits AgentEndEvent BEFORE clearing the flag,
    # reproducing the real ordering (is_streaming still True at emit time).
    session.emit(AgentEndEvent(aborted=False, generation=1))

    assert handle._fold.projection.streaming is False, (
        "projection stuck streaming=True after AgentEndEvent — the per-event "
        "refresh clobbered the fold with the not-yet-cleared is_streaming flag"
    )
    assert handle._fold.projection.stop_reason == "completed"
    assert handle._fold.projection.activity == ""


@pytest.mark.asyncio
async def test_command_boundary_reconcile_cannot_restick_after_abort() -> None:
    """F1 regression: on the abort/error path the session emits AgentEndEvent
    INLINE while its ``is_streaming`` flag is still True (it clears several
    awaits later, in the turn's ``finally``). A mobile command landing in that
    window runs ``refresh`` → ``_reconcile_streaming`` with the stale True. The
    fold must ignore it because it has already folded a terminal event, so the
    projection stays ``streaming=False`` instead of re-sticking to "in
    progress" with no later event to correct it.
    """
    from local_operator.harness.types import AgentEndEvent, AgentStartEvent

    handle, session = make_handle()
    handle.subscribe(lambda: None)

    session.is_streaming = True
    session.emit(AgentStartEvent(generation=1))
    # Aborted turn: end event folded, but the session flag has NOT cleared yet.
    session.emit(AgentEndEvent(aborted=True, generation=1))
    assert handle._fold.projection.streaming is False

    # A command lands before the finally clears is_streaming: reconcile sees
    # the stale True and must NOT raise streaming back up.
    await handle.refresh()
    assert handle._fold.projection.streaming is False, (
        "command-boundary reconcile re-stuck streaming=True from the stale "
        "is_streaming flag on the abort path"
    )

    # A genuine next turn still reconciles up normally (latch cleared on start).
    session.emit(AgentStartEvent(generation=2))
    assert handle._fold.projection.streaming is True


@pytest.mark.asyncio
async def test_attach_seeds_streaming_from_flag_for_mid_turn_subscriber() -> None:
    """A phone that subscribes mid-turn never saw the AgentStartEvent, so the
    fold alone would open on a stale ``streaming=False``. Attach seeds the live
    flag once from the session so the working line paints immediately."""
    handle, session = make_handle()
    session.is_streaming = True
    handle.subscribe(lambda: None)
    assert handle._fold.projection.streaming is True
