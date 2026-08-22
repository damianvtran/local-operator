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
import json

import pytest

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.mobile.owned import OwnedSessionHandle


class FakeSession:
    """The slice of Session the OwnedSessionHandle touches in these tests."""

    def __init__(self) -> None:
        self.session_id = "sess-1"
        self.model_label = "test/model"
        self.model = None
        self.conversation_name = ""
        self.is_streaming = False
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
    def subscribe(self, handler):  # pragma: no cover - not exercised here
        return lambda: None

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
async def test_ask_gate_projects_serializable_option_labels() -> None:
    """An owned ask WITH options must project a JSON-serializable card.

    Regression: the gate used to push ``options=list(first.options)`` \u2014 raw
    AskOption pydantic models \u2014 into PendingRequest, whose ``to_json`` is
    ``asdict`` and leaves those models as objects ``json.dumps`` cannot encode.
    A daemon-owned session that asked a question with options therefore crashed
    the very projection push meant to show the card on the phone. The gate now
    projects option LABELS (strings), which is also what the web card renders
    and hands back in ask_answer. This test fails on the old code (the
    ``json.dumps`` below raises ``TypeError: Object of type AskOption is not
    JSON serializable``) and passes now.
    """
    handle, _ = make_handle(auto_approve=False)
    gate = handle._ask_gate

    question = AskQuestion(
        id="stale",
        question="What should happen to the stale rows?",
        options=[
            AskOption(label="Drop them", description="nothing reads the column"),
            AskOption(label="Backfill", description="slower, keeps history"),
        ],
    )
    asked = asyncio.ensure_future(gate([question]))
    await asyncio.sleep(0)  # let the gate enqueue its card

    pending = handle._fold.projection.pending
    assert pending is not None
    assert pending.kind == "ask"
    assert pending.options == ["Drop them", "Backfill"]

    # The whole point: the projection round-trips over the wire. This is the
    # line that raised before the fix.
    wire = json.dumps(handle._fold.projection.to_json())
    assert '"Drop them"' in wire and '"Backfill"' in wire

    # Answer it back with a label, exactly as the phone does, so the parked
    # gate resolves under the question's id and the test leaves nothing hanging.
    await handle.ask_answer(pending.request_id, "Backfill")
    assert await asyncio.wait_for(asked, 1) == {"stale": ["Backfill"]}


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
