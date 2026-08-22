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
async def test_ask_gate_projects_serializable_options_with_descriptions() -> None:
    """An owned ask WITH options must project a JSON-serializable card that
    carries each option's consequence line (U3).

    Regression origin: the gate once pushed raw AskOption pydantic models into
    PendingRequest, whose ``to_json`` is ``asdict`` and leaves those models as
    objects ``json.dumps`` cannot encode — crashing the projection push. The
    wire now carries {label, description} dicts (still JSON-serializable), and
    the phone renders both. The ``json.dumps`` below is what raised on the old
    object-valued shape.
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
    assert [o.label for o in pending.options] == ["Drop them", "Backfill"]
    assert [o.description for o in pending.options] == [
        "nothing reads the column",
        "slower, keeps history",
    ]
    assert pending.secret is False
    assert (pending.question_index, pending.question_total) == (0, 1)

    # The whole point: the projection round-trips over the wire. This is the
    # line that raised on the old object-valued shape.
    wire = json.dumps(handle._fold.projection.to_json())
    assert '"Drop them"' in wire and '"nothing reads the column"' in wire

    # Answer it back with a label, exactly as the phone does, so the parked
    # gate resolves under the question's id and the test leaves nothing hanging.
    await handle.ask_answer(pending.request_id, "Backfill")
    assert await asyncio.wait_for(asked, 1) == {"stale": ["Backfill"]}


@pytest.mark.asyncio
async def test_ask_gate_projects_secret_flag_without_the_value() -> None:
    """D1/U2: a secret ask projects secret=True and no options (paste field),
    and the pasted value never rides the projection or the wire."""
    handle, _ = make_handle(auto_approve=False)
    gate = handle._ask_gate

    question = AskQuestion(id="OPENAI_API_KEY", question="Paste your key", secret=True)
    asked = asyncio.ensure_future(gate([question]))
    await asyncio.sleep(0)

    pending = handle._fold.projection.pending
    assert pending is not None
    assert pending.secret is True
    assert pending.options == []

    await handle.ask_answer(pending.request_id, "sk-topsecret")
    # The value never appeared on the wire while the card was live or after.
    assert "sk-topsecret" not in json.dumps(handle._fold.projection.to_json())
    assert await asyncio.wait_for(asked, 1) == {"OPENAI_API_KEY": ["sk-topsecret"]}


@pytest.mark.asyncio
async def test_ask_gate_asks_multiple_questions_one_at_a_time() -> None:
    """U1: an owned multi-question ask projects and resolves question by
    question — answering Q1 advances to Q2 rather than settling the whole set,
    and the gate returns BOTH answers only after the last is answered."""
    handle, _ = make_handle(auto_approve=False)
    gate = handle._ask_gate

    q1 = AskQuestion(
        id="env",
        question="Which environment?",
        options=[AskOption(label="prod"), AskOption(label="staging")],
    )
    q2 = AskQuestion(
        id="confirm",
        question="Confirm?",
        options=[AskOption(label="yes"), AskOption(label="no")],
    )
    asked = asyncio.ensure_future(gate([q1, q2]))
    await asyncio.sleep(0)

    first = handle._fold.projection.pending
    assert first is not None
    assert first.title == "Which environment?"
    assert (first.question_index, first.question_total) == (0, 2)

    await handle.ask_answer(first.request_id, "prod")
    # The answer resolves via call_soon_threadsafe; let the gate resume, pop Q1,
    # and push Q2 across a few loop turns before inspecting the new front card.
    for _ in range(5):
        await asyncio.sleep(0)
    assert not asked.done(), "the gate resolved after only Q1 (U1 truncation)"

    second = handle._fold.projection.pending
    assert second is not None
    assert second.title == "Confirm?"
    assert (second.question_index, second.question_total) == (1, 2)
    # A distinct request id per question: Q1's stale request must no longer
    # resolve anything.
    assert second.request_id != first.request_id

    await handle.ask_answer(second.request_id, "yes")
    assert await asyncio.wait_for(asked, 1) == {"env": ["prod"], "confirm": ["yes"]}
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
