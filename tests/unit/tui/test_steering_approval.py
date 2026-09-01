"""Mid-turn steering, Esc-to-stop, and the in-TUI tool-approval prompt.

These three exist because of one shared defect class: the TUI owned the terminal
but not the interactions that needed it.

- Typing during a turn called ``prompt()``, which the session REJECTS while a
  turn holds its lock, so the text was thrown away behind an error notice.
- Esc was bound to nothing.
- Approvals fell through to the factory's stdin gate, which cannot be answered
  while Textual holds the terminal in raw mode — the turn parked on ``input()``
  forever, which is what the user saw as a frozen agent.

The assertions are about the observable contract (what the session is told, what
the frame says, whether the awaited future settles), never about how the app
routes messages internally.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import pytest
from rich.cells import cell_len
from textual.binding import Binding
from textual.document._document import Selection

from local_operator.harness.types import (
    ToolCallComposeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.resume import TRANSCRIPT_NAME
from local_operator.tui import app as app_module
from local_operator.tui.app import (
    DOUBLE_INTERRUPT_WINDOW_S,
    DOUBLE_STOP_WINDOW_S,
    OperatorApp,
)
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageEnd,
    AssistantMessageStart,
    ToolComposing,
    ToolEnded,
    ToolStarted,
    TurnStarted,
)
from local_operator.tui.widgets.approval import (
    APPROVAL_CHOICES,
    APPROVAL_KEY_BY_LABEL,
    CHOICES,
    HAZARD_GLYPH,
    OUTSIDE_MARKER,
    PROMPT_GLYPH,
    UNRESOLVABLE_MARKER,
    ApprovalBlock,
    ApprovalPrompt,
)
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    TranscriptView,
    UserBlock,
    WorkingBlock,
)

from .test_app_pilot import FakeSession, _factory
from .test_transcript_selection import _cell, _composer_copy


@pytest.fixture
def unraceable_answer_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the wall clock out of the composer's answer-key hold.

    ``ANSWER_KEY_HOLD_S`` parks a routed answer key for 180 ms so that the
    first character of a typed word cannot authorise anything: the hold is
    released early by the SECOND keystroke, and commits as an answer only if
    that keystroke never comes. In production that budget is enormous — 180 ms
    is most of a second of typing — but in a pilot test the gap between two
    ``pilot.press`` calls is not a human's inter-key interval. It is however
    long the harness takes to round-trip a key through the message pump, and
    that is contention-sensitive: measured here at **85-112 ms against the
    180 ms budget**, a margin of only ~90 ms on an idle 14-core box. A loaded
    4-vCPU CI runner erases it, the timer fires between two characters of
    ``and also drop the index``, and the leading ``a`` answers the prompt
    instead of being typed — the exact failure seen on CI as
    ``editor='d also drop the index'`` with the approval resolved.

    That is a test-harness race, not a product defect: no human types the
    second character of a word 180 ms after the first and means the first as
    an answer. So the tests that TYPE A WORD stretch the hold far beyond any
    pilot round trip, which makes "the second keystroke arrived first" true by
    construction rather than by timing luck.

    It does not weaken the STRUCTURAL guard: the hold still exists, the key
    is still parked, and the assertions are unchanged, so reintroducing the F3
    defect (route the key immediately, no hold at all) still turns these tests
    red. What the stretch does remove is their sensitivity to the hold's
    DURATION — with it applied, any value from 1 ms to 30 s passes, and a 1 ms
    hold is no protection at all for a real user. That is a real hole and it
    is why :func:`test_the_answer_key_hold_is_long_enough_to_be_a_hold` exists:
    the constant is pinned by its own test, against the human-typing facts it
    was derived from, so this fixture cannot hide a regression in it.

    Not every test here can take the stretch. The ones that assert the hold
    COMMITS — a deliberate single keypress answering the question — need the
    real timer to fire and deliberately do not use this fixture. The three that
    do use it either type a word (the second keystroke cancels the hold) or end
    it explicitly with Enter/Escape/an external settle, so in every case the
    hold is resolved by an EVENT rather than by the clock running out.

    It is deliberately NOT applied to
    ``test_a_prompt_arriving_mid_sentence_does_not_take_the_caret``, which
    types ``please `` before the question arrives: a non-empty composer stands
    the routing down entirely, so no hold is ever created and the fixture would
    be inert. Attaching it there would advertise a dependency the test does not
    have (review round 2).
    """
    monkeypatch.setattr(app_module, "ANSWER_KEY_HOLD_S", 30.0)


def test_the_answer_key_hold_is_long_enough_to_be_a_hold() -> None:
    """The hold's DURATION is a safety property, so pin it here.

    ``unraceable_answer_hold`` stretches this constant to take the wall clock
    out of three pilot tests, which necessarily makes those tests blind to its
    value: with the fixture applied, a 1 ms hold passes them just as happily
    as 180 ms. A 1 ms hold is not a hold — no keystroke pair arrives that
    close — so it silently reopens the F3 hazard where the first character of
    ``yes do it`` authorises a pending ``rm -rf``.

    The bounds are the ones the constant was derived from, and each end is a
    real failure rather than a preference:

    * Too SHORT and the hold cannot span a human's inter-key interval, so a
      typed word commits its first character as an answer. 60 wpm is ~200 ms
      per character and the burst inside a word is far shorter; 100 ms is
      below any of that.
    * Too LONG and a deliberate single keypress feels unanswered — the user
      presses ``y`` and watches the question sit there. 400 ms is the outer
      edge of "immediate".

    A unit test rather than a pilot one on purpose: it asserts a fact about
    the constant, needs no app, and cannot itself flake.

    It pins the VALUE; the assertion below pins the WIRING, because a correct
    constant nobody reads is worth nothing (review round 2, F2-1). Together
    they mean the hold cannot be shortened either by retuning the number or by
    bypassing it at the call site.
    """
    assert 0.100 <= app_module.ANSWER_KEY_HOLD_S <= 0.400, (
        f"ANSWER_KEY_HOLD_S is {app_module.ANSWER_KEY_HOLD_S}s: outside the range where it is "
        "both long enough to span typing and short enough to feel immediate"
    )
    # The timer must be armed FROM the constant, not from a literal that has
    # drifted away from it. Read the source of the one call site rather than
    # mocking ``set_timer``: the wiring is a static fact, so a static check
    # cannot race and needs no app.
    hold_source = inspect.getsource(app_module.OperatorApp._hold_answer_key)
    assert "set_timer(ANSWER_KEY_HOLD_S" in hold_source, (
        "_hold_answer_key no longer arms its timer from ANSWER_KEY_HOLD_S, so the bound "
        f"asserted above guards nothing:\n{hold_source}"
    )


async def _focus_composer(pilot: Any, app: OperatorApp) -> None:
    """Put the caret in the composer and wait until it is verifiably there.

    ``pilot.click(Editor)`` is a COORDINATE click: it reads the widget's region
    and posts a mouse event at its centre. Between the read and the event the
    dock can reflow — the prompt host appears the moment a question mounts and
    the boot card clamps away — so under load the event lands on the prompt
    instead of the composer, and the prompt's own ``on_click`` takes focus.
    The next keystroke then answers the question: measured as the leading
    ``n`` of ``no wait`` vanishing from the buffer while the approval resolved
    ``False``. The click is not the thing under test in any of these cases —
    "the user is typing in the composer" is — and the mid-sentence twin of
    these tests already establishes the direct idiom (``editor.focus()``).

    So focus the composer the way the app would and wait on the CONDITION a
    typing user actually has: the screen's focused widget is the Editor. The
    ceiling is a deadlock guard, not a timing assumption.
    """
    app.query_one(Editor).focus()
    for _ in range(100):
        if isinstance(app.screen.focused, Editor):
            return
        await pilot.pause(0.02)
    raise AssertionError("the composer never took focus")


class SteerableSession(FakeSession):
    """A fake whose streaming state the test drives, recording steer calls."""

    def __init__(self) -> None:
        super().__init__()
        self.steers: list[str] = []
        self.streaming = False
        self.approval_handler: Any | None = None

    @property
    def is_streaming(self) -> bool:
        return self.streaming

    def steer_message(self, message: Any) -> None:
        # Record the text the old `steer` override did and let the base fake
        # hold the object, so the recall seam sees a real queue.
        self.steers.append(message.text)
        super().steer_message(message)

    def set_approval_handler(self, handler: object | None) -> None:
        self.approval_handler = handler


def _ask_question(qid: str, text: str, *, multi: bool = False):  # type: ignore[no-untyped-def]
    """A minimal `ask` question, for tests that need one beside an approval."""
    from local_operator.harness.types import AskOption, AskQuestion

    return AskQuestion(
        id=qid,
        question=text,
        multi=multi,
        options=[
            AskOption(label="Drop", description=""),
            AskOption(label="Backfill", description=""),
        ],
    )


def _approval_gate(session: SteerableSession) -> Callable[[str, str], Awaitable[bool]]:
    """The handler the app installed, narrowed to non-optional.

    Asserting here rather than in each test keeps "the app installed a gate" a
    single failure with one message, instead of an AttributeError per call site.
    """
    handler = session.approval_handler
    assert handler is not None, "the app never installed its approval handler"
    return cast("Callable[[str, str], Awaitable[bool]]", handler)


async def _booted_gate(
    pilot: Any, session: SteerableSession
) -> Callable[[str, str], Awaitable[bool]]:
    """Wait for the app to install its approval gate, then return it.

    Waits on the CONDITION rather than on a duration. The fixed
    ``pause(0.25)`` these tests used is a bet on how long boot takes, and the
    bet is lost for the FIRST app built in a process: session creation runs in
    a worker, and the first one through pays the import and construction cost
    the later ones do not. Measured on an unmodified tree, the first app in a
    process installed its handler at ~0.39s and every later one at ~0.30s — so
    each of these tests passed when the file ran whole and failed when run
    alone, on `main` as much as here.

    Relationship to :func:`_boot`, which waits for the app to ADOPT a session:
    this helper SUBSUMES that wait and does not need to be paired with it.
    `_adopt_session` is synchronous and assigns `self._session` before it calls
    `session.set_approval_handler(...)`, so a handler that exists proves an
    adoption that already happened. The 13 tests here that call this alone are
    correct as written; do not add a `_boot` in front of them.

    `_boot` is what a test needs when it SUBMITS without ever touching the gate
    — there the adoption is the only fact in question, and waiting on a handler
    the test never uses would be waiting on the wrong thing.
    """
    for _ in range(100):
        if session.approval_handler is not None:
            return _approval_gate(session)
        await pilot.pause(0.02)
    raise AssertionError("the app never installed its approval handler")


def rows(app: OperatorApp) -> list[str]:
    """The painted frame as plain text, one entry per row."""
    return [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]


async def _settled_rows(pilot: Any, app: OperatorApp, ceiling: int = 200) -> list[str]:
    """The painted frame, read only once it has stopped reflowing.

    Tests that assert on a row's REFLOWED GEOMETRY — which choices were shed at
    a given width, whether anything bleeds past the terminal, whether a receipt
    stays on one row — must not read a mid-reflow frame. A single ``pilot.pause()``
    yields exactly one idle tick, and an approval row that is still collapsing to
    fit a narrow width paints a truncated intermediate frame (a dangling ``…`` or
    a half-shed clause) on that tick. Reading it makes a width-parametrized test
    fail intermittently under load with a truncated row the settled frame never
    shows — the failure mode that a fixed ``pause(0.2)`` used to paper over by
    outlasting the reflow.

    Waits for the CONDITION the geometry assertions depend on instead: the frame
    is stable when two consecutive idle ticks paint the identical row set. The
    ceiling is a deadlock guard, not a timing assumption, so a slow or contended
    machine costs nothing and a genuinely stuck frame still fails rather than
    hanging.
    """
    previous: list[str] = []
    for _ in range(ceiling):
        await pilot.pause()
        current = rows(app)
        if current == previous:
            return current
        previous = current
    return previous


async def _wait_for_row(pilot: Any, app: OperatorApp, needle: str, ceiling: int = 200) -> None:
    """Wait until ``needle`` appears in the painted frame.

    A settled approval future does NOT mean its receipt is on screen. The card
    resolves its own future first and the transcript row is written by a later
    frame, so a test that presses a key, awaits the future and then reads
    ``rows(app)`` is asserting one frame too early. It usually wins that race
    and loses it under load: the census caught exactly this as
    ``assert any("denied" in row ...)`` failing on a contended xdist worker,
    where the sibling ``y`` test survived only because its focus poll happened
    to spend the extra ticks.

    The ceiling is a deadlock guard rather than a timing assumption, so a slow
    machine costs nothing while a receipt that never paints still fails.
    """
    for _ in range(ceiling):
        if any(needle in row for row in rows(app)):
            return
        await pilot.pause()
    raise AssertionError(f"{needle!r} never appeared in the painted frame")


async def _boot(pilot: Any, app: OperatorApp) -> None:
    """Wait for the session to be ADOPTED, rather than for a fixed duration.

    Every test here begins by submitting into a booted app, and the boot is
    asynchronous: ``OperatorApp`` awaits a session factory and adopts the result.
    Waiting a flat ``pause(0.25)`` for that assumed a duration nothing enforces
    — measured on this machine, adoption takes ~0.03s on a warm run and **2.3s**
    on a cold one (first import of the session graph). On the cold run the first
    submit landed before ``self._session`` existed, was dropped on the floor, and
    the test failed on an assertion two lines later with no hint that the boot
    was the cause: ``assert [] == ['first task']``.

    That made the whole file load-dependent. It failed deterministically on a
    cold cache and passed on a warm one, which is exactly the shape that reads as
    "flaky" and gets re-run instead of fixed.

    Polls the same condition the app's own readiness depends on, with a generous
    ceiling: the ceiling is a deadlock guard, not a timing assumption, so a slow
    machine costs nothing and a genuinely broken boot still fails rather than
    hanging.
    """
    for _ in range(400):
        if app._session is not None:
            return
        await pilot.pause()
    raise AssertionError("the app never adopted a session")


async def _submit(pilot: Any, app: OperatorApp, text: str) -> None:
    app.query_one(Editor).text = text
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
async def test_mid_turn_submit_steers_instead_of_prompting() -> None:
    """A turn is already running: the text rides the steering queue.

    The session's own contract is that ``prompt()`` raises while streaming, so
    this is not a preference — re-prompting is the bug, and the receipt tells the
    user their words were kept rather than dropped.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "first task")
        assert session.prompts == ["first task"]

        session.streaming = True
        await _submit(pilot, app, "actually use pygame-ce")

        assert session.steers == ["actually use pygame-ce"]
        assert session.prompts == ["first task"]  # no second prompt attempted
        painted = rows(app)
        assert any("actually use pygame-ce" in row for row in painted)
        assert any("queued" in row for row in painted)


@pytest.mark.asyncio
async def test_escape_stops_a_running_turn() -> None:
    """Esc aborts while streaming, and does nothing at all when idle.

    "Does nothing when idle" is the load-bearing half: Esc must never clear the
    composer, because that would discard typed text on the key people press to
    cancel.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)

        editor = app.query_one(Editor)
        editor.text = "kept"
        await pilot.press("escape")
        await pilot.pause()
        assert session.aborts == []
        assert editor.text == "kept"
        # Focus MUST stay in the composer. TextArea binds Escape to `blur`, which
        # made the first press move focus out of the input while looking like it
        # did nothing — every keystroke after it went nowhere.
        assert app.screen.focused is editor

        session.streaming = True
        await pilot.press("escape")
        await pilot.pause()
        assert session.aborts == ["interrupted"]
        assert app.screen.focused is editor


@pytest.mark.asyncio
async def test_empty_assistant_message_mounts_no_block() -> None:
    """A tool-use turn carries no prose, so it must not spend rows on a block.

    Every Anthropic tool turn opens a message and goes straight to the calls;
    mounting the block eagerly cost two rows (the empty block plus the blank row
    the spacing rule opens above a new kind), which read as a hole between the
    lead-in and the tool ledger.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "make a game")
        app.post_message(TurnStarted())
        app.post_message(AssistantMessageStart())
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(
                    tool_call_id="c0", tool_name="bash", args={"command": "ls -la"}
                )
            )
        )
        await pilot.pause()

        assert not app.query(AssistantBlock)
        painted = rows(app)
        # The working line TRAILS the row it reports on — it is pinned to the
        # foot of the transcript — with exactly ONE blank row between them: the
        # air that says "this is the live status, not another ledger entry".
        # An eagerly mounted empty prose block would open a second hole.
        # Found by its spinner head rather than by its words, because the
        # ledger row above it also carries the tool's name.
        working = next(
            index
            for index, row in enumerate(painted)
            if row.strip()[:1] in set(WorkingBlock._SPINNER)
        )
        tool = next(
            index for index, row in enumerate(painted) if "bash" in row and index != working
        )
        assert working - tool == 2


@pytest.mark.asyncio
async def test_approval_prompt_resolves_from_a_keystroke() -> None:
    """The awaited future settles from the UI, which is the anti-freeze contract."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        ask = await _booted_gate(pilot, session)
        pending = asyncio.ensure_future(ask("bash", "run: rm -rf /tmp/x"))
        # Waited on the CONDITION, not a duration. The card takes focus on
        # mount (the composer is empty here), but a fixed pause is a bet on how
        # many frames that takes and CI lost it on the slower runner.
        for _ in range(100):
            if isinstance(app.screen.focused, ApprovalPrompt):
                break
            await pilot.pause(0.02)
        prompt = app.query_one(ApprovalPrompt)
        assert app.screen.focused is prompt  # else the keys go to the composer
        # The three answers are offered as a LIST, not only as letters to aim
        # at a focused widget, and the session-wide one names its scope.
        assert any("Allow all" in row for row in rows(app))
        # And the question is anchored in the dock, above the composer, rather
        # than in the transcript where a busy turn scrolls it out of reach.
        assert prompt.region.y > app.query_one(TranscriptView).region.y

        await pilot.press("y")
        assert await asyncio.wait_for(pending, 2) is True
        # The future resolves BEFORE the widget comes down: the card settles
        # its own future and the awaiting frame then unmounts it and restores
        # focus, which is a mount round trip rather than a synchronous step. A
        # fixed pause is a bet on that taking one frame.
        for _ in range(100):
            if isinstance(app.screen.focused, Editor):
                break
            await pilot.pause(0.02)
        assert isinstance(app.screen.focused, Editor)  # focus handed back
        assert not app.query(ApprovalPrompt)  # the question is gone
        assert any("allowed" in row for row in rows(app))  # decision kept


@pytest.mark.asyncio
async def test_n_denies_one_tool_and_lets_the_turn_continue() -> None:
    """``n`` is the per-call refusal: this tool is refused, the turn carries on."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        ask = await _booted_gate(pilot, session)
        session.streaming = True
        pending = asyncio.ensure_future(ask("write", "write: /etc/hosts"))
        # Same condition wait the ``y`` sibling makes: mounting the card and
        # moving focus to it is a mount round trip, so a single pause is a bet
        # on that costing one frame. Pressing early types into the composer.
        for _ in range(100):
            if isinstance(app.screen.focused, ApprovalPrompt):
                break
            await pilot.pause(0.02)

        await pilot.press("n")
        assert await asyncio.wait_for(pending, 2) is False
        assert session.aborts == []  # only the tool was refused
        await _wait_for_row(pilot, app, "denied")


@pytest.mark.asyncio
async def test_escape_denies_the_prompt_and_stops_the_turn() -> None:
    """Esc means stop, uniformly — including while a question is on screen.

    It deliberately does NOT mean "answer this one question": Esc denying only
    the front prompt cost the user one press per concurrent approval before the
    run actually stopped, and each press looked like it had done nothing.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        ask = await _booted_gate(pilot, session)
        session.streaming = True
        first = asyncio.ensure_future(ask("write", "write: /etc/hosts"))
        second = asyncio.ensure_future(ask("bash", "run: rm -rf /"))
        await pilot.pause()

        await pilot.press("escape")
        # BOTH asks are settled by the one press, and the turn is stopped.
        assert await asyncio.wait_for(first, 2) is False
        assert await asyncio.wait_for(second, 2) is False
        assert session.aborts == ["interrupted"]
        # No question is left on screen for a turn that has been stopped.
        assert not [prompt for prompt in app.query(ApprovalPrompt) if not prompt.answered]


@pytest.mark.asyncio
async def test_allow_all_latches_for_the_session() -> None:
    """``A`` answers every later ask without a second prompt."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        ask = await _booted_gate(pilot, session)
        first = asyncio.ensure_future(ask("bash", "run: make"))
        await pilot.pause()
        await pilot.press("A")
        assert await asyncio.wait_for(first, 2) is True
        await pilot.pause()

        # No prompt is mounted for the second ask at all.
        before = len(app.query(ApprovalPrompt))
        second = asyncio.ensure_future(ask("write", "write: out.txt"))
        assert await asyncio.wait_for(second, 2) is True
        await pilot.pause()
        assert len(app.query(ApprovalPrompt)) == before


@pytest.mark.asyncio
async def test_interrupt_denies_a_parked_approval() -> None:
    """Ctrl+C must not leave the engine awaiting a future nobody will answer.

    A turn parked in the approval callback cannot observe the abort signal until
    the callback returns, so the prompt has to be settled first or the abort is
    silently ineffective — the same hang, reached a different way.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        ask = await _booted_gate(pilot, session)
        session.streaming = True
        pending = asyncio.ensure_future(ask("bash", "run: sleep 99"))
        await pilot.pause()

        app.action_interrupt()
        assert await asyncio.wait_for(pending, 2) is False
        assert session.aborts == ["interrupted"]


@pytest.mark.asyncio
async def test_clearing_the_transcript_settles_a_pending_approval() -> None:
    """The prompt's widget is about to be removed; the awaiting turn is denied."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        ask = await _booted_gate(pilot, session)
        pending = asyncio.ensure_future(ask("bash", "run: make"))
        await pilot.pause()

        app.query_one(TranscriptView).clear_blocks()
        await pilot.pause()
        assert await asyncio.wait_for(pending, 2) is False


@pytest.mark.asyncio
async def test_escape_closes_an_open_picker_before_it_stops_anything() -> None:
    """Esc precedence: the editor's open list wins, the turn-stop is the fallback.

    Binding Esc with ``priority=True`` broke this — Textual matches priority
    bindings BEFORE dispatching the key to the focused widget, so the picker's
    own Esc handler (and its ``event.stop()``) never ran and the command/model
    lists could not be dismissed at all. The binding must bubble instead, which
    is what these two halves pin.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True  # a stop IS available, so precedence is observable

        editor = app.query_one(Editor)
        editor.text = "/"
        await pilot.pause()
        assert editor.picker.is_open()

        await pilot.press("escape")
        await pilot.pause()
        assert not editor.picker.is_open()  # the list closed…
        assert session.aborts == []  # …and the turn was NOT stopped

        # The very next Esc stops the turn — no dead press in between, and focus
        # never leaves the composer.
        await pilot.press("escape")
        await pilot.pause()
        assert session.aborts == ["interrupted"]
        assert app.screen.focused is editor


@pytest.mark.asyncio
async def test_a_queued_ask_is_denied_rather_than_re_asked_after_a_stop() -> None:
    """A stop settles the asks BEHIND the front prompt too.

    Settling only the visible prompt woke the queued asker, which then mounted a
    brand-new question for a turn that had already been aborted — and on the
    teardown path, into a screen that was going away. Write/exec tier tools are
    not interruptible, so the runner parked on the callback is settled by nothing
    but this future: the re-asked question was genuinely live.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        ask = await _booted_gate(pilot, session)
        session.streaming = True
        first = asyncio.ensure_future(ask("bash", "run: one"))
        second = asyncio.ensure_future(ask("bash", "run: two"))
        await pilot.pause()
        assert len(app.query(ApprovalPrompt)) == 1  # serialized, not stacked

        app.action_interrupt()
        assert await asyncio.wait_for(first, 2) is False
        assert await asyncio.wait_for(second, 2) is False
        await pilot.pause()
        # No fresh question was mounted for the stopped turn.
        assert not app.query(ApprovalPrompt)


@pytest.mark.asyncio
async def test_allow_all_is_seen_by_an_already_queued_ask() -> None:
    """The policy must be readable the instant the future resolves.

    Latching through a posted message put the flag several pump hops behind the
    future's awaiters, so the queued asker read a stale policy and re-asked —
    immediately after the user pressed "allow all".
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        ask = await _booted_gate(pilot, session)
        first = asyncio.ensure_future(ask("bash", "run: one"))
        second = asyncio.ensure_future(ask("write", "write: two"))
        await pilot.pause()

        await pilot.press("A")
        assert await asyncio.wait_for(first, 2) is True
        assert await asyncio.wait_for(second, 2) is True  # never asked again
        await pilot.pause()
        assert not app.query(ApprovalPrompt)


@pytest.mark.asyncio
async def test_lowercase_a_does_not_disarm_the_gate() -> None:
    """ "Allow all" needs Shift: the card holds focus, and ``a`` is a letter.

    A user who did not notice focus moved and kept typing their next instruction
    would otherwise hit the most common letter in English and disable the
    approval gate for the whole session, with no receipt.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        ask = await _booted_gate(pilot, session)
        pending = asyncio.ensure_future(ask("bash", "run: one"))
        for _ in range(100):
            if isinstance(app.screen.focused, ApprovalPrompt):
                break
            await pilot.pause(0.02)

        await pilot.press("a")
        await pilot.pause()
        assert not pending.done()  # the question is still being asked
        assert app._approve_all is False
        # The card KEEPS focus and keeps the question answerable.
        #
        # It used to hand focus to the composer and re-post the keystroke there,
        # which is the defect this rework removes: from that point the letters
        # the card still advertised went into the prompt buffer as text, so a
        # user who pressed `y` watched a `y` appear in the composer while the
        # tool went on waiting. The stray letter is dropped instead — this card
        # is a question, not a text field, and the composer is one keystroke
        # away by click or Tab.
        prompt = app.query_one(ApprovalPrompt)
        assert app.screen.focused is prompt
        assert app.query_one(Editor).text == ""

        # And the answer keys still work, with no click needed to restore focus.
        await pilot.press("y")
        assert await asyncio.wait_for(pending, 2) is True


@pytest.mark.asyncio
async def test_approvals_command_restores_prompting() -> None:
    """``/approvals ask`` is the way back from a latched "allow all"."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command("/approvals auto")
        await pilot.pause()
        assert app._approve_all is True
        # The band's alarm is a bare `!` in the trailing cell now — the session
        # name took the words that used to be there.
        assert any(row.rstrip().endswith("!") for row in rows(app))

        app._run_slash_command("/approvals ask")
        await pilot.pause()
        assert app._approve_all is False
        assert not any(row.rstrip().endswith("!") for row in rows(app))
        # And the gate really asks again.
        pending = asyncio.ensure_future(_approval_gate(session)("bash", "run: x"))
        await pilot.pause()
        assert app.query(ApprovalPrompt)
        await pilot.press("n")
        assert await asyncio.wait_for(pending, 2) is False


@pytest.mark.asyncio
async def test_ctrl_c_twice_exits_and_offers_the_resume_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One Ctrl+C stops the work and keeps the session; two leave.

    The resume line is what makes the second press safe to offer: the session is
    on disk (the transcript is appended per turn), so quitting is recoverable.
    The transcript is planted here because the hint is gated on it existing.
    """
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    transcript = tmp_path / "sessions" / "sess" / TRANSCRIPT_NAME
    transcript.parent.mkdir(parents=True)
    transcript.write_text("{}\n", encoding="utf-8")

    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert session.aborts == ["interrupted"]
        assert app.is_running  # one press NEVER exits
        assert any("ctrl+c again to exit" in row for row in rows(app))
        # The transcript line promises recoverability; it deliberately does NOT
        # carry the command, which belongs in the exit block printed to the
        # terminal (the alt screen is discarded, so a copyable command in a frame
        # is the unreachable copy).
        assert any("the session can be resumed" in row for row in rows(app))
        assert not any("--resume" in row for row in rows(app))

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.is_running

    assert app.resume_hint() == "lop --resume sess"


@pytest.mark.asyncio
async def test_a_slow_second_ctrl_c_is_a_fresh_interrupt_not_an_exit() -> None:
    """Two interrupts a minute apart must not quit the app.

    The window is what keeps the gesture a deliberate double-tap; without it, a
    user interrupting two turns in a row would exit by accident.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True

        await pilot.press("ctrl+c")
        await pilot.pause()
        # Age the first press past the window without waiting for it.
        app._last_interrupt_at -= DOUBLE_INTERRUPT_WINDOW_S + 1
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running
        assert session.aborts == ["interrupted", "interrupted"]
        # One hint, not one per press: it is replaced rather than repeated.
        assert len([row for row in rows(app) if "ctrl+c again to exit" in row]) == 1


@pytest.mark.asyncio
async def test_an_empty_message_end_keeps_the_streamed_prose() -> None:
    """An empty authoritative text must not erase what the deltas painted."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "say something")
        app.post_message(TurnStarted())
        app.post_message(AssistantMessageStart())
        app.post_message(AssistantDelta("hello from the model"))
        # Wait on the PAINTED condition, not on one idle tick: a delta posted
        # to the message pump lands on a later frame, and a single pause is a
        # bet on that being the very next tick. Under load the frame read here
        # precedes the paint and the row is absent (caught by the 12-hog
        # harness on an otherwise-green tree). _wait_for_row's ceiling is a
        # deadlock guard, so a contended runner costs nothing.
        await _wait_for_row(pilot, app, "hello from the model")

        app.post_message(AssistantMessageEnd(""))
        # The contract is that the empty authoritative text erases nothing —
        # so the assertion must read the frame AFTER the end handler ran, and
        # a single pause is a bet on that being the next tick. The handler's
        # observable fact is that it clears ``_streaming_block``; the frame
        # itself is identical before and after (that is the point of the
        # fix), so settling the frame would not prove it ran. Wait on the
        # condition, then read the frame the user would see.
        for _ in range(200):
            if app._streaming_block is None:
                break
            await pilot.pause()
        assert app._streaming_block is None, "the empty end was never processed"
        await pilot.pause()
        assert any("hello from the model" in row for row in rows(app))
        assert len(app.query(AssistantBlock)) == 1


def test_the_answer_keys_match_the_bindings_and_the_hint_row() -> None:
    """Three spellings of the same fact must agree.

    ``_ANSWER_KEYS`` is written out by hand (deriving it from ``BINDINGS`` does
    not type-check, since that attribute also accepts Binding objects), so this
    is what stops it drifting from the keys that actually answer — a key missing
    from the set would be typed into the composer instead of answering, and a
    stale entry would swallow a keystroke that should have reached the composer.
    The hint row is checked too: a choice nobody is told about is not on offer.
    """
    # Same narrowing the widget cannot do at class scope: BINDINGS is typed as
    # accepting Binding objects too, and only this test needs to read the keys.
    bound = {
        binding.key if isinstance(binding, Binding) else binding[0]
        for binding in ApprovalBlock.BINDINGS
    }
    assert ApprovalBlock._ANSWER_KEYS == bound

    advertised = {key for key, _ in CHOICES}
    # Esc is advertised but is NOT this widget's key — the app owns it as "stop".
    assert advertised - {"esc"} == bound
    assert "a" not in bound, "allow-all must need Shift; see CHOICES"

    # The LIVE prompt is a different widget now, and it answers to the same
    # letters. A drift here is invisible until someone presses a key that the
    # card advertises and nothing handles, so the two are pinned to each other
    # rather than each to a literal.
    assert ApprovalPrompt._ANSWER_KEYS == bound
    # Every letter that answers is also a row the user can pick without knowing
    # it, which is the point of the list: a key-only answer is undiscoverable.
    assert {key for _label, key, _why in APPROVAL_CHOICES} == bound
    # And every row maps back to exactly one key, so a click and a keystroke
    # cannot disagree about what was authorised.
    assert set(APPROVAL_KEY_BY_LABEL.values()) == bound
    assert len(APPROVAL_KEY_BY_LABEL) == len(APPROVAL_CHOICES)


@pytest.mark.asyncio
async def test_clearing_the_transcript_does_not_disarm_later_prompts() -> None:
    """``/clear`` settles the visible question; it does not stop the turn.

    Latching the turn-wide deny flag here denied every later write/exec tool of
    the SAME run with no prompt at all — silently, and while ``/approvals``
    reported that tools would prompt. ``/clear`` clears a screen; it is not a stop.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        ask = await _booted_gate(pilot, session)
        session.streaming = True

        first = asyncio.ensure_future(ask("bash", "run: one"))
        await pilot.pause()
        app.query_one(TranscriptView).clear_blocks()
        await pilot.pause()
        assert await asyncio.wait_for(first, 2) is False  # its widget is gone
        assert session.aborts == []  # the turn was NOT stopped

        # The next tool of the same run still gets to ask.
        second = asyncio.ensure_future(ask("write", "write: two"))
        await pilot.pause()
        assert app.query(ApprovalPrompt), "a later tool of a live turn must still ask"
        await pilot.press("y")
        assert await asyncio.wait_for(second, 2) is True


@pytest.mark.asyncio
async def test_approvals_ask_clears_a_latched_deny() -> None:
    """``/approvals ask`` promises prompting; it must deliver it.

    The command previously only touched the allow-all flag, so after a stop had
    latched the deny flag it printed "tools will prompt again" while the next ask
    was refused without a prompt.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        app._deny_queued_approvals()  # what a stop leaves behind
        assert app._approvals_are_denied(app._turn_epoch) is True

        app._run_slash_command("/approvals ask")
        await pilot.pause()
        assert app._approvals_are_denied(app._turn_epoch) is False
        pending = asyncio.ensure_future(_approval_gate(session)("bash", "run: x"))
        await pilot.pause()
        assert app.query(ApprovalPrompt)
        await pilot.press("y")
        assert await asyncio.wait_for(pending, 2) is True


@pytest.mark.asyncio
async def test_the_resume_hint_is_withheld_until_the_session_is_on_disk() -> None:
    """A hint must not advertise a command ``--resume`` will refuse.

    ``--resume`` requires the transcript to exist (a typo must fail rather than
    open an empty session that looks resumed), so quitting before the first turn
    persisted would otherwise print a command guaranteed to be rejected.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        # FakeSession's id has no directory on disk, so there is nothing to offer.
        assert app.resume_hint() == ""
        await pilot.press("ctrl+c")
        await pilot.pause()
        painted = rows(app)
        assert any("ctrl+c again to exit" in row for row in painted)
        assert not any("--resume" in row for row in painted)


@pytest.mark.parametrize("width", [110, 80, 60, 52, 46, 40, 30, 24, 20, 16])
@pytest.mark.asyncio
async def test_the_prompt_never_hides_its_target_or_overflows(width: int) -> None:
    """A safety prompt must not say LESS the more dangerous the ask is.

    The hazard clause used to be appended before the detail budget was computed,
    so at 52 columns an outside-workspace prompt dropped the target path entirely
    and ended on a dangling em dash — while the same prompt without a hazard
    truncated gracefully. 52 columns is a split pane, not an edge case.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(width, 24)) as pilot:
        await _boot(pilot, app)
        view = app.query_one(TranscriptView)
        view.append_block(
            ApprovalBlock("write_file", "[outside workspace] write: /Users/x/deep/config.yml")
        )
        painted = await _settled_rows(pilot, app)

        # Nothing bleeds past the terminal at any width.
        assert not [row for row in painted if cell_len(row) > width]
        question = next((row for row in painted if "write_file" in row), "")
        if question:
            # The hazard degrades to a marker rather than eating the target, so a
            # row that still has room for a path shows one.
            assert not question.rstrip().endswith("—"), "hazard left a dangling clause"


@pytest.mark.parametrize("width", [80, 46, 40, 34, 30, 24])
@pytest.mark.asyncio
async def test_the_answered_receipt_stays_on_one_row(width: int) -> None:
    """The receipt must never wrap onto the composer's gutter.

    Rich honours ``Text.no_wrap`` for a Group's children but not for a bare Text
    handed to a Static, so returning the answered row unwrapped painted 2-4 rows
    with the continuation at column 2 — the ❯ column — while the pending two-row
    form was fine.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(width, 24)) as pilot:
        await _boot(pilot, app)
        block = ApprovalBlock("write_file", "[outside workspace] write: /Users/x/deep/config.yml")
        app.query_one(TranscriptView).append_block(block)
        await pilot.pause()
        block.resolve(True, answer="y")
        painted = await _settled_rows(pilot, app)

        # Found by the outcome GLYPH, not by the word: below ~32 columns the
        # receipt sheds `allowed` to keep the hazard clause, which is the same
        # trade the live ask makes with `allow` and is the behaviour under test
        # two rows up. What must never change is that it stays on ONE row.
        assert len([row for row in painted if row.strip().startswith("✓")]) == 1
        assert not [row for row in painted if cell_len(row) > width]
        assert block.spans_multiple_rows() is False


@pytest.mark.parametrize("width", [110, 60, 46, 30, 24, 16])
@pytest.mark.asyncio
async def test_the_hint_row_sheds_whole_choices(width: int) -> None:
    """Never half a choice: a key whose consequence was cut off is not an offer.

    Truncating left rows ending `A allow all …` and then `A …`. Shedding also
    encodes priority — the session-wide switch goes first, then the global stop
    key, so the two per-call ANSWERS survive furthest.

    Below the width where even one labelled choice fits, the row becomes the
    ``n/y`` legend: both answers, no labels. That rung is deliberately ABOVE the
    single labelled choice, because `n deny` spends ten cells advertising how to
    refuse and nothing else, while `n/y` spends five naming both ways out.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(width, 24)) as pilot:
        await _boot(pilot, app)
        app.query_one(TranscriptView).append_block(ApprovalBlock("bash", "run: make"))
        # Read the SETTLED frame: this asserts on the hint row's reflowed geometry
        # (which choices were shed at this width), and a mid-reflow frame paints a
        # truncated `…` the next assertion forbids. See _settled_rows.
        painted = await _settled_rows(pilot, app)
        hint = next((row.strip() for row in painted if "deny" in row or "n/y" in row), "")
        assert hint, "the prompt must always advertise at least one answer"
        assert "…" not in hint

        if hint == "n/y":
            # The keys alone, in the same order, both answers present.
            return
        # Every advertised item is whole: any key shown is followed by its label.
        for key, label in CHOICES:
            if f"{key} " in hint:
                assert f"{key} {label}" in hint, f"{key} advertised without its label"
        # The refusal is always the first thing offered, never the yes-word.
        assert hint.startswith("n deny")


@pytest.mark.parametrize("width", [110, 80, 60, 52, 46, 40, 36, 32, 30, 28, 26, 24, 20, 16])
@pytest.mark.asyncio
async def test_a_dangerous_ask_never_paints_like_a_safe_one(width: int) -> None:
    """The hazard is the LAST thing shed, at every width the app can be run at.

    The failure this pins is silent: below ~32 columns the outside-workspace
    prompt rendered BYTE-IDENTICAL to the same call inside the workspace, because
    the warning was being shed to buy one or two cells of path tail (`…s`). A
    prompt that says less is fine; one that says the same thing about a different
    risk is not, and no user can catch it because there is nothing on screen to
    notice. The ladder concedes in this order: full words -> `!` marker -> the
    word "allow" (said twice already, by the glyph and the hint row) -> characters
    off the tool name. The marker itself never goes.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(width, 24)) as pilot:
        await _boot(pilot, app)
        view = app.query_one(TranscriptView)
        view.append_block(ApprovalBlock("write_file", f"{OUTSIDE_MARKER} write: /tmp/x/keys"))
        view.append_block(ApprovalBlock("write_file", "write: /tmp/x/keys"))
        painted = await _settled_rows(pilot, app)
        # Matched on the leading glyph, not the tool name: at the narrowest widths
        # the hazard IS the glyph (`!` for `?`), which is the behaviour under test
        # rather than a reason to miss the row.
        asks = [row.strip() for row in painted if row.strip()[:1] in (PROMPT_GLYPH, HAZARD_GLYPH)]
        assert len(asks) == 2, asks
        hazardous, safe = asks
        assert hazardous != safe, f"identical at {width}: {hazardous!r}"
        # The risk reached the user in one of the three forms the ladder allows.
        assert hazardous.startswith(HAZARD_GLYPH) or "!" in hazardous or "outside" in hazardous
        assert not [row for row in painted if cell_len(row) > width]


@pytest.mark.parametrize(
    ("detail", "expected_tail"),
    [
        ("write: /Users/nobody/.ssh/authorized_keys", "authorized_keys"),
        ("write: /Users/nobody/Documents/notes.md", "notes.md"),
    ],
)
@pytest.mark.asyncio
async def test_a_narrow_prompt_keeps_the_end_of_the_path(detail: str, expected_tail: str) -> None:
    """Truncating a path from the RIGHT answers a question nobody asked.

    `/Users/<name>/` is boilerplate every path on the machine shares; the
    basename is the whole difference between the ask a user must refuse and the
    one they can wave through. Right-truncation at 52 columns rendered
    `~/.ssh/authorized_keys` and `~/Documents/notes.md` identically.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(52, 24)) as pilot:
        await _boot(pilot, app)
        app.query_one(TranscriptView).append_block(ApprovalBlock("write_file", detail))
        painted = await _settled_rows(pilot, app)
        ask = next(row for row in painted if "write_file" in row)
        assert expected_tail in ask, ask


@pytest.mark.asyncio
async def test_a_turn_start_racing_the_stop_cannot_revive_a_denied_ask() -> None:
    """The deny latch belongs to the turn that armed it, not to a wall clock.

    The latch used to be a bare flag cleared by the next `TurnStarted`. A
    `TurnStarted` already sitting in the message pump when the stop landed was
    therefore dispatched BEFORE the parked asker woke, so the asker re-read a
    cleared latch and mounted a fresh question for the turn the user had just
    stopped — and answering it ran the tool. Epochs make the ordering
    irrelevant: the asker compares against the turn it entered in.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        first = asyncio.ensure_future(_approval_gate(session)("bash", "run: one"))
        await pilot.pause()
        second = asyncio.ensure_future(_approval_gate(session)("execute", "run: rm -rf /"))
        await pilot.pause()

        # The race: a turn boundary is already queued when the stop lands.
        app.post_message(TurnStarted())
        app.action_interrupt()
        await pilot.pause()

        assert await asyncio.wait_for(first, 2) is False
        assert await asyncio.wait_for(second, 2) is False
        painted = [strip.text for strip in app.screen._compositor.render_strips()]
        assert not [row for row in painted if "rm -rf /" in row and "?" in row]


@pytest.mark.asyncio
async def test_approvals_auto_answers_the_question_on_screen() -> None:
    """ "Every tool runs without asking" must include the one already asking.

    Setting the mode without settling the live prompt left the loudest notice in
    the app next to a question still waiting, with the tool behind it parked on
    a future nothing would settle.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _booted_gate(pilot, session)
        session.streaming = True
        pending = asyncio.ensure_future(_approval_gate(session)("bash", "run: make"))
        await pilot.pause()
        assert app.query(ApprovalPrompt)

        app._run_slash_command("/approvals auto")
        await pilot.pause()
        assert await asyncio.wait_for(pending, 2) is True
        painted = [strip.text for strip in app.screen._compositor.render_strips()]
        assert [row for row in painted if "allowed" in row]


@pytest.mark.asyncio
async def test_the_prompt_cannot_be_repainted_from_a_tool_argument() -> None:
    """Model-controlled text reaches this row; it must reach it INERT.

    The JSON dump this prompt used to show escaped control characters as a side
    effect of being JSON. Moving to a real sentence (`run: <command>`) removed
    that accident, and with it the only thing stopping a command argument from
    carrying live CSI onto the terminal: erase-line plus cursor-up inside an
    approval prompt can wipe the row above and paint a forged receipt over it,
    which is the one row in the app where a forgery is worth the most.

    Two further effects the width work depends on: `cell_len` counts escape
    bytes, so an un-stripped string mis-measures the ladder, and truncation can
    cut a sequence in half and leave a bare introducer on the terminal.
    """
    payload = "echo hi\x1b[2K\x1b[1A\x1b[32m✓ allowed bash  run: echo hi\x1b[0m"
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(90, 24)) as pilot:
        await _boot(pilot, app)
        block = ApprovalBlock("bash", f"run: {payload}")
        app.query_one(TranscriptView).append_block(block)
        await pilot.pause()

        assert "\x1b" not in block.description
        assert "\x1b" not in block.tool_name
        painted = [strip.text for strip in app.screen._compositor.render_strips()]
        assert not [row for row in painted if "\x1b" in row]
        # The forged receipt text may survive as PROSE — it is inert once the
        # escapes are gone — but it must not be able to move the cursor.
        assert not [row for row in painted if "[2K" in row or "[1A" in row]


@pytest.mark.asyncio
async def test_a_pending_prompt_does_not_widen_the_tool_ledger() -> None:
    """The ledger's name column is sized by rows that RAN, not by questions.

    An approval prompt carries a `tool_name` and draws nothing in that column,
    so counting it shifted every settled summary right for a call that had not
    run — and refusing the call did not give the cells back.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(90, 30)) as pilot:
        await _boot(pilot, app)
        view = app.query_one(TranscriptView)
        view.append_block(ToolCard("t1", "bash", {}, ""))
        await pilot.pause()
        settled = view.tool_name_col

        block = ApprovalBlock("mcp__linear_create_initiative", "run: x")
        view.append_block(block)
        await pilot.pause()
        assert view.tool_name_col == settled

        block.resolve(False, answer="n")
        await pilot.pause()
        assert view.tool_name_col == settled


@pytest.mark.asyncio
async def test_clearing_the_transcript_forgets_the_ledger_width() -> None:
    """A derived measurement must not outlive the blocks it was derived from."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 30)) as pilot:
        await _boot(pilot, app)
        view = app.query_one(TranscriptView)
        view.append_block(ToolCard("t1", "mcp__linear_create_initiative", {}, ""))
        await pilot.pause()
        widened = view.tool_name_col
        assert widened > 8

        view.clear_blocks()
        await pilot.pause()
        assert view.tool_name_col == 8


@pytest.mark.asyncio
async def test_a_tall_notice_is_separated_from_what_precedes_it() -> None:
    """Adaptive spacing reads BOTH neighbours, not just the one above.

    Asking only "was the previous block tall?" separated tall→short correctly
    and left short→tall packed flush — the same wall, built in the other order.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(40, 30)) as pilot:
        await _boot(pilot, app)
        view = app.query_one(TranscriptView)
        short = NoticeBlock("resumed session 4c1f", "info")
        tall = NoticeBlock(
            "tool approvals: auto - /approvals ask restores prompting for the rest "
            "of this session, including write and command tools",
            "warning",
        )
        view.append_block(short)
        view.append_block(tall)
        await pilot.pause()
        assert tall.spans_multiple_rows()
        assert tall.has_class("gap-above")


@pytest.mark.asyncio
async def test_a_call_being_dictated_shows_a_row_that_moves() -> None:
    """The reported freeze: a large `write` painted NOTHING while it streamed.

    A tool call does not exist until its last argument token arrives, so the
    transcript held still for as long as the model took to dictate one — minutes
    for a file — and the only reasonable reading of that frame was a hung agent.

    Two halves, and the second is not decoration: providers open the call and
    then go silent before delivering the arguments in a burst (measured at 80
    seconds on a real Anthropic stream), so a byte counter alone is static text
    through exactly the pause that needs explaining. The clock is what moves.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(ToolComposing(ToolCallComposeEvent(tool_call_id="c1", tool_name="write")))
        await pilot.pause()
        painted = [strip.text for strip in app.screen._compositor.render_strips()]
        assert [row for row in painted if "composing" in row]
        # No size until there is one: a `0 B` that never moves reads as stuck.
        assert not [row for row in painted if "0 B" in row]

        app.post_message(
            ToolComposing(
                ToolCallComposeEvent(tool_call_id="c1", tool_name="write", argument_bytes=14079)
            )
        )
        await pilot.pause()
        painted = [strip.text for strip in app.screen._compositor.render_strips()]
        assert [row for row in painted if "13.7 KB" in row]

        # The execution ADOPTS that row rather than mounting a second one.
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(
                    tool_call_id="c1", tool_name="write", args={"path": "/tmp/x"}
                )
            )
        )
        await pilot.pause()
        painted = [strip.text for strip in app.screen._compositor.render_strips()]
        assert len([row for row in painted if "write" in row]) == 1
        assert not [row for row in painted if "composing" in row]


@pytest.mark.asyncio
async def test_an_adopted_row_times_the_tool_not_the_dictation() -> None:
    """The duration on a receipt is how long the TOOL took.

    The row is mounted when the model starts dictating, which for a large call
    is minutes before anything runs. Left alone, the clock that started then was
    still running when the tool finished, so a `write` that executed in
    milliseconds settled as `✓ 101s` — and the ledger beneath it, whose rows were
    never composed, measured something else entirely under the same glyph.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(ToolComposing(ToolCallComposeEvent(tool_call_id="c1", tool_name="write")))
        await pilot.pause(0.55)  # "dictation" time that must not be billed
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(tool_call_id="c1", tool_name="write", args={"path": "/x"})
            )
        )
        await pilot.pause(0.05)
        app.post_message(
            ToolEnded(
                ToolExecutionEndEvent(
                    tool_call_id="c1",
                    tool_name="write",
                    result=ToolResult(tool_call_id="c1", tool_name="write", content=[]),
                )
            )
        )
        await pilot.pause()
        card = next(iter(app.query(ToolCard)))
        assert card._duration is not None
        assert card._duration < 0.4, f"the dictation was billed to the tool: {card._duration}"


@pytest.mark.asyncio
async def test_the_composing_row_sheds_its_label_before_its_facts() -> None:
    """Below the width where both fit, the two things that MOVE survive.

    Truncating instead protected `composing…` — boilerplate next to a row that
    is already visibly live — and cut the byte count and the clock, so below 39
    columns the row stopped changing at all and two calls dictating 12 KB and
    61 B painted identically.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(34, 24)) as pilot:
        await _boot(pilot, app)
        app.post_message(
            ToolComposing(
                ToolCallComposeEvent(tool_call_id="c1", tool_name="write", argument_bytes=12700)
            )
        )
        await pilot.pause()
        painted = [strip.text for strip in app.screen._compositor.render_strips()]
        row = next(row for row in painted if "12.4 KB" in row)
        assert "composing" not in row


@pytest.mark.asyncio
async def test_an_interrupted_record_sheds_its_label_like_the_live_row() -> None:
    """Three different interrupted records must not paint identically.

    `mark_interrupted` set the summary but not the composing facts, so the
    label-shed ladder — gated on the composing STATE — could not reach the row
    the live one turns into, and 40 columns down three materially different
    records were byte-identical.
    """
    from local_operator.tui.widgets.tool_card import ToolCard

    seen: set[str] = set()
    for size in (12, 19_199, 4_011_000):
        card = ToolCard("c1", "write", {}, None)
        card.set_composing(size)
        card.mark_interrupted()
        seen.add(card._summary)

    assert len(seen) == 3, seen
    # And the facts are what survives the shed, not the prose.
    assert all("composed" in text for text in seen)


@pytest.mark.asyncio
async def test_a_dictated_name_moves_its_own_row_but_not_the_shared_column() -> None:
    """Two rules meet on this row and the narrower one wins.

    The ROW must follow the tool name as its fragments arrive — a provider that
    splits `mcp__linear_create_issue` left `mcp` on screen with the wrong icon for
    the whole dictation. The shared COLUMN must not: the name is model-controlled
    and arrives in pieces, so one announced 200-character name took the column to
    its cap and shifted every settled receipt beside it, and the width outlived
    the row when it settled as `never sent`. Same argument the column already
    makes for a pending approval — a name earns the column once the call it names
    has started.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(110, 24)) as pilot:
        await _boot(pilot, app)
        view = app.query_one(TranscriptView)
        settled = ToolCard("s1", "read", {}, None)
        view.append_block(settled)
        settled.mark_done("read a file")
        await pilot.pause()

        def settled_row() -> str:
            return next(
                (s.text for s in app.screen._compositor.render_strips() if "read a file" in s.text),
                "",
            )

        before = settled_row()
        app.post_message(ToolComposing(ToolCallComposeEvent(tool_call_id="c1", tool_name="mcp")))
        await pilot.pause()
        first = next(
            (s.text for s in app.screen._compositor.render_strips() if "composing" in s.text), ""
        )
        app.post_message(
            ToolComposing(ToolCallComposeEvent(tool_call_id="c1", tool_name="mcp__" + "z" * 200))
        )
        await pilot.pause()
        renamed = next(
            (s.text for s in app.screen._compositor.render_strips() if "composing" in s.text), ""
        )
        during = settled_row()

    # The row itself followed the fragment.
    assert first != renamed, (first, renamed)
    # The settled receipt beside it did not move a cell.
    assert before == during, (before, during)


@pytest.mark.asyncio
async def test_a_marker_rung_exists_between_the_words_and_the_glyph() -> None:
    """The hazard ladder has THREE rungs, and the middle one must be reachable.

    A test asserting only "the grammar never returns as the frame narrows" passes
    whether or not the middle rung exists, so it defends nothing: collapsing the
    marker rung into the glyph rung is monotone too. What distinguishes them is
    that some width must exist where the row still leads with the prompt glyph
    `?` — it is a question — AND carries a separate `!` for the hazard, and that
    those widths form ONE contiguous band.

    Both halves are load-bearing, and the first alone is not enough: with the
    floor rung concatenated AFTER the peers the marker band still exists at 22
    widths, in two runs either side of a glyph island, and a non-emptiness
    assertion passes. Measured across the three structures — head one run
    [[28, 57]], peers-first two runs, floor-first none.
    """
    both: list[int] = []
    # A fresh app per width: `screen.styles.width` does not re-run the frame
    # decision the block makes from its own size, so an in-place resize measures
    # the first frame over and over.
    for width in range(90, 19, -1):
        app = OperatorApp(lambda: _factory(SteerableSession()))
        async with app.run_test(size=(width, 20)) as pilot:
            await _boot(pilot, app)
            view = app.query_one(TranscriptView)
            view.append_block(ApprovalBlock("write_file", f"{OUTSIDE_MARKER} write: /etc/hosts"))
            # Wait for the row to SETTLE, not for 50ms and not merely for it to
            # be non-empty. A flat pause is a guess about how long a compositor
            # pass takes, and when it lost the race the width was measured off a
            # frame the user never sees, dropped out of `both`, and SPLIT THE
            # BAND — so `len(runs) == 1` failed with a plausible two-run result
            # ([49, 39, 28]) that pointed at the hazard ladder instead of at the
            # clock. Measured ~1-2 failures in 9 full-suite runs on unmodified
            # main, and 2 in 8 under CPU contention.
            #
            # Waiting for a NON-EMPTY row does not fix it, which is the trap
            # here: instrumenting the failing sweep shows `empties=0` and ~33
            # frames carrying the intermediate "words" rung — `?` present, `!`
            # absent. That frame is non-empty, so a first-non-empty read accepts
            # it and classifies the width false exactly as an unpainted frame
            # would. The defect was never emptiness; it is reading a frame that
            # has not finished reflowing.
            #
            # So the read is repeated until it stops changing: two consecutive
            # identical rows with a pause between them. That is the same rule
            # the rest of this suite follows — wait for the condition, not for a
            # duration — and it holds the band under the load that splits it.
            # The ceiling is a deadlock guard rather than a timing assumption.
            row = ""
            for _ in range(200):
                await pilot.pause()
                current = next(
                    (
                        strip.text
                        for strip in app.screen._compositor.render_strips()
                        if "write_file" in strip.text
                    ),
                    "",
                )
                if current and current == row:
                    break
                row = current
            assert row, f"the approval row never painted at width {width}"
        if PROMPT_GLYPH in row and HAZARD_GLYPH in row:
            both.append(width)

    assert both, "no width leads with `?` and carries a separate `!`"
    # ONE band, not merely a non-empty set. Collapsing the floor rung into the
    # peers leaves 22 widths in TWO runs with a glyph island between them — the
    # hazard hopping slots that the two-list split exists to prevent — and a
    # non-emptiness assertion passes straight through it. Measured: head one run
    # [[28, 57]], peers-first two runs [[34, 38], [41, 57]], floor-first none.
    runs = [w for w in both if w - 1 not in both]
    assert len(runs) == 1, (runs, both)


@pytest.mark.asyncio
async def test_the_two_hazards_spell_out_different_sentences() -> None:
    """One boolean was carrying two reasons and only one sentence.

    A target that cannot be resolved escalates exactly like one outside the
    workspace — and must, since nothing can be said about where it is. But it is
    visibly under the workspace root, so `outside the workspace — write:
    /ws/a\\x00b` argues with itself, and a clause the user can see is wrong is one
    they learn to ignore on the genuine escape.

    Only the WORDS rung differs. At the widths where the clause collapses to `!`
    the row makes no location claim and is already honest for both, which is what
    keeps the ladder a single shape.
    """
    cases = {
        "outside": (f"{OUTSIDE_MARKER} write: /etc/hosts", "outside the workspace"),
        "unresolvable": (f"{UNRESOLVABLE_MARKER} write: '/ws/a\\x00b'", "unresolvable"),
    }
    for label, (detail, expected) in cases.items():
        app = OperatorApp(lambda: _factory(SteerableSession()))
        async with app.run_test(size=(96, 20)) as pilot:
            await _boot(pilot, app)
            app.query_one(TranscriptView).append_block(ApprovalBlock("write_file", detail))
            # Settled read, for the reason its two neighbours document: a
            # mid-reflow frame is non-empty and carries the WRONG clause, which
            # is precisely what this test distinguishes. Measured here at
            # 163-169ms against the `pause(0.1)` this replaced — the same
            # overdraft as the threshold test below, found by measuring rather
            # than by waiting for it to fail in CI.
            row = ""
            for _ in range(400):
                await pilot.pause()
                current = next(
                    (
                        strip.text
                        for strip in app.screen._compositor.render_strips()
                        if "write_file" in strip.text
                    ),
                    "",
                )
                if current and current == row:
                    break
                row = current
            assert row, f"the approval row never painted for {label}"
        assert expected in row, (label, row)
        # The parser's own bracket token must never reach the sentence.
        assert "[" not in row, (label, row)
        # And the two must not paint the same clause.
        if label == "unresolvable":
            assert "outside the workspace" not in row, row


@pytest.mark.asyncio
async def test_the_verbose_threshold_measures_this_row_s_own_clause() -> None:
    """`_verbose_min_width` counts the clause the row will actually paint.

    Assuming the longer of the two costs the shorter one its verb: measuring
    `outside the workspace — ` for a row that spells `unresolvable — ` puts the
    threshold 9 columns too high, so `allow` is shed while the explicit form
    still fits. Measured band 56-64; at 56 the correct row is
    `? allow write_file  unresolvable — write: '/ws/a\\x00b'` and the assumed one
    drops `allow`.

    The outside-workspace clause is unaffected (threshold 63 either way), which is
    why the whole suite stayed green with this reverted — the regression lives
    only on the clause introduced alongside it.
    """
    detail = f"{UNRESOLVABLE_MARKER} write: '/ws/a\\x00b'"
    kept: list[int] = []
    for width in range(66, 61, -1):
        app = OperatorApp(lambda: _factory(SteerableSession()))
        async with app.run_test(size=(width, 20)) as pilot:
            await _boot(pilot, app)
            app.query_one(TranscriptView).append_block(ApprovalBlock("write_file", detail))
            # Read the SETTLED row, not the row after a fixed pause. This is the
            # same rule the hazard-ladder test three functions up already
            # follows, and for the identical reason it documents: the defect is
            # never emptiness, it is reading a frame that has not finished
            # reflowing. A mid-reflow frame is non-empty and sheds `allow`
            # exactly as a genuine threshold regression would, so the flat pause
            # cannot tell the two apart.
            #
            # The pause it replaced was `0.05`. Measured on this machine the row
            # settles at 160-210ms across this width band, so the test was
            # passing on a 3-4x overdraft that nothing enforced: it survived only
            # because the branch's old `pause(0.25)` boot happened to leave that
            # much slack behind it, and main's `_boot` (~67ms, condition-based)
            # correctly stopped donating it. Polling removes the bet rather than
            # re-tuning it. The ceiling is a deadlock guard, not a timing
            # assumption.
            row = ""
            for _ in range(400):
                await pilot.pause()
                current = next(
                    (
                        strip.text
                        for strip in app.screen._compositor.render_strips()
                        if "write_file" in strip.text
                    ),
                    "",
                )
                if current and current == row:
                    break
                row = current
            assert row, f"the approval row never painted at width {width}"
        if "allow" in row:
            kept.append(width)
        # Whatever the verb does, the clause the row measured is the one it paints.
        assert "unresolvable" in row, (width, row)

    # Every width in the band keeps the verb — the point of measuring the real
    # clause. Measuring the longer clause instead sheds `allow` from 64 down,
    # so a single missing width here is that threshold returning.
    assert kept == list(range(66, 61, -1)), kept


@pytest.mark.asyncio
async def test_a_background_jobs_approval_survives_the_parents_stop_latch() -> None:
    """The deny latch is TURN-scoped; a background job is not.

    ``_turn_epoch`` advances on a parent ``TurnStarted`` and nothing else, so a
    subagent still running after its parent's turn ended carried that turn's
    dead epoch. Any stop latched during that turn then denied the child's
    write/exec tools — with no card mounted, so the user saw a tool that simply
    did not work rather than a decision anyone made.

    Refusing the latch for a job costs no stop that exists: ``Session.abort``
    aborts only the parent's turn signal and never touches ``self.jobs``, and
    ``action_stop`` is gated on the parent streaming, so Esc could not reach a
    background child either way.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        for _ in range(12):
            await pilot.pause()
        # Arm the latch exactly as a stop during the parent's turn does.
        app._turn_epoch = 1
        app._approvals_denied_epoch = 1

        # The parent's own ask is still refused, and silently — that is the
        # behaviour the latch exists for and it must not change.
        assert await app.request_tool_approval("bash", "rm -rf /tmp/x") is False

        # The child's ask reaches the user instead of vanishing.
        task = asyncio.create_task(app.request_tool_approval("bash", "echo hi", job_id="job-7"))
        for _ in range(10):
            await pilot.pause()
        assert not task.done(), "a job's approval was denied without asking"
        assert app._approval is not None, "no card was mounted for the job"
        # `resolve` takes the DECISION, not the keystroke: passing "y" happened
        # to be truthy, so this asserted the right outcome for the wrong reason.
        app._approval.resolve(True)
        assert await task, "the answered job approval did not resolve truthy"


@pytest.mark.asyncio
async def test_a_prompt_never_eats_a_half_typed_prompt() -> None:
    """A question arriving mid-draft must not cost the user their sentence.

    A prompt is raised by the AGENT, so it lands whenever the tool call
    happens — often while the user is part-way through typing. Two things have
    to hold: the composer's text survives untouched, and the CARET stays where
    the user put it, so the sentence can be finished where it was left.

    The card yields its usual focus grab to a non-empty composer for the
    stronger reason recorded in `test_a_prompt_arriving_mid_sentence_…`: the
    answer keys are ordinary characters, so taking the caret mid-sentence feeds
    the rest of the sentence to the card, and on an approval `y` authorises the
    call. Nothing is lost by yielding, because the advertised keys are routed
    from the composer.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        ask = await _booted_gate(pilot, session)
        editor = app.query_one(Editor)
        editor.focus()
        editor.load_text("please clean up the stale rows and then")
        await pilot.pause()

        pending = asyncio.ensure_future(ask("bash", "run: rm -rf ./build"))
        for _ in range(100):
            if app.query(ApprovalPrompt):
                break
            await pilot.pause(0.02)
        await pilot.pause()

        # The question is up, the draft is intact, and the caret never moved.
        assert app.query(ApprovalPrompt), "no prompt was raised"
        assert isinstance(app.screen.focused, Editor)
        assert editor.text == "please clean up the stale rows and then"

        # The draft is what makes this the interesting case: with text in the
        # buffer the routing stands down entirely, so `y` is TYPED, not taken.
        await pilot.press("y")
        await pilot.pause()
        assert not pending.done(), "a keystroke answered while a draft was open"
        # Typed into the buffer (at the caret, which `load_text` leaves at the
        # start) rather than taken as an answer. What matters is that the draft
        # is intact and one character longer, not where the caret happened to be.
        typed = app.query_one(Editor).text
        assert "please clean up the stale rows and then" in typed, typed
        assert typed.count("y") == "please clean up the stale rows and then".count("y") + 1

        # Clearing the draft hands the keys back: the card is answerable again.
        app.query_one(Editor).load_text("")
        await pilot.pause()
        await pilot.press("y")
        assert await asyncio.wait_for(pending, 2) is True


@pytest.mark.asyncio
async def test_the_answer_keys_work_from_the_composer() -> None:
    """The reported defect: keys the card advertises must actually answer it.

    A prompt is raised while the user is looking at the composer, they press
    the key the card is showing them, and it lands in the draft as text while
    the tool goes on waiting. The keys are ROUTED to the live prompt now, with
    the caret left where it is — see `route_key_to_live_prompt` for why moving
    focus instead was worse than the bug.
    """
    for key, expected in (("y", True), ("n", False), ("A", True)):
        session = SteerableSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            ask = await _booted_gate(pilot, session)
            pending = asyncio.ensure_future(ask("bash", "run: rm -rf ./build"))
            for _ in range(100):
                if app.query(ApprovalPrompt):
                    break
                await pilot.pause(0.02)
            await pilot.pause()

            await _focus_composer(pilot, app)
            await pilot.press(key)
            assert await asyncio.wait_for(pending, 2) is expected, key
            # The key answered instead of being typed.
            assert app.query_one(Editor).text == "", key


@pytest.mark.asyncio
async def test_typing_a_steer_at_a_live_prompt_never_answers_it(
    unraceable_answer_hold: None,
) -> None:
    """A sentence typed at a question is a steer, not an authorisation.

    This is the hazard the routing takes on, and it is the reason a routed key
    is held for one keystroke rather than acted on immediately. An earlier
    revision moved FOCUS to the prompt whenever the composer was empty — which
    is exactly when a user is about to start typing — so the first character of
    `yes do it` authorised a pending `rm -rf` and left `es do it` in the buffer.

    Every phrase here begins with a character that WOULD answer if it stood
    alone, which is the whole point: the ambiguity is real and is resolved by
    what follows.
    """
    for phrase in ("yes do it", "no wait", "and also drop the index"):
        session = SteerableSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            ask = await _booted_gate(pilot, session)
            pending = asyncio.ensure_future(ask("bash", "run: rm -rf /Users/x/project/data"))
            for _ in range(100):
                if app.query(ApprovalPrompt):
                    break
                await pilot.pause(0.02)
            await pilot.pause()

            await _focus_composer(pilot, app)
            for character in phrase:
                await pilot.press("space" if character == " " else character)
            await pilot.pause()

            # Nothing was authorised...
            assert not pending.done(), (phrase, "typing a steer answered the prompt")
            assert app._approve_all is False, phrase
            # ...and the whole sentence survived, including its first character.
            assert app.query_one(Editor).text == phrase, phrase

            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, Exception):
                pass


@pytest.mark.asyncio
async def test_a_narrow_card_still_names_what_it_is_authorising() -> None:
    """An approval that cannot state its subject must not look answerable.

    Measured before the fix at 60x16: the card rendered `❯ 1. Allow`, a
    `showing 1–1 of 3` count, and the key hints — an authorisation prompt for
    `rm -rf` that never named the tool, the command, or the target, with the
    cursor parked on the permissive option. The question now outranks both the
    option rows and the count, so the last thing to go is what is being asked.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 16)) as pilot:
        ask = await _booted_gate(pilot, session)
        pending = asyncio.ensure_future(ask("bash", "run: rm -rf /Users/x/project/data"))
        for _ in range(100):
            if app.query(ApprovalPrompt):
                break
            await pilot.pause(0.02)
        await pilot.pause()

        lines = app.query(ApprovalPrompt).first().render_lines_for_test()
        assert lines, "the card drew nothing at a size it can draw at"
        text = "\n".join(lines)
        # The subject is on screen: the tool AND its target.
        assert "bash" in text, lines
        assert "rm -rf" in text, lines
        # And the exit is still stated.
        assert "esc" in lines[-1], lines

        pending.cancel()
        try:
            await pending
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_the_card_advertises_only_keys_that_do_something() -> None:
    """A key offered where it does nothing is a lie the card tells once.

    Two of them, both found by driving the real card: `1-9 jump` was printed
    unconditionally, so a card windowed down to one visible row still offered a
    range where `5`/`7`/`9` did nothing; and the approval letters `y`/`n`/`A`
    were live bindings rendered nowhere at all, which cost "allow all" its only
    discoverable shortcut.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        ask = await _booted_gate(pilot, session)
        pending = asyncio.ensure_future(ask("bash", "run: make"))
        for _ in range(100):
            if app.query(ApprovalPrompt):
                break
            await pilot.pause(0.02)
        await pilot.pause()

        prompt = app.query(ApprovalPrompt).first()
        lines = prompt.render_lines_for_test()
        text = "\n".join(lines)
        # Every answer key is printed beside the row it answers...
        for label, key, _why in APPROVAL_CHOICES:
            assert f"{key}." in text, (key, lines)
            assert label in text, (label, lines)
        # ...and the digit range is not advertised, because these rows are
        # addressed by letter and the digits would name a different keyboard.
        assert "1-9" not in text, lines

        pending.cancel()
        try:
            await pending
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_a_card_that_shows_no_options_cannot_approve() -> None:
    """An affirmative answer must never come from a card that offered nothing.

    On a terminal too short for the list the card shrinks to the question
    alone — and the answer letters went on working, so `y` approved
    `rm -rf …` from a frame that displayed no options at all. At height 13 it
    was worse still: the card's entire content was `esc deny`, the word for
    refusing, and `y` approved from it (D9, design round 2).

    Denial stays available in every state, because it is the safe direction and
    because Escape means stop everywhere in this app regardless of what the
    footer says.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 13)) as pilot:
        ask = await _booted_gate(pilot, session)
        pending = asyncio.ensure_future(ask("bash", "run: rm -rf /Users/x/project/data"))
        for _ in range(100):
            if app.query(ApprovalPrompt):
                break
            await pilot.pause(0.02)
        await pilot.pause()

        prompt = app.query(ApprovalPrompt).first()
        lines = prompt.render_lines_for_test()
        assert not prompt.visible_rows, (lines, "this size is meant to draw no options")
        # It still names what it is asking about...
        assert "rm -rf" in "\n".join(lines), lines

        # ...and the permissive keys are refused while it cannot show them.
        await pilot.press("y")
        await pilot.pause()
        assert not pending.done(), "y approved from a card showing no options"
        await pilot.press("A")
        await pilot.pause()
        assert not pending.done(), "A approved from a card showing no options"
        assert app._approve_all is False, "the session gate was disarmed invisibly"
        # Enter is refused for the same reason: the cursor is on a preselected
        # row the user was never shown.
        await pilot.press("enter")
        await pilot.pause()
        assert not pending.done(), "enter committed an unseen selection"

        # Denial is always available.
        await pilot.press("n")
        assert await asyncio.wait_for(pending, 2) is False


@pytest.mark.asyncio
async def test_a_prompt_hidden_by_a_shrink_comes_back_on_re_grow() -> None:
    """A shrink must not be a one-way door onto an unanswerable question.

    A hidden widget is not laid out and so receives no resize event, so a card
    that hid itself on a terminal too short to draw it could never learn the
    terminal had grown back. The question stayed invisible for the rest of the
    turn while the tool went on waiting (D10, design round 2).
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        ask = await _booted_gate(pilot, session)
        pending = asyncio.ensure_future(ask("bash", "run: make"))
        for _ in range(100):
            if app.query(ApprovalPrompt):
                break
            await pilot.pause(0.02)
        await pilot.pause()
        prompt = app.query(ApprovalPrompt).first()
        assert prompt.render_lines_for_test(), "no card to begin with"

        await pilot.resize_terminal(40, 6)
        for _ in range(14):
            await pilot.pause()
        assert not prompt.render_lines_for_test(), "the card should hide when it cannot draw"

        await pilot.resize_terminal(100, 30)
        for _ in range(20):
            await pilot.pause()
        assert prompt.render_lines_for_test(), "the question never came back"
        assert app.query_one("#prompt-host").display

        # And it is answerable again.
        await pilot.press("y")
        assert await asyncio.wait_for(pending, 2) is True


@pytest.mark.asyncio
async def test_a_held_answer_key_resolves_cleanly_on_every_second_key(
    unraceable_answer_hold: None,
) -> None:
    """The hold has three endings, and each has to leave a coherent state.

    A routed answer key is parked for one keystroke so that typing a word
    beginning with `y` cannot authorise anything. That parking window is new
    machinery, and what arrives next decides what the keystroke MEANT:

    - a printable key: the user was typing, so both characters land in the
      composer and nothing is answered;
    - Enter: the answer was deliberate, so it is taken. Releasing into the
      buffer first would submit a prompt the user never finished, and letting
      Enter through afterwards dropped the character entirely — measured as an
      empty submit and a lost `y`;
    - Escape: stop. The prompt settles as a denial and the composer is left
      EMPTY, rather than holding a stray `y` for a question that has gone.
    """

    async def _live(pilot, app, session):
        ask = await _booted_gate(pilot, session)
        pending = asyncio.ensure_future(ask("bash", "run: rm -rf ./build"))
        for _ in range(100):
            if app.query(ApprovalPrompt):
                break
            await pilot.pause(0.02)
        await pilot.pause()
        await _focus_composer(pilot, app)
        return pending

    # Enter takes the answer.
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        pending = await _live(pilot, app, session)
        await pilot.press("y")
        await pilot.press("enter")
        assert await asyncio.wait_for(pending, 2) is True
        assert app.query_one(Editor).text == "", "the held key was typed as well as taken"

    # Escape denies and leaves nothing behind.
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        pending = await _live(pilot, app, session)
        await pilot.press("y")
        await pilot.press("escape")
        assert await asyncio.wait_for(pending, 2) is False
        await pilot.pause()
        assert app.query_one(Editor).text == "", "a stray character outlived the question"

    # A printable key means it was typing all along.
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        pending = await _live(pilot, app, session)
        await pilot.press("y")
        await pilot.press("e")
        await pilot.pause()
        assert not pending.done(), "typing answered the prompt"
        assert app.query_one(Editor).text == "ye"
        pending.cancel()
        try:
            await pending
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_a_held_key_never_answers_a_question_it_was_not_meant_for(
    unraceable_answer_hold: None,
) -> None:
    """A key parked against a question that then settles must not answer another.

    The window is short, but a stop, a `/clear` or a teardown can land inside
    it — and the next tool of the same turn may raise its own prompt. A hold
    that survived would authorise a call the user never saw.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        ask = await _booted_gate(pilot, session)
        first = asyncio.ensure_future(ask("bash", "run: one"))
        for _ in range(100):
            if app.query(ApprovalPrompt):
                break
            await pilot.pause(0.02)
        await pilot.pause()
        await _focus_composer(pilot, app)

        await pilot.press("y")
        assert app._held_answer_key is not None, "the key was not held"
        # The question settles from somewhere else entirely, mid-hold.
        assert app._approval is not None
        app._approval.resolve(False, answer="n")
        assert await asyncio.wait_for(first, 2) is False
        await pilot.pause()

        # Nothing is left parked, and the session gate was not disarmed.
        assert app._held_answer_key is None
        assert app._approve_all is False


@pytest.mark.asyncio
async def test_a_prompt_arriving_mid_sentence_does_not_take_the_caret() -> None:
    """A question can land while the user is typing, and must not hijack it.

    This is the mount-time twin of the routing hazard. A prompt is raised by
    the AGENT, so it can arrive at any moment — including between two
    characters of a sentence. Taking the caret then starts feeding the rest of
    that sentence to the card, where the answer keys are ordinary letters, and
    on an approval the first `y` AUTHORISES the call. Measured before the fix:
    typing `please ` then `yes do it` through the mount approved
    `rm -rf /Users/x/project/data` and left `please es do it` in the buffer
    (D12, design round 3).

    Nothing is lost by yielding: the advertised keys are routed from the
    composer, and the footer names only the ones that work from there.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        ask = await _booted_gate(pilot, session)
        editor = app.query_one(Editor)
        editor.focus()
        for character in "please ":
            await pilot.press("space" if character == " " else character)

        pending = asyncio.ensure_future(ask("bash", "run: rm -rf /Users/x/project/data"))
        for _ in range(10):
            await pilot.pause(0.02)
        # The question is up, and the caret has NOT moved.
        assert app.query(ApprovalPrompt), "no prompt was raised"
        assert isinstance(app.screen.focused, Editor)

        for character in "yes do it":
            await pilot.press("space" if character == " " else character)
        await pilot.pause()

        assert not pending.done(), "typing through the mount answered the prompt"
        assert app._approve_all is False
        assert app.query_one(Editor).text == "please yes do it"

        pending.cancel()
        try:
            await pending
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_a_resize_while_typing_never_moves_the_keyboard() -> None:
    """A resize is not an instruction to answer the question.

    The re-focus that brings a prompt back after a shrink (D10) fired on EVERY
    resize, with no check on where the caret was. So a one-column resize while
    the user was typing took the keyboard, and the next character of their
    sentence answered the question — on an approval, a `y` meant as text
    AUTHORISED `rm -rf /data`. The composer's one-keystroke hold is a correct
    defence and was simply bypassed, because it only guards keys that arrive at
    the composer (F10, agent review round 7).

    The two cases are cleanly distinguishable: hiding a widget leaves focus as
    None, which is the state D10 repairs; a typing user leaves it as the
    Editor.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        ask = await _booted_gate(pilot, session)
        editor = app.query_one(Editor)
        editor.focus()
        for character in "wait a":
            await pilot.press("space" if character == " " else character)
        await pilot.pause()

        pending = asyncio.ensure_future(ask("bash", "run: rm -rf /data"))
        for _ in range(100):
            if app.query(ApprovalPrompt):
                break
            await pilot.pause(0.02)
        await pilot.pause()
        assert isinstance(app.screen.focused, Editor)

        await pilot.resize_terminal(99, 30)
        for _ in range(10):
            await pilot.pause(0.05)
        assert isinstance(app.screen.focused, Editor), "a resize took the keyboard"

        await pilot.press("y")
        await pilot.pause()
        assert not pending.done(), "a resize let a typed character authorise the call"
        assert app._approve_all is False
        assert app.query_one(Editor).text == "wait ay"

        pending.cancel()
        try:
            await pending
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_answering_one_prompt_does_not_move_the_caret_off_a_draft() -> None:
    """A question settling elsewhere must not hand the keyboard to the other one.

    With an ask and an approval both live, answering the approval hands the
    keyboard to the surviving question — but only if the DEPARTING card had it.
    If the user is typing, removing the answered card makes Textual focus the
    next focusable node, which is the surviving prompt, and the rest of the
    sentence becomes an answer (F11, agent review round 7 — F10's defect
    through the overlap door).
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        ask = await _booted_gate(pilot, session)
        asked = asyncio.create_task(
            app.request_user_choice([_ask_question("stale", "Which rollout?")])
        )
        for _ in range(12):
            await pilot.pause(0.02)
        approving = asyncio.ensure_future(ask("bash", "run: make"))
        for _ in range(20):
            await pilot.pause(0.02)

        editor = app.query_one(Editor)
        editor.focus()
        for character in "hold on":
            await pilot.press("space" if character == " " else character)
        await pilot.pause()
        assert isinstance(app.screen.focused, Editor)

        # The approval settles from somewhere else entirely.
        assert app._approval is not None
        app._approval.resolve(False, answer="n")
        assert await asyncio.wait_for(approving, 2) is False
        for _ in range(14):
            await pilot.pause(0.05)

        # The caret stayed put and the draft is intact.
        assert isinstance(app.screen.focused, Editor), "the caret moved to the other question"
        assert app.query_one(Editor).text == "hold on"

        asked.cancel()
        try:
            await asked
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_nothing_takes_the_keyboard_from_someone_mid_sentence() -> None:
    """The invariant, stated once over every path that can move focus.

    Six separate findings across this review were the same defect reaching the
    caret by a different door: the mount (D12), an empty buffer (F9), a
    rewording deletion (D18), a resize (F10), a sibling prompt settling (F11),
    and the overlap hand-off. Each was fixed where it was found, which is how
    the next door stayed open long enough for a reviewer to walk through it.

    So this asserts the property rather than the instances: with a draft in the
    composer and a question live, NOTHING moves the caret. A new path that
    forgets the rule fails here even if nobody thought to write a test for that
    path — which is the point.
    """
    draft = "mid sentence"

    async def _drafting(app, pilot, ask, *, multi=False, approval=False, overlap=False):
        editor = app.query_one(Editor)
        editor.focus()
        for character in draft:
            await pilot.press("space" if character == " " else character)
        await pilot.pause()
        futures = []
        if overlap:
            futures.append(
                asyncio.create_task(app.request_user_choice([_ask_question("stale", "Which?")]))
            )
            for _ in range(12):
                await pilot.pause(0.02)
            futures.append(asyncio.ensure_future(ask("bash", "run: make")))
        elif approval:
            futures.append(asyncio.ensure_future(ask("bash", "run: make")))
        else:
            futures.append(
                asyncio.create_task(
                    app.request_user_choice([_ask_question("stale", "Which?", multi=multi)])
                )
            )
        for _ in range(20):
            await pilot.pause(0.02)
        return futures

    async def _mount(app, pilot):
        return None

    async def _resize_one_column(app, pilot):
        await pilot.resize_terminal(99, 30)

    async def _shrink_below_minimum_and_grow(app, pilot):
        await pilot.resize_terminal(40, 6)
        for _ in range(8):
            await pilot.pause(0.05)
        await pilot.resize_terminal(100, 30)

    async def _poll_tick(app, pilot):
        app._refresh_band()

    async def _settle_the_sibling(app, pilot):
        assert app._approval is not None
        app._approval.resolve(False, answer="n")

    async def _clear_the_transcript(app, pilot):
        app.action_clear_transcript()

    cases = (
        ("mount, ask", _mount, {}),
        ("mount, approval", _mount, {"approval": True}),
        ("mount, multi-select", _mount, {"multi": True}),
        ("a one-column resize", _resize_one_column, {"approval": True}),
        ("a shrink past the minimum and back", _shrink_below_minimum_and_grow, {}),
        ("the 1 Hz poll", _poll_tick, {}),
        ("a sibling prompt settling", _settle_the_sibling, {"overlap": True}),
        ("clearing the transcript", _clear_the_transcript, {}),
    )

    for label, action, kwargs in cases:
        session = SteerableSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            ask = await _booted_gate(pilot, session)
            futures = await _drafting(app, pilot, ask, **kwargs)
            await action(app, pilot)
            for _ in range(12):
                await pilot.pause(0.05)

            assert isinstance(app.screen.focused, Editor), f"{label} took the keyboard"
            assert app.query_one(Editor).text == draft, f"{label} disturbed the draft"

            for future in futures:
                future.cancel()
                try:
                    await future
                except (asyncio.CancelledError, Exception):
                    pass


# ---------------------------------------------------------------------------
# The Esc ladder: stop the turn, then (on a second press) the subagents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_esc_stops_the_turn_and_offers_to_stop_the_subagents() -> None:
    """One press stops the parent and SAYS that children are still running.

    The offer is the load-bearing half. A stop that silently left three agents
    burning tokens is the defect this ladder exists to fix, and a user has no
    way to discover the wider stop unless the first press names it.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 2

        await pilot.press("escape")
        await pilot.pause()

        assert session.aborts == ["interrupted"]
        # The children are NOT stopped by the first press.
        assert session.subagent_cancels == []
        assert any("2 subagents still running" in row for row in rows(app))
        assert any("esc again to stop them" in row for row in rows(app))


@pytest.mark.asyncio
async def test_a_second_esc_stops_the_subagents_and_confirms_it() -> None:
    """The escalation press cancels the children and reports the count."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 2

        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        assert session.subagent_cancels == ["interrupted"]
        assert any("stopped 2 subagents" in row for row in rows(app))
        # One line, not two: the stale "still running" row would read as current.
        assert not any("still running" in row for row in rows(app))


@pytest.mark.asyncio
async def test_a_slow_second_esc_does_not_stop_the_subagents() -> None:
    """The offer expires, so a much later Esc is a fresh first press.

    Without the window, an Esc pressed minutes after an unrelated stop would
    silently kill every child — the expensive, unrecoverable action arriving
    with no warning attached to the keystroke that triggered it.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 1

        await pilot.press("escape")
        await pilot.pause()
        # Age the offer past its window without waiting for it.
        assert app._stop_offered_at is not None, "the first press must make the offer"
        app._stop_offered_at -= DOUBLE_STOP_WINDOW_S + 1
        await pilot.press("escape")
        await pilot.pause()

        assert session.subagent_cancels == [], "an expired offer must not escalate"


@pytest.mark.asyncio
async def test_esc_with_no_children_says_nothing_about_subagents() -> None:
    """The common case stays silent: no ladder, no line, no second rung."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True

        await pilot.press("escape")
        await pilot.pause()

        assert session.aborts == ["interrupted"]
        # Inspect the ladder state, not every screen cell: the cwd or branch may
        # legitimately contain the word "subagent" and is unrelated chrome.
        assert app._stop_notice is None
        assert app._stop_offered_at is None


@pytest.mark.asyncio
async def test_esc_stops_subagents_even_when_the_parent_turn_has_ended() -> None:
    """Children outlive the turn that launched them, so Esc must still reach
    them when the parent is already idle — that is the exact state a user is
    in when they notice work still running and press Esc."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = False  # the parent finished; children did not
        session.running_children = 3

        await pilot.press("escape")
        await pilot.pause()
        assert any("3 subagents still running" in row for row in rows(app))

        await pilot.press("escape")
        await pilot.pause()
        assert session.subagent_cancels == ["interrupted"]
        assert any("stopped 3 subagents" in row for row in rows(app))


@pytest.mark.asyncio
async def test_esc_with_nothing_running_still_never_clears_the_composer() -> None:
    """The rule the ladder must not break: Esc is not a way to lose a draft."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.load_text("a half-typed prompt")

        await pilot.press("escape")
        await pilot.pause()

        assert editor.text == "a half-typed prompt"
        assert session.aborts == []


# ---------------------------------------------------------------------------
# The Ctrl+C ladder: a draft is cleared before the exit ladder starts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ctrl_c_clears_a_draft_before_arming_the_exit_ladder() -> None:
    """Ctrl+C on a half-typed prompt means "scrap that", not "start exiting".

    This also removes the ladder's sharpest edge: the quitting press used to be
    two taps away while the user was typing, so a reflexive double-tap at the
    composer could close the app with a drafted prompt on screen.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.load_text("draft to be scrapped")

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert editor.text == ""
        assert not any("ctrl+c again to exit" in row for row in rows(app))
        # The ladder was NOT armed: the press was spent on the composer.
        assert app._last_interrupt_at == 0.0
        assert app.is_running
        # And the text is recoverable rather than destroyed.
        assert "draft to be scrapped" in editor.prompt_history()


@pytest.mark.asyncio
async def test_two_ctrl_c_presses_from_a_draft_do_not_exit() -> None:
    """The first press clears; the SECOND is a first press for the ladder."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.query_one(Editor).load_text("draft")

        await pilot.press("ctrl+c")
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app.is_running, "a double-tap from a draft must never quit"
        assert any("ctrl+c again to exit" in row for row in rows(app))


@pytest.mark.asyncio
async def test_ctrl_c_on_an_empty_composer_keeps_the_exit_ladder() -> None:
    """The existing gesture is unchanged when there is no draft to clear."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert session.aborts == ["interrupted"]
        assert any("ctrl+c again to exit" in row for row in rows(app))

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.is_running


@pytest.mark.asyncio
async def test_whitespace_only_draft_is_not_treated_as_a_draft() -> None:
    """Spaces and newlines are not something a user asked to keep, so the key
    keeps its interrupt meaning rather than being silently swallowed."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        app.query_one(Editor).load_text("   \n  ")

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert session.aborts == ["interrupted"]
        assert any("ctrl+c again to exit" in row for row in rows(app))


@pytest.mark.asyncio
async def test_clearing_the_transcript_disarms_the_stop_offer() -> None:
    """`/clear` removes the offer's line, so the offer must go with it.

    Two failures at once otherwise: `_replace_stop_notice` would try to remove
    a block the transcript no longer holds, and an escalation would stay armed
    whose terms the user can no longer read.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 2

        await pilot.press("escape")
        await pilot.pause()
        assert app._stop_offered_at is not None

        app.action_clear_transcript()
        await pilot.pause()
        assert app._stop_offered_at is None
        assert app._stop_notice is None

        # And the next Esc is a FIRST press, not an escalation.
        await pilot.press("escape")
        await pilot.pause()
        assert session.subagent_cancels == []


@pytest.mark.asyncio
async def test_a_stop_offer_does_not_survive_into_a_new_session() -> None:
    """The sharpest cross-session leak in this family: an armed escalation.

    Carried across a session swap, an Esc pressed just after `/new` would take
    up an offer made about the OLD conversation's children and cancel the NEW
    session's subagents — destroying work on the strength of a count the user
    was shown for a different conversation.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 2

        await pilot.press("escape")
        await pilot.pause()
        assert app._stop_offered_at is not None, "the offer must be armed first"

        await app._reload_session()
        await pilot.pause()

        assert app._stop_offered_at is None
        assert app._stop_notice is None


@pytest.mark.asyncio
async def test_ctrl_c_never_files_an_aside_question_in_prompt_history() -> None:
    """Review round 1, M2. The aside's "off the record" is a contract.

    The card prints "off the record — nothing here joins the chat", and
    `set_records_history(False)` is how that is enforced. `remember_draft`
    bypassed the flag, so Ctrl+C filed a question the user deliberately kept
    out of the conversation into the recallable history — one `up` and one
    Enter from being sent to the agent as a real turn.

    The press closes the card instead, which is the path that already means
    "done with this aside" and hands back the borrowed main-chat draft.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        app._open_aside()
        await pilot.pause()
        editor.load_text("is my salary competitive?")
        await pilot.pause()
        assert editor._records_history is False, "the aside must suppress recording"

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert not any(
            "salary" in entry for entry in editor.prompt_history()
        ), "the off-the-record question reached the recallable history"
        assert app.is_running


# ---------------------------------------------------------------------------
# Design review round 1: what the ladder SAYS when its own state has moved on.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_late_second_esc_is_acknowledged_rather_than_silently_ignored() -> None:
    """A press that misses the window must not repaint an identical row.

    Past DOUBLE_STOP_WINDOW_S the press falls through to the ordinary stop,
    which re-arms the offer. Reprinting the same string there made the press
    indistinguishable from a dropped keystroke — the rendered frame was byte
    for byte the same — on the very key this change exists to stop making
    silent promises (D1). The re-armed offer now states the constraint that
    defeated the press, so the user learns why nothing was stopped.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 2

        await pilot.press("escape")
        await pilot.pause()
        assert any("esc again to stop them" in row for row in rows(app))

        # The window lapses without a second press.
        app._stop_offered_at = time.monotonic() - (DOUBLE_STOP_WINDOW_S + 1)
        await pilot.press("escape")
        await pilot.pause()

        # Nothing was stopped — that part is by design — but the row CHANGED,
        # so the press is visibly acknowledged.
        assert session.subagent_cancels == []
        assert any(
            "too slow; esc again within" in row for row in rows(app)
        ), "the late press repainted an identical row and read as a dropped key"


@pytest.mark.asyncio
async def test_children_finishing_between_presses_is_not_reported_as_a_denial() -> None:
    """ "no subagents were running" must not contradict the offer (D2).

    `cancel_subagents` recounts at press time, so children that settle inside
    the window return zero. Printing a flat denial there replaces a row that
    just said "2 subagents still running" and reads as "the first message was
    wrong" — re-raising the credibility problem this change exists to fix,
    when what actually happened is better news than the offer implied.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 2

        await pilot.press("escape")
        await pilot.pause()

        # They finish on their own before the user's second press lands.
        session.running_children = 0
        await pilot.press("escape")
        await pilot.pause()

        painted = rows(app)
        assert any(
            "finished before the stop landed" in row for row in painted
        ), "the confirmation did not explain that the children had finished"
        assert not any(
            "no subagents were running" in row for row in painted
        ), "the confirmation flatly denied the offer the user had just read"


@pytest.mark.asyncio
async def test_a_stop_that_spares_background_jobs_says_so() -> None:
    """Escalating reads as "stop all of it", and `bash` jobs deliberately
    survive it (D3). The one moment that is worth a word is the confirmation,
    and only when such a job actually exists."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 1
        session.running_bash_jobs = 1

        await pilot.press("escape")
        await pilot.pause()
        app._stop_offered_at = time.monotonic()
        await pilot.press("escape")
        await pilot.pause()

        painted = rows(app)
        assert any("stopped 1 subagent" in row for row in painted)
        assert any(
            "background job" in row and "jobs cancel" in row for row in painted
        ), "the spared background job was not named"


@pytest.mark.asyncio
async def test_the_confirmation_stays_quiet_when_nothing_was_spared() -> None:
    """The `bash` clause is conditional: unconditional, it would be noise on
    every stop (D3)."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 1
        session.running_bash_jobs = 0

        await pilot.press("escape")
        await pilot.pause()
        app._stop_offered_at = time.monotonic()
        await pilot.press("escape")
        await pilot.pause()

        assert not any("background job" in row for row in rows(app))


@pytest.mark.asyncio
async def test_the_offer_retires_its_promise_when_the_window_closes() -> None:
    """An instruction that no key will honour must not stand (D4).

    Nothing cleared the row, so a transcript kept a warning-amber
    "esc again to stop them" for the rest of the session, scrolling up through
    the history with no visible expiry. The FACT survives — the children really
    are still running — only the promise goes, and it drops out of the warning
    weight because it is no longer something to act on within a window.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 2

        await pilot.press("escape")
        await pilot.pause()
        assert any("esc again to stop them" in row for row in rows(app))

        # Wait out the real window rather than reaching into the timer.
        await pilot.pause(DOUBLE_STOP_WINDOW_S + 0.5)

        painted = rows(app)
        assert not any(
            "esc again to stop them" in row for row in painted
        ), "the expired offer went on advertising an escalation no key honours"
        assert any(
            "2 subagents still running" in row for row in painted
        ), "the surviving fact was dropped along with the promise"


@pytest.mark.asyncio
async def test_taking_the_offer_is_not_undone_by_the_expiry_timer() -> None:
    """The timer is keyed on the stamp it was armed with, so it can never
    retire an offer that was already taken, nor a newer one armed since."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 2

        await pilot.press("escape")
        await pilot.pause()
        app._stop_offered_at = time.monotonic()
        await pilot.press("escape")
        await pilot.pause()
        assert session.subagent_cancels == ["interrupted"]

        # The first press's timer fires somewhere in here and must do nothing.
        await pilot.pause(DOUBLE_STOP_WINDOW_S + 0.5)

        assert any(
            "stopped 2 subagents" in row for row in rows(app)
        ), "the expiry timer overwrote the confirmation of a stop that happened"


@pytest.mark.asyncio
async def test_the_ladder_row_keeps_its_place_in_the_transcript() -> None:
    """Replacing the row must not move it (D5).

    `_replace_stop_notice` used to remove and re-append, so the row jumped to
    the transcript end and landed BELOW notices that had arrived after it —
    a row the user had already read reordering itself past later ones. The
    replacement now restates in place, which keeps both the replace-don't-stack
    decision and a chronological transcript.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 2

        await pilot.press("escape")
        await pilot.pause()
        # A notice arrives after the offer, as the aborted turn's own does.
        app._append_block(NoticeBlock("interrupted", "info"))
        await pilot.pause()

        app._stop_offered_at = time.monotonic()
        await pilot.press("escape")
        await pilot.pause()

        painted = [row for row in rows(app) if row.strip()]
        ladder = next(i for i, row in enumerate(painted) if "stopped 2 subagents" in row)
        later = next(i for i, row in enumerate(painted) if "interrupted" in row)
        assert ladder < later, "the restated ladder row jumped below a later notice"


@pytest.mark.asyncio
async def test_ctrl_c_says_where_the_draft_went() -> None:
    """Filing the draft into history is the point of this rung, but all the
    user sees is their sentence vanishing — and recoverability nobody can
    discover gets retyped instead of recalled (D6)."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "a half-typed prompt I do not want to lose"
        await pilot.pause()

        await pilot.press("ctrl+c")
        await pilot.pause()

        painted = rows(app)
        assert any(
            "draft cleared" in row and "↑ to recover" in row for row in painted
        ), "the draft vanished with nothing to say it was recoverable"
        # And the exit ladder is still NOT armed by this press.
        assert not any("ctrl+c again" in row for row in painted)
        assert app.is_running


@pytest.mark.asyncio
async def test_an_armed_offer_never_outranks_a_waiting_approval() -> None:
    """A permission gate takes Escape ahead of the escalation (R1).

    Round 1's F5 was the ask picker outranking an approval; the escalation
    branch was a second door to the same inversion. With the offer armed and
    an `rm -rf /` approval on screen, Escape has to answer the PROMPT — the
    thing the engine is blocked on and the most recently raised surface — not
    silently destroy every delegated child on a press the user could
    reasonably have aimed at the approval in front of them.

    The offer survives unspent, so the next press still escalates.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        ask = await _booted_gate(pilot, session)
        session.streaming = True
        session.running_children = 2

        pending = asyncio.ensure_future(ask("bash", "run: rm -rf /"))
        for _ in range(100):
            if isinstance(app.screen.focused, ApprovalPrompt):
                break
            await pilot.pause(0.02)
        assert app._live_prompt() is app._approval, "the approval is not the live prompt"

        # Arm the ladder exactly as a first Esc during the turn would.
        app._stop_offered_at = time.monotonic()
        app._stop_offer_count = 2

        await pilot.press("escape")
        await pilot.pause()

        assert (
            session.subagent_cancels == []
        ), "Escape escalated past a waiting approval and killed the subagents"
        assert await asyncio.wait_for(pending, 2) is False, "the approval was not denied"
        pending.cancel()


@pytest.mark.asyncio
async def test_the_late_row_promises_the_number_of_presses_it_actually_costs() -> None:
    """The re-armed offer must not overstate the cost by one (D9).

    The late press itself re-arms the ladder, so by the time its row is painted
    the offer is live and the very NEXT single press escalates. Saying "press
    esc twice" there described the ladder in general rather than the state the
    user is actually in — harmless to obey, but wrong on the one key this
    change exists to make honest.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 2

        await pilot.press("escape")
        await pilot.pause()

        # The window lapses, so this press cannot escalate — it re-arms.
        app._stop_offered_at = time.monotonic() - (DOUBLE_STOP_WINDOW_S + 1)
        await pilot.press("escape")
        await pilot.pause()
        assert session.subagent_cancels == []
        assert any("esc again within" in row for row in rows(app))

        # ONE further press, exactly as the row now promises.
        await pilot.press("escape")
        await pilot.pause()

        assert session.subagent_cancels == [
            "interrupted"
        ], "the row promised one more press and one more press did not escalate"


@pytest.mark.asyncio
async def test_a_buried_ladder_row_moves_to_where_the_user_is_looking() -> None:
    """Review round 3, R3-2. Restating in place must not repaint off-screen.

    `_stop_notice` outlives the turn that created it, so a later press can find
    it far up in scrollback. D5's unconditional in-place restate repainted it
    there and left the visible frame unchanged — the silent no-op D1 was filed
    for, reintroduced on a different axis. When the row is no longer the
    transcript's tail it is dropped and re-appended, which keeps both the
    replace-don't-stack promise and the guarantee that a press is visible.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 2

        await pilot.press("escape")
        await pilot.pause()
        first = app._stop_notice
        assert first is not None

        # The conversation carries on until the row is genuinely off screen —
        # asserted, not assumed, since a short transcript would not scroll and
        # the test would then be exercising the visible path by accident.
        for turn in range(30):
            app._append_block(UserBlock(f"another question {turn}"))
            prose = AssistantBlock()
            prose.update_text(f"another answer {turn} " + "padding " * 20)
            app._append_block(prose)
        await pilot.pause()
        await pilot.pause()
        assert not app._is_on_screen(first), "the row never scrolled out of sight"

        # A later press must produce a row the user can actually see.
        session.streaming = True
        await pilot.press("escape")
        await pilot.pause()
        # The append scrolls the transcript; let the layout settle before
        # asking where the row landed.
        for _ in range(5):
            await pilot.pause()

        blocks = app._transcript_view().blocks()
        assert blocks[-1] is app._stop_notice, "the ladder row did not move to the tail"
        assert first not in blocks, "the buried row was left behind as a duplicate"
        # And the new row is where the user is actually looking, which is the
        # whole point: a press that repaints only off-screen rows is a press
        # the user cannot tell from a dropped keystroke.
        moved = app._stop_notice
        assert moved is not None
        assert app._is_on_screen(moved), "the moved row is still not visible"


@pytest.mark.asyncio
async def test_the_expired_row_recounts_rather_than_restating_a_stale_number() -> None:
    """Review round 3, R3-3. The surviving row is present tense, so it must be
    presently true.

    The expiry used to restate the count captured when the offer was ARMED, in
    the calm receipt weight — so children that finished during the 4s window
    left the transcript asserting "2 subagents still running" as settled fact.
    Better than leaving an amber offer up forever, but one step short.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        session.running_children = 2

        await pilot.press("escape")
        await pilot.pause()
        assert any("2 subagents still running" in row for row in rows(app))

        # They finish on their own inside the window.
        session.running_children = 0
        await pilot.pause(DOUBLE_STOP_WINDOW_S + 0.5)

        painted = rows(app)
        assert not any(
            "still running" in row for row in painted
        ), "the expired row asserted running subagents that had already finished"


@pytest.mark.asyncio
async def test_a_stale_highlight_copies_but_cannot_arm_the_exit_ladder() -> None:
    """D17, design round 5. A stale range is a copy now — and ONLY a copy.

    This test used to assert that a stale highlight "does not divert Ctrl+C
    from the draft". The explicit-copy gesture changed the answer at the
    source: highlight-then-Ctrl+C IS the composer's copy now, so the press
    takes the range — deliberately, because a live range means the user can
    see what Ctrl+C will act on, which is the property the old selection
    gate lacked. What D17 actually protects survives unchanged: no abort,
    the draft untouched (it is not being scrapped, it is being quoted), the
    exit ladder NOT armed, and the reflexive second tap still not an exit.
    The draft rung this test used to exercise is covered on clean state by
    `test_ctrl_c_with_no_selection_never_reaches_the_interrupt` in
    `test_transcript_selection.py`.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "a half-typed prompt I do not want to lose"
        await pilot.pause()

        # A highlight with no gesture in flight: the caret simply sits inside a
        # range the user selected earlier.
        editor.selection = Selection((0, 0), (0, 10))
        await pilot.pause()
        assert editor.selected_text, "the stale highlight must be live to mean anything"
        assert not getattr(editor, "_selecting", False), "no drag should be in flight"

        await pilot.press("ctrl+c")
        await pilot.pause()

        # The copy takes the key, and nothing else moves: no abort, the draft
        # still there, and the ladder unarmed.
        assert session.aborts == [], "a stale highlight diverted the key to the interrupt"
        assert editor.text == "a half-typed prompt I do not want to lose", "the draft was touched"

        # And the second press must not quit, which is the damage D17 named.
        # The range is still live, so this press copies again — it must NOT
        # have started counting rungs.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running, "a second reflexive tap exited the app"
        assert session.aborts == [], "the second tap became an abort"


@pytest.mark.asyncio
async def test_moving_the_caret_after_a_copy_gives_ctrl_c_back_to_the_draft() -> None:
    """D20, design review round 6. The deferral must retire when the copy does.

    `_copied` alone retires only on an EDIT, so clicking elsewhere or pressing
    an arrow left it set with nothing on screen: the user saw a plain draft,
    got the interrupt instead, and the second reflexive tap quit with the draft
    lost. That is the D17 damage again, reached by swapping one invisible
    long-lived flag for another.

    The guard is therefore `_copied` AND a live highlight. The pair is
    self-retiring and, more to the point, VISIBLE — the key means "interrupt"
    exactly while the highlight the copy took is on screen, which is the same
    window in which the user perceives a copy at all.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        session.streaming = True
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "a half-typed prompt I do not want to lose"
        await pilot.pause()
        await pilot.pause()

        # A REAL explicit copy, driven through the widget's own key path
        # rather than by hand-setting flags. Round 19 (MAJOR-1) caught a
        # hand-set stand-in going vacuous when the predicate behind
        # `copy_in_flight` changed underneath it. The press claims no gesture
        # window at all (a highlight outlives the press, and deferring the
        # next Ctrl+C for that long is D17/D20); what this test pins is that
        # collapsing the highlight hands the key back to the draft.
        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 10))
        assert editor.selected_text, "the copy's own highlight must be on screen"
        assert editor._copied, "the receipt must be armed for this to mean anything"

        # Moving the caret collapses the highlight — the copy is over, and the
        # user can see that it is.
        await pilot.press("right")
        await pilot.pause()
        # The HIGHLIGHT is what this step is about, so it keeps its own direct
        # assertion: `copy_in_flight` is already False before the press (the
        # explicit copy claims no gesture window — see `Editor.action_copy`), so
        # it cannot witness the collapse on its own and would leave this step
        # asserting nothing about the caret move it describes (review round 1,
        # F1).
        assert not editor.selected_text, "the highlight should be gone"
        # The copy is not in flight either, so the key is back with the draft
        # rung; the RECEIPT flag survives, because the toast it drives is still
        # true (the text really is on the clipboard). Asserting both is what
        # keeps the two lifetimes from being fused again — doing so cost a stale
        # receipt once already (R18-1). `copy_in_flight` rather than a private
        # flag: it is the predicate the app's Ctrl+C rung actually consults, so
        # this stays a statement about the guarded behaviour instead of about a
        # field's name.
        assert not editor.copy_in_flight, "the copy should no longer divert the key"
        assert editor._copied, "the receipt is still true and must not be retired here"

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert session.aborts == [], "a retired copy still diverted the key"
        assert editor.text == "", "the draft was not cleared"
        assert "a half-typed prompt I do not want to lose" in editor.prompt_history()

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running, "the second tap quit with the draft lost"
