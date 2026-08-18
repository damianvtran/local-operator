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
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from rich.cells import cell_len
from textual.binding import Binding

from local_operator.harness.types import (
    ImageContent,
    ToolCallComposeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.resume import TRANSCRIPT_NAME
from local_operator.tui.app import DOUBLE_INTERRUPT_WINDOW_S, OperatorApp
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
    WorkingBlock,
)

from .test_app_pilot import FakeSession, _factory


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

    def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        self.steers.append(text)

    def set_approval_handler(self, handler: object | None) -> None:
        self.approval_handler = handler


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
    """
    for _ in range(100):
        if session.approval_handler is not None:
            return _approval_gate(session)
        await pilot.pause(0.02)
    raise AssertionError("the app never installed its approval handler")


def rows(app: OperatorApp) -> list[str]:
    """The painted frame as plain text, one entry per row."""
    return [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]


async def _submit(pilot: Any, app: OperatorApp, text: str) -> None:
    app.query_one(Editor).text = text
    await pilot.press("enter")
    await pilot.pause(0.1)


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
        await pilot.pause(0.25)
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
        await pilot.pause(0.25)

        editor = app.query_one(Editor)
        editor.text = "kept"
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert session.aborts == []
        assert editor.text == "kept"
        # Focus MUST stay in the composer. TextArea binds Escape to `blur`, which
        # made the first press move focus out of the input while looking like it
        # did nothing — every keystroke after it went nowhere.
        assert app.screen.focused is editor

        session.streaming = True
        await pilot.press("escape")
        await pilot.pause(0.1)
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
        await pilot.pause(0.25)
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
        await pilot.pause(0.3)

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
        ask = await _booted_gate(pilot, session)
        pending = asyncio.ensure_future(ask("bash", "run: rm -rf /tmp/x"))
        await pilot.pause(0.3)
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
        await pilot.pause(0.1)
        assert isinstance(app.screen.focused, Editor)  # focus handed back
        assert not app.query(ApprovalPrompt)  # the question is gone
        assert any("allowed" in row for row in rows(app))  # decision kept


@pytest.mark.asyncio
async def test_n_denies_one_tool_and_lets_the_turn_continue() -> None:
    """``n`` is the per-call refusal: this tool is refused, the turn carries on."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        ask = await _booted_gate(pilot, session)
        session.streaming = True
        pending = asyncio.ensure_future(ask("write", "write: /etc/hosts"))
        await pilot.pause(0.3)

        await pilot.press("n")
        assert await asyncio.wait_for(pending, 2) is False
        assert session.aborts == []  # only the tool was refused
        assert any("denied" in row for row in rows(app))


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
        ask = await _booted_gate(pilot, session)
        session.streaming = True
        first = asyncio.ensure_future(ask("write", "write: /etc/hosts"))
        second = asyncio.ensure_future(ask("bash", "run: rm -rf /"))
        await pilot.pause(0.3)

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
        ask = await _booted_gate(pilot, session)
        first = asyncio.ensure_future(ask("bash", "run: make"))
        await pilot.pause(0.3)
        await pilot.press("A")
        assert await asyncio.wait_for(first, 2) is True
        await pilot.pause(0.1)

        # No prompt is mounted for the second ask at all.
        before = len(app.query(ApprovalPrompt))
        second = asyncio.ensure_future(ask("write", "write: out.txt"))
        assert await asyncio.wait_for(second, 2) is True
        await pilot.pause(0.1)
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
        ask = await _booted_gate(pilot, session)
        session.streaming = True
        pending = asyncio.ensure_future(ask("bash", "run: sleep 99"))
        await pilot.pause(0.3)

        app.action_interrupt()
        assert await asyncio.wait_for(pending, 2) is False
        assert session.aborts == ["interrupted"]


@pytest.mark.asyncio
async def test_clearing_the_transcript_settles_a_pending_approval() -> None:
    """The prompt's widget is about to be removed; the awaiting turn is denied."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        ask = await _booted_gate(pilot, session)
        pending = asyncio.ensure_future(ask("bash", "run: make"))
        await pilot.pause(0.3)

        app.query_one(TranscriptView).clear_blocks()
        await pilot.pause(0.1)
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
        await pilot.pause(0.25)
        session.streaming = True  # a stop IS available, so precedence is observable

        editor = app.query_one(Editor)
        editor.text = "/"
        await pilot.pause(0.2)
        assert editor.picker.is_open()

        await pilot.press("escape")
        await pilot.pause(0.1)
        assert not editor.picker.is_open()  # the list closed…
        assert session.aborts == []  # …and the turn was NOT stopped

        # The very next Esc stops the turn — no dead press in between, and focus
        # never leaves the composer.
        await pilot.press("escape")
        await pilot.pause(0.1)
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
        ask = await _booted_gate(pilot, session)
        session.streaming = True
        first = asyncio.ensure_future(ask("bash", "run: one"))
        second = asyncio.ensure_future(ask("bash", "run: two"))
        await pilot.pause(0.3)
        assert len(app.query(ApprovalPrompt)) == 1  # serialized, not stacked

        app.action_interrupt()
        assert await asyncio.wait_for(first, 2) is False
        assert await asyncio.wait_for(second, 2) is False
        await pilot.pause(0.2)
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
        ask = await _booted_gate(pilot, session)
        first = asyncio.ensure_future(ask("bash", "run: one"))
        second = asyncio.ensure_future(ask("write", "write: two"))
        await pilot.pause(0.3)

        await pilot.press("A")
        assert await asyncio.wait_for(first, 2) is True
        assert await asyncio.wait_for(second, 2) is True  # never asked again
        await pilot.pause(0.2)
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
        ask = await _booted_gate(pilot, session)
        pending = asyncio.ensure_future(ask("bash", "run: one"))
        await pilot.pause(0.3)

        await pilot.press("a")
        await pilot.pause(0.2)
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
        await pilot.pause(0.25)
        app._run_slash_command("/approvals auto")
        await pilot.pause(0.1)
        assert app._approve_all is True
        assert any("auto-approve" in row for row in rows(app))  # band says so

        app._run_slash_command("/approvals ask")
        await pilot.pause(0.1)
        assert app._approve_all is False
        assert not any("auto-approve" in row for row in rows(app))
        # And the gate really asks again.
        pending = asyncio.ensure_future(_approval_gate(session)("bash", "run: x"))
        await pilot.pause(0.3)
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
        await pilot.pause(0.25)
        session.streaming = True

        await pilot.press("ctrl+c")
        await pilot.pause(0.1)
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
        await pilot.pause(0.3)
        assert not app.is_running

    assert app.resume_hint() == "local-operator --resume sess"


@pytest.mark.asyncio
async def test_a_slow_second_ctrl_c_is_a_fresh_interrupt_not_an_exit() -> None:
    """Two interrupts a minute apart must not quit the app.

    The window is what keeps the gesture a deliberate double-tap; without it, a
    user interrupting two turns in a row would exit by accident.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        session.streaming = True

        await pilot.press("ctrl+c")
        await pilot.pause(0.1)
        # Age the first press past the window without waiting for it.
        app._last_interrupt_at -= DOUBLE_INTERRUPT_WINDOW_S + 1
        await pilot.press("ctrl+c")
        await pilot.pause(0.2)
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
        await pilot.pause(0.25)
        await _submit(pilot, app, "say something")
        app.post_message(TurnStarted())
        app.post_message(AssistantMessageStart())
        app.post_message(AssistantDelta("hello from the model"))
        await pilot.pause(0.2)
        assert any("hello from the model" in row for row in rows(app))

        app.post_message(AssistantMessageEnd(""))
        await pilot.pause(0.2)
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
        ask = await _booted_gate(pilot, session)
        session.streaming = True

        first = asyncio.ensure_future(ask("bash", "run: one"))
        await pilot.pause(0.3)
        app.query_one(TranscriptView).clear_blocks()
        await pilot.pause(0.2)
        assert await asyncio.wait_for(first, 2) is False  # its widget is gone
        assert session.aborts == []  # the turn was NOT stopped

        # The next tool of the same run still gets to ask.
        second = asyncio.ensure_future(ask("write", "write: two"))
        await pilot.pause(0.3)
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
        await pilot.pause(0.25)
        session.streaming = True
        app._deny_queued_approvals()  # what a stop leaves behind
        assert app._approvals_are_denied(app._turn_epoch) is True

        app._run_slash_command("/approvals ask")
        await pilot.pause(0.1)
        assert app._approvals_are_denied(app._turn_epoch) is False
        pending = asyncio.ensure_future(_approval_gate(session)("bash", "run: x"))
        await pilot.pause(0.3)
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
        await pilot.pause(0.25)
        # FakeSession's id has no directory on disk, so there is nothing to offer.
        assert app.resume_hint() == ""
        await pilot.press("ctrl+c")
        await pilot.pause(0.1)
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
        await pilot.pause(0.2)
        view = app.query_one(TranscriptView)
        view.append_block(
            ApprovalBlock("write_file", "[outside workspace] write: /Users/x/deep/config.yml")
        )
        await pilot.pause(0.2)
        painted = [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]

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
        await pilot.pause(0.2)
        block = ApprovalBlock("write_file", "[outside workspace] write: /Users/x/deep/config.yml")
        app.query_one(TranscriptView).append_block(block)
        await pilot.pause(0.1)
        block.resolve(True, answer="y")
        await pilot.pause(0.2)
        painted = [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]

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
        await pilot.pause(0.2)
        app.query_one(TranscriptView).append_block(ApprovalBlock("bash", "run: make"))
        await pilot.pause(0.2)
        painted = [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]
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
        await pilot.pause(0.2)
        view = app.query_one(TranscriptView)
        view.append_block(ApprovalBlock("write_file", f"{OUTSIDE_MARKER} write: /tmp/x/keys"))
        view.append_block(ApprovalBlock("write_file", "write: /tmp/x/keys"))
        await pilot.pause(0.2)
        painted = [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]
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
        await pilot.pause(0.2)
        app.query_one(TranscriptView).append_block(ApprovalBlock("write_file", detail))
        await pilot.pause(0.2)
        painted = [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]
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
        await pilot.pause(0.25)
        session.streaming = True
        first = asyncio.ensure_future(_approval_gate(session)("bash", "run: one"))
        await pilot.pause(0.25)
        second = asyncio.ensure_future(_approval_gate(session)("execute", "run: rm -rf /"))
        await pilot.pause(0.15)

        # The race: a turn boundary is already queued when the stop lands.
        app.post_message(TurnStarted())
        app.action_interrupt()
        await pilot.pause(0.4)

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
        await _booted_gate(pilot, session)
        session.streaming = True
        pending = asyncio.ensure_future(_approval_gate(session)("bash", "run: make"))
        await pilot.pause(0.3)
        assert app.query(ApprovalPrompt)

        app._run_slash_command("/approvals auto")
        await pilot.pause(0.2)
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
        await pilot.pause(0.2)
        block = ApprovalBlock("bash", f"run: {payload}")
        app.query_one(TranscriptView).append_block(block)
        await pilot.pause(0.2)

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
        await pilot.pause(0.25)
        view = app.query_one(TranscriptView)
        view.append_block(ToolCard("t1", "bash", {}, ""))
        await pilot.pause(0.1)
        settled = view.tool_name_col

        block = ApprovalBlock("mcp__linear_create_initiative", "run: x")
        view.append_block(block)
        await pilot.pause(0.1)
        assert view.tool_name_col == settled

        block.resolve(False, answer="n")
        await pilot.pause(0.1)
        assert view.tool_name_col == settled


@pytest.mark.asyncio
async def test_clearing_the_transcript_forgets_the_ledger_width() -> None:
    """A derived measurement must not outlive the blocks it was derived from."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.25)
        view = app.query_one(TranscriptView)
        view.append_block(ToolCard("t1", "mcp__linear_create_initiative", {}, ""))
        await pilot.pause(0.1)
        widened = view.tool_name_col
        assert widened > 8

        view.clear_blocks()
        await pilot.pause(0.1)
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
        await pilot.pause(0.25)
        view = app.query_one(TranscriptView)
        short = NoticeBlock("resumed session 4c1f", "info")
        tall = NoticeBlock(
            "tool approvals: auto - /approvals ask restores prompting for the rest "
            "of this session, including write and command tools",
            "warning",
        )
        view.append_block(short)
        view.append_block(tall)
        await pilot.pause(0.3)
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
        await pilot.pause(0.2)
        app.post_message(ToolComposing(ToolCallComposeEvent(tool_call_id="c1", tool_name="write")))
        await pilot.pause(0.1)
        painted = [strip.text for strip in app.screen._compositor.render_strips()]
        assert [row for row in painted if "composing" in row]
        # No size until there is one: a `0 B` that never moves reads as stuck.
        assert not [row for row in painted if "0 B" in row]

        app.post_message(
            ToolComposing(
                ToolCallComposeEvent(tool_call_id="c1", tool_name="write", argument_bytes=14079)
            )
        )
        await pilot.pause(0.1)
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
        await pilot.pause(0.1)
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
        await pilot.pause(0.2)
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
        await pilot.pause(0.1)
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
        await pilot.pause(0.2)
        app.post_message(
            ToolComposing(
                ToolCallComposeEvent(tool_call_id="c1", tool_name="write", argument_bytes=12700)
            )
        )
        await pilot.pause(0.1)
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
        view = app.query_one(TranscriptView)
        settled = ToolCard("s1", "read", {}, None)
        view.append_block(settled)
        settled.mark_done("read a file")
        await pilot.pause(0.1)

        def settled_row() -> str:
            return next(
                (s.text for s in app.screen._compositor.render_strips() if "read a file" in s.text),
                "",
            )

        before = settled_row()
        app.post_message(ToolComposing(ToolCallComposeEvent(tool_call_id="c1", tool_name="mcp")))
        await pilot.pause(0.15)
        first = next(
            (s.text for s in app.screen._compositor.render_strips() if "composing" in s.text), ""
        )
        app.post_message(
            ToolComposing(ToolCallComposeEvent(tool_call_id="c1", tool_name="mcp__" + "z" * 200))
        )
        await pilot.pause(0.15)
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
            view = app.query_one(TranscriptView)
            view.append_block(ApprovalBlock("write_file", f"{OUTSIDE_MARKER} write: /etc/hosts"))
            await pilot.pause(0.05)
            row = next(
                (
                    strip.text
                    for strip in app.screen._compositor.render_strips()
                    if "write_file" in strip.text
                ),
                "",
            )
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
            app.query_one(TranscriptView).append_block(ApprovalBlock("write_file", detail))
            await pilot.pause(0.1)
            row = next(
                (s.text for s in app.screen._compositor.render_strips() if "write_file" in s.text),
                "",
            )
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
            app.query_one(TranscriptView).append_block(ApprovalBlock("write_file", detail))
            await pilot.pause(0.05)
            row = next(
                (s.text for s in app.screen._compositor.render_strips() if "write_file" in s.text),
                "",
            )
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
    """A question arriving mid-sentence must not cost the user their draft.

    This is the risk the anchored prompt takes on. It grabs focus so its answer
    keys work — the whole point of the rework — and it does so while the user
    may be in the middle of typing, since an approval is raised by the agent
    rather than by them. Two things therefore have to hold: the composer's text
    survives untouched, and focus comes back to it once the question is
    answered, so the sentence can be finished where it was left.

    The old prompt got the first right and the second wrong in the opposite
    direction: it handed focus AWAY on any printable key, which is what made
    its own advertised answer keys stop working.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        ask = await _booted_gate(pilot, session)
        editor = app.query_one(Editor)
        editor.focus()
        editor.load_text("please clean up the stale rows and then")
        await pilot.pause(0.1)

        pending = asyncio.ensure_future(ask("bash", "run: rm -rf ./build"))
        # Waited on the CONDITION rather than a duration. A fixed
        # `pause(0.3)` is a bet on how long the mount takes, and CI lost it on
        # the slower of the two Python versions: the prompt had not taken focus
        # yet and the assertion below read the composer.
        for _ in range(100):
            if isinstance(app.screen.focused, ApprovalPrompt):
                break
            await pilot.pause(0.02)
        # The question took focus (or its keys would type into the draft)...
        assert isinstance(app.screen.focused, ApprovalPrompt)
        # ...and left the draft exactly as it was.
        assert editor.text == "please clean up the stale rows and then"

        await pilot.press("y")
        assert await asyncio.wait_for(pending, 2) is True
        # Same reason as above: the hand-back happens on the way out of the
        # gate, which is a mount/unmount round trip rather than a synchronous
        # step, so the wait is on the condition.
        for _ in range(100):
            if isinstance(app.screen.focused, Editor):
                break
            await pilot.pause(0.02)
        # Focus is handed back, and the sentence is still there to finish.
        assert isinstance(app.screen.focused, Editor)
        assert app.query_one(Editor).text == "please clean up the stale rows and then"
