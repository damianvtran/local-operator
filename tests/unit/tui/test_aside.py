"""The ``/btw`` aside — the overlay, the composer it borrows, and its contract.

The contract these tests exist to pin, because everything else about the
feature is downstream of it:

    An aside READS the conversation and never writes to it. The question and
    the answer reach the model for exactly one request; they never enter the
    transcript and never enter the model's context for the main conversation.
    Esc discards the exchange and hands the user's draft back. ``^f`` is the
    only door out, and it is the user's.

``test_an_aside_exchange_leaves_no_trace_in_the_conversation`` and
``test_forking_is_the_only_way_an_aside_reaches_the_conversation`` are the two
halves of that sentence. A change that makes either fail is a change to what
``/btw`` means, not a broken test.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any
from unittest.mock import patch

import pytest

from local_operator.harness.types import Message, Usage
from local_operator.tui.app import RESIZE_REFIT_DELAY_S, OperatorApp
from local_operator.tui.widgets import aside_panel as aside_panel_module
from local_operator.tui.widgets.aside_panel import (
    ASIDE_COPY_KEY,
    PANEL_HEIGHT_MARGIN,
    AsidePanel,
    AsideTurn,
)
from local_operator.tui.widgets.editor import (
    ASIDE_PLACEHOLDER,
    DEFAULT_PLACEHOLDER,
    SHELL_PLACEHOLDER,
    Editor,
)
from local_operator.tui.widgets.transcript import TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class AsideSession(FakeSession):
    """A fake session that can answer an aside and adopt one.

    Records what the aside asked (so a test can prove the live conversation was
    handed to the model) and what, if anything, it later wrote back.
    """

    def __init__(self, answer: str = "because the file is generated.") -> None:
        super().__init__()
        self.answer = answer
        self.aside_calls: list[list[Any]] = []
        #: What `^f` promoted into the conversation. NOT ``adopted`` — the base
        #: `FakeSession` already has one, typed as a list of message LISTS, and
        #: a mutable attribute cannot be narrowed in a subclass. The name is
        #: also the better one: this is what forking produced.
        self.forked: list[Message] = []
        self.streaming = False
        self.fail: str | None = None
        self._history = [Message.user("port the loop"), Message.assistant("done.")]

    @property
    def is_streaming(self) -> bool:
        return self.streaming

    async def complete_aside(self, turns, *, on_delta=None, on_usage=None) -> str:  # noqa: ANN001
        self.aside_calls.append(list(turns))
        if self.fail is not None:
            raise RuntimeError(self.fail)
        if on_delta is not None:
            on_delta(self.answer)
        if on_usage is not None:
            on_usage(Usage(input_tokens=100, output_tokens=20))
        return self.answer

    async def adopt_aside(self, messages) -> None:  # noqa: ANN001
        if self.streaming:
            raise RuntimeError("cannot adopt an aside while a turn is running")
        self.forked.extend(messages)
        self._history.extend(messages)


async def _settle_boot(pilot, app: OperatorApp, session: FakeSession) -> None:
    """Wait for the boot worker to ADOPT the session and paint its history.

    Boot is a worker; one ``pause()`` is a bet that it has finished, and on a
    loaded CI runner it loses (~850 ms seen for the first app in a process).
    A test that reads the transcript's block count before adoption sees 0,
    then blames the aside for the 2 rows the replayed history paints. Wait on
    the adoption edge the way ``test_app_pilot`` does, never on the clock.
    """
    for _ in range(200):
        await pilot.pause()
        if app._session is session and len(app.query_one(TranscriptView).blocks()) >= len(
            session.history()
        ):
            return
    raise AssertionError("the boot worker never adopted the session")


async def _open_with_question(pilot, app: OperatorApp, question: str) -> AsidePanel:
    """Type ``/btw <question>`` the way a user does, and settle the answer."""
    app.query_one(Editor).focus()
    await pilot.pause()
    app.query_one(Editor).load_text(f"/btw {question}")
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()
    return app.query_one(AsidePanel)


# -- opening ---------------------------------------------------------------
@pytest.mark.asyncio
async def test_btw_opens_the_overlay_over_the_conversation() -> None:
    """``/btw`` raises the card without disturbing the chat underneath it."""
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("h", "i", "enter")
        await pilot.pause()
        before = len(app.query_one(TranscriptView).blocks())

        panel = await _open_with_question(pilot, app, "why sed and not edit?")

        assert panel.is_open
        # The transcript is still the transcript: same blocks, still displayed.
        assert len(app.query_one(TranscriptView).blocks()) == before
        assert app.query_one(TranscriptView).display
        rendered = "\n".join(panel.render_lines_for_test())
        assert "off the record" in rendered
        assert "why sed and not edit?" in rendered
        assert session.answer in rendered


@pytest.mark.asyncio
async def test_bare_btw_opens_an_empty_aside_and_waits() -> None:
    """No question on the line is not an error — it is the other half of the form."""
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        app.query_one(Editor).load_text("/btw")
        await pilot.press("enter")
        await pilot.pause()

        panel = app.query_one(AsidePanel)
        assert panel.is_open
        assert panel.turns == []
        assert session.aside_calls == []
        assert app.query_one(Editor).placeholder == ASIDE_PLACEHOLDER


@pytest.mark.asyncio
async def test_the_aside_answers_from_the_live_conversation() -> None:
    """The model is handed the session's context, not a bare question."""
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _open_with_question(pilot, app, "what did I ask you to do?")

        # `complete_aside` appends these to the live context itself, so what
        # the app must supply is the question — wrapped in the off-the-record
        # framing, which is what stops the model treating it as a new task.
        assert len(session.aside_calls) == 1
        sent = session.aside_calls[0][-1]
        assert sent.role == "user"
        assert "what did I ask you to do?" in sent.text
        assert "OFF" in sent.text and "RECORD" in sent.text


@pytest.mark.asyncio
async def test_closing_the_aside_returns_the_composer_to_bang_mode() -> None:
    """The aside borrows the composer and disables bang-mode for the life of
    the card, but the half-typed command comes back IN the mode it was typed
    in — a `echo hi` returned to a composer that silently left bang-mode
    would arm Enter with a shell command aimed at the model."""
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.press("!")
        for key in "echo hi":
            await pilot.press("space" if key == " " else key)
        assert editor.shell_mode is True
        await pilot.press("f8")
        await pilot.pause()
        assert editor.shell_mode is False
        await pilot.press("escape")  # the user's door out of the card
        await pilot.pause()
        assert editor.shell_mode is True
        assert editor.text == "echo hi"
        assert editor.placeholder == SHELL_PLACEHOLDER


@pytest.mark.asyncio
async def test_bang_inside_the_aside_is_the_start_of_a_question() -> None:
    """Bang-mode is a main-chat gesture. A ``!`` typed into the aside is
    the first character of a question, not a command — the card promised
    off the record, and eating the bang would both break that and run a
    shell command the user never asked the conversation to see."""
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _open_with_question(pilot, app, "why?")
        editor.clear_content()
        await pilot.press("!")
        await pilot.pause()
        assert editor.shell_mode is False
        assert editor.text == "!"
        assert editor.placeholder == ASIDE_PLACEHOLDER


# -- the contract ----------------------------------------------------------
@pytest.mark.asyncio
async def test_an_aside_exchange_leaves_no_trace_in_the_conversation() -> None:
    """THE contract. Ask, answer, dismiss: the conversation is untouched.

    Untouched on both surfaces that matter — the visible transcript AND the
    history the model is replayed. If this ever fails, ``/btw`` has stopped
    being an aside and become a prompt that hides itself behind a popup.
    """
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("h", "i", "enter")
        await pilot.pause()
        blocks_before = len(app.query_one(TranscriptView).blocks())
        history_before = list(session.history())

        panel = await _open_with_question(pilot, app, "are the subagents stuck?")
        # A second turn INSIDE the aside, then dismissal.
        app.query_one(Editor).load_text("and the first one?")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert len(panel.turns) == 2
        await pilot.press("escape")
        await pilot.pause()

        assert len(app.query_one(TranscriptView).blocks()) == blocks_before
        assert session.history() == history_before
        assert session.prompts == ["hi"]
        assert session.forked == []


@pytest.mark.asyncio
async def test_a_turn_taken_inside_the_aside_stays_inside_the_aside() -> None:
    """Enter while the card is up asks the aside, never the conversation."""
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _settle_boot(pilot, app, session)
        # The fake boots with a replayed history, so the transcript is not
        # empty — the claim is that the aside adds nothing to it.
        before = len(app.query_one(TranscriptView).blocks())
        assert before == len(session.history()), "read before the history was painted"
        panel = await _open_with_question(pilot, app, "first?")

        app.query_one(Editor).load_text("second?")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert [turn.question for turn in panel.turns] == ["first?", "second?"]
        assert session.prompts == []
        assert len(app.query_one(TranscriptView).blocks()) == before
        # The follow-up carries the earlier exchange, so "and why is that?"
        # resolves against what was already said HERE.
        follow_up = session.aside_calls[-1]
        assert any(m.role == "user" and m.text == "first?" for m in follow_up)
        assert any(m.role == "assistant" and m.text == session.answer for m in follow_up)


@pytest.mark.asyncio
async def test_an_aside_over_a_live_turn_carries_the_sentence_on_screen() -> None:
    """ "What are you doing right now?" is the question this surface exists for.

    The loop appends an assistant message to the context only once it settles,
    so mid-turn the sentence the user can SEE is the one thing missing from
    what the model would be shown. A rendered frame caught this path handing
    the model a bound method instead (``AssistantBlock.text`` is a method, not
    a property), which no other test reached.
    """
    session = AsideSession()
    session.streaming = True
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._ensure_streaming_block().update_text("Shifting the ingress weight to 25% now")
        await pilot.pause()

        panel = await _open_with_question(pilot, app, "what are you doing right now?")

        sent = session.aside_calls[-1]
        assert any(
            m.role == "assistant" and m.text == "Shifting the ingress weight to 25% now"
            for m in sent
        )
        assert panel.turns[-1].state == "done"
        # Forking is refused mid-turn, so the card must not advertise ^f.
        assert "^f" not in panel.render_lines_for_test()[-1]


@pytest.mark.asyncio
async def test_a_slash_command_typed_inside_the_aside_is_a_question() -> None:
    """The card is a MODE: while it is up the composer belongs to it."""
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _settle_boot(pilot, app, session)
        before = len(app.query_one(TranscriptView).blocks())
        panel = await _open_with_question(pilot, app, "first?")

        app.query_one(Editor).load_text("/usage")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert [turn.question for turn in panel.turns] == ["first?", "/usage"]
        assert len(app.query_one(TranscriptView).blocks()) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (60, 24)])
async def test_the_card_shares_the_composer_s_column(size) -> None:  # noqa: ANN001
    """The card and the composer are ONE unit, so they share a column.

    The aside's input IS the composer directly below it. Drawn as a narrow
    centred card over a full-width dock, they read as a floating dialog and an
    unrelated bar \u2014 the relationship stated backwards. Same left edge, same
    right edge, resting on it with no gap, is what says otherwise.
    """
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press("f8")
        await pilot.pause()
        await pilot.pause()

        card = app.query_one(AsidePanel).region
        shell = app.query_one("#input-shell").region
        assert (card.x, card.right) == (shell.x, shell.right)
        assert card.bottom == shell.y, "the card rests ON the composer, with no gap"


@pytest.mark.asyncio
async def test_width_parity_survives_a_live_resize() -> None:
    """A settled frame differs from a first paint, and this card is placed twice.

    Textual sends no resize event to a card in a ``width: auto`` host, and the
    dock's re-arrange lands after the refresh callbacks — so the card
    re-measures on a short timer (``RESIZE_REFIT_DELAY_S``).

    POLLED to a deadline rather than slept past once. The claim is that parity
    is restored, not that it is restored within some exact number of
    milliseconds; a fixed sleep asserts the second and goes red under a loaded
    suite while the product is fine.
    """
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("f8")
        await pilot.pause()
        await pilot.resize_terminal(60, 24)

        deadline = time.monotonic() + 5.0
        while True:
            await pilot.pause()
            card = app.query_one(AsidePanel).region
            shell = app.query_one("#input-shell").region
            if (card.x, card.right) == (shell.x, shell.right):
                break
            assert (
                time.monotonic() < deadline
            ), f"the card never re-measured: card={card} shell={shell}"
            await asyncio.sleep(RESIZE_REFIT_DELAY_S)


@pytest.mark.asyncio
async def test_the_card_rests_on_the_end_of_the_chat_not_on_top_of_it() -> None:
    """The card covers the NEWEST turns, which is the context it asks about.

    Receding the conversation says nothing about the part the card hides, and
    at a short terminal what it hid was the last thing the user asked. The
    transcript gives up the card's rows instead, so the card rests on the end
    of the chat, and takes them back on esc.

    "Takes them back" is asserted as the absence of the INLINE rule rather than
    as a number. The sheet's own bottom padding is the conversation's trailing
    row of ground and it differs per layout (1 in the conversation, 0 under the
    boot splash — see `TranscriptView` in the tcss), so a constant here would
    pin the wrong layout's answer and, before that row existed, silently passed
    while the close path wrote a hardcoded 0 over the sheet.
    """
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 24)) as pilot:
        await pilot.pause()
        transcript = app._transcript_view()
        assert not transcript.styles.inline.has_rule("padding")

        await _open_with_question(pilot, app, "are the subagents stuck?")
        card = app.query_one(AsidePanel)
        assert transcript.styles.padding.bottom == card.region.height > 0

        await pilot.press("escape")
        await pilot.pause()
        assert not transcript.styles.inline.has_rule("padding")


@pytest.mark.asyncio
async def test_an_aside_question_never_enters_the_prompt_history() -> None:
    """ "esc discards it" has to be true of UP as well as of the transcript.

    ``Editor._submit`` records history BEFORE it posts, so the aside had to
    borrow the recording off rather than unwind it afterwards. Reproduced
    before the fix: after Esc the question was in history, UP recalled it, and
    the next Enter sent it to the agent as a real turn — on a card whose title
    says the opposite. It also buried the last thing the user really said.
    """
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        editor = app.query_one(Editor)
        await pilot.pause()
        editor.focus()
        await pilot.pause()
        await pilot.press("h", "i", "enter")
        await pilot.pause()
        assert editor.prompt_history() == ["hi"]

        await _open_with_question(pilot, app, "are the subagents stuck?")
        editor.load_text("and the first one?")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        # Neither question is recallable, and UP still returns the last REAL
        # prompt rather than an aside the user was told had been discarded.
        assert editor.prompt_history() == ["hi"]
        # And the composer records again the moment the aside gives it back.
        editor.load_text("carry on")
        await pilot.press("enter")
        await pilot.pause()
        assert editor.prompt_history() == ["hi", "carry on"]


@pytest.mark.asyncio
async def test_ctrl_c_dismisses_the_aside_so_its_own_warning_is_readable() -> None:
    """The exit hint is the only warning before a second press quits the app.

    Appended behind a card that floats one elevation step over the transcript,
    it is drawn where it cannot be read — the same failure the fork refusals
    were moved onto the card to avoid. Ctrl+C keeps its meaning; it stops doing
    it invisibly.
    """
    session = AsideSession()
    session.streaming = True
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _open_with_question(pilot, app, "what are you doing?")

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert not app.query_one(AsidePanel).is_open
        assert session.aborts == ["interrupted"]
        # No inline reservation left: the sheet owns the resting row again (see
        # `test_the_card_rests_on_the_end_of_the_chat_not_on_top_of_it`).
        assert not app._transcript_view().styles.inline.has_rule("padding")


@pytest.mark.asyncio
@pytest.mark.parametrize("act", ["clear", "approval"])
async def test_the_aside_yields_to_anything_that_acts_on_the_conversation(
    act: str,
) -> None:
    """A floating card must not outlive the conversation it is a question about.

    ``ctrl+l`` wipes the ledger the aside is asking about; a tool approval is a
    transcript block that would mount behind the card while taking focus off
    the composer the card is pointed at. Both are the general form of the
    ctrl+c case, and both already yield the full-page subagent view.
    """
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _open_with_question(pilot, app, "why?")
        assert app.query_one(AsidePanel).is_open

        if act == "clear":
            app.action_clear_transcript()
        else:
            app.run_worker(app.request_tool_approval("bash", "rm -rf"), thread=False)
        await pilot.pause()
        await pilot.pause()

        assert not app.query_one(AsidePanel).is_open
        editor = app.query_one(Editor)
        assert editor.placeholder == DEFAULT_PLACEHOLDER
        assert not app._transcript_view().styles.inline.has_rule("padding")


# -- leaving ---------------------------------------------------------------
@pytest.mark.asyncio
async def test_escape_restores_the_main_chat_and_its_half_typed_prompt() -> None:
    """F8 mid-draft, ask, Esc: the draft is back and the chat is intact.

    The draft is what makes the aside a place you STEP INTO. Losing it would
    make the feature cost a retype every time, which is exactly the tax that
    stops anyone from using a side channel at all.
    """
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.load_text("port the compaction gate to")
        await pilot.press("f8")
        await pilot.pause()

        assert app.query_one(AsidePanel).is_open
        assert editor.text == ""
        assert editor.placeholder == ASIDE_PLACEHOLDER

        editor.load_text("does compaction run mid-turn?")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        # Half-typed text in the aside at the moment Esc lands: it is a
        # question the user decided not to ask, and it must not be carried
        # into the main composer where Enter would send it to the agent.
        editor.load_text("and does it")
        await pilot.press("escape")
        await pilot.pause()

        assert not app.query_one(AsidePanel).is_open
        assert editor.text == "port the compaction gate to"
        assert editor.placeholder == DEFAULT_PLACEHOLDER
        assert session.prompts == []


@pytest.mark.asyncio
async def test_escape_dismisses_the_aside_before_it_stops_the_turn() -> None:
    """A card promising ``esc close`` must not abort the agent instead."""
    session = AsideSession()
    session.streaming = True
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _open_with_question(pilot, app, "what are you doing?")

        await pilot.press("escape")
        await pilot.pause()

        assert not app.query_one(AsidePanel).is_open
        assert session.aborts == []
        # And the NEXT Esc is the stop it always was.
        await pilot.press("escape")
        await pilot.pause()
        assert session.aborts == ["interrupted"]


@pytest.mark.asyncio
async def test_reopening_the_aside_opens_an_empty_one() -> None:
    """Dismiss means discard. A card that says "off the record" cannot hoard."""
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = await _open_with_question(pilot, app, "first?")
        assert len(panel.turns) == 1
        await pilot.press("escape")
        await pilot.pause()

        await pilot.press("f8")
        await pilot.pause()

        assert app.query_one(AsidePanel).is_open
        assert app.query_one(AsidePanel).turns == []


# -- forking ---------------------------------------------------------------
@pytest.mark.asyncio
async def test_forking_is_the_only_way_an_aside_reaches_the_conversation() -> None:
    """``^f`` promotes the exchange as ordinary turns, and closes the card.

    The other half of the contract. The question is adopted VERBATIM — the
    off-the-record wrapper is scaffolding for one request, and a transcript
    carrying it would replay "this never happened" forever.
    """
    session = AsideSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _open_with_question(pilot, app, "why sed and not edit?")

        await pilot.press("ctrl+f")
        await pilot.pause()
        await pilot.pause()

        assert not app.query_one(AsidePanel).is_open
        assert [(m.role, m.text) for m in session.forked] == [
            ("user", "why sed and not edit?"),
            ("assistant", session.answer),
        ]
        # And it is on screen as a real exchange, not as a notice about one.
        transcript = app.query_one(TranscriptView)
        assert len(transcript.blocks()) >= 2


@pytest.mark.asyncio
async def test_forking_is_refused_while_a_turn_is_running() -> None:
    """The loop owns the message list mid-turn; a splice makes it unsendable."""
    session = AsideSession()
    session.streaming = True
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _open_with_question(pilot, app, "what are you doing?")

        await pilot.press("ctrl+f")
        await pilot.pause()

        assert session.forked == []
        assert app.query_one(AsidePanel).is_open


# -- the card itself -------------------------------------------------------
def test_the_card_states_its_contract_and_its_exit() -> None:
    """Both facts a user needs about a popup that talks to the agent."""
    panel = AsidePanel()
    panel.display = True
    panel._turns = [AsideTurn(question="q?", answer="a.", state="done")]
    rendered = panel.render_lines_for_test()

    assert "off the record" in rendered[0]
    assert "esc" in rendered[-1]


def test_the_fork_key_is_advertised_only_when_it_would_work() -> None:
    """A footer offering a key that fails is worse than one that omits it."""
    panel = AsidePanel()
    panel.display = True
    panel._turns = [AsideTurn(question="q?", state="running")]
    assert "^f" not in panel.render_lines_for_test()[-1]

    panel._turns = [AsideTurn(question="q?", answer="a.", state="done")]
    panel.set_fork_available(True)
    assert "^f" in panel.render_lines_for_test()[-1]


def test_a_failed_aside_is_never_forkable() -> None:
    """Forking is "keep this exchange", and a failure is not one."""
    panel = AsidePanel()
    panel.display = True
    generation = panel.ask("q?")
    panel.fail_answer(generation, "provider exploded")

    assert panel.fork_messages() == []
    assert "provider exploded" in "\n".join(panel.render_lines_for_test())


def test_a_long_exchange_pins_the_owning_question_and_counts_questions() -> None:
    """The newest rows are what is being read; the rest announces itself.

    The cut is by ROW — cutting on turn boundaries made a turn the smallest
    addressable unit, so an answer taller than the card had a middle no gesture
    could reach. What the boundary cut was protecting is kept: the window pins
    the question that OWNS its rows, so the card never opens on a mid-sentence
    continuation at the answer indent with no question above it, which reads as
    the start of a new answer.

    Counted in questions here, because whole questions are above the window and
    a user remembers asking three things without counting the rows they wrapped
    to. Inside one answer there is no question to count and the marker switches
    to lines; that is
    ``test_one_long_answer_states_the_rows_it_withheld``.

    The answer is a LIST because the answer renders as markdown: twelve
    repetitions of a bare word are one paragraph to a markdown parser, which
    folds to a single row and leaves nothing to overflow.
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [
        AsideTurn(question=f"question {index}?", answer="- line\n" * 12, state="done")
        for index in range(6)
    ]
    rendered = panel.render_lines_for_test()

    assert any("↑ 4 earlier questions · scroll" in line for line in rendered)
    # The newest question and its whole answer are on screen, which is where a
    # reader who has not scrolled should be.
    assert any(line.startswith("▌ question 5?") for line in rendered)

    # Scrolled back INTO question 4's answer, that question is pinned above the
    # fragment rather than leaving it unattributed.
    panel._scroll_by(8)
    rendered = panel.render_lines_for_test()
    assert rendered[3].startswith("▌ question 4?")
    assert rendered[4].strip() == ""
    assert rendered[5].strip() == "• line"


def test_one_long_answer_states_the_rows_it_withheld() -> None:
    """A single turn taller than the card must ADMIT the cut, in lines.

    This is the defect the card shipped with: with one turn the question count
    is zero, so the marker was skipped entirely and a truncated answer read as
    a complete one. The count is in lines because there is no question above to
    count, and it states a quantity rather than a bare "…more" — two more lines
    and fifty are a different decision about whether to go looking.
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [
        AsideTurn(question="why?", answer="\n".join(f"row {i}" for i in range(200)), state="done")
    ]

    rendered = panel.render_lines_for_test()
    marker = [line for line in rendered if "earlier" in line]
    assert marker and "line" in marker[0] and "question" not in marker[0]
    assert re.search(r"↑ \d+ earlier lines · scroll", marker[0])
    # The owning question is pinned under it, so the fragment has an owner.
    assert any(line.startswith("▌ why?") for line in rendered)

    # At the top of the exchange there is nothing withheld and no marker.
    panel._scroll_back_rows = panel._max_scroll_back()
    assert not any("earlier" in line for line in panel.render_lines_for_test())


def test_every_row_of_a_long_answer_is_reachable_by_scrolling() -> None:
    """The fix: the scroll unit is a ROW, so no row is unaddressable.

    Under the turn-index model ``_max_scroll_back`` was ``len(turns) - 1``,
    which is 0 for one turn — 184 of 200 rows could not be reached by any
    gesture. Drives the same entry point the wheel handlers do.
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [
        AsideTurn(
            question="why?",
            answer="\n".join(f"ROW-{i:03d}" for i in range(200)),
            state="done",
        )
    ]

    seen: set[str] = set()
    for _ in range(400):
        seen.update(panel.render_lines_for_test())
        before = panel._scroll_back_rows
        panel._scroll_by(1)
        if panel._scroll_back_rows == before:
            break

    blob = "\n".join(seen)
    assert {i for i in range(200) if f"ROW-{i:03d}" in blob} == set(range(200))
    # And the card never grew to do it: one height at every offset.
    heights = set()
    for offset in range(panel._max_scroll_back() + 1):
        panel._scroll_back_rows = offset
        heights.add(len(panel._compose_rows()) + panel._fit()[1])
    assert len(heights) == 1
    assert max(heights) <= panel._fit()[0] - PANEL_HEIGHT_MARGIN


def test_a_reader_scrolled_back_is_not_dragged_by_a_streaming_answer() -> None:
    """The anchor rule, testable for the first time.

    Under the turn-index model a streaming answer was one turn, so max
    scroll-back was 0 and a reader could not BE scrolled back inside it — the
    state ``append_answer``'s comment protects was unreachable. In rows it is
    real, and holding the offset number still is not enough: the offset counts
    back from the tail, so rows arriving at the tail slide the window forward
    under a reader who is holding still.
    """
    panel = AsidePanel()
    panel.display = True
    generation = panel.ask("why?")
    for index in range(200):
        panel.append_answer(generation, f"ROW-{index:03d}\n")

    panel._scroll_by(120)
    parked = [line for line in panel.render_lines_for_test() if "ROW-" in line]
    for index in range(200, 260):
        panel.append_answer(generation, f"ROW-{index:03d}\n")

    assert [line for line in panel.render_lines_for_test() if "ROW-" in line] == parked

    # A reader AT the tail still follows the stream — that is the other half of
    # the rule, and a fix that pinned everyone would break it.
    following = AsidePanel()
    following.display = True
    generation = following.ask("why?")
    for index in range(50):
        following.append_answer(generation, f"ROW-{index:03d}\n")
    assert "ROW-049" in "\n".join(following.render_lines_for_test())
    for index in range(50, 90):
        following.append_answer(generation, f"ROW-{index:03d}\n")
    assert "ROW-089" in "\n".join(following.render_lines_for_test())
    assert following._scroll_back_rows == 0


def test_scroll_page_is_the_keyboards_way_in_and_reaches_what_the_wheel_does() -> None:
    """``scroll_page`` pages by the rows SHOWN, not by the budget.

    The card overlays its marker and pinned question on the window's top rows,
    so a page of ``_fit()[2]`` would step over exactly the rows the overlay was
    covering. Returns False rather than moving when there is nothing to scroll;
    the caller discards it, so this is a signal and not a receipt.
    """
    closed = AsidePanel()
    assert closed.scroll_page(down=False) is False

    short = AsidePanel()
    short.display = True
    short._turns = [AsideTurn(question="q?", answer="one line", state="done")]
    assert short.scroll_page(down=False) is False

    panel = AsidePanel()
    panel.display = True
    panel._turns = [
        AsideTurn(
            question="why?",
            answer="\n".join(f"ROW-{i:03d}" for i in range(200)),
            state="done",
        )
    ]

    seen: set[str] = set()
    for _ in range(400):
        seen.update(panel.render_lines_for_test())
        if not panel.scroll_page(down=False):
            break
    blob = "\n".join(seen)
    assert {i for i in range(200) if f"ROW-{i:03d}" in blob} == set(range(200))

    # Clamped at the top, and it says so rather than moving.
    assert panel.scroll_page(down=False) is False
    assert panel.scroll_page(down=True) is True

    # FORWARD reaches everything too, which is not implied by the sweep above.
    # Paging forward moves the window the other way from the overlay it has to
    # step around, so a step sized against the CURRENT window lands past rows
    # the destination will cover and nothing ever paints them: measured at
    # budget 8, the jump 51 -> 43 stepped 8 while 6 rows were visible and rows
    # 8-9 fell in the gap. The two directions have to agree about which rows
    # exist.
    panel._scroll_back_rows = panel._max_scroll_back()
    seen = set()
    for _ in range(400):
        seen.update(panel.render_lines_for_test())
        if not panel.scroll_page(down=True):
            break
    blob = "\n".join(seen)
    assert {i for i in range(200) if f"ROW-{i:03d}" in blob} == set(range(200))
    assert panel._scroll_back_rows == 0


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_a_one_row_budget_still_shows_the_answer_not_just_a_marker(budget: int) -> None:
    """At the smallest sizes CONTENT wins the last row, not the marker.

    ``_fit()`` returns 1 for 4 to 7 rows above the dock, and for 8 or 10 with a
    notice showing, so this is a short terminal or a full dock rather than a
    hypothetical. The overlay is drawn OVER the window's top rows, so an
    overlay as large as the budget covers every row it names — measured before
    the fix, budget 1 painted ``↑ 60 earlier lines · scroll`` and nothing else
    at every offset, which is the defect this work exists to remove, sharpened.
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [
        AsideTurn(
            question="why?", answer="\n".join(f"ROW-{i:03d}" for i in range(59)), state="done"
        )
    ]

    with patch.object(AsidePanel, "_fit", return_value=(budget + 6, 0, budget)):
        seen: set[str] = set()
        for offset in range(panel._max_scroll_back() + 1):
            panel._scroll_back_rows = offset
            rendered = panel.render_lines_for_test()
            body = rendered[2:-2]
            assert len(body) == budget, "the card must paint exactly its budget"
            # A row of the EXCHANGE, never a marker on its own. At the very top
            # of a one-row window that row is the question itself, which is
            # content the reader asked for; the failure being guarded is a card
            # whose only row describes the text instead of showing any.
            assert any(
                "ROW-" in line or line.startswith("▌") for line in body
            ), f"only a marker at offset {offset}: {body}"
            seen.update(rendered)

        # And every row is still reachable at these sizes.
        blob = "\n".join(seen)
        assert {i for i in range(59) if f"ROW-{i:03d}" in blob} == set(range(59))


def test_copy_text_takes_the_whole_exchange_off_the_dataclass() -> None:
    """The copy payload is the TEXT, not the painted rows.

    The rows are the windowed subset the reader can already see and they carry
    the card's chrome — a copy key that returned the screen would fail on
    exactly the long answer that makes someone reach for it. Includes a running
    turn, which is the window where ``^f`` is refused.
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [
        AsideTurn(
            question="first?", answer="\n".join(f"ROW-{i:03d}" for i in range(200)), state="done"
        ),
        AsideTurn(question="second?", answer="still arriving", state="running"),
    ]

    payload = panel.copy_text()
    for index in range(200):
        assert f"ROW-{index:03d}" in payload
    assert "still arriving" in payload
    assert "first?" in payload and "second?" in payload
    # No chrome: no title, no rule, no hint row, no overflow marker, no spine.
    for chrome in ("Aside", "off the record", "esc discard", "─", "▌", "↑ "):
        assert chrome not in payload
    # Fork still refuses the half exchange it always refused.
    assert panel.fork_messages() == [("first?", panel._turns[0].answer)]

    assert AsidePanel().copy_text() == ""


def test_copy_text_drops_failed_and_cancelled_turns_on_their_STATE() -> None:
    """A turn that streamed text and then failed must not be copied as an answer.

    Filtering on ``turn.answer`` rather than on ``turn.state`` looks equivalent
    and is not: a failed turn KEEPS whatever streamed before it died, so the
    text test copies words the model never stood behind, formatted exactly like
    words it did. ``fork_messages`` drops the same turns for the same reason.
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [
        AsideTurn(question="ok?", answer="a good answer", state="done"),
        AsideTurn(
            question="failed?",
            answer="partial text before it died",
            state="error",
            error="provider exploded",
        ),
        AsideTurn(question="superseded?", answer="half a thought", state="cancelled"),
        AsideTurn(question="live?", answer="still arriving", state="running"),
    ]

    payload = panel.copy_text()
    assert "a good answer" in payload
    assert "still arriving" in payload
    assert "partial text before it died" not in payload
    assert "half a thought" not in payload
    assert "provider exploded" not in payload
    assert "failed?" not in payload and "superseded?" not in payload


def test_the_footer_advertises_copy_beside_fork() -> None:
    """Copy is discoverable, because the alternative is forking to keep a line.

    The two keys differ in the thing the card's title promises: ``^f`` writes
    the exchange onto the record, ``ctrl+r`` writes it to the clipboard and
    leaves the record alone. Copy shows whenever there is a turn, NOT on
    fork's condition — it works while the answer streams, which is when ``^f``
    is refused.
    """
    panel = AsidePanel()
    panel.display = True
    panel.open()
    assert f"{ASIDE_COPY_KEY} copy" not in panel.render_lines_for_test()[-1]

    generation = panel.ask("why?")
    panel.append_answer(generation, "still streaming")
    footer = panel.render_lines_for_test()[-1]
    assert f"{ASIDE_COPY_KEY} copy" in footer
    assert "^f" not in footer

    panel.settle_answer(generation, "settled")
    panel.set_fork_available(True)
    footer = panel.render_lines_for_test()[-1]
    assert footer.index(ASIDE_COPY_KEY) < footer.index("^f")
    assert footer.startswith("esc discard, back to the chat")


def test_the_wheel_walks_back_through_earlier_questions() -> None:
    """The overflow marker names content, so something has to reach it.

    The wheel and not ↑/↓: those belong to the focused composer's prompt
    history, and the aside's premise is that the user keeps typing there.

    A LIST for the reason
    ``test_a_long_exchange_cuts_on_a_turn_boundary_and_counts_questions``
    records: the answer is markdown, so a tall turn has to be tall AS markdown.
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [
        AsideTurn(question=f"question {index}?", answer="- line\n" * 12, state="done")
        for index in range(6)
    ]
    assert "question 5?" in "\n".join(panel.render_lines_for_test())

    panel._scroll_by(1)
    assert "question 4?" in "\n".join(panel.render_lines_for_test())

    # A new answer snaps back to the tail: a card parked in history while the
    # user's newest question was answered would hide the answer.
    generation = panel.ask("question 6?")
    panel.append_answer(generation, "fresh")
    assert "question 6?" in "\n".join(panel.render_lines_for_test())


def test_a_fork_refusal_is_stated_on_the_card_not_in_the_transcript() -> None:
    """A warning row in the ledger IS the trace the card promises not to leave."""
    panel = AsidePanel()
    panel.display = True
    panel.open()
    panel.set_notice("ask something first — there is nothing to fork")

    assert any("nothing to fork" in line for line in panel.render_lines_for_test())
    # And it clears the moment the user does the thing it asked for.
    panel.ask("why?")
    assert not any("nothing to fork" in line for line in panel.render_lines_for_test())


# -- the answer is markdown, the question is not ---------------------------
def test_the_answer_renders_as_markdown_not_as_its_own_source() -> None:
    """Model prose on the card is prose, the same as everywhere else it shows.

    The card used to wrap the answer as plain text, so the markdown a model
    writes reached the user as literal source — backticks around `code`,
    asterisks around **bold**, a `-` where the transcript draws a bullet. The
    same string handed to an `AssistantBlock` by `^f` rendered properly, so
    identical words changed shape the moment the user kept them.
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [
        AsideTurn(
            question="how does it work?",
            answer=(
                "Call `run_turn()`, which is **not** re-entrant:\n"
                "\n"
                "- drains deltas\n"
                "- flushes cards\n"
            ),
            state="done",
        )
    ]
    body = "\n".join(panel.render_lines_for_test())

    # The markup is gone...
    assert "`run_turn()`" not in body
    assert "**not**" not in body
    assert "- drains deltas" not in body
    # ...and what it MEANT is on the card.
    assert "run_turn()" in body
    assert "not" in body
    assert "•" in body, "a list item renders as a bullet, as it does in the transcript"
    assert "drains deltas" in body


def test_the_question_is_never_markdown_rendered() -> None:
    """The question is the user's own typed input echoed back.

    `_question_rows` mirrors `UserBlock`'s spine deliberately, and running the
    user's keystrokes through a markdown parser would eat the asterisks and
    underscores out of a question that is ABOUT them — "what does **kwargs
    mean" is a question this surface exists to be asked.
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [
        AsideTurn(question="what does **kwargs mean in `f(**kwargs)`?", answer="a.", state="done")
    ]
    body = "\n".join(panel.render_lines_for_test())

    assert "**kwargs" in body
    assert "`f(**kwargs)`" in body


@pytest.mark.parametrize("width", [12, 26, 46])
def test_every_wrapped_question_row_keeps_the_transcript_spine(width: int) -> None:
    from rich.console import Console
    from rich.style import Style

    panel = AsidePanel()
    console = Console()
    mark = Style(color="red", dim=True)
    prose = Style(color="white", dim=False)
    question = "What does **kwargs mean in `f(**kwargs)` when the question wraps?"
    rows = panel._question_rows(question, width, mark, prose)

    assert len(rows) > 1
    for row in rows:
        assert row.plain.startswith("▌ ")
        assert row.cell_len <= width
        assert row.get_style_at_offset(console, 0) == mark
        assert row.get_style_at_offset(console, 2) == prose
    assert "".join(row.plain[2:] for row in rows).replace(" ", "") == question.replace(" ", "")


@pytest.mark.parametrize(
    "turn, expected",
    [
        (AsideTurn("question", answer="Short answer.", state="done"), "Short answer."),
        (AsideTurn("question", state="running"), "thinking…"),
        (AsideTurn("question", error="Failure.", state="error"), "! Failure."),
        (AsideTurn("question", state="cancelled"), "no answer"),
        (AsideTurn("question", answer="Partial answer.", state="cancelled"), "Partial answer."),
    ],
)
def test_each_turn_separates_its_question_from_answer_side_rows(
    turn: AsideTurn, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = AsidePanel()
    panel.display = True
    monkeypatch.setattr(panel, "_content_width", lambda: 26)
    turn.question = "This question needs several wrapped rows to finish"
    panel._turns = [turn, AsideTurn("Next question?", answer="Next answer.", state="done")]
    flat = panel._flat_body()
    for start, end in flat.heads:
        assert all(row.plain.startswith("▌ ") for row in flat.lines[start:end])
        assert flat.lines[end].plain == ""
        assert flat.lines[end + 1].plain.strip(), "exactly one question/answer separator"
    assert flat.lines[flat.heads[0][1] + 1].plain.strip() == expected
    assert len(flat.lines) == len(flat.owners), "separators participate in row windowing"


def test_an_error_is_shown_verbatim_rather_than_parsed() -> None:
    """A provider's failure text is a diagnostic, not markup to interpret."""
    panel = AsidePanel()
    panel.display = True
    generation = panel.ask("why?")
    panel.fail_answer(generation, "no handler for `__call__` in **provider**")
    body = "\n".join(panel.render_lines_for_test())

    assert "`__call__`" in body
    assert "**provider**" in body


def test_the_rendered_rows_are_the_rows_the_card_is_pinned_to() -> None:
    """Markdown changes how many rows an answer occupies, and the card is
    sized to its CONTENT — so the count has to be the real, final one.

    `_repaint` pins `styles.height` to the painted rows plus the gutter. A
    renderer that returned rows the card did not paint (or vice versa) would
    clip its own last line or reserve a row of empty overlay over the chat.
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [
        AsideTurn(
            question="q?",
            answer="intro:\n\n- one\n- two\n\nand a `tail`.\n",
            state="done",
        )
    ]
    rows = panel._compose_rows()

    assert len(rows) == len(panel.render_lines_for_test())
    # Every row is one painted line: a row carrying an embedded newline would
    # make the height pin disagree with what lands on screen.
    assert not any("\n" in row.plain for row in rows)


def test_a_half_arrived_fence_never_swallows_the_prose_before_it() -> None:
    """A dangling ``` or ** is a normal intermediate state while streaming.

    The parser reads an unclosed fence as a code block running to the end of
    the text, which restyles the tail — the transcript's own `AssistantBlock`
    does exactly the same mid-stream. What must NOT happen is text going
    missing: the sentence the user has already read stays on the card through
    every intermediate state.
    """
    panel = AsidePanel()
    panel.display = True
    generation = panel.ask("how do I re-enter?")
    answer = "Stop it first:\n\n```python\nloop.stop()\n```\n\nThen **re-enter** it.\n"

    seen = ""
    for index in range(0, len(answer), 4):
        seen += answer[index : index + 4]
        panel.append_answer(generation, answer[index : index + 4])
        body = "\n".join(panel.render_lines_for_test())
        if "Stop it first" in seen:
            assert "Stop it first" in body, f"prose vanished at {seen!r}"

    panel.settle_answer(generation, answer)
    body = "\n".join(panel.render_lines_for_test())
    assert "loop.stop()" in body
    assert "Then re-enter it." in body
    assert "```" not in body


def test_a_settled_answer_is_not_re_rendered_on_every_delta() -> None:
    """`_repaint` runs per streamed delta and repaints the WHOLE card.

    Without a cache the turns ABOVE the streaming one are re-rendered from
    text that cannot have changed, and the cost is linear in the exchange's
    length: measured at a 120-column card, 0.75 ms per repaint with nothing
    above it against 17.6 ms at 21 turns, past the 30 ms bar
    `test_tui_responsiveness` holds the loop to. The cache is keyed on
    `(text, width, theme epoch)` — the three inputs `flatten` bakes in.
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [
        AsideTurn(question=f"q{index}?", answer=f"answer `{index}`", state="done")
        for index in range(4)
    ]
    panel.render_lines_for_test()
    settled = dict(panel._answer_cache)
    assert settled, "settled answers are cached"

    # A delta on a NEW turn reuses every settled render rather than redoing it.
    panel._turns.append(AsideTurn(question="q4?", answer="streaming"))
    panel.render_lines_for_test()
    assert all(panel._answer_cache.get(key) is rows for key, rows in settled.items())


def test_the_cache_cannot_grow_without_bound_while_an_answer_streams() -> None:
    """Keyed on the TEXT, so every delta mints a fresh key.

    An insert-only cache would hold every prefix of every answer for as long
    as the card is open. The paint drops whatever it did not touch, which
    bounds it at the visible turns without needing a size policy.
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [AsideTurn(question="q?")]
    answer = "a sentence that arrives a few characters at a time, as they do.\n"

    sizes = []
    for index in range(0, len(answer), 4):
        panel._turns[-1].answer += answer[index : index + 4]
        panel.render_lines_for_test()
        sizes.append(len(panel._answer_cache))

    assert max(sizes) == 1, f"one live answer, one entry — saw {max(sizes)}"


def test_dismissing_the_aside_drops_the_rendered_answers_too() -> None:
    """ "No trace" covers the render cache, not just the turns.

    `close()` has to clear it explicitly: `_repaint` returns early on a hidden
    card, so the paint-scoped prune never runs to drop it.
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [AsideTurn(question="q?", answer="an answer with `code`", state="done")]
    panel.render_lines_for_test()
    assert panel._answer_cache

    panel.close()
    assert panel._answer_cache == {}


@pytest.mark.asyncio
async def test_a_resize_refolds_the_answer_rather_than_reusing_the_old_width() -> None:
    """`flatten` bakes the width in, so the cache key carries it.

    A cache keyed on the text alone would paint the old fold at the new width
    — rows either overflowing the card's column or stopping short of it — and
    nothing about the card would say why. Driven through the real app because
    the width comes from the composer's column, which only a laid-out screen
    has.
    """
    session = AsideSession(
        answer=(
            "A sentence long enough to fold differently at sixty columns than at "
            "a hundred and twenty, with `code` in it."
        ),
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        panel = await _open_with_question(pilot, app, "how wide is this?")
        wide = [key[1] for key in panel._answer_cache]
        assert wide, "the settled answer is cached at the width it was folded at"

        await pilot.resize_terminal(60, 24)
        deadline = time.monotonic() + 5.0
        while True:
            await pilot.pause()
            narrow = [key[1] for key in panel._answer_cache]
            if narrow and narrow != wide:
                break
            assert time.monotonic() < deadline, f"the answer never refolded: {narrow} == {wide}"
            await asyncio.sleep(RESIZE_REFIT_DELAY_S)

        # And the rows it painted fit the card it was refolded for.
        assert max(narrow) < max(wide)
        rows = panel.render_lines_for_test()
        assert max(len(row) for row in rows) <= panel.panel_width()


def test_a_repeat_paint_does_not_re_render_the_exchange() -> None:
    """The card must not do work proportional to the exchange on every paint.

    The guard this replaces asserted the narrower fact that a paint renders
    only the turns it can show, which the TURN walk could promise because it
    stopped at the first turn that did not fit. A ROW cut cannot: to know where
    a row window starts you need the row counts of everything above it, so
    `_flat_body` renders the whole exchange once by construction. Asserting the
    old ratio now would be asserting the row window does not exist.

    What the guard was actually protecting is the cost, and that survives
    intact — the card repaints on every streamed delta, the turn count is
    uncapped, so what must not scale is the REPEAT paint. That is what is
    pinned here, and it is pinned the same way: as a ratio, so the test says
    what is wrong (work proportional to the whole exchange) rather than how
    fast this machine is.

    Measured at a 120x40 card over realistic answers, the first paint costs
    31 ms at 120 turns and 78 ms at 300 — real, and inherent to the row cut —
    while every later paint is 0.02 ms and a streamed delta is 0.74 ms, because
    `_flat_body` is memoised on `(width, theme epoch, revision)` and
    `_markdown_rows` caches underneath it. Against the 30 ms `STALL_MS` bar in
    `tests/unit/test_tui_responsiveness.py`, the cost that repeats is what
    decides whether the user feels it.

    Both halves are asserted, because they fail differently: a missed
    `_invalidate_flat` makes the memo stale (wrong rows, caught by the
    correctness tests around it), while a memo keyed on something that changes
    every paint makes it useless (caught here).
    """
    panel = AsidePanel()
    panel.display = True
    panel._turns = [
        AsideTurn(question=f"question {index}?", answer=f"answer `{index}` here", state="done")
        for index in range(50)
    ]
    panel._invalidate_flat()

    # Two counters, because the two caches fail independently and the cheap
    # one masks the expensive one. `flatten` is the markdown render, which
    # `_answer_cache` already prevents repeating; `_answer_rows` is the
    # per-turn ASSEMBLY the flatten memo is what avoids. Counting only the
    # former passes with the memo deleted outright — the assembly is what
    # scales with the exchange once the renders are cached.
    assembled: list[int] = []
    rendered_md: list[int] = []
    original_rows = AsidePanel._answer_rows
    original_flatten = aside_panel_module.flatten

    def counted_rows(self, turn, width, dim, danger):
        assembled.append(1)
        return original_rows(self, turn, width, dim, danger)

    def counted_flatten(*args, **kwargs):
        rendered_md.append(1)
        return original_flatten(*args, **kwargs)

    panel.render_lines_for_test()  # the one paint that pays for the flatten

    AsidePanel._answer_rows = counted_rows
    aside_panel_module.flatten = counted_flatten
    try:
        panel.render_lines_for_test()
        assert assembled == [], (
            f"a repeat paint re-assembled {len(assembled)} answers — the flatten "
            "memo is not holding, so every paint walks the whole exchange"
        )

        # A streamed delta drops the memo, so the exchange IS re-assembled —
        # and that is the point of the second counter: assembly comes back from
        # `_answer_cache`, so no matter how long the exchange is, exactly one
        # answer reaches the markdown renderer. This is what fails if the memo
        # is ever made to evict that cache.
        generation = panel.ask("what changed?")
        rendered_md.clear()
        panel.append_answer(generation, "a token")
        panel.render_lines_for_test()
        assert len(rendered_md) <= 1, (
            f"a streamed delta rendered {len(rendered_md)} answers — only the "
            "live turn's text changed, so the rest must come from the cache"
        )
    finally:
        AsidePanel._answer_rows = original_rows
        aside_panel_module.flatten = original_flatten


def test_the_flatten_memo_is_dropped_when_the_exchange_changes() -> None:
    """A memo that outlives its input paints rows the exchange no longer has.

    The risk the memo introduces, asserted directly rather than through a
    height: every mutation path has to reach `_invalidate_flat`, and the one
    that is easy to get wrong is `append_answer`, which measures the flattened
    height on BOTH sides of the delta to hold a parked reader still.
    """
    panel = AsidePanel()
    panel.display = True
    generation = panel.ask("why?")
    panel.append_answer(generation, "- one\n")
    before = len(panel._flat_body().lines)

    panel.append_answer(generation, "- two\n- three\n")
    after = len(panel._flat_body().lines)
    assert after > before, "the flatten memo survived a streamed delta"

    panel.settle_answer(generation, "- one\n")
    assert len(panel._flat_body().lines) < after, "the memo survived a settle"

    panel.close()
    panel.open()
    assert panel._flat_body().lines == [], "the memo survived the card closing"
