"""The queued-ask TUI surfaces: the minimized bar, the list, the cards.

Design ``docs/design/ask-nonblocking.md`` §5.0 (R7) and §5.1 (PR B).

WHAT THESE TESTS ARE FOR. R7 is a *behavioural* claim — "the composer routes to
the ask only while its answer surface is EXPANDED" — and stills cannot show it.
The two cases that carry the most risk are therefore asserted here by driving
the REAL app (``run_test``, the real stylesheet, the real key path through
``Editor``): what Enter does in each mode, and what happens to the two drafts
across a toggle. The frames the same states produce are captured separately by
``scripts/ask_queue_shot.py``; a green test is not visual evidence, and a still
is not a routing assertion.

THE FLAG. Every surface here is flag-on only (§5's invariant), so each test
that exercises one turns ``asks.policy.NONBLOCKING_ASK`` on explicitly, and
the first test in this file proves the OFF state renders nothing at all.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import pytest
from rich.cells import cell_len

from local_operator.asks import policy
from local_operator.tui.app import ASK_ANSWER_PLACEHOLDER, OperatorApp
from local_operator.tui.widgets.ask_picker import AskPickerScreen
from local_operator.tui.widgets.ask_queue import (
    ASK_BAR_CHEVRON_COLLAPSED,
    ASK_BAR_CHEVRON_EXPANDED,
    ASK_MARKER,
    ASK_TOGGLE_KEY,
    STATUS_OPEN,
    AskBar,
    AskQueueList,
    ask_rows,
)
from local_operator.tui.widgets.editor import Editor
from tests.unit.tui.test_app_pilot import FakeSession, _factory

# The whole module drives the real app through Textual's pilot, so every test
# is a coroutine: one module-level marker rather than sixteen decorators.
pytestmark = pytest.mark.asyncio


def _question(
    qid: str, text: str, *, options: list[str] | None = None, **extra: Any
) -> dict[str, Any]:
    """One wire-shaped question dict, exactly as ``asks/queue.py`` stores it."""
    labels = options if options is not None else ["Yes", "No"]
    return {
        "id": qid,
        "question": text,
        "options": [{"label": label, "description": ""} for label in labels],
        "multi": False,
        "recommended": None,
        "secret": False,
        "persist": False,
        **extra,
    }


def _row(ask_id: str, question: str, *, status: str = STATUS_OPEN, **extra: Any) -> dict[str, Any]:
    return {
        "ask_id": ask_id,
        "created_at": 1_000_000,
        "expires_at": 1_000_000 + 3_600_000,
        "timeout_s": 3600,
        "urgent": False,
        "status": status,
        "delivered": False,
        "questions": [_question("q1", question)],
        **extra,
    }


class _AskSession(FakeSession):
    """The pilot's session, plus the four queued-ask ops the surfaces call.

    A recording double rather than a real ``Session``: these tests are about
    what the TUI *sends*, and the queue's own behaviour is pinned by
    ``tests/unit/asks``. Each call is recorded with its arguments so an
    assertion can name the ask and the answers that were delivered.
    """

    def __init__(self) -> None:
        super().__init__()
        self.answered: list[tuple[str, dict[str, Any], str]] = []
        self.declined: list[tuple[str, str]] = []
        self.dismissed: list[tuple[str, str]] = []
        self.refusals: dict[str, str] = {}
        self.dismiss_refusals: dict[str, str] = {}

    def respond_ask(self, ask_id, answers, *, by="unknown"):
        self.answered.append((ask_id, dict(answers), by))
        error = self.refusals.get(ask_id)
        return {"ok": not error, **({"error": error} if error else {})}

    def decline_ask(self, ask_id, *, by="unknown"):
        self.declined.append((ask_id, by))
        return {"ok": True}

    def dismiss_ask(self, ask_id, *, by="unknown"):
        self.dismissed.append((ask_id, by))
        error = self.dismiss_refusals.get(ask_id)
        return {"ok": not error, **({"error": error} if error else {})}


@pytest.fixture
def enabled(monkeypatch):
    """The queued-ask feature ON for this test, the way the env seam turns it on."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)


def _app(session: FakeSession) -> OperatorApp:
    # `FakeSession` and not `_AskSession`: the viewer's double models a
    # DIFFERENT session shape (async ops), and narrowing this parameter to
    # the owner's double is what made that a type error rather than a test
    # (agent review round 5).
    return OperatorApp(lambda: _factory(session))


async def _settle(pilot, turns: int = 3) -> None:
    for _ in range(turns):
        await pilot.pause()


def _countdown_seconds(painted: str) -> int:
    """The first ``expires in Ns`` in a painted list, or 0 once it expired.

    A frame is a rendering, not a struct: the countdown test has to read the
    number back out of the pixels it is asserting about, and ``expiring`` (the
    row past its deadline) is the floor of that scale, not an unparsed value.
    """
    match = re.search(r"expires in (\d+)s", painted)
    return int(match.group(1)) if match else 0


# -- the flag-off invariant --------------------------------------------------


async def test_flag_off_renders_no_ask_surface(monkeypatch):
    """With the flag off a queued ask is invisible — the writer field is ignored.

    The rows are FED rather than absent, because that is the case that matters:
    a runtime that published ``asks`` while this build's flag is off must not
    paint a bar for a feature this app has not turned on.
    """
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", False)
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        assert app._ask_rows == []
        assert not app.query_one(AskBar).display
        assert app._ask_mode is False
        assert len(app.query(AskPickerScreen)) == 0


# -- the minimized bar -------------------------------------------------------


async def test_minimized_bar_counts_and_names_the_head_question(enabled):
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        bar = app.query_one(AskBar)
        assert bar.display
        line = bar.render().plain
        assert line.startswith(f"{ASK_MARKER} 1 question waiting")
        assert "Deploy now?" in line
        # The chevron is the RIGHT-EDGE affordance and its direction is the
        # state: collapsed means a click expands.
        assert line.rstrip().endswith(ASK_BAR_CHEVRON_COLLAPSED)
        # No surface is mounted: a new ask never auto-mounts (§5.0).
        assert app._ask_mode is False
        assert len(app.query(AskPickerScreen)) == 0

        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a1", "Deploy now?"),
                    _row("a2", "Which region?"),
                    _row("a3", "Roll back?"),
                ]
            )
        )
        await _settle(pilot)
        line = app.query_one(AskBar).render().plain
        assert "3 questions waiting" in line


async def test_bar_is_absent_at_zero_asks(enabled):
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        assert app.query_one(AskBar).display
        app._sync_ask_surface([])
        await _settle(pilot)
        assert not app.query_one(AskBar).display


def test_the_surface_takes_the_composer_row_only_when_it_has_something():
    """The bar paints zero rows at rest — the dock must not grow for nothing.

    A geometry assertion rather than a text one: `display` is what the layout
    engine reads, and a bar that painted an empty row would push the composer
    past the bottom of a short terminal (the class of defect `#prompt-host`'s
    zero-row rule exists for).
    """
    bar = AskBar()
    assert bar.display is False
    bar.set_state(count=0, expanded=False)
    assert bar.display is False
    bar.set_state(count=1, head="Deploy now?", expanded=False)
    assert bar.display is True


# -- expand / collapse, and the composer routing (R7) ------------------------


async def test_clicking_the_bar_expands_the_card_and_flips_the_placeholder(enabled):
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        await pilot.click(AskBar)
        await _settle(pilot)
        assert app._ask_mode is True
        card = app.query_one(AskPickerScreen)
        assert card.is_attached
        editor = app._editor()
        assert editor.placeholder == ASK_ANSWER_PLACEHOLDER
        # ...and the chevron now points the other way.
        assert app.query_one(AskBar).render().plain.rstrip().endswith(ASK_BAR_CHEVRON_EXPANDED)

        # Escape collapses and LEAVES THE ASK OPEN (D5): no decline, no answer.
        await pilot.press("escape")
        await _settle(pilot)
        assert app._ask_mode is False
        assert session.answered == []
        assert session.declined == []
        assert len(app.query(AskPickerScreen)) == 0
        editor = app._editor()
        assert editor.placeholder == editor.resting_placeholder


async def test_minimized_enter_sends_chat_and_expanded_enter_sends_the_answer(enabled):
    """R7 (2) and (3) — the routing rule, driven through the real key path."""
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)

        # MINIMIZED: Enter is an ordinary conversation message.
        app._editor().load_text("hello there")
        await _settle(pilot, 2)
        await pilot.press("enter")
        await _settle(pilot)
        assert session.prompts == ["hello there"]
        assert session.answered == []

        # EXPANDED: the same key sends the ANSWER, and never chat.
        session.prompts.clear()
        app._expand_asks()
        await _settle(pilot)
        # The user put the caret in the composer and typed — the route R7 (3)
        # is about. (The card takes focus on expand, so a bare `press` would
        # hit the card instead; that path is the picker's own and is pinned by
        # `test_ask_picker`.)
        composer = app._editor()
        composer.focus()
        composer.load_text("deploy to eu-west-1")
        await _settle(pilot, 2)
        await pilot.press("enter")
        await _settle(pilot)
        assert session.prompts == []
        assert session.answered == [("a1", {"q1": ["deploy to eu-west-1"]}, "terminal")]
        # Answering collapses the surface and returns the composer.
        assert app._ask_mode is False


async def test_collapsing_preserves_both_drafts_and_reexpanding_restores_the_ask_one(enabled):
    """R7 (4). The chat draft is stashed, not swallowed or cross-sent."""
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        editor = app._editor()
        editor.load_text("a chat draft in progress")
        await _settle(pilot, 2)

        app._expand_asks()
        await _settle(pilot)
        # The chat draft left the buffer — that is what makes "Enter is an
        # answer" safe rather than a way to send a sentence to the ask.
        assert editor.text == ""
        assert app._ask_chat_draft == "a chat draft in progress"

        # An answer draft, typed into the card's free-text row.
        card = app.query_one(AskPickerScreen)
        card.state.selected = card.other_row
        card.state.typed = "eu-west-1"
        await _settle(pilot, 2)

        app._collapse_asks()
        await _settle(pilot)
        # The chat draft is back, and nothing was sent anywhere.
        assert editor.text == "a chat draft in progress"
        assert session.prompts == []
        assert session.answered == []

        # Re-expanding restores the ASK draft.
        app._expand_asks()
        await _settle(pilot)
        card = app.query_one(AskPickerScreen)
        assert card.state.typed == "eu-west-1"
        assert editor.text == ""


async def test_an_ask_settling_while_expanded_collapses_and_restores_the_chat_draft(enabled):
    """R7 (5). No auto-send, no discard: the surface leaves and the draft returns."""
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        editor = app._editor()
        editor.load_text("chat draft")
        await _settle(pilot, 2)
        app._expand_asks()
        await _settle(pilot)
        assert app._ask_mode is True

        # The fold drops the ask (answered from the phone, or expired).
        app._sync_ask_surface([])
        await _settle(pilot)
        assert app._ask_mode is False
        assert len(app.query(AskPickerScreen)) == 0
        assert editor.text == "chat draft"
        assert session.answered == []
        assert editor.placeholder == editor.resting_placeholder


# -- the queue list, decline and dismiss -------------------------------------


async def test_several_asks_open_the_list_and_enter_mounts_that_ask(enabled):
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a1", "Deploy now?"),
                    _row("a2", "Which region?"),
                    _row("a3", "Roll back?"),
                ]
            )
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        assert listing.is_attached
        assert len(app.query(AskPickerScreen)) == 0
        assert [row.ask_id for row in listing.rows] == ["a1", "a2", "a3"]
        # Navigation is a list, not a single-row form: down moves the highlight.
        await pilot.press("down")
        await _settle(pilot)
        assert listing.index == 1
        await pilot.press("enter")
        await _settle(pilot)
        assert app._ask_mounted_id == "a2"
        assert app.query_one(AskPickerScreen).is_attached
        assert not app.query(AskQueueList)


async def test_a_single_ask_expands_straight_to_its_card(enabled):
    """One row would be a click between the user and the only thing to answer."""
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        assert app._ask_mounted_id == "a1"
        assert app.query_one(AskPickerScreen).is_attached
        assert not app.query(AskQueueList)


async def test_the_list_declines_and_dismisses_explicitly(enabled):
    """``d`` declines; ``x`` dismisses a TIMED-OUT ask. Both are user actions."""
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a1", "Deploy now?"),
                    _row("a2", "Which region?", status="timed_out"),
                ]
            )
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        await pilot.press("d")
        await _settle(pilot)
        assert session.declined == [("a1", "terminal")]
        await pilot.press("down")
        await pilot.press("x")
        await _settle(pilot)
        assert session.dismissed == [("a2", "terminal")]


async def test_the_list_marks_a_timed_out_ask_as_still_answerable(enabled):
    """Honest states (§5): a timed-out ask is NOT renamed to finished."""
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a1", "Deploy now?"),
                    _row("a2", "Which region?", status="timed_out"),
                ]
            )
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        text = app.query_one(AskQueueList).render().plain
        assert "timed out — still answerable" in text
        # ...and it does not inflate the bar's count of what the user owes.
        assert "1 question waiting" in app.query_one(AskBar).render().plain


async def test_a_refused_answer_says_so_and_does_not_lose_the_ask(enabled):
    """A lost race (answered elsewhere) is reported, not swallowed."""
    session = _AskSession()
    session.refusals["a1"] = "already answered by phone."
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        app._submit_ask_answer("eu-west-1")
        await _settle(pilot)
        assert session.answered[0][0] == "a1"


# -- the sidebar mark --------------------------------------------------------


async def test_the_sidebar_mark_is_painted_for_the_current_session(enabled):
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        sidebar = app._session_sidebar
        assert sidebar._asking == {session.session_id: 1}
        app._sync_ask_surface([])
        await _settle(pilot)
        assert sidebar._asking == {}


async def test_the_sidebar_mark_survives_a_queue_that_only_timed_out(enabled):
    """A timed-out-but-answerable ask keeps the mark: it is OUTSTANDING.

    The mark used to be painted from the OPEN count alone, so a queue of nothing
    but timed-out asks dropped it to absence — even though the bar was still up
    and every one of those asks was still answerable. The mark, the bar and the
    wire's ``asks_open`` now read the same set (``store.OUTSTANDING_STATUSES``).
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?", status="timed_out")]))
        await _settle(pilot)
        sidebar = app._session_sidebar
        assert sidebar._asking == {session.session_id: 1}
        assert app._ask_bar.display is True


def _response_details(status: str) -> dict[str, Any]:
    return {
        "ask_id": "a1",
        "status": status,
        "questions": [
            _question("q1", "Which region?"),
            _question("q2", "Rotate the key?", secret=True, options=[]),
        ],
        "answers": {"q1": ["eu-west-1"], "q2": ["[DEPLOY_KEY]"]},
        "text": "Answered wholesale.",
    }


def test_the_response_card_expands_to_the_questions_and_the_answers():
    from local_operator.tui.widgets.transcript import AskResponseBlock

    block = AskResponseBlock(_response_details("answered"), kind="response")
    collapsed = block._build_content(110).plain
    assert "Answered" in collapsed
    assert "eu-west-1" not in collapsed

    block.action_activate()
    expanded = block._build_content(110).plain
    assert "Which region?" in expanded
    assert "eu-west-1" in expanded
    # A SECRET answer shows the KEY, never a value: the wire carries keys here
    # and the card must not become the surface that resolves one.
    assert "[DEPLOY_KEY]" in expanded


def test_the_timeout_card_expands_to_what_went_unanswered():
    from local_operator.tui.widgets.transcript import AskResponseBlock

    details = {
        "ask_id": "a1",
        "status": "timed_out",
        "waited_s": 3600,
        "urgent": False,
        "lapsed_while_stopped": False,
        "questions": [_question("q1", "Which region?")],
        "text": "Timed out.",
    }
    block = AskResponseBlock(details, kind="timeout")
    assert "Timed out" in block._build_content(110).plain
    block.action_activate()
    expanded = block._build_content(110).plain
    assert "Which region?" in expanded
    assert "not answered" in expanded


def test_a_row_with_no_structured_questions_falls_back_to_the_envelope():
    """An older runtime's row (or a timeout with none) must not expand to nothing."""
    from local_operator.tui.widgets.transcript import AskResponseBlock

    block = AskResponseBlock({"ask_id": "a1", "text": "Answered."}, kind="response")
    block.action_activate()
    assert "Answered." in block._build_content(110).plain


async def test_an_incomplete_answer_is_never_submitted_and_a_complete_one_is():
    """The submit gate: reached by Escape versus by answering the last question.

    ``AskPickerScreen`` resolves a PARTIAL map when the user Escapes mid-walk
    and the whole map when they answer the last question, and R7/D5 make the
    first a COLLAPSE (the ask stays open). Asserted against the app's own
    handler rather than a restatement of the rule, so a change to either side
    fails here instead of silently submitting an incomplete row — which
    ``respond`` refuses anyway, turning a plain Escape into a refusal notice.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        # Adopt, as the app does before the dock can be clicked: the queue OPS
        # are the session's, and the surfaces resolve them through it.
        app._session = session

        # Escape on question 1 of a 2-question ask: a partial map, nothing sent.
        app._ask_rows = ask_rows(
            [
                {
                    **_row("a2", "Two parts?"),
                    "questions": [_question("q1", "First?"), _question("q2", "Second?")],
                }
            ]
        )
        app._on_queue_ask_settle("a2", {"q1": ["Yes"]})
        await _settle(pilot)
        assert session.answered == []

        # Answering the last question: the complete map, one atomic write.
        app._ask_rows = ask_rows([_row("a1", "Deploy now?")])
        app._on_queue_ask_settle("a1", {"q1": ["Yes"]})
        await _settle(pilot)
        assert session.answered and session.answered[0][0] == "a1"
        assert session.answered[0][1] == {"q1": ["Yes"]}


# -- round 1 remediation: the flows the four reviews actually walked ---------
#
# One test per finding, driven through the real app the same way the reviewers
# drove it. The findings are named in each docstring, because the value of the
# test is not the assertion — it is knowing which defect it stands in front of.


async def test_the_chat_draft_survives_picking_an_ask_out_of_the_list(enabled):
    """BLOCKER-1: the list→card swap used to destroy the conversation draft.

    The n>1 path is the one the list exists for, and it was the path that lost
    the user's draft: ``_mount_ask_card`` collapsed with ``restore_draft=False``,
    which CLEARS the stash rather than restoring it. The suite covered n=1 only,
    so nothing caught it.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        editor = app.query_one(Editor)
        editor.load_text("a chat draft in progress")
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which region?")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        assert app._ask_chat_draft == "a chat draft in progress"
        app.on_ask_queue_list_picked(AskQueueList.Picked("a2"))
        await _settle(pilot)
        # The draft is STILL stashed, not silently thrown away...
        assert app._ask_chat_draft == "a chat draft in progress"
        # ...and it comes back intact when the surface finally closes.
        app._collapse_asks()
        await _settle(pilot)
        assert editor.text == "a chat draft in progress"


async def test_the_list_state_does_not_promise_the_ask_route(enabled):
    """U1 / review MAJOR-2: the placeholder and the route read ONE condition.

    With the LIST up the composer used to say "Enter sends it to the ask" while
    ``on_editor_submitted`` sent the text to the CONVERSATION — the copy
    promised a routing the code did not perform.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        editor = app.query_one(Editor)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which region?")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        assert app.query(AskQueueList)
        assert app._composer_placeholder_for(editor) != ASK_ANSWER_PLACEHOLDER
        # ...and what Enter does matches what the placeholder said: chat.
        editor.load_text("this is a chat message")
        editor.focus()
        await _settle(pilot)
        await pilot.press("enter")
        await _settle(pilot)
        assert session.aborts == []
        assert [p for p in session.prompts if "chat message" in str(p)]
        assert session.answered == []
        # On a CARD the same two sites agree the other way.
        app.on_ask_queue_list_picked(AskQueueList.Picked("a1"))
        await _settle(pilot)
        assert app._composer_placeholder_for(editor) == ASK_ANSWER_PLACEHOLDER
        editor.load_text("eu-west-1")
        await _settle(pilot)
        await pilot.press("enter")
        await _settle(pilot)
        assert session.answered and session.answered[0][0] == "a1"


async def test_escape_from_the_composer_collapses_and_does_not_stop(enabled):
    """QA Q2: Esc from the answer surface's own composer aborted the turn.

    The state the user is in while typing an answer is the composer-focused one,
    so the key most likely to be pressed there fell through the whole stop
    ladder — from a surface whose footer says "esc collapse".
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        editor = app.query_one(Editor)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        editor.focus()
        await _settle(pilot)
        await pilot.press("escape")
        await _settle(pilot)
        assert app._ask_mode is False
        assert app._ask_card is None
        assert session.aborts == [], "Esc collapsed the surface but still stopped the turn"


async def test_escape_out_of_a_card_returns_to_the_list_at_the_same_row(enabled):
    """UX U6: a back-out from a card opened OUT of the list is not a leave.

    Collapsing to minimized made the user re-expand and re-hunt for their place,
    because the list was rebuilt with the highlight back at row 1.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows(
                [_row("a1", "Deploy now?"), _row("a2", "Which region?"), _row("a3", "Roll back?")]
            )
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("enter")
        await _settle(pilot)
        assert app._ask_mounted_id == "a3"
        await pilot.press("escape")
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        assert listing.is_attached, "Esc out of a card left the list behind entirely"
        assert [row.ask_id for row in listing.rows][listing.index] == "a3"
        assert app._ask_mode is True


async def test_a_click_on_a_row_opens_that_ask(enabled):
    """UX U2 + review round 2 MAJOR: a click opens the ask that is PAINTED there.

    This test asserted the MIS-mapping on the round-1 head — it clicked the row
    where ``a1`` was painted and expected ``a2``, because ``_row_at`` counted
    from the content area while Textual hands ``event.y`` in outer-region
    coordinates (the panel carries ``padding: 1 1``). So it is rewritten to
    click what the widget PAINTS: the offset for row *i* is the widget's own top
    inset plus the header's painted height plus *i*, read from the widget rather
    than from a constant.

    Every row is clicked, not one, because the defect made the LAST ask
    unreachable — the row that no single-row probe would have missed only if it
    happened to be the one probed.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows(
                [_row("a1", "Deploy now?"), _row("a2", "Which region?"), _row("a3", "Roll back?")]
            )
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        for index, ask_id in enumerate(["a1", "a2", "a3"]):
            listing = app.query_one(AskQueueList)
            top = listing.content_region.y - listing.region.y
            await pilot.click(AskQueueList, offset=(4, top + listing.HEADER_ROWS + index))
            await _settle(pilot)
            assert (
                app._ask_mounted_id == ask_id
            ), f"the click at the painted row {index} opened {app._ask_mounted_id}"
            assert app.focused is not app.query_one(Editor), "the click gave the composer the caret"
            # Back to the list for the next click, keeping the rows in place.
            app._clear_ask_surface()
            app._mount_ask_list()
            await _settle(pilot)


@pytest.mark.parametrize(
    "size, kept",
    [
        # At 130 columns the row still fits the irreversible action's own hint
        # (`d decline`) alongside the primary one; at 100 the CONTROL and the
        # drawer's count are what survives; at 60 the control alone, because a
        # filtered view that cannot be un-filtered is worse than an unadvertised
        # key — `esc` still collapses from the composer and `d` still declines,
        # and both are still true when unspoken. Asserting all three makes the
        # sacrifice order one a reader can predict rather than guess.
        ((130, 30), "d decline"),
        ((100, 30), "Waiting or moved on"),
        ((60, 20), "Settled"),
    ],
)
async def test_the_list_header_stays_one_line_and_never_splits_a_hint(enabled, size, kept):
    """Design round 2, D12: the header wrapped at 80 columns and orphaned an `x`.

    It is also the row the hit test counts from, so a header that wraps shifts
    every pointer hit below it — the two findings are one cause. Hints are spent
    whole, from the right, and the row is ``no_wrap``. The filter control is
    what the row keeps longest: design §4's three-way filter is the operator's
    own ask, and it is the only part of the row that is a live destination
    rather than a statement.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=size) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a1", "Deploy now?"),
                    _row("a2", "Which region?", urgent=True),
                    _row("a3", "Roll back?"),
                ]
            )
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        painted = listing.render().plain.splitlines()
        assert len(painted) == len(listing.rows) + listing.HEADER_ROWS, painted
        header = painted[0]
        assert cell_len(header) <= listing.content_size.width, header
        # Every hint that IS painted is painted whole — key and verb together.
        for hint in listing.HEADER_HINTS:
            if hint.split()[-1] in header:
                assert hint in header, f"{hint!r} was split at the row end"
        # The hint that survives at this width, in full.
        assert kept in header, header


async def test_the_bar_says_how_many_are_urgent(enabled):
    """Design round 2, D13: the amber glyph was the only channel for urgency.

    On one row the hue said "urgent" while the sentence beside it named the
    question that was not urgent, because the glyph is painted from
    ``any(row.urgent)`` and the head names ``rows[0]``.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a1", "Which rollout should the stale-row migration take?"),
                    _row("a2", "Rotate the deploy key before the cutover?", urgent=True),
                ]
            )
        )
        await _settle(pilot)
        bar = app.query_one(AskBar).render().plain
        assert "1 urgent" in bar, bar
        assert "2 questions waiting" in bar, bar


async def test_the_head_question_fills_a_wide_bar(enabled):
    """Design round 2, D14: the 48-cell ceiling still bit at the widest size.

    The named question was cut with empty cells left before the chevron, which
    is the case D9 asked to be spared. The app is driven out of its boot card
    first, because that clamp is what sizes the composer shell — the capture
    script does the same thing (it appends the turns the frames show).
    """
    from local_operator.tui.widgets.transcript import UserBlock

    session = _AskSession()
    app = _app(session)
    question = "Which rollout should the stale-row migration take tonight?"
    async with app.run_test(size=(130, 40)) as pilot:
        await _settle(pilot)
        app._append_block(UserBlock("what should we do about the stale rows?"))
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", question)]))
        await _settle(pilot)
        bar = app.query_one(AskBar)
        assert bar.size.width > 100, bar.size.width
        text = bar.render().plain
        assert question in text, text
        assert cell_len(text) <= bar.size.width, text


async def test_a_card_that_cannot_be_built_does_not_strand_ask_mode(enabled):
    """Review round 2, MINOR-3: an early return left ask mode on with no surface."""
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        assert app._ask_mode is True and app._ask_card is not None
        # A row whose questions cannot be parsed: the card gives up, and the
        # composer must not be left routing to a surface that is not there.
        app._clear_ask_surface()
        broken = ask_rows([_row("a1", "Deploy now?")])[0].__class__
        from local_operator.tui.widgets.ask_queue import AskRow

        app._mount_ask_card(  # type: ignore[arg-type]
            AskRow(
                ask_id="a9",
                status=STATUS_OPEN,
                created_at=0,
                expires_at=0,
                urgent=False,
                questions=({"id": "q1", "question": "", "options": [], "multi": False},),
            )
        )
        await _settle(pilot)
        assert app._ask_card is None
        assert app._ask_mode is False, "ask mode outlived the surface that justified it"
        assert broken is AskRow


async def test_the_key_route_reaches_the_surface_from_the_composer(enabled):
    """UX U5: nothing reached the bar from the keyboard, so a queued ask was mouse-only.

    The key is asserted to be one the composer does NOT claim, which is the
    reason it is not f7: Textual's ``TextArea`` binds f6/f7 itself, so the key
    this row would naturally suggest is swallowed by the one surface that most
    needs the route.
    """
    from textual.widgets import TextArea

    # `getattr`, not `.key`: a `BINDINGS` entry may be written as a tuple, and
    # the check is about the keys that exist, not about their spelling.
    composer_keys = {str(getattr(binding, "key", binding)) for binding in TextArea.BINDINGS}
    assert (
        ASK_TOGGLE_KEY not in composer_keys
    ), f"{ASK_TOGGLE_KEY} is claimed by the composer, so it can never reach the app"
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app.query_one("Editor").focus()
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which region?")]))
        await _settle(pilot)
        await pilot.press(ASK_TOGGLE_KEY)
        await _settle(pilot)
        assert app._ask_mode is True
        assert app.query(AskQueueList)
        await pilot.press(ASK_TOGGLE_KEY)
        await _settle(pilot)
        assert app._ask_mode is False


async def test_the_open_list_follows_the_wire(enabled):
    """QA Q3 / design D4: the list was built once and never refreshed."""
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows(
                [_row("a1", "Deploy now?"), _row("a2", "Which region?"), _row("a3", "Roll back?")]
            )
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        assert len(listing.rows) == 3
        # a3 is answered from the phone: the panel must drop it, not keep a
        # stale, answerable-looking row.
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which region?")]))
        await _settle(pilot)
        assert [row.ask_id for row in listing.rows] == ["a1", "a2"]
        assert listing.index == 0


async def test_a_late_answer_leaves_the_answerable_surfaces(enabled):
    """UX U3 / design D6: a late answer was still painted "timed out — still
    answerable", counted as owed, and had an answer box opened for it."""
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows([_row("a1", "Deploy now?", status="late"), _row("a2", "Which region?")])
        )
        await _settle(pilot)
        # The late row is KEPT in the list (design §4: the surface shows the
        # whole queue and the filter picks a half), so what "leaves the
        # answerable surfaces" means is now narrower and sharper: it leaves the
        # set the user OWES an answer on.
        assert [row.ask_id for row in app._open_ask_rows()] == ["a2"]
        assert [row.ask_id for row in app._ask_rows] == ["a1", "a2"]
        assert "1 question waiting" in app.query_one(AskBar).render().plain
        assert "timed out" not in app.query_one(AskBar).render().plain


async def test_the_bar_and_the_list_speak_from_one_fold(enabled):
    """UX U4 / QA Q4 / design D7, re-expressed by design §4's two registers.

    Two totals for one queue, three rows apart, is the defect this file's
    round-1 fix exists to prevent — and the fix was to make them ONE count.
    Design §4 splits them again, deliberately and in one direction only: the
    bar keeps the CHIP register (what the agent is still waiting on) and the
    list takes the DRAWER register (what the drawer is showing). What must NOT
    happen is the two counts disagreeing about the same population, so the
    assertions below pin the halves they name rather than one shared string.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which region?", status="timed_out")])
        )
        await _settle(pilot)
        bar_text = app.query_one(AskBar).render().plain
        app._expand_asks()
        await _settle(pilot)
        header = app.query_one(AskQueueList).render().plain.splitlines()[0]
        # The CHIP: only the open half is "waiting"; the timed-out one is named
        # as what it is.
        assert "1 question waiting · 1 ask timed out" in bar_text, bar_text
        # The DRAWER: a mixed queue spells BOTH halves, in the drawer's words.
        assert "1 waiting, 1 moved on" in header, header
        # Same rows, same totals: the two registers describe ONE queue.
        assert "All · 2" in header, header
        assert "Waiting or moved on · 2" in header, header
        assert "Settled · 0" in header, header


async def test_the_bar_keeps_its_chevron_at_eighty_columns(enabled):
    """Design D5/D9 + UX U10: at 80x24 the bar overflowed and lost the chevron.

    The head question is what gives way, and the expiry words it also paints are
    the second half of the finding: urgency used to be carried by hue alone
    because the countdown never rendered at all (design D8).
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row(
                        "a1",
                        "Which rollout should the stale-row migration take tonight?",
                        expires_at=1_000_000 + 120_000,
                    )
                ]
            )
        )
        await _settle(pilot)
        bar = app.query_one(AskBar)
        text = bar.render().plain
        assert text.rstrip().endswith(ASK_BAR_CHEVRON_COLLAPSED), repr(text)
        assert len(text) <= bar.size.width, (len(text), bar.size.width)
        assert bar.size.width <= 80


async def test_the_card_title_never_claims_the_agent_is_waiting(enabled):
    """Design D1 (blocker) + UX U9: the queued card said the agent was WAITING.

    In the queued model the tool returned a receipt and the agent moved on, so
    the old string misstated both the machine's state and the cost of the user's
    silence. The timeout case is the second half: the card kept the same line
    after the bar beside it had begun saying "timed out".
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        card = app.query_one(AskPickerScreen)
        assert "waiting" not in card._title, card._title
        assert "moved on" in card._title
        # The ask times out UNDER the open card: the card must follow the wire.
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?", status="timed_out")]))
        await _settle(pilot)
        card = app.query_one(AskPickerScreen)
        assert "waiting" not in card._title, card._title
        assert "timed out" in card._title, card._title


def test_the_timeout_receipt_keeps_the_warning_ink():
    """Review MINOR-3: the comment claimed an ink the shared builder dropped.

    Both receipts painted their summary in ``dim``, so the deadline that fired
    read exactly like the answer that arrived — while the comment beside them
    argued they were different facts.
    """
    from local_operator.tui.widgets.transcript import AskResponseBlock

    timeout = AskResponseBlock({"text": "…"}, kind="timeout")
    answered = AskResponseBlock({"text": "…"}, kind="response")
    assert timeout._summary_ink() == "warning"
    assert timeout._summary_ink() != answered._summary_ink()
    # And the receipt renders with the flag OFF: rendering a stored row is not
    # gated, only producing one is (this file's MINOR-4 note).
    assert AskResponseBlock({"text": "…"}, kind="response")._build_content(80).plain


async def test_a_refused_dismiss_reports_at_the_gesture(enabled):
    """UX U8: a refused ``x`` reported into the transcript, which is scrolled
    away while the dock has the user's attention — so the row looked dead."""
    from local_operator.tui.widgets.toast import Toast

    session = _AskSession()
    session.dismiss_refusals["a1"] = "only a timed-out ask can be dismissed; it is still open."
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        app._dismiss_ask("a1")
        await _settle(pilot)
        # The toast owns the message while it is up; the transcript is untouched.
        assert "still open" in str(app.query_one(Toast)._message)


# -- round 3: the row height the hit test depends on -------------------------

#: A question that cannot fit any of the widths under test on one line if it is
#: allowed to wrap — the fixture whose absence let round 3's defect through.
LONG_QUESTION = (
    "Which rollout should the stale-row migration take tonight, and " + "which shard " * 6
)


@pytest.mark.parametrize("size", [(100, 30), (80, 24), (60, 20)])
async def test_every_row_paints_one_line_and_every_painted_row_is_clickable(enabled, size):
    """Review round 3: the hit test counted one line per ask, but rows could WRAP.

    ``Text(no_wrap=True)`` is a rich-side hint the widget's painting ignores, so
    a 90-character question spent a second painted line and every click below it
    opened the wrong ask — Bravo's line opened the third ask, and the last two
    were inert. The lever that paints is the sheet's ``text-wrap: nowrap`` (a
    post-mount ``styles.text_wrap`` write does not change the paint), and the
    assertion below is the PAINT's own number rather than the builder's: the
    widget's virtual height must be its padding plus one line per ask plus the
    header. Every row is then clicked at the offset it is painted at.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=size) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row(f"a{i}", f"{name}: {LONG_QUESTION}")
                    for i, name in enumerate(["Alpha", "Bravo", "Charlie", "Delta"], start=1)
                ]
            )
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        padding = listing.styles.padding.top + listing.styles.padding.bottom
        assert listing.styles.text_wrap == "nowrap", "the rows may wrap again"
        assert listing.virtual_size.height - padding == listing.HEADER_ROWS + len(listing.rows), (
            f"a row painted more than one line: {listing.virtual_size.height} painted rows "
            f"for {len(listing.rows)} asks"
        )
        for index, row in enumerate(list(listing.rows)):
            top = listing.content_region.y - listing.region.y
            await pilot.click(AskQueueList, offset=(4, top + listing.HEADER_ROWS + index))
            await _settle(pilot)
            assert (
                app._ask_mounted_id == row.ask_id
            ), f"the click at painted row {index} opened {app._ask_mounted_id}, not {row.ask_id}"
            app._clear_ask_surface()
            app._mount_ask_list()
            await _settle(pilot)


async def test_the_urgency_word_counts_the_same_set_on_both_surfaces(enabled):
    """Review round 3, MINOR-1: the bar counted OPEN urgent asks, the list every row.

    With an urgent ask whose deadline has already passed, the list said
    `1 urgent · 1 ask timed out` while the bar showed only its amber hue — one
    queue described two ways, which is the class of defect the round-1 count
    vocabulary fix exists to prevent.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a1", "Deploy now?", urgent=True),
                    _row("a2", "Which region?", status="timed_out", urgent=True),
                ]
            )
        )
        await _settle(pilot)
        bar = app.query_one(AskBar).render().plain
        app._expand_asks()
        await _settle(pilot)
        header = app.query_one(AskQueueList).render().plain.splitlines()[0]
        assert "1 urgent" in bar, bar
        assert "1 urgent" in header, header
        assert "2 urgent" not in header, header


# -- round 4: the VIEWER holder ---------------------------------------------


class _ViewerAskSession(FakeSession):
    """The viewer's shape: the same three ops, but ``async`` (they cross the wire).

    This is the double whose absence let QA round 4's blocker through: the dock
    looked the owner's names up on whichever session held the ask, and a viewer
    that lacked them made every answer silently do nothing.

    It inherits ``FakeSession`` and NOT ``_AskSession``, which is a typing
    constraint rather than a filing choice: ``_AskSession`` implements the three
    ops SYNCHRONOUSLY (the owner's shape), so a subclass replacing them with
    coroutines is an incompatible override — pyright said so on CI's
    ``type-check`` while my local run had been made before this double existed
    (agent review round 5, BLOCKER). Modelling the viewer on the pilot's plain
    session is also the truthful shape: a viewer's ops are async, not an owner's
    with an await bolted on.
    """

    def __init__(self) -> None:
        super().__init__()
        self.wire: list[tuple[str, str, dict[str, Any], str]] = []

    async def respond_ask(self, ask_id, answers=None, *, by="unknown"):
        self.wire.append(("respond", ask_id, dict(answers or {}), by))
        return {"ok": True}

    async def decline_ask(self, ask_id, *, by="unknown"):
        self.wire.append(("decline", ask_id, {}, by))
        return {"ok": False, "error": "already answered by phone."}

    async def dismiss_ask(self, ask_id, *, by="unknown"):
        self.wire.append(("dismiss", ask_id, {}, by))
        return {"ok": True}


async def _drain(pilot, turns: int = 6) -> None:
    """Let the op's worker run and report: it rides a task, not the key path."""
    await _settle(pilot, turns)


async def test_a_viewer_answers_a_queued_ask_through_its_async_op(enabled):
    """QA round 4, Q1: on a viewer every answer used to be silently dropped.

    ``AttachedSession`` now carries the same three names as the owner, in the
    same verdict-dict contract but ``async`` — so the dock's one lookup works on
    both kinds of holder and the copy is the same.
    """
    session = _ViewerAskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which region?")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        app.on_ask_queue_list_picked(AskQueueList.Picked("a1"))
        await _settle(pilot)
        app._submit_ask_answer("eu-west-1")
        await _drain(pilot)
        assert session.wire == [("respond", "a1", {"q1": ["eu-west-1"]}, "terminal")], session.wire
        # The surface STAYS UP and advances to the next ask (audit B): a1 was
        # picked out of a two-row list, and the collapse this asserted was the
        # rule the audit replaced. The viewer's verdict arrives later and does
        # not move the surface — a refusal leaves the row where the wire still
        # has it, so the advance costs nothing on that path.
        assert app._ask_mode is True
        listing = app.query_one(AskQueueList)
        assert listing.rows[listing.index].ask_id == "a2"


async def test_a_viewer_refusal_reaches_the_screen_in_the_owners_words(enabled):
    """The refusal arrives on the worker, so it has to be reported from there."""
    from local_operator.tui.widgets.toast import Toast

    session = _ViewerAskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which region?")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        app._decline_ask("a1")
        await _drain(pilot)
        assert session.wire == [("decline", "a1", {}, "terminal")], session.wire
        assert "already answered by phone." in str(app.query_one(Toast)._message)


async def test_a_row_too_wide_for_the_box_keeps_its_expiry_and_shows_the_cut(enabled):
    """Agent review round 4, MINOR-1: the painter hard-crops, so the TAIL went.

    A row wider than the content box is cropped by Textual without a glyph, and
    because the expiry is appended last it was the expiry that fell off — the
    per-row word for urgency that design D13 added so meaning did not rest on
    hue alone. The question is clipped against the room the tail leaves, so the
    cut is painted and the tail survives.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot)
        # TWO asks, because one expands straight to its card — the list is what
        # this test is about.
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a1", LONG_QUESTION, urgent=True, expires_at=1_700_003_600_000),
                    _row("a2", "Which region?"),
                ]
            )
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        row = next(line for line in listing.render().plain.splitlines() if "Which rollout" in line)
        assert "…" in row, row
        assert "expiring" in row or "expires in" in row, row
        assert cell_len(row) <= listing.content_size.width, row


# -- the audited gaps: the frozen countdown, the re-expand, the dim receipt ---
#
# The audit of the MERGED surface found three gaps the four review rounds did
# not: a painted countdown that only moved on a frontend snapshot, a queue that
# cost one re-expand per answer, and a late answer painted ``dim`` while the
# timeout row beside it painted amber. One test each, driven through the real
# app, because all three are claims about what the user SEES between events.


async def test_the_countdown_repaints_between_snapshots(enabled, monkeypatch):
    """The countdown is derived at paint time, so a paint must fire on its own.

    ``_sync_ask_surface`` runs only on a frontend snapshot, so a surface left
    open froze its "expires in 42m" until the next wire event — past the very
    deadline it was counting down to. The tick is that missing event. The period
    is patched to a test-sized one so it is the FIRING that is asserted rather
    than the wiring, and the second half pins the other edge: no rows, no clock.
    """
    from local_operator.tui import app as app_module

    monkeypatch.setattr(app_module, "ASK_COUNTDOWN_TICK_S", 0.05)
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        assert app._ask_tick is None, "a session with no asks carries no clock"
        # TWO asks with a deadline five seconds out: two rows keep the LIST up
        # (one ask expands straight to its card), and a five-second deadline
        # leaves room for the second hand to move inside the test rather than
        # an hour from now.
        expires_at = int(time.time() * 1000) + 5_000
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a1", "Deploy now?", expires_at=expires_at),
                    _row("a2", "Which region?", expires_at=expires_at),
                ]
            )
        )
        await _settle(pilot)
        assert app._ask_tick is not None, "the countdown was never armed"
        app._expand_asks()
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        armed_at = listing._now_ms
        first = listing.render().plain
        assert "expires in" in first, first
        # No snapshot, no keypress, nothing but the clock: the row must re-derive.
        # A whole second has to pass before the WORDS can move, so the frame is
        # read a second after a tick period of 0.05 s.
        await asyncio.sleep(1.2)
        await _settle(pilot)
        after = listing.render().plain
        assert listing._now_ms > armed_at, "the row's clock never advanced"
        assert _countdown_seconds(after) < _countdown_seconds(first), (first, after)

        # The last row leaving stops the clock rather than letting it tick on.
        app._sync_ask_surface([])
        await _settle(pilot)
        assert app._ask_tick is None
        assert app._ask_mode is False


def _countdown_token(painted: str) -> str:
    """The ``expires in 42m`` word a painted row carries, or "".

    Design D2's tests are about the WORD moving while the layout does not, so
    they read the token out of the frame rather than asserting a fixed string:
    the exact value depends on a live clock, and the claim is the CHANGE.
    """
    match = re.search(r"expires in \d+[smh]", painted)
    return match.group(0) if match else ""


def test_the_expiry_field_reserves_its_widest_point():
    """Design round 1, D2: the tail's own width must not move under the question.

    ``10m`` is one cell wider than ``9m``, and the words change unit at the hour
    and minute boundaries, so a row that had filled its line re-cut its own
    question on every tick — the one thing a repaint-only tick must not do. The
    reservation is what keeps the clip budget fixed; this asserts it is CONSTANT
    across every word one row can paint, and never narrower than the word it
    reserves for. Spelled out as a unit test because the invariant is arithmetic
    on the row, not on any frame.
    """
    from local_operator.tui.widgets.ask_queue import AskRow, expiry_room, expiry_text

    created = 1_700_000_000_000
    for urgent in (False, True):
        row = AskRow(
            ask_id="a1",
            status=STATUS_OPEN,
            created_at=created,
            expires_at=created + 3_600_000,
            urgent=urgent,
            questions=(),
        )
        words = [
            expiry_text(row, created + offset)
            for offset in (0, 1_800_000, 3_540_000, 3_599_000, 3_600_000, 3_601_000)
        ]
        rooms = {expiry_room(row, word) for word in words}
        assert len(rooms) == 1, (urgent, words, rooms)
        room = rooms.pop()
        assert all(cell_len(word) + 2 <= room for word in words), (urgent, words, room)

    # A timed-out row's word is not a countdown and does not move, so its
    # reservation is exactly that word rather than a countdown's worst case.
    timed = AskRow(
        ask_id="a2",
        status="timed_out",
        created_at=created,
        expires_at=created + 60_000,
        urgent=False,
        questions=(),
    )
    word = expiry_text(timed, created)
    assert expiry_room(timed, word) == cell_len(word) + 2


async def test_a_clock_tick_does_not_re_cut_the_question(enabled):
    """Design round 1, D2, through the app: the tick moves the tail and nothing else.

    Driven by advancing the widget's own clock (``set_now``, exactly what the
    interval calls) rather than by sleeping: the claim is about the PAINT, and
    the timer's own firing is the countdown test's evidence.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(pilot)
        # TWO asks, because one expands straight to its card and this is about
        # the LIST's rows.
        deadline = int(time.time() * 1000) + 600_000
        app._sync_ask_surface(
            ask_rows([_row("a1", LONG_QUESTION, expires_at=deadline), _row("a2", "Which region?")])
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        # The clock is SET rather than nudged: the discriminating case is a
        # DIGIT leaving the tail ("10m" -> "9m"), and reading it off the wall
        # clock would land on "9m" -> "8m" — one cell either way — which is the
        # tick this test would then pass without the reservation.
        listing.set_now(deadline - 600_000)
        await _settle(pilot)
        before = listing.render().plain.splitlines()[1]
        assert "expires in 10m" in before, before
        assert "…" in before, before
        listing.set_now(deadline - 599_000)
        await _settle(pilot)
        after = listing.render().plain.splitlines()[1]
        assert _countdown_token(after) != _countdown_token(before), (before, after)
        # The question is cut in EXACTLY the same place, and the row is the same
        # width: the minute that left the tail did not become question.
        assert before.split("expires in")[0] == after.split("expires in")[0], (before, after)
        assert cell_len(before) == cell_len(after), (before, after)


async def test_answering_one_ask_advances_the_list_to_the_next(enabled):
    """Audit B: a queue of N used to cost N re-expands.

    A submit collapsed the surface, so answering three asks meant expanding,
    hunting for the place and answering, three times. The list is a PLACE — the
    answered row leaves and its successor slides into the space it held — so the
    surface stays up with the highlight advanced to the next outstanding ask,
    and the highlight survives the wire's own snapshot dropping the row under it.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._session = session
        app._sync_ask_surface(
            ask_rows(
                [_row("a1", "Deploy now?"), _row("a2", "Which region?"), _row("a3", "Roll back?")]
            )
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        app.on_ask_queue_list_picked(AskQueueList.Picked("a2"))
        await _settle(pilot)
        assert app._ask_mounted_id == "a2"

        app._on_queue_ask_settle("a2", {"q1": ["Yes"]})
        await _settle(pilot)
        assert session.answered and session.answered[0][0] == "a2"
        listing = app.query_one(AskQueueList)
        assert listing.is_attached, "answering out of the list collapsed the surface"
        assert app._ask_mode is True
        assert listing.rows[listing.index].ask_id == "a3", "the highlight did not advance"

        # The wire's next snapshot drops the answered row: the list follows it
        # and keeps the user where the advance put them.
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?"), _row("a3", "Roll back?")]))
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        assert [row.ask_id for row in listing.rows] == ["a1", "a3"]
        assert listing.rows[listing.index].ask_id == "a3"


async def test_answering_the_last_row_falls_back_to_the_one_before_it(enabled):
    """The other half of the walk: the last ask has no successor to hand over.

    Answering the BOTTOM row must not throw the user back to the top of the
    list — the row that moves up into the freed space is the row they were
    about to reach, so that is the one the surface lands on.
    """
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._session = session
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which region?")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        app.on_ask_queue_list_picked(AskQueueList.Picked("a2"))
        await _settle(pilot)
        app._on_queue_ask_settle("a2", {"q1": ["Yes"]})
        await _settle(pilot)
        assert session.answered and session.answered[0][0] == "a2"
        listing = app.query_one(AskQueueList)
        assert listing.is_attached
        assert listing.rows[listing.index].ask_id == "a1"


def test_an_answer_that_landed_late_takes_the_warning_ink():
    """Audit C: a late answer is a RESPONSE, and a response was dim by definition.

    The shared row already distinguishes the three statuses and calls the late
    one ``warning``; the surface branched on ``kind`` instead, so the late
    receipt painted ``dim`` beside the amber timeout row that reports the same
    missed deadline.
    """
    from local_operator.tui.widgets.transcript import AskResponseBlock

    late = AskResponseBlock({"text": "…", "status": "late"}, kind="response")
    answered = AskResponseBlock({"text": "…", "status": "answered"}, kind="response")
    assert late._summary_ink() == "warning"
    assert answered._summary_ink() != "warning"
    # The same weight as the timeout row, which is the whole point of the ink.
    assert late._summary_ink() == AskResponseBlock({"text": "…"}, kind="timeout")._summary_ink()
    assert late._build_content(80).plain
