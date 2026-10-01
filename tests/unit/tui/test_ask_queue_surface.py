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

from typing import Any

import pytest

from local_operator.asks import policy
from local_operator.tui.app import ASK_ANSWER_PLACEHOLDER, OperatorApp
from local_operator.tui.widgets.ask_picker import AskPickerScreen
from local_operator.tui.widgets.ask_queue import (
    ASK_BAR_CHEVRON_COLLAPSED,
    ASK_BAR_CHEVRON_EXPANDED,
    ASK_MARKER,
    STATUS_OPEN,
    AskBar,
    AskQueueList,
    ask_rows,
)
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

    def respond_ask(self, ask_id, answers, *, by="unknown"):
        self.answered.append((ask_id, dict(answers), by))
        error = self.refusals.get(ask_id)
        return {"ok": not error, **({"error": error} if error else {})}

    def decline_ask(self, ask_id, *, by="unknown"):
        self.declined.append((ask_id, by))
        return {"ok": True}

    def dismiss_ask(self, ask_id, *, by="unknown"):
        self.dismissed.append((ask_id, by))
        return {"ok": True}


@pytest.fixture
def enabled(monkeypatch):
    """The queued-ask feature ON for this test, the way the env seam turns it on."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)


def _app(session: _AskSession) -> OperatorApp:
    return OperatorApp(lambda: _factory(session))


async def _settle(pilot, turns: int = 3) -> None:
    for _ in range(turns):
        await pilot.pause()


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
        assert sidebar._asking == (session.session_id, 1)
        app._sync_ask_surface([])
        await _settle(pilot)
        assert sidebar._asking == (session.session_id, 0)


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
