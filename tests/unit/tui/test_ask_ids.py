"""Ask ids in agent prose: link ink, a one-cell state mark, and the click.

Design §3/§5 of the ask-id work (T-3, T-4). An agent names an ask BY ID — "the
old one, a-745e, is withdrawn" — and before this the id was plain prose: no
state, no route to the ask, and (backticked) an ink that looked like a link and
did nothing.

WHAT THESE TESTS ARE FOR. The three claims a still cannot make: the mark follows
the ASK's state at paint time, the mark costs exactly ONE cell (so an answer
never reflows the paragraph), and the painted id is a real click target driven
by the app's own click broker rather than a rich ``link=`` handed to the
terminal.
"""

from __future__ import annotations

import pytest
from rich.style import Style
from rich.text import Text
from textual.selection import Selection

from local_operator.tui.ask_ink import (
    NBSP,
    decorate_source,
    mark_for_status,
    strip_marks,
    style_marks,
)
from tests.unit.tui.conftest import await_booted
from tests.unit.tui.test_ask_queue_surface import _app, _AskSession, _row, ask_rows

pytestmark = pytest.mark.asyncio


@pytest.fixture
def enabled(monkeypatch):
    """The queued-ask feature ON, the way the env seam turns it on."""
    from local_operator.asks import policy
    from local_operator.tui import ask_open_policy

    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    # The open-by-default policy stays off for the same reason the queue
    # surface's file turns it off: these fixtures are dated 1970 and would be
    # read as pending-on-open, opening the surface before the test touches it.
    monkeypatch.setattr(ask_open_policy, "AUTO_OPEN", False)


#: One conversation's asks, as the app's ``_ask_mark_states`` map carries them.
STATES = {
    "a-745e": "withdrawn",
    "a-e53b": "open",
    "a-1c09": "answered",
}


# -- the source transform ----------------------------------------------------


def test_a_resolvable_id_gets_its_mark_and_an_unknown_one_does_not():
    text = "The old one, a-745e, is withdrawn. The new one is a-e53b. An a-9999 is not ours."
    painted = decorate_source(text, STATES)
    assert f"a-745e{NBSP}○" in painted
    assert f"a-e53b{NBSP}●" in painted
    # An id this conversation does not know stays exactly as the model wrote it:
    # ids are unique per conversation only, and the same a-9999 is a real ask in
    # another session's log.
    assert "a-9999" in painted and f"a-9999{NBSP}" not in painted


def test_the_decoration_is_idempotent_across_streaming_flushes():
    once = decorate_source("see a-e53b", STATES)
    twice = decorate_source(once, STATES)
    assert once == twice, (once, twice)


def test_a_fenced_code_block_is_left_alone_but_inline_code_is_decorated():
    text = "Ask `a-e53b` and note:\n\n" "```sh\n" "lop sessions asks a-e53b\n" "```\n"
    painted = decorate_source(text, STATES)
    assert f"`a-e53b`{NBSP}●" in painted
    assert "asks a-e53b\n" in painted
    assert f"a-e53b{NBSP}" not in painted.split("```")[1]


def test_an_id_inside_a_url_is_not_a_citation():
    text = "see https://example.test/a-e53b for the log"
    assert decorate_source(text, STATES) == text


def test_a_word_that_merely_looks_like_an_id_is_only_decorated_when_it_resolves():
    text = "a-cafe is a word here, and a ab-cafe is not a token"
    assert decorate_source(text, STATES) == text
    resolved = decorate_source("a-cafe", {"a-cafe": "open"})
    assert resolved == f"a-cafe{NBSP}●"


def test_each_state_gets_its_own_one_cell_mark():
    assert mark_for_status("open") == "●"
    assert mark_for_status("answered") == "✓"
    assert mark_for_status("late") == "✓"
    for closed in ("withdrawn", "declined", "dismissed", "expired", "timed_out"):
        assert mark_for_status(closed) == "○", closed
    # Unknown statuses cannot ask the reader for anything.
    assert mark_for_status("something_new") == "○"


def test_the_mark_is_one_cell_so_a_state_change_never_reflows():
    waiting = decorate_source("the ask a-e53b", {"a-e53b": "open"})
    answered = decorate_source("the ask a-e53b", {"a-e53b": "answered"})
    from rich.cells import cell_len

    assert cell_len(waiting) == cell_len(answered)


# -- the paint ---------------------------------------------------------------


def _styled(marked: str, action=lambda ask_id: f"app.open_ask_id({ask_id!r})") -> Text:
    text = Text(marked)
    style_marks(text, action=action)
    return text


def test_the_id_takes_the_link_ink_underline_and_the_click_action():
    text = _styled(f"x a-e53b{NBSP}●")
    at = text.plain.index("a-e53b")
    span = next(span for span in text.spans if span.start <= at < span.end)
    style = span.style
    assert isinstance(style, Style)
    from local_operator.tui import theme as theme_mod

    colour = style.color
    assert colour is not None
    resolved = colour if hasattr(colour, "get_truecolor") else None
    assert resolved is not None
    assert resolved.get_truecolor().hex.lower() == (theme_mod.semantic_color("signal").lower())
    assert style.underline
    # The click target is Textual ACTION META, never a rich link: a link makes
    # Textual emit an OSC-8 URL, and a `lop-ask:` scheme would then be offered
    # to the desktop as something to open.
    assert style.meta == {"@click": "app.open_ask_id('a-e53b')"}
    assert getattr(style, "link", None) is None


def test_the_mark_takes_the_state_ink():
    from local_operator.tui import theme as theme_mod

    text = _styled(f"a-e53b{NBSP}●")
    at = text.plain.index("●")
    span = next(span for span in text.spans if span.start <= at < span.end)
    style = span.style
    assert isinstance(style, Style) and style.color is not None
    assert style.color.get_truecolor().hex.lower() == theme_mod.semantic_color("warning").lower()


def test_an_action_of_none_leaves_the_id_unpainted():
    text = _styled(f"a-e53b{NBSP}●", action=lambda ask_id: None)
    assert text.spans == []


def test_strip_marks_returns_the_models_words():
    marked = f"the old one, a-745e{NBSP}○, is withdrawn"
    assert strip_marks(marked) == "the old one, a-745e, is withdrawn"
    # Text with no marks at all is returned untouched.
    assert strip_marks("plain prose") == "plain prose"


# -- the block ---------------------------------------------------------------


def test_the_block_decorates_only_when_the_app_has_the_ask():
    from local_operator.tui.widgets.assistant import AssistantBlock

    block = AssistantBlock()
    block.update_text("the old one, a-745e, is withdrawn")
    fresh = block._flat_whole(80).plain
    assert f"a-745e{NBSP}" not in fresh
    block.set_ask_states(STATES)
    painted = block._flat_whole(80).plain
    assert f"a-745e{NBSP}○" in painted
    # The MESSAGE is untouched: /copy reads `text()`, and the mark is paint.
    assert block.text() == "the old one, a-745e, is withdrawn"


def test_a_state_change_repaints_the_marks_in_place():
    from local_operator.tui.widgets.assistant import AssistantBlock

    block = AssistantBlock()
    block.update_text("waiting on a-e53b")
    block.set_ask_states({"a-e53b": "open"})
    assert f"a-e53b{NBSP}●" in block._flat_whole(80).plain
    block.set_ask_states({"a-e53b": "answered"})
    assert f"a-e53b{NBSP}✓" in block._flat_whole(80).plain


def test_a_selection_copy_drops_the_painted_mark():
    from local_operator.tui.widgets.assistant import AssistantBlock

    block = AssistantBlock()
    block.update_text("the old one, a-745e, is withdrawn")
    block.set_ask_states(STATES)
    block._flat_whole(80)
    copied = block.get_selection(Selection(None, None))
    assert copied is not None
    body, _sep = copied
    assert "a-745e" in body
    assert NBSP not in body and "○" not in body


# -- the click, through the real app -----------------------------------------


async def test_clicking_a_cited_id_opens_the_list_on_that_ask(enabled):
    """The whole route: painted meta → the app's click broker → the list.

    The style is read back from the REAL compositor at the id's own cells (so
    the meta is provably on the painted cell, not on a Text the test built), and
    the click is brokered through the widget's own ``broker_event`` — the call
    ``Widget._on_click`` makes — because Textual's pilot posts mouse events
    WITHOUT the style the screen fills in, so a pilot click would exercise a
    broker with nothing to read.
    """
    from textual import events
    from textual.actions import parse

    from local_operator.tui.widgets.ask_queue import AskQueueList
    from local_operator.tui.widgets.assistant import AssistantBlock

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a-e53b", "Ship the relay fix tonight?"),
                    _row("a-745e", "Ship the relay fix as 0.71.2?", status="withdrawn"),
                ]
            )
        )
        await pilot.pause()
        prose = AssistantBlock()
        prose.update_text("The old one, a-745e, is withdrawn.")
        app._append_block(prose)
        await pilot.pause()
        plain = prose._render().plain
        assert f"a-745e{NBSP}○" in plain, plain
        at = plain.index("a-745e")
        row = plain[:at].count("\n")
        col = at - (plain.rfind("\n", 0, at) + 1)
        x, y = prose.region.x + col, prose.region.y + row
        # The PAINTED cell carries the action, and the mark beside it too.
        style = app.screen.get_style_at(x, y)
        assert style.meta.get("@click") == "app.open_ask_id('a-745e')", style.meta
        assert app.screen.get_style_at(x + len("a-745e") + 1, y).meta.get("@click") == (
            "app.open_ask_id('a-745e')"
        )
        # ...and it is a string Textual's own parser accepts, into this app's
        # action with the id as its argument.
        assert parse(style.meta["@click"]) == ("app", "open_ask_id", ("a-745e",))

        click = events.Click(
            widget=prose,
            x=col,
            y=row,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=x,
            screen_y=y,
            style=style,
        )
        await prose.broker_event("click", click)
        await pilot.pause()
        listing = app.query_one(AskQueueList)
        current = listing.current()
        assert current is not None and current.ask_id == "a-745e"
