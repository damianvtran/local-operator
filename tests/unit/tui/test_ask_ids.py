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
from textual.content import Content
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
        rendered = prose._render()
        assert isinstance(rendered, Content)
        plain = rendered.plain
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


# -- `/ask <id>` (step 4) ----------------------------------------------------
#
# The typed door to the same reveal a click uses. Driven through the REAL
# editor and Enter, because the command's parsing lives in the submit path and
# calling the handler directly would skip the half a typo would land in.


async def _submit(pilot, app, text: str) -> None:
    """Type ``text`` into the composer and press Enter (the reported path)."""
    from local_operator.tui.widgets.editor import Editor

    editor = app.query_one(Editor)
    # The ask list takes the keyboard when it opens, so a second command typed
    # "into the composer" would land on the list's own Enter. Focus it the way
    # the user does — the composer is where a slash command is typed.
    editor.focus()
    editor.text = text
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


async def test_ask_alone_opens_the_list(enabled):
    from local_operator.tui.widgets.ask_queue import AskQueueList

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        app._sync_ask_surface(ask_rows([_row("a-e53b", "Ship it?"), _row("a-9f20", "Rotate?")]))
        await pilot.pause()
        await _submit(pilot, app, "/ask")
        assert len(app.query(AskQueueList)) == 1
        # A second `/ask` re-points the surface rather than closing it: the
        # toggle is f4's own, and a typed command that shut the list it had
        # just shown would read as a failure.
        await _submit(pilot, app, "/ask")
        assert len(app.query(AskQueueList)) == 1


async def test_ask_with_an_id_lands_the_cursor_on_that_ask(enabled):
    from local_operator.tui.widgets.ask_queue import AskQueueList

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a-e53b", "Ship it?"),
                    _row("a-9f20", "Rotate the relay token?"),
                ]
            )
        )
        await pilot.pause()
        await _submit(pilot, app, "/ask a-9f20")
        listing = app.query_one(AskQueueList)
        current = listing.current()
        assert current is not None and current.ask_id == "a-9f20"
        # The bare suffix is the same command: the prefix is what a person
        # reads past, not what they should have to retype.
        await _submit(pilot, app, "/ask e53b")
        landed = app.query_one(AskQueueList).current()
        assert landed is not None and landed.ask_id == "a-e53b"


async def test_ask_on_a_settled_id_switches_to_all_with_a_notice(enabled):
    from local_operator.i18n.keys import tui_asks as ask_keys
    from local_operator.i18n.text import tr
    from local_operator.tui.widgets.ask_queue import FILTER_ALL, AskQueueList
    from local_operator.tui.widgets.transcript import NoticeBlock

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a-e53b", "Ship it?"),
                    _row("a-745e", "Ship the relay fix tonight?", status="withdrawn"),
                ]
            )
        )
        await pilot.pause()
        await _submit(pilot, app, "/ask a-745e")
        listing = app.query_one(AskQueueList)
        assert listing.filter_value == FILTER_ALL
        current = listing.current()
        assert current is not None and current.ask_id == "a-745e"
        receipt = tr("tui.asks")(ask_keys.list_showing_all_settled(ask_id="a-745e"))
        assert app.query(NoticeBlock)[-1].text() == receipt


async def test_ask_on_an_unknown_id_says_so_and_opens_nothing(enabled):
    from local_operator.i18n.keys import tui_asks as ask_keys
    from local_operator.i18n.text import tr
    from local_operator.tui.widgets.ask_queue import AskQueueList
    from local_operator.tui.widgets.transcript import NoticeBlock

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        app._sync_ask_surface(ask_rows([_row("a-e53b", "Ship it?")]))
        await pilot.pause()
        await _submit(pilot, app, "/ask a-1234")
        assert len(app.query(AskQueueList)) == 0
        receipt = tr("tui.asks")(ask_keys.list_no_such_ask(ask_id="a-1234"))
        assert app.query(NoticeBlock)[-1].text() == receipt


async def test_an_id_from_another_conversation_is_not_looked_up(enabled):
    """Q1, decided: ids are unique per conversation only.

    The same id in another session's log is a DIFFERENT ask, so a fleet-wide
    search would open a stranger's question from a typed id — and the sentence
    the refusal prints says "in this conversation" for that reason.
    """
    from local_operator.i18n.keys import tui_asks as ask_keys
    from local_operator.i18n.text import tr
    from local_operator.tui.widgets.ask_queue import AskQueueList
    from local_operator.tui.widgets.transcript import NoticeBlock

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        app._ask_fleet_rows = ask_rows([_row("a-1234", "Someone else's question?", status="open")])
        app._sync_ask_surface(ask_rows([_row("a-e53b", "Ship it?")]))
        await pilot.pause()
        await _submit(pilot, app, "/ask a-1234")
        assert len(app.query(AskQueueList)) == 0
        receipt = tr("tui.asks")(ask_keys.list_no_such_ask(ask_id="a-1234"))
        assert app.query(NoticeBlock)[-1].text() == receipt


async def test_help_lists_the_ask_command(enabled):
    """/help names the command and says what it does — the one place a hint
    dropped from a narrow header still appears (design §3/§6)."""
    from rich.console import Console

    from local_operator.i18n.keys import wire_slash
    from local_operator.i18n.text import tr

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        # The block `/help` appends, rendered through the same rich console the
        # frame uses: the table is a GRID, so a substring check on a composed
        # string would not be the thing the user reads.
        console = Console(width=116, record=True, force_terminal=False)
        console.print(app._help_block().renderable)
        text = console.export_text()
        assert "/ask" in text, text[:2000]
        description = tr("wire.slash")(wire_slash.ask())
        assert description in text, (description, text[:2000])


# -- round 1 remediation: the PAINTED cells, not the built Text ---------------


async def test_the_painted_cells_carry_the_intended_ink(enabled):
    """D-1/Q1: Textual used to repaint our spans body-coloured and underlined.

    Textual overlays its link style on any span whose meta carries ``@click``
    (``textual/widget.py``'s ``_Styled``), which replaced the ink
    ``style_marks`` set — the id and all three marks painted body ink with an
    underline, so the state colour was gone. The unit tests read the ``Text``
    before that overlay, so they passed while the screen differed. This reads
    the COMPOSITOR: the cells the terminal is handed.
    """
    from local_operator.tui import theme as theme_mod
    from local_operator.tui.widgets.assistant import AssistantBlock

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a-e53b", "Ship it?"),
                    _row("a-745e", "Ship 0.71.2?", status="withdrawn"),
                    _row("a-1c09", "Smoke first?", status="answered"),
                ]
            )
        )
        await pilot.pause()
        prose = AssistantBlock()
        prose.update_text(
            "The old one, a-745e, is withdrawn; the new one is a-e53b. "
            "The relay smoke, a-1c09, ran. See https://example.test/x for the log."
        )
        app._append_block(prose)
        await pilot.pause()
        rendered = prose._render()
        assert isinstance(rendered, Content)
        plain = rendered.plain

        def painted(needle: str):
            at = plain.index(needle)
            line_no = plain[:at].count("\n")
            col = at - (plain.rfind("\n", 0, at) + 1)
            return app.screen.get_style_at(prose.region.x + col, prose.region.y + line_no)

        def hexed(value: str) -> str:
            return value.lower()

        signal = hexed(theme_mod.semantic_color("signal"))
        id_style = painted("a-745e")
        assert id_style.color is not None
        assert hexed(id_style.color.get_truecolor().hex) == signal, id_style
        assert id_style.underline, "the id lost its link underline"
        assert id_style.meta.get("@click") == "app.open_ask_id('a-745e')"
        # Each mark keeps its STATE ink, and none of them is underlined: the
        # underline is the link cue and belongs to the id alone.
        for mark, token in (("○", "muted"), ("●", "warning"), ("✓", "success")):
            style = painted(mark)
            assert style.color is not None
            assert hexed(style.color.get_truecolor().hex) == hexed(
                theme_mod.semantic_color(token)
            ), (mark, style, token)
            assert not style.underline, (mark, style)
            assert style.meta.get("@click"), mark
        # A real URL in the same paragraph is untouched: link ink, underlined,
        # and no click action (the transcript's own handler owns it).
        url_style = painted("https://example.test/x")
        assert url_style.color is not None
        assert hexed(url_style.color.get_truecolor().hex) == signal
        assert url_style.underline
        assert "@click" not in url_style.meta


# -- partial selections (M-1/Q2) ---------------------------------------------


async def test_a_partial_drag_never_copies_the_painted_mark(enabled):
    """M-1/Q2: the glyph path copied the NBSP and the mark.

    The decoration is cells the reader never typed. Three drags over one
    paragraph, at the exact columns QA reproduced them at, and a select-all.
    """
    from textual.geometry import Offset

    from local_operator.tui.widgets.assistant import AssistantBlock

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        app._sync_ask_surface(
            ask_rows([_row("a-745e", "Ship it?", status="withdrawn"), _row("a-e53b", "x")])
        )
        await pilot.pause()
        prose = AssistantBlock()
        prose.update_text("The old one, a-745e, is withdrawn; the new one is a-e53b.")
        app._append_block(prose)
        await pilot.pause()
        rendered = prose._render()
        assert isinstance(rendered, Content)
        plain = rendered.plain.split("\n")[0]
        at = plain.index("a-745e")
        nbsp = plain.index(NBSP)
        mark = nbsp + 1
        end = plain.index(";")

        def take(a: int, b: int) -> str:
            got = prose.get_selection(Selection.from_offsets(Offset(a, 0), Offset(b, 0)))
            assert got is not None
            return got[0]

        # id + mark + what follows: the id and the prose survive, the pair does not.
        assert take(at, mark + 3) == "a-745e,"
        # a drag starting inside the id and ending ON the mark.
        assert take(at + 2, mark + 1) == "745e"
        # a drag starting ON the mark: the orphan glyph goes too.
        assert take(mark, end) == ", is withdrawn"
        for fragment in ("a-745e,", "745e", ", is withdrawn"):
            assert NBSP not in fragment and not any(glyph in fragment for glyph in "●✓○")
        # Select-all is already clean (the markdown source path).
        whole = prose.get_selection(Selection(None, None))
        assert whole is not None and NBSP not in whole[0] and "○" not in whole[0]
        # `/copy` reads the message source, which never carried a mark.
        assert prose.text() == "The old one, a-745e, is withdrawn; the new one is a-e53b."


# -- the reveal returns to this conversation (M-2/Q3) ------------------------


async def test_a_reveal_with_the_fleet_list_open_lands_on_this_conversation(enabled):
    """M-2/Q3: the highlight was resolved inside FLEET scope.

    Ids are unique per session only, so the first fleet row with the id can be
    another conversation's question — one ``enter`` away from answering it.
    """
    from local_operator.tui.widgets.ask_queue import (
        SCOPE_FLEET,
        SCOPE_SESSION,
        AskQueueList,
    )

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        app._ask_fleet_rows = ask_rows(
            [_row("a-745e", "OTHER conversation: rotate keys?", session_id="sess-OTHER")]
        )
        app._ask_scope = SCOPE_FLEET
        app._sync_ask_surface(
            ask_rows([_row("a-745e", "This one is withdrawn?", status="withdrawn")])
        )
        await pilot.pause()
        app._reveal_ask("a-745e")
        await pilot.pause()
        listing = app.query_one(AskQueueList)
        current = listing.current()
        assert current is not None
        assert current.ask_id == "a-745e"
        # The row it landed on is THIS conversation's: the fleet copy carries a
        # session id, the session-scope rows do not.
        assert current.session_id == "", current
        assert current.head_question == "This one is withdrawn?", current
        assert app._ask_scope == SCOPE_SESSION
        assert app._ask_fleet_rows == []


async def test_an_id_missing_here_is_not_found_even_with_the_fleet_list_open(enabled):
    """The narrow half of Q3: the fleet rows must not answer for this one."""
    from local_operator.i18n.keys import tui_asks as ask_keys
    from local_operator.i18n.text import tr
    from local_operator.tui.widgets.ask_queue import SCOPE_FLEET, AskQueueList
    from local_operator.tui.widgets.transcript import NoticeBlock

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        app._ask_fleet_rows = ask_rows([_row("a-beef", "Another conversation's?")])
        app._ask_scope = SCOPE_FLEET
        app._sync_ask_surface(ask_rows([_row("a-e53b", "Ship it?"), _row("a-9f20", "x")]))
        await pilot.pause()
        app._reveal_ask("a-beef")
        await pilot.pause()
        assert len(app.query(AskQueueList)) == 0
        assert app.query(NoticeBlock)[-1].text() == tr("tui.asks")(
            ask_keys.list_no_such_ask(ask_id="a-beef")
        )


# -- the perf pre-check (review minor) ---------------------------------------


async def test_a_state_change_repaints_only_blocks_that_cite_an_id():
    """Review minor: 0.47 s on 300 blocks, most of them unable to draw a mark."""
    from local_operator.tui.widgets.assistant import AssistantBlock

    citing = AssistantBlock()
    citing.update_text("the old one, a-745e, is withdrawn")
    citing.set_ask_states({"a-745e": "open"})
    citing._flat_whole(80)

    silent = AssistantBlock()
    silent.update_text("nothing here is an ask id")
    silent.set_ask_states({"a-745e": "open"})
    silent._flat_whole(80)

    flattens = {"citing": 0, "silent": 0}

    def counting(name, block):
        real = block._flat_whole

        def wrapper(width=None):
            flattens[name] += 1
            return real(width)

        block._flat_whole = wrapper  # type: ignore[method-assign]

    counting("citing", citing)
    counting("silent", silent)
    citing.set_ask_states({"a-745e": "answered"})
    silent.set_ask_states({"a-745e": "answered"})
    assert flattens == {"citing": 1, "silent": 0}, flattens
    # The citing block really did repaint in place.
    assert f"a-745e{NBSP}✓" in citing._flat_whole(80).plain
    # ...and the silent one still took the map, so a later text change renders
    # the right state rather than a stale one.
    assert silent._ask_states == {"a-745e": "answered"}


# -- decorate/strip agreement (review minor) ---------------------------------


def test_a_backticked_id_round_trips_and_never_collects_two_marks():
    from local_operator.tui.ask_ink import strip_marks

    source = "ask `a-745e` or a-e53b, not `a-dead`"
    states = {"a-745e": "withdrawn", "a-e53b": "open"}
    once = decorate_source(source, states)
    assert once == f"ask `a-745e`{NBSP}○ or a-e53b{NBSP}●, not `a-dead`"
    # The second flush must find the mark AFTER the closing backtick.
    assert decorate_source(once, states) == once
    # ...and stripping the SOURCE shape gives the model's words back exactly.
    assert strip_marks(once) == source
    # The painted shape (no backticks) still strips.
    assert strip_marks(f"a-745e{NBSP}○") == "a-745e"


# -- the refusal echoes what was typed (nit) ---------------------------------


async def test_ask_zzz_reports_the_id_as_typed(enabled):
    from local_operator.i18n.keys import tui_asks as ask_keys
    from local_operator.i18n.text import tr
    from local_operator.tui.widgets.transcript import NoticeBlock

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        app._sync_ask_surface(ask_rows([_row("a-e53b", "Ship it?"), _row("a-9f20", "x")]))
        await pilot.pause()
        await _submit(pilot, app, "/ask zzz")
        # No invented `a-` prefix in the sentence: the user never typed one.
        assert app.query(NoticeBlock)[-1].text() == tr("tui.asks")(
            ask_keys.list_no_such_ask(ask_id="zzz")
        )


# -- round 2 remediation -----------------------------------------------------


def test_a_wrapped_pair_drops_its_orphan_mark():
    """N-1: the orphan check used a bare ``rstrip()``, and NBSP is whitespace.

    Rich may fold between the no-break space and the mark, leaving the mark at
    column 0 of the next row. The check that recognises it looked for a row
    ending in NBSP — through ``str.rstrip``, which strips NBSP itself, so it
    never fired and a wrapped id still copied its mark.
    """
    from local_operator.tui.ask_ink import without_marks

    # Every position the fold can land in for one pair.
    assert without_marks("x a-745e", start=0, end=-1) == "x a-745e"
    assert without_marks("x a-745e\u00a0", start=0, end=-1) == "x a-745e"
    assert without_marks("● tail", start=0, end=-1, previous_row="x a-745e\u00a0") == " tail"
    assert without_marks("● tail", start=0, end=-1, previous_row="x a-745e\u00a0 ") == " tail"
    # A row that merely ENDS in padding is not an orphan: the mark is the row's
    # own text and is kept.
    assert without_marks("● tail", start=0, end=-1, previous_row="plain row") == "● tail"


async def test_a_drag_on_a_wrapped_pair_copies_no_decoration(enabled):
    """The integration half of N-1, over a real rich fold.

    ``refit_width`` publishes the lane, so the rows below are rich's own fold at
    50 cells — one where the no-break space ends a row and the mark opens the
    next (measured; the pair rides the fold at exactly this width).
    """
    from textual.geometry import Offset

    from local_operator.tui.widgets.assistant import AssistantBlock

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        app._sync_ask_surface(
            ask_rows([_row("a-745e", "Ship it?", status="withdrawn"), _row("a-e53b", "x")])
        )
        await pilot.pause()
        prose = AssistantBlock()
        prose.update_text(
            "The old one, a-745e, is withdrawn and the relay smoke has already "
            "run; the new question is a-e53b and it is still open."
        )
        app._append_block(prose)
        await pilot.pause()
        prose.refit_width(50)
        rendered = prose._render()
        assert isinstance(rendered, Content)
        rows = rendered.plain.split("\n")
        # The fold this test exists for: NBSP ends one row, the mark opens the next.
        fold = next(
            i for i in range(len(rows) - 1) if rows[i].endswith(NBSP) and rows[i + 1][:1] in "●✓○"
        )
        got = prose.get_selection(Selection.from_offsets(Offset(0, fold + 1), Offset(12, fold + 1)))
        assert got is not None
        assert got[0].strip().startswith("and it is"), got
        # A drag that sweeps both rows stays clean too.
        got = prose.get_selection(Selection.from_offsets(Offset(0, fold), Offset(12, fold + 1)))
        assert got is not None
        assert NBSP not in got[0] and got[0][:1] not in "●✓○", got


def test_without_marks_slices_cells_not_characters():
    """Minor: a wide glyph's two cells were read as one character."""
    from local_operator.tui.ask_ink import without_marks

    # 日本語 is six cells, then a space: cell 7 is the `a`.
    assert without_marks("日本語 a x", start=7, end=-1) == "a x"
    assert without_marks("日本語 a x", start=8, end=-1) == " x"
    # ...and the removal path stays consistent with it.
    assert without_marks("日本語 a-745e\u00a0○ tail", start=9, end=-1) == "745e tail"


async def test_an_id_leaving_the_map_clears_its_mark(enabled):
    """N-2/Q5: the pre-check read the NEW map only, so a leaving id kept its mark."""
    from local_operator.tui.widgets.assistant import AssistantBlock

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await await_booted(app, pilot)
        app._sync_ask_surface(
            ask_rows([_row("a-e53b", "Ship it?"), _row("a-745e", "Old?", status="withdrawn")])
        )
        await pilot.pause()
        citing = AssistantBlock()
        citing.update_text("cites a-745e only")
        silent = AssistantBlock()
        silent.update_text("nothing cites an id here")
        app._append_block(citing)
        app._append_block(silent)
        await pilot.pause()
        assert f"a-745e{NBSP}○" in citing._flat_whole(120).plain
        calls = {"silent": 0}
        real = silent._flat_whole

        def counting(width=None):
            calls["silent"] += 1
            return real(width)

        silent._flat_whole = counting  # type: ignore[method-assign]
        # The id LEAVES the map: the block that cited it must repaint without it.
        app._sync_ask_surface(ask_rows([_row("a-e53b", "Ship it?")]))
        await pilot.pause()
        assert f"a-745e{NBSP}○" not in citing._flat_whole(120).plain
        assert calls["silent"] == 0, "a block citing no id was re-rendered"
        # ...and so must every block when the queue empties.
        app._sync_ask_surface(ask_rows([]))
        await pilot.pause()
        assert NBSP not in citing._flat_whole(120).plain
