"""The `/goal` overlay: the body it prints, and the keys it binds.

Two halves, and both matter. The BODY is a pure function of the record, so it is
asserted directly — including the one thing the operator asked for by name, a
REAL strikethrough (an SGR 9 style on the goal's own span, not a colour or a tag
standing in for it) and its fit at 80 and 100 columns. The WIRING is asserted
through the app, because "`/goal` with no argument opens this" and "`d` strikes
the goal" are claims about the command and the keymap, not about a formatter.

The keys are audited rather than guessed: the panel is the focused widget while
it is open, so `d`/`c` are consulted before the app's bindings — and neither is
added to the REMAPPABLE, PERSISTED vocabulary in `local_operator/keymap.py`,
which is what would make a new global action a config migration.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from rich.style import Style
from textual.content import Content

from local_operator.session.goal_judge import MAX_GOAL_CONTINUATIONS
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.goal_panel import (
    MAX_HISTORY_ROWS,
    PANEL_PADDING_CELLS,
    PANEL_PADDING_ROWS,
    GoalPanel,
    build_goal_body,
    clamp_history_rows,
    judge_line,
)

from .test_app_pilot import FakeSession, _factory

GOAL = "land the OAuth refresh fix"


def _card_body(panel: GoalPanel) -> Content:
    """The content the card is carrying, narrowed for the type gate.

    ``Widget.render()`` is typed as a union of renderables, so reaching `.plain`
    or `.spans` through it is a type error even though this card always renders a
    ``Content`` — and the isinstance is not decoration, it is the claim every card
    assertion here rests on. (It renders a ``Content`` rather than the rich
    ``Text`` the formatter builds: Textual wraps what a ``Static`` is given.)
    """
    rendered = panel.render()
    assert isinstance(rendered, Content)
    return rendered


def _card_text(panel: GoalPanel) -> str:
    """The card's text. Kept apart from the renderable so the difference the
    frames taught us stays explicit: this is what is IN THE BODY, and a row's
    index against `panel.region.height` is what is inside the PAINTED BOX."""
    return _card_body(panel).plain


def _entry(text: str, status: str = "done") -> dict[str, str]:
    return {
        "id": f"id-{text}",
        "text": text,
        "status": status,
        "created_at": "2026-09-20T10:00:00+00:00",
        "settled_at": "2026-09-21T11:30:00+00:00",
        "reason": "the judge said so" if status == "done" else "",
    }


def _binding_keys() -> set[str]:
    """The keys the card binds, from a tuple OR a ``Binding`` entry.

    ``BINDINGS`` is typed as a union of both spellings (Textual accepts either),
    so unpacking it as tuples is a type error even though this class declares
    only tuples — and the audit below is exactly the kind of thing that should
    keep reading them.
    """
    keys: set[str] = set()
    for entry in GoalPanel.BINDINGS:
        if isinstance(entry, tuple):
            keys.add(entry[0])
        else:
            keys.add(entry.key)
    return keys


def _struck_spans(body) -> list[str]:  # noqa: ANN001
    """The text carried by spans whose style actually STRIKES it."""
    out: list[str] = []
    for span in body.spans:
        style = span.style
        if style is not None and getattr(style, "strike", False):
            out.append(body.plain[span.start : span.end])
    return out


def _body(**kwargs: object):  # noqa: ANN201
    params = {
        "goal": GOAL,
        "status": "active",
        "judge": {"state": "continuing", "run": 2, "verdict": "continue", "reason": "more to do"},
        "history": [],
        "width": 80,
        "cap": MAX_GOAL_CONTINUATIONS,
    }
    params.update(kwargs)
    return build_goal_body(**params)  # type: ignore[arg-type]


# --- the body ----------------------------------------------------------------


def test_an_active_goal_is_shown_unstruck_with_its_judge_state() -> None:
    body = _body()
    assert GOAL in body.plain
    assert "— active" in body.plain
    assert "judge: continuing" in body.plain
    assert f"run 2/{MAX_GOAL_CONTINUATIONS}" in body.plain
    assert GOAL not in _struck_spans(body), "an active goal must not be struck"


def test_a_done_goal_is_really_struck_through() -> None:
    """The operator's ask: `Style(strike=True)`, not a colour or a tag alone."""
    body = _body(status="done", judge={"state": "done", "verdict": "achieved", "reason": "done"})
    assert "— done" in body.plain
    struck = _struck_spans(body)
    assert GOAL in struck
    # ...and the TAG is not struck, so a struck row stays readable — the rule the
    # to-do rows already follow.
    assert not any("— done" in text for text in struck)


def test_only_finished_history_rows_are_struck() -> None:
    """`superseded` was replaced, not completed — a strike would claim otherwise."""
    history = [_entry("the finished one"), _entry("the abandoned one", "superseded")]
    body = _body(status="", judge=None, history=history)
    struck = _struck_spans(body)
    assert "the finished one" in struck
    assert "the abandoned one" not in struck
    assert "superseded" in body.plain


def test_no_goal_says_so_and_offers_the_command() -> None:
    body = _body(goal="", status="", judge=None)
    assert "no goal set" in body.plain
    assert "/goal <text>" in body.plain
    assert "settled  none yet" in body.plain


def test_a_follower_is_not_offered_the_two_actions() -> None:
    """The keys WRITE the record; a reader must not be taught a refused key."""
    watcher = _body(actions=False)
    assert "d done" not in watcher.plain
    assert "c clear" not in watcher.plain
    assert "q close" in watcher.plain
    assert "d done" in _body().plain


def test_the_history_list_is_bounded_and_says_what_it_dropped() -> None:
    history = [_entry(f"goal {index}") for index in range(MAX_HISTORY_ROWS + 3)]
    body = _body(status="", judge=None, history=history)
    assert "3 more settled" in body.plain
    assert body.plain.count("✓ ") == MAX_HISTORY_ROWS


def test_the_body_fits_its_width_at_80_and_100_columns() -> None:
    """The panel's two supported widths, measured in CELLS (not characters).

    ``width`` here is the CONTENT box (the caller takes the padding off), and the
    property that matters is that NOTHING WRAPS: a line one cell too long takes a
    second row, and a card that pinned its height to its line count then loses
    its last row off the bottom.
    """
    history = [_entry("a settled goal with a reasonably long name")]
    for screen in (80, 100):
        width = screen - PANEL_PADDING_CELLS
        body = _body(width=width, history=history)
        longest = max(cell_len(line) for line in body.plain.split("\n"))
        assert longest <= width, f"{screen}: {longest} cells in a {width}-cell box"


def test_clamp_history_rows_never_takes_the_card_off_the_screen() -> None:
    assert clamp_history_rows(MAX_HISTORY_ROWS, available_rows=40) == MAX_HISTORY_ROWS
    assert clamp_history_rows(MAX_HISTORY_ROWS, available_rows=12) < MAX_HISTORY_ROWS
    assert clamp_history_rows(MAX_HISTORY_ROWS, available_rows=0) == 0
    # A SQUEEZED card has spent its gutter, so the same ground holds more rows:
    # a budget still charging for a gutter the sheet has taken away shows fewer
    # settled goals than the terminal has room for.
    squeezed = clamp_history_rows(MAX_HISTORY_ROWS, available_rows=12, gutter_rows=0)
    assert squeezed > clamp_history_rows(MAX_HISTORY_ROWS, available_rows=12)


def test_the_judge_line_names_the_wire_state_and_labels_the_fraction() -> None:
    """One vocabulary, one separator, and a fraction that says of what.

    The card called the same state `stopped` in the minute the notice called it
    `stalled`, joined a ` · ` onto a clause that already opened with `— `, and
    printed a bare `2/12` (design D5/D7, UX U8). The state word is the WIRE's now.
    """
    line = judge_line(
        {
            "state": "stalled",
            "run": MAX_GOAL_CONTINUATIONS,
            "reason": "stopped after 12 continuations",
        },
        cap=MAX_GOAL_CONTINUATIONS,
    )
    assert line.plain == (
        f"judge: stalled · run {MAX_GOAL_CONTINUATIONS}/{MAX_GOAL_CONTINUATIONS}"
        " — stopped after 12 continuations"
    ), "no `· —` doubling, and the fraction is labelled"
    # THE ONE STATE THAT ASKS SOMETHING IS NOT PAINTED LIKE THE PASSIVE ONES: a
    # resolved theme colour for the stall, plain `dim` for everything else.
    assert isinstance(line.style, Style) and line.style.color is not None
    assert judge_line({"state": "waiting"}, cap=3).style == "dim"
    # An idle judge has not RUN; it is not a card with no goal on it (UX U7).
    assert judge_line({"state": "idle"}, cap=3).plain == "judge: idle (the judge has not run)"
    # An unknown member of the closed vocabulary (a newer writer's) prints as
    # itself rather than being dropped or crashing the card.
    assert judge_line({"state": "pondering"}, cap=3).plain == "judge: pondering"
    assert judge_line(None, cap=3).plain == "judge: —"


def test_the_action_keys_are_the_panels_own_and_not_global() -> None:
    """Audited, not assumed: a global action would be a persisted keymap id."""
    from local_operator import keymap

    bindings = _binding_keys()
    assert {"d", "c", "q", "escape"} <= bindings
    remappable = {action.id for action in keymap.KEY_ACTIONS}
    assert all(f"keymap.{key}" not in remappable for key in bindings)
    # The card is focusable: a binding on a widget nothing can focus is dead.
    assert GoalPanel.can_focus is True


# --- the wiring --------------------------------------------------------------


def _armed() -> FakeSession:
    session = FakeSession()
    session.arm_goal(GOAL)
    return session


@pytest.mark.asyncio
async def test_the_bare_command_opens_the_overlay() -> None:
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        # The LOCAL handler is what a TUI runs for its own session.
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        panel = app.query_one(GoalPanel)
        assert panel.is_open
        assert GOAL in _card_text(panel)
        assert app.focused is panel


@pytest.mark.asyncio
async def test_the_done_key_strikes_the_goal_and_records_it() -> None:
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        assert session.goal_status == "done"
        assert [row["text"] for row in session.history_view()] == [GOAL]
        panel = app.query_one(GoalPanel)
        assert panel.is_open, "the card stays up so the struck state is visible"
        body = _card_body(panel)
        assert "— done" in body.plain
        assert GOAL in _struck_spans(body), "the done goal must be struck on the card"


@pytest.mark.asyncio
async def test_the_clear_key_arms_first_and_only_the_second_press_erases() -> None:
    """DESIGN-UX §2.1: a recorded act is one press; an erased act is two.

    The card fired an immediate erase on ONE `c` — a key directly under `d` on a
    QWERTY board, on a surface that holds focus, with no undo and no history entry
    to recover from (design D4 / UX U2). The first press must do nothing but
    rehearse.
    """
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        # ARMED, and nothing has happened to the record.
        assert session.goal == GOAL, "one press must not erase anything"
        assert session.goal_status == "active"
        panel = app.query_one(GoalPanel)
        assert "c again to clear · esc cancels" in _card_text(panel)
        await pilot.press("c")
        await pilot.pause()
        assert session.goal == ""
        assert session.goal_status == ""
        # CLEAR records nothing: that is the whole difference from mark-done.
        assert session.history_view() == []


@pytest.mark.asyncio
async def test_escape_cancels_an_armed_clear() -> None:
    """The rehearsal has to be abandonable, or it is just a slower delete."""
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert session.goal == GOAL
        # Reopening the card starts UNARMED: an arm that outlived the card would
        # fire the next `c` at a goal the user has not rehearsed deleting.
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        panel = app.query_one(GoalPanel)
        assert "c again to clear" not in _card_text(panel)
        assert session.goal == GOAL


@pytest.mark.asyncio
async def test_the_card_is_tall_enough_for_its_own_content() -> None:
    """The border-box trap: a pinned height that forgets the gutter CLIPS rows.

    Measured on the first captured frame of this card, and the reason this test
    exists: it painted its title, rule, goal and judge rows and then ran out,
    with `settled` and the key hint invisible while still occupying height. A
    test that only read `render().plain` saw nothing wrong — the content was
    right and the box was too small — so the assertion has to be about the BOX.
    """
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        panel = app.query_one(GoalPanel)
        rows = _card_text(panel).split("\n")
        # `region` is the OUTER box: the content plus the stylesheet's own two
        # padding rows, which is what Textual's border-box sizing hands out.
        assert panel.region.height >= len(rows) + PANEL_PADDING_ROWS
        assert panel.region.width <= 100
        # AND NO LINE WRAPS. A line wider than the content box takes a second
        # row, which is the same clipping by another route — measured on the same
        # captured frame, where a `─` rule built to the OUTER width wrapped and
        # pushed the key hint out of the card.
        for line in rows:
            assert cell_len(line) <= panel.content_size.width, line[:40]


@pytest.mark.asyncio
async def test_escape_closes_the_card_and_returns_focus() -> None:
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.query_one(GoalPanel).is_open is False
        assert app.focused is not None

    for width in (80, 100):
        session = _armed()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(width, 40)) as pilot:
            await pilot.pause()
            app._cmd_goal("", lambda body, kind="info": None)
            await pilot.pause()
            region = app.query_one(GoalPanel).region
            assert region.width <= width, f"{width}: card is {region.width} cells wide"
            assert region.height > 0


# --- the findings that needed the REAL card ---------------------------------


def test_a_done_card_at_80_columns_does_not_wrap_its_hint_row_away() -> None:
    """Design D2's fixture: what the old width test's 26-cell goal hid.

    The goal row budgeted `inner - 2` for its text and then appended `  — done`
    and, on a settled row, a 19-cell stamp — so at 80 columns a realistic 54-cell
    goal measured 74 cells in a 66-cell box, wrapped, and took the key hint off
    the card entirely. The old test passed because its fixture goal was short.
    """
    long_goal = " ".join(["objective"] * 6)[:54]
    assert len(long_goal) == 54
    for screen in (80, 100):
        width = screen - PANEL_PADDING_CELLS
        body = _body(
            width=width,
            goal=long_goal,
            status="done",
            judge={"state": "done", "verdict": "achieved", "reason": "met"},
            history=[_entry(long_goal), _entry("the replaced one", "superseded")],
        )
        for line in body.plain.split("\n"):
            assert cell_len(line) <= width, f"{screen}: {cell_len(line)} cells in {width}"
        # ...and the rows that used to vanish are still on the card.
        assert "c dismiss · q close" in body.plain
        assert "2026-09-21 11:30" in body.plain


@pytest.mark.asyncio
async def test_the_card_counts_the_settled_goals_it_did_not_paint() -> None:
    """Design D1 / UX U1 through the APP's own repaint.

    The unit test that used to cover this called the formatter directly, so it
    stayed green while the card painted six of eleven settled goals and said
    nothing about the five it dropped — the widget pre-sliced the list, which made
    the formatter's dropped-rows branch unreachable in the app.
    """
    session = _armed()
    for index in range(11):
        session.arm_goal(f"objective number {index}")
        session.mark_goal_done()
    session.arm_goal(GOAL)

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        total = len(session.history_view())
        assert total > MAX_HISTORY_ROWS
        panel = app.query_one(GoalPanel)
        assert f"… {total - MAX_HISTORY_ROWS} more settled" in _card_text(panel)


@pytest.mark.asyncio
async def test_a_short_terminal_says_the_list_was_clipped_rather_than_none() -> None:
    """The worse half of D1: `settled  none yet` over a record with eleven.

    Measured at 80×14, where the card's ground is seven rows: the budget for
    settled rows is zero, and the OLD card turned that into a claim that the
    record holds none. The notice row is reserved out of the same budget now, and
    this asserts it is inside the PAINTED box rather than merely in the body —
    the clipped rows are the ones a screenshot would not show.
    """
    session = _armed()
    for index in range(11):
        session.arm_goal(f"objective number {index}")
        session.mark_goal_done()
    session.arm_goal(GOAL)

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 14)) as pilot:
        await pilot.pause()
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        panel = app.query_one(GoalPanel)
        body = _card_text(panel)
        assert "none yet" not in body, "the record has eleven; the card must not deny them"
        assert "more settled" in body
        # ...and it is PAINTED here: with no room for a row, the notice rides the
        # heading, which is the row that fits where two did not.
        lines = body.split("\n")
        notice = next(index for index, line in enumerate(lines) if "more settled" in line)
        assert notice < panel.region.height, "the notice must be inside the painted box"
        # On the ground that cannot hold the notice either, the card is clipped
        # from the bottom rather than allowed to paint over the composer — the
        # chrome (title, goal, judge, heading) is the floor it keeps.
        assert panel.region.y + panel.region.height <= app.query_one("#input-shell").region.y


@pytest.mark.asyncio
async def test_the_clip_notice_is_painted_once_the_ground_can_hold_it() -> None:
    """The other half of D1, measured inside the box rather than in the body.

    At 80×16 the ground holds the notice, and this asserts the row is PAINTED —
    index < height — because a row the body carries but the box clips is a row no
    user reads, which is the distinction design round 1's frames were made with.
    """
    session = _armed()
    for index in range(11):
        session.arm_goal(f"objective number {index}")
        session.mark_goal_done()
    session.arm_goal(GOAL)

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 16)) as pilot:
        await pilot.pause()
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        panel = app.query_one(GoalPanel)
        lines = _card_text(panel).split("\n")
        notice = next(index for index, line in enumerate(lines) if "more settled" in line)
        assert notice < panel.region.height, "the notice must be inside the painted box"


@pytest.mark.asyncio
async def test_a_clipped_record_says_so_on_the_card() -> None:
    """`goal_history_truncated` had no consumer anywhere under `local_operator/tui/`.

    The wire drops a record's settled list when the frame cannot carry it, and the
    flag is what keeps a reader from taking an empty list for "no completed
    goals". The card reads it off the session's frontend state, the same way the
    footer reads its catalogue sibling.
    """
    from types import SimpleNamespace

    session = _armed()
    # The flag the wire sets when a frame could not carry the list; read off the
    # session's frontend state, exactly as the footer reads its catalogue sibling.
    session.frontend_state = SimpleNamespace(  # type: ignore[attr-defined]
        goal_history_truncated=True
    )

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        panel = app.query_one(GoalPanel)
        body = _card_text(panel)
        assert "older settled goals were dropped from this record" in body
        assert "none yet" not in body, "a clipped record cannot be reported as empty"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 14), (80, 12), (60, 12)])
async def test_the_card_never_covers_the_docked_composer(size: tuple[int, int]) -> None:
    """Design D3 / UX U5, asserted as geometry rather than left to the clamp.

    The card pinned NINE rows regardless of the ground and covered `#input-dock`
    at every one of these sizes — with only the `❯` chevron left visible beside
    it — while the sibling usage card spent its gutter and stayed clear. The
    invariant is the helper's own sentence: the card covers no docked surface.
    """
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        panel = app.query_one(GoalPanel)
        card_bottom = panel.region.y + panel.region.height
        shell = app.query_one("#input-shell")
        assert (
            card_bottom <= shell.region.y
        ), f"{size}: card bottom {card_bottom} covers the composer at {shell.region.y}"
        # ...and it is the SQUEEZE that bought that, not a card that simply
        # refused to paint its own content.
        assert panel.has_class("-squeezed")
