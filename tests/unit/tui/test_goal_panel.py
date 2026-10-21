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

import asyncio
import time

import pytest
from rich.cells import cell_len
from rich.style import Style
from textual.content import Content

from local_operator.session.goal_judge import MAX_GOAL_CONTINUATIONS
from local_operator.tui import app as app_module
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
        # Content delivery can trail the key's state change under shard load;
        # assert the rendered rehearsal, not the first (possibly empty) frame.
        deadline = time.monotonic() + 5
        while "c again to clear · esc cancels" not in _card_text(panel):
            assert time.monotonic() < deadline, "the armed clear hint never rendered (5s)"
            await pilot.pause()
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
        # THE ADOPTION PRECONDITION, not a longer wait: `_cmd_goal` returns at
        # its own `session is None` guard, so a `/goal` issued before adoption
        # never opens the card at all and `_card_text` is `''` — the exact
        # `assert '… dropped …' in ''` this cell recorded on main. The sibling
        # tests above already wait through `_open_card` for this reason; this
        # one was the only caller that still issued the command after a single
        # `pause()`. Waiting for the card that is about to be measured is the
        # precondition, not a flake workaround.
        panel = await _open_card(pilot, app, session)
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
        # THE CARD HAS TO EXIST BEFORE ITS GEOMETRY MEANS ANYTHING. The failure
        # this test reported under CI load ("the card never settled its squeeze
        # decision": fit=(5, 0, 0), squeezed=False, region=0x0, display=False)
        # was NOT a repaint losing a race with the assertion: the app had not
        # adopted the session yet, so `/goal` hit its own `session is None`
        # guard and returned before ever opening the card (`_open_goal_panel`
        # never ran — a probe counted zero refresh calls on that path). The card
        # was never shown, so the settle loop below spent its whole budget
        # waiting for a class on a card that did not exist.
        panel = await _open_card(pilot, app, session)
        shell = app.query_one("#input-shell")
        # WAIT FOR THE DECISION, NOT A FRAME. `_fit` reads the DOCK's laid-out
        # height, so the first repaint (at mount) can see a ground the final dock
        # never leaves. `on_resize` repaints when that settles; one `pause()`
        # asserted the class before the repaint had run under CI load — the failure
        # was always the class, never the geometry (run 39286870604, job
        # 118199298717, `[size1]`). This waits for the class to agree with the
        # fit that is now in force.
        for _ in range(200):
            rows, gutter, _shown = panel._fit()
            if panel.region.height and panel.has_class("-squeezed") == (gutter == 0):
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        else:
            raise AssertionError(
                f"{size}: the card never settled its squeeze decision: "
                f"fit={panel._fit()!r}, squeezed={panel.has_class('-squeezed')}, "
                f"region={panel.region!r}, display={panel.display}, "
                f"styles={panel.styles.width!r}x{panel.styles.height!r}, app={app.size!r}"
            )
        card_bottom = panel.region.y + panel.region.height
        assert (
            card_bottom <= shell.region.y
        ), f"{size}: card bottom {card_bottom} covers the composer at {shell.region.y}"
        # ...and the class tells the truth about the FIT that produced this: below
        # the squeeze threshold the gutter is spent, at or above it the card can
        # afford the gutter and is not required to squeeze. Asserting the class is
        # ALWAYS present was the test assuming a ground that 80x12 does not always
        # have, not the invariant the card owes.
        rows, gutter, _shown = panel._fit()
        assert panel.has_class("-squeezed") == (
            gutter == 0
        ), f"{size}: squeeze class does not match fit rows={rows} gutter={gutter}"


@pytest.mark.asyncio
async def test_a_resize_while_open_repaints_the_card_for_the_new_ground() -> None:
    """A terminal resize must re-fit the card, not keep its old ground.

    THE SIBLING'S RULE (``UsagePanel.on_resize``) that this card lacked. The card
    is opened at 100x40, where it can afford its gutter, then the terminal shrinks
    to 80x12, where it cannot. Without a resize repaint both the pinned height and
    the squeeze class stay at the 100x40 decision — on a shorter ground that puts
    the card over the docked composer, which is the one invariant the card owes.
    """
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        # The same adoption precondition as the sibling above: a card opened
        # before the app adopted the session is never shown, so the gutter
        # assertion would describe a card that does not exist. (This test's own
        # recorded failure on `main` was a DIFFERENT signature - "card bottom 22
        # covers the composer at 6" after the resize - which this precondition
        # does not explain and does not claim to fix.)
        panel = await _open_card(pilot, app, session)
        assert panel.has_class("-squeezed") is False, "the wide card should afford its gutter"

        await pilot.resize_terminal(80, 12)
        shell = app.query_one("#input-shell")
        # The OLD wait broke the moment `_fit()` and `-squeezed` AGREED, and
        # before the resize was delivered they agree about the OLD ground: the
        # panel still reports `(29, 2, …)`, the class is still absent, and the
        # loop returned on its first pass. That is the recorded main failure —
        # `card bottom 22 covers the composer at 6` — measured before either
        # number had a chance to change (this test's own archived CI log shows
        # the app already at 80x12 by the time the assertion ran).
        #
        # So the wait now has three parts: the app must REPORT the new size, the
        # card must have re-fit to it, and the invariant the card owes — never
        # cover a docked surface — is asserted on EVERY pass, not only at the
        # end. The product's `ComposerDock.BoxChanged` seam repaints the card
        # from the dock's OWN resize (after its region is real) rather than only
        # from the 50 ms debounce, so the invariant holds from the first settled
        # frame; a regression that returns to measuring the pre-resize dock
        # fails here on the first pass instead of surviving to the final assert.
        for _ in range(200):
            await pilot.pause()
            card_bottom = panel.region.y + panel.region.height
            assert card_bottom <= shell.region.y, (
                f"after resize: card bottom {card_bottom} covers the composer at "
                f"{shell.region.y} (app={app.size!r}, region={panel.region!r})"
            )
            rows, gutter, _shown = panel._fit()
            if (
                app.size == (80, 12)
                and panel.region.height
                and panel.has_class("-squeezed") == (gutter == 0)
            ):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError(
                "the card never re-fit after the resize: "
                f"fit={panel._fit()!r}, squeezed={panel.has_class('-squeezed')}, "
                f"region={panel.region!r}, display={panel.display}, "
                f"styles={panel.styles.width!r}x{panel.styles.height!r}, app={app.size!r}"
            )


@pytest.mark.asyncio
async def test_a_resize_refits_the_card_from_the_dock_without_the_debounce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dock's own resize is the authoritative re-fit trigger, not a 50 ms guess.

    Measured at 100x40 -> 80x12 with the debounce timer neutralised: BOTH the
    resize handler and the timer can run before the screen has arranged the
    dock, and `overlay.rows_above_dock` then returns the PREVIOUS frame's
    ceiling — 29, then 33 rows on a 12-row screen. The card kept its 9-row
    ground and its gutter, covering the composer at row 6, until a later pass
    happened to measure real numbers: 0.7 s in the archived CI trace.

    `ComposerDock` now reports its own `Resize` (`BoxChanged`) and the app
    re-measures the overlay cards there, after the dock's region is real. With
    the timer out of the picture nothing else can repair the overlap, so a
    regression that drops the dock seam reopens it here rather than passing on
    a timer that happened to win the race.
    """
    monkeypatch.setattr(app_module, "RESIZE_REFIT_DELAY_S", 30.0)
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        panel = await _open_card(pilot, app, session)
        shell = app.query_one("#input-shell")
        await pilot.resize_terminal(80, 12)
        for _ in range(100):
            await pilot.pause()
            bottom = panel.region.y + panel.region.height
            assert bottom <= shell.region.y, (
                "the debounce is 30 s, so only the dock's BoxChanged can re-fit; "
                f"the card bottom is {bottom} over the composer at {shell.region.y}"
            )
            if panel.has_class("-squeezed") and app.size == (80, 12):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError(f"the card never re-fit: {panel._fit()!r}")


@pytest.mark.asyncio
async def test_a_height_only_resize_refits_the_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bottom-docked composer MOVES without resizing; the card must follow.

    QA round 1, Q-1: at 100x40 -> 100x12 the dock stays 75x5 and only its `y`
    changes, so Textual sends ONE `Resize` — before the arrange pass — and no
    later event exists. A seam that keys on the dock's size therefore cannot
    see this resize at all.

    The debounce is neutralised so only the arrange-following ground watcher
    can repair the stale ceiling; on the previous head this failed with the
    recorded `card bottom is 22 over the composer at 6`.
    """
    monkeypatch.setattr(app_module, "RESIZE_REFIT_DELAY_S", 30.0)
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        panel = await _open_card(pilot, app, session)
        shell = app.query_one("#input-shell")
        await pilot.resize_terminal(100, 12)
        for _ in range(100):
            await pilot.pause()
            bottom = panel.region.y + panel.region.height
            assert bottom <= shell.region.y, (
                "height-only resize: the card bottom is "
                f"{bottom} over the composer at {shell.region.y}"
            )
            if panel.has_class("-squeezed") and app.size == (100, 12):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError(f"the card never re-fit: {panel._fit()!r}")


async def _adopted(pilot, app: OperatorApp, session: object) -> None:  # noqa: ANN001
    """Wait until the app has ADOPTED ``session``, not merely booted.

    One ``pause()`` is not enough under load: a card opened before adoption is
    drawn from no session at all, and an assertion over it passes or skips for
    a reason that has nothing to do with the card.
    """
    for _ in range(60):
        await pilot.pause()
        if app._session is session:
            return
    raise AssertionError("the app never adopted the session")


async def _open_card(pilot, app: OperatorApp, session: object) -> GoalPanel:  # noqa: ANN001
    """Open the goal card and return it ONCE IT IS ON SCREEN.

    THE PRECONDITION, not a longer wait. ``_cmd_goal`` returns at its own
    ``session is None`` guard, so a ``/goal`` issued before adoption never
    reaches ``_open_goal_panel`` and the card is never shown at all — measured
    under CI load as ``display=False, region=0x0`` after a 200-iteration settle
    loop had already spent its budget waiting for a class on a card that did not
    exist. Adoption and the first paint are two separate events, so this waits
    for both and returns the panel the caller is about to measure.
    """
    await _adopted(pilot, app, session)
    app._cmd_goal("", lambda body, kind="info": None)
    panel = app.query_one(GoalPanel)
    for _ in range(200):
        if panel.is_open and panel.region.height:
            return panel
        await pilot.pause()
    raise AssertionError(
        "the card was never displayed: "
        f"open={panel.is_open}, region={panel.region!r}, "
        f"adopted={app._session is session}, app={app.size!r}"
    )


@pytest.mark.asyncio
async def test_a_viewers_card_reads_the_record_off_the_wire() -> None:
    """Agent review round 2, MINOR-6: a viewer's card was drawn from nothing.

    `_goal_record_for` is ``None`` on anything that does not own the session (the
    record's WRITES are the owner's), and the card read status, judge and history
    off that ``None`` — so every attached viewer painted `judge: —` and `settled
    none yet` beside a record its own frame carried. The frame here holds a judge
    mid-run and one settled goal that the fake's own record does NOT, so a card
    that still read the record would show neither.
    """
    from types import SimpleNamespace

    session = _armed()
    session.runtime_locality = "this-machine"  # type: ignore[misc]
    session.frontend_state = SimpleNamespace(  # type: ignore[attr-defined]
        goal_status="active",
        goal_judge={"state": "continuing", "run": 2},
        goal_history=[_entry("ship the wire-only objective")],
        goal_history_truncated=False,
    )

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _adopted(pilot, app, session)
        app._open_goal_panel()
        await pilot.pause()
        body = _card_text(app.query_one(GoalPanel))
    assert f"judge: continuing · run 2/{MAX_GOAL_CONTINUATIONS}" in body
    assert "ship the wire-only objective" in body
    assert "none yet" not in body
    # ...and it is still a READ-ONLY card: the source moved, the ownership did not.
    assert body.rstrip().endswith("q close")
    assert "d done" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("height", range(16, 26))
async def test_the_hint_row_is_painted_whatever_the_record_drops(height: int) -> None:
    """Agent review round 2, MINOR-9, measured inside the painted box.

    Two lines rode OUTSIDE the row budget the card is pinned to: the wire's
    clipped-record note (MINOR-9 as reported) and the goal row itself, which
    `_CHROME_ROWS` never counted — and a record longer than `MAX_HISTORY_ROWS`
    printed its `… N more settled` line on a tall terminal with no row reserved
    for it either. Each surplus line cost the body's LAST one, the key hint:
    before the fix this cell failed at every height here. From 16 rows up the
    ground holds the chrome, so every height must paint the whole body.
    """
    from types import SimpleNamespace

    session = _armed()
    for index in range(MAX_HISTORY_ROWS + 3):
        session.arm_goal(f"objective number {index}")
        session.mark_goal_done()
    session.arm_goal(GOAL)
    session.frontend_state = SimpleNamespace(  # type: ignore[attr-defined]
        goal_history_truncated=True
    )

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, height)) as pilot:
        await _adopted(pilot, app, session)
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        panel = app.query_one(GoalPanel)
        lines = _card_text(panel).split("\n")
        painted = panel.content_region.height
    assert len(lines) <= painted, f"{height}: body {len(lines)} rows, box paints {painted}"
    assert lines[-1].endswith("q close"), "the hint is the body's last line"
    assert "older settled goals were dropped" in "\n".join(lines)
    assert "more settled" in "\n".join(lines)
