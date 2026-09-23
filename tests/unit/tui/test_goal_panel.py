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

from local_operator.session.goal_judge import MAX_GOAL_CONTINUATIONS
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.goal_panel import (
    MAX_HISTORY_ROWS,
    GoalPanel,
    build_goal_body,
    clamp_history_rows,
    judge_line,
)

from .test_app_pilot import FakeSession, _factory

GOAL = "land the OAuth refresh fix"


def _entry(text: str, status: str = "done") -> dict[str, str]:
    return {
        "id": f"id-{text}",
        "text": text,
        "status": status,
        "created_at": "2026-09-20T10:00:00+00:00",
        "settled_at": "2026-09-21T11:30:00+00:00",
        "reason": "the judge said so" if status == "done" else "",
    }


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
    assert f"2/{MAX_GOAL_CONTINUATIONS}" in body.plain
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
    assert "c delete" not in watcher.plain
    assert "q close" in watcher.plain
    assert "d done" in _body().plain


def test_the_history_list_is_bounded_and_says_what_it_dropped() -> None:
    history = [_entry(f"goal {index}") for index in range(MAX_HISTORY_ROWS + 3)]
    body = _body(status="", judge=None, history=history)
    assert "3 more settled" in body.plain
    assert body.plain.count("✓ ") == MAX_HISTORY_ROWS


def test_the_body_fits_its_width_at_80_and_100_columns() -> None:
    """The panel's two supported widths, measured in CELLS (not characters)."""
    history = [_entry("a settled goal with a reasonably long name")]
    for width in (80, 100):
        body = _body(width=width, history=history)
        longest = max(cell_len(line) for line in body.plain.split("\n"))
        assert longest <= width, f"{width}: {longest} cells"


def test_clamp_history_rows_never_takes_the_card_off_the_screen() -> None:
    assert clamp_history_rows(MAX_HISTORY_ROWS, available_rows=40) == MAX_HISTORY_ROWS
    assert clamp_history_rows(MAX_HISTORY_ROWS, available_rows=12) < MAX_HISTORY_ROWS
    assert clamp_history_rows(MAX_HISTORY_ROWS, available_rows=0) == 0


def test_the_judge_line_prints_the_caps_fraction_and_the_reason() -> None:
    line = judge_line(
        {
            "state": "stalled",
            "run": MAX_GOAL_CONTINUATIONS,
            "reason": "stopped after 12 continuations",
        },
        cap=MAX_GOAL_CONTINUATIONS,
    )
    assert f"12/{MAX_GOAL_CONTINUATIONS}" in line.plain
    assert "stopped after 12 continuations" in line.plain
    assert line.plain.startswith("judge: stopped")
    # An unknown member of the closed vocabulary (a newer writer's) prints as
    # itself rather than being dropped or crashing the card.
    assert judge_line({"state": "pondering"}, cap=3).plain == "judge: pondering"
    assert judge_line(None, cap=3).plain == "judge: —"


def test_the_action_keys_are_the_panels_own_and_not_global() -> None:
    """Audited, not assumed: a global action would be a persisted keymap id."""
    from local_operator import keymap

    bindings = {key for key, _, _ in GoalPanel.BINDINGS}
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
        assert GOAL in panel.render().plain
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
        body = panel.render()
        assert "— done" in body.plain
        assert GOAL in _struck_spans(body), "the done goal must be struck on the card"


@pytest.mark.asyncio
async def test_the_delete_key_deletes_without_recording_anything() -> None:
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app._cmd_goal("", lambda body, kind="info": None)
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert session.goal == ""
        assert session.goal_status == ""
        # DELETE records nothing: that is the whole difference from mark-done.
        assert session.history_view() == []


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
