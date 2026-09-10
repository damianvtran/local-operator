"""The CLICK and TAB focus paths must be guarded the way the Esc path is.

QA regression suite for the composer-focus change. Each test here FAILS on
``fix/composer-focus-default`` and PASSES on ``origin/main``, which is the
property that makes it a regression test rather than a restatement of the code.

WHAT THIS COVERS THAT THE SHIPPED SUITE DOES NOT. ``test_composer_focus.py``
tests ``_focus_is_claimed()`` as a PREDICATE
(``test_focus_is_claimed_covers_every_overlay``) and tests the one caller that
consults it — ``_return_focus_to_composer``, on the Esc path
(``test_esc_does_not_steal_focus_from_a_live_prompt``). The change adds THREE
more focus routes, and none of them calls the predicate:

* ``ComposerDock.on_click``      (app.py) — checks only ``editor.can_focus``
* ``TranscriptBlock.action_focus_composer`` (transcript.py) — same
* ``TranscriptView.on_click``    (transcript.py) — same

``can_focus`` is a weaker guard than ``_focus_is_claimed()``: it is False only
while the composer is READ-ONLY (the subagent page, the login prompt). It is
True while an approval, an ask picker, the aside, or the focused sidebar owns
the keyboard — so those four surfaces are unprotected on the click and Tab
routes.

WHY THE DOCK CLICK REACHES A LIVE PROMPT AT ALL. ``#prompt-host`` is mounted
INSIDE ``#input-dock`` (app.py, ``compose``), so the approval card and the ask
picker are DESCENDANTS of the container whose ``on_click`` now grabs focus.
``Click`` bubbles, and the handler deliberately does not call ``event.stop()``,
so a click on the card's own padding — the gesture a user makes to read the
question — arrives at ``ComposerDock.on_click`` and takes the keyboard away
from the thing being clicked.

Measured at 120x40, approval card up, clicking ``prompt-host.region + (1, 1)``:
origin/main left ``ApprovalPrompt`` focused; this branch left ``Editor``
focused with the approval still unanswered.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.tool_card import ToolCard

from .test_app_pilot import FakeSession, _factory


def _app(**kwargs: Any) -> OperatorApp:
    return OperatorApp(lambda: _factory(FakeSession()), **kwargs)


async def _boot(pilot: Any, app: OperatorApp) -> None:
    """Pause until the session is attached — the app is not usable before that."""
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


def _clamped(app: OperatorApp, offset: tuple[int, int]) -> tuple[int, int]:
    """Keep a derived offset inside what ``Pilot.click`` will accept.

    ``screen.size.region`` is two cells smaller per axis than ``screen.region``
    (``pilot.py:440-443``), and the dock's last rows fall in that gap.
    """
    x, y = offset
    bounds = app.screen.size.region
    return (min(max(x, 0), bounds.width - 1), min(max(y, 0), bounds.height - 1))


def _dock_pad(app: OperatorApp) -> tuple[int, int]:
    """A dead dock cell — column 2 of the shell's top row.

    Derived at call time rather than hardcoded: the boot and settled layouts put
    the shell at different origins, so a literal coordinate is wrong in one of
    them.
    """
    shell = app.query_one("#input-shell")
    return _clamped(app, (shell.region.x + 2, shell.region.y))


async def _a_card(pilot: Any, app: OperatorApp) -> ToolCard:
    """One tool card in the ledger — the row that took focus in the report."""
    app._append_block(ToolCard("t1", "bash", {"command": "ls"}))
    await pilot.pause()
    await pilot.pause()
    return next(iter(app.query(ToolCard).results()))


async def _a_multi_select(pilot: Any, app: OperatorApp) -> Any:
    """A live multi-select ask picker: the one question routed keys cannot reach.

    MULTI-SELECT specifically, for the reason
    ``test_esc_does_not_steal_focus_from_a_live_prompt`` records: an approval
    advertises ``y``/``n``/``A``, which the composer routes, so it does not need
    the caret. A multi-select is answered by Space and Enter, which the composer
    would swallow, so ``_prompt_wants_the_keyboard`` pulls focus to the card on
    purpose — and it is therefore the only surface that can be robbed of it.
    """
    from local_operator.harness.types import AskOption, AskQuestion

    question = AskQuestion(
        id="rows",
        question="Which rows should be dropped?",
        options=[
            AskOption(label="Stale", description="nothing reads them"),
            AskOption(label="Orphaned", description="no parent row"),
        ],
        multi=True,
    )
    # A worker rather than a bare `create_task`: a local handle goes out of
    # scope when this helper returns and the loop may collect the task out from
    # under the picker.
    app.run_worker(app.request_user_choice([question]), thread=False)
    for _ in range(8):
        await pilot.pause()
    picker = app._ask_screen
    assert picker is not None, "premise: the picker is up"
    return picker


@pytest.mark.asyncio
async def test_a_dock_click_does_not_steal_focus_from_a_live_approval() -> None:
    """THE blocker: clicking the dock while a destructive approval is up.

    The approval is mounted inside the dock, so this is not an exotic gesture —
    it is a user clicking near or on the question they are being asked, which is
    exactly where their eyes and pointer already are.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app.run_worker(
            app.request_tool_approval("bash", "rm -rf /Users/me/project/data"), thread=False
        )
        for _ in range(8):
            await pilot.pause()
        prompt = app._approval
        assert prompt is not None, "premise: the approval card is up"
        assert app._focus_is_claimed(), "premise: the predicate says the keyboard is claimed"

        await pilot.click(offset=_dock_pad(app))
        for _ in range(3):
            await pilot.pause()

        assert not app.query_one(Editor).has_focus, (
            "a dock click took the keyboard off a live approval; the composer now "
            "holds the caret while an unanswered `rm -rf` waits behind it"
        )


@pytest.mark.asyncio
async def test_clicking_the_approval_card_itself_leaves_it_focused() -> None:
    """The gesture is "read the question", and it must not disarm the question."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app.run_worker(app.request_tool_approval("bash", "rm -rf /"), thread=False)
        for _ in range(8):
            await pilot.pause()
        prompt = app._approval
        assert prompt is not None, "premise: the approval card is up"

        host = app.query_one("#prompt-host")
        await pilot.click(offset=_clamped(app, (host.region.x + 1, host.region.y + 1)))
        for _ in range(3):
            await pilot.pause()

        assert not app.query_one(Editor).has_focus, (
            "clicking ON the approval card moved focus to the composer"
        )


@pytest.mark.asyncio
async def test_a_dock_click_leaves_a_multi_select_answerable() -> None:
    """The consequence, not just the focus location.

    After the theft the card's advertised answer keys go to the composer:
    ``space`` types a space into the buffer and ``enter`` SUBMITS it. The
    question stays unsettled with the tool still waiting, and the user's only
    remaining route is a gesture nothing advertises.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        picker = await _a_multi_select(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        # Tab is the documented explicit handover for this card.
        await pilot.press("tab")
        for _ in range(3):
            await pilot.pause()
        assert not editor.has_focus, "premise: Tab handed the keyboard to the picker"

        await pilot.click(offset=_dock_pad(app))
        for _ in range(3):
            await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(4):
            await pilot.pause()

        assert picker.settled, (
            "the multi-select never settled: a dock click took the keyboard back "
            "and its Space/Enter answers were typed into the composer instead"
        )


@pytest.mark.asyncio
async def test_a_dock_click_does_not_steal_focus_from_the_sidebar() -> None:
    """The F9 sidebar is a focused surface with its own arrow-key navigation."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _a_card(pilot, app)
        app._session_sidebar.focus()
        for _ in range(3):
            await pilot.pause()
        assert app._session_sidebar.has_focus, "premise: the sidebar holds focus"

        await pilot.click(offset=_dock_pad(app))
        for _ in range(3):
            await pilot.pause()

        assert app._session_sidebar.has_focus, (
            "a dock click took the keyboard off the focused sidebar"
        )


@pytest.mark.asyncio
async def test_tab_from_a_row_does_not_steal_focus_from_a_live_prompt() -> None:
    """The Tab route is guarded by ``can_focus`` alone, which a live prompt passes.

    ``KeyPromptBlock`` and ``ApprovalBlock`` both descend from
    ``TranscriptBlock``, so both inherit the new unconditional ``tab`` binding —
    a prompt row now has a one-press exit that hands the keyboard away from the
    question it is asking.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        card = await _a_card(pilot, app)
        picker = await _a_multi_select(pilot, app)
        assert picker is not None

        card.focus()
        await pilot.pause()
        await pilot.press("tab")
        for _ in range(3):
            await pilot.pause()

        assert not app.query_one(Editor).has_focus, (
            "Tab from a transcript row took the keyboard while a multi-select "
            "was live, making its Space/Enter answers unreachable"
        )


@pytest.mark.asyncio
async def test_a_blank_transcript_click_does_not_steal_focus_from_a_live_prompt() -> None:
    """The third unguarded route: ``TranscriptView.on_click``."""
    from local_operator.tui.widgets.transcript import TranscriptView

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _a_card(pilot, app)
        picker = await _a_multi_select(pilot, app)
        assert picker is not None
        editor = app.query_one(Editor)
        assert not editor.has_focus, "premise: the picker holds the keyboard"

        view = app.query_one(TranscriptView)
        site = _clamped(app, (view.region.right - 3, view.region.y + view.region.height - 2))
        await pilot.click(offset=site)
        for _ in range(3):
            await pilot.pause()

        assert not editor.has_focus, (
            "a click on blank transcript took the keyboard off a live multi-select"
        )


@pytest.mark.asyncio
async def test_a_dock_click_does_not_steal_focus_from_the_login_key_prompt(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The surface where stealing focus puts an API key into the transcript.

    ``test_focus_is_claimed_by_a_live_key_prompt`` asserts the PREDICATE answers
    True here, and its docstring names this as "the one surface where stealing
    focus put an API key into the transcript in plain text". The click route
    never asks the predicate, so the protection does not reach this gesture.

    The composer is NOT read-only while this prompt is up — measured
    ``can_focus=True read_only=False`` — so the ``can_focus`` guard that the
    click path does use passes straight through. After the click, the key the
    user types goes into the composer's buffer UNMASKED, where the key prompt
    would have masked it.
    """
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    from .test_login_key_prompt import _LoginSession, _run_login

    monkeypatch.setattr("webbrowser.open", lambda *a, **k: True)
    app = OperatorApp(
        lambda: _factory(_LoginSession()),
        provider_controller=ProviderController(AuthStore(tmp_path / "auth.db")),
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _run_login(pilot, app, "alibaba")
        assert app._key_prompt is not None, "premise: the key prompt is up"

        await pilot.click(offset=_dock_pad(app))
        for _ in range(3):
            await pilot.pause()

        assert not app.query_one(Editor).has_focus, (
            "a dock click took the keyboard off the login key prompt; the key the "
            "user types next lands unmasked in the composer buffer"
        )


# WITHDRAWN: `test_a_key_in_flight_does_not_land_where_a_click_moved_focus`.
#
# It asserted that a `space` following a dock click must not reach the composer,
# and it was wrong — kept as a note because the way it was wrong is instructive.
#
# It contradicted `test_composer_focus.py:128`
# (`test_clicking_the_shell_padding_returns_focus_to_the_composer`), which
# asserts that the IDENTICAL gesture MUST focus the Editor. That test encodes
# the user's actual bug report and is the fail-first guard for this MR. Both
# cannot hold, and the one encoding the reported defect wins.
#
# The diagnosis error is visible in the evidence the withdrawn test itself
# quoted: on origin/main the card expanded and the buffer stayed `'draft'`.
# That is not correct focus ordering — it is the click being SWALLOWED by the
# dead frame this MR removes. The card expanded because it never lost focus.
# The test passed on main for the same reason the bug exists on main, so
# pinning it would have pinned the defect.
#
# There was also no race: `await pilot.click(...)` settles `set_focus(Editor)`
# fully before `pilot.press("space")` runs, with or without an intervening
# `pause()`. A user who clicks the input and then presses space should get a
# space in the input; that is the feature, not corruption.
#
# The design's "focus must never move under a key already in flight" rule is
# about focus moving with NO user gesture behind it — a turn ending, a timer,
# a condition going true while the user sits still. A click IS the user's
# gesture, so the rule does not reach that case. The test below is what that
# rule actually asks for.


@pytest.mark.asyncio
async def test_the_guard_holds_when_both_events_land_in_one_drain() -> None:
    """The guard must survive a click and a key delivered without an intervening settle.

    This is the in-flight shape the design's rule genuinely asks for, and it
    differs from the withdrawn test above in the one way that matters: a live
    claimant exists, so ``_focus_is_claimed()`` is True and the correct outcome
    is "focus does not move" REGARDLESS of ordering. There is no competing
    assertion here — no test wants a dock click to focus the composer while a
    multi-select is unanswered.

    What it protects against: a guard that is correct when the predicate is
    evaluated on a settled frame, but is bypassed when the click handler runs
    against state that has not caught up. Both events go into the same drain
    with no ``pause()`` between them, in both orders.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _a_card(pilot, app)
        picker = await _a_multi_select(pilot, app)
        editor = app.query_one(Editor)
        assert not editor.has_focus, "premise: the picker holds the keyboard"

        # Click then key, no settle between.
        await pilot.click(offset=_dock_pad(app))
        await pilot.press("space")
        for _ in range(4):
            await pilot.pause()
        assert not editor.has_focus, (
            "a dock click in the same drain as a keypress took the keyboard off "
            "a live multi-select"
        )
        assert editor.text == "", (
            f"the multi-select's `space` answer was typed into the composer: "
            f"{editor.text!r}"
        )

        # And the reverse order, which exercises the other interleaving.
        await pilot.press("space")
        await pilot.click(offset=_dock_pad(app))
        for _ in range(4):
            await pilot.pause()
        assert not editor.has_focus, (
            "a keypress in the same drain as a dock click took the keyboard off "
            "a live multi-select"
        )
        assert not picker.settled or picker.is_attached, (
            "the picker was settled by keys it never received"
        )
