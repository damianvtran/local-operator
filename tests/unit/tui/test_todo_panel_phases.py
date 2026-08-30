"""Phased-todo panel behaviour (design §6/§7).

The single-phase back-compat guarantee is guarded by the existing
``test_band_panels.py`` goldens (a flat store still renders headerless and
byte-identical). These tests cover the NEW surface: multi-phase headers +
indent, the ``select_collapsed`` walking viewport ported from omp, the
view-only auto-hide, and the ``ctrl+t`` expand/collapse toggle.

They drive the REAL ``OperatorApp`` (not a reduced host) wherever CSS or the
row budget matters, so the panel is exercised the way a user reaches it — the
same discipline ``test_band_panels.py`` documents. Pure-policy helpers
(``select_collapsed``/``_active_phase_index``/``_phase_settled``) are asserted
directly because they carry no rendering state.
"""

from __future__ import annotations

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.todo_panel import (
    TODO_PHASE_HIDE_DELAY_S,
    TodoPanel,
    _active_phase_index,
    _as_phases,
    _phase_settled,
    select_collapsed,
    todo_items,
)

# Reuse the band tests' app-boot harness rather than re-declaring a FakeSession:
# one fake, one factory, so a drift in the protocol surfaces in one place.
# ``_clean_todo_store`` is an autouse fixture — imported so it runs here too.
from tests.unit.tui.test_band_panels import (  # noqa: F401
    FakeSession,
    _async_factory,
    _clean_todo_store,
)


def _item(text: str, status: str, **extra: str) -> dict[str, str]:
    return {"text": text, "status": status, **extra}


def _affordance_text(panel: TodoPanel) -> str:
    """The collapse/expand control's text — its OWN widget now (defect 2).

    The affordance was split out of the body Static into ``TodoAffordance`` so a
    ``:hover`` and an ``on_click`` can scope to just that row, so a test that
    used to read it off ``panel._body.content`` reads it here instead. Returns
    ``""`` when the control is hidden (a terminal too short to afford its row)."""
    if not panel._affordance.display:
        return ""
    return str(panel._affordance.content)


# --------------------------------------------------------------------------- #
# select_collapsed — the walking viewport (omp selectCollapsedTodos, todo.ts)
# --------------------------------------------------------------------------- #


def test_select_collapsed_active_under_cap_shows_all() -> None:
    items = [_item("a", "pending"), _item("b", "pending")]
    shown, hidden = select_collapsed(items, 5)
    assert [i["text"] for i in shown] == ["a", "b"]
    assert hidden == 0


def test_select_collapsed_active_over_cap_counts_hidden_open() -> None:
    items = [_item(str(n), "pending") for n in range(8)]
    shown, hidden = select_collapsed(items, 5)
    assert [i["text"] for i in shown] == ["0", "1", "2", "3", "4"]
    assert hidden == 3  # 8 open - 5 shown


def test_select_collapsed_out_of_order_completion_keeps_one_closed_lead() -> None:
    # ``done`` accepts any named item, so a closed row can sit between open ones.
    # The last closed row is kept as an additive lead (never costs an open row),
    # so finishing work out of sequence stays visible.
    items = [_item("a", "pending"), _item("b", "done"), _item("c", "pending")]
    shown, hidden = select_collapsed(items, 5)
    assert [(i["text"], i["status"]) for i in shown] == [
        ("b", "done"),  # the closed lead
        ("a", "pending"),
        ("c", "pending"),
    ]
    assert hidden == 0


def test_select_collapsed_settled_phase_selects_over_its_own_closed_rows() -> None:
    # No open work left: fall back to the closed rows so the settled phase still
    # renders something (omp selectWithinCap over the closed base).
    items = [_item("a", "done"), _item("b", "dropped")]
    shown, hidden = select_collapsed(items, 5)
    assert [i["text"] for i in shown] == ["a", "b"]
    assert hidden == 0


def test_active_phase_index_is_earliest_open_even_when_worked_ahead() -> None:
    # A later phase completed while an earlier one still has open work: the
    # pointer is the EARLIEST open phase, not the last touched.
    phases = [
        {"name": "A", "items": [_item("a", "pending")]},
        {"name": "B", "items": [_item("b", "done")]},
    ]
    assert _active_phase_index(phases) == 0
    phases = [
        {"name": "A", "items": [_item("a", "done")]},
        {"name": "B", "items": [_item("b", "pending")]},
    ]
    assert _active_phase_index(phases) == 1
    # Everything settled → last phase, so a done plan still anchors somewhere.
    phases = [
        {"name": "A", "items": [_item("a", "done")]},
        {"name": "B", "items": [_item("b", "dropped")]},
    ]
    assert _active_phase_index(phases) == 1


def test_phase_settled_excludes_blocked() -> None:
    # The auto-hide safety invariant: blocked is open work, never closed.
    assert _phase_settled({"name": "x", "items": [_item("a", "done")]}) is True
    assert (
        _phase_settled({"name": "x", "items": [_item("a", "done"), _item("b", "dropped")]}) is True
    )
    assert (
        _phase_settled({"name": "x", "items": [_item("a", "done"), _item("b", "blocked")]}) is False
    )
    assert (
        _phase_settled({"name": "x", "items": [_item("a", "done"), _item("b", "pending")]}) is False
    )
    assert _phase_settled({"name": "x", "items": []}) is False


def test_as_phases_coerces_flat_and_passes_phased_through() -> None:
    flat = [_item("a", "pending")]
    coerced = _as_phases(flat)
    assert coerced == [{"name": "Todos", "items": [_item("a", "pending")]}]
    phased = [{"name": "Auth", "items": [_item("a", "pending")]}]
    assert _as_phases(phased) is phased


def test_todo_items_returns_phases_and_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.tools import builtin

    builtin.TODO_STORE.clear()
    builtin.TODO_STORE["sess"] = [
        {"name": "Auth", "items": [_item("a", "pending")]},
    ]
    phases = todo_items("sess")
    assert phases[0]["name"] == "Auth"
    # Mutating the returned copy must not touch the store (the tool mutates in
    # place, so the panel holding originals would repaint stale).
    phases[0]["items"][0]["status"] = "done"
    assert builtin.TODO_STORE["sess"][0]["items"][0]["status"] == "pending"
    builtin.TODO_STORE.clear()


# --------------------------------------------------------------------------- #
# Multi-phase render — headers, indent, marks (driven through the real app)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_multi_phase_renders_headers_indent_and_marks() -> None:
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {
                "name": "Foundation",
                "items": [_item("wire the config", "done"), _item("add the migration", "pending")],
            },
            {
                "name": "Auth",
                "items": [_item("token exchange", "pending")],
            },
        ]
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        lines = str(panel._body.content).split("\n")
        # Root progression line reads as a STAGE POINTER, not a completion
        # fraction (D1/U1): ``stage 1/2`` cannot be misread as "1 of 2 done"
        # against the ``0/N`` phase headers below it. The stage total is the
        # absolute phase count (2), not the collapsed view's phase count.
        assert lines[0] == "Todos · stage 1/2"
        assert "Foundation · 1/2" in lines
        assert "Auth · 0/1" in lines
        # Items are indented one two-space gutter beneath their phase header.
        assert "  - [x] wire the config" in lines
        assert "  - [ ] add the migration" in lines
        assert "  - [ ] token exchange" in lines


@pytest.mark.asyncio
async def test_single_phase_renders_headerless() -> None:
    """A lone phase — the flat back-compat case — draws NO phase header and no
    indent: the root ``Todos · n/total resolved`` line and flush ``- [ ]`` rows,
    exactly as the pre-phases panel did."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # A flat store (item dicts at top level) coerces to one implicit phase.
        builtin.TODO_STORE["sess"] = [
            _item("wire the band", "done"),
            _item("capture frames", "pending"),
        ]
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        lines = str(panel._body.content).split("\n")
        assert lines[0] == "Todos · 1/2 resolved"
        # No phase header, no indent — rows sit flush.
        assert lines[1] == "- [x] wire the band"
        assert lines[2] == "- [ ] capture frames"
        assert not any(ln.startswith("  ") for ln in lines)


# --------------------------------------------------------------------------- #
# Auto-hide — view only, never the store (design §7.4)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_settled_phase_hidden_after_delay_but_not_before() -> None:
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"name": "Auth", "items": [_item("token exchange", "pending")]},
            {"name": "Cleanup", "items": [_item("remove the shim", "done")]},
        ]
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        # Freshly settled: Cleanup is still visible (its clock has not elapsed).
        assert "Cleanup · 1/1" in str(panel._body.content)

        # Age its settle time past the threshold — the settable seam the design
        # calls for — then repaint. Now it is hidden from the VIEW.
        panel._settled_since["Cleanup"] = (
            panel._settled_since["Cleanup"] - TODO_PHASE_HIDE_DELAY_S - 1
        )
        panel._shown = None  # force past the equality guard
        app._refresh_band()
        await pilot.pause()
        body = str(panel._body.content)
        assert "Cleanup" not in body
        # The affordance (its own widget now) confesses the hidden item.
        assert "+1 done" in _affordance_text(panel)
        # The store is untouched — hiding is view-only.
        assert len(builtin.TODO_STORE["sess"]) == 2
        assert builtin.TODO_STORE["sess"][1]["items"][0]["status"] == "done"
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_phase_with_a_pending_item_is_never_hidden() -> None:
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"name": "Auth", "items": [_item("token exchange", "pending")]},
            {"name": "Build", "items": [_item("compile", "done"), _item("link", "pending")]},
        ]
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        # Build has an open item, so it never gets a settle timer at all.
        assert "Build" not in panel._settled_since
        # Even forcibly ageing a (wrongly) seeded entry cannot hide it, because
        # the next sync clears the timer for any phase with open work.
        panel._settled_since["Build"] = 0.0
        panel._shown = None
        app._refresh_band()
        await pilot.pause()
        assert "Build" not in panel._hidden_phase_names(todo_items("sess"))
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_regaining_open_work_restarts_the_hide_clock() -> None:
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"name": "Auth", "items": [_item("token exchange", "pending")]},
            {"name": "Cleanup", "items": [_item("remove the shim", "done")]},
        ]
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        assert "Cleanup" in panel._settled_since
        # Cleanup reopens (a re-init/add adds a pending item): the clock clears.
        builtin.TODO_STORE["sess"][1]["items"].append(_item("also drop the env", "pending"))
        app._refresh_band()
        await pilot.pause()
        assert "Cleanup" not in panel._settled_since
        builtin.TODO_STORE.clear()


# --------------------------------------------------------------------------- #
# ctrl+t expand/collapse (design §7.6)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ctrl_t_toggles_and_expanded_reveals_a_hidden_settled_phase() -> None:
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"name": "Auth", "items": [_item("token exchange", "pending")]},
            {"name": "Cleanup", "items": [_item("remove the shim", "done")]},
        ]
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        # Hide Cleanup by ageing its clock.
        panel._settled_since["Cleanup"] = (
            panel._settled_since["Cleanup"] - TODO_PHASE_HIDE_DELAY_S - 1
        )
        panel._shown = None
        app._refresh_band()
        await pilot.pause()
        assert "Cleanup" not in str(panel._body.content)

        # ctrl+t through the real key path.
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert panel._expanded is True
        body = str(panel._body.content)
        assert "Cleanup · 1/1" in body  # expanded reveals the auto-hidden phase
        assert "ctrl+t to collapse" in _affordance_text(panel)

        # Toggle back.
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert panel._expanded is False
        assert "Cleanup" not in str(panel._body.content)
        builtin.TODO_STORE.clear()


def _big_multi_phase() -> list[dict[str, object]]:
    """A ~16-item multi-phase list mirroring the user's report (Discovery/
    Implementation/Validation), some done/blocked/dropped."""
    return [
        {
            "name": "Discovery",
            "items": [_item(f"d{n}", "done") for n in range(3)],
        },
        {
            "name": "Implementation",
            "items": [
                _item("i0", "done"),
                *[_item(f"i{n}", "pending") for n in range(1, 5)],
                _item("i5", "blocked", reason="waiting on review"),
            ],
        },
        {
            "name": "Validation",
            "items": [
                *[_item(f"v{n}", "pending") for n in range(6)],
                _item("v6", "dropped"),
            ],
        },
    ]


@pytest.mark.asyncio
async def test_expanded_reveals_every_item_collapsed_hid_on_a_normal_terminal() -> None:
    """Defect 1: on a normal-height terminal, ``ctrl+t`` must GROW the panel so a
    ~16-item multi-phase list is shown in FULL — the collapsed view's ``+N more``
    marker is gone and every hidden item is now painted.

    The original design bounded expanded by the SAME ``MAX_TODO_ROWS + 2`` budget
    as collapsed, so expand was a no-op in practice (the user's report). Expanded
    now takes rows from the transcript (``1fr``, scrolls) down to a small floor,
    which is what lets the whole list appear."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = _big_multi_phase()
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)

        # Collapsed: the walking viewport hides items behind a "+N more" count.
        collapsed = str(panel._body.content)
        assert "more" in _affordance_text(panel)
        assert collapsed.count("- [") < 16  # not everything is shown

        await pilot.press("ctrl+t")
        await pilot.pause()
        assert panel._expanded is True
        body = str(panel._body.content)
        # Every item is painted — one row per todo, all three phase headers.
        assert body.count("- [") == 16
        for header in ("Discovery · 3/3", "Implementation · 1/6", "Validation · 1/7"):
            assert header in body, header
        # No "+N more" remainder to confess: expanded hides nothing.
        assert "more" not in _affordance_text(panel)
        assert _affordance_text(panel) == "ctrl+t to collapse"
        # The list fits without a SCREEN scrollbar, and the composer is on screen.
        assert app.screen.virtual_size == app.screen.size
        assert app.screen.show_vertical_scrollbar is False
        assert app.query_one("#input-shell").region.height > 0
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_expanded_scrolls_when_the_list_exceeds_the_bound() -> None:
    """Defect 1, overflow case: an expanded list longer than the panel's height
    budget must stay REACHABLE by scrolling — the body never silently clips under
    ``Screen { overflow: hidden }`` and never pushes the composer off screen."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    # A short terminal makes even a modest list exceed the expanded bound.
    async with app.run_test(size=(100, 14)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = _big_multi_phase()
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        await pilot.press("ctrl+t")
        await pilot.pause()

        scroll = app.screen.query_one("#todo-scroll")
        # The full list is painted into the body, and the scroll region can reach
        # all of it (its virtual height exceeds its shown height).
        assert str(panel._body.content).count("- [") == 16
        assert scroll.max_scroll_y > 0
        assert scroll.show_vertical_scrollbar is True
        # The SCREEN itself never scrolls, and the composer stays visible.
        assert app.screen.virtual_size == app.screen.size
        assert app.query_one("#input-shell").region.height > 0
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_predicted_rows_matches_the_capped_paint_in_each_state() -> None:
    """``predicted_rows`` must report the panel's REAL occupied height in both
    states, or the band mis-budgets and reflows on the first frame. Expanded's
    over-long list is SCROLLED, so the prediction is the capped height, not the
    full line count."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 14)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = _big_multi_phase()
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        await pilot.press("ctrl+t")
        await pilot.pause()
        # The body holds all 16 items, but the panel is capped by its budget: the
        # prediction is the capped height, never the raw 16+ line count.
        assert str(panel._body.content).count("- [") == 16
        assert panel.predicted_rows() <= panel._body_rows()
        assert panel.predicted_rows() < 16
        builtin.TODO_STORE.clear()


# --------------------------------------------------------------------------- #
# Affordance is a separate clickable widget (defect 2)
# --------------------------------------------------------------------------- #


class _StopEvent:
    """A minimal click event: records whether ``stop()`` was called.

    The handler contract is exercised directly rather than through
    ``pilot.click``, whose mouse geometry against this docked band resolves a
    frame early under load and made the assertion flaky. What matters here is the
    HANDLER — ``event.stop()`` then the single-source toggle — not that Textual
    can route a synthetic mouse press to the right cell, which its own tests
    already cover."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_affordance_is_a_separate_widget_and_click_toggles() -> None:
    """Defect 2: the affordance is its OWN widget (so hover/click scope to just
    that row), and its ``on_click`` toggles expand through the SAME path
    ``ctrl+t`` uses, stopping the event first so the click never also scrolls the
    transcript behind the dock (band mouse-isolation rule). The list body Static
    carries no such handler, so a click in the list cannot toggle."""
    from local_operator.tools import builtin
    from local_operator.tui.widgets.todo_panel import TodoAffordance

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = _big_multi_phase()
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)

        # The affordance is a distinct widget, pinned OUTSIDE the scroll body.
        affordance = app.query_one(TodoAffordance)
        assert affordance is panel._affordance
        assert affordance.styles.pointer == "pointer"
        assert "ctrl+t to expand" in str(affordance.content)
        # It is NOT inside the scroll body — a scroll never moves the control.
        assert panel._body not in affordance.walk_children()

        # on_click stops the event (mouse isolation) and toggles through the
        # panel's single source of truth, the same path ``ctrl+t`` reaches.
        assert panel._expanded is False
        event = _StopEvent()
        affordance.on_click(event)
        await pilot.pause()
        assert event.stopped is True
        assert panel._expanded is True
        # Clicking again collapses — one control, both directions.
        affordance.on_click(_StopEvent())
        await pilot.pause()
        assert panel._expanded is False
        # And the toggle never scrolled the transcript behind the dock.
        assert app.screen.virtual_size == app.screen.size

        # The list body has no click-to-toggle: only the affordance does.
        assert getattr(panel._body, "on_click", None) is None
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_affordance_hover_ground_is_the_shared_overlay_step() -> None:
    """Defect 2: pointing at the affordance lights it with the app's shared hover
    ground (``$lo-overlay``, the same step ``SubagentRow:hover`` and
    ``ToolCard:hover`` use), and only the affordance carries that rule — hover is
    scoped to the button, so a stray hover over the list never lights it.

    The rule's PRESENCE is asserted here (the rendered hover is proven by a
    captured frame); driving ``pilot.hover`` and reading a computed colour proved
    timing-flaky under load and is Textual's contract to keep, not this test's."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = _big_multi_phase()
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        affordance = panel._affordance
        assert affordance.styles.pointer == "pointer"

        # A `#todo-affordance:hover` rule sets `background: $lo-overlay`, and the
        # base `#todo-affordance` rule does NOT set a background (so the hover is
        # a real change, not a no-op) — the same ground the other band hovers use.
        base_rule = None
        hover_rule = None
        for rule in app.stylesheet.rules:
            sel = ",".join(str(s.css) for s in rule.selector_set)
            if sel == "#todo-affordance":
                base_rule = rule
            elif sel == "#todo-affordance:hover":
                hover_rule = rule
        assert base_rule is not None and hover_rule is not None
        assert not base_rule.styles.has_rule("background")
        assert hover_rule.styles.has_rule("background")
        # The hover ground IS `$lo-overlay` (the `overlay` semantic token), the
        # app's shared hover step.
        from local_operator.tui.theme import semantic_color

        assert hover_rule.styles.background.hex.lower() == semantic_color("overlay").lower()
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_toggle_still_forces_a_repaint_past_the_equality_guard() -> None:
    """The equality-guard repaint discipline survives the widget split: a
    ``ctrl+t`` toggle changes what is shown with NO store change, so ``_shown``
    must drop and the panel must repaint (defect regression guard)."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = _big_multi_phase()
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        before = str(panel._body.content)
        # A redundant sync with no store change is a no-op (guard holds).
        guard = panel._shown
        app._refresh_band()
        assert panel._shown == guard
        # A toggle drops the guard and repaints to a different body.
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert str(panel._body.content) != before
        builtin.TODO_STORE.clear()


# --------------------------------------------------------------------------- #
# Row budget — headers count, marker counts items, short terminal fits composer
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_headers_count_toward_budget_and_composer_survives_a_short_terminal() -> None:
    """A phase header is a render row: it is flattened into the row list BEFORE
    the cap arithmetic, so a header can never push the composer off the bottom
    of a short terminal (design §6.4). The body never exceeds its budget and its
    top edge stays on screen."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 14)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"name": f"Phase {p}", "items": [_item(f"task {p}.{n}", "pending") for n in range(4)]}
            for p in range(5)
        ]
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        body = app.screen.query_one("#todo-body")
        painted = str(panel._body.content).split("\n")
        # The paint fits the budget: no clipping above the top edge, and no
        # scrollbar (virtual size equals actual size).
        assert body.region.y >= 0
        assert len(painted) <= panel._body_rows()
        assert app.screen.virtual_size == app.screen.size
        # The composer is still on screen below the band.
        assert app.query_one("#input-shell").region.y >= body.region.y
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_short_terminal_keeps_at_least_one_item_visible() -> None:
    """On a terminal so short the budget is 2 rows (root + 1), the panel must
    still paint an ITEM, not a bare root + phase header (U2).

    The walking viewport puts a phase HEADER at ``body[0]``, so a naive
    ``body[:cap]`` kept the header and painted zero todos — the panel read as
    empty though items existed, strictly worse than the flat list it replaces
    (which keeps the item at the same height). ``_fit_body`` trades the header
    for the first item so a real todo is always visible."""
    from local_operator.tools import builtin

    for height in (12, 14):
        session = FakeSession()
        app = OperatorApp(_async_factory(session))
        async with app.run_test(size=(100, height)) as pilot:
            await pilot.pause()
            builtin.TODO_STORE["sess"] = [
                {
                    "name": f"Phase {p}",
                    "items": [_item(f"task {p}.{n}", "pending") for n in range(2)],
                }
                for p in range(3)
            ]
            app._refresh_band()
            await pilot.pause()
            panel = app.query_one(TodoPanel)
            painted = str(panel._body.content).split("\n")
            # At least one indented item row is on screen — the panel never reads
            # as empty while todos exist.
            assert any(
                ln.startswith("  - [") for ln in painted
            ), f"h={height}: no item row painted, panel reads empty: {painted}"
            # Budget still respected: composer stays below the band and no
            # silent scrollbar appears. (``virtual_size == size`` is NOT asserted
            # at h=12: the whole app's chrome exceeds ten content rows there for
            # the FLAT list too, so it is a pre-existing app-height condition, not
            # a phased-panel regression — the invariant the panel owns is "no
            # scrollbar", which holds.)
            assert len(painted) <= panel._body_rows()
            assert app.screen.show_vertical_scrollbar is False

            # EXPANDED must ALSO keep an item on screen at the floor (U6): the
            # pinned affordance row must not starve the last item. Assert an
            # actual item row sits in the initial scroll viewport, not merely
            # that some line was painted (a header-only panel would clear that).
            await pilot.press("ctrl+t")
            await pilot.pause()
            scroll = app.screen.query_one("#todo-scroll")
            exp_lines = str(panel._body.content).split("\n")
            window = exp_lines[scroll.scroll_offset.y : scroll.scroll_offset.y + scroll.size.height]
            assert any(
                "- [" in ln for ln in window
            ), f"h={height}: expanded viewport shows no item row: {window}"
            builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_root_stage_total_is_absolute_not_the_collapsed_view() -> None:
    """The root ``stage N/M`` denominator counts ALL phases, not the visible
    ones (code-review NIT under D1). An auto-hidden settled phase must not make
    a three-phase plan read ``stage 2/2`` — the stage total is a fact about the
    plan, not about what currently fits on screen."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"name": "Foundation", "items": [_item("a", "done"), _item("b", "done")]},
            {"name": "Auth", "items": [_item("c", "pending"), _item("d", "pending")]},
            {"name": "Verification", "items": [_item("e", "pending")]},
        ]
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        # Age Foundation past the hide delay so it auto-hides from the view.
        panel._settled_since["Foundation"] = (
            panel._settled_since["Foundation"] - TODO_PHASE_HIDE_DELAY_S - 1
        )
        panel._shown = None
        app._refresh_band()
        await pilot.pause()
        lines = str(panel._body.content).split("\n")
        # Foundation is hidden, but the stage total is still 3, not 2.
        assert "Foundation" not in str(panel._body.content)
        assert lines[0] == "Todos · stage 2/3"
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_lone_named_phase_shows_its_header_matching_the_receipt() -> None:
    """A single EXPLICITLY-named phase renders WITH its header in the dock,
    matching the ``view`` receipt (U5). Only the implicit ``\"Todos\"`` phase is
    headerless — the panel and receipt gate on the same predicate now."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"name": "Foundation", "items": [_item("a", "pending"), _item("b", "done")]},
        ]
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        body = str(panel._body.content)
        # The chosen name survives in the dock, spelled the same as the receipt.
        assert "Foundation · 1/2" in body
        # The implicit flat store still routes headerless (back-compat).
        builtin.TODO_STORE["sess"] = [_item("x", "done"), _item("y", "pending")]
        app._refresh_band()
        await pilot.pause()
        flat = str(panel._body.content).split("\n")
        assert flat[0] == "Todos · 1/2 resolved"
        assert flat[1] == "- [x] x"
        builtin.TODO_STORE.clear()


# --------------------------------------------------------------------------- #
# Round-1 remediation: flat expand (M1), expanded-floor (U1), focus guard (U3),
# keyboard scroll (U2), no dangling header (D1)
# --------------------------------------------------------------------------- #


def _flat(n: int) -> list[dict[str, object]]:
    """A flat (implicit ``\"Todos\"``) store of ``n`` pending items — the DEFAULT
    shape a ``todo init items=[...]`` produces, which routes through
    ``_build_flat``."""
    return [{"name": "Todos", "items": [_item(f"item {i}", "pending") for i in range(n)]}]


@pytest.mark.asyncio
async def test_flat_expand_reveals_every_item_and_mounts_affordance() -> None:
    """M1: ``ctrl+t`` on a FLAT list (the common non-phased case) must GROW the
    panel to paint every item AND mount the clickable affordance — the round-1
    regression was that ``_build_flat`` ignored ``self._expanded``, so expand was
    a no-op and no control was ever shown for the default todo shape.

    Collapsed stays byte-identical (a body ``… N more todos`` marker, no
    affordance widget); expanded paints all items with no marker and shows the
    ``ctrl+t to collapse`` control."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = _flat(15)
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)

        # Collapsed: capped, marker is a BODY line, NO affordance widget.
        collapsed = str(panel._body.content)
        assert collapsed.count("- [") < 15
        assert "more todos" in collapsed
        assert panel._affordance.display is False

        await pilot.press("ctrl+t")
        await pilot.pause()
        assert panel._expanded is True
        body = str(panel._body.content)
        # Every item painted, no remainder marker.
        assert body.count("- [") == 15
        assert "more todos" not in body
        # The affordance is now mounted and clickable.
        assert panel._affordance.display is True
        assert _affordance_text(panel) == "ctrl+t to collapse"
        assert panel._affordance.styles.pointer == "pointer"
        # Fits without a SCREEN scrollbar; composer stays on screen.
        assert app.screen.virtual_size == app.screen.size
        assert app.query_one("#input-shell").region.height > 0
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_flat_collapsed_is_byte_identical_with_no_affordance() -> None:
    """M1 back-compat guard: the flat COLLAPSED path is unchanged — the marker is
    a body line and the affordance widget stays hidden, so the existing goldens
    hold."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 14)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = [
            {"text": f"step {n} of the plan", "status": "pending"} for n in range(1, 13)
        ]
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        lines = str(panel._body.content).split("\n")
        assert lines[0] == "Todos · 0/12 resolved"
        assert lines[-1] == "… 10 more todos"
        # Collapsed flat mounts NO affordance widget (back-compat).
        assert panel._affordance.display is False
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_expanded_never_shows_fewer_rows_than_collapsed_across_heights() -> None:
    """U1: expand must NEVER paint fewer todo rows than collapsed at the same
    height. Below ~24 rows the transcript floor drove the grown share below the
    collapsed budget, so ``ctrl+t`` shrank to a 1-row porthole — worse than
    collapsed. The expanded budget is now floored at the collapsed budget, so at
    every height expanded shows AT LEAST what collapsed did and scrolls the rest.
    """
    from local_operator.tools import builtin

    for height in (12, 14, 20, 24, 40):
        session = FakeSession()
        app = OperatorApp(_async_factory(session))
        async with app.run_test(size=(100, height)) as pilot:
            await pilot.pause()
            builtin.TODO_STORE["sess"] = _big_multi_phase()
            app._refresh_band()
            await pilot.pause()
            panel = app.query_one(TodoPanel)
            scroll = app.screen.query_one("#todo-scroll")
            total = str(panel._body.content).count("- [")
            collapsed_shown = min(scroll.size.height, total)
            await pilot.press("ctrl+t")
            await pilot.pause()
            expanded_shown = min(scroll.size.height, total)
            assert expanded_shown >= collapsed_shown, (
                f"h={height}: expanded shows {expanded_shown} < collapsed " f"{collapsed_shown}"
            )
            # Composer stays on screen at every height.
            assert app.query_one("#input-shell").region.height > 0
            # No SCREEN scrollbar at h>=14. At h=12 the whole app's chrome
            # exceeds ten content rows for the FLAT list too (a pre-existing
            # app-height condition the flat short-terminal golden documents), so
            # the invariant the panel owns — "expanded never overflows worse than
            # collapsed" — is asserted as expanded virtual_size <= collapsed's.
            if height >= 14:
                assert app.screen.virtual_size == app.screen.size
            builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_todo_scroll_does_not_take_focus_from_the_composer() -> None:
    """U3: clicking the list body must NOT trap focus in the scroll region. The
    scroll is a status surface (``can_focus = False``); a body click leaves the
    composer focused, so the next typed message lands in the composer instead of
    vanishing into a widget that does nothing with keystrokes (the same class as
    the ``TranscriptView`` focus bug the app already guards against)."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 14)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = _big_multi_phase()
        app._refresh_band()
        await pilot.pause()
        await pilot.press("ctrl+t")
        await pilot.pause()
        scroll = app.screen.query_one("#todo-scroll")
        assert scroll.can_focus is False
        # Click the body, then type: the text must reach the composer.
        await pilot.click("#todo-body")
        await pilot.pause()
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.pause()
        assert app._editor().text == "hello"
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_keyboard_scroll_reaches_the_expanded_overflow() -> None:
    """U2: with the scroll region non-focusable, a keyboard user reaches the
    expanded overflow via ``ctrl+down``/``ctrl+up`` (``scroll_todos_*``), which
    page the SAME region the wheel drives. Focus stays on the composer
    throughout, and the actions no-op unless an expanded list overflows."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 14)) as pilot:
        await pilot.pause()
        builtin.TODO_STORE["sess"] = _big_multi_phase()
        app._refresh_band()
        await pilot.pause()
        panel = app.query_one(TodoPanel)

        # Collapsed: the scroll helper is a no-op (nothing to reveal that way).
        assert panel.scroll_expanded(down=True) is False

        await pilot.press("ctrl+t")
        await pilot.pause()
        scroll = app.screen.query_one("#todo-scroll")
        assert scroll.max_scroll_y > 0  # the list overflows

        assert scroll.scroll_offset.y == 0
        await pilot.press("ctrl+down")
        await pilot.pause()
        assert scroll.scroll_offset.y > 0  # keyboard reached the overflow
        # Focus never left the composer — the scroll region cannot hold it.
        assert app.screen.focused is app._editor()
        await pilot.press("ctrl+up")
        await pilot.pause()
        assert scroll.scroll_offset.y == 0
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_collapsed_view_has_no_childless_phase_header() -> None:
    """D1: the walking-viewport slice can admit the next phase's HEADER but run
    out of budget before any of its items, leaving an empty group
    (``Validation · 1/6`` with nothing beneath, then ``+N more``). ``_fit_body``
    now drops a trailing childless header; its count is implied by the root stage
    line and the ``+N more`` total, so the hidden count stays honest."""
    from rich.text import Text as _Text

    panel = TodoPanel()

    def _h(name: str) -> tuple[_Text, bool, bool]:
        return (_Text(name), False, False)

    def _i(name: str) -> tuple[_Text, bool, bool]:
        return (_Text(name), True, True)

    body = [_h("Implementation"), _i("i0"), _i("i1"), _i("i2"), _h("Validation"), _i("v0")]
    # cap admits the Validation header but no Validation item.
    kept, dropped = panel._fit_body(list(body), 5)
    kept_texts = [t.plain for t, _is_item, _o in kept]
    assert kept_texts == ["Implementation", "i0", "i1", "i2"]
    assert "Validation" not in kept_texts
    # Every dropped item is still confessed to the caller for the +N count.
    assert any(is_item for _t, is_item, _o in dropped)
    # And when the cap DOES reach an item, the header stays.
    kept2, _ = panel._fit_body(list(body), 6)
    assert [t.plain for t, _i2, _o in kept2] == [
        "Implementation",
        "i0",
        "i1",
        "i2",
        "Validation",
        "v0",
    ]


@pytest.mark.asyncio
async def test_expanded_shows_an_item_at_the_floor_flat_and_phased() -> None:
    """U6: at the collapsed floor (h=12 both shapes, h=13 phased) ``ctrl+t`` must
    still paint at least one real ITEM row in the initial scroll viewport, not
    just the header + ``ctrl+t to collapse`` affordance.

    The affordance is pinned OUTSIDE the scroll region, so at ``budget - 1`` rows
    it stole the single row the first item would occupy — expand painted zero
    todos where collapsed showed one, the "expand shows nothing" defect returning
    at the extreme floor. This asserts an actual ``- [`` item row is WITHIN the
    scroll viewport (top ``scroll.size.height`` body lines), not merely that the
    painted line count cleared a bar the header alone would clear."""
    from local_operator.tools import builtin

    def _items_in_viewport(app: "OperatorApp", panel: TodoPanel) -> int:
        scroll = app.screen.query_one("#todo-scroll")
        lines = str(panel._body.content).split("\n")
        top = scroll.scroll_offset.y
        window = lines[top : top + scroll.size.height]
        return sum(1 for ln in window if "- [" in ln)

    flat_store = [_item(f"foundation task {n}", "pending") for n in range(15)]
    phased_store = _big_multi_phase()
    cases = [
        ("flat", flat_store, 12),
        ("flat", flat_store, 13),
        ("phased", phased_store, 12),
        ("phased", phased_store, 13),
    ]
    for shape, store, height in cases:
        session = FakeSession()
        app = OperatorApp(_async_factory(session))
        async with app.run_test(size=(100, height)) as pilot:
            await pilot.pause()
            builtin.TODO_STORE["sess"] = [dict(p) for p in store] if shape == "phased" else store
            app._refresh_band()
            await pilot.pause()
            panel = app.query_one(TodoPanel)
            collapsed_items = _items_in_viewport(app, panel)
            await pilot.press("ctrl+t")
            await pilot.pause()
            expanded_items = _items_in_viewport(app, panel)
            assert expanded_items >= 1, (
                f"{shape} h={height}: expanded viewport shows no item row: "
                f"{str(panel._body.content).split(chr(10))}"
            )
            # Never worse than collapsed at the same height (the U1 contract, held
            # at the floor by the U6 guard).
            assert expanded_items >= collapsed_items, (
                f"{shape} h={height}: expanded {expanded_items} < collapsed " f"{collapsed_items}"
            )
            # The rest stays reachable — the guard trims chrome, never drops an
            # item — so the overflow scrolls rather than being lost.
            scroll = app.screen.query_one("#todo-scroll")
            assert scroll.max_scroll_y > 0
            assert app.query_one("#input-shell").region.height > 0
            builtin.TODO_STORE.clear()


# --------------------------------------------------------------------------- #
# Expanded overflow/position footer (#264 U4, #265 U5)
# --------------------------------------------------------------------------- #


async def _booted_panel(  # type: ignore[no-untyped-def]
    app: OperatorApp, pilot, phases: list[dict[str, object]]
) -> TodoPanel:
    """Seed the store once the SESSION exists, repaint, and hand back the panel.

    Waits for ``app._session`` rather than a frame count. The app paints before
    its session exists (the factory is awaited in a boot worker) and the panel
    reads the store keyed by ``session_id``, so a repaint landing in that window
    finds no todos and hides the panel — which reads as an assertion about
    footers failing for a reason that has nothing to do with footers.
    ``test_band_panels.py`` documents the same race and waits the same way.
    """
    for _ in range(200):
        await pilot.pause()
        if app._session is not None:
            break
    from local_operator.tools import builtin

    builtin.TODO_STORE["sess"] = phases
    app._refresh_band()
    await pilot.pause()
    return app.query_one(TodoPanel)


async def _settled_expand(app: OperatorApp, pilot) -> TodoPanel:  # type: ignore[no-untyped-def]
    """Press ``ctrl+t`` and wait for the scroll region's virtual size to settle.

    The panel repaints in the tick the flag flips (``request_toggle`` ->
    ``_refresh_band``), but ``max_scroll_y`` is a function of a virtual size the
    compositor computes a frame later. A single ``pause`` therefore reads
    ``max_scroll_y == 0`` on a list that does overflow, which is the shape of a
    flake rather than a defect — the footer's own text is already correct at that
    point. Settling here keeps the OVERFLOW PRECONDITION honest: a test asserting
    an overflow footer must first prove the list actually overflows.
    """
    panel = app.query_one(TodoPanel)
    await pilot.press("ctrl+t")
    for _ in range(40):
        await pilot.pause()
        if panel._scroll.max_scroll_y > 0:
            break
    return panel


def _long_multi_phase() -> list[dict[str, object]]:
    """A 17-item plan that overflows the expanded budget on a 100x30 terminal.

    Deliberately longer than ``_big_multi_phase``: that one FITS at a normal
    height (it is the fixture proving expand reveals everything), so it can never
    exercise the overflow footer. The counts here are asserted against
    ``sum(len(phase items))`` rather than restated, so a fixture edit cannot make
    a wrong position number look right.
    """
    return [
        {"name": "Discovery", "items": [_item(f"discovery task {n}", "done") for n in range(4)]},
        {
            "name": "Implementation",
            "items": [
                _item("implementation task 0", "done"),
                *[_item(f"implementation task {n}", "pending") for n in range(1, 6)],
                _item("implementation task 6", "blocked", reason="waiting on review"),
            ],
        },
        {
            "name": "Validation",
            "items": [
                *[_item(f"validation task {n}", "pending") for n in range(5)],
                _item("validation task 5", "dropped"),
            ],
        },
    ]


@pytest.mark.asyncio
async def test_expanded_overflow_footer_states_the_remainder_below() -> None:
    """U4 + U5 as ONE footer: an expanded list that overflows reads
    ``↓ N more · ctrl+t to collapse``.

    The scrollbar thumb was the only signal content continued — invisible to a
    keyboard-only reader and easy to miss on a short window. The number is a
    REMAINDER, not a position (UX round 1, U2): ``N of M`` reads as "the item I
    am looking at" by every terminal convention, and no visible row carries an
    ordinal that could confirm it. A remainder is checkable by scrolling, and it
    mirrors the collapsed ``+N more`` the reader saw one keypress earlier."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        phases = _long_multi_phase()
        total = sum(len(phase["items"]) for phase in phases)  # type: ignore[arg-type]
        await _booted_panel(app, pilot, phases)
        panel = await _settled_expand(app, pilot)

        # Precondition: this list really does overflow, or the footer is vacuous.
        assert panel._scroll.max_scroll_y > 0
        footer = _affordance_text(panel)
        assert footer.endswith("ctrl+t to collapse"), footer
        assert footer.startswith("↓ "), footer
        assert " more · " in footer, footer
        # The remainder counts TODOS, not painted rows: phase headers and the
        # root line are chrome and must not inflate it. At the top of the scroll
        # some todos are visible, so the remainder is strictly between 0 and all.
        remaining = int(footer.split(" more")[0].removeprefix("↓ ").strip())
        assert 0 < remaining < total, footer
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_expanded_footer_absent_when_the_whole_list_is_visible() -> None:
    """No overflow, no cue. An expanded list the reader can see in full is not a
    place to be lost in, and a permanent ``5 of 5`` would be chrome that never
    changes — collapsed does not confess a remainder it does not have either."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        panel = await _booted_panel(
            app,
            pilot,
            [
                {"name": "Auth", "items": [_item("token exchange", "pending")]},
                {"name": "Cleanup", "items": [_item("remove the shim", "pending")]},
            ],
        )
        await pilot.press("ctrl+t")
        await pilot.pause()

        assert panel._scroll.max_scroll_y == 0  # precondition: nothing hidden
        assert _affordance_text(panel) == "ctrl+t to collapse"
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_expanded_footer_tracks_the_keyboard_scroll_and_drops_the_arrow() -> None:
    """The position follows ``ctrl+down``/``ctrl+up`` (U2's real key path), and
    reaching the end drops the ``↓`` while the count reads ``M of M``.

    A scroll repaints nothing on its own — ``sync``'s equality guard covers the
    store, the budget, the expanded flag and the hidden set, none of which a
    scroll moves — so this is the regression guard for the ``scroll_y`` watcher.
    A footer frozen at its first-paint numbers would be worse than none: it would
    claim more content below after the reader had already reached the end."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        phases = _long_multi_phase()
        total = sum(len(phase["items"]) for phase in phases)  # type: ignore[arg-type]
        await _booted_panel(app, pilot, phases)
        panel = await _settled_expand(app, pilot)
        assert panel._scroll.max_scroll_y > 0  # precondition: it really overflows
        at_top = _affordance_text(panel)
        assert at_top.startswith("↓ "), at_top

        # The REAL binding, not the scroll API: this is the path a user takes.
        await pilot.press("ctrl+down")
        for _ in range(12):
            await pilot.pause()
        assert panel._scroll.scroll_y == panel._scroll.max_scroll_y
        # At the end there is no remainder, so the prefix is gone ENTIRELY and
        # the row is byte-identical to the no-overflow footer — both mean
        # "nothing is hidden" (UX round 1, U4). Absence can carry that meaning
        # only because the widest-fit rule stops the prefix vanishing for any
        # other reason.
        at_end = _affordance_text(panel)
        assert at_end == "ctrl+t to collapse", at_end
        assert str(total) not in at_end, at_end

        await pilot.press("ctrl+up")
        for _ in range(12):
            await pilot.pause()
        assert _affordance_text(panel) == at_top
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_expanded_footer_sheds_whole_before_it_eats_the_hotkey() -> None:
    """Narrow width: the position prefix is dropped ENTIRE, never truncated into
    the ``ctrl+t`` token.

    The row is ``no_wrap``/``ellipsis`` and clipped against the screen, so a
    footer that merely grew would spend a narrow width on the count and leave
    ``ctrl+t to colla…`` — keeping the summary and shedding the affordance, the
    exact inversion ``usage_panel._hint_row`` documents. The hotkey is the only
    signal the toggle exists; position is polish."""
    from local_operator.tools import builtin

    session = FakeSession()
    for width in (16, 20, 40):
        session = FakeSession()
        app = OperatorApp(_async_factory(session))
        async with app.run_test(size=(width, 30)) as pilot:
            panel = await _booted_panel(app, pilot, _long_multi_phase())
            await pilot.press("ctrl+t")
            await pilot.pause()

            footer = _affordance_text(panel)
            # ``ctrl+t`` survives at every width — the one non-negotiable token.
            assert "ctrl+t" in footer, f"w={width}: {footer!r}"
            if " more · " in footer:
                # Wide enough for both: the hotkey phrase stays WHOLE behind the
                # prefix, never clipped to pay for the count.
                assert footer.endswith("ctrl+t to collapse"), f"w={width}: {footer!r}"
            elif footer.startswith("↓"):
                # Too narrow for the number but not for the cue: the NUMBER is
                # what goes, never the arrow. The arrow is U4's finding; the
                # count is only U5's orientation.
                assert footer == "↓ · ctrl+t to collapse", f"w={width}: {footer!r}"
            else:
                # Shed WHOLE: the row is exactly what it was before this feature,
                # with no count fragment and no orphaned separator.
                assert footer.startswith("ctrl+t to"), f"w={width}: {footer!r}"
                assert "·" not in footer, f"w={width}: {footer!r}"
            builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_collapsed_affordance_is_unchanged_by_the_expanded_footer() -> None:
    """The collapsed row keeps its ``+N more · ctrl+t to expand`` exactly: the
    new footer is an EXPANDED-only addition, and the collapsed goldens
    (``test_band_panels.py``) must not move."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        panel = await _booted_panel(app, pilot, _long_multi_phase())
        assert panel._expanded is False
        footer = _affordance_text(panel)
        assert footer.endswith("· ctrl+t to expand"), footer
        assert footer.startswith("+"), footer
        assert "↓" not in footer, footer
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_expanded_footer_arrow_follows_the_body_not_the_todo_count() -> None:
    """Design round 1, D2: the arrow answers "is there more to SEE", so it is
    derived from the viewport, never from the todo remainder.

    A plan whose last phase has no items (a shape the tool schema accepts, and
    the natural rendering of work not yet started) puts a phase HEADER below the
    fold with no todo behind it. Deriving the arrow from the todo count painted
    "nothing below" while that header was hidden — contradicting the scrollbar
    thumb in the same frame, which is precisely the failure #264 exists to
    prevent. The counts stay todo-based; only the arrow moved."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted_panel(
            app,
            pilot,
            [
                {"name": "Alpha", "items": [_item(f"alpha {n}", "pending") for n in range(9)]},
                {"name": "Beta", "items": [_item(f"beta {n}", "pending") for n in range(6)]},
                # The whole point: a trailing phase contributing a header row and
                # no item rows, so body lines outlast todo rows.
                {"name": "Gamma", "items": []},
            ],
        )
        panel = await _settled_expand(app, pilot)
        max_scroll = panel._scroll.max_scroll_y
        assert max_scroll > 0

        for offset in range(max_scroll + 1):
            panel._scroll.scroll_to(y=offset, animate=False)
            for _ in range(8):
                await pilot.pause()
            footer = _affordance_text(panel)
            more_below = offset < max_scroll
            assert ("↓" in footer) is more_below, f"y={offset}: {footer!r}"
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_expanded_footer_keeps_the_hotkey_column_fixed_while_scrolling() -> None:
    """Design round 1, D1: the ``ctrl+t`` token must not move under a reader who
    is only scrolling.

    D1 measured a two-cell jump when the arrow dropped under the old ``N of M``
    wording. ``N more`` sheds the whole prefix at the end instead, but the same
    defect class survives the rewording as a one-cell shift at every
    power-of-ten boundary (``↓ 10 more`` -> ``↓ 9 more``) — mid-scroll, with the
    reader watching. The remainder is padded to the widest number's width so the
    column is pinned across every position the prefix is shown at."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _booted_panel(app, pilot, _long_multi_phase())
        panel = await _settled_expand(app, pilot)
        assert panel._scroll.max_scroll_y > 0

        columns = set()
        crossed_ten = False
        for offset in range(panel._scroll.max_scroll_y + 1):
            panel._scroll.scroll_to(y=offset, animate=False)
            for _ in range(8):
                await pilot.pause()
            footer = _affordance_text(panel)
            if "↓" not in footer:
                continue  # the end state deliberately sheds the prefix
            columns.add(footer.index("ctrl+t"))
            remaining = int(footer.split(" more")[0].removeprefix("↓ ").strip())
            crossed_ten |= remaining < 10
        # The scroll must actually cross a digit boundary, or this proves nothing.
        assert crossed_ten, "fixture never drops below ten remaining"
        assert len(columns) == 1, f"ctrl+t moved between columns {sorted(columns)}"
        builtin.TODO_STORE.clear()


@pytest.mark.asyncio
async def test_expanded_footer_cue_never_vanishes_mid_scroll_at_narrow_widths() -> None:
    """UX round 1, U1: the cue's presence is a property of the width and the list
    length, never of the number's current digit count.

    Width-testing the prefix against its CURRENT text made the cue survive
    ``↓ 9 more`` and vanish at ``↓ 10 more``, so a reader at 34 columns was shown
    the exact pre-change "this is everything" row with todos still hidden. That
    is worse than never showing a cue, because absence had been taught to mean
    something. Sizing on the widest remainder makes the cue stable: shown at
    every position, or at none."""
    from local_operator.tools import builtin

    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(34, 16)) as pilot:
        await _booted_panel(app, pilot, _long_multi_phase())
        panel = await _settled_expand(app, pilot)
        max_scroll = panel._scroll.max_scroll_y
        assert max_scroll > 0

        # Every position before the end must carry the cue — no gaps.
        blind = []
        for offset in range(max_scroll):
            panel._scroll.scroll_to(y=offset, animate=False)
            for _ in range(8):
                await pilot.pause()
            footer = _affordance_text(panel)
            if "↓" not in footer:
                blind.append((offset, footer))
        assert not blind, f"cue vanished mid-scroll at {blind}"
        builtin.TODO_STORE.clear()
