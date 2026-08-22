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
        # Root progression line, then a header per phase with per-phase done/total.
        assert lines[0] == "Todos · 1/2"
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
        assert "+1 done" in body  # the affordance confesses the hidden item
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
        assert "ctrl+t to collapse" in body

        # Toggle back.
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert panel._expanded is False
        assert "Cleanup" not in str(panel._body.content)
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
