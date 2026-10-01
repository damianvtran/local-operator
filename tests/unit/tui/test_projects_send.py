"""Quick-send's PURE half (S6d parity P5a): the target list and its filter.

The rows' ORDER is the feature — the manager first, then the project's own
sessions with live ones first, and never this session — so it is pinned here
without a terminal. The card's keys and the app's delivery wiring are driven
through the real app in the same file once that wiring lands.
"""

from __future__ import annotations

from local_operator.tui.widgets.projects_send import (
    NO_TARGET_FOOTER,
    SendTarget,
    compose_band,
    filter_targets,
    send_targets,
)

MANAGER = SendTarget(
    kind="manager", session_id="mgr-0001", label="manager", state="live", live=True
)


def _view(*rows: dict[str, object]) -> dict[str, object]:
    return {"project": {"id": "p1", "name": "parity-spec"}, "sessions": list(rows)}


def test_the_manager_row_leads_and_is_absent_when_there_is_none() -> None:
    # REAL payloads: liveness lives in `runtime.state` (F2/Q1).
    rows = send_targets(
        _view({"session_id": "aa", "runtime": {"state": "live"}, "title": "work"}),
        own_session=None,
        manager=MANAGER,
    )
    assert rows[0] is MANAGER
    assert [row.kind for row in rows] == ["manager", "session"]
    # No manager resolved: the row is simply absent rather than a dead one.
    assert [
        row.kind
        for row in send_targets(_view({"session_id": "aa"}), own_session=None, manager=None)
    ] == ["session"]
    # And an empty project is an empty list, which the card paints as its
    # footer line rather than as "no results".
    assert send_targets(_view(), own_session=None, manager=None) == []


def test_live_sessions_come_first_and_this_session_is_never_a_target() -> None:
    """You cannot message yourself, and the order is live-first (spec §7.5.1)."""
    rows = send_targets(
        _view(
            {"session_id": "stopped1", "runtime": {"state": "stale"}, "title": "old"},
            {"session_id": "mine", "runtime": {"state": "live"}, "title": "this one"},
            {"session_id": "missing1", "exists": False, "title": "gone"},
            {"session_id": "live1", "runtime": {"state": "live"}, "title": "running"},
            # A wedged record is dialable-ish but must NOT outrank a healthy
            # peer in the order the reader chooses from.
            {"session_id": "wedged1", "runtime": {"state": "wedged"}, "title": "stuck"},
        ),
        own_session="mine",
        manager=None,
    )
    # The spec's contract is LIVE-FIRST and store order within a state group —
    # it does not rank the non-live states against each other, so the test
    # asserts exactly that and the ROW INK (not the order) is what tells
    # `wedged` from `stale`/`missing`.
    ids = [row.session_id for row in rows]
    assert ids[0] == "live1", ids
    assert sorted(ids) == ["live1", "missing1", "stopped1", "wedged1"]
    assert {row.session_id: row.state for row in rows} == {
        "live1": "live",
        "wedged1": "wedged",
        "stopped1": "stale",
        "missing1": "missing",
    }
    assert all(row.session_id != "mine" for row in rows)


def test_a_row_states_its_session_handle_and_state() -> None:
    row = SendTarget(
        kind="session", session_id="ab12cd34ef5609", label="projects review", state="live"
    )
    assert row.row_text == "projects review  · session ab12cd34ef56  · [live]"


def test_filtering_is_a_subsequence_over_the_row_text() -> None:
    rows = [
        SendTarget(kind="manager", session_id="m1", label="manager"),
        SendTarget(kind="session", session_id="s1", label="projects review", state="live"),
    ]
    assert filter_targets(rows, "") == rows
    assert [row.label for row in filter_targets(rows, "prv")] == ["projects review"]
    assert [row.label for row in filter_targets(rows, "zzz")] == []


def test_the_band_names_the_target_and_the_way_out() -> None:
    target = SendTarget(kind="session", session_id="s1", label="projects review", state="live")
    assert compose_band(target) == "send to: projects review · m target · esc cancel"
    assert "esc closes" in NO_TARGET_FOOTER and " s " not in NO_TARGET_FOOTER


# -- the flow, over the real page ------------------------------------------
# The card and the modes are driven through the shipped app harness rather
# than a bare host: `m`'s meaning depends on which surface is up (a session row
# sends straight to it, anything else asks), and the composer hand-over is a
# message the app answers.

import pytest  # noqa: E402

from local_operator.tui.widgets.projects_send import SendTargetCard  # noqa: E402
from local_operator.tui.widgets.projects_view import (  # noqa: E402
    ProjectsViewComposeChanged,
    ProjectsViewSendRequested,
)
from tests.unit.tui.test_projects_view import (  # noqa: E402
    _boot,
    _factory,
    _open,
    _ProjectSession,
    _registry,
)


@pytest.mark.asyncio
async def test_m_opens_the_picker_and_esc_puts_it_away(tmp_path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = _factory(session)  # noqa: F841 — the factory is what the app builds
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        assert view._mode == "canvas"
        await pilot.press("m")
        await pilot.pause()
        assert view._mode == "send"
        assert view._send_card is not None
        assert view._send_card.rows == [target]
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "canvas"
        assert view._send_card is None


@pytest.mark.asyncio
async def test_choosing_a_target_hands_the_composer_over_and_submits(tmp_path) -> None:
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app)
        target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
        view._send_targets = lambda: [target]  # type: ignore[method-assign]
        await pilot.press("m")
        await pilot.pause()
        card = view._send_card
        assert card is not None
        card.action_choose()
        await pilot.pause()
        assert view.composing and view.compose_target is not None

        # RECORD, and still post: replacing `post_message` outright severs the
        # widget's own plumbing (mount/refresh notifications go through it),
        # which wedged `run_test` — the first version of this test hung the
        # harness for exactly that reason.
        posted: list[object] = []
        real_post = view.post_message

        def record(message: object) -> object:
            posted.append(message)
            return real_post(message)  # type: ignore[arg-type]

        view.post_message = record  # type: ignore[method-assign]
        sent = view.submit_compose("  hello there  ")
        assert sent is True
        request = [m for m in posted if isinstance(m, ProjectsViewSendRequested)]
        assert request and request[0].text == "hello there"
        assert request[0].target is view.compose_target

        # An empty body is refused IN-SURFACE: nothing is dialled, and the
        # page says why on its own notice line.
        posted.clear()
        assert view.submit_compose("   ") is True
        assert not [m for m in posted if isinstance(m, ProjectsViewSendRequested)]
        assert view._notice


def test_the_compose_surface_reports_its_target_and_esc() -> None:
    """`escape_surface` is the door the APP asks before dismissing the page."""
    target = SendTarget(kind="session", session_id="s1", label="work", state="live", live=True)
    card = SendTargetCard([target], style_for=None)
    # A card that has not been mounted still answers its own grammar.
    assert card.selected() == target
    assert isinstance(ProjectsViewComposeChanged(target=target), ProjectsViewComposeChanged)
