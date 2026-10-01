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
    rows = send_targets(
        _view({"session_id": "aa", "live": True, "title": "work"}),
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
            {"session_id": "stopped1", "live": False, "title": "old"},
            {"session_id": "mine", "live": True, "title": "this one"},
            {"session_id": "missing1", "exists": False, "title": "gone"},
            {"session_id": "live1", "live": True, "title": "running"},
        ),
        own_session="mine",
        manager=None,
    )
    assert [row.session_id for row in rows] == ["live1", "stopped1", "missing1"]
    assert [row.state for row in rows] == ["live", "stopped", "missing"]
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
    assert NO_TARGET_FOOTER == "no target? s starts a session"
