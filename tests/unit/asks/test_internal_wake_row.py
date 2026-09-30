"""The hidden internal wake kinds, and every human surface that must subtract them.

The ask queue rides the wake engine for its deadlines (design
``docs/design/ask-nonblocking.md`` §2.2, D10), which means an ``ask_timeout`` row
appears in the same ``schedules`` lists every wake surface reads. The failure this
file guards is the one the design's risk 4 names: a human surface that misses the
new predicate leaks a hidden timer as if it were a reminder someone set — so the
subtraction is ONE named predicate (``is_internal_wake_row``) and the surfaces
that must ask it are pinned here rather than trusted to have been found.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from local_operator.harness.wake_types import WakeSchedule
from local_operator.wakes import patience, store

ROOT = Path(__file__).resolve().parents[3]


def test_the_predicate_covers_patience_and_ask_timeout_and_nothing_else():
    assert store.is_internal_wake_row({"kind": "patience"}) is True
    assert store.is_internal_wake_row({"kind": "ask_timeout"}) is True
    assert store.is_internal_wake_row({"kind": "scheduled"}) is False
    assert store.is_internal_wake_row({}) is False
    assert store.is_internal_wake_row(None) is False


def test_the_predicate_reads_models_and_dicts_alike():
    """The session holds models and the index/supervisor hold dicts, and BOTH
    must be filtered by the same call — the review finding that produced
    ``is_patience_row``."""
    row = WakeSchedule(
        id="a-1",
        message="ask a-1 deadline",
        next_due_at=1_700_000_000_000,
        created_at=1_699_000_000_000,
        kind="ask_timeout",
        hidden=True,
    )
    assert store.is_internal_wake_row(row) is True
    assert store.is_internal_wake_row(row.model_dump()) is True


def test_an_ask_timeout_row_validates_as_a_wake_schedule():
    """The ``kind`` Literal accepts it, so a booted runtime can load the row —
    without this the queue silently degrades to its in-runtime timer."""
    row = WakeSchedule(
        id="ask-timeout-a-1",
        message="ask a-1 deadline",
        next_due_at=1_700_000_000_000,
        created_at=1_699_000_000_000,
        kind="ask_timeout",
        hidden=True,
    )
    assert WakeSchedule.model_validate(row.model_dump()).kind == "ask_timeout"


def test_a_pre_ask_build_would_drop_the_row_rather_than_misfire():
    """An OLD build's ``kind`` Literal is the two-value one, and ``extra`` is
    forbidden — so it drops the row on load and the queue falls back to its timer
    rather than firing a wake whose kind it does not know."""
    with pytest.raises(Exception):
        WakeSchedule.model_validate(
            {
                "id": "ask-timeout-a-1",
                "message": "ask a-1 deadline",
                "next_due_at": 1_700_000_000_000,
                "created_at": 1_699_000_000_000,
                "kind": "some_future_kind",
                "hidden": True,
            }
        )


def test_scheduled_rows_drops_both_internal_kinds():
    rows = [
        {"id": "w1", "kind": "scheduled", "message": "standup"},
        {"id": "patience-1", "kind": "patience", "message": "wait"},
        {"id": "ask-timeout-a-1", "kind": "ask_timeout", "message": "ask a-1 deadline"},
    ]
    assert [row["id"] for row in store.scheduled_rows(rows)] == ["w1"]
    assert [row["id"] for row in patience.scheduled_rows(rows)] == ["w1"]


def test_the_patience_module_re_exports_the_one_predicate():
    """``wakes/patience.py`` re-exports both, so a caller that already imports the
    patience engine gets the union rather than a second spelling."""
    assert patience.is_internal_wake_row({"kind": "ask_timeout"}) is True
    assert patience.is_internal_wake_row({"kind": "scheduled"}) is False


def test_the_narrow_predicate_still_means_patience_only():
    """The cap, the cancel path and the watermark delivery are ABOUT patience; a
    union there would make the ask queue's timer cancellable by ``patience``."""
    assert store.is_patience_row({"kind": "patience"}) is True
    assert store.is_patience_row({"kind": "ask_timeout"}) is False
    assert patience.is_patience_row({"kind": "ask_timeout"}) is False


def test_every_listing_surface_goes_through_the_one_filter():
    """A source sweep, because the failure mode is a MISSED call site: the
    surfaces that list wakes for a human all read ``scheduled_rows``, and none of
    them filters ``scheduler.schedules`` by hand."""
    surfaces = [
        ROOT / "local_operator/tools/builtin.py",
        ROOT / "local_operator/tui/widgets/wake_panel.py",
        ROOT / "local_operator/server/utils/desktop_feed.py",
        ROOT / "local_operator/server/routes/desktop_wakes.py",
        ROOT / "local_operator/cli.py",
        ROOT / "local_operator/session/catalog.py",
    ]
    for path in surfaces:
        assert "scheduled_rows" in path.read_text(), path


def test_the_session_subtracts_the_union_from_the_visible_catch_up_fold():
    """``Session._prepare_missed_wake_catchup`` is the one human-visible fold that
    filtered patience directly; an ``ask_timeout`` fire reaching it would render a
    hidden timer as a wake the user set."""
    from local_operator.session import session as session_module

    source = inspect.getsource(session_module.Session._prepare_missed_wake_catchup)
    assert "is_internal_wake_row" in source
    assert "is_patience_row" not in source


def test_the_delivery_path_still_routes_ask_timeout_rows():
    """The counterpart rule: the delivery path must NOT subtract them — a hidden
    timer still has to be delivered (as a reconcile, never as a note)."""
    from local_operator.session import session as session_module

    source = inspect.getsource(session_module.Session._deliver_wake)
    assert "is_ask_timeout_row" in source
