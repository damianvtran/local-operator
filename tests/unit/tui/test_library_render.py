"""Pure tests for the ``/library`` page's state and formatting.

The page's widget tests drive the real app in ``test_library_view``; these
cover the decisions that do not need a terminal: which query a set of controls
adds up to, how a sort cycles, how a size/date reads, and how the composer
prompt names an asset. The service remains the authority on filtering — these
tests assert the QUERY the page would send, not a second implementation of the
filter.
"""

from __future__ import annotations

import time

from local_operator.tui import library_render
from local_operator.tui.library_render import LibraryState


def test_sort_cycles_recent_name_size_type_and_returns_to_recent() -> None:
    state = LibraryState()
    seen = [state.cycle_sort() for _ in range(4)]
    assert seen == ["name", "size", "type", "recent"]
    assert state.sort == "recent"


def test_type_is_the_reader_word_for_the_services_kind_sort() -> None:
    state = LibraryState(sort="type")
    assert state.query().sort == "kind"
    assert library_render.service_sort("recent") == "recent"


def test_a_category_filter_is_exclusive_and_clears_on_the_same_row() -> None:
    state = LibraryState()
    state.select_category("image")
    assert state.query().categories == frozenset({"image"})
    state.select_category("image")
    assert state.query().categories == frozenset()


def test_each_status_row_maps_to_its_own_query_field() -> None:
    state = LibraryState()
    state.select_status("highlighted")
    query = state.query()
    assert (query.highlighted, query.organized, query.favorite) == (True, None, None)
    state.select_status("unorganized")
    query = state.query()
    assert (query.highlighted, query.organized, query.favorite) == (None, False, None)
    state.select_status("favorites")
    query = state.query()
    assert (query.highlighted, query.organized, query.favorite) == (None, None, True)


def test_a_tag_filters_by_id_and_displays_its_name() -> None:
    """QA round 1: the service compares tag IDs; the page paints names."""
    state = LibraryState()
    state.select_tag("58e2791934", "q3")
    assert state.query().tag == "58e2791934"
    assert state.tag_label == "q3"
    state.select_tag("58e2791934", "q3")
    assert state.query().tag is None
    assert state.tag_label is None


def test_search_is_trimmed_and_the_cursor_resets() -> None:
    state = LibraryState(cursor=5)
    state.set_query("  chart  ")
    assert state.query().q == "chart"
    assert state.cursor == 0


def test_the_cursor_clamps_at_both_ends() -> None:
    state = LibraryState(cursor=2)
    assert state.move(-10, 5) == 0
    assert state.move(2, 5) == 2
    assert state.move(10, 5) == 4
    # An empty list still has one row position, never a negative index.
    assert LibraryState(cursor=3).clamp(0) == 0


def test_filtered_is_about_any_control_not_the_row_count() -> None:
    assert LibraryState().filtered() is False
    assert LibraryState(q=" x ").filtered() is True
    assert LibraryState(category="image").filtered() is True
    assert LibraryState(status="favorites").filtered() is True
    assert LibraryState(tag="q3").filtered() is True


def test_size_reads_in_the_units_the_service_reports() -> None:
    assert library_render.format_size(0) == "0 B"
    assert library_render.format_size(1024) == "1.00 KB"
    assert library_render.format_size(8_412_000) == "8.02 MB"
    assert library_render.format_size(None) == "—"


def test_when_is_relative_for_a_week_then_a_date(monkeypatch) -> None:
    now = 1_800_000_000.0
    assert library_render.format_when(now - 10, now=now) == "now"
    assert library_render.format_when(now - 120, now=now) == "2m"
    assert library_render.format_when(now - 7200, now=now) == "2h"
    assert library_render.format_when(now - 3 * 86_400, now=now) == "3d"
    assert library_render.format_when(now - 10 * 86_400, now=now) == time.strftime(
        "%Y-%m-%d", time.localtime(now - 10 * 86_400)
    )
    assert library_render.format_when(None) == ""


def test_the_tag_grammar_splits_on_commas_and_keeps_the_first_spelling() -> None:
    assert library_render.tag_names(" q3, deck ,, Q3\nboard ") == ["q3", "deck", "board"]
    assert library_render.tag_names("   ") == []
    assert library_render.too_many_tags([f"t{i}" for i in range(13)]) is True
    assert library_render.too_many_tags([f"t{i}" for i in range(12)]) is False


def test_the_composer_reference_names_a_path_where_the_index_has_one() -> None:
    row = {
        "id": "la_0000000000000001",
        "name": "chart.png",
        "kind": "image",
        "ref": {"path": "/tmp/chart.png"},
        "occurrences": [{"session_id": "abc", "session_title": "board prep"}],
    }
    shape, values = library_render.ask_reference(row)
    assert shape == "path"
    assert values == {
        "name": "chart.png",
        "path": "/tmp/chart.png",
        "kind": "image",
        "session": "board prep",
    }


def test_a_digest_only_asset_is_named_by_id_not_by_an_invented_path() -> None:
    row = {
        "id": "la_0000000000000002",
        "name": "mood board",
        "kind": "image",
        "ref": {"path": None},
        "occurrences": [],
    }
    shape, values = library_render.ask_reference(row)
    assert shape == "digest"
    assert values == {"name": "mood board", "id": "la_0000000000000002", "kind": "image"}


def test_category_options_put_all_first_with_the_catalog_total() -> None:
    counts = {"category": {"image": 2, "document": 3}}
    rows = library_render.category_options(counts)
    assert rows[0] == ("", 5)
    assert ("image", 2) in rows
    assert ("video", 0) in rows


def test_tag_groups_keep_the_stores_category_order_and_never_drop_a_tag() -> None:
    groups = library_render.tag_groups(
        [
            {"name": "deck", "category": "project", "count": 2},
            {"name": "q3", "category": "topic", "count": 1},
            {"name": "odd", "category": "made-up", "count": 1},
        ]
    )
    names = [name for name, _ in groups]
    assert names == ["topic", "project", "custom"]
    assert groups[0][1][0]["name"] == "q3"
