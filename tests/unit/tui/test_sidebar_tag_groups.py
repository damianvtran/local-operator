"""TUI tag presentation: the persisted pref and the grouped sidebar rows.

The STORE semantics (toggle/create/primary) live in
``tests/unit/session/test_session_tags.py``. This file covers the two things
that are specific to the terminal: the durable `/tags` preference and the
sidebar's section → primary-tag buckets, including the geometry risk the design
preflight names — a tag heading is a painted row, not selectable, and it must
consume capacity like every other heading.
"""

from __future__ import annotations

from pathlib import Path

from local_operator.resume import SessionRow
from local_operator.tui.session_catalog import CatalogEntry, rank_entries
from local_operator.tui.sidebar_tags import read_tag_groups, write_tag_groups
from local_operator.tui.widgets.session_sidebar import SessionSidebar

A = ("aaaaaaaaaa", "Alpha")
B = ("bbbbbbbbbb", "Beta")
C = ("cccccccccc", "Gamma")


def _entry(entry_id: str, mtime: float, name: str, *, busy: bool = False) -> CatalogEntry:
    return CatalogEntry(
        SessionRow(entry_id, mtime, name, live_state="busy" if busy else ""),
    )


def _sidebar(
    entries: list[CatalogEntry],
    primaries: dict[str, str],
    *,
    enabled: bool = True,
    categories: tuple[tuple[str, str], ...] = (A, B, C),
) -> SessionSidebar:
    sidebar = SessionSidebar()
    sidebar.set_entries(entries)
    sidebar.set_tag_groups(enabled, categories, primaries)
    return sidebar


def _headers(sidebar: SessionSidebar) -> list[str]:
    return [
        kind
        for kind, entry in sidebar._display_rows()
        if entry is None and kind.startswith("header:")
    ]


def _row_ids(sidebar: SessionSidebar) -> list[str]:
    return [entry.id for _kind, entry in sidebar._display_rows() if entry is not None]


def test_tag_groups_are_persisted_and_default_off(tmp_path: Path) -> None:
    assert read_tag_groups(tmp_path) is False
    assert write_tag_groups(tmp_path, True) is True
    assert read_tag_groups(tmp_path) is True
    # Identical bytes are not rewritten: a poll or a repeated `/tags on` must
    # not touch the file for nothing.
    assert write_tag_groups(tmp_path, True) is False
    assert write_tag_groups(tmp_path, False) is True
    assert read_tag_groups(tmp_path) is False


def test_malformed_tag_group_preference_reads_as_off(tmp_path: Path) -> None:
    (tmp_path / "sidebar-tags.json").write_text("{not json")
    assert read_tag_groups(tmp_path) is False
    (tmp_path / "sidebar-tags.json").write_text('{"version": 99, "group_by_tag": true}')
    assert read_tag_groups(tmp_path) is False


def test_sections_subdivide_by_primary_tag_then_untagged() -> None:
    # All three rows are ACTIVE, so one outer section and three bucket headings.
    entries = [
        _entry("111111111111", 30, "Alpha row", busy=True),
        _entry("222222222222", 20, "Beta row", busy=True),
        _entry("333333333333", 10, "Untagged row", busy=True),
    ]
    sidebar = _sidebar(
        entries,
        {"111111111111": A[0], "222222222222": B[0]},
    )
    assert _headers(sidebar) == [
        "header:active",
        f"header:tag:{A[0]}",
        f"header:tag:{B[0]}",
        "header:tag:untagged",
    ]
    # Each conversation is painted ONCE, in base order within its bucket.
    assert _row_ids(sidebar) == ["111111111111", "222222222222", "333333333333"]


def test_secondary_tags_never_duplicate_and_primary_wins() -> None:
    entries = [
        _entry("111111111111", 20, "Primary Alpha", busy=True),
        _entry("222222222222", 10, "Primary Beta", busy=True),
    ]
    # 111 is assigned Alpha AND Beta but its primary is Alpha; only Alpha may
    # bucket it. 222's primary is Beta.
    sidebar = _sidebar(entries, {"111111111111": A[0], "222222222222": B[0]})
    assert _row_ids(sidebar) == ["111111111111", "222222222222"]
    assert _headers(sidebar) == [
        "header:active",
        f"header:tag:{A[0]}",
        f"header:tag:{B[0]}",
    ]


def test_occupied_untagged_bucket_is_explicit_and_empty_ones_are_omitted() -> None:
    only_alpha = _sidebar(
        [_entry("111111111111", 10, "Alpha", busy=True)],
        {"111111111111": A[0]},
    )
    assert "header:tag:untagged" not in _headers(only_alpha)

    all_untagged = _sidebar([_entry("111111111111", 10, "Loose", busy=True)], {})
    assert _headers(all_untagged) == ["header:active", "header:tag:untagged"]


def test_unknown_primary_is_untagged_not_its_own_bucket() -> None:
    """A stale id cannot invent a heading; the widget drops it at adoption."""
    sidebar = _sidebar(
        [_entry("111111111111", 10, "Stale", busy=True)],
        {"111111111111": "dddddddddd"},
    )
    assert _headers(sidebar) == ["header:active", "header:tag:untagged"]


def test_off_restores_the_exact_flat_order_and_chrome() -> None:
    entries = [
        _entry("111111111111", 30, "Alpha", busy=True),
        _entry("222222222222", 20, "Beta"),
        _entry("333333333333", 10, "Loose"),
    ]
    sidebar = _sidebar(entries, {"111111111111": A[0], "222222222222": B[0]})
    assert _headers(sidebar) == [
        "header:active",
        f"header:tag:{A[0]}",
        "header:previous",
        f"header:tag:{B[0]}",
        "header:tag:untagged",
    ]

    sidebar.set_tag_groups(False)
    assert tuple(sidebar.entries) == rank_entries(entries)
    assert _headers(sidebar) == [
        "header:active",
        "header:previous",
    ]


def test_tag_headings_are_counted_by_the_capacity_arithmetic() -> None:
    """The frame and `page_size` must agree about how many non-row lines there are.

    This is the sidebar's recorded risk: section chrome was counted in one place
    and painted in another. Tag headings join the SAME key, so the count of
    non-row display lines and `_chrome_for` have to agree exactly.
    """
    entries = [
        _entry("111111111111", 30, "Alpha", busy=True),
        _entry("222222222222", 20, "Beta", busy=True),
        _entry("333333333333", 10, "Loose", busy=True),
    ]
    sidebar = _sidebar(entries, {"111111111111": A[0], "222222222222": B[0]})
    painted_chrome = sum(1 for kind, _entry in sidebar._display_rows() if kind != "entry")
    assert painted_chrome == sidebar._chrome_for(entries)


def test_tag_headings_are_not_selectable_rows() -> None:
    entries = [
        _entry("111111111111", 30, "Alpha", busy=True),
        _entry("222222222222", 20, "Loose", busy=True),
    ]
    sidebar = _sidebar(entries, {"111111111111": A[0]})
    rows = sidebar._display_rows()
    heading_y = next(y for y, (kind, _entry) in enumerate(rows) if kind.startswith("header:tag:"))
    assert sidebar._entry_at(heading_y) is None
    entry_y = next(y for y, (kind, _entry) in enumerate(rows) if kind == "entry")
    assert sidebar._entry_at(entry_y) is not None
