"""The scoped, cursor-paged catalogue: one group's page, walked to exhaustion.

WHY A SCOPE IS A DIFFERENT QUESTION FROM A SMALLER PAGE. The sidebar used to read
the whole store (``limit=500``) and filter the page it happened to hold by each
row's binding, so a group's rows could be missing from it entirely — on the
operator's store ``team:lopdev`` held 434 conversations and a 500-row page carried
283 of them, while the group's badge read 283 and its expanded pane drew "No chats
yet" over 151 real rows. A scoped page answers "this group's top N" instead, which
is a different slice of a different ranking, and that is what most of this file
pins.

WHAT THE CURSOR IS AND IS NOT. It is the ranking KEY of the page's last row
(``(tier, wake_rank, -created_at, id)``), so a walk resumes by POSITION and a
deleted anchor breaks nothing — no offset is involved and none could be, because
``load_catalog`` re-derives the ranking on every call and there is no stable row
501. It is NOT a snapshot: ``rank``'s first two terms move with the poll, so a row
whose tier changes between two page reads may be skipped or re-sent. The walk
tests below assert the property that IS promised (no duplicate and no gap while
the store is unchanged) and the tests above them assert the tolerance that covers
the rest (an unusable or foreign cursor answers the scope's first page rather than
an error).

The binding census rides the same memoized read the scope filter needs
(``_BINDING_MEMO``), so its counts are asserted against an INDEPENDENT walk of the
store -- reading every ``attachment.json`` directly -- rather than against the
cache the implementation used to build them.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from local_operator.resume import read_session_attachment, write_session_attachment
from local_operator.session import catalog
from local_operator.session.catalog import (
    CatalogueScope,
    catalogue_page,
    decode_cursor,
    encode_cursor,
    in_scope,
    load_catalog,
)
from local_operator.session.creation import ensure_session_created_at


def _session(
    root: Path,
    session_id: str,
    *,
    team: str = "",
    agent: str = "",
    created: float = 1_000.0,
    stamp: float = 1_000.0,
) -> Path:
    """One visible conversation, with the binding the sidebar groups it by.

    ``created`` is stamped through the store's own writer because it is the
    immutable ordering key the walk is asserted against, and ``stamp`` is the
    transcript's activity. A sidecar rather than the ``st_birthtime`` fallback:
    that fallback is macOS-only, so a test that relied on it would order
    correctly here and collapse to the id tie-break in CI (the note in
    ``test_catalog_scan_cost._session`` says the same, for the same reason).
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    ensure_session_created_at(directory, created)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    os.utime(directory / "transcript.jsonl", (stamp, stamp))
    if team or agent:
        # Through the real writer: the census reads what a session actually
        # stores, so a hand-written file could agree with a wrong reader.
        write_session_attachment(directory, team=team, agent=agent, goal="")
    return directory


def _bindings(root: Path) -> dict[str, tuple[str, str]]:
    """Every visible session's binding, read STRAIGHT from the store.

    The independent authority the census is asserted against: it reads each
    ``attachment.json`` itself and never touches ``catalog._BINDING_MEMO``.
    """
    found: dict[str, tuple[str, str]] = {}
    for directory in (root / "sessions").iterdir():
        stored = read_session_attachment(directory)
        found[directory.name] = ("", "") if stored is None else (stored.team, stored.agent)
    return found


EIGHT_CHATS_PER_TEAM = 8


@pytest.fixture(autouse=True)
def _fresh_memos() -> Any:
    """No test inherits another's memo: the store differs every time.

    Cleared rather than left alone because a stale key would make a cold-read
    assertion pass for the wrong reason, which is the exact failure mode
    ``_BIRTH_MEMO``'s own tests guard against.
    """
    catalog._BINDING_MEMO.clear()
    yield
    catalog._BINDING_MEMO.clear()


class TestThePageIsTheScopesOwnTopN:
    """A scoped page is a slice of the SCOPE's ranking, not the head's filtered."""

    def test_a_group_page_holds_only_that_groups_rows(self, tmp_path: Path) -> None:
        for index in range(EIGHT_CHATS_PER_TEAM):
            _session(tmp_path, f"lop{index:05d}", team="lopdev", created=5_000.0 + index)
        for index in range(EIGHT_CHATS_PER_TEAM):
            _session(tmp_path, f"min{index:05d}", team="minervadev", created=4_000.0 + index)

        page = catalogue_page(tmp_path, scope=CatalogueScope("team", "lopdev"), limit=50)

        assert [entry.id for entry in page.entries] == [
            f"lop{index:05d}" for index in reversed(range(EIGHT_CHATS_PER_TEAM))
        ]

    def test_the_page_is_the_scopes_top_n_not_the_head_page_filtered(self, tmp_path: Path) -> None:
        """The defect this exists to remove, as an executable statement.

        A newer team fills the head page completely. Filtering that page for the
        OLDER team finds nothing — which is the "No chats yet" over real chats —
        while the older team's own first page holds its newest row.
        """
        for index in range(6):
            _session(tmp_path, f"new{index:04d}", team="newteam", created=9_000.0 + index)
        _session(tmp_path, "old0000", team="oldteam", created=1_000.0)

        head = load_catalog(tmp_path, limit=6)
        assert "old0000" not in [entry.id for entry in head], (
            "the head page is full of the newer team, so filtering THAT page for the "
            "older one finds nothing -- the 'No chats yet' drawn over a real conversation"
        )

        page = catalogue_page(tmp_path, scope=CatalogueScope("team", "oldteam"), limit=6)
        assert [entry.id for entry in page.entries] == ["old0000"]

    def test_an_agent_scope_excludes_team_attached_rows(self, tmp_path: Path) -> None:
        """The ``not team`` half of the renderer's rule, which is load-bearing.

        A session can carry BOTH names — attached to a team and to a profile —
        and the sidebar draws it under the TEAM only. Without the ``not team``
        half it would also appear under an agent group the UI never draws it in,
        which is a row in the wrong place rather than a missing row.
        """
        _session(tmp_path, "bothteam0001", team="lopdev", agent="reviewer")
        _session(tmp_path, "soloagent001", agent="reviewer")

        page = catalogue_page(tmp_path, scope=CatalogueScope("agent", "reviewer"), limit=50)

        assert [entry.id for entry in page.entries] == ["soloagent001"]
        team_page = catalogue_page(tmp_path, scope=CatalogueScope("team", "lopdev"), limit=50)
        assert [entry.id for entry in team_page.entries] == ["bothteam0001"]

    def test_unbound_sessions_are_in_no_scope(self, tmp_path: Path) -> None:
        _session(tmp_path, "unbound00001")
        _session(tmp_path, "bound0000001", team="lopdev")

        for scope in (CatalogueScope("team", "lopdev"), CatalogueScope("agent", "reviewer")):
            page = catalogue_page(tmp_path, scope=scope, limit=50)
            assert "unbound00001" not in [entry.id for entry in page.entries]

    def test_an_empty_scope_is_an_empty_page_not_an_error(self, tmp_path: Path) -> None:
        """A team with no conversations is a legitimate state, not a 404.

        The name is deliberately NOT validated against the live registry: a team
        can be renamed or deleted while its sessions keep the name their
        ``attachment.json`` was written under.
        """
        _session(tmp_path, "bound0000001", team="lopdev")

        page = catalogue_page(tmp_path, scope=CatalogueScope("team", "deleted-team"), limit=50)

        assert page.entries == []
        assert page.next_cursor is None
        assert page.cursor_missing is False

    def test_the_head_page_is_unchanged_without_a_scope(self, tmp_path: Path) -> None:
        """No scope, no cursor: the same listing ``load_catalog`` gives, in order."""
        for index in range(5):
            _session(tmp_path, f"chat{index:05d}", created=1_000.0 + index)

        page = catalogue_page(tmp_path, limit=3)
        expected = load_catalog(tmp_path, limit=3)

        assert [entry.id for entry in page.entries] == [entry.id for entry in expected]

    def test_an_excluded_id_refills_the_page_from_behind_it(self, tmp_path: Path) -> None:
        """C1 (round-2 review): an exclusion must not eat a page slot.

        ``exclude_ids`` is applied inside the ranking→window step, so the
        window, the truncation verdict and ``next_cursor`` are computed over
        the filtered list. Moving the same filter to the assembled page — the
        first cut of the draft-listing fix — answered a store with rows to
        spare with a short page and, at ``limit=1``, an EMPTY one.
        """
        for index in range(5):
            _session(tmp_path, f"chat{index:05d}", created=2_000.0 + index, stamp=2_000.0 + index)

        full = catalogue_page(tmp_path, limit=5)
        ordered = [entry.id for entry in full.entries]
        assert len(ordered) == 5
        first_id = ordered[0]

        excluded = catalogue_page(tmp_path, limit=2, exclude_ids={first_id})
        assert [entry.id for entry in excluded.entries] == ordered[1:3]

        one = catalogue_page(tmp_path, limit=1, exclude_ids={first_id})
        assert [entry.id for entry in one.entries] == ordered[1:2]
        assert one.next_cursor is not None, "more rows still follow the refilled page"


class TestTheWalkIsTotalAndUnique:
    """Paging a scope to exhaustion returns exactly the scope's rows."""

    def test_a_cursor_walk_covers_the_scope_exactly_once(self, tmp_path: Path) -> None:
        """The property the design asks to be proven BY CONSTRUCTION.

        Paged to exhaustion, the union of the pages is the same id SET (and the
        same order) as one big unscoped read of that scope: no duplicates and no
        gaps, at page sizes that do not divide the population.
        """
        for index in range(11):
            _session(tmp_path, f"lop{index:05d}", team="lopdev", created=5_000.0 + index)
        for index in range(4):
            _session(tmp_path, f"min{index:05d}", team="minervadev", created=4_000.0 + index)
        _session(tmp_path, "unbound00001")
        bindings = _bindings(tmp_path)
        scope = CatalogueScope("team", "lopdev")
        expected = [
            entry.id
            for entry in load_catalog(tmp_path, limit=200)
            if in_scope(bindings[entry.id], scope)
        ]
        assert len(expected) == 11

        seen: list[str] = []
        cursor: str | None = None
        pages = 0
        while True:
            page = catalogue_page(tmp_path, scope=scope, cursor=cursor, limit=4)
            pages += 1
            seen += [entry.id for entry in page.entries]
            assert (page.next_cursor is not None) == (len(page.entries) == 4)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
            assert pages < 10, "the walk did not terminate"

        assert pages == 3, "11 rows at 4 per page is three pages"
        assert seen == expected
        assert len(set(seen)) == len(seen), "a walk must not repeat a row"

    def test_a_walk_resumes_by_key_so_a_deleted_anchor_breaks_nothing(self, tmp_path: Path) -> None:
        """The cursor names a POSITION, so the row it was minted from may vanish.

        This is the whole reason the token carries the rank tuple rather than an
        offset or an id: with an offset, deleting one row above the boundary
        skips exactly one unreached row.
        """
        for index in range(6):
            _session(tmp_path, f"chat{index:05d}", team="lopdev", created=1_000.0 + index)

        first = catalogue_page(tmp_path, scope=CatalogueScope("team", "lopdev"), limit=3)
        assert [entry.id for entry in first.entries] == ["chat00005", "chat00004", "chat00003"]
        assert first.next_cursor is not None

        # The anchor itself goes away, and so does the row just above it.
        import shutil

        shutil.rmtree(tmp_path / "sessions" / "chat00003")
        shutil.rmtree(tmp_path / "sessions" / "chat00004")

        second = catalogue_page(
            tmp_path, scope=CatalogueScope("team", "lopdev"), cursor=first.next_cursor, limit=3
        )

        assert [entry.id for entry in second.entries] == ["chat00002", "chat00001", "chat00000"]
        assert second.next_cursor is None

    def test_the_cursor_is_minted_exactly_when_the_scope_holds_more(self, tmp_path: Path) -> None:
        """``(next_cursor is not None) == truncated``, on both sides of the edge."""
        for index in range(4):
            _session(tmp_path, f"chat{index:05d}", team="lopdev", created=1_000.0 + index)

        exact = catalogue_page(tmp_path, scope=CatalogueScope("team", "lopdev"), limit=4)
        assert len(exact.entries) == 4 and exact.next_cursor is None

        one_more = catalogue_page(tmp_path, scope=CatalogueScope("team", "lopdev"), limit=3)
        assert len(one_more.entries) == 3 and one_more.next_cursor is not None

    def test_the_last_page_of_a_walk_mints_no_cursor(self, tmp_path: Path) -> None:
        for index in range(3):
            _session(tmp_path, f"chat{index:05d}", team="lopdev", created=1_000.0 + index)

        last = catalogue_page(tmp_path, scope=CatalogueScope("team", "lopdev"), limit=5)

        assert last.next_cursor is None
        assert [entry.id for entry in last.entries] == ["chat00002", "chat00001", "chat00000"]

    def test_the_pinned_extras_ride_the_first_page_only(self, tmp_path: Path) -> None:
        """A pinned row is an EXTRA on one page of a walk, never on every page.

        ``pinned_off_page`` answers "which pinned conversations is this page
        missing" so a client holding ONE page can still draw the Pinned section --
        a promise about the page it paints first. Appended from ``ranked[limit:]``,
        the same row was re-appended on every page of a walk that still had it
        below (QA measured one pinned id twice over a seven-page walk), so a
        walk's surplus grew with the pin's distance down the listing. The row's
        OWN later position stays as it is: the row union is id-keyed and the
        design says a pinned row "may additionally be in the head".
        """
        for index in range(4):
            _session(tmp_path, f"lop{index:05d}", team="lopdev", created=5_000.0 + index)
        older = "oldpin00001"
        _session(tmp_path, older, team="lopdev", created=1_000.0)

        first = catalogue_page(tmp_path, limit=2, pinned_off_page=[older])

        assert [entry.id for entry in first.entries] == ["lop00003", "lop00002", older]
        assert first.next_cursor is not None

        surpluses: list[list[str]] = []
        cursor: str | None = first.next_cursor
        while cursor is not None:
            page = catalogue_page(tmp_path, cursor=cursor, limit=2, pinned_off_page=[older])
            surpluses.append([entry.id for entry in page.entries[2:]])
            cursor = page.next_cursor

        # The pin ranks last, so the first page's surplus is the one extra there
        # is; every page after it is the scope's own continuation and nothing
        # else, however far down the pin is.
        assert surpluses == [[], []], surpluses


class TestAnUnusableCursorIsNotAnError:
    """Bad input is answered with the scope's first page and a flag, never a 4xx."""

    @pytest.mark.parametrize(
        "token",
        [
            "not base64 at all !!",
            "e30",  # `{}` — no version, no position
            "eyJ2Ijo5OTksInMiOm51bGwsImsiOls0LDIsMSwiaWQiXX0",  # an unknown version
            "eyJ2IjoxLCJzIjpbInRlYW0iLCJsb3BkZXYiXX0",  # a version, no `k`
            "eyJ2IjoxLCJzIjpbInRlYW0iXSwiaiI6WzAsMCwxLCJpZCJdfQ",  # a torn scope
        ],
    )
    def test_an_unreadable_cursor_answers_the_first_page(self, tmp_path: Path, token: str) -> None:
        for index in range(4):
            _session(tmp_path, f"chat{index:05d}", team="lopdev", created=1_000.0 + index)
        scope = CatalogueScope("team", "lopdev")

        page = catalogue_page(tmp_path, scope=scope, cursor=token, limit=2)

        assert page.cursor_missing is True
        assert [entry.id for entry in page.entries] == ["chat00003", "chat00002"]

    def test_an_empty_cursor_is_absence_not_a_bad_token(self, tmp_path: Path) -> None:
        """``?cursor=`` is what a client sends for "no cursor", as with the scope.

        ``cursor_missing`` therefore stays FALSE: there was no cursor to lose, and
        a client that had sent one and seen the flag raised would re-read from the
        top on every poll.
        """
        for index in range(4):
            _session(tmp_path, f"chat{index:05d}", team="lopdev", created=1_000.0 + index)

        page = catalogue_page(tmp_path, scope=CatalogueScope("team", "lopdev"), cursor="", limit=2)

        assert page.cursor_missing is False
        assert [entry.id for entry in page.entries] == ["chat00003", "chat00002"]

    def test_a_cursor_from_another_scope_answers_this_scopes_first_page(
        self, tmp_path: Path
    ) -> None:
        """A foreign token is not an error either — it is a lost place.

        Resuming a filtered list from an unfiltered position (or another team's)
        would silently serve rows the scope does not contain, which is worse than
        re-reading from the top.
        """
        for index in range(4):
            _session(tmp_path, f"lop{index:05d}", team="lopdev", created=5_000.0 + index)
        for index in range(4):
            _session(tmp_path, f"min{index:05d}", team="minervadev", created=1_000.0 + index)
        other = catalogue_page(tmp_path, scope=CatalogueScope("team", "minervadev"), limit=2)
        assert other.next_cursor is not None

        page = catalogue_page(
            tmp_path, scope=CatalogueScope("team", "lopdev"), cursor=other.next_cursor, limit=2
        )

        assert page.cursor_missing is True
        assert [entry.id for entry in page.entries] == ["lop00003", "lop00002"]

    def test_an_unscoped_cursor_is_not_usable_for_a_scoped_page(self, tmp_path: Path) -> None:
        for index in range(4):
            _session(tmp_path, f"chat{index:05d}", team="lopdev", created=1_000.0 + index)
        head = catalogue_page(tmp_path, limit=2)
        assert head.next_cursor is not None

        page = catalogue_page(
            tmp_path, scope=CatalogueScope("team", "lopdev"), cursor=head.next_cursor, limit=2
        )

        assert page.cursor_missing is True
        assert [entry.id for entry in page.entries] == ["chat00003", "chat00002"]

    def test_the_head_itself_can_be_walked_by_cursor(self, tmp_path: Path) -> None:
        """The unscoped listing is a scope too — the chat region's tail pages it."""
        for index in range(5):
            _session(tmp_path, f"chat{index:05d}", created=1_000.0 + index)

        first = catalogue_page(tmp_path, limit=2)
        second = catalogue_page(tmp_path, cursor=first.next_cursor, limit=2)
        third = catalogue_page(tmp_path, cursor=second.next_cursor, limit=2)

        assert [entry.id for entry in first.entries] == ["chat00004", "chat00003"]
        assert second.cursor_missing is False
        assert [entry.id for entry in second.entries] == ["chat00002", "chat00001"]
        assert [entry.id for entry in third.entries] == ["chat00000"]
        assert third.next_cursor is None

    def test_a_cursor_from_another_scope_is_still_a_valid_token(self, tmp_path: Path) -> None:
        """The refusal is the SCOPE's, not the token's: decoding succeeds."""
        decoded = decode_cursor(encode_cursor(CatalogueScope("team", "lopdev"), (0, 2, -5.0, "x")))

        assert decoded is not None
        assert decoded.scope == CatalogueScope("team", "lopdev")
        assert decoded.key == (0, 2, -5.0, "x")


class TestTheCensusMatchesAnIndependentWalk:
    """``with_counts`` is asserted against the store, not against its own memo."""

    def test_counts_are_only_present_when_asked_for(self, tmp_path: Path) -> None:
        _session(tmp_path, "chat00000001", team="lopdev")
        _session(tmp_path, "chat00000002")

        assert catalogue_page(tmp_path, limit=50).counts is None
        assert catalogue_page(tmp_path, limit=50, with_counts=True).counts is not None

    def test_every_group_total_and_unbound_count_matches_the_store(self, tmp_path: Path) -> None:
        for index in range(5):
            _session(tmp_path, f"lop{index:05d}", team="lopdev", created=5_000.0 + index)
        for index in range(3):
            _session(tmp_path, f"min{index:05d}", team="minervadev", created=4_000.0 + index)
        _session(tmp_path, "bothteam0001", team="lopdev", agent="reviewer")
        _session(tmp_path, "soloagent001", agent="reviewer")
        _session(tmp_path, "unbound00001")
        _session(tmp_path, "unbound00002")

        counts = catalogue_page(tmp_path, limit=50, with_counts=True).counts
        assert counts is not None

        expected = Counter(
            ("team", team) if team else ("agent", agent)
            for team, agent in _bindings(tmp_path).values()
            if team or agent
        )
        assert {(tally.kind, tally.name): tally.total for tally in counts.scopes} == dict(expected)
        assert counts.total == len(_bindings(tmp_path))
        assert counts.unbound == 2
        assert counts.total == sum(expected.values()) + counts.unbound

    def test_the_census_is_drawn_from_the_whole_listing_not_the_page(self, tmp_path: Path) -> None:
        """What a COLLAPSED group's badge reads, which is why it is not page-scoped."""
        for index in range(9):
            _session(tmp_path, f"lop{index:05d}", team="lopdev", created=5_000.0 + index)

        page = catalogue_page(
            tmp_path, scope=CatalogueScope("team", "lopdev"), limit=2, with_counts=True
        )

        assert len(page.entries) == 2
        assert page.counts is not None
        assert [(tally.name, tally.total) for tally in page.counts.scopes] == [("lopdev", 9)]

    def test_the_counts_follow_a_reattached_session(self, tmp_path: Path) -> None:
        """The memo is keyed on the file, so a real re-attach moves the numbers.

        ``write_session_attachment`` publishes through a pid-named temp and
        ``os.replace``, which is what makes the new value land under a NEW inode
        key: a memo that keyed on the parsed value's identity, or that trusted a
        cached name, would keep reporting the old group forever.
        """
        directory = _session(tmp_path, "chat00000001", team="lopdev")
        first = catalogue_page(tmp_path, limit=50, with_counts=True).counts
        assert first is not None
        assert [(tally.name, tally.total) for tally in first.scopes] == [("lopdev", 1)]

        write_session_attachment(directory, team="", agent="reviewer", goal="")

        counts = catalogue_page(tmp_path, limit=50, with_counts=True).counts
        assert counts is not None
        assert [(tally.kind, tally.name, tally.total) for tally in counts.scopes] == [
            ("agent", "reviewer", 1)
        ]
        assert counts.unbound == 0

    def test_a_removed_binding_removes_the_binding_not_the_row(self, tmp_path: Path) -> None:
        directory = _session(tmp_path, "chat00000001", team="lopdev")
        (directory / "attachment.json").unlink()

        counts = catalogue_page(tmp_path, limit=50, with_counts=True).counts

        assert counts is not None
        assert counts.scopes == ()
        assert counts.unbound == 1

    def test_the_scope_order_is_stable_and_by_descending_size(self, tmp_path: Path) -> None:
        for index in range(3):
            _session(tmp_path, f"aaa{index:04d}", team="aaa", created=5_000.0 + index)
        for index in range(2):
            _session(tmp_path, f"bbb{index:04d}", team="bbb", created=4_000.0 + index)
        _session(tmp_path, "ccc0000", agent="ccc")

        counts = catalogue_page(tmp_path, limit=50, with_counts=True).counts
        assert counts is not None

        assert [(tally.kind, tally.name, tally.total) for tally in counts.scopes] == [
            ("team", "aaa", 3),
            ("team", "bbb", 2),
            ("agent", "ccc", 1),
        ]


class TestTheScopePredicate:
    """The renderer's own grouping rule, pinned where it is spelled.

    ``in_scope`` is what both the page filter and the census classify a row by, so
    its two halves are asserted directly rather than only through a page: the
    ``not team`` half in particular is the one a future reader would call a bug
    and "fix".
    """

    def test_a_team_scope_matches_on_the_team_half_alone(self) -> None:
        team = CatalogueScope("team", "lopdev")

        assert in_scope(("lopdev", ""), team) is True
        assert in_scope(("lopdev", "reviewer"), team) is True
        assert in_scope(("", "lopdev"), team) is False
        assert in_scope(("minervadev", ""), team) is False

    def test_an_agent_scope_excludes_a_team_attached_session(self) -> None:
        agent = CatalogueScope("agent", "reviewer")

        assert in_scope(("", "reviewer"), agent) is True
        assert in_scope(("lopdev", "reviewer"), agent) is False, (
            "a session attached to a team is drawn under the team, so it must not "
            "also appear in an agent group the sidebar never renders it in"
        )
        assert in_scope(("", ""), agent) is False


class TestTheCensusCostIsTheMemoizedOne:
    """T5, as counts rather than as time: one read per session, then stats only."""

    @staticmethod
    def _counting_attachment_reads(run: Any) -> tuple[Any, int]:
        """Count ``attachment.json`` opens during ``run`` (the read the memo saves)."""
        import builtins
        import io

        opened = [0]
        real_open = io.open

        def counting(file: Any, *args: Any, **kwargs: Any) -> Any:
            if str(file).endswith("attachment.json"):
                opened[0] += 1
            return real_open(file, *args, **kwargs)

        original = builtins.open, io.open
        builtins.open = io.open = counting  # type: ignore[assignment]
        try:
            return run(), opened[0]
        finally:
            builtins.open, io.open = original  # type: ignore[assignment]

    def test_a_cold_census_reads_each_binding_once_and_a_warm_one_reads_none(
        self, tmp_path: Path
    ) -> None:
        for index in range(30):
            _session(tmp_path, f"chat{index:08d}", team="lopdev", created=1_000.0 + index)

        _cold, cold_reads = self._counting_attachment_reads(
            lambda: catalogue_page(tmp_path, limit=5, with_counts=True)
        )
        _warm, warm_reads = self._counting_attachment_reads(
            lambda: catalogue_page(tmp_path, limit=5, with_counts=True)
        )

        assert cold_reads == 30, cold_reads
        assert warm_reads == 0, warm_reads

    def test_a_scoped_page_reads_the_same_census_the_counts_need(self, tmp_path: Path) -> None:
        """A scoped page cannot be filtered without every candidate's binding.

        Stated as a count because it is the honest version of "a scoped page costs
        what an unscoped page costs": it costs what the HEAD answer the app
        actually sends costs (``with_counts=true``), because the census is the
        same read. A plain unscoped page that asks for no counts reads none.
        """
        for index in range(30):
            _session(tmp_path, f"chat{index:08d}", team="lopdev", created=1_000.0 + index)
        scope = CatalogueScope("team", "lopdev")

        _plain, plain_reads = self._counting_attachment_reads(
            lambda: catalogue_page(tmp_path, limit=5)
        )
        # Each measurement starts from a cold memo, because the question is what
        # a FIRST scoped page costs: the memo is what makes the second one free.
        catalog._BINDING_MEMO.clear()
        _scoped, scoped_reads = self._counting_attachment_reads(
            lambda: catalogue_page(tmp_path, scope=scope, limit=5)
        )
        catalog._BINDING_MEMO.clear()
        _counted, counted_reads = self._counting_attachment_reads(
            lambda: catalogue_page(tmp_path, limit=5, with_counts=True)
        )

        assert plain_reads == 0, plain_reads
        assert scoped_reads == counted_reads == 30

    def test_the_memo_is_pruned_to_this_calls_candidates(self, tmp_path: Path) -> None:
        """A bound on the map rather than a cache that grows with the store's past."""
        for index in range(5):
            _session(tmp_path, f"chat{index:08d}", team="lopdev", created=1_000.0 + index)
        catalogue_page(tmp_path, limit=5, with_counts=True)
        assert len(catalog._BINDING_MEMO[str(tmp_path / "sessions")]) == 5

        import shutil

        for index in range(3):
            shutil.rmtree(tmp_path / "sessions" / f"chat{index:08d}")
        catalogue_page(tmp_path, limit=5, with_counts=True)

        assert len(catalog._BINDING_MEMO[str(tmp_path / "sessions")]) == 2


class TestTheCursorToken:
    """The token's shape, and the tolerance that lets it change later."""

    def test_the_token_is_base64url_json_and_rejects_an_unknown_version(self) -> None:
        import base64

        token = encode_cursor(None, (1, 2, -3.5, "abc"))
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
        assert json.loads(raw) == {"v": 1, "s": None, "k": [1, 2, 3.5, "abc"]}
        assert "=" not in token, "padding is stripped so the token is URL-safe"

        bumped = (
            base64.urlsafe_b64encode(
                json.dumps({"v": 2, "s": None, "k": [1, 2, 3.5, "abc"]}).encode()
            )
            .decode()
            .rstrip("=")
        )
        assert decode_cursor(bumped) is None

    @pytest.mark.parametrize(
        "key",
        [
            [0, 2, "soon", "abc"],
            [0, 2, 3.5],
            [True, 2, 3.5, "abc"],
            [0, False, 3.5, "abc"],
        ],
    )
    def test_a_rank_tuple_of_the_wrong_shape_is_refused(self, key: list[Any]) -> None:
        import base64

        token = (
            base64.urlsafe_b64encode(json.dumps({"v": 1, "s": None, "k": key}).encode())
            .decode()
            .rstrip("=")
        )

        assert decode_cursor(token) is None

    def test_an_over_long_token_is_refused_before_it_is_decoded(self) -> None:
        assert decode_cursor("a" * (catalog.CURSOR_MAX_LENGTH + 1)) is None
