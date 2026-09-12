"""One definition of "which conversation matches", shared by every surface.

The bug these pin: the TUI's ``/resume`` picker could find a session by a word
said inside it, while the phone's search and the desktop chat search — which
read the same store, for the same user, from the same machine — could not, and
neither ranked what it found. ``local_operator.session.session_search`` is now
the one implementation all three call, so these tests stand on the mechanics
itself rather than on any one surface: admission (name, id, exact body, bounded
soft), tiering (name > id > body > soft), the recency tie-break, and the gate
that decides whether the expensive soft tier runs at all.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from local_operator.harness.types import Message, MessageRole, TextContent
from local_operator.resume import SessionRow, write_session_title
from local_operator.session.search_index import SoftSearchIndex
from local_operator.session.session_search import (
    PRECISE_HITS_ENOUGH,
    RANK_BODY,
    RANK_ID,
    RANK_NAME,
    RANK_SOFT,
    filter_rows,
    matched_in_body,
    rank_rows,
    search_rows,
    search_store,
    soft_tier_wanted,
)
from local_operator.session.transcript import Transcript


def _write(session_dir: Path, *turns: tuple[MessageRole, str]) -> None:
    """Build a real transcript through the real writer, not a hand-rolled file."""

    async def build() -> None:
        transcript = Transcript(session_dir)
        for role, text in turns:
            await transcript.append_message(Message(role=role, content=[TextContent(text=text)]))

    asyncio.run(build())


def _row(session_id: str, name: str, mtime: float, *, forked: bool = False) -> SessionRow:
    return SessionRow(session_id, mtime, name, forked=forked)


class SpySoft(SoftSearchIndex):
    """A soft index that records whether anyone asked it anything."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def search(self, digests: dict[str, str], query: str) -> set[str]:  # type: ignore[override]
        self.calls.append(query)
        return super().search(digests, query)


# --- admission and ranking --------------------------------------------------


def test_a_name_hit_outranks_a_body_hit_and_recency_does_not_decide_it():
    """The whole point of the tiers: the row whose LABEL contains the query is
    the answer, even when an older name hit and a newer body hit compete."""
    rows = [
        _row("bbbb2222", "Unrelated work", 300.0),  # newest: body only
        _row("aaaa1111", "Retention sweep design", 100.0),  # oldest: name
    ]
    matches = search_rows(rows, "retention", digests={"bbbb2222": "the retention sweep runs"})

    assert [m.row.id for m in matches] == ["aaaa1111", "bbbb2222"]
    assert [m.rank for m in matches] == [RANK_NAME, RANK_BODY]
    # And only the row whose visible name does NOT explain it is marked.
    assert [m.body_match for m in matches] == [False, True]


def test_recency_breaks_ties_within_a_tier():
    """Rows arrive newest-first, so a stable sort on the tier alone keeps
    recency order inside every tier — which is what the field expects when two
    matches are equally relevant."""
    rows = [
        _row("cccc3333", "Retention sweep design", 300.0),
        _row("bbbb2222", "Retention sweep rewrite", 200.0),
        _row("aaaa1111", "Retention sweep notes", 100.0),
    ]
    matches = search_rows(rows, "retention")

    assert [m.row.id for m in matches] == ["cccc3333", "bbbb2222", "aaaa1111"]
    assert {m.rank for m in matches} == {RANK_NAME}


def test_the_id_is_searchable_and_sits_below_the_name():
    rows = [_row("aaaa1111", "Unrelated", 300.0), _row("bbbb2222", "aaaa1111 report", 100.0)]
    matches = search_rows(rows, "aaaa1111")

    assert [m.rank for m in matches] == [RANK_NAME, RANK_ID]
    assert [m.row.id for m in matches] == ["bbbb2222", "aaaa1111"]


def test_a_typo_finds_the_row_and_is_marked_as_a_body_match():
    """``classifer`` -> ``classifier``: the bound that makes a mistyped search
    work WITHOUT letting unrelated words collide."""
    rows = [_row("aaaa1111", "Improve ADM Classifier Throughput", 100.0)]
    digests = {"aaaa1111": "Improve ADM Classifier Throughput the classifier is slow"}
    matches = search_rows(rows, "classifer", digests=digests)

    assert [m.row.id for m in matches] == ["aaaa1111"]
    # Ranked as a soft hit, but STILL explained: the visible name does contain
    # the words, just not the typo as typed, so the row is admitted on its
    # conversation and says so.
    assert matches[0].rank == RANK_SOFT
    assert matches[0].body_match is True


def test_word_order_does_not_matter_for_a_soft_match():
    rows = [_row("aaaa1111", "Throughput classifier work", 100.0)]
    matches = search_rows(
        rows,
        "classifier throughput",
        digests={"aaaa1111": "throughput classifier work continued"},
    )
    assert [m.row.id for m in matches] == ["aaaa1111"]


def test_an_unrelated_query_admits_nothing():
    """The bound on soft matching has to actually bound it: a search that
    returns every row is indistinguishable from a broken one."""
    rows = [_row("aaaa1111", "Retention sweep design", 100.0)]
    assert search_rows(rows, "kubernetes", digests={"aaaa1111": "retention sweep design"}) == []


def test_an_empty_query_is_not_a_search():
    """Every row is the answer, in the order it arrived (recency), with limit
    honoured and nothing marked as a body match."""
    rows = [_row("cccc3333", "C", 300.0), _row("bbbb2222", "B", 200.0)]
    matches = search_rows(rows, "   ", digests={"cccc3333": "c"}, limit=1)

    assert [m.row.id for m in matches] == ["cccc3333"]
    assert matches[0].rank == RANK_NAME and matches[0].body_match is False


def test_a_caller_without_an_index_keeps_the_name_and_id_search():
    """Degradation, not failure: a host with no digest index (a test, an
    embedder, a store that could not be digested) still filters by name and id
    exactly as it did before body search existed."""
    rows = [_row("aaaa1111", "Retention sweep design", 100.0), _row("bbbb2222", "Other", 50.0)]

    assert [m.row.id for m in search_rows(rows, "retention")] == ["aaaa1111"]
    assert [m.row.id for m in search_rows(rows, "aaaa1111")] == ["aaaa1111"]
    assert search_rows(rows, "retention sweep design here") == []


def test_a_fork_tag_is_matchable_and_ranks_on_the_name():
    """What is displayed has to be searchable: a row rendered with ``[fork]``
    is found by typing it, and it is a precise name hit rather than a fuzzy
    body one."""
    rows = [_row("aaaa1111", "Refactor the loader", 100.0, forked=True)]
    matches = search_rows(rows, "fork")

    assert [m.row.id for m in matches] == ["aaaa1111"]
    assert matches[0].rank == RANK_NAME
    assert matched_in_body(matches[0].row, "fork", {"aaaa1111"}) is False


# --- the soft-tier gate -----------------------------------------------------


def test_the_soft_tier_is_skipped_once_the_name_answers_precisely():
    """Below the floor the expensive tier runs; at or above it the query is
    already answered, and fuzzy additions would only dilute it."""
    rows = [_row(f"id{i}", f"retention sweep {i}", 100.0 - i) for i in range(PRECISE_HITS_ENOUGH)]
    spy = SpySoft()

    assert soft_tier_wanted(rows, "retention") is False
    search_rows(rows, "retention", digests={row.id: "body" for row in rows}, soft=spy)
    assert spy.calls == []

    # One row short of the floor: the tier is consulted, because one precise
    # hit is as often incidental as deliberate.
    assert soft_tier_wanted(rows[:-1], "retention") is True
    search_rows(rows[:-1], "retention", digests={row.id: "body" for row in rows[:-1]}, soft=spy)
    assert spy.calls == ["retention"]


def test_an_incidental_body_hit_does_not_silence_the_tier():
    """The gate counts NAME and ID hits only. Gating on "did anything match"
    let one incidental body hit anywhere in the store silence the tier for the
    whole query, which lost the typo it exists to rescue."""
    rows = [_row("aaaa1111", "Some unrelated title", 100.0)]
    digests = {"aaaa1111": "we discussed retention once in passing"}

    assert soft_tier_wanted(rows, "retention") is True
    # And the row it admits is still ranked as a body hit.
    matches = search_rows(rows, "retention", digests=digests)
    assert [m.rank for m in matches] == [RANK_BODY]


def test_the_soft_set_admits_a_row_the_exact_tier_missed():
    """``filter_rows`` takes the caller's admitted set, so folding the soft set
    in is what puts a typo'd row on screen — the exact body set alone would not
    have it."""
    rows = [_row("aaaa1111", "Improve ADM Classifier Throughput", 100.0)]
    assert filter_rows(rows, "classifer", set()) == []
    assert [row.id for row in filter_rows(rows, "classifer", {"aaaa1111"})] == ["aaaa1111"]


# --- the one-shot store entry point ----------------------------------------


def test_search_store_finds_a_session_by_a_word_only_its_conversation_holds(tmp_path: Path):
    """End to end over a real store: the reported failure, through the entry
    point the phone and the desktop call."""
    _write(
        tmp_path / "sessions" / "aaaa1111",
        ("user", "hey can you look at this thing"),
        ("assistant", "The retention sweep is evicting live session directories."),
    )

    matches = search_store(tmp_path, "retention")

    assert [m.row.id for m in matches] == ["aaaa1111"]
    # The row's visible name is its opener, which says nothing about retention,
    # so the row is explained as a body match rather than appearing arbitrary.
    assert matches[0].row.name == "hey can you look at this thing"
    assert matches[0].body_match is True
    assert matches[0].rank == RANK_BODY


def test_a_name_the_session_no_longer_wears_is_still_searchable(tmp_path: Path):
    """A topic-pivot session is findable by the subject it ENDED on. The
    digest folds every name the session has borne, which is the only reason a
    word that is now only in its history matches at all."""
    session = tmp_path / "sessions" / "aaaa1111"
    _write(session, ("user", "opener"))
    write_session_title(
        session,
        "Cleanup work",
        user_set=True,
        past_names=["Cleanup work", "Retention sweep design"],
    )

    matches = search_store(tmp_path, "retention")

    assert [m.row.id for m in matches] == ["aaaa1111"]
    # Not the name it is displayed with, so it must say WHY it is here.
    assert matches[0].row.name == "Cleanup work"
    assert matches[0].body_match is True


def test_search_store_scans_the_whole_store_rather_than_a_page(tmp_path: Path):
    """A cap would make a session past it unfindable — indistinguishable from
    one that was deleted — which is the bug the picker documents at its own
    call site. The only cap here is on the RESULT, never on the scan."""
    for index in range(6):
        _write(
            tmp_path / "sessions" / f"session{index}",
            ("user", f"unrelated opener {index}"),
            ("assistant", f"a note about the migration phase {index}"),
        )

    everything = search_store(tmp_path, "migration")
    limited = search_store(tmp_path, "migration", limit=2)

    assert len(everything) == 6
    assert [m.row.id for m in limited] == [m.row.id for m in everything[:2]]


def test_search_store_answers_an_empty_query_with_the_store_in_recency_order(tmp_path: Path):
    for index in range(3):
        _write(tmp_path / "sessions" / f"session{index}", ("user", f"opener {index}"))

    matches = search_store(tmp_path, "", limit=2)

    assert len(matches) == 2
    assert all(m.body_match is False for m in matches)
    # Newest first, the same order the list surface shows.
    assert [m.row.mtime for m in matches] == sorted((m.row.mtime for m in matches), reverse=True)


def test_concurrent_searches_do_not_answer_from_each_others_corpus(tmp_path: Path):
    """The shared accelerator and the exact-search memo are process-wide, and
    the server calls this module from worker threads. A search built from
    another store's corpus returns the WRONG rows, so the shared path is
    serialized and this pins that two stores searched at once each answer
    about themselves."""
    for letter in ("aaa", "bbb"):
        _write(
            tmp_path / letter / "sessions" / f"{letter}1111",
            ("user", f"opener for {letter}"),
            ("assistant", f"the {letter} keyword appears only here"),
        )

    results: dict[str, list[str]] = {}
    errors: list[BaseException] = []

    def run(letter: str) -> None:
        try:
            for _ in range(5):
                hits = search_store(tmp_path / letter, f"{letter} keyword")
                results[letter] = [m.row.id for m in hits]
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(letter,)) for letter in ("aaa", "bbb")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert results == {"aaa": ["aaa1111"], "bbb": ["bbb1111"]}


def test_a_ranked_row_is_the_same_row_the_filter_admitted(tmp_path: Path):
    """``rank_rows`` must not reorder rows into something the filter would not
    have shown, and it must not drop one either."""
    _write(tmp_path / "sessions" / "aaaa1111", ("user", "retention sweep design"))
    rows = search_store(tmp_path, "", limit=10)
    assert [row.id for row in rank_rows([m.row for m in rows], "retention")] == ["aaaa1111"]
