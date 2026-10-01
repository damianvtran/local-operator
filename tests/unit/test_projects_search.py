"""``projects_search`` — fold, match tiers, weights, order, memoisation.

The ranking rules this file pins are the operator's ("title is higher value
than description higher value than updates") and the architecture note's §2;
the memoisation pins are STRUCTURAL (fold/tokenise call counts), not timing
(AGENTS.md, "Prefer a structural invariant to a numeric one") — the wall
figures live in ``scripts/bench_projects_search.py`` and the PR.

Every test starts from fresh module-level indexes (the ``fresh_indexes``
fixture): the singletons are process-wide by design, so a test that asserts
call counts must own their initial state.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from uuid import uuid4

import pytest

import local_operator.projects_search as ps
from local_operator.projects import Project
from local_operator.projects_search import ProjectSearchMatch, search_projects


@pytest.fixture(autouse=True)
def fresh_indexes(monkeypatch):
    """Give each test its own empty per-field singletons."""

    monkeypatch.setattr(ps, "_INDEXES", {field: ps._FieldIndex() for field in ps._FIELDS})


def _pid() -> str:
    return uuid4().hex[:12]


def _row(project_id: str, name: str, **fields: Any) -> Project:
    """One minimal valid row; only what the test sets is populated."""

    return Project(id=project_id, name=name, **fields)


def _update(text: str) -> dict[str, str]:
    return {"at": "2026-01-01T00:00:00Z", "text": text, "by": "operator"}


def _ids(matches: list[ProjectSearchMatch]) -> list[str]:
    return [match.id for match in matches]


def test_fold_matches_diacritics_and_case_in_both_directions():
    """``Café`` is found by ``cafe``, and ``cafe`` by ``CAFÉ`` — one fold."""

    row = _row(_pid(), "cafe-ops", description="Café launch checklist")
    assert _ids(search_projects([row], "cafe")) == [row.id]
    assert _ids(search_projects([row], "CAFÉ")) == [row.id]
    other = _row(_pid(), "uber-rollout", description="über migration")
    assert _ids(search_projects([other], "uber")) == [other.id]


def test_prefixes_and_bounded_typos_match_per_token():
    row = _row(_pid(), "adm", title="Improve ADM Classifier Throughput")
    assert search_projects([row], "coord") == []  # sanity: no false positive
    assert _ids(search_projects([row], "through")) == [row.id]  # prefix
    assert _ids(search_projects([row], "classifer")) == [row.id]  # typo, distance 2
    assert _ids(search_projects([row], "classifr")) == [row.id]  # typo, distance 1
    # Below the 4-char floor there is no fuzzy tier: "adn" is one edit from
    # "adm" but must not match it.
    assert search_projects([row], "adn") == []


def test_word_order_is_free_and_tokens_may_land_in_different_fields():
    row = _row(
        _pid(), "kafka-migration", updates=[_update("cutover planning for the billing service")]
    )
    # Order-independent AND across fields: "billing" lives in updates, "kafka" in the name.
    assert _ids(search_projects([row], "billing kafka")) == [row.id]
    (match,) = search_projects([row], "kafka billing")
    # name: 8 x 1 token; updates: 2 x 1 token — and NO phrase bonus, because the
    # whole query is not contiguous in either field (a split phrase is counts only).
    assert match.score == 10.0
    assert match.fields == ("name", "updates")
    # A token that matches nothing excludes the row ("excludes rather than ranks").
    assert search_projects([row], "kafka zzz") == []


def test_title_outranks_description_outranks_updates():
    """The operator's tiers, as order AND as numbers (the tunable dict, pinned)."""

    in_name = _row(_pid(), "kafka-ingest", updated_at=1.0)
    in_desc = _row(_pid(), "streams", description="kafka drop-in", updated_at=1.0)
    in_updates = _row(_pid(), "pipeline", updates=[_update("kafka checklists")], updated_at=1.0)
    matches = search_projects([in_updates, in_desc, in_name], "kafka")
    assert _ids(matches) == [in_name.id, in_desc.id, in_updates.id]
    by_id = {match.id: match for match in matches}
    assert by_id[in_name.id].score == 16.0  # 8 x 1 token + 8 phrase
    assert by_id[in_desc.id].score == 8.0  # 4 + 4
    assert by_id[in_updates.id].score == 4.0  # 2 + 2
    assert by_id[in_name.id].fields == ("name",)
    assert by_id[in_desc.id].fields == ("description",)
    assert by_id[in_updates.id].fields == ("updates",)


def test_phrase_bonus_needs_contiguity_and_punctuation_is_not_stripped():
    phrase = _row(_pid(), "beta-cut", description="coordination links ship Tuesday", updated_at=1.0)
    split = _row(
        _pid(), "beta-cut-2", description="links coordination ship Tuesday", updated_at=1.0
    )
    hyphen = _row(
        _pid(), "beta-cut-3", description="coordination-links ship Tuesday", updated_at=1.0
    )
    matches = search_projects([phrase, split, hyphen], "coordination links")
    by_id = {match.id: match for match in matches}
    # Contiguous: 2 tokens x 4 + 4 bonus.
    assert by_id[phrase.id].score == 12.0
    # Swapped order matches (token-AND) but earns no phrase bonus.
    assert by_id[split.id].score == 8.0
    # The documented v1 limit: punctuation inside the phrase misses the bonus.
    assert by_id[hyphen.id].score == 8.0
    assert matches[0].id == phrase.id


def test_tags_people_and_progress_fields_score_and_are_attributed():
    tagged = _row(_pid(), "rollout", tags=["q4"], updated_at=1.0)
    owned = _row(_pid(), "audit", owner="rhea", updated_at=1.0)
    teamed = _row(_pid(), "migration", team="atlas", updated_at=1.0)
    progressed = _row(
        _pid(), "intake", progress="rhea unblocked", progress_reported_by="operator", updated_at=1.0
    )
    matches = search_projects([tagged, owned, teamed, progressed], "rhea")
    by_id = {match.id: match for match in matches}
    assert by_id[owned.id].fields == ("owner",)
    assert by_id[owned.id].score == 6.0  # 3 + 3 phrase
    assert by_id[progressed.id].fields == ("progress",)
    assert by_id[progressed.id].score == 4.0  # 2 + 2 phrase
    (tag_match,) = search_projects([tagged], "q4")
    assert tag_match.fields == ("tags",) and tag_match.score == 8.0
    (team_match,) = search_projects([teamed], "atlas")
    assert team_match.fields == ("team",) and team_match.score == 6.0
    # ``progress_reported_by`` rides the progress field (the note's "also covers").
    plain = _row(_pid(), "plain", progress_reported_by="operator")
    (reporter,) = search_projects([plain], "operator")
    assert reporter.fields == ("progress",)


def test_the_id_field_is_searchable_and_ranks_weakest():
    row = _row("deadbeefcafe", "opaque")
    (match,) = search_projects([row], "deadbeef")
    assert match.fields == ("id",)
    assert match.score == 2.0  # 1 x token + 1 phrase


def test_score_is_weight_times_tokens_plus_a_phrase_bonus():
    row = _row(_pid(), "beta", title="beta daily cut")
    # One token, prefix match, contiguous: 8 + 8.
    assert search_projects([row], "bet")[0].score == 16.0
    # One token, typo match (not a substring): 8, no bonus.
    assert search_projects([row], "betta")[0].score == 8.0


def test_a_token_in_two_fields_counts_and_lists_in_both():
    row = _row(_pid(), "beta-cut", description="beta notes")
    (match,) = search_projects([row], "beta")
    assert match.score == 24.0  # name: 8 + 8 phrase; description: 4 + 4 phrase
    assert match.fields == ("name", "description")


def test_the_display_title_does_not_hide_the_addressing_name():
    row = _row(_pid(), "payments-v2", title="Billing rewrite")
    assert _ids(search_projects([row], "payments")) == [row.id]
    assert _ids(search_projects([row], "billing")) == [row.id]
    (match,) = search_projects([row], "payments")
    assert match.name == "Billing rewrite"  # display_name: the title wins


def test_the_order_is_total_and_byte_identical_for_identical_inputs():
    rows = [
        _row(_pid(), f"proj-{i:02d}", description="shared keyword", updated_at=float(i % 3))
        for i in range(12)
    ]
    first = search_projects(rows, "keyword")
    again = search_projects(rows, "keyword")
    assert [(m.id, m.score, m.fields) for m in first] == [(m.id, m.score, m.fields) for m in again]
    # Row order in the input must not change the output order.
    shuffled = rows[7:] + rows[:7]
    assert _ids(search_projects(shuffled, "keyword")) == _ids(first)


def test_ties_break_on_updated_at_then_display_name():
    newer = _row(_pid(), "alpha", description="keyword here", updated_at=5.0)
    older = _row(_pid(), "beta", description="keyword here", updated_at=1.0)
    assert _ids(search_projects([older, newer], "keyword")) == [newer.id, older.id]
    # Same score and stamp: display name asc — and the TITLE is the display
    # name, so this pair orders opposite to its raw names on purpose.
    zulu = _row(_pid(), "a-raw", title="Zulu", description="keyword", updated_at=2.0)
    alpha = _row(_pid(), "b-raw", title="Alpha", description="keyword", updated_at=2.0)
    assert _ids(search_projects([zulu, alpha], "keyword")) == [alpha.id, zulu.id]


def test_the_id_is_the_last_resort_tiebreak():
    """Titles CAN collide, so equal everything-but-id must fall to the id, asc."""

    later = _row(
        "ffffffffffff", "twin", title="Same Display", description="keyword", updated_at=2.0
    )
    earlier = _row(
        "000000000001", "twin", title="Same Display", description="keyword", updated_at=2.0
    )
    assert _ids(search_projects([later, earlier], "keyword")) == [earlier.id, later.id]
    assert _ids(search_projects([earlier, later], "keyword")) == [earlier.id, later.id]


def test_limit_truncates_after_ranking():
    rows = [
        _row(_pid(), f"proj-{i:02d}", description="keyword", updated_at=float(10 - i))
        for i in range(5)
    ]
    full = search_projects(rows, "keyword")
    assert len(full) == 5
    assert _ids(search_projects(rows, "keyword", limit=2)) == _ids(full)[:2]


def test_an_empty_or_punctuation_only_query_answers_nothing():
    row = _row(_pid(), "alpha")
    assert search_projects([row], "") == []
    assert search_projects([row], "   ") == []
    assert search_projects([row], "!!! ---") == []


def test_repeat_searches_reuse_the_fold_and_reindex_only_what_changed(monkeypatch):
    """Structural memoisation pin: warm calls re-fold NOTHING but the query;

    a change folds exactly ONE field. Counts, not timings — a regression that
    drops the raw-first freshness (re-folding the whole corpus per request)
    fails here deterministically.
    """

    calls = {"fold": 0, "tokenize": 0}
    real_fold, real_tokenize = ps.normalize, ps._tokenize

    def counting_fold(text: str) -> str:
        calls["fold"] += 1
        return real_fold(text)

    def counting_tokenize(text: str) -> list[str]:
        calls["tokenize"] += 1
        return real_tokenize(text)

    monkeypatch.setattr(ps, "normalize", counting_fold)
    monkeypatch.setattr(ps, "_tokenize", counting_tokenize)

    row = _row(_pid(), "alpha", description="first text")
    search_projects([row], "alpha")
    # One fold per field, plus the query's own fold/tokenisation.
    assert calls["fold"] == len(ps._FIELDS) + 1
    assert calls["tokenize"] == len(ps._FIELDS) + 1

    calls["fold"] = calls["tokenize"] = 0
    search_projects([row], "alpha")
    # Unchanged content: only the query itself is folded/tokenised again.
    assert calls["fold"] == 1
    assert calls["tokenize"] == 1

    calls["fold"] = calls["tokenize"] = 0
    changed = row.model_copy(update={"description": "second text"})
    search_projects([changed], "second")
    # Exactly the changed field is re-folded, beside the query.
    assert calls["fold"] == 2
    assert calls["tokenize"] == 2


def test_changed_text_is_reindexed_and_removed_rows_are_pruned():
    row = _row(_pid(), "alpha", description="kafka migration")
    assert _ids(search_projects([row], "kafka")) == [row.id]
    # The store changed: the old term is gone, the new one is findable.
    updated = row.model_copy(update={"description": "rabbit migration"})
    assert search_projects([updated], "kafka") == []
    assert _ids(search_projects([updated], "rabbit")) == [row.id]
    # The row left the store: the cache is pruned to the live set.
    assert search_projects([], "rabbit") == []
    assert search_projects([], "alpha") == []


def test_the_lock_is_what_keeps_concurrent_searches_from_crossing_stores(monkeypatch):
    """The falsifier for ``_LOCK``, forced at its switch point.

    The race window is between one thread's ``sync`` and its scoring pass —
    bytecodes wide, so ordinary scheduling almost never lands in it. The switch
    point is therefore FORCED: ``resolved`` sleeps after computing, widening
    that window deliberately. With the lock, each call is atomic across its own
    store and every answer is right; with ``_LOCK`` removed, the other thread's
    ``sync`` prunes this store's ids mid-flight and the answer comes back
    EMPTY. Verified both ways (see the PR): passes with the lock, fails with
    it removed.
    """

    stores = {
        letter: [
            _row(_pid(), f"{letter}-store", description=f"the {letter} keyword appears only here")
        ]
        for letter in ("aaa", "bbb")
    }
    real_resolved = ps._FieldIndex.resolved

    def switching(self, tokens):
        result = real_resolved(self, tokens)
        time.sleep(0.0002)  # the switch point: another thread's sync lands here
        return result

    monkeypatch.setattr(ps._FieldIndex, "resolved", switching)
    wrong: list[str] = []

    def run(letter: str) -> None:
        other = "bbb" if letter == "aaa" else "aaa"
        for _ in range(24):
            hits = search_projects(stores[letter], f"{letter} keyword")
            if _ids(hits) != [stores[letter][0].id]:
                wrong.append(f"{letter} answered {_ids(hits)} (other store: {other})")

    threads = [threading.Thread(target=run, args=(letter,)) for letter in ("aaa", "bbb")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert wrong == [], f"cross-store answers: {wrong[:5]}"
