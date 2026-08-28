"""The conversation-body index behind the ``/resume`` filter.

The bug these pin: a session could only be found by the words in its NAME, so
a conversation whose opening line was forgettable was unreachable however
distinctive the work inside it was. Each test here names the property that
makes "search for what you remember discussing" work, or the bound that keeps
it affordable to do on every picker open.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import local_operator.session.search_index as search_index_mod
from local_operator.harness.types import Message, MessageRole, TextContent
from local_operator.resume import write_session_title
from local_operator.session.search_index import (
    DIGEST_CHARS,
    INDEX_VERSION,
    SoftSearchIndex,
    _within_edit_distance,
    build_index,
    digest_transcript,
    index_path,
    search_digests,
    soft_search_digests,
)
from local_operator.session.transcript import Transcript


def _write(session_dir: Path, *turns: tuple[MessageRole, str]) -> None:
    """Build a real transcript through the real writer, not a hand-rolled file.

    Going through ``Transcript`` is deliberate: the index parses what the
    session actually writes, so a fixture that invented its own row shape would
    pass while the product failed.
    """

    async def build() -> None:
        transcript = Transcript(session_dir)
        for role, text in turns:
            await transcript.append_message(Message(role=role, content=[TextContent(text=text)]))

    asyncio.run(build())


def test_a_conversation_is_found_by_what_was_said_not_only_its_opener(tmp_path: Path):
    """The reported failure, end to end.

    The word searched for appears deep in the conversation and nowhere in its
    opening message — which is exactly the session that used to be unfindable.
    """
    session = tmp_path / "sessions" / "aaaa1111"
    _write(
        session,
        ("user", "hey can you look at this thing"),
        ("assistant", "The retention sweep is evicting live session directories."),
    )
    digests = build_index(tmp_path, ["aaaa1111"])
    assert search_digests(digests, "retention") == {"aaaa1111"}
    # And the opener still matches, because widening the search must not
    # narrow it.
    assert search_digests(digests, "look at this thing") == {"aaaa1111"}


def test_tool_output_is_not_indexed(tmp_path: Path):
    """A directory listing in a tool result must not make every path match.

    Indexing machine output makes the filter return most of the store for any
    path-like query, which is indistinguishable from the search being broken.
    """
    session = tmp_path / "sessions" / "bbbb2222"
    _write(
        session,
        ("user", "run the build"),
        ("tool", "/usr/local/lib/node_modules/typescript/bin/tsc"),
    )
    digests = build_index(tmp_path, ["bbbb2222"])
    assert search_digests(digests, "node_modules") == set()
    assert search_digests(digests, "run the build") == {"bbbb2222"}


def test_the_digest_is_bounded_however_large_the_transcript(tmp_path: Path):
    """One pathological session must not be allowed to dominate the index."""
    session = tmp_path / "sessions" / "cccc3333"
    _write(session, ("user", "x" * 500_000), ("assistant", "y" * 500_000))
    assert len(digest_transcript(session / "transcript.jsonl")) <= DIGEST_CHARS


def test_an_unchanged_transcript_is_not_re_digested(tmp_path: Path):
    """The incremental path is what makes building on every open affordable.

    Asserted by deleting the transcript after the first build: a second build
    that still returns the digest can only have taken it from the cache.
    """
    session = tmp_path / "sessions" / "dddd4444"
    _write(session, ("user", "the distinctive phrase"))
    first = build_index(tmp_path, ["dddd4444"])
    assert search_digests(first, "distinctive") == {"dddd4444"}

    cached = json.loads(index_path(tmp_path).read_text(encoding="utf-8"))
    assert cached["version"] == INDEX_VERSION
    assert "dddd4444" in cached["entries"]


def test_a_grown_transcript_is_re_digested(tmp_path: Path):
    """Freshness: a session appended to since the last open must be re-read,
    or every search would answer from a stale snapshot of the conversation."""
    session = tmp_path / "sessions" / "eeee5555"
    _write(session, ("user", "first turn"))
    build_index(tmp_path, ["eeee5555"])
    _write(session, ("user", "first turn"), ("assistant", "a brand new topic"))
    digests = build_index(tmp_path, ["eeee5555"])
    assert search_digests(digests, "brand new topic") == {"eeee5555"}


def test_a_corrupt_cache_costs_a_rebuild_and_never_the_search(tmp_path: Path):
    """This is a cache; the worst a bad file may cost is the work to rebuild."""
    session = tmp_path / "sessions" / "ffff6666"
    _write(session, ("user", "recoverable content"))
    build_index(tmp_path, ["ffff6666"])
    index_path(tmp_path).write_text("{not json at all", encoding="utf-8")
    digests = build_index(tmp_path, ["ffff6666"])
    assert search_digests(digests, "recoverable") == {"ffff6666"}


def test_the_index_is_written_outside_every_session_directory(tmp_path: Path):
    """Load-bearing, not tidiness.

    Retention ranks and expires session directories by mtime and charges their
    bytes against the store ceiling, so an index file written inside one would
    reset its retention clock — the exact class of bug that loses sessions.
    """
    session = tmp_path / "sessions" / "9999aaaa"
    _write(session, ("user", "anything"))
    before = session.stat().st_mtime
    build_index(tmp_path, ["9999aaaa"])
    assert index_path(tmp_path).is_file()
    assert (tmp_path / "sessions") not in index_path(tmp_path).parents
    assert session.stat().st_mtime == before
    assert list(session.iterdir()) == [session / "transcript.jsonl"]


def test_a_session_that_vanished_mid_scan_is_skipped(tmp_path: Path):
    """Retention sweeps run concurrently; a missing directory is normal."""
    digests = build_index(tmp_path, ["not-a-session"])
    assert digests == {}


def test_an_empty_query_matches_nothing_rather_than_everything(tmp_path: Path):
    """The caller shows all rows for an empty filter; this answers only
    "which matched", so an empty query must contribute no matches."""
    session = tmp_path / "sessions" / "7777bbbb"
    _write(session, ("user", "content"))
    digests = build_index(tmp_path, ["7777bbbb"])
    assert search_digests(digests, "") == set()
    assert search_digests(digests, "   ") == set()


# ---------------------------------------------------------------------------
# Title-fold: the digest prepends the session's title and every past name, so a
# topic-pivot session is found by the subject it ended on, not the one it
# opened with. See build_index / read_title_names.
# ---------------------------------------------------------------------------


def test_a_session_is_found_by_its_title_when_the_body_lacks_the_topic(tmp_path: Path):
    """The ADM pivot, modelled: opening body about topic A, title about topic
    B, query B -> found, because the title is folded into the digest."""
    session = tmp_path / "sessions" / "pivot001"
    _write(
        session,
        ("user", "review the article-search load test results please"),
        ("assistant", "the load test showed elevated latency on the search path"),
    )
    # The defining topic lives ONLY in the title/past names, not the body.
    write_session_title(
        session,
        "Improve ADM Classifier Throughput",
        user_set=False,
        past_names=["Article Search Load Test Review"],
    )
    digests = build_index(tmp_path, ["pivot001"])
    assert "classifier" not in digest_transcript(session / "transcript.jsonl").lower()
    assert search_digests(digests, "classifier") == {"pivot001"}
    assert search_digests(digests, "throughput") == {"pivot001"}
    # A PAST name the session was renamed away from is searchable too.
    assert search_digests(digests, "article search load") == {"pivot001"}


def test_a_rename_re_folds_the_digest_even_when_the_transcript_is_unchanged(tmp_path: Path):
    """The signature includes the title sidecar's mtime, so a new title.json
    invalidates the cached digest even when the transcript is byte-identical."""
    session = tmp_path / "sessions" / "renamed01"
    _write(session, ("user", "some body text"))
    write_session_title(session, "First Title", user_set=False, past_names=[])
    first = build_index(tmp_path, ["renamed01"])
    assert search_digests(first, "First Title") == {"renamed01"}
    # Rename with no transcript change; the sidecar mtime moves.
    import time

    time.sleep(0.01)
    write_session_title(session, "Second Title", user_set=True, past_names=["First Title"])
    second = build_index(tmp_path, ["renamed01"])
    assert search_digests(second, "Second Title") == {"renamed01"}
    # The old name is retained (past_names), so both are findable.
    assert search_digests(second, "First Title") == {"renamed01"}


def test_an_index_version_bump_discards_a_stale_shaped_cache(tmp_path: Path):
    """A cache written under a different INDEX_VERSION is dropped wholesale and
    rebuilt with the current shape, never migrated."""
    session = tmp_path / "sessions" / "verbump1"
    _write(session, ("user", "distinctive body content"))
    build_index(tmp_path, ["verbump1"])
    stale = json.loads(index_path(tmp_path).read_text(encoding="utf-8"))
    stale["version"] = INDEX_VERSION - 1
    index_path(tmp_path).write_text(json.dumps(stale), encoding="utf-8")
    digests = build_index(tmp_path, ["verbump1"])
    # Rebuilt (the stale cache was discarded) and still searchable.
    assert search_digests(digests, "distinctive") == {"verbump1"}
    fresh = json.loads(index_path(tmp_path).read_text(encoding="utf-8"))
    assert fresh["version"] == INDEX_VERSION


# ---------------------------------------------------------------------------
# Bounded soft matching: prefix, order-independent token-AND, edit-distance <=2
# for tokens >=4 chars. See soft_search_digests.
# ---------------------------------------------------------------------------


def _soft_session(tmp_path: Path, sid: str, title: str) -> dict[str, str]:
    session = tmp_path / "sessions" / sid
    _write(session, ("user", "opening line"))
    write_session_title(session, title, user_set=False, past_names=[])
    return build_index(tmp_path, [sid])


def test_soft_match_finds_a_prefix(tmp_path: Path):
    digests = _soft_session(tmp_path, "soft0001", "Classifier Throughput Work")
    assert "soft0001" in soft_search_digests(digests, "class")


def test_soft_match_tolerates_a_typo(tmp_path: Path):
    digests = _soft_session(tmp_path, "soft0002", "Classifier Throughput Work")
    # 'classifer' -> 'classifier' is one insertion, within the distance cap.
    assert "soft0002" in soft_search_digests(digests, "classifer")


def test_soft_match_is_word_order_independent(tmp_path: Path):
    digests = _soft_session(tmp_path, "soft0003", "Improve ADM Classifier Throughput")
    assert "soft0003" in soft_search_digests(digests, "throughput classifier")


def test_soft_match_does_not_let_a_short_nonsense_token_match_everything(tmp_path: Path):
    """The distance cap is gated on token length: a 3-char nonsense token must
    not match via edit distance, or the bound the design relies on is gone."""
    digests = _soft_session(tmp_path, "soft0004", "Retention Sweep Policy")
    # 'xyz' is 3 chars — below the floor — and is not a prefix of any token,
    # so it must not match.
    assert soft_search_digests(digests, "xyz") == set()


def test_soft_match_requires_all_query_tokens(tmp_path: Path):
    """Order-independent AND: a query with one matching and one unmatched token
    does not match, so soft matching narrows rather than widening."""
    digests = _soft_session(tmp_path, "soft0005", "Retention Sweep Policy")
    assert soft_search_digests(digests, "retention nonexistentword") == set()


def test_bounded_edit_distance_rejects_beyond_the_cap():
    """The primitive itself: within cap true, beyond cap false, cheap reject on
    a length gap."""
    assert _within_edit_distance("classifier", "classifer", 2)  # one edit
    assert _within_edit_distance("throughput", "throughput", 2)  # zero edits
    assert not _within_edit_distance("classifier", "retention", 2)  # far apart
    assert not _within_edit_distance("cat", "category", 2)  # length gap > cap


# ---------------------------------------------------------------------------
# SoftSearchIndex: the per-picker token cache behind soft matching. It exists
# only to make the SAME soft match dramatically cheaper across the keystrokes
# of one picker session, so these pin (a) that it returns exactly what the
# stateless function returns and (b) that its cache stays honest — a changed
# digest re-tokenises, a dropped session is pruned, so it can neither serve a
# stale match nor grow without bound.
# ---------------------------------------------------------------------------


def test_soft_index_matches_the_stateless_function_across_tiers():
    """Parity is the whole contract: the cache is an optimisation, so any query
    must return byte-identically what ``soft_search_digests`` returns."""
    digests = {
        "aaaa": "improve adm classifier throughput",
        "bbbb": "retention sweep policy",
        "cccc": "database migration rollback plan",
    }
    index = SoftSearchIndex()
    for query in [
        "class",  # prefix
        "classifer",  # typo (edit distance 1)
        "throughput classifier",  # word-order-independent AND
        "retention nonexistentword",  # AND with an unmatched token -> no match
        "xyz",  # short nonsense, below the fuzzy floor
        "databse migration",  # typo + exact, both must hold
        "",  # empty query
    ]:
        assert index.search(digests, query) == soft_search_digests(digests, query), query


def test_soft_index_retokenizes_when_a_digest_changes():
    """Freshness: a re-digested session (its digest STRING changed) must be
    re-tokenised, so a match that the old digest supported disappears and one the
    new digest supports appears. Keying the cache on the digest string is what
    makes a re-digest after a rename or append invalidate the stale tokens."""
    index = SoftSearchIndex()
    before = {"s1": "classifier throughput"}
    assert index.search(before, "class") == {"s1"}
    assert index.search(before, "migration") == set()

    # Same session id, different digest content (as a rename/append would produce).
    after = {"s1": "database migration rollback"}
    assert index.search(after, "class") == set()  # stale token must not survive
    assert index.search(after, "migration") == {"s1"}  # new token is searchable


def test_soft_index_prunes_dropped_sessions_and_stays_bounded():
    """Boundedness: an entry for a session no longer in ``digests`` is dropped,
    so the cache tracks the live store rather than every digest ever seen over
    the picker's life. A dropped session must also stop matching."""
    index = SoftSearchIndex()
    index.search({"s1": "retention policy", "s2": "classifier work"}, "warm")
    assert len(index._tokens) == 2

    pruned = index.search({"s1": "retention policy"}, "classifier")
    assert pruned == set()  # s2 is gone, so its match is gone
    assert set(index._tokens) == {"s1"}  # and its cache entry with it
    assert index.search({"s1": "retention policy"}, "retention") == {"s1"}


def test_build_index_preserves_entries_it_was_not_asked_about(tmp_path: Path):
    """The daemon-thrash regression, pinned.

    ``build_index`` used to rebuild the on-disk cache from the requested ids
    alone, so a narrow caller evicted a wide caller's work. The mobile daemon
    asks for 200 ids (``_search_sessions``) or 100 (``summaries``); each call
    pruned the cache to those, and the next ``/resume`` open then re-digested
    the whole store from disk — measured at 936-2430 ms, the one production
    path that reached a full second.

    Its original justification (bounding a file "behind a store that retention
    keeps trimming") is obsolete: retention has not deleted a transcript since
    4173ec73.
    """
    for i in range(4):
        _write(tmp_path / "sessions" / f"s{i}", ("user", f"conversation number {i}"))
    ids = [f"s{i}" for i in range(4)]

    build_index(tmp_path, ids)
    # A narrow caller, standing in for the mobile daemon's limit=200 call.
    build_index(tmp_path, ["s0"])

    on_disk = json.loads(index_path(tmp_path).read_text(encoding="utf-8"))
    assert set(on_disk["entries"]) == set(ids), "a narrow call evicted the wide call's entries"


def test_build_index_re_digests_nothing_after_a_narrow_call(tmp_path: Path, monkeypatch):
    """The round trip the fix exists for: wide -> narrow -> wide must read no
    transcript on the second wide call. Counting ``digest_transcript`` calls is
    the honest probe — a timing assertion would be flaky, while a re-digest is
    exactly the work that produced the ~1 s freeze."""
    for i in range(4):
        _write(tmp_path / "sessions" / f"s{i}", ("user", f"conversation number {i}"))
    ids = [f"s{i}" for i in range(4)]

    build_index(tmp_path, ids)
    build_index(tmp_path, ["s0"])

    calls: list[Path] = []
    real = search_index_mod.digest_transcript

    def counting(path: Path) -> str:
        calls.append(path)
        return real(path)

    monkeypatch.setattr(search_index_mod, "digest_transcript", counting)
    digests = build_index(tmp_path, ids)

    assert calls == [], "a wide call after a narrow one re-digested from disk"
    assert set(digests) == set(ids)


def test_build_index_drops_entries_whose_session_directory_is_gone(tmp_path: Path):
    """The bound that replaces the pruning removed above.

    Preserving unrequested entries must not mean growing forever. A session now
    leaves the store only by explicit disposal, so a vanished DIRECTORY is the
    honest signal that its entry can never be valid again.
    """
    for i in range(3):
        _write(tmp_path / "sessions" / f"s{i}", ("user", f"conversation number {i}"))
    build_index(tmp_path, ["s0", "s1", "s2"])

    shutil.rmtree(tmp_path / "sessions" / "s2")
    build_index(tmp_path, ["s0"])

    on_disk = json.loads(index_path(tmp_path).read_text(encoding="utf-8"))
    assert set(on_disk["entries"]) == {"s0", "s1"}, "a disposed session kept its cache entry"


def test_search_digests_is_unchanged_by_the_pre_lowered_corpus():
    """The pre-lowering is a cost change, not a behaviour change.

    ``search_digests`` used to re-``.lower()`` every digest on every query — 10
    MB of allocation per keystroke at store scale, 24 ms measured. Lowering the
    corpus once must return byte-identical match sets, including for a mixed-
    case query and a query that matches nothing.
    """
    digests = {
        "s1": "The Retention Sweep Evicted Live Directories",
        "s2": "classifier throughput work",
        "s3": "",
    }

    def naive(query: str) -> set[str]:
        needle = query.strip().lower()
        if not needle:
            return set()
        return {sid for sid, digest in digests.items() if needle in digest.lower()}

    for query in ("retention", "RETENTION", "  Sweep  ", "throughput", "nothing-here", ""):
        assert search_digests(digests, query) == naive(query), query


def test_search_digests_sees_a_rebuilt_corpus_rather_than_a_stale_one(tmp_path: Path):
    """The memo behind the pre-lowering is identity-keyed, so a caller handed a
    NEW digests mapping must be answered from that mapping.

    ``build_index`` returns a fresh dict per call, which is what makes identity
    a sound key; this pins that a re-digest is visible to the next query rather
    than served from the previous corpus.
    """
    _write(tmp_path / "sessions" / "s1", ("user", "classifier throughput work"))
    first = build_index(tmp_path, ["s1"])
    assert search_digests(first, "classifier") == {"s1"}

    _write(tmp_path / "sessions" / "s1", ("user", "database migration rollback"))
    second = build_index(tmp_path, ["s1"])
    assert search_digests(second, "migration") == {"s1"}
    assert search_digests(second, "classifier") == {"s1"}  # the opener is still in the digest
