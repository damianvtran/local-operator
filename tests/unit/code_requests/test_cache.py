"""The cache: disk round-trips, bounds, TTLs, dirty marks, the two throttles."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from local_operator.code_requests import cache
from local_operator.code_requests.adapters.base import FetchOutcome
from local_operator.code_requests.refs import Ref, parse_any


def _ref(url: str) -> Ref:
    """A parsed ref, typed: ``parse_any`` returns ``Ref | None`` and a module
    constant narrowed by an ``assert`` does not stay narrowed inside functions,
    which is exactly where these tests consume it."""
    ref = parse_any(url)
    assert ref is not None, url
    return ref


REF = _ref("https://github.com/o/r/pull/7")
GLREF = _ref("https://gitlab.com/g/s/p/-/merge_requests/4")


@pytest.fixture(autouse=True)
def _clean_memory():
    cache._reset_for_tests()
    yield
    cache._reset_for_tests()


def test_entry_round_trips_memory_and_disk(tmp_path: Path) -> None:
    entry = {
        "key": REF.key,
        "pieces": {"summary": {"title": "t"}},
        "validators": {"summary": {"etag": "x"}},
    }
    assert cache.write_entry(tmp_path, entry, ref=REF)
    read = cache.read_entry(tmp_path, REF)
    assert read is not None and read["pieces"]["summary"]["title"] == "t"
    assert read["schema"] == cache.ENTRY_SCHEMA
    # A memory hit survives deleting the disk copy; the disk hit repopulates
    # memory after a restart (simulated by clearing the memory layer).
    cache._reset_for_tests()
    assert cache.read_entry(tmp_path, REF) is not None
    cache.drop_entry(tmp_path, REF)
    assert cache.read_entry(tmp_path, REF) is None


def test_schema_mismatch_is_ignored_not_misread(tmp_path: Path) -> None:
    path = cache.entry_path(tmp_path, host=REF.host, project=REF.project, number=REF.number)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": cache.ENTRY_SCHEMA + 1, "pieces": {}}), encoding="utf-8")
    assert cache.read_entry(tmp_path, REF) is None


def test_bound_keeps_newest_comments_and_caps_bodies(tmp_path: Path) -> None:
    long_body = "x" * (cache.COMMENT_BODY_MAX + 50)
    comments = [{"id": f"c{i}", "body": long_body, "created_at": i} for i in range(60)]
    entry = {"pieces": {"comments": comments, "summary": {"title": "t"}}}
    cache.write_entry(tmp_path, entry, ref=REF)
    read = cache.read_entry(tmp_path, REF)
    assert read is not None
    stored = read["pieces"]["comments"]
    assert len(stored) == cache.COMMENT_KEEP_MAX
    # Newest kept: the last id survives, the earliest do not.
    assert stored[-1]["id"] == "c59"
    assert stored[0]["id"] == f"c{60 - cache.COMMENT_KEEP_MAX}"
    # Body capped with the marker, and the marker is honest about truncation.
    assert stored[0]["body"].endswith("[truncated]")
    assert len(stored[0]["body"]) == cache.COMMENT_BODY_MAX + len("\n[truncated]")


def test_ttl_table_matches_the_design(tmp_path: Path) -> None:
    open_pending = {"state": "open", "ci": {"status": "pending"}}
    open_settled = {"state": "open", "ci": {"status": "success"}}
    assert cache.ttl_seconds(open_pending, forge="github") == 60.0
    assert cache.ttl_seconds(open_pending, forge="gitlab") == 90.0
    assert cache.ttl_seconds(open_settled, forge="github") == 300.0
    assert cache.ttl_seconds({"state": "merged"}, forge="github") == 86400.0
    assert cache.ttl_seconds(None, forge="github") is None


def test_is_expired_follows_checked_at() -> None:
    now = time.time()
    entry = {"state": "open", "ci": {"status": "success"}, "checked_at": now - 301}
    assert cache.is_expired(entry, forge="github")
    fresh = {"state": "open", "ci": {"status": "success"}, "checked_at": now}
    assert not cache.is_expired(fresh, forge="github")


def test_dirty_marks_merge_consume_and_respect_since(tmp_path: Path) -> None:
    key = REF.key
    assert cache.mark_dirty(tmp_path, "s1", keys=[key])
    assert cache.mark_dirty(tmp_path, "s1", keys=["other"])
    info = cache.read_dirty(tmp_path, "s1")
    assert info["keys"] == [key, "other"]
    snapshot = info["at"]
    # A mark that landed AFTER the snapshot survives the consume (the race the
    # ``since`` guard exists for).
    time.sleep(0.01)
    assert cache.mark_dirty(tmp_path, "s1", keys=["late"])
    cache.clear_dirty(tmp_path, "s1", keys=[key], since=snapshot)
    assert cache.read_dirty(tmp_path, "s1")["keys"] == [key, "other", "late"]
    later = cache.read_dirty(tmp_path, "s1")["at"]
    cache.clear_dirty(tmp_path, "s1", keys=[key, "other"], since=later)
    assert cache.read_dirty(tmp_path, "s1")["keys"] == ["late"]


def test_dirty_all_is_a_noop_without_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A session with no tracked rows must not litter the store: the mark's own
    # ledger check gates it (``ledger.read_index`` returns None here).
    assert cache.mark_dirty(tmp_path, "s1", all_rows=True) is False
    assert cache.read_dirty(tmp_path, "s1")["all"] is False


def test_cooling_until_reset_and_exponential_fallback() -> None:
    # REAL time here, not a synthetic epoch: ``cooling_map`` stamps against the
    # wall clock, so a 1970-era value would be filtered out as already-lifted.
    now = time.time()
    # A GitHub reset header wins.
    until = cache.note_rate_limited("github.com", reset_at=now + 120, now=now)
    assert until == now + 120
    assert cache.cooling_map() == {"github.com": now + 120}
    cache.note_host_success("github.com")
    assert cache.cooling_map() == {}
    # No headers: 30 s, then 60 s for an immediate second offence (doubling).
    first = cache.note_rate_limited("gitlab.com", now=now)
    assert first == now + cache.BACKOFF_BASE_S
    second = cache.note_rate_limited("gitlab.com", now=now)
    assert second == now + 2 * cache.BACKOFF_BASE_S
    # Retry-After wins when present.
    third = cache.note_rate_limited("gitlab.com", retry_after=45, now=now)
    assert third == now + 45


def test_key_backoff_doubles_and_clears() -> None:
    now = 2_000_000.0
    assert cache.note_key_failure("k", now=now) == now + cache.BACKOFF_BASE_S
    assert cache.note_key_failure("k", now=now) == now + 2 * cache.BACKOFF_BASE_S
    assert cache.key_backoff_until("k", now=now) is not None
    cache.clear_key_backoff("k")
    assert cache.key_backoff_until("k", now=now) is None


def test_merge_pieces_replaces_keeps_and_preserves_stored_validators() -> None:
    stored = {
        "pieces": {"summary": {"title": "old"}, "comments": [{"id": "c1"}]},
        "validators": {"summary": {"etag": "s1"}, "comments": {"etag": "c1"}},
    }
    outcome = FetchOutcome()
    outcome.pieces["summary"] = {"title": "new"}
    outcome.not_modified.add("comments")
    outcome.validators["summary"] = {"etag": "s2"}
    pieces, validators = cache.merge_pieces(stored, outcome)
    assert pieces["summary"]["title"] == "new"
    assert pieces["comments"] == [{"id": "c1"}]  # untouched piece kept
    assert validators["comments"] == {"etag": "c1"}  # its validator kept
    assert validators["summary"] == {"etag": "s2"}  # replaced by the fetch


def test_fetch_dir_is_per_host_and_project_encoded(tmp_path: Path) -> None:
    path = cache.entry_path(tmp_path, host="gitlab.com", project="a/b/c", number=4)
    assert path.parent.name == "gitlab.com"
    assert path.name == "a%2Fb%2Fc__4.json"
    dirty = cache.dirty_path(tmp_path, "abcdef123456")
    assert dirty.parent.name == ".dirty"
