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
from pathlib import Path

from local_operator.harness.types import Message, TextContent
from local_operator.session.search_index import (
    DIGEST_CHARS,
    INDEX_VERSION,
    build_index,
    digest_transcript,
    index_path,
    search_digests,
)
from local_operator.session.transcript import Transcript


def _write(session_dir: Path, *turns: tuple[str, str]) -> None:
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
