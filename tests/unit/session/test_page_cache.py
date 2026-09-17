"""The process-wide decoded-page cache: identity, single-flight, and its bounds.

Four properties, and each test here is one of them:

* a hit reads NOTHING (asserted on a counting instrument, never on a clock);
* the key is the FILE'S IDENTITY, so a same-size replacement by a different
  inode returns the new content while a rollback that restores the old bytes may
  legitimately hit;
* concurrent callers for one key share ONE read — and a follower after an append
  does not get the stale rows;
* the byte bound is a working instrument, which takes a walker that descends
  into the page rather than stopping at it.

House rules apply throughout: no ``sleep``-then-assert, no wall-clock threshold
(a numeric ceiling here would measure the machine — this box runs at load
average ~200), and every wait is on a completion the code itself publishes.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.session import page_cache as page_cache_module
from local_operator.session.page_cache import (
    PAGE_CACHE_ENTRIES,
    PageKey,
    TranscriptPageCache,
    load_transcript_page,
    page_cache,
    page_key,
    reset_page_cache,
    retained_bytes,
)
from local_operator.session.transcript import (
    ENTRY_MESSAGE,
    TRANSCRIPT_FILENAME,
    TranscriptEntry,
    read_transcript_page,
)

#: Loop turns a follower needs to JOIN an in-flight read. Its join is a loop
#: step with no I/O behind it, so a turn count is the honest bound: contention
#: stretches how long a turn takes, never how many the join needs (AGENTS.md,
#: "Wait on the event, never on the clock").
_JOIN_TURNS = 8

#: Backstop for a wait that must never time out. Not a latency budget: the
#: assertion is always the state the wait produces, and this only turns a
#: regression into a named failure instead of a hung suite.
_BACKSTOP_S = 30.0


@pytest.fixture(autouse=True)
def _empty_cache():
    """A process-wide cache is process-wide state: no test may inherit another's."""
    reset_page_cache()
    yield
    reset_page_cache()


def _journal(directory: Path, ids: list[str]) -> Path:
    """Write ``ids`` as one message row each, in order, and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TRANSCRIPT_FILENAME
    path.write_text(
        "".join(
            TranscriptEntry(
                row_id,
                float(index),
                ENTRY_MESSAGE,
                {"role": "user", "content": row_id},
            ).to_json()
            + "\n"
            for index, row_id in enumerate(ids)
        ),
        encoding="utf-8",
    )
    return path


def _key(directory: Path, **kwargs: Any) -> PageKey:
    """``page_key`` for a journal this test just wrote: the absent arm cannot be taken."""
    key = page_key(directory, **kwargs)
    assert key is not None, "the fixture's journal is missing"
    return key


def _ids(page: Any) -> list[str]:
    return [row.id for row in page.entries]


@pytest.mark.asyncio
async def test_a_hit_reads_nothing_and_returns_the_same_page(tmp_path, monkeypatch):
    """The cache's whole purpose, asserted where a clock cannot fake it.

    One read for two loads, counted at the reader. The name of the mutation this
    catches is "the key lost a field" or "put() never stored anything": either
    one makes the second load a MISS and takes the count to two.
    """
    directory = tmp_path / "sess"
    _journal(directory, ["a", "b", "c"])
    reads = 0
    real = page_cache_module.read_transcript_page

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal reads
        reads += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(page_cache_module, "read_transcript_page", counting)

    first = await load_transcript_page(directory, limit=10)
    second = await load_transcript_page(directory, limit=10)

    assert reads == 1, "the second load went back to disk"
    assert _ids(second) == ["a", "b", "c"]
    assert second is first
    assert page_cache().hits == 1 and page_cache().misses == 1


@pytest.mark.asyncio
async def test_a_cursor_page_and_a_tail_page_of_one_file_do_not_collide(tmp_path):
    """The cursor fields are part of the key, not decoration on it."""
    directory = tmp_path / "sess"
    _journal(directory, ["a", "b", "c", "d"])

    tail = await load_transcript_page(directory, limit=4)
    before = await load_transcript_page(directory, before_id="c", limit=4)
    through = await load_transcript_page(directory, through_id="b", limit=4)

    assert _ids(tail) == ["a", "b", "c", "d"]
    assert _ids(before) == ["a", "b"]
    assert _ids(through) == ["a", "b"]
    # …and the tail page is still the same cached object afterwards, so the
    # three keys above genuinely coexisted rather than overwriting each other.
    assert await load_transcript_page(directory, limit=4) is tail


@pytest.mark.asyncio
async def test_an_append_is_a_miss_and_never_serves_a_stale_tail(tmp_path):
    """Size in the key, driven by the append the operator actually makes."""
    directory = tmp_path / "sess"
    path = _journal(directory, ["a", "b"])
    assert _ids(await load_transcript_page(directory, limit=10)) == ["a", "b"]

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            TranscriptEntry("c", 9.0, ENTRY_MESSAGE, {"role": "user", "content": "c"}).to_json()
            + "\n"
        )

    appended = await load_transcript_page(directory, limit=10)

    assert _ids(appended) == ["a", "b", "c"], "the append was served a stale page"
    assert page_cache().misses == 2


@pytest.mark.asyncio
async def test_a_same_size_replacement_serves_the_new_content(tmp_path):
    """The ``compact_file`` shape: a different inode behind an identical length.

    This is the case a key of ``(size, cursors)`` alone — or a per-``Transcript``
    generation counter — cannot see, and it is the reason the key carries
    ``st_ino``: ``compact_file`` writes a temp file and ``os.replace``s it, so
    every byte of the file can change while its length does not.
    """
    directory = tmp_path / "sess"
    path = _journal(directory, ["old1", "old2"])
    assert _ids(await load_transcript_page(directory, limit=10)) == ["old1", "old2"]

    replacement = tmp_path / "replacement"
    _journal(replacement, ["new1", "new2"])
    assert (
        replacement / TRANSCRIPT_FILENAME
    ).stat().st_size == path.stat().st_size, "the fixture must be same-size, or this proves nothing"
    os.replace(replacement / TRANSCRIPT_FILENAME, path)

    assert _ids(await load_transcript_page(directory, limit=10)) == ["new1", "new2"]


@pytest.mark.asyncio
async def test_a_rollback_to_a_cached_size_answers_with_the_cached_page(tmp_path):
    """The one case where a hit on an old key is CORRECT, not stale.

    ``_commit``'s failure path truncates the journal back to the size it had
    before the batch (``os.truncate``, same inode). The restored bytes ARE the
    bytes the cached page was decoded from, so the cached page describes the file
    that is now there — which is why size is in the key rather than mtime: an
    append under ``preserve_mtime`` rewinds the clock on purpose and would leave
    an mtime-keyed cache serving the page the rollback just removed.
    """
    directory = tmp_path / "sess"
    path = _journal(directory, ["a", "b"])
    original = path.stat().st_size
    first = await load_transcript_page(directory, limit=10)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            TranscriptEntry("c", 9.0, ENTRY_MESSAGE, {"role": "user", "content": "c"}).to_json()
            + "\n"
        )
    assert _ids(await load_transcript_page(directory, limit=10)) == ["a", "b", "c"]

    os.truncate(path, original)
    restored = await load_transcript_page(directory, limit=10)

    assert _ids(restored) == _ids(read_transcript_page(directory, limit=10)) == ["a", "b"]
    assert restored is first, "the restored size is the cached key, and that is the point"


@pytest.mark.asyncio
async def test_two_concurrent_loads_issue_one_read(tmp_path, monkeypatch):
    """Single-flight, with the read parked so the follower must join it.

    Without the shared task both callers read the same bytes, and on a 261 MB
    conversation their two parses contend with the loop through the GIL — which
    is what makes the duplicate more than wasted work. The leader is parked
    INSIDE the read, so the follower has nothing to race against: it either joins
    the flight or starts a second read, and the count says which.
    """
    directory = tmp_path / "sess"
    _journal(directory, ["a", "b", "c"])
    reads = 0
    entered, release = threading.Event(), threading.Event()
    real = page_cache_module.read_transcript_page

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal reads
        reads += 1
        entered.set()
        if not release.wait(_BACKSTOP_S):
            raise AssertionError("the test never released the parked read")
        return real(*args, **kwargs)

    monkeypatch.setattr(page_cache_module, "read_transcript_page", counting)

    first = asyncio.create_task(load_transcript_page(directory, limit=10))
    assert await asyncio.to_thread(entered.wait, _BACKSTOP_S), "the read never started"
    second = asyncio.create_task(load_transcript_page(directory, limit=10))
    for _ in range(_JOIN_TURNS):
        await asyncio.sleep(0)
    release.set()
    leader, follower = await asyncio.wait_for(asyncio.gather(first, second), _BACKSTOP_S)

    assert reads == 1, "the follower started its own read of the same bytes"
    assert leader is follower
    assert _ids(leader) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_a_follower_after_an_append_does_not_share_the_stale_read(tmp_path, monkeypatch):
    """A shared read is keyed on the file, so a write during it is not shared.

    The reader is parked across an append: the flight's read answers from the
    file as it was, and its publication is therefore refused (the post-read stat
    moved), so the next request decodes the file that is now there. A cache that
    published anyway would serve rows the journal no longer ends with — and
    nothing about the served page would look wrong.
    """
    directory = tmp_path / "sess"
    path = _journal(directory, ["a", "b"])
    entered, release = threading.Event(), threading.Event()
    real = page_cache_module.read_transcript_page

    def parked(*args: Any, **kwargs: Any) -> Any:
        page = real(*args, **kwargs)
        entered.set()
        if not release.wait(_BACKSTOP_S):
            raise AssertionError("the test never released the parked read")
        return page

    monkeypatch.setattr(page_cache_module, "read_transcript_page", parked)

    inflight = asyncio.create_task(load_transcript_page(directory, limit=10))
    assert await asyncio.to_thread(entered.wait, _BACKSTOP_S), "the read never ran"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            TranscriptEntry("c", 9.0, ENTRY_MESSAGE, {"role": "user", "content": "c"}).to_json()
            + "\n"
        )
    release.set()
    stale = await asyncio.wait_for(inflight, _BACKSTOP_S)
    assert _ids(stale) == ["a", "b"], "the in-flight read is still an honest answer for its key"

    assert page_cache().entry_count == 0, "a read that spanned the append was published"
    assert _ids(await load_transcript_page(directory, limit=10)) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_a_page_too_large_to_admit_is_a_miss_and_not_an_error(tmp_path, monkeypatch):
    """An over-budget page degrades to today's behaviour: it is simply re-read."""
    directory = tmp_path / "sess"
    _journal(directory, ["a", "b"])
    tiny = TranscriptPageCache(entries=PAGE_CACHE_ENTRIES, byte_budget=1024)
    monkeypatch.setattr(page_cache_module, "_PAGE_CACHE", tiny)

    first = await load_transcript_page(directory, limit=10)
    second = await load_transcript_page(directory, limit=10)

    assert _ids(first) == ["a", "b"] and _ids(second) == ["a", "b"]
    assert tiny.entry_count == 0 and tiny.accounted_bytes == 0
    assert tiny.oversize == 2, "an unadmitted page must be counted, or the bound is silent"
    assert second is not first


@pytest.mark.asyncio
async def test_a_page_larger_than_the_entries_cap_evicts_the_oldest(tmp_path):
    """The LRU bound, at its entry dimension."""
    cache = TranscriptPageCache(entries=2, byte_budget=64 * 1024 * 1024)

    def key(sid: str) -> PageKey:
        return _key(tmp_path / sid, limit=10)

    for name in ("one", "two"):
        _journal(tmp_path / name, ["a"])
        cache.put(key(name), await load_transcript_page(tmp_path / name, limit=10))
    assert cache.entry_count == 2

    _journal(tmp_path / "three", ["a"])
    cache.put(key("three"), await load_transcript_page(tmp_path / "three", limit=10))

    assert cache.entry_count == 2
    assert cache.get(key("one")) is None, "the oldest entry was not the one evicted"
    assert cache.get(key("three")) is not None


def test_two_sequential_event_loops_can_each_load(tmp_path):
    """The façade is loop-owned, and nothing about it may outlive a loop.

    ``tests/unit/tui/*`` create and tear down loops inside one process, so a
    flight entry awaited from a different loop would raise ``RuntimeError`` on an
    arbitrary subset of runs rather than fail honestly (design risk 5).
    """
    directory = tmp_path / "sess"
    _journal(directory, ["a", "b"])

    first = asyncio.run(load_transcript_page(directory, limit=10))
    second = asyncio.run(load_transcript_page(directory, limit=10))

    assert _ids(first) == ["a", "b"] and _ids(second) == ["a", "b"]


def test_a_flight_orphaned_by_a_closed_loop_is_treated_as_absent(tmp_path):
    """The loop tag, driven in the direction that FAILS.

    A flight that outlives its loop is exactly what a torn-down test loop leaves
    behind, and awaiting it would be the intermittent ``RuntimeError`` the tag
    exists to remove. Planted rather than provoked: the entry is created on a loop
    that then closes with the task still pending, which is the state the check has
    to read as absent.
    """
    directory = tmp_path / "sess"
    _journal(directory, ["a", "b"])
    key = _key(directory, limit=10)

    async def orphan() -> None:
        async def never() -> None:
            await asyncio.Event().wait()

        # ``cast`` because a deliberately PLANTED entry is not a real transcript
        # read: the point is the map's loop tag, not the task's result type.
        page_cache_module._FLIGHTS[key] = (
            asyncio.get_running_loop(),
            cast("Any", asyncio.create_task(never())),
        )

    asyncio.run(orphan())
    assert key in page_cache_module._FLIGHTS, "the fixture did not plant a flight"

    page = asyncio.run(load_transcript_page(directory, limit=10))

    assert _ids(page) == ["a", "b"]


@pytest.mark.asyncio
async def test_invalidate_drops_one_session_and_leaves_the_others(tmp_path):
    """A session that is gone must not keep its pages until the LRU turns over."""
    _journal(tmp_path / "one", ["a"])
    _journal(tmp_path / "two", ["b"])
    await load_transcript_page(tmp_path / "one", limit=10)
    await load_transcript_page(tmp_path / "two", limit=10)
    assert page_cache().entry_count == 2

    page_cache().invalidate(tmp_path / "one")

    assert page_cache().entry_count == 1
    assert page_cache().get(_key(tmp_path / "two", limit=10)) is not None
    assert page_cache().get(_key(tmp_path / "one", limit=10)) is None


@pytest.mark.asyncio
async def test_the_facade_keeps_the_readers_preconditions(tmp_path):
    """One validator, so the sync reader and the façade cannot drift apart.

    The ORDER matters as much as the set: the desktop route distinguishes these
    by TYPE — ``history`` catches ``FileNotFoundError`` and reconciles, while a
    request carrying both cursors is a programming error its tests pin — so a
    façade that checked the file first would change what a client observes.
    """
    directory = tmp_path / "sess"
    _journal(directory, ["a"])

    with pytest.raises(ValueError, match="choose before_id or through_id, not both"):
        await load_transcript_page(directory, before_id="a", through_id="b")
    with pytest.raises(ValueError, match="limit must be at least 1"):
        await load_transcript_page(directory, limit=0)
    with pytest.raises(FileNotFoundError):
        await load_transcript_page(tmp_path / "absent")
    # Wrong twice: the programming error wins, exactly as it does in the reader.
    with pytest.raises(ValueError, match="choose before_id or through_id, not both"):
        await load_transcript_page(tmp_path / "absent", before_id="a", through_id="b")


def test_retained_bytes_sees_dataclass_payloads(tmp_path):
    """The instrument check, and it is written to fail against a wrong walker.

    ``TranscriptPage`` and ``TranscriptEntry`` are plain dataclasses, so a walk
    that descends only into pydantic models and containers stops AT the page: a
    400 KB page would be accounted as a few dozen bytes, the byte bound would be
    inert, and the process-wide cache would be unbounded on a long-lived server.
    Asserted as "the figure exceeds the page's own serialized size" because a
    dead instrument returns a READING, not an error (AGENTS.md §Timing).
    """
    directory = tmp_path / "sess"
    directory.mkdir()
    (directory / TRANSCRIPT_FILENAME).write_text(
        TranscriptEntry(
            "big",
            1.0,
            ENTRY_MESSAGE,
            {"role": "user", "content": "x" * 400_000},
        ).to_json()
        + "\n",
        encoding="utf-8",
    )
    page = read_transcript_page(directory, limit=10)
    serialized = len(page.entries[0].to_json())

    accounted = retained_bytes((_key(directory), page), 64 * 1024 * 1024)

    assert accounted is not None and accounted > serialized, (
        "the walker cannot see the page's rows: accounted "
        f"{accounted} against {serialized} serialized bytes"
    )
    # And the bound is enforced at that same figure, not at the page object's.
    assert retained_bytes((_key(directory), page), serialized) is None


@pytest.mark.asyncio
async def test_the_byte_budget_admits_a_page_the_size_of_a_real_conversation(tmp_path):
    """The calibration test. It exists because the calibration was wrong once.

    The budget is in ACCOUNTED bytes while every page size anyone quotes is a
    serialized one, and the instrument charges more — measured 1.7-2.7x over six
    real conversations. Sized from the serialized figures alone the cache admitted
    NOTHING for an ordinary conversation: the benchmark's own oversize tally
    counted two refused pages on a 9 MB session whose 100-row page accounts
    4.6 MiB, so the "bounded cache" was a bound on an empty one.

    The fixture therefore asserts its OWN discriminating band — accounted, over
    the old budget and under this one — because a calibration test whose fixture
    drifted below the old bound would pass for the wrong reason and say nothing.
    The page it builds is between the ordinary (2.0-5.4 MiB accounted) and the
    enormous (84 MiB) sizes measured on the operator's store.
    """
    directory = tmp_path / "sess"
    directory.mkdir()
    filler = "y" * 45_000
    (directory / TRANSCRIPT_FILENAME).write_text(
        "".join(
            TranscriptEntry(
                f"r{index}",
                float(index),
                ENTRY_MESSAGE,
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": filler}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 20},
                },
            ).to_json()
            + "\n"
            for index in range(100)
        ),
        encoding="utf-8",
    )
    page = read_transcript_page(directory, limit=100)
    accounted = retained_bytes((_key(directory, limit=100), page), 1 << 40)

    assert accounted is not None
    assert 4 * 1024 * 1024 < accounted < 24 * 1024 * 1024, (
        f"the fixture must sit between the old 4 MiB budget and this one; "
        f"accounted {accounted} bytes"
    )
    await load_transcript_page(directory, limit=100)

    assert page_cache().oversize == 0, "a page the size of a real one was refused"
    assert page_cache().entry_count == 1
